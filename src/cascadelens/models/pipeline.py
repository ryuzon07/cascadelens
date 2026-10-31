"""Train, freeze and evaluate the rankers on saved splits.

``train_folds`` and ``freeze_config`` read training and validation cases only. ``evaluate`` is the
one route to test cases: it goes through the Part 4 runner and the freeze check.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cascadelens.baselines.base import CaseHandle
from cascadelens.baselines.runner import (
    GROUPED,
    Dataset,
    SplitRun,
    load_dataset,
    load_split,
    run_folds,
    truth_table,
)
from cascadelens.config import BaselineConfig, RankerConfig, config_hash, load_config
from cascadelens.eval.metrics import retrieval_recall_at_k
from cascadelens.features.config import FeaturesConfig
from cascadelens.models.artifacts import Provenance, save_artifact
from cascadelens.models.data import FeatureSource, RankerError
from cascadelens.models.freeze import FrozenRecord, frozen_choices, resolve, write_frozen
from cascadelens.models.ranker import FittedFold, TwoStageRanker
from cascadelens.models.single_stage import SingleStageRanker
from cascadelens.models.tuning import Choice
from cascadelens.results import git_commit, write_result
from cascadelens.splits.report import assert_no_group_overlap
from cascadelens.splits.splitters import Split

METHODS: dict[str, type[TwoStageRanker]] = {
    method.name: method for method in (TwoStageRanker, SingleStageRanker)
}
FAMILIES = ("group_run", "lofo", "loso")
RUNS_DIR = Path("experiments/ranker_clean/runs")


def load_ranker_config(path: Path, root: Path = Path()) -> RankerConfig:
    """Read a ranker config, relative to ``root`` when not absolute."""
    target = path if path.is_absolute() else root / path
    if not target.is_file():
        raise RankerError(f"No ranker config at {target}")
    return load_config(target, RankerConfig)


def as_baseline_config(method: str, config: RankerConfig) -> BaselineConfig:
    """Wrap a ranker config so the runner can carry it."""
    if method not in METHODS:
        raise RankerError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
    return BaselineConfig(
        name=method, supervised=True, seeds=config.seeds, params=config.model_dump(mode="json")
    )


def feature_settings(config: RankerConfig, root: Path = Path()) -> FeaturesConfig:
    """Read the feature config the ranker names."""
    return load_config(root / config.features_config, FeaturesConfig)


def make_ranker(
    method: str,
    config: RankerConfig,
    dataset: Dataset,
    group_map: dict[str, str],
    frozen: dict[str, Choice] | None,
    root: Path = Path(),
) -> TwoStageRanker:
    """Build a ranker over the Part 6 store of this dataset."""
    if method not in METHODS:
        raise RankerError(f"Unknown method {method!r}; choose from {sorted(METHODS)}")
    source = FeatureSource(feature_settings(config, root), root)
    source.verify(dataset.checksum)
    return METHODS[method](source, group_map, frozen)


def _train_val(split: Split, index: int, dataset: Dataset) -> tuple[list[CaseHandle], ...]:
    fold = split.folds[index]
    return dataset.handles(fold.train_ids), dataset.handles(fold.val_ids)


def _check_fold(split: Split, index: int, group_map: dict[str, str]) -> None:
    fold = split.folds[index]
    if split.name in GROUPED:
        assert_no_group_overlap(fold.train_ids, fold.val_ids, group_map)
        assert_no_group_overlap(fold.val_ids, fold.test_ids, group_map)


def train_folds(
    config_path: Path, split_name: str, seed: int, method: str, root: Path = Path()
) -> list[tuple[Path, FittedFold]]:
    """Train and save one model per fold. No test case is read."""
    config = load_ranker_config(config_path, root)
    wrapped = as_baseline_config(method, config)
    dataset = load_dataset(root)
    split, group_map = load_split(split_name, dataset, root)
    ranker = make_ranker(method, config, dataset, group_map, None, root)
    features = feature_settings(config, root)
    saved = []
    for index in range(len(split.folds)):
        _check_fold(split, index, group_map)
        train, val = _train_val(split, index, dataset)
        ranker.fit(train, wrapped, val, seed)
        fold = ranker.folds[-1]
        provenance = Provenance(
            method,
            split_name,
            seed,
            index,
            config_hash(config),
            config_hash(features),
            dataset.checksum,
            features.model_dump(mode="json"),
        )
        saved.append((save_artifact(fold, provenance, root), fold))
    return saved


def freeze_config(config_path: Path, name: str, root: Path = Path()) -> Path:
    """Search every fold of every split family on validation, and record the choices."""
    config = load_ranker_config(config_path, root)
    dataset = load_dataset(root)
    families: dict[str, dict[str, list[dict[str, object]]]] = {}
    for split_name in FAMILIES:
        split, group_map = load_split(split_name, dataset, root)
        families[split_name] = {}
        for method in METHODS:
            ranker = make_ranker(method, config, dataset, group_map, None, root)
            entries = []
            for index in range(len(split.folds)):
                _check_fold(split, index, group_map)
                train, val = _train_val(split, index, dataset)
                entries.append({"fold": index, **ranker.search_fold(train, val, config).to_dict()})
            families[split_name][method] = entries
    record = FrozenRecord(
        name=name,
        config_hash=config_hash(config),
        feature_config_hash=config_hash(feature_settings(config, root)),
        data_checksum=dataset.checksum,
        git_commit=git_commit(),
        families=families,
    )
    return write_frozen(record, root)


@dataclass(frozen=True)
class RankerRun:
    """One method on one split and seed, on test cases."""

    run: SplitRun
    folds: list[FittedFold]
    status: str


def stage1_table(folds: list[FittedFold]) -> pl.DataFrame:
    """Return every test case's Stage 1 scores with its fold, empty for the single-stage model."""
    tables = [
        table.with_columns(fold=pl.lit(index))
        for index, fold in enumerate(folds)
        for table in fold.stage1
    ]
    if tables:
        return pl.concat(tables)
    return pl.DataFrame(
        schema={"case_id": pl.String, "service": pl.String, "score": pl.Float64, "fold": pl.Int32}
    )


def fold_details(
    folds: list[FittedFold],
    outcomes: pl.DataFrame,
    truth: pl.DataFrame,
    k_choices: list[int],
) -> list[dict[str, object]]:
    """Return per-fold retrieval recall and the validation-to-test gap."""
    rows: list[dict[str, object]] = []
    for index, fold in enumerate(folds):
        tested = outcomes.filter(pl.col("fold") == index)
        test_top1 = float(tested["hit@1"].mean())  # type: ignore[arg-type]
        row: dict[str, object] = {
            "fold": index,
            "k": fold.choice.k,
            "params": fold.choice.params,
            "best_iteration": fold.booster.best_iteration,
            "search_val_top1": fold.choice.val_top1,
            "val_top1": fold.val_top1,
            "test_top1": test_top1,
            "val_test_gap": fold.val_top1 - test_top1,
        }
        if fold.stage1:
            stage1 = pl.concat(fold.stage1)
            labels = truth.filter(pl.col("case_id").is_in(tested["case_id"].to_list()))
            for k in sorted({*k_choices, *([fold.choice.k] if fold.choice.k else [])}):
                row[f"recall@{k}"] = retrieval_recall_at_k(stage1, labels, k).mean
            assert fold.choice.k is not None
            row["recall_at_chosen_k"] = row[f"recall@{fold.choice.k}"]
        rows.append(row)
    return rows


def evaluate(
    config_path: Path,
    split_name: str,
    seed: int,
    method: str,
    exploratory: bool,
    root: Path = Path(),
    ranker: TwoStageRanker | None = None,
) -> RankerRun:
    """Report test results through the runner, once the freeze check allows it."""
    config = load_ranker_config(config_path, root)
    wrapped = as_baseline_config(method, config)
    status, record = resolve(config, (split_name,), (method,), exploratory, root)
    dataset = load_dataset(root)
    split, group_map = load_split(split_name, dataset, root)
    model = ranker
    if model is None:
        model = make_ranker(
            method, config, dataset, group_map, frozen_choices(record, split_name, method), root
        )
    run = run_folds(model, wrapped, split, group_map, dataset, seed)
    folds = model.take_folds()
    truth = truth_table(dataset.records, group_map)
    details = fold_details(folds, run.outcomes, truth, config.retrieval.top_k_choices)
    metrics: dict[str, object] = {**run.metrics, "status": status, "fold_details": details}
    directory = root / RUNS_DIR / method / split_name / f"seed_{seed}"
    (directory / "cache").mkdir(parents=True, exist_ok=True)
    run.predictions.write_parquet(directory / "cache" / "predictions.parquet")
    write_result(directory / "result.json", metrics, config, seed, dataset.checksum)
    return RankerRun(SplitRun(run.predictions, run.outcomes, metrics), folds, status)
