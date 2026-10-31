"""Run a baseline over every fold of a saved split and record the result.

Layout of a run under ``experiments/runs/<name>/<split>/seed_<seed>/``::

    result.json                 metrics, with config hash, git commit, data checksum and seed
    cache/predictions.parquet   every prediction, not committed
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cascadelens.baselines.base import Baseline, BaselineError, CaseHandle, SummaryCache
from cascadelens.baselines.frequency import FrequencyPrior
from cascadelens.baselines.lightgbm_flat import LightGBMFlat
from cascadelens.baselines.rule import RuleOnset, RuleUpstream
from cascadelens.config import BaselineConfig, load_config
from cascadelens.data.errors import DataError
from cascadelens.data.io import CASES_FILE, read_index, read_manifest
from cascadelens.data.schema import CaseRecord
from cascadelens.eval.metrics import missing_truth_count, per_case_outcomes
from cascadelens.results import write_result
from cascadelens.seed import set_seed
from cascadelens.splits.report import assert_no_group_overlap
from cascadelens.splits.splitters import Split
from cascadelens.splits.store import SPLIT_FILE, read_split

BASELINES: dict[str, type[Baseline]] = {
    method.name: method for method in (FrequencyPrior, RuleOnset, RuleUpstream, LightGBMFlat)
}
PROCESSED_DIR = Path("data/processed")
SPLITS_DIR = Path("data/splits")
CONFIG_DIR = Path("configs/baselines")
RUNS_DIR = Path("experiments/runs")
GROUPED = ("group_run", "lofo", "loso")


@dataclass(frozen=True)
class Dataset:
    """The case index and its checksum, shared by every method."""

    processed_dir: Path
    records: dict[str, CaseRecord]
    checksum: str

    def handles(self, ids: tuple[str, ...]) -> list[CaseHandle]:
        """Return handles for ``ids``, in that order."""
        return [CaseHandle.stored(self.records[i], self.processed_dir) for i in ids]


@dataclass(frozen=True)
class SplitRun:
    """The outputs of one baseline on one split and seed."""

    predictions: pl.DataFrame
    outcomes: pl.DataFrame
    metrics: dict[str, object]


def load_dataset(root: Path = Path()) -> Dataset:
    """Read the processed index and manifest."""
    processed = root / PROCESSED_DIR
    if not (processed / CASES_FILE).is_file():
        raise BaselineError(f"No cases index in {processed}; run `cascadelens data build` first")
    try:
        checksum = read_manifest(processed).digest()
    except (DataError, FileNotFoundError) as error:
        raise BaselineError(f"Cannot read the manifest in {processed}: {error}") from error
    return Dataset(processed, {r.case_id: r for r in read_index(processed)}, checksum)


def load_split(name: str, dataset: Dataset, root: Path = Path()) -> tuple[Split, dict[str, str]]:
    """Read a saved split, refusing one built from another dataset version."""
    directory = root / SPLITS_DIR / name
    if not (directory / SPLIT_FILE).is_file():
        raise BaselineError(f"No split in {directory}; run `cascadelens splits build` first")
    saved, group_map = read_split(directory)
    if saved.data_manifest_checksum != dataset.checksum:
        raise BaselineError(f"Split {name} was built from a different dataset version")
    return saved.split, group_map


def load_baseline_config(name: str, root: Path = Path()) -> BaselineConfig:
    """Read the config of a known baseline."""
    if name not in BASELINES:
        raise BaselineError(f"Unknown baseline {name!r}; choose from {sorted(BASELINES)}")
    path = root / CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise BaselineError(f"No config for {name} at {path}")
    return load_config(path, BaselineConfig)


def truth_table(records: dict[str, CaseRecord], group_map: dict[str, str]) -> pl.DataFrame:
    """Return ground truth with ``case_id, root_cause_service, group_id``."""
    ordered = sorted(records)
    return pl.DataFrame(
        {
            "case_id": ordered,
            "root_cause_service": [records[case].root_cause_service for case in ordered],
            "group_id": [group_map[case] for case in ordered],
        }
    )


def _summarize(
    outcomes: pl.DataFrame, predictions: pl.DataFrame, split: Split, config: BaselineConfig
) -> dict[str, object]:
    def means(frame: pl.DataFrame) -> dict[str, float]:
        return {
            "top1": float(frame["hit@1"].mean()),  # type: ignore[arg-type]
            "top3": float(frame["hit@3"].mean()),  # type: ignore[arg-type]
            "avg5": float(frame["avg@5"].mean()),  # type: ignore[arg-type]
        }

    return {
        "split": split.name,
        "leaky_split": split.leaky,
        "method": config.name,
        "n_cases": outcomes.height,
        "n_predictions": predictions.height,
        "missing_truth": missing_truth_count(outcomes),
        **means(outcomes),
        "folds": [
            {
                "fold": index,
                "held_out": fold.held_out,
                "n_cases": len(fold.test_ids),
                **means(outcomes.filter(pl.col("fold") == index)),
            }
            for index, fold in enumerate(split.folds)
        ],
    }


def run_folds(
    baseline: Baseline,
    config: BaselineConfig,
    split: Split,
    group_map: dict[str, str],
    dataset: Dataset,
    seed: int,
) -> SplitRun:
    """Fit on each fold's training cases and predict its test cases only."""
    set_seed(seed)
    tables: list[pl.DataFrame] = []
    frames: list[pl.DataFrame] = []
    truth = truth_table(dataset.records, group_map)
    for index, fold in enumerate(split.folds):
        if split.name in GROUPED:
            assert_no_group_overlap(fold.train_ids, fold.test_ids, group_map)
            assert_no_group_overlap(fold.val_ids, fold.test_ids, group_map)
        baseline.fit(dataset.handles(fold.train_ids), config, dataset.handles(fold.val_ids), seed)
        predictions = pl.concat(baseline.predict(c) for c in dataset.handles(fold.test_ids))
        tested = truth.filter(pl.col("case_id").is_in(fold.test_ids))
        tables.append(predictions.with_columns(fold=pl.lit(index)))
        frames.append(per_case_outcomes(predictions, tested).with_columns(fold=pl.lit(index)))
    all_predictions = pl.concat(tables)
    outcomes = pl.concat(frames).sort("case_id")
    return SplitRun(all_predictions, outcomes, _summarize(outcomes, all_predictions, split, config))


def run_baseline(
    name: str,
    split_name: str,
    seed: int,
    root: Path = Path(),
    cache: SummaryCache | None = None,
    baseline: Baseline | None = None,
) -> SplitRun:
    """Run one baseline on one saved split, and write its predictions and result JSON."""
    config = load_baseline_config(name, root)
    dataset = load_dataset(root)
    split, group_map = load_split(split_name, dataset, root)
    method = baseline if baseline is not None else BASELINES[name](cache)
    run = run_folds(method, config, split, group_map, dataset, seed)
    directory = root / RUNS_DIR / name / split_name / f"seed_{seed}"
    (directory / "cache").mkdir(parents=True, exist_ok=True)
    run.predictions.write_parquet(directory / "cache" / "predictions.parquet")
    write_result(directory / "result.json", run.metrics, config, seed, dataset.checksum)
    return run
