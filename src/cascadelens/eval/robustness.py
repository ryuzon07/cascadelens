"""The perturbation sweep (PRD 9.1 and 9.3).

Three stages, each resumable:

``derive``  Feature tables and service summaries of every perturbed case, cached by content hash.
``train``   One unit per (variant, split): fit each fold and seed, then score the fold's test cases
            at every grid point. A unit's outcomes file is named by a hash of its inputs, so a
            finished unit is skipped.
``report``  Tables, curves, S1 to S3 and ``docs/robustness_summary.md``, from saved outcomes.

Perturbations reach test cases only. Training and validation use clean cases, plus masked copies in
training for augmented variants. P4 and P5 keep data outside the new window, and feature code does
not crop, so window points are cropped here before any feature is computed.
"""

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import polars as pl

from cascadelens.baselines.base import Baseline, CaseHandle, SummaryCache
from cascadelens.baselines.frequency import FrequencyPrior
from cascadelens.baselines.lightgbm_flat import LightGBMFlat
from cascadelens.baselines.published.baro import Baro
from cascadelens.baselines.rule import RuleOnset, RuleUpstream
from cascadelens.baselines.runner import (
    GROUPED,
    Dataset,
    load_baseline_config,
    load_dataset,
    load_split,
    truth_table,
)
from cascadelens.config import (
    BaselineConfig,
    RankerConfig,
    StrictModel,
    canonical_json,
    config_hash,
    load_config,
)
from cascadelens.data.schema import Case, CaseRecord
from cascadelens.eval.bootstrap import (
    diff_of_diffs_bootstrap,
    group_bootstrap_ci,
    paired_group_bootstrap,
)
from cascadelens.eval.metrics import per_case_outcomes
from cascadelens.eval.plots import CurveSeries, robustness_curves
from cascadelens.eval.significance import label_comparisons, wilcoxon_signed_rank
from cascadelens.features.config import FeaturesConfig, GroupName
from cascadelens.features.context import FeatureInput
from cascadelens.features.registry import build_case_features
from cascadelens.features.simple import SignalSettings, service_summary
from cascadelens.models.augment import (
    AugmentConfig,
    AugmentedFlat,
    AugmentedRanker,
    AugmentedSingleStage,
    AugmentedSource,
    AugmentStore,
)
from cascadelens.models.data import FeatureSource, RankerError
from cascadelens.models.freeze import (
    EXPLORATORY,
    FROZEN,
    FrozenRecord,
    find_frozen,
    frozen_choices,
    write_frozen,
)
from cascadelens.models.pipeline import METHODS, feature_settings, load_ranker_config
from cascadelens.models.ranker import FittedFold, TwoStageRanker
from cascadelens.models.single_stage import SingleStageRanker
from cascadelens.models.tuning import Choice
from cascadelens.perturb.base import Perturbation, Stage, crop_to_window
from cascadelens.perturb.registry import REFERENCES, build, grid, load_suite
from cascadelens.perturb.store import PerturbedStore
from cascadelens.results import git_commit, write_result
from cascadelens.splits.report import assert_no_group_overlap
from cascadelens.splits.splitters import Fold

EXPERIMENT_DIR = Path("experiments/robustness")
DERIVED_DIR = Path("data/robustness")
SUMMARY_PATH = Path("docs/robustness_summary.md")
CLEAN = "clean"
FAMILIES = ("group_run", "lofo", "loso")
METRICS = (("hit@1", "Top-1"), ("hit@3", "Top-3"), ("avg@5", "Avg@5"))
ENGINE_VERSION = 1


class Variant(StrictModel):
    """One trained or fixed method of the sweep."""

    name: str
    kind: Literal["two_stage", "single_stage", "flat", "frequency", "rule", "rule_upstream", "baro"]
    augment: bool = False
    drop_group: GroupName | None = None
    points: Literal["all", "s3"] = "all"


class CurveMethods(StrictModel):
    """Methods drawn on the curves, one panel per kind."""

    supervised: list[str]
    unsupervised: list[str]


class S3Point(StrictModel):
    """The grid point of S3."""

    id: str
    rate: float


class RobustnessConfig(StrictModel):
    """Settings of the sweep."""

    seed: int = 0
    n_resamples: int = 10_000
    alpha: float = 0.05
    s2_margin: float = 0.12
    workers: int = 6
    ranker_config: str = "configs/ranker.yaml"
    augment_config: str = "configs/augment.yaml"
    ranker: str
    plain_ranker: str
    prior: str
    primary: list[str]
    s3_point: S3Point
    s3_baselines: list[str]
    curve_methods: CurveMethods
    variants: list[Variant]

    def variant(self, name: str) -> Variant:
        """Return a variant by name."""
        for item in self.variants:
            if item.name == name:
                return item
        raise RankerError(f"Unknown variant {name!r}")


@dataclass(frozen=True)
class Point:
    """One grid point: a perturbation with parameters, or the clean data."""

    key: str
    identifier: str
    params: dict[str, Any]
    realism: str

    def steps(self) -> list[Perturbation[Any]]:
        """Return the perturbation, or nothing for clean data."""
        return [] if self.identifier == CLEAN else [build(self.identifier, self.params)]


def point_key(identifier: str, params: Mapping[str, Any]) -> str:
    """Return a readable key of a grid point."""
    text = ",".join(f"{key}={params[key]}" for key in sorted(params))
    return f"{identifier}:{text}"


CLEAN_POINT = Point(CLEAN, CLEAN, {}, "clean")


def suite_points(root: Path = Path()) -> list[Point]:
    """Return clean data and every transform of the registry."""
    points = [CLEAN_POINT]
    for spec, params in grid(load_suite(root)):
        points.append(Point(point_key(spec.id, params), spec.id, params, spec.realism))
    return points


def s3_key(config: RobustnessConfig, points: Sequence[Point]) -> str:
    """Return the key of the S3 grid point."""
    for point in points:
        if (
            point.identifier == config.s3_point.id
            and point.params.get("rate") == config.s3_point.rate
        ):
            return point.key
    raise RankerError("The suite has no S3 grid point")


class Derived:
    """Feature tables and summaries of a grid point, on disk."""

    def __init__(
        self,
        point: Point,
        features: FeaturesConfig,
        seed: int,
        checksum: str,
        root: Path = Path(),
    ) -> None:
        self.point = point
        self.features = features
        self.store = PerturbedStore(point.steps(), seed, checksum, root)
        name = f"{self.store.directory.name}_{config_hash(features)[:16]}"
        self.directory = root / DERIVED_DIR / name

    def path(self, kind: str, case_id: str, suffix: str = "parquet") -> Path:
        """Return where one file lives."""
        return self.directory / kind / f"{case_id}.{suffix}"

    def case(self, record: CaseRecord) -> Case:
        """Return the perturbed case, cropped to its window for window perturbations."""
        case = self.store.case(record)
        window = any(step.stage == Stage.WINDOW for step in self.store.pipeline.steps)
        return crop_to_window(case) if window else case

    def ensure(self, record: CaseRecord) -> None:
        """Compute the tables of one case if they are missing."""
        target = self.path("records", record.case_id, "json")
        if target.is_file():
            return
        case = self.case(record)
        view = FeatureInput.from_record(case.record(), case.metrics)
        _write_parquet(
            build_case_features(view, self.features), self.path("features", case.case_id)
        )
        summary = service_summary(
            case.metrics, case.window_start, case.window_end, SignalSettings()
        )
        _write_parquet(summary, self.path("summaries", case.case_id))
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(f".{os.getpid()}.partial")
        partial.write_text(case.record().model_dump_json(), encoding="utf-8")
        os.replace(partial, target)

    def table(self, case_id: str) -> pl.DataFrame:
        """Return a stored feature table."""
        return pl.read_parquet(self.path("features", case_id))

    def summary(self, case_id: str) -> pl.DataFrame:
        """Return a stored service summary."""
        return pl.read_parquet(self.path("summaries", case_id))

    def record(self, case_id: str) -> CaseRecord:
        """Return a stored perturbed record."""
        text = self.path("records", case_id, "json").read_text(encoding="utf-8")
        return CaseRecord.model_validate_json(text)


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f".{os.getpid()}.partial")
    frame.write_parquet(partial)
    os.replace(partial, path)


def _derive_one(job: tuple[Point, FeaturesConfig, int, str, Path, CaseRecord]) -> None:
    """Derive one case. Runs in a worker process, so it takes plain values."""
    point, features, seed, checksum, root, record = job
    Derived(point, features, seed, checksum, root).ensure(record)


def derive_points(
    points: Sequence[Point],
    features: FeaturesConfig,
    seed: int,
    dataset: Dataset,
    root: Path = Path(),
    workers: int = 1,
) -> None:
    """Build every missing table. Run it again to resume."""
    records = list(dataset.records.values())
    for point in points:
        derived = Derived(point, features, seed, dataset.checksum, root)
        derived.store.write_spec()
        jobs = [(point, features, seed, dataset.checksum, root, record) for record in records]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_derive_one, jobs, chunksize=4))
        else:
            for job in jobs:
                _derive_one(job)


class StoredSummaries(SummaryCache):
    """Service summaries of a grid point, read from disk."""

    def __init__(self, derived: Derived) -> None:
        super().__init__()
        self.derived = derived

    def get(self, case: CaseHandle, settings: SignalSettings) -> pl.DataFrame:
        """Return the stored summary. Every baseline here uses the default settings."""
        if settings != SignalSettings():
            raise RankerError("Stored summaries use the default signal settings")
        return self.derived.summary(case.record.case_id)


def _unloaded() -> pl.DataFrame:
    raise RankerError("This handle has no metrics: test scoring reads derived tables")


class Arm(Protocol):
    """A method that can be fitted on a fold and scored on perturbed test cases."""

    seeds: list[int]

    def fit(self, fold: Fold, seed: int) -> None:
        """Fit on the clean training cases of ``fold``."""
        ...

    def score(self, derived: Derived, case_ids: Sequence[str]) -> pl.DataFrame:
        """Score test cases at a grid point."""
        ...


class RankerArm:
    """A ranker scored on stored feature tables."""

    def __init__(
        self,
        ranker: TwoStageRanker,
        config: BaselineConfig,
        dataset: Dataset,
        tables: dict[tuple[str, str], pl.DataFrame],
    ) -> None:
        self.ranker = ranker
        self.config = config
        self.dataset = dataset
        self.seeds = config.seeds
        self.tables = tables
        self.fold: FittedFold | None = None

    def fit(self, fold: Fold, seed: int) -> None:
        """Fit through the Part 7 interface."""
        train = self.dataset.handles(fold.train_ids)
        self.ranker.fit(train, self.config, self.dataset.handles(fold.val_ids), seed)
        self.fold = self.ranker.take_folds()[0]

    def score(self, derived: Derived, case_ids: Sequence[str]) -> pl.DataFrame:
        """Rank each case from its stored table."""
        assert self.fold is not None
        frames = []
        for case_id in case_ids:
            key = (derived.point.key, case_id)
            if key not in self.tables:
                self.tables[key] = derived.table(case_id)
            frames.append(self.fold.rank(self.tables[key])[0])
        return pl.concat(frames)


class BaselineArm:
    """A Part 4 baseline scored on stored summaries."""

    def __init__(self, baseline: Baseline, config: BaselineConfig, dataset: Dataset) -> None:
        self.baseline = baseline
        self.config = config
        self.dataset = dataset
        self.seeds = config.seeds
        self.training = baseline.cache

    def fit(self, fold: Fold, seed: int) -> None:
        """Fit on clean cases, with the in-memory cache of training summaries."""
        self.baseline.cache = self.training
        train = self.dataset.handles(fold.train_ids)
        self.baseline.fit(train, self.config, self.dataset.handles(fold.val_ids), seed)

    def score(self, derived: Derived, case_ids: Sequence[str]) -> pl.DataFrame:
        """Predict each case from its stored summary and perturbed record."""
        self.baseline.cache = StoredSummaries(derived)
        handles = [CaseHandle(derived.record(case_id), _unloaded) for case_id in case_ids]
        return pl.concat(self.baseline.predict(handle) for handle in handles)


class FrequencyArm:
    """The frequency prior. It reads no telemetry, so every grid point gives the same scores."""

    def __init__(self, config: BaselineConfig, dataset: Dataset) -> None:
        self.prior = FrequencyPrior()
        self.config = config
        self.dataset = dataset
        self.seeds = config.seeds
        self.scores: pl.DataFrame | None = None

    def fit(self, fold: Fold, seed: int) -> None:
        """Count training labels."""
        self.prior.fit(self.dataset.handles(fold.train_ids), self.config, (), seed)
        self.scores = None

    def score(self, derived: Derived, case_ids: Sequence[str]) -> pl.DataFrame:
        """Score once per fold."""
        if self.scores is None:
            handles = self.dataset.handles(tuple(case_ids))
            self.scores = pl.concat(self.prior.predict(handle) for handle in handles)
        return self.scores


class BaroArm:
    """BARO on perturbed cases. A new instance per point, because it memoizes by case id."""

    def __init__(self, config: BaselineConfig, dataset: Dataset) -> None:
        self.config = config
        self.dataset = dataset
        self.seeds = config.seeds
        self.failures = 0

    def fit(self, fold: Fold, seed: int) -> None:
        """Fit is a no-op: the method is unsupervised."""

    def score(self, derived: Derived, case_ids: Sequence[str]) -> pl.DataFrame:
        """Score each case from its metrics."""
        method = Baro()
        method.fit((), self.config, (), 0)
        frames = []
        for case_id in case_ids:
            case = derived.case(self.dataset.records[case_id])
            frames.append(method.predict(CaseHandle.from_case(case)))
        self.failures += len(method.failures)
        return pl.concat(frames)


@dataclass
class Context:
    """What the stages share."""

    config: RobustnessConfig
    ranker: RankerConfig
    augment: AugmentConfig
    features: FeaturesConfig
    dataset: Dataset
    points: list[Point]
    root: Path
    status: str
    record: FrozenRecord | None
    tables: dict[tuple[str, str], pl.DataFrame] = field(default_factory=dict)

    @property
    def perturb_seed(self) -> int:
        """Return the suite seed."""
        return load_suite(self.root).seed

    def derived(self, point: Point) -> Derived:
        """Return the tables of a grid point."""
        return Derived(point, self.features, self.perturb_seed, self.dataset.checksum, self.root)

    def augment_store(self) -> AugmentStore:
        """Return the store of masked copies."""
        return AugmentStore(self.augment, self.features, self.dataset.checksum, self.root)


def load_context(exploratory: bool, root: Path = Path()) -> Context:
    """Read every config, and refuse a config that is neither frozen nor exploratory."""
    config = load_config(root / EXPERIMENT_DIR / "config.yaml", RobustnessConfig)
    ranker = load_ranker_config(Path(config.ranker_config), root)
    record = None if exploratory else find_frozen(ranker, root)
    if not exploratory and record is None:
        raise RankerError(
            f"Config {config_hash(ranker)[:12]} is not frozen; run `cascadelens freeze` or pass "
            "--exploratory"
        )
    return Context(
        config,
        ranker,
        load_config(root / config.augment_config, AugmentConfig),
        feature_settings(ranker, root),
        load_dataset(root),
        suite_points(root),
        root,
        EXPLORATORY if exploratory else FROZEN,
        record,
    )


def variant_settings(variant: Variant, base: RankerConfig) -> RankerConfig:
    """Return the ranker config of a variant, with its feature group removed."""
    if variant.drop_group is None:
        return base
    kept = [group for group in base.groups if group != variant.drop_group]
    return RankerConfig.model_validate({**base.model_dump(mode="json"), "groups": kept})


def make_ranker(
    variant: Variant,
    context: Context,
    groups: dict[str, str],
    frozen: dict[str, Choice] | None,
) -> TwoStageRanker:
    """Build the ranker of a variant."""
    source = FeatureSource(context.features, context.root)
    source.verify(context.dataset.checksum)
    if not variant.augment:
        kind = SingleStageRanker if variant.kind == "single_stage" else TwoStageRanker
        return kind(source, groups, frozen)
    store = context.augment_store()
    augmented = AugmentedSingleStage if variant.kind == "single_stage" else AugmentedRanker
    return augmented(AugmentedSource(source, store), store, groups, frozen)


def _wrap(variant: Variant, settings: RankerConfig) -> BaselineConfig:
    return BaselineConfig(
        name=variant.name,
        supervised=True,
        seeds=settings.seeds,
        params=settings.model_dump(mode="json"),
    )


def build_arm(variant: Variant, split: str, group_map: dict[str, str], context: Context) -> Arm:
    """Return the arm of a variant on a split."""
    dataset, root = context.dataset, context.root
    if variant.kind in ("two_stage", "single_stage"):
        settings = variant_settings(variant, context.ranker)
        frozen = None
        if context.record is not None:
            frozen = frozen_choices(context.record, split, variant.name)
        ranker = make_ranker(variant, context, group_map, frozen)
        return RankerArm(ranker, _wrap(variant, settings), dataset, context.tables)
    if variant.kind == "flat":
        config = load_baseline_config("lightgbm_flat", root)
        method: Baseline = (
            AugmentedFlat(context.augment_store()) if variant.augment else LightGBMFlat()
        )
        return BaselineArm(method, config, dataset)
    if variant.kind == "frequency":
        return FrequencyArm(load_baseline_config("frequency", root), dataset)
    if variant.kind == "baro":
        return BaroArm(load_config(root / "configs/baselines/baro.yaml", BaselineConfig), dataset)
    rule = RuleOnset() if variant.kind == "rule" else RuleUpstream()
    return BaselineArm(rule, load_baseline_config(variant.kind, root), dataset)


def unit_points(variant: Variant, split: str, context: Context) -> list[Point]:
    """Return the points a unit scores. LOFO and LOSO are P9 and P10, so clean data only."""
    if split != "group_run":
        return [CLEAN_POINT]
    if variant.points == "s3":
        wanted = {CLEAN, s3_key(context.config, context.points)}
        return [point for point in context.points if point.key in wanted]
    return context.points


def unit_path(variant: Variant, split: str, context: Context) -> Path:
    """Return the outcomes file of a unit, named by a hash of everything that shapes it."""
    payload = canonical_json(
        {
            "augment": context.augment.model_dump(mode="json"),
            "data": context.dataset.checksum,
            "engine": ENGINE_VERSION,
            "features": config_hash(context.features),
            "points": [point.key for point in unit_points(variant, split, context)],
            "ranker": config_hash(context.ranker),
            "split": split,
            "status": context.status,
            "variant": variant.model_dump(mode="json"),
        }
    )
    name = hashlib.sha256(payload.encode()).hexdigest()[:24]
    root = context.root / EXPERIMENT_DIR / "cache" / "outcomes"
    return root / variant.name / split / f"{name}.parquet"


def assert_test_only(fold: Fold, case_ids: Sequence[str]) -> None:
    """Fail unless the scored cases are test cases of ``fold``."""
    stray = set(case_ids) - set(fold.test_ids)
    if stray or set(fold.test_ids) & (set(fold.train_ids) | set(fold.val_ids)):
        raise RankerError("The evaluation path may perturb test cases only")


def run_unit(variant: Variant, split_name: str, context: Context) -> Path:
    """Fit one variant on one split and score every point, unless that is already done."""
    path = unit_path(variant, split_name, context)
    if path.is_file():
        return path
    split, group_map = load_split(split_name, context.dataset, context.root)
    truth = truth_table(context.dataset.records, group_map)
    arm = build_arm(variant, split_name, group_map, context)
    points = unit_points(variant, split_name, context)
    frames = []
    for index, fold in enumerate(split.folds):
        if split_name in GROUPED:
            assert_no_group_overlap(fold.train_ids, fold.test_ids, group_map)
            assert_no_group_overlap(fold.val_ids, fold.test_ids, group_map)
        tested = truth.filter(pl.col("case_id").is_in(fold.test_ids))
        context.tables.clear()
        for seed in arm.seeds:
            arm.fit(fold, seed)
            for point in points:
                assert_test_only(fold, fold.test_ids)
                predictions = arm.score(context.derived(point), fold.test_ids)
                frames.append(
                    per_case_outcomes(predictions, tested).with_columns(
                        method=pl.lit(variant.name),
                        split=pl.lit(split_name),
                        point=pl.lit(point.key),
                        fold=pl.lit(index),
                        seed=pl.lit(seed),
                    )
                )
    context.tables.clear()
    _write_parquet(pl.concat(frames), path)
    return path


def derive_stage(context: Context) -> None:
    """Build masked copies and every perturbed table."""
    records = list(context.dataset.records.values())
    context.augment_store().build(records, context.config.workers)
    derive_points(
        context.points,
        context.features,
        context.perturb_seed,
        context.dataset,
        context.root,
        context.config.workers,
    )


def train_stage(context: Context, names: Sequence[str] | None = None) -> list[Path]:
    """Run every unit that is not finished."""
    chosen = list(names) if names else [v.name for v in context.config.variants]
    paths = []
    for name in chosen:
        for split in FAMILIES:
            variant = context.config.variant(name)
            if variant.points == "s3" and split != "group_run":
                continue
            paths.append(run_unit(variant, split, context))
    return paths


def load_outcomes(context: Context) -> pl.DataFrame:
    """Read every finished unit of the current inputs."""
    frames = []
    for variant in context.config.variants:
        for split in FAMILIES:
            if variant.points == "s3" and split != "group_run":
                continue
            path = unit_path(variant, split, context)
            if not path.is_file():
                raise RankerError(f"Unit {variant.name} on {split} has not run")
            frames.append(pl.read_parquet(path))
    return pl.concat(frames)


def seed_averaged(outcomes: pl.DataFrame, column: str) -> pl.DataFrame:
    """Return one row per case, averaged over seeds and sorted by case."""
    return (
        outcomes.group_by("case_id", "group_id")
        .agg(pl.col(column).cast(pl.Float64).mean().alias("value"))
        .sort("case_id")
    )


def _select(frame: pl.DataFrame, method: str, split: str, point: str) -> pl.DataFrame:
    return frame.filter(
        (pl.col("method") == method) & (pl.col("split") == split) & (pl.col("point") == point)
    )


def aligned(arms: Iterable[pl.DataFrame], column: str = "hit@1") -> list[pl.DataFrame]:
    """Seed-average each arm, and fail unless all cover the same cases in the same order."""
    averaged = [seed_averaged(arm, column) for arm in arms]
    first = averaged[0]["case_id"].to_list()
    if not first or any(item["case_id"].to_list() != first for item in averaged[1:]):
        raise RankerError("Compared arms must cover the same cases")
    return averaged


def paired(
    first: pl.DataFrame,
    second: pl.DataFrame,
    config: RobustnessConfig,
    margin: float = 0.0,
    column: str = "hit@1",
) -> dict[str, float]:
    """Return ``first + margin - second`` with the paired bootstrap and the Wilcoxon check."""
    a, b = aligned([first, second], column)
    shifted = a["value"] + margin
    result = paired_group_bootstrap(
        shifted, b["value"], a["group_id"], config.n_resamples, config.seed, config.alpha
    )
    check = wilcoxon_signed_rank(shifted, b["value"], a["group_id"])
    return {
        "estimate": result.estimate,
        "lower": result.lower,
        "upper": result.upper,
        "p_value": result.p_value,
        "wilcoxon_p": check.p_value,
    }


def difference_of_differences(
    ranker_clean: pl.DataFrame,
    baseline_clean: pl.DataFrame,
    ranker_lossy: pl.DataFrame,
    baseline_lossy: pl.DataFrame,
    config: RobustnessConfig,
) -> dict[str, float]:
    """Return S3: the ranker's margin at the loss point minus its margin on clean data."""
    a, b, c, d = aligned([ranker_lossy, baseline_lossy, ranker_clean, baseline_clean])
    result = diff_of_diffs_bootstrap(
        a["value"],
        b["value"],
        c["value"],
        d["value"],
        a["group_id"],
        config.n_resamples,
        config.seed,
        config.alpha,
    )
    return {
        "estimate": result.estimate,
        "lower": result.lower,
        "upper": result.upper,
        "p_value": result.p_value,
    }


def scores(outcomes: pl.DataFrame, config: RobustnessConfig) -> dict[str, dict[str, float]]:
    """Return each metric with its group-bootstrap interval, over seed-averaged cases."""
    result = {}
    for column, label in METRICS:
        averaged = seed_averaged(outcomes, column)
        interval = group_bootstrap_ci(
            averaged["value"],
            averaged["group_id"],
            config.n_resamples,
            config.seed,
            config.alpha,
        )
        result[label] = {
            "estimate": interval.estimate,
            "lower": interval.lower,
            "upper": interval.upper,
        }
    return result


def grid_rows(frame: pl.DataFrame, context: Context) -> list[dict[str, Any]]:
    """Return every method at every group_run point, with its drop from clean."""
    rows = []
    base = frame.filter(pl.col("split") == "group_run")
    for variant in context.config.variants:
        clean = scores(_select(base, variant.name, "group_run", CLEAN), context.config)
        for point in unit_points(variant, "group_run", context):
            found = scores(_select(base, variant.name, "group_run", point.key), context.config)
            before, after = clean["Top-1"]["estimate"], found["Top-1"]["estimate"]
            rows.append(
                {
                    "method": variant.name,
                    "point": point.key,
                    "perturbation": point.identifier,
                    "realism": point.realism,
                    **found,
                    "relative_drop": (before - after) / before if before else None,
                }
            )
    return rows


def fold_rows(frame: pl.DataFrame, context: Context) -> list[dict[str, Any]]:
    """Return P9 and P10 per fold, never pooled."""
    rows = []
    for identifier, reference in REFERENCES.items():
        part = frame.filter(pl.col("split") == reference.split)
        for variant in context.config.variants:
            chosen = part.filter(pl.col("method") == variant.name)
            for fold in sorted(chosen["fold"].unique().to_list()):
                cases = chosen.filter(pl.col("fold") == fold)
                rows.append(
                    {
                        "method": variant.name,
                        "perturbation": identifier,
                        "split": reference.split,
                        "fold": fold,
                        "realism": reference.realism,
                        "cases": cases["case_id"].n_unique(),
                        **scores(cases, context.config),
                    }
                )
    return rows


def best_clean_baseline(frame: pl.DataFrame, context: Context) -> tuple[str, float]:
    """Return the non-ranker method with the best clean Top-1 on group_run."""
    ranked = (
        frame.filter((pl.col("split") == "group_run") & (pl.col("point") == CLEAN))
        .group_by("method")
        .agg(pl.col("hit@1").cast(pl.Float64).mean())
    )
    names = set(context.config.s3_baselines)
    allowed = ranked.filter(pl.col("method").is_in(sorted(names))).sort(
        ["hit@1", "method"], descending=[True, False]
    )
    return str(allowed["method"][0]), float(allowed["hit@1"][0])


def hypotheses(frame: pl.DataFrame, context: Context) -> dict[str, Any]:
    """Return S1, S2 and S3, with Holm over the primary comparisons available."""
    config = context.config
    key = s3_key(config, context.points)
    runs = frame.filter(pl.col("split") == "group_run")

    def arm(method: str, point: str = CLEAN) -> pl.DataFrame:
        return _select(runs, method, "group_run", point)

    best, best_top1 = best_clean_baseline(frame, context)
    result: dict[str, Any] = {"s2_comparator": {"method": best, "top1": best_top1}}
    for name, ranker in (("ranker", config.ranker), ("plain_ranker", config.plain_ranker)):
        result[name] = {
            "method": ranker,
            "S1": paired(arm(ranker), arm(config.prior), config),
            "S2": paired(arm(ranker), arm(best), config, config.s2_margin),
            "difference_from_best": paired(arm(ranker), arm(best), config),
        }
    s3 = {
        baseline: difference_of_differences(
            arm(config.ranker),
            arm(baseline),
            arm(config.ranker, key),
            arm(baseline, key),
            config,
        )
        for baseline in config.s3_baselines
    }
    result["S3"] = s3
    labelled = {
        "S1": ("S1", result["ranker"]["S1"]["p_value"]),
        "S2": ("S2", result["ranker"]["S2"]["p_value"]),
        **{f"S3 vs {name}": ("S3", entry["p_value"]) for name, entry in s3.items()},
    }
    table = label_comparisons(labelled, config.primary)
    result["holm"] = dict(zip(table["label"], table["adjusted_p"], strict=True))
    masked = config.variant("ablate_mask").name
    result["mask_gain_at_s3_point"] = paired(arm(config.ranker, key), arm(masked, key), config)
    return result


def ablation(frame: pl.DataFrame, context: Context) -> list[dict[str, Any]]:
    """Return the full ranker minus each ablated variant: effect sizes, with no claims."""
    config = context.config
    runs = frame.filter(pl.col("split") == "group_run")
    key = s3_key(config, context.points)
    rows = []
    for variant in config.variants:
        if variant.drop_group is None and variant.name != "single_stage_augmented":
            continue
        entry: dict[str, Any] = {"variant": variant.name}
        for label, point in (("clean", CLEAN), ("p2_30", key)):
            entry[label] = paired(
                _select(runs, config.ranker, "group_run", point),
                _select(runs, variant.name, "group_run", point),
                config,
            )
        rows.append(entry)
    return rows


def curves(rows: Sequence[dict[str, Any]], context: Context) -> None:
    """Draw one figure per perturbation and panel, labelled with its realism."""
    directory = context.root / EXPERIMENT_DIR / "figures"
    identifiers = sorted({r["perturbation"] for r in rows if r["perturbation"] != CLEAN})
    panels = (
        ("supervised", context.config.curve_methods.supervised),
        ("unsupervised", context.config.curve_methods.unsupervised),
    )
    for identifier in identifiers:
        tag = next(r["realism"] for r in rows if r["perturbation"] == identifier)
        keys = [r["point"] for r in rows if r["perturbation"] == identifier]
        ordered = list(dict.fromkeys(keys))
        for panel, methods in panels:
            series = []
            for method in methods:
                base = next(
                    (r for r in rows if r["method"] == method and r["point"] == CLEAN), None
                )
                points = [r for r in rows if r["method"] == method and r["point"] in ordered]
                if base is None or not points:
                    continue
                entries = [base, *sorted(points, key=lambda r: ordered.index(r["point"]))]
                top1 = [100 * entry["Top-1"]["estimate"] for entry in entries]
                series.append(
                    CurveSeries(
                        method,
                        list(range(len(entries))),
                        top1,
                        [100 * entry["Top-1"]["lower"] for entry in entries],
                        [100 * entry["Top-1"]["upper"] for entry in entries],
                    )
                )
            if series:
                robustness_curves(
                    series,
                    f"{identifier} grid point, 0 is clean, order as in the table",
                    "Top-1 (%)",
                    tag,
                    directory / f"{identifier}_{panel}",
                )


def report_stage(context: Context) -> dict[str, Any]:
    """Write tables, curves and the summary from saved outcomes."""
    frame = load_outcomes(context)
    rows = grid_rows(frame, context)
    payload: dict[str, Any] = {
        "status": context.status,
        "ranker_config_hash": config_hash(context.ranker),
        "augment_config_hash": config_hash(context.augment),
        "grid": rows,
        "held_out_folds": fold_rows(frame, context),
        "hypotheses": hypotheses(frame, context),
        "ablation": ablation(frame, context),
    }
    curves(rows, context)
    directory = context.root / EXPERIMENT_DIR
    write_result(
        directory / "results.json",
        payload,
        context.config,
        context.config.seed,
        context.dataset.checksum,
    )
    _write_text(context.root / SUMMARY_PATH, summary_text(payload, context))
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def _interval(entry: Mapping[str, float]) -> str:
    return f"{_pct(entry['estimate'])} [{_pct(entry['lower'])}, {_pct(entry['upper'])}]"


def _difference(label: str, entry: Mapping[str, float], verdict: str, adjusted: object) -> str:
    holm = "-" if adjusted is None else f"{float(str(adjusted)):.4f}"
    return f"| {label} | {_interval(entry)} | {entry['p_value']:.4f} | {holm} | {verdict} |"


def s1_verdict(entry: Mapping[str, float]) -> str:
    """S1 passes when the paired interval lies above zero."""
    return "pass" if entry["lower"] > 0 else "fail"


def s2_verdict(entry: Mapping[str, float]) -> str:
    """S2 passes when the margin-shifted interval lies above zero."""
    return "pass" if entry["lower"] > 0 else "fail"


def s3_verdict(entry: Mapping[str, float]) -> str:
    """S3 passes when the margin grows under loss, with the interval above zero."""
    return "pass" if entry["lower"] > 0 else "fail"


def summary_text(payload: Mapping[str, Any], context: Context) -> str:
    """Return ``docs/robustness_summary.md``."""
    config = context.config
    found = payload["hypotheses"]
    ranker, holm = found["ranker"], found["holm"]
    best = found["s2_comparator"]["method"]
    lines = [
        "# Robustness summary",
        "",
        f"Generated by `cascadelens experiments robustness`. **Status: {payload['status']}.**",
        "",
    ]
    if payload["status"] != FROZEN:
        lines += [
            "The config is not frozen, so verdicts below preview the pre-registered tests.",
            "",
        ]
    lines += [
        f"The CascadeLens ranker is `{config.ranker}`. Differences are Top-1 points on group_run "
        "with paired group-bootstrap intervals, over seed-averaged cases. Perturbations are "
        "simulated and reach test cases only; P9 and P10 are held-out splits. The benchmark has a "
        "static topology, so this measures an imperfect graph, not a changing one.",
        "",
        "Holm runs over the primary comparisons available here. S4 needs Part 10 and S7 is an "
        "observed result, so the final adjusted values can only rise.",
        "",
        "## Primary hypotheses",
        "",
        "| Criterion | Difference [95% CI] | p | Holm p | Verdict |",
        "| --- | --- | --- | --- | --- |",
        _difference(
            "S1: ranker minus frequency", ranker["S1"], s1_verdict(ranker["S1"]), holm["S1"]
        ),
        _difference(
            f"S2: ranker minus {best}, plus {_pct(config.s2_margin)} margin",
            ranker["S2"],
            s2_verdict(ranker["S2"]),
            holm["S2"],
        ),
    ]
    for name, entry in found["S3"].items():
        lines.append(
            _difference(
                f"S3: margin growth over {name}, P2 at 30% (simulated)",
                entry,
                s3_verdict(entry),
                holm[f"S3 vs {name}"],
            )
        )
    lines += [
        "",
        f"Without the margin, the ranker minus {best} is "
        f"{_interval(ranker['difference_from_best'])} points. The plain Part 7 ranker has S1 "
        f"{_interval(found['plain_ranker']['S1'])} and that same difference "
        f"{_interval(found['plain_ranker']['difference_from_best'])}.",
        "",
        "## Ablation (exploratory)",
        "",
        "Full ranker minus the ablated variant, in Top-1 points. Effect sizes only.",
        "",
        "| Variant | Clean | P2 at 30% (simulated) |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {row['variant']} | {_interval(row['clean'])} | {_interval(row['p2_30'])} |"
        for row in payload["ablation"]
    ]
    lines += [
        "",
        "`ablate_mask` shows what mask features add under P2. P2's gaps come from ground truth, so "
        "a gain there may be the gaps revealing the fault, not robustness: "
        f"{_interval(found['mask_gain_at_s3_point'])} points.",
        "",
        "## Top-1 by perturbation (exploratory)",
        "",
        "| Point | Label | " + " | ".join(config.curve_methods.supervised) + " |",
        "| --- | --- | " + " | ".join("---" for _ in config.curve_methods.supervised) + " |",
    ]
    table: dict[str, dict[str, str]] = {}
    labels: dict[str, str] = {}
    for row in payload["grid"]:
        table.setdefault(row["point"], {})[row["method"]] = _pct(row["Top-1"]["estimate"])
        labels[row["point"]] = row["realism"]
    for point, entries in table.items():
        cells = " | ".join(entries.get(m, "-") for m in config.curve_methods.supervised)
        lines.append(f"| {point} | {labels[point]} | {cells} |")
    lines += [
        "",
        "Full tables with intervals and relative drops are in "
        "`experiments/robustness/results.json`, and curves in `experiments/robustness/figures/`.",
        "",
    ]
    return "\n".join(lines)


def freeze_variants(name: str, root: Path = Path()) -> Path:
    """Record the validation choice of every ranker variant on every split family.

    This reads training and validation cases only. Part 7's method names are recorded as well, so
    `experiments ranker-clean` accepts the same frozen config.
    """
    context = load_context(True, root)
    records = list(context.dataset.records.values())
    context.augment_store().build(records, context.config.workers)
    families: dict[str, dict[str, list[dict[str, Any]]]] = {}
    ranked = [v for v in context.config.variants if v.kind in ("two_stage", "single_stage")]
    extra = [
        Variant(name=method, kind="two_stage" if method == "two_stage_ranker" else "single_stage")
        for method in METHODS
    ]
    for split_name in FAMILIES:
        split, group_map = load_split(split_name, context.dataset, root)
        families[split_name] = {}
        for variant in [*ranked, *extra]:
            settings = variant_settings(variant, context.ranker)
            ranker = make_ranker(variant, context, group_map, None)
            entries = []
            for index, fold in enumerate(split.folds):
                train = context.dataset.handles(fold.train_ids)
                if isinstance(ranker, AugmentedRanker):
                    train = ranker.expanded(train)
                val = context.dataset.handles(fold.val_ids)
                entries.append(
                    {"fold": index, **ranker.search_fold(train, val, settings).to_dict()}
                )
            families[split_name][variant.name] = entries
    record = FrozenRecord(
        name=name,
        config_hash=config_hash(context.ranker),
        feature_config_hash=config_hash(context.features),
        data_checksum=context.dataset.checksum,
        git_commit=git_commit(),
        families=families,
    )
    return write_frozen(record, root)


def write_json(path: Path, payload: object) -> None:
    """Write stable JSON."""
    _write_text(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
