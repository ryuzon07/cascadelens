"""Masking augmentation, for training cases only (PRD 8.4).

Masks are applied to the raw series, features are recomputed, and the copies join the training set
as extra cases. Copies keep the group of their source, so they can never cross a split boundary.
Validation and test cases are never augmented. Mask features stay in the model, and missing values
stay NaN.

**Why failure-correlated masking reads no ground truth.** Part 8's P2 simulates how a failure
damages telemetry, so it may read the true root cause. A training mask must not: it would write the
label into the mask features, and the model would learn to read gaps instead of the fault. So
"near" here means close to the earliest-onset service and "after" means after the earliest detected
onset, both computed from the data alone. The same procedure can therefore run at inference. To
make this hold by construction, masks are computed on a scrubbed copy of the case whose label and
injection time are placeholders.
"""

import hashlib
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
from pydantic import Field

from cascadelens.baselines.base import BaselineError, CaseHandle
from cascadelens.baselines.lightgbm_flat import LightGBMFlat
from cascadelens.config import BaselineConfig, StrictModel, canonical_json, config_hash
from cascadelens.data.io import read_case
from cascadelens.data.schema import Case, CaseRecord
from cascadelens.features.config import FeaturesConfig
from cascadelens.features.context import FeatureInput, onsets, scored_samples
from cascadelens.features.registry import build_case_features
from cascadelens.features.simple import SignalSettings
from cascadelens.models.data import TableSource
from cascadelens.models.ranker import TwoStageRanker
from cascadelens.models.tuning import Choice
from cascadelens.perturb.base import SERIES, null_rows, observed, rounded, with_fields
from cascadelens.perturb.topology import hop_distances
from cascadelens.seed import derive_case_seed

AUGMENTED_DIR = Path("data/augmented")
COPY_MARK = ".aug"
Kind = Literal["random", "failure"]
SCRUBBED_SERVICE = "unknown"
SCRUBBED_TIME = 0


class AugmentConfig(StrictModel):
    """How many copies to make and how hard to mask them."""

    seed: int = Field(default=0, ge=0)
    random_copies: int = Field(default=1, ge=0)
    failure_copies: int = Field(default=1, ge=0)
    rate_low: float = Field(default=0.1, gt=0, le=1)
    rate_high: float = Field(default=0.5, gt=0, le=1)
    near_hops: int = Field(default=1, ge=0)
    near_boost: float = Field(default=3.0, ge=1)
    after_boost: float = Field(default=3.0, ge=1)
    block_seconds: int = Field(default=30, ge=1)


def scrubbed(case: Case) -> Case:
    """Return ``case`` with placeholders where its ground truth was."""
    return with_fields(case, root_cause_service=SCRUBBED_SERVICE, inject_time=SCRUBBED_TIME)


def _generator(case_id: str, kind: Kind, index: int, config: AugmentConfig) -> np.random.Generator:
    params = config.model_dump(mode="json")
    return np.random.default_rng(derive_case_seed(case_id, f"augment_{kind}", params, index))


def mask_random(case: Case, rate: float, rng: np.random.Generator) -> pl.DataFrame:
    """Drop ``rate`` of the series, then ``rate`` of the services whole."""
    metrics = case.metrics
    series = metrics.filter(observed(metrics)).select(*SERIES).unique().sort(*SERIES)
    count = rounded(rate * series.height)
    chosen = series[np.sort(rng.choice(series.height, size=count, replace=False)).tolist()]
    services = sorted(metrics["service"].unique().to_list())
    gone = rng.choice(len(services), size=rounded(rate * len(services)), replace=False)
    whole = [services[index] for index in gone.tolist()]
    flagged = metrics.join(
        chosen.with_columns(drop=pl.lit(True)), on=list(SERIES), how="left", maintain_order="left"
    )
    rows = flagged["drop"].fill_null(False) | flagged["service"].is_in(whole)
    return null_rows(metrics, rows.to_numpy())


def data_onset(case: Case) -> tuple[str, int] | None:
    """Return the earliest-onset service and its time, from the data, or None without an onset."""
    view = FeatureInput.from_record(case.record(), case.metrics)
    settings = SignalSettings()
    found = onsets(scored_samples(view, settings), settings)
    if found.is_empty():
        return None
    service, position = found.sort("onset", "service").row(0)
    span = case.window_end - case.window_start
    cut = case.window_start + settings.baseline_fraction * span
    return str(service), round(cut + position * (case.window_end - cut))


def mask_failure(
    case: Case, rate: float, config: AugmentConfig, rng: np.random.Generator
) -> pl.DataFrame:
    """Drop blocks with a probability rising near and after the earliest detected onset."""
    onset = data_onset(case)
    if onset is None:
        return case.metrics
    service, moment = onset
    distances = hop_distances(case, service)
    near = sorted(name for name, hops in distances.items() if hops <= config.near_hops)
    block = (pl.col("timestamp") - case.window_start) // config.block_seconds
    start = case.window_start + block * config.block_seconds
    frame = case.metrics.with_columns(
        block=block,
        near=pl.col("service").is_in(near),
        after=start >= moment,
        real=observed(case.metrics),
    )
    keys = frame.select(*SERIES, "block").unique(maintain_order=True)
    draws = keys.with_columns(draw=pl.Series(rng.random(keys.height)))
    probability = (
        pl.lit(rate)
        * pl.when("near").then(config.near_boost).otherwise(1.0)
        * pl.when("after").then(config.after_boost).otherwise(1.0)
    ).clip(0.0, 1.0)
    joined = frame.join(draws, on=[*SERIES, "block"], how="left", maintain_order="left")
    rows = joined.select((pl.col("draw") < probability) & pl.col("real")).to_series()
    return null_rows(case.metrics, rows.to_numpy())


def augment_case(case: Case, kind: Kind, index: int, config: AugmentConfig) -> Case:
    """Return one masked copy. It keeps the label, the run and every other field of its source."""
    rng = _generator(case.case_id, kind, index, config)
    rate = float(rng.uniform(config.rate_low, config.rate_high))
    blind = scrubbed(case)
    if kind == "random":
        metrics = mask_random(blind, rate, rng)
    else:
        metrics = mask_failure(blind, rate, config, rng)
    return with_fields(case, case_id=copy_id(case.case_id, kind, index), metrics=metrics)


def copy_id(case_id: str, kind: Kind, index: int) -> str:
    """Return the id of a copy."""
    return f"{case_id}{COPY_MARK}.{kind}{index}"


def source_id(copy: str) -> str:
    """Return the id of a copy's source."""
    return copy.split(COPY_MARK, 1)[0]


def copy_plan(config: AugmentConfig) -> list[tuple[Kind, int]]:
    """Return the copies each case gets."""
    random_copies: list[tuple[Kind, int]] = [("random", i) for i in range(config.random_copies)]
    failure_copies: list[tuple[Kind, int]] = [("failure", i) for i in range(config.failure_copies)]
    return [*random_copies, *failure_copies]


def assert_same_group(pairs: Sequence[tuple[str, str]], groups: Mapping[str, str]) -> None:
    """Fail unless every (copy, source) pair shares a group."""
    for copy, source in pairs:
        if groups[copy] != groups[source]:
            raise BaselineError(f"Copy {copy} left the group of {source}")


def _build_copy(
    job: tuple[CaseRecord, str, str, AugmentConfig, FeaturesConfig, Kind, int],
) -> str:
    """Build one copy. Runs in a worker process, so it takes plain values."""
    record, processed, directory, config, features, kind, index = job
    target = Path(directory)
    identifier = copy_id(record.case_id, kind, index)
    table_path = target / "features" / f"{identifier}.parquet"
    if not table_path.is_file():
        copy = augment_case(read_case(record, Path(processed)), kind, index, config)
        metrics_file = target / "metrics" / f"{identifier}.parquet"
        metrics_file.parent.mkdir(parents=True, exist_ok=True)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pending = metrics_file.with_suffix(f".{os.getpid()}.partial")
        copy.metrics.write_parquet(pending)
        os.replace(pending, metrics_file)
        view = FeatureInput.from_record(copy.record(), copy.metrics)
        pending = table_path.with_suffix(f".{os.getpid()}.partial")
        build_case_features(view, features).write_parquet(pending)
        os.replace(pending, table_path)
    return identifier


class AugmentStore:
    """Masked copies of every case, built once and cached by content hash."""

    def __init__(
        self, config: AugmentConfig, features: FeaturesConfig, checksum: str, root: Path = Path()
    ) -> None:
        self.config = config
        self.features = features
        payload = canonical_json([config.model_dump(mode="json"), config_hash(features), checksum])
        name = hashlib.sha256(payload.encode()).hexdigest()
        self.directory = root / AUGMENTED_DIR / name
        self.processed = root / "data" / "processed"

    def build(self, records: Sequence[CaseRecord], workers: int = 1) -> None:
        """Build every missing copy, resuming a partial build."""
        jobs = [
            (
                record,
                str(self.processed),
                str(self.directory),
                self.config,
                self.features,
                kind,
                index,
            )
            for record in records
            for kind, index in copy_plan(self.config)
        ]
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                list(pool.map(_build_copy, jobs, chunksize=4))
        else:
            for job in jobs:
                _build_copy(job)

    def handles(self, cases: Sequence[CaseHandle]) -> list[CaseHandle]:
        """Return handles of the copies of ``cases``, which must be training cases."""
        result = []
        for case in cases:
            for kind, index in copy_plan(self.config):
                identifier = copy_id(case.record.case_id, kind, index)
                path = self.directory / "metrics" / f"{identifier}.parquet"
                record = case.record.model_copy(update={"case_id": identifier})
                result.append(CaseHandle(record, partial(pl.read_parquet, path)))
        return result


class AugmentedSource:
    """Feature tables of copies from the store, and of everything else from ``base``."""

    def __init__(self, base: TableSource, store: AugmentStore) -> None:
        self.base = base
        self.store = store
        self._tables: dict[str, pl.DataFrame] = {}

    def table(self, case: CaseHandle) -> pl.DataFrame:
        """Return the feature table of ``case``."""
        identifier = case.record.case_id
        if COPY_MARK not in identifier:
            return self.base.table(case)
        if identifier not in self._tables:
            path = self.store.directory / "features" / f"{identifier}.parquet"
            self._tables[identifier] = pl.read_parquet(path)
        return self._tables[identifier]


class AugmentedRanker(TwoStageRanker):
    """The CascadeLens ranker: Part 7's trainer, given masked copies as extra training cases."""

    name = "two_stage_augmented"

    def __init__(
        self,
        source: TableSource,
        store: AugmentStore,
        groups: Mapping[str, str] | None = None,
        frozen: Mapping[str, Choice] | None = None,
    ) -> None:
        super().__init__(source, groups, frozen)
        self.store = store

    def expanded(self, train_cases: Sequence[CaseHandle]) -> list[CaseHandle]:
        """Return the training cases with their copies, each copy in its source's group."""
        copies = self.store.handles(train_cases)
        sources = {
            c.record.case_id: self.groups.get(c.record.case_id, c.record.run_id)
            for c in train_cases
        }
        self.groups.update(sources)
        pairs = [(c.record.case_id, source_id(c.record.case_id)) for c in copies]
        self.groups.update({copy: self.groups[source] for copy, source in pairs})
        assert_same_group(pairs, self.groups)
        return [*train_cases, *copies]

    def fit(
        self,
        train_cases: Sequence[CaseHandle],
        config: BaselineConfig,
        val_cases: Sequence[CaseHandle] = (),
        seed: int = 0,
    ) -> None:
        """Train on the training cases and their copies. Validation is left clean."""
        super().fit(self.expanded(train_cases), config, val_cases, seed)


class AugmentedSingleStage(AugmentedRanker):
    """The single-stage ablation with augmentation."""

    name = "single_stage_augmented"
    retrieves = False


class AugmentedFlat(LightGBMFlat):
    """The augmentation-matched flat baseline: the same copies, still flat features only."""

    name = "lightgbm_flat_masking"

    def __init__(self, store: AugmentStore, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.store = store

    def fit(
        self,
        train_cases: Sequence[CaseHandle],
        config: BaselineConfig,
        val_cases: Sequence[CaseHandle] = (),
        seed: int = 0,
    ) -> None:
        """Train on the training cases and their copies, with the flat baseline's own search."""
        copies = self.store.handles(train_cases)
        super().fit([*train_cases, *copies], config, val_cases, seed)
