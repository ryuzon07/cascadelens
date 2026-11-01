from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from perturb_factory import make_case
from ranker_factory import settings, synthetic, wrapped

from cascadelens.baselines.base import BaselineError, CaseHandle
from cascadelens.config import RankerConfig
from cascadelens.data.schema import Case
from cascadelens.features.config import FeaturesConfig
from cascadelens.features.context import FeatureInput
from cascadelens.features.registry import build_case_features
from cascadelens.models.augment import (
    AugmentConfig,
    AugmentedRanker,
    AugmentedSource,
    AugmentStore,
    assert_same_group,
    augment_case,
    copy_id,
    copy_plan,
    data_onset,
    source_id,
)
from cascadelens.models.tuning import Choice
from cascadelens.perturb.base import metrics_bytes, with_fields

CONFIG = AugmentConfig()
SHIFT = 12.0


def faulty_case(number: int = 0, root: str = "svc-3") -> Case:
    """Return a case where the second service shifts first, so onset differs from the label."""
    case = make_case(number, root=root)
    shifted = case.metrics.with_columns(
        pl.when((pl.col("service") == "svc-1") & (pl.col("timestamp") > 1120))
        .then(pl.col("value") + SHIFT)
        .otherwise(pl.col("value"))
        .alias("value")
    )
    return with_fields(case, metrics=shifted)


@pytest.mark.parametrize("kind", ["random", "failure"])
def test_masks_use_no_ground_truth(kind: str) -> None:
    case = faulty_case()
    other = with_fields(case, root_cause_service="svc-7", inject_time=case.window_start + 5)
    first = augment_case(case, kind, 0, CONFIG)  # type: ignore[arg-type]
    second = augment_case(other, kind, 0, CONFIG)  # type: ignore[arg-type]
    assert metrics_bytes(first.metrics) == metrics_bytes(second.metrics)
    assert first.root_cause_service == case.root_cause_service
    assert second.root_cause_service == "svc-7"


def test_failure_masks_follow_the_detected_onset_not_the_label() -> None:
    case = faulty_case()
    onset = data_onset(case)
    assert onset is not None
    assert onset[0] == "svc-1"
    masked = augment_case(case, "failure", 0, AugmentConfig(rate_low=0.3, rate_high=0.3))
    missing = masked.metrics.group_by("service").agg(gone=pl.col("value").is_null().mean())
    share = dict(missing.iter_rows())
    assert share["svc-1"] > share["svc-7"]
    assert share["svc-1"] > share["svc-5"]


@pytest.mark.parametrize("kind", ["random", "failure"])
def test_masks_are_seeded(kind: str) -> None:
    case = faulty_case()
    first = augment_case(case, kind, 0, CONFIG)  # type: ignore[arg-type]
    same = augment_case(case, kind, 0, CONFIG)  # type: ignore[arg-type]
    other = augment_case(case, kind, 1, CONFIG)  # type: ignore[arg-type]
    assert metrics_bytes(first.metrics) == metrics_bytes(same.metrics)
    assert metrics_bytes(first.metrics) != metrics_bytes(other.metrics)
    changed = augment_case(case, kind, 0, AugmentConfig(seed=1))  # type: ignore[arg-type]
    assert metrics_bytes(first.metrics) != metrics_bytes(changed.metrics)


def test_random_masks_change_nan_share_and_mask_features() -> None:
    case = faulty_case()
    copy = augment_case(case, "random", 0, AugmentConfig(rate_low=0.5, rate_high=0.5))
    config = FeaturesConfig(groups=["signal", "mask"])
    clean = build_case_features(FeatureInput.from_record(case.record(), case.metrics), config)
    masked = build_case_features(FeatureInput.from_record(copy.record(), copy.metrics), config)
    columns = [c for c in clean.columns if c.startswith("signal_")]

    def nan_share(table: pl.DataFrame) -> float:
        return float(np.isnan(table.select(columns).to_numpy()).mean())

    assert nan_share(masked) > nan_share(clean)
    assert masked["mask_missing_fraction"].mean() > clean["mask_missing_fraction"].mean()  # type: ignore[operator]
    assert masked["mask_missing_fraction"].max() == 1.0
    assert copy.metrics.height == case.metrics.height
    assert not copy.metrics.filter(pl.col("value") == 0.0).height


def test_copy_ids_round_trip_and_follow_the_plan() -> None:
    assert source_id(copy_id("re1-ob_a_cpu_1", "failure", 0)) == "re1-ob_a_cpu_1"
    assert copy_plan(AugmentConfig(random_copies=2, failure_copies=1)) == [
        ("random", 0),
        ("random", 1),
        ("failure", 0),
    ]


def test_group_check_rejects_a_copy_that_moved() -> None:
    assert_same_group([("a.aug.random0", "a")], {"a.aug.random0": "g", "a": "g"})
    with pytest.raises(BaselineError, match="left the group"):
        assert_same_group([("a.aug.random0", "a")], {"a.aug.random0": "h", "a": "g"})


class SpyRanker(AugmentedRanker):
    """Records what training receives and stubs out the training itself."""

    seen: list[str]

    def search_fold(
        self,
        train_cases: Sequence[CaseHandle],
        val_cases: Sequence[CaseHandle],
        settings: RankerConfig,
    ) -> Choice:
        self.seen = [case.record.case_id for case in train_cases]
        self.seen_val = [case.record.case_id for case in val_cases]
        return super().search_fold(train_cases, val_cases, settings)


def test_copies_join_training_only_and_keep_their_group(tmp_path: Path) -> None:
    cases, source = synthetic(120)
    config = AugmentConfig(random_copies=1, failure_copies=0)
    store = AugmentStore(config, FeaturesConfig(groups=["signal"]), "x", tmp_path)
    tables = dict(source.tables)
    for case in cases[:80]:
        identifier = copy_id(case.record.case_id, "random", 0)
        tables[identifier] = source.tables[case.record.case_id].with_columns(
            pl.lit(identifier).alias("case_id")
        )

    class Tables:
        def table(self, case: CaseHandle) -> pl.DataFrame:
            return tables[case.record.case_id]

    groups = {c.record.case_id: f"group-{i // 2}" for i, c in enumerate(cases)}
    ranker = SpyRanker(Tables(), store, groups)
    ranker.fit(cases[:80], wrapped(settings()), cases[80:100], 0)
    train_ids = {c.record.case_id for c in cases[:80]}
    copies = [case for case in ranker.seen if case not in train_ids]
    assert len(copies) == 80
    assert all(source_id(case) in train_ids for case in copies)
    assert all(ranker.groups[case] == groups[source_id(case)] for case in copies)
    assert not any(".aug" in case for case in ranker.seen_val)
    predictions = ranker.predict(cases[100])
    assert not predictions.is_empty()


def test_augmented_source_serves_stored_copies(tmp_path: Path) -> None:
    cases, source = synthetic(2)
    store = AugmentStore(AugmentConfig(), FeaturesConfig(groups=["signal"]), "x", tmp_path)
    wrapper = AugmentedSource(source, store)
    assert wrapper.table(cases[0]).equals(source.table(cases[0]))
    identifier = copy_id(cases[0].record.case_id, "random", 0)
    path = store.directory / "features" / f"{identifier}.parquet"
    path.parent.mkdir(parents=True)
    source.table(cases[1]).write_parquet(path)
    handle = store.handles([cases[0]])[0]
    assert wrapper.table(handle).equals(source.table(cases[1]))
