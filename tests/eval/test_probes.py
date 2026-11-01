from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest
from ranker_factory import settings, synthetic

from cascadelens.baselines.base import CaseHandle
from cascadelens.eval.probes import (
    FoldCases,
    ProbesConfig,
    ShuffledSource,
    detect_cliff,
    flagged_above_floor,
    identity_probe,
    injection_position,
    score_ranker,
    scored_top1,
    top1_ci,
    uniform_floor,
    window_points,
)
from cascadelens.eval.robustness import CLEAN_POINT, Point, point_key
from cascadelens.models.data import RankerError
from cascadelens.splits.splitters import Fold

CONFIG = ProbesConfig(
    n_resamples=200,
    offsets=[-30, -15, 15, 30],
    random_window_jitter=[60],
    window_methods=["ranker"],
)
GRAPH_MARKER = "graph_personalized_pagerank"


def _folds(cases: Sequence[CaseHandle]) -> list[FoldCases]:
    return [(cases[:160], cases[160:200], cases[200:240])]


def _groups(cases: Sequence[CaseHandle]) -> dict[str, str]:
    return {c.record.case_id: c.record.run_id for c in cases}


def test_identity_probe_drops_when_graph_features_identify_the_label() -> None:
    cases, source = synthetic(240, marker=GRAPH_MARKER)
    result = identity_probe(source, settings(), _folds(cases), _groups(cases), CONFIG)
    assert result["graph_and_structural"]["drop"]["estimate"] > 0.5
    assert result["structural_only"]["drop"]["estimate"] < 0.1
    assert result["passed"] is False


def test_identity_probe_stays_flat_when_they_do_not() -> None:
    cases, source = synthetic(240)
    result = identity_probe(source, settings(), _folds(cases), _groups(cases), CONFIG)
    assert abs(result["graph_and_structural"]["drop"]["estimate"]) < 0.05
    assert result["passed"] is True


def test_shuffling_is_seeded_and_keeps_the_values() -> None:
    cases, source = synthetic(2)
    wrapper = ShuffledSource(source, [GRAPH_MARKER])
    first = wrapper.table(cases[0])
    assert first.equals(wrapper.table(cases[0]))
    assert sorted(first[GRAPH_MARKER]) == sorted(source.table(cases[0])[GRAPH_MARKER])
    assert not first[GRAPH_MARKER].equals(source.table(cases[0])[GRAPH_MARKER])


def test_mask_style_flag_compares_the_interval_with_the_floor() -> None:
    assert flagged_above_floor({"lower": 0.2}, 0.05)
    assert not flagged_above_floor({"lower": 0.03}, 0.05)
    assert uniform_floor([4, 4]) == 0.25


def test_group_only_settings_run_on_one_group() -> None:
    cases, source = synthetic(240, marker=GRAPH_MARKER)
    config = settings()
    only = type(config).model_validate({**config.model_dump(mode="json"), "groups": ["graph"]})
    outcomes = score_ranker(source, only, _folds(cases), _groups(cases))
    assert top1_ci(outcomes, CONFIG)["estimate"] == 1.0


def test_cliff_detection_finds_a_spike_and_ignores_a_slope() -> None:
    spike = {-30: 0.5, -15: 0.5, 0: 0.9, 15: 0.5, 30: 0.5}
    slope = {-30: 0.6, -15: 0.7, 0: 0.8, 15: 0.7, 30: 0.6}
    assert detect_cliff(spike, 0.05)["cliff"] is True
    assert detect_cliff(slope, 0.05)["cliff"] is False
    with pytest.raises(RankerError):
        detect_cliff({1: 0.5}, 0.05)


def test_window_probe_detects_an_injected_cliff() -> None:
    cases, _ = synthetic(20)
    truth = pl.DataFrame(
        {
            "case_id": [c.record.case_id for c in cases],
            "root_cause_service": [c.record.root_cause_service for c in cases],
            "group_id": [c.record.run_id for c in cases],
        }
    )
    fold = Fold(train_ids=("x",), val_ids=("y",), test_ids=tuple(truth["case_id"].to_list()))
    points = [CLEAN_POINT, *window_points(CONFIG)]

    def score(point: Point, case_ids: Sequence[str]) -> pl.DataFrame:
        """Name the truth at offset 0 alone, as a model that reads the window would."""
        chosen = truth.filter(pl.col("case_id").is_in(case_ids))
        service = "wrong" if point is not CLEAN_POINT else None
        return chosen.select(
            "case_id",
            pl.coalesce(pl.lit(service, pl.String), "root_cause_service").alias("service"),
            pl.lit(1.0).alias("score"),
        )

    top1 = scored_top1(lambda f, s: None, score, [fold], [0], points, lambda p: p, truth)
    curve = {0: top1["clean"]}
    curve |= {o: top1[point_key("P4", {"offset_seconds": o})] for o in CONFIG.offsets}
    assert detect_cliff(curve, CONFIG.cliff_threshold)["cliff"] is True


def test_injection_position_reports_the_middle() -> None:
    cases, _ = synthetic(4)
    report = injection_position([c.record for c in cases])
    assert report["median"] == pytest.approx(0.5)
    assert report["share_within_2pct_of_middle"] == 1.0
    assert np.isfinite(report["median"])
