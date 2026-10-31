from collections.abc import Sequence

import numpy as np
import polars as pl
import pytest
from ranker_factory import ALL_GROUPS, SIGNAL, settings, synthetic, wrapped

from cascadelens.baselines.base import CaseHandle
from cascadelens.eval.metrics import per_case_outcomes, retrieval_recall_at_k
from cascadelens.features.context import FeatureError
from cascadelens.features.registry import feature_columns
from cascadelens.models.data import (
    RankerError,
    RankingMatrix,
    assert_no_identity,
    build_matrix,
    model_features,
)
from cascadelens.models.importance import dominant_structural, gain_shares, group_shares
from cascadelens.models.ranker import TwoStageRanker
from cascadelens.models.retrieval import (
    HeuristicScorer,
    candidates,
    cross_fit_scores,
    score_frame,
    top_k_rows,
)
from cascadelens.models.single_stage import SingleStageRanker
from cascadelens.models.tuning import sample_configurations


def _truth(cases: Sequence[CaseHandle]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": [c.record.case_id for c in cases],
            "root_cause_service": [c.record.root_cause_service for c in cases],
            "group_id": [c.record.run_id for c in cases],
        }
    )


def _top1(method: type[TwoStageRanker], shuffled: bool = False) -> float:
    cases, source = synthetic(240, shuffled=shuffled)
    train, val, test = cases[:160], cases[160:200], cases[200:]
    model = method(source)
    model.fit(train, wrapped(settings()), val, 0)
    predictions = pl.concat(model.predict(case) for case in test)
    return float(per_case_outcomes(predictions, _truth(test))["hit@1"].mean())  # type: ignore[arg-type]


def _matrix(count: int = 8) -> tuple[RankingMatrix, list[CaseHandle]]:
    handles, source = synthetic(count)
    tables = [source.tables[c.record.case_id] for c in handles]
    truth = {c.record.case_id: c.record.root_cause_service for c in handles}
    return build_matrix(tables, truth, ALL_GROUPS), handles


def test_groups_and_labels_line_up_with_rows() -> None:
    matrix, cases = _matrix()
    assert matrix.sizes == (6,) * 8
    assert matrix.features.shape[0] == matrix.labels.size == len(matrix.services) == 48
    for (start, end), case in zip(matrix.spans(), cases, strict=True):
        assert matrix.labels[start:end].sum() == 1
        row = start + int(np.argmax(matrix.labels[start:end]))
        assert matrix.services[row] == case.record.root_cause_service


def test_input_matrix_has_no_identity_column() -> None:
    matrix, _ = _matrix(2)
    assert not {"service", "case_id"} & set(matrix.columns)
    assert list(matrix.columns) == feature_columns(ALL_GROUPS)
    with pytest.raises(FeatureError):
        assert_no_identity(["signal_cpu_max_z", "service"])


@pytest.mark.parametrize("column", ["service_id", "signal_service_id", "mask_name"])
def test_adding_an_identity_column_fails_the_build(column: str) -> None:
    handles, source = synthetic(2)
    table = source.tables[handles[0].record.case_id].with_columns(pl.lit(1.0).alias(column))
    truth = {handles[0].record.case_id: handles[0].record.root_cause_service}
    with pytest.raises(FeatureError):
        build_matrix([table], truth, ALL_GROUPS)
    with pytest.raises(FeatureError):
        model_features(table, ["signal_cpu_max_z"])


def test_feature_groups_switch_off() -> None:
    handles, source = synthetic(2)
    tables = [source.tables[c.record.case_id] for c in handles]
    truth = {c.record.case_id: c.record.root_cause_service for c in handles}
    matrix = build_matrix(tables, truth, ["signal"])
    assert all(column.startswith("signal_") for column in matrix.columns)


def test_recall_is_monotone_in_k_and_capped_by_case_size() -> None:
    matrix, cases = _matrix(20)
    scores = np.random.default_rng(0).normal(size=matrix.labels.size)
    tables = [
        score_frame(case.record.case_id, matrix.services[start:end], scores[start:end])
        for case, (start, end) in zip(cases, matrix.spans(), strict=True)
    ]
    frame, truth = pl.concat(tables), _truth(cases)
    recalls = [retrieval_recall_at_k(frame, truth, k).mean for k in range(1, 9)]
    assert recalls == sorted(recalls)
    assert recalls[5] == recalls[7] == 1.0
    _, sizes = top_k_rows(scores, matrix.sizes, 8)
    assert sizes == [6] * 20


def test_unretrieved_truth_stays_a_miss_in_stage_two() -> None:
    cases, source = synthetic(120)
    model = TwoStageRanker(source)
    model.fit(cases[:80], wrapped(settings(scorer="heuristic", choices=(1,))), cases[80:100], 0)
    test = cases[100:]
    predictions = pl.concat(model.predict(case) for case in test)
    assert predictions.group_by("case_id").len()["len"].to_list() == [1] * 20
    outcomes = per_case_outcomes(predictions, _truth(test))
    assert outcomes.height == 20
    assert outcomes["rank_of_truth"].null_count() > 0
    assert outcomes.filter(pl.col("rank_of_truth").is_null())["hit@1"].sum() == 0


def test_case_without_positive_is_kept_in_candidates_and_dropped_for_training() -> None:
    matrix, _ = _matrix(4)
    chosen = candidates(matrix, -matrix.labels.astype(np.float64), 2)
    assert chosen.sizes == (2,) * 4
    assert chosen.labels.sum() == 0
    assert chosen.with_positive().sizes == ()


def test_cross_fitting_is_stable_and_needs_two_groups() -> None:
    matrix, cases = _matrix(40)
    groups = {c.record.case_id: c.record.run_id for c in cases}
    config = settings().retrieval
    assert np.array_equal(
        cross_fit_scores(matrix, groups, config, 0, 1),
        cross_fit_scores(matrix, groups, config, 0, 1),
    )
    with pytest.raises(RankerError, match="two groups"):
        cross_fit_scores(matrix, dict.fromkeys(groups, "one"), config, 0, 1)


def test_heuristic_prefers_high_z_and_early_onset() -> None:
    scorer = HeuristicScorer(["signal_cpu_max_z", "temporal_onset_rank"])
    scores = scorer.score(np.array([[5.0, 1.0], [5.0, 2.0], [np.nan, np.nan]]))
    assert scores[0] > scores[1] > scores[2]
    with pytest.raises(RankerError):
        HeuristicScorer(["signal_cpu_max_z"])


@pytest.mark.parametrize("method", [TwoStageRanker, SingleStageRanker])
def test_a_perfect_feature_reaches_top1_of_one(method: type[TwoStageRanker]) -> None:
    assert _top1(method) == 1.0


@pytest.mark.parametrize("method", [TwoStageRanker, SingleStageRanker])
def test_shuffled_labels_fall_to_chance(method: type[TwoStageRanker]) -> None:
    assert _top1(method, shuffled=True) < 0.4


def test_same_seed_gives_identical_predictions() -> None:
    cases, source = synthetic(120)
    outputs = []
    for _ in range(2):
        model = TwoStageRanker(source)
        model.fit(cases[:80], wrapped(settings()), cases[80:100], 3)
        outputs.append(pl.concat(model.predict(case) for case in cases[100:]))
    assert outputs[0].equals(outputs[1])


def test_different_seed_changes_the_tuning_sample() -> None:
    assert sample_configurations(5, 0, (5, 8)) == sample_configurations(5, 0, (5, 8))
    assert sample_configurations(5, 0, (5, 8)) != sample_configurations(5, 1, (5, 8))
    assert all(k is None for _, k in sample_configurations(5, 0, None))
    assert {k for _, k in sample_configurations(30, 0, (5, 8))} == {5, 8}


def test_seeds_of_one_fold_share_the_search() -> None:
    cases, source = synthetic(120)
    model = TwoStageRanker(source)
    for seed in (0, 1):
        model.fit(cases[:80], wrapped(settings()), cases[80:100], seed)
    first, second = model.take_folds()
    assert first.choice == second.choice
    assert model.folds == []


def test_ranker_needs_validation_cases_and_fitting() -> None:
    cases, source = synthetic(10)
    with pytest.raises(RankerError, match="validation"):
        TwoStageRanker(source).fit(cases, wrapped(settings()), (), 0)
    with pytest.raises(RankerError, match="fitted"):
        TwoStageRanker(source).predict(cases[0])


def test_importance_sums_by_group_and_flags_structural_dominance() -> None:
    cases, source = synthetic(120)
    model = TwoStageRanker(source)
    model.fit(cases[:80], wrapped(settings()), cases[80:100], 0)
    shares = gain_shares(model.folds[-1].booster)
    assert sum(shares.values()) == pytest.approx(1.0)
    assert group_shares(shares)["signal"] > 0.5
    assert dominant_structural({"structural_betweenness": 0.5, SIGNAL: 0.5}, 0.25) == [
        "structural_betweenness"
    ]
    assert dominant_structural({SIGNAL: 1.0}, 0.25) == []
