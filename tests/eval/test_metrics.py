from typing import Any

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cascadelens.eval.errors import EvalError
from cascadelens.eval.metrics import (
    avg_at_5,
    missing_truth_count,
    per_case_outcomes,
    ranked,
    retrieval_recall_at_k,
    top_k,
)

# Ranks of the truth, read from the scores: c1 first, c2 second, c3 third, c4 sixth, c5 absent.
SCORES = {
    "c1": {"a": 5.0, "b": 1.0},
    "c2": {"a": 5.0, "b": 4.0},
    "c3": {"a": 5.0, "b": 4.0, "c": 3.0},
    "c4": {"a": 6.0, "b": 5.0, "c": 4.0, "d": 3.0, "e": 2.0, "f": 1.0},
    "c5": {"a": 2.0, "b": 1.0},
}
LABELS = {"c1": "a", "c2": "b", "c3": "c", "c4": "f", "c5": "z"}


def predictions(scores: dict[str, dict[str, float]]) -> pl.DataFrame:
    rows = [(c, s, v) for c, entries in scores.items() for s, v in entries.items()]
    return pl.DataFrame(rows, schema=["case_id", "service", "score"], orient="row")


def truth(labels: dict[str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": list(labels),
            "root_cause_service": list(labels.values()),
            "group_id": [f"g-{case}" for case in labels],
        }
    )


PRED, TRUTH = predictions(SCORES), truth(LABELS)


def test_rank_of_truth_on_the_five_case_example() -> None:
    outcomes = per_case_outcomes(PRED, TRUTH)

    assert outcomes["rank_of_truth"].to_list() == [1, 2, 3, 6, None]
    assert outcomes["hit@1"].to_list() == [1, 0, 0, 0, 0]
    assert outcomes["hit@3"].to_list() == [1, 1, 1, 0, 0]
    assert outcomes["hit@5"].to_list() == [1, 1, 1, 0, 0]
    assert outcomes["group_id"][0] == "g-c1"


@pytest.mark.parametrize(
    ("k", "expected"), [(1, 1 / 5), (2, 2 / 5), (3, 3 / 5), (5, 3 / 5), (6, 4 / 5)]
)
def test_top_k_means(k: int, expected: float) -> None:
    assert top_k(PRED, TRUTH, k).mean == pytest.approx(expected)


def test_top_k_is_per_case() -> None:
    assert top_k(PRED, TRUTH, 2).per_case["hit"].to_list() == [1, 1, 0, 0, 0]


def test_avg_at_5_is_the_mean_of_top_1_to_5() -> None:
    # Top-1..5 are 1/5, 2/5, 3/5, 3/5, 3/5, whose mean is 12/25.
    assert avg_at_5(PRED, TRUTH) == pytest.approx(12 / 25)


def test_avg_at_5_per_case_averages_to_the_metric() -> None:
    per_case = per_case_outcomes(PRED, TRUTH)["avg@5"].to_list()

    assert per_case == pytest.approx([1.0, 0.8, 0.6, 0.0, 0.0])


def test_missing_truth_is_a_counted_miss() -> None:
    result = top_k(PRED, TRUTH, 5)

    assert result.missing == 1
    assert result.per_case["hit"][4] == 0
    assert missing_truth_count(per_case_outcomes(PRED, TRUTH)) == 1


def test_case_with_no_predictions_at_all_is_a_counted_miss() -> None:
    result = top_k(PRED.filter(pl.col("case_id") != "c5"), TRUTH, 1)

    assert result.per_case.height == 5
    assert result.missing == 1


def test_ties_break_by_ascending_service_name() -> None:
    pred = predictions({"c": {"zeta": 1.0, "alpha": 1.0, "mid": 1.0}})

    assert ranked(pred).sort("rank")["service"].to_list() == ["alpha", "mid", "zeta"]


def test_tie_breaking_ignores_row_order() -> None:
    pred = predictions({"c1": {"b": 1.0, "a": 1.0}, "c2": {"a": 2.0, "b": 2.0}})
    labels = truth({"c1": "b", "c2": "b"})

    first = per_case_outcomes(pred, labels)
    second = per_case_outcomes(pred.reverse(), labels)

    assert first.equals(second)
    assert first["rank_of_truth"].to_list() == [2, 2]


def test_a_dict_of_lists_is_accepted_like_a_pandas_frame() -> None:
    pred: dict[str, list[Any]] = {"case_id": ["c", "c"], "service": ["a", "b"], "score": [1, 2]}
    labels = {"case_id": ["c"], "root_cause_service": ["b"], "group_id": ["g"]}

    assert top_k(pred, labels, 1).mean == 1.0


def test_recall_at_k() -> None:
    result = retrieval_recall_at_k(PRED, TRUTH, 2)

    assert result.mean == pytest.approx(2 / 5)
    assert result.missing == 1


@pytest.mark.parametrize(
    ("pred", "labels", "message"),
    [
        (PRED.drop("score"), TRUTH, "Missing columns"),
        (PRED.with_columns(score=pl.lit(float("nan"))), TRUTH, "null or NaN"),
        (pl.concat([PRED, PRED]), TRUTH, "more than once in the predictions"),
        (PRED, TRUTH.filter(pl.col("case_id") != "c1"), "unknown case c1"),
        (PRED, pl.concat([TRUTH, TRUTH]), "more than once in the ground truth"),
        (PRED, TRUTH.clear(), "empty"),
    ],
)
def test_bad_inputs_fail_loudly(pred: pl.DataFrame, labels: pl.DataFrame, message: str) -> None:
    with pytest.raises(EvalError, match=message):
        per_case_outcomes(pred, labels)


def test_k_must_be_positive() -> None:
    with pytest.raises(EvalError, match="at least 1"):
        top_k(PRED, TRUTH, 0)


@st.composite
def datasets(draw: st.DrawFn) -> tuple[pl.DataFrame, pl.DataFrame]:
    services = ["a", "b", "c", "d", "e", "f", "g"]
    scores: dict[str, dict[str, float]] = {}
    labels: dict[str, str] = {}
    for index in range(draw(st.integers(1, 6))):
        case = f"c{index}"
        names = draw(st.lists(st.sampled_from(services), min_size=1, unique=True))
        scores[case] = {n: draw(st.floats(-5, 5, allow_nan=False)) for n in names}
        labels[case] = draw(st.sampled_from(services))
    return predictions(scores), truth(labels)


@settings(deadline=None)
@given(datasets())
def test_metrics_stay_in_range_and_top_k_is_monotone(
    data: tuple[pl.DataFrame, pl.DataFrame],
) -> None:
    pred, labels = data

    means = [top_k(pred, labels, k).mean for k in range(1, 9)]

    assert all(0 <= mean <= 1 for mean in means)
    assert means == sorted(means)
    assert 0 <= avg_at_5(pred, labels) <= 1
    assert avg_at_5(pred, labels) == pytest.approx(sum(means[:5]) / 5)
