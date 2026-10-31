import math

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cascadelens.eval.calibration import (
    aurc,
    brier,
    case_confidence,
    ece,
    oracle_abstention,
    random_abstention,
    reliability_table,
    risk_coverage,
    selective_accuracy,
)
from cascadelens.eval.errors import EvalError


def scores(rows: list[tuple[str, str, float]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["case_id", "service", "score"], orient="row")


def test_softmax_of_two_scores() -> None:
    table = case_confidence(scores([("c", "a", math.log(3)), ("c", "b", 0.0)]))

    assert table["service"][0] == "a"
    assert table["confidence"][0] == pytest.approx(0.75)


def test_temperature_flattens_confidence() -> None:
    table = case_confidence(scores([("c", "a", 2 * math.log(3)), ("c", "b", 0.0)]), temperature=2)

    assert table["confidence"][0] == pytest.approx(0.75)


def test_softmax_survives_huge_scores() -> None:
    table = case_confidence(scores([("c", "a", 1e6), ("c", "b", 1e6 - math.log(3))]))

    assert table["confidence"][0] == pytest.approx(0.75)


def test_tied_scores_share_confidence_and_rank_by_name() -> None:
    table = case_confidence(scores([("c", "b", 1.0), ("c", "a", 1.0)]))

    assert table.row(0) == ("c", "a", 0.5)


def test_temperature_must_be_positive() -> None:
    with pytest.raises(EvalError, match="positive"):
        case_confidence(scores([("c", "a", 1.0)]), temperature=0)


def test_ece_on_known_bins() -> None:
    # Bin (0.2, 0.3]: confidence 0.25, accuracy 1/2, gap 0.25, weight 2/5.
    # Bin (0.8, 0.9]: confidence 0.85, accuracy 2/3, gap 11/60, weight 3/5.
    conf = [0.25, 0.25, 0.85, 0.85, 0.85]
    correct = [1, 0, 1, 1, 0]

    assert ece(conf, correct) == pytest.approx(0.4 * 0.25 + 0.6 * (0.85 - 2 / 3))


def test_ece_is_zero_when_calibrated() -> None:
    assert ece([0.5, 0.5], [1, 0]) == 0.0


def test_ece_bin_count_changes_the_grouping() -> None:
    conf, correct = [0.25, 0.75], [0, 1]

    assert ece(conf, correct, bins=1) == 0.0
    assert ece(conf, correct, bins=2) == pytest.approx(0.25)


def test_brier_on_a_trivial_example() -> None:
    assert brier([1.0, 0.0, 0.5], [1, 1, 1]) == pytest.approx((0 + 1 + 0.25) / 3)


def test_reliability_table_keeps_empty_bins() -> None:
    table = reliability_table([0.25, 0.25, 0.85], [1, 0, 1], bins=10)

    assert table.height == 10
    assert table["count"].to_list() == [0, 0, 2, 0, 0, 0, 0, 0, 1, 0]
    assert table["accuracy"][2] == 0.5
    assert table["confidence"][2] == pytest.approx(0.25)
    assert table["gap"][8] == pytest.approx(0.15)
    assert table["accuracy"][0] is None


def test_zero_confidence_lands_in_the_first_bin() -> None:
    assert reliability_table([0.0], [0], bins=4)["count"].to_list() == [1, 0, 0, 0]


def test_risk_coverage_by_hand() -> None:
    # Most confident first: right, wrong, right.
    curve = risk_coverage([0.9, 0.5, 0.1], [1, 0, 1])

    np.testing.assert_allclose(curve.coverage, [1 / 3, 2 / 3, 1])
    np.testing.assert_allclose(curve.risk, [0, 1 / 2, 1 / 3])


def test_informative_confidence_matches_the_oracle() -> None:
    correct = np.array([1, 1, 1, 0, 0])
    conf = correct * 0.5 + 0.25

    curve = risk_coverage(conf, correct)
    oracle = oracle_abstention(correct)

    np.testing.assert_allclose(curve.risk, oracle.risk)
    # Risks are 0, 0, 0, 1/4 and 2/5.
    assert aurc(curve) == pytest.approx(aurc(oracle)) == pytest.approx((0.25 + 0.4) / 5)


def test_random_confidence_gives_aurc_near_the_error_rate() -> None:
    rng = np.random.default_rng(0)
    correct = rng.random(20_000) < 0.7

    area = aurc(risk_coverage(rng.random(20_000), correct))

    assert area == pytest.approx(1 - correct.mean(), abs=0.01)


def test_random_reference_is_flat_at_the_error_rate() -> None:
    assert aurc(random_abstention([1, 1, 1, 0])) == pytest.approx(0.25)


def test_tied_confidence_gives_the_flat_curve_in_any_order() -> None:
    for correct in ([1, 0, 0, 1], [0, 0, 1, 1]):
        np.testing.assert_allclose(risk_coverage([0.5] * 4, correct).risk, [0.5] * 4)


def test_selective_accuracy_by_hand() -> None:
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [1, 1, 0, 0]

    assert selective_accuracy(conf, correct, 0.5) == 1.0
    assert selective_accuracy(conf, correct, 0.75) == pytest.approx(2 / 3)
    assert selective_accuracy(conf, correct, 1.0) == 0.5


def test_selective_accuracy_keeps_at_least_one_case() -> None:
    assert selective_accuracy([0.9, 0.1], [1, 0], 0.01) == 1.0


@pytest.mark.parametrize(
    ("conf", "correct", "message"),
    [
        ([0.5], [1, 0], "Shapes differ"),
        ([], [], "No cases"),
        ([1.5], [1], r"\[0, 1\]"),
        ([float("nan")], [1], r"\[0, 1\]"),
        ([0.5], [2], "0 or 1"),
    ],
)
def test_bad_inputs_fail_loudly(conf: list[float], correct: list[int], message: str) -> None:
    with pytest.raises(EvalError, match=message):
        brier(conf, correct)


def test_coverage_must_be_valid() -> None:
    with pytest.raises(EvalError, match="coverage"):
        selective_accuracy([0.5], [1], 0)


PAIRS = st.lists(st.tuples(st.floats(0, 1), st.integers(0, 1)), min_size=1, max_size=40)


@settings(deadline=None)
@given(PAIRS)
def test_calibration_metrics_stay_in_range(pairs: list[tuple[float, int]]) -> None:
    conf, correct = zip(*pairs, strict=True)

    assert 0 <= ece(conf, correct) <= 1
    assert 0 <= brier(conf, correct) <= 1
    assert 0 <= selective_accuracy(conf, correct, 0.5) <= 1
    assert aurc(oracle_abstention(correct)) <= aurc(risk_coverage(conf, correct)) + 1e-12
