import math
import time

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cascadelens.eval.bootstrap import (
    diff_of_diffs_bootstrap,
    ece_difference_ci,
    group_bootstrap_ci,
    paired_group_bootstrap,
    resample_counts,
    selective_gain_bootstrap,
)
from cascadelens.eval.calibration import ece, selective_accuracy
from cascadelens.eval.errors import EvalError

RNG = np.random.default_rng(0)
VALUES = (RNG.random(100) < 0.6).astype(float)
ONE_PER_CASE = np.arange(100)


def test_one_group_per_case_matches_a_case_bootstrap() -> None:
    result = group_bootstrap_ci(VALUES, ONE_PER_CASE, n=10_000, seed=1)

    picks = np.random.default_rng(2).integers(0, 100, size=(10_000, 100))
    means = VALUES[picks].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])

    assert result.lower == pytest.approx(lower, abs=0.01)
    assert result.upper == pytest.approx(upper, abs=0.01)


def test_all_cases_in_one_group_is_degenerate() -> None:
    result = group_bootstrap_ci(VALUES, np.zeros(100), n=200)

    assert result.n_groups == 1
    assert result.lower == result.upper == result.estimate == VALUES.mean()


def test_group_resampling_never_splits_a_group() -> None:
    # Two groups of opposite constant values can only give means of 0, 1/2 or 1.
    values = [0.0, 0.0, 1.0, 1.0]
    counts = resample_counts(2, 1000, 0)

    assert set(counts[:, 1]) == {0.0, 1.0, 2.0}
    assert group_bootstrap_ci(values, ["a", "a", "b", "b"], n=1000).lower in (0.0, 0.5)
    assert group_bootstrap_ci(values, ["a", "a", "b", "b"], n=1000).upper in (0.5, 1.0)


def test_clustered_groups_widen_the_interval() -> None:
    values = np.repeat((np.random.default_rng(3).random(20) < 0.5).astype(float), 5)

    grouped = group_bootstrap_ci(values, np.repeat(np.arange(20), 5))
    naive = group_bootstrap_ci(values, np.arange(100))

    assert grouped.upper - grouped.lower > 1.5 * (naive.upper - naive.lower)


def test_seeded_results_reproduce() -> None:
    assert group_bootstrap_ci(VALUES, ONE_PER_CASE, n=500, seed=4) == group_bootstrap_ci(
        VALUES, ONE_PER_CASE, n=500, seed=4
    )
    assert group_bootstrap_ci(VALUES, ONE_PER_CASE, n=500, seed=4) != group_bootstrap_ci(
        VALUES, ONE_PER_CASE, n=500, seed=5
    )


def test_group_labels_only_need_to_be_distinct() -> None:
    labels = [f"group-{i % 7}" for i in range(100)]
    numbers = [i % 7 for i in range(100)]

    assert group_bootstrap_ci(VALUES, labels, n=500) == group_bootstrap_ci(VALUES, numbers, n=500)


def test_identical_inputs_give_zero_difference() -> None:
    result = paired_group_bootstrap(VALUES, VALUES, ONE_PER_CASE, n=500)

    assert result.estimate == 0
    assert result.lower <= 0 <= result.upper
    assert result.p_value == 1.0
    assert not result.excludes_zero()


def test_constant_shift_excludes_zero() -> None:
    result = paired_group_bootstrap(VALUES + 1, VALUES, ONE_PER_CASE, n=500)

    assert result.estimate == pytest.approx(1)
    assert result.lower == pytest.approx(1)
    assert result.excludes_zero()
    assert result.p_value == pytest.approx(2 / 501)


def test_paired_comparison_removes_shared_noise() -> None:
    noise = np.random.default_rng(5).normal(size=200)
    shift = np.random.default_rng(6).normal(scale=0.01, size=200)

    paired = paired_group_bootstrap(noise + 0.05 + shift, noise, np.arange(200))

    assert paired.excludes_zero()


def test_sign_flip_negates_the_difference() -> None:
    other = VALUES[::-1]

    first = paired_group_bootstrap(VALUES, other, ONE_PER_CASE, n=500)
    second = paired_group_bootstrap(other, VALUES, ONE_PER_CASE, n=500)

    assert second.estimate == pytest.approx(-first.estimate)
    assert second.lower == pytest.approx(-first.upper)


def test_diff_of_diffs_by_hand() -> None:
    # Margins: 0.5 when perturbed and 0.25 when clean, in every group.
    groups = [0, 0, 1, 1]
    result = diff_of_diffs_bootstrap(
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0],
        [0.75, 0.75, 0.75, 0.75],
        groups,
        n=200,
    )

    assert result.estimate == pytest.approx(0.25)
    assert result.lower == pytest.approx(0.25)
    assert result.upper == pytest.approx(0.25)


def test_diff_of_diffs_resamples_all_arms_with_the_same_groups() -> None:
    # The clean and perturbed margins are equal per group, so only one shared draw gives zero.
    margin = np.random.default_rng(7).random(50)
    zeros = np.zeros(50)

    result = diff_of_diffs_bootstrap(margin, zeros, margin, zeros, np.arange(50), n=300)

    assert result.lower == result.upper == 0


def test_ece_difference_matches_the_point_estimate() -> None:
    conf = np.array([0.9, 0.9, 0.9, 0.9, 0.2, 0.2])
    better = np.array([0.5, 0.5, 0.5, 0.5, 0.2, 0.2])
    correct = [1, 0, 1, 0, 0, 0]

    result = ece_difference_ci(conf, better, correct, [0, 0, 1, 1, 2, 2], n=500)

    assert result.estimate == pytest.approx(ece(better, correct) - ece(conf, correct))
    assert result.estimate < 0


def test_ece_difference_of_identical_confidences_is_zero() -> None:
    conf = np.random.default_rng(8).random(60)
    correct = conf > 0.5

    result = ece_difference_ci(conf, conf, correct, np.arange(60) % 6, n=200)

    assert (result.estimate, result.lower, result.upper, result.p_value) == (0, 0, 0, 1)


def test_resampled_ece_reduces_to_ece_for_one_group() -> None:
    conf = np.array([0.15, 0.25, 0.35, 0.85])
    correct = np.array([0, 1, 0, 1])

    result = ece_difference_ci(conf, np.zeros(4), correct, np.zeros(4), n=10)

    assert result.lower == pytest.approx(ece(np.zeros(4), correct) - ece(conf, correct))


def test_selective_gain_matches_the_point_estimate() -> None:
    conf = [0.9, 0.8, 0.7, 0.6]
    correct = [1, 1, 0, 0]

    result = selective_gain_bootstrap(conf, correct, [0, 1, 2, 3], 0.5, n=500)

    assert result.estimate == pytest.approx(selective_accuracy(conf, correct, 0.5) - 0.5)


def test_selective_gain_of_uninformative_confidence_is_zero() -> None:
    correct = (np.random.default_rng(9).random(100) < 0.5).astype(float)

    result = selective_gain_bootstrap(np.full(100, 0.5), correct, np.arange(100) % 20, n=300)

    assert result.estimate == pytest.approx(0)
    assert result.lower == pytest.approx(0, abs=1e-12)
    assert result.upper == pytest.approx(0, abs=1e-12)


def test_selective_gain_detects_informative_confidence() -> None:
    rng = np.random.default_rng(10)
    correct = (rng.random(300) < 0.5).astype(float)
    conf = 0.25 + 0.5 * correct + rng.uniform(-0.1, 0.1, 300)

    assert selective_gain_bootstrap(conf, correct, np.arange(300) % 60).lower > 0.2


def test_ten_thousand_resamples_on_400_cases_take_seconds() -> None:
    values = np.random.default_rng(11).random(400)
    groups = np.arange(400)

    start = time.perf_counter()
    paired_group_bootstrap(values, values[::-1], groups)
    diff_of_diffs_bootstrap(values, values[::-1], values[::-1], values, groups)
    ece_difference_ci(values, values[::-1], values > 0.5, groups)

    assert time.perf_counter() - start < 10


@pytest.mark.parametrize(
    ("values", "groups", "message"),
    [
        ([1.0], [], "non-empty"),
        ([1.0, 2.0], [0], "aligned"),
        ([math.nan], [0], "finite"),
    ],
)
def test_bad_inputs_fail_loudly(values: list[float], groups: list[int], message: str) -> None:
    with pytest.raises(EvalError, match=message):
        group_bootstrap_ci(values, groups, n=10)


def test_bad_parameters_fail_loudly() -> None:
    with pytest.raises(EvalError, match="alpha"):
        group_bootstrap_ci([1.0], [0], alpha=0)
    with pytest.raises(EvalError, match="at least 1"):
        group_bootstrap_ci([1.0], [0], n=0)


@settings(deadline=None, max_examples=25)
@given(st.lists(st.integers(0, 1), min_size=2, max_size=30), st.integers(1, 5))
def test_intervals_stay_in_range_and_contain_the_mean_of_binary_data(
    hits: list[int], group_size: int
) -> None:
    groups = [i // group_size for i in range(len(hits))]

    result = group_bootstrap_ci(hits, groups, n=200)

    assert 0 <= result.lower <= result.upper <= 1
    assert result.estimate == pytest.approx(sum(hits) / len(hits))
