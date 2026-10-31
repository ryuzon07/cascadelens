import polars as pl
import pytest

from cascadelens.eval.errors import EvalError
from cascadelens.eval.seeds import seed_averaged_outcomes, summarize_seeds
from cascadelens.eval.significance import (
    holm_correction,
    label_comparisons,
    wilcoxon_signed_rank,
)


def test_holm_against_a_hand_computed_example() -> None:
    # Sorted: 0.01 * 4 = 0.04, 0.02 * 3 = 0.06, 0.03 * 2 = 0.06, 0.04 * 1 -> running max 0.06.
    adjusted = holm_correction([0.04, 0.01, 0.03, 0.02], ["d", "a", "c", "b"])

    assert list(adjusted) == ["d", "a", "c", "b"]
    assert adjusted == pytest.approx({"a": 0.04, "b": 0.06, "c": 0.06, "d": 0.06})


def test_holm_caps_at_one() -> None:
    assert holm_correction([0.6, 0.7], ["a", "b"]) == {"a": 1.0, "b": 1.0}


def test_holm_with_one_comparison_changes_nothing() -> None:
    assert holm_correction([0.03], ["a"]) == {"a": 0.03}


@pytest.mark.parametrize(
    ("pvalues", "labels"), [([0.1], ["a", "b"]), ([0.1, 0.2], ["a", "a"]), ([1.5], ["a"])]
)
def test_holm_refuses_bad_inputs(pvalues: list[float], labels: list[str]) -> None:
    with pytest.raises(EvalError):
        holm_correction(pvalues, labels)


def test_labels_come_from_the_config_list_and_only_primaries_are_corrected() -> None:
    table = label_comparisons(
        {
            "ranker vs prior": ("S1", 0.01),
            "ranker margin": ("S3", 0.04),
            "ece": ("S5", 0.001),
        }
    )

    assert table["role"].to_list() == ["primary", "primary", "exploratory"]
    assert table["adjusted_p"].to_list()[:2] == pytest.approx([0.02, 0.04])
    assert table["adjusted_p"][2] is None


def test_labels_honour_a_custom_primary_list() -> None:
    table = label_comparisons({"x": ("S5", 0.01)}, primary=["S5"])

    assert table["role"][0] == "primary"
    assert table["adjusted_p"][0] == 0.01


def test_wilcoxon_by_hand() -> None:
    # Differences 1..6, all positive: W = 0 and the exact two-sided p is 2 / 2^6.
    result = wilcoxon_signed_rank([1, 2, 3, 4, 5, 6], [0] * 6)

    assert result.statistic == 0
    assert result.p_value == pytest.approx(2 / 64)
    assert result.n_nonzero == 6


def test_wilcoxon_drops_zero_differences() -> None:
    assert wilcoxon_signed_rank([1, 0, 2], [0, 0, 0]).n_nonzero == 2


def test_wilcoxon_with_no_differences_is_not_significant() -> None:
    assert wilcoxon_signed_rank([1, 1], [1, 1]).p_value == 1.0


def test_wilcoxon_can_run_on_group_means() -> None:
    result = wilcoxon_signed_rank([1, 1, 2, 2], [0, 0, 0, 0], ["a", "a", "b", "b"])

    assert result.n_nonzero == 2


def test_wilcoxon_refuses_misaligned_inputs() -> None:
    with pytest.raises(EvalError, match="aligned"):
        wilcoxon_signed_rank([1], [1, 2])
    with pytest.raises(EvalError, match="group ids"):
        wilcoxon_signed_rank([1], [1], [1, 2])


def seed_table() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "case_id": ["a", "b"] * 2,
            "group_id": ["g1", "g2"] * 2,
            "seed": [0, 0, 1, 1],
            "value": [1.0, 0.0, 0.0, 0.0],
        }
    )


def test_seed_average_is_per_case() -> None:
    table = seed_averaged_outcomes(seed_table(), min_seeds=2)

    assert table["case_id"].to_list() == ["a", "b"]
    assert table["value"].to_list() == [0.5, 0.0]


def test_summary_reports_spread_and_the_seed_averaged_interval() -> None:
    summary = summarize_seeds(seed_table(), min_seeds=2, n=200)

    assert summary.seed_means["value"].to_list() == [0.5, 0.0]
    assert summary.spread == pytest.approx(0.5 / 2**0.5)
    assert summary.ci.estimate == 0.25
    assert summary.ci.n_groups == 2


def test_one_seed_has_no_spread() -> None:
    table = seed_table().filter(pl.col("seed") == 0)

    assert summarize_seeds(table, min_seeds=1, n=50).spread == 0


def test_too_few_seeds_fail() -> None:
    with pytest.raises(EvalError, match="at least 5 seeds"):
        seed_averaged_outcomes(seed_table())


def test_a_missing_case_seed_pair_fails() -> None:
    with pytest.raises(EvalError, match="Case b is not present once per seed"):
        seed_averaged_outcomes(seed_table().head(3), min_seeds=2)


def test_a_case_in_two_groups_fails() -> None:
    table = seed_table().with_columns(
        group_id=pl.when(pl.col("seed") == 1).then(pl.lit("other")).otherwise("group_id")
    )

    with pytest.raises(EvalError, match="several group ids"):
        seed_averaged_outcomes(table, min_seeds=2)


def test_missing_columns_fail() -> None:
    with pytest.raises(EvalError, match="Missing columns"):
        seed_averaged_outcomes(seed_table().drop("seed"))
