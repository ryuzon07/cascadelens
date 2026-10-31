import polars as pl
import pytest
from case_factory import Row, default_rows, make_case, metrics_frame

from cascadelens.data.errors import DataValidationError
from cascadelens.data.validate import CaseValidator, validate_cases


def checks_of(error: DataValidationError) -> set[str]:
    return {issue.check for issue in error.issues}


def test_valid_case_passes() -> None:
    assert validate_cases([make_case()]) == []


def test_label_service_must_exist_in_metrics() -> None:
    with pytest.raises(DataValidationError, match=r"re1-ob_db_cpu_1.*'ghost'") as caught:
        validate_cases([make_case(root_cause_service="ghost")])

    assert checks_of(caught.value) == {"label_in_metrics"}


@pytest.mark.parametrize(("start", "end"), [(500, 600), (120, 100)], ids=["no-samples", "reversed"])
def test_empty_window_fails(start: int, end: int) -> None:
    with pytest.raises(DataValidationError, match="no samples in window") as caught:
        validate_cases([make_case(window_start=start, window_end=end)])

    assert checks_of(caught.value) == {"empty_window"}


def test_window_of_only_nulls_is_empty() -> None:
    rows: list[Row] = [(s, m, t, None) for s, m, t, _ in default_rows() if t == 100]
    rows += [row for row in default_rows() if row[2] != 100]
    case = make_case(metrics=metrics_frame(rows), window_start=100, window_end=100)

    with pytest.raises(DataValidationError) as caught:
        validate_cases([case])

    assert "empty_window" in checks_of(caught.value)


@pytest.mark.parametrize("timestamps", [(100, 120, 110), (100, 110, 110)], ids=["back", "repeat"])
def test_non_monotonic_timestamps_fail(timestamps: tuple[int, int, int]) -> None:
    rows: list[Row] = [("db", "cpu", timestamp, 1.0) for timestamp in timestamps]
    rows += [row for row in default_rows() if row[:2] != ("db", "cpu")]

    with pytest.raises(DataValidationError, match="series db/cpu") as caught:
        validate_cases([make_case(metrics=metrics_frame(rows))])

    assert checks_of(caught.value) == {"monotonic_timestamps"}


def test_timestamps_are_checked_per_series() -> None:
    # Series restart at the same time, so a global check would wrongly fail here.
    assert validate_cases([make_case()]) == []


def test_duplicate_case_id_fails() -> None:
    with pytest.raises(DataValidationError, match="case_id already used") as caught:
        validate_cases([make_case(), make_case()])

    assert checks_of(caught.value) == {"duplicate_case_id"}


def test_uncovered_topology_is_logged_but_does_not_fail() -> None:
    issues = validate_cases([make_case(topology=[{"caller": "api", "callee": "gateway"}])])

    assert [(issue.check, issue.severity) for issue in issues] == [("topology_coverage", "warning")]
    assert "['db']" in issues[0].reason


def test_missing_topology_is_logged() -> None:
    issues = validate_cases([make_case(topology=None)])

    assert [issue.reason for issue in issues] == ["no topology available"]


def test_nan_only_series_on_the_label_service_fails() -> None:
    rows = [(s, m, t, None if (s, m) == ("db", "mem") else v) for s, m, t, v in default_rows()]

    with pytest.raises(DataValidationError, match="label service series db/mem") as caught:
        validate_cases([make_case(metrics=metrics_frame(rows))])

    assert checks_of(caught.value) == {"nan_only_series"}


def test_nan_only_series_on_other_services_are_logged_once() -> None:
    rows = [(s, m, t, None if s == "api" else v) for s, m, t, v in default_rows()]

    issues = validate_cases([make_case(metrics=metrics_frame(rows))])

    assert [(i.check, i.severity) for i in issues] == [("nan_only_series", "warning")]
    assert issues[0].reason == "2 series without values on 1 service(s): ['api']"


def test_float_nan_series_also_fails() -> None:
    metrics = metrics_frame(default_rows()).with_columns(
        pl.when(pl.col("metric") == "mem")
        .then(float("nan"))
        .otherwise(pl.col("value"))
        .alias("value")
    )

    with pytest.raises(DataValidationError) as caught:
        validate_cases([make_case(metrics=metrics)])

    assert checks_of(caught.value) == {"nan_only_series"}


@pytest.mark.parametrize("inject_time", [99, 121], ids=["before", "after"])
def test_inject_time_outside_data_fails(inject_time: int) -> None:
    with pytest.raises(DataValidationError, match="outside data range") as caught:
        validate_cases([make_case(inject_time=inject_time)])

    assert checks_of(caught.value) == {"inject_time_in_data"}


def test_all_failures_are_reported_together() -> None:
    bad = make_case(root_cause_service="ghost", inject_time=1)

    with pytest.raises(DataValidationError) as caught:
        validate_cases([bad])

    assert checks_of(caught.value) == {"label_in_metrics", "inject_time_in_data"}


def test_validator_keeps_state_between_cases() -> None:
    validator = CaseValidator()

    assert validator.check(make_case(case_id="a")) == validator.check(make_case(case_id="b"))
    assert [i.check for i in validator.check(make_case(case_id="a"))] == ["duplicate_case_id"]
