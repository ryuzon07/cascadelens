"""Ingestion checks. Errors name the case and the reason, and stop the build."""

from collections.abc import Iterable

import polars as pl

from cascadelens.data.errors import DataValidationError, Issue
from cascadelens.data.schema import Case


def check_label_in_metrics(case: Case) -> list[Issue]:
    """The ground-truth service must have metrics."""
    if case.metrics.filter(pl.col("service") == case.root_cause_service).is_empty():
        reason = f"label service {case.root_cause_service!r} has no metrics"
        return [Issue(case.case_id, "label_in_metrics", reason)]
    return []


def check_window_not_empty(case: Case) -> list[Issue]:
    """The window must be ordered and hold at least one real sample."""
    samples = case.metrics.filter(
        pl.col("timestamp").is_between(case.window_start, case.window_end)
        & pl.col("value").is_not_null()
    )
    if case.window_start > case.window_end or samples.is_empty():
        reason = f"no samples in window [{case.window_start}, {case.window_end}]"
        return [Issue(case.case_id, "empty_window", reason)]
    return []


def check_timestamps_monotonic(case: Case) -> list[Issue]:
    """Timestamps must strictly increase within each series, in stored order."""
    broken = (
        case.metrics.group_by("service", "metric", maintain_order=True)
        .agg(pl.col("timestamp").diff().min().alias("smallest_step"))
        .filter(pl.col("smallest_step") <= 0)
    )
    return [
        Issue(
            case.case_id,
            "monotonic_timestamps",
            f"series {row['service']}/{row['metric']} has a step of {row['smallest_step']}",
        )
        for row in broken.iter_rows(named=True)
    ]


def check_no_nan_only_series(case: Case) -> list[Issue]:
    """Series without a single value fail for the label service and are logged for the rest.

    The real RE1-TT data has Istio series that are empty for services that took no traffic.
    That is genuine missing telemetry, which this project studies, so it is kept and logged
    once per case. An empty series on the root-cause service would corrupt the label's
    evidence and still fails.
    """
    empty = (
        case.metrics.group_by("service", "metric", maintain_order=True)
        .agg(pl.col("value").is_not_null().and_(pl.col("value").is_not_nan()).any().alias("real"))
        .filter(~pl.col("real"))
    )
    labelled = empty.filter(pl.col("service") == case.root_cause_service)
    issues = [
        Issue(
            case.case_id,
            "nan_only_series",
            f"label service series {row['service']}/{row['metric']} has no values",
        )
        for row in labelled.iter_rows(named=True)
    ]
    others = empty.filter(pl.col("service") != case.root_cause_service)
    if not others.is_empty():
        services = sorted(others["service"].unique().to_list())
        reason = f"{others.height} series without values on {len(services)} service(s): {services}"
        issues.append(Issue(case.case_id, "nan_only_series", reason, "warning"))
    return issues


def check_inject_time_in_data(case: Case) -> list[Issue]:
    """The injection must fall inside the recorded data."""
    timestamp = pl.col("timestamp")
    first, last = case.metrics.select(timestamp.min().alias("first"), timestamp.max()).row(0)
    if first is None or not first <= case.inject_time <= last:
        reason = f"inject_time {case.inject_time} is outside data range [{first}, {last}]"
        return [Issue(case.case_id, "inject_time_in_data", reason)]
    return []


def check_topology_coverage(case: Case) -> list[Issue]:
    """Log, without failing, any metric service the topology does not cover."""
    if case.topology is None:
        reason = "no topology available"
        return [Issue(case.case_id, "topology_coverage", reason, "warning")]
    nodes = {name for edge in case.topology for name in (edge.caller, edge.callee)}
    services = set(case.metrics["service"].unique().to_list())
    missing = sorted(services - nodes)
    if missing:
        reason = f"metric services not in topology: {missing}"
        return [Issue(case.case_id, "topology_coverage", reason, "warning")]
    return []


CASE_CHECKS = (
    check_label_in_metrics,
    check_window_not_empty,
    check_timestamps_monotonic,
    check_no_nan_only_series,
    check_inject_time_in_data,
    check_topology_coverage,
)


class CaseValidator:
    """Check cases one at a time, so that a suite need not fit in memory."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def check(self, case: Case) -> list[Issue]:
        """Return every issue with ``case``, including a clash with any earlier case."""
        issues = [issue for check in CASE_CHECKS for issue in check(case)]
        if case.case_id in self._seen:
            issues.append(Issue(case.case_id, "duplicate_case_id", "case_id already used"))
        self._seen.add(case.case_id)
        return issues


def raise_on_errors(issues: Iterable[Issue]) -> None:
    """Raise DataValidationError listing every error in ``issues``."""
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        raise DataValidationError(errors)


def validate_cases(cases: Iterable[Case]) -> list[Issue]:
    """Check ``cases``, raising on any error and returning the logged warnings."""
    validator = CaseValidator()
    issues = [issue for case in cases for issue in validator.check(case)]
    raise_on_errors(issues)
    return issues
