"""Graph, timeline and coverage of one case, computed from the features' own context."""

import polars as pl

from cascadelens.api.schemas import (
    CoverageCell,
    GraphEdge,
    GraphNode,
    IncidentCoverage,
    IncidentGraph,
    IncidentTimeline,
    NodeState,
    ServiceTimeline,
    TopologySource,
)
from cascadelens.features.context import CaseContext
from cascadelens.features.simple import metric_family
from cascadelens.inference import DiagnoseResult

SUSPECTS = 3


def topology_source(context: CaseContext) -> TopologySource:
    """Return where the case's call graph came from."""
    if not context.case.topology:
        return "none"
    return "inferred" if context.topology_inferred else "provided"


def _with_telemetry(context: CaseContext) -> set[str]:
    present = context.case.metrics.filter(pl.col("value").is_not_null())
    return set(present["service"].unique().to_list())


def _state(service: str, rank: dict[str, int], live: set[str], context: CaseContext) -> NodeState:
    if service not in live:
        return "no_telemetry"
    place = rank.get(service)
    if place == 0:
        return "top_suspect"
    if place is not None and place < SUSPECTS:
        return "suspect"
    return "anomalous" if service in context.onsets else "healthy"


def _on_path(context: CaseContext, top: str) -> set[str]:
    """Return the services that reach the top suspect through anomalous callers."""
    edges = context.case.topology or ()
    reached = {top}
    growing = True
    while growing:
        callers = {e.caller for e in edges if e.callee in reached and e.caller in context.onsets}
        growing = not callers <= reached
        reached |= callers
    return reached


def build_graph(context: CaseContext, result: DiagnoseResult) -> IncidentGraph:
    """Return nodes and edges, with each node's state taken from the diagnosis."""
    rank = {item.service: place for place, item in enumerate(result.ranking)}
    score = {item.service: item.score for item in result.ranking}
    live = _with_telemetry(context)
    nodes = [
        GraphNode(
            service=service,
            state=_state(service, rank, live, context),
            score=score.get(service),
            has_telemetry=service in live,
        )
        for service in context.services
    ]
    path = _on_path(context, result.ranking[0].service) if result.ranking else set()
    edges = [
        GraphEdge(
            caller=edge.caller,
            callee=edge.callee,
            inferred=context.topology_inferred,
            on_anomalous_path=edge.caller in path and edge.callee in path,
        )
        for edge in context.case.topology or ()
    ]
    return IncidentGraph(nodes=nodes, edges=edges, topology_source=topology_source(context))


def build_timeline(context: CaseContext) -> IncidentTimeline:
    """Return each service's onset, last anomaly and any point where its telemetry stopped."""
    case = context.case
    span = case.window_end - case.window_start
    cut = case.window_start + context.signal.baseline_fraction * span
    after = max(case.window_end - cut, 1.0)
    last_anomaly = dict(
        context.samples.filter(~pl.col("in_baseline") & (pl.col("z") > context.signal.z_threshold))
        .group_by("service")
        .agg(pl.col("timestamp").max())
        .iter_rows()
    )
    last_sample = dict(
        case.metrics.filter(pl.col("value").is_not_null())
        .group_by("service")
        .agg(pl.col("timestamp").max())
        .iter_rows()
    )
    services = []
    for service in context.services:
        onset = context.onsets.get(service)
        seen = last_sample.get(service)
        stopped = seen is not None and case.window_end - seen > context.mask.stop_gap_seconds
        services.append(
            ServiceTimeline(
                service=service,
                onset=None if onset is None else round(cut + onset * after),
                anomalous_until=last_anomaly.get(service) if onset is not None else None,
                telemetry_stopped=seen if stopped else None,
                has_telemetry=seen is not None,
            )
        )
    return IncidentTimeline(
        window_start=case.window_start,
        window_end=case.window_end,
        baseline_end=round(cut),
        services=services,
    )


def build_coverage(context: CaseContext) -> IncidentCoverage:
    """Return the share of expected samples present, per service and metric family."""
    metrics = context.case.metrics
    moments = metrics["timestamp"].n_unique()
    series = metrics.group_by("service", "metric").agg(valid=pl.col("value").is_not_null().sum())
    totals: dict[tuple[str, str], tuple[int, int]] = {}
    for service, metric, valid in series.iter_rows():
        key = (service, metric_family(metric))
        samples, count = totals.get(key, (0, 0))
        totals[key] = (samples + valid, count + 1)
    families = sorted({family for _, family in totals})
    cells = []
    for service in context.services:
        for family in families:
            samples, count = totals.get((service, family), (0, 0))
            share = samples / (count * moments) if count else 0.0
            cells.append(
                CoverageCell(service=service, family=family, coverage=share, samples=samples)
            )
    present = sum(samples for samples, _ in totals.values())
    expected = sum(count for _, count in totals.values()) * moments
    return IncidentCoverage(
        services=context.services,
        families=families,
        cells=cells,
        overall=present / expected if expected else 0.0,
    )
