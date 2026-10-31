"""Temporal features: when each service turned anomalous, relative to the case.

Onset positions are fractions of the post-baseline window. Rank 1 is the earliest, and tied services
share a rank. ``topology_agreement`` is one value per case: the Spearman correlation between onset
position and distance from the earliest-onset services along the direction faults propagate. Tied
earliest services are all sources, so no name decides anything. It is NaN without topology, or
when fewer than three services can be compared.
"""

import math

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
import polars as pl
from scipy import stats  # type: ignore[import-untyped]

from cascadelens.features.context import CaseContext
from cascadelens.features.graphs import call_graph

PREFIX = "temporal_"
COLUMNS = (
    f"{PREFIX}onset",
    f"{PREFIX}onset_rank",
    f"{PREFIX}onset_delta",
    f"{PREFIX}topology_agreement",
)
MINIMUM_PAIRS = 3


def _agreement(context: CaseContext) -> float:
    graph = call_graph(context.case)
    if graph is None or not context.onsets:
        return math.nan
    placed = {s: o for s, o in context.onsets.items() if s in graph}
    if not placed:
        return math.nan
    earliest = min(placed.values())
    sources = {service for service, onset in placed.items() if onset == earliest}
    distances = nx.multi_source_dijkstra_path_length(graph.reverse(copy=False), sources)
    reached = sorted(service for service in placed if service in distances)
    if len(reached) < MINIMUM_PAIRS:
        return math.nan
    order = np.array([placed[service] for service in reached])
    steps = np.array([distances[service] for service in reached], dtype=np.float64)
    if np.ptp(order) == 0 or np.ptp(steps) == 0:
        return math.nan
    return float(stats.spearmanr(order, steps).statistic)


def temporal_features(context: CaseContext) -> pl.DataFrame:
    """Return one row per service; services with no onset are null."""
    onsets = [context.onsets.get(service) for service in context.services]
    onset = pl.col("temporal_onset")
    earliest = min(context.onsets.values(), default=None)
    return (
        context.frame()
        .with_columns(
            pl.Series("temporal_onset", onsets, dtype=pl.Float64),
        )
        .with_columns(
            temporal_onset_rank=onset.rank("min").cast(pl.Float64),
            temporal_onset_delta=onset - earliest if earliest is not None else onset,
            temporal_topology_agreement=pl.lit(_agreement(context), dtype=pl.Float64),
        )
    )
