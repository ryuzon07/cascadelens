"""Graph features: neighbour anomalies and PageRank, plus structure-only features.

Two groups live here so ablations can drop them apart. ``graph`` depends on which services are
anomalous in this case. ``structural`` depends on the graph alone, which is fixed per system, so it
can fingerprint a service; the identity probe covers it (PRD 8.1, 6.5).

A case with no topology gets NaN everywhere and ``graph_topology_available`` = 0. A service absent
from an existing graph gets NaN too, while the flag stays 1: the flag describes the case.
"""

import math

import networkx as nx  # type: ignore[import-untyped]
import polars as pl

from cascadelens.features.context import CaseContext
from cascadelens.features.graphs import call_graph

GRAPH_PREFIX = "graph_"
STRUCTURAL_PREFIX = "structural_"
GRAPH_COLUMNS = (
    f"{GRAPH_PREFIX}upstream_anomalous_fraction",
    f"{GRAPH_PREFIX}downstream_anomalous_fraction",
    f"{GRAPH_PREFIX}personalized_pagerank",
    f"{GRAPH_PREFIX}topology_available",
)
STRUCTURAL_COLUMNS = (
    f"{STRUCTURAL_PREFIX}in_degree",
    f"{STRUCTURAL_PREFIX}out_degree",
    f"{STRUCTURAL_PREFIX}betweenness",
)


def _fraction(neighbours: list[str], anomalous: set[str]) -> float:
    if not neighbours:
        return math.nan
    return sum(node in anomalous for node in neighbours) / len(neighbours)


def _table(context: CaseContext, columns: dict[str, list[float]]) -> pl.DataFrame:
    frames = {name: pl.Series(name, values, dtype=pl.Float64) for name, values in columns.items()}
    return context.frame().with_columns(*frames.values())


def graph_features(context: CaseContext) -> pl.DataFrame:
    """Return anomaly-dependent graph features per service."""
    graph = call_graph(context.case)
    count = len(context.services)
    if graph is None:
        empty = {name: [math.nan] * count for name in GRAPH_COLUMNS}
        return _table(context, {**empty, GRAPH_COLUMNS[3]: [0.0] * count})
    anomalous = set(context.onsets) & set(graph)
    reversed_graph = graph.reverse(copy=False)
    ranks: dict[str, float] = {}
    if anomalous:
        seeds = {node: 1.0 if node in anomalous else 0.0 for node in graph}
        ranks = nx.pagerank(
            reversed_graph, alpha=context.graph.pagerank_alpha, personalization=seeds
        )
    upstream, downstream = [], []
    for service in context.services:
        known = service in graph
        upstream.append(
            _fraction(sorted(graph.successors(service)), anomalous) if known else math.nan
        )
        downstream.append(
            _fraction(sorted(graph.predecessors(service)), anomalous) if known else math.nan
        )
    return _table(
        context,
        {
            GRAPH_COLUMNS[0]: upstream,
            GRAPH_COLUMNS[1]: downstream,
            GRAPH_COLUMNS[2]: [ranks.get(service, math.nan) for service in context.services],
            GRAPH_COLUMNS[3]: [1.0] * count,
        },
    )


def structural_features(context: CaseContext) -> pl.DataFrame:
    """Return degree and betweenness per service, which ignore anomalies entirely."""
    graph = call_graph(context.case)
    count = len(context.services)
    if graph is None:
        return _table(context, {name: [math.nan] * count for name in STRUCTURAL_COLUMNS})
    between = nx.betweenness_centrality(graph)

    def value(service: str, number: float) -> float:
        return number if service in graph else math.nan

    return _table(
        context,
        {
            STRUCTURAL_COLUMNS[0]: [
                value(s, graph.in_degree(s) if s in graph else 0) for s in context.services
            ],
            STRUCTURAL_COLUMNS[1]: [
                value(s, graph.out_degree(s) if s in graph else 0) for s in context.services
            ],
            STRUCTURAL_COLUMNS[2]: [between.get(s, math.nan) for s in context.services],
        },
    )
