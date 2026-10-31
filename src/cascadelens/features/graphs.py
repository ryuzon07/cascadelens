"""The call graph of a case, shared by the temporal and graph features.

Edges run caller to callee. Faults propagate the other way, so throughout this package a service's
upstream is what it calls and its downstream is what calls it, as in ``baselines/rule.py``.
"""

import networkx as nx  # type: ignore[import-untyped]

from cascadelens.features.context import FeatureInput


def call_graph(case: FeatureInput) -> "nx.DiGraph[str] | None":
    """Return the call graph, or None when the case has no usable topology."""
    if not case.topology:
        return None
    graph: nx.DiGraph[str] = nx.DiGraph()
    # Sorted, so results do not depend on the order edges are stored in.
    graph.add_edges_from(
        sorted((edge.caller, edge.callee) for edge in case.topology if edge.caller != edge.callee)
    )
    return graph
