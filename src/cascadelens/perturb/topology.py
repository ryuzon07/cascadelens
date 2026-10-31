"""Topology perturbations: P6, P7 and P8.

These model an imperfect graph. The benchmark has a static topology, so nothing here claims to model
a changing one.

RE1 records no call graph, so a case with no recorded topology works on the graph inferred with the
Part 6 settings, which is what the clean feature pipeline uses for it. That is why P6 and P7 still
have edges to edit on RE1. The result is stored in ``Case.topology``.
"""

import math
from typing import Literal

import numpy as np
import polars as pl
from pydantic import Field

from cascadelens.config import StrictModel
from cascadelens.data.schema import Case, TopologyEdge
from cascadelens.features.context import FeatureInput, scored_samples
from cascadelens.features.inference import InferenceSettings, infer_edges, service_signals
from cascadelens.features.simple import SignalSettings
from cascadelens.perturb.base import Edges, Outcome, Perturbation, Stage, rounded, with_fields


class EdgeRateParams(StrictModel):
    """Share of the case's edges to change, rounded half up to a count."""

    rate: float = Field(gt=0, le=1)


class InferredParams(StrictModel):
    """How P8 infers a graph.

    ``correlation`` links services by absolute same-time correlation and makes each link two-way.
    ``lagged`` is the Granger-style option: it also tests lags up to ``max_lag_bins``, and the
    service that leads is taken as the callee.
    """

    method: Literal["correlation", "lagged"] = "lagged"
    threshold: float = Field(default=0.6, gt=0, le=1)
    bin_seconds: int = Field(default=15, ge=1)
    max_lag_bins: int = Field(default=3, ge=1)


def infer_from_metrics(case: Case, settings: InferenceSettings) -> Edges | None:
    """Infer a graph from pre-incident samples, through the view that holds no label."""
    view = FeatureInput.from_record(case.record(), case.metrics)
    samples = scored_samples(view, SignalSettings())
    chosen = samples.filter(pl.col("family").is_in(settings.families))
    services, signals = service_signals(chosen, settings.bin_seconds)
    return infer_edges(services, signals, settings)


def working_graph(case: Case) -> Edges | None:
    """Return the recorded graph, or the one the clean pipeline infers."""
    return case.topology if case.topology else infer_from_metrics(case, InferenceSettings())


def _edge(pair: tuple[str, str]) -> TopologyEdge:
    return TopologyEdge(caller=pair[0], callee=pair[1])


def _pairs(edges: Edges) -> list[tuple[str, str]]:
    return sorted({(edge.caller, edge.callee) for edge in edges})


class EdgeDeletion(Perturbation[EdgeRateParams]):
    """P6: delete ``round(rate * edges)`` edges, uniformly at random."""

    id = "P6"
    name = "edge_deletion"
    stage = Stage.TOPOLOGY
    schema = EdgeRateParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Delete edges."""
        pairs = _pairs(working_graph(case) or ())
        count = rounded(self.params.rate * len(pairs))
        gone = set(rng.choice(len(pairs), size=count, replace=False).tolist()) if count else set()
        kept = tuple(_edge(pair) for index, pair in enumerate(pairs) if index not in gone)
        return Outcome(
            with_fields(case, topology=kept or None),
            edges_before=len(pairs),
            edges_removed=count,
        )


class EdgeAddition(Perturbation[EdgeRateParams]):
    """P7: add ``round(rate * edges)`` directed edges between existing services, none a copy."""

    id = "P7"
    name = "edge_addition"
    stage = Stage.TOPOLOGY
    schema = EdgeRateParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Add spurious edges."""
        pairs = _pairs(working_graph(case) or ())
        present = set(pairs)
        services = sorted(set(case.metrics["service"].unique().to_list()))
        open_pairs = [
            (first, second)
            for first in services
            for second in services
            if first != second and (first, second) not in present
        ]
        count = min(rounded(self.params.rate * len(pairs)), len(open_pairs))
        chosen = rng.choice(len(open_pairs), size=count, replace=False).tolist() if count else []
        added = [open_pairs[index] for index in chosen]
        edges = tuple(_edge(pair) for pair in sorted([*pairs, *added]))
        return Outcome(
            with_fields(case, topology=edges or None),
            edges_before=len(pairs),
            edges_added=count,
        )


class InferredTopology(Perturbation[InferredParams]):
    """P8: replace the graph with one inferred from pre-incident metrics, with no ground truth.

    On RE1 with default settings this equals the graph the clean pipeline already infers. It only
    becomes a real change on a benchmark with a recorded graph, or with other settings.
    """

    id = "P8"
    name = "inferred_topology"
    stage = Stage.TOPOLOGY
    schema = InferredParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Infer the graph. ``rng`` is unused: inference is deterministic."""
        settings = InferenceSettings(
            threshold=self.params.threshold,
            bin_seconds=self.params.bin_seconds,
            max_lag_bins=self.params.max_lag_bins if self.params.method == "lagged" else 0,
        )
        edges = infer_from_metrics(case, settings)
        before = set(_pairs(working_graph(case) or ()))
        after = set(_pairs(edges or ()))
        return Outcome(
            with_fields(case, topology=edges),
            edges_before=len(before),
            edges_added=len(after - before),
            edges_removed=len(before - after),
            details={"edges_after": len(after), "method": self.params.method},
        )


def hop_distances(case: Case, source: str) -> dict[str, float]:
    """Return undirected hops from ``source`` in the working graph, inf when unreachable."""
    neighbours: dict[str, set[str]] = {}
    for edge in working_graph(case) or ():
        neighbours.setdefault(edge.caller, set()).add(edge.callee)
        neighbours.setdefault(edge.callee, set()).add(edge.caller)
    services = set(case.metrics["service"].unique().to_list()) | set(neighbours) | {source}
    distances = dict.fromkeys(services, math.inf)
    distances[source] = 0.0
    frontier = {source}
    hops = 0
    while frontier:
        hops += 1
        reached = {n for s in frontier for n in neighbours.get(s, ()) if distances[n] == math.inf}
        for service in reached:
            distances[service] = float(hops)
        frontier = reached
    return distances
