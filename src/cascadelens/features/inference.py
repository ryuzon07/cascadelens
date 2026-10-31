"""Infer a call graph from one case's metrics, for benchmarks that record none.

RE1 has no topology. This builds a correlation graph from the pre-incident part of the window only,
so the fault's own spread cannot become structure and nothing here depends on what failed. It reads
no label, no injection time and no other case.

Each service becomes one signal: the mean signed robust z-score of its series in ``families``, in
time bins. Latency is the default because it is what travels along calls. Repetitions of one
deployment share about 40 to 58% of their latency edges, against 0 to 15% when every metric is
averaged at 5 s, and that label-free stability is how the settings were chosen. Two services are
linked when their absolute correlation reaches ``threshold``, at lag 0 or at any lag up
to ``max_lag_bins``. Direction comes from lead and lag. The service that leads is taken as the
callee, because faults travel from callee to caller. When neither clearly leads, the edge runs both
ways, so no service name ever decides a direction.

This graph shows statistical association, not real calls. Features built from it carry
``graph_topology_inferred`` = 1.
"""

import numpy as np
import numpy.typing as npt
import polars as pl

from cascadelens.config import StrictModel
from cascadelens.data.schema import TopologyEdge
from cascadelens.features.simple import Z_CAP

MINIMUM_BINS = 8
Matrix = npt.NDArray[np.float64]


class InferenceSettings(StrictModel):
    """How to infer a graph. Set by stability across repetitions, never by root-cause accuracy."""

    families: tuple[str, ...] = ("latency",)
    bin_seconds: int = 15
    threshold: float = 0.6
    max_lag_bins: int = 3
    direction_margin: float = 0.05


def service_signals(samples: pl.DataFrame, bin_seconds: int) -> tuple[list[str], Matrix]:
    """Return services and a time-by-service matrix from the pre-incident samples."""
    baseline = samples.filter("in_baseline")
    if baseline.is_empty():
        return [], np.empty((0, 0))
    signed = (pl.col("value") - pl.col("median")) / pl.col("scale")
    binned = (
        baseline.with_columns(
            signal=signed.clip(-Z_CAP, Z_CAP),
            bin=(pl.col("timestamp") - pl.col("timestamp").min()) // bin_seconds,
        )
        .group_by("service", "bin", maintain_order=True)
        .agg(pl.col("signal").mean())
        .pivot(on="service", index="bin", values="signal")
        .sort("bin")
        .fill_null(0.0)
    )
    services = sorted(column for column in binned.columns if column != "bin")
    return services, binned.select(services).to_numpy().astype(np.float64)


def _standardized(signals: Matrix) -> tuple[Matrix, npt.NDArray[np.bool_]]:
    centered = signals - signals.mean(axis=0)
    deviation = signals.std(axis=0)
    varying = deviation > 0
    scaled = np.divide(centered, deviation, out=np.zeros_like(centered), where=varying)
    return scaled, varying


def infer_edges(
    services: list[str], signals: Matrix, settings: InferenceSettings
) -> tuple[TopologyEdge, ...] | None:
    """Return caller-to-callee edges, or None when the signals cannot support a graph."""
    rows = signals.shape[0]
    if len(services) < 2 or rows < MINIMUM_BINS:
        return None
    unit, varying = _standardized(signals)
    same_time = np.abs(unit.T @ unit) / rows
    leads = np.zeros_like(same_time)
    for lag in range(1, min(settings.max_lag_bins, rows - MINIMUM_BINS + 1) + 1):
        overlap = rows - lag
        leads = np.maximum(leads, np.abs(unit[:-lag].T @ unit[lag:]) / overlap)
    edges: set[tuple[str, str]] = set()
    for first in range(len(services)):
        for second in range(first + 1, len(services)):
            if not (varying[first] and varying[second]):
                continue
            ahead, behind = leads[first, second], leads[second, first]
            if max(same_time[first, second], ahead, behind) < settings.threshold:
                continue
            if ahead - behind > settings.direction_margin:
                edges.add((services[second], services[first]))
            elif behind - ahead > settings.direction_margin:
                edges.add((services[first], services[second]))
            else:
                edges |= {
                    (services[first], services[second]),
                    (services[second], services[first]),
                }
    if not edges:
        return None
    return tuple(TopologyEdge(caller=caller, callee=callee) for caller, callee in sorted(edges))


def infer_topology(
    samples: pl.DataFrame, settings: InferenceSettings
) -> tuple[TopologyEdge, ...] | None:
    """Infer the graph of one case from its scored samples."""
    chosen = samples.filter(pl.col("family").is_in(settings.families))
    services, signals = service_signals(chosen, settings.bin_seconds)
    return infer_edges(services, signals, settings)
