"""A hand-built case: db fails first, and the fault climbs api then web. idle stays quiet."""

import numpy as np
import polars as pl

from cascadelens.config import load_config
from cascadelens.data.schema import METRICS_SCHEMA, Case, TopologyEdge
from cascadelens.features.config import FeaturesConfig
from cascadelens.features.context import FeatureInput

SAMPLES = 400
START = 1000
END = START + SAMPLES - 1
ONSETS = {"db": 200, "api": 220, "web": 240, "idle": None}
EDGES = (("web", "api"), ("api", "db"), ("web", "idle"))


def make_metrics(onsets: dict[str, int | None] | None = None, seed: int = 0) -> pl.DataFrame:
    """Return cpu, latency and a cumulative error counter per service."""
    chosen = onsets if onsets is not None else ONSETS
    rng = np.random.default_rng(seed)
    frames = []
    for service in sorted(chosen):
        onset = chosen[service]
        noise = rng.normal(scale=0.2, size=(3, SAMPLES))
        shift = np.zeros(SAMPLES) if onset is None else (np.arange(SAMPLES) >= onset) * 20.0
        series = {
            "cpu": 10.0 + noise[0] + shift,
            "latency": 50.0 + noise[1] + shift,
            "request-error-total": np.cumsum(1.0 + noise[2]),
        }
        for metric, values in series.items():
            frames.append(
                pl.DataFrame(
                    {
                        "service": [service] * SAMPLES,
                        "metric": [metric] * SAMPLES,
                        "timestamp": np.arange(START, END + 1),
                        "value": values,
                    },
                    schema=METRICS_SCHEMA,
                )
            )
    return pl.concat(frames)


def edges(pairs: tuple[tuple[str, str], ...] = EDGES) -> tuple[TopologyEdge, ...]:
    return tuple(TopologyEdge(caller=caller, callee=callee) for caller, callee in pairs)


def make_input(
    metrics: pl.DataFrame | None = None,
    topology: tuple[TopologyEdge, ...] | None = None,
    known: tuple[str, ...] = tuple(sorted(ONSETS)),
    case_id: str = "case-1",
) -> FeatureInput:
    return FeatureInput(
        case_id=case_id,
        metrics=metrics if metrics is not None else make_metrics(),
        window_start=START,
        window_end=END,
        topology=topology,
        known_services=known,
    )


def make_case(case_id: str, root_cause: str, system: str, inject_time: int = START + 200) -> Case:
    onsets: dict[str, int | None] = {s: None for s in ONSETS}
    onsets[root_cause] = 200
    return Case(
        case_id=case_id,
        suite="suite",
        system=system,
        fault_type="cpu",
        run_id=f"{system}/{root_cause}",
        repetition=1,
        root_cause_service=root_cause,
        inject_time=inject_time,
        window_start=START,
        window_end=END,
        service_names={service: service for service in ONSETS},
        topology=edges(),
        metrics=make_metrics(onsets, seed=sum(map(ord, case_id))),
    )


def config(**overrides: object) -> FeaturesConfig:
    fields: dict[str, object] = {"groups": ["signal", "temporal", "graph", "structural", "mask"]}
    return FeaturesConfig.model_validate({**fields, **overrides})


def row(table: pl.DataFrame, service: str) -> dict[str, float]:
    return table.filter(pl.col("service") == service).to_dicts()[0]


def shipped_config(path: str) -> FeaturesConfig:
    return load_config(path, FeaturesConfig)
