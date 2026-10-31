"""Small generated cases for perturbation tests, with a known chain topology."""

import itertools
from typing import Any

import numpy as np
import polars as pl

from cascadelens.data.schema import METRICS_SCHEMA, Case, TopologyEdge

START = 1000
SAMPLES = 240
METRICS = ("cpu", "mem-used", "latency-50", "latency-90")
SERVICES = tuple(f"svc-{index}" for index in range(8))
CHAIN = tuple(
    TopologyEdge(caller=first, callee=second) for first, second in itertools.pairwise(SERVICES)
)


def make_case(
    number: int = 0, root: str = "svc-0", topology: tuple[TopologyEdge, ...] | None = CHAIN
) -> Case:
    rng = np.random.default_rng(number)
    rows: list[tuple[str, str, int, float]] = []
    for service in SERVICES:
        for metric in METRICS:
            values = 5.0 + rng.normal(size=SAMPLES)
            rows += [(service, metric, START + t, float(values[t])) for t in range(SAMPLES)]
    return Case(
        case_id=f"case-{number}",
        suite="suite",
        system="shop",
        fault_type="cpu",
        run_id=f"run-{number}",
        repetition=0,
        root_cause_service=root,
        inject_time=START + SAMPLES // 2,
        window_start=START,
        window_end=START + SAMPLES - 1,
        topology=topology,
        metrics=pl.DataFrame(rows, schema=METRICS_SCHEMA, orient="row"),
    )


EXAMPLES: dict[str, dict[str, Any]] = {
    "P1": {"rate": 0.3},
    "P2": {"rate": 0.3},
    "P3": {"family": "latency"},
    "P4": {"offset_seconds": 30},
    "P5": {"max_jitter_seconds": 20},
    "P6": {"rate": 0.3},
    "P7": {"rate": 0.3},
    "P8": {"method": "lagged"},
    "P11": {"factor": 5},
    "P12": {"max_offset_seconds": 5, "late_fraction": 0.2},
    "P13": {"mode": "rename", "service_fraction": 0.5},
}
