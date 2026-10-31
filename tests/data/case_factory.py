"""Hand-built cases for tests; ``make_case`` gives a case that passes every check."""

from pathlib import Path
from typing import Any

import polars as pl

from cascadelens.data.schema import METRICS_SCHEMA, Case

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "rcaeval"
FIXTURE_SUITES = ("RE1-OB", "RE1-SS")


Row = tuple[str, str, int, float | None]


def metrics_frame(rows: list[Row]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=METRICS_SCHEMA, orient="row")


def default_rows() -> list[Row]:
    return [
        (service, metric, timestamp, 1.0)
        for service in ("api", "db")
        for metric in ("cpu", "mem")
        for timestamp in (100, 110, 120)
    ]


def make_case(**overrides: object) -> Case:
    fields: dict[str, Any] = {
        "case_id": "re1-ob_db_cpu_1",
        "suite": "RE1-OB",
        "system": "online-boutique",
        "fault_type": "cpu",
        "run_id": "RE1-OB/db_cpu",
        "repetition": 1,
        "root_cause_service": "db",
        "inject_time": 110,
        "window_start": 100,
        "window_end": 120,
        "service_names": {"api": "API", "db": "db"},
        "topology": [{"caller": "api", "callee": "db"}],
        "metrics": metrics_frame(default_rows()),
    }
    return Case(**{**fields, **overrides})
