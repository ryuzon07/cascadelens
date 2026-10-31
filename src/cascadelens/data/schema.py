"""The unified Case schema (PRD section 6.2).

``inject_time`` is GROUND TRUTH ONLY. It exists so that labels can be audited and so that the
window-leak probe can be scored. It must never reach a model, a feature or an onset detector
(PRD section 8.1); downstream code must read metrics and window bounds only.
"""

import re

import polars as pl
from pydantic import ConfigDict, Field, field_validator

from cascadelens.config import StrictModel

METRICS_SCHEMA = pl.Schema(
    {
        "service": pl.String,
        "metric": pl.String,
        "timestamp": pl.Int64,
        "value": pl.Float64,
    }
)

CASE_ID_PATTERN = r"^[a-z0-9][a-z0-9_.-]*$"


def normalize_service_name(name: str) -> str:
    """Return the canonical form of a service name: lower case, with ``-`` as the separator."""
    normalized = re.sub(r"[\s_]+", "-", name.strip().lower())
    if not normalized:
        raise ValueError(f"Service name {name!r} is empty after normalization")
    return normalized


class TopologyEdge(StrictModel):
    """One directed call between two normalized service names."""

    caller: str = Field(min_length=1)
    callee: str = Field(min_length=1)


class CaseRecord(StrictModel):
    """Everything about a case except its metric samples: one row of the cases index."""

    case_id: str = Field(pattern=CASE_ID_PATTERN, description="Globally unique")
    suite: str = Field(min_length=1, description="Benchmark subset, such as RE1-OB")
    system: str = Field(min_length=1, description="Application name")
    fault_type: str = Field(min_length=1, description="cpu, mem, delay, loss, etc.")
    run_id: str = Field(min_length=1, description="Injection run; the primary group key")
    repetition: int = Field(ge=0, description="The dataset's repetition number within run_id")
    root_cause_service: str = Field(min_length=1, description="Normalized ground-truth label")
    inject_time: int = Field(description="Unix seconds. GROUND TRUTH ONLY; never a model input")
    window_start: int = Field(description="Unix seconds; first sample of the analysis window")
    window_end: int = Field(description="Unix seconds; last sample of the analysis window")
    service_names: dict[str, str] = Field(
        default_factory=dict, description="Normalized service name to the name in the raw data"
    )
    topology: tuple[TopologyEdge, ...] | None = Field(
        default=None, description="Call graph; None means the source provides none"
    )
    logs: str | None = Field(default=None, description="Stretch modality; unused for now")
    traces: str | None = Field(default=None, description="Stretch modality; unused for now")


class Case(CaseRecord):
    """A case with its metrics, as a long table of (service, metric, timestamp, value).

    A missing sample is a row whose ``value`` is null, never NaN, so "missing" has one spelling.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid", frozen=True)

    metrics: pl.DataFrame

    @field_validator("metrics")
    @classmethod
    def _metrics_match_schema(cls, metrics: pl.DataFrame) -> pl.DataFrame:
        if metrics.schema != METRICS_SCHEMA:
            raise ValueError(f"metrics must have schema {METRICS_SCHEMA}, got {metrics.schema}")
        return metrics

    def record(self) -> CaseRecord:
        """Return this case without its metrics."""
        return CaseRecord.model_validate(self.model_dump(exclude={"metrics"}))

    def equals(self, other: "Case") -> bool:
        """Compare field by field; ``==`` is unusable because a DataFrame has no truth value."""
        return self.record() == other.record() and self.metrics.equals(other.metrics)
