"""What feature code may see about a case, and the signal shared by the groups.

``FeatureInput`` is the only way case data reaches a feature group. It has no field for the label
or the injection time, so a group cannot read them.

The window is the first and last timestamp of the recording, computed without ``inject_time``.
It still depends on the injection indirectly: RCAEval recorded about as long before the injection
as after it, so the injection sits near the middle of the window (docs/data_audit.md). The
pre-incident part is therefore the first ``baseline_fraction`` of the window, set in config and
kept well short of the middle. The split reads the window bounds only. It never reads the injection
time, and a shifted window would shift it.
"""

from dataclasses import dataclass

import polars as pl

from cascadelens.config import StrictModel
from cascadelens.data.schema import CaseRecord, TopologyEdge
from cascadelens.features.simple import (
    MAD_TO_SIGMA,
    RELATIVE_SCALE_FLOOR,
    SERIES,
    Z_CAP,
    SignalSettings,
    metric_family,
)

KEYS = ("case_id", "service")


class FeatureError(Exception):
    """A feature table or build is not valid."""


class GraphSettings(StrictModel):
    """Graph feature settings."""

    pagerank_alpha: float = 0.85


class MaskSettings(StrictModel):
    """Missingness feature settings."""

    stop_gap_seconds: int = 30


@dataclass(frozen=True)
class FeatureInput:
    """The fields of a case that features may read."""

    case_id: str
    metrics: pl.DataFrame
    window_start: int
    window_end: int
    topology: tuple[TopologyEdge, ...] | None
    known_services: tuple[str, ...]

    @classmethod
    def from_record(cls, record: CaseRecord, metrics: pl.DataFrame) -> "FeatureInput":
        """Copy the allowed fields from an index row."""
        return cls(
            case_id=record.case_id,
            metrics=metrics,
            window_start=record.window_start,
            window_end=record.window_end,
            topology=record.topology,
            known_services=tuple(sorted(record.service_names)),
        )


def all_services(case: FeatureInput) -> list[str]:
    """Return every service known to the case, sorted, even one with no telemetry left."""
    found = set(case.metrics["service"].unique().to_list()) | set(case.known_services)
    for edge in case.topology or ():
        found |= {edge.caller, edge.callee}
    return sorted(found)


def scored_samples(case: FeatureInput, settings: SignalSettings) -> pl.DataFrame:
    """Return every finite sample with its series baseline and absolute robust z-score.

    Cumulative ``-total`` counters are differenced first. Baselines use samples before the cut
    only. A series with no baseline sample is dropped, so its features stay NaN.
    """
    value = pl.col("value")
    series = list(SERIES)
    span = case.window_end - case.window_start
    cut = case.window_start + settings.baseline_fraction * span
    metrics = case.metrics["metric"].unique().to_list()
    families = pl.DataFrame(
        {"metric": metrics, "family": [metric_family(metric) for metric in metrics]},
        schema={"metric": pl.String, "family": pl.String},
    )
    rates = (
        case.metrics.filter(value.is_finite())
        .sort([*series, "timestamp"])
        .with_columns(
            pl.when(pl.col("metric").str.ends_with("-total"))
            .then(value.diff().over(series))
            .otherwise(value)
            .alias("value")
        )
        .drop_nulls("value")
        .with_columns(in_baseline=pl.col("timestamp") < cut)
    )
    baseline = (
        rates.filter("in_baseline")
        .group_by(series, maintain_order=True)
        .agg(median=value.median(), mad=(value - value.median()).abs().median())
    )
    floored = pl.max_horizontal(
        MAD_TO_SIGMA * pl.col("mad"), RELATIVE_SCALE_FLOOR * pl.col("median").abs()
    )
    return (
        rates.join(baseline, on=series, how="inner", maintain_order="left")
        .join(families, on="metric", how="left", maintain_order="left")
        .with_columns(scale=pl.when(floored > 0).then(floored).otherwise(1.0))
        .with_columns(z=((value - pl.col("median")).abs() / pl.col("scale")).clip(0, Z_CAP))
        .with_columns(position=(pl.col("timestamp") - cut) / max(case.window_end - cut, 1.0))
    )


def onsets(samples: pl.DataFrame, settings: SignalSettings) -> pl.DataFrame:
    """Return, per service, the earliest position of a sustained anomaly after the baseline."""
    run = settings.min_consecutive
    after = samples.filter(~pl.col("in_baseline"))
    sustained = (
        (pl.col("z") > settings.z_threshold).cast(pl.Int32).rolling_sum(run).over(list(SERIES))
    )
    return (
        after.with_columns(sustained=sustained)
        .filter(pl.col("sustained") == run)
        .group_by("service", maintain_order=True)
        .agg(onset=pl.col("position").min())
    )


@dataclass(frozen=True)
class CaseContext:
    """One case with the work its groups share."""

    case: FeatureInput
    services: list[str]
    samples: pl.DataFrame
    onsets: dict[str, float]
    signal: SignalSettings
    graph: GraphSettings
    mask: MaskSettings

    @classmethod
    def build(
        cls,
        case: FeatureInput,
        signal: SignalSettings,
        graph: GraphSettings,
        mask: MaskSettings,
    ) -> "CaseContext":
        """Score the case once."""
        samples = scored_samples(case, signal)
        found = onsets(samples, signal)
        return cls(
            case=case,
            services=all_services(case),
            samples=samples,
            onsets=dict(found.iter_rows()),
            signal=signal,
            graph=graph,
            mask=mask,
        )

    def frame(self) -> pl.DataFrame:
        """Return the single-column frame groups join onto."""
        return pl.DataFrame({"service": self.services}, schema={"service": pl.String})
