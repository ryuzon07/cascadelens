"""Minimal per-service signal features for the simple baselines.

The functions here take metrics and window bounds only, so ``inject_time`` cannot reach them. The
pre-incident baseline is the first ``baseline_fraction`` of the window. Keep it well short of the
middle: the injection sits near 50% of every recording (see docs/data_audit.md), so a split at the
middle would turn the window into a copy of ``inject_time``.

Durations are fractions of the window, because recording length in seconds tracks fault type.
Metrics ending in ``-total`` are cumulative counters and are differenced before scoring.

There are no service names or IDs, no graph features and no mask features. Families a service has
no data for stay null.
"""

import polars as pl

from cascadelens.config import StrictModel

FAMILIES = ("error", "latency", "cpu", "memory", "disk", "network", "traffic", "other")
FAMILY_WORDS = (
    ("error", ("error",)),
    ("latency", ("latency",)),
    ("cpu", ("cpu",)),
    ("memory", ("mem",)),
    ("disk", ("disk", "-fs-")),
    ("network", ("network",)),
    ("traffic", ("load", "request", "bytes")),
)
STATISTICS = ("max_z", "relative_change", "anomaly_duration", "peak_magnitude")
FEATURE_COLUMNS = tuple(f"{family}_{stat}" for family in FAMILIES for stat in STATISTICS)
SERIES = ("service", "metric")
MAD_TO_SIGMA = 1.4826
MIN_SCALE = 1e-9
RELATIVE_SCALE_FLOOR = 0.01
Z_CAP = 1000.0


class SignalSettings(StrictModel):
    """How anomalies are scored. Fixed in config, never tuned on a test fold."""

    baseline_fraction: float = 0.25
    z_threshold: float = 3.0
    min_consecutive: int = 5


def metric_family(metric: str) -> str:
    """Return the family of a metric name, using the first matching word."""
    for family, words in FAMILY_WORDS:
        if any(word in metric for word in words):
            return family
    return "other"


def _scored_samples(
    metrics: pl.DataFrame, window_start: int, window_end: int, settings: SignalSettings
) -> pl.DataFrame:
    """Return the samples after the baseline, with robust z-scores from the baseline."""
    value = pl.col("value")
    series = list(SERIES)
    cut = window_start + settings.baseline_fraction * (window_end - window_start)
    rates = (
        metrics.filter(value.is_finite())
        .sort([*series, "timestamp"])
        .with_columns(
            pl.when(pl.col("metric").str.ends_with("-total"))
            .then(value.diff().over(series))
            .otherwise(value)
            .alias("value")
        )
        .drop_nulls("value")
    )
    baseline = (
        rates.filter(pl.col("timestamp") < cut)
        .group_by(series)
        .agg(median=value.median(), mad=(value - value.median()).abs().median())
    )
    floored = pl.max_horizontal(
        MAD_TO_SIGMA * pl.col("mad"), RELATIVE_SCALE_FLOOR * pl.col("median").abs()
    )
    # A flat zero baseline has no scale. Like RobustScaler, fall back to 1, because a tiny scale
    # would send every small change to the cap and tie most services.
    scale = pl.when(floored > 0).then(floored).otherwise(1.0)
    return (
        rates.filter(pl.col("timestamp") >= cut)
        .join(baseline, on=series, how="inner")
        .with_columns(z=((value - pl.col("median")).abs() / scale).clip(0, Z_CAP))
        .with_columns(position=(pl.col("timestamp") - cut) / max(window_end - cut, 1.0))
    )


def _family_columns(samples: pl.DataFrame, threshold: float) -> pl.DataFrame:
    value = pl.col("value")
    per_series = (
        samples.group_by(SERIES)
        .agg(
            max_z=pl.col("z").max(),
            relative_change=(value.median() - pl.col("median").first()).abs()
            / (pl.col("median").first().abs() + MIN_SCALE),
            anomaly_duration=(pl.col("z") > threshold).mean(),
            peak_magnitude=(value - pl.col("median")).abs().max().log1p(),
        )
        .with_columns(family=pl.col("metric").map_elements(metric_family, return_dtype=pl.String))
    )
    return (
        per_series.group_by("service", "family")
        .agg(pl.col(stat).max() for stat in STATISTICS)
        .unpivot(index=["service", "family"], on=list(STATISTICS), variable_name="statistic")
        .with_columns(column=pl.concat_str("family", "statistic", separator="_"))
        .pivot(on="column", index="service", values="value")
    )


def _onsets(samples: pl.DataFrame, settings: SignalSettings) -> pl.DataFrame:
    """Return, per service, the earliest position of a sustained anomaly in any metric."""
    run = settings.min_consecutive
    sustained = (
        (pl.col("z") > settings.z_threshold)
        .cast(pl.Int32)
        .rolling_sum(run)
        .over(list(SERIES), order_by="timestamp")
    )
    return (
        samples.with_columns(sustained=sustained)
        .filter(pl.col("sustained") == run)
        .group_by("service")
        .agg(onset=pl.col("position").min())
    )


def service_summary(
    metrics: pl.DataFrame, window_start: int, window_end: int, settings: SignalSettings
) -> pl.DataFrame:
    """Return one row per service of ``metrics``, sorted by service.

    Columns are ``service``, ``onset`` (0 to 1, null when no sustained anomaly), ``max_z``, and
    ``FEATURE_COLUMNS``. Services with no scorable series are kept with nulls.
    """
    services = metrics.select("service").unique().sort("service")
    samples = _scored_samples(metrics, window_start, window_end, settings)
    peaks = samples.group_by("service").agg(max_z=pl.col("z").max())
    summary = (
        services.join(_onsets(samples, settings), on="service", how="left")
        .join(peaks, on="service", how="left")
        .join(_family_columns(samples, settings.z_threshold), on="service", how="left")
    )
    absent = [column for column in FEATURE_COLUMNS if column not in summary.columns]
    return summary.with_columns(pl.lit(None, dtype=pl.Float64).alias(c) for c in absent).select(
        "service",
        pl.col("onset").cast(pl.Float64),
        pl.col("max_z").cast(pl.Float64),
        *[pl.col(column).cast(pl.Float64) for column in FEATURE_COLUMNS],
    )
