"""Signal features: per service and metric family, from each series against its own baseline.

Scores are computed per series, then a family takes the maximum of its series. Baseline median and
MAD take the median of its series. ``anomaly_duration`` is the share of post-baseline samples past
the threshold, not a count: recording length tracks fault type (docs/data_audit.md), so a count
would leak it. The pre-incident part is defined in ``features/context.py``.

``changepoint`` is the CUSUM statistic of the whole window: the largest absolute cumulative
deviation from the series mean, divided by the baseline scale and the root of the sample count.
"""

import polars as pl

from cascadelens.features.context import CaseContext
from cascadelens.features.simple import FAMILIES, MIN_SCALE, SERIES, Z_CAP

PREFIX = "signal_"
SCORES = ("max_z", "changepoint", "relative_change", "anomaly_duration", "peak_magnitude")
BASELINES = ("baseline_median", "baseline_mad")
STATISTICS = (*BASELINES, *SCORES)
COLUMNS = tuple(f"{PREFIX}{family}_{stat}" for family in FAMILIES for stat in STATISTICS)


def signal_features(context: CaseContext) -> pl.DataFrame:
    """Return one row per service with every signal column, null where there is no data."""
    value = pl.col("value")
    series = list(SERIES)
    threshold = context.signal.z_threshold
    samples = context.samples
    drift = (value - value.mean().over(series)).cum_sum().over(series)
    changepoints = (
        samples.with_columns(drift=drift)
        .group_by(series, maintain_order=True)
        .agg(
            changepoint=(
                pl.col("drift").abs().max() / (pl.col("scale").first() * pl.len().sqrt())
            ).clip(0, Z_CAP)
        )
    )
    per_series = (
        samples.filter(~pl.col("in_baseline"))
        .group_by(series, maintain_order=True)
        .agg(
            family=pl.col("family").first(),
            baseline_median=pl.col("median").first(),
            baseline_mad=pl.col("mad").first(),
            max_z=pl.col("z").max(),
            relative_change=(value.median() - pl.col("median").first()).abs()
            / (pl.col("median").first().abs() + MIN_SCALE),
            anomaly_duration=(pl.col("z") > threshold).mean(),
            peak_magnitude=(value - pl.col("median")).abs().max().log1p(),
        )
        .join(changepoints, on=series, how="left", maintain_order="left")
    )
    table = context.frame()
    for family in FAMILIES:
        part = (
            per_series.filter(pl.col("family") == family)
            .group_by("service", maintain_order=True)
            .agg(
                *[pl.col(stat).median().alias(f"{PREFIX}{family}_{stat}") for stat in BASELINES],
                *[pl.col(stat).max().alias(f"{PREFIX}{family}_{stat}") for stat in SCORES],
            )
        )
        table = table.join(part, on="service", how="left", maintain_order="left")
    return table.select("service", *[pl.col(c).cast(pl.Float64) for c in COLUMNS])
