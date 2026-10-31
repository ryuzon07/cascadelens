"""Missingness features: what telemetry a service has, never filled in.

These columns are never NaN: they describe the absence that other groups leave as NaN. A service
with no telemetry at all has every mask at 0, a missing fraction of 1, and counts as stopped.
"""

import polars as pl

from cascadelens.features.context import CaseContext
from cascadelens.features.simple import FAMILIES, metric_family

PREFIX = "mask_"
MISSING_COLUMN = f"{PREFIX}missing_fraction"
STOPPED_COLUMN = f"{PREFIX}stopped_early"
AVAILABLE_COLUMNS = tuple(f"{PREFIX}{family}_available" for family in FAMILIES)
COLUMNS = (*AVAILABLE_COLUMNS, MISSING_COLUMN, STOPPED_COLUMN)


def mask_features(context: CaseContext) -> pl.DataFrame:
    """Return one row per service with availability, missing share and the stopped flag."""
    case = context.case
    metrics = case.metrics
    instants = metrics["timestamp"].n_unique()
    observed = metrics.filter(pl.col("value").is_finite())
    pairs = metrics.select("service", "metric").unique(maintain_order=True)
    expected = pairs.group_by("service", maintain_order=True).len("series")
    seen = observed.group_by("service", maintain_order=True).agg(
        samples=pl.len(), last_seen=pl.col("timestamp").max()
    )
    limit = case.window_end - context.mask.stop_gap_seconds
    table = (
        context.frame()
        .join(expected, on="service", how="left", maintain_order="left")
        .join(seen, on="service", how="left", maintain_order="left")
        .with_columns(
            (1 - pl.col("samples").fill_null(0) / (pl.col("series") * instants))
            .fill_null(1.0)
            .alias(MISSING_COLUMN),
            (pl.col("last_seen") < limit).fill_null(True).cast(pl.Float64).alias(STOPPED_COLUMN),
        )
    )
    live = observed.select("service", "metric").unique(maintain_order=True)
    names = live["metric"].unique().to_list()
    families = dict(zip(names, map(metric_family, names), strict=True))
    present = {(service, families[metric]) for service, metric in live.iter_rows()}
    for family, column in zip(FAMILIES, AVAILABLE_COLUMNS, strict=True):
        flags = [float((service, family) in present) for service in context.services]
        table = table.with_columns(pl.Series(column, flags, dtype=pl.Float64))
    return table.select("service", *[pl.col(c).cast(pl.Float64) for c in COLUMNS])
