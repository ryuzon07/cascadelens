"""Convert between our long metrics table and the wide table RCAEval expects.

RCAEval takes one ``time`` column and one column per series, named ``<service>_<metric>``.
Normalized service names never hold ``_``, but the map is kept anyway, so a column never has to be
parsed to find its service.

The frame stays in Polars here. Pandas appears only inside ``call_library``, so everything else
works without the ``published`` extra.
"""

from dataclasses import dataclass
from typing import Any

import polars as pl

from cascadelens.baselines.base import BaselineError
from cascadelens.data.schema import METRICS_SCHEMA

TIME = "time"
# The series closest to RCAEval's simple_data.csv, which is derived and absent from our cases.
KEY_METRICS = (
    "container-cpu-usage-seconds-total",
    "container-memory-usage-bytes",
    "istio-latency-90",
    "istio-request-total",
    "istio-error-total",
)
ONLINE_BOUTIQUE_METRICS = ("cpu", "mem", "latency", "load", "error")


@dataclass(frozen=True)
class WideFrame:
    """A wide table, with the service and metric behind each series column."""

    frame: pl.DataFrame
    series: dict[str, tuple[str, str]]

    def service_of(self, column: str) -> str:
        """Return the exact service name of a column."""
        return self.series[column][0]


def key_metrics(metrics: pl.DataFrame) -> pl.DataFrame:
    """Keep a small metric set chosen by name alone, with cumulative counters differenced.

    Node exporters and sidecar internals drop out, because they have none of these metrics.
    Their services then stay unranked. The choice reads no label and no anomaly time.
    """
    value = pl.col("value")
    return (
        metrics.filter(pl.col("metric").is_in((*KEY_METRICS, *ONLINE_BOUTIQUE_METRICS)))
        .sort("service", "metric", "timestamp")
        .with_columns(
            pl.when(pl.col("metric").str.ends_with("-total"))
            .then(value.diff().over("service", "metric"))
            .otherwise(value)
            .alias("value")
        )
    )


def to_wide(metrics: pl.DataFrame) -> WideFrame:
    """Pivot a long table to one row per timestamp. Columns are sorted; gaps stay null."""
    if metrics.is_empty():
        raise BaselineError("A case has no metrics")
    named = metrics.with_columns(column=pl.concat_str("service", "metric", separator="_"))
    pairs = named.select("column", "service", "metric").unique().sort("column")
    if pairs["column"].n_unique() != pairs.height:
        raise BaselineError("Two series map to one column name")
    series = {row[0]: (row[1], row[2]) for row in pairs.iter_rows()}
    frame = (
        named.pivot(on="column", index="timestamp", values="value")
        .rename({"timestamp": TIME})
        .sort(TIME)
        .select(TIME, *sorted(series))
    )
    return WideFrame(frame, series)


def to_long(wide: WideFrame) -> pl.DataFrame:
    """Invert ``to_wide``, keeping nulls as missing samples."""
    frames = [
        wide.frame.select(
            pl.lit(wide.series[column][0]).alias("service"),
            pl.lit(wide.series[column][1]).alias("metric"),
            pl.col(TIME).alias("timestamp"),
            pl.col(column).alias("value"),
        )
        for column in wide.frame.columns
        if column != TIME
    ]
    return pl.concat(frames).cast(METRICS_SCHEMA).sort("service", "metric", "timestamp")


def fill_gaps(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill gaps as RCAEval's ``main.py`` does: infinities to null, forward-fill, then 0."""
    return frame.with_columns(
        pl.when(pl.col(column).is_infinite())
        .then(None)
        .otherwise(pl.col(column))
        .fill_nan(None)
        .forward_fill()
        .fill_null(0.0)
        .alias(column)
        for column in frame.columns
        if column != TIME
    )


def call_library(method: str, frame: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
    """Call ``RCAEval.e2e.<method>`` on a frame. This is the one place the library is touched."""
    try:
        from RCAEval import e2e
    except ImportError as error:
        raise BaselineError("RCAEval is not installed; run `uv sync --extra published`") from error
    if not hasattr(e2e, method):
        raise BaselineError(f"RCAEval has no method {method!r} in this environment")
    result: dict[str, Any] = getattr(e2e, method)(frame.to_pandas(), **kwargs)
    return result
