"""Data perturbations: loss (P1, P2, P3, P11, P12) and schema drift (P13)."""

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import polars as pl
from pydantic import Field

from cascadelens.config import StrictModel
from cascadelens.data.schema import Case
from cascadelens.features.simple import FAMILIES, metric_family
from cascadelens.perturb.base import (
    SERIES,
    Outcome,
    Perturbation,
    Stage,
    null_rows,
    observed,
    rounded,
    with_fields,
)
from cascadelens.perturb.topology import hop_distances

CALIBRATION_STEPS = 60


class RandomLossParams(StrictModel):
    """P1. ``rate`` means a different thing per level.

    ``series``: exactly ``round(rate * series)`` of the (service, metric) series that hold data lose
    every sample, so rate is a share of series.
    ``sample``: each real sample drops independently with probability ``rate``, so rate is an
    expected share of samples.
    """

    rate: float = Field(gt=0, le=1)
    level: Literal["series", "sample"] = "series"


class RandomMetricLoss(Perturbation[RandomLossParams]):
    """P1: uniform loss over series or over samples."""

    id = "P1"
    name = "random_metric_loss"
    stage = Stage.DATA
    schema = RandomLossParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Drop series or samples."""
        metrics = case.metrics
        real = metrics.select(observed(metrics).alias("real"))["real"].to_numpy()
        if self.params.level == "sample":
            return Outcome(
                with_fields(
                    case,
                    metrics=null_rows(
                        metrics, real & (rng.random(metrics.height) < self.params.rate)
                    ),
                )
            )
        series = metrics.filter(observed(metrics)).select(*SERIES).unique().sort(*SERIES)
        count = rounded(self.params.rate * series.height)
        chosen = series[np.sort(rng.choice(series.height, size=count, replace=False)).tolist()]
        marked = metrics.join(
            chosen.with_columns(drop=pl.lit(True)),
            on=list(SERIES),
            how="left",
            maintain_order="left",
        )["drop"].fill_null(False)
        return Outcome(with_fields(case, metrics=null_rows(metrics, marked.to_numpy())))


class FailureLossParams(StrictModel):
    """P2. ``rate`` is the expected share of real samples lost, whatever the boosts are.

    The boosts multiply drop odds: a block near the fault weighs ``near_boost`` and a block after
    the fault ``after_boost``, on top of a weight of 1. The base probability is solved per case so
    the expected share equals ``rate``. Loss comes in blocks of ``block_seconds`` per series,
    because independent gaps on 1 s data would leave every robust statistic unchanged.
    """

    rate: float = Field(gt=0, le=1)
    near_boost: float = Field(default=3.0, ge=1)
    after_boost: float = Field(default=3.0, ge=1)
    near_hops: int = Field(default=1, ge=0)
    block_seconds: int = Field(default=30, ge=1)


def solve_base_rate(sizes: Sequence[float], weights: Sequence[float], rate: float) -> float:
    """Return ``c`` so that the size-weighted mean of ``min(1, c * weight)`` equals ``rate``."""
    share = np.asarray(sizes, dtype=np.float64)
    share = share / share.sum()
    weight = np.asarray(weights, dtype=np.float64)
    low, high = 0.0, 1.0 / float(weight[share > 0].min())
    for _ in range(CALIBRATION_STEPS):
        middle = (low + high) / 2
        if float((share * np.minimum(1.0, middle * weight)).sum()) < rate:
            low = middle
        else:
            high = middle
    return high


class FailureCorrelatedLoss(Perturbation[FailureLossParams]):
    """P2: loss concentrated near the fault and after it.

    This is the one perturbation that reads ground truth, because failures damage telemetry. Near
    means within ``near_hops`` undirected hops of ``root_cause_service`` in the working graph, which
    includes the service itself. After means a block that starts at or after ``inject_time``. Both
    stay inside this function: the output has the same fields as the input.
    """

    id = "P2"
    name = "failure_correlated_loss"
    stage = Stage.DATA
    schema = FailureLossParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Drop blocks with probability rising near and after the fault."""
        params = self.params
        distances = hop_distances(case, case.root_cause_service)
        near = sorted(s for s, hops in distances.items() if hops <= params.near_hops)
        block = (pl.col("timestamp") - case.window_start) // params.block_seconds
        start = case.window_start + block * params.block_seconds
        frame = case.metrics.with_columns(
            block=block,
            near=pl.col("service").is_in(near),
            after=start >= case.inject_time,
            real=observed(case.metrics),
        )
        cells = (
            frame.filter("real").group_by("near", "after").agg(size=pl.len()).sort("near", "after")
        )
        if cells.is_empty():
            return Outcome(case, details={"base_rate": 0.0})

        def weight(is_near: bool, is_after: bool) -> float:
            return (params.near_boost if is_near else 1.0) * (
                params.after_boost if is_after else 1.0
            )

        weights = [weight(n, a) for n, a in cells.select("near", "after").iter_rows()]
        base = solve_base_rate(cells["size"].to_list(), weights, params.rate)
        keys = frame.select(*SERIES, "block").unique(maintain_order=True)
        draws = keys.with_columns(draw=pl.Series(rng.random(keys.height)))
        probability = (
            pl.lit(base)
            * pl.when("near").then(params.near_boost).otherwise(1.0)
            * pl.when("after").then(params.after_boost).otherwise(1.0)
        ).clip(0.0, 1.0)
        joined = frame.join(draws, on=[*SERIES, "block"], how="left", maintain_order="left")
        marked = joined.with_columns(drop=(pl.col("draw") < probability) & pl.col("real"))
        return Outcome(
            with_fields(case, metrics=null_rows(case.metrics, marked["drop"].to_numpy())),
            details={
                "base_rate": base,
                "near_services": len(near),
                "near_rate": _share(marked.filter("near")),
                "far_rate": _share(marked.filter(~pl.col("near"))),
                "before_rate": _share(marked.filter(~pl.col("after"))),
                "after_rate": _share(marked.filter("after")),
            },
        )


def _share(frame: pl.DataFrame) -> float | None:
    real = frame.filter("real")
    return None if real.is_empty() else float(real["drop"].mean())  # type: ignore[arg-type]


class ModalityParams(StrictModel):
    """P3. Logs and traces are stretch modalities."""

    family: Literal["error", "latency", "cpu", "memory", "disk", "network", "traffic"] = "latency"
    modality: Literal["metrics", "logs", "traces"] = "metrics"


class WholeModalityLoss(Perturbation[ModalityParams]):
    """P3: drop one metric family for every service."""

    id = "P3"
    name = "whole_modality_loss"
    stage = Stage.DATA
    schema = ModalityParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Drop the family."""
        if self.params.modality != "metrics":
            raise NotImplementedError(
                f"P3 on {self.params.modality} is a stretch modality: RE1 has metrics only"
            )
        names = case.metrics["metric"].unique().to_list()
        chosen = [name for name in names if metric_family(name) == self.params.family]
        marked = case.metrics["metric"].is_in(chosen).to_numpy()
        return Outcome(with_fields(case, metrics=null_rows(case.metrics, marked)))


class DownsampleParams(StrictModel):
    """P11. Keep one instant in ``factor``."""

    factor: int = Field(ge=2)


class SamplingRateDegradation(Perturbation[DownsampleParams]):
    """P11: keep every ``factor``-th instant of the case, and mark the others missing.

    Rows stay, so mask features see the coarser rate. The first instant is always kept.
    """

    id = "P11"
    name = "sampling_rate_degradation"
    stage = Stage.DATA
    schema = DownsampleParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Thin every series on the same instants."""
        instants = case.metrics["timestamp"].unique().sort()
        kept = instants.gather_every(self.params.factor)
        marked = (~case.metrics["timestamp"].is_in(kept.to_list())).to_numpy()
        return Outcome(with_fields(case, metrics=null_rows(case.metrics, marked)))


class DelayParams(StrictModel):
    """P12. Clock offsets are whole seconds, uniform on the range. Late samples are a share."""

    max_offset_seconds: int = Field(default=15, ge=0)
    late_fraction: float = Field(default=0.1, ge=0, lt=1)


class DelayedSeries(Perturbation[DelayParams]):
    """P12: a clock offset per service, and late samples missing from the window.

    Offsets move timestamps, so a service's rows can run past the window. Those rows stay.
    """

    id = "P12"
    name = "delayed_series"
    stage = Stage.DATA
    schema = DelayParams

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Offset clocks, then drop late samples."""
        limit = self.params.max_offset_seconds
        services = sorted(case.metrics["service"].unique().to_list())
        offsets = dict(
            zip(services, rng.integers(-limit, limit + 1, size=len(services)).tolist(), strict=True)
        )
        shifted = case.metrics.with_columns(
            pl.col("timestamp") + pl.col("service").replace_strict(offsets, return_dtype=pl.Int64)
        )
        real = shifted.select(observed(shifted).alias("real"))["real"].to_numpy()
        late = real & (rng.random(shifted.height) < self.params.late_fraction)
        return Outcome(
            with_fields(case, metrics=null_rows(shifted, late)), details={"offsets": offsets}
        )


class DriftParams(StrictModel):
    """P13. ``service_fraction`` of services change, rounded half up to a count."""

    mode: Literal["rename", "rescale"] = "rename"
    service_fraction: float = Field(default=0.3, gt=0, le=1)
    factor: float = Field(default=1000.0, gt=0)


class MetricDrift(Perturbation[DriftParams]):
    """P13: permute metric names within a family, or rescale units.

    ``rename`` applies a cycle to each family that has two or more metrics on a chosen service, so
    every name in it moves. ``rescale`` multiplies a chosen service's series by ``factor``. Both are
    recorded in the manifest and can be undone from it.
    """

    id = "P13"
    name = "metric_renaming_unit_drift"
    stage = Stage.DATA
    schema = DriftParams

    def _chosen(self, case: Case, rng: np.random.Generator) -> list[str]:
        services = sorted(case.metrics["service"].unique().to_list())
        count = rounded(self.params.service_fraction * len(services))
        return sorted(services[i] for i in rng.choice(len(services), size=count, replace=False))

    def _rename(
        self, case: Case, chosen: list[str], rng: np.random.Generator
    ) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
        mapping: list[dict[str, Any]] = []
        pairs = case.metrics.filter(pl.col("service").is_in(chosen)).select(*SERIES).unique()
        for service, metrics in sorted(
            pairs.group_by("service").agg(pl.col("metric").sort()).iter_rows()
        ):
            for family in FAMILIES:
                names = [m for m in metrics if metric_family(m) == family]
                if len(names) < 2:
                    continue
                order = rng.permutation(len(names)).tolist()
                cycle = [names[index] for index in order]
                for position, name in enumerate(cycle):
                    target = cycle[(position + 1) % len(cycle)]
                    mapping.append({"service": service, "from": name, "to": target})
        if not mapping:
            return case.metrics, mapping
        table = pl.DataFrame(
            mapping, schema={"service": pl.String, "from": pl.String, "to": pl.String}
        )
        metrics_table = (
            case.metrics.join(
                table.rename({"from": "metric"}), on=list(SERIES), how="left", maintain_order="left"
            )
            .with_columns(pl.coalesce("to", "metric").alias("metric"))
            .select(case.metrics.columns)
            .sort(*SERIES, "timestamp", maintain_order=True)
        )
        return metrics_table, mapping

    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Rename or rescale."""
        chosen = self._chosen(case, rng)
        if self.params.mode == "rename":
            metrics, mapping = self._rename(case, chosen, rng)
            return Outcome(with_fields(case, metrics=metrics), details={"mapping": mapping})
        scaled = case.metrics.with_columns(
            pl.when(pl.col("service").is_in(chosen))
            .then(pl.col("value") * self.params.factor)
            .otherwise(pl.col("value"))
            .alias("value")
        )
        details = {"mapping": [{"service": s, "factor": self.params.factor} for s in chosen]}
        return Outcome(with_fields(case, metrics=scaled), details=details)
