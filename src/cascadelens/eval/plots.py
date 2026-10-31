"""Matplotlib figures, saved as PNG and SVG with no styling dependency.

Figures use the object interface, so no global backend state is touched. Hues follow a fixed order
and never depend on rank; identity is always also carried by a legend. Output is byte-stable for
the same input and library versions.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import polars as pl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from cascadelens.eval.calibration import RiskCoverage
from cascadelens.eval.errors import EvalError

Numbers = npt.ArrayLike
LegendLocation = Literal["best", "upper left", "lower right"]
SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED_INK = "#52514e"
GRID = "#e4e3df"
LINE_WIDTH = 1.6
BAND_ALPHA = 0.16
FIGURE_SIZE = (7.0, 4.2)
DPI = 200


@dataclass(frozen=True)
class CurveSeries:
    """One method on a robustness plot."""

    label: str
    x: Numbers
    y: Numbers
    lower: Numbers
    upper: Numbers


@dataclass(frozen=True)
class PowerCurve:
    """Power of one scenario over true differences, as fractions."""

    label: str
    difference: Numbers
    power: Numbers


def _axes(title: str, xlabel: str, ylabel: str) -> tuple[Figure, Axes]:
    figure = Figure(figsize=FIGURE_SIZE, facecolor=SURFACE, layout="constrained")
    axes = figure.add_subplot()
    axes.set_facecolor(SURFACE)
    axes.set_title(title, color=INK, loc="left", fontsize=11)
    axes.set_xlabel(xlabel, color=MUTED_INK)
    axes.set_ylabel(ylabel, color=MUTED_INK)
    axes.grid(visible=True, color=GRID, linewidth=0.7)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED_INK, length=0)
    for side, spine in axes.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(GRID)
    return figure, axes


def _legend(axes: Axes, location: LegendLocation = "best") -> None:
    axes.legend(frameon=False, labelcolor=MUTED_INK, fontsize=9, loc=location)


def _color(index: int) -> str:
    if index >= len(SERIES_COLORS):
        raise EvalError(f"At most {len(SERIES_COLORS)} series fit one chart; use small multiples")
    return SERIES_COLORS[index]


def save_figure(figure: Figure, stem: Path) -> tuple[Path, Path]:
    """Save ``stem`` as PNG and SVG and return both paths."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    png, svg = stem.with_suffix(".png"), stem.with_suffix(".svg")
    with mpl.rc_context({"svg.hashsalt": "cascadelens"}):
        figure.savefig(png, dpi=DPI, metadata={"Software": None})
        figure.savefig(svg, metadata={"Date": None, "Creator": None})
    return png, svg


def robustness_curves(
    series: Sequence[CurveSeries], xlabel: str, ylabel: str, tag: str, stem: Path
) -> tuple[Path, Path]:
    """Plot a metric against perturbation intensity with CI bands.

    ``tag`` is the "simulated" or "held-out" label PRD 9.1 requires on every figure.
    """
    figure, axes = _axes(f"{ylabel} versus {xlabel} ({tag})", xlabel, ylabel)
    for index, item in enumerate(series):
        color = _color(index)
        axes.fill_between(item.x, item.lower, item.upper, color=color, alpha=BAND_ALPHA, lw=0)
        axes.plot(item.x, item.y, color=color, linewidth=LINE_WIDTH, marker="o", label=item.label)
    if len(series) > 1:
        _legend(axes)
    return save_figure(figure, stem)


def reliability_diagram(table: pl.DataFrame, title: str, stem: Path) -> tuple[Path, Path]:
    """Plot accuracy per confidence bin from ``calibration.reliability_table``."""
    figure, axes = _axes(title, "Mean confidence", "Accuracy")
    filled = table.drop_nulls("accuracy")
    axes.plot([0, 1], [0, 1], color=MUTED_INK, linewidth=1, linestyle="--", label="Perfect")
    axes.plot(
        filled["confidence"].to_numpy(),
        filled["accuracy"].to_numpy(),
        color=_color(0),
        linewidth=LINE_WIDTH,
        marker="o",
        label="Observed",
    )
    axes.set_xlim(0, 1)
    axes.set_ylim(0, 1)
    _legend(axes, "upper left")
    return save_figure(figure, stem)


def risk_coverage_curves(
    curves: Mapping[str, RiskCoverage],
    random: RiskCoverage,
    oracle: RiskCoverage,
    title: str,
    stem: Path,
) -> tuple[Path, Path]:
    """Plot risk against coverage for each model, with random and oracle references."""
    figure, axes = _axes(title, "Coverage", "Error rate among kept cases")
    for index, (label, curve) in enumerate(curves.items()):
        axes.plot(
            curve.coverage, curve.risk, color=_color(index), linewidth=LINE_WIDTH, label=label
        )
    axes.plot(random.coverage, random.risk, color=MUTED_INK, linestyle="--", label="Random")
    axes.plot(oracle.coverage, oracle.risk, color=MUTED_INK, linestyle=":", label="Oracle")
    axes.set_xlim(0, 1)
    _legend(axes)
    return save_figure(figure, stem)


def bars_with_ci(
    labels: Sequence[str],
    values: Numbers,
    lower: Numbers,
    upper: Numbers,
    ylabel: str,
    title: str,
    stem: Path,
) -> tuple[Path, Path]:
    """Plot one bar per label with CI whiskers, from a zero baseline."""
    centers = np.asarray(values, dtype=np.float64)
    whiskers = np.vstack(
        [
            centers - np.asarray(lower, dtype=np.float64),
            np.asarray(upper, dtype=np.float64) - centers,
        ]
    )
    figure, axes = _axes(title, "", ylabel)
    positions = np.arange(len(labels))
    axes.bar(positions, centers, width=0.55, color=_color(0))
    axes.errorbar(positions, centers, yerr=whiskers, fmt="none", ecolor=INK, capsize=3, lw=1)
    axes.set_xticks(positions, labels)
    axes.grid(visible=False, axis="x")
    return save_figure(figure, stem)


def power_curves(curves: Sequence[PowerCurve], target: float, stem: Path) -> tuple[Path, Path]:
    """Plot power against the true Top-1 difference, in points, with the target marked."""
    figure, axes = _axes(
        "Power of the paired group bootstrap (simulated)",
        "True Top-1 difference (points)",
        "Power",
    )
    for index, curve in enumerate(curves):
        points = 100 * np.asarray(curve.difference, dtype=np.float64)
        axes.plot(points, curve.power, color=_color(index), linewidth=LINE_WIDTH, label=curve.label)
    axes.axhline(target, color=MUTED_INK, linewidth=1, linestyle="--", label=f"{target:.0%} power")
    axes.set_ylim(0, 1.02)
    _legend(axes, "lower right")
    return save_figure(figure, stem)
