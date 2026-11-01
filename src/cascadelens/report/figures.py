"""Figures drawn from result JSONs. Each title carries its simulated or held-out label."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cascadelens.eval.plots import CurveSeries, bars_with_ci, robustness_curves
from cascadelens.report.sources import Store
from cascadelens.report.tables import CLEAN, CURVE_METHODS, RANKER, UNSUPERVISED, _grid


@dataclass(frozen=True)
class FigureEntry:
    """A figure and the source it came from."""

    path: str
    source: str


def _leakage(store: Store, out: Path) -> FigureEntry:
    metrics = store("leakage").metrics["scores"]
    labels, values, lower, upper = [], [], [], []
    for method in sorted(metrics):
        for split in ("group_run", "random"):
            entry = metrics[method][split]["Top-1"]
            labels.append(f"{method}\n{split}")
            values.append(100 * entry["estimate"])
            lower.append(100 * entry["lower"])
            upper.append(100 * entry["upper"])
    stem = out / "leakage"
    title = "Top-1 on a random split and on group_run (real cases)"
    bars_with_ci(labels, values, lower, upper, "Top-1 (%)", title, stem)
    return FigureEntry("docs/results/figures/leakage.png", "leakage")


def _p2_points(store: Store) -> list[str]:
    rows = store("robustness").metrics["grid"]
    found = [r["point"] for r in rows if r["method"] == RANKER and r["perturbation"] == "P2"]
    return [CLEAN, *found]


def _curves(store: Store, methods: Sequence[str], name: str, tag: str, out: Path) -> FigureEntry:
    grid = _grid(store)
    points = _p2_points(store)
    rates = np.array([0.0, *(float(p.rsplit("rate=", 1)[1]) for p in points[1:])]) * 100
    series = []
    for method in methods:
        entries = [grid[(method, point)]["Top-1"] for point in points]
        series.append(
            CurveSeries(
                method,
                rates,
                np.array([100 * e["estimate"] for e in entries]),
                np.array([100 * e["lower"] for e in entries]),
                np.array([100 * e["upper"] for e in entries]),
            )
        )
    robustness_curves(series, "Failure-correlated loss (%)", "Top-1 (%)", tag, out / name)
    return FigureEntry(f"docs/results/figures/{name}.png", "robustness")


def _selective(store: Store, out: Path) -> FigureEntry:
    results = store("calibration").metrics["results"]
    series = []
    for variant, entry in results.items():
        selective = entry[CLEAN]["selective"]
        coverages = np.array([100 * float(c) for c in selective])
        series.append(
            CurveSeries(
                variant,
                coverages,
                np.array([100 * (1 - e["estimate"]) for e in selective.values()]),
                np.array([100 * (1 - e["upper"]) for e in selective.values()]),
                np.array([100 * (1 - e["lower"]) for e in selective.values()]),
            )
        )
    robustness_curves(series, "Coverage (%)", "Error rate (%)", "clean", out / "risk_coverage")
    return FigureEntry("docs/results/figures/risk_coverage.png", "calibration")


def all_figures(store: Store, out: Path) -> list[FigureEntry]:
    """Draw every figure and return the entries."""
    out.mkdir(parents=True, exist_ok=True)
    supervised = [method for method in CURVE_METHODS if method != "frequency"]
    return [
        _leakage(store, out),
        _curves(store, supervised, "robustness_p2_supervised", "simulated", out),
        _curves(store, UNSUPERVISED, "robustness_p2_unsupervised", "simulated", out),
        _selective(store, out),
    ]
