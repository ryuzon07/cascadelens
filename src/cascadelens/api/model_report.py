"""Read committed result JSONs for the Model screen. A missing value stays null."""

import json
from pathlib import Path
from typing import Any

import yaml

from cascadelens.api.schemas import (
    CheckValue,
    CoveragePoint,
    ReliabilityBin,
    ResultValue,
    RobustnessPoint,
)

FROZEN_DIR = Path("configs/frozen")
CALIBRATION_RESULTS = "experiments/calibration/results.json"
RANKER_RESULTS = "experiments/ranker_clean/results.json"
ROBUSTNESS_RESULTS = "experiments/robustness/results.json"
PROBES = {
    "Identity probe": "experiments/probes/identity.json",
    "Graph-only probe": "experiments/probes/graph_only.json",
    "Mask-only probe": "experiments/probes/mask_only.json",
    "Window probe": "experiments/probes/window.json",
}
CLEAN = "clean"
TOPK = ("Top-1", "Top-3", "Avg@5")
METHODS = {
    "ranker": "Ranker",
    "ranker_augmented": "Ranker, augmented",
    "lightgbm_flat": "LightGBM flat",
    "lightgbm_flat_masking": "LightGBM flat, masking",
}
CALIBRATED_METHOD = "ranker_augmented"


def read_metrics(path: Path) -> dict[str, Any] | None:
    """Return a result file's metrics, or None if it is absent."""
    if not path.is_file():
        return None
    metrics: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    return metrics


def frozen_config_hash(config_hash: str, root: Path) -> str | None:
    """Return the model's config hash if a frozen record holds it, or null."""
    for path in sorted((root / FROZEN_DIR).glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            record = yaml.safe_load(handle)
        if record.get("config_hash") == config_hash:
            return config_hash
    return None


def _value(name: str, found: dict[str, float] | None, source: str) -> ResultValue:
    found = found or {}
    return ResultValue(
        name=name,
        estimate=found.get("estimate"),
        lower=found.get("lower"),
        upper=found.get("upper"),
        source=source,
    )


def _clean_rows(robustness: dict[str, Any] | None) -> list[ResultValue]:
    rows = {
        row["method"]: row for row in (robustness or {}).get("grid", []) if row["point"] == CLEAN
    }
    return [
        _value(f"{label} {metric} (clean)", rows.get(method, {}).get(metric), ROBUSTNESS_RESULTS)
        for method, label in METHODS.items()
        for metric in TOPK
    ]


def _criteria(robustness: dict[str, Any] | None) -> list[ResultValue]:
    found = (robustness or {}).get("hypotheses", {}).get("ranker", {})
    return [
        _value("S1 robustness gain under loss", found.get("S1"), ROBUSTNESS_RESULTS),
        _value("S2 clean non-inferiority", found.get("S2"), ROBUSTNESS_RESULTS),
    ]


def _calibration(calibration: dict[str, Any] | None) -> list[ResultValue]:
    clean = (calibration or {}).get("results", {}).get(CALIBRATED_METHOD, {}).get(CLEAN, {})
    return [
        _value("ECE after calibration", clean.get("ece_after"), CALIBRATION_RESULTS),
        _value("Brier after calibration", clean.get("brier_after"), CALIBRATION_RESULTS),
    ]


def result_values(root: Path) -> tuple[str | None, list[ResultValue]]:
    """Return the result status and every value, each with the file it came from."""
    ranker = read_metrics(root / RANKER_RESULTS)
    robustness = read_metrics(root / ROBUSTNESS_RESULTS)
    calibration = read_metrics(root / CALIBRATION_RESULTS)
    status = ranker["status"] if ranker else None
    return status, [*_clean_rows(robustness), *_criteria(robustness), *_calibration(calibration)]


def _clean_calibration(root: Path) -> dict[str, Any]:
    calibration = read_metrics(root / CALIBRATION_RESULTS) or {}
    clean: dict[str, Any] = calibration.get("results", {}).get(CALIBRATED_METHOD, {}).get(CLEAN, {})
    return clean


def reliability_bins(root: Path) -> list[ReliabilityBin] | None:
    """Return reliability bins, or None: Part 10 saved them only as figures so far."""
    bins = _clean_calibration(root).get("reliability")
    return [ReliabilityBin.model_validate(item) for item in bins] if bins else None


def risk_coverage(root: Path) -> list[CoveragePoint]:
    """Return Top-1 on the kept cases at each coverage, lowest coverage first."""
    selective = _clean_calibration(root).get("selective", {})
    return [
        CoveragePoint(coverage=float(coverage), **selective[coverage])
        for coverage in sorted(selective, key=float)
    ]


def robustness_points(root: Path) -> list[RobustnessPoint]:
    """Return Top-1 of the compared methods at every grid point, in the committed order."""
    robustness = read_metrics(root / ROBUSTNESS_RESULTS) or {}
    return [
        RobustnessPoint(
            method=row["method"],
            perturbation=row["perturbation"],
            point=row["point"],
            realism=row["realism"],
            **row["Top-1"],
        )
        for row in robustness.get("grid", [])
        if row["method"] in METHODS
    ]


def check_values(root: Path) -> list[CheckValue]:
    """Return the leakage probes and the calibration verdicts."""
    checks = []
    for name, source in PROBES.items():
        probe = read_metrics(root / source)
        checks.append(
            CheckValue(name=name, passed=probe["passed"] if probe else None, source=source)
        )
    calibration = read_metrics(root / CALIBRATION_RESULTS)
    verdicts = calibration["verdicts"] if calibration else {}
    for name in ("S4", "S5"):
        checks.append(CheckValue(name=name, passed=verdicts.get(name), source=CALIBRATION_RESULTS))
    return checks
