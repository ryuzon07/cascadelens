"""Rank the services of one case with a saved model, with confidence and abstention."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cascadelens.data.schema import METRICS_SCHEMA
from cascadelens.eval.calibration import case_confidence
from cascadelens.features.context import FeatureInput
from cascadelens.features.registry import build_case_features
from cascadelens.models.artifacts import load_artifact
from cascadelens.models.calibration import load_calibration
from cascadelens.models.data import RankerError

DISCLAIMER = "ranking of likely origin services, not proof of causation"


@dataclass(frozen=True)
class Diagnosis:
    """A ranking with its top-1 confidence, the abstention flag and notes for the reader."""

    predictions: pl.DataFrame
    confidence: float
    calibrated: bool
    abstain: bool
    notes: list[str]


def _read_metrics(case_file: Path) -> pl.DataFrame:
    if not case_file.is_file():
        raise RankerError(f"No case file at {case_file}")
    metrics = pl.read_parquet(case_file)
    if metrics.schema != METRICS_SCHEMA:
        raise RankerError(f"{case_file} must have schema {METRICS_SCHEMA}, got {metrics.schema}")
    if metrics.is_empty():
        raise RankerError(f"{case_file} holds no samples")
    return metrics


def diagnose_case(case_file: Path, model_dir: Path) -> pl.DataFrame:
    """Rank a long metrics table saved as Parquet, using its first and last sample as the window."""
    return _rank(case_file, _read_metrics(case_file), model_dir)


def _rank(case_file: Path, metrics: pl.DataFrame, model_dir: Path) -> pl.DataFrame:
    model, features = load_artifact(model_dir)
    case = FeatureInput(
        case_id=case_file.stem,
        metrics=metrics,
        window_start=int(metrics["timestamp"].min()),  # type: ignore[arg-type]
        window_end=int(metrics["timestamp"].max()),  # type: ignore[arg-type]
        topology=None,
        known_services=(),
    )
    predictions, _ = model.rank(build_case_features(case, features))
    return predictions


def services_with_gaps(metrics: pl.DataFrame) -> int:
    """Return how many services have a null or NaN sample."""
    gaps = metrics.filter(pl.col("value").is_null() | pl.col("value").is_nan())
    return gaps["service"].n_unique()


def diagnose_report(case_file: Path, model_dir: Path) -> Diagnosis:
    """Rank a case, and add confidence, abstention and notes. It works without a calibrator."""
    metrics = _read_metrics(case_file)
    predictions = _rank(case_file, metrics, model_dir)
    saved = load_calibration(model_dir)
    notes = [f"{services_with_gaps(metrics)} services had missing telemetry.", DISCLAIMER]
    if saved is None:
        confidence = float(case_confidence(predictions)["confidence"][0])
        notes.insert(0, "Confidence is uncalibrated: no calibrator is saved with this model.")
        return Diagnosis(predictions, confidence, False, False, notes)
    confidence = float(saved.calibrator.apply(predictions)["confidence"][0])
    return Diagnosis(predictions, confidence, True, confidence < saved.tau, notes)
