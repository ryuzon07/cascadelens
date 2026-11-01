"""Score one case with a saved model, with confidence and abstention."""

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cascadelens.data.schema import METRICS_SCHEMA
from cascadelens.eval.calibration import case_confidence
from cascadelens.features.config import FeaturesConfig
from cascadelens.features.context import FeatureInput
from cascadelens.features.registry import build_case_features
from cascadelens.models.artifacts import load_artifact
from cascadelens.models.calibration import SavedCalibration, load_calibration
from cascadelens.models.data import RankerError
from cascadelens.models.ranker import FittedFold

DISCLAIMER = "ranking of likely origin services, not proof of causation"


@dataclass(frozen=True)
class Scored:
    """What the model made of a case: features, ranking, confidence and the abstention flag."""

    table: pl.DataFrame
    predictions: pl.DataFrame
    confidence: float
    calibrated: bool
    abstain: bool


@dataclass(frozen=True)
class Diagnosis:
    """A ranking with its top-1 confidence, the abstention flag and notes for the reader."""

    predictions: pl.DataFrame
    confidence: float
    calibrated: bool
    abstain: bool
    notes: list[str]


def score_case(
    case: FeatureInput,
    model: FittedFold,
    features: FeaturesConfig,
    saved: SavedCalibration | None,
) -> Scored:
    """Run feature extraction, both stages, calibration and abstention on one case."""
    table = build_case_features(case, features)
    predictions, _ = model.rank(table)
    if saved is None:
        confidence = float(case_confidence(predictions)["confidence"][0])
        return Scored(table, predictions, confidence, False, False)
    confidence = float(saved.calibrator.apply(predictions)["confidence"][0])
    return Scored(table, predictions, confidence, True, confidence < saved.tau)


def read_metrics(case_file: Path) -> pl.DataFrame:
    """Read a long metrics table saved as Parquet."""
    if not case_file.is_file():
        raise RankerError(f"No case file at {case_file}")
    metrics = pl.read_parquet(case_file)
    if metrics.schema != METRICS_SCHEMA:
        raise RankerError(f"{case_file} must have schema {METRICS_SCHEMA}, got {metrics.schema}")
    if metrics.is_empty():
        raise RankerError(f"{case_file} holds no samples")
    return metrics


def _view(case_file: Path, metrics: pl.DataFrame) -> FeatureInput:
    return FeatureInput(
        case_id=case_file.stem,
        metrics=metrics,
        window_start=int(metrics["timestamp"].min()),  # type: ignore[arg-type]
        window_end=int(metrics["timestamp"].max()),  # type: ignore[arg-type]
        topology=None,
        known_services=(),
    )


def diagnose_case(case_file: Path, model_dir: Path) -> pl.DataFrame:
    """Rank a metrics file, using its first and last sample as the window."""
    return diagnose_report(case_file, model_dir).predictions


def services_with_gaps(metrics: pl.DataFrame) -> int:
    """Return how many services have a null or NaN sample."""
    gaps = metrics.filter(pl.col("value").is_null() | pl.col("value").is_nan())
    return gaps["service"].n_unique()


def diagnose_report(case_file: Path, model_dir: Path) -> Diagnosis:
    """Rank a case, and add confidence, abstention and notes. It works without a calibrator."""
    metrics = read_metrics(case_file)
    model, features = load_artifact(model_dir)
    scored = score_case(_view(case_file, metrics), model, features, load_calibration(model_dir))
    notes = [f"{services_with_gaps(metrics)} services had missing telemetry.", DISCLAIMER]
    if not scored.calibrated:
        notes.insert(0, "Confidence is uncalibrated: no calibrator is saved with this model.")
    return Diagnosis(
        scored.predictions, scored.confidence, scored.calibrated, scored.abstain, notes
    )
