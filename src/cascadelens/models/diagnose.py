"""Rank the services of one case with a saved model. No confidence or abstention yet."""

from pathlib import Path

import polars as pl

from cascadelens.data.schema import METRICS_SCHEMA
from cascadelens.features.context import FeatureInput
from cascadelens.features.registry import build_case_features
from cascadelens.models.artifacts import load_artifact
from cascadelens.models.data import RankerError


def diagnose_case(case_file: Path, model_dir: Path) -> pl.DataFrame:
    """Rank a long metrics table saved as Parquet, using its first and last sample as the window."""
    if not case_file.is_file():
        raise RankerError(f"No case file at {case_file}")
    metrics = pl.read_parquet(case_file)
    if metrics.schema != METRICS_SCHEMA:
        raise RankerError(f"{case_file} must have schema {METRICS_SCHEMA}, got {metrics.schema}")
    if metrics.is_empty():
        raise RankerError(f"{case_file} holds no samples")
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
