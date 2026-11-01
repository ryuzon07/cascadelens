"""The one inference path: the CLI, the API and the demo all call ``diagnose``.

This module is read-only analysis. It opens no socket and reads no ground truth: ``CaseInput`` has
no field for a label or an injection time, and a sample file's ``simulation`` block is dropped here.
"""

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from cascadelens.config import StrictModel
from cascadelens.data.schema import METRICS_SCHEMA, TopologyEdge
from cascadelens.eval.metrics import ranked
from cascadelens.features.config import FeaturesConfig
from cascadelens.features.context import CaseContext, FeatureInput
from cascadelens.features.mask import MISSING_COLUMN
from cascadelens.features.registry import REGISTRY, build_case_features
from cascadelens.models.artifacts import load_artifact, read_sidecar
from cascadelens.models.calibration import SavedCalibration, load_calibration, model_hash
from cascadelens.models.data import RankerError, model_features
from cascadelens.models.diagnose import DISCLAIMER, score_case
from cascadelens.models.ranker import FittedFold

ONSET_COLUMN = "temporal_onset_rank"
UPSTREAM_COLUMN = "graph_upstream_anomalous_fraction"
AVAILABLE_COLUMN = "graph_topology_available"
INFERRED_COLUMN = "graph_topology_inferred"
TOP_GROUPS = 3
DEFAULT_TOLERANCE_SECONDS = 60
VERSION_LENGTH = 12


class Window(StrictModel):
    """The analysis window, in Unix seconds."""

    start: int
    end: int

    @model_validator(mode="after")
    def _ordered(self) -> "Window":
        if self.start >= self.end:
            raise ValueError("window start must be before window end")
        return self


class MetricRow(StrictModel):
    """One sample. A null value means the sample is missing."""

    service: str = Field(min_length=1, max_length=200)
    metric: str = Field(min_length=1, max_length=200)
    ts: int
    value: float | None

    @model_validator(mode="after")
    def _finite(self) -> "MetricRow":
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("metric values must be finite or null")
        return self


class CasePayload(StrictModel):
    """A case as JSON: the body of ``POST /diagnose`` and the form of a sample file."""

    case_id: str = Field(default="case", min_length=1, max_length=200)
    topology: list[TopologyEdge] | None = None
    metrics: list[MetricRow] = Field(min_length=1)
    window: Window


@dataclass(frozen=True)
class CaseInput:
    """What inference may read about a case."""

    case_id: str
    metrics: pl.DataFrame
    window_start: int
    window_end: int
    topology: tuple[TopologyEdge, ...] | None = None


def case_from_payload(
    payload: CasePayload, tolerance: int = DEFAULT_TOLERANCE_SECONDS
) -> CaseInput:
    """Check a payload against its window and return the case. An empty edge list means none."""
    low, high = payload.window.start - tolerance, payload.window.end + tolerance
    stray = sum(not low <= row.ts <= high for row in payload.metrics)
    if stray:
        raise RankerError(f"{stray} samples are more than {tolerance} s outside the window")
    metrics = pl.DataFrame(
        {
            "service": [row.service for row in payload.metrics],
            "metric": [row.metric for row in payload.metrics],
            "timestamp": [row.ts for row in payload.metrics],
            "value": [row.value for row in payload.metrics],
        },
        schema=METRICS_SCHEMA,
    )
    if metrics.select("service", "metric", "timestamp").is_duplicated().any():
        raise RankerError("A (service, metric, ts) sample appears more than once")
    edges = tuple(payload.topology) if payload.topology else None
    return CaseInput(payload.case_id, metrics, payload.window.start, payload.window.end, edges)


def read_case(path: Path) -> CaseInput:
    """Read a metrics Parquet file, or a JSON case whose ``simulation`` block is dropped."""
    if not path.is_file():
        raise RankerError(f"No case file at {path}")
    if path.suffix == ".json":
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        raw.pop("simulation", None)
        try:
            return case_from_payload(CasePayload.model_validate(raw))
        except ValidationError as error:
            raise RankerError(
                f"{path} is not a valid case: {error.error_count()} errors"
            ) from error
    metrics = pl.read_parquet(path)
    if metrics.schema != METRICS_SCHEMA:
        raise RankerError(f"{path} must have schema {METRICS_SCHEMA}, got {metrics.schema}")
    if metrics.is_empty():
        raise RankerError(f"{path} holds no samples")
    start, end = metrics["timestamp"].min(), metrics["timestamp"].max()
    return CaseInput(path.stem, metrics, int(start), int(end))  # type: ignore[arg-type]


@dataclass(frozen=True)
class LoadedModel:
    """A saved model, its calibrator and its version, loaded once."""

    directory: Path
    fold: FittedFold
    features: FeaturesConfig
    calibration: SavedCalibration | None
    version: str


def load_model(directory: Path) -> LoadedModel:
    """Load a saved model and any calibrator beside it, with a clear error if it is missing."""
    if not directory.is_dir():
        raise RankerError(f"No model directory at {directory}")
    fold, features = load_artifact(directory)
    version = model_hash(directory)[:VERSION_LENGTH]
    return LoadedModel(directory, fold, features, load_calibration(directory), version)


class GroupContribution(BaseModel):
    """The summed LightGBM contribution of one feature group to a service's score."""

    model_config = ConfigDict(frozen=True)

    group: str
    contribution: float


class FeatureContribution(BaseModel):
    """One feature's value and its LightGBM contribution to a service's score."""

    model_config = ConfigDict(frozen=True)

    feature: str
    label: str
    group: str
    value: float | None
    contribution: float


class Evidence(BaseModel):
    """Why a service ranks where it does. Null means the case could not support the value."""

    model_config = ConfigDict(frozen=True)

    onset_rank: float | None
    upstream_anom_ratio: float | None
    missing_frac: float | None
    top_groups: list[GroupContribution]


class RankedService(BaseModel):
    """One candidate service."""

    model_config = ConfigDict(frozen=True)

    service: str
    score: float
    evidence: Evidence


class Timing(BaseModel):
    """How long inference took, measured inside ``diagnose``."""

    model_config = ConfigDict(frozen=True)

    total_seconds: float


class DiagnoseResult(BaseModel):
    """A ranking of likely origin services, for a person to check."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    ranking: list[RankedService]
    confidence: float
    calibrated: bool
    abstain: bool
    notes: list[str]
    model_version: str
    disclaimer: str = DISCLAIMER
    timing: Timing


def _number(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)


def _group_of(column: str) -> str:
    for name, group in REGISTRY.items():
        if column.startswith(group.prefix):
            return name
    raise RankerError(f"Column {column!r} belongs to no feature group")


def _contribution_matrix(table: pl.DataFrame, model: FittedFold) -> npt.NDArray[np.float64]:
    booster: lgb.Booster = model.booster
    raw = booster.predict(model_features(table, model.columns), pred_contrib=True)
    return np.asarray(raw, dtype=np.float64)[:, :-1]


def contributions(table: pl.DataFrame, model: FittedFold) -> dict[str, list[GroupContribution]]:
    """Return each service's largest feature-group contributions from Stage 2."""
    values = _contribution_matrix(table, model)
    groups = [_group_of(column) for column in model.columns]
    result = {}
    for service, row in zip(table["service"].to_list(), values, strict=True):
        totals: dict[str, float] = {}
        for group, value in zip(groups, row.tolist(), strict=True):
            totals[group] = totals.get(group, 0.0) + value
        largest = sorted(totals.items(), key=lambda item: (-abs(item[1]), item[0]))[:TOP_GROUPS]
        result[service] = [GroupContribution(group=g, contribution=v) for g, v in largest]
    return result


def feature_label(column: str) -> str:
    """Return a feature's name as words, without its group prefix."""
    prefix = next(g.prefix for g in REGISTRY.values() if column.startswith(g.prefix))
    return column.removeprefix(prefix).replace("_", " ").capitalize()


def feature_contributions(
    case_input: CaseInput, model: LoadedModel, service: str
) -> list[FeatureContribution]:
    """Return every Stage 2 feature's contribution to one service's score, largest first."""
    table = build_case_features(_feature_input(case_input), model.features)
    row = table.with_row_index("row").filter(pl.col("service") == service)
    if row.is_empty():
        raise RankerError(f"The case has no service {service!r}")
    index = int(row["row"][0])
    values = _contribution_matrix(table, model.fold)[index].tolist()
    found = [
        FeatureContribution(
            feature=column,
            label=feature_label(column),
            group=_group_of(column),
            value=_number(row[column][0]),
            contribution=contribution,
        )
        for column, contribution in zip(model.fold.columns, values, strict=True)
    ]
    return sorted(found, key=lambda item: (-abs(item.contribution), item.feature))


def _ranking(
    table: pl.DataFrame, predictions: pl.DataFrame, model: FittedFold
) -> list[RankedService]:
    rows = {row["service"]: row for row in table.iter_rows(named=True)}
    groups = contributions(table, model)
    ranking = []
    for service, score in ranked(predictions).select("service", "score").iter_rows():
        row = rows[service]
        evidence = Evidence(
            onset_rank=_number(row.get(ONSET_COLUMN)),
            upstream_anom_ratio=_number(row.get(UPSTREAM_COLUMN)),
            missing_frac=_number(row.get(MISSING_COLUMN)),
            top_groups=groups[service],
        )
        ranking.append(RankedService(service=service, score=score, evidence=evidence))
    return ranking


def _notes(table: pl.DataFrame, ranked_count: int, calibrated: bool) -> list[str]:
    missing = (
        table[MISSING_COLUMN].fill_nan(1.0).fill_null(1.0) if MISSING_COLUMN in table else None
    )
    gaps = int((missing > 0).sum()) if missing is not None else 0
    notes = [f"telemetry missing for {gaps} of {table.height} services"]
    available = AVAILABLE_COLUMN in table and float(table[AVAILABLE_COLUMN].max() or 0) > 0  # type: ignore[arg-type]
    if not available:
        notes.append("topology missing: graph evidence is unavailable")
    elif INFERRED_COLUMN in table and float(table[INFERRED_COLUMN].max() or 0) > 0:  # type: ignore[arg-type]
        notes.append("topology inferred from metrics, not recorded")
    else:
        notes.append("topology provided with the case")
    if ranked_count < table.height:
        notes.append(f"{ranked_count} of {table.height} services passed retrieval and were ranked")
    if not calibrated:
        notes.append("confidence is uncalibrated: no calibrator is saved with this model")
    return notes


def _feature_input(case_input: CaseInput) -> FeatureInput:
    moment = pl.col("timestamp")
    metrics = case_input.metrics.with_columns(pl.col("value").fill_nan(None)).filter(
        (moment >= case_input.window_start) & (moment <= case_input.window_end)
    )
    if metrics.is_empty():
        raise RankerError("The case has no samples inside its window")
    return FeatureInput(
        case_id=case_input.case_id,
        metrics=metrics,
        window_start=case_input.window_start,
        window_end=case_input.window_end,
        topology=case_input.topology,
        known_services=(),
    )


def case_context(case_input: CaseInput, model: LoadedModel) -> CaseContext:
    """Return the context features read: onsets, scored samples and any inferred graph."""
    settings = model.features
    return CaseContext.build(
        _feature_input(case_input),
        settings.signal,
        settings.graph,
        settings.mask,
        settings.topology,
    )


def diagnose(case_input: CaseInput, model_dir: Path | LoadedModel) -> DiagnoseResult:
    """Run feature extraction, Stage 1, Stage 2, calibration and abstention on one case.

    Samples outside the window are left out, as the perturbed evaluation does. Pass a
    ``LoadedModel`` to avoid loading the model again for each case.
    """
    started = time.perf_counter()
    model = model_dir if isinstance(model_dir, LoadedModel) else load_model(model_dir)
    view = _feature_input(case_input)
    scored = score_case(view, model.fold, model.features, model.calibration)
    ranking = _ranking(scored.table, scored.predictions, model.fold)
    return DiagnoseResult(
        case_id=case_input.case_id,
        ranking=ranking,
        confidence=scored.confidence,
        calibrated=scored.calibrated,
        abstain=scored.abstain,
        notes=_notes(scored.table, len(ranking), scored.calibrated),
        model_version=model.version,
        timing=Timing(total_seconds=time.perf_counter() - started),
    )


def model_details(model: LoadedModel) -> dict[str, Any]:
    """Return the provenance a saved model records."""
    sidecar = read_sidecar(model.directory)
    return {
        "model_version": model.version,
        "method": sidecar["method"],
        "split": sidecar["split"],
        "seed": sidecar["seed"],
        "fold": sidecar["fold"],
        "training_data_checksum": sidecar["data_checksum"],
        "feature_config_hash": sidecar["feature_config_hash"],
        "config_hash": sidecar["config_hash"],
        "calibrated": model.calibration is not None,
    }
