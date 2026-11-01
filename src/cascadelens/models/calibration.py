"""Turn scores into a case confidence: temperature scaling, isotonic regression and the choice.

Everything here is fitted on validation predictions and is given no test case. Validation folds of
a split overlap the test cases of its other folds, so validation predictions are pooled across the
seeds of one fold, never across folds. Each pooled prediction comes from a model that did not train
on that case, though validation did steer that model's early stopping and search.

Temperature scaling maximises the likelihood of the true service among the candidates. Cases whose
truth was not retrieved have no candidate to score, so they are left out of the fit and reported as
``retrieval_miss_rate``. A softmax over candidates cannot express "none of these", which leaves
confidence slightly high by about that rate.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import Field
from scipy.optimize import minimize_scalar  # type: ignore[import-untyped]
from scipy.special import logsumexp  # type: ignore[import-untyped]
from sklearn.isotonic import IsotonicRegression  # type: ignore[import-untyped]

from cascadelens.config import StrictModel
from cascadelens.eval.bootstrap import paired_group_bootstrap
from cascadelens.eval.calibration import case_confidence
from cascadelens.eval.metrics import per_case_outcomes
from cascadelens.models.data import RankerError

Floats = npt.NDArray[np.float64]
Bools = npt.NDArray[np.bool_]
ARTIFACT = "calibration.json"
FORMAT_VERSION = 1
LOG_TEMPERATURE_BOUND = 7.0
CLIP = 1e-6
COVERAGES = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)
KEY_SEPARATOR = "@seed"


class CalibrationConfig(StrictModel):
    """Settings of the calibrator and of abstention."""

    seed: int = 0
    method: Literal["auto", "temperature", "isotonic"] = "auto"
    cross_fit_folds: int = 5
    n_resamples: int = 10_000
    alpha: float = 0.05
    target_coverage: float = 0.5
    coverages: list[float] = Field(default_factory=lambda: list(COVERAGES))


class Calibrator(Protocol):
    """Anything that gives each case its calibrated top-1 confidence."""

    @property
    def kind(self) -> str:
        """Return the name of this calibrator."""
        ...

    def apply(self, pred: pl.DataFrame) -> pl.DataFrame:
        """Return ``case_id, service, confidence`` for each case."""
        ...

    def parameters(self) -> dict[str, Any]:
        """Return what is needed to rebuild this calibrator."""
        ...


def pool_seeds(
    predictions: Mapping[int, pl.DataFrame], truth: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Stack one fold's predictions from several seeds, keying each case by case and seed."""
    pred, labels = [], []
    for seed, frame in predictions.items():
        key = pl.col("case_id") + pl.lit(f"{KEY_SEPARATOR}{seed}")
        pred.append(frame.with_columns(key.alias("case_id")))
        scored = truth.filter(pl.col("case_id").is_in(frame["case_id"].unique().to_list()))
        labels.append(scored.with_columns(key.alias("case_id")))
    return pl.concat(pred), pl.concat(labels)


def _score_rows(pred: pl.DataFrame, truth: pl.DataFrame) -> tuple[Floats, Bools, Floats]:
    """Return padded scores by case, the candidate mask, and the truth's score in retrieved cases.

    Cases whose truth has no score are dropped here; ``retrieval_miss_rate`` counts them.
    """
    labels = truth.select("case_id", pl.col("root_cause_service").alias("service"))
    held = labels.join(pred, on=["case_id", "service"], how="inner").sort("case_id")
    if held.is_empty():
        raise RankerError("No validation case has its true service among the candidates")
    rows = pred.join(held.select("case_id"), on="case_id", how="semi").sort("case_id", "service")
    sizes = rows.group_by("case_id", maintain_order=True).len()["len"].to_numpy()
    width = int(sizes.max())
    mask = np.arange(width)[None, :] < sizes[:, None]
    scores = np.zeros(mask.shape, dtype=np.float64)
    scores[mask] = rows["score"].to_numpy()
    return scores, mask, held["score"].to_numpy().astype(np.float64)


def negative_log_likelihood(pred: pl.DataFrame, truth: pl.DataFrame, temperature: float) -> float:
    """Return the mean NLL of the true service among candidates, over retrieved cases."""
    return _nll(*_score_rows(pred, truth), temperature)


def _nll(scores: Floats, mask: Bools, true_scores: Floats, temperature: float) -> float:
    totals = logsumexp(scores / temperature, axis=1, b=mask.astype(np.float64))
    return float(np.mean(totals - true_scores / temperature))


def retrieval_miss_rate(pred: pl.DataFrame, truth: pl.DataFrame) -> float:
    """Return the share of cases whose true service is not among the candidates."""
    outcomes = per_case_outcomes(pred, truth)
    return float(outcomes["rank_of_truth"].is_null().mean())  # type: ignore[arg-type]


@dataclass(frozen=True)
class TemperatureScaler:
    """A softmax over candidate scores at a temperature."""

    temperature: float = 1.0
    kind: str = "temperature"

    @classmethod
    def fit(cls, pred: pl.DataFrame, truth: pl.DataFrame) -> "TemperatureScaler":
        """Minimise the NLL over the temperature, in log space so that it stays positive."""
        rows = _score_rows(pred, truth)
        result = minimize_scalar(
            lambda log_t: _nll(*rows, float(np.exp(log_t))),
            bounds=(-LOG_TEMPERATURE_BOUND, LOG_TEMPERATURE_BOUND),
            method="bounded",
            options={"xatol": 1e-6},
        )
        return cls(float(np.exp(result.x)))

    def apply(self, pred: pl.DataFrame) -> pl.DataFrame:
        """Return each case's confidence at the fitted temperature."""
        return case_confidence(pred, self.temperature)

    def parameters(self) -> dict[str, Any]:
        """Return the temperature."""
        return {"temperature": self.temperature}


@dataclass(frozen=True)
class IsotonicCalibrator:
    """A monotone map from raw top-1 confidence to the chance of being right."""

    inputs: tuple[float, ...]
    outputs: tuple[float, ...]
    kind: str = "isotonic"

    @classmethod
    def fit(cls, conf: npt.ArrayLike, correct: npt.ArrayLike) -> "IsotonicCalibrator":
        """Fit on raw confidences and correctness. Outside the fitted range the map is flat."""
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(np.asarray(conf, dtype=np.float64), np.asarray(correct, dtype=np.float64))
        return cls(tuple(model.X_thresholds_.tolist()), tuple(model.y_thresholds_.tolist()))

    def transform(self, conf: npt.ArrayLike) -> Floats:
        """Map raw confidences."""
        mapped: Floats = np.interp(np.asarray(conf, dtype=np.float64), self.inputs, self.outputs)
        return mapped

    def apply(self, pred: pl.DataFrame) -> pl.DataFrame:
        """Return each case's mapped confidence."""
        frame = case_confidence(pred)
        return frame.with_columns(confidence=pl.Series(self.transform(frame["confidence"])))

    def parameters(self) -> dict[str, Any]:
        """Return the fitted steps."""
        return {"inputs": list(self.inputs), "outputs": list(self.outputs)}


def correctness(pred: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    """Return ``case_id, group_id, correct`` by case, sorted as ``case_confidence`` is."""
    outcomes = per_case_outcomes(pred, truth)
    return outcomes.select("case_id", "group_id", pl.col("hit@1").cast(pl.Float64).alias("correct"))


def fit_calibrator(kind: str, pred: pl.DataFrame, truth: pl.DataFrame) -> Calibrator:
    """Fit one kind of calibrator."""
    if kind == "temperature":
        return TemperatureScaler.fit(pred, truth)
    if kind == "isotonic":
        raw = case_confidence(pred)["confidence"]
        return IsotonicCalibrator.fit(raw, correctness(pred, truth)["correct"])
    raise RankerError(f"Unknown calibrator {kind!r}")


def log_loss(conf: npt.ArrayLike, correct: npt.ArrayLike) -> Floats:
    """Return each case's binary log loss of top-1 confidence against correctness."""
    p = np.clip(np.asarray(conf, dtype=np.float64), CLIP, 1 - CLIP)
    y = np.asarray(correct, dtype=np.float64)
    losses: Floats = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return losses


def group_folds(group_ids: list[str], folds: int, seed: int) -> dict[str, int]:
    """Assign each group to a cross-fitting fold, deterministically."""
    groups = sorted(set(group_ids))
    if len(groups) < 2:
        raise RankerError("Cross-fitting needs at least two validation groups")
    shuffled = np.random.default_rng(seed).permutation(len(groups))
    count = min(folds, len(groups))
    return {groups[int(position)]: index % count for index, position in enumerate(shuffled)}


def cross_fit_losses(
    kind: str, pred: pl.DataFrame, truth: pl.DataFrame, config: CalibrationConfig
) -> pl.DataFrame:
    """Return each validation case's log loss from a calibrator fitted on other groups."""
    assigned = group_folds(truth["group_id"].to_list(), config.cross_fit_folds, config.seed)
    labelled = truth.with_columns(
        part=pl.col("group_id").replace_strict(assigned, return_dtype=pl.Int64)
    )
    frames = []
    for part in sorted(set(assigned.values())):
        held = labelled.filter(pl.col("part") == part).drop("part")
        used = labelled.filter(pl.col("part") != part).drop("part")
        fitted = fit_calibrator(kind, pred.join(used, on="case_id", how="semi"), used)
        scored = pred.join(held, on="case_id", how="semi")
        frame = fitted.apply(scored).join(correctness(scored, held), on="case_id")
        frames.append(
            frame.select(
                "case_id",
                "group_id",
                loss=pl.Series(log_loss(frame["confidence"], frame["correct"])),
            )
        )
    return pl.concat(frames).sort("case_id")


@dataclass(frozen=True)
class Selection:
    """The chosen calibrator and the evidence for the choice."""

    calibrator: Calibrator
    report: dict[str, Any]


def select_calibrator(
    pred: pl.DataFrame, truth: pl.DataFrame, config: CalibrationConfig
) -> Selection:
    """Choose on validation only: isotonic must beat temperature on held-out log loss.

    Temperature has one parameter, so it is the default, and isotonic needs a paired interval that
    lies wholly below zero.
    """
    temperature = cross_fit_losses("temperature", pred, truth, config)
    isotonic = cross_fit_losses("isotonic", pred, truth, config)
    test = paired_group_bootstrap(
        isotonic["loss"],
        temperature["loss"],
        temperature["group_id"],
        config.n_resamples,
        config.seed,
        config.alpha,
    )
    wins = test.upper < 0
    kind = "isotonic" if wins else "temperature"
    forced = config.method != "auto"
    if forced:
        kind = config.method
    fitted = fit_calibrator(kind, pred, truth)
    report = {
        "chosen": kind,
        "forced": forced,
        "log_loss_temperature": float(temperature["loss"].mean()),  # type: ignore[arg-type]
        "log_loss_isotonic": float(isotonic["loss"].mean()),  # type: ignore[arg-type]
        "isotonic_minus_temperature": {
            "estimate": test.estimate,
            "lower": test.lower,
            "upper": test.upper,
        },
        "temperature": TemperatureScaler.fit(pred, truth).temperature,
        "retrieval_miss_rate": retrieval_miss_rate(pred, truth),
        "validation_cases": truth.height,
        "validation_top1": float(correctness(pred, truth)["correct"].mean()),  # type: ignore[arg-type]
    }
    return Selection(fitted, report)


def rebuild(kind: str, parameters: Mapping[str, Any]) -> Calibrator:
    """Rebuild a calibrator from saved parameters."""
    if kind == "temperature":
        return TemperatureScaler(float(parameters["temperature"]))
    if kind == "isotonic":
        return IsotonicCalibrator(tuple(parameters["inputs"]), tuple(parameters["outputs"]))
    raise RankerError(f"Unknown calibrator {kind!r}")


def model_hash(directory: Path) -> str:
    """Return a hash of the saved boosters, so that a calibrator cannot outlive its model."""
    digest = hashlib.sha256()
    for path in sorted(directory.glob("stage*.txt")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SavedCalibration:
    """A calibrator and its thresholds, as saved beside a model."""

    calibrator: Calibrator
    tau: float
    target_coverage: float
    thresholds: list[dict[str, float]]
    report: dict[str, Any]


def save_calibration(saved: SavedCalibration, directory: Path) -> Path:
    """Write the artifact beside its model, tied to the hash of that model."""
    payload = {
        "format_version": FORMAT_VERSION,
        "kind": saved.calibrator.kind,
        "model_hash": model_hash(directory),
        "parameters": saved.calibrator.parameters(),
        "report": saved.report,
        "target_coverage": saved.target_coverage,
        "tau": saved.tau,
        "thresholds": saved.thresholds,
    }
    path = directory / ARTIFACT
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def load_calibration(directory: Path) -> SavedCalibration | None:
    """Read the artifact of a model, or nothing if it is absent. A stale one is refused."""
    path = directory / ARTIFACT
    if not path.is_file():
        return None
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    if payload["format_version"] != FORMAT_VERSION:
        raise RankerError(f"Calibrator format {payload['format_version']} is not supported")
    if payload["model_hash"] != model_hash(directory):
        raise RankerError(f"The calibrator in {directory} belongs to a different model")
    return SavedCalibration(
        rebuild(payload["kind"], payload["parameters"]),
        payload["tau"],
        payload["target_coverage"],
        payload["thresholds"],
        payload["report"],
    )
