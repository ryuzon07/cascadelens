"""The perturbation contract (PRD 9.1).

A perturbation returns a new case and never changes its input. All randomness comes from a generator
seeded from (case_id, perturbation, parameters, seed), so the same inputs give the same bytes.
Labels, identifiers and ``inject_time`` never change. Window fields change only in P4 and P5, which
record the originals in the manifest.

Dropped data stays in the table as a row whose value is null, the one spelling of missing in the
Case schema. Rows are never removed and nothing is imputed, so Part 6 mask features see every drop.
A method that cannot cope with gaps declares an imputation in its own config.

Stages run in a fixed order when perturbations compose: data loss, then window, then topology.
"""

import hashlib
import io
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, ClassVar, Generic, Literal, TypeVar

import numpy as np
import numpy.typing as npt
import polars as pl

from cascadelens.config import StrictModel, canonical_json
from cascadelens.data.schema import Case, TopologyEdge
from cascadelens.seed import derive_case_seed

Realism = Literal["simulated", "held-out"]
ParamsT = TypeVar("ParamsT", bound=StrictModel)
Edges = tuple[TopologyEdge, ...]
SERIES = ("service", "metric")


class PerturbationError(Exception):
    """A perturbation could not be configured or applied."""


class Stage(IntEnum):
    """Composition order: data loss first, then window, then topology."""

    DATA = 0
    WINDOW = 1
    TOPOLOGY = 2


class ManifestRow(StrictModel):
    """What one application did. It stays beside the data and never reaches a model."""

    case_id: str
    perturbation: str
    name: str
    parameters: dict[str, Any]
    seed: int
    realism: Realism
    samples_before: int
    samples_dropped: int
    series_before: int
    series_dropped: int
    edges_before: int
    edges_added: int
    edges_removed: int
    details: dict[str, Any]
    output_hash: str


@dataclass(frozen=True)
class Outcome:
    """The result of a transform, before the generic counts are added."""

    case: Case
    edges_before: int = 0
    edges_added: int = 0
    edges_removed: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Applied:
    """A perturbed case with its manifest row."""

    case: Case
    row: ManifestRow


def with_fields(case: Case, **changes: object) -> Case:
    """Return a validated copy of ``case`` with some fields replaced."""
    fields: dict[str, Any] = {**case.record().model_dump(), "metrics": case.metrics}
    return Case(**{**fields, **changes})


def rounded(value: float) -> int:
    """Round half up, so a count does not depend on banker's rounding."""
    return int(np.floor(value + 0.5))


def observed(metrics: pl.DataFrame) -> pl.Expr:
    """Return the expression that marks a real sample."""
    return pl.col("value").is_not_null() & pl.col("value").is_finite()


def null_rows(metrics: pl.DataFrame, rows: npt.NDArray[np.bool_]) -> pl.DataFrame:
    """Return ``metrics`` with the marked rows set to missing."""
    mark = pl.Series("drop", rows, dtype=pl.Boolean)
    return metrics.with_columns(pl.when(mark).then(None).otherwise(pl.col("value")).alias("value"))


def crop_to_window(case: Case) -> Case:
    """Return ``case`` with only the samples inside its window.

    P4 and P5 keep data outside the new window, because a later shift may need it. Feature code does
    not crop, so read a windowed case through this when the outside must stay out of reach.
    """
    inside = pl.col("timestamp").is_between(case.window_start, case.window_end)
    return with_fields(case, metrics=case.metrics.filter(inside))


def metrics_bytes(metrics: pl.DataFrame) -> bytes:
    """Return the Parquet bytes of a metrics table."""
    buffer = io.BytesIO()
    metrics.write_parquet(buffer)
    return buffer.getvalue()


def output_hash(case: Case, data: bytes | None = None) -> str:
    """Hash a whole case, its metrics and every other field."""
    digest = hashlib.sha256(canonical_json(case.record().model_dump(mode="json")).encode())
    digest.update(metrics_bytes(case.metrics) if data is None else data)
    return digest.hexdigest()


def _counts(metrics: pl.DataFrame) -> tuple[int, int]:
    real = metrics.filter(observed(metrics))
    return real.height, real.select(*SERIES).n_unique()


class Perturbation(ABC, Generic[ParamsT]):
    """One perturbation with fixed parameters."""

    id: ClassVar[str]
    name: ClassVar[str]
    realism: ClassVar[Realism] = "simulated"
    stage: ClassVar[Stage]
    schema: ClassVar[type[StrictModel]]

    def __init__(self, params: ParamsT) -> None:
        self.params = params

    @classmethod
    def build(cls, params: Mapping[str, Any]) -> "Perturbation[ParamsT]":
        """Return an instance with validated parameters."""
        return cls(cls.schema.model_validate(dict(params)))  # type: ignore[arg-type]

    def parameters(self) -> dict[str, Any]:
        """Return the parameters, defaults included."""
        return self.params.model_dump(mode="json")

    def generator(self, case: Case, seed: int) -> np.random.Generator:
        """Return the generator of this (case, perturbation, parameters, seed)."""
        return np.random.default_rng(
            derive_case_seed(case.case_id, self.name, self.parameters(), seed)
        )

    @abstractmethod
    def transform(self, case: Case, rng: np.random.Generator) -> Outcome:
        """Return the perturbed case."""

    def run(self, case: Case, seed: int) -> Applied:
        """Apply the perturbation and describe what it changed."""
        outcome = self.transform(case, self.generator(case, seed))
        samples, series = _counts(case.metrics)
        samples_after, series_after = _counts(outcome.case.metrics)
        row = ManifestRow(
            case_id=case.case_id,
            perturbation=self.id,
            name=self.name,
            parameters=self.parameters(),
            seed=seed,
            realism=self.realism,
            samples_before=samples,
            samples_dropped=samples - samples_after,
            series_before=series,
            series_dropped=series - series_after,
            edges_before=outcome.edges_before,
            edges_added=outcome.edges_added,
            edges_removed=outcome.edges_removed,
            details=outcome.details,
            output_hash=output_hash(outcome.case),
        )
        return Applied(outcome.case, row)

    def apply(self, case: Case, seed: int) -> Case:
        """Return the perturbed case."""
        return self.run(case, seed).case
