"""The interface every baseline implements.

Each method gets the same windows, splits and cases. Unsupervised baselines keep ``fit`` a no-op
and must not read a training label. No baseline may read ``inject_time``.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import ClassVar

import polars as pl

from cascadelens.config import BaselineConfig
from cascadelens.data.io import metrics_path
from cascadelens.data.schema import Case, CaseRecord
from cascadelens.features.simple import SignalSettings, service_summary

PREDICTION_SCHEMA = pl.Schema({"case_id": pl.String, "service": pl.String, "score": pl.Float64})


class BaselineError(Exception):
    """A baseline could not be configured or run."""


class CaseHandle:
    """A case whose metrics load on demand, so a fold does not hold every series in memory."""

    def __init__(self, record: CaseRecord, load: Callable[[], pl.DataFrame]) -> None:
        self.record = record
        self._load = load

    @classmethod
    def from_case(cls, case: Case) -> "CaseHandle":
        """Wrap a case that is already in memory."""
        return cls(case.record(), lambda: case.metrics)

    @classmethod
    def stored(cls, record: CaseRecord, processed_dir: Path) -> "CaseHandle":
        """Refer to a case stored in the processed directory."""
        path = metrics_path(processed_dir, record.case_id)
        return cls(record, lambda: pl.read_parquet(path))

    def metrics(self) -> pl.DataFrame:
        """Load the long metrics table."""
        return self._load()


class SummaryCache:
    """Service summaries by case and settings, shared across folds and methods of one dataset."""

    def __init__(self) -> None:
        self._summaries: dict[tuple[str, SignalSettings], pl.DataFrame] = {}

    def get(self, case: CaseHandle, settings: SignalSettings) -> pl.DataFrame:
        """Return the summary of ``case``, computing it once."""
        key = (case.record.case_id, settings)
        if key not in self._summaries:
            record = case.record
            self._summaries[key] = service_summary(
                case.metrics(), record.window_start, record.window_end, settings
            )
        return self._summaries[key]


def score_table(case_id: str, scores: Mapping[str, float]) -> pl.DataFrame:
    """Return predictions sorted by descending score, ties by ascending service name."""
    return pl.DataFrame(
        {
            "case_id": [case_id] * len(scores),
            "service": list(scores),
            "score": list(scores.values()),
        },
        schema=PREDICTION_SCHEMA,
    ).sort(["score", "service"], descending=[True, False])


class Baseline(ABC):
    """A method that ranks the services of a case. Higher scores are more likely root causes."""

    name: ClassVar[str]
    supervised: ClassVar[bool]

    def __init__(self, cache: SummaryCache | None = None) -> None:
        self.cache = cache if cache is not None else SummaryCache()

    @abstractmethod
    def fit(
        self,
        train_cases: Sequence[CaseHandle],
        config: BaselineConfig,
        val_cases: Sequence[CaseHandle] = (),
        seed: int = 0,
    ) -> None:
        """Learn from the training cases; validation is for model selection only."""

    @abstractmethod
    def predict(self, case: CaseHandle) -> pl.DataFrame:
        """Score every service of ``case`` as ``case_id, service, score``."""
