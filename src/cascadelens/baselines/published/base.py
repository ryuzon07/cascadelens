"""The shared wrapper: unsupervised, failure-tolerant and timed.

A case that fails is logged with its ``case_id`` and counted, and scores every service equally, so
it can only hurt the method. Runtime per case is recorded. Library calls are memoized per case and
seed, so one seed costs one pass over the data however many splits it is scored on.
"""

import logging
import time
from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from cascadelens.baselines.base import Baseline, CaseHandle, SummaryCache, score_table
from cascadelens.baselines.published.adapter import WideFrame, fill_gaps, key_metrics, to_wide
from cascadelens.config import BaselineConfig, StrictModel
from cascadelens.seed import derive_case_seed, set_seed

LOGGER = logging.getLogger(__name__)
LIBRARY_DATASET = "re1"


class PublishedParams(StrictModel):
    """Settings of a published wrapper."""

    method: str
    cut_fraction: float | None = None
    metric_set: Literal["all", "key"] = "all"


@dataclass(frozen=True)
class CaseFailure:
    """One case the library could not score."""

    case_id: str
    error: str


class PublishedBaseline(Baseline):
    """Rank services with a library method. ``fit`` never touches a training case."""

    supervised = False

    def __init__(self, cache: SummaryCache | None = None) -> None:
        super().__init__(cache)
        self.params = PublishedParams(method=self.name)
        self.seed = 0
        self.failures: list[CaseFailure] = []
        self.runtimes: dict[str, float] = {}
        self._tables: dict[tuple[str, int], pl.DataFrame] = {}

    def fit(
        self,
        train_cases: Sequence[CaseHandle],
        config: BaselineConfig,
        val_cases: Sequence[CaseHandle] = (),
        seed: int = 0,
    ) -> None:
        """Read the settings only."""
        self.params = PublishedParams.model_validate(config.params)
        self.seed = seed

    @abstractmethod
    def scores(self, case: CaseHandle, wide: WideFrame) -> dict[str, float]:
        """Return a score per column the method ranked; higher is more suspicious."""

    def predict(self, case: CaseHandle) -> pl.DataFrame:
        """Score every service, as the maximum over its columns; unranked ones come last."""
        key = (case.record.case_id, self.seed)
        if key not in self._tables:
            self._tables[key] = self._score_case(case)
        return self._tables[key]

    def _score_case(self, case: CaseHandle) -> pl.DataFrame:
        case_id = case.record.case_id
        metrics = case.metrics()
        services = sorted(metrics["service"].unique().to_list())
        set_seed(derive_case_seed(case_id, self.name, {}, self.seed))
        start = time.perf_counter()
        try:
            chosen = key_metrics(metrics) if self.params.metric_set == "key" else metrics
            wide = to_wide(chosen)
            filled = WideFrame(fill_gaps(wide.frame), wide.series)
            ranked = self.scores(case, filled)
            by_service: dict[str, float] = {}
            for column, score in ranked.items():
                service = filled.service_of(column)
                by_service[service] = max(score, by_service.get(service, score))
            last = min(by_service.values(), default=0.0) - 1.0
            table = {service: by_service.get(service, last) for service in services}
        except Exception as error:
            LOGGER.warning("%s failed on %s: %s", self.name, case_id, error)
            self.failures.append(CaseFailure(case_id, f"{type(error).__name__}: {error}"))
            table = dict.fromkeys(services, 0.0)
        self.runtimes[case_id] = time.perf_counter() - start
        return score_table(case_id, table)


def check_output(output: dict[str, Any]) -> None:
    """Raise when the library's ``rca`` decorator swallowed an error.

    On failure it prints to stderr and returns the plain column order with ``adj == []``, which
    would otherwise look like a real ranking.
    """
    adjacency = output.get("adj")
    if isinstance(adjacency, list) and not adjacency:
        raise RuntimeError("the library fell back to column order")
