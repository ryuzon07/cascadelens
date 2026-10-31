"""The frequency prior: the most common root-cause service per system in training.

A service's score is its share of its system's training labels, and 0 when never seen there. When
none of a case's services was seen for that system, as always under LOSO, every service scores the
same: this is the random-among-candidates floor (PRD 6.4, rule 5). Its expected Top-1 is one over
the number of candidates. The metric itself breaks the tie by service name, so report that expected
floor next to the measured Top-1 of those folds.
"""

from collections import Counter, defaultdict
from collections.abc import Sequence

import polars as pl

from cascadelens.baselines.base import Baseline, CaseHandle, score_table
from cascadelens.config import BaselineConfig


class FrequencyPrior(Baseline):
    """Rank services by how often they were the root cause in training."""

    name = "frequency"
    supervised = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._counts: dict[str, Counter[str]] = {}

    def fit(
        self,
        train_cases: Sequence[CaseHandle],
        config: BaselineConfig,
        val_cases: Sequence[CaseHandle] = (),
        seed: int = 0,
    ) -> None:
        """Count labels per system. Validation is unused: there is nothing to select."""
        counts: dict[str, Counter[str]] = defaultdict(Counter)
        for case in train_cases:
            counts[case.record.system][case.record.root_cause_service] += 1
        self._counts = dict(counts)

    def predict(self, case: CaseHandle) -> pl.DataFrame:
        """Score by training share, or uniformly when no candidate was seen in training."""
        services = sorted(case.metrics()["service"].unique().to_list())
        counts = self._counts.get(case.record.system, Counter())
        total = sum(counts.values())
        if not any(counts[service] for service in services):
            return score_table(case.record.case_id, dict.fromkeys(services, 1 / len(services)))
        return score_table(
            case.record.case_id, {service: counts[service] / total for service in services}
        )
