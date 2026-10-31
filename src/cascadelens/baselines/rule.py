"""Unsupervised rule-based baselines. Neither reads a label or ``inject_time``.

``RuleOnset`` scores earlier sustained anomaly onset higher. ``RuleUpstream`` takes the highest
anomaly score among anomalous services with no anomalous upstream, where upstream follows failure
propagation: a fault travels from a callee to its callers, so a service's upstream is what it
calls. Without a topology no service can be excluded, so it reduces to the highest anomaly score.
RCAEval RE1 has no topology, so on that data it is a plain max-anomaly ranking.
"""

from collections.abc import Sequence

import polars as pl

from cascadelens.baselines.base import Baseline, CaseHandle, score_table
from cascadelens.config import BaselineConfig
from cascadelens.features.simple import SignalSettings

NO_ONSET_SCORE = -2.0


class _Rule(Baseline):
    supervised = False

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._settings = SignalSettings()

    def fit(
        self,
        train_cases: Sequence[CaseHandle],
        config: BaselineConfig,
        val_cases: Sequence[CaseHandle] = (),
        seed: int = 0,
    ) -> None:
        """Read the settings only. Training cases are never touched."""
        self._settings = SignalSettings.model_validate(config.params)


class RuleOnset(_Rule):
    """Rank services by earliest sustained anomaly; services without one come last."""

    name = "rule"

    def predict(self, case: CaseHandle) -> pl.DataFrame:
        """Score each service by minus its onset position."""
        summary = self.cache.get(case, self._settings)
        return score_table(
            case.record.case_id,
            {
                row["service"]: NO_ONSET_SCORE if row["onset"] is None else -row["onset"]
                for row in summary.iter_rows(named=True)
            },
        )


class RuleUpstream(_Rule):
    """Rank anomalous services with no anomalous upstream first, each group by anomaly score."""

    name = "rule_upstream"

    def predict(self, case: CaseHandle) -> pl.DataFrame:
        """Score by peak z-score, below every candidate for services with anomalous upstream."""
        summary = self.cache.get(case, self._settings)
        rows = list(summary.iter_rows(named=True))
        peaks = {row["service"]: row["max_z"] or 0.0 for row in rows}
        anomalous = {row["service"] for row in rows if row["onset"] is not None}
        blocked = {
            edge.caller
            for edge in case.record.topology or ()
            if edge.callee in anomalous and edge.callee != edge.caller
        }
        penalty = max(peaks.values(), default=0.0) + 1.0
        return score_table(
            case.record.case_id,
            {
                service: peak - penalty if service in blocked else peak
                for service, peak in peaks.items()
            },
        )
