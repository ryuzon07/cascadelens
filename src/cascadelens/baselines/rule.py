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
from cascadelens.features.simple import FEATURE_COLUMNS, SignalSettings

NO_ONSET_SCORE = -2.0
DURATION_COLUMNS = [column for column in FEATURE_COLUMNS if column.endswith("anomaly_duration")]


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
        """Score by rank of peak z-score, then duration, with anomalous-upstream services last.

        Peaks often tie at the z-score cap, so the longest anomaly breaks the tie before names do.
        """
        summary = self.cache.get(case, self._settings)
        rows = list(summary.iter_rows(named=True))
        durations = summary.select(
            pl.max_horizontal(pl.col(DURATION_COLUMNS)).fill_null(0.0)
        ).to_series()
        keys = {
            row["service"]: (row["max_z"] or 0.0, duration)
            for row, duration in zip(rows, durations, strict=True)
        }
        positions = {key: float(index) for index, key in enumerate(sorted(set(keys.values())))}
        anomalous = {row["service"] for row in rows if row["onset"] is not None}
        blocked = {
            edge.caller
            for edge in case.record.topology or ()
            if edge.callee in anomalous and edge.callee != edge.caller
        }
        penalty = float(len(positions))
        return score_table(
            case.record.case_id,
            {
                service: positions[key] - penalty if service in blocked else positions[key]
                for service, key in keys.items()
            },
        )
