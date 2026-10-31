"""BARO's RobustScorer from RCAEval (Pham, Ha, Zhang, FSE 2024).

The method needs an anomaly time. Here it is ``window_start + cut_fraction * (window_end -
window_start)``, computed from the window only, and ``inject_time`` is never read. With the
configured ``cut_fraction`` of 0.25 it matches the rule baseline's pre-incident part.

``baro_midpoint`` uses 0.5. The injection sits near the middle of the recordings (see
docs/data_audit.md), so that time is ``inject_time`` in disguise. It exists as a labelled
sensitivity check for the window-leak probe, never as a headline.
"""

import math

from cascadelens.baselines.base import BaselineError, CaseHandle
from cascadelens.baselines.published.adapter import WideFrame, call_library
from cascadelens.baselines.published.base import LIBRARY_DATASET, PublishedBaseline, check_output


class Baro(PublishedBaseline):
    """BARO with an anomaly time derived from the window."""

    name = "baro"

    def anomaly_time(self, case: CaseHandle) -> int:
        """Return the split between normal and anomalous rows, from the window bounds alone."""
        fraction = self.params.cut_fraction
        if fraction is None:
            raise BaselineError(f"{self.name} needs cut_fraction in its config")
        record = case.record
        return round(record.window_start + fraction * (record.window_end - record.window_start))

    def scores(self, case: CaseHandle, wide: WideFrame) -> dict[str, float]:
        """Return BARO's score per column, dropping non-finite ones."""
        output = call_library(
            "baro",
            wide.frame,
            inject_time=self.anomaly_time(case),
            dataset=LIBRARY_DATASET,
            dk_select_useful=False,
        )
        check_output(output)
        return {c: s for c, s in output["scores"].items() if math.isfinite(s)}


class BaroKeyMetrics(Baro):
    """BARO on the key metric set, to compare with ``lingam_randomwalk`` on equal inputs."""

    name = "baro_key_metrics"


class BaroMidpoint(Baro):
    """BARO split at the window middle. Sensitivity check only: see the module docstring."""

    name = "baro_midpoint"
