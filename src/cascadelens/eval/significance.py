"""Signed-rank checks, Holm correction and the primary or exploratory label (PRD 9.3)."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy import stats  # type: ignore[import-untyped]

from cascadelens.eval.calibration import ArrayLike
from cascadelens.eval.errors import EvalError

PRIMARY_CRITERIA = ("S1", "S2", "S3", "S4", "S7")


@dataclass(frozen=True)
class WilcoxonResult:
    """A signed-rank result. ``n_nonzero`` counts the differences that were ranked."""

    statistic: float
    p_value: float
    n_nonzero: int


def wilcoxon_signed_rank(
    a_values: ArrayLike, b_values: ArrayLike, group_ids: ArrayLike | None = None
) -> WilcoxonResult:
    """Run a two-sided Wilcoxon signed-rank test, dropping zero differences.

    This is the check of PRD 9.3, not the primary test. It assumes independent units, and cases of
    one group are not independent, so per-case p-values run too small. Pass ``group_ids`` to test
    mean differences per group instead. With no non-zero difference the p-value is 1.
    """
    first = np.asarray(a_values, dtype=np.float64)
    second = np.asarray(b_values, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise EvalError(f"Expected aligned values, got {first.shape} and {second.shape}")
    differences = first - second
    if group_ids is not None:
        labels = np.asarray(group_ids)
        if labels.shape != differences.shape:
            raise EvalError(f"Expected {differences.size} group ids, got shape {labels.shape}")
        _, positions = np.unique(labels, return_inverse=True)
        differences = np.bincount(positions, weights=differences) / np.bincount(positions)
    nonzero = int(np.count_nonzero(differences))
    if nonzero == 0:
        return WilcoxonResult(0.0, 1.0, 0)
    result = stats.wilcoxon(differences, zero_method="wilcox", alternative="two-sided")
    return WilcoxonResult(float(result.statistic), float(result.pvalue), nonzero)


def holm_correction(pvalues: Sequence[float], labels: Sequence[str]) -> dict[str, float]:
    """Return Holm-adjusted p-values by label, in the order given.

    With p-values sorted ascending, the i-th adjusted value of m is the running maximum of
    (m - i + 1) * p, capped at 1.
    """
    if len(pvalues) != len(labels) or len(set(labels)) != len(labels):
        raise EvalError("Need one unique label per p-value")
    if not all(0 <= p <= 1 for p in pvalues):
        raise EvalError("p-values must lie in [0, 1]")
    total = len(pvalues)
    adjusted = [0.0] * total
    running = 0.0
    for step, index in enumerate(sorted(range(total), key=lambda i: pvalues[i])):
        running = max(running, min(1.0, (total - step) * pvalues[index]))
        adjusted[index] = running
    return dict(zip(labels, adjusted, strict=True))


def label_comparisons(
    comparisons: Mapping[str, tuple[str, float]],
    primary: Sequence[str] = PRIMARY_CRITERIA,
) -> pl.DataFrame:
    """Mark each comparison primary or exploratory, and Holm-correct the primary family.

    ``comparisons`` maps a label to its (criterion, raw p-value). ``adjusted_p`` is set for primary
    comparisons only: PRD 9.3 shows exploratory results as effect sizes, without significance
    claims, so they get no adjusted value.
    """
    primary_labels = [label for label, (crit, _) in comparisons.items() if crit in primary]
    adjusted = holm_correction([comparisons[label][1] for label in primary_labels], primary_labels)
    return pl.DataFrame(
        {
            "label": list(comparisons),
            "criterion": [criterion for criterion, _ in comparisons.values()],
            "role": ["primary" if label in adjusted else "exploratory" for label in comparisons],
            "p_value": [p_value for _, p_value in comparisons.values()],
            "adjusted_p": [adjusted.get(label) for label in comparisons],
        },
        schema_overrides={"adjusted_p": pl.Float64},
    )
