"""Group bootstrap (PRD section 9.3).

Groups are resampled with replacement and cases never are: a resample is a vector of counts, one
per group, drawn once and applied to every arm, so all arms of a comparison see the same groups.
Each statistic is the mean over the cases of the resampled groups. Intervals are percentile
intervals. With few groups these run a little narrow; ``power`` measures the effect at the real
group counts.

Bootstrap p-values are two-sided: twice the smaller share of resampled differences on either side
of zero, inclusive, with the usual (count + 1) / (n + 1) correction so a p-value is never zero.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from cascadelens.eval.calibration import (
    ArrayLike,
    Floats,
    aurc,
    bin_index,
    ece,
    random_abstention,
    risk_coverage,
    selective_accuracy,
    tie_averaged_correct,
    validated,
)
from cascadelens.eval.errors import EvalError

N_RESAMPLES = 10_000
ALPHA = 0.05
AURC_BATCH = 500


@dataclass(frozen=True)
class BootstrapResult:
    """A point estimate with its percentile interval."""

    estimate: float
    lower: float
    upper: float
    n_resamples: int
    n_groups: int
    alpha: float

    def excludes_zero(self) -> bool:
        """Return whether zero lies outside the interval."""
        return self.lower > 0 or self.upper < 0


@dataclass(frozen=True)
class PairedResult(BootstrapResult):
    """A paired difference, with a two-sided bootstrap p-value."""

    p_value: float = 1.0


def group_positions(group_ids: ArrayLike) -> tuple[npt.NDArray[np.intp], int]:
    """Return each case's position among the sorted unique groups, and the group count."""
    labels = np.asarray(group_ids)
    if labels.ndim != 1 or labels.size == 0:
        raise EvalError("group_ids must be a non-empty one-dimensional sequence")
    unique, positions = np.unique(labels, return_inverse=True)
    return positions, int(unique.size)


def resample_counts(n_groups: int, n: int, seed: int) -> npt.NDArray[np.float64]:
    """Return an ``n`` by ``n_groups`` matrix of how often each resample drew each group."""
    if n < 1:
        raise EvalError(f"n must be at least 1, got {n}")
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n).astype(np.float64)


def _values(values: ArrayLike, cases: int) -> Floats:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (cases,):
        raise EvalError(f"Expected {cases} aligned per-case values, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EvalError("Per-case values must be finite")
    return array


def _check_alpha(alpha: float) -> None:
    if not 0 < alpha < 1:
        raise EvalError(f"alpha must lie in (0, 1), got {alpha}")


def _resampled_means(
    values: Floats, positions: npt.NDArray[np.intp], counts: npt.NDArray[np.float64]
) -> Floats:
    n_groups = counts.shape[1]
    sums = np.bincount(positions, weights=values, minlength=n_groups)
    sizes = np.bincount(positions, minlength=n_groups).astype(np.float64)
    means: Floats = (counts @ sums) / (counts @ sizes)
    return means


def _interval(resampled: Floats, alpha: float) -> tuple[float, float]:
    lower, upper = np.quantile(resampled, [alpha / 2, 1 - alpha / 2])
    return float(lower), float(upper)


def _p_value(resampled: Floats) -> float:
    n = resampled.size
    side = min(int(np.sum(resampled <= 0)), int(np.sum(resampled >= 0)))
    return min(1.0, 2 * (side + 1) / (n + 1))


def _paired(estimate: float, resampled: Floats, n_groups: int, alpha: float) -> PairedResult:
    lower, upper = _interval(resampled, alpha)
    return PairedResult(
        estimate=estimate,
        lower=lower,
        upper=upper,
        n_resamples=resampled.size,
        n_groups=n_groups,
        alpha=alpha,
        p_value=_p_value(resampled),
    )


def group_bootstrap_ci(
    per_case_values: ArrayLike,
    group_ids: ArrayLike,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> BootstrapResult:
    """Return the mean of ``per_case_values`` with a group-bootstrap interval."""
    _check_alpha(alpha)
    positions, n_groups = group_positions(group_ids)
    values = _values(per_case_values, positions.size)
    resampled = _resampled_means(values, positions, resample_counts(n_groups, n, seed))
    lower, upper = _interval(resampled, alpha)
    return BootstrapResult(float(values.mean()), lower, upper, n, n_groups, alpha)


def paired_group_bootstrap(
    a_values: ArrayLike,
    b_values: ArrayLike,
    group_ids: ArrayLike,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> PairedResult:
    """Return the mean of ``a - b`` on aligned cases, with its interval and p-value.

    For the S2 margin test, compare ``lower`` with minus the margin.
    """
    _check_alpha(alpha)
    positions, n_groups = group_positions(group_ids)
    difference = _values(a_values, positions.size) - _values(b_values, positions.size)
    resampled = _resampled_means(difference, positions, resample_counts(n_groups, n, seed))
    return _paired(float(difference.mean()), resampled, n_groups, alpha)


def diff_of_diffs_bootstrap(
    ranker_perturbed: ArrayLike,
    baseline_perturbed: ArrayLike,
    ranker_clean: ArrayLike,
    baseline_clean: ArrayLike,
    group_ids: ArrayLike,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> PairedResult:
    """Return S3: (ranker - baseline at perturbed) minus (ranker - baseline at clean).

    The four arms are aligned by case and each resample applies the same groups to all of them.
    A positive value means the ranker's margin grows under the perturbation.
    """
    _check_alpha(alpha)
    positions, n_groups = group_positions(group_ids)
    cases = positions.size
    margin_perturbed = _values(ranker_perturbed, cases) - _values(baseline_perturbed, cases)
    margin_clean = _values(ranker_clean, cases) - _values(baseline_clean, cases)
    counts = resample_counts(n_groups, n, seed)
    resampled = _resampled_means(margin_perturbed, positions, counts) - _resampled_means(
        margin_clean, positions, counts
    )
    return _paired(float((margin_perturbed - margin_clean).mean()), resampled, n_groups, alpha)


def _resampled_ece(
    confidences: Floats,
    outcomes: Floats,
    positions: npt.NDArray[np.intp],
    counts: npt.NDArray[np.float64],
    bins: int,
) -> Floats:
    """Return the ECE of every resample, using per-group, per-bin totals."""
    n_groups = counts.shape[1]
    cells = positions * bins + bin_index(confidences, bins)
    shape = n_groups * bins
    gap = np.bincount(cells, weights=outcomes - confidences, minlength=shape)
    gaps_by_group = gap.reshape(n_groups, bins)
    sizes = np.bincount(positions, minlength=n_groups).astype(np.float64)
    errors: Floats = np.abs(counts @ gaps_by_group).sum(axis=1) / (counts @ sizes)
    return errors


def ece_difference_ci(
    conf_before: ArrayLike,
    conf_after: ArrayLike,
    correct: ArrayLike,
    group_ids: ArrayLike,
    bins: int = 10,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> PairedResult:
    """Return S5: ECE after calibration minus ECE before, so negative is an improvement.

    Calibration keeps the ranking, so both arms share ``correct``. The same groups are resampled
    for both, and the bins are fixed.
    """
    _check_alpha(alpha)
    before, outcomes = validated(conf_before, correct)
    after, _ = validated(conf_after, correct)
    positions, n_groups = group_positions(group_ids)
    if positions.size != before.size:
        raise EvalError(f"Expected {before.size} group ids, got {positions.size}")
    counts = resample_counts(n_groups, n, seed)
    resampled = _resampled_ece(after, outcomes, positions, counts, bins) - _resampled_ece(
        before, outcomes, positions, counts, bins
    )
    estimate = ece(after, outcomes, bins) - ece(before, outcomes, bins)
    return _paired(estimate, resampled, n_groups, alpha)


def selective_gain_bootstrap(
    conf: ArrayLike,
    correct: ArrayLike,
    group_ids: ArrayLike,
    coverage: float = 0.5,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> PairedResult:
    """Return S4: selective Top-1 at ``coverage`` minus full-coverage Top-1.

    Random abstention has an expected selective accuracy equal to full-coverage accuracy, so this
    one difference is also the gain over random abstention. The kept share is recomputed inside
    every resample from the resampled case weights.
    """
    _check_alpha(alpha)
    if not 0 < coverage <= 1:
        raise EvalError(f"coverage must lie in (0, 1], got {coverage}")
    confidences, outcomes = validated(conf, correct)
    positions, n_groups = group_positions(group_ids)
    if positions.size != confidences.size:
        raise EvalError(f"Expected {confidences.size} group ids, got {positions.size}")
    counts = resample_counts(n_groups, n, seed)
    selective, full = _resampled_selective(confidences, outcomes, positions, counts, coverage)
    _, hits = tie_averaged_correct(confidences, outcomes)
    kept = max(1, round(coverage * confidences.size))
    estimate = float(hits[:kept].mean() - hits.mean())
    return _paired(estimate, selective - full, n_groups, alpha)


def _resampled_selective(
    confidences: Floats,
    outcomes: Floats,
    positions: npt.NDArray[np.intp],
    counts: npt.NDArray[np.float64],
    coverage: float,
) -> tuple[Floats, Floats]:
    """Return each resample's selective accuracy and full-coverage accuracy."""
    order = np.argsort(-confidences, kind="stable")
    _, hits = tie_averaged_correct(confidences, outcomes)
    weights = counts[:, positions[order]]
    cases = np.cumsum(weights, axis=1)
    correct_cases = np.cumsum(weights * hits, axis=1)
    target = np.maximum(1.0, np.round(coverage * cases[:, -1]))
    last = np.argmax(cases >= target[:, None], axis=1)
    rows = np.arange(counts.shape[0])
    selective: Floats = correct_cases[rows, last] / cases[rows, last]
    full: Floats = correct_cases[:, -1] / cases[:, -1]
    return selective, full


def _scored(
    conf: ArrayLike, correct: ArrayLike, group_ids: ArrayLike, alpha: float
) -> tuple[Floats, Floats, npt.NDArray[np.intp], int]:
    _check_alpha(alpha)
    confidences, outcomes = validated(conf, correct)
    positions, n_groups = group_positions(group_ids)
    if positions.size != confidences.size:
        raise EvalError(f"Expected {confidences.size} group ids, got {positions.size}")
    return confidences, outcomes, positions, n_groups


def ece_ci(
    conf: ArrayLike,
    correct: ArrayLike,
    group_ids: ArrayLike,
    bins: int = 10,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> BootstrapResult:
    """Return ECE with a group-bootstrap interval, over fixed bins."""
    confidences, outcomes, positions, n_groups = _scored(conf, correct, group_ids, alpha)
    resampled = _resampled_ece(
        confidences, outcomes, positions, resample_counts(n_groups, n, seed), bins
    )
    lower, upper = _interval(resampled, alpha)
    return BootstrapResult(ece(confidences, outcomes, bins), lower, upper, n, n_groups, alpha)


def selective_accuracy_ci(
    conf: ArrayLike,
    correct: ArrayLike,
    group_ids: ArrayLike,
    coverage: float,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> BootstrapResult:
    """Return selective accuracy at ``coverage``, with the kept share recomputed per resample."""
    confidences, outcomes, positions, n_groups = _scored(conf, correct, group_ids, alpha)
    counts = resample_counts(n_groups, n, seed)
    selective, _ = _resampled_selective(confidences, outcomes, positions, counts, coverage)
    lower, upper = _interval(selective, alpha)
    estimate = selective_accuracy(confidences, outcomes, coverage)
    return BootstrapResult(estimate, lower, upper, n, n_groups, alpha)


def aurc_difference_ci(
    conf: ArrayLike,
    correct: ArrayLike,
    group_ids: ArrayLike,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> PairedResult:
    """Return AURC minus the AURC of random abstention, so negative is an improvement.

    Random abstention is flat at the error rate, so each resample compares with its own rate.
    Resamples run in batches, because a batch costs resamples times cases in memory.
    """
    confidences, outcomes, positions, n_groups = _scored(conf, correct, group_ids, alpha)
    order = np.argsort(-confidences, kind="stable")
    _, hits = tie_averaged_correct(confidences, outcomes)
    counts = resample_counts(n_groups, n, seed)
    batches = []
    for start in range(0, n, AURC_BATCH):
        weights = counts[start : start + AURC_BATCH][:, positions[order]]
        cases = np.cumsum(weights, axis=1)
        errors = np.cumsum(weights * (1.0 - hits), axis=1)
        risk = np.divide(errors, cases, out=np.zeros_like(errors), where=cases > 0)
        area = (weights * risk).sum(axis=1) / cases[:, -1]
        batches.append(area - errors[:, -1] / cases[:, -1])
    estimate = aurc(risk_coverage(confidences, outcomes)) - aurc(random_abstention(outcomes))
    return _paired(estimate, np.concatenate(batches), n_groups, alpha)
