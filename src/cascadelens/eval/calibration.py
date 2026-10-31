"""Confidence, calibration and risk-coverage (PRD sections 8.5 and 9.2).

Fitting temperature scaling or isotonic regression does not belong here. Everything below scores
confidences it is given. ``correct`` is whether the top-ranked service was right, normally the
``hit@1`` column of ``per_case_outcomes``.

Confidence ties are resolved by expectation, not by order: within a block of equal confidence each
case carries the block's mean error, which is the expected result of abstaining at random inside
the block. Curves therefore do not depend on the order of the input.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from cascadelens.eval.errors import EvalError
from cascadelens.eval.metrics import Table, ranked

Floats = npt.NDArray[np.float64]
ArrayLike = npt.ArrayLike
DEFAULT_BINS = 10


def case_confidence(pred: Table, temperature: float = 1.0) -> pl.DataFrame:
    """Return each case's top-1 confidence, a softmax over its scores at ``temperature``.

    Columns are ``case_id, service, confidence``, with one row per case sorted by ``case_id``.
    ``service`` is the top-ranked service, using the tie-break of ``metrics.ranked``.
    """
    if temperature <= 0:
        raise EvalError(f"temperature must be positive, got {temperature}")
    # Subtracting the case maximum keeps the exponentials finite for any score scale.
    weight = ((pl.col("score") - pl.col("score").max().over("case_id")) / temperature).exp()
    return (
        ranked(pred)
        .with_columns(confidence=weight / weight.sum().over("case_id"))
        .filter(pl.col("rank") == 1)
        .select("case_id", "service", "confidence")
        .sort("case_id")
    )


def validated(conf: ArrayLike, correct: ArrayLike) -> tuple[Floats, Floats]:
    """Return ``conf`` and ``correct`` as float arrays, refusing inputs that cannot be scored."""
    confidences = np.asarray(conf, dtype=np.float64)
    outcomes = np.asarray(correct, dtype=np.float64)
    if confidences.ndim != 1 or confidences.shape != outcomes.shape:
        raise EvalError(f"Shapes differ: {confidences.shape} and {outcomes.shape}")
    if confidences.size == 0:
        raise EvalError("No cases to score")
    if not np.all((confidences >= 0) & (confidences <= 1)):
        raise EvalError("Confidences must lie in [0, 1]")
    if not np.all((outcomes == 0) | (outcomes == 1)):
        raise EvalError("Correctness must be 0 or 1")
    return confidences, outcomes


def bin_index(confidences: Floats, bins: int) -> npt.NDArray[np.intp]:
    """Return equal-width bins over [0, 1]; each is (lower, upper], and the first includes 0."""
    if bins < 1:
        raise EvalError(f"bins must be at least 1, got {bins}")
    edges = np.linspace(0.0, 1.0, bins + 1)
    return np.clip(np.searchsorted(edges, confidences, side="left") - 1, 0, bins - 1)


def reliability_table(
    conf: ArrayLike, correct: ArrayLike, bins: int = DEFAULT_BINS
) -> pl.DataFrame:
    """Return one row per bin: bounds, count, mean confidence, accuracy and absolute gap.

    Empty bins are kept with null means, so a plot shows where there is no evidence.
    """
    confidences, outcomes = validated(conf, correct)
    index = bin_index(confidences, bins)
    counts = np.bincount(index, minlength=bins)
    conf_sums = np.bincount(index, weights=confidences, minlength=bins)
    hit_sums = np.bincount(index, weights=outcomes, minlength=bins)
    filled = counts > 0
    mean_conf = [float(c / n) if n else None for c, n in zip(conf_sums, counts, strict=True)]
    accuracy = [float(h / n) if n else None for h, n in zip(hit_sums, counts, strict=True)]
    return pl.DataFrame(
        {
            "bin": np.arange(bins),
            "lower": np.arange(bins) / bins,
            "upper": np.arange(1, bins + 1) / bins,
            "count": counts.astype(np.int64),
            "confidence": mean_conf,
            "accuracy": accuracy,
            "gap": [
                abs(float(h - c)) / int(n) if ok else None
                for h, c, n, ok in zip(hit_sums, conf_sums, counts, filled, strict=True)
            ],
        },
        schema_overrides={"confidence": pl.Float64, "accuracy": pl.Float64, "gap": pl.Float64},
    )


def ece(conf: ArrayLike, correct: ArrayLike, bins: int = DEFAULT_BINS) -> float:
    """Return the case-weighted mean of |accuracy - confidence| over equal-width bins."""
    confidences, outcomes = validated(conf, correct)
    index = bin_index(confidences, bins)
    conf_sums = np.bincount(index, weights=confidences, minlength=bins)
    hit_sums = np.bincount(index, weights=outcomes, minlength=bins)
    return float(np.abs(hit_sums - conf_sums).sum() / confidences.size)


def brier(conf: ArrayLike, correct: ArrayLike) -> float:
    """Return the mean squared error of the top-1 confidence against correctness."""
    confidences, outcomes = validated(conf, correct)
    return float(np.mean((confidences - outcomes) ** 2))


@dataclass(frozen=True)
class RiskCoverage:
    """A risk-coverage curve, with one point per number of cases kept, ascending."""

    coverage: Floats
    risk: Floats


def tie_averaged_correct(confidences: Floats, outcomes: Floats) -> tuple[Floats, Floats]:
    """Sort by descending confidence, giving each tied block its mean correctness."""
    order = np.argsort(-confidences, kind="stable")
    sorted_conf = confidences[order]
    sorted_hits = outcomes[order]
    starts = np.flatnonzero(np.r_[True, sorted_conf[1:] != sorted_conf[:-1]])
    sizes = np.diff(np.r_[starts, sorted_conf.size])
    block_means = np.add.reduceat(sorted_hits, starts) / sizes
    return sorted_conf, np.repeat(block_means, sizes)


def _curve(errors: Floats) -> RiskCoverage:
    kept = np.arange(1, errors.size + 1, dtype=np.float64)
    return RiskCoverage(coverage=kept / errors.size, risk=np.cumsum(errors) / kept)


def risk_coverage(conf: ArrayLike, correct: ArrayLike) -> RiskCoverage:
    """Return the error rate among the kept cases as the least confident are abstained."""
    confidences, outcomes = validated(conf, correct)
    _, hits = tie_averaged_correct(confidences, outcomes)
    return _curve(1.0 - hits)


def oracle_abstention(correct: ArrayLike) -> RiskCoverage:
    """Return the best achievable curve: every wrong case is abstained before any right one."""
    outcomes = np.asarray(correct, dtype=np.float64)
    return _curve(np.sort(1.0 - outcomes))


def random_abstention(correct: ArrayLike) -> RiskCoverage:
    """Return the expected curve of random abstention: flat at the overall error rate.

    This is the exact expectation, not a simulation, so it needs no seed.
    """
    outcomes = np.asarray(correct, dtype=np.float64)
    return _curve(np.full(outcomes.size, 1.0 - float(outcomes.mean())))


def aurc(curve: RiskCoverage) -> float:
    """Return the area under a risk-coverage curve, as the mean risk over its points.

    This is the discrete form with one point per kept-case count, so the oracle and flat curves
    have exact values. Lower is better.
    """
    return float(curve.risk.mean())


def selective_accuracy(conf: ArrayLike, correct: ArrayLike, coverage: float) -> float:
    """Return accuracy on the most confident ``coverage`` share of cases, at least one case."""
    if not 0 < coverage <= 1:
        raise EvalError(f"coverage must lie in (0, 1], got {coverage}")
    curve = risk_coverage(conf, correct)
    kept = max(1, round(coverage * curve.risk.size))
    return 1.0 - float(curve.risk[kept - 1])
