"""Aggregation across model seeds (PRD section 9.3, item 4).

This is the plan's chosen procedure: average each case's outcome over the seeds, then bootstrap
groups over those seed-averaged outcomes. Seed spread is reported next to the interval, not folded
into it, so the interval expresses sampling of groups and the spread expresses training noise.
"""

from dataclasses import dataclass

import polars as pl

from cascadelens.eval.bootstrap import ALPHA, N_RESAMPLES, BootstrapResult, group_bootstrap_ci
from cascadelens.eval.errors import EvalError

MIN_SEEDS = 5
COLUMNS = ("case_id", "group_id", "seed", "value")


@dataclass(frozen=True)
class SeedSummary:
    """A seed-averaged metric. ``seed_means`` holds one row per seed."""

    seed_means: pl.DataFrame
    spread: float
    ci: BootstrapResult


def seed_averaged_outcomes(outcomes: pl.DataFrame, min_seeds: int = MIN_SEEDS) -> pl.DataFrame:
    """Return one row per case with its mean ``value`` over seeds, sorted by ``case_id``.

    ``outcomes`` has columns ``case_id, group_id, seed, value``. Every case must appear exactly
    once per seed, so a mean never covers different seeds for different cases. Use
    ``min_seeds=1`` for a deterministic method.
    """
    absent = sorted(set(COLUMNS) - set(outcomes.columns))
    if absent:
        raise EvalError(f"Missing columns: {absent}")
    seeds = outcomes["seed"].n_unique()
    if seeds < min_seeds:
        raise EvalError(f"Need at least {min_seeds} seeds, got {seeds}")
    per_case = outcomes.group_by("case_id").agg(
        pl.col("seed").len().alias("rows"),
        pl.col("seed").n_unique().alias("seeds"),
        pl.col("group_id").n_unique().alias("groups"),
        pl.col("group_id").first(),
        pl.col("value").mean(),
    )
    bad = per_case.filter((pl.col("rows") != seeds) | (pl.col("seeds") != seeds))
    if not bad.is_empty():
        raise EvalError(f"Case {bad.sort('case_id')['case_id'][0]} is not present once per seed")
    bad = per_case.filter(pl.col("groups") != 1)
    if not bad.is_empty():
        raise EvalError(f"Case {bad.sort('case_id')['case_id'][0]} has several group ids")
    return per_case.select("case_id", "group_id", "value").sort("case_id")


def summarize_seeds(
    outcomes: pl.DataFrame,
    min_seeds: int = MIN_SEEDS,
    n: int = N_RESAMPLES,
    seed: int = 0,
    alpha: float = ALPHA,
) -> SeedSummary:
    """Return per-seed means, their standard deviation and the seed-averaged group bootstrap."""
    averaged = seed_averaged_outcomes(outcomes, min_seeds)
    seed_means = outcomes.group_by("seed").agg(pl.col("value").mean()).sort("seed")
    deviation = seed_means["value"].std() if seed_means.height > 1 else 0.0
    return SeedSummary(
        seed_means=seed_means,
        spread=float(deviation),  # type: ignore[arg-type]
        ci=group_bootstrap_ci(averaged["value"], averaged["group_id"], n, seed, alpha),
    )
