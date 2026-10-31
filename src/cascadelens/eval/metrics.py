"""Ranking metrics on the common data contract (PRD section 9.2).

Predictions have columns ``case_id, service, score`` and a higher score means more likely root
cause. Ground truth has columns ``case_id, root_cause_service, group_id``.

Ties are broken by ascending service name, never randomly, so a metric is a pure function of its
input. A scorer that returns one constant therefore ranks services alphabetically, which is
deterministic but not neutral; report such a scorer as what it is.

Every case in the ground truth counts. A case whose true service has no prediction, or that has no
predictions at all, is a miss with a null ``rank_of_truth``. Such misses are counted, not dropped.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import polars as pl

from cascadelens.eval.errors import EvalError

PREDICTION_COLUMNS = ("case_id", "service", "score")
TRUTH_COLUMNS = ("case_id", "root_cause_service", "group_id")
OUTCOME_KS = (1, 3, 5)
AVG_K = 5


class ColumnTable(Protocol):
    """Anything indexable by column name, such as a pandas DataFrame or a dict of lists."""

    def __getitem__(self, key: str, /) -> Iterable[Any]: ...


Table = pl.DataFrame | ColumnTable


def as_frame(table: Table, columns: Sequence[str]) -> pl.DataFrame:
    """Return the named columns of ``table`` as a Polars frame."""
    if isinstance(table, pl.DataFrame):
        absent = sorted(set(columns) - set(table.columns))
        if absent:
            raise EvalError(f"Missing columns: {absent}")
        return table.select(columns)
    try:
        return pl.DataFrame({column: list(table[column]) for column in columns})
    except KeyError as error:
        raise EvalError(f"Missing column: {error}") from error


def _predictions(pred: Table) -> pl.DataFrame:
    frame = as_frame(pred, PREDICTION_COLUMNS).with_columns(pl.col("score").cast(pl.Float64))
    if frame.select(pl.col("score").is_null() | pl.col("score").is_nan()).to_series().any():
        raise EvalError("Scores must not be null or NaN")
    if frame.select(pl.struct("case_id", "service").is_duplicated()).to_series().any():
        raise EvalError("A (case_id, service) pair appears more than once in the predictions")
    return frame


def _truth(truth: Table) -> pl.DataFrame:
    frame = as_frame(truth, TRUTH_COLUMNS)
    if frame.is_empty():
        raise EvalError("Ground truth is empty")
    if frame["case_id"].is_duplicated().any():
        raise EvalError("A case_id appears more than once in the ground truth")
    return frame


def ranked(pred: Table) -> pl.DataFrame:
    """Add a 1-based ``rank`` per case: highest score first, ties by ascending service name."""
    return (
        _predictions(pred)
        .sort(["case_id", "score", "service"], descending=[False, True, False])
        .with_columns(rank=pl.int_range(1, pl.len() + 1, dtype=pl.Int64).over("case_id"))
    )


def per_case_outcomes(pred: Table, truth: Table) -> pl.DataFrame:
    """Return one row per true case, sorted by ``case_id``, for paired comparisons.

    Columns are ``case_id, group_id, rank_of_truth, hit@1, hit@3, hit@5, avg@5``. The hits are 0 or
    1 and ``avg@5`` is the case's own Avg@5, so its mean is ``avg_at_5``.
    """
    labels = _truth(truth)
    ranks = ranked(pred)
    unknown = ranks.join(labels, on="case_id", how="anti")
    if not unknown.is_empty():
        raise EvalError(f"Predictions for unknown case {unknown['case_id'][0]}")
    joined = labels.join(
        ranks.select("case_id", pl.col("service").alias("root_cause_service"), "rank"),
        on=["case_id", "root_cause_service"],
        how="left",
    )
    rank = pl.col("rank_of_truth")
    return (
        joined.rename({"rank": "rank_of_truth"})
        .select(
            "case_id",
            "group_id",
            "rank_of_truth",
            *[(rank <= k).fill_null(False).cast(pl.Int8).alias(f"hit@{k}") for k in OUTCOME_KS],
            ((AVG_K + 1 - rank).clip(0, AVG_K) / AVG_K).fill_null(0.0).alias(f"avg@{AVG_K}"),
        )
        .sort("case_id")
    )


def missing_truth_count(outcomes: pl.DataFrame) -> int:
    """Count cases whose true service had no prediction."""
    return outcomes["rank_of_truth"].null_count()


@dataclass(frozen=True)
class TopK:
    """A Top-k result. ``per_case`` has columns ``case_id, hit``."""

    k: int
    per_case: pl.DataFrame
    mean: float
    missing: int


def top_k(pred: Table, truth: Table, k: int) -> TopK:
    """Return per-case 0/1 hits for "the true service is ranked in the top ``k``", and the mean."""
    if k < 1:
        raise EvalError(f"k must be at least 1, got {k}")
    outcomes = per_case_outcomes(pred, truth)
    per_case = outcomes.select(
        "case_id", hit=(pl.col("rank_of_truth") <= k).fill_null(False).cast(pl.Int8)
    )
    return TopK(k, per_case, float(per_case["hit"].mean()), missing_truth_count(outcomes))  # type: ignore[arg-type]


def avg_at_5(pred: Table, truth: Table) -> float:
    """Return Avg@5, the mean of Top-1, Top-2, Top-3, Top-4 and Top-5.

    This averages Top-k over k = 1..5 (PRD 9.2). It is not MRR and not a mean over services. A case
    ranked r <= 5 scores (6 - r) / 5, and any other case scores 0.
    """
    return float(per_case_outcomes(pred, truth)[f"avg@{AVG_K}"].mean())  # type: ignore[arg-type]


def retrieval_recall_at_k(candidates: Table, truth: Table, k: int) -> TopK:
    """Return the share of cases whose true service is in Stage 1's top ``k`` candidates.

    ``candidates`` follows the prediction contract, scored by Stage 1. This is an upper bound on
    Stage 2 Top-1 when Stage 2 sees exactly those ``k`` candidates.
    """
    return top_k(candidates, truth, k)
