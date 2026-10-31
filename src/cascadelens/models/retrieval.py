"""Stage 1: pick the candidate services of a case (PRD 8.2).

Stage 2 trains on candidates. If those came from a scorer fitted on the same cases, truth would be
retrieved almost always in training and less often at test. So training candidates come from
out-of-fold scores, cross-fitted over whole groups. Validation and test cases use a scorer fitted on
all training cases, which has not seen them.
"""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl

from cascadelens.baselines.base import score_table
from cascadelens.config import RetrievalConfig
from cascadelens.models.data import Floats, RankerError, RankingMatrix
from cascadelens.seed import derive_case_seed, lightgbm_seed_params

ONSET_RANK = "temporal_onset_rank"
MAX_Z_SUFFIX = "_max_z"
NO_ONSET_RANK = 1000.0


class Scorer(Protocol):
    """A Stage 1 scorer. Higher means more likely a candidate."""

    def score(self, features: Floats) -> Floats:
        """Score one row per service."""
        ...


class BoosterScorer:
    """A LightGBM binary classifier."""

    def __init__(self, booster: lgb.Booster) -> None:
        self.booster = booster

    def score(self, features: Floats) -> Floats:
        """Return the predicted probability of each row."""
        return np.asarray(self.booster.predict(features), dtype=np.float64)


class HeuristicScorer:
    """The fallback: largest z-score minus onset rank, with no training."""

    def __init__(self, columns: Sequence[str]) -> None:
        self.columns = tuple(columns)
        self._onset = [i for i, c in enumerate(columns) if c == ONSET_RANK]
        self._max_z = [i for i, c in enumerate(columns) if c.endswith(MAX_Z_SUFFIX)]
        if not self._onset or not self._max_z:
            raise RankerError("The heuristic scorer needs the temporal and signal groups")

    def score(self, features: Floats) -> Floats:
        """Return the score of each row. A service with no signal or onset ranks last."""
        z_scores = np.nan_to_num(features[:, self._max_z], nan=0.0).max(axis=1)
        onset = np.nan_to_num(features[:, self._onset[0]], nan=NO_ONSET_RANK)
        scores: Floats = z_scores - onset
        return scores


def _fit_booster(
    features: Floats,
    labels: npt.NDArray[np.int32],
    columns: Sequence[str],
    settings: RetrievalConfig,
    seed: int,
    threads: int,
) -> BoosterScorer:
    params: dict[str, Any] = {
        "objective": "binary",
        "verbosity": -1,
        "num_threads": threads,
        **lightgbm_seed_params(seed),
        **settings.params,
    }
    data = lgb.Dataset(features, label=labels, feature_name=list(columns))
    return BoosterScorer(lgb.train(params, data, num_boost_round=settings.num_boost_round))


def fit_scorer(matrix: RankingMatrix, settings: RetrievalConfig, seed: int, threads: int) -> Scorer:
    """Fit the configured scorer on every row of ``matrix``."""
    if settings.scorer == "heuristic":
        return HeuristicScorer(matrix.columns)
    return _fit_booster(matrix.features, matrix.labels, matrix.columns, settings, seed, threads)


def _assign_parts(
    case_ids: Sequence[str], groups: Mapping[str, str], parts: int, seed: int
) -> list[int]:
    """Give each case a part, keeping every group whole."""
    names = sorted({groups[case] for case in case_ids})
    if len(names) < 2:
        raise RankerError("Cross-fitting needs at least two groups")
    rng = np.random.default_rng(derive_case_seed("stage1", "cross_fit", {}, seed))
    order = rng.permutation(len(names)).tolist()
    count = min(parts, len(names))
    part_of = {names[index]: position % count for position, index in enumerate(order)}
    return [part_of[groups[case]] for case in case_ids]


def cross_fit_scores(
    matrix: RankingMatrix,
    groups: Mapping[str, str],
    settings: RetrievalConfig,
    seed: int,
    threads: int,
) -> Floats:
    """Score each case with a scorer fitted on other groups."""
    if settings.scorer == "heuristic":
        return HeuristicScorer(matrix.columns).score(matrix.features)
    parts = _assign_parts(matrix.case_ids, groups, settings.cross_fit_folds, seed)
    scores = np.zeros(matrix.labels.size, dtype=np.float64)
    spans = list(matrix.spans())
    for part in sorted(set(parts)):
        held = [i for i, value in enumerate(parts) if value == part]
        kept = [i for i, value in enumerate(parts) if value != part]
        scorer = fit_scorer(matrix.cases(kept), settings, seed, threads)
        rows = np.concatenate([np.arange(*spans[i], dtype=np.intp) for i in held])
        scores[rows] = scorer.score(matrix.features[rows])
    return scores


def top_k_rows(
    scores: Floats, sizes: Sequence[int], k: int
) -> tuple[npt.NDArray[np.intp], list[int]]:
    """Return the best ``k`` rows of each case, in row order, capped by the case size.

    Rows are sorted by service within a case, so a stable sort breaks ties by service name, as the
    metrics do.
    """
    if k < 1:
        raise RankerError(f"k must be at least 1, got {k}")
    chosen: list[npt.NDArray[np.intp]] = []
    kept: list[int] = []
    start = 0
    for size in sizes:
        count = min(k, size)
        best = np.argsort(-scores[start : start + size], kind="stable")[:count]
        chosen.append(np.sort(best) + start)
        kept.append(count)
        start += size
    return np.concatenate(chosen), kept


def candidates(matrix: RankingMatrix, scores: Floats, k: int) -> RankingMatrix:
    """Return the top ``k`` documents of every case."""
    rows, sizes = top_k_rows(scores, matrix.sizes, k)
    return matrix.take(rows, sizes)


def score_frame(case_id: str, services: Sequence[str], scores: Floats) -> pl.DataFrame:
    """Return Stage 1 scores of one case, in the prediction contract."""
    return score_table(case_id, dict(zip(services, scores.tolist(), strict=True)))
