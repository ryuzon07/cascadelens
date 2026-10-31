"""Stage 2: LambdaRank over Stage 1 candidates, one query per case (PRD 8.3).

The ranker implements the baseline interface, so it runs through the same runner, windows and splits
as every baseline. ``predict`` returns candidates only: a case whose truth was not retrieved has no
prediction for it, and the metrics count that as a miss.

One search runs per fold with a fixed tuning seed, so every model seed of a fold uses the same
choice, as for the flat baseline. Reuse one instance across seeds so the search runs once.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar

import lightgbm as lgb
import numpy as np
import polars as pl

from cascadelens.baselines.base import Baseline, CaseHandle, SummaryCache, score_table
from cascadelens.config import BaselineConfig, RankerConfig, RerankerConfig
from cascadelens.models.data import (
    Floats,
    RankerError,
    RankingMatrix,
    TableSource,
    build_matrix,
    model_features,
)
from cascadelens.models.retrieval import (
    Scorer,
    candidates,
    cross_fit_scores,
    fit_scorer,
    score_frame,
    top_k_rows,
)
from cascadelens.models.tuning import Choice, fold_key, sample_configurations, search
from cascadelens.seed import lightgbm_seed_params

AVG_K = 5


class MeanNdcgStop:
    """Stop when the mean of the validation NDCGs has not improved for ``patience`` rounds.

    NDCG@1 alone plateaus at once on a small validation fold, and LightGBM's own callback can only
    follow one metric or any metric. Ties keep the earlier round.
    """

    order = 30

    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = -np.inf
        self.best_round = 0
        self.best_results: list[Any] = []

    def __call__(self, env: lgb.callback.CallbackEnv) -> None:
        """Track the best round, and end training when patience runs out or rounds do."""
        results = env.evaluation_result_list or []
        value = float(np.mean([item[2] for item in results]))
        if value > self.best:
            self.best, self.best_round, self.best_results = value, env.iteration, list(results)
        last = env.iteration == env.end_iteration - 1
        if last or env.iteration - self.best_round >= self.patience:
            raise lgb.callback.EarlyStopException(self.best_round, self.best_results)


def train_ranker(
    train: RankingMatrix,
    val: RankingMatrix,
    params: Mapping[str, Any],
    settings: RerankerConfig,
    seed: int,
    threads: int,
) -> lgb.Booster:
    """Train one LambdaRank model, early-stopped on the validation queries with a positive."""
    fitted, stopping = train.with_positive(), val.with_positive()
    if not fitted.sizes or not stopping.sizes:
        raise RankerError("Training and validation each need a case whose truth was retrieved")
    full = {
        "objective": settings.objective,
        "metric": "ndcg",
        "eval_at": settings.ndcg_at,
        "verbosity": -1,
        "num_threads": threads,
        "bagging_freq": 1,
        "bagging_by_query": True,
        **lightgbm_seed_params(seed),
        **settings.params,
        **params,
    }
    data = lgb.Dataset(
        fitted.features,
        label=fitted.labels,
        group=list(fitted.sizes),
        feature_name=list(fitted.columns),
        params={"feature_pre_filter": False},
    )
    held = lgb.Dataset(
        stopping.features,
        label=stopping.labels,
        group=list(stopping.sizes),
        reference=data,
        params={"feature_pre_filter": False},
    )
    return lgb.train(
        full,
        data,
        num_boost_round=settings.num_boost_round,
        valid_sets=[held],
        valid_names=["val"],
        callbacks=[MeanNdcgStop(settings.early_stopping_rounds)],
    )


def query_scores(booster: lgb.Booster, matrix: RankingMatrix) -> tuple[float, float]:
    """Return Top-1 and Avg@5 over all cases of ``matrix``. A case with no positive is a miss."""
    scores = np.asarray(booster.predict(matrix.features), dtype=np.float64)
    top1, avg5 = 0.0, 0.0
    for start, end in matrix.spans():
        order = np.argsort(-scores[start:end], kind="stable")
        positive = np.flatnonzero(matrix.labels[start:end][order])
        if positive.size:
            rank = int(positive[0]) + 1
            top1 += float(rank == 1)
            avg5 += max(0, AVG_K + 1 - rank) / AVG_K
    count = len(matrix.sizes)
    return top1 / count, avg5 / count


@dataclass
class FittedFold:
    """The models of one fold and seed, with what they did on validation."""

    scorer: Scorer | None
    booster: lgb.Booster
    choice: Choice
    columns: tuple[str, ...]
    val_top1: float
    stage1: list[pl.DataFrame] = field(default_factory=list)

    def rank(self, table: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame | None]:
        """Return Stage 2 predictions and, when the method retrieves, Stage 1 scores."""
        case_id = str(table["case_id"][0])
        services = table["service"].to_list()
        features = model_features(table, self.columns)
        retrieved = None
        rows = np.arange(len(services), dtype=np.intp)
        if self.scorer is not None:
            if self.choice.k is None:
                raise RankerError("A retrieving model needs a K")
            first = self.scorer.score(features)
            retrieved = score_frame(case_id, services, first)
            rows, _ = top_k_rows(first, [len(services)], self.choice.k)
        scores: Floats = np.asarray(self.booster.predict(features[rows]), dtype=np.float64)
        names = [services[row] for row in rows.tolist()]
        return score_table(case_id, dict(zip(names, scores.tolist(), strict=True))), retrieved


class _Prepared:
    """Train and validation matrices of one fold, with candidates made once per K."""

    def __init__(
        self,
        train: RankingMatrix,
        val: RankingMatrix,
        train_scores: Floats | None,
        val_scores: Floats | None,
    ) -> None:
        self.train, self.val = train, val
        self.train_scores, self.val_scores = train_scores, val_scores
        self._pairs: dict[int | None, tuple[RankingMatrix, RankingMatrix]] = {}

    def pair(self, k: int | None) -> tuple[RankingMatrix, RankingMatrix]:
        """Return the training and validation documents for ``k``."""
        if k not in self._pairs:
            if k is None or self.train_scores is None or self.val_scores is None:
                self._pairs[k] = (self.train, self.val)
            else:
                self._pairs[k] = (
                    candidates(self.train, self.train_scores, k),
                    candidates(self.val, self.val_scores, k),
                )
        return self._pairs[k]


class TwoStageRanker(Baseline):
    """Stage 1 retrieval followed by LambdaRank on the candidates."""

    name = "two_stage_ranker"
    supervised = True
    retrieves: ClassVar[bool] = True

    def __init__(
        self,
        source: TableSource,
        groups: Mapping[str, str] | None = None,
        frozen: Mapping[str, Choice] | None = None,
        cache: SummaryCache | None = None,
    ) -> None:
        super().__init__(cache)
        self.source = source
        self.groups = dict(groups) if groups is not None else {}
        self.frozen = frozen
        self.folds: list[FittedFold] = []
        self._chosen: dict[str, Choice] = {}

    def _matrix(self, cases: Sequence[CaseHandle], settings: RankerConfig) -> RankingMatrix:
        truth = {case.record.case_id: case.record.root_cause_service for case in cases}
        return build_matrix([self.source.table(case) for case in cases], truth, settings.groups)

    def _group_map(self, cases: Sequence[CaseHandle]) -> dict[str, str]:
        ids = (case.record for case in cases)
        return {r.case_id: self.groups.get(r.case_id, r.run_id) for r in ids}

    def _prepare(
        self,
        train: RankingMatrix,
        val: RankingMatrix,
        groups: Mapping[str, str],
        settings: RankerConfig,
        seed: int,
    ) -> tuple[_Prepared, Scorer | None]:
        if not self.retrieves:
            return _Prepared(train, val, None, None), None
        retrieval, threads = settings.retrieval, settings.num_threads
        scorer = fit_scorer(train, retrieval, seed, threads)
        out_of_fold = cross_fit_scores(train, groups, retrieval, seed, threads)
        return _Prepared(train, val, out_of_fold, scorer.score(val.features)), scorer

    def search_fold(
        self,
        train_cases: Sequence[CaseHandle],
        val_cases: Sequence[CaseHandle],
        settings: RankerConfig,
    ) -> Choice:
        """Choose a configuration for one fold. This reads training and validation cases only."""
        if not val_cases:
            raise RankerError(f"{self.name} needs validation cases for its search")
        key = fold_key(
            [c.record.case_id for c in train_cases],
            [c.record.case_id for c in val_cases],
            settings,
        )
        if self.frozen is not None:
            if key not in self.frozen:
                raise RankerError(f"The frozen record has no choice for fold {key[:12]}")
            return self.frozen[key]
        if key not in self._chosen:
            seed = settings.tuning.seed
            prepared, _ = self._prepare(
                self._matrix(train_cases, settings),
                self._matrix(val_cases, settings),
                self._group_map(train_cases),
                settings,
                seed,
            )
            choices = settings.retrieval.top_k_choices if self.retrieves else None

            def evaluate(params: dict[str, Any], k: int | None) -> tuple[float, float]:
                train, val = prepared.pair(k)
                booster = train_ranker(
                    train, val, params, settings.reranker, seed, settings.num_threads
                )
                return query_scores(booster, val)

            options = sample_configurations(settings.tuning.budget, seed, choices)
            self._chosen[key] = search(key, options, evaluate)
        return self._chosen[key]

    def fit(
        self,
        train_cases: Sequence[CaseHandle],
        config: BaselineConfig,
        val_cases: Sequence[CaseHandle] = (),
        seed: int = 0,
    ) -> None:
        """Take the fold's choice, then train the final models on the training cases."""
        settings = RankerConfig.model_validate(config.params)
        choice = self.search_fold(train_cases, val_cases, settings)
        train_matrix = self._matrix(train_cases, settings)
        prepared, scorer = self._prepare(
            train_matrix,
            self._matrix(val_cases, settings),
            self._group_map(train_cases),
            settings,
            seed,
        )
        train, val = prepared.pair(choice.k)
        booster = train_ranker(
            train, val, choice.params, settings.reranker, seed, settings.num_threads
        )
        top1, _ = query_scores(booster, val)
        self.folds.append(FittedFold(scorer, booster, choice, train_matrix.columns, top1))

    def predict(self, case: CaseHandle) -> pl.DataFrame:
        """Score the candidates of the case."""
        if not self.folds:
            raise RankerError(f"{self.name} must be fitted before predicting")
        fold = self.folds[-1]
        predictions, retrieved = fold.rank(self.source.table(case))
        if retrieved is not None:
            fold.stage1.append(retrieved)
        return predictions

    def take_folds(self) -> list[FittedFold]:
        """Return the fitted folds since the last call, in fold order."""
        folds, self.folds = self.folds, []
        return folds
