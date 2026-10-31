"""The flat LightGBM baseline: LambdaRank on per-service signal features, one query per case.

Hyperparameters are chosen on the validation cases, which the splitters carve from training groups
only. One search runs per fold with a fixed ``search_seed``, so every model seed of a fold uses the
same choice. The final model trains on the training cases alone.
"""

import hashlib
from collections.abc import Sequence
from typing import Any

import lightgbm as lgb
import numpy as np
import numpy.typing as npt
import polars as pl

from cascadelens.baselines.base import Baseline, BaselineError, CaseHandle, score_table
from cascadelens.config import BaselineConfig, StrictModel, canonical_json
from cascadelens.features.simple import FEATURE_COLUMNS, SignalSettings
from cascadelens.seed import derive_case_seed, lightgbm_seed_params


class FlatParams(StrictModel):
    """Settings of the flat baseline."""

    signal: SignalSettings = SignalSettings()
    search_budget: int = 30
    search_seed: int = 0
    num_boost_round: int = 100
    num_threads: int = 4
    masking_augmentation: bool = False


def sample_search_space(budget: int, seed: int) -> list[dict[str, Any]]:
    """Draw ``budget`` configurations from the small space of PRD section 8.3."""
    rng = np.random.default_rng(derive_case_seed("lightgbm_flat", "search", {}, seed))
    return [
        {
            "learning_rate": float(10 ** rng.uniform(-2, -0.5)),
            "num_leaves": int(rng.integers(4, 32)),
            "min_data_in_leaf": int(rng.integers(5, 50)),
            "lambda_l1": float(10 ** rng.uniform(-3, 1)),
            "lambda_l2": float(10 ** rng.uniform(-3, 1)),
            "feature_fraction": float(rng.uniform(0.5, 1.0)),
        }
        for _ in range(budget)
    ]


class _Matrix:
    """Features, labels and query sizes of some cases, with services in summary order."""

    def __init__(self, summaries: Sequence[pl.DataFrame], labels: Sequence[str]) -> None:
        self.features = pl.concat(s.select(FEATURE_COLUMNS) for s in summaries).to_numpy()
        self.groups = [summary.height for summary in summaries]
        self.labels = np.concatenate(
            [
                (summary["service"] == label).to_numpy().astype(np.int32)
                for summary, label in zip(summaries, labels, strict=True)
            ]
        )


def _top1(model: lgb.Booster, matrix: _Matrix) -> float:
    scores = np.asarray(model.predict(matrix.features), dtype=np.float64)
    hits, start = 0, 0
    for size in matrix.groups:
        best = int(np.argmax(scores[start : start + size]))
        hits += int(matrix.labels[start + best])
        start += size
    return hits / len(matrix.groups)


class LightGBMFlat(Baseline):
    """Rank services with LambdaRank on ``FEATURE_COLUMNS``."""

    name = "lightgbm_flat"
    supervised = True

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._params = FlatParams()
        self._model: lgb.Booster | None = None
        self._chosen: dict[str, dict[str, Any]] = {}

    def _matrix(self, cases: Sequence[CaseHandle]) -> _Matrix:
        summaries = [self.cache.get(case, self._params.signal) for case in cases]
        return _Matrix(summaries, [case.record.root_cause_service for case in cases])

    def _train(self, matrix: _Matrix, params: dict[str, Any], seed: int) -> lgb.Booster:
        settings = {
            "objective": "lambdarank",
            "metric": "none",
            "verbosity": -1,
            "num_threads": self._params.num_threads,
            **lightgbm_seed_params(seed),
            **params,
        }
        data = lgb.Dataset(
            matrix.features,
            label=matrix.labels,
            group=matrix.groups,
            feature_name=list(FEATURE_COLUMNS),
            params={"feature_pre_filter": False},
        )
        return lgb.train(settings, data, num_boost_round=self._params.num_boost_round)

    def _select(self, train: _Matrix, val: _Matrix, key: str) -> dict[str, Any]:
        if key not in self._chosen:
            options = sample_search_space(self._params.search_budget, self._params.search_seed)
            seed = self._params.search_seed
            scores = [_top1(self._train(train, option, seed), val) for option in options]
            self._chosen[key] = options[int(np.argmax(scores))]
        return self._chosen[key]

    def fit(
        self,
        train_cases: Sequence[CaseHandle],
        config: BaselineConfig,
        val_cases: Sequence[CaseHandle] = (),
        seed: int = 0,
    ) -> None:
        """Choose hyperparameters on validation, then train on the training cases."""
        self._params = FlatParams.model_validate(config.params)
        if self._params.masking_augmentation:
            raise BaselineError("Masking augmentation is not implemented yet")
        if not val_cases:
            raise BaselineError("lightgbm_flat needs validation cases for its search")
        ids = (
            sorted(case.record.case_id for case in train_cases),
            sorted(case.record.case_id for case in val_cases),
        )
        key = hashlib.sha256(
            canonical_json([ids, self._params.model_dump(mode="json")]).encode()
        ).hexdigest()
        train = self._matrix(train_cases)
        chosen = self._select(train, self._matrix(val_cases), key)
        self._model = self._train(train, chosen, seed)

    def chosen_params(self) -> list[dict[str, Any]]:
        """Return the configurations selected so far, in fold order."""
        return list(self._chosen.values())

    def predict(self, case: CaseHandle) -> pl.DataFrame:
        """Score every service of the case."""
        if self._model is None:
            raise BaselineError("lightgbm_flat must be fitted before predicting")
        summary = self.cache.get(case, self._params.signal)
        features = summary.select(FEATURE_COLUMNS).to_numpy()
        scores: npt.NDArray[np.float64] = np.asarray(
            self._model.predict(features), dtype=np.float64
        )
        return score_table(
            case.record.case_id,
            dict(zip(summary["service"].to_list(), scores.tolist(), strict=True)),
        )
