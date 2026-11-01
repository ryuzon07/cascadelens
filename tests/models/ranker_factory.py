"""Synthetic feature tables and small projects for ranker tests."""

import json
import shutil
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from cascadelens.baselines.base import CaseHandle
from cascadelens.config import BaselineConfig, RankerConfig
from cascadelens.data.schema import CaseRecord
from cascadelens.features.config import GroupName
from cascadelens.features.registry import feature_columns

ALL_GROUPS: list[GroupName] = ["signal", "temporal", "graph", "structural", "mask"]
CONFIGS = Path(__file__).parents[2] / "configs"
SIGNAL = "signal_cpu_max_z"


class DictSource:
    """Feature tables held in memory."""

    def __init__(self, tables: Mapping[str, pl.DataFrame]) -> None:
        self.tables = tables

    def table(self, case: CaseHandle) -> pl.DataFrame:
        return self.tables[case.record.case_id]


def settings(
    scorer: str = "lightgbm", budget: int = 2, choices: tuple[int, ...] = (2, 3)
) -> RankerConfig:
    return RankerConfig.model_validate(
        {
            "seeds": [0, 1],
            "retrieval": {
                "scorer": scorer,
                "top_k_choices": list(choices),
                "cross_fit_folds": 2,
                "num_boost_round": 15,
                "params": {"min_data_in_leaf": 5, "num_leaves": 7},
            },
            "reranker": {"num_boost_round": 40, "early_stopping_rounds": 10},
            "tuning": {"budget": budget, "seed": 0},
            "num_threads": 1,
        }
    )


def wrapped(config: RankerConfig, name: str = "two_stage_ranker") -> BaselineConfig:
    return BaselineConfig(
        name=name, supervised=True, seeds=config.seeds, params=config.model_dump(mode="json")
    )


def record(case_id: str, truth: str, run_id: str) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        suite="suite",
        system="shop",
        fault_type="cpu",
        run_id=run_id,
        repetition=0,
        root_cause_service=truth,
        inject_time=100,
        window_start=0,
        window_end=200,
    )


def _no_metrics() -> pl.DataFrame:
    raise AssertionError("Synthetic cases have no metrics")


def synthetic(
    cases: int, services: int = 6, seed: int = 0, shuffled: bool = False, marker: str = SIGNAL
) -> tuple[list[CaseHandle], DictSource]:
    """Return cases whose ``marker`` marks the truth, or an unrelated service when ``shuffled``."""
    rng = np.random.default_rng(seed)
    names = [f"svc-{index}" for index in range(services)]
    columns = feature_columns(ALL_GROUPS)
    handles, tables = [], {}
    for number in range(cases):
        case_id = f"case-{number}"
        truth = names[int(rng.integers(services))]
        marked = names[int(rng.integers(services))] if shuffled else truth
        data: dict[str, object] = {
            "case_id": [case_id] * services,
            "service": names,
            **{c: rng.normal(size=services).tolist() for c in columns},
        }
        data[marker] = [float(name == marked) for name in names]
        tables[case_id] = pl.DataFrame(data)
        handles.append(CaseHandle(record(case_id, truth, f"run-{number // 2}"), _no_metrics))
    return handles, DictSource(tables)


def write_project_configs(root: Path) -> None:
    """Add the feature, ranker and experiment configs to a ``build_project`` directory."""
    shutil.copy(CONFIGS / "features.yaml", root / "configs" / "features.yaml")
    clean = root / "experiments" / "baselines_clean"
    clean.mkdir(parents=True, exist_ok=True)
    shutil.copy(CONFIGS.parent / "experiments" / "baselines_clean" / "config.yaml", clean)
    config = settings().model_dump(mode="json")
    (root / "configs" / "ranker.yaml").write_text(json.dumps(config), encoding="utf-8")
    directory = root / "experiments" / "ranker_clean"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.yaml").write_text(
        "seed: 0\nn_resamples: 200\nalpha: 0.05\n"
        "methods: [two_stage_ranker, single_stage_ranker]\n",
        encoding="utf-8",
    )
