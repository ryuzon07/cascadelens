"""Typed experiment configs, YAML loading and stable config hashing."""

import hashlib
import json
from pathlib import Path
from typing import Any, Literal, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict, Field

SplitStrategy = Literal["random", "group_run", "lofo", "loso"]
GridValue = int | float | str


class StrictModel(BaseModel):
    """Immutable base model that rejects unknown keys, so config typos fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DataConfig(StrictModel):
    """Where benchmark data lives and which suites are ingested."""

    source: str
    suites: list[str] = Field(default_factory=list)
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"


class SplitsConfig(StrictModel):
    """Which splitting protocols are generated."""

    seed: int = Field(ge=0)
    strategies: list[SplitStrategy]
    group_key: Literal["run_id"] = "run_id"
    n_folds: int = Field(default=5, ge=2)
    val_fraction: float = Field(default=0.2, gt=0, lt=1)
    near_duplicate_threshold: float = Field(default=0.98, ge=0, le=1)
    signature_bins: int = Field(default=64, ge=2)
    processed_dir: str = "data/processed"
    output_dir: str = "data/splits"


class PerturbationSpec(StrictModel):
    """One perturbation and its parameter grid."""

    id: str = Field(pattern=r"^P\d+$")
    name: str
    realism: str
    params: dict[str, list[GridValue]] = Field(default_factory=dict)


class PerturbationsConfig(StrictModel):
    """The perturbation suite."""

    seed: int = Field(ge=0)
    perturbations: list[PerturbationSpec]


class RetrievalConfig(StrictModel):
    """Stage 1 candidate retrieval."""

    top_k_choices: list[int] = Field(min_length=1)
    scorer: Literal["lightgbm", "heuristic"] = "lightgbm"
    cross_fit_folds: int = Field(default=5, ge=2)
    num_boost_round: int = Field(default=200, ge=1)
    params: dict[str, Any] = Field(default_factory=dict)


class RerankerConfig(StrictModel):
    """Stage 2 LightGBM reranker."""

    objective: Literal["lambdarank"] = "lambdarank"
    params: dict[str, Any] = Field(default_factory=dict)
    ndcg_at: list[int] = Field(default=[1, 3, 5], min_length=1)
    num_boost_round: int = Field(default=500, ge=1)
    early_stopping_rounds: int = Field(default=30, ge=1)


class TuningConfig(StrictModel):
    """The fixed-budget random search."""

    budget: int = Field(default=30, ge=1)
    seed: int = Field(default=0, ge=0)


class RankerConfig(StrictModel):
    """The two-stage CascadeLens ranker."""

    seeds: list[int] = Field(min_length=1)
    retrieval: RetrievalConfig
    reranker: RerankerConfig
    tuning: TuningConfig = TuningConfig()
    groups: list[Literal["signal", "temporal", "graph", "structural", "mask"]] = Field(
        default=["signal", "temporal", "graph", "structural", "mask"], min_length=1
    )
    features_config: str = "configs/features.yaml"
    num_threads: int = Field(default=4, ge=1)
    structural_dominance: float = Field(default=0.25, gt=0, le=1)


class BaselineConfig(StrictModel):
    """One baseline method."""

    name: str
    supervised: bool
    seeds: list[int] = Field(min_length=1)
    params: dict[str, Any] = Field(default_factory=dict)


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_config(path: str | Path, schema: type[ConfigT]) -> ConfigT:
    """Read a YAML file and validate it against ``schema``."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping")
    return schema.model_validate(raw)


def canonical_json(value: object) -> str:
    """Serialize ``value`` to one canonical JSON string, independent of key order."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def config_hash(config: BaseModel) -> str:
    """Return a SHA-256 hex digest of the validated config, defaults included."""
    payload = canonical_json(config.model_dump(mode="json"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
