"""Fixed-budget random search (PRD 7.1 and 8.3).

Selection reads validation cases only, and those are carved from training groups. K is drawn inside
each configuration rather than multiplying them, so the budget stays equal to the flat baseline's.
"""

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cascadelens.config import RankerConfig, canonical_json, config_hash
from cascadelens.models.data import RankerError
from cascadelens.seed import derive_case_seed


@dataclass(frozen=True)
class Choice:
    """The selected configuration of one fold, and what validation said about it."""

    key: str
    params: dict[str, Any]
    k: int | None
    val_top1: float
    val_avg5: float

    def to_dict(self) -> dict[str, Any]:
        """Return plain values for YAML and JSON."""
        return asdict(self)


def fold_key(train_ids: Sequence[str], val_ids: Sequence[str], config: RankerConfig) -> str:
    """Identify one fold under one config."""
    payload = canonical_json([sorted(train_ids), sorted(val_ids), config_hash(config)])
    return hashlib.sha256(payload.encode()).hexdigest()


def sample_configurations(
    budget: int, seed: int, k_choices: Sequence[int] | None
) -> list[tuple[dict[str, Any], int | None]]:
    """Draw ``budget`` hyperparameter sets, each with a K when the method retrieves."""
    rng = np.random.default_rng(derive_case_seed("ranker", "search", {}, seed))
    drawn: list[tuple[dict[str, Any], int | None]] = []
    for _ in range(budget):
        params: dict[str, Any] = {
            "learning_rate": float(10 ** rng.uniform(-2, -0.5)),
            "num_leaves": int(rng.integers(4, 32)),
            "min_data_in_leaf": int(rng.integers(5, 50)),
            "lambda_l1": float(10 ** rng.uniform(-3, 1)),
            "lambda_l2": float(10 ** rng.uniform(-3, 1)),
            "feature_fraction": float(rng.uniform(0.5, 1.0)),
            "bagging_fraction": float(rng.uniform(0.5, 1.0)),
        }
        drawn.append((params, int(rng.choice(k_choices)) if k_choices else None))
    return drawn


def search(
    key: str,
    options: Sequence[tuple[dict[str, Any], int | None]],
    evaluate: Callable[[dict[str, Any], int | None], tuple[float, float]],
) -> Choice:
    """Return the option with the best validation Top-1, then Avg@5, then the earliest drawn."""
    if not options:
        raise RankerError("The search has no configurations")
    best: Choice | None = None
    for params, k in options:
        top1, avg5 = evaluate(params, k)
        if best is None or (top1, avg5) > (best.val_top1, best.val_avg5):
            best = Choice(key, params, k, top1, avg5)
    assert best is not None
    return best
