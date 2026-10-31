"""Seeding helpers for reproducible runs."""

import hashlib
import random
from collections.abc import Mapping
from typing import Any

import numpy as np

from cascadelens.config import canonical_json

MAX_NUMPY_SEED = 2**32 - 1
CASE_SEED_BITS = 31  # LightGBM stores seeds as signed 32-bit integers.


def set_seed(seed: int) -> None:
    """Seed the process-global Python and NumPy generators.

    LightGBM has no global generator; pass ``lightgbm_seed_params(seed)`` to each model.
    """
    if not 0 <= seed <= MAX_NUMPY_SEED:
        raise ValueError(f"seed must be in [0, {MAX_NUMPY_SEED}], got {seed}")
    random.seed(seed)
    np.random.seed(seed)


def lightgbm_seed_params(seed: int) -> dict[str, Any]:
    """Return LightGBM parameters that make training deterministic for ``seed``."""
    return {"seed": seed, "deterministic": True, "force_row_wise": True}


def derive_case_seed(case_id: str, perturbation: str, params: Mapping[str, Any], seed: int) -> int:
    """Derive a stable seed from (case_id, perturbation, params, seed).

    Unlike ``hash()``, the result does not change between processes or platforms.
    """
    payload = canonical_json(
        {"case_id": case_id, "perturbation": perturbation, "params": params, "seed": seed}
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") >> (64 - CASE_SEED_BITS)
