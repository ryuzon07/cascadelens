"""Gain-based feature importance by feature group, per fold."""

from collections.abc import Iterable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from cascadelens.features.registry import REGISTRY

STRUCTURAL = "structural"


def _group_of(column: str) -> str:
    for name, group in REGISTRY.items():
        if column.startswith(group.prefix):
            return name
    return "other"


def gain_shares(booster: lgb.Booster) -> dict[str, float]:
    """Return each column's share of total gain, summing to 1 unless the model made no split."""
    gains = np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)
    total = float(gains.sum())
    names: list[str] = booster.feature_name()
    return {n: float(g) / total if total > 0 else 0.0 for n, g in zip(names, gains, strict=True)}


def group_shares(shares: Mapping[str, float]) -> dict[str, float]:
    """Sum column shares by feature group."""
    totals: dict[str, float] = {}
    for column, share in shares.items():
        name = _group_of(column)
        totals[name] = totals.get(name, 0.0) + share
    return dict(sorted(totals.items()))


def mean_shares(shares: Iterable[Mapping[str, float]]) -> dict[str, float]:
    """Average column shares across models, such as the seeds of one fold."""
    frames = list(shares)
    return {column: float(np.mean([s[column] for s in frames])) for column in frames[0]}


def top_columns(shares: Mapping[str, float], count: int = 10) -> dict[str, float]:
    """Return the ``count`` largest columns, ties by name."""
    ordered = sorted(shares.items(), key=lambda item: (-item[1], item[0]))
    return dict(ordered[:count])


def dominant_structural(shares: Mapping[str, float], threshold: float) -> list[str]:
    """Return the structural columns holding at least ``threshold`` of total gain."""
    return sorted(
        column
        for column, share in shares.items()
        if _group_of(column) == STRUCTURAL and share >= threshold
    )


def structural_warnings(
    entries: Sequence[tuple[str, Mapping[str, float]]], threshold: float
) -> list[str]:
    """Return one message per labelled model where a structural column dominates."""
    return [
        f"WARNING: {label}: {column} holds {shares[column]:.1%} of gain, so the model may be "
        "fingerprinting nodes"
        for label, shares in entries
        for column in dominant_structural(shares, threshold)
    ]
