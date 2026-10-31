"""Every perturbation by id, and the parameter grid from ``configs/perturbations.yaml``.

P9 and P10 are held-out conditions, not transforms. They appear here as references to the LOFO and
LOSO splits so the suite enumerates them, and they have no ``apply``.
"""

import itertools
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cascadelens.config import PerturbationsConfig, PerturbationSpec, load_config
from cascadelens.perturb.base import Perturbation, PerturbationError, Realism
from cascadelens.perturb.loss import (
    DelayedSeries,
    FailureCorrelatedLoss,
    MetricDrift,
    RandomMetricLoss,
    SamplingRateDegradation,
    WholeModalityLoss,
)
from cascadelens.perturb.topology import EdgeAddition, EdgeDeletion, InferredTopology
from cascadelens.perturb.window import WindowNoise, WindowShift

CONFIG_PATH = Path("configs/perturbations.yaml")


@dataclass(frozen=True)
class SplitReference:
    """A held-out condition that a split provides."""

    id: str
    name: str
    split: str
    realism: Realism = "held-out"


PERTURBATIONS: dict[str, type[Perturbation[Any]]] = {
    item.id: item
    for item in (
        RandomMetricLoss,
        FailureCorrelatedLoss,
        WholeModalityLoss,
        WindowShift,
        WindowNoise,
        EdgeDeletion,
        EdgeAddition,
        InferredTopology,
        SamplingRateDegradation,
        DelayedSeries,
        MetricDrift,
    )
}
REFERENCES = {
    item.id: item
    for item in (
        SplitReference("P9", "unseen_fault_type", "lofo"),
        SplitReference("P10", "unseen_system", "loso"),
    )
}


def build(identifier: str, params: Mapping[str, Any]) -> Perturbation[Any]:
    """Return a perturbation by id or name."""
    for item in PERTURBATIONS.values():
        if identifier in (item.id, item.name):
            return item.build(params)
    for reference in REFERENCES.values():
        if identifier in (reference.id, reference.name):
            raise PerturbationError(
                f"{reference.id} is the {reference.split} split, not a transform"
            )
    raise PerturbationError(f"Unknown perturbation {identifier!r}")


def parse_params(items: Sequence[str]) -> dict[str, str]:
    """Parse ``key=value`` strings. The parameter schema converts the values."""
    pairs = (item.split("=", 1) for item in items)
    parsed = {}
    for pair in pairs:
        if len(pair) != 2:
            raise PerturbationError(f"Expected key=value, got {pair[0]!r}")
        parsed[pair[0]] = pair[1]
    return parsed


def load_suite(root: Path = Path()) -> PerturbationsConfig:
    """Read the suite config."""
    return load_config(root / CONFIG_PATH, PerturbationsConfig)


def expand(spec: PerturbationSpec) -> Iterator[dict[str, Any]]:
    """Yield every parameter combination of one entry."""
    keys = sorted(spec.params)
    for values in itertools.product(*(spec.params[key] for key in keys)):
        yield dict(zip(keys, values, strict=True))


def grid(config: PerturbationsConfig) -> Iterator[tuple[PerturbationSpec, dict[str, Any]]]:
    """Yield every transform of the suite with its parameters, leaving out split references."""
    for spec in config.perturbations:
        if spec.id in PERTURBATIONS:
            for params in expand(spec):
                yield spec, params
