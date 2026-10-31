"""Compose perturbations in the documented order: data loss, then window, then topology.

Order does not depend on how steps are listed. Within a stage, steps run in the order given. Each
step draws from its own generator, so composing equals applying the steps by hand in that order.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from cascadelens.data.schema import Case
from cascadelens.perturb.base import ManifestRow, Perturbation


@dataclass(frozen=True)
class Pipelined:
    """A perturbed case with one manifest row per step."""

    case: Case
    rows: list[ManifestRow]


class Pipeline:
    """Perturbations that run together."""

    def __init__(self, steps: Sequence[Perturbation[Any]]) -> None:
        self.steps = sorted(steps, key=lambda step: step.stage)

    def run(self, case: Case, seed: int) -> Pipelined:
        """Apply every step in order."""
        rows = []
        for step in self.steps:
            applied = step.run(case, seed)
            case = applied.case
            rows.append(applied.row)
        return Pipelined(case, rows)
