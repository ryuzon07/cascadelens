"""Apply Part 8 perturbations to a loaded case for the demo's sliders.

Failure-correlated loss reads a label to simulate damage, so it needs a sample's ``simulation``
block. The label stays inside the perturbation and is never shown or returned.
"""

import json
from dataclasses import dataclass
from typing import Any

import polars as pl

from cascadelens.data.schema import Case
from cascadelens.inference import CaseInput, CasePayload, case_from_payload
from cascadelens.models.data import RankerError
from cascadelens.perturb.base import Perturbation, Stage, crop_to_window
from cascadelens.perturb.pipeline import Pipeline
from cascadelens.perturb.registry import build

SEED = 0
PLACEHOLDER = "unknown"


@dataclass(frozen=True)
class Simulation:
    """The simulated fault of a sample case."""

    root_cause_service: str
    inject_time: int


@dataclass(frozen=True)
class Damage:
    """Slider positions. Rates are shares and the shift is in seconds."""

    random_loss: float = 0.0
    failure_loss: float = 0.0
    window_shift: int = 0
    edge_deletion: float = 0.0


def parse_sample(text: str) -> tuple[CaseInput, Simulation | None]:
    """Return a JSON case and its simulation block, if it has one."""
    raw: dict[str, Any] = json.loads(text)
    block = raw.pop("simulation", None)
    simulation = Simulation(**block) if block else None
    return case_from_payload(CasePayload.model_validate(raw)), simulation


def _as_case(case: CaseInput, simulation: Simulation | None) -> Case:
    return Case(
        case_id=case.case_id,
        suite="demo",
        system="demo",
        fault_type=PLACEHOLDER,
        run_id=case.case_id,
        repetition=0,
        root_cause_service=simulation.root_cause_service if simulation else PLACEHOLDER,
        inject_time=simulation.inject_time if simulation else case.window_start,
        window_start=case.window_start,
        window_end=case.window_end,
        topology=case.topology,
        metrics=case.metrics.with_columns(pl.col("value").fill_nan(None)),
    )


def apply_damage(case: CaseInput, simulation: Simulation | None, damage: Damage) -> CaseInput:
    """Return the case after the chosen perturbations, without changing the input."""
    if damage.failure_loss and simulation is None:
        raise RankerError("Failure-correlated loss needs a sample case with a label")
    steps: list[Perturbation[Any]] = []
    if damage.random_loss:
        steps.append(build("P1", {"level": "sample", "rate": damage.random_loss}))
    if damage.failure_loss:
        steps.append(build("P2", {"rate": damage.failure_loss}))
    if damage.window_shift:
        steps.append(build("P4", {"offset_seconds": damage.window_shift}))
    if damage.edge_deletion and case.topology:
        steps.append(build("P6", {"rate": damage.edge_deletion}))
    if not steps:
        return case
    return perturbed_case(case, simulation, steps, SEED)


def perturbed_case(
    case: CaseInput, simulation: Simulation | None, steps: list[Perturbation[Any]], seed: int
) -> CaseInput:
    """Return the case after ``steps``, cropped to any window they moved."""
    changed = Pipeline(steps).run(_as_case(case, simulation), seed).case
    if any(step.stage == Stage.WINDOW for step in steps):
        changed = crop_to_window(changed)
    return CaseInput(
        case.case_id,
        changed.metrics,
        changed.window_start,
        changed.window_end,
        changed.topology,
    )
