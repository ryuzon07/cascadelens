"""Simulate a change to a copy of an incident's case. Nothing here writes to the store.

Each request runs in a small thread pool. A thread cannot be stopped once it has started, so a
timed-out run keeps its slot until it ends, and new requests get a 503 until then.
"""

import asyncio
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from pydantic import ValidationError

from cascadelens.api.failures import ApiFailure
from cascadelens.api.incidents import IncidentService, model_with_threshold
from cascadelens.api.schemas import (
    CurvePoint,
    Decision,
    RankChange,
    Scenario,
    ScenarioService,
    SettingsUpdate,
    WhatIfRequest,
    WhatIfResult,
)
from cascadelens.api.store import DiagnosisRecord, Incident
from cascadelens.config import PerturbationsConfig, load_config
from cascadelens.demo.perturb import perturbed_case
from cascadelens.inference import CaseInput, DiagnoseResult, LoadedModel, diagnose
from cascadelens.perturb.base import Perturbation, PerturbationError
from cascadelens.perturb.registry import CONFIG_PATH, PERTURBATIONS, build

WORKERS = 2
WINDOW_SECONDS = 60.0
TOPOLOGY_PERTURBATIONS = ("P6", "P7")
# The numeric parameter behind each curve. Its values come from configs/perturbations.yaml.
INTENSITY = {
    "P1": "rate",
    "P4": "offset_seconds",
    "P5": "max_jitter_seconds",
    "P6": "rate",
    "P7": "rate",
    "P11": "factor",
}
T = TypeVar("T")


def decision_from(result: DiagnoseResult) -> Decision:
    """Return a diagnosis's decision."""
    if not result.calibrated:
        return "uncalibrated"
    return "abstained" if result.abstain else "confident"


def _scenario(result: DiagnoseResult) -> Scenario:
    ranking = [
        ScenarioService(service=item.service, rank=place + 1, score=item.score)
        for place, item in enumerate(result.ranking)
    ]
    return Scenario(ranking=ranking, confidence=result.confidence, decision=decision_from(result))


def rank_changes(original: Scenario, scenario: Scenario) -> list[RankChange]:
    """Return how each service moved, in original order and then any newly ranked ones."""
    before = {item.service: item.rank for item in original.ranking}
    after = {item.service: item.rank for item in scenario.ranking}
    changes = []
    for service in [*before, *(name for name in after if name not in before)]:
        first, second = before.get(service), after.get(service)
        moved = None if first is None or second is None else first - second
        changes.append(
            RankChange(service=service, original_rank=first, scenario_rank=second, change=moved)
        )
    return changes


def _build(name: str, params: dict[str, Any]) -> Perturbation[Any]:
    try:
        return build(name, params)
    except (PerturbationError, ValidationError) as error:
        message = f"{name} does not accept those parameters"
        raise ApiFailure(422, "invalid_parameters", message) from error


def _diagnose(
    case: CaseInput, step: Perturbation[Any], seed: int, model: LoadedModel
) -> DiagnoseResult:
    try:
        return diagnose(perturbed_case(case, None, [step], seed), model)
    except PerturbationError as error:
        raise ApiFailure(422, "invalid_parameters", str(error)) from error


def _curve(
    request: WhatIfRequest,
    case: CaseInput,
    model: LoadedModel,
    root: Path,
    original: DiagnoseResult,
) -> tuple[str | None, list[CurvePoint]]:
    parameter = INTENSITY.get(request.perturbation)
    if parameter is None:
        return None, []
    config = load_config(root / CONFIG_PATH, PerturbationsConfig)
    spec = next(item for item in config.perturbations if item.id == request.perturbation)
    points = [
        CurvePoint(
            intensity=0.0,
            confidence=original.confidence,
            top_service=original.ranking[0].service,
            decision=decision_from(original),
        )
    ]
    for intensity in spec.params[parameter]:
        params = {**request.params, parameter: intensity}
        result = _diagnose(case, _build(request.perturbation, params), request.seed, model)
        if result.ranking:
            points.append(
                CurvePoint(
                    intensity=float(intensity),
                    confidence=result.confidence,
                    top_service=result.ranking[0].service,
                    decision=decision_from(result),
                )
            )
    return parameter, sorted(points, key=lambda point: point.intensity)


def run_whatif(
    service: IncidentService,
    incident: Incident,
    record: DiagnosisRecord,
    request: WhatIfRequest,
    root: Path,
) -> WhatIfResult:
    """Run one scenario under the settings of the stored diagnosis."""
    settings = SettingsUpdate.model_validate_json(record.settings_json)
    case = service.case_of(incident, settings)
    if request.perturbation in TOPOLOGY_PERTURBATIONS and not case.topology:
        raise ApiFailure(422, "no_topology", "This incident has no call graph to change")
    model = model_with_threshold(service.model, settings.abstain_threshold)
    step = _build(request.perturbation, request.params)
    original = DiagnoseResult.model_validate_json(record.result_json)
    changed = _diagnose(case, step, request.seed, model)
    parameter, curve = _curve(request, case, model, root, original)
    before, after = _scenario(original), _scenario(changed)
    return WhatIfResult(
        perturbation=request.perturbation,
        name=PERTURBATIONS[request.perturbation].name,
        realism=PERTURBATIONS[request.perturbation].realism,
        parameters=step.parameters(),
        seed=request.seed,
        original=before,
        scenario=after,
        rank_changes=rank_changes(before, after),
        intensity_parameter=parameter,
        curve=curve,
        abstain_threshold=record.abstain_threshold,
    )


class WhatIfRunner:
    """Runs scenarios off the event loop, with a timeout, a busy limit and a rate limit."""

    def __init__(self, timeout_seconds: float, per_minute: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.per_minute = per_minute
        self._pool = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="whatif")
        self._slots = threading.BoundedSemaphore(WORKERS)
        self._calls: defaultdict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _check_rate(self, client: str) -> None:
        moment = time.monotonic()
        with self._lock:
            calls = self._calls[client]
            while calls and moment - calls[0] > WINDOW_SECONDS:
                calls.popleft()
            if len(calls) >= self.per_minute:
                message = "Too many what-if runs. Wait a minute and try again."
                raise ApiFailure(429, "whatif_rate_limited", message)
            calls.append(moment)

    async def run(self, client: str, work: Callable[[], T]) -> T:
        """Return ``work()``, or fail with 429 or 503."""
        self._check_rate(client)
        if not self._slots.acquire(blocking=False):
            message = "What-if is busy. Try again shortly."
            raise ApiFailure(503, "whatif_busy", message)
        future = self._pool.submit(work)
        future.add_done_callback(lambda _: self._slots.release())
        try:
            return await asyncio.wait_for(asyncio.wrap_future(future), self.timeout_seconds)
        except TimeoutError as error:
            message = f"The scenario took longer than {self.timeout_seconds:g} s"
            raise ApiFailure(503, "whatif_timeout", message) from error
