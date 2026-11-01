"""Create and re-run incidents. Every diagnosis goes through ``inference.diagnose``."""

import json
from dataclasses import dataclass, replace
from typing import cast

from cascadelens.api.schemas import (
    Decision,
    IncidentCreate,
    IncidentDetail,
    IncidentSummary,
    SettingsUpdate,
    Severity,
    Status,
    TopologySource,
)
from cascadelens.api.store import DiagnosisRecord, Incident, Store, incident_id, now
from cascadelens.api.views import topology_source
from cascadelens.features.context import CaseContext
from cascadelens.inference import (
    CaseInput,
    CasePayload,
    DiagnoseResult,
    LoadedModel,
    case_context,
    case_from_payload,
    diagnose,
)
from cascadelens.models.data import RankerError


def apply_settings(case: CaseInput, settings: SettingsUpdate) -> CaseInput:
    """Return the case the model should see, keeping any shorter window around its middle."""
    changed = case if settings.use_provided_topology else replace(case, topology=None)
    length = settings.window_seconds
    if length is None or length >= case.window_end - case.window_start:
        return changed
    start = (case.window_start + case.window_end - length) // 2
    return replace(changed, window_start=start, window_end=start + length)


def model_with_threshold(model: LoadedModel, threshold: float | None) -> LoadedModel:
    """Return the model with its abstention threshold replaced, if it has a calibrator."""
    if threshold is None or model.calibration is None:
        return model
    return replace(model, calibration=replace(model.calibration, tau=threshold))


def decision_of(incident: Incident) -> Decision:
    """Return the decision. An uncalibrated confidence cannot clear a threshold."""
    if not incident.calibrated:
        return "uncalibrated"
    return "abstained" if incident.abstain else "confident"


def summary(incident: Incident) -> IncidentSummary:
    """Return an incident's list row."""
    assert incident.seq is not None
    return IncidentSummary(
        id=incident_id(incident.seq),
        title=incident.title,
        severity=cast(Severity, incident.severity),
        status=cast(Status, incident.status),
        owner=incident.owner,
        is_demo=incident.is_demo,
        window_start=incident.window_start,
        window_end=incident.window_end,
        created_at=incident.created_at,
        top_service=incident.top_service,
        top_score=incident.top_score,
        confidence=incident.confidence,
        decision=decision_of(incident),
    )


def stored_case(incident: Incident) -> CasePayload:
    """Return the case an incident was created from."""
    return CasePayload.model_validate_json(incident.case_json)


def detail(incident: Incident, record: DiagnosisRecord) -> IncidentDetail:
    """Return an incident with one of its diagnoses."""
    case = stored_case(incident)
    return IncidentDetail(
        **summary(incident).model_dump(),
        service_count=len({row.service for row in case.metrics}),
        sample_count=len(case.metrics),
        topology_source=cast(TopologySource, record.topology_source),
        diagnosed_at=record.created_at,
        abstain_threshold=record.abstain_threshold,
        diagnosis=DiagnoseResult.model_validate_json(record.result_json),
    )


@dataclass(frozen=True)
class IncidentService:
    """Runs the model for the API and the demo seeder."""

    store: Store
    model: LoadedModel
    window_tolerance_seconds: int

    def current_settings(self) -> SettingsUpdate:
        """Return the decision settings in force."""
        saved = self.store.settings()
        return SettingsUpdate(
            abstain_threshold=saved.abstain_threshold,
            window_seconds=saved.window_seconds,
            use_provided_topology=saved.use_provided_topology,
        )

    def _case(self, payload: CasePayload, settings: SettingsUpdate) -> CaseInput:
        case = case_from_payload(payload, self.window_tolerance_seconds)
        return apply_settings(case, settings)

    def _run(
        self, payload: CasePayload, settings: SettingsUpdate, created_at: int
    ) -> tuple[DiagnoseResult, DiagnosisRecord]:
        case = self._case(payload, settings)
        model = model_with_threshold(self.model, settings.abstain_threshold)
        result = diagnose(case, model)
        if not result.ranking:
            raise RankerError("No service passed retrieval, so there is nothing to rank")
        record = DiagnosisRecord(
            incident_seq=0,
            created_at=created_at,
            result_json=result.model_dump_json(),
            settings_json=settings.model_dump_json(),
            topology_source=topology_source(case_context(case, model)),
            abstain_threshold=model.calibration.tau if model.calibration else None,
        )
        return result, record

    def create(
        self,
        body: IncidentCreate,
        status: Status = "open",
        owner: str | None = None,
        is_demo: bool = False,
    ) -> Incident:
        """Diagnose a case and save it. A case the model refuses saves nothing."""
        created = now()
        payload = CasePayload.model_validate(body.model_dump(exclude={"title", "severity"}))
        result, record = self._run(payload, self.current_settings(), created)
        incident = Incident(
            title=body.title,
            severity=body.severity,
            status=status,
            owner=owner,
            is_demo=is_demo,
            window_start=payload.window.start,
            window_end=payload.window.end,
            created_at=created,
            case_json=payload.model_dump_json(),
            top_service=result.ranking[0].service,
            top_score=result.ranking[0].score,
            confidence=result.confidence,
            abstain=result.abstain,
            calibrated=result.calibrated,
        )
        return self.store.add(incident, record)

    def rerun(self, incident: Incident) -> Incident:
        """Diagnose an incident again with the current model and settings."""
        assert incident.seq is not None
        result, record = self._run(stored_case(incident), self.current_settings(), now())
        record.incident_seq = incident.seq
        incident.top_service = result.ranking[0].service
        incident.top_score = result.ranking[0].score
        incident.confidence = result.confidence
        incident.abstain = result.abstain
        incident.calibrated = result.calibrated
        return self.store.add_diagnosis(incident, record)

    def case_of(self, incident: Incident, settings: SettingsUpdate) -> CaseInput:
        """Return an incident's case as the model saw it under ``settings``."""
        return self._case(stored_case(incident), settings)

    def context(self, incident: Incident, record: DiagnosisRecord) -> CaseContext:
        """Return the context of a stored diagnosis, under the settings it was made with."""
        settings = SettingsUpdate.model_validate(json.loads(record.settings_json))
        return case_context(self.case_of(incident, settings), self.model)
