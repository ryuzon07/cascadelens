"""The ``/api/v1`` routes behind the web UI. Read-only analysis, with no outbound request."""

from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.routing import APIRoute

from cascadelens.api.failures import ApiFailure
from cascadelens.api.feedback import history, label, new_feedback
from cascadelens.api.incidents import IncidentService, decision_of, detail, summary
from cascadelens.api.model_report import (
    CALIBRATED_METHOD,
    check_values,
    frozen_config_hash,
    reliability_bins,
    result_values,
    risk_coverage,
    robustness_points,
)
from cascadelens.api.schemas import (
    ErrorResponse,
    FeedbackCreate,
    FeedbackLabel,
    HistoryReport,
    IncidentCoverage,
    IncidentCreate,
    IncidentDetail,
    IncidentGraph,
    IncidentPage,
    IncidentTimeline,
    Integration,
    IntegrationList,
    ModelReport,
    ServiceEvidence,
    SettingsUpdate,
    SettingsView,
    Severity,
    Status,
    WhatIfRequest,
    WhatIfResult,
)
from cascadelens.api.store import DiagnosisRecord, Incident, StoredSettings, parse_incident_id
from cascadelens.api.views import build_coverage, build_evidence, build_graph, build_timeline
from cascadelens.api.whatif import WhatIfRunner, run_whatif
from cascadelens.features.context import CaseContext
from cascadelens.inference import (
    DiagnoseResult,
    LoadedModel,
    feature_contributions,
    model_details,
)

READ_ONLY = "CascadeLens is read-only: these records are configuration only and connect to nothing."
INTEGRATIONS = (
    ("Prometheus", "metrics"),
    ("Alertmanager", "alerts"),
    ("PagerDuty", "paging"),
    ("Slack", "notifications"),
)
ERRORS: dict[int | str, dict[str, type[ErrorResponse]]] = {
    code: {"model": ErrorResponse} for code in (404, 413, 422, 429, 500, 503)
}
ReadBody = Callable[[Request], Awaitable[bytes]]


def limited_route(read_body: ReadBody) -> type[APIRoute]:
    """Return a route class that stops reading a body as soon as it passes the size limit."""

    class LimitedRoute(APIRoute):
        def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
            handle = super().get_route_handler()

            async def limited(request: Request) -> Response:
                request._body = await read_body(request)
                return await handle(request)

            return limited

    return LimitedRoute


def _found(service: IncidentService, text: str) -> Incident:
    seq = parse_incident_id(text)
    incident = None if seq is None else service.store.incident(seq)
    if incident is None:
        raise ApiFailure(404, "incident_not_found", f"No incident {text[:40]}")
    return incident


def _settings_view(saved: SettingsUpdate, model: LoadedModel) -> SettingsView:
    threshold = model.calibration.tau if model.calibration else None
    chosen = saved.abstain_threshold if saved.abstain_threshold is not None else threshold
    return SettingsView(
        **saved.model_dump(),
        model_threshold=threshold,
        effective_threshold=chosen if model.calibration else None,
    )


def _diagnosis_routes(
    router: APIRouter, service: IncidentService, root: Path, runner: WhatIfRunner
) -> None:
    def latest_of(text: str) -> tuple[Incident, DiagnosisRecord]:
        incident = _found(service, text)
        assert incident.seq is not None
        return incident, service.store.latest_diagnosis(incident.seq)

    @router.get("/incidents/{incident_id}/evidence/{service_name}")
    def incident_evidence(incident_id: str, service_name: str) -> ServiceEvidence:
        """Return one service's feature contributions and metric series."""
        incident, record = latest_of(incident_id)
        context = service.context(incident, record)
        if service_name not in context.services:
            raise ApiFailure(404, "service_not_found", "That service is not part of this incident")
        settings = SettingsUpdate.model_validate_json(record.settings_json)
        case = service.case_of(incident, settings)
        result = DiagnoseResult.model_validate_json(record.result_json)
        features = feature_contributions(case, service.model, service_name)
        return build_evidence(context, result, service_name, features)

    @router.post("/incidents/{incident_id}/whatif")
    async def incident_whatif(
        incident_id: str, body: WhatIfRequest, request: Request
    ) -> WhatIfResult:
        """Simulate a change to a copy of the case. Nothing is stored."""
        incident, record = latest_of(incident_id)
        client = request.client.host if request.client else "unknown"
        return await runner.run(client, lambda: run_whatif(service, incident, record, body, root))

    @router.post("/incidents/{incident_id}/feedback", status_code=201)
    def incident_feedback(incident_id: str, body: FeedbackCreate) -> FeedbackLabel:
        """Store a verdict on the latest diagnosis and close the incident."""
        incident, record = latest_of(incident_id)
        result = DiagnoseResult.model_validate_json(record.result_json)
        services = service.context(incident, record).services
        verdict = new_feedback(body, incident, result, decision_of(incident), services)
        return label(service.store.add_feedback(incident, verdict), incident)

    @router.get("/history")
    def get_history(since: int | None = None) -> HistoryReport:
        """Return labelled incidents, weekly outcomes and match rates with intervals."""
        return history(service.store.latest_feedback(since))


def _incident_routes(router: APIRouter, service: IncidentService) -> None:
    def latest(incident: Incident) -> IncidentDetail:
        assert incident.seq is not None
        return detail(incident, service.store.latest_diagnosis(incident.seq))

    def stored(text: str) -> tuple[Incident, DiagnoseResult]:
        incident = _found(service, text)
        assert incident.seq is not None
        record = service.store.latest_diagnosis(incident.seq)
        return incident, DiagnoseResult.model_validate_json(record.result_json)

    def context_of(incident: Incident) -> CaseContext:
        assert incident.seq is not None
        return service.context(incident, service.store.latest_diagnosis(incident.seq))

    @router.get("/incidents")
    def list_incidents(
        status: Status | None = None,
        severity: Severity | None = None,
        incident_service: Annotated[str | None, Query(alias="service", max_length=200)] = None,
        since: int | None = None,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        cursor: Annotated[str | None, Query(pattern=r"^\d{1,18}$")] = None,
    ) -> IncidentPage:
        """List incidents, newest first, with the top suspect and the decision."""
        rows = service.store.incidents(
            status,
            severity,
            incident_service,
            since,
            int(cursor) if cursor else None,
            limit + 1,
        )
        page = rows[:limit]
        more = len(rows) > limit
        return IncidentPage(
            items=[summary(row) for row in page],
            next_cursor=str(page[-1].seq) if more else None,
        )

    @router.post("/incidents", status_code=201)
    def create_incident(body: IncidentCreate) -> IncidentDetail:
        """Diagnose a case and store it as a new incident."""
        return latest(service.create(body))

    @router.get("/incidents/{incident_id}")
    def get_incident(incident_id: str) -> IncidentDetail:
        """Return an incident with its latest diagnosis."""
        return latest(_found(service, incident_id))

    @router.post("/incidents/{incident_id}/rerun")
    def rerun_incident(incident_id: str) -> IncidentDetail:
        """Diagnose an incident again with the current model and settings."""
        return latest(service.rerun(_found(service, incident_id)))

    @router.get("/incidents/{incident_id}/graph")
    def incident_graph(incident_id: str) -> IncidentGraph:
        """Return the dependency graph, with node states from the latest diagnosis."""
        incident, result = stored(incident_id)
        return build_graph(context_of(incident), result)

    @router.get("/incidents/{incident_id}/timeline")
    def incident_timeline(incident_id: str) -> IncidentTimeline:
        """Return each service's onset, anomalous period and any telemetry stop."""
        return build_timeline(context_of(_found(service, incident_id)))

    @router.get("/incidents/{incident_id}/coverage")
    def incident_coverage(incident_id: str) -> IncidentCoverage:
        """Return telemetry coverage per service and metric family."""
        return build_coverage(context_of(_found(service, incident_id)))


def build_router(
    service: IncidentService, root: Path, read_body: ReadBody, runner: WhatIfRunner
) -> APIRouter:
    """Return every ``/api/v1`` route."""
    router = APIRouter(prefix="/api/v1", route_class=limited_route(read_body), responses=ERRORS)
    _incident_routes(router, service)
    _diagnosis_routes(router, service, root, runner)
    model = service.model

    @router.get("/settings")
    def get_settings() -> SettingsView:
        """Return the decision settings."""
        return _settings_view(service.current_settings(), model)

    @router.put("/settings")
    def put_settings(body: SettingsUpdate) -> SettingsView:
        """Change the decision settings. Stored diagnoses keep the settings they were made with."""
        service.store.save_settings(StoredSettings(**body.model_dump()))
        return _settings_view(service.current_settings(), model)

    @router.get("/integrations")
    def integrations() -> IntegrationList:
        """Return configuration records. Nothing connects to these systems."""
        items = [Integration(name=name, kind=kind) for name, kind in INTEGRATIONS]
        return IntegrationList(items=items, read_only=READ_ONLY)

    @router.get("/model")
    def model_report() -> ModelReport:
        """Return model provenance and results from committed result JSONs."""
        details = model_details(model)
        status, results = result_values(root)
        return ModelReport(
            **details,
            abstain_threshold=model.calibration.tau if model.calibration else None,
            frozen_config_hash=frozen_config_hash(details["config_hash"], root),
            result_status=status,
            results=results,
            checks=check_values(root),
            calibrated_method=CALIBRATED_METHOD,
            reliability=reliability_bins(root),
            risk_coverage=risk_coverage(root),
            robustness=robustness_points(root),
        )

    return router
