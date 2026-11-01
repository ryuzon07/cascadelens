"""The ``/api/v1`` routes behind the web UI. Read-only analysis, with no outbound request."""

from collections.abc import Awaitable, Callable, Coroutine
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Query, Request, Response
from fastapi.routing import APIRoute

from cascadelens.api.incidents import IncidentService, detail, summary
from cascadelens.api.model_report import check_values, frozen_config_hash, result_values
from cascadelens.api.schemas import (
    ErrorResponse,
    IncidentCoverage,
    IncidentCreate,
    IncidentDetail,
    IncidentGraph,
    IncidentPage,
    IncidentTimeline,
    Integration,
    IntegrationList,
    ModelReport,
    SettingsUpdate,
    SettingsView,
    Severity,
    Status,
)
from cascadelens.api.store import Incident, StoredSettings, parse_incident_id
from cascadelens.api.views import build_coverage, build_graph, build_timeline
from cascadelens.features.context import CaseContext
from cascadelens.inference import DiagnoseResult, LoadedModel, model_details

READ_ONLY = "CascadeLens is read-only: these records are configuration only and connect to nothing."
INTEGRATIONS = (
    ("Prometheus", "metrics"),
    ("Alertmanager", "alerts"),
    ("PagerDuty", "paging"),
    ("Slack", "notifications"),
)
ERRORS: dict[int | str, dict[str, type[ErrorResponse]]] = {
    code: {"model": ErrorResponse} for code in (404, 413, 422, 500)
}
ReadBody = Callable[[Request], Awaitable[bytes]]


class ApiFailure(Exception):
    """A failure with a code a client can act on."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


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


def build_router(service: IncidentService, root: Path, read_body: ReadBody) -> APIRouter:
    """Return every ``/api/v1`` route."""
    router = APIRouter(prefix="/api/v1", route_class=limited_route(read_body), responses=ERRORS)
    _incident_routes(router, service)
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
        )

    return router
