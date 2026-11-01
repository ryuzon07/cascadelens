"""Request and response bodies of ``/api/v1``. No model here has a ground-truth field."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from cascadelens.config import StrictModel
from cascadelens.inference import CasePayload, DiagnoseResult, FeatureContribution
from cascadelens.models.diagnose import DISCLAIMER

Severity = Literal["critical", "high", "medium", "low"]
Status = Literal["open", "needs_feedback", "closed"]
Decision = Literal["confident", "abstained", "uncalibrated"]
TopologySource = Literal["provided", "inferred", "none"]
NodeState = Literal["top_suspect", "suspect", "anomalous", "healthy", "no_telemetry"]
Outcome = Literal["top_1", "top_3", "missed", "unknown"]
WhatIfPerturbation = Literal["P1", "P3", "P4", "P5", "P6", "P7", "P8", "P11", "P12", "P13"]


class ErrorDetail(BaseModel):
    """A failure a client can act on."""

    code: str
    message: str


class ErrorResponse(BaseModel):
    """The body of every 4xx and 5xx under ``/api/v1``."""

    error: ErrorDetail
    disclaimer: str = DISCLAIMER


class IncidentCreate(CasePayload):
    """A case, the same body as ``POST /diagnose``, with a title and a severity."""

    title: str = Field(min_length=1, max_length=200)
    severity: Severity


class IncidentSummary(BaseModel):
    """One row of the incident list."""

    id: str
    title: str
    severity: Severity
    status: Status
    owner: str | None
    is_demo: bool
    window_start: int
    window_end: int
    created_at: int
    top_service: str
    top_score: float
    confidence: float
    decision: Decision
    disclaimer: str = DISCLAIMER


class IncidentPage(BaseModel):
    """A page of incidents. ``next_cursor`` is null on the last page."""

    items: list[IncidentSummary]
    next_cursor: str | None
    disclaimer: str = DISCLAIMER


class IncidentDetail(IncidentSummary):
    """An incident with its latest diagnosis."""

    service_count: int
    sample_count: int
    topology_source: TopologySource
    diagnosed_at: int
    abstain_threshold: float | None
    diagnosis: DiagnoseResult


class GraphNode(BaseModel):
    """One service. ``state`` comes from the diagnosis, never from the service's name."""

    service: str
    state: NodeState
    score: float | None
    has_telemetry: bool


class GraphEdge(BaseModel):
    """One call. ``on_anomalous_path`` means anomalous callers lead along it to the top suspect."""

    caller: str
    callee: str
    inferred: bool
    on_anomalous_path: bool


class IncidentGraph(BaseModel):
    """The dependency graph of an incident."""

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    topology_source: TopologySource
    disclaimer: str = DISCLAIMER


class ServiceTimeline(BaseModel):
    """When a service turned anomalous and when its telemetry stopped, in Unix seconds."""

    service: str
    onset: int | None
    anomalous_until: int | None
    telemetry_stopped: int | None
    has_telemetry: bool


class IncidentTimeline(BaseModel):
    """Every service's timeline. Before ``baseline_end`` is the pre-incident reference."""

    window_start: int
    window_end: int
    baseline_end: int
    services: list[ServiceTimeline]
    disclaimer: str = DISCLAIMER


class CoverageCell(BaseModel):
    """The share of expected samples present for one service and metric family."""

    service: str
    family: str
    coverage: float = Field(ge=0, le=1)
    samples: int


class IncidentCoverage(BaseModel):
    """Telemetry coverage of an incident."""

    services: list[str]
    families: list[str]
    cells: list[CoverageCell]
    overall: float = Field(ge=0, le=1)
    disclaimer: str = DISCLAIMER


class GroupValue(BaseModel):
    """The summed contribution of one feature group."""

    group: str
    contribution: float


class MetricPoint(BaseModel):
    """One plotted sample. A null value means that no sample is present."""

    ts: int
    value: float | None


class MetricSeries(BaseModel):
    """One metric of a service, downsampled for plotting."""

    metric: str
    family: str
    points: list[MetricPoint]


class ServiceEvidence(BaseModel):
    """Why a service ranks where it does. Contributions explain the score, not the cause."""

    service: str
    rank: int | None
    score: float | None
    groups: list[GroupValue]
    features: list[FeatureContribution]
    series: list[MetricSeries]
    disclaimer: str = DISCLAIMER


class WhatIfRequest(StrictModel):
    """A simulated change to a copy of the case.

    P2 is absent because it needs a label, which incidents do not hold. P9 and P10 are held-out
    splits, not transforms.
    """

    perturbation: WhatIfPerturbation
    params: dict[str, float | int | str] = Field(default_factory=dict)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)


class ScenarioService(BaseModel):
    """One service's place in a ranking."""

    service: str
    rank: int
    score: float


class Scenario(BaseModel):
    """A ranking and its decision."""

    ranking: list[ScenarioService]
    confidence: float
    decision: Decision


class RankChange(BaseModel):
    """How a service moved. A null rank means it was not ranked, and up is positive."""

    service: str
    original_rank: int | None
    scenario_rank: int | None
    change: int | None


class CurvePoint(BaseModel):
    """One intensity of the perturbation. Zero is the original case."""

    intensity: float
    confidence: float
    top_service: str
    decision: Decision


class WhatIfResult(BaseModel):
    """Original and simulated rankings. Nothing here is stored."""

    perturbation: str
    name: str
    realism: str
    parameters: dict[str, Any]
    seed: int
    original: Scenario
    scenario: Scenario
    rank_changes: list[RankChange]
    intensity_parameter: str | None
    curve: list[CurvePoint]
    abstain_threshold: float | None
    disclaimer: str = DISCLAIMER


class FeedbackCreate(StrictModel):
    """A person's verdict on the latest diagnosis."""

    correct: bool
    actual_service: str | None = Field(default=None, min_length=1, max_length=200)
    note: str = Field(default="", max_length=2000)


class FeedbackLabel(BaseModel):
    """A stored verdict. ``actual_service`` is what a person confirmed, not benchmark truth."""

    incident_id: str
    title: str
    correct: bool
    predicted_service: str
    actual_service: str | None
    outcome: Outcome
    confidence: float
    decision: Decision
    note: str
    labelled_by: str
    created_at: int
    is_demo: bool
    disclaimer: str = DISCLAIMER


class WeekOutcomes(BaseModel):
    """How many labels of each outcome fell in the week starting on this Monday."""

    week_start: int
    top_1: int
    top_3: int
    missed: int
    unknown: int


class MatchRate(BaseModel):
    """A match rate over labelled incidents, with a group-bootstrap interval."""

    estimate: float
    lower: float
    upper: float
    labels: int


class HistoryReport(BaseModel):
    """Labelled incidents. A null rate means no label supports it."""

    items: list[FeedbackLabel]
    weeks: list[WeekOutcomes]
    top_1: MatchRate | None
    top_3: MatchRate | None
    includes_demo: bool
    disclaimer: str = DISCLAIMER


class SettingsUpdate(StrictModel):
    """Decision settings. They affect new and re-run diagnoses, never stored ones.

    A null threshold uses the saved model's. A null window length uses each case's whole window.
    """

    abstain_threshold: float | None = Field(default=None, ge=0, le=1)
    window_seconds: int | None = Field(default=None, ge=60, le=3600)
    use_provided_topology: bool = True


class SettingsView(SettingsUpdate):
    """The settings and the threshold a new diagnosis would use."""

    model_threshold: float | None
    effective_threshold: float | None


class Integration(BaseModel):
    """A configuration record. Nothing connects to this system."""

    name: str
    kind: str
    status: Literal["configuration only"] = "configuration only"
    configured: bool = False


class IntegrationList(BaseModel):
    """Every configuration record."""

    items: list[Integration]
    read_only: str


class ResultValue(BaseModel):
    """A committed result. Null values mean that it is missing, and the source says where from."""

    name: str
    estimate: float | None
    lower: float | None
    upper: float | None
    source: str


class CheckValue(BaseModel):
    """A pass or fail from a committed result. Null means the result is missing."""

    name: str
    passed: bool | None
    source: str


class ModelReport(BaseModel):
    """The saved model's provenance and the committed results behind the Model screen."""

    model_version: str
    method: str
    split: str
    seed: int
    fold: int
    calibrated: bool
    abstain_threshold: float | None
    training_data_checksum: str
    feature_config_hash: str
    config_hash: str
    frozen_config_hash: str | None
    result_status: str | None
    results: list[ResultValue]
    checks: list[CheckValue]
    disclaimer: str = DISCLAIMER
