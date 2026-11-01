import json
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from cascadelens.api.main import ApiSettings, create_app
from cascadelens.api.schemas import ErrorResponse
from cascadelens.api.views import MAX_POINTS, TOP_FEATURES, _thinned
from cascadelens.api.whatif import WhatIfRunner, rank_changes
from cascadelens.inference import feature_label
from cascadelens.models.diagnose import DISCLAIMER

REPO = Path(__file__).parents[2]
SAMPLES = REPO / "demo_data"
GROUND_TRUTH = {"root_cause_service", "inject_time", "simulation"}
API = "/api/v1"
FIRST = f"{API}/incidents/INC-0001"


def create_body(name: str = "payment-cpu") -> dict[str, Any]:
    raw: dict[str, Any] = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    raw.pop("simulation", None)
    return {**raw, "title": "Slow checkout", "severity": "high"}


def assert_safe(value: object) -> None:
    if isinstance(value, dict):
        assert not GROUND_TRUTH & set(value)
        for item in value.values():
            assert_safe(item)
    elif isinstance(value, list):
        for item in value:
            assert_safe(item)
    elif isinstance(value, str):
        assert not any(word in value for word in GROUND_TRUTH)


def assert_failure(response: Any, status: int, code: str) -> None:  # noqa: ANN401
    assert response.status_code == status
    ErrorResponse.model_validate(response.json())
    assert response.json()["error"]["code"] == code
    assert response.json()["disclaimer"] == DISCLAIMER


@pytest.fixture
def client(trained_model: Path) -> Iterator[TestClient]:
    settings = ApiSettings(model_dir=trained_model, root=REPO, whatif_per_minute=1000)
    with TestClient(create_app(settings)) as test_client:
        assert test_client.post(f"{API}/incidents", json=create_body()).status_code == 201
        yield test_client


def top_of(client: TestClient) -> str:
    found: str = client.get(FIRST).json()["top_service"]
    return found


def test_evidence_explains_a_service(client: TestClient) -> None:
    top = top_of(client)
    evidence = client.get(f"{FIRST}/evidence/{top}").json()
    assert evidence["rank"] == 1
    assert evidence["disclaimer"] == DISCLAIMER
    assert evidence["groups"]
    assert 0 < len(evidence["features"]) <= TOP_FEATURES
    sizes = [abs(item["contribution"]) for item in evidence["features"]]
    assert sizes == sorted(sizes, reverse=True)
    assert all(item["label"] for item in evidence["features"])
    assert {item["metric"] for item in evidence["series"]} == {"cpu", "error", "latency", "mem"}
    assert_safe(evidence)


def test_evidence_groups_agree_with_the_diagnosis(client: TestClient) -> None:
    detail = client.get(FIRST).json()
    top = detail["diagnosis"]["ranking"][0]
    evidence = client.get(f"{FIRST}/evidence/{top['service']}").json()
    groups = {item["group"]: item["contribution"] for item in evidence["groups"]}
    for item in top["evidence"]["top_groups"]:
        assert groups[item["group"]] == pytest.approx(item["contribution"])


def test_evidence_of_an_unknown_service_is_404(client: TestClient) -> None:
    assert_failure(client.get(f"{FIRST}/evidence/nothing"), 404, "service_not_found")
    assert_failure(client.get(f"{API}/incidents/INC-9/evidence/x"), 404, "incident_not_found")


def test_a_silent_service_shows_its_gap(trained_model: Path) -> None:
    settings = ApiSettings(model_dir=trained_model, root=REPO)
    with TestClient(create_app(settings)) as silent:
        silent.post(f"{API}/incidents", json=create_body("payment-silent"))
        evidence = silent.get(f"{FIRST}/evidence/payment").json()
    values = [point["value"] for point in evidence["series"][0]["points"]]
    assert None in values
    assert values[0] is not None


def test_long_series_are_thinned_and_keep_gaps() -> None:
    import polars as pl

    size = MAX_POINTS * 5
    rows = pl.DataFrame({"timestamp": list(range(size)), "value": [None] * size})
    rows = rows.with_columns(
        pl.when(pl.col("timestamp") < MAX_POINTS * 2).then(1.0).otherwise(None).alias("value")
    )
    thinned = _thinned(rows)
    assert thinned.height == MAX_POINTS
    assert thinned["value"][0] == 1.0
    assert thinned["value"][-1] is None


def test_feature_labels_are_words() -> None:
    assert feature_label("temporal_onset_rank") == "Onset rank"


def test_whatif_returns_both_rankings_and_stores_nothing(client: TestClient) -> None:
    before = client.get(FIRST).json()
    response = client.post(
        f"{FIRST}/whatif", json={"perturbation": "P1", "params": {"rate": 0.5}, "seed": 3}
    )
    assert response.status_code == 200
    result = response.json()
    assert result["realism"] == "simulated"
    assert result["disclaimer"] == DISCLAIMER
    assert result["parameters"]["rate"] == 0.5
    assert result["original"]["confidence"] == before["confidence"]
    assert result["original"]["ranking"][0]["service"] == before["top_service"]
    assert result["rank_changes"]
    assert_safe(result)
    assert client.get(FIRST).json() == before
    assert len(client.get(f"{API}/incidents").json()["items"]) == 1


def test_whatif_is_repeatable(client: TestClient) -> None:
    body = {"perturbation": "P1", "params": {"rate": 0.5}, "seed": 3}
    first = client.post(f"{FIRST}/whatif", json=body).json()
    assert client.post(f"{FIRST}/whatif", json=body).json() == first
    other = client.post(f"{FIRST}/whatif", json={**body, "seed": 4}).json()
    assert other["seed"] == 4


def test_whatif_curve_starts_at_the_original(client: TestClient) -> None:
    result = client.post(
        f"{FIRST}/whatif", json={"perturbation": "P1", "params": {"rate": 0.3}}
    ).json()
    assert result["intensity_parameter"] == "rate"
    intensities = [point["intensity"] for point in result["curve"]]
    assert intensities == [0.0, 0.1, 0.2, 0.3, 0.5]
    assert result["curve"][0]["confidence"] == result["original"]["confidence"]


@pytest.mark.parametrize(
    ("perturbation", "params"),
    [
        ("P3", {"family": "latency"}),
        ("P4", {"offset_seconds": 60}),
        ("P5", {"max_jitter_seconds": 30}),
        ("P6", {"rate": 0.5}),
        ("P7", {"rate": 0.2}),
        ("P8", {"method": "correlation"}),
        ("P11", {"factor": 2}),
        ("P12", {}),
        ("P13", {"mode": "rescale"}),
    ],
)
def test_whatif_runs_every_offered_perturbation(
    client: TestClient, perturbation: str, params: dict[str, Any]
) -> None:
    response = client.post(f"{FIRST}/whatif", json={"perturbation": perturbation, "params": params})
    assert response.status_code == 200, response.text
    assert_safe(response.json())


@pytest.mark.parametrize("perturbation", ["P2", "P9", "P10", "P99"])
def test_whatif_refuses_what_it_cannot_run(client: TestClient, perturbation: str) -> None:
    response = client.post(
        f"{FIRST}/whatif", json={"perturbation": perturbation, "params": {"rate": 0.3}}
    )
    assert_failure(response, 422, "validation_error")


@pytest.mark.parametrize(
    "params", [{"rate": 2.0}, {"rate": 0}, {}, {"rate": 0.1, "surprise": 1}, {"level": "x"}]
)
def test_whatif_rejects_bad_parameters(client: TestClient, params: dict[str, Any]) -> None:
    response = client.post(f"{FIRST}/whatif", json={"perturbation": "P1", "params": params})
    assert_failure(response, 422, "invalid_parameters")


def test_whatif_needs_a_graph_to_change_edges(trained_model: Path) -> None:
    settings = ApiSettings(model_dir=trained_model, root=REPO)
    with TestClient(create_app(settings)) as bare:
        body = create_body()
        body.pop("topology")
        bare.post(f"{API}/incidents", json=body)
        settings_body = {"use_provided_topology": True}
        bare.put(f"{API}/settings", json=settings_body)
        response = bare.post(
            f"{FIRST}/whatif", json={"perturbation": "P6", "params": {"rate": 0.3}}
        )
    assert_failure(response, 422, "no_topology")


def test_whatif_of_an_unknown_incident_is_404(client: TestClient) -> None:
    response = client.post(f"{API}/incidents/INC-9/whatif", json={"perturbation": "P12"})
    assert_failure(response, 404, "incident_not_found")


def test_whatif_is_rate_limited(trained_model: Path) -> None:
    settings = ApiSettings(model_dir=trained_model, root=REPO, whatif_per_minute=1)
    with TestClient(create_app(settings)) as limited:
        limited.post(f"{API}/incidents", json=create_body())
        assert limited.post(f"{FIRST}/whatif", json={"perturbation": "P12"}).status_code == 200
        second = limited.post(f"{FIRST}/whatif", json={"perturbation": "P12"})
    assert_failure(second, 429, "whatif_rate_limited")


@pytest.mark.anyio
async def test_runner_times_out_and_stays_busy_until_the_work_ends() -> None:
    from cascadelens.api.failures import ApiFailure

    release = threading.Event()
    runner = WhatIfRunner(timeout_seconds=0.05, per_minute=100)
    with pytest.raises(ApiFailure) as first:
        await runner.run("a", release.wait)
    assert first.value.code == "whatif_timeout"
    with pytest.raises(ApiFailure):
        await runner.run("a", release.wait)
    with pytest.raises(ApiFailure) as busy:
        await runner.run("a", release.wait)
    assert busy.value.code == "whatif_busy"
    assert busy.value.status == 503
    release.set()
    for _ in range(100):
        if runner._slots.acquire(blocking=False):
            break
        time.sleep(0.01)
    else:
        pytest.fail("The slot was not released")
    runner._slots.release()
    assert await runner.run("a", lambda: 7) == 7


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def test_rank_changes_handle_new_and_lost_services() -> None:
    from cascadelens.api.schemas import Scenario, ScenarioService

    def scenario(*names: str) -> Scenario:
        ranking = [
            ScenarioService(service=name, rank=place + 1, score=0.0)
            for place, name in enumerate(names)
        ]
        return Scenario(ranking=ranking, confidence=0.5, decision="confident")

    changes = {item.service: item for item in rank_changes(scenario("a", "b"), scenario("b", "c"))}
    assert changes["b"].change == 1
    assert changes["a"].scenario_rank is None
    assert changes["a"].change is None
    assert changes["c"].original_rank is None


def test_correct_feedback_closes_the_incident_and_reaches_history(client: TestClient) -> None:
    top = top_of(client)
    response = client.post(f"{FIRST}/feedback", json={"correct": True, "note": "Confirmed."})
    assert response.status_code == 201
    saved = response.json()
    assert saved["actual_service"] == top
    assert saved["outcome"] == "top_1"
    assert saved["disclaimer"] == DISCLAIMER
    assert client.get(FIRST).json()["status"] == "closed"
    report = client.get(f"{API}/history").json()
    assert [item["incident_id"] for item in report["items"]] == ["INC-0001"]
    assert report["items"][0]["note"] == "Confirmed."
    assert report["top_1"]["estimate"] == 1.0
    assert report["top_1"]["labels"] == 1
    assert report["includes_demo"] is False
    assert sum(week["top_1"] for week in report["weeks"]) == 1
    assert_safe(report)


def test_feedback_grades_the_named_service(client: TestClient) -> None:
    ranking = client.get(FIRST).json()["diagnosis"]["ranking"]
    if len(ranking) >= 2:
        saved = client.post(
            f"{FIRST}/feedback", json={"correct": False, "actual_service": ranking[1]["service"]}
        ).json()
        assert saved["outcome"] == "top_3"
    report = client.get(f"{API}/history").json()
    assert report["top_1"]["estimate"] == 0.0
    assert report["top_3"]["estimate"] == 1.0


def test_unknown_cause_counts_against_top_1_only(client: TestClient) -> None:
    saved = client.post(f"{FIRST}/feedback", json={"correct": False}).json()
    assert saved["outcome"] == "unknown"
    assert saved["actual_service"] is None
    report = client.get(f"{API}/history").json()
    assert report["top_1"]["estimate"] == 0.0
    assert report["top_3"] is None


def test_newer_feedback_supersedes_older(client: TestClient) -> None:
    client.post(f"{FIRST}/feedback", json={"correct": False})
    client.post(f"{FIRST}/feedback", json={"correct": True})
    report = client.get(f"{API}/history").json()
    assert len(report["items"]) == 1
    assert report["items"][0]["correct"] is True


def test_feedback_survives_a_rerun_with_its_snapshot(client: TestClient) -> None:
    client.post(f"{FIRST}/feedback", json={"correct": True})
    label = client.get(f"{API}/history").json()["items"][0]
    assert client.post(f"{FIRST}/rerun").status_code == 200
    assert client.get(f"{API}/history").json()["items"][0] == label


def test_invalid_feedback_changes_nothing(client: TestClient) -> None:
    top = top_of(client)
    before = client.get(FIRST).json()
    assert_failure(
        client.post(f"{FIRST}/feedback", json={"correct": True, "actual_service": "cart" + "x"}),
        422,
        "unknown_service",
    )
    assert_failure(
        client.post(f"{FIRST}/feedback", json={"correct": False, "actual_service": top}),
        422,
        "conflicting_feedback",
    )
    other = next(
        item["service"] for item in before["diagnosis"]["ranking"] if item["service"] != top
    )
    assert_failure(
        client.post(f"{FIRST}/feedback", json={"correct": True, "actual_service": other}),
        422,
        "conflicting_feedback",
    )
    for bad in ({}, {"correct": True, "note": "x" * 3000}, {"correct": True, "extra": 1}):
        assert_failure(client.post(f"{FIRST}/feedback", json=bad), 422, "validation_error")
    assert client.get(FIRST).json() == before
    assert client.get(f"{API}/history").json()["items"] == []


def test_feedback_cannot_smuggle_ground_truth(client: TestClient) -> None:
    response = client.post(
        f"{FIRST}/feedback", json={"correct": True, "root_cause_service": "payment"}
    )
    assert_failure(response, 422, "validation_error")
    assert "root_cause_service" not in response.text


def test_empty_history_has_no_rates(client: TestClient) -> None:
    report = client.get(f"{API}/history").json()
    assert report["items"] == []
    assert report["weeks"] == []
    assert report["top_1"] is None
    assert report["top_3"] is None


def test_history_since_filters(client: TestClient) -> None:
    client.post(f"{FIRST}/feedback", json={"correct": True})
    future = int(time.time()) + 3600
    assert client.get(f"{API}/history", params={"since": future}).json()["items"] == []
    assert client.get(f"{API}/history", params={"since": 0}).json()["items"]


def test_analysis_opens_no_socket(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("A socket connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    top = top_of(client)
    assert client.get(f"{FIRST}/evidence/{top}").status_code == 200
    body = {"perturbation": "P1", "params": {"rate": 0.3}}
    assert client.post(f"{FIRST}/whatif", json=body).status_code == 200
    assert client.post(f"{FIRST}/feedback", json={"correct": True}).status_code == 201
    assert client.get(f"{API}/history").status_code == 200
