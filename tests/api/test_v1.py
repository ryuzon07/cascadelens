import json
import re
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from cascadelens.api.incidents import IncidentService, apply_settings, model_with_threshold
from cascadelens.api.main import ApiSettings, create_app
from cascadelens.api.schemas import SettingsUpdate
from cascadelens.api.seed import DEMO_INCIDENTS, seed_demo
from cascadelens.api.store import Store, incident_id, open_engine, parse_incident_id
from cascadelens.inference import DEFAULT_TOLERANCE_SECONDS, load_model, read_case
from cascadelens.models.diagnose import DISCLAIMER

REPO = Path(__file__).parents[2]
SAMPLES = REPO / "demo_data"
SNAPSHOT = REPO / "docs/openapi.json"
GROUND_TRUTH = {"root_cause_service", "inject_time", "simulation"}
API = "/api/v1"


def sample(name: str = "payment-cpu") -> dict[str, Any]:
    raw: dict[str, Any] = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    raw.pop("simulation", None)
    return raw


def create_body(name: str = "payment-cpu") -> dict[str, Any]:
    return {**sample(name), "title": "Slow checkout", "severity": "high"}


def assert_safe(value: object) -> None:
    """Fail on a ground-truth key or string anywhere in a response."""
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
    body = response.json()
    assert body["error"]["code"] == code
    assert body["error"]["message"]
    assert body["disclaimer"] == DISCLAIMER
    assert_safe(body)


@pytest.fixture
def client(trained_model: Path) -> Iterator[TestClient]:
    settings = ApiSettings(model_dir=trained_model, root=REPO, max_body_bytes=2_000_000)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def seeded(client: TestClient) -> TestClient:
    for name in ("payment-cpu", "cart-memory-no-topology", "payment-silent"):
        assert client.post(f"{API}/incidents", json=create_body(name)).status_code == 201
    return client


def incident_ids(client: TestClient) -> list[str]:
    return [item["id"] for item in client.get(f"{API}/incidents").json()["items"]]


def test_created_incident_keeps_its_diagnosis(client: TestClient) -> None:
    response = client.post(f"{API}/incidents", json=create_body())
    assert response.status_code == 201
    created = response.json()
    assert created["id"] == "INC-0001"
    assert created["status"] == "open"
    assert created["is_demo"] is False
    assert created["disclaimer"] == DISCLAIMER
    assert created["diagnosis"]["disclaimer"] == DISCLAIMER
    assert created["top_service"] == created["diagnosis"]["ranking"][0]["service"]
    assert client.get(f"{API}/incidents/INC-0001").json() == created


def test_incident_matches_the_diagnose_endpoint(client: TestClient) -> None:
    created = client.post(f"{API}/incidents", json=create_body()).json()
    direct = client.post("/diagnose", json=sample()).json()
    assert created["diagnosis"]["ranking"] == direct["ranking"]
    assert created["diagnosis"]["confidence"] == direct["confidence"]


def invalid_bodies() -> list[dict[str, Any]]:
    return [
        {**create_body(), "severity": "extreme"},
        {**create_body(), "title": ""},
        {**create_body(), "root_cause_service": "payment"},
        {**create_body(), "simulation": {"inject_time": 1}},
        {**create_body(), "metrics": []},
        sample(),
    ]


@pytest.mark.parametrize("body", invalid_bodies())
def test_invalid_create_stores_nothing(client: TestClient, body: dict[str, Any]) -> None:
    assert_failure(client.post(f"{API}/incidents", json=body), 422, "validation_error")
    assert incident_ids(client) == []


def test_a_case_the_model_refuses_stores_nothing(client: TestClient) -> None:
    repeated = create_body()
    repeated["metrics"].append(repeated["metrics"][0])
    assert_failure(client.post(f"{API}/incidents", json=repeated), 422, "invalid_case")
    assert incident_ids(client) == []


def test_oversized_body_is_refused(trained_model: Path) -> None:
    settings = ApiSettings(model_dir=trained_model, root=REPO, max_body_bytes=1000)
    with TestClient(create_app(settings)) as small:
        declared = small.post(f"{API}/incidents", json=create_body())
        streamed = small.post(f"{API}/incidents", content=(b"x" * 500 for _ in range(4)))
    assert_failure(declared, 413, "body_too_large")
    assert_failure(streamed, 413, "body_too_large")


@pytest.mark.parametrize("name", ["INC-9999", "nonsense", "INC-", "INC-١٢"])
@pytest.mark.parametrize("suffix", ["", "/graph", "/timeline", "/coverage"])
def test_unknown_incident_is_404(client: TestClient, name: str, suffix: str) -> None:
    assert_failure(client.get(f"{API}/incidents/{name}{suffix}"), 404, "incident_not_found")


def test_rerun_of_unknown_incident_is_404(client: TestClient) -> None:
    assert_failure(client.post(f"{API}/incidents/INC-9999/rerun"), 404, "incident_not_found")


def test_unknown_v1_route_uses_the_error_shape(client: TestClient) -> None:
    assert_failure(client.get(f"{API}/nothing"), 404, "not_found")


def test_list_pages_without_repeats(seeded: TestClient) -> None:
    first = seeded.get(f"{API}/incidents", params={"limit": 2}).json()
    assert [item["id"] for item in first["items"]] == ["INC-0003", "INC-0002"]
    assert first["next_cursor"] == "2"
    second = seeded.get(f"{API}/incidents", params={"limit": 2, "cursor": "2"}).json()
    assert [item["id"] for item in second["items"]] == ["INC-0001"]
    assert second["next_cursor"] is None
    assert first["disclaimer"] == DISCLAIMER


@pytest.mark.parametrize(
    "query", [{"limit": 0}, {"limit": 500}, {"cursor": "x"}, {"status": "lost"}, {"severity": "no"}]
)
def test_list_rejects_bad_queries(client: TestClient, query: dict[str, Any]) -> None:
    assert_failure(client.get(f"{API}/incidents", params=query), 422, "validation_error")


def test_list_filters(seeded: TestClient) -> None:
    def count(**query: Any) -> int:  # noqa: ANN401
        return len(seeded.get(f"{API}/incidents", params=query).json()["items"])

    top = seeded.get(f"{API}/incidents/INC-0001").json()
    assert count() == 3
    assert count(status="open") == 3
    assert count(status="closed") == 0
    assert count(severity="high") == 3
    assert count(severity="low") == 0
    assert count(service=top["top_service"]) >= 1
    assert count(service="nothing") == 0
    assert count(since=top["window_start"]) == 3
    assert count(since=top["window_end"] + 1) == 0


def test_rerun_uses_new_settings_and_keeps_the_old_record(seeded: TestClient) -> None:
    before = seeded.get(f"{API}/incidents/INC-0001").json()
    threshold = 1.0
    assert seeded.put(f"{API}/settings", json={"abstain_threshold": threshold}).status_code == 200
    untouched = seeded.get(f"{API}/incidents/INC-0001").json()
    assert untouched == before
    after = seeded.post(f"{API}/incidents/INC-0001/rerun").json()
    assert after["abstain_threshold"] == threshold
    assert after["diagnosis"]["abstain"] is (after["diagnosis"]["confidence"] < threshold)
    assert after["diagnosis"]["ranking"] == before["diagnosis"]["ranking"]


def test_settings_defaults_and_validation(client: TestClient) -> None:
    current = client.get(f"{API}/settings").json()
    assert current["abstain_threshold"] is None
    assert current["window_seconds"] is None
    assert current["use_provided_topology"] is True
    assert current["effective_threshold"] == current["model_threshold"]
    for bad in (
        {"abstain_threshold": 1.5},
        {"abstain_threshold": -0.1},
        {"window_seconds": 5},
        {"window_seconds": 99999},
        {"surprise": 1},
    ):
        assert_failure(client.put(f"{API}/settings", json=bad), 422, "validation_error")
    assert client.get(f"{API}/settings").json() == current


def test_settings_round_trip(client: TestClient) -> None:
    wanted = {"abstain_threshold": 0.4, "window_seconds": 300, "use_provided_topology": False}
    saved = client.put(f"{API}/settings", json=wanted).json()
    assert {key: saved[key] for key in wanted} == wanted
    assert saved["effective_threshold"] == 0.4
    assert client.get(f"{API}/settings").json() == saved


def test_no_topology_setting_reaches_the_graph(client: TestClient) -> None:
    client.put(f"{API}/settings", json={"use_provided_topology": False})
    created = client.post(f"{API}/incidents", json=create_body()).json()
    assert created["topology_source"] != "provided"
    graph = client.get(f"{API}/incidents/{created['id']}/graph").json()
    assert all(edge["inferred"] for edge in graph["edges"])


def test_window_setting_keeps_the_middle(trained_model: Path) -> None:
    case = read_case(SAMPLES / "payment-cpu.json")
    middle = (case.window_start + case.window_end) // 2
    short = apply_settings(case, SettingsUpdate(window_seconds=100))
    assert short.window_end - short.window_start == 100
    assert abs((short.window_start + short.window_end) // 2 - middle) <= 1
    longer = apply_settings(case, SettingsUpdate(window_seconds=3000))
    assert (longer.window_start, longer.window_end) == (case.window_start, case.window_end)


def test_threshold_override_leaves_the_loaded_model_alone(trained_model: Path) -> None:
    model = load_model(trained_model)
    assert model.calibration is not None
    tau = model.calibration.tau
    changed = model_with_threshold(model, 0.123)
    assert changed.calibration is not None
    assert changed.calibration.tau == 0.123
    assert model.calibration.tau == tau
    assert model_with_threshold(model, None) is model


def test_graph_states_come_from_the_diagnosis(seeded: TestClient) -> None:
    detail = seeded.get(f"{API}/incidents/INC-0001").json()
    graph = seeded.get(f"{API}/incidents/INC-0001/graph").json()
    states = {node["service"]: node["state"] for node in graph["nodes"]}
    assert states[detail["top_service"]] == "top_suspect"
    assert list(states.values()).count("top_suspect") == 1
    assert graph["topology_source"] == "provided"
    assert graph["disclaimer"] == DISCLAIMER
    assert {(e["caller"], e["callee"]) for e in graph["edges"]} == {
        (e["caller"], e["callee"]) for e in sample()["topology"]
    }
    assert any(edge["on_anomalous_path"] for edge in graph["edges"])
    assert not any(edge["inferred"] for edge in graph["edges"])
    assert_safe(graph)


def test_missing_topology_gives_nodes_without_edges(seeded: TestClient) -> None:
    graph = seeded.get(f"{API}/incidents/INC-0002/graph").json()
    assert graph["nodes"]
    if graph["topology_source"] == "none":
        assert graph["edges"] == []


def test_timeline_finds_the_onset_and_the_stop(seeded: TestClient) -> None:
    timeline = seeded.get(f"{API}/incidents/INC-0003/timeline").json()
    rows = {row["service"]: row for row in timeline["services"]}
    assert timeline["window_start"] < timeline["baseline_end"] < timeline["window_end"]
    assert rows["payment"]["telemetry_stopped"] is not None
    assert rows["frontend"]["telemetry_stopped"] is None
    assert timeline["disclaimer"] == DISCLAIMER
    assert_safe(timeline)
    visible = seeded.get(f"{API}/incidents/INC-0001/timeline").json()
    onsets = [row["onset"] for row in visible["services"] if row["onset"] is not None]
    assert onsets
    for onset in onsets:
        assert visible["baseline_end"] <= onset <= visible["window_end"]


def test_coverage_measures_the_gap(seeded: TestClient) -> None:
    full = seeded.get(f"{API}/incidents/INC-0001/coverage").json()
    assert full["overall"] == 1.0
    assert full["families"] == ["cpu", "error", "latency", "memory"]
    assert len(full["cells"]) == len(full["services"]) * len(full["families"])
    assert full["disclaimer"] == DISCLAIMER
    gaps = seeded.get(f"{API}/incidents/INC-0003/coverage").json()
    assert gaps["overall"] < 1.0
    cells = {(cell["service"], cell["family"]): cell["coverage"] for cell in gaps["cells"]}
    assert cells[("payment", "cpu")] == pytest.approx(0.5)
    assert cells[("frontend", "cpu")] == 1.0


def test_integrations_are_configuration_only(client: TestClient) -> None:
    response = client.get(f"{API}/integrations").json()
    assert response["items"]
    for item in response["items"]:
        assert item["status"] == "configuration only"
        assert item["configured"] is False
    assert "read-only" in response["read_only"]


def test_model_report_reads_committed_results(client: TestClient) -> None:
    report = client.get(f"{API}/model").json()
    assert report["result_status"] == "exploratory"
    assert report["frozen_config_hash"] is None
    assert report["abstain_threshold"] is not None
    assert report["disclaimer"] == DISCLAIMER
    committed = json.loads((REPO / "experiments/robustness/results.json").read_text("utf-8"))
    expected = committed["metrics"]["hypotheses"]["ranker"]["S1"]
    found = next(row for row in report["results"] if row["name"].startswith("S1"))
    assert found["estimate"] == expected["estimate"]
    assert found["source"] == "experiments/robustness/results.json"
    for row in report["results"]:
        assert row["estimate"] is not None
        assert (REPO / row["source"]).is_file()
    checks = {check["name"]: check["passed"] for check in report["checks"]}
    assert checks["Mask-only probe"] is False
    assert checks["S4"] is True
    assert_safe(report)


def test_missing_results_stay_null(trained_model: Path, tmp_path: Path) -> None:
    settings = ApiSettings(model_dir=trained_model, root=tmp_path)
    with TestClient(create_app(settings)) as empty:
        report = empty.get(f"{API}/model").json()
    assert report["result_status"] is None
    assert report["results"]
    assert all(row["estimate"] is None for row in report["results"])
    assert all(check["passed"] is None for check in report["checks"])


def test_every_response_is_free_of_ground_truth(seeded: TestClient) -> None:
    for incident in incident_ids(seeded):
        for suffix in ("", "/graph", "/timeline", "/coverage"):
            assert_safe(seeded.get(f"{API}/incidents/{incident}{suffix}").json())
    assert_safe(seeded.get(f"{API}/incidents").json())
    assert_safe(seeded.get(f"{API}/settings").json())
    assert_safe(seeded.get(f"{API}/integrations").json())


def test_seeding_loads_each_sample_once(trained_model: Path) -> None:
    store = Store(open_engine(None))
    service = IncidentService(store, load_model(trained_model), DEFAULT_TOLERANCE_SECONDS)
    assert seed_demo(service, REPO) == len(DEMO_INCIDENTS)
    assert seed_demo(service, REPO) == 0
    rows = store.incidents()
    assert len(rows) == len(DEMO_INCIDENTS)
    assert all(row.is_demo for row in rows)
    for row in rows:
        assert_safe(json.loads(row.case_json))
        assert json.loads(row.case_json)["case_id"] == "sample"


def test_demo_titles_do_not_name_the_simulated_fault() -> None:
    for name, title, *_ in DEMO_INCIDENTS:
        raw = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
        culprit = (raw.get("simulation") or {}).get("root_cause_service")
        if culprit:
            assert culprit not in title.lower()


def test_incidents_survive_a_restart(trained_model: Path, tmp_path: Path) -> None:
    settings = ApiSettings(
        model_dir=trained_model, root=REPO, database=tmp_path / "nested" / "app.db"
    )
    with TestClient(create_app(settings)) as first:
        created = first.post(f"{API}/incidents", json=create_body()).json()
    with TestClient(create_app(settings)) as second:
        assert second.get(f"{API}/incidents/{created['id']}").json() == created


def test_ids_round_trip() -> None:
    assert parse_incident_id(incident_id(42)) == 42
    assert parse_incident_id("42") is None


def test_cors_allows_only_the_configured_origin(trained_model: Path) -> None:
    allowed = "http://localhost:5173"
    settings = ApiSettings(model_dir=trained_model, root=REPO, cors_origins=(allowed,))
    with TestClient(create_app(settings)) as origin_client:
        good = origin_client.get(f"{API}/settings", headers={"origin": allowed})
        bad = origin_client.get(f"{API}/settings", headers={"origin": "http://example.test"})
    assert good.headers["access-control-allow-origin"] == allowed
    assert "access-control-allow-origin" not in bad.headers


def test_no_socket_is_opened(seeded: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("A socket connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert seeded.post(f"{API}/incidents", json=create_body()).status_code == 201
    assert seeded.post(f"{API}/incidents/INC-0001/rerun").status_code == 200
    assert seeded.get(f"{API}/incidents/INC-0001/graph").status_code == 200
    assert seeded.get(f"{API}/model").status_code == 200


def test_committed_openapi_matches_the_app(client: TestClient) -> None:
    snapshot = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    assert client.get("/openapi.json").json() == snapshot


def test_openapi_has_no_dangling_references(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    text = json.dumps(schema)
    assert "#/$defs/" not in text
    for name in sorted(set(re.findall(r"#/components/schemas/(\w+)", text))):
        assert name in schema["components"]["schemas"]
