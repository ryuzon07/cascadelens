import ast
import json
import socket
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from cascadelens.api.main import ApiSettings, create_app
from cascadelens.cli import app as cli_app
from cascadelens.inference import CaseInput, DiagnoseResult, load_model, read_case
from cascadelens.inference import diagnose as run_diagnosis
from cascadelens.models.data import RankerError
from cascadelens.models.diagnose import DISCLAIMER

REPO = Path(__file__).parents[2]
SAMPLES = REPO / "demo_data"
GROUND_TRUTH = {"root_cause_service", "inject_time"}
SAFE_CODE = [
    REPO / "src/cascadelens/inference.py",
    *sorted((REPO / "src/cascadelens/api").glob("*.py")),
    *sorted((REPO / "src/cascadelens/demo").glob("*.py")),
]
BANNED_MODULES = {
    "aiohttp",
    "ftplib",
    "http",
    "httpx",
    "os.system",
    "paramiko",
    "requests",
    "smtplib",
    "socket",
    "subprocess",
    "urllib",
    "urllib3",
    "websockets",
}


def body(name: str = "payment-cpu") -> dict[str, Any]:
    raw: dict[str, Any] = json.loads((SAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    raw.pop("simulation", None)
    return raw


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


@pytest.fixture(scope="module")
def client(trained_model: Path) -> Iterator[TestClient]:
    settings = ApiSettings(model_dir=trained_model, root=REPO, max_body_bytes=2_000_000)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_health_reports_a_loaded_model(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["disclaimer"] == DISCLAIMER


def test_model_info_has_provenance_and_headline_metrics(client: TestClient) -> None:
    info = client.get("/model-info").json()
    assert info["training_data_checksum"]
    assert info["feature_config_hash"]
    assert info["frozen_config_hash"] is None
    assert info["headline_metrics"]["two_stage_top1_group_run"]["estimate"] > 0
    assert info["disclaimer"] == DISCLAIMER
    assert_safe(info)


def test_diagnose_returns_a_ranking(client: TestClient) -> None:
    response = client.post("/diagnose", json=body())
    assert response.status_code == 200
    result = response.json()
    assert result["ranking"]
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["abstain"], bool)
    assert result["disclaimer"] == DISCLAIMER
    evidence = result["ranking"][0]["evidence"]
    assert {"onset_rank", "upstream_anom_ratio", "missing_frac", "top_groups"} <= set(evidence)
    assert evidence["top_groups"]
    assert_safe(result)


def invalid_cases() -> list[tuple[str, dict[str, Any]]]:
    late = body()
    late["metrics"][0]["ts"] = late["window"]["end"] + 100_000
    repeated = body()
    repeated["metrics"].append(repeated["metrics"][0])
    return [
        ("empty metrics", {**body(), "metrics": []}),
        ("start after end", {**body(), "window": {"start": 10, "end": 5}}),
        ("empty window", {**body(), "window": {"start": 5, "end": 5}}),
        ("timestamp far outside", late),
        ("repeated sample", repeated),
        ("label sent in", {**body(), "root_cause_service": "payment"}),
        ("not a case", {"hello": 1}),
    ]


@pytest.mark.parametrize(("name", "payload"), invalid_cases(), ids=lambda item: str(item)[:20])
def test_invalid_payloads_get_a_message(
    client: TestClient, name: str, payload: dict[str, Any]
) -> None:
    response = client.post("/diagnose", json=payload)
    assert response.status_code == 422, name
    assert response.json()["detail"]
    assert response.json()["disclaimer"] == DISCLAIMER
    assert_safe(response.json())


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_non_finite_values_are_rejected(client: TestClient, constant: str) -> None:
    raw = body()
    raw["metrics"][0]["value"] = "__BAD__"
    bad = json.dumps(raw).replace('"__BAD__"', constant)
    response = client.post("/diagnose", content=bad, headers={"content-type": "application/json"})
    assert response.status_code == 422
    assert response.json()["disclaimer"] == DISCLAIMER


def test_broken_json_is_rejected(client: TestClient) -> None:
    response = client.post("/diagnose", content=b'{"metrics": [')
    assert response.status_code == 422
    assert "JSON" in response.json()["detail"]


def test_oversized_body_gets_413(trained_model: Path) -> None:
    settings = ApiSettings(model_dir=trained_model, root=REPO, max_body_bytes=1000)
    with TestClient(create_app(settings)) as small:
        response = small.post("/diagnose", json=body())
        streamed = small.post("/diagnose", content=(b"x" * 500 for _ in range(4)))
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]
    assert response.json()["disclaimer"] == DISCLAIMER
    assert streamed.status_code == 413


def test_unknown_routes_carry_the_disclaimer(client: TestClient) -> None:
    response = client.get("/nothing")
    assert response.status_code == 404
    assert response.json()["disclaimer"] == DISCLAIMER


def test_missing_model_is_a_clear_startup_error(tmp_path: Path) -> None:
    with pytest.raises(RankerError, match="No model directory"):
        create_app(ApiSettings(model_dir=tmp_path / "absent"))


def test_cors_is_off_by_default(client: TestClient) -> None:
    response = client.get("/health", headers={"origin": "http://example.test"})
    assert "access-control-allow-origin" not in response.headers


def test_no_socket_is_opened_while_handling_requests(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, trained_model: Path
) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("A socket connection was attempted")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    assert client.post("/diagnose", json=body()).status_code == 200
    assert client.get("/model-info").status_code == 200
    assert client.get("/health").status_code == 200
    run_diagnosis(read_case(SAMPLES / "payment-cpu.json"), trained_model)


@pytest.mark.parametrize("path", SAFE_CODE, ids=lambda path: path.name)
def test_safe_code_imports_nothing_that_reaches_out(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for name in names:
            assert name.split(".")[0] not in BANNED_MODULES, f"{path.name} imports {name}"


def test_cli_and_api_give_identical_rankings(client: TestClient, trained_model: Path) -> None:
    posted = client.post("/diagnose", json=body()).json()
    result = CliRunner().invoke(
        cli_app,
        [
            "diagnose",
            "--case",
            str(SAMPLES / "payment-cpu.json"),
            "--model",
            str(trained_model),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    printed = json.loads(result.output)
    assert printed["ranking"] == posted["ranking"]
    assert printed["confidence"] == posted["confidence"]
    assert printed["disclaimer"] == DISCLAIMER


def test_cli_table_prints_the_disclaimer(trained_model: Path) -> None:
    result = CliRunner().invoke(
        cli_app,
        ["diagnose", "--case", str(SAMPLES / "payment-cpu.json"), "--model", str(trained_model)],
    )
    assert result.exit_code == 0
    assert DISCLAIMER in result.output
    assert "1. " in result.output


def test_abstaining_prints_the_banner(
    trained_model: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def abstaining(case: CaseInput, model: Path) -> DiagnoseResult:
        return run_diagnosis(case, model).model_copy(update={"abstain": True})

    monkeypatch.setattr("cascadelens.cli.run_diagnosis", abstaining)
    result = CliRunner().invoke(
        cli_app,
        ["diagnose", "--case", str(SAMPLES / "payment-cpu.json"), "--model", str(trained_model)],
    )
    assert "ABSTAIN: low confidence" in result.output


def test_output_is_deterministic_apart_from_timing(trained_model: Path) -> None:
    case = read_case(SAMPLES / "payment-cpu.json")
    model = load_model(trained_model)
    first = run_diagnosis(case, model).model_dump(exclude={"timing"})
    second = run_diagnosis(case, model).model_dump(exclude={"timing"})
    assert first == second


@pytest.mark.parametrize("path", ["/docs", "/redoc"])
def test_pages_that_load_public_scripts_are_off(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 404
