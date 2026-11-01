from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("sqlmodel")

from fastapi.testclient import TestClient

from cascadelens.api.main import ApiSettings, create_app
from cascadelens.models.diagnose import DISCLAIMER

REPO = Path(__file__).parents[2]
PAGE = "<!doctype html><title>page</title>"


@pytest.fixture
def ui(tmp_path: Path) -> Path:
    folder = tmp_path / "dist"
    (folder / "assets").mkdir(parents=True)
    (folder / "index.html").write_text(PAGE, encoding="utf-8")
    (folder / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("private", encoding="utf-8")
    return folder


@pytest.fixture
def client(trained_model: Path, ui: Path) -> Iterator[TestClient]:
    settings = ApiSettings(model_dir=trained_model, root=REPO, ui_dir=ui)
    with TestClient(create_app(settings)) as test_client:
        yield test_client


def test_serves_the_page_and_its_assets(client: TestClient) -> None:
    assert client.get("/").text == PAGE
    assert client.get("/assets/app.js").text == "console.log(1)"


@pytest.mark.parametrize("path", ["/incidents", "/incidents/INC-0001/what-if", "/model"])
def test_client_routes_get_the_page(client: TestClient, path: str) -> None:
    assert client.get(path).text == PAGE


def test_api_routes_win(client: TestClient) -> None:
    assert client.get("/health").json()["model_loaded"] is True
    assert client.get("/api/v1/settings").status_code == 200
    assert client.get("/openapi.json").json()["info"]["title"] == "CascadeLens"


@pytest.mark.parametrize("path", ["/api/v1/nothing", "/api/v1"])
def test_unknown_api_paths_stay_json_errors(client: TestClient, path: str) -> None:
    response = client.get(path)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"
    assert response.json()["disclaimer"] == DISCLAIMER


@pytest.mark.parametrize(
    "path", ["/../secret.txt", "/assets/../../secret.txt", "/%2e%2e/secret.txt", "/..%2fsecret.txt"]
)
def test_nothing_outside_the_folder_is_served(client: TestClient, path: str) -> None:
    assert "private" not in client.get(path).text


def test_ui_stays_off_unless_asked(trained_model: Path) -> None:
    settings = ApiSettings(model_dir=trained_model, root=REPO)
    with TestClient(create_app(settings)) as bare:
        assert bare.get("/").status_code == 404
        assert bare.get("/incidents").status_code == 404


def test_a_missing_build_is_a_clear_error(trained_model: Path, tmp_path: Path) -> None:
    settings = ApiSettings(model_dir=trained_model, root=REPO, ui_dir=tmp_path / "absent")
    with pytest.raises(RuntimeError, match="make web-build"):
        create_app(settings)


def test_the_schema_is_the_same_with_the_ui(client: TestClient, trained_model: Path) -> None:
    with_ui = client.get("/openapi.json").json()
    settings = ApiSettings(model_dir=trained_model, root=REPO)
    with TestClient(create_app(settings)) as bare:
        assert bare.get("/openapi.json").json() == with_ui
