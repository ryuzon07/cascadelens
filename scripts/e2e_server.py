"""Serve the built UI on a fresh database for the Playwright tests. Needs the api extra.

It trains the tiny model the API tests use, on synthetic data, so it needs no benchmark data and
works in CI. ``web/playwright.config.ts`` runs it as ``e2e_server.py <port> <ui folder>``.
"""

import sys
import tempfile
from pathlib import Path

import uvicorn

from cascadelens.api.incidents import IncidentService
from cascadelens.api.main import ApiSettings, create_app
from cascadelens.api.seed import seed_demo
from cascadelens.api.store import Store, open_engine
from cascadelens.inference import DEFAULT_TOLERANCE_SECONDS, load_model
from cascadelens.models.abstain import calibrate_split
from cascadelens.models.artifacts import latest_artifact

REPO = Path(__file__).parents[1]


def tiny_model(root: Path) -> Path:
    """Train and calibrate a small model on synthetic cases."""
    for factory in sorted((REPO / "tests").rglob("*factory.py")):
        sys.path.insert(0, str(factory.parent))
    from baseline_factory import build_project
    from ranker_factory import write_project_configs

    build_project(root)
    write_project_configs(root)
    (root / "configs" / "calibration.yaml").write_text(
        "seed: 0\nn_resamples: 200\n", encoding="utf-8"
    )
    calibrate_split(Path("configs/ranker.yaml"), "group_run", "two_stage_ranker", root)
    return latest_artifact(root)


def main() -> None:
    """Build everything in a temporary folder and serve it."""
    port, ui = int(sys.argv[1]), Path(sys.argv[2])
    work = Path(tempfile.mkdtemp(prefix="cascadelens-e2e-"))
    directory = tiny_model(work / "model")
    database = work / "app.db"
    store = Store(open_engine(database))
    seed_demo(IncidentService(store, load_model(directory), DEFAULT_TOLERANCE_SECONDS), REPO)
    settings = ApiSettings(model_dir=directory, root=REPO, database=database, ui_dir=ui)
    uvicorn.run(create_app(settings), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
