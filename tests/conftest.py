import sys
from pathlib import Path

import pytest

for folder in ("baselines", "models", "perturb"):
    sys.path.insert(0, str(Path(__file__).parent / folder))


@pytest.fixture(scope="session")
def trained_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return a tiny saved model with a calibrator, for the API and demo tests."""
    from baseline_factory import build_project
    from ranker_factory import write_project_configs

    from cascadelens.models.abstain import calibrate_split
    from cascadelens.models.artifacts import latest_artifact

    root = tmp_path_factory.mktemp("served")
    build_project(root)
    write_project_configs(root)
    (root / "configs" / "calibration.yaml").write_text(
        "seed: 0\nn_resamples: 200\n", encoding="utf-8"
    )
    calibrate_split(Path("configs/ranker.yaml"), "group_run", "two_stage_ranker", root)
    return latest_artifact(root)
