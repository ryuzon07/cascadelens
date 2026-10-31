import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from baseline_factory import build_project
from typer.testing import CliRunner

from cascadelens.baselines.base import SummaryCache
from cascadelens.baselines.experiments import baselines_clean, leakage_demo
from cascadelens.cli import app

RUNNER = CliRunner()
EXPERIMENTS = Path(__file__).parents[2] / "experiments"


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("experiments")
    build_project(root)
    for name in ("leakage_demo", "baselines_clean"):
        target = root / "experiments" / name
        target.mkdir(parents=True)
        shutil.copy(EXPERIMENTS / name / "config.yaml", target / "config.yaml")
        path = target / "config.yaml"
        text = path.read_text(encoding="utf-8").replace("10000", "200")
        path.write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def cache() -> SummaryCache:
    return SummaryCache()


def read(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def test_leakage_demo_writes_every_output(project: Path, cache: SummaryCache) -> None:
    payload = leakage_demo(project, cache)

    directory = project / "experiments" / "leakage_demo"
    note = (directory / "note.md").read_text(encoding="utf-8")
    assert (directory / "figures" / "leakage_top1.png").is_file()
    assert (directory / "figures" / "leakage_top1.svg").is_file()
    assert "Paired difference, random minus group_run" in note
    assert "paired gap of" in note
    saved = read(directory / "results.json")
    assert set(saved["metrics"]["scores"]) == {"frequency", "lightgbm_flat"}
    assert set(saved["metrics"]["paired_differences"]["frequency"]) == {"Top-1", "Top-3", "Avg@5"}
    assert saved["data_checksum"] != "unknown"
    assert payload["scores"] == saved["metrics"]["scores"]


def test_baselines_clean_separates_tables_and_reports_folds(
    project: Path, cache: SummaryCache
) -> None:
    payload = baselines_clean(project, cache)

    directory = project / "experiments" / "baselines_clean"
    text = (directory / "results.md").read_text(encoding="utf-8")
    assert text.index("Supervised: group_run") < text.index("Unsupervised: group_run")
    assert "| rule | loso | shop |" in text
    assert "| frequency | lofo | cpu |" in text
    folds: Any = payload["folds"]
    assert len(folds) == 4 * (4 + 3)
    assert read(directory / "results.json")["config_hash"]


def test_cli_reports_a_missing_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "experiments" / "leakage_demo"
    target.mkdir(parents=True)
    (target / "config.yaml").write_text("methods: [frequency]\n", encoding="utf-8")

    result = RUNNER.invoke(app, ["experiments", "leakage-demo"])

    assert result.exit_code == 1
    assert "leakage demo failed" in result.output
