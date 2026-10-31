from pathlib import Path

import pytest
from baseline_factory import build_project
from typer.testing import CliRunner

from cascadelens.baselines.runner import load_dataset
from cascadelens.cli import app
from cascadelens.data.io import metrics_path
from cascadelens.perturb.base import PerturbationError, output_hash
from cascadelens.perturb.registry import build
from cascadelens.perturb.store import PerturbedStore, apply_split, apply_to, spec_hash
from cascadelens.perturb.summary import SUMMARY_PATH, summarize

SUITE = """seed: 0
perturbations:
  - {id: P1, name: random_metric_loss, realism: simulated, params: {rate: [0.2]}}
  - {id: P2, name: failure_correlated_loss, realism: simulated, params: {rate: [0.2]}}
  - {id: P6, name: edge_deletion, realism: simulated, params: {rate: [0.2]}}
  - {id: P9, name: unseen_fault_type, realism: held-out, params: {split: [lofo]}}
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("perturb")
    build_project(root)
    (root / "configs" / "perturbations.yaml").write_text(SUITE, encoding="utf-8")
    return root


def test_apply_writes_a_cached_dataset_and_manifest(project: Path) -> None:
    first = apply_split("P2", {"rate": 0.3}, "group_run", 0, project)
    assert len(first.rows) == 96
    assert (first.directory / "manifest.jsonl").is_file()
    assert first.rows[0].realism == "simulated"
    path = metrics_path(first.directory, first.rows[0].case_id)
    stamp = path.stat().st_mtime_ns
    second = apply_split("P2", {"rate": 0.3}, "group_run", 0, project)
    assert second.directory == first.directory
    assert second.rows == first.rows
    assert path.stat().st_mtime_ns == stamp


def test_stored_case_matches_a_direct_application(project: Path) -> None:
    dataset = load_dataset(project)
    step = build("P1", {"rate": 0.2})
    store = PerturbedStore([step], 0, dataset.checksum, project)
    record = next(iter(dataset.records.values()))
    stored = store.case(record)
    assert output_hash(stored) == store.ensure(record)[0].output_hash


def test_workers_give_the_same_cache(project: Path) -> None:
    dataset = load_dataset(project)
    ids = sorted(dataset.records)[:4]
    serial = apply_to("P12", {}, 3, ids, project, workers=1)
    parallel = apply_to("P12", {}, 4, ids, project, workers=2)
    assert serial.directory != parallel.directory
    assert [r.case_id for r in serial.rows] == [r.case_id for r in parallel.rows]


def test_topology_steps_store_no_metrics(project: Path) -> None:
    result = apply_split("P6", {"rate": 0.2}, "group_run", 0, project)
    assert not (result.directory / "metrics").exists()


def test_hash_changes_with_parameters_and_seed() -> None:
    low, high = build("P1", {"rate": 0.1}), build("P1", {"rate": 0.2})
    assert spec_hash([low], 0, "x") != spec_hash([high], 0, "x")
    assert spec_hash([low], 0, "x") != spec_hash([low], 1, "x")
    assert spec_hash([low], 0, "x") != spec_hash([low], 0, "y")


def test_split_references_cannot_be_applied(project: Path) -> None:
    with pytest.raises(PerturbationError, match="split"):
        apply_split("P9", {}, "group_run", 0, project)


def test_summary_reports_labels_and_no_changing_topology_claim(project: Path) -> None:
    lines = summarize(project, max_cases=12)
    assert [line["id"] for line in lines] == ["P1", "P2", "P6"]
    assert all(line["realism"] == "simulated" for line in lines)
    text = (project / SUMMARY_PATH).read_text(encoding="utf-8")
    assert "imperfect graph, not a changing one" in text
    assert "held-out" in text


def test_cli_lists_the_suite(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    result = CliRunner().invoke(app, ["perturb", "list"])
    assert result.exit_code == 0
    assert "P9 unseen_fault_type [held-out, split reference]" in result.output


def test_cli_apply_reports_failure(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    runner = CliRunner()
    assert runner.invoke(app, ["perturb", "apply", "--name", "P9"]).exit_code == 1
    assert runner.invoke(app, ["perturb", "apply", "--name", "P1"]).exit_code == 1
    done = runner.invoke(app, ["perturb", "apply", "--name", "P1", "--params", "rate=0.2"])
    assert done.exit_code == 0
