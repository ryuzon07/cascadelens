import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from baseline_factory import build_project
from ranker_factory import write_project_configs
from typer.testing import CliRunner

from cascadelens.baselines.experiments import baselines_clean
from cascadelens.baselines.runner import load_dataset, load_split
from cascadelens.cli import app
from cascadelens.data.io import metrics_path
from cascadelens.models.artifacts import latest_artifact, load_artifact, read_sidecar
from cascadelens.models.data import FeatureSource, RankerError
from cascadelens.models.diagnose import diagnose_case
from cascadelens.models.experiment import best_baseline, ranker_clean
from cascadelens.models.freeze import find_frozen
from cascadelens.models.pipeline import (
    evaluate,
    feature_settings,
    freeze_config,
    load_ranker_config,
    train_folds,
)

CONFIG = Path("configs/ranker.yaml")


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("ranker")
    build_project(root)
    write_project_configs(root)
    return root


@pytest.mark.parametrize("split", ["group_run", "lofo", "loso"])
def test_validation_never_overlaps_test(project: Path, split: str) -> None:
    saved, _ = load_split(split, load_dataset(project), project)
    for fold in saved.folds:
        assert fold.val_ids
        assert not set(fold.val_ids) & set(fold.test_ids)
        assert not set(fold.val_ids) & set(fold.train_ids)


def test_evaluation_refuses_a_config_that_is_not_frozen(project: Path) -> None:
    with pytest.raises(RankerError, match="not frozen"):
        evaluate(CONFIG, "group_run", 0, "two_stage_ranker", False, project)
    with pytest.raises(RankerError, match="not frozen"):
        ranker_clean(False, project)


def test_cli_refuses_without_exploratory(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    result = CliRunner().invoke(app, ["evaluate", "--split", "group_run"])
    assert result.exit_code == 1
    assert "not frozen" in result.output


def test_exploratory_results_are_tagged_and_keep_every_case(project: Path) -> None:
    result = evaluate(CONFIG, "group_run", 0, "two_stage_ranker", True, project)
    assert result.status == "exploratory"
    path = project / "experiments/ranker_clean/runs/two_stage_ranker/group_run/seed_0/result.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["metrics"]["status"] == "exploratory"
    assert saved["metrics"]["n_cases"] == 96
    for fold in saved["metrics"]["fold_details"]:
        assert fold["k"] in (2, 3)
        assert fold["recall@2"] <= fold["recall@3"]
        assert fold["test_top1"] <= fold["recall_at_chosen_k"]


def test_freeze_records_choices_and_evaluation_reuses_them(project: Path, tmp_path: Path) -> None:
    root = tmp_path / "copy"
    shutil.copytree(project, root)
    path = freeze_config(CONFIG, "candidate", root)
    assert path == root / "configs" / "frozen" / "candidate.yaml"
    record = find_frozen(load_ranker_config(CONFIG, root), root)
    assert record is not None
    assert set(record.families) == {"group_run", "lofo", "loso"}
    assert set(record.families["loso"]) == {"two_stage_ranker", "single_stage_ranker"}
    result = evaluate(CONFIG, "group_run", 0, "two_stage_ranker", False, root)
    assert result.status == "frozen"
    entries = record.families["group_run"]["two_stage_ranker"]
    assert [f.choice.params for f in result.folds] == [e["params"] for e in entries]
    with pytest.raises(RankerError, match="already exists"):
        freeze_config(CONFIG, "candidate", root)


def test_saved_artifact_gives_identical_predictions(project: Path) -> None:
    saved = train_folds(CONFIG, "group_run", 0, "two_stage_ranker", project)
    directory, fitted = saved[0]
    loaded, features = load_artifact(directory)
    assert features == feature_settings(load_ranker_config(CONFIG, project), project)
    sidecar = read_sidecar(directory)
    assert {"config_hash", "feature_config_hash", "data_checksum"} <= set(sidecar)
    assert directory.name == f"v{sidecar['config_hash'][:12]}"
    dataset = load_dataset(project)
    split, _ = load_split("group_run", dataset, project)
    source = FeatureSource(features, project)
    for case in dataset.handles(split.folds[0].test_ids[:5]):
        table = source.table(case)
        assert fitted.rank(table)[0].equals(loaded.rank(table)[0])
    case_file = metrics_path(project / "data" / "processed", split.folds[0].test_ids[0])
    ranked = diagnose_case(case_file, directory)
    assert 1 <= ranked.height <= 4
    assert ranked["score"].is_sorted(descending=True)
    assert latest_artifact(project).parts[-2].startswith("fold_")


def test_diagnose_rejects_a_bad_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(path)
    with pytest.raises(RankerError, match="schema"):
        diagnose_case(path, tmp_path)
    with pytest.raises(RankerError, match="No case file"):
        diagnose_case(tmp_path / "none.parquet", tmp_path)


def test_ranker_clean_writes_tagged_results(project: Path) -> None:
    baselines_clean(project)
    reproduction = project / "experiments" / "reproduction" / "evaluation"
    reproduction.mkdir(parents=True, exist_ok=True)
    (reproduction / "results.json").write_text(
        json.dumps({"metrics": {"group_run": [{"method": "baro", "Top-1": {"estimate": 0.0}}]}}),
        encoding="utf-8",
    )
    assert best_baseline(project)[0] != "baro"
    payload = ranker_clean(True, project)
    directory = project / "experiments" / "ranker_clean"
    assert payload["status"] == "exploratory"
    assert (directory / "results.json").is_file()
    assert (directory / "figures" / "ranker_top1.png").is_file()
    assert "Status: exploratory" in (directory / "results.md").read_text(encoding="utf-8")
    assert {row["method"] for row in payload["group_run"]} == {
        "two_stage_ranker",
        "single_stage_ranker",
    }
    comparisons = payload["comparisons"]
    assert comparisons["S2_vs_best_baseline"]["estimate"] == pytest.approx(
        comparisons["S2_unshifted_difference"]["estimate"] + 0.12
    )
    assert sum(payload["importance"][0]["groups"].values()) == pytest.approx(1.0)
