import json
from pathlib import Path
from typing import Any

import pytest
from split_factory import (
    build_fixture_dataset,
    identity_groups,
    index_records,
    near_duplicate_dataset,
    splits_config,
)
from typer.testing import CliRunner

from cascadelens.cli import app
from cascadelens.config import config_hash
from cascadelens.data.io import MANIFEST_FILE, Manifest, read_manifest, write_manifest
from cascadelens.splits.errors import SplitError
from cascadelens.splits.groups import build_groups
from cascadelens.splits.report import build_report
from cascadelens.splits.splitters import Fold, Split, group_run_split, random_split
from cascadelens.splits.store import (
    CLUSTER_FILE,
    GROUPS_FILE,
    REPORT_FILE,
    SPLIT_FILE,
    build_splits,
    check_splits,
    read_split,
)

RUNNER = CliRunner()
RECORDS = index_records()
GROUPS = identity_groups(RECORDS)
SUMMARY = build_groups(RECORDS, lambda _: pytest.fail("no candidates"), 0.98, 8).summary
CONFIG = splits_config()


def split_dir(project: Path, name: str) -> Path:
    return project / "data" / "splits" / name


def read_json(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def leak_a_test_case(project: Path) -> None:
    path = split_dir(project, "group_run") / SPLIT_FILE
    saved = read_json(path)
    fold = saved["split"]["folds"][0]
    fold["train_ids"] = [*fold["train_ids"], fold["test_ids"][0]]
    path.write_text(json.dumps(saved), encoding="utf-8")


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    build_fixture_dataset(tmp_path)
    (tmp_path / "splits.yaml").write_text(
        "seed: 0\nstrategies: [random, group_run, lofo, loso]\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_report_passes_for_a_grouped_split() -> None:
    split = group_run_split(RECORDS, GROUPS, 5, 0.2, 0)

    report = build_report(split, RECORDS, GROUPS, SUMMARY, "cfg", "data")

    assert report.passed
    assert not report.leaky_by_design
    assert report.overall.train_test == report.overall.folds_with_overlap == 0


def test_report_fails_for_random_with_per_fold_overlap() -> None:
    report = build_report(random_split(RECORDS, 5, 0.2, 0), RECORDS, GROUPS, SUMMARY, "c", "d")

    assert not report.passed
    assert report.leaky_by_design
    assert report.overall.train_test == sum(len(f.train_test_overlap) for f in report.folds)
    assert report.overall.train_test > 0


def test_report_counts_match_the_fold() -> None:
    split = group_run_split(RECORDS, GROUPS, 5, 0.2, 0)
    fold = split.folds[0]

    entry = build_report(split, RECORDS, GROUPS, SUMMARY, "c", "d").folds[0]

    assert entry.test.cases == len(fold.test_ids)
    assert entry.test.groups == len({GROUPS[case_id] for case_id in fold.test_ids})
    assert sum(entry.test.labels.values()) == len(fold.test_ids)
    assert entry.train.systems == 3
    assert entry.train.fault_types == 4


def test_report_lists_test_labels_missing_from_train() -> None:
    first, second = RECORDS[0], next(r for r in RECORDS if r.root_cause_service == "db")
    third = next(r for r in RECORDS if r.root_cause_service == "cart")
    fold = Fold(train_ids=(first.case_id,), val_ids=(third.case_id,), test_ids=(second.case_id,))
    split = Split(name="group_run", seed=0, folds=(fold,))

    report = build_report(split, RECORDS, GROUPS, SUMMARY, "c", "d")

    assert report.folds[0].test_labels_unseen_in_train == ("db",)


def test_a_hand_built_leak_in_a_grouped_split_is_reported() -> None:
    twins = [r.case_id for r in RECORDS if r.run_id == RECORDS[0].run_id]
    fold = Fold(train_ids=(twins[0],), val_ids=(RECORDS[-1].case_id,), test_ids=(twins[1],))
    split = Split(name="group_run", seed=0, folds=(fold,))

    report = build_report(split, RECORDS, GROUPS, SUMMARY, "c", "d")

    assert not report.passed
    assert report.folds[0].train_test_overlap == (RECORDS[0].run_id,)


def test_build_writes_every_artifact_with_provenance(project: Path) -> None:
    reports = build_splits(CONFIG)

    processed = project / "data" / "processed"
    checksum = read_manifest(processed).digest()
    assert [report.split for report in reports] == ["random", "group_run", "lofo", "loso"]
    assert (project / "data" / "splits" / CLUSTER_FILE).is_file()
    for name in ("random", "group_run", "lofo", "loso"):
        directory = split_dir(project, name)
        assert (directory / GROUPS_FILE).is_file()
        saved = read_json(directory / SPLIT_FILE)
        report = read_json(directory / REPORT_FILE)
        for payload in (saved, report):
            assert payload["config_hash"] == config_hash(CONFIG)
            assert payload["data_manifest_checksum"] == checksum
    assert read_json(split_dir(project, "random") / REPORT_FILE)["passed"] is False
    assert read_json(split_dir(project, "group_run") / REPORT_FILE)["passed"] is True


def test_saved_split_round_trips(project: Path) -> None:
    build_splits(CONFIG)

    saved, group_map = read_split(split_dir(project, "group_run"))

    assert saved.split.name == "group_run"
    assert len(group_map) == 10


def test_build_is_byte_identical_when_repeated(project: Path) -> None:
    def snapshot() -> dict[Path, bytes]:
        root = project / "data" / "splits"
        return {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}

    build_splits(CONFIG)
    first = snapshot()
    build_splits(CONFIG)

    assert snapshot() == first


def test_build_without_data_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(SplitError, match="data build"):
        build_splits(CONFIG, tmp_path)


def test_failed_build_keeps_the_previous_one(project: Path) -> None:
    build_splits(CONFIG)
    before = (split_dir(project, "group_run") / SPLIT_FILE).read_bytes()

    with pytest.raises(SplitError, match="cannot fill"):
        build_splits(splits_config(n_folds=50))

    assert (split_dir(project, "group_run") / SPLIT_FILE).read_bytes() == before


def test_build_refuses_to_replace_foreign_contents(project: Path) -> None:
    output = project / "data" / "splits"
    output.mkdir()
    (output / "keep.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(SplitError, match="previous build"):
        build_splits(CONFIG)

    assert (output / "keep.txt").is_file()
    assert not (project / "data" / "splits.staging").exists()


def test_check_passes_and_skips_random(project: Path) -> None:
    build_splits(CONFIG)

    results = check_splits(CONFIG)

    assert [result.name for result in results] == ["group_run", "lofo", "loso"]
    assert all(not result.problems for result in results)


def test_check_finds_a_planted_overlap(project: Path) -> None:
    build_splits(CONFIG)
    leak_a_test_case(project)

    results = {result.name: result.problems for result in check_splits(CONFIG)}

    assert any("train/test share 1 group" in problem for problem in results["group_run"])
    assert results["loso"] == ()


def test_check_flags_a_changed_dataset(project: Path) -> None:
    build_splits(CONFIG)
    manifest = project / "data" / "processed" / MANIFEST_FILE
    payload = read_json(manifest)
    payload["suites"] = ["RE1-OB"]
    payload.pop("digest")

    write_manifest(Manifest.model_validate(payload), manifest.parent)

    assert all("different dataset version" in r.problems[0] for r in check_splits(CONFIG))


def test_check_flags_a_changed_config(project: Path) -> None:
    build_splits(CONFIG)

    results = check_splits(splits_config(seed=1))

    assert all("different splits config" in result.problems[0] for result in results)


def test_check_without_saved_splits_fails(project: Path) -> None:
    with pytest.raises(SplitError, match="No saved grouped splits"):
        check_splits(CONFIG)


def test_cli_build_and_check(project: Path) -> None:
    built = RUNNER.invoke(app, ["splits", "build", "--config", "splits.yaml"])
    checked = RUNNER.invoke(app, ["splits", "check", "--config", "splits.yaml"])

    assert built.exit_code == 0, built.output
    assert "random: 5 fold(s), overlap FAIL (leaky by design)" in built.output
    assert "group_run: 5 fold(s), overlap pass" in built.output
    assert checked.exit_code == 0, checked.output


def test_cli_check_exits_non_zero_on_overlap(project: Path) -> None:
    RUNNER.invoke(app, ["splits", "build", "--config", "splits.yaml"])
    leak_a_test_case(project)

    checked = RUNNER.invoke(app, ["splits", "check", "--config", "splits.yaml"])

    assert checked.exit_code == 1
    assert "group_run: FAIL" in checked.output


def test_cli_build_reports_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "splits.yaml"
    config.write_text("seed: 0\nstrategies: [loso]\n", encoding="utf-8")

    built = RUNNER.invoke(app, ["splits", "build", "--config", str(config)])

    assert built.exit_code == 1
    assert "splits build failed" in built.output


def test_near_duplicates_flow_through_to_the_report(tmp_path: Path) -> None:
    project = tmp_path
    near_duplicate_dataset(project / "data" / "processed")

    reports = build_splits(CONFIG, project)

    grouped = [report for report in reports if not report.leaky_by_design]
    assert all(report.passed for report in grouped)
    assert grouped[0].clusters.merged[0].group_id == "alpha/api_cpu"
    assert grouped[0].clusters.merged[0].cases == 4
