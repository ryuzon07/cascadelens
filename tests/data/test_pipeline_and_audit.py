import shutil
from pathlib import Path

import pytest
from case_factory import FIXTURE_DIR, FIXTURE_SUITES
from typer.testing import CliRunner

from cascadelens.cli import app
from cascadelens.config import DataConfig
from cascadelens.data.audit import (
    AuditData,
    CaseStats,
    RawInventory,
    collect,
    find_concerns,
    inventory_raw,
    overlapping_cases,
    render,
    run_audit,
)
from cascadelens.data.errors import DataError, DataValidationError, Issue, RawLayoutError
from cascadelens.data.io import CASES_FILE, MANIFEST_FILE, read_cases, read_manifest
from cascadelens.data.pipeline import build

RUNNER = CliRunner()


@pytest.fixture
def project(tmp_path: Path) -> Path:
    shutil.copytree(FIXTURE_DIR, tmp_path / "data" / "raw")
    return tmp_path


CONFIG = DataConfig(source="rcaeval", suites=list(FIXTURE_SUITES))


def stat(**overrides: object) -> CaseStats:
    fields: dict[str, object] = {
        "case_id": "c",
        "suite": "RE1-OB",
        "system": "online-boutique",
        "fault_type": "cpu",
        "run_id": "r",
        "root_cause_service": "a",
        "services": 2,
        "metrics_per_service": (3, 3),
        "missing_fraction": 0.0,
        "window_seconds": 100,
        "sample_step": 5,
        "inject_position": 0.5,
        "window_start": 0,
        "window_end": 100,
        "has_topology": False,
    }
    return CaseStats(**{**fields, **overrides})  # type: ignore[arg-type]


def test_build_writes_everything(project: Path) -> None:
    warnings = build(CONFIG, project)

    processed = project / "data" / "processed"
    assert len(warnings) == 10
    assert (processed / CASES_FILE).is_file()
    assert len(list(read_cases(processed))) == 10
    assert len(read_manifest(processed).files) == 20


def test_build_is_byte_identical_when_repeated(project: Path) -> None:
    processed = project / "data" / "processed"
    build(CONFIG, project)
    first = {p: p.read_bytes() for p in sorted(processed.rglob("*")) if p.is_file()}
    build(CONFIG, project)
    second = {p: p.read_bytes() for p in sorted(processed.rglob("*")) if p.is_file()}

    assert first == second


def test_failed_build_keeps_the_previous_build(project: Path) -> None:
    build(CONFIG, project)
    manifest = (project / "data" / "processed" / MANIFEST_FILE).read_bytes()
    bad = project / "data" / "raw" / "RE1-OB" / "adservice_cpu" / "1" / "inject_time.txt"
    bad.write_text("5\n", encoding="utf-8")

    with pytest.raises(DataValidationError, match=r"re1-ob_adservice_cpu_1.*outside data range"):
        build(CONFIG, project)

    assert (project / "data" / "processed" / MANIFEST_FILE).read_bytes() == manifest
    assert not (project / "data" / "processed.staging").exists()


def test_load_failures_stop_the_build(project: Path) -> None:
    case = project / "data" / "raw" / "RE1-OB" / "adservice_cpu" / "1"
    (case / "data.csv").unlink()

    with pytest.raises(DataValidationError, match="RE1-OB/adservice_cpu/1: load"):
        build(CONFIG, project)


def test_build_refuses_to_replace_foreign_contents(project: Path) -> None:
    processed = project / "data" / "processed"
    processed.mkdir()
    (processed / "keep.txt").write_text("mine", encoding="utf-8")

    with pytest.raises(DataError, match="previous build"):
        build(CONFIG, project)

    assert (processed / "keep.txt").is_file()


def test_build_without_raw_data_fails(tmp_path: Path) -> None:
    with pytest.raises(RawLayoutError, match="does not exist"):
        build(CONFIG, tmp_path)


def test_build_without_suites_fails(project: Path) -> None:
    with pytest.raises(DataError, match="No suites"):
        build(DataConfig(source="rcaeval"), project)


def test_build_with_unknown_source_fails(project: Path) -> None:
    with pytest.raises(DataError, match="Unsupported"):
        build(DataConfig(source="other", suites=["RE1-OB"]), project)


def test_audit_has_every_requested_section(project: Path) -> None:
    output = project / "docs" / "data_audit.md"

    data = run_audit(CONFIG, output, project)

    text = output.read_text(encoding="utf-8")
    assert len(data.stats) == 10
    for heading in (
        "## Raw structure found",
        "## How fields are derived",
        "## Systems",
        "## Fault types",
        "## Injected services per system",
        "## Cases per (system, fault type)",
        "## Runs per system",
        "## Cases per run",
        "## Services per case",
        "## Metrics per service",
        "## Missing-sample fraction",
        "## Window length (seconds)",
        "## Topology availability per system",
        "## Concerns",
    ):
        assert heading in text
    assert text.rstrip().rsplit("## ", 1)[-1].startswith("Concerns")


def test_audit_is_deterministic(project: Path) -> None:
    assert render(collect(CONFIG, project), "raw") == render(collect(CONFIG, project), "raw")


def test_audit_without_data_says_so(tmp_path: Path) -> None:
    output = tmp_path / "audit.md"

    data = run_audit(CONFIG, output, tmp_path)

    assert data.stats == []
    assert "No raw data found" in output.read_text(encoding="utf-8")


def test_inventory_reports_an_unknown_layout(tmp_path: Path) -> None:
    case = tmp_path / "re1ob_adservice_cpu_1"
    case.mkdir()
    (case / "metrics.json").write_text("{}", encoding="utf-8")
    (case / "call_graph.csv").write_text("a,b\n", encoding="utf-8")

    inventory = inventory_raw(tmp_path)

    assert inventory.top_level == ("re1ob_adservice_cpu_1",)
    assert dict(inventory.extensions) == {".csv": 1, ".json": 1}
    assert inventory.topology_candidates == ("re1ob_adservice_cpu_1/call_graph.csv",)


def concerns_for(*stats: CaseStats) -> str:
    return "\n".join(find_concerns(AuditData(RawInventory(True), ("RE1-OB",), list(stats))))


def test_concern_tiny_groups() -> None:
    assert "Tiny group: online-boutique/cpu" in concerns_for(stat())


def test_concern_label_imbalance() -> None:
    stats = [stat(root_cause_service="a") for _ in range(7)] + [stat(root_cause_service="b")]

    assert "Label imbalance in online-boutique" in concerns_for(*stats)


def test_concern_window_tied_to_inject_time() -> None:
    assert "Window tied to inject_time in RE1-OB" in concerns_for(stat(), stat())


def test_no_tied_window_concern_when_position_varies() -> None:
    stats = [stat(inject_position=0.3), stat(inject_position=0.7)]

    assert "Window tied" not in concerns_for(*stats)


def test_tied_window_tolerates_a_few_truncated_cases() -> None:
    stats = [stat() for _ in range(9)] + [stat(inject_position=0.59)]

    assert "9 of 10 cases" in concerns_for(*stats)


def test_concern_window_length_varies_without_tracking_fault_type() -> None:
    stats = [stat(window_seconds=720), stat(window_seconds=4200)]

    assert "2 distinct lengths from 720 to 4200 s, not by fault type" in concerns_for(*stats)


def test_concern_window_length_tracks_fault_type() -> None:
    stats = [
        stat(fault_type="cpu", window_seconds=4200),
        stat(fault_type="cpu", window_seconds=4200),
        stat(fault_type="loss", window_seconds=720),
    ]

    text = concerns_for(*stats)

    assert "Window length tracks fault type in RE1-OB" in text
    assert "cpu=4200, loss=720" in text


def test_concern_for_repairs_and_empty_series_and_ignored_files() -> None:
    issues = [
        Issue("c1", "load_repair", "dropped 3 trailing row(s)", "warning"),
        Issue("c2", "nan_only_series", "8 series without values", "warning"),
    ]
    inventory = RawInventory(True, file_names=(("data.csv", 5), ("simple_data.csv", 2)))
    data = AuditData(inventory, ("RE1-OB",), [stat()], issues)

    text = "\n".join(find_concerns(data))

    assert "1 case(s) needed a raw-file repair" in text
    assert "1 case(s) hold series with no values" in text
    assert "never reads: simple_data.csv (2)" in text


def test_render_lists_repairs_and_groups_warnings() -> None:
    issues = [
        Issue("c1", "load_repair", "dropped 3 trailing row(s)", "warning"),
        Issue("c2", "load_repair", "dropped 5 trailing row(s)", "warning"),
    ]
    data = AuditData(RawInventory(True), ("RE1-OB",), [stat()], issues)

    text = render(data, "raw")

    assert "## Raw-file repairs" in text
    assert "| load_repair | 2 | c1 | dropped 3 trailing row(s) |" in text


def test_one_case_is_not_evidence_of_a_tied_window() -> None:
    assert "Window tied" not in concerns_for(stat())


def test_concern_missing_topology() -> None:
    assert "No topology for any case" in concerns_for(stat())


def test_concern_case_count_differs_from_documentation() -> None:
    assert "RE1-OB has 1 cases; the README states 125" in concerns_for(stat())


def test_concern_near_duplicate_runs() -> None:
    assert "Runs hold up to 2 repetitions" in concerns_for(stat(), stat())


def test_overlapping_cases_are_counted() -> None:
    assert overlapping_cases([stat(), stat(window_start=50, window_end=150)]) == 1
    assert overlapping_cases([stat(), stat(window_start=101, window_end=150)]) == 0


def test_cli_build_and_audit(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    config = project / "data.yaml"
    config.write_text("source: rcaeval\nsuites: [RE1-OB, RE1-SS]\n", encoding="utf-8")

    built = RUNNER.invoke(app, ["data", "build", "--config", str(config)])
    audited = RUNNER.invoke(app, ["data", "audit", "--config", str(config)])

    assert built.exit_code == 0, built.output
    assert audited.exit_code == 0, audited.output
    assert (project / "docs" / "data_audit.md").is_file()


def test_cli_fails_loudly_without_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "data.yaml"
    config.write_text("source: rcaeval\nsuites: [RE1-OB]\n", encoding="utf-8")

    built = RUNNER.invoke(app, ["data", "build", "--config", str(config)])
    audited = RUNNER.invoke(app, ["data", "audit", "--config", str(config)])

    assert built.exit_code == 1
    assert "does not exist" in built.output
    assert audited.exit_code == 1
