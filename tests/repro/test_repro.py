import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from cascadelens.models.diagnose import DISCLAIMER
from cascadelens.report.build import (
    DOCUMENTS,
    HEADLINE_PATH,
    PROVENANCE_PATH,
    TEMPLATES,
    build,
    check,
    generate,
    render,
    verify,
)
from cascadelens.report.sources import REPRODUCTION_LOG, SOURCES, ReportError, Store
from cascadelens.report.tables import MARKER, Table, all_tables, full_run, headline
from cascadelens.report.values import Value, all_values, s6_values

REPO = Path(__file__).parents[2]
GENERATED_BLOCK = re.compile(r"<!-- generated:(\w+);.*?-->(.*?)<!-- /generated -->", re.DOTALL)
PROTOCOL_DECIMALS = {"2405.09330", "2408.13729", "1.11.0", "0.05", "1.00", "3.11"}


def script(name: str) -> ModuleType:
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    """Return a copy of the result JSONs with S1 changed, so a read value is visible."""
    for relative in SOURCES.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    path = tmp_path / SOURCES["robustness"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["metrics"]["hypotheses"]["ranker"]["S1"] |= {"estimate": 0.5, "lower": 0.4, "upper": 0.6}
    path.write_text(json.dumps(raw), encoding="utf-8")
    (tmp_path / "docs").mkdir()
    return tmp_path


def test_generators_read_the_fixture_result_files(fixture_root: Path) -> None:
    rows = headline(Store(fixture_root)).rows
    assert rows[0][0] == "S1"
    assert rows[0][2] == "50.0 [40.0, 60.0]"


def test_build_writes_tables_figures_and_provenance(fixture_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    written = build(fixture_root, out)
    assert written
    assert verify(out) == []
    assert (out / "docs/results/figures/leakage.png").read_bytes().startswith(b"\x89PNG")
    assert "50.0 [40.0, 60.0]" in (out / "README.md").read_text(encoding="utf-8")


def test_generation_is_deterministic(fixture_root: Path) -> None:
    assert generate(fixture_root).files == generate(fixture_root).files


def test_verify_reports_a_missing_criterion(fixture_root: Path, tmp_path: Path) -> None:
    out = tmp_path / "out"
    build(fixture_root, out)
    path = out / HEADLINE_PATH
    path.write_text(path.read_text(encoding="utf-8").replace("| S4", "| S9"), encoding="utf-8")
    assert verify(out) == ["Headline table lacks S4"]


def test_a_missing_result_file_is_a_clear_error(fixture_root: Path) -> None:
    (fixture_root / SOURCES["calibration"]).unlink()
    with pytest.raises(ReportError, match="make all"):
        all_tables(Store(fixture_root))


def test_an_unknown_placeholder_is_an_error() -> None:
    with pytest.raises(ReportError, match="Unknown value"):
        render("{{nothing}}", {}, {})


def test_render_fills_tables_and_values() -> None:
    table = Table("t", "T", ("A",), [("1",)], ("baro",))
    text = render("{{v}} {{table:t}}", {"t": table}, {"v": Value("x", "baro")})
    assert text.startswith("x <!-- generated:t;")


def test_committed_documents_match_the_result_jsons() -> None:
    assert check(REPO) == []


def test_number_sync_catches_an_edited_readme(tmp_path: Path) -> None:
    for relative in SOURCES.values():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / relative, target)
    (tmp_path / "docs").mkdir()
    build(tmp_path)
    path = tmp_path / "README.md"
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("68.7 [58.8, 77.9]", "99.9 [99.0, 100.0]"), encoding="utf-8")
    assert check(tmp_path) == ["README.md"]


@pytest.mark.parametrize("name", ["README.md", "docs/REPORT.md", "MODEL_CARD.md"])
def test_every_table_in_a_document_has_provenance(name: str) -> None:
    text = (REPO / name).read_text(encoding="utf-8")
    trace = (REPO / PROVENANCE_PATH).read_text(encoding="utf-8")
    for table, _ in GENERATED_BLOCK.findall(text):
        assert f"| {table} |" in trace, table
    outside = GENERATED_BLOCK.sub("", text)
    rows = [line for line in outside.splitlines() if line.startswith("|")]
    assert not [line for line in rows if re.search(r"\d", line)]


@pytest.mark.parametrize("template", sorted(DOCUMENTS))
def test_templates_type_no_result_by_hand(template: str) -> None:
    text = (TEMPLATES / template).read_text(encoding="utf-8")
    decimals = set(re.findall(r"\d+\.\d+(?:\.\d+)?", text)) - PROTOCOL_DECIMALS
    assert not decimals


def test_every_marker_is_listed_in_the_claims_file() -> None:
    claims = (REPO / "docs/CLAIMS_TODO.md").read_text(encoding="utf-8")
    paths = [*sorted((REPO / "docs").rglob("*.md")), REPO / "README.md"]
    for path in paths:
        relative = path.relative_to(REPO).as_posix()
        if relative == "docs/CLAIMS_TODO.md":
            continue
        for found in MARKER.findall(path.read_text(encoding="utf-8")):
            assert f"`{relative}` | `{found}`" in claims


def test_documents_carry_the_disclaimer_and_avoid_overclaims() -> None:
    for name in ("README.md", "docs/REPORT.md", "MODEL_CARD.md"):
        text = (REPO / name).read_text(encoding="utf-8")
        assert DISCLAIMER in text
        assert "production-ready" not in text.lower()


def test_perturbation_rows_are_labelled() -> None:
    table = (REPO / "docs/results/tables/robustness_full.md").read_text(encoding="utf-8")
    for line in table.splitlines()[4:]:
        assert line.startswith(("| clean", "| P")), line
        assert line.split("|")[2].strip() in ("clean", "simulated")


def test_s6_needs_a_full_run_that_passed(tmp_path: Path) -> None:
    path = tmp_path / REPRODUCTION_LOG
    path.parent.mkdir(parents=True)
    header = "| Date | Kind | Commit | Machine | Runtime (s) | Outcome |\n"
    path.write_text(header + "| 2026-11-01 | smoke | `a` | x | 1 | pass |\n", encoding="utf-8")
    assert full_run(tmp_path) is None
    path.write_text(header + "| 2026-11-01 | full | `a` | x | 1 | fail |\n", encoding="utf-8")
    assert full_run(tmp_path) is None
    path.write_text(header + "| 2026-11-01 | full | `a` | x | 1 | pass |\n", encoding="utf-8")
    assert full_run(tmp_path) == "2026-11-01 full"


def test_s6_prose_follows_the_log(fixture_root: Path) -> None:
    path = fixture_root / REPRODUCTION_LOG
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| Date | Kind | Commit | Machine | Runtime (s) | Outcome |\n"
    path.write_text(header, encoding="utf-8")
    assert "not demonstrated" in s6_values(Store(fixture_root))["s6_limitation"].text
    path.write_text(header + "| 2026-11-01 | full | `a` | x | 1 | pass |\n", encoding="utf-8")
    values = s6_values(Store(fixture_root))
    assert "not demonstrated" not in values["s6_limitation"].text
    assert values["s6_todo"].text.startswith("- [x]")


def test_no_template_states_the_s6_verdict_by_hand() -> None:
    for template in DOCUMENTS:
        text = (TEMPLATES / template).read_text(encoding="utf-8")
        assert "not demonstrated" not in text
        assert "not been demonstrated" not in text


def test_headline_says_s6_is_not_demonstrated_without_a_logged_run() -> None:
    if full_run(REPO) is None:
        text = (REPO / HEADLINE_PATH).read_text(encoding="utf-8")
        assert "not demonstrated" in text


def test_prose_values_are_all_traced() -> None:
    trace = (REPO / PROVENANCE_PATH).read_text(encoding="utf-8")
    for name in all_values(Store(REPO)):
        assert f"| {name} |" in trace


def test_stages_run_once_and_rerun_when_a_config_changes(tmp_path: Path) -> None:
    runner = script("run_stages")
    (tmp_path / "configs").mkdir()
    config = tmp_path / "configs" / "a.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    pipeline = [
        runner.Stage("one", (("first",),)),
        runner.Stage("two", (("second",),)),
    ]
    calls: list[tuple[str, ...]] = []

    def record(command: tuple[str, ...], root: Path) -> None:
        calls.append(command)

    assert runner.run_stages(tmp_path, pipeline, record) == ["one", "two"]
    assert runner.run_stages(tmp_path, pipeline, record) == []
    config.write_text("x: 2\n", encoding="utf-8")
    assert runner.run_stages(tmp_path, pipeline, record) == ["one", "two"]
    assert len(calls) == 4


def test_a_failed_stage_resumes_there(tmp_path: Path) -> None:
    runner = script("run_stages")
    pipeline = [runner.Stage("one", (("first",),)), runner.Stage("two", (("second",),))]

    def fail_second(command: tuple[str, ...], root: Path) -> None:
        if command == ("second",):
            raise RuntimeError

    with pytest.raises(RuntimeError):
        runner.run_stages(tmp_path, pipeline, fail_second)
    assert runner.run_stages(tmp_path, pipeline, lambda command, root: None) == ["two"]


def test_freezing_changes_the_ranker_stage_fingerprint(tmp_path: Path) -> None:
    runner = script("run_stages")
    named = {stage.name: stage for stage in runner.stages(True, True)}
    frozen = {stage.name: stage for stage in runner.stages(False, True)}
    assert runner.fingerprint(tmp_path, named["ranker"], "") != runner.fingerprint(
        tmp_path, frozen["ranker"], ""
    )


def test_pipeline_runs_in_the_required_order() -> None:
    names = [stage.name for stage in script("run_stages").stages(True, True)]
    assert names == [
        "data",
        "verify-data",
        "splits",
        "leakage",
        "baselines",
        "published",
        "features",
        "ranker",
        "perturbations",
        "robustness",
        "probes",
        "calibration",
        "report",
    ]


def test_environment_check_explains_where_data_goes(tmp_path: Path) -> None:
    problems = script("check_env").data_problems(tmp_path)
    assert "data/raw/RE1-OB" in problems[0]
    assert "accept" in problems[0]


def test_nothing_downloads_unless_asked() -> None:
    text = (REPO / "scripts/reproduce_all.sh").read_text(encoding="utf-8")
    assert 'if [ "$download" = "1" ]' in text
    assert "download_data" not in (REPO / "scripts/run_stages.py").read_text(encoding="utf-8")


def test_base_image_is_pinned_and_runs_as_non_root() -> None:
    text = (REPO / "Dockerfile").read_text(encoding="utf-8")
    assert "@sha256:" in text
    assert "USER cascadelens" in text
    assert "make" in text


def test_smoke_finishes_inside_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    smoke = script("smoke")
    started = time.perf_counter()
    smoke.main()
    assert time.perf_counter() - started < 300


def test_a_full_run_compares_tables_with_the_committed_ones() -> None:
    text = (REPO / "scripts/clean_clone_check.sh").read_text(encoding="utf-8")
    assert "git diff --exit-code" in text
    assert "docs/results/headline_table.md" in text
