import importlib.util
import shutil
from pathlib import Path

import polars as pl
import pytest
from case_factory import FIXTURE_DIR, FIXTURE_SUITES

from cascadelens.data.errors import CaseLoadError, Issue, RawLayoutError
from cascadelens.data.loaders.rcaeval import (
    CaseSource,
    discover_cases,
    find_suite_dir,
    iter_cases,
    load_case,
    load_case_with_notes,
    system_of,
)
from cascadelens.data.schema import Case
from cascadelens.data.validate import validate_cases

SCRIPT = Path(__file__).parents[2] / "scripts" / "make_fixture.py"


def load_all() -> list[Case]:
    items = list(iter_cases(FIXTURE_DIR, FIXTURE_SUITES))
    assert not [item for item in items if isinstance(item, Issue)]
    return [item for item in items if isinstance(item, Case)]


def write_raw(
    root: Path, csv: str = "time,a_cpu\n1,0.5\n2,0.6\n", inject: str | None = "1\n"
) -> CaseSource:
    path = root / "RE1-OB" / "a_cpu" / "1"
    path.mkdir(parents=True)
    (path / "data.csv").write_text(csv, encoding="utf-8")
    if inject is not None:
        (path / "inject_time.txt").write_text(inject, encoding="utf-8")
    return CaseSource("RE1-OB", "a", "cpu", 1, path)


def test_fixture_has_ten_cases_on_two_systems() -> None:
    cases = load_all()

    assert len(cases) == 10
    assert {case.system for case in cases} == {"online-boutique", "sock-shop"}


def test_fixture_passes_validation_with_only_topology_warnings() -> None:
    issues = validate_cases(load_all())

    assert len(issues) == 10
    assert {issue.check for issue in issues} == {"topology_coverage"}


def test_labels_and_ids_come_from_the_path() -> None:
    case = next(c for c in load_all() if c.case_id == "re1-ss_front-end_loss_1")

    assert case.suite == "RE1-SS"
    assert case.system == "sock-shop"
    assert case.root_cause_service == "front-end"
    assert case.fault_type == "loss"
    assert case.repetition == 1


def test_repetitions_of_a_fault_share_one_run_id() -> None:
    runs = {case.case_id: case.run_id for case in load_all()}

    assert (
        runs["re1-ob_adservice_cpu_1"] == runs["re1-ob_adservice_cpu_2"] == "RE1-OB/adservice_cpu"
    )
    assert runs["re1-ob_adservice_mem_1"] != runs["re1-ob_adservice_cpu_1"]


def test_run_id_differs_across_suites(tmp_path: Path) -> None:
    source = write_raw(tmp_path)
    other = tmp_path / "RE2-OB"
    shutil.copytree(tmp_path / "RE1-OB", other)
    copy = CaseSource("RE2-OB", "a", "cpu", 1, other / "a_cpu" / "1")

    assert load_case(source).run_id != load_case(copy).run_id


def test_window_is_the_recording_extent_and_ignores_inject_time(tmp_path: Path) -> None:
    source = write_raw(tmp_path / "x", "time,a_cpu\n10,0.5\n20,0.6\n30,0.7\n", "15\n")
    moved = write_raw(tmp_path / "y", "time,a_cpu\n10,0.5\n20,0.6\n30,0.7\n", "25\n")

    first, second = load_case(source), load_case(moved)

    assert (first.window_start, first.window_end) == (10, 30)
    assert (second.window_start, second.window_end) == (10, 30)
    assert first.inject_time != second.inject_time


def test_series_keep_file_order_so_validation_can_see_disorder(tmp_path: Path) -> None:
    source = write_raw(tmp_path, "time,a_cpu\n2,0.5\n1,0.6\n", "1\n")

    assert load_case(source).metrics["timestamp"].to_list() == [2, 1]


def test_missing_cells_and_nan_become_null(tmp_path: Path) -> None:
    source = write_raw(tmp_path, "time,a_cpu\n1,\n2,nan\n3,0.5\n")

    assert load_case(source).metrics["value"].to_list() == [None, None, 0.5]


def test_repeated_time_column_is_dropped() -> None:
    case = next(c for c in load_all() if c.case_id == "re1-ob_adservice_cpu_2")

    assert "time" not in case.metrics["service"].to_list()
    assert case.metrics.select("service", "metric").unique().height == 9


def test_repeated_time_in_another_number_format_is_the_same_instant(tmp_path: Path) -> None:
    source = write_raw(tmp_path, "time,a_cpu,time,b_cpu\n1,0.5,1.0,0.7\n2,0.6,2.0,0.8\n")

    case, notes = load_case_with_notes(source)

    assert notes == []
    assert sorted(case.metrics["service"].unique().to_list()) == ["a", "b"]


def test_repeated_time_on_another_clock_drops_its_columns(tmp_path: Path) -> None:
    csv = "time,a_cpu,time,b_cpu,c_cpu\n1,0.5,100,0.7,0.9\n2,0.6,101,0.8,1.0\n"

    case, notes = load_case_with_notes(write_raw(tmp_path, csv))

    assert case.metrics["service"].unique().to_list() == ["a"]
    assert [(note.severity, note.check) for note in notes] == [("warning", "load_repair")] * 2
    assert "disagrees" in notes[0].reason
    assert "dropped 2 column(s)" in notes[1].reason


def test_columns_after_a_matching_repeated_time_are_kept(tmp_path: Path) -> None:
    csv = "time,a_cpu,time,b_cpu,time,c_cpu\n1,0.5,1,0.7,9,0.9\n2,0.6,2,0.8,9,1.0\n"

    case, _ = load_case_with_notes(write_raw(tmp_path, csv))

    assert sorted(case.metrics["service"].unique().to_list()) == ["a", "b"]


def test_untimed_tail_rows_are_dropped_and_reported(tmp_path: Path) -> None:
    source = write_raw(tmp_path, "time,a_cpu,b_cpu\n1,0.5,\n2,0.6,\n,,0.9\n,,\n")

    case, notes = load_case_with_notes(source)

    assert sorted(case.metrics["timestamp"].unique().to_list()) == [1, 2]
    assert (case.window_start, case.window_end) == (1, 2)
    assert [note.reason for note in notes] == [
        "dropped 2 trailing row(s) with no timestamp, 1 of them holding values"
    ]
    assert notes[0].case_id == case.case_id


def test_untimed_row_before_the_end_fails(tmp_path: Path) -> None:
    source = write_raw(tmp_path, "time,a_cpu\n1,0.5\n,0.6\n3,0.7\n")

    with pytest.raises(CaseLoadError, match="before the last timestamped row"):
        load_case(source)


def test_repairs_are_yielded_before_their_case(tmp_path: Path) -> None:
    write_raw(tmp_path, "time,a_cpu\n1,0.5\n,0.6\n")

    items = list(iter_cases(tmp_path, ["RE1-OB"]))

    assert [type(item) for item in items] == [Issue, Case]
    assert isinstance(items[0], Issue)
    assert items[0].severity == "warning"


def test_service_names_are_normalized_with_a_mapping(tmp_path: Path) -> None:
    root = tmp_path / "RE1-OB" / "Front_End_cpu" / "1"
    root.mkdir(parents=True)
    (root / "data.csv").write_text("time,Cart-Svc_cpu\n1,0.5\n", encoding="utf-8")
    (root / "inject_time.txt").write_text("1\n", encoding="utf-8")

    case = load_case(CaseSource("RE1-OB", "Front_End", "cpu", 1, root))

    assert case.root_cause_service == "front-end"
    assert case.service_names == {"cart-svc": "Cart-Svc", "front-end": "Front_End"}
    assert case.metrics["service"].to_list() == ["cart-svc"]


def test_metric_name_keeps_everything_after_the_first_underscore(tmp_path: Path) -> None:
    case = load_case(write_raw(tmp_path, "time,a_lat_90\n1,0.5\n"))

    assert case.metrics.row(0) == ("a", "lat_90", 1, 0.5)


def test_colliding_service_names_fail(tmp_path: Path) -> None:
    source = write_raw(tmp_path, "time,Foo_cpu,foo_mem\n1,0.5,0.6\n")

    with pytest.raises(CaseLoadError, match="both normalize"):
        load_case(source)


def test_simple_metrics_is_the_fallback_file(tmp_path: Path) -> None:
    source = write_raw(tmp_path)
    (source.path / "data.csv").rename(source.path / "simple_metrics.csv")

    assert load_case(source).metrics.height == 2


def test_data_csv_wins_when_both_files_exist(tmp_path: Path) -> None:
    source = write_raw(tmp_path)
    (source.path / "simple_metrics.csv").write_text("time,b_cpu\n1,0.5\n", encoding="utf-8")

    assert load_case(source).metrics["service"].unique().to_list() == ["a"]


@pytest.mark.parametrize(
    ("csv", "message"),
    [
        ("a_cpu\n0.5\n", "no 'time' column"),
        ("time,a_cpu\n1,oops\n", "non-numeric"),
        ("time,a_cpu\n1.5,0.5\n", "fractional"),
        ("time,a_cpu\n1700000000000,0.5\n", "not in Unix seconds"),
        ("time,a_cpu\n", "no rows"),
        ("time\n1\n", "no metric columns"),
        ("time,acpu\n1,0.5\n", "not <service>_<metric>"),
        ("time,,a_cpu\n1,0,0.5\n", "unnamed column"),
        ("time,a_cpu,a_cpu\n1,0.5,0.6\n", "duplicate metric column"),
    ],
    ids=lambda value: value.replace("\n", " ")[:24],
)
def test_bad_metric_files_fail(tmp_path: Path, csv: str, message: str) -> None:
    with pytest.raises(CaseLoadError, match=message):
        load_case(write_raw(tmp_path, csv))


@pytest.mark.parametrize("inject", [None, "", "soon\n"], ids=["missing", "empty", "text"])
def test_bad_inject_time_fails(tmp_path: Path, inject: str | None) -> None:
    with pytest.raises(CaseLoadError, match=r"inject_time\.txt"):
        load_case(write_raw(tmp_path, inject=inject))


def test_a_bad_case_becomes_an_issue_and_the_rest_still_load(tmp_path: Path) -> None:
    write_raw(tmp_path)
    bad = tmp_path / "RE1-OB" / "b_cpu" / "1"
    bad.mkdir(parents=True)

    items = list(iter_cases(tmp_path, ["RE1-OB"]))

    assert sum(isinstance(item, Case) for item in items) == 1
    issues = [item for item in items if isinstance(item, Issue)]
    assert [issue.case_id for issue in issues] == ["RE1-OB/b_cpu/1"]


def test_a_name_rejected_by_the_schema_becomes_an_issue(tmp_path: Path) -> None:
    path = tmp_path / "RE1-OB" / "bad!name_cpu" / "1"
    path.mkdir(parents=True)
    (path / "data.csv").write_text("time,a_cpu\n1,0.5\n", encoding="utf-8")
    (path / "inject_time.txt").write_text("1\n", encoding="utf-8")

    items = list(iter_cases(tmp_path, ["RE1-OB"]))

    assert [type(item) for item in items] == [Issue]


def test_repetitions_sort_numerically(tmp_path: Path) -> None:
    for repetition in ("10", "2"):
        path = tmp_path / "RE1-OB" / "a_cpu" / repetition
        path.mkdir(parents=True)

    sources, _ = discover_cases(tmp_path / "RE1-OB", "RE1-OB")

    assert [source.repetition for source in sources] == [2, 10]


def test_readme_style_flat_layout_is_refused(tmp_path: Path) -> None:
    case = tmp_path / "RE1-OB" / "re1ob_adservice_cpu_1"
    case.mkdir(parents=True)
    (case / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RawLayoutError, match="data audit"):
        discover_cases(tmp_path / "RE1-OB", "RE1-OB")


def test_stray_directories_are_reported(tmp_path: Path) -> None:
    write_raw(tmp_path)
    (tmp_path / "RE1-OB" / "notes").mkdir()

    _, issues = discover_cases(tmp_path / "RE1-OB", "RE1-OB")

    assert ["not <service>_<fault>" in issue.reason for issue in issues] == [True]


def test_missing_raw_dir_is_refused(tmp_path: Path) -> None:
    with pytest.raises(RawLayoutError, match="does not exist"):
        find_suite_dir(tmp_path / "absent", "RE1-OB")


def test_suite_may_sit_one_level_down(tmp_path: Path) -> None:
    (tmp_path / "RE1" / "RE1-OB").mkdir(parents=True)

    assert find_suite_dir(tmp_path, "RE1-OB") == tmp_path / "RE1" / "RE1-OB"


def test_ambiguous_suite_location_is_refused(tmp_path: Path) -> None:
    (tmp_path / "RE1-OB").mkdir()
    (tmp_path / "copy" / "RE1-OB").mkdir(parents=True)

    with pytest.raises(RawLayoutError, match="Expected one directory"):
        find_suite_dir(tmp_path, "RE1-OB")


def test_unknown_suite_is_refused() -> None:
    with pytest.raises(RawLayoutError, match="Unknown suite"):
        system_of("RE1-XX")


def test_loading_is_deterministic() -> None:
    first, second = load_all(), load_all()

    assert [case.case_id for case in first] == sorted(case.case_id for case in first)
    assert all(a.equals(b) for a, b in zip(first, second, strict=True))


def test_metrics_table_has_the_long_schema() -> None:
    case = load_all()[0]

    assert case.metrics.columns == ["service", "metric", "timestamp", "value"]
    assert case.metrics.schema["timestamp"] == pl.Int64


def test_committed_fixture_matches_the_generator(tmp_path: Path) -> None:
    spec = importlib.util.spec_from_file_location("make_fixture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    module.generate(tmp_path / "fresh")

    fresh = sorted(path for path in (tmp_path / "fresh").rglob("*") if path.is_file())
    assert [path.relative_to(tmp_path / "fresh") for path in fresh] == [
        path.relative_to(FIXTURE_DIR) for path in sorted(FIXTURE_DIR.rglob("*")) if path.is_file()
    ]
    for path in fresh:
        committed = FIXTURE_DIR / path.relative_to(tmp_path / "fresh")
        assert committed.read_bytes().replace(b"\r\n", b"\n") == path.read_bytes()
