import json
from pathlib import Path

import pytest
from case_factory import FIXTURE_DIR, FIXTURE_SUITES, default_rows, make_case, metrics_frame

from cascadelens.data.errors import DataError
from cascadelens.data.io import (
    MANIFEST_FILE,
    build_manifest,
    changed_files,
    read_cases,
    read_index,
    read_manifest,
    sha256_file,
    write_cases,
    write_manifest,
)
from cascadelens.data.loaders.rcaeval import iter_cases
from cascadelens.data.schema import Case


def fixture_cases() -> list[Case]:
    cases = list(iter_cases(FIXTURE_DIR, FIXTURE_SUITES))
    assert all(isinstance(case, Case) for case in cases)
    return cases  # type: ignore[return-value]


def test_round_trip_is_identical(tmp_path: Path) -> None:
    rows = [*default_rows()[:-1], ("db", "mem", 120, None)]
    cases = [
        make_case(metrics=metrics_frame(rows), logs="logs.csv"),
        make_case(case_id="re1-ob_db_cpu_2", repetition=2, topology=None),
    ]

    write_cases(cases, tmp_path)
    loaded = list(read_cases(tmp_path))

    assert len(loaded) == 2
    assert all(original.equals(copy) for original, copy in zip(cases, loaded, strict=True))
    assert loaded[0].metrics["value"].null_count() == 1
    assert loaded[1].topology is None


def test_empty_topology_stays_distinct_from_missing(tmp_path: Path) -> None:
    write_cases([make_case(topology=[])], tmp_path)

    assert read_index(tmp_path)[0].topology == ()


def test_round_trip_on_fixture(tmp_path: Path) -> None:
    cases = fixture_cases()

    write_cases(cases, tmp_path)
    loaded = {case.case_id: case for case in read_cases(tmp_path)}

    assert len(loaded) == 10
    assert all(case.equals(loaded[case.case_id]) for case in cases)


def test_index_has_one_row_per_case_sorted(tmp_path: Path) -> None:
    write_cases([make_case(case_id="b"), make_case(case_id="a")], tmp_path)

    assert [record.case_id for record in read_index(tmp_path)] == ["a", "b"]


def test_output_is_byte_identical_between_runs(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    write_cases(fixture_cases(), first)
    write_cases(reversed(fixture_cases()), second)

    names = sorted(path.relative_to(first) for path in first.rglob("*.parquet"))
    assert names == sorted(path.relative_to(second) for path in second.rglob("*.parquet"))
    assert all((first / name).read_bytes() == (second / name).read_bytes() for name in names)


def test_checksum_matches_known_value(tmp_path: Path) -> None:
    path = tmp_path / "file"
    path.write_bytes(b"abc")

    assert sha256_file(path) == ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


def test_manifest_is_stable_and_complete() -> None:
    first = build_manifest("rcaeval", FIXTURE_DIR, {s: FIXTURE_DIR / s for s in FIXTURE_SUITES})
    reordered = build_manifest(
        "rcaeval", FIXTURE_DIR, {s: FIXTURE_DIR / s for s in reversed(FIXTURE_SUITES)}
    )

    assert first == reordered
    assert first.digest() == reordered.digest()
    assert len(first.files) == 20
    assert [entry.path for entry in first.files] == sorted(entry.path for entry in first.files)
    assert all("\\" not in entry.path for entry in first.files)


def test_manifest_order_is_case_sensitive_on_every_platform(tmp_path: Path) -> None:
    suite = tmp_path / "RE1-OB"
    suite.mkdir()
    for name in ("b.txt", "C.txt", "a.txt"):
        (suite / name).write_text(name, encoding="utf-8")

    manifest = build_manifest("rcaeval", tmp_path, {"RE1-OB": suite})

    assert [entry.path.rsplit("/")[-1] for entry in manifest.files] == ["C.txt", "a.txt", "b.txt"]


def test_manifest_digest_changes_with_any_file(tmp_path: Path) -> None:
    suite = tmp_path / "RE1-OB"
    suite.mkdir()
    (suite / "a.txt").write_text("one", encoding="utf-8")
    before = build_manifest("rcaeval", tmp_path, {"RE1-OB": suite})

    (suite / "a.txt").write_text("two", encoding="utf-8")
    after = build_manifest("rcaeval", tmp_path, {"RE1-OB": suite})

    assert before.digest() != after.digest()
    assert changed_files(before, tmp_path) == ["RE1-OB/a.txt"]
    assert changed_files(after, tmp_path) == []


def test_changed_files_reports_deleted_sources(tmp_path: Path) -> None:
    suite = tmp_path / "RE1-OB"
    suite.mkdir()
    (suite / "a.txt").write_text("one", encoding="utf-8")
    manifest = build_manifest("rcaeval", tmp_path, {"RE1-OB": suite})

    (suite / "a.txt").unlink()

    assert changed_files(manifest, tmp_path) == ["RE1-OB/a.txt"]


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = build_manifest("rcaeval", FIXTURE_DIR, {"RE1-OB": FIXTURE_DIR / "RE1-OB"})

    write_manifest(manifest, tmp_path)

    assert read_manifest(tmp_path) == manifest


def test_tampered_manifest_is_refused(tmp_path: Path) -> None:
    manifest = build_manifest("rcaeval", FIXTURE_DIR, {"RE1-OB": FIXTURE_DIR / "RE1-OB"})
    write_manifest(manifest, tmp_path)
    path = tmp_path / MANIFEST_FILE
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["files"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataError, match="digest"):
        read_manifest(tmp_path)
