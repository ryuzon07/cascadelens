import json
from pathlib import Path

import polars as pl
import pytest
from feature_factory import config, make_case, shipped_config
from typer.testing import CliRunner

from cascadelens.cli import app
from cascadelens.config import config_hash
from cascadelens.data.io import MANIFEST_FILE, Manifest, read_manifest, write_cases, write_manifest
from cascadelens.features.checks import check_table
from cascadelens.features.context import FeatureError
from cascadelens.features.store import (
    CASES_DIR,
    CHECKS_FILE,
    METADATA_FILE,
    build_features,
    check_features,
    feature_dir,
)

RUNNER = CliRunner()
CONFIGS = Path(__file__).parents[2] / "configs"


def project(root: Path) -> None:
    cases = [
        make_case(f"{system}-{service}-{rep}", service, system)
        for system in ("shop", "bank")
        for service in ("db", "api", "web")
        for rep in range(2)
    ]
    processed = root / "data" / "processed"
    write_cases(cases, processed)
    write_manifest(Manifest(source="s", suites=(), files=()), processed)


def snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def test_shipped_config_is_valid() -> None:
    assert shipped_config(str(CONFIGS / "features.yaml")).signal.baseline_fraction == 0.25


def test_build_writes_tables_tied_to_the_dataset(tmp_path: Path) -> None:
    project(tmp_path)
    settings = config()

    summary = build_features(settings, tmp_path)

    directory = tmp_path / "data" / "features" / config_hash(settings)
    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    assert summary.directory == directory
    assert summary.cases == 12
    assert metadata["config_hash"] == config_hash(settings)
    assert (
        metadata["data_manifest_checksum"]
        == read_manifest(tmp_path / "data" / "processed").digest()
    )
    assert len(list((directory / CASES_DIR).glob("*.parquet"))) == 12
    assert not list(directory.rglob("*.partial"))


def test_building_twice_gives_identical_bytes(tmp_path: Path) -> None:
    project(tmp_path)
    settings = config()

    first = snapshot(build_features(settings, tmp_path).directory)
    second = snapshot(build_features(settings, tmp_path).directory)

    assert first == second


def test_worker_processes_give_identical_bytes(tmp_path: Path) -> None:
    project(tmp_path)
    settings = config()

    first = snapshot(build_features(settings, tmp_path, workers=1).directory)
    second = snapshot(build_features(settings, tmp_path, workers=2).directory)

    assert first == second


def test_group_choice_changes_the_directory(tmp_path: Path) -> None:
    assert feature_dir(config(), tmp_path) != feature_dir(config(groups=["signal"]), tmp_path)


def test_build_without_data_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(FeatureError, match="data build"):
        build_features(config(), tmp_path)


def rewrite_manifest(root: Path) -> None:
    processed = root / "data" / "processed"
    write_manifest(Manifest(source="other", suites=(), files=()), processed)
    assert (processed / MANIFEST_FILE).is_file()


def test_build_refuses_tables_of_another_dataset(tmp_path: Path) -> None:
    project(tmp_path)
    build_features(config(), tmp_path)
    rewrite_manifest(tmp_path)

    with pytest.raises(FeatureError, match="different dataset"):
        build_features(config(), tmp_path)


def test_check_refuses_another_dataset(tmp_path: Path) -> None:
    project(tmp_path)
    build_features(config(), tmp_path)
    rewrite_manifest(tmp_path)

    with pytest.raises(FeatureError, match="different dataset"):
        check_features(config(), tmp_path)


def test_check_finds_a_missing_table_and_a_partial_build(tmp_path: Path) -> None:
    project(tmp_path)
    directory = build_features(config(), tmp_path).directory
    next((directory / CASES_DIR).glob("*.parquet")).unlink()

    with pytest.raises(FeatureError, match="missing"):
        check_features(config(), tmp_path)

    (directory / METADATA_FILE).unlink()
    with pytest.raises(FeatureError, match="No complete build"):
        check_features(config(), tmp_path)


def test_check_writes_a_report(tmp_path: Path) -> None:
    project(tmp_path)
    directory = build_features(config(), tmp_path).directory

    report = check_features(config(), tmp_path)

    saved = json.loads((directory / CHECKS_FILE).read_text(encoding="utf-8"))
    assert saved["cases"] == report.cases == 12
    assert "structural_in_degree" in report.nan_share


def test_structural_columns_are_flagged_because_the_graph_is_fixed(tmp_path: Path) -> None:
    project(tmp_path)
    build_features(config(), tmp_path)

    report = check_features(config(), tmp_path)

    assert "structural_betweenness" in report.near_fingerprints["shop"]


def table(values: dict[str, list[float]], services: list[str], cases: list[str]) -> pl.DataFrame:
    return pl.DataFrame({"case_id": cases, "service": services, **values})


def repeated(column: list[float]) -> pl.DataFrame:
    return table(
        {"signal_x": column * 2},
        ["a", "b", "c", "d"] * 2,
        ["c1"] * 4 + ["c2"] * 4,
    )


def test_constant_and_all_nan_columns() -> None:
    frame = repeated([1.0, 1.0, 1.0, 1.0]).with_columns(signal_y=pl.lit(float("nan")))

    report = check_table(frame)

    assert report.constant == ["signal_x"]
    assert report.all_nan == ["signal_y"]
    assert report.nan_share["signal_y"] == 1
    assert report.nan_share["signal_x"] == 0


def test_row_order_column_is_flagged() -> None:
    assert check_table(repeated([1.0, 2.0, 3.0, 4.0])).row_order == ["signal_x"]


def test_shuffled_column_is_not_flagged_as_row_order() -> None:
    frame = table(
        {"signal_x": [3.0, 1.0, 4.0, 2.0, 2.0, 4.0, 1.0, 3.0]},
        ["a", "b", "c", "d"] * 2,
        ["c1"] * 4 + ["c2"] * 4,
    )

    report = check_table(frame)

    assert report.row_order == []
    assert report.exact_fingerprints == {}
    assert report.near_fingerprints == {}


def test_exact_fingerprint_needs_stability_and_uniqueness() -> None:
    stable = repeated([5.0, 2.0, 9.0, 1.0])
    shared = repeated([5.0, 5.0, 9.0, 1.0])

    assert check_table(stable).exact_fingerprints == {"all": ["signal_x"]}
    assert check_table(shared).exact_fingerprints == {}


def test_near_fingerprint_allows_small_noise() -> None:
    frame = table(
        {"signal_x": [5.0, 2.0, 9.0, 1.0, 5.001, 2.001, 9.001, 1.001]},
        ["a", "b", "c", "d"] * 2,
        ["c1"] * 4 + ["c2"] * 4,
    )

    report = check_table(frame)

    assert report.near_fingerprints == {"all": ["signal_x"]}
    assert report.exact_fingerprints == {}


def test_fingerprints_are_scoped_by_system() -> None:
    frame = repeated([5.0, 2.0, 9.0, 1.0]).with_columns(system=pl.lit("shop"))

    assert list(check_table(frame).exact_fingerprints) == ["shop"]


def test_cli_build_and_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project(tmp_path)
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "features.yaml"
    path.write_text("groups: [signal, mask]\n", encoding="utf-8")

    built = RUNNER.invoke(app, ["features", "build", "--config", str(path)])
    checked = RUNNER.invoke(app, ["features", "check", "--config", str(path)])

    assert built.exit_code == 0, built.output
    assert "12 case(s)" in built.output
    assert checked.exit_code == 0, checked.output


def test_cli_reports_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "features.yaml"
    path.write_text("groups: [signal]\n", encoding="utf-8")

    built = RUNNER.invoke(app, ["features", "build", "--config", str(path)])
    checked = RUNNER.invoke(app, ["features", "check", "--config", str(path)])

    assert built.exit_code == 1
    assert checked.exit_code == 1
    assert "features check failed" in checked.output
