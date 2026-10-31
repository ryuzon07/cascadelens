"""Feature tables on disk, tied to one config and one dataset version.

Layout under ``data/features/<config_hash>/``::

    cases/<case_id>.parquet    one table per case, written as soon as it is built
    metadata.json              config, data manifest checksum and columns, written last
    checks.json                the latest leakage self-check

Cases are processed one at a time, or in worker processes, and nothing is held in memory across
cases. A file only appears complete: each is written beside its target and renamed. No file records
when it was built, so the same inputs give identical bytes.
"""

import json
import os
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cascadelens.config import config_hash
from cascadelens.data.errors import DataError
from cascadelens.data.io import CASES_FILE, metrics_path, read_index, read_manifest
from cascadelens.data.schema import CaseRecord
from cascadelens.features.checks import CheckReport, check_table
from cascadelens.features.config import FeaturesConfig
from cascadelens.features.context import FeatureError, FeatureInput
from cascadelens.features.registry import build_case_features, feature_columns

CASES_DIR = "cases"
METADATA_FILE = "metadata.json"
CHECKS_FILE = "checks.json"


@dataclass(frozen=True)
class BuildSummary:
    """What a build wrote."""

    directory: Path
    cases: int
    columns: int
    bytes: int


def feature_dir(config: FeaturesConfig, root: Path = Path()) -> Path:
    """Return where this config's tables live."""
    return root / config.output_dir / config_hash(config)


def _dataset(config: FeaturesConfig, root: Path) -> tuple[list[CaseRecord], str]:
    processed = root / config.processed_dir
    if not (processed / CASES_FILE).is_file():
        raise FeatureError(f"No cases index in {processed}; run `cascadelens data build` first")
    try:
        return read_index(processed), read_manifest(processed).digest()
    except (DataError, FileNotFoundError) as error:
        raise FeatureError(f"Cannot read the dataset in {processed}: {error}") from error


def _write_json(payload: object, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _build_one(job: tuple[CaseRecord, str, str, FeaturesConfig]) -> str:
    """Build and write one case. Runs in a worker process, so it takes plain values."""
    record, metrics_file, target, config = job
    case = FeatureInput.from_record(record, pl.read_parquet(metrics_file))
    output = Path(target)
    partial = output.with_suffix(".partial")
    build_case_features(case, config).write_parquet(partial)
    os.replace(partial, output)
    return record.case_id


def _check_previous(directory: Path, checksum: str) -> None:
    path = directory / METADATA_FILE
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["data_manifest_checksum"] != checksum:
            raise FeatureError(f"{directory} holds tables of a different dataset; remove it first")


def build_features(config: FeaturesConfig, root: Path = Path(), workers: int = 1) -> BuildSummary:
    """Build every case and write the metadata last, so a partial build has none."""
    records, checksum = _dataset(config, root)
    directory = feature_dir(config, root)
    _check_previous(directory, checksum)
    tables = directory / CASES_DIR
    tables.mkdir(parents=True, exist_ok=True)
    (directory / METADATA_FILE).unlink(missing_ok=True)
    processed = root / config.processed_dir
    jobs = [
        (
            record,
            str(metrics_path(processed, record.case_id)),
            str(tables / f"{record.case_id}.parquet"),
            config,
        )
        for record in records
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            list(pool.map(_build_one, jobs, chunksize=4))
    else:
        for job in jobs:
            _build_one(job)
    columns = feature_columns(config.groups)
    _write_json(
        {
            "cases": len(jobs),
            "columns": columns,
            "config": config.model_dump(mode="json"),
            "config_hash": config_hash(config),
            "data_manifest_checksum": checksum,
        },
        directory / METADATA_FILE,
    )
    size = sum(path.stat().st_size for path in tables.glob("*.parquet"))
    return BuildSummary(directory, len(jobs), len(columns), size)


def read_tables(directory: Path, case_ids: Sequence[str]) -> Iterator[pl.DataFrame]:
    """Yield saved tables one at a time, in the order given."""
    for case_id in case_ids:
        yield pl.read_parquet(directory / CASES_DIR / f"{case_id}.parquet")


def check_features(config: FeaturesConfig, root: Path = Path()) -> CheckReport:
    """Verify a saved build against the dataset, run the self-check and save its report."""
    records, checksum = _dataset(config, root)
    directory = feature_dir(config, root)
    path = directory / METADATA_FILE
    if not path.is_file():
        raise FeatureError(f"No complete build in {directory}; run `cascadelens features build`")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if metadata["data_manifest_checksum"] != checksum:
        raise FeatureError(f"{directory} was built from a different dataset version")
    ids = [record.case_id for record in records]
    absent = [i for i in ids if not (directory / CASES_DIR / f"{i}.parquet").is_file()]
    if absent:
        raise FeatureError(f"{len(absent)} case table(s) are missing, such as {absent[0]}")
    expected = ["case_id", "service", *metadata["columns"]]
    frames = []
    for record, table in zip(records, read_tables(directory, ids), strict=True):
        if table.columns != expected:
            raise FeatureError(f"Case {record.case_id} has unexpected columns")
        frames.append(table.with_columns(pl.lit(record.system).alias("system")))
    report = check_table(pl.concat(frames))
    _write_json(report.to_dict(), directory / CHECKS_FILE)
    return report
