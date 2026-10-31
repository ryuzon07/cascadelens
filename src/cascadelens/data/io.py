"""Parquet storage for cases and the checksum manifest.

Layout under the processed directory::

    cases.parquet               one row per case, sorted by case_id
    metrics/<case_id>.parquet   the long metrics table of that case
    manifest.json               SHA-256 of every source file
    config_hash.txt             hash of the data config used for the build

Output is byte-stable for the same input and library versions: rows are stored in a fixed order
and nothing records the time of the build.
"""

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import polars as pl
from polars.datatypes import DataType, DataTypeClass
from pydantic import Field

from cascadelens.config import StrictModel, canonical_json
from cascadelens.data.errors import DataError
from cascadelens.data.schema import Case, CaseRecord

CASES_FILE = "cases.parquet"
METRICS_DIR = "metrics"
MANIFEST_FILE = "manifest.json"
CHUNK_BYTES = 1024 * 1024

_NAME_MAPPING = pl.List(pl.Struct({"normalized": pl.String, "original": pl.String}))
_EDGES = pl.List(pl.Struct({"caller": pl.String, "callee": pl.String}))
_CASES_COLUMNS: dict[str, DataType | DataTypeClass] = {
    "case_id": pl.String,
    "suite": pl.String,
    "system": pl.String,
    "fault_type": pl.String,
    "run_id": pl.String,
    "repetition": pl.Int64,
    "root_cause_service": pl.String,
    "inject_time": pl.Int64,
    "window_start": pl.Int64,
    "window_end": pl.Int64,
    "service_names": _NAME_MAPPING,
    "topology": _EDGES,
    "logs": pl.String,
    "traces": pl.String,
}
CASES_SCHEMA = pl.Schema(_CASES_COLUMNS)


def metrics_path(root: Path, case_id: str) -> Path:
    """Return where the metrics of ``case_id`` are stored."""
    return root / METRICS_DIR / f"{case_id}.parquet"


def write_metrics(case: Case, root: Path) -> None:
    """Write the metrics of one case."""
    target = metrics_path(root, case.case_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    case.metrics.write_parquet(target)


def _to_row(record: CaseRecord) -> dict[str, object]:
    row = record.model_dump(mode="python")
    row["service_names"] = [
        {"normalized": normalized, "original": original}
        for normalized, original in sorted(record.service_names.items())
    ]
    return row


def _from_row(row: dict[str, object]) -> CaseRecord:
    names = row["service_names"]
    assert isinstance(names, list)
    fields = {**row, "service_names": {item["normalized"]: item["original"] for item in names}}
    return CaseRecord.model_validate(fields)


def write_index(records: Iterable[CaseRecord], root: Path) -> None:
    """Write the cases index."""
    rows = [_to_row(record) for record in sorted(records, key=lambda record: record.case_id)]
    root.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, schema=CASES_SCHEMA).write_parquet(root / CASES_FILE)


def read_index(root: Path) -> list[CaseRecord]:
    """Read the cases index."""
    frame = pl.read_parquet(root / CASES_FILE)
    return [_from_row(row) for row in frame.iter_rows(named=True)]


def read_case(record: CaseRecord, root: Path) -> Case:
    """Join an index row with its metrics."""
    metrics = pl.read_parquet(metrics_path(root, record.case_id))
    return Case(**record.model_dump(), metrics=metrics)


def read_cases(root: Path) -> Iterator[Case]:
    """Yield every stored case, one at a time, in case_id order."""
    for record in read_index(root):
        yield read_case(record, root)


def write_cases(cases: Iterable[Case], root: Path) -> None:
    """Write cases and their index."""
    records = []
    for case in cases:
        write_metrics(case, root)
        records.append(case.record())
    write_index(records, root)


class SourceFile(StrictModel):
    """Checksum of one source file."""

    path: str
    size: int = Field(ge=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class Manifest(StrictModel):
    """Checksums of the source files, with paths relative to the raw directory."""

    source: str
    suites: tuple[str, ...]
    files: tuple[SourceFile, ...]

    def digest(self) -> str:
        """Return one checksum for the whole dataset, for ``write_result``."""
        payload = canonical_json(self.model_dump(mode="json"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(source: str, raw_dir: Path, suite_dirs: dict[str, Path]) -> Manifest:
    """Checksum every file under the given suite directories."""
    # Sorted as text: Path ordering ignores case on Windows, which would change the digest.
    files = sorted(
        (
            path
            for suite_dir in suite_dirs.values()
            for path in suite_dir.rglob("*")
            if path.is_file()
        ),
        key=lambda path: path.relative_to(raw_dir).as_posix(),
    )
    return Manifest(
        source=source,
        suites=tuple(sorted(suite_dirs)),
        files=tuple(
            SourceFile(
                path=path.relative_to(raw_dir).as_posix(),
                size=path.stat().st_size,
                sha256=sha256_file(path),
            )
            for path in files
        ),
    )


def write_manifest(manifest: Manifest, root: Path) -> None:
    """Write the manifest as stable, readable JSON."""
    payload = {"digest": manifest.digest(), **manifest.model_dump(mode="json")}
    root.mkdir(parents=True, exist_ok=True)
    with (root / MANIFEST_FILE).open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def read_manifest(root: Path) -> Manifest:
    """Read a manifest, refusing one whose digest does not match its contents."""
    payload = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    digest = payload.pop("digest")
    manifest = Manifest.model_validate(payload)
    if manifest.digest() != digest:
        raise DataError(f"Manifest {root / MANIFEST_FILE} does not match its digest")
    return manifest


def changed_files(manifest: Manifest, raw_dir: Path) -> Sequence[str]:
    """Return the paths of source files that are missing or no longer match the manifest."""
    return [
        entry.path
        for entry in manifest.files
        if not (raw_dir / entry.path).is_file() or sha256_file(raw_dir / entry.path) != entry.sha256
    ]
