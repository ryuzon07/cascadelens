"""Lazy, content-addressed storage of perturbed cases.

A perturbed dataset lives under ``data/perturbed/<hash>/``, where the hash covers the steps, their
parameters, the seed and the dataset checksum::

    spec.json                  what the hash covers
    records/<case_id>.json     the case without its metrics, with its manifest rows
    metrics/<case_id>.parquet  absent when only topology changed: the processed file is used
    manifest.jsonl             the rows of the last ``apply_split``, sorted by case

A case is built the first time it is asked for, one at a time, and each file appears whole. Topology
steps store no metrics, which is what keeps the full grid small.
"""

import hashlib
import json
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from cascadelens.baselines.base import BaselineError
from cascadelens.baselines.runner import PROCESSED_DIR, load_dataset, load_split
from cascadelens.config import canonical_json
from cascadelens.data.io import metrics_path, read_case
from cascadelens.data.schema import Case, CaseRecord
from cascadelens.perturb.base import (
    ManifestRow,
    Perturbation,
    PerturbationError,
    Stage,
    metrics_bytes,
)
from cascadelens.perturb.pipeline import Pipeline
from cascadelens.perturb.registry import build

PERTURBED_DIR = Path("data/perturbed")
ENGINE_VERSION = 1
RECORDS_DIR = "records"
METRICS_DIR = "metrics"
SPEC_FILE = "spec.json"
MANIFEST_FILE = "manifest.jsonl"


@dataclass(frozen=True)
class Summary:
    """What an apply wrote."""

    directory: Path
    rows: list[ManifestRow]
    bytes: int


def spec_payload(steps: Sequence[Perturbation[Any]], seed: int, checksum: str) -> dict[str, Any]:
    """Return what identifies a perturbed dataset."""
    return {
        "data_checksum": checksum,
        "engine_version": ENGINE_VERSION,
        "seed": seed,
        "steps": [
            {"id": step.id, "parameters": step.parameters(), "realism": step.realism}
            for step in Pipeline(steps).steps
        ],
    }


def spec_hash(steps: Sequence[Perturbation[Any]], seed: int, checksum: str) -> str:
    """Return the name of a perturbed dataset's directory."""
    return hashlib.sha256(canonical_json(spec_payload(steps, seed, checksum)).encode()).hexdigest()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(f"{path.suffix}.{os.getpid()}.partial")
    partial.write_bytes(data)
    os.replace(partial, path)


class PerturbedStore:
    """One perturbed dataset, built on demand."""

    def __init__(
        self,
        steps: Sequence[Perturbation[Any]],
        seed: int,
        checksum: str,
        root: Path = Path(),
    ) -> None:
        self.pipeline = Pipeline(steps)
        self.seed = seed
        self.processed = root / PROCESSED_DIR
        self.directory = root / PERTURBED_DIR / spec_hash(steps, seed, checksum)
        self.payload = spec_payload(steps, seed, checksum)

    @property
    def keeps_metrics(self) -> bool:
        """Return whether the source metrics stay valid, so none need storing."""
        return all(step.stage == Stage.TOPOLOGY for step in self.pipeline.steps)

    def _record_path(self, case_id: str) -> Path:
        return self.directory / RECORDS_DIR / f"{case_id}.json"

    def write_spec(self) -> None:
        """Write the spec once. Workers find it already there, as ``apply_to`` writes it first."""
        path = self.directory / SPEC_FILE
        if not path.is_file():
            _write(path, (json.dumps(self.payload, indent=2, sort_keys=True) + "\n").encode())

    def ensure(self, record: CaseRecord) -> list[ManifestRow]:
        """Build a case if it is not stored, and return its manifest rows."""
        path = self._record_path(record.case_id)
        if not path.is_file():
            result = self.pipeline.run(read_case(record, self.processed), self.seed)
            if not self.keeps_metrics:
                target = metrics_path(self.directory, record.case_id)
                _write(target, metrics_bytes(result.case.metrics))
            payload = {
                "record": result.case.record().model_dump(mode="json"),
                "rows": [row.model_dump(mode="json") for row in result.rows],
            }
            self.write_spec()
            _write(path, canonical_json(payload).encode())
        stored = json.loads(path.read_text(encoding="utf-8"))
        return [ManifestRow.model_validate(row) for row in stored["rows"]]

    def case(self, record: CaseRecord) -> Case:
        """Return a perturbed case, building it if needed."""
        self.ensure(record)
        stored = json.loads(self._record_path(record.case_id).read_text(encoding="utf-8"))
        source = self.processed if self.keeps_metrics else self.directory
        metrics = pl.read_parquet(metrics_path(source, record.case_id))
        return Case(**CaseRecord.model_validate(stored["record"]).model_dump(), metrics=metrics)

    def size(self) -> int:
        """Return the bytes on disk."""
        return sum(p.stat().st_size for p in self.directory.rglob("*") if p.is_file())


def _ensure_one(job: tuple[str, dict[str, Any], int, str, Path, CaseRecord]) -> list[ManifestRow]:
    """Build one case. Runs in a worker process, so it takes plain values."""
    identifier, params, seed, checksum, root, record = job
    return PerturbedStore([build(identifier, params)], seed, checksum, root).ensure(record)


def apply_to(
    identifier: str,
    params: dict[str, Any],
    seed: int,
    case_ids: Sequence[str],
    root: Path = Path(),
    workers: int = 1,
) -> Summary:
    """Build a single perturbation over some cases, and write their manifest."""
    dataset = load_dataset(root)
    step = build(identifier, params)
    full = step.parameters()
    store = PerturbedStore([step], seed, dataset.checksum, root)
    store.write_spec()
    jobs = [
        (identifier, full, seed, dataset.checksum, root, dataset.records[case])
        for case in sorted(case_ids)
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            built = list(pool.map(_ensure_one, jobs, chunksize=4))
    else:
        built = [_ensure_one(job) for job in jobs]
    rows = [row for group in built for row in group]
    lines = "".join(row.model_dump_json() + "\n" for row in rows)
    _write(store.directory / MANIFEST_FILE, lines.encode())
    return Summary(store.directory, rows, store.size())


def apply_split(
    identifier: str,
    params: dict[str, Any],
    split_name: str,
    seed: int,
    root: Path = Path(),
    workers: int = 1,
) -> Summary:
    """Perturb the test cases of every fold of a split."""
    dataset = load_dataset(root)
    try:
        split, _ = load_split(split_name, dataset, root)
    except BaselineError as error:
        raise PerturbationError(str(error)) from error
    ids = sorted({case for fold in split.folds for case in fold.test_ids})
    return apply_to(identifier, params, seed, ids, root, workers)
