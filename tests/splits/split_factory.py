"""Generated datasets for split tests, including one with planted near-duplicates."""

import shutil
import zlib
from pathlib import Path

import numpy as np
import polars as pl

from cascadelens.config import DataConfig, SplitsConfig
from cascadelens.data.io import Manifest, write_cases, write_manifest
from cascadelens.data.pipeline import build
from cascadelens.data.schema import METRICS_SCHEMA, Case, CaseRecord

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "rcaeval"
FIXTURE_SUITES = ["RE1-OB", "RE1-SS"]
SAMPLES = 200
SYSTEMS = ("alpha", "beta", "gamma")
SERVICES = ("api", "cart", "db")
FAULTS = ("cpu", "delay", "mem", "loss")
REPLAY_SUFFIX = "_replay"


def make_record(
    case_id: str,
    run_id: str,
    system: str = "alpha",
    fault_type: str = "cpu",
    root_cause_service: str = "api",
) -> CaseRecord:
    return CaseRecord(
        case_id=case_id,
        suite="suite",
        system=system,
        fault_type=fault_type,
        run_id=run_id,
        repetition=1,
        root_cause_service=root_cause_service,
        inject_time=SAMPLES // 2,
        window_start=0,
        window_end=SAMPLES - 1,
    )


def index_records(reps: int = 2) -> list[CaseRecord]:
    """Return records for every (system, service, fault), with ``reps`` cases per run."""
    return [
        make_record(
            f"{system}_{service}_{fault}_{rep}",
            f"{system}/{service}_{fault}",
            system,
            fault,
            service,
        )
        for system in SYSTEMS
        for service in SERVICES
        for fault in FAULTS
        for rep in range(reps)
    ]


def identity_groups(records: list[CaseRecord]) -> dict[str, str]:
    return {record.case_id: record.run_id for record in records}


def _metrics(run_seed: int, noise_seed: int) -> pl.DataFrame:
    base = np.random.default_rng(run_seed).normal(size=SAMPLES)
    jitter = np.random.default_rng(noise_seed).normal(scale=0.01, size=SAMPLES)
    rows = [
        (service, metric, step, float(scale * (base[step] + jitter[step])))
        for service, metric, scale in (("api", "cpu", 1.0), ("db", "mem", 100.0))
        for step in range(SAMPLES)
    ]
    return pl.DataFrame(rows, schema=METRICS_SCHEMA, orient="row")


def near_duplicate_dataset(root: Path) -> tuple[list[CaseRecord], str]:
    """Write a dataset where one run was recorded twice under different run_ids.

    Returns the records and the run that has a replay. Every run's series come from its own seed,
    so unrelated runs do not correlate. The replay copies ``alpha/api_cpu`` but gets another
    run_id. The run ``beta/api_cpu`` also copies it, yet differs in system, so it must not merge.
    """
    replayed = "alpha/api_cpu"
    records = index_records()
    records += [
        make_record(
            f"alpha_api_cpu_replay_{rep}", f"{replayed}{REPLAY_SUFFIX}", "alpha", "cpu", "api"
        )
        for rep in range(2)
    ]
    cases = []
    for number, record in enumerate(records):
        source = record.run_id.removesuffix(REPLAY_SUFFIX)
        if record.run_id == "beta/api_cpu":
            source = replayed
        metrics = _metrics(zlib.crc32(source.encode()), number + 1)
        cases.append(Case(**record.model_dump(), metrics=metrics))
    write_cases(cases, root)
    write_manifest(Manifest(source="generated", suites=("suite",), files=()), root)
    return records, replayed


def build_fixture_dataset(project: Path) -> None:
    """Ingest the checked-in RCAEval-shaped fixture into ``project/data/processed``."""
    shutil.copytree(FIXTURE_DIR, project / "data" / "raw")
    build(DataConfig(source="rcaeval", suites=FIXTURE_SUITES), project)


def splits_config(**overrides: object) -> SplitsConfig:
    fields: dict[str, object] = {
        "seed": 0,
        "strategies": ["random", "group_run", "lofo", "loso"],
    }
    return SplitsConfig.model_validate({**fields, **overrides})
