"""Generated cases for baseline tests: flat telemetry with one service that shifts first."""

import shutil
import zlib
from pathlib import Path

import numpy as np
import polars as pl

from cascadelens.baselines.base import CaseHandle
from cascadelens.config import SplitsConfig
from cascadelens.data.io import Manifest, write_cases, write_manifest
from cascadelens.data.schema import METRICS_SCHEMA, Case, TopologyEdge
from cascadelens.splits.store import build_splits

SAMPLES = 200
START = 1000
CONFIGS = Path(__file__).parents[2] / "configs" / "baselines"


def make_metrics(onsets: dict[str, int | None], seed: int = 0) -> pl.DataFrame:
    """Return cpu and a cumulative counter per service; a service shifts at its onset step."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[str, str, int, float]] = []
    for service, onset in onsets.items():
        noise = rng.normal(scale=0.1, size=SAMPLES)
        shift = np.zeros(SAMPLES) if onset is None else (np.arange(SAMPLES) >= onset) * 10.0
        cpu = 5.0 + noise + shift
        counter = np.cumsum(1.0 + noise + shift)
        for step in range(SAMPLES):
            rows.append((service, "cpu", START + step, float(cpu[step])))
            rows.append((service, "request-total", START + step, float(counter[step])))
    return pl.DataFrame(rows, schema=METRICS_SCHEMA, orient="row")


def make_case(
    case_id: str = "case-1",
    root_cause: str = "db",
    onsets: dict[str, int | None] | None = None,
    system: str = "shop",
    fault_type: str = "cpu",
    run_id: str = "run-1",
    inject_time: int = START + 100,
    topology: tuple[TopologyEdge, ...] | None = None,
    seed: int = 0,
) -> Case:
    chosen = onsets if onsets is not None else {"api": 120, "cache": None, "db": 100}
    return Case(
        case_id=case_id,
        suite="suite",
        system=system,
        fault_type=fault_type,
        run_id=run_id,
        repetition=1,
        root_cause_service=root_cause,
        inject_time=inject_time,
        window_start=START,
        window_end=START + SAMPLES - 1,
        topology=topology,
        metrics=make_metrics(chosen, seed),
    )


def handle(case: Case) -> CaseHandle:
    return CaseHandle.from_case(case)


def build_project(root: Path) -> list[Case]:
    """Write a small processed dataset, its splits and the shipped baseline configs."""
    cases = []
    services = ("api", "cart", "db", "pay")
    for system in ("shop", "bank", "mail"):
        for root_cause in services:
            for fault in ("cpu", "mem", "delay", "loss"):
                run = f"{system}/{root_cause}_{fault}"
                for repetition in range(2):
                    onsets: dict[str, int | None] = {
                        service: 100 if service == root_cause else 130 for service in services
                    }
                    cases.append(
                        make_case(
                            f"{system}-{root_cause}-{fault}-{repetition}",
                            root_cause,
                            onsets,
                            system,
                            fault,
                            run,
                            seed=zlib.crc32(f"{run}{repetition}".encode()),
                        )
                    )
    processed = root / "data" / "processed"
    write_cases(cases, processed)
    write_manifest(Manifest(source="generated", suites=("suite",), files=()), processed)
    build_splits(
        SplitsConfig(seed=0, strategies=["random", "group_run", "lofo", "loso"]),
        root,
    )
    target = root / "configs" / "baselines"
    shutil.copytree(CONFIGS, target)
    for name, budget in (("lightgbm_flat", 2),):
        path = target / f"{name}.yaml"
        text = path.read_text(encoding="utf-8").replace(
            "search_budget: 30", f"search_budget: {budget}"
        )
        path.write_text(
            text.replace("num_boost_round: 100", "num_boost_round: 10"), encoding="utf-8"
        )
    return cases
