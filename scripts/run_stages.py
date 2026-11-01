"""Run the full pipeline in order, skipping a stage whose inputs have not changed.

A stage's fingerprint covers the configs, source, lock file, the earlier stage's fingerprint and
any data files it names. Fingerprints live in .cache/stages.json. Set FROZEN=1 once a config is
frozen, and SKIP_PUBLISHED=1 to leave out the published baselines, which need about two hours.
Pass --dry-run to list what would run, and --force to run everything.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[1]
STAMPS = Path(".cache/stages.json")
SHARED_INPUTS = ("configs", "src", "experiments/*/config.yaml", "uv.lock")
Command = tuple[str, ...]


@dataclass(frozen=True)
class Stage:
    """One step of the pipeline."""

    name: str
    commands: tuple[Command, ...]
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()


def _lens(*words: str) -> Command:
    return ("cascadelens", *words)


def stages(exploratory: bool, published: bool) -> list[Stage]:
    """Return every stage in order."""
    flag = ("--exploratory",) if exploratory else ()
    probes = ("identity", "graph-only", "mask-only", "window")
    pipeline = [
        Stage(
            "data",
            (
                _lens("data", "build", "--config", "configs/data.yaml"),
                _lens("data", "audit", "--config", "configs/data.yaml"),
            ),
            ("data/raw",),
            ("data/processed/cases.parquet",),
        ),
        Stage(
            "verify-data",
            ((sys.executable, "scripts/check_env.py", "--verify-data"),),
        ),
        Stage(
            "splits",
            (
                _lens("splits", "build", "--config", "configs/splits.yaml"),
                _lens("splits", "check", "--config", "configs/splits.yaml"),
            ),
        ),
        Stage(
            "leakage",
            (_lens("experiments", "leakage-demo"),),
            outputs=("experiments/leakage_demo/results.json",),
        ),
        Stage(
            "baselines",
            (
                _lens("experiments", "baselines-clean"),
                _lens("eval", "power", "--config", "configs/eval.yaml"),
            ),
            outputs=("experiments/baselines_clean/results.json",),
        ),
    ]
    if published:
        pipeline.append(
            Stage(
                "published",
                (
                    _lens("published", "reproduce", "--name", "baro"),
                    _lens("published", "reproduce", "--name", "lingam_randomwalk"),
                    _lens("published", "evaluate"),
                ),
                outputs=("experiments/reproduction/evaluation/results.json",),
            )
        )
    pipeline += [
        Stage(
            "features",
            (
                _lens("features", "build", "--config", "configs/features.yaml", "--workers", "4"),
                _lens("features", "check", "--config", "configs/features.yaml"),
            ),
        ),
        Stage(
            "ranker",
            (_lens("experiments", "ranker-clean", *flag),),
            outputs=("experiments/ranker_clean/results.json",),
        ),
        Stage("perturbations", (_lens("perturb", "summarize", "--workers", "4"),)),
        Stage(
            "robustness",
            (_lens("experiments", "robustness", "--stage", "all", *flag),),
            outputs=("experiments/robustness/results.json",),
        ),
        Stage(
            "probes",
            tuple(_lens("probes", name, *flag) for name in probes),
            outputs=("experiments/probes/window.json",),
        ),
        Stage(
            "calibration",
            (_lens("experiments", "calibration", *flag),),
            outputs=("experiments/calibration/results.json",),
        ),
        Stage(
            "report",
            ((sys.executable, "-m", "cascadelens.report", "build"),),
            outputs=("docs/results/headline_table.md",),
        ),
    ]
    return pipeline


def _files(root: Path, patterns: Sequence[str]) -> list[Path]:
    found = []
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            found += sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    return [path for path in found if "__pycache__" not in path.parts]


def fingerprint(root: Path, stage: Stage, previous: str) -> str:
    """Return a hash of a stage's commands and inputs, chained to the stage before it."""
    digest = hashlib.sha256(previous.encode())
    commands = [[part for part in command if part != sys.executable] for command in stage.commands]
    digest.update(json.dumps([stage.name, commands, stage.inputs]).encode())
    for path in _files(root, SHARED_INPUTS):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    for path in _files(root, stage.inputs):
        status = path.stat()
        digest.update(f"{path.relative_to(root).as_posix()}:{status.st_size}".encode())
    return digest.hexdigest()


def run_stages(
    root: Path,
    pipeline: Sequence[Stage],
    runner: Callable[[Command, Path], None],
    dry_run: bool = False,
    force: bool = False,
) -> list[str]:
    """Run the stages that are out of date and return their names.

    A stage's stamp is saved only after all its commands succeed, so a stopped run resumes there.
    """
    path = root / STAMPS
    saved: dict[str, str] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    previous, ran = "", []
    for stage in pipeline:
        print_name = f"[{stage.name}]"
        current = fingerprint(root, stage, previous)
        previous = current
        finished = all((root / item).is_file() for item in stage.outputs)
        if not force and saved.get(stage.name) == current and finished:
            print(f"{print_name} up to date", flush=True)
            continue
        ran.append(stage.name)
        print(f"{print_name} {'would run' if dry_run else 'running'}", flush=True)
        if dry_run:
            continue
        for command in stage.commands:
            runner(command, root)
        saved[stage.name] = current
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(saved, indent=2, sort_keys=True), encoding="utf-8")
    return ran


def _run(command: Command, root: Path) -> None:
    print("  " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def main() -> int:
    """Run the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    exploratory = os.environ.get("FROZEN", "0") != "1"
    published = os.environ.get("SKIP_PUBLISHED", "0") != "1"
    try:
        run_stages(ROOT, stages(exploratory, published), _run, arguments.dry_run, arguments.force)
    except subprocess.CalledProcessError as error:
        print(
            f"Stage failed: {' '.join(error.cmd)}. Fix it and run again to resume.", file=sys.stderr
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
