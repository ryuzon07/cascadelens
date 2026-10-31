"""Generate the synthetic RCAEval-shaped fixture in tests/fixtures/rcaeval/.

Ten invented cases across two systems, for CI. Nothing here comes from the real dataset; the
tree mirrors the layout read by cascadelens.data.loaders.rcaeval. Output is byte-identical on
every run. Usage: python scripts/make_fixture.py [target_directory]
"""

import random
import shutil
import sys
from pathlib import Path

DEFAULT_TARGET = Path(__file__).parents[1] / "tests" / "fixtures" / "rcaeval"
SEED = 20261031
SAMPLES = 40
START = 1_700_000_000
STEP = 5
INJECT_INDEX = 24
METRICS = ("cpu", "mem", "latency")
# (suite, services, [(service, fault, repetition), ...])
SUITES = (
    (
        "RE1-OB",
        ("adservice", "cartservice", "frontend"),
        (
            ("adservice", "cpu", 1),
            ("adservice", "cpu", 2),
            ("adservice", "mem", 1),
            ("cartservice", "delay", 1),
            ("cartservice", "delay", 2),
        ),
    ),
    (
        "RE1-SS",
        ("carts", "front-end", "user"),
        (
            ("carts", "cpu", 1),
            ("carts", "cpu", 2),
            ("front-end", "loss", 1),
            ("user", "mem", 1),
            ("user", "mem", 2),
        ),
    ),
)
# (suite, service, fault, repetition) of the case that repeats the time column, as RCAEval's
# main.py says some RE1-OB cases do.
REPEATED_TIME_CASE = ("RE1-OB", "adservice", "cpu", 2)
MISSING_SAMPLE_RATE = 0.05


def write_case(
    target: Path,
    rng: random.Random,
    services: tuple[str, ...],
    origin: int,
    fault_service: str,
    repeat_time: bool,
) -> None:
    """Write one case's data.csv and inject_time.txt."""
    target.mkdir(parents=True)
    columns = [f"{service}_{metric}" for service in services for metric in METRICS]
    header = ["time", *columns, *(["time"] if repeat_time else [])]
    rows = [",".join(header)]
    for index in range(SAMPLES):
        timestamp = origin + index * STEP
        cells = [str(timestamp)]
        for column in columns:
            value = rng.uniform(0.2, 0.4)
            if column.startswith(f"{fault_service}_") and index >= INJECT_INDEX:
                value += 1.0
            missing = rng.random() < MISSING_SAMPLE_RATE
            cells.append("" if missing else f"{value:.4f}")
        rows.append(",".join([*cells, *([str(timestamp)] if repeat_time else [])]))
    (target / "data.csv").write_bytes(("\n".join(rows) + "\n").encode("utf-8"))
    inject_time = origin + INJECT_INDEX * STEP
    (target / "inject_time.txt").write_bytes(f"{inject_time}\n".encode())


def generate(target: Path) -> None:
    """Replace ``target`` with a freshly generated fixture."""
    if target.exists():
        shutil.rmtree(target)
    rng = random.Random(SEED)
    origin = START
    for suite, services, cases in SUITES:
        for service, fault, repetition in cases:
            directory = target / suite / f"{service}_{fault}" / str(repetition)
            repeat_time = (suite, service, fault, repetition) == REPEATED_TIME_CASE
            write_case(directory, rng, services, origin, service, repeat_time)
            origin += 2 * SAMPLES * STEP


if __name__ == "__main__":
    generate(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TARGET)
