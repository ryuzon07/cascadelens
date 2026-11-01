"""Check the machine and the data before a full run, with next steps for each problem.

The committed data manifest is the data checksum recorded in the result JSONs, so ``--verify-data``
compares it with the processed data after ``data build``.
"""

import argparse
import ctypes
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SUITES = ("RE1-OB", "RE1-SS", "RE1-TT")
PYTHON = (3, 11)
MIN_DISK_GB = 10.0
MIN_RAM_GB = 12.0
RAW = Path("data/raw")
COMMITTED = Path("experiments/baselines_clean/results.json")
PLACEMENT = """Place the RE1 suites at:
  data/raw/RE1-OB/<service>_<fault>/<n>/data.csv
  data/raw/RE1-SS/...
  data/raw/RE1-TT/...
Get them from https://github.com/phamquiluan/RCAEval and accept the dataset's terms yourself.
Or rerun with --download once you have accepted them."""


class _MemoryStatus(ctypes.Structure):
    _fields_ = [
        ("length", ctypes.c_ulong),
        ("load", ctypes.c_ulong),
        ("total", ctypes.c_ulonglong),
        ("rest", ctypes.c_ulonglong * 6),
    ]


def ram_gb() -> float | None:
    """Return total memory in GB, or None if it cannot be read here."""
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024**2
    if sys.platform == "win32":
        status = _MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.total / 1024**3
    return None


def machine_problems(root: Path) -> list[str]:
    """Return what is wrong with this machine for a full run."""
    problems = []
    if sys.version_info[:2] != PYTHON:
        problems.append(f"Python {PYTHON[0]}.{PYTHON[1]} is required. Run `make install`.")
    free = shutil.disk_usage(root).free / 1024**3
    if free < MIN_DISK_GB:
        problems.append(
            f"{free:.1f} GB free; a full run needs {MIN_DISK_GB:.0f} GB. Free up space."
        )
    memory = ram_gb()
    if memory is not None and memory < MIN_RAM_GB:
        problems.append(f"{memory:.1f} GB of RAM; a full run needs {MIN_RAM_GB:.0f} GB.")
    return problems


def data_problems(root: Path) -> list[str]:
    """Return which suites are missing."""
    missing = [suite for suite in SUITES if not (root / RAW / suite).is_dir()]
    return [f"Missing {', '.join(missing)}.\n{PLACEMENT}"] if missing else []


def verify_data(root: Path) -> list[str]:
    """Compare the processed data with the checksum the committed results used."""
    from cascadelens.baselines.runner import load_dataset

    expected = json.loads((root / COMMITTED).read_text(encoding="utf-8"))["data_checksum"]
    found = load_dataset(root).checksum
    if found == expected:
        return []
    return [
        f"Processed data checksum {found[:12]} differs from the committed {expected[:12]}. "
        "Check the suites in data/raw against the dataset's source, then rerun `make all`."
    ]


def main() -> int:
    """Print problems and return non-zero if there are any."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-data", action="store_true")
    parser.add_argument("--machine-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.verify_data:
        problems = verify_data(ROOT)
    else:
        problems = machine_problems(ROOT)
        if not arguments.machine_only:
            problems += data_problems(ROOT)
    for problem in problems:
        print(problem, file=sys.stderr)
    if not problems:
        print("Environment checks passed.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
