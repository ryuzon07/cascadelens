"""Run the pipeline end to end on the checked-in fixture, in a temporary directory.

This is the 10-case smoke run of PRD 10.2. It needs no downloaded data, and any failing step stops
it with a non-zero exit, including group overlap found by ``splits check``.
"""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "rcaeval"
DATA_CONFIG = "source: rcaeval\nsuites: [RE1-OB, RE1-SS]\n"
STEPS = (
    ("data", "build", "--config", "configs/data.yaml"),
    ("splits", "build", "--config", "configs/splits.yaml"),
    ("splits", "check", "--config", "configs/splits.yaml"),
    ("features", "build", "--config", "configs/features.yaml"),
    ("features", "check", "--config", "configs/features.yaml"),
    ("baselines", "run", "--name", "frequency", "--split", "group_run"),
    ("baselines", "run", "--name", "rule", "--split", "group_run"),
)


def main() -> None:
    """Run every step, stopping at the first failure."""
    start = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="cascadelens-smoke-") as folder:
        project = Path(folder)
        shutil.copytree(FIXTURE, project / "data" / "raw")
        shutil.copytree(ROOT / "configs", project / "configs")
        (project / "configs" / "data.yaml").write_text(DATA_CONFIG, encoding="utf-8")
        for step in STEPS:
            print("cascadelens", *step, flush=True)
            subprocess.run(["cascadelens", *step], cwd=project, check=True)
    print(f"Smoke run passed in {time.perf_counter() - start:.1f} s.")


if __name__ == "__main__":
    main()
