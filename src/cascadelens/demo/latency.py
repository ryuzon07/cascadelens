"""Measure end-to-end diagnose time and write docs/demo/latency.md.

Run ``python -m cascadelens.demo.latency``. Times include reading the case file and feature
extraction. The model is loaded once, as the API does, and its load time is reported apart. Local
benchmark cases are measured too if they exist, because the samples are far smaller.
"""

import platform
import statistics
import time
from collections.abc import Sequence
from pathlib import Path

from cascadelens.inference import LoadedModel, diagnose, load_model, read_case
from cascadelens.models.artifacts import latest_artifact
from cascadelens.results import git_commit

SAMPLES_DIR = Path("demo_data")
BENCHMARK_DIR = Path("data/processed/metrics")
OUTPUT = Path("docs/demo/latency.md")
TARGET_SECONDS = 2.0
SAMPLE_REPEATS = 20
BENCHMARK_CASES = 40


def timed(paths: Sequence[Path], model: LoadedModel, repeats: int) -> list[float]:
    """Return the seconds of each read-and-diagnose run."""
    times = []
    for _ in range(repeats):
        for path in paths:
            started = time.perf_counter()
            diagnose(read_case(path), model)
            times.append(time.perf_counter() - started)
    return times


def summary(times: Sequence[float]) -> tuple[float, float]:
    """Return the median and the 95th percentile."""
    ordered = sorted(times)
    return statistics.median(ordered), ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def _row(name: str, times: Sequence[float]) -> str:
    median, p95 = summary(times)
    verdict = "meets" if p95 < TARGET_SECONDS else "misses"
    return f"| {name} | {len(times)} | {median:.3f} | {p95:.3f} | {verdict} |"


def main() -> None:
    """Measure and write the note."""
    started = time.perf_counter()
    directory = latest_artifact()
    model = load_model(directory)
    load_seconds = time.perf_counter() - started
    samples = sorted(SAMPLES_DIR.glob("*.json"))
    rows = [_row("Synthetic samples (1,800 rows)", timed(samples, model, SAMPLE_REPEATS))]
    benchmark = sorted(BENCHMARK_DIR.glob("*.parquet"))
    if benchmark:
        step = max(1, len(benchmark) // BENCHMARK_CASES)
        chosen = benchmark[::step][:BENCHMARK_CASES]
        name = f"Benchmark cases, Parquet, local only ({len(chosen)} cases)"
        rows.append(_row(name, timed(chosen, model, 1)))
    lines = [
        "# Latency",
        "",
        f"Target: under {TARGET_SECONDS:.0f} s for one case end to end, feature extraction "
        "included, on a laptop CPU.",
        "",
        f"Measured by `python -m cascadelens.demo.latency` at commit `{git_commit()[:12]}` on "
        f"{platform.platform()}, Python {platform.python_version()}.",
        "",
        "| Input | Runs | Median (s) | p95 (s) | Against target |",
        "| --- | --- | --- | --- | --- |",
        *rows,
        "",
        f"Loading the model takes {load_seconds:.3f} s and happens once at API startup, so it is "
        "not in the table. A single CLI call pays it, plus Python's own startup.",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
