"""Write the synthetic sample cases in demo_data/. No benchmark data goes into them.

Run ``python -m cascadelens.demo.samples``. A sample holds the fields of ``POST /diagnose``, and
those with a ``simulation`` block also name the simulated fault. Only the failure-correlated
slider of the demo reads that block; ``read_case`` drops it.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np

SAMPLES_DIR = Path("demo_data")
SERVICES = ("frontend", "checkout", "payment", "catalog", "cart")
CALLS = (
    ("frontend", "checkout"),
    ("frontend", "catalog"),
    ("frontend", "cart"),
    ("checkout", "payment"),
    ("checkout", "cart"),
)
METRICS = {"cpu": 30.0, "mem": 200.0, "latency": 0.05, "error": 0.2}
START = 1_700_000_000
STEP = 5
SAMPLES = 90
FAULT_AT = SAMPLES // 2


def _series(rng: np.random.Generator, base: float, effect: float) -> list[float]:
    values = base * (1 + rng.normal(0, 0.03, SAMPLES))
    values[FAULT_AT:] *= effect
    return [round(float(value), 3) for value in values]


def sample_case(
    name: str,
    culprit: str,
    metric: str,
    seed: int,
    topology: bool = True,
    labelled: bool = True,
    silent: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Return one case: ``culprit`` degrades at the midpoint, and its callers slow down less."""
    rng = np.random.default_rng(seed)
    callers = {caller for caller, callee in CALLS if callee == culprit}
    rows = []
    for service in SERVICES:
        for family, base in METRICS.items():
            effect = 1.0
            if service == culprit and family == metric:
                effect = 8.0
            elif service in callers and family == "latency":
                effect = 2.5
            for index, value in enumerate(_series(rng, base, effect)):
                gone = service in silent and index >= FAULT_AT
                rows.append(
                    {
                        "service": service,
                        "metric": family,
                        "ts": START + STEP * index,
                        "value": None if gone else value,
                    }
                )
    case: dict[str, Any] = {
        "case_id": name,
        "metrics": rows,
        "window": {"start": START, "end": START + STEP * (SAMPLES - 1)},
    }
    if topology:
        case["topology"] = [{"caller": a, "callee": b} for a, b in CALLS]
    if labelled:
        case["simulation"] = {
            "root_cause_service": culprit,
            "inject_time": START + STEP * FAULT_AT,
        }
    return case


def all_samples() -> list[dict[str, Any]]:
    """Return every sample, deterministically."""
    return [
        sample_case("payment-cpu", "payment", "cpu", 1),
        sample_case("catalog-latency", "catalog", "latency", 2),
        sample_case("cart-memory-no-topology", "cart", "mem", 3, topology=False),
        sample_case("payment-silent", "payment", "latency", 4, silent=("payment", "checkout")),
        sample_case("checkout-unlabelled", "checkout", "cpu", 5, labelled=False),
    ]


def write_samples(directory: Path = SAMPLES_DIR) -> list[Path]:
    """Write every sample as compact JSON."""
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for case in all_samples():
        path = directory / f"{case['case_id']}.json"
        text = json.dumps(case, sort_keys=True, separators=(",", ":"))
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
        paths.append(path)
    return paths


if __name__ == "__main__":
    for written in write_samples():
        print(written)
