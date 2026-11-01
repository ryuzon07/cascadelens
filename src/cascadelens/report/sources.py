"""Committed result JSONs, and number formats shared by the generators."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SOURCES = {
    "leakage": "experiments/leakage_demo/results.json",
    "baselines": "experiments/baselines_clean/results.json",
    "ranker": "experiments/ranker_clean/results.json",
    "robustness": "experiments/robustness/results.json",
    "calibration": "experiments/calibration/results.json",
    "reproduction": "experiments/reproduction/evaluation/results.json",
    "baro": "experiments/reproduction/baro/results.json",
    "lingam": "experiments/reproduction/lingam_randomwalk/results.json",
    "probe_identity": "experiments/probes/identity.json",
    "probe_graph": "experiments/probes/graph_only.json",
    "probe_mask": "experiments/probes/mask_only.json",
    "probe_window": "experiments/probes/window.json",
}
REPRODUCTION_LOG = "docs/results/reproduction_log.md"


class ReportError(Exception):
    """A result file is missing, or a document cannot be generated."""


@dataclass(frozen=True)
class Source:
    """One result JSON and its provenance fields."""

    key: str
    path: str
    metrics: dict[str, Any]
    config_hash: str
    git_commit: str
    data_checksum: str
    seed: int


class Store:
    """Read each result JSON once."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.loaded: dict[str, Source] = {}

    def __call__(self, key: str) -> Source:
        """Return one source."""
        if key not in self.loaded:
            path = self.root / SOURCES[key]
            if not path.is_file():
                raise ReportError(f"Missing result file {path}; run `make all` first")
            raw = json.loads(path.read_text(encoding="utf-8"))
            self.loaded[key] = Source(
                key,
                SOURCES[key],
                raw["metrics"],
                raw["config_hash"],
                raw["git_commit"],
                raw["data_checksum"],
                raw["seed"],
            )
        return self.loaded[key]


def pct(value: float) -> str:
    """Return a fraction as percentage points."""
    return f"{100 * value:.1f}"


def ci(entry: Mapping[str, float]) -> str:
    """Return an estimate in points, with its interval."""
    return f"{pct(entry['estimate'])} [{pct(entry['lower'])}, {pct(entry['upper'])}]"


def fraction(entry: Mapping[str, float]) -> str:
    """Return an estimate as a fraction, with its interval."""
    return f"{entry['estimate']:.3f} [{entry['lower']:.3f}, {entry['upper']:.3f}]"


def p_value(value: float) -> str:
    """Return a p-value."""
    return f"{value:.4f}"
