"""Result JSONs that trace every reported number to its config, commit and data."""

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cascadelens.config import config_hash

UNKNOWN = "unknown"


def git_commit() -> str:
    """Return the current git commit, or ``"unknown"`` outside a repository."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return UNKNOWN
    return completed.stdout.strip() or UNKNOWN


def write_result(
    path: str | Path,
    metrics: Mapping[str, Any],
    config: BaseModel,
    seed: int,
    data_checksum: str = UNKNOWN,
) -> None:
    """Save ``metrics`` with the provenance fields required for traceability."""
    payload = {
        "config": config.model_dump(mode="json"),
        "config_hash": config_hash(config),
        "data_checksum": data_checksum,
        "git_commit": git_commit(),
        "metrics": dict(metrics),
        "seed": seed,
    }
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
