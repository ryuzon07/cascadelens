import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from cascadelens.config import DataConfig, config_hash
from cascadelens.results import git_commit, write_result

CONFIG = DataConfig(source="rcaeval")


def read(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return payload


def test_result_contains_provenance_fields(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "result.json"

    write_result(path, {"top1": 0.5}, CONFIG, seed=3, data_checksum="abc")

    assert read(path) == {
        "config": CONFIG.model_dump(mode="json"),
        "config_hash": config_hash(CONFIG),
        "data_checksum": "abc",
        "git_commit": git_commit(),
        "metrics": {"top1": 0.5},
        "seed": 3,
    }


def test_data_checksum_defaults_to_unknown(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    write_result(path, {}, CONFIG, seed=0)

    assert read(path)["data_checksum"] == "unknown"


def test_same_inputs_give_identical_bytes(tmp_path: Path) -> None:
    first, second = tmp_path / "a.json", tmp_path / "b.json"

    write_result(first, {"b": 1, "a": 2}, CONFIG, seed=0)
    write_result(second, {"a": 2, "b": 1}, CONFIG, seed=0)

    assert first.read_bytes() == second.read_bytes()


def test_nan_metrics_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_result(tmp_path / "result.json", {"top1": float("nan")}, CONFIG, seed=0)


@pytest.mark.parametrize(
    "error", [FileNotFoundError("git"), subprocess.CalledProcessError(128, "git")]
)
def test_git_commit_falls_back_to_unknown(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(subprocess, "run", fail)

    assert git_commit() == "unknown"
