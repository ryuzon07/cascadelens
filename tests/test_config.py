from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from cascadelens.config import (
    BaselineConfig,
    DataConfig,
    PerturbationsConfig,
    RankerConfig,
    SplitsConfig,
    canonical_json,
    config_hash,
    load_config,
)

CONFIGS = Path(__file__).parents[1] / "configs"


def write_yaml(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("name", "schema"),
    [
        ("data.yaml", DataConfig),
        ("splits.yaml", SplitsConfig),
        ("perturbations.yaml", PerturbationsConfig),
        ("ranker.yaml", RankerConfig),
    ],
)
def test_shipped_configs_are_valid(name: str, schema: type[BaseModel]) -> None:
    assert isinstance(load_config(CONFIGS / name, schema), schema)


@pytest.mark.parametrize("path", sorted((CONFIGS / "baselines").glob("*.yaml")), ids=str)
def test_shipped_baseline_configs_are_valid(path: Path) -> None:
    assert load_config(path, BaselineConfig).name == path.stem


def test_hash_ignores_key_order_and_comments(tmp_path: Path) -> None:
    first = write_yaml(tmp_path / "a.yaml", "source: rcaeval\nsuites: [one, two]\n")
    second = write_yaml(tmp_path / "b.yaml", "# comment\nsuites: [one, two]\nsource: rcaeval\n")

    assert config_hash(load_config(first, DataConfig)) == config_hash(
        load_config(second, DataConfig)
    )


def test_hash_changes_with_values() -> None:
    assert config_hash(DataConfig(source="a")) != config_hash(DataConfig(source="b"))


def test_hash_includes_defaults() -> None:
    assert config_hash(DataConfig(source="a")) == config_hash(
        DataConfig(source="a", raw_dir="data/raw")
    )


def test_hash_is_a_fixed_value() -> None:
    # Guards against changes in serialization, which would orphan recorded config hashes.
    assert config_hash(DataConfig(source="rcaeval")) == (
        "b7aaf1288079e768dbc0a7fdc1957bf0dabb9f9cf11fb92bbaf64267997abf95"
    )


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"rate": float("nan")})


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "typo.yaml", "source: rcaeval\nsuits: []\n")

    with pytest.raises(ValidationError):
        load_config(path, DataConfig)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = write_yaml(tmp_path / "list.yaml", "- one\n- two\n")

    with pytest.raises(ValueError, match="mapping"):
        load_config(path, DataConfig)


def test_stage_two_runs_on_one_thread_so_bagging_is_repeatable() -> None:
    config = load_config(CONFIGS / "ranker.yaml", RankerConfig)
    assert config.reranker.params["num_threads"] == 1
