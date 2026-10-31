"""The test-fold freeze (PRD 6.4, rule 3).

``freeze`` records, per split family, the configuration chosen on validation for each fold, under
the config hash. Test results of a config with no such record are refused unless the caller asks
for exploratory results, and those are tagged in their JSON.
"""

from pathlib import Path
from typing import Any

import yaml

from cascadelens.config import RankerConfig, StrictModel, config_hash
from cascadelens.models.data import RankerError
from cascadelens.models.tuning import Choice

FROZEN_DIR = Path("configs/frozen")
FROZEN = "frozen"
EXPLORATORY = "exploratory"


class FrozenRecord(StrictModel):
    """What a freeze wrote. ``families`` maps split to method to one choice per fold."""

    name: str
    config_hash: str
    feature_config_hash: str
    data_checksum: str
    git_commit: str
    families: dict[str, dict[str, list[dict[str, Any]]]]


def write_frozen(record: FrozenRecord, root: Path = Path()) -> Path:
    """Write a record, refusing to replace an existing one."""
    path = root / FROZEN_DIR / f"{record.name}.yaml"
    if path.exists():
        raise RankerError(f"{path} already exists; a frozen config is not rewritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(record.model_dump(mode="json"), handle, sort_keys=True)
    return path


def find_frozen(config: RankerConfig, root: Path = Path()) -> FrozenRecord | None:
    """Return the record of ``config``, if any."""
    wanted = config_hash(config)
    for path in sorted((root / FROZEN_DIR).glob("*.yaml")):
        with path.open(encoding="utf-8") as handle:
            record = FrozenRecord.model_validate(yaml.safe_load(handle))
        if record.config_hash == wanted:
            return record
    return None


def resolve(
    config: RankerConfig,
    splits: tuple[str, ...],
    methods: tuple[str, ...],
    exploratory: bool,
    root: Path = Path(),
) -> tuple[str, FrozenRecord | None]:
    """Return the status of a test evaluation, or refuse it.

    An exploratory evaluation searches again and ignores any record.
    """
    if exploratory:
        return EXPLORATORY, None
    record = find_frozen(config, root)
    if record is None:
        raise RankerError(
            f"Config {config_hash(config)[:12]} is not frozen; run `cascadelens freeze` or pass "
            "--exploratory"
        )
    for split in splits:
        for method in methods:
            if method not in record.families.get(split, {}):
                raise RankerError(f"Frozen record {record.name} has no {method} on {split}")
    return FROZEN, record


def frozen_choices(
    record: FrozenRecord | None, split: str, method: str
) -> dict[str, Choice] | None:
    """Return a record's choices of one split and method by fold key."""
    if record is None:
        return None
    entries = record.families[split][method]
    return {
        entry["key"]: Choice(
            entry["key"], entry["params"], entry["k"], entry["val_top1"], entry["val_avg5"]
        )
        for entry in entries
    }
