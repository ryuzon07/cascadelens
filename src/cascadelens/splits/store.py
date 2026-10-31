"""Saved splits, tied to one dataset version.

Layout under the output directory::

    cluster_summary.json            what the near-duplicate step did
    <name>/split.json               the folds, with the config hash and data manifest checksum
    <name>/groups.parquet           case_id to group_id, as used by that split
    <name>/leakage_report.json      overlap, counts and the pass flag

A build goes through a staging directory, so a failure leaves any previous build untouched.
"""

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from pydantic import BaseModel

from cascadelens.config import SplitsConfig, StrictModel, config_hash
from cascadelens.data.errors import DataError
from cascadelens.data.io import CASES_FILE, read_index, read_manifest
from cascadelens.data.schema import CaseRecord
from cascadelens.splits.errors import SplitError
from cascadelens.splits.groups import groups_from_config
from cascadelens.splits.report import LeakageReport, build_report, shared_groups
from cascadelens.splits.splitters import GROUPED_STRATEGIES, Split, make_split

SPLIT_FILE = "split.json"
GROUPS_FILE = "groups.parquet"
REPORT_FILE = "leakage_report.json"
CLUSTER_FILE = "cluster_summary.json"


class SavedSplit(StrictModel):
    """A split with what it was built from."""

    config_hash: str
    data_manifest_checksum: str
    split: Split


@dataclass(frozen=True)
class CheckResult:
    """The outcome of re-verifying one saved split."""

    name: str
    problems: tuple[str, ...]


def _write_json(model: BaseModel, path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(model.model_dump(mode="json"), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _load_dataset(processed_dir: Path) -> tuple[list[CaseRecord], str]:
    if not (processed_dir / CASES_FILE).is_file():
        raise SplitError(f"No cases index in {processed_dir}; run `cascadelens data build` first")
    try:
        return read_index(processed_dir), read_manifest(processed_dir).digest()
    except (DataError, FileNotFoundError) as error:
        raise SplitError(f"Cannot read the dataset in {processed_dir}: {error}") from error


def _replace_directory(source: Path, target: Path) -> None:
    """Move ``source`` to ``target``, refusing to delete anything that is not a prior build."""
    if target.exists():
        if any(target.iterdir()) and not (target / CLUSTER_FILE).is_file():
            raise SplitError(f"{target} is not empty and does not look like a previous build")
        shutil.rmtree(target)
    source.rename(target)


def build_splits(config: SplitsConfig, root: Path = Path()) -> list[LeakageReport]:
    """Build, report and save every configured strategy."""
    if not config.strategies:
        raise SplitError("No strategies selected in the splits config")
    processed_dir = root / config.processed_dir
    records, checksum = _load_dataset(processed_dir)
    grouping = groups_from_config(records, processed_dir, config)
    digest = config_hash(config)
    splits = [
        make_split(strategy, records, grouping.group_map, config)
        for strategy in dict.fromkeys(config.strategies)
    ]
    reports = [
        build_report(split, records, grouping.group_map, grouping.summary, digest, checksum)
        for split in splits
    ]

    output_dir = root / config.output_dir
    staging = output_dir.with_name(f"{output_dir.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)
    groups = pl.DataFrame(
        sorted(grouping.group_map.items()), schema=["case_id", "group_id"], orient="row"
    )
    try:
        for split, report in zip(splits, reports, strict=True):
            directory = staging / split.name
            directory.mkdir(parents=True)
            saved = SavedSplit(config_hash=digest, data_manifest_checksum=checksum, split=split)
            _write_json(saved, directory / SPLIT_FILE)
            _write_json(report, directory / REPORT_FILE)
            groups.write_parquet(directory / GROUPS_FILE)
        _write_json(grouping.summary, staging / CLUSTER_FILE)
        _replace_directory(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return reports


def read_split(directory: Path) -> tuple[SavedSplit, dict[str, str]]:
    """Read one saved split and its group map."""
    saved = SavedSplit.model_validate_json((directory / SPLIT_FILE).read_text(encoding="utf-8"))
    frame = pl.read_parquet(directory / GROUPS_FILE)
    return saved, dict(zip(frame["case_id"], frame["group_id"], strict=True))


def _verify(
    directory: Path, records: list[CaseRecord], digest: str, checksum: str
) -> tuple[str, ...]:
    saved, group_map = read_split(directory)
    problems: list[str] = []
    if saved.data_manifest_checksum != checksum:
        problems.append("built from a different dataset version")
    if saved.config_hash != digest:
        problems.append("built from a different splits config")
    run_groups: dict[str, set[str]] = {}
    for record in records:
        if record.case_id not in group_map:
            problems.append(f"case {record.case_id} has no group")
            continue
        run_groups.setdefault(record.run_id, set()).add(group_map[record.case_id])
    problems += [f"run {run} spans {len(g)} groups" for run, g in run_groups.items() if len(g) > 1]
    known = {record.case_id for record in records}
    for index, fold in enumerate(saved.split.folds):
        parts = {"train": fold.train_ids, "val": fold.val_ids, "test": fold.test_ids}
        problems += [
            f"fold {index}: {part} is empty or holds unknown cases"
            for part, ids in parts.items()
            if not ids or not known.issuperset(ids)
        ]
        for first, second in (("train", "test"), ("train", "val"), ("val", "test")):
            shared = shared_groups(
                (i for i in parts[first] if i in group_map),
                (i for i in parts[second] if i in group_map),
                group_map,
            )
            if shared:
                problems.append(f"fold {index}: {first}/{second} share {len(shared)} group(s)")
    return tuple(problems)


def check_splits(config: SplitsConfig, root: Path = Path()) -> list[CheckResult]:
    """Re-verify every saved grouped split from its files, against the current dataset."""
    records, checksum = _load_dataset(root / config.processed_dir)
    digest = config_hash(config)
    output_dir = root / config.output_dir
    results = []
    for name in GROUPED_STRATEGIES:
        directory = output_dir / name
        if (directory / SPLIT_FILE).is_file():
            results.append(CheckResult(name, _verify(directory, records, digest, checksum)))
    if not results:
        raise SplitError(f"No saved grouped splits in {output_dir}")
    return results
