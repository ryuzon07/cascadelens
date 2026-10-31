"""The build step: load, validate, write and checksum."""

import shutil
from pathlib import Path

from cascadelens.config import DataConfig, config_hash
from cascadelens.data.errors import DataError, Issue, RawLayoutError
from cascadelens.data.io import (
    CASES_FILE,
    build_manifest,
    write_index,
    write_manifest,
    write_metrics,
)
from cascadelens.data.loaders.rcaeval import find_suite_dir, iter_cases
from cascadelens.data.schema import CaseRecord
from cascadelens.data.validate import CaseValidator, raise_on_errors

SUPPORTED_SOURCE = "rcaeval"


def check_config(config: DataConfig) -> None:
    """Refuse a config the loader cannot honour."""
    if config.source != SUPPORTED_SOURCE:
        raise DataError(f"Unsupported data source {config.source!r}")
    if not config.suites:
        raise DataError("No suites selected in the data config")


def _replace_directory(source: Path, target: Path) -> None:
    """Move ``source`` to ``target``, refusing to delete anything that is not a prior build."""
    if target.exists():
        contents = list(target.iterdir())
        if contents and not (target / CASES_FILE).is_file():
            raise DataError(f"{target} is not empty and does not look like a previous build")
        shutil.rmtree(target)
    source.rename(target)


def build(config: DataConfig, root: Path = Path()) -> list[Issue]:
    """Run the build and return the logged warnings.

    Output goes to a staging directory, so a failed build leaves any previous build untouched.
    """
    check_config(config)
    raw_dir = root / config.raw_dir
    processed_dir = root / config.processed_dir
    suite_dirs = {suite: find_suite_dir(raw_dir, suite) for suite in config.suites}
    staging = processed_dir.with_name(f"{processed_dir.name}.staging")
    if staging.exists():
        shutil.rmtree(staging)

    issues: list[Issue] = []
    records: list[CaseRecord] = []
    validator = CaseValidator()
    try:
        for item in iter_cases(raw_dir, config.suites):
            if isinstance(item, Issue):
                issues.append(item)
                continue
            issues.extend(validator.check(item))
            write_metrics(item, staging)
            records.append(item.record())
        raise_on_errors(issues)
        if not records:
            raise RawLayoutError(f"No cases found in {raw_dir}")
        write_index(records, staging)
        write_manifest(build_manifest(config.source, raw_dir, suite_dirs), staging)
        (staging / "config_hash.txt").write_bytes(f"{config_hash(config)}\n".encode())
        _replace_directory(staging, processed_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return issues
