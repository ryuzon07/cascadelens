"""Training matrices built from feature tables.

Labels are joined here, outside the feature code, from the case's ``root_cause_service``. The
``service`` column is a join key only: ``RankingMatrix`` keeps it next to the numbers so predictions
can be named, and it never enters ``features``.
"""

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import polars as pl

from cascadelens.baselines.base import BaselineError, CaseHandle
from cascadelens.features.config import FeaturesConfig, GroupName
from cascadelens.features.context import KEYS, FeatureError, FeatureInput
from cascadelens.features.registry import build_case_features, feature_columns, to_model_matrix
from cascadelens.features.store import CASES_DIR, METADATA_FILE, feature_dir

Floats = npt.NDArray[np.float64]
Ints = npt.NDArray[np.int32]


class RankerError(BaselineError):
    """The ranker could not be configured, trained or evaluated."""


class TableSource(Protocol):
    """Anything that returns the feature table of a case."""

    def table(self, case: CaseHandle) -> pl.DataFrame:
        """Return ``case_id, service`` and the feature columns."""
        ...


class FeatureSource:
    """Feature tables from the Part 6 store, built on the spot for a case that is not stored."""

    def __init__(self, config: FeaturesConfig, root: Path = Path()) -> None:
        self.config = config
        self.directory = feature_dir(config, root)
        self._tables: dict[str, pl.DataFrame] = {}

    def verify(self, checksum: str) -> None:
        """Refuse a store built from another dataset version."""
        path = self.directory / METADATA_FILE
        if path.is_file():
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata["data_manifest_checksum"] != checksum:
                raise RankerError(f"{self.directory} holds features of another dataset version")

    def table(self, case: CaseHandle) -> pl.DataFrame:
        """Return the table of ``case``, reading or building it once."""
        case_id = case.record.case_id
        if case_id not in self._tables:
            path = self.directory / CASES_DIR / f"{case_id}.parquet"
            self._tables[case_id] = (
                pl.read_parquet(path)
                if path.is_file()
                else build_case_features(
                    FeatureInput.from_record(case.record, case.metrics()), self.config
                )
            )
        return self._tables[case_id]


def assert_no_identity(columns: Sequence[str]) -> None:
    """Fail if a key column would reach the model."""
    leaked = sorted(set(columns) & set(KEYS))
    if leaked:
        raise FeatureError(f"Identity columns would reach the model: {leaked}")


def model_columns(table: pl.DataFrame, groups: Sequence[GroupName]) -> list[str]:
    """Return the enabled columns, after ``to_model_matrix`` has vetted the whole table."""
    available = set(to_model_matrix(table).columns)
    columns = feature_columns(groups)
    absent = [column for column in columns if column not in available]
    if absent:
        raise FeatureError(f"The table lacks enabled columns, such as {absent[0]}")
    assert_no_identity(columns)
    return columns


def model_features(table: pl.DataFrame, columns: Sequence[str]) -> Floats:
    """Return the input matrix of one table, vetted and in ``columns`` order."""
    assert_no_identity(columns)
    return to_model_matrix(table).select(columns).to_numpy().astype(np.float64)


@dataclass(frozen=True)
class RankingMatrix:
    """Rows of several cases, contiguous by case, with one query per case."""

    case_ids: tuple[str, ...]
    services: tuple[str, ...]
    features: Floats
    labels: Ints
    sizes: tuple[int, ...]
    columns: tuple[str, ...]

    def spans(self) -> Iterator[tuple[int, int]]:
        """Yield the row range of each case."""
        start = 0
        for size in self.sizes:
            yield start, start + size
            start += size

    def take(self, rows: npt.NDArray[np.intp], sizes: Sequence[int]) -> "RankingMatrix":
        """Return ``rows``, still contiguous by case, with the new case sizes."""
        return RankingMatrix(
            self.case_ids,
            tuple(self.services[row] for row in rows.tolist()),
            self.features[rows],
            self.labels[rows],
            tuple(sizes),
            self.columns,
        )

    def cases(self, keep: Sequence[int]) -> "RankingMatrix":
        """Return the cases at the positions in ``keep``."""
        spans = list(self.spans())
        empty = np.empty(0, dtype=np.intp)
        rows = np.concatenate([empty, *(np.arange(*spans[i], dtype=np.intp) for i in keep)])
        taken = self.take(rows, [self.sizes[i] for i in keep])
        return RankingMatrix(
            tuple(self.case_ids[i] for i in keep),
            taken.services,
            taken.features,
            taken.labels,
            taken.sizes,
            taken.columns,
        )

    def with_positive(self) -> "RankingMatrix":
        """Return the cases that have a positive row.

        A query with no positive gives LambdaRank nothing to learn from, and NDCG scores it as
        perfect. Such a case is left out of Stage 2 training and of early stopping only. It stays
        in every evaluation, where it is a miss.
        """
        keep = [i for i, (s, e) in enumerate(self.spans()) if int(self.labels[s:e].sum()) > 0]
        return self.cases(keep)


def build_matrix(
    tables: Sequence[pl.DataFrame], truth: Mapping[str, str], groups: Sequence[GroupName]
) -> RankingMatrix:
    """Stack feature tables, and join the label of each row from ``truth``."""
    if not tables:
        raise RankerError("No feature tables to build a matrix from")
    columns = model_columns(tables[0], groups)
    case_ids: list[str] = []
    services: list[str] = []
    labels: list[Ints] = []
    for table in tables:
        case_id = str(table["case_id"][0])
        names = table["service"].to_list()
        case_ids.append(case_id)
        services += names
        labels.append(np.array([n == truth[case_id] for n in names], dtype=np.int32))
    return RankingMatrix(
        tuple(case_ids),
        tuple(services),
        np.vstack([model_features(table, columns) for table in tables]),
        np.concatenate(labels),
        tuple(table.height for table in tables),
        tuple(columns),
    )
