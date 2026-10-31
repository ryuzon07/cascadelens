"""Grouping cases by run, then merging runs that hold near-duplicate recordings.

Only runs sharing (system, fault_type, root_cause_service) are compared, and only their cases have
a signature computed, so the metric files of every other case are never opened. Each signature is
one short vector, not a raw series. ``inject_time`` is not read.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
from pydantic import Field

from cascadelens.config import SplitsConfig, StrictModel
from cascadelens.data.io import metrics_path
from cascadelens.data.schema import CaseRecord

Signature = npt.NDArray[np.float64]
SignatureSource = Callable[[CaseRecord], Signature]


class MergedCluster(StrictModel):
    """Several runs merged into one group."""

    group_id: str
    run_ids: tuple[str, ...]
    cases: int = Field(ge=1)


class ClusterSummary(StrictModel):
    """What the clustering did, for the leakage report."""

    threshold: float
    signature_bins: int
    cases: int
    runs: int
    groups: int
    group_sizes: dict[str, int] = Field(description="Cases in a group, to number of such groups")
    runs_with_several_cases: int
    candidate_run_pairs: int = Field(description="Run pairs eligible for merging")
    compared_case_pairs: int
    undefined_signatures: int = Field(description="Candidate cases with a flat signature")
    max_cross_run_correlation: float | None
    merged: tuple[MergedCluster, ...]
    notes: tuple[str, ...]


class Grouping(StrictModel):
    """A group for every case, with the summary of how it was reached."""

    group_map: dict[str, str]
    summary: ClusterSummary


class UnionFind:
    """Disjoint sets of strings; the root of a set is its smallest member."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, first: str, second: str) -> None:
        roots = sorted((self.find(first), self.find(second)))
        self._parent[roots[1]] = roots[0]


def case_signature(
    metrics: pl.LazyFrame, window_start: int, window_end: int, bins: int
) -> Signature:
    """Average every z-scored series into ``bins`` slices of the recording.

    Bins are relative to the recording, so cases of different lengths stay comparable. Flat series
    have no z-score and contribute nothing; a bin with no samples is zero.
    """
    value = pl.col("value")
    series = ["service", "metric"]
    span = max(window_end - window_start, 1)
    binned = (
        metrics.filter(value.is_finite())
        .with_columns(z=(value - value.mean().over(series)) / value.std().over(series))
        .filter(pl.col("z").is_finite())
        .with_columns(bin=((pl.col("timestamp") - window_start) * bins // span).clip(0, bins - 1))
        .group_by("bin")
        .agg(pl.col("z").mean())
        .collect()
    )
    signature = np.zeros(bins, dtype=np.float64)
    signature[binned["bin"].to_numpy()] = binned["z"].to_numpy()
    return signature


def stored_signature(root: Path, bins: int) -> SignatureSource:
    """Return a function computing signatures from the processed directory ``root``."""

    def compute(record: CaseRecord) -> Signature:
        metrics = pl.scan_parquet(metrics_path(root, record.case_id))
        return case_signature(metrics, record.window_start, record.window_end, bins)

    return compute


def _correlations(signatures: Sequence[Signature]) -> tuple[npt.NDArray[np.float64], list[bool]]:
    """Return pairwise Pearson correlations and which signatures have any variance."""
    matrix = np.vstack(signatures)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    defined = norms[:, 0] > 0
    unit = np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)
    return unit @ unit.T, defined.tolist()


def _render_notes(runs: int, cases: int, candidates: int, merged: int) -> tuple[str, ...]:
    notes: list[str] = []
    if runs < cases:
        notes.append(
            f"run_id does not identify one injection: {cases} cases share {runs} runs, "
            "so repetitions are grouped together before any clustering."
        )
    if candidates == 0:
        notes.append(
            "No two runs share (system, fault_type, root_cause_service), so the near-duplicate "
            "step had nothing to compare and is a no-op on this dataset."
        )
    elif merged == 0:
        notes.append("No run pair reached the threshold; every group is one run.")
    return tuple(notes)


def build_groups(
    records: Sequence[CaseRecord], signature_of: SignatureSource, threshold: float, bins: int
) -> Grouping:
    """Assign a ``group_id`` to every case.

    Two runs merge when any pair of their cases reaches ``threshold``, and merging is transitive.
    The ``group_id`` is the smallest merged run_id, so it does not depend on processing order.
    """
    runs = sorted({record.run_id for record in records})
    forest = UnionFind(runs)
    candidates: dict[tuple[str, str, str], list[CaseRecord]] = defaultdict(list)
    for record in records:
        candidates[(record.system, record.fault_type, record.root_cause_service)].append(record)

    candidate_run_pairs = compared = undefined = 0
    best: float | None = None
    for key in sorted(candidates):
        members = sorted(candidates[key], key=lambda record: record.case_id)
        run_count = len({member.run_id for member in members})
        if run_count < 2:
            continue
        candidate_run_pairs += run_count * (run_count - 1) // 2
        correlations, defined = _correlations([signature_of(member) for member in members])
        undefined += defined.count(False)
        for i, first in enumerate(members):
            for j in range(i + 1, len(members)):
                second = members[j]
                if first.run_id == second.run_id or not (defined[i] and defined[j]):
                    continue
                compared += 1
                correlation = float(correlations[i, j])
                best = correlation if best is None else max(best, correlation)
                if correlation >= threshold:
                    forest.union(first.run_id, second.run_id)

    group_map = {record.case_id: forest.find(record.run_id) for record in records}
    members_of: dict[str, set[str]] = defaultdict(set)
    for run in runs:
        members_of[forest.find(run)].add(run)
    sizes = Counter(group_map.values())
    run_sizes = Counter(record.run_id for record in records)
    histogram = Counter(sizes.values())
    merged = tuple(
        MergedCluster(group_id=group, run_ids=tuple(sorted(members)), cases=sizes[group])
        for group, members in sorted(members_of.items())
        if len(members) > 1
    )
    summary = ClusterSummary(
        threshold=threshold,
        signature_bins=bins,
        cases=len(records),
        runs=len(runs),
        groups=len(sizes),
        group_sizes={str(size): histogram[size] for size in sorted(histogram)},
        runs_with_several_cases=sum(size > 1 for size in run_sizes.values()),
        candidate_run_pairs=candidate_run_pairs,
        compared_case_pairs=compared,
        undefined_signatures=undefined,
        max_cross_run_correlation=best,
        merged=merged,
        notes=_render_notes(len(runs), len(records), candidate_run_pairs, len(merged)),
    )
    return Grouping(group_map=group_map, summary=summary)


def groups_from_config(
    records: Sequence[CaseRecord], processed_dir: Path, config: SplitsConfig
) -> Grouping:
    """Run the clustering with the configured threshold on stored metrics."""
    return build_groups(
        records,
        stored_signature(processed_dir, config.signature_bins),
        config.near_duplicate_threshold,
        config.signature_bins,
    )
