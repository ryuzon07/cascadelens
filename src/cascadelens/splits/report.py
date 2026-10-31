"""The leakage report (PRD section 6.4) and the no-leak assertion other parts import."""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence

from pydantic import Field

from cascadelens.config import SplitStrategy, StrictModel
from cascadelens.data.schema import CaseRecord
from cascadelens.splits.errors import LeakageError
from cascadelens.splits.groups import ClusterSummary
from cascadelens.splits.splitters import Fold, GroupMap, Split


def _groups_of(ids: Iterable[str], group_map: GroupMap) -> set[str]:
    ids = list(ids)
    unknown = sorted(case_id for case_id in ids if case_id not in group_map)
    if unknown:
        raise LeakageError(f"{len(unknown)} case(s) have no group, such as {unknown[0]}")
    return {group_map[case_id] for case_id in ids}


def shared_groups(first: Iterable[str], second: Iterable[str], group_map: GroupMap) -> list[str]:
    """Return the groups present on both sides, sorted."""
    return sorted(_groups_of(first, group_map) & _groups_of(second, group_map))


def assert_no_group_overlap(
    train_ids: Iterable[str], test_ids: Iterable[str], group_map: GroupMap
) -> None:
    """Raise ``LeakageError`` if any group appears on both sides, or a case has no group."""
    shared = shared_groups(train_ids, test_ids, group_map)
    if shared:
        raise LeakageError(f"{len(shared)} group(s) on both sides: {shared}")


class PartCounts(StrictModel):
    """Size and label mix of one partition of a fold."""

    cases: int
    groups: int
    systems: int
    fault_types: int
    labels: dict[str, int] = Field(description="root_cause_service to number of cases")


class FoldReport(StrictModel):
    """Overlap and counts of one fold."""

    fold: int
    held_out: str | None
    train_test_overlap: tuple[str, ...]
    train_val_overlap: tuple[str, ...]
    val_test_overlap: tuple[str, ...]
    train: PartCounts
    val: PartCounts
    test: PartCounts
    test_labels_unseen_in_train: tuple[str, ...]


class OverlapTotals(StrictModel):
    """Overlap across all folds, counting a group once per fold it leaks in."""

    train_test: int
    train_val: int
    val_test: int
    folds_with_overlap: int


class LeakageReport(StrictModel):
    """Everything ``leakage_report.json`` holds for one split."""

    split: SplitStrategy
    leaky_by_design: bool
    passed: bool = Field(description="True when no group is shared between any two partitions")
    config_hash: str
    data_manifest_checksum: str
    overall: OverlapTotals
    folds: tuple[FoldReport, ...]
    clusters: ClusterSummary


def _counts(
    ids: Sequence[str], records: Mapping[str, CaseRecord], group_map: GroupMap
) -> PartCounts:
    members = [records[case_id] for case_id in ids]
    labels = Counter(record.root_cause_service for record in members)
    return PartCounts(
        cases=len(members),
        groups=len(_groups_of(ids, group_map)),
        systems=len({record.system for record in members}),
        fault_types=len({record.fault_type for record in members}),
        labels=dict(sorted(labels.items())),
    )


def _fold_report(
    index: int, fold: Fold, records: Mapping[str, CaseRecord], group_map: GroupMap
) -> FoldReport:
    train = _counts(fold.train_ids, records, group_map)
    test = _counts(fold.test_ids, records, group_map)
    return FoldReport(
        fold=index,
        held_out=fold.held_out,
        train_test_overlap=tuple(shared_groups(fold.train_ids, fold.test_ids, group_map)),
        train_val_overlap=tuple(shared_groups(fold.train_ids, fold.val_ids, group_map)),
        val_test_overlap=tuple(shared_groups(fold.val_ids, fold.test_ids, group_map)),
        train=train,
        val=_counts(fold.val_ids, records, group_map),
        test=test,
        test_labels_unseen_in_train=tuple(sorted(set(test.labels) - set(train.labels))),
    )


def build_report(
    split: Split,
    records: Sequence[CaseRecord],
    group_map: GroupMap,
    clusters: ClusterSummary,
    config_hash: str,
    data_manifest_checksum: str,
) -> LeakageReport:
    """Measure one split. A failed report is data, not an error: ``random`` is meant to fail."""
    by_id = {record.case_id: record for record in records}
    folds = tuple(
        _fold_report(index, fold, by_id, group_map) for index, fold in enumerate(split.folds)
    )
    overall = OverlapTotals(
        train_test=sum(len(fold.train_test_overlap) for fold in folds),
        train_val=sum(len(fold.train_val_overlap) for fold in folds),
        val_test=sum(len(fold.val_test_overlap) for fold in folds),
        folds_with_overlap=sum(
            bool(fold.train_test_overlap or fold.train_val_overlap or fold.val_test_overlap)
            for fold in folds
        ),
    )
    return LeakageReport(
        split=split.name,
        leaky_by_design=split.leaky,
        passed=overall.folds_with_overlap == 0,
        config_hash=config_hash,
        data_manifest_checksum=data_manifest_checksum,
        overall=overall,
        folds=folds,
        clusters=clusters,
    )
