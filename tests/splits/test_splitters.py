from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
from split_factory import (
    FAULTS,
    SYSTEMS,
    build_fixture_dataset,
    identity_groups,
    index_records,
    make_record,
    near_duplicate_dataset,
    splits_config,
)

from cascadelens.data.io import read_index
from cascadelens.data.schema import CaseRecord
from cascadelens.splits import LeakageError, SplitError, assert_no_group_overlap
from cascadelens.splits.groups import groups_from_config
from cascadelens.splits.report import shared_groups
from cascadelens.splits.splitters import (
    GROUPED_STRATEGIES,
    Split,
    group_run_split,
    lofo_split,
    loso_split,
    make_split,
    random_split,
)

RECORDS = index_records()
GROUPS = identity_groups(RECORDS)
CONFIG = splits_config()


def assert_clean(split: Split, group_map: dict[str, str]) -> None:
    for fold in split.folds:
        assert fold.train_ids and fold.val_ids and fold.test_ids
        assert_no_group_overlap(fold.train_ids, fold.test_ids, group_map)
        assert_no_group_overlap(fold.train_ids, fold.val_ids, group_map)
        assert_no_group_overlap(fold.val_ids, fold.test_ids, group_map)


@pytest.mark.parametrize("strategy", GROUPED_STRATEGIES)
def test_no_group_overlap_on_the_checked_in_fixture(strategy: str, tmp_path: Path) -> None:
    build_fixture_dataset(tmp_path)
    processed = tmp_path / "data" / "processed"
    records = read_index(processed)
    grouping = groups_from_config(records, processed, CONFIG)

    split = make_split(strategy, records, grouping.group_map, CONFIG)  # type: ignore[arg-type]

    assert_clean(split, grouping.group_map)


@pytest.mark.parametrize("strategy", GROUPED_STRATEGIES)
def test_no_group_overlap_on_near_duplicates(strategy: str, tmp_path: Path) -> None:
    near_duplicate_dataset(tmp_path)
    records = read_index(tmp_path)
    group_map = groups_from_config(records, tmp_path, CONFIG).group_map

    split = make_split(strategy, records, group_map, CONFIG)  # type: ignore[arg-type]

    assert_clean(split, group_map)
    for fold in split.folds:
        original = "alpha_api_cpu_0" in fold.test_ids
        replay = "alpha_api_cpu_replay_0" in fold.test_ids
        assert original == replay


def test_random_split_leaks_when_runs_hold_several_cases() -> None:
    split = random_split(RECORDS, 5, 0.2, 0)

    assert split.leaky
    assert any(shared_groups(fold.train_ids, fold.test_ids, GROUPS) for fold in split.folds)


def test_grouped_splits_are_not_marked_leaky() -> None:
    assert not group_run_split(RECORDS, GROUPS, 5, 0.2, 0).leaky


@pytest.mark.parametrize(
    "build",
    [
        lambda seed: random_split(RECORDS, 5, 0.2, seed),
        lambda seed: group_run_split(RECORDS, GROUPS, 5, 0.2, seed),
    ],
    ids=["random", "group_run"],
)
def test_seed_controls_random_and_group_run(build: Callable[[int], Split]) -> None:
    assert build(0) == build(0)
    assert build(0) != build(1)


@pytest.mark.parametrize(
    "build",
    [
        lambda records: lofo_split(records, GROUPS, 0.2, 0),
        lambda records: loso_split(records, GROUPS, 0.2, 0),
        lambda records: group_run_split(records, GROUPS, 5, 0.2, 0),
    ],
    ids=["lofo", "loso", "group_run"],
)
def test_input_order_does_not_change_a_split(build: Callable[[list[CaseRecord]], Split]) -> None:
    assert build(RECORDS) == build(list(reversed(RECORDS)))


def test_seed_changes_lofo_and_loso_validation_only() -> None:
    first, second = lofo_split(RECORDS, GROUPS, 0.2, 0), lofo_split(RECORDS, GROUPS, 0.2, 1)

    assert [fold.test_ids for fold in first.folds] == [fold.test_ids for fold in second.folds]
    assert [fold.val_ids for fold in first.folds] != [fold.val_ids for fold in second.folds]


def test_every_case_is_in_exactly_one_group_run_test_fold() -> None:
    split = group_run_split(RECORDS, GROUPS, 5, 0.2, 0)

    counts = Counter(case_id for fold in split.folds for case_id in fold.test_ids)

    assert counts == Counter(record.case_id for record in RECORDS)


def test_group_run_partitions_every_case_within_a_fold() -> None:
    for fold in group_run_split(RECORDS, GROUPS, 5, 0.2, 0).folds:
        assert sorted((*fold.train_ids, *fold.val_ids, *fold.test_ids)) == sorted(GROUPS)


def test_group_run_is_stratified_by_system() -> None:
    by_id = {record.case_id: record.system for record in RECORDS}

    for fold in group_run_split(RECORDS, GROUPS, 3, 0.2, 0).folds:
        systems = Counter(by_id[case_id] for case_id in fold.test_ids)
        assert set(systems) == set(SYSTEMS)
        assert max(systems.values()) - min(systems.values()) <= 2


@pytest.mark.parametrize("seed", range(5))
def test_group_run_puts_each_service_in_each_fold_once(seed: int) -> None:
    keys = {r.case_id: (r.system, r.root_cause_service) for r in RECORDS}

    for fold in group_run_split(RECORDS, GROUPS, 4, 0.2, seed).folds:
        assert len({GROUPS[case_id] for case_id in fold.test_ids}) == 9
        assert len({keys[case_id] for case_id in fold.test_ids}) == 9


@pytest.mark.parametrize("seed", range(30))
def test_validation_never_takes_a_service_away_from_training(seed: int) -> None:
    records = [
        make_record(
            f"{system}-{service}-{fault}", f"{system}/{service}_{fault}", system, fault, service
        )
        for system in SYSTEMS
        for service in ("a", "b", "c", "d", "e")
        for fault in ("cpu", "delay", "disk", "loss", "mem")
    ]
    groups = identity_groups(records)
    labels = {record.case_id: (record.system, record.root_cause_service) for record in records}

    for fold in group_run_split(records, groups, 5, 0.2, seed).folds:
        trained = {labels[case_id] for case_id in fold.train_ids}
        assert {labels[case_id] for case_id in fold.test_ids} <= trained
        assert {labels[case_id] for case_id in fold.val_ids} <= trained
        assert len(fold.val_ids) == 12


@pytest.mark.parametrize("seed", range(5))
def test_group_run_spreads_services_evenly_with_fewer_folds_than_groups(seed: int) -> None:
    counts = Counter(
        (index, record.system, record.root_cause_service)
        for index, fold in enumerate(group_run_split(RECORDS, GROUPS, 3, 0.2, seed).folds)
        for record in RECORDS
        if record.case_id in fold.test_ids
    )

    assert max(counts.values()) - min(counts.values()) <= 2


def test_group_run_tolerates_a_system_with_too_few_groups() -> None:
    records = [*RECORDS, make_record("lonely", "lonely-run", "delta")]
    groups = identity_groups(records)

    assert_clean(group_run_split(records, groups, 5, 0.2, 0), groups)


def test_validation_comes_from_training_groups_only() -> None:
    for split in (
        group_run_split(RECORDS, GROUPS, 5, 0.2, 0),
        lofo_split(RECORDS, GROUPS, 0.2, 0),
        loso_split(RECORDS, GROUPS, 0.2, 0),
    ):
        for fold in split.folds:
            test_groups = {GROUPS[case_id] for case_id in fold.test_ids}
            assert not test_groups & {GROUPS[case_id] for case_id in fold.val_ids}


def test_validation_size_follows_the_fraction() -> None:
    fold = group_run_split(RECORDS, GROUPS, 5, 0.25, 0).folds[0]

    assert len(fold.val_ids) == pytest.approx(0.25 * (len(RECORDS) - len(fold.test_ids)), abs=4)


def test_loso_has_one_fold_per_system_and_holds_all_of_it_out() -> None:
    split = loso_split(RECORDS, GROUPS, 0.2, 0)
    systems = {record.case_id: record.system for record in RECORDS}

    assert [fold.held_out for fold in split.folds] == sorted(SYSTEMS)
    for fold in split.folds:
        assert {systems[case_id] for case_id in fold.test_ids} == {fold.held_out}
        assert fold.held_out not in {systems[case_id] for case_id in fold.train_ids}


def test_lofo_has_one_fold_per_fault_type_and_holds_all_of_it_out() -> None:
    split = lofo_split(RECORDS, GROUPS, 0.2, 0)
    faults = {record.case_id: record.fault_type for record in RECORDS}

    assert [fold.held_out for fold in split.folds] == sorted(FAULTS)
    for fold in split.folds:
        assert {faults[case_id] for case_id in fold.test_ids} == {fold.held_out}
        assert fold.held_out not in {faults[case_id] for case_id in fold.train_ids}


def test_a_group_spanning_fault_types_never_straddles_a_lofo_fold() -> None:
    records = [
        make_record("mixed-1", "mixed", fault_type="cpu"),
        make_record("mixed-2", "mixed", fault_type="mem"),
        *RECORDS,
    ]
    groups = identity_groups(records)

    assert_clean(lofo_split(records, groups, 0.2, 0), groups)


def test_loso_with_one_system_fails_loudly() -> None:
    records = [r for r in RECORDS if r.system == SYSTEMS[0]]

    with pytest.raises(SplitError, match=r"loso fold 0 \(alpha\): no groups left for training"):
        loso_split(records, GROUPS, 0.2, 0)


def test_lofo_with_one_fault_type_fails_loudly() -> None:
    records = [r for r in RECORDS if r.fault_type == FAULTS[0]]

    with pytest.raises(SplitError, match=r"lofo fold 0 \(cpu\): no groups left for training"):
        lofo_split(records, GROUPS, 0.2, 0)


def test_too_few_groups_for_the_test_folds_fails_loudly() -> None:
    records = RECORDS[:4]

    with pytest.raises(SplitError, match=r"2 group\(s\) cannot fill 5 test folds"):
        group_run_split(records, GROUPS, 5, 0.2, 0)


def test_too_few_cases_for_the_random_folds_fails_loudly() -> None:
    with pytest.raises(SplitError, match=r"4 case\(s\) cannot fill 5 test folds"):
        random_split(RECORDS[:4], 5, 0.2, 0)


def test_empty_validation_fails_with_the_reason() -> None:
    records = [
        make_record("a", "run-a", "alpha"),
        make_record("b", "run-b", "beta"),
        make_record("c", "run-c", "beta", fault_type="mem"),
    ]
    only_one_left = [record for record in records if record.case_id != "c"]

    with pytest.raises(SplitError, match=r"validation set is empty; 1 training group"):
        loso_split(only_one_left, identity_groups(records), 0.2, 0)


def test_a_case_without_a_group_fails_loudly() -> None:
    with pytest.raises(SplitError, match="has no group"):
        loso_split(RECORDS, {}, 0.2, 0)


def test_helper_passes_on_disjoint_groups() -> None:
    assert_no_group_overlap(["a"], ["b"], {"a": "g1", "b": "g2"})


def test_helper_raises_on_a_shared_group() -> None:
    with pytest.raises(LeakageError, match=r"1 group\(s\) on both sides: \['g'\]"):
        assert_no_group_overlap(["a"], ["b"], {"a": "g", "b": "g"})


def test_helper_refuses_a_case_it_cannot_place() -> None:
    with pytest.raises(LeakageError, match="no group"):
        assert_no_group_overlap(["a"], ["b"], {"a": "g"})


def test_helper_accepts_generators() -> None:
    assert_no_group_overlap((i for i in ["a"]), (i for i in ["b"]), {"a": "g1", "b": "g2"})
