"""The four splitting protocols of PRD section 6.3.

Splitters read the cases index and a ``group_map`` only. Validation always comes out of the
training groups, and a fold that would be empty raises ``SplitError``.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np
from pydantic import Field

from cascadelens.config import SplitsConfig, SplitStrategy, StrictModel
from cascadelens.data.schema import CaseRecord
from cascadelens.seed import derive_case_seed
from cascadelens.splits.errors import SplitError

GroupMap = Mapping[str, str]
GROUPED_STRATEGIES: tuple[SplitStrategy, ...] = ("group_run", "lofo", "loso")


class Fold(StrictModel):
    """Case ids of one fold. ``held_out`` names what LOFO or LOSO left out."""

    train_ids: tuple[str, ...]
    val_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    held_out: str | None = None


class Split(StrictModel):
    """A splitting protocol applied to one dataset."""

    name: SplitStrategy
    seed: int = Field(ge=0)
    leaky: bool = False
    folds: tuple[Fold, ...]


def _rng(name: str, purpose: str, fold: int, seed: int) -> np.random.Generator:
    return np.random.default_rng(derive_case_seed(name, purpose, {"fold": fold}, seed))


def _shuffled(items: Iterable[str], rng: np.random.Generator) -> list[str]:
    ordered = sorted(items)
    return [ordered[int(index)] for index in rng.permutation(len(ordered))]


def _groups_of(records: Sequence[CaseRecord], group_map: GroupMap) -> dict[str, list[CaseRecord]]:
    groups: dict[str, list[CaseRecord]] = defaultdict(list)
    for record in sorted(records, key=lambda record: record.case_id):
        if record.case_id not in group_map:
            raise SplitError(f"Case {record.case_id} has no group")
        groups[group_map[record.case_id]].append(record)
    return dict(groups)


def _spread_by_service(members: Sequence[str], services: Mapping[str, str]) -> list[str]:
    """Reorder groups so a prefix takes from every service before taking twice from any.

    Services with the most groups come first, so the last group of a service is taken last.
    """
    sizes = Counter(services[group] for group in members)
    seen: Counter[str] = Counter()
    ranks = {}
    for group in members:
        ranks[group] = (seen[services[group]], -sizes[services[group]])
        seen[services[group]] += 1
    return sorted(members, key=lambda group: ranks[group])


def _carve_validation(
    groups: Iterable[str],
    strata: Mapping[str, str],
    services: Mapping[str, str],
    fraction: float,
    rng: np.random.Generator,
) -> set[str]:
    """Pick validation groups per stratum, leaving at least one group of each for training.

    Groups are spread over services, so validation does not take a service out of training.
    """
    by_stratum: dict[str, list[str]] = defaultdict(list)
    for group in groups:
        by_stratum[strata[group]].append(group)
    chosen: set[str] = set()
    for stratum in sorted(by_stratum):
        members = _spread_by_service(_shuffled(by_stratum[stratum], rng), services)
        count = min(len(members) - 1, max(1, round(fraction * len(members))))
        chosen.update(members[:count])
    return chosen


def _fold(
    label: str,
    groups: Mapping[str, Sequence[CaseRecord]],
    test_ids: Iterable[str],
    test_groups: set[str],
    strata: Mapping[str, str],
    fraction: float,
    rng: np.random.Generator,
    held_out: str | None = None,
) -> Fold:
    """Build one fold; a group touching the test set never reaches train or validation."""
    tests = tuple(sorted(test_ids))
    if not tests:
        raise SplitError(f"{label}: test set is empty")
    remaining = sorted(set(groups) - test_groups)
    if not remaining:
        raise SplitError(f"{label}: no groups left for training after holding out the test set")
    services = {group: groups[group][0].root_cause_service for group in remaining}
    val_groups = _carve_validation(remaining, strata, services, fraction, rng)
    if not val_groups:
        raise SplitError(
            f"{label}: validation set is empty; {len(remaining)} training group(s) spread over "
            f"{len({strata[group] for group in remaining})} stratum/strata leave none to spare"
        )
    pool = [record.case_id for group in remaining for record in groups[group]]
    val_ids = {record.case_id for group in val_groups for record in groups[group]}
    return Fold(
        train_ids=tuple(sorted(set(pool) - val_ids)),
        val_ids=tuple(sorted(val_ids)),
        test_ids=tests,
        held_out=held_out,
    )


def _system_strata(groups: Mapping[str, Sequence[CaseRecord]]) -> dict[str, str]:
    return {group: members[0].system for group, members in groups.items()}


def random_split(
    records: Sequence[CaseRecord], n_folds: int, val_fraction: float, seed: int
) -> Split:
    """Split by case, ignoring groups. For the leakage demonstration only."""
    if len(records) < n_folds:
        raise SplitError(f"random: {len(records)} case(s) cannot fill {n_folds} test folds")
    name: SplitStrategy = "random"
    singletons = {record.case_id: [record] for record in records}
    strata = dict.fromkeys(singletons, "all")
    order = _shuffled(singletons, _rng(name, "folds", 0, seed))
    folds = []
    for index in range(n_folds):
        test = set(order[index::n_folds])
        rng = _rng(name, "validation", index, seed)
        folds.append(
            _fold(f"random fold {index}", singletons, test, test, strata, val_fraction, rng)
        )
    return Split(name=name, seed=seed, leaky=True, folds=tuple(folds))


def _assign_group_folds(
    groups: Mapping[str, Sequence[CaseRecord]],
    strata: Mapping[str, str],
    n_folds: int,
    rng: np.random.Generator,
) -> list[list[str]]:
    """Deal groups into folds, largest first.

    Within a stratum, balance services first, so a label's test cases never crowd one fold, then
    fault types, then the stratum and the fold as a whole.
    """
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    cases: Counter[tuple[int, str, str]] = Counter()
    for stratum in sorted(set(strata.values())):
        members = _shuffled((group for group in groups if strata[group] == stratum), rng)
        members.sort(key=lambda group: len(groups[group]), reverse=True)
        offset = int(rng.integers(n_folds))
        for group in members:
            first = groups[group][0]
            keys = (
                ("service", f"{stratum}/{first.root_cause_service}"),
                ("fault", f"{stratum}/{first.fault_type}"),
                ("stratum", stratum),
                ("fold", ""),
            )
            costs = [
                ([cases[fold, *key] for key in keys], (fold - offset) % n_folds, fold)
                for fold in range(n_folds)
            ]
            target = min(costs)[2]
            folds[target].append(group)
            for key in keys:
                cases[target, *key] += len(groups[group])
    return folds


def group_run_split(
    records: Sequence[CaseRecord],
    group_map: GroupMap,
    n_folds: int,
    val_fraction: float,
    seed: int,
) -> Split:
    """Grouped K-fold on ``group_id``, balanced by system and service where group counts allow."""
    name: SplitStrategy = "group_run"
    groups = _groups_of(records, group_map)
    if len(groups) < n_folds:
        raise SplitError(f"group_run: {len(groups)} group(s) cannot fill {n_folds} test folds")
    strata = _system_strata(groups)
    assigned = _assign_group_folds(groups, strata, n_folds, _rng(name, "folds", 0, seed))
    folds = []
    for index, test_groups in enumerate(assigned):
        test = [record.case_id for group in test_groups for record in groups[group]]
        rng = _rng(name, "validation", index, seed)
        label = f"group_run fold {index}"
        folds.append(_fold(label, groups, test, set(test_groups), strata, val_fraction, rng))
    return Split(name=name, seed=seed, folds=tuple(folds))


def _leave_one_out(
    name: SplitStrategy,
    records: Sequence[CaseRecord],
    group_map: GroupMap,
    key: Callable[[CaseRecord], str],
    val_fraction: float,
    seed: int,
) -> Split:
    """Hold out each value of ``key``; groups that touch it are removed from training entirely."""
    groups = _groups_of(records, group_map)
    strata = _system_strata(groups)
    folds = []
    for index, value in enumerate(sorted({key(record) for record in records})):
        held = [record for record in records if key(record) == value]
        test_groups = {group_map[record.case_id] for record in held}
        rng = _rng(name, "validation", index, seed)
        label = f"{name} fold {index} ({value})"
        folds.append(
            _fold(
                label,
                groups,
                [record.case_id for record in held],
                test_groups,
                strata,
                val_fraction,
                rng,
                held_out=value,
            )
        )
    return Split(name=name, seed=seed, folds=tuple(folds))


def lofo_split(
    records: Sequence[CaseRecord], group_map: GroupMap, val_fraction: float, seed: int
) -> Split:
    """Leave one fault type out."""
    return _leave_one_out(
        "lofo", records, group_map, lambda record: record.fault_type, val_fraction, seed
    )


def loso_split(
    records: Sequence[CaseRecord], group_map: GroupMap, val_fraction: float, seed: int
) -> Split:
    """Leave one system out."""
    return _leave_one_out(
        "loso", records, group_map, lambda record: record.system, val_fraction, seed
    )


def make_split(
    strategy: SplitStrategy,
    records: Sequence[CaseRecord],
    group_map: GroupMap,
    config: SplitsConfig,
) -> Split:
    """Run the splitter named by ``strategy`` with the configured parameters."""
    if strategy == "random":
        return random_split(records, config.n_folds, config.val_fraction, config.seed)
    if strategy == "group_run":
        return group_run_split(records, group_map, config.n_folds, config.val_fraction, config.seed)
    if strategy == "lofo":
        return lofo_split(records, group_map, config.val_fraction, config.seed)
    return loso_split(records, group_map, config.val_fraction, config.seed)
