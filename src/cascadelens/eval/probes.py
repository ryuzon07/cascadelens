"""The four leakage probes of PRD 6.5. Each writes a JSON result and a short Markdown note.

| Probe | Pass condition from the PRD |
| --- | --- |
| Identity | Shuffling graph features across services makes a small drop |
| Graph-only | Graph features alone stay near the frequency prior on group_run and LOSO |
| Window leak | Accuracy falls smoothly with offset, with no cliff at zero |
| Mask-only | Reported, and flagged if strongly predictive on clean data |
"""

import itertools
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from cascadelens.baselines.base import CaseHandle
from cascadelens.baselines.runner import load_split, truth_table
from cascadelens.config import BaselineConfig, RankerConfig, StrictModel, load_config
from cascadelens.eval.bootstrap import group_bootstrap_ci, paired_group_bootstrap
from cascadelens.eval.metrics import per_case_outcomes
from cascadelens.eval.robustness import (
    CLEAN_POINT,
    Context,
    Point,
    build_arm,
    derive_points,
    load_context,
    point_key,
    seed_averaged,
    write_json,
)
from cascadelens.features.config import GroupName
from cascadelens.features.registry import REGISTRY, feature_columns
from cascadelens.models.data import FeatureSource, RankerError, TableSource
from cascadelens.models.ranker import TwoStageRanker
from cascadelens.results import write_result
from cascadelens.seed import derive_case_seed
from cascadelens.splits.report import assert_no_group_overlap
from cascadelens.splits.splitters import Fold

PROBES_DIR = Path("experiments/probes")
FoldCases = tuple[Sequence[CaseHandle], Sequence[CaseHandle], Sequence[CaseHandle]]
GRAPH_GROUPS: list[GroupName] = ["graph", "structural"]


class ProbesConfig(StrictModel):
    """Settings of the probes."""

    seed: int = 0
    n_resamples: int = 10_000
    alpha: float = 0.05
    identity_drop_limit: float = 0.05
    graph_floor_margin: float = 0.10
    cliff_threshold: float = 0.05
    offsets: list[int]
    random_window_jitter: list[int]
    window_methods: list[str]


class ShuffledSource:
    """A table source with some columns shuffled across services, within each case."""

    def __init__(self, base: TableSource, columns: Sequence[str], seed: int = 0) -> None:
        self.base = base
        self.columns = list(columns)
        self.seed = seed

    def table(self, case: CaseHandle) -> pl.DataFrame:
        """Return the table with one permutation per column."""
        table = self.base.table(case)
        rng = np.random.default_rng(
            derive_case_seed(case.record.case_id, "identity_probe", {}, self.seed)
        )
        shuffled = [
            pl.Series(column, table[column].to_numpy()[rng.permutation(table.height)])
            for column in self.columns
        ]
        return table.with_columns(shuffled)


def score_ranker(
    source: TableSource,
    settings: RankerConfig,
    folds: Sequence[FoldCases],
    groups: Mapping[str, str],
    ranker_type: type[TwoStageRanker] = TwoStageRanker,
) -> pl.DataFrame:
    """Fit a ranker on every fold and seed, and return per-case outcomes on test cases."""
    ranker = ranker_type(source, groups)
    config = BaselineConfig(
        name=ranker.name,
        supervised=True,
        seeds=settings.seeds,
        params=settings.model_dump(mode="json"),
    )
    frames = []
    for index, (train, val, test) in enumerate(folds):
        truth = pl.DataFrame(
            {
                "case_id": [c.record.case_id for c in test],
                "root_cause_service": [c.record.root_cause_service for c in test],
                "group_id": [groups.get(c.record.case_id, c.record.run_id) for c in test],
            }
        )
        for seed in settings.seeds:
            ranker.fit(train, config, val, seed)
            predictions = pl.concat(ranker.predict(case) for case in test)
            ranker.take_folds()
            frames.append(
                per_case_outcomes(predictions, truth).with_columns(
                    fold=pl.lit(index), seed=pl.lit(seed)
                )
            )
    return pl.concat(frames)


def top1_ci(outcomes: pl.DataFrame, config: ProbesConfig) -> dict[str, float]:
    """Return seed-averaged Top-1 with its group-bootstrap interval."""
    averaged = seed_averaged(outcomes, "hit@1")
    result = group_bootstrap_ci(
        averaged["value"], averaged["group_id"], config.n_resamples, config.seed, config.alpha
    )
    return {"estimate": result.estimate, "lower": result.lower, "upper": result.upper}


def paired_drop(
    base: pl.DataFrame, shuffled: pl.DataFrame, config: ProbesConfig
) -> dict[str, float]:
    """Return base minus shuffled Top-1, paired over seed-averaged cases."""
    first, second = seed_averaged(base, "hit@1"), seed_averaged(shuffled, "hit@1")
    if first["case_id"].to_list() != second["case_id"].to_list():
        raise RankerError("Paired probes must cover the same cases")
    result = paired_group_bootstrap(
        first["value"],
        second["value"],
        first["group_id"],
        config.n_resamples,
        config.seed,
        config.alpha,
    )
    return {
        "estimate": result.estimate,
        "lower": result.lower,
        "upper": result.upper,
        "p_value": result.p_value,
    }


def identity_probe(
    source: TableSource,
    settings: RankerConfig,
    folds: Sequence[FoldCases],
    groups: Mapping[str, str],
    config: ProbesConfig,
) -> dict[str, Any]:
    """Compare a ranker with versions whose graph columns are shuffled within each case."""
    base = score_ranker(source, settings, folds, groups)
    graph = feature_columns(GRAPH_GROUPS)
    structural = list(REGISTRY["structural"].columns)
    result: dict[str, Any] = {"base_top1": top1_ci(base, config)}
    for label, columns in (("graph_and_structural", graph), ("structural_only", structural)):
        shuffled = score_ranker(ShuffledSource(source, columns), settings, folds, groups)
        drop = paired_drop(base, shuffled, config)
        result[label] = {"top1": top1_ci(shuffled, config), "drop": drop}
    result["passed"] = bool(
        result["graph_and_structural"]["drop"]["estimate"] < config.identity_drop_limit
        and result["structural_only"]["drop"]["estimate"] < config.identity_drop_limit
    )
    return result


def uniform_floor(cases: Sequence[int]) -> float:
    """Return the expected Top-1 of picking uniformly among each case's services."""
    return float(np.mean([1 / count for count in cases]))


def flagged_above_floor(top1: Mapping[str, float], floor: float) -> bool:
    """Return whether the interval lies clearly above the floor."""
    return top1["lower"] > floor


def detect_cliff(curve: Mapping[int, float], threshold: float) -> dict[str, Any]:
    """Return whether accuracy spikes at offset 0 relative to its neighbours and the curve.

    A cliff is a gap at zero above ``threshold`` that is also over twice the largest step between
    other adjacent offsets. A model that decays smoothly with distance has neither.
    """
    if 0 not in curve:
        raise RankerError("The window curve needs offset 0")
    ordered = sorted(curve)
    position = ordered.index(0)
    neighbours = [curve[ordered[i]] for i in (position - 1, position + 1) if 0 <= i < len(ordered)]
    gap = curve[0] - float(np.mean(neighbours))
    away = [o for o in ordered if o != 0]
    steps = [
        abs(curve[first] - curve[second])
        for first, second in itertools.pairwise(away)
        if first * second > 0
    ]
    largest = max(steps, default=0.0)
    return {
        "gap_at_zero": gap,
        "largest_other_step": largest,
        "cliff": bool(gap > threshold and gap > 2 * largest),
    }


def injection_position(records: Sequence[Any]) -> dict[str, float]:
    """Return where the injection sits in the window, for the window-definition report."""
    positions = np.array(
        [(r.inject_time - r.window_start) / (r.window_end - r.window_start) for r in records]
    )
    return {
        "median": float(np.median(positions)),
        "share_within_2pct_of_middle": float(np.mean(np.abs(positions - 0.5) <= 0.02)),
    }


def _write(
    name: str, payload: dict[str, Any], note: str, config: ProbesConfig, root: Path, checksum: str
) -> None:
    directory = root / PROBES_DIR
    write_result(directory / f"{name}.json", payload, config, config.seed, checksum)
    path = directory / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(note + "\n")


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def _ci(entry: Mapping[str, float]) -> str:
    return f"{_pct(entry['estimate'])} [{_pct(entry['lower'])}, {_pct(entry['upper'])}]"


def _folds(split: str, context: Context) -> tuple[list[FoldCases], dict[str, str]]:
    saved, groups = load_split(split, context.dataset, context.root)
    folds: list[FoldCases] = []
    for fold in saved.folds:
        assert_no_group_overlap(fold.val_ids, fold.test_ids, groups)
        handles = context.dataset.handles
        folds.append((handles(fold.train_ids), handles(fold.val_ids), handles(fold.test_ids)))
    return folds, groups


def _settings(context: Context, groups: list[GroupName] | None = None) -> RankerConfig:
    if groups is None:
        return context.ranker
    return RankerConfig.model_validate({**context.ranker.model_dump(mode="json"), "groups": groups})


def _load(exploratory: bool, root: Path) -> tuple[ProbesConfig, Context, FeatureSource]:
    config = load_config(root / PROBES_DIR / "config.yaml", ProbesConfig)
    context = load_context(exploratory, root)
    source = FeatureSource(context.features, root)
    source.verify(context.dataset.checksum)
    return config, context, source


def run_identity(exploratory: bool, root: Path = Path()) -> dict[str, Any]:
    """Run the identity probe on group_run."""
    config, context, source = _load(exploratory, root)
    folds, groups = _folds("group_run", context)
    result = identity_probe(source, context.ranker, folds, groups, config)
    result["status"] = context.status
    note = (
        f"# Identity probe ({context.status})\n\n"
        "Pass condition (PRD 6.5): shuffling graph features across services within a case makes "
        f"a small drop, here under {_pct(config.identity_drop_limit)} points.\n\n"
        f"Base Top-1: {_ci(result['base_top1'])}.\n\n"
        f"Drop with graph and structural shuffled: {_ci(result['graph_and_structural']['drop'])}."
        f"\n\nDrop with structural only shuffled: {_ci(result['structural_only']['drop'])}.\n\n"
        f"**{'Passed' if result['passed'] else 'Failed'}.**"
    )
    _write("identity", result, note, config, root, context.dataset.checksum)
    return result


def _case_sizes(source: TableSource, folds: Sequence[FoldCases]) -> list[int]:
    return [source.table(case).height for _, _, test in folds for case in test]


def _prior_top1(split: str, context: Context) -> float:
    path = context.root / "experiments/baselines_clean/results.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    if split == "group_run":
        return float(
            next(r for r in rows["group_run"] if r["method"] == "frequency")["Top-1"]["estimate"]
        )
    entries = [r for r in rows["folds"] if r["method"] == "frequency" and r["split"] == split]
    return float(np.mean([r["floor"] for r in entries]))


def run_graph_only(exploratory: bool, root: Path = Path()) -> dict[str, Any]:
    """Train on graph and structural features alone, on group_run and LOSO."""
    config, context, source = _load(exploratory, root)
    settings = _settings(context, GRAPH_GROUPS)
    result: dict[str, Any] = {"status": context.status}
    lines = [f"# Graph-only probe ({context.status})", ""]
    lines += [
        "Pass condition (PRD 6.5): near the frequency-prior floor on group_run and LOSO, here "
        f"within {_pct(config.graph_floor_margin)} points.",
        "",
        "Under LOSO the frequency prior is undefined, so its value here is the uniform floor.",
        "",
    ]
    passed = True
    for split in ("group_run", "loso"):
        folds, groups = _folds(split, context)
        outcomes = score_ranker(source, settings, folds, groups)
        top1 = top1_ci(outcomes, config)
        prior = _prior_top1(split, context)
        floor = uniform_floor(_case_sizes(source, folds))
        near = bool(top1["estimate"] <= prior + config.graph_floor_margin)
        passed = passed and near
        result[split] = {
            "top1": top1,
            "frequency_prior": prior,
            "uniform_floor": floor,
            "near": near,
        }
        lines.append(
            f"{split}: Top-1 {_ci(top1)}, frequency prior {_pct(prior)}, uniform floor "
            f"{_pct(floor)}, {'near' if near else 'not near'} the floor."
        )
        lines.append("")
    result["passed"] = passed
    lines.append(f"**{'Passed' if passed else 'Failed'}.**")
    _write("graph_only", result, "\n".join(lines), config, root, context.dataset.checksum)
    return result


def run_mask_only(exploratory: bool, root: Path = Path()) -> dict[str, Any]:
    """Train on mask features alone on clean data."""
    config, context, source = _load(exploratory, root)
    folds, groups = _folds("group_run", context)
    outcomes = score_ranker(source, _settings(context, ["mask"]), folds, groups)
    top1 = top1_ci(outcomes, config)
    floor = uniform_floor(_case_sizes(source, folds))
    flagged = flagged_above_floor(top1, floor)
    result = {
        "status": context.status,
        "top1": top1,
        "uniform_floor": floor,
        "flagged": flagged,
        "passed": not flagged,
    }
    note = (
        f"# Mask-only probe ({context.status})\n\n"
        "Pass condition (PRD 6.5): reported; if strongly predictive on clean data, flagged as a "
        "possible collection artifact. Flagged here when the interval clears the uniform floor."
        f"\n\nTop-1 {_ci(top1)}, uniform floor {_pct(floor)}.\n\n"
        f"**{'Flagged as a possible collection artifact' if flagged else 'Not flagged'}.**"
    )
    _write("mask_only", result, note, config, root, context.dataset.checksum)
    return result


def window_points(config: ProbesConfig, realism: str = "simulated") -> list[Point]:
    """Return the offsets, and random window bounds."""
    points = [
        Point(point_key("P4", {"offset_seconds": o}), "P4", {"offset_seconds": o}, realism)
        for o in config.offsets
    ]
    points += [
        Point(point_key("P5", {"max_jitter_seconds": j}), "P5", {"max_jitter_seconds": j}, realism)
        for j in config.random_window_jitter
    ]
    return points


def scored_top1(
    fit: Callable[[Fold, int], None],
    score: Callable[[Any, Sequence[str]], pl.DataFrame],
    folds: Sequence[Fold],
    seeds: Sequence[int],
    points: Sequence[Point],
    derived: Callable[[Point], Any],
    truth: pl.DataFrame,
) -> dict[str, float]:
    """Return mean Top-1 at every point, fitting on clean cases and scoring test cases."""
    hits: dict[str, list[float]] = {point.key: [] for point in points}
    for fold in folds:
        tested = truth.filter(pl.col("case_id").is_in(fold.test_ids))
        for seed in seeds:
            fit(fold, seed)
            for point in points:
                predictions = score(derived(point), fold.test_ids)
                outcomes = per_case_outcomes(predictions, tested)
                hits[point.key] += outcomes["hit@1"].cast(pl.Float64).to_list()
    return {key: float(np.mean(values)) for key, values in hits.items()}


def run_window(exploratory: bool, root: Path = Path()) -> dict[str, Any]:
    """Score trained models across window offsets and random bounds."""
    config, context, _ = _load(exploratory, root)
    points = window_points(config)
    suite_seed = context.perturb_seed
    derive_points(
        points,
        context.features,
        suite_seed,
        context.dataset,
        root,
        context.config.workers,
    )
    split, groups = load_split("group_run", context.dataset, root)
    truth = truth_table(context.dataset.records, groups)
    every = [CLEAN_POINT, *points]
    result: dict[str, Any] = {
        "status": context.status,
        "injection_position": injection_position(list(context.dataset.records.values())),
        "methods": {},
    }
    lines = [
        f"# Window-leak probe ({context.status})",
        "",
        "Pass condition (PRD 6.5): accuracy falls smoothly with offset; a cliff at zero means the "
        "window encodes `inject_time`. Windows are cropped before features. Simulated.",
        "",
        "Window bounds are computed from the recording, without reading `inject_time`, but the "
        f"recordings were centred on the injection: median position "
        f"{result['injection_position']['median']:.3f}, and "
        f"{_pct(result['injection_position']['share_within_2pct_of_middle'])}% of cases within 2% "
        "of the middle. So the definition has no direct dependence and a strong indirect one.",
        "",
        "| Method | Clean | Cliff | " + " | ".join(p.key for p in points) + " |",
        "| --- | --- | --- | " + " | ".join("---" for _ in points) + " |",
    ]
    passed = True
    for name in config.window_methods:
        variant = context.config.variant(name)
        arm = build_arm(variant, "group_run", groups, context)
        top1 = scored_top1(
            arm.fit, arm.score, split.folds, arm.seeds, every, context.derived, truth
        )
        context.tables.clear()
        curve = {0: top1[CLEAN_POINT.key]}
        curve |= {o: top1[point_key("P4", {"offset_seconds": o})] for o in config.offsets}
        cliff = detect_cliff(curve, config.cliff_threshold)
        passed = passed and not cliff["cliff"]
        result["methods"][name] = {
            "top1": top1,
            "curve": {str(k): v for k, v in curve.items()},
            **cliff,
        }
        cells = " | ".join(_pct(top1[p.key]) for p in points)
        lines.append(f"| {name} | {_pct(top1['clean'])} | {cliff['cliff']} | {cells} |")
    result["passed"] = passed
    lines += ["", f"**{'Passed' if passed else 'Failed'}.**"]
    _write("window", result, "\n".join(lines), config, root, context.dataset.checksum)
    return result


def run_all(exploratory: bool, root: Path = Path()) -> dict[str, dict[str, Any]]:
    """Run every probe and write their index."""
    results = {
        "identity": run_identity(exploratory, root),
        "graph_only": run_graph_only(exploratory, root),
        "window": run_window(exploratory, root),
        "mask_only": run_mask_only(exploratory, root),
    }
    write_json(
        root / PROBES_DIR / "index.json",
        {name: entry["passed"] for name, entry in results.items()},
    )
    return results
