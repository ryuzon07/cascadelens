"""The clean-data ranker experiment: S1, S2, the ablation, retrieval and importance.

Baselines are read from their cached predictions, never rerun, so this experiment leaves Parts 4
and 5 untouched and compares against exactly what they reported. The S2 comparator is the best
baseline of any kind at evaluation time, as PREREGISTRATION.md requires.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from cascadelens.baselines.experiments import (
    METRICS,
    ExperimentConfig,
    MethodRuns,
    _averaged,
    _cell,
    _fold_rows,
    _paired,
    _pct,
    _summaries,
    _table,
    _with_held_out,
    _write_text,
)
from cascadelens.baselines.runner import RUNS_DIR as BASELINE_RUNS_DIR
from cascadelens.baselines.runner import Dataset, load_dataset, load_split, truth_table
from cascadelens.config import config_hash, load_config
from cascadelens.eval.bootstrap import paired_group_bootstrap
from cascadelens.eval.metrics import per_case_outcomes
from cascadelens.eval.plots import bars_with_ci
from cascadelens.eval.significance import wilcoxon_signed_rank
from cascadelens.models.data import RankerError
from cascadelens.models.freeze import frozen_choices, resolve
from cascadelens.models.importance import (
    gain_shares,
    group_shares,
    mean_shares,
    structural_warnings,
    top_columns,
)
from cascadelens.models.pipeline import (
    FAMILIES,
    evaluate,
    load_ranker_config,
    make_ranker,
    stage1_table,
)
from cascadelens.models.ranker import FittedFold
from cascadelens.results import write_result

EXPERIMENT_DIR = Path("experiments/ranker_clean")
BASELINE_RESULTS = (
    Path("experiments/baselines_clean/results.json"),
    Path("experiments/reproduction/evaluation/results.json"),
)
PRIMARY_FAMILY_SIZE = 5
RANKER, SINGLE = "two_stage_ranker", "single_stage_ranker"


class RankerExperimentConfig(ExperimentConfig):
    """Settings of the experiment."""

    s2_margin: float = 0.12
    prior: str = "frequency"
    ranker_config: str = "configs/ranker.yaml"


@dataclass(frozen=True)
class SeedRuns:
    """One method on one split, over every seed."""

    outcomes: pl.DataFrame
    stage1: pl.DataFrame
    predictions: pl.DataFrame
    details: list[list[dict[str, Any]]]
    folds: list[list[FittedFold]]


def best_baseline(root: Path = Path()) -> tuple[str, float]:
    """Return the baseline of any kind with the best Top-1 on group_run."""
    scores: dict[str, float] = {}
    for path in BASELINE_RESULTS:
        with (root / path).open(encoding="utf-8") as handle:
            rows = json.load(handle)["metrics"]["group_run"]
        scores |= {row["method"]: row["Top-1"]["estimate"] for row in rows}
    if not scores:
        raise RankerError("No baseline results to compare with")
    name = max(sorted(scores), key=lambda method: scores[method])
    return name, scores[name]


def cached_outcomes(
    method: str, split: str, dataset: Dataset, truth: pl.DataFrame, root: Path = Path()
) -> pl.DataFrame:
    """Return per-case outcomes of a baseline from its cached predictions."""
    directory = root / BASELINE_RUNS_DIR / method / split
    frames = []
    for path in sorted(directory.glob("seed_*/cache/predictions.parquet")):
        result = json.loads((path.parents[1] / "result.json").read_text(encoding="utf-8"))
        if result["data_checksum"] != dataset.checksum:
            raise RankerError(f"{path} was made from another dataset version")
        predictions = pl.read_parquet(path)
        frames.append(
            per_case_outcomes(predictions, truth).with_columns(seed=pl.lit(result["seed"]))
        )
    if not frames:
        raise RankerError(f"No cached predictions in {directory}; run the baseline first")
    return pl.concat(frames)


def compare(
    first: pl.DataFrame,
    second: pl.DataFrame,
    column: str,
    config: RankerExperimentConfig,
    margin: float = 0.0,
) -> dict[str, float]:
    """Return ``first + margin - second`` with the paired bootstrap and the Wilcoxon check."""
    a, b = _averaged(first, column), _averaged(second, column)
    if a["case_id"].to_list() != b["case_id"].to_list():
        raise RankerError("Compared methods must cover the same cases")
    shifted = a["value"] + margin
    result = paired_group_bootstrap(
        shifted, b["value"], a["group_id"], config.n_resamples, config.seed, config.alpha
    )
    check = wilcoxon_signed_rank(shifted, b["value"], a["group_id"])
    return {
        **_paired(result),
        "wilcoxon_p": check.p_value,
        "holm_upper_bound": min(1.0, PRIMARY_FAMILY_SIZE * result.p_value),
    }


def _seed_runs(
    method: str,
    split: str,
    config_path: Path,
    exploratory: bool,
    dataset: Dataset,
    root: Path,
) -> SeedRuns:
    ranker_config = load_ranker_config(config_path, root)
    _, record = resolve(ranker_config, (split,), (method,), exploratory, root)
    _, group_map = load_split(split, dataset, root)
    model = make_ranker(
        method,
        ranker_config,
        dataset,
        group_map,
        frozen_choices(record, split, method),
        root,
    )
    outcomes, stage1, details, folds, predictions = [], None, [], [], None
    for seed in ranker_config.seeds:
        result = evaluate(config_path, split, seed, method, exploratory, root, model)
        outcomes.append(result.run.outcomes.with_columns(seed=pl.lit(seed)))
        stage1 = stage1_table(result.folds)
        predictions = result.run.predictions
        metrics = result.run.metrics["fold_details"]
        assert isinstance(metrics, list)
        details.append(metrics)
        folds.append(result.folds)
    assert stage1 is not None and predictions is not None
    return SeedRuns(pl.concat(outcomes), stage1, predictions, details, folds)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def fold_report(
    method: str, split: str, runs: SeedRuns, sizes: dict[int, int]
) -> list[dict[str, Any]]:
    """Average per-fold details over seeds. K and hyperparameters are shared across seeds."""
    rows = []
    for index, first in enumerate(runs.details[0]):
        seeds = [details[index] for details in runs.details]
        row: dict[str, Any] = {
            "method": method,
            "split": split,
            "fold": index,
            "cases": sizes[index],
            "k": first["k"],
            "params": first["params"],
            "best_iteration": _mean([s["best_iteration"] for s in seeds]),
        }
        for key in ("search_val_top1", "val_top1", "test_top1", "val_test_gap"):
            row[key] = _mean([s[key] for s in seeds])
        for key in first:
            if key.startswith("recall"):
                row[key] = _mean([s[key] for s in seeds])
        rows.append(row)
    return rows


def _weighted(rows: Sequence[dict[str, Any]], key: str) -> float:
    total = sum(row["cases"] for row in rows)
    return float(sum(row[key] * row["cases"] for row in rows) / total)


def importance_report(
    method: str, split: str, runs: SeedRuns, threshold: float
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return gain shares per fold, averaged over seeds, and any structural warnings."""
    rows = []
    labelled = []
    for index in range(len(runs.folds[0])):
        shares = mean_shares(gain_shares(seeded[index].booster) for seeded in runs.folds)
        rows.append(
            {
                "method": method,
                "split": split,
                "fold": index,
                "groups": group_shares(shares),
                "top_columns": top_columns(shares),
            }
        )
        labelled.append((f"{method} {split} fold {index}", shares))
    return rows, structural_warnings(labelled, threshold)


def _pooled_group_importance(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    return mean_shares(row["groups"] for row in rows)


def _pooled_row(
    method: str, runs: SeedRuns, floor: float, config: RankerExperimentConfig
) -> dict[str, Any]:
    return {
        "method": method,
        "split": "group_run",
        "fold": "pooled",
        "cases": runs.outcomes["case_id"].n_unique(),
        "floor": floor,
        **_summaries(runs.outcomes, config),
    }


def _all_services(runs: SeedRuns) -> pl.DataFrame:
    return runs.stage1 if not runs.stage1.is_empty() else runs.predictions


def _floor(runs: SeedRuns) -> float:
    sizes = _all_services(runs).group_by("case_id").len()
    return float((1 / sizes["len"]).mean())  # type: ignore[arg-type]


def _difference_line(label: str, entry: dict[str, float]) -> str:
    return (
        f"| {label} | {_pct(entry['estimate'])} [{_pct(entry['lower'])}, {_pct(entry['upper'])}] "
        f"| {entry['p_value']:.4f} | {entry['wilcoxon_p']:.4f} |"
    )


def _gap_table(rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        "| Method | Split | Fold | K | Recall@K | Val Top-1 | Test Top-1 | Val minus test |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        recall = row.get("recall_at_chosen_k")
        lines.append(
            f"| {row['method']} | {row['split']} | {row['fold']} | {row['k'] or '-'} "
            f"| {'-' if recall is None else _pct(recall)} | {_pct(row['val_top1'])} "
            f"| {_pct(row['test_top1'])} | {_pct(row['val_test_gap'])} |"
        )
    return lines


def ranker_clean(exploratory: bool, root: Path = Path()) -> dict[str, Any]:
    """Run both rankers on every split family and compare them with the baselines."""
    directory = root / EXPERIMENT_DIR
    config = load_config(directory / "config.yaml", RankerExperimentConfig)
    config_path = Path(config.ranker_config)
    ranker_config = load_ranker_config(config_path, root)
    methods = tuple(config.methods)
    status, _ = resolve(ranker_config, FAMILIES, methods, exploratory, root)
    dataset = load_dataset(root)
    _, group_map = load_split("group_run", dataset, root)
    truth = truth_table(dataset.records, group_map)
    comparator, comparator_top1 = best_baseline(root)
    best = cached_outcomes(comparator, "group_run", dataset, truth, root)
    prior = cached_outcomes(config.prior, "group_run", dataset, truth, root)

    pooled: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    importance: list[dict[str, Any]] = []
    warnings: list[str] = []
    arms: dict[str, pl.DataFrame] = {}
    for method in methods:
        for split in FAMILIES:
            runs = _seed_runs(method, split, config_path, exploratory, dataset, root)
            sizes = dict(runs.outcomes.group_by("fold").agg(pl.col("case_id").n_unique()).rows())
            gaps += fold_report(method, split, runs, sizes)
            rows, messages = importance_report(
                method, split, runs, ranker_config.structural_dominance
            )
            importance += rows
            warnings += messages
            if split == "group_run":
                arms[method] = runs.outcomes
                pooled.append(_pooled_row(method, runs, _floor(runs), config))
            else:
                held = MethodRuns(method, split, runs.outcomes, _all_services(runs))
                fold_rows += _fold_rows(method, _with_held_out(held, root, split), split, config)

    primary = arms[RANKER]
    comparisons: dict[str, Any] = {
        "S1_vs_frequency": {
            label: compare(primary, prior, column, config) for column, label in METRICS
        },
        "S2_vs_best_baseline": compare(primary, best, "hit@1", config, config.s2_margin),
        "S2_unshifted_difference": compare(primary, best, "hit@1", config),
        "two_stage_vs_single_stage": {
            label: compare(primary, arms[SINGLE], column, config) for column, label in METRICS
        },
    }
    s1 = comparisons["S1_vs_frequency"]["Top-1"]
    s2 = comparisons["S2_vs_best_baseline"]
    recalls = {
        split: _weighted(
            [g for g in gaps if g["method"] == RANKER and g["split"] == split],
            "recall_at_chosen_k",
        )
        for split in FAMILIES
    }
    payload: dict[str, Any] = {
        "status": status,
        "ranker_config_hash": config_hash(ranker_config),
        "s2_comparator": {"method": comparator, "top1": comparator_top1},
        "s2_margin": config.s2_margin,
        "s1_interval_excludes_zero": s1["lower"] > 0 or s1["upper"] < 0,
        "s2_interval_above_zero": s2["lower"] > 0,
        "group_run": pooled,
        "folds": fold_rows,
        "fold_details": gaps,
        "retrieval_recall_at_chosen_k": recalls,
        "comparisons": comparisons,
        "importance": importance,
        "group_importance_group_run": {
            method: _pooled_group_importance(
                [r for r in importance if r["method"] == method and r["split"] == "group_run"]
            )
            for method in methods
        },
        "warnings": warnings,
    }
    chart = [
        *pooled,
        {"method": comparator, **_summaries(best, config)},
        {"method": config.prior, **_summaries(prior, config)},
    ]
    bars_with_ci(
        [str(row["method"]) for row in chart],
        [100 * row["Top-1"]["estimate"] for row in chart],
        [100 * row["Top-1"]["lower"] for row in chart],
        [100 * row["Top-1"]["upper"] for row in chart],
        "Top-1 (%)",
        f"Top-1 on group_run, 95% group-bootstrap intervals ({status})",
        directory / "figures" / "ranker_top1",
    )
    _write_text(directory / "results.md", _report(payload, chart, config) + "\n")
    write_result(directory / "results.json", payload, config, config.seed, dataset.checksum)
    return payload


def _report(
    payload: dict[str, Any], chart: Sequence[dict[str, Any]], config: RankerExperimentConfig
) -> str:
    comparisons = payload["comparisons"]
    status = payload["status"]
    comparator = payload["s2_comparator"]["method"]
    lines = [
        "# Clean-data ranker",
        "",
        f"Generated by `cascadelens experiments ranker-clean`. **Status: {status}.**",
        "",
    ]
    if status != "frozen":
        lines += [
            "The config is not frozen, so nothing here is a pre-registered test result. S1 and S2 "
            "below are previews of the planned tests.",
            "",
        ]
    lines += [
        "Scores are percentages with 95% group-bootstrap intervals, averaged per case over seeds. "
        "A case whose truth was not retrieved is a miss. Comparisons are paired, with a Wilcoxon "
        "check on group means. Holm over the five primary criteria needs S3, S4 and S7, so raw "
        "p-values are shown here and `holm_upper_bound` in the JSON is the worst case.",
        "",
        "## Group_run",
        "",
        "| Method | Top-1 | Top-3 | Avg@5 |",
        "| --- | --- | --- | --- |",
    ]
    for row in chart:
        cells = " | ".join(_cell(row[label]) for _, label in METRICS)
        lines.append(f"| {row['method']} | {cells} |")
    lines += [
        "",
        "![Top-1 on group_run](figures/ranker_top1.png)",
        "",
        "## Paired comparisons on group_run",
        "",
        "| Comparison | Difference, points [95% CI] | Bootstrap p | Wilcoxon p |",
        "| --- | --- | --- | --- |",
        _difference_line(
            "S1: ranker minus frequency, Top-1", comparisons["S1_vs_frequency"]["Top-1"]
        ),
        _difference_line(
            f"S2: ranker minus {comparator}, plus {_pct(config.s2_margin)} margin",
            comparisons["S2_vs_best_baseline"],
        ),
        _difference_line(
            f"Ranker minus {comparator}, no margin", comparisons["S2_unshifted_difference"]
        ),
    ]
    lines += [
        _difference_line(f"Two-stage minus single-stage, {label}", entry)
        for label, entry in comparisons["two_stage_vs_single_stage"].items()
    ]
    lines += ["", "S2 needs the margin-shifted interval to lie above zero.", ""]
    lines += _table("LOFO and LOSO per fold", payload["folds"])
    lines += [
        "The random floor is the expected Top-1 of picking uniformly among a case's services.",
        "",
        "## Validation versus test, and retrieval recall",
        "",
        "Means over seeds. K and hyperparameters were chosen on validation for each fold. "
        "Validation also set early stopping, so it runs optimistic. Recall@K bounds Top-1.",
        "",
        *_gap_table(payload["fold_details"]),
        "",
        "## Gain share by feature group, group_run",
        "",
        "| Method | Group | Share of gain |",
        "| --- | --- | --- |",
    ]
    for method, groups in payload["group_importance_group_run"].items():
        lines += [f"| {method} | {g} | {_pct(share)} |" for g, share in groups.items()]
    lines += ["", "Per-fold shares and top columns are in `results.json`.", "", "## Warnings", ""]
    lines += [f"- {message}" for message in payload["warnings"]] or ["None."]
    return "\n".join(lines)
