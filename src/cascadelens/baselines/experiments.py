"""The leakage demonstration and the clean-data baseline tables.

Each experiment folder holds ``config.yaml``, a result JSON and a generated Markdown file.
Stochastic methods run over their configured seeds; per-case outcomes are averaged over seeds and
then bootstrapped over groups (``eval.seeds``). Every interval resamples groups, on every split, so
the random split is judged by the same unit as the grouped ones.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from cascadelens.baselines.base import SummaryCache
from cascadelens.baselines.runner import BASELINES, load_baseline_config, run_baseline
from cascadelens.config import StrictModel, load_config
from cascadelens.data.io import read_manifest
from cascadelens.eval.bootstrap import BootstrapResult, PairedResult, paired_group_bootstrap
from cascadelens.eval.plots import bars_with_ci
from cascadelens.eval.seeds import summarize_seeds
from cascadelens.results import write_result

METRICS = (("hit@1", "Top-1"), ("hit@3", "Top-3"), ("avg@5", "Avg@5"))
LEAKAGE_DIR = Path("experiments/leakage_demo")
CLEAN_DIR = Path("experiments/baselines_clean")


class ExperimentConfig(StrictModel):
    """Settings of one experiment."""

    seed: int = 0
    n_resamples: int = 10_000
    alpha: float = 0.05
    methods: list[str]


@dataclass(frozen=True)
class MethodRuns:
    """A method on a split: per-seed outcomes, stacked."""

    method: str
    split: str
    outcomes: pl.DataFrame
    predictions: pl.DataFrame


def run_seeds(method: str, split: str, root: Path, cache: SummaryCache) -> MethodRuns:
    """Run every configured seed of ``method``, reusing one instance so a fold searches once."""
    config = load_baseline_config(method, root)
    baseline = BASELINES[method](cache)
    frames = []
    predictions = None
    for seed in config.seeds:
        run = run_baseline(method, split, seed, root, baseline=baseline)
        frames.append(run.outcomes.with_columns(seed=pl.lit(seed)))
        predictions = run.predictions
    assert predictions is not None
    return MethodRuns(method, split, pl.concat(frames), predictions)


def _value_table(outcomes: pl.DataFrame, column: str) -> pl.DataFrame:
    return outcomes.select(
        "case_id", "group_id", "seed", pl.col(column).cast(pl.Float64).alias("value")
    )


def _summary(outcomes: pl.DataFrame, column: str, config: ExperimentConfig) -> dict[str, float]:
    seeds = outcomes["seed"].n_unique()
    summary = summarize_seeds(
        _value_table(outcomes, column), seeds, config.n_resamples, config.seed, config.alpha
    )
    return {
        "estimate": summary.ci.estimate,
        "lower": summary.ci.lower,
        "upper": summary.ci.upper,
        "seed_spread": summary.spread,
    }


def _summaries(outcomes: pl.DataFrame, config: ExperimentConfig) -> dict[str, dict[str, float]]:
    return {label: _summary(outcomes, column, config) for column, label in METRICS}


def _averaged(outcomes: pl.DataFrame, column: str) -> pl.DataFrame:
    return (
        outcomes.group_by("case_id", "group_id")
        .agg(pl.col(column).cast(pl.Float64).mean().alias("value"))
        .sort("case_id")
    )


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def _cell(entry: dict[str, float]) -> str:
    return f"{_pct(entry['estimate'])} [{_pct(entry['lower'])}, {_pct(entry['upper'])}]"


def _paired(difference: PairedResult) -> dict[str, float]:
    return {
        "estimate": difference.estimate,
        "lower": difference.lower,
        "upper": difference.upper,
        "p_value": difference.p_value,
    }


def _inflation(
    leaky: pl.DataFrame, grouped: pl.DataFrame, column: str, config: ExperimentConfig
) -> PairedResult:
    first, second = _averaged(leaky, column), _averaged(grouped, column)
    return paired_group_bootstrap(
        first["value"],
        second["value"],
        second["group_id"],
        config.n_resamples,
        config.seed,
        config.alpha,
    )


def _sentence(method: str, entries: dict[str, dict[str, float]], gap: BootstrapResult) -> str:
    verdict = (
        "inflated" if gap.lower > 0 else "lowered" if gap.upper < 0 else "had no clear effect on"
    )
    return (
        f"For {method}, Top-1 was {_pct(entries['random']['estimate'])}% on the random split and "
        f"{_pct(entries['group_run']['estimate'])}% on group_run, a paired gap of "
        f"{_pct(gap.estimate)} points (95% CI {_pct(gap.lower)} to {_pct(gap.upper)}), so the "
        f"random split {verdict} it."
    )


def leakage_demo(root: Path = Path(), cache: SummaryCache | None = None) -> dict[str, object]:
    """Compare each method on the random split and on group_run, with identical settings."""
    directory = root / LEAKAGE_DIR
    config = load_config(directory / "config.yaml", ExperimentConfig)
    store = cache if cache is not None else SummaryCache()
    runs = {
        (method, split): run_seeds(method, split, root, store)
        for method in config.methods
        for split in ("random", "group_run")
    }
    scores: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    gaps: dict[str, dict[str, dict[str, float]]] = {}
    lines = ["# Leakage demonstration", "", "Generated by `cascadelens experiments leakage-demo`."]
    table = ["", "| Method | Split | Top-1 | Top-3 | Avg@5 |", "| --- | --- | --- | --- | --- |"]
    gap_table = [
        "",
        "Paired difference, random minus group_run, in points:",
        "",
        "| Method | Metric | Difference [95% CI] | p |",
        "| --- | --- | --- | --- |",
    ]
    sentences = []
    for method in config.methods:
        leaky = runs[method, "random"].outcomes
        grouped = runs[method, "group_run"].outcomes
        scores[method] = {
            "random": _summaries(leaky, config),
            "group_run": _summaries(grouped, config),
        }
        gaps[method] = {}
        for split in ("random", "group_run"):
            entry = scores[method][split]
            cells = " | ".join(_cell(entry[label]) for _, label in METRICS)
            table.append(f"| {method} | {split} | {cells} |")
        for column, label in METRICS:
            gap = _inflation(leaky, grouped, column, config)
            gaps[method][label] = _paired(gap)
            gap_table.append(
                f"| {method} | {label} | {_pct(gap.estimate)} "
                f"[{_pct(gap.lower)}, {_pct(gap.upper)}] | {gap.p_value:.4f} |"
            )
            if label == "Top-1":
                top1 = {split: scores[method][split]["Top-1"] for split in scores[method]}
                sentences.append(_sentence(method, top1, gap))
    lines += ["", " ".join(sentences), *table, *gap_table]
    lines += [
        "",
        "Scores are percentages with 95% group-bootstrap intervals. Both splits use the same "
        "groups, cases and settings.",
        "",
        "![Top-1 on the random split and on group_run](figures/leakage_top1.png)",
    ]
    labels = [f"{m}\n{s}" for m in config.methods for s in ("random", "group_run")]
    entries = [scores[m][s]["Top-1"] for m in config.methods for s in ("random", "group_run")]
    bars_with_ci(
        labels,
        [100 * entry["estimate"] for entry in entries],
        [100 * entry["lower"] for entry in entries],
        [100 * entry["upper"] for entry in entries],
        "Top-1 (%)",
        "Top-1 by split, with 95% group-bootstrap intervals",
        directory / "figures" / "leakage_top1",
    )
    _write_text(directory / "note.md", "\n".join(lines) + "\n")
    payload: dict[str, object] = {"scores": scores, "paired_differences": gaps}
    checksum = read_manifest(root / "data" / "processed").digest()
    write_result(directory / "results.json", payload, config, config.seed, checksum)
    return payload


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _floor(predictions: pl.DataFrame) -> float:
    """Return the expected Top-1 of picking uniformly among each case's services."""
    candidates = predictions.group_by("case_id").len()
    return float((1 / candidates["len"]).mean())  # type: ignore[arg-type]


def _fold_rows(
    method: str, runs: MethodRuns, split_name: str, config: ExperimentConfig
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    held = runs.outcomes.select("fold").unique().sort("fold")["fold"].to_list()
    names = _fold_names(runs, split_name)
    for fold in held:
        part = runs.outcomes.filter(pl.col("fold") == fold)
        tested = runs.predictions.filter(pl.col("fold") == fold)
        rows.append(
            {
                "method": method,
                "split": split_name,
                "fold": names.get(fold, str(fold)),
                "cases": part["case_id"].n_unique(),
                "floor": _floor(tested),
                **_summaries(part, config),
            }
        )
    return rows


def _fold_names(runs: MethodRuns, split_name: str) -> dict[int, str]:
    key = {"lofo": "fault_type", "loso": "system"}.get(split_name)
    if key is None or key not in runs.outcomes.columns:
        return {}
    pairs = runs.outcomes.select("fold", key).unique()
    return dict(pairs.iter_rows())


def _with_held_out(runs: MethodRuns, root: Path, split: str) -> MethodRuns:
    if split not in ("lofo", "loso"):
        return runs
    key = "fault_type" if split == "lofo" else "system"
    index = pl.read_parquet(root / "data" / "processed" / "cases.parquet").select("case_id", key)
    return MethodRuns(
        runs.method, runs.split, runs.outcomes.join(index, on="case_id"), runs.predictions
    )


def _table(title: str, rows: Sequence[dict[str, object]]) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| Method | Split | Fold | Cases | Random floor | Top-1 | Top-3 | Avg@5 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        cells = " | ".join(_cell(row[label]) for _, label in METRICS)  # type: ignore[arg-type]
        floor = row["floor"]
        assert isinstance(floor, float)
        lines.append(
            f"| {row['method']} | {row['split']} | {row['fold']} | {row['cases']} "
            f"| {_pct(floor)} | {cells} |"
        )
    return [*lines, ""]


def baselines_clean(root: Path = Path(), cache: SummaryCache | None = None) -> dict[str, object]:
    """Tabulate every baseline on group_run, and per fold on LOFO and LOSO."""
    directory = root / CLEAN_DIR
    config = load_config(directory / "config.yaml", ExperimentConfig)
    store = cache if cache is not None else SummaryCache()
    pooled: list[dict[str, object]] = []
    folds: list[dict[str, object]] = []
    for method in config.methods:
        for split in ("group_run", "lofo", "loso"):
            runs = _with_held_out(run_seeds(method, split, root, store), root, split)
            if split == "group_run":
                pooled.append(
                    {
                        "method": method,
                        "split": split,
                        "fold": "pooled",
                        "cases": runs.outcomes["case_id"].n_unique(),
                        "floor": _floor(runs.predictions),
                        **_summaries(runs.outcomes, config),
                    }
                )
            else:
                folds += _fold_rows(method, runs, split, config)
    lines = [
        "# Clean-data baselines",
        "",
        "Generated by `cascadelens experiments baselines-clean`. Scores are percentages with 95% "
        "group-bootstrap intervals, averaged per case over each method's seeds.",
        "",
        "Supervised and unsupervised methods are separate tables; comparing across them is only "
        "descriptive (PRD 7.1). LOFO and LOSO are reported per fold and not pooled (PRD 9.3). The "
        "random floor is the expected Top-1 of picking uniformly among a case's services.",
        "",
        "Under LOSO the frequency prior has no seen service, so it ties all candidates and its "
        "measured Top-1 only reflects the tie-break by service name. Read the random floor "
        "instead (PRD 6.4, rule 5). RE1 has no topology, so `rule_upstream` here is a plain "
        "max-anomaly ranking.",
        "",
    ]
    for title, supervised in (("Supervised", True), ("Unsupervised", False)):
        chosen = {name for name in config.methods if BASELINES[name].supervised == supervised}
        lines += _table(f"{title}: group_run", [r for r in pooled if r["method"] in chosen])
        lines += _table(f"{title}: LOFO per fold", _only(folds, chosen, "lofo"))
        lines += _table(f"{title}: LOSO per fold", _only(folds, chosen, "loso"))
    _write_text(directory / "results.md", "\n".join(lines).rstrip() + "\n")
    payload: dict[str, object] = {"group_run": pooled, "folds": folds}
    checksum = read_manifest(root / "data" / "processed").digest()
    write_result(directory / "results.json", payload, config, config.seed, checksum)
    return payload


def _only(
    rows: Sequence[dict[str, object]], methods: set[str], split: str
) -> list[dict[str, object]]:
    return [row for row in rows if row["method"] in methods and row["split"] == split]
