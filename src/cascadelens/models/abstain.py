"""Abstention: thresholds chosen on validation, the calibrate command and the Part 10 experiment.

The calibrator and tau of a fold come from that fold's clean validation cases, pooled across seeds.
Test cases, clean or perturbed, are scored once with what validation produced. Nothing is refitted
on perturbed data.

Rows of all seeds are stacked for the metrics. Groups stay the resampling unit, so stacking adds no
false precision, and each group carries one row per seed and case.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from cascadelens.baselines.runner import GROUPED, load_dataset, load_split, truth_table
from cascadelens.config import StrictModel, canonical_json, config_hash, load_config
from cascadelens.eval.bootstrap import (
    BootstrapResult,
    PairedResult,
    aurc_difference_ci,
    ece_ci,
    ece_difference_ci,
    group_bootstrap_ci,
    selective_accuracy_ci,
    selective_gain_bootstrap,
)
from cascadelens.eval.calibration import (
    aurc,
    brier,
    case_confidence,
    ece,
    oracle_abstention,
    random_abstention,
    reliability_table,
    risk_coverage,
    selective_accuracy,
)
from cascadelens.eval.metrics import per_case_outcomes
from cascadelens.eval.plots import reliability_diagram, risk_coverage_curves
from cascadelens.eval.robustness import (
    CLEAN,
    CLEAN_POINT,
    FAMILIES,
    Context,
    Point,
    build_arm,
    load_context,
    s3_key,
)
from cascadelens.eval.significance import holm_correction
from cascadelens.models.artifacts import Provenance, artifact_dir, load_artifact
from cascadelens.models.calibration import (
    CalibrationConfig,
    Calibrator,
    SavedCalibration,
    pool_seeds,
    save_calibration,
    select_calibrator,
)
from cascadelens.models.data import FeatureSource, RankerError
from cascadelens.models.pipeline import (
    feature_settings,
    load_ranker_config,
    train_folds,
)
from cascadelens.results import write_result
from cascadelens.splits.report import assert_no_group_overlap
from cascadelens.splits.splitters import Fold

CALIBRATION_CONFIG = Path("configs/calibration.yaml")
EXPERIMENT_DIR = Path("experiments/calibration")
ROBUSTNESS_RESULTS = Path("experiments/robustness/results.json")
SUMMARY_PATH = Path("docs/calibration_summary.md")
ENGINE_VERSION = 1
HELD_OUT = "held-out"
FIGURE_POINTS = ("clean", "s3", "P4:offset_seconds=-120")


@dataclass(frozen=True)
class Threshold:
    """A confidence threshold and the coverage it gives on validation."""

    target: float
    tau: float
    realised: float

    def to_dict(self) -> dict[str, float]:
        """Return plain values."""
        return {"target": self.target, "tau": self.tau, "realised": self.realised}


def select_tau(conf: npt.ArrayLike, target: float) -> Threshold:
    """Return the tau whose validation coverage is nearest ``target``, preferring higher coverage.

    Cases at or above tau are kept. Tied confidences stay together, so a calibrator with plateaus
    can miss the target by more than one case; the realised value records that.
    """
    if not 0 < target <= 1:
        raise RankerError(f"Target coverage must lie in (0, 1], got {target}")
    values = np.sort(np.asarray(conf, dtype=np.float64))[::-1]
    if values.size == 0:
        raise RankerError("No validation cases to choose a threshold on")
    taus, first = np.unique(-values, return_index=True)
    kept = np.r_[first[1:], values.size] / values.size
    best = int(np.argmin(np.abs(kept - target) - 1e-12 * kept))
    return Threshold(target, float(-taus[best]), float(kept[best]))


def threshold_table(conf: npt.ArrayLike, coverages: Sequence[float]) -> list[Threshold]:
    """Return tau for each coverage."""
    return [select_tau(conf, coverage) for coverage in coverages]


def abstain_flags(conf: npt.ArrayLike, tau: float) -> npt.NDArray[np.bool_]:
    """Return which cases abstain: those with confidence below tau."""
    flags: npt.NDArray[np.bool_] = np.asarray(conf, dtype=np.float64) < tau
    return flags


def assert_validation_only(fold: Fold, case_ids: Sequence[str]) -> None:
    """Fail unless a calibrator reads only validation cases of ``fold``."""
    used = set(case_ids)
    if used - set(fold.val_ids) or used & set(fold.test_ids):
        raise RankerError("A calibrator may be fitted on validation cases only")


def load_calibration_config(root: Path = Path()) -> CalibrationConfig:
    """Read the calibrator settings."""
    return load_config(root / CALIBRATION_CONFIG, CalibrationConfig)


def fit_fold(
    predictions: Mapping[int, pl.DataFrame],
    truth: pl.DataFrame,
    fold: Fold,
    config: CalibrationConfig,
) -> SavedCalibration:
    """Fit one fold's calibrator and tau on validation predictions pooled across seeds."""
    for frame in predictions.values():
        assert_validation_only(fold, frame["case_id"].unique().to_list())
    pred, labels = pool_seeds(predictions, truth)
    selection = select_calibrator(pred, labels, config)
    confidences = selection.calibrator.apply(pred)["confidence"].to_numpy()
    target = select_tau(confidences, config.target_coverage)
    table = threshold_table(confidences, config.coverages)
    return SavedCalibration(
        selection.calibrator,
        target.tau,
        target.realised,
        [item.to_dict() for item in table],
        {**selection.report, "target_coverage": config.target_coverage},
    )


def calibrate_split(
    config_path: Path, split_name: str, method: str, root: Path = Path()
) -> list[Path]:
    """Fit and save a calibrator for every saved model of a split, training any that is missing."""
    config = load_ranker_config(config_path, root)
    settings = load_calibration_config(root)
    features = feature_settings(config, root)
    dataset = load_dataset(root)
    split, group_map = load_split(split_name, dataset, root)
    truth = truth_table(dataset.records, group_map)
    source = FeatureSource(features, root)
    source.verify(dataset.checksum)
    saved: list[Path] = []
    for index, fold in enumerate(split.folds):
        directories, predictions = [], {}
        handles = dataset.handles(fold.val_ids)
        for seed in config.seeds:
            provenance = Provenance(
                method,
                split_name,
                seed,
                index,
                config_hash(config),
                config_hash(features),
                dataset.checksum,
                features.model_dump(mode="json"),
            )
            directory = artifact_dir(provenance, root)
            if not (directory / "sidecar.json").is_file():
                train_folds(config_path, split_name, seed, method, root)
            model, _ = load_artifact(directory)
            predictions[seed] = pl.concat(model.rank(source.table(case))[0] for case in handles)
            directories.append(directory)
        calibration = fit_fold(predictions, truth, fold, settings)
        saved.extend(save_calibration(calibration, directory) for directory in directories)
    return saved


class ExperimentConfig(StrictModel):
    """Settings of the calibration experiment."""

    seed: int = 0
    n_resamples: int = 10_000
    alpha: float = 0.05
    bins: int = 10
    calibration_config: str = str(CALIBRATION_CONFIG)
    primary: str
    variants: list[str]
    selective_coverages: list[float]
    s4_coverage: float = 0.5


def evaluation_points(split: str, context: Context) -> list[Point]:
    """Return clean data, and on group_run also P2 at 30% and every P4 shift."""
    if split != "group_run":
        return [CLEAN_POINT]
    wanted = s3_key(context.config, context.points)
    return [
        point
        for point in context.points
        if point.key in (CLEAN, wanted) or point.identifier == "P4"
    ]


def unit_path(
    variant: str,
    split: str,
    context: Context,
    settings: CalibrationConfig,
) -> Path:
    """Return the rows of a unit, named by a hash of everything that shapes them."""
    payload = canonical_json(
        {
            "augment": context.augment.model_dump(mode="json"),
            "calibration": settings.model_dump(mode="json"),
            "data": context.dataset.checksum,
            "engine": ENGINE_VERSION,
            "features": config_hash(context.features),
            "points": [point.key for point in evaluation_points(split, context)],
            "ranker": config_hash(context.ranker),
            "split": split,
            "status": context.status,
            "variant": context.config.variant(variant).model_dump(mode="json"),
        }
    )
    name = hashlib.sha256(payload.encode()).hexdigest()[:24]
    return context.root / EXPERIMENT_DIR / "cache" / "rows" / variant / split / f"{name}.parquet"


def scored_rows(
    predictions: pl.DataFrame, tested: pl.DataFrame, calibrator: Calibrator, tau: float
) -> pl.DataFrame:
    """Return each test case's raw and calibrated confidence, correctness and abstention."""
    outcomes = per_case_outcomes(predictions, tested)
    calibrated = calibrator.apply(predictions)["confidence"].to_numpy()
    return outcomes.select(
        "case_id",
        "group_id",
        pl.col("hit@1").cast(pl.Float64).alias("correct"),
        raw=case_confidence(predictions)["confidence"],
        calibrated=pl.Series(calibrated),
        abstain=pl.Series(abstain_flags(calibrated, tau)),
    )


def run_unit(variant: str, split_name: str, context: Context, settings: CalibrationConfig) -> Path:
    """Fit on validation, then score every point of one variant on one split, unless done."""
    path = unit_path(variant, split_name, context, settings)
    if path.is_file():
        return path
    split, group_map = load_split(split_name, context.dataset, context.root)
    truth = truth_table(context.dataset.records, group_map)
    arm = build_arm(context.config.variant(variant), split_name, group_map, context)
    points = evaluation_points(split_name, context)
    clean = context.derived(CLEAN_POINT)
    frames, fits = [], []
    for index, fold in enumerate(split.folds):
        if split_name in GROUPED:
            assert_no_group_overlap(fold.val_ids, fold.test_ids, group_map)
        tested = truth.filter(pl.col("case_id").is_in(fold.test_ids))
        context.tables.clear()
        validation: dict[int, pl.DataFrame] = {}
        scored: dict[tuple[str, int], pl.DataFrame] = {}
        for seed in arm.seeds:
            arm.fit(fold, seed)
            validation[seed] = arm.score(clean, fold.val_ids)
            for point in points:
                scored[point.key, seed] = arm.score(context.derived(point), fold.test_ids)
        fitted = fit_fold(validation, truth, fold, settings)
        fits.append({"fold": index, "tau": fitted.tau, **fitted.report})
        fits[-1]["thresholds"] = fitted.thresholds
        for (key, seed), predictions in scored.items():
            rows = scored_rows(predictions, tested, fitted.calibrator, fitted.tau)
            frames.append(
                rows.with_columns(
                    variant=pl.lit(variant),
                    split=pl.lit(split_name),
                    point=pl.lit(key),
                    fold=pl.lit(index),
                    seed=pl.lit(seed),
                )
            )
    context.tables.clear()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(fits, indent=2), encoding="utf-8")
    partial = path.with_suffix(".partial")
    pl.concat(frames).write_parquet(partial)
    partial.replace(path)
    return path


def _interval(result: BootstrapResult) -> dict[str, float]:
    return {"estimate": result.estimate, "lower": result.lower, "upper": result.upper}


def _paired(result: PairedResult) -> dict[str, float]:
    return {**_interval(result), "p_value": result.p_value}


def seed_spread(rows: pl.DataFrame, config: ExperimentConfig) -> dict[str, dict[str, float]]:
    """Return the mean and spread across seeds of ECE after and Top-1 at the target."""
    eces, selective = [], []
    for _, part in rows.group_by("seed", maintain_order=True):
        eces.append(ece(part["calibrated"], part["correct"], config.bins))
        selective.append(
            selective_accuracy(part["calibrated"], part["correct"], config.s4_coverage)
        )
    return {
        name: {"mean": float(np.mean(values)), "std": float(np.std(values))}
        for name, values in (("ece_after", eces), ("selective_top1", selective))
    }


def measures(rows: pl.DataFrame, config: ExperimentConfig) -> dict[str, Any]:
    """Return every calibration and abstention measure of a slice of rows."""
    raw, calibrated = rows["raw"].to_numpy(), rows["calibrated"].to_numpy()
    correct, groups = rows["correct"].to_numpy(), rows["group_id"].to_list()
    resampling = (config.n_resamples, config.seed, config.alpha)
    curve = risk_coverage(calibrated, correct)
    kept = rows.filter(~pl.col("abstain"))
    realised = kept.height / rows.height
    result: dict[str, Any] = {
        "rows": rows.height,
        "groups": len(set(groups)),
        "top1": _interval(group_bootstrap_ci(correct, groups, *resampling)),
        "ece_before": _interval(ece_ci(raw, correct, groups, config.bins, *resampling)),
        "ece_after": _interval(ece_ci(calibrated, correct, groups, config.bins, *resampling)),
        "ece_difference": _paired(
            ece_difference_ci(raw, calibrated, correct, groups, config.bins, *resampling)
        ),
        "brier_before": _interval(group_bootstrap_ci((raw - correct) ** 2, groups, *resampling)),
        "brier_after": _interval(
            group_bootstrap_ci((calibrated - correct) ** 2, groups, *resampling)
        ),
        "aurc": aurc(curve),
        "aurc_random": aurc(random_abstention(correct)),
        "aurc_oracle": aurc(oracle_abstention(correct)),
        "aurc_raw": aurc(risk_coverage(raw, correct)),
        "aurc_minus_random": _paired(aurc_difference_ci(calibrated, correct, groups, *resampling)),
        "selective": {
            str(coverage): _interval(
                selective_accuracy_ci(calibrated, correct, groups, coverage, *resampling)
            )
            for coverage in config.selective_coverages
        },
        "selective_gain": _paired(
            selective_gain_bootstrap(calibrated, correct, groups, config.s4_coverage, *resampling)
        ),
        "realised_coverage": realised,
        "top1_on_kept": float(kept["correct"].to_numpy().mean()) if kept.height else None,
        "seed_spread": seed_spread(rows, config),
    }
    assert np.isclose(brier(calibrated, correct), result["brier_after"]["estimate"])
    return result


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")


def _figures(
    rows: pl.DataFrame, variant: str, label: str, tag: str, config: ExperimentConfig, stem: Path
) -> None:
    correct = rows["correct"].to_numpy()
    for column, name in (("raw", "before"), ("calibrated", "after")):
        table = reliability_table(rows[column], correct, config.bins)
        title = f"Reliability of {variant}, {name} calibration, {label} ({tag})"
        reliability_diagram(table, title, stem.with_name(f"{stem.name}_reliability_{name}"))
    risk_coverage_curves(
        {variant: risk_coverage(rows["calibrated"], correct)},
        random_abstention(correct),
        oracle_abstention(correct),
        f"Risk versus coverage of {variant}, {label} ({tag})",
        stem.with_name(f"{stem.name}_risk_coverage"),
    )


def fold_slices(rows: pl.DataFrame) -> list[tuple[int, pl.DataFrame]]:
    """Return each fold's rows."""
    return [(int(key[0]), part) for key, part in rows.group_by("fold", maintain_order=True)]


def coverage_by_fold(rows: pl.DataFrame) -> list[float]:
    """Return realised test coverage per fold."""
    return [1.0 - float(part["abstain"].mean()) for _, part in fold_slices(rows)]  # type: ignore[arg-type]


def primary_family(s4_p: float, root: Path) -> dict[str, float]:
    """Return Holm-adjusted p-values over S1, S2, S3 and S4. S7 is an observed result."""
    path = root / ROBUSTNESS_RESULTS
    pvalues = {"S4": s4_p}
    if path.is_file():
        found = json.loads(path.read_text(encoding="utf-8"))["metrics"]["hypotheses"]
        pvalues["S1"] = found["ranker"]["S1"]["p_value"]
        pvalues["S2"] = found["ranker"]["S2"]["p_value"]
        for name, entry in found["S3"].items():
            pvalues[f"S3:{name}"] = entry["p_value"]
    labels = sorted(pvalues)
    return holm_correction([pvalues[label] for label in labels], labels)


def verdicts(
    primary: Mapping[str, Any], holm: Mapping[str, float], alpha: float
) -> dict[str, bool]:
    """Return S4 and S5 on clean group_run."""
    gain = primary["selective_gain"]
    return {
        "S4": gain["lower"] > 0 and holm["S4"] < alpha,
        "S5": primary["ece_difference"]["upper"] < 0,
    }


def analyse(frame: pl.DataFrame, context: Context, config: ExperimentConfig) -> dict[str, Any]:
    """Return every measure, and draw the figures."""
    figures = context.root / EXPERIMENT_DIR / "figures"
    shown = s3_key(context.config, context.points)
    results: dict[str, Any] = {}
    for variant in config.variants:
        mine = frame.filter(pl.col("variant") == variant)
        entry: dict[str, Any] = {}
        runs = mine.filter(pl.col("split") == "group_run")
        for key, part in runs.group_by("point", maintain_order=True):
            point = str(key[0])
            tag = "clean" if point == CLEAN else "simulated"
            entry[point] = {
                "tag": tag,
                **measures(part, config),
                "fold_coverage": coverage_by_fold(part),
            }
            if point in (CLEAN, shown) or point == FIGURE_POINTS[2]:
                stem = figures / f"{variant}_group_run_{_slug(point)}"
                _figures(part, variant, point, tag, config, stem)
        for split in ("lofo", "loso"):
            part = mine.filter(pl.col("split") == split)
            entry[split] = {
                "tag": HELD_OUT,
                "pooled": measures(part, config),
                "folds": {str(index): measures(rows, config) for index, rows in fold_slices(part)},
            }
            _figures(part, variant, split, HELD_OUT, config, figures / f"{variant}_{split}")
        results[variant] = entry
    return results


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def _ci(entry: Mapping[str, float]) -> str:
    return f"{_pct(entry['estimate'])} [{_pct(entry['lower'])}, {_pct(entry['upper'])}]"


def _ece(entry: Mapping[str, float]) -> str:
    return f"{entry['estimate']:.3f} [{entry['lower']:.3f}, {entry['upper']:.3f}]"


def _row(name: str, entry: Mapping[str, Any], config: ExperimentConfig) -> str:
    selective = entry["selective"][str(config.s4_coverage)]
    cells = [
        name,
        _ece(entry["ece_before"]),
        _ece(entry["ece_after"]),
        _ece(entry["ece_difference"]),
        _ci(entry["top1"]),
        _ci(selective),
        _ci(entry["selective_gain"]),
        f"{entry['aurc']:.3f} / {entry['aurc_random']:.3f} / {entry['aurc_oracle']:.3f}",
        _pct(entry["realised_coverage"]),
    ]
    return "| " + " | ".join(cells) + " |"


HEADER = (
    "| Slice | ECE before | ECE after | ECE difference | Top-1 | Top-1 at 50% | Gain over full "
    "and random | AURC / random / oracle | Test coverage |\n| " + " | ".join(["---"] * 9) + " |"
)


def _choices(fits: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    lines = []
    for split, items in fits.items():
        chosen = [item["chosen"] for item in items]
        temperatures = ", ".join(f"{item['temperature']:.3f}" for item in items)
        misses = max(item["retrieval_miss_rate"] for item in items)
        lines.append(
            f"- {split}: {chosen.count('temperature')} temperature, "
            f"{chosen.count('isotonic')} isotonic. Temperatures {temperatures}. "
            f"Largest validation retrieval miss rate {_pct(misses)}%."
        )
    return lines


def _caveats(fits: Mapping[str, list[dict[str, Any]]]) -> list[str]:
    empty = [
        f"{split} fold {item['fold']}"
        for split, items in fits.items()
        for item in items
        if item["validation_top1"] == 1
    ]
    return [
        "## Caveats",
        "",
        "- ECE before uses a softmax at temperature 1, but ranking scores have no natural scale, "
        "so the before value is partly arbitrary and S5 is an easy bar.",
        "- Stacking seeds averages their errors, so stacked ECE sits below the mean ECE of single "
        "seeds; `seed_spread` in the results has that value.",
        "- Confidence is compared across folds that have different calibrators.",
        "- A zero-width interval means one tied block of confidences covers that coverage.",
        "- Folds whose validation predictions were all correct give the calibrator no errors to "
        f"learn from, so confidence saturates: {', '.join(empty) if empty else 'none'}.",
        "",
    ]


def render_summary(payload: Mapping[str, Any], config: ExperimentConfig) -> str:
    """Return the note with the S4 and S5 verdicts, including failures."""
    results, status = payload["results"], payload["status"]
    primary = results[config.primary]
    clean = primary[CLEAN]
    verdict = payload["verdicts"]
    gain = clean["selective_gain"]
    lines = [
        "# Calibration summary",
        "",
        f"Generated by `cascadelens experiments calibration`. **Status: {status}.**",
        "",
        "The config is not frozen, so verdicts below preview the pre-registered tests."
        if status != "frozen"
        else "The config is frozen.",
        "",
        f"The CascadeLens ranker is `{config.primary}`. Each fold's calibrator and tau come from "
        "that fold's validation cases, pooled across seeds. They are not pooled across folds, "
        "because validation cases of one fold are test cases of another. Validation also steered "
        "the ranker's early stopping and search, so it may look better than test. Nothing was "
        "refitted on perturbed data.",
        "",
        "Rows of all seeds are stacked and groups are resampled. ECE is always shown with its "
        "interval, and the reliability diagrams are in `experiments/calibration/figures/`. "
        "Random abstention's expected accuracy equals full-coverage accuracy, so one difference "
        "answers both S4 comparisons.",
        "",
        "## Verdicts",
        "",
        "| Criterion | Result [95% CI] | p | Holm p | Verdict |",
        "| --- | --- | --- | --- | --- |",
        f"| S4: Top-1 at 50% coverage minus full coverage and random abstention | {_ci(gain)} | "
        f"{gain['p_value']:.4f} | {payload['holm']['S4']:.4f} | "
        f"{'pass' if verdict['S4'] else 'fail'} |",
        f"| S5: ECE after minus ECE before (secondary) | {_ece(clean['ece_difference'])} | "
        f"{clean['ece_difference']['p_value']:.4f} | - | {'pass' if verdict['S5'] else 'fail'} |",
        "",
        f"Holm runs over S1, S2, S3 and S4, {len(payload['holm'])} tests. S7 is an observed "
        "result, so it is not in the family.",
        "",
        f"AURC minus random abstention: {_ece(clean['aurc_minus_random'])}. AURC "
        f"{clean['aurc']:.3f}, random {clean['aurc_random']:.3f}, oracle "
        f"{clean['aurc_oracle']:.3f}.",
        "",
        "## Calibrator choice",
        "",
        "Isotonic is chosen only if its cross-fitted validation log loss beats temperature "
        "scaling with a paired interval wholly below zero.",
        "",
        *_choices(payload["fits"][config.primary]),
        "",
        *_caveats(payload["fits"][config.primary]),
        "## Results",
        "",
        "Percent for Top-1, fractions for ECE and AURC. Clean and held-out rows are real "
        "benchmark cases, P2 and P4 are simulated.",
        "",
    ]
    for variant in config.variants:
        entry = results[variant]
        lines += [f"### {variant}", "", HEADER]
        for key, item in entry.items():
            if key in ("lofo", "loso"):
                continue
            lines.append(_row(f"{key} ({item['tag']})", item, config))
        for split in ("lofo", "loso"):
            lines.append(_row(f"{split} pooled (held-out)", entry[split]["pooled"], config))
            for index, item in entry[split]["folds"].items():
                lines.append(_row(f"{split} fold {index} (held-out)", item, config))
        lines.append("")
    lines += [
        "## Selective Top-1, clean group_run",
        "",
        "| Coverage | " + " | ".join(config.variants) + " |",
        "| " + " | ".join(["---"] * (len(config.variants) + 1)) + " |",
    ]
    for coverage in config.selective_coverages:
        cells = [_ci(results[v][CLEAN]["selective"][str(coverage)]) for v in config.variants]
        lines.append(f"| {_pct(coverage)}% | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Coverage drift",
        "",
        f"Tau targets {_pct(payload['target_coverage'])}% coverage on clean validation.",
        "",
        "| Slice | Test coverage | Range across folds |",
        "| --- | --- | --- |",
    ]
    for key, item in primary.items():
        if key not in ("lofo", "loso"):
            folds = item["fold_coverage"]
            span = f"{_pct(min(folds))} to {_pct(max(folds))}"
            lines.append(f"| {key} ({item['tag']}) | {_pct(item['realised_coverage'])} | {span} |")
    for split in ("lofo", "loso"):
        folds = [item["realised_coverage"] for item in primary[split]["folds"].values()]
        pooled = primary[split]["pooled"]["realised_coverage"]
        span = f"{_pct(min(folds))} to {_pct(max(folds))}"
        lines.append(f"| {split} (held-out) | {_pct(pooled)} | {span} |")
    return "\n".join(lines) + "\n"


def run_calibration(exploratory: bool, root: Path = Path()) -> dict[str, Any]:
    """Run every unmade unit, then measure, plot and write the summary."""
    context = load_context(exploratory, root)
    config = load_config(root / EXPERIMENT_DIR / "config.yaml", ExperimentConfig)
    settings = load_config(root / config.calibration_config, CalibrationConfig)
    frames = []
    fits: dict[str, dict[str, Any]] = {}
    for variant in config.variants:
        fits[variant] = {}
        for split in FAMILIES:
            path = run_unit(variant, split, context, settings)
            frames.append(pl.read_parquet(path))
            fits[variant][split] = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    results = analyse(pl.concat(frames), context, config)
    clean = results[config.primary][CLEAN]
    holm = primary_family(clean["selective_gain"]["p_value"], root)
    payload = {
        "status": context.status,
        "results": results,
        "fits": fits,
        "holm": holm,
        "verdicts": verdicts(clean, holm, config.alpha),
        "target_coverage": settings.target_coverage,
        "ranker_config_hash": config_hash(context.ranker),
        "calibration_config_hash": config_hash(settings),
    }
    directory = root / EXPERIMENT_DIR
    write_result(directory / "results.json", payload, config, config.seed, context.dataset.checksum)
    target = root / SUMMARY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_summary(payload, config), encoding="utf-8", newline="\n")
    return payload
