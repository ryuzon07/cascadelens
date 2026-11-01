import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from baseline_factory import build_project
from ranker_factory import write_project_configs
from typer.testing import CliRunner

from cascadelens.baselines.experiments import baselines_clean
from cascadelens.baselines.runner import load_dataset, load_split
from cascadelens.cli import app
from cascadelens.data.io import metrics_path
from cascadelens.eval.bootstrap import (
    aurc_difference_ci,
    ece_ci,
    selective_accuracy_ci,
    selective_gain_bootstrap,
)
from cascadelens.eval.calibration import (
    aurc,
    case_confidence,
    ece,
    oracle_abstention,
    random_abstention,
    risk_coverage,
    selective_accuracy,
)
from cascadelens.eval.robustness import derive_stage, load_context
from cascadelens.models import abstain
from cascadelens.models.abstain import (
    abstain_flags,
    assert_validation_only,
    calibrate_split,
    fit_fold,
    run_calibration,
    select_tau,
    threshold_table,
)
from cascadelens.models.artifacts import latest_artifact
from cascadelens.models.calibration import (
    ARTIFACT,
    CalibrationConfig,
    IsotonicCalibrator,
    SavedCalibration,
    TemperatureScaler,
    load_calibration,
    model_hash,
    negative_log_likelihood,
    pool_seeds,
    retrieval_miss_rate,
    save_calibration,
    select_calibrator,
)
from cascadelens.models.data import RankerError
from cascadelens.models.diagnose import DISCLAIMER, diagnose_report
from cascadelens.splits.splitters import Fold

SETTINGS = CalibrationConfig(n_resamples=200)
SUITE = """seed: 0
perturbations:
  - {id: P2, name: failure_correlated_loss, realism: simulated, params: {rate: [0.3]}}
  - {id: P4, name: window_shift, realism: simulated, params: {offset_seconds: [30]}}
  - {id: P9, name: unseen_fault_type, realism: held-out, params: {split: [lofo]}}
  - {id: P10, name: unseen_system, realism: held-out, params: {split: [loso]}}
"""
ROBUSTNESS = """seed: 0
n_resamples: 200
workers: 1
ranker: ranker_augmented
plain_ranker: ranker
prior: frequency
primary: [S1, S2, S3, S4, S7]
s3_point: {id: P2, rate: 0.3}
s3_baselines: [lightgbm_flat]
curve_methods: {supervised: [ranker], unsupervised: []}
variants:
  - {name: ranker, kind: two_stage}
  - {name: ranker_augmented, kind: two_stage, augment: true}
  - {name: lightgbm_flat, kind: flat}
"""
EXPERIMENT = """seed: 0
n_resamples: 200
primary: ranker_augmented
variants: [ranker_augmented, lightgbm_flat]
selective_coverages: [1.0, 0.5]
"""


def scored_cases(
    cases: int, scale: float, seed: int = 0, services: int = 4
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Draw truth from a softmax of the scores, then multiply scores by ``scale``.

    With scale 1 the scores are calibrated. With a larger scale they are overconfident.
    """
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 1.5, size=(cases, services))
    chance = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    picks = [int(rng.choice(services, p=row)) for row in chance]
    rows: dict[str, list[object]] = {"case_id": [], "service": [], "score": []}
    for case in range(cases):
        for service in range(services):
            rows["case_id"].append(f"c{case:04d}")
            rows["service"].append(f"s{service}")
            rows["score"].append(float(logits[case, service] * scale))
    truth = pl.DataFrame(
        {
            "case_id": [f"c{case:04d}" for case in range(cases)],
            "root_cause_service": [f"s{pick}" for pick in picks],
            "group_id": [f"g{case // 3}" for case in range(cases)],
        }
    )
    return pl.DataFrame(rows), truth


def test_temperature_scaling_lowers_nll_on_overconfident_scores() -> None:
    pred, truth = scored_cases(1500, 4.0)
    fitted = TemperatureScaler.fit(pred, truth)
    assert fitted.temperature == pytest.approx(4.0, rel=0.2)
    before = negative_log_likelihood(pred, truth, 1.0)
    after = negative_log_likelihood(pred, truth, fitted.temperature)
    assert after < before


def test_temperature_recovers_one_on_calibrated_scores() -> None:
    pred, truth = scored_cases(3000, 1.0)
    assert TemperatureScaler.fit(pred, truth).temperature == pytest.approx(1.0, abs=0.12)


def test_temperature_one_is_the_identity() -> None:
    pred, _ = scored_cases(50, 2.0)
    assert TemperatureScaler(1.0).apply(pred).equals(case_confidence(pred))


def test_fitted_temperature_ignores_test_labels() -> None:
    pred, truth = scored_cases(600, 3.0)
    validation = truth.head(400)
    used = pred.join(validation, on="case_id", how="semi")
    first = TemperatureScaler.fit(used, validation)
    flipped = truth.with_columns(pl.col("root_cause_service").reverse())
    again = flipped.with_columns(
        pl.when(pl.int_range(pl.len()) < 400)
        .then(truth["root_cause_service"])
        .otherwise(pl.col("root_cause_service"))
        .alias("root_cause_service")
    ).head(400)
    assert TemperatureScaler.fit(used, again).temperature == first.temperature


def test_cases_outside_the_candidates_are_left_out_and_reported() -> None:
    pred, truth = scored_cases(200, 2.0)
    missed = truth["case_id"][:20].to_list()
    truth_rows = pl.col("root_cause_service")
    thinned = pred.join(truth, on="case_id").filter(
        ~(pl.col("case_id").is_in(missed) & (pl.col("service") == truth_rows))
    )
    thinned = thinned.select("case_id", "service", "score")
    assert retrieval_miss_rate(thinned, truth) == pytest.approx(0.1)
    assert TemperatureScaler.fit(thinned, truth).temperature > 0


def test_isotonic_is_monotone_and_bounded() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=400)
    correct = (rng.uniform(size=400) < conf**2).astype(float)
    fitted = IsotonicCalibrator.fit(conf, correct)
    mapped = fitted.transform(np.linspace(-0.5, 1.5, 501))
    assert np.all(np.diff(mapped) >= 0)
    assert mapped.min() >= 0
    assert mapped.max() <= 1


def test_selection_is_deterministic_and_can_be_forced() -> None:
    pred, truth = scored_cases(300, 3.0)
    first = select_calibrator(pred, truth, SETTINGS)
    second = select_calibrator(pred, truth, SETTINGS)
    assert first.report == second.report
    assert first.report["chosen"] == "temperature"
    forced = select_calibrator(pred, truth, SETTINGS.model_copy(update={"method": "isotonic"}))
    assert forced.report["chosen"] == "isotonic"
    assert forced.report["forced"]


def test_pooling_keeps_seeds_apart() -> None:
    pred, truth = scored_cases(10, 1.0)
    pooled, labels = pool_seeds({0: pred, 1: pred}, truth)
    assert pooled["case_id"].n_unique() == 20
    assert labels.height == 20


@pytest.mark.parametrize("target", [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3])
def test_threshold_is_within_one_case_of_target(target: float) -> None:
    conf = np.random.default_rng(1).uniform(size=137)
    threshold = select_tau(conf, target)
    assert abs(threshold.realised - target) <= 1 / conf.size
    assert 1 - abstain_flags(conf, threshold.tau).mean() == pytest.approx(threshold.realised)


def test_threshold_table_lists_each_coverage() -> None:
    table = threshold_table(np.linspace(0, 1, 100), [0.9, 0.5])
    assert [item.target for item in table] == [0.9, 0.5]
    assert table[0].tau < table[1].tau


def test_tied_confidences_stay_together() -> None:
    threshold = select_tau([0.9, 0.9, 0.9, 0.2], 0.5)
    assert threshold.tau == 0.9
    assert threshold.realised == 0.75


def test_selective_accuracy_at_full_coverage_is_overall_accuracy() -> None:
    rng = np.random.default_rng(0)
    conf, correct = rng.uniform(size=90), rng.integers(0, 2, size=90)
    assert selective_accuracy(conf, correct, 1.0) == pytest.approx(correct.mean())


def test_perfect_confidence_matches_the_oracle() -> None:
    correct = np.array([1, 1, 0, 1, 0, 0, 1, 1, 1, 0], dtype=float)
    assert aurc(risk_coverage(correct, correct)) == pytest.approx(aurc(oracle_abstention(correct)))


def test_random_confidence_is_near_the_error_rate() -> None:
    rng = np.random.default_rng(0)
    correct = (rng.uniform(size=5000) < 0.7).astype(float)
    assert aurc(risk_coverage(rng.uniform(size=5000), correct)) == pytest.approx(0.3, abs=0.02)


def test_uninformative_confidence_does_not_beat_random_abstention() -> None:
    rng = np.random.default_rng(3)
    correct = (rng.uniform(size=900) < 0.7).astype(float)
    groups = [f"g{index // 5}" for index in range(900)]
    gain = selective_gain_bootstrap(rng.uniform(size=900), correct, groups, 0.5, 2000)
    assert gain.lower <= 0 <= gain.upper


def test_new_intervals_match_their_point_estimates() -> None:
    rng = np.random.default_rng(0)
    conf = rng.uniform(size=200)
    correct = (rng.uniform(size=200) < conf).astype(float)
    groups = [f"g{index // 4}" for index in range(200)]
    result = ece_ci(conf, correct, groups, n=500)
    assert result.estimate == pytest.approx(ece(conf, correct))
    assert result.lower <= result.upper
    selective = selective_accuracy_ci(conf, correct, groups, 0.5, n=500)
    assert selective.estimate == pytest.approx(selective_accuracy(conf, correct, 0.5))
    area = aurc_difference_ci(conf, correct, groups, n=1200)
    expected = aurc(risk_coverage(conf, correct)) - aurc(random_abstention(correct))
    assert area.estimate == pytest.approx(expected)
    assert area.upper < 0


def test_fit_refuses_test_cases() -> None:
    fold = Fold(train_ids=("a",), val_ids=("b",), test_ids=("c",))
    assert_validation_only(fold, ["b"])
    with pytest.raises(RankerError, match="validation"):
        assert_validation_only(fold, ["b", "c"])
    with pytest.raises(RankerError, match="validation"):
        assert_validation_only(fold, ["a"])


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("calibration")
    build_project(root)
    write_project_configs(root)
    baselines_clean(root)
    (root / "configs" / "perturbations.yaml").write_text(SUITE, encoding="utf-8")
    (root / "configs" / "augment.yaml").write_text(
        "seed: 0\nrandom_copies: 1\nfailure_copies: 0\n", encoding="utf-8"
    )
    (root / "configs" / "calibration.yaml").write_text(
        "seed: 0\nn_resamples: 200\n", encoding="utf-8"
    )
    for name, text in (("robustness", ROBUSTNESS), ("calibration", EXPERIMENT)):
        directory = root / "experiments" / name
        directory.mkdir(parents=True)
        (directory / "config.yaml").write_text(text, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def calibrated(project: Path) -> Path:
    calibrate_split(Path("configs/ranker.yaml"), "group_run", "two_stage_ranker", project)
    return project


def _case_file(root: Path, gaps: bool = False) -> Path:
    dataset = load_dataset(root)
    record = next(iter(dataset.records.values()))
    metrics = pl.read_parquet(metrics_path(root / "data" / "processed", record.case_id))
    if gaps:
        service = metrics["service"][0]
        metrics = metrics.with_columns(
            pl.when(pl.col("service") == service)
            .then(None)
            .otherwise(pl.col("value"))
            .alias("value")
        )
    path = root / "one_case.parquet"
    metrics.write_parquet(path)
    return path


def test_artifact_reloads_to_identical_outputs(tmp_path: Path) -> None:
    pred, truth = scored_cases(120, 3.0)
    selection = select_calibrator(pred, truth, SETTINGS)
    saved = SavedCalibration(selection.calibrator, 0.5, 0.5, [], selection.report)
    (tmp_path / "stage2.txt").write_text("model", encoding="utf-8")
    save_calibration(saved, tmp_path)
    loaded = load_calibration(tmp_path)
    assert loaded is not None
    assert loaded.calibrator.apply(pred).equals(selection.calibrator.apply(pred))
    (tmp_path / "stage2.txt").write_text("another model", encoding="utf-8")
    with pytest.raises(RankerError, match="different model"):
        load_calibration(tmp_path)


def test_calibrators_come_from_each_folds_validation_only(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = load_dataset(project)
    split, _ = load_split("group_run", dataset, project)
    seen: list[set[str]] = []

    def spy(pred: pl.DataFrame, truth: pl.DataFrame, config: CalibrationConfig) -> object:
        seen.append({value.split("@seed")[0] for value in truth["case_id"].to_list()})
        return real(pred, truth, config)

    real = select_calibrator
    monkeypatch.setattr(abstain, "select_calibrator", spy)
    calibrate_split(Path("configs/ranker.yaml"), "group_run", "two_stage_ranker", project)
    assert len(seen) == len(split.folds)
    for used, fold in zip(seen, split.folds, strict=True):
        assert used == set(fold.val_ids)
        assert not used & set(fold.test_ids)


def test_fit_fold_refuses_test_predictions() -> None:
    pred, truth = scored_cases(30, 2.0)
    ids = tuple(truth["case_id"].to_list())
    fold = Fold(train_ids=(), val_ids=ids[:20], test_ids=ids[20:])
    with pytest.raises(RankerError, match="validation"):
        fit_fold({0: pred}, truth, fold, SETTINGS)


def test_each_model_gets_an_artifact_tied_to_its_hash(calibrated: Path) -> None:
    paths = list((calibrated / "experiments/ranker_clean/cache/models").rglob(ARTIFACT))
    assert len(paths) == 10
    for path in paths:
        saved = json.loads(path.read_text(encoding="utf-8"))
        assert saved["model_hash"] == model_hash(path.parent)


def test_diagnose_reports_confidence_abstention_and_disclaimer(calibrated: Path) -> None:
    report = diagnose_report(_case_file(calibrated, gaps=True), latest_artifact(calibrated))
    assert 0 <= report.confidence <= 1
    assert isinstance(report.abstain, bool)
    assert report.calibrated
    assert DISCLAIMER == "ranking of likely origin services, not proof of causation"
    assert DISCLAIMER in report.notes
    assert "1 services had missing telemetry." in report.notes


def test_diagnose_works_without_a_calibrator(
    calibrated: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = latest_artifact(calibrated)
    moved = directory / "set_aside.json"
    (directory / ARTIFACT).rename(moved)
    try:
        report = diagnose_report(_case_file(calibrated), directory)
        monkeypatch.chdir(calibrated)
        result = CliRunner().invoke(
            app, ["diagnose", "--case", str(calibrated / "one_case.parquet")]
        )
    finally:
        moved.rename(directory / ARTIFACT)
    assert not report.calibrated
    assert report.abstain is False
    assert "uncalibrated" in report.notes[0]
    assert result.exit_code == 0
    assert "Confidence (uncalibrated)" in result.output
    assert DISCLAIMER in result.output


def test_experiment_writes_verdicts_and_stays_clean_fitted(calibrated: Path) -> None:
    derive_stage(load_context(True, calibrated))
    first = run_calibration(True, calibrated)
    second = run_calibration(True, calibrated)
    assert first["results"] == second["results"]
    clean = first["results"]["ranker_augmented"]["clean"]
    assert clean["tag"] == "clean"
    assert clean["ece_difference"]["lower"] <= clean["ece_difference"]["upper"]
    assert set(first["verdicts"]) == {"S4", "S5"}
    rows = next((calibrated / "experiments/calibration/cache/rows").rglob("group_run/*.parquet"))
    frame = pl.read_parquet(rows)
    assert frame["point"].n_unique() == 3
    fits = json.loads(rows.with_suffix(".json").read_text(encoding="utf-8"))
    assert len(fits) == 5
    text = (calibrated / "docs/calibration_summary.md").read_text(encoding="utf-8")
    assert "S4:" in text
    assert "S5:" in text
    assert "simulated" in text
    assert "held-out" in text
    assert list((calibrated / "experiments/calibration/figures").glob("*reliability_after.png"))


def test_s4_joins_the_holm_family_of_the_robustness_results(tmp_path: Path) -> None:
    entry = {"p_value": 0.01}
    hypotheses = {"ranker": {"S1": entry, "S2": entry}, "S3": {"lightgbm_flat": entry}}
    path = tmp_path / "experiments/robustness/results.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"metrics": {"hypotheses": hypotheses}}), encoding="utf-8")
    holm = abstain.primary_family(0.5, tmp_path)
    assert set(holm) == {"S1", "S2", "S3:lightgbm_flat", "S4"}
    assert holm["S4"] == pytest.approx(0.5)
