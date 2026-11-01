import json
from pathlib import Path

import polars as pl
import pytest
from baseline_factory import build_project
from ranker_factory import write_project_configs
from typer.testing import CliRunner

import cascadelens.eval.bootstrap as bootstrap
from cascadelens.baselines.experiments import baselines_clean
from cascadelens.cli import app
from cascadelens.config import RankerConfig
from cascadelens.eval import robustness
from cascadelens.eval.robustness import (
    CLEAN,
    RankerArm,
    RobustnessConfig,
    aligned,
    assert_test_only,
    derive_stage,
    difference_of_differences,
    freeze_variants,
    load_context,
    load_outcomes,
    report_stage,
    suite_points,
    train_stage,
    unit_path,
    variant_settings,
)
from cascadelens.models.data import RankerError
from cascadelens.models.freeze import find_frozen
from cascadelens.splits.splitters import Fold

SUITE = """seed: 0
perturbations:
  - {id: P2, name: failure_correlated_loss, realism: simulated, params: {rate: [0.3]}}
  - {id: P4, name: window_shift, realism: simulated, params: {offset_seconds: [30]}}
  - {id: P9, name: unseen_fault_type, realism: held-out, params: {split: [lofo]}}
  - {id: P10, name: unseen_system, realism: held-out, params: {split: [loso]}}
"""
CONFIG = """seed: 0
n_resamples: 200
alpha: 0.05
workers: 1
ranker: ranker_augmented
plain_ranker: ranker
prior: frequency
primary: [S1, S2, S3, S4, S7]
s3_point: {id: P2, rate: 0.3}
s3_baselines: [lightgbm_flat, lightgbm_flat_masking, frequency, rule]
curve_methods:
  supervised: [ranker_augmented, ranker, lightgbm_flat_masking, lightgbm_flat, frequency]
  unsupervised: [rule]
variants:
  - {name: ranker, kind: two_stage}
  - {name: ranker_augmented, kind: two_stage, augment: true}
  - {name: lightgbm_flat, kind: flat}
  - {name: lightgbm_flat_masking, kind: flat, augment: true}
  - {name: ablate_mask, kind: two_stage, augment: true, drop_group: mask}
  - {name: single_stage_augmented, kind: single_stage, augment: true}
  - {name: frequency, kind: frequency}
  - {name: rule, kind: rule}
"""


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("robustness")
    build_project(root)
    write_project_configs(root)
    baselines_clean(root)
    (root / "configs" / "perturbations.yaml").write_text(SUITE, encoding="utf-8")
    (root / "configs" / "augment.yaml").write_text(
        "seed: 0\nrandom_copies: 1\nfailure_copies: 0\n", encoding="utf-8"
    )
    directory = root / "experiments" / "robustness"
    directory.mkdir(parents=True)
    (directory / "config.yaml").write_text(CONFIG, encoding="utf-8")
    return root


@pytest.fixture(scope="module")
def swept(project: Path) -> Path:
    context = load_context(True, project)
    derive_stage(context)
    train_stage(context)
    return project


def test_sweep_refuses_a_config_that_is_not_frozen(project: Path) -> None:
    with pytest.raises(RankerError, match="not frozen"):
        load_context(False, project)


def test_points_start_clean_and_leave_out_split_references(project: Path) -> None:
    points = suite_points(project)
    assert points[0].key == CLEAN
    assert {p.identifier for p in points} == {CLEAN, "P2", "P4"}


def test_outcomes_line_up_across_methods(swept: Path) -> None:
    frame = load_outcomes(load_context(True, swept))
    runs = frame.filter(pl.col("split") == "group_run")
    tallies = runs.group_by("method", "point").agg(pl.col("case_id").n_unique())
    assert set(tallies["case_id"].to_list()) == {96}
    point = runs.filter(pl.col("point") != CLEAN)["point"][0]
    arms = [
        runs.filter((pl.col("method") == m) & (pl.col("point") == point))
        for m in ("ranker_augmented", "lightgbm_flat_masking", "rule")
    ]
    ids = [a["case_id"].to_list() for a in aligned(arms)]
    assert ids[0] == ids[1] == ids[2]


def test_held_out_splits_score_clean_data_only(swept: Path) -> None:
    frame = load_outcomes(load_context(True, swept))
    assert set(frame.filter(pl.col("split") == "loso")["point"].unique()) == {CLEAN}


def test_finished_units_are_skipped(swept: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    context = load_context(True, swept)

    def refuse(*args: object) -> None:
        raise AssertionError("A finished unit must not run again")

    monkeypatch.setattr(robustness, "build_arm", refuse)
    assert len(train_stage(context)) == 24


def test_a_config_change_gives_a_new_unit(swept: Path) -> None:
    context = load_context(True, swept)
    variant = context.config.variant("ranker")
    before = unit_path(variant, "group_run", context)
    context.status = "frozen"
    assert unit_path(variant, "group_run", context) != before


def test_scoring_reads_test_cases_only(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = project
    context = load_context(True, root)
    variant = context.config.variant("ranker")
    path = unit_path(variant, "lofo", context)
    path.unlink(missing_ok=True)
    seen: list[tuple[str, ...]] = []
    original = RankerArm.score

    def spy(self: RankerArm, derived: object, case_ids: tuple[str, ...]) -> pl.DataFrame:
        seen.append(tuple(case_ids))
        return original(self, derived, case_ids)  # type: ignore[arg-type]

    monkeypatch.setattr(RankerArm, "score", spy)
    train_stage(context, ["ranker"])
    assert seen
    from cascadelens.baselines.runner import load_split

    split, _ = load_split("lofo", context.dataset, root)
    tests = {fold.test_ids for fold in split.folds}
    assert all(case_ids in tests for case_ids in seen)


def test_the_test_only_check_refuses_training_cases() -> None:
    fold = Fold(train_ids=("a",), val_ids=("b",), test_ids=("c",))
    assert_test_only(fold, ("c",))
    with pytest.raises(RankerError, match="test cases only"):
        assert_test_only(fold, ("a",))


def test_difference_of_differences_resamples_all_arms_once(
    swept: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = load_context(True, swept)
    frame = load_outcomes(context).filter(pl.col("split") == "group_run")
    key = next(p.key for p in context.points if p.identifier == "P2")

    def arm(method: str, point: str) -> pl.DataFrame:
        return frame.filter((pl.col("method") == method) & (pl.col("point") == point))

    calls: list[int] = []
    original = bootstrap.resample_counts

    def counted(n_groups: int, n: int, seed: int) -> object:
        calls.append(n)
        return original(n_groups, n, seed)

    monkeypatch.setattr(bootstrap, "resample_counts", counted)
    result = difference_of_differences(
        arm("ranker_augmented", CLEAN),
        arm("rule", CLEAN),
        arm("ranker_augmented", key),
        arm("rule", key),
        context.config,
    )
    assert calls == [200]
    assert result["lower"] <= result["estimate"] <= result["upper"]
    with pytest.raises(RankerError, match="same cases"):
        aligned([arm("rule", CLEAN), arm("rule", CLEAN).head(10)])


def test_report_labels_and_tags_everything(swept: Path) -> None:
    context = load_context(True, swept)
    payload = report_stage(context)
    assert payload["status"] == "exploratory"
    assert {"S1", "S2"} <= set(payload["hypotheses"]["ranker"])
    assert set(payload["hypotheses"]["S3"]) == set(context.config.s3_baselines)
    holm = payload["hypotheses"]["holm"]
    assert all(value is not None for value in holm.values())
    assert {row["realism"] for row in payload["held_out_folds"]} == {"held-out"}
    assert {r["realism"] for r in payload["grid"]} == {"clean", "simulated"}
    text = (swept / "docs" / "robustness_summary.md").read_text(encoding="utf-8")
    assert "Status: exploratory" in text
    assert "not a changing one" in text
    assert "S3: margin growth over lightgbm_flat_masking" in text
    figures = swept / "experiments" / "robustness" / "figures"
    assert (figures / "P2_supervised.png").is_file()
    assert (figures / "P4_unsupervised.png").is_file()
    saved = json.loads((swept / "experiments/robustness/results.json").read_text(encoding="utf-8"))
    assert saved["metrics"]["status"] == "exploratory"


def test_ablation_removes_one_group(swept: Path) -> None:
    context = load_context(True, swept)
    settings = variant_settings(context.config.variant("ablate_mask"), context.ranker)
    assert isinstance(settings, RankerConfig)
    assert "mask" not in settings.groups
    assert variant_settings(context.config.variant("ranker"), context.ranker) == context.ranker


def test_freeze_helper_records_every_ranker_variant(swept: Path, tmp_path: Path) -> None:
    import shutil

    root = tmp_path / "copy"
    shutil.copytree(swept, root)
    freeze_variants("candidate", root)
    context = load_context(False, root)
    assert context.status == "frozen"
    record = find_frozen(context.ranker, root)
    assert record is not None
    assert {"ranker_augmented", "ablate_mask", "two_stage_ranker"} <= set(record.families["loso"])
    assert isinstance(load_context(False, root).config, RobustnessConfig)
    variant = context.config.variant("ranker_augmented")
    frozen = robustness.run_unit(variant, "loso", context)
    assert frozen.is_file()


def test_cli_refuses_without_exploratory(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)
    runner = CliRunner()
    first = runner.invoke(app, ["experiments", "robustness", "--stage", "report"])
    assert first.exit_code == 1
    assert "not frozen" in first.output
    assert runner.invoke(app, ["probes", "identity"]).exit_code == 1
