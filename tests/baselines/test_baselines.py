import json
from pathlib import Path

import polars as pl
import pytest
from baseline_factory import build_project, handle, make_case
from typer.testing import CliRunner

from cascadelens.baselines.base import Baseline, BaselineError, SummaryCache, score_table
from cascadelens.baselines.frequency import FrequencyPrior
from cascadelens.baselines.lightgbm_flat import FlatParams, LightGBMFlat, sample_search_space
from cascadelens.baselines.rule import RuleOnset, RuleUpstream
from cascadelens.baselines.runner import BASELINES, load_baseline_config, run_baseline
from cascadelens.cli import app
from cascadelens.config import BaselineConfig
from cascadelens.data.schema import METRICS_SCHEMA, Case, TopologyEdge
from cascadelens.eval.metrics import per_case_outcomes
from cascadelens.features.simple import FEATURE_COLUMNS

RUNNER = CliRunner()
SIMPLE = BaselineConfig(name="x", supervised=False, seeds=[0])
FLAT = BaselineConfig(
    name="lightgbm_flat",
    supervised=True,
    seeds=[0],
    params={"search_budget": 2, "num_boost_round": 10, "num_threads": 1},
)


def train_cases() -> list[Case]:
    return [
        make_case(f"t{index}", root, {"api": 100, "db": 120, "web": 140}, run_id=f"run-{index}")
        for index, root in enumerate(["api", "api", "db", "web", "db", "api"])
    ]


def val_cases() -> list[Case]:
    return [make_case("v0", "db", {"api": 120, "db": 100, "web": 140}, run_id="run-v")]


def fitted(method: type[Baseline]) -> Baseline:
    baseline = method()
    config = FLAT if method is LightGBMFlat else SIMPLE
    baseline.fit([handle(c) for c in train_cases()], config, [handle(c) for c in val_cases()], 0)
    return baseline


def test_rule_config_files_exist_for_every_baseline() -> None:
    for name in BASELINES:
        assert load_baseline_config(name, Path(__file__).parents[2]).name == name


@pytest.mark.parametrize("method", list(BASELINES.values()))
def test_every_baseline_returns_all_services_once(method: type[Baseline]) -> None:
    case = make_case(onsets={"api": 120, "cache": None, "db": 100})

    table = fitted(method).predict(handle(case))

    assert table.columns == ["case_id", "service", "score"]
    assert sorted(table["service"].to_list()) == ["api", "cache", "db"]
    assert table["case_id"].unique().to_list() == ["case-1"]
    assert table["score"].is_finite().all()


@pytest.mark.parametrize("method", list(BASELINES.values()))
def test_output_is_sorted_with_ties_by_name_and_repeatable(method: type[Baseline]) -> None:
    case = make_case(onsets={"z": None, "m": None, "a": None})

    first = fitted(method).predict(handle(case))
    second = fitted(method).predict(handle(case))

    assert first.equals(second)
    assert first.equals(first.sort(["score", "service"], descending=[True, False]))


def test_score_table_breaks_ties_by_service_name() -> None:
    assert score_table("c", {"b": 1.0, "a": 1.0, "c": 2.0})["service"].to_list() == ["c", "a", "b"]


def test_frequency_regression_on_a_fixed_toy_dataset() -> None:
    # Training labels are api 3, db 2, web 1, so the prior always answers api.
    baseline = fitted(FrequencyPrior)
    tests = [
        make_case("a", "api", {"api": 100, "db": 100, "web": 100}),
        make_case("b", "api", {"api": 100, "db": 100, "web": 100}),
        make_case("c", "db", {"api": 100, "db": 100, "web": 100}),
        make_case("d", "web", {"api": 100, "db": 100, "web": 100}),
    ]
    predictions = pl.concat(baseline.predict(handle(c)) for c in tests)
    truth = pl.DataFrame(
        {
            "case_id": [c.case_id for c in tests],
            "root_cause_service": [c.root_cause_service for c in tests],
            "group_id": [c.case_id for c in tests],
        }
    )

    assert per_case_outcomes(predictions, truth)["hit@1"].mean() == 0.5
    assert baseline.predict(handle(tests[0]))["score"].to_list() == pytest.approx(
        [1 / 2, 1 / 3, 1 / 6]
    )


def test_frequency_gives_zero_to_a_service_unseen_in_a_seen_system() -> None:
    table = fitted(FrequencyPrior).predict(handle(make_case(onsets={"api": 100, "new": 100})))

    assert dict(table.select("service", "score").iter_rows()) == {"api": 0.5, "new": 0.0}


def test_frequency_falls_back_to_uniform_for_an_unseen_system() -> None:
    case = make_case(system="other", onsets={"p": 100, "q": 100, "r": 100, "s": 100})

    table = fitted(FrequencyPrior).predict(handle(case))

    assert table["score"].to_list() == [0.25] * 4
    assert table["service"].to_list() == ["p", "q", "r", "s"]


def test_frequency_falls_back_when_no_candidate_was_seen() -> None:
    table = fitted(FrequencyPrior).predict(handle(make_case(onsets={"p": 100, "q": 100})))

    assert table["score"].to_list() == [0.5, 0.5]


def test_rule_ranks_the_earliest_service_first() -> None:
    table = fitted(RuleOnset).predict(handle(make_case()))

    assert table["service"].to_list() == ["db", "api", "cache"]


@pytest.mark.parametrize("method", [RuleOnset, RuleUpstream])
def test_rules_ignore_inject_time(method: type[Baseline]) -> None:
    first = fitted(method).predict(handle(make_case(inject_time=1100)))
    second = fitted(method).predict(handle(make_case(inject_time=1199)))

    assert first.equals(second)


@pytest.mark.parametrize("method", [RuleOnset, RuleUpstream])
def test_unsupervised_output_survives_shuffled_labels(method: type[Baseline]) -> None:
    labels = ["web", "db", "api", "api", "web", "db"]
    shuffled = [
        make_case(c.case_id, label, {"api": 100, "db": 120, "web": 140}, run_id=c.run_id)
        for c, label in zip(train_cases(), labels, strict=True)
    ]
    first, second = fitted(method), method()
    second.fit([handle(c) for c in shuffled], SIMPLE)
    case = handle(make_case())

    assert first.predict(case).equals(second.predict(case))


@pytest.mark.parametrize("method", [RuleOnset, RuleUpstream])
def test_unsupervised_fit_does_not_load_training_cases(method: type[Baseline]) -> None:
    class Forbidden:
        @property
        def record(self) -> None:
            raise AssertionError("training case touched")

    method().fit([Forbidden()], SIMPLE)  # type: ignore[list-item]


def test_rule_upstream_without_topology_is_max_anomaly() -> None:
    case = make_case(onsets={"a": 100, "b": None})

    assert fitted(RuleUpstream).predict(handle(case))["service"][0] == "a"


def test_rule_upstream_breaks_capped_ties_by_duration_not_name() -> None:
    metrics = pl.concat(
        [
            pl.DataFrame(
                {
                    "service": [service] * 200,
                    "metric": ["cpu"] * 200,
                    "timestamp": list(range(1000, 1200)),
                    "value": [0.0] * start + [5000.0] * (200 - start),
                },
                schema=METRICS_SCHEMA,
            )
            for service, start in (("aaa", 180), ("zzz", 100))
        ]
    )
    case = make_case().model_copy(update={"metrics": metrics})

    table = fitted(RuleUpstream).predict(handle(case))

    assert table["service"].to_list() == ["zzz", "aaa"]


def test_rule_upstream_duration_never_overrides_a_real_peak_difference() -> None:
    def series(service: str, level: float, start: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "service": [service] * 200,
                "metric": ["cpu"] * 200,
                "timestamp": list(range(1000, 1200)),
                "value": [1.0, 2.0, 3.0, 2.0] * 15 + [2.0] * (start - 60) + [level] * (200 - start),
            },
            schema=METRICS_SCHEMA,
        )

    metrics = pl.concat([series("short", 40.0, 190), series("long", 20.0, 100)])
    case = make_case().model_copy(update={"metrics": metrics})

    assert fitted(RuleUpstream).predict(handle(case))["service"][0] == "short"


def test_rule_upstream_demotes_a_service_with_anomalous_callee() -> None:
    topology = (TopologyEdge(caller="api", callee="db"),)
    case = make_case(onsets={"api": 60, "db": 100}, topology=topology)
    open_case = make_case(onsets={"api": 60, "db": 100}, topology=())

    assert fitted(RuleUpstream).predict(handle(case))["service"][0] == "db"
    assert fitted(RuleUpstream).predict(handle(open_case)).height == 2


def test_flat_features_hold_no_identity_graph_or_mask_columns() -> None:
    banned = ("service", "name", "id", "degree", "rank", "between", "mask", "missing", "system")

    for column in FEATURE_COLUMNS:
        assert not any(word in column.lower().split("_") for word in banned)
    assert all(c.split("_", 1)[1] in _statistics() for c in FEATURE_COLUMNS)


def _statistics() -> set[str]:
    return {"max_z", "relative_change", "anomaly_duration", "peak_magnitude"}


def test_flat_training_is_identical_on_the_same_seed() -> None:
    first, second = fitted(LightGBMFlat), fitted(LightGBMFlat)
    case = handle(make_case())

    assert first.predict(case).equals(second.predict(case))


def test_flat_seed_changes_the_model_but_not_the_search() -> None:
    baseline = LightGBMFlat(SummaryCache())
    train = [handle(c) for c in train_cases()]
    val = [handle(c) for c in val_cases()]

    baseline.fit(train, FLAT, val, 0)
    baseline.fit(train, FLAT, val, 1)

    assert len(baseline.chosen_params()) == 1


def test_search_space_is_deterministic_and_sized_by_the_budget() -> None:
    assert sample_search_space(5, 1) == sample_search_space(5, 1)
    assert sample_search_space(5, 1) != sample_search_space(5, 2)
    assert len(sample_search_space(30, 0)) == FlatParams().search_budget


def test_flat_requires_validation_cases() -> None:
    with pytest.raises(BaselineError, match="validation"):
        LightGBMFlat().fit([handle(c) for c in train_cases()], FLAT)


def test_flat_refuses_masking_for_now() -> None:
    config = FLAT.model_copy(update={"params": {"masking_augmentation": True}})

    with pytest.raises(BaselineError, match="Masking"):
        LightGBMFlat().fit([handle(c) for c in train_cases()], config)


def test_flat_must_be_fitted_first() -> None:
    with pytest.raises(BaselineError, match="fitted"):
        LightGBMFlat().predict(handle(make_case()))


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("project")
    build_project(root)
    return root


def test_runner_writes_a_result_with_every_required_field(project: Path) -> None:
    run = run_baseline("frequency", "group_run", 0, project)

    directory = project / "experiments" / "runs" / "frequency" / "group_run" / "seed_0"
    payload = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    assert {"config_hash", "git_commit", "data_checksum", "seed", "metrics"} <= set(payload)
    assert payload["data_checksum"] != "unknown"
    assert payload["metrics"]["n_cases"] == 96
    assert len(payload["metrics"]["folds"]) == 5
    assert (directory / "cache" / "predictions.parquet").is_file()
    assert run.outcomes.height == 96


def test_runner_tests_every_case_once(project: Path) -> None:
    run = run_baseline("rule", "group_run", 0, project)

    assert run.outcomes["case_id"].n_unique() == 96
    assert run.metrics["top1"] == 1.0


def test_runner_refuses_a_split_from_another_dataset(tmp_path: Path) -> None:
    build_project(tmp_path)
    path = tmp_path / "data" / "splits" / "group_run" / "split.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["data_manifest_checksum"] = "other"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BaselineError, match="different dataset"):
        run_baseline("frequency", "group_run", 0, tmp_path)


def test_runner_refuses_an_unknown_baseline(project: Path) -> None:
    with pytest.raises(BaselineError, match="Unknown baseline"):
        run_baseline("baro", "group_run", 0, project)


def test_loso_frequency_uses_the_uniform_floor(project: Path) -> None:
    run = run_baseline("frequency", "loso", 0, project)

    assert run.predictions["score"].unique().to_list() == [0.25]


def test_cli_run(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)

    result = RUNNER.invoke(
        app, ["baselines", "run", "--name", "rule", "--split", "lofo", "--seed", "0"]
    )

    assert result.exit_code == 0, result.output
    assert "Top-1 1.000" in result.output


def test_cli_run_fails_cleanly(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(project)

    result = RUNNER.invoke(app, ["baselines", "run", "--name", "rule", "--split", "nope"])

    assert result.exit_code == 1
    assert "baselines run failed" in result.output
