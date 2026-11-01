import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import polars as pl
import pytest
from baseline_factory import build_project, handle, make_case, make_metrics
from typer.testing import CliRunner

from cascadelens.baselines.base import BaselineError
from cascadelens.baselines.published import adapter, baro, lingam_randomwalk
from cascadelens.baselines.published.adapter import (
    fill_gaps,
    key_metrics,
    to_long,
    to_wide,
)
from cascadelens.baselines.published.baro import Baro, BaroMidpoint
from cascadelens.baselines.published.base import PublishedBaseline
from cascadelens.baselines.published.evaluate import PUBLISHED, evaluate_published
from cascadelens.baselines.published.lingam_randomwalk import LingamRandomWalk
from cascadelens.baselines.published.reproduce import (
    PUBLISHED_AVG5,
    compare,
    render,
)
from cascadelens.cli import app
from cascadelens.config import BaselineConfig
from cascadelens.data.schema import METRICS_SCHEMA, Case

RUNNER = CliRunner()
CONFIGS = Path(__file__).parents[2] / "configs" / "baselines"
EXPERIMENT = Path(__file__).parents[2] / "experiments" / "reproduction"
Call = dict[str, Any]


def config(cut: float | None = 0.25, method: str = "baro") -> BaselineConfig:
    params = {"method": method, "cut_fraction": cut}
    return BaselineConfig(name=method, supervised=False, seeds=[0], params=params)


def fake_baro(calls: list[Call]) -> Any:  # noqa: ANN401
    def run(method: str, frame: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        calls.append({"method": method, "frame": frame, **kwargs})
        columns = [c for c in frame.columns if c != "time"]
        return {"scores": {c: float(i) for i, c in enumerate(columns)}, "ranks": columns[::-1]}

    return run


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[Call]:
    recorded: list[Call] = []
    monkeypatch.setattr(baro, "call_library", fake_baro(recorded))
    monkeypatch.setattr(lingam_randomwalk, "call_library", fake_baro(recorded))
    return recorded


def fitted(method: type[PublishedBaseline]) -> PublishedBaseline:
    baseline = method()
    baseline.fit([], config())
    return baseline


def test_wide_frame_round_trips() -> None:
    metrics = make_metrics({"api": 100, "db": None})

    back = to_long(to_wide(metrics))

    expected = metrics.sort("service", "metric", "timestamp")
    assert back.equals(expected)


def test_round_trip_keeps_missing_samples_and_hyphenated_names() -> None:
    metrics = pl.DataFrame(
        [
            ("ts-auth-service", "container-cpu-usage-seconds-total", 1, 1.5),
            ("ts-auth-service", "container-cpu-usage-seconds-total", 2, None),
            ("carts-db", "cpu", 1, 2.0),
            ("carts-db", "cpu", 2, 3.0),
        ],
        schema=METRICS_SCHEMA,
        orient="row",
    )

    wide = to_wide(metrics)

    assert to_long(wide).equals(metrics.sort("service", "metric", "timestamp"))
    assert wide.service_of("carts-db_cpu") == "carts-db"
    assert wide.service_of("ts-auth-service_container-cpu-usage-seconds-total") == (
        "ts-auth-service"
    )


def test_wide_frame_has_time_first_and_sorted_rows() -> None:
    wide = to_wide(make_metrics({"b": None, "a": None}).reverse())

    assert wide.frame.columns[0] == "time"
    assert wide.frame["time"].is_sorted()
    assert wide.frame.columns[1:] == sorted(wide.frame.columns[1:])


def test_gaps_fill_like_main_py() -> None:
    frame = pl.DataFrame(
        {
            "time": [1, 2, 3, 4],
            "a": [None, 1.0, float("inf"), 2.0],
            "b": [1.0, float("nan"), None, 2.0],
        }
    )

    filled = fill_gaps(frame)

    assert filled["a"].to_list() == [0.0, 1.0, 1.0, 2.0]
    assert filled["b"].to_list() == [1.0, 1.0, 1.0, 2.0]


def test_key_metrics_filter_by_name_and_difference_counters() -> None:
    metrics = pl.DataFrame(
        [
            ("api", "istio-request-total", 1, 10.0),
            ("api", "istio-request-total", 2, 13.0),
            ("api", "container-memory-usage-bytes", 1, 7.0),
            ("192-168-0-1-9100", "node-cpu-seconds-total", 1, 1.0),
        ],
        schema=METRICS_SCHEMA,
        orient="row",
    )

    table = key_metrics(metrics)

    counter = table.filter(pl.col("metric") == "istio-request-total")
    assert table["service"].unique().to_list() == ["api"]
    assert counter["value"].to_list() == [None, 3.0]


def test_key_metric_wrappers_still_return_every_service(calls: list[Call]) -> None:
    baseline = LingamRandomWalk()
    params = {"method": "lingam_randomwalk", "metric_set": "key"}
    baseline.fit([], BaselineConfig(name="x", supervised=False, seeds=[0], params=params))
    case = make_case(onsets={"api": 100, "db": 100})

    table = baseline.predict(handle(case))

    assert sorted(table["service"].to_list()) == ["api", "db"]
    assert "api_cpu" in calls[0]["frame"].columns
    assert "api_request-total" not in calls[0]["frame"].columns


def test_empty_metrics_fail_loudly() -> None:
    with pytest.raises(BaselineError, match="no metrics"):
        to_wide(pl.DataFrame(schema=METRICS_SCHEMA))


def test_a_missing_library_says_what_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(__import__("sys").modules, "RCAEval", None)

    with pytest.raises(BaselineError, match="uv sync --extra published"):
        adapter.call_library("baro", pl.DataFrame({"time": [1]}))


def test_importing_the_library_leaves_the_plot_style_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "RCAEval"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "e2e.py").write_text(
        "import matplotlib.pyplot as plt\n"
        'plt.style.use("fivethirtyeight")\n'
        "def baro(data):\n"
        "    return {}\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(tmp_path)
    for name in ("RCAEval", "RCAEval.e2e"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    before = dict(mpl.rcParams)

    try:
        assert adapter.call_library("baro", pl.DataFrame({"time": [1]})) == {}
        assert dict(mpl.rcParams) == before
    finally:
        for name in ("RCAEval", "RCAEval.e2e"):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("method", list(PUBLISHED.values()))
def test_wrappers_return_every_service_once(
    method: type[PublishedBaseline], calls: list[Call]
) -> None:
    case = make_case(onsets={"api": 120, "cache": None, "db": 100})

    table = fitted(method).predict(handle(case))

    assert table.columns == ["case_id", "service", "score"]
    assert sorted(table["service"].to_list()) == ["api", "cache", "db"]
    assert table["score"].is_finite().all()


@pytest.mark.parametrize("method", list(PUBLISHED.values()))
def test_ties_break_by_name_and_output_repeats(
    method: type[PublishedBaseline], monkeypatch: pytest.MonkeyPatch
) -> None:
    def tied(name: str, frame: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"scores": {}, "ranks": []}

    monkeypatch.setattr(baro, "call_library", tied)
    monkeypatch.setattr(lingam_randomwalk, "call_library", tied)
    case = handle(make_case(onsets={"z": None, "m": None, "a": None}))

    first = fitted(method).predict(case)

    assert first["service"].to_list() == ["a", "m", "z"]
    assert first.equals(fitted(method).predict(case))


def test_service_score_is_its_best_column(monkeypatch: pytest.MonkeyPatch) -> None:
    def scored(name: str, frame: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"scores": {"db_cpu": 1.0, "db_request-total": 9.0, "api_cpu": 5.0}}

    monkeypatch.setattr(baro, "call_library", scored)

    table = fitted(Baro).predict(handle(make_case(onsets={"api": 100, "db": 100, "idle": None})))

    assert dict(table.select("service", "score").iter_rows()) == {
        "db": 9.0,
        "api": 5.0,
        "idle": 4.0,
    }


def test_baro_time_comes_from_the_window_alone(calls: list[Call]) -> None:
    baseline = Baro()
    baseline.fit([], config(0.25))

    baseline.predict(handle(make_case(inject_time=1100)))

    assert calls[0]["inject_time"] == round(1000 + 0.25 * 199)


def test_baro_output_ignores_inject_time(calls: list[Call]) -> None:
    fitted(Baro).predict(handle(make_case(inject_time=1100)))
    fitted(Baro).predict(handle(make_case(inject_time=1190)))

    assert calls[0]["inject_time"] == calls[1]["inject_time"]


def test_midpoint_variant_splits_at_the_middle(calls: list[Call]) -> None:
    baseline = BaroMidpoint()
    baseline.fit([], config(0.5))

    baseline.predict(handle(make_case()))

    assert calls[0]["inject_time"] == round(1000 + 0.5 * 199)


def test_baro_needs_a_cut_fraction(calls: list[Call]) -> None:
    baseline = Baro()
    baseline.fit([], config(None))

    baseline.predict(handle(make_case()))

    assert "cut_fraction" in baseline.failures[0].error


def test_lingam_gets_no_anomaly_time(calls: list[Call]) -> None:
    fitted(LingamRandomWalk).predict(handle(make_case()))

    assert calls[0]["inject_time"] is None
    assert calls[0]["n_iter"] == len(calls[0]["frame"].columns) - 1


@pytest.mark.parametrize("method", list(PUBLISHED.values()))
def test_output_survives_shuffled_labels(
    method: type[PublishedBaseline], calls: list[Call]
) -> None:
    def training(labels: list[str]) -> list[Case]:
        return [make_case(f"t{i}", label) for i, label in enumerate(labels)]

    first, second = method(), method()
    first.fit([handle(c) for c in training(["api", "db"])], config())
    second.fit([handle(c) for c in training(["db", "api"])], config())
    case = handle(make_case())

    assert first.predict(case).equals(second.predict(case))


@pytest.mark.parametrize("method", list(PUBLISHED.values()))
def test_fit_never_touches_training_cases(method: type[PublishedBaseline]) -> None:
    class Forbidden:
        @property
        def record(self) -> None:
            raise AssertionError("training case touched")

    method().fit([Forbidden()], config())  # type: ignore[list-item]


def test_a_failing_case_is_logged_counted_and_does_not_crash(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def broken(name: str, frame: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        raise ValueError("singular matrix")

    monkeypatch.setattr(baro, "call_library", broken)
    baseline = fitted(Baro)

    table = baseline.predict(handle(make_case("bad-case")))

    assert table["score"].to_list() == [0.0, 0.0, 0.0]
    assert baseline.failures[0].case_id == "bad-case"
    assert "singular matrix" in baseline.failures[0].error
    assert "bad-case" in caplog.text


def test_the_librarys_silent_fallback_counts_as_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fallback(name: str, frame: pl.DataFrame, **kwargs: Any) -> dict[str, Any]:  # noqa: ANN401
        return {"adj": [], "node_names": [], "ranks": ["api_cpu"]}

    monkeypatch.setattr(lingam_randomwalk, "call_library", fallback)
    baseline = fitted(LingamRandomWalk)

    baseline.predict(handle(make_case()))

    assert "column order" in baseline.failures[0].error


def test_runtime_is_recorded_and_a_repeat_is_memoized(calls: list[Call]) -> None:
    baseline = fitted(Baro)
    case = handle(make_case())

    baseline.predict(case)
    baseline.predict(case)

    assert len(calls) == 1
    assert baseline.runtimes["case-1"] >= 0


def test_compare_applies_the_tolerance_to_the_system_mean() -> None:
    published = PUBLISHED_AVG5["baro"]
    faults = ("cpu", "mem", "disk", "delay", "loss")
    scores = {suite: dict(zip(faults, values, strict=True)) for suite, values in published.items()}
    scores["RE1-OB"] = {fault: value - 0.04 for fault, value in scores["RE1-OB"].items()}
    scores["RE1-TT"] = {fault: value - 0.06 for fault, value in scores["RE1-TT"].items()}

    rows = compare("baro", scores)

    assert rows["RE1-OB"]["passed"] is True
    assert rows["RE1-SS"]["difference"] == pytest.approx(0)
    assert rows["RE1-TT"]["passed"] is False


def test_report_shows_the_verdict_and_notes() -> None:
    faults = ("cpu", "mem", "disk", "delay", "loss")
    scores = {
        suite: dict(zip(faults, values, strict=True))
        for suite, values in PUBLISHED_AVG5["baro"].items()
    }
    rows = compare("baro", scores)
    one = dict.fromkeys(rows, 0.5)

    text = render("baro", rows, one, 0, one, 20, "data.csv", "A cause.")

    assert "**Result: PASS**" in text
    assert "A cause." in text
    assert "Table 5" in text


def test_published_numbers_match_the_plan() -> None:
    plan = (Path(__file__).parents[2] / "docs" / "reproduction" / "PLAN.md").read_text("utf-8")

    for values in PUBLISHED_AVG5["baro"].values():
        assert " | ".join(f"{value:.2f}" for value in values) in plan


def test_primary_baro_keeps_identical_inputs() -> None:
    assert "metric_set" not in (CONFIGS / "baro.yaml").read_text(encoding="utf-8")


def test_shipped_published_configs_name_a_known_method() -> None:
    assert {path.stem for path in CONFIGS.glob("*.yaml")} >= set(PUBLISHED)


@pytest.fixture
def project(tmp_path: Path, calls: list[Call]) -> Path:
    build_project(tmp_path)
    target = tmp_path / "experiments" / "reproduction" / "evaluation"
    target.mkdir(parents=True)
    source = (EXPERIMENT / "evaluation" / "config.yaml").read_text(encoding="utf-8")
    (target / "config.yaml").write_text(source.replace("10000", "100"), encoding="utf-8")
    return tmp_path


def test_evaluation_writes_every_required_field(project: Path) -> None:
    evaluate_published(project)

    directory = project / "experiments" / "reproduction" / "evaluation"
    payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    assert {"config_hash", "git_commit", "data_checksum", "seed", "metrics"} <= set(payload)
    assert payload["data_checksum"] != "unknown"
    assert payload["metrics"]["operational"]["baro"]["failed_cases"] == 0
    assert payload["metrics"]["operational"]["baro"]["mean_seconds_per_case"] >= 0
    text = (directory / "results.md").read_text(encoding="utf-8")
    assert "Unsupervised: group_run" in text
    assert "| baro | loso | shop |" in text
    assert "not a fair score" in text


def test_cli_reports_unknown_reproduction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    result = RUNNER.invoke(app, ["published", "reproduce", "--name", "nope"])

    assert result.exit_code == 1
    assert "published reproduce failed" in result.output


def test_real_library_on_the_synthetic_case() -> None:
    pytest.importorskip("RCAEval")
    baseline = fitted(Baro)

    table = baseline.predict(handle(make_case(onsets={"api": 150, "db": 100, "idle": None})))

    assert baseline.failures == []
    assert table.height == 3
