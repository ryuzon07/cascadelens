import ast
import math
import time
from pathlib import Path

import polars as pl
import pytest
from feature_factory import END, ONSETS, START, config, edges, make_input, make_metrics, row

from cascadelens.features import graph, mask, signal, temporal
from cascadelens.features.config import GroupName
from cascadelens.features.context import FeatureError, FeatureInput
from cascadelens.features.registry import (
    REGISTRY,
    build_case_features,
    feature_columns,
    to_model_matrix,
)
from cascadelens.features.simple import FEATURE_COLUMNS

PACKAGE = Path(__file__).parents[2] / "src" / "cascadelens" / "features"
TABLE = build_case_features(make_input(topology=edges()), config())
GROUPS: list[GroupName] = ["signal", "temporal", "graph", "structural", "mask"]


def isnan(value: float) -> bool:
    return value is not None and math.isnan(value)


def test_output_contract() -> None:
    assert TABLE.columns[:2] == ["case_id", "service"]
    assert TABLE["service"].to_list() == sorted(ONSETS)
    assert all(
        dtype == pl.Float64 for dtype in TABLE.select(pl.exclude("case_id", "service")).dtypes
    )


def test_earliest_service_has_rank_one_and_the_rest_follow() -> None:
    assert row(TABLE, "db")["temporal_onset_rank"] == 1
    assert row(TABLE, "api")["temporal_onset_rank"] == 2
    assert row(TABLE, "web")["temporal_onset_rank"] == 3
    assert isnan(row(TABLE, "idle")["temporal_onset_rank"])


def test_onset_delta_is_relative_to_the_earliest() -> None:
    assert row(TABLE, "db")["temporal_onset_delta"] == 0
    assert 0 < row(TABLE, "api")["temporal_onset_delta"] < row(TABLE, "web")["temporal_onset_delta"]


def test_z_scores_are_high_only_for_the_origin_and_its_downstream() -> None:
    for service in ("db", "api", "web"):
        assert row(TABLE, service)["signal_cpu_max_z"] > 20
    assert row(TABLE, "idle")["signal_cpu_max_z"] < 6


def test_upstream_and_downstream_fractions() -> None:
    api = row(TABLE, "api")

    assert api["graph_upstream_anomalous_fraction"] == 1
    assert api["graph_downstream_anomalous_fraction"] == 1
    assert isnan(row(TABLE, "db")["graph_upstream_anomalous_fraction"])
    assert row(TABLE, "web")["graph_upstream_anomalous_fraction"] == 0.5


def test_pagerank_and_structure() -> None:
    assert row(TABLE, "db")["graph_personalized_pagerank"] > 0
    assert row(TABLE, "web")["structural_out_degree"] == 2
    assert row(TABLE, "db")["structural_in_degree"] == 1
    assert row(TABLE, "api")["structural_betweenness"] > 0
    assert row(TABLE, "db")["graph_topology_available"] == 1


def test_topology_agreement_is_perfect_when_onset_follows_distance() -> None:
    assert row(TABLE, "db")["temporal_topology_agreement"] == pytest.approx(1.0)


def test_signal_statistics_by_hand() -> None:
    db = row(TABLE, "db")

    assert db["signal_cpu_baseline_median"] == pytest.approx(10, abs=0.1)
    assert db["signal_cpu_relative_change"] == pytest.approx(2, abs=0.05)
    # 200 of the 300 post-baseline samples are after the onset.
    assert db["signal_cpu_anomaly_duration"] == pytest.approx(200 / 300, abs=0.01)
    assert db["signal_cpu_peak_magnitude"] == pytest.approx(math.log1p(21), abs=0.1)
    assert db["signal_cpu_changepoint"] > row(TABLE, "idle")["signal_cpu_changepoint"]


def test_cumulative_counters_are_not_anomalies() -> None:
    assert row(TABLE, "idle")["signal_error_max_z"] < 10


def test_inject_time_cannot_reach_features() -> None:
    assert not {"inject_time", "root_cause_service"} & set(FeatureInput.__dataclass_fields__)


@pytest.mark.parametrize("path", sorted(PACKAGE.glob("*.py")), ids=lambda p: p.name)
def test_feature_code_never_names_the_label_or_injection(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    used = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    used |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Module | ast.FunctionDef | ast.ClassDef)
        and node.body
        and isinstance(node.body[0], ast.Expr)
    }
    used |= {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    }
    assert not used & {"inject_time", "root_cause_service", "run_id", "fault_type"}


def relabel(table: pl.DataFrame, names: dict[str, str]) -> pl.DataFrame:
    return table.with_columns(pl.col("service").replace(names))


def test_relabelling_permutes_rows_but_changes_no_value() -> None:
    names = {"db": "zulu", "api": "alpha", "web": "mike", "idle": "bravo"}
    metrics = relabel(make_metrics(), names)
    renamed_edges = edges(tuple((names[e.caller], names[e.callee]) for e in edges()))
    case = make_input(metrics, renamed_edges, tuple(sorted(names.values())))

    renamed = build_case_features(case, config())

    assert renamed["service"].to_list() == sorted(names.values())
    for old, new in names.items():
        first, second = row(TABLE, old), row(renamed, new)
        for column in feature_columns(GROUPS):
            assert first[column] == pytest.approx(second[column], nan_ok=True), column


def test_tied_earliest_services_do_not_depend_on_names() -> None:
    onsets: dict[str, int | None] = {"a": 200, "b": 200, "c": 240, "d": 260}
    pairs = (("c", "a"), ("c", "b"), ("d", "c"))
    first = build_case_features(
        make_input(make_metrics(onsets), edges(pairs), tuple(onsets)), config()
    )
    swap = {"a": "b", "b": "a"}
    second = build_case_features(
        make_input(relabel(make_metrics(onsets), swap), edges(pairs), tuple(onsets)), config()
    )

    assert row(first, "c")["temporal_topology_agreement"] == pytest.approx(
        row(second, "c")["temporal_topology_agreement"]
    )


def test_dropping_a_services_metrics_gives_nan_signal_and_correct_masks() -> None:
    metrics = make_metrics().filter(pl.col("service") != "api")

    table = build_case_features(make_input(metrics, edges()), config())

    api = row(table, "api")
    assert table.height == 4
    assert all(isnan(api[c]) for c in signal.COLUMNS)
    assert isnan(api["temporal_onset"])
    assert api["mask_cpu_available"] == 0
    assert api["mask_missing_fraction"] == 1
    assert api["mask_stopped_early"] == 1
    assert row(table, "db")["mask_cpu_available"] == 1
    assert row(table, "db")["mask_missing_fraction"] == 0


def test_null_samples_count_as_missing_and_stay_nan() -> None:
    metrics = make_metrics().with_columns(
        pl.when(pl.col("service") == "web").then(None).otherwise("value").alias("value")
    )

    web = row(build_case_features(make_input(metrics), config()), "web")

    assert isnan(web["signal_cpu_max_z"])
    assert web["mask_missing_fraction"] == 1
    assert web["mask_cpu_available"] == 0


def test_partial_missing_fraction_and_family_mask() -> None:
    cut = pl.col("timestamp") < START + 100
    drop = (pl.col("service") == "db") & (pl.col("metric") == "latency") & cut
    table = build_case_features(make_input(make_metrics().filter(~drop)), config())

    db = row(table, "db")
    assert db["mask_latency_available"] == 1
    assert db["mask_missing_fraction"] == pytest.approx(100 / 1200)


def test_telemetry_stopping_early_is_flagged_by_the_configured_gap() -> None:
    metrics = make_metrics().filter((pl.col("service") != "db") | (pl.col("timestamp") < END - 50))

    assert row(build_case_features(make_input(metrics), config()), "db")["mask_stopped_early"] == 1
    relaxed = config(mask={"stop_gap_seconds": 60})
    assert row(build_case_features(make_input(metrics), relaxed), "db")["mask_stopped_early"] == 0
    assert row(build_case_features(make_input(metrics), config()), "api")["mask_stopped_early"] == 0


def test_a_series_with_no_baseline_samples_stays_nan() -> None:
    metrics = make_metrics().filter(
        (pl.col("service") != "web") | (pl.col("timestamp") >= START + 200)
    )

    web = row(build_case_features(make_input(metrics), config()), "web")

    assert isnan(web["signal_cpu_baseline_median"])


@pytest.mark.parametrize("topology", [None, ()])
def test_missing_topology_gives_nan_and_a_flag(topology: tuple[()] | None) -> None:
    table = build_case_features(make_input(topology=topology), config())

    for service in ONSETS:
        values = row(table, service)
        assert values["graph_topology_available"] == 0
        assert isnan(values["graph_personalized_pagerank"])
        assert isnan(values["structural_betweenness"])
        assert isnan(values["temporal_topology_agreement"])


def test_a_service_missing_from_the_graph_is_nan_while_the_flag_stays() -> None:
    table = build_case_features(make_input(topology=edges((("api", "db"),))), config())

    idle = row(table, "idle")
    assert isnan(idle["structural_in_degree"])
    assert isnan(idle["graph_personalized_pagerank"])
    assert idle["graph_topology_available"] == 1


def test_a_service_only_in_the_graph_still_gets_a_row() -> None:
    table = build_case_features(make_input(topology=edges((("api", "ghost"),))), config())

    assert "ghost" in table["service"].to_list()
    assert row(table, "ghost")["mask_missing_fraction"] == 1


def test_no_anomaly_gives_nan_pagerank_not_an_error() -> None:
    quiet: dict[str, int | None] = {"a": None, "b": None}

    table = build_case_features(
        make_input(make_metrics(quiet), edges((("a", "b"),)), ("a", "b")), config()
    )

    assert isnan(row(table, "a")["graph_personalized_pagerank"])
    assert isnan(row(table, "a")["temporal_onset_rank"])


def test_nothing_is_imputed() -> None:
    table = build_case_features(make_input(), config())

    assert table["graph_personalized_pagerank"].is_nan().all()
    assert table.null_count().sum_horizontal()[0] == 0


@pytest.mark.parametrize("name", GROUPS)
def test_switching_a_group_off_removes_exactly_its_columns(name: GroupName) -> None:
    others = [group for group in GROUPS if group != name]

    table = build_case_features(make_input(topology=edges()), config(groups=others))

    assert set(TABLE.columns) - set(table.columns) == set(REGISTRY[name].columns)
    assert not any(c.startswith(REGISTRY[name].prefix) for c in table.columns)


@pytest.mark.parametrize("name", GROUPS)
def test_a_group_alone_has_its_prefix_on_every_column(name: GroupName) -> None:
    table = build_case_features(make_input(topology=edges()), config(groups=[name]))

    assert list(to_model_matrix(table).columns) == list(REGISTRY[name].columns)
    assert all(c.startswith(REGISTRY[name].prefix) for c in REGISTRY[name].columns)


def test_group_modules_export_their_prefix() -> None:
    assert signal.PREFIX == "signal_"
    assert temporal.PREFIX == "temporal_"
    assert graph.STRUCTURAL_PREFIX == "structural_"
    assert mask.PREFIX == "mask_"


def test_model_matrix_drops_the_keys() -> None:
    matrix = to_model_matrix(TABLE)

    assert "service" not in matrix.columns
    assert "case_id" not in matrix.columns
    assert matrix.width == TABLE.width - 2


@pytest.mark.parametrize(
    "column", ["service_id", "signal_service_index", "graph_service_hash", "signal_cpu_name"]
)
def test_model_matrix_rejects_service_like_columns(column: str) -> None:
    table = TABLE.with_columns(pl.lit(1.0).alias(column))

    with pytest.raises(FeatureError):
        to_model_matrix(table)


def test_model_matrix_rejects_text_and_unknown_columns() -> None:
    with pytest.raises(FeatureError, match="not numeric"):
        to_model_matrix(TABLE.with_columns(pl.lit("x").alias("signal_note")))
    with pytest.raises(FeatureError, match="no feature group"):
        to_model_matrix(TABLE.with_columns(pl.lit(1.0).alias("mystery")))


def test_part4_simple_features_are_untouched() -> None:
    assert len(FEATURE_COLUMNS) == 32
    assert not set(FEATURE_COLUMNS) & set(TABLE.columns)


def test_one_case_builds_in_about_a_second() -> None:
    case = make_input(topology=edges())
    start = time.perf_counter()

    build_case_features(case, config())

    assert time.perf_counter() - start < 1.5
