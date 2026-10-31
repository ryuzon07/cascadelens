import math

import numpy as np
import polars as pl
import pytest
from feature_factory import END, START, config, edges, make_input, row

from cascadelens.data.schema import METRICS_SCHEMA
from cascadelens.features.inference import InferenceSettings, infer_edges
from cascadelens.features.registry import build_case_features, feature_columns

SETTINGS = InferenceSettings()
BINS = 80
INFERRED = {"topology": {"source": "inferred_if_missing", "inference": {"bin_seconds": 5}}}


def noise(seed: int, size: int = BINS) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=size)


def links(services: list[str], signals: np.ndarray) -> set[tuple[str, str]]:
    found = infer_edges(services, signals, SETTINGS)
    return {(edge.caller, edge.callee) for edge in found or ()}


def test_the_service_that_leads_is_the_callee() -> None:
    base = noise(0)
    follower = np.r_[0.0, 0.0, base[:-2]]

    assert links(["db", "api"], np.c_[base, follower]) == {("api", "db")}


def test_direction_follows_the_signal_not_the_name() -> None:
    base = noise(0)
    follower = np.r_[0.0, 0.0, base[:-2]]

    assert links(["api", "db"], np.c_[base, follower]) == {("db", "api")}


def test_no_clear_leader_gives_both_directions() -> None:
    base = noise(1)

    assert links(["a", "b"], np.c_[base, base]) == {("a", "b"), ("b", "a")}


def test_unrelated_services_get_no_edge() -> None:
    assert infer_edges(["a", "b"], np.c_[noise(2), noise(3)], SETTINGS) is None


def test_flat_services_get_no_edge() -> None:
    base = noise(4)
    signals = np.c_[base, base, np.zeros(BINS)]

    assert links(["a", "b", "flat"], signals) == {("a", "b"), ("b", "a")}


@pytest.mark.parametrize("bins", [0, 7])
def test_too_little_baseline_gives_no_graph(bins: int) -> None:
    assert infer_edges(["a", "b"], np.zeros((bins, 2)), SETTINGS) is None


def test_one_service_gives_no_graph() -> None:
    assert infer_edges(["a"], noise(5).reshape(-1, 1), SETTINGS) is None


def correlated_metrics(after_cut: float = 0.0) -> pl.DataFrame:
    """Return services sharing one signal before the cut, with an optional later shift."""
    steps = END - START + 1
    shared = np.repeat(noise(6, steps // 5 + 1), 5)[:steps]
    frames = []
    for number, service in enumerate(("a", "b", "c")):
        values = 10 + shared + noise(number + 7, steps) * 0.01
        values[steps // 2 :] += after_cut * (number == 0)
        frames.append(
            pl.DataFrame(
                {
                    "service": [service] * steps,
                    "metric": ["latency"] * steps,
                    "timestamp": np.arange(START, END + 1),
                    "value": values,
                },
                schema=METRICS_SCHEMA,
            )
        )
    return pl.concat(frames)


def inferred_table(metrics: pl.DataFrame) -> pl.DataFrame:
    case = make_input(metrics, None, ("a", "b", "c"))
    return build_case_features(case, config(**INFERRED))


def test_inferred_graph_fills_graph_features_and_is_flagged() -> None:
    table = inferred_table(correlated_metrics())

    values = row(table, "a")
    assert values["graph_topology_available"] == 1
    assert values["graph_topology_inferred"] == 1
    assert values["structural_in_degree"] == 2


def test_the_incident_cannot_change_the_inferred_graph() -> None:
    first = inferred_table(correlated_metrics(0.0))
    second = inferred_table(correlated_metrics(500.0))

    columns = ["structural_in_degree", "structural_out_degree", "structural_betweenness"]
    assert first.select(columns).equals(second.select(columns))


def test_a_recorded_graph_wins_over_inference() -> None:
    table = build_case_features(make_input(topology=edges()), config(**INFERRED))

    assert row(table, "web")["graph_topology_inferred"] == 0
    assert row(table, "web")["structural_out_degree"] == 2


def test_the_default_config_never_infers() -> None:
    values = row(inferred_table_default(), "a")

    assert values["graph_topology_available"] == 0
    assert values["graph_topology_inferred"] == 0
    assert math.isnan(values["structural_in_degree"])


def inferred_table_default() -> pl.DataFrame:
    return build_case_features(make_input(correlated_metrics(), None, ("a", "b", "c")), config())


def test_flat_signals_stay_flagged_as_no_topology() -> None:
    table = inferred_table(correlated_metrics().with_columns(value=pl.lit(1.0)))

    assert row(table, "a")["graph_topology_available"] == 0
    assert row(table, "a")["graph_topology_inferred"] == 0


def test_relabelling_does_not_change_inferred_features() -> None:
    names = {"a": "z", "b": "y", "c": "x"}
    metrics = correlated_metrics().with_columns(pl.col("service").replace(names))

    first = inferred_table(correlated_metrics())
    second = build_case_features(make_input(metrics, None, ("x", "y", "z")), config(**INFERRED))

    columns = feature_columns(["graph", "structural", "signal"])
    for old, new in names.items():
        left, right = row(first, old), row(second, new)
        for column in columns:
            assert left[column] == pytest.approx(right[column], nan_ok=True), column


def test_only_the_configured_families_build_the_graph() -> None:
    metrics = correlated_metrics().with_columns(metric=pl.lit("cpu"))

    table = inferred_table(metrics)

    assert row(table, "a")["graph_topology_available"] == 0
