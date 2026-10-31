from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from perturb_factory import CHAIN, EXAMPLES, SERVICES, START, make_case

from cascadelens.config import load_config
from cascadelens.data.schema import METRICS_SCHEMA, Case
from cascadelens.features.config import FeaturesConfig
from cascadelens.features.context import FeatureInput
from cascadelens.features.inference import InferenceSettings
from cascadelens.features.registry import build_case_features
from cascadelens.features.simple import metric_family
from cascadelens.perturb.base import (
    PerturbationError,
    Stage,
    crop_to_window,
    metrics_bytes,
    observed,
    output_hash,
    rounded,
)
from cascadelens.perturb.loss import FailureLossParams, solve_base_rate
from cascadelens.perturb.pipeline import Pipeline
from cascadelens.perturb.registry import (
    PERTURBATIONS,
    REFERENCES,
    build,
    grid,
    load_suite,
    parse_params,
)
from cascadelens.perturb.topology import hop_distances

CASE = make_case()
REPO = Path(__file__).parents[2]
KEPT = ("case_id", "suite", "system", "fault_type", "run_id", "root_cause_service", "inject_time")


def real(case: Case) -> pl.DataFrame:
    return case.metrics.filter(observed(case.metrics))


@pytest.mark.parametrize("identifier", sorted(EXAMPLES))
def test_same_inputs_give_identical_bytes(identifier: str) -> None:
    step = build(identifier, EXAMPLES[identifier])
    first, second = step.apply(CASE, 0), step.apply(CASE, 0)
    assert metrics_bytes(first.metrics) == metrics_bytes(second.metrics)
    assert output_hash(first) == output_hash(second)


@pytest.mark.parametrize("identifier", ["P1", "P2", "P5", "P6", "P7", "P12", "P13"])
def test_a_different_seed_changes_the_output(identifier: str) -> None:
    step = build(identifier, EXAMPLES[identifier])
    assert output_hash(step.apply(CASE, 0)) != output_hash(step.apply(CASE, 1))


@pytest.mark.parametrize("identifier", sorted(EXAMPLES))
def test_input_is_not_mutated(identifier: str) -> None:
    case = make_case(1)
    before = metrics_bytes(case.metrics), case.record()
    build(identifier, EXAMPLES[identifier]).apply(case, 0)
    assert (metrics_bytes(case.metrics), case.record()) == before


PARAMETERS: dict[str, Any] = {
    "P1": st.fixed_dictionaries(
        {"rate": st.floats(0.01, 1.0), "level": st.sampled_from(["series", "sample"])}
    ),
    "P2": st.fixed_dictionaries(
        {
            "rate": st.floats(0.01, 1.0),
            "near_boost": st.floats(1.0, 6.0),
            "after_boost": st.floats(1.0, 6.0),
        }
    ),
    "P3": st.fixed_dictionaries(
        {"family": st.sampled_from(["cpu", "latency", "memory", "disk", "error"])}
    ),
    "P4": st.fixed_dictionaries({"offset_seconds": st.integers(-120, 120)}),
    "P5": st.fixed_dictionaries({"max_jitter_seconds": st.integers(1, 60)}),
    "P6": st.fixed_dictionaries({"rate": st.floats(0.01, 1.0)}),
    "P7": st.fixed_dictionaries({"rate": st.floats(0.01, 1.0)}),
    "P8": st.fixed_dictionaries({"method": st.sampled_from(["correlation", "lagged"])}),
    "P11": st.fixed_dictionaries({"factor": st.integers(2, 10)}),
    "P12": st.fixed_dictionaries(
        {"max_offset_seconds": st.integers(0, 30), "late_fraction": st.floats(0.0, 0.9)}
    ),
    "P13": st.fixed_dictionaries(
        {
            "mode": st.sampled_from(["rename", "rescale"]),
            "service_fraction": st.floats(0.05, 1.0),
        }
    ),
}


@pytest.mark.parametrize("identifier", sorted(PARAMETERS))
def test_labels_and_identifiers_never_change(identifier: str) -> None:
    @settings(max_examples=8, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    @given(params=PARAMETERS[identifier], seed=st.integers(0, 2**31 - 1))
    def check(params: dict[str, Any], seed: int) -> None:
        output = build(identifier, params).apply(CASE, seed)
        for name in KEPT:
            assert getattr(output, name) == getattr(CASE, name)
        assert output.metrics.schema == METRICS_SCHEMA
        assert type(output) is Case
        assert set(output.model_fields_set) <= set(Case.model_fields)
        if identifier not in ("P4", "P5"):
            assert output.window_start == CASE.window_start
            assert output.window_end == CASE.window_end

    check()


@pytest.mark.parametrize("level", ["series", "sample"])
@pytest.mark.parametrize("rate", [0.1, 0.3, 0.5])
def test_p1_hits_the_nominal_rate(level: str, rate: float) -> None:
    rows = [build("P1", {"rate": rate, "level": level}).run(make_case(i), 0).row for i in range(30)]
    if level == "series":
        assert {r.series_dropped for r in rows} == {rounded(rate * 32)}
    realised = sum(r.samples_dropped for r in rows) / sum(r.samples_before for r in rows)
    assert realised == pytest.approx(rate, abs=0.03)


@pytest.mark.parametrize("rate", [0.1, 0.2, 0.3, 0.5])
def test_p2_hits_the_nominal_rate_and_favours_the_fault(rate: float) -> None:
    rows = [build("P2", {"rate": rate}).run(make_case(i), 0).row for i in range(40)]
    realised = sum(r.samples_dropped for r in rows) / sum(r.samples_before for r in rows)
    assert realised == pytest.approx(rate, abs=0.03)
    near = np.array([r.details["near_rate"] for r in rows])
    far = np.array([r.details["far_rate"] for r in rows])
    assert (near > far).mean() > 0.9
    before = np.array([r.details["before_rate"] for r in rows])
    after = np.array([r.details["after_rate"] for r in rows])
    assert (after > before).mean() > 0.9


def test_p2_expected_rate_survives_saturation() -> None:
    sizes, weights = [10.0, 90.0], [9.0, 1.0]
    base = solve_base_rate(sizes, weights, 0.5)
    assert 0.1 * min(1, base * 9) + 0.9 * min(1, base) == pytest.approx(0.5)


def test_p2_defines_near_by_hops_from_the_root() -> None:
    assert hop_distances(CASE, "svc-0")["svc-1"] == 1
    assert hop_distances(CASE, "svc-0")["svc-3"] == 3
    assert FailureLossParams(rate=0.1).near_hops == 1


def test_p2_adds_no_field_that_reveals_ground_truth() -> None:
    output = build("P2", {"rate": 0.3}).apply(CASE, 0)
    assert output.metrics.columns == CASE.metrics.columns
    assert output.metrics.height == CASE.metrics.height
    same = CASE.record().model_dump()
    assert output.record().model_dump() == same
    assert output.metrics.select("service", "metric", "timestamp").equals(
        CASE.metrics.select("service", "metric", "timestamp")
    )


def test_p3_drops_one_family_and_stubs_other_modalities() -> None:
    output = build("P3", {"family": "latency"}).apply(CASE, 0)
    assert {metric_family(m) for m in real(output)["metric"].unique()} == {"cpu", "memory"}
    for modality in ("logs", "traces"):
        with pytest.raises(NotImplementedError, match="stretch"):
            build("P3", {"modality": modality}).apply(CASE, 0)


def test_p4_moves_bounds_keeps_outside_data_and_pads() -> None:
    later = build("P4", {"offset_seconds": 30}).run(CASE, 0)
    assert later.case.window_start == CASE.window_start + 30
    assert later.case.window_end == CASE.window_end + 30
    assert later.row.details["original_window_start"] == CASE.window_start
    assert later.row.details["original_window_end"] == CASE.window_end
    assert later.row.samples_dropped == 0
    assert real(later.case).height == real(CASE).height
    outside = later.case.metrics.filter(pl.col("timestamp") < later.case.window_start)
    assert outside["value"].null_count() == 0
    assert outside.height == 30 * 32
    padded = later.case.metrics.filter(pl.col("timestamp") > CASE.window_end)
    assert padded["value"].null_count() == padded.height == 30 * 32
    cropped = crop_to_window(later.case)
    assert int(cropped.metrics["timestamp"].min()) == later.case.window_start  # type: ignore[arg-type]


def test_p5_moves_each_bound_within_the_range() -> None:
    moved = {
        (o.window_start - CASE.window_start, o.window_end - CASE.window_end)
        for o in (build("P5", {"max_jitter_seconds": 20}).apply(CASE, s) for s in range(12))
    }
    assert len({a for a, _ in moved}) > 1
    assert any(a != b for a, b in moved)
    assert all(abs(a) <= 20 and abs(b) <= 20 for a, b in moved)


@pytest.mark.parametrize("rate", [0.1, 0.2, 0.3])
def test_p6_edge_count_is_exact_to_rounding(rate: float) -> None:
    applied = build("P6", {"rate": rate}).run(CASE, 0)
    assert applied.row.edges_removed == rounded(rate * len(CHAIN))
    assert len(applied.case.topology or ()) == len(CHAIN) - applied.row.edges_removed


@pytest.mark.parametrize("rate", [0.1, 0.2, 0.5])
def test_p7_edge_count_is_exact_and_between_existing_services(rate: float) -> None:
    applied = build("P7", {"rate": rate}).run(CASE, 0)
    edges = applied.case.topology or ()
    assert len(edges) == len(CHAIN) + rounded(rate * len(CHAIN))
    assert len({(e.caller, e.callee) for e in edges}) == len(edges)
    assert all(e.caller in SERVICES and e.callee in SERVICES for e in edges)
    assert all(e.caller != e.callee for e in edges)


def _correlated(root: str) -> Case:
    rng = np.random.default_rng(0)
    shared = rng.normal(size=800).cumsum()
    rows = [
        (service, "latency-50", START + t, float(shared[t] + 0.05 * rng.normal()))
        for service in ("a", "b", "c")
        for t in range(800)
    ]
    base = make_case(root="a", topology=None)
    fields = {**base.record().model_dump(), "root_cause_service": root, "window_end": START + 799}
    return Case(**fields, metrics=pl.DataFrame(rows, schema=METRICS_SCHEMA, orient="row"))


@pytest.mark.parametrize("method", ["correlation", "lagged"])
def test_p8_uses_no_ground_truth(method: str) -> None:
    step = build("P8", {"method": method})
    first, second = step.apply(_correlated("a"), 0), step.apply(_correlated("c"), 0)
    assert first.topology
    assert first.topology == second.topology
    moved = _correlated("a")
    shifted = Case(**{**moved.record().model_dump(), "inject_time": START}, metrics=moved.metrics)
    assert step.apply(shifted, 0).topology == first.topology


def test_p8_correlation_gives_two_way_edges() -> None:
    edges = build("P8", {"method": "correlation"}).apply(_correlated("a"), 0).topology or ()
    pairs = {(e.caller, e.callee) for e in edges}
    assert pairs == {(b, a) for a, b in pairs}


def test_edge_steps_fall_back_to_the_inferred_graph() -> None:
    case = _correlated("a")
    applied = build("P6", {"rate": 0.5}).run(case, 0)
    assert applied.row.edges_before > 0
    assert applied.row.edges_removed > 0


def test_defaults_match_the_part_six_inference() -> None:
    features = load_config(REPO / "configs" / "features.yaml", FeaturesConfig)
    assert features.topology.inference == InferenceSettings(
        families=features.topology.inference.families
    )


@pytest.mark.parametrize("factor", [2, 5])
def test_p11_keeps_one_instant_in_factor(factor: int) -> None:
    output = build("P11", {"factor": factor}).apply(CASE, 0)
    assert output.metrics.height == CASE.metrics.height
    kept = real(output)["timestamp"].unique().sort()
    assert kept.diff().drop_nulls().unique().to_list() == [factor]


def test_p12_offsets_services_and_drops_late_samples() -> None:
    applied = build("P12", {"max_offset_seconds": 5, "late_fraction": 0.2}).run(CASE, 0)
    offsets = applied.row.details["offsets"]
    assert all(abs(v) <= 5 for v in offsets.values())
    for service, offset in offsets.items():
        first = applied.case.metrics.filter(pl.col("service") == service)["timestamp"].min()
        assert first == CASE.window_start + offset
    assert applied.row.samples_dropped / applied.row.samples_before == pytest.approx(0.2, abs=0.02)


def test_p13_rename_mapping_is_invertible() -> None:
    applied = build("P13", {"mode": "rename", "service_fraction": 0.5}).run(CASE, 0)
    mapping = applied.row.details["mapping"]
    assert mapping
    assert len({(m["service"], m["from"]) for m in mapping}) == len(mapping)
    assert len({(m["service"], m["to"]) for m in mapping}) == len(mapping)
    assert all(metric_family(m["from"]) == metric_family(m["to"]) for m in mapping)
    assert all(m["from"] != m["to"] for m in mapping)
    table = pl.DataFrame(mapping).rename({"to": "metric", "from": "original"})
    restored = (
        applied.case.metrics.join(table, on=["service", "metric"], how="left")
        .with_columns(pl.coalesce("original", "metric").alias("metric"))
        .select(CASE.metrics.columns)
        .sort("service", "metric", "timestamp")
    )
    assert restored.equals(CASE.metrics.sort("service", "metric", "timestamp"))


def test_p13_rescale_mapping_is_invertible() -> None:
    applied = build("P13", {"mode": "rescale", "service_fraction": 0.5, "factor": 1000.0}).run(
        CASE, 0
    )
    mapping = {m["service"]: m["factor"] for m in applied.row.details["mapping"]}
    assert len(mapping) == 4
    restored = applied.case.metrics.with_columns(
        pl.col("value")
        / pl.col("service").replace_strict(mapping, default=1.0, return_dtype=pl.Float64)
    )
    assert np.allclose(restored["value"].to_numpy(), CASE.metrics["value"].to_numpy())


def test_composition_order_is_fixed_and_equals_manual_steps() -> None:
    loss = build("P1", {"rate": 0.3})
    shift = build("P4", {"offset_seconds": 30})
    edges = build("P6", {"rate": 0.3})
    pipeline = Pipeline([edges, shift, loss])
    assert [s.stage for s in pipeline.steps] == [Stage.DATA, Stage.WINDOW, Stage.TOPOLOGY]
    both = Pipeline([shift, loss]).run(CASE, 0)
    manual = shift.apply(loss.apply(CASE, 0), 0)
    assert output_hash(both.case) == output_hash(manual)
    assert [r.perturbation for r in both.rows] == ["P1", "P4"]


@pytest.mark.parametrize("identifier", ["P1", "P2", "P3", "P11", "P12"])
def test_dropped_data_is_marked_missing(identifier: str) -> None:
    applied = build(identifier, EXAMPLES[identifier]).run(CASE, 0)
    output = applied.case.metrics
    assert output.height == CASE.metrics.height
    assert output["value"].null_count() == applied.row.samples_dropped > 0
    assert output.filter(pl.col("value") == 0.0).is_empty()
    assert not output["value"].is_nan().any()


def test_dropped_data_reaches_the_mask_features() -> None:
    settings_ = FeaturesConfig(groups=["mask"])
    tables = [
        build_case_features(FeatureInput.from_record(c.record(), c.metrics), settings_)
        for c in (CASE, build("P3", {"family": "latency"}).apply(CASE, 0))
    ]
    assert tables[0]["mask_latency_available"].min() == 1.0
    assert tables[1]["mask_latency_available"].max() == 0.0
    assert tables[1]["mask_missing_fraction"].min() == pytest.approx(0.5)


def test_registry_matches_the_shipped_config() -> None:
    suite = load_suite(REPO)
    ids = {spec.id for spec in suite.perturbations}
    assert ids == set(PERTURBATIONS) | set(REFERENCES)
    for spec in suite.perturbations:
        if spec.id in PERTURBATIONS:
            assert PERTURBATIONS[spec.id].name == spec.name
            assert PERTURBATIONS[spec.id].realism == spec.realism == "simulated"
        else:
            assert REFERENCES[spec.id].realism == spec.realism == "held-out"
            assert spec.params["split"] == [REFERENCES[spec.id].split]
    for spec, params in grid(suite):
        build(spec.id, params)


@pytest.mark.parametrize("identifier", ["P9", "P10", "unseen_system"])
def test_split_references_have_no_apply(identifier: str) -> None:
    with pytest.raises(PerturbationError, match="split"):
        build(identifier, {})


def test_bad_input_is_refused() -> None:
    with pytest.raises(PerturbationError, match="Unknown"):
        build("P99", {})
    with pytest.raises(PerturbationError, match="key=value"):
        parse_params(["rate"])
    assert build("P2", parse_params(["rate=0.3"])).parameters()["rate"] == 0.3
