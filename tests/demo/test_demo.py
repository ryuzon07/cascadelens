import json
from pathlib import Path

import polars as pl
import pytest
from typer.testing import CliRunner

from cascadelens.cli import app as cli_app
from cascadelens.demo.figure import case_graph, render_figure
from cascadelens.demo.latency import summary
from cascadelens.demo.perturb import Damage, apply_damage, parse_sample
from cascadelens.demo.samples import all_samples
from cascadelens.inference import diagnose, load_model, read_case
from cascadelens.models.calibration import ARTIFACT, SavedCalibration, TemperatureScaler
from cascadelens.models.data import RankerError
from cascadelens.models.diagnose import DISCLAIMER

REPO = Path(__file__).parents[1].parent
SAMPLES = REPO / "demo_data"
APP = REPO / "src/cascadelens/demo/app.py"


def text(name: str) -> str:
    return (SAMPLES / f"{name}.json").read_text(encoding="utf-8")


def test_committed_samples_match_their_generator() -> None:
    for case in all_samples():
        saved = json.loads(text(case["case_id"]))
        assert saved == json.loads(json.dumps(case))


def test_read_case_drops_the_simulation_block() -> None:
    case = read_case(SAMPLES / "payment-cpu.json")
    assert not hasattr(case, "simulation")
    _, simulation = parse_sample(text("payment-cpu"))
    assert simulation is not None
    assert parse_sample(text("checkout-unlabelled"))[1] is None


def test_missing_topology_gets_a_note(trained_model: Path) -> None:
    result = diagnose(read_case(SAMPLES / "cart-memory-no-topology.json"), trained_model)
    assert result.ranking
    assert any("topology" in note and "provided" not in note for note in result.notes)


def test_heavy_missing_telemetry_gets_a_note(trained_model: Path) -> None:
    case = read_case(SAMPLES / "payment-silent.json")
    result = diagnose(case, trained_model)
    assert "telemetry missing for 2 of 5 services" in result.notes
    silent = case.metrics.with_columns(value=pl.lit(None, dtype=pl.Float64))
    empty = diagnose(
        type(case)(case.case_id, silent, case.window_start, case.window_end), trained_model
    )
    assert empty.ranking
    assert 0 <= empty.confidence <= 1
    assert "telemetry missing for 5 of 5 services" in empty.notes


def test_low_confidence_sets_the_abstain_flag(trained_model: Path) -> None:
    model = load_model(trained_model)
    never = SavedCalibration(TemperatureScaler(1.0), 1.1, 0.5, [], {})
    strict = type(model)(model.directory, model.fold, model.features, never, model.version)
    assert diagnose(read_case(SAMPLES / "payment-silent.json"), strict).abstain


def test_uncalibrated_model_says_so(trained_model: Path) -> None:
    model = load_model(trained_model)
    bare = type(model)(model.directory, model.fold, model.features, None, model.version)
    result = diagnose(read_case(SAMPLES / "payment-cpu.json"), bare)
    assert not result.calibrated
    assert not result.abstain
    assert any("uncalibrated" in note for note in result.notes)
    assert (trained_model / ARTIFACT).is_file()


@pytest.mark.parametrize("name", ["payment-cpu", "cart-memory-no-topology"])
def test_demo_figure_writes_a_png(trained_model: Path, tmp_path: Path, name: str) -> None:
    out = tmp_path / "fig.png"
    result = CliRunner().invoke(
        cli_app,
        [
            "demo-figure",
            "--case",
            str(SAMPLES / f"{name}.json"),
            "--out",
            str(out),
            "--model",
            str(trained_model),
        ],
    )
    assert result.exit_code == 0
    assert out.read_bytes().startswith(b"\x89PNG")


def _figure_text(name: str, trained_model: Path, abstain: bool) -> list[str]:
    case = read_case(SAMPLES / f"{name}.json")
    result = diagnose(case, trained_model).model_copy(update={"abstain": abstain})
    figure = render_figure(case, result)
    found = [item.get_text() for axes in figure.axes for item in axes.texts]
    assert figure.get_supxlabel() == DISCLAIMER
    return found


def test_figure_shows_the_banner_and_the_missing_topology(trained_model: Path) -> None:
    found = _figure_text("cart-memory-no-topology", trained_model, True)
    assert "! ABSTAIN: low confidence" in found
    assert any("No topology" in item for item in found)
    assert "! ABSTAIN: low confidence" not in _figure_text("payment-cpu", trained_model, False)


def test_no_topology_graph_has_nodes_and_no_edges(trained_model: Path) -> None:
    case = read_case(SAMPLES / "cart-memory-no-topology.json")
    graph = case_graph(case, diagnose(case, trained_model))
    assert graph.number_of_nodes() == 5
    assert graph.number_of_edges() == 0


def test_sliders_change_the_case_and_leave_the_input(trained_model: Path) -> None:
    case, simulation = parse_sample(text("payment-cpu"))
    before = case.metrics.clone()
    random = apply_damage(case, simulation, Damage(random_loss=0.3))
    failure = apply_damage(case, simulation, Damage(failure_loss=0.3))
    shifted = apply_damage(case, simulation, Damage(window_shift=60))
    trimmed = apply_damage(case, simulation, Damage(edge_deletion=0.3))
    assert case.metrics.equals(before)
    assert random.metrics["value"].null_count() > 0
    assert failure.metrics["value"].null_count() > 0
    assert shifted.window_start == case.window_start + 60
    assert trimmed.topology is not None
    assert case.topology is not None
    assert len(trimmed.topology) < len(case.topology)
    for changed in (random, failure, shifted, trimmed):
        assert diagnose(changed, trained_model).ranking
    assert apply_damage(case, simulation, Damage()) is case


def test_failure_loss_needs_a_label() -> None:
    case, simulation = parse_sample(text("checkout-unlabelled"))
    assert simulation is None
    with pytest.raises(RankerError, match="label"):
        apply_damage(case, simulation, Damage(failure_loss=0.3))


def test_latency_catches_an_order_of_magnitude_regression(trained_model: Path) -> None:
    model = load_model(trained_model)
    case = read_case(SAMPLES / "payment-cpu.json")
    times = [diagnose(case, model).timing.total_seconds for _ in range(5)]
    assert summary(times)[0] < 5.0


def test_streamlit_app_imports_and_renders(
    trained_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CASCADELENS_MODEL_DIR", str(trained_model))
    monkeypatch.setenv("CASCADELENS_SAMPLES_DIR", str(SAMPLES))
    screen = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not screen.exception
    assert screen.info[0].value == DISCLAIMER
    assert any(item.value == DISCLAIMER for item in screen.caption)


def test_streamlit_sliders_rerun_diagnose_and_label_slider_locks(
    trained_model: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CASCADELENS_MODEL_DIR", str(trained_model))
    monkeypatch.setenv("CASCADELENS_SAMPLES_DIR", str(SAMPLES))
    screen = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not screen.sidebar.slider[0].disabled
    screen.sidebar.slider[0].set_value(0.3).run()
    screen.sidebar.slider[1].set_value(0.3).run()
    screen.sidebar.slider[2].set_value(60).run()
    assert not screen.exception
    screen.sidebar.selectbox[0].set_value("checkout-unlabelled").run()
    assert not screen.exception
    assert screen.sidebar.slider[0].disabled
    assert any(item.value == DISCLAIMER for item in screen.info)
