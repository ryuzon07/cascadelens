"""The Streamlit demo. Start it with ``python -m cascadelens.demo``, which keeps it local."""

import os
from pathlib import Path

import polars as pl
import streamlit as st

from cascadelens.demo.figure import BANNER, render_figure
from cascadelens.demo.perturb import Damage, Simulation, apply_damage, parse_sample
from cascadelens.demo.samples import SAMPLES_DIR
from cascadelens.inference import CaseInput, DiagnoseResult, LoadedModel, diagnose, load_model
from cascadelens.models.artifacts import latest_artifact
from cascadelens.models.diagnose import DISCLAIMER

MODEL_VARIABLE = "CASCADELENS_MODEL_DIR"
SAMPLES_VARIABLE = "CASCADELENS_SAMPLES_DIR"
UPLOAD = "Upload a JSON case"


@st.cache_resource
def _model() -> LoadedModel:
    configured = os.environ.get(MODEL_VARIABLE)
    return load_model(Path(configured) if configured else latest_artifact())


def _choose_case() -> tuple[CaseInput, Simulation | None] | None:
    folder = Path(os.environ.get(SAMPLES_VARIABLE, str(SAMPLES_DIR)))
    samples = {path.stem: path for path in sorted(folder.glob("*.json"))}
    choice = st.sidebar.selectbox("Case", [*samples, UPLOAD])
    if choice != UPLOAD:
        return parse_sample(samples[choice].read_text(encoding="utf-8"))
    uploaded = st.sidebar.file_uploader("JSON case", type="json")
    return parse_sample(uploaded.getvalue().decode("utf-8")) if uploaded else None


def _damage(case: CaseInput, simulation: Simulation | None) -> Damage:
    st.sidebar.subheader("Simulated damage")
    labelled = simulation is not None
    failure = st.sidebar.slider(
        "Failure-correlated loss",
        0.0,
        0.5,
        0.0,
        0.1,
        disabled=not labelled,
        help="Uses the sample's stored label to simulate damage, so it needs a labelled sample.",
    )
    if not labelled:
        st.sidebar.caption("Failure-correlated loss is off: this case has no label.")
    return Damage(
        random_loss=st.sidebar.slider("Random loss", 0.0, 0.5, 0.0, 0.1),
        failure_loss=failure if labelled else 0.0,
        window_shift=st.sidebar.slider("Window shift (s)", -120, 120, 0, 30),
        edge_deletion=st.sidebar.slider(
            "Edge deletion", 0.0, 0.3, 0.0, 0.1, disabled=case.topology is None
        ),
    )


def _ranking_table(result: DiagnoseResult) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "service": [item.service for item in result.ranking],
            "score": [round(item.score, 3) for item in result.ranking],
            "onset_rank": [item.evidence.onset_rank for item in result.ranking],
            "upstream_anom_ratio": [item.evidence.upstream_anom_ratio for item in result.ranking],
            "missing_frac": [item.evidence.missing_frac for item in result.ranking],
            "top_groups": [
                ", ".join(f"{g.group} {g.contribution:+.2f}" for g in item.evidence.top_groups)
                for item in result.ranking
            ],
        }
    )


def _show(case: CaseInput, result: DiagnoseResult, clean: DiagnoseResult) -> None:
    if result.abstain:
        st.error(BANNER)
    top, confidence = st.columns(2)
    top.metric("Top suspect", result.ranking[0].service, f"was {clean.ranking[0].service}", "off")
    confidence.metric(
        "Confidence" if result.calibrated else "Confidence (uncalibrated)",
        f"{result.confidence:.2f}",
        f"{result.confidence - clean.confidence:+.2f} from undamaged",
    )
    st.pyplot(render_figure(case, result))
    st.dataframe(_ranking_table(result), hide_index=True)
    for note in result.notes:
        st.caption(note)


def main() -> None:
    """Render one screen."""
    st.set_page_config(page_title="CascadeLens demo")
    st.title("CascadeLens")
    st.info(DISCLAIMER)
    chosen = _choose_case()
    if chosen is None:
        st.write("Choose a sample case, or upload one.")
        return
    case, simulation = chosen
    damage = _damage(case, simulation)
    model = _model()
    damaged = apply_damage(case, simulation, damage)
    _show(damaged, diagnose(damaged, model), diagnose(case, model))
    st.caption(DISCLAIMER)


if __name__ == "__main__":
    main()
