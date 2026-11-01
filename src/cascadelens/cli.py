"""Command-line entry point."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from cascadelens import __version__
from cascadelens.baselines.cli import app as baselines_app
from cascadelens.baselines.cli import experiments_app
from cascadelens.baselines.published.cli import app as published_app
from cascadelens.data.cli import app as data_app
from cascadelens.demo.figure import BANNER, save_figure
from cascadelens.eval.cli import app as eval_app
from cascadelens.eval.robustness_cli import app as probes_app
from cascadelens.eval.robustness_cli import run_robustness
from cascadelens.features.cli import app as features_app
from cascadelens.inference import DiagnoseResult, read_case
from cascadelens.inference import diagnose as run_diagnosis
from cascadelens.models.artifacts import latest_artifact
from cascadelens.models.cli import (
    FAILURES,
    calibrate,
    evaluate,
    freeze,
    run_calibration,
    run_ranker_clean,
    train,
)
from cascadelens.perturb.cli import app as perturb_app
from cascadelens.splits.cli import app as splits_app

app = typer.Typer(help="CascadeLens experiment pipeline.", no_args_is_help=True)
app.add_typer(baselines_app, name="baselines")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(features_app, name="features")
app.add_typer(experiments_app, name="experiments")
app.add_typer(published_app, name="published")
app.add_typer(perturb_app, name="perturb")
app.add_typer(probes_app, name="probes")
app.add_typer(splits_app, name="splits")
app.command()(train)
app.command()(evaluate)
app.command()(freeze)
app.command()(calibrate)
experiments_app.command("calibration")(run_calibration)
experiments_app.command("ranker-clean")(run_ranker_clean)
experiments_app.command("robustness")(run_robustness)

NOT_IMPLEMENTED = "not implemented"
ModelOption = Annotated[
    Path | None, typer.Option("--model", help="Saved model directory; the latest by default.")
]
CaseOption = Annotated[
    Path, typer.Option("--case", help="A case as JSON, or a metrics table as Parquet.")
]


class OutputFormat(StrEnum):
    """How diagnose prints."""

    table = "table"
    json = "json"


def _print_table(result: DiagnoseResult) -> None:
    typer.echo(f"Model: {result.model_version}")
    if result.abstain:
        typer.echo(BANNER)
    label = "calibrated" if result.calibrated else "uncalibrated"
    typer.echo(f"Confidence ({label}): {result.confidence:.4f}")
    typer.echo(f"Abstain: {str(result.abstain).lower()}")
    for rank, item in enumerate(result.ranking, start=1):
        evidence = item.evidence
        groups = ", ".join(f"{g.group} {g.contribution:+.2f}" for g in evidence.top_groups)
        typer.echo(
            f"{rank}. {item.service}  {item.score:.4f}  onset_rank={evidence.onset_rank} "
            f"upstream_anom_ratio={evidence.upstream_anom_ratio} "
            f"missing_frac={evidence.missing_frac}  [{groups}]"
        )
    for note in result.notes:
        typer.echo(f"Note: {note}")
    typer.echo(f"Note: {result.disclaimer}")


def _failed(command: str, error: Exception) -> typer.Exit:
    typer.echo(f"{command} failed: {error}", err=True)
    return typer.Exit(code=1)


@app.command()
def diagnose(
    case: CaseOption,
    model: ModelOption = None,
    output: Annotated[
        OutputFormat, typer.Option("--format", help="table or json.")
    ] = OutputFormat.table,
) -> None:
    """Rank likely origin services for a single case."""
    try:
        directory = model if model is not None else latest_artifact()
        result = run_diagnosis(read_case(case), directory)
    except FAILURES as error:
        raise _failed("diagnose", error) from error
    if output is OutputFormat.json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        _print_table(result)


@app.command("demo-figure")
def demo_figure(
    case: CaseOption,
    out: Annotated[Path, typer.Option("--out", help="PNG to write.")],
    model: ModelOption = None,
) -> None:
    """Draw the dependency graph with suspects coloured by score."""
    try:
        directory = model if model is not None else latest_artifact()
        loaded = read_case(case)
        save_figure(loaded, run_diagnosis(loaded, directory), out)
    except FAILURES as error:
        raise _failed("demo-figure", error) from error
    typer.echo(f"Wrote {out}.")


@app.command()
def serve(
    port: Annotated[int, typer.Option("--port", help="Port to listen on.")] = 8000,
    host: Annotated[
        str, typer.Option("--host", help="Address. Local by default; never expose it publicly.")
    ] = "127.0.0.1",
    model: ModelOption = None,
    max_body_bytes: Annotated[
        int, typer.Option("--max-body-bytes", help="Largest accepted request body.")
    ] = 25_000_000,
) -> None:
    """Start the local read-only API. Needs the api extra."""
    try:
        import uvicorn

        from cascadelens.api.main import ApiSettings, create_app
    except ImportError as error:
        typer.echo("serve needs the api extra: uv sync --extra api", err=True)
        raise typer.Exit(code=1) from error
    try:
        directory = model if model is not None else latest_artifact()
        settings = ApiSettings(model_dir=directory, max_body_bytes=max_body_bytes)
        application = create_app(settings)
    except FAILURES as error:
        raise _failed("serve", error) from error
    uvicorn.run(application, host=host, port=port)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def ingest() -> None:
    """Ingest benchmark data into the unified Case schema."""
    typer.echo(NOT_IMPLEMENTED)


@app.command()
def report() -> None:
    """Build result tables and figures."""
    typer.echo(NOT_IMPLEMENTED)
