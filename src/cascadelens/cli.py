"""Command-line entry point."""

import typer

from cascadelens import __version__
from cascadelens.baselines.cli import app as baselines_app
from cascadelens.baselines.cli import experiments_app
from cascadelens.baselines.published.cli import app as published_app
from cascadelens.data.cli import app as data_app
from cascadelens.eval.cli import app as eval_app
from cascadelens.eval.robustness_cli import app as probes_app
from cascadelens.eval.robustness_cli import run_robustness
from cascadelens.features.cli import app as features_app
from cascadelens.models.cli import (
    calibrate,
    diagnose,
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
app.command()(diagnose)
experiments_app.command("calibration")(run_calibration)
experiments_app.command("ranker-clean")(run_ranker_clean)
experiments_app.command("robustness")(run_robustness)

NOT_IMPLEMENTED = "not implemented"


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
