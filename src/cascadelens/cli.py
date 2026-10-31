"""Command-line entry point."""

import typer

from cascadelens import __version__
from cascadelens.baselines.cli import app as baselines_app
from cascadelens.baselines.cli import experiments_app
from cascadelens.data.cli import app as data_app
from cascadelens.eval.cli import app as eval_app
from cascadelens.splits.cli import app as splits_app

app = typer.Typer(help="CascadeLens experiment pipeline.", no_args_is_help=True)
app.add_typer(baselines_app, name="baselines")
app.add_typer(data_app, name="data")
app.add_typer(eval_app, name="eval")
app.add_typer(experiments_app, name="experiments")
app.add_typer(splits_app, name="splits")

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
def perturb() -> None:
    """Generate seeded perturbed Case variants."""
    typer.echo(NOT_IMPLEMENTED)


@app.command()
def train() -> None:
    """Train the CascadeLens ranker."""
    typer.echo(NOT_IMPLEMENTED)


@app.command()
def evaluate() -> None:
    """Compute metrics and significance tests."""
    typer.echo(NOT_IMPLEMENTED)


@app.command()
def report() -> None:
    """Build result tables and figures."""
    typer.echo(NOT_IMPLEMENTED)


@app.command()
def diagnose() -> None:
    """Rank likely origin services for a single case."""
    typer.echo(NOT_IMPLEMENTED)
