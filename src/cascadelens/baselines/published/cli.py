"""The ``cascadelens published`` commands."""

from typing import Annotated

import typer

from cascadelens.baselines.base import BaselineError
from cascadelens.baselines.published.evaluate import evaluate_published
from cascadelens.baselines.published.reproduce import reproduce as run_reproduction
from cascadelens.splits.errors import SplitError

app = typer.Typer(help="Published baselines. Needs the published extra.", no_args_is_help=True)


@app.callback()
def group() -> None:
    """Published baselines."""


@app.command()
def reproduce(name: Annotated[str, typer.Option("--name", help="Method name.")]) -> None:
    """Run the paper's protocol and compare with the published numbers."""
    try:
        run_reproduction(name)
    except BaselineError as error:
        typer.echo(f"published reproduce failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote docs/reproduction/{name}.md.")


@app.command()
def evaluate() -> None:
    """Run the configured methods on group_run, LOFO and LOSO."""
    try:
        evaluate_published()
    except (BaselineError, SplitError) as error:
        typer.echo(f"published evaluate failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Wrote experiments/reproduction/evaluation.")
