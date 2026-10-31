"""The ``cascadelens baselines`` and ``cascadelens experiments`` commands."""

from typing import Annotated

import typer

from cascadelens.baselines.base import BaselineError
from cascadelens.baselines.experiments import baselines_clean, leakage_demo
from cascadelens.baselines.runner import run_baseline
from cascadelens.splits.errors import SplitError

app = typer.Typer(help="Simple baselines.", no_args_is_help=True)
experiments_app = typer.Typer(help="Reproducible experiments.", no_args_is_help=True)


@app.callback()
def group() -> None:
    """Simple baselines."""


@app.command()
def run(
    name: Annotated[str, typer.Option("--name", help="Baseline name.")],
    split: Annotated[str, typer.Option("--split", help="Saved split name.")],
    seed: Annotated[int, typer.Option("--seed", help="Run seed.")] = 0,
) -> None:
    """Run one baseline on every fold of one split."""
    try:
        result = run_baseline(name, split, seed)
    except (BaselineError, SplitError) as error:
        typer.echo(f"baselines run failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    metrics = result.metrics
    typer.echo(
        f"{name} on {split}, seed {seed}: Top-1 {metrics['top1']:.3f}, "
        f"Top-3 {metrics['top3']:.3f}, Avg@5 {metrics['avg5']:.3f}"
    )


@experiments_app.command("leakage-demo")
def run_leakage_demo() -> None:
    """Compare the random split with group_run."""
    try:
        leakage_demo()
    except (BaselineError, SplitError) as error:
        typer.echo(f"leakage demo failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Wrote experiments/leakage_demo.")


@experiments_app.command("baselines-clean")
def run_baselines_clean() -> None:
    """Tabulate every baseline on group_run, LOFO and LOSO."""
    try:
        baselines_clean()
    except (BaselineError, SplitError) as error:
        typer.echo(f"baselines clean failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo("Wrote experiments/baselines_clean.")
