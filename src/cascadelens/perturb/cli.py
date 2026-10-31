"""The ``cascadelens perturb`` commands."""

from typing import Annotated

import typer
from pydantic import ValidationError

from cascadelens.baselines.base import BaselineError
from cascadelens.perturb.base import PerturbationError
from cascadelens.perturb.registry import PERTURBATIONS, load_suite, parse_params
from cascadelens.perturb.store import apply_split
from cascadelens.perturb.summary import SUMMARY_PATH, summarize

app = typer.Typer(help="Seeded perturbations of the cases.", no_args_is_help=True)
FAILURES = (BaselineError, PerturbationError, ValidationError)


@app.callback()
def group() -> None:
    """Seeded perturbations of the cases."""


@app.command("list")
def list_suite() -> None:
    """Print every perturbation, its grid and its realism label."""
    for spec in load_suite().perturbations:
        kind = "transform" if spec.id in PERTURBATIONS else "split reference"
        typer.echo(f"{spec.id} {spec.name} [{spec.realism}, {kind}] {spec.params}")


@app.command()
def apply(
    name: Annotated[str, typer.Option("--name", help="Perturbation id or name.")],
    params: Annotated[
        list[str] | None, typer.Option("--params", help="key=value; repeat for several.")
    ] = None,
    split: Annotated[str, typer.Option("--split", help="Saved split name.")] = "group_run",
    seed: Annotated[int, typer.Option("--seed", help="Perturbation seed.")] = 0,
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = 1,
) -> None:
    """Perturb the test cases of a split, reusing whatever is cached."""
    try:
        result = apply_split(name, parse_params(params or []), split, seed, workers=workers)
    except FAILURES as error:
        typer.echo(f"perturb apply failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{len(result.rows)} case(s) in {result.directory}, {result.bytes / 1e6:.0f} MB.")


@app.command("summarize")
def summarize_grid(
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = 1,
    max_cases: Annotated[
        int | None, typer.Option("--max-cases", help="Evenly spaced cases; all by default.")
    ] = None,
) -> None:
    """Report realised rates against nominal over the grid in the suite config."""
    try:
        lines = summarize(workers=workers, max_cases=max_cases)
    except FAILURES as error:
        typer.echo(f"perturb summarize failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {SUMMARY_PATH} for {len(lines)} grid point(s).")
