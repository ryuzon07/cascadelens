"""The ``experiments robustness`` and ``probes`` commands."""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal

import typer

from cascadelens.baselines.base import BaselineError
from cascadelens.eval import probes
from cascadelens.eval.robustness import derive_stage, load_context, report_stage, train_stage
from cascadelens.features.context import FeatureError
from cascadelens.perturb.base import PerturbationError
from cascadelens.splits.errors import SplitError

FAILURES = (BaselineError, FeatureError, PerturbationError, SplitError)
ExploratoryOption = Annotated[
    bool, typer.Option("--exploratory", help="Allow a config that is not frozen, and tag results.")
]
app = typer.Typer(help="Leakage probes (PRD 6.5).", no_args_is_help=True)


@app.callback()
def group() -> None:
    """Leakage probes (PRD 6.5)."""


def run_robustness(
    stage: Annotated[
        Literal["all", "derive", "train", "report"],
        typer.Option("--stage", help="Run one stage. Finished work is skipped."),
    ] = "all",
    variant: Annotated[
        list[str] | None, typer.Option("--variant", help="Train only these; repeat for several.")
    ] = None,
    exploratory: ExploratoryOption = False,
) -> None:
    """Run the perturbation sweep in resumable stages."""
    try:
        context = load_context(exploratory)
        if stage in ("all", "derive"):
            derive_stage(context)
        if stage in ("all", "train"):
            train_stage(context, variant)
        if stage in ("all", "report"):
            report_stage(context)
    except FAILURES as error:
        typer.echo(f"robustness failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Finished stage {stage} ({context.status}).")


def _probe(name: str, run: Callable[[bool, Path], dict[str, Any]], exploratory: bool) -> None:
    try:
        result = run(exploratory, Path())
    except FAILURES as error:
        typer.echo(f"{name} probe failed to run: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{name}: {'passed' if result['passed'] else 'FAILED or flagged'}.")


@app.command()
def identity(exploratory: ExploratoryOption = False) -> None:
    """Shuffle graph features across services and retrain."""
    _probe("identity", probes.run_identity, exploratory)


@app.command("graph-only")
def graph_only(exploratory: ExploratoryOption = False) -> None:
    """Train on graph and structural features alone."""
    _probe("graph-only", probes.run_graph_only, exploratory)


@app.command()
def window(exploratory: ExploratoryOption = False) -> None:
    """Score across window offsets and random bounds."""
    _probe("window", probes.run_window, exploratory)


@app.command("mask-only")
def mask_only(exploratory: ExploratoryOption = False) -> None:
    """Train on mask features alone on clean data."""
    _probe("mask-only", probes.run_mask_only, exploratory)
