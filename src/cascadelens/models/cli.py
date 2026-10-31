"""The ranker commands: train, evaluate, freeze, diagnose and experiments ranker-clean."""

from pathlib import Path
from typing import Annotated

import typer

from cascadelens.baselines.base import BaselineError
from cascadelens.features.context import FeatureError
from cascadelens.models.artifacts import latest_artifact
from cascadelens.models.diagnose import diagnose_case
from cascadelens.models.experiment import ranker_clean
from cascadelens.models.pipeline import evaluate as evaluate_method
from cascadelens.models.pipeline import freeze_config, train_folds
from cascadelens.splits.errors import SplitError

FAILURES = (BaselineError, FeatureError, SplitError)
ConfigOption = Annotated[Path, typer.Option("--config", help="Ranker config.")]
SplitOption = Annotated[str, typer.Option("--split", help="Saved split name.")]
SeedOption = Annotated[int, typer.Option("--seed", help="Model seed.")]
MethodOption = Annotated[
    str, typer.Option("--method", help="two_stage_ranker or single_stage_ranker.")
]
ExploratoryOption = Annotated[
    bool, typer.Option("--exploratory", help="Allow a config that is not frozen, and tag results.")
]


def _fail(command: str, error: Exception) -> typer.Exit:
    typer.echo(f"{command} failed: {error}", err=True)
    return typer.Exit(code=1)


def train(
    config: ConfigOption = Path("configs/ranker.yaml"),
    split: SplitOption = "group_run",
    seed: SeedOption = 0,
    method: MethodOption = "two_stage_ranker",
) -> None:
    """Train the ranker on each fold and save it. Reads no test case."""
    try:
        saved = train_folds(config, split, seed, method)
    except FAILURES as error:
        raise _fail("train", error) from error
    for index, (directory, fold) in enumerate(saved):
        typer.echo(
            f"Fold {index}: validation Top-1 {fold.val_top1:.3f}, K {fold.choice.k}, {directory}"
        )


def evaluate(
    config: ConfigOption = Path("configs/ranker.yaml"),
    split: SplitOption = "group_run",
    seed: SeedOption = 0,
    method: MethodOption = "two_stage_ranker",
    exploratory: ExploratoryOption = False,
) -> None:
    """Report test results of a frozen config, or of any config with --exploratory."""
    try:
        result = evaluate_method(config, split, seed, method, exploratory)
    except FAILURES as error:
        raise _fail("evaluate", error) from error
    metrics = result.run.metrics
    typer.echo(
        f"{method} on {split}, seed {seed} ({result.status}): Top-1 {metrics['top1']:.3f}, "
        f"Top-3 {metrics['top3']:.3f}, Avg@5 {metrics['avg5']:.3f}"
    )


def freeze(
    name: Annotated[str, typer.Option("--name", help="Name of configs/frozen/<name>.yaml.")],
    config: ConfigOption = Path("configs/ranker.yaml"),
) -> None:
    """Record the choices made on validation for every fold of every split family."""
    try:
        path = freeze_config(config, name)
    except FAILURES as error:
        raise _fail("freeze", error) from error
    typer.echo(f"Wrote {path}.")


def diagnose(
    case: Annotated[Path, typer.Option("--case", help="Metrics table of one case, as Parquet.")],
    model: Annotated[
        Path | None, typer.Option("--model", help="Saved model; the latest by default.")
    ] = None,
) -> None:
    """Rank likely origin services for a single case."""
    try:
        directory = model if model is not None else latest_artifact()
        predictions = diagnose_case(case, directory)
    except FAILURES as error:
        raise _fail("diagnose", error) from error
    typer.echo(f"Model: {directory}")
    for rank, (service, score) in enumerate(
        predictions.select("service", "score").iter_rows(), start=1
    ):
        typer.echo(f"{rank}. {service}  {score:.4f}")


def run_ranker_clean(exploratory: ExploratoryOption = False) -> None:
    """Run both rankers on every split family and compare them with the baselines."""
    try:
        payload = ranker_clean(exploratory)
    except FAILURES as error:
        raise _fail("ranker clean", error) from error
    for message in payload["warnings"]:
        typer.echo(message)
    typer.echo(f"Wrote experiments/ranker_clean ({payload['status']}).")
