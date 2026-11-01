"""The ranker commands: train, evaluate, freeze, calibrate and the experiments."""

from pathlib import Path
from typing import Annotated

import typer

from cascadelens.baselines.base import BaselineError
from cascadelens.eval.robustness import freeze_variants
from cascadelens.features.context import FeatureError
from cascadelens.models.abstain import calibrate_split
from cascadelens.models.abstain import run_calibration as calibration_experiment
from cascadelens.models.experiment import ranker_clean
from cascadelens.models.pipeline import evaluate as evaluate_method
from cascadelens.models.pipeline import train_folds
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
) -> None:
    """Record the validation choices of every ranker variant on every split family."""
    try:
        path = freeze_variants(name)
    except FAILURES as error:
        raise _fail("freeze", error) from error
    typer.echo(f"Wrote {path}.")


def calibrate(
    config: ConfigOption = Path("configs/ranker.yaml"),
    split: SplitOption = "group_run",
    method: MethodOption = "two_stage_ranker",
) -> None:
    """Fit a calibrator and tau on validation cases for each saved model. Reads no test case."""
    try:
        saved = calibrate_split(config, split, method)
    except FAILURES as error:
        raise _fail("calibrate", error) from error
    typer.echo(f"Wrote {len(saved)} calibrators for {method} on {split}.")


def run_calibration(exploratory: ExploratoryOption = False) -> None:
    """Measure calibration and abstention on clean, held-out and perturbed test cases."""
    try:
        payload = calibration_experiment(exploratory)
    except FAILURES as error:
        raise _fail("calibration", error) from error
    typer.echo(f"Wrote experiments/calibration ({payload['status']}).")


def run_ranker_clean(exploratory: ExploratoryOption = False) -> None:
    """Run both rankers on every split family and compare them with the baselines."""
    try:
        payload = ranker_clean(exploratory)
    except FAILURES as error:
        raise _fail("ranker clean", error) from error
    for message in payload["warnings"]:
        typer.echo(message)
    typer.echo(f"Wrote experiments/ranker_clean ({payload['status']}).")
