"""The ``cascadelens splits`` commands."""

from pathlib import Path
from typing import Annotated

import typer

from cascadelens.config import SplitsConfig, load_config
from cascadelens.splits.errors import SplitError
from cascadelens.splits.store import build_splits, check_splits

app = typer.Typer(help="Grouped splits and leakage reports.", no_args_is_help=True)

ConfigOption = Annotated[Path, typer.Option("--config", help="Splits config YAML.")]
DEFAULT_CONFIG = Path("configs/splits.yaml")


@app.command()
def build(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Build every configured split and its leakage report."""
    settings = load_config(config, SplitsConfig)
    try:
        reports = build_splits(settings)
    except SplitError as error:
        typer.echo(f"splits build failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    for report in reports:
        verdict = "pass" if report.passed else "FAIL"
        note = " (leaky by design)" if report.leaky_by_design else ""
        typer.echo(f"{report.split}: {len(report.folds)} fold(s), overlap {verdict}{note}")
    if any(not report.passed and not report.leaky_by_design for report in reports):
        typer.echo("A grouped split has overlap; see its leakage_report.json.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Wrote {settings.output_dir}.")


@app.command()
def check(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Re-verify saved grouped splits. Exits non-zero on any overlap or stale split."""
    try:
        results = check_splits(load_config(config, SplitsConfig))
    except SplitError as error:
        typer.echo(f"splits check failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    for result in results:
        typer.echo(f"{result.name}: {'FAIL' if result.problems else 'pass'}")
        for problem in result.problems:
            typer.echo(f"  {problem}", err=True)
    if any(result.problems for result in results):
        raise typer.Exit(code=1)
