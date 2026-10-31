"""The ``cascadelens features`` commands."""

from pathlib import Path
from typing import Annotated

import typer

from cascadelens.config import load_config
from cascadelens.features.config import FeaturesConfig
from cascadelens.features.context import FeatureError
from cascadelens.features.store import build_features, check_features

app = typer.Typer(help="Feature tables and leakage self-checks.", no_args_is_help=True)

ConfigOption = Annotated[Path, typer.Option("--config", help="Features config YAML.")]
DEFAULT_CONFIG = Path("configs/features.yaml")


@app.command()
def build(
    config: ConfigOption = DEFAULT_CONFIG,
    workers: Annotated[int, typer.Option("--workers", help="Worker processes.")] = 1,
) -> None:
    """Build a feature table per case."""
    settings = load_config(config, FeaturesConfig)
    try:
        summary = build_features(settings, workers=workers)
    except FeatureError as error:
        typer.echo(f"features build failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(
        f"Wrote {summary.cases} case(s) with {summary.columns} columns, "
        f"{summary.bytes / 1e6:.1f} MB, to {summary.directory}."
    )


@app.command()
def check(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Verify a build and report leakage warnings. Exits non-zero only if a build is invalid."""
    try:
        report = check_features(load_config(config, FeaturesConfig))
    except FeatureError as error:
        typer.echo(f"features check failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"{report.cases} case(s), {report.rows} rows")
    typer.echo(f"All NaN: {len(report.all_nan)}; constant: {len(report.constant)}")
    typer.echo(f"Row order: {report.row_order}")
    for label, found in (
        ("Exact fingerprints", report.exact_fingerprints),
        ("Near fingerprints", report.near_fingerprints),
    ):
        for system, columns in found.items():
            typer.echo(f"{label} in {system}: {columns}")
