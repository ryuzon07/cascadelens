"""The ``cascadelens eval`` commands."""

from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from cascadelens.config import load_config
from cascadelens.eval.config import EvalConfig
from cascadelens.eval.errors import EvalError
from cascadelens.eval.power import write_power_report

app = typer.Typer(help="Evaluation and statistics.", no_args_is_help=True)

ConfigOption = Annotated[Path, typer.Option("--config", help="Evaluation config YAML.")]
DEFAULT_CONFIG = Path("configs/eval.yaml")


@app.callback()
def group() -> None:
    """Evaluation and statistics."""


@app.command()
def power(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Simulate power on the real group counts and write the report."""
    settings = load_config(config, EvalConfig)
    try:
        result = write_power_report(settings)
    except EvalError as error:
        typer.echo(f"eval power failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    for row in result.minimum_detectable.filter(
        pl.col("alpha") == settings.power.alphas[0]
    ).iter_rows(named=True):
        value = row["minimum_detectable"]
        shown = "not reached" if value is None else f"{100 * value:.1f} points"
        typer.echo(f"{row['scenario']}: minimum detectable difference {shown}")
    typer.echo(f"Wrote {settings.power.output}.")
