"""The ``cascadelens data`` commands."""

from pathlib import Path
from typing import Annotated

import typer

from cascadelens.config import DataConfig, load_config
from cascadelens.data.audit import run_audit
from cascadelens.data.errors import DataError
from cascadelens.data.pipeline import build as run_build

app = typer.Typer(help="Data ingestion and audit.", no_args_is_help=True)

ConfigOption = Annotated[Path, typer.Option("--config", help="Data config YAML.")]
DEFAULT_CONFIG = Path("configs/data.yaml")
DEFAULT_AUDIT = Path("docs/data_audit.md")


@app.command()
def build(config: ConfigOption = DEFAULT_CONFIG) -> None:
    """Load, validate and write cases, then write the manifest."""
    settings = load_config(config, DataConfig)
    try:
        warnings = run_build(settings)
    except DataError as error:
        typer.echo(f"data build failed: {error}", err=True)
        raise typer.Exit(code=1) from error
    typer.echo(f"Wrote {settings.processed_dir} with {len(warnings)} logged warning(s).")


@app.command()
def audit(
    config: ConfigOption = DEFAULT_CONFIG,
    output: Annotated[Path, typer.Option("--output", help="Report path.")] = DEFAULT_AUDIT,
) -> None:
    """Write the data audit. Exits non-zero unless every selected case loaded cleanly."""
    data = run_audit(load_config(config, DataConfig), output)
    typer.echo(f"Wrote {output} covering {len(data.stats)} case(s).")
    failed = data.layout_error or any(issue.severity == "error" for issue in data.issues)
    if failed or not data.stats:
        typer.echo("data audit found problems; see the Concerns section.", err=True)
        raise typer.Exit(code=1)
