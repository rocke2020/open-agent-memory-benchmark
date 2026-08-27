"""Offline-first OAMB command composition root."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .contracts.schema import generate_schemas, schema_drift

app = typer.Typer(
    name="oamb",
    help="Open Agent Memory Benchmark evidence and comparison tooling.",
    no_args_is_help=True,
)
schema_app = typer.Typer(help="Generate or verify checked-in public JSON Schemas.")
app.add_typer(schema_app, name="schema")


@schema_app.command("build")
def schema_build(
    output: Annotated[Path, typer.Option("--output", help="Schema output directory.")] = Path(
        "schemas"
    ),
) -> None:
    """Generate every registered public contract schema."""

    written = generate_schemas(output)
    typer.echo(f"generated {len(written)} schemas in {output}")


@schema_app.command("check")
def schema_check(
    output: Annotated[Path, typer.Option("--output", help="Tracked schema directory.")] = Path(
        "schemas"
    ),
) -> None:
    """Fail when tracked schemas differ from deterministic generation."""

    drift = schema_drift(output)
    if drift:
        for item in drift:
            typer.echo(item, err=True)
        raise typer.Exit(code=1)
    typer.echo(f"schema check passed: {output}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
