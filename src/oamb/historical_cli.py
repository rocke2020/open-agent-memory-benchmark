"""Installed CLI flow for pinned external historical evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from oamb.artifacts.atomic import atomic_write_bytes
from oamb.contracts.evidence import ValidationResult
from oamb.contracts.external import ExternalHistoricalEvidence
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.states import ValidationDisposition
from oamb.external_evidence.amb import (
    import_historical_amb_result,
    load_curated_evidence,
    load_packaged_curated_evidence,
)
from oamb.external_evidence.reduce import reduce_external_historical_report
from oamb.external_evidence.validation import validate_external_historical_evidence

external_app = typer.Typer(
    help="Import, validate, and report pinned external historical evidence.",
    no_args_is_help=True,
)


@external_app.command("import")
def import_external_historical_evidence(
    source: Annotated[
        Path,
        typer.Option("--source", help="Pinned producer result JSON path."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only restricted factual evidence JSON."),
    ],
) -> None:
    """Verify the complete raw source, then retain only allowlisted facts."""

    try:
        record = import_historical_amb_result(source)
    except (OSError, ValueError):
        raise typer.BadParameter("external raw source does not match the fixed profile") from None
    try:
        written = atomic_write_bytes(
            output, canonical_json_bytes(record), trusted_root=output.parent
        )
    except OSError:
        raise typer.BadParameter("external evidence output could not be published") from None
    typer.echo(f"external evidence: {record.external_evidence_id}")
    typer.echo(f"created: {str(written.created).lower()}")


@external_app.command("validate")
def validate_external_historical_evidence_command(
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only evidence validation JSON."),
    ],
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Restricted factual evidence JSON; defaults to the packaged frozen pack.",
        ),
    ] = None,
) -> None:
    """Run the fixed five-rule external evidence profile."""

    record = _load_external_record(input_path)
    validation = validate_external_historical_evidence(record)
    try:
        written = atomic_write_bytes(
            output,
            canonical_json_bytes(validation),
            trusted_root=output.parent,
        )
    except OSError:
        raise typer.BadParameter("external validation output could not be published") from None
    typer.echo(f"validation: {validation.disposition.value}")
    typer.echo(f"created: {str(written.created).lower()}")
    if validation.disposition != ValidationDisposition.VALIDATED:
        raise typer.Exit(code=1)


@external_app.command("report")
def report_external_historical_evidence(
    validation_path: Annotated[
        Path,
        typer.Option("--validation", help="External evidence validation JSON path."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Create-only report derivation parent."),
    ],
    input_path: Annotated[
        Path | None,
        typer.Option(
            "--input",
            help="Restricted factual evidence JSON; defaults to the packaged frozen pack.",
        ),
    ] = None,
    audience: Annotated[
        str,
        typer.Option("--audience", help="Report audience: public or local."),
    ] = "public",
) -> None:
    """Freshly validate, reduce, export-validate, and seal the external report."""

    from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash
    from oamb.reporting.publication import build_report_derivation
    from oamb.reporting.roots import build_report_spec

    if audience not in {"public", "local"}:
        raise typer.BadParameter("audience must be 'public' or 'local'")
    record = _load_external_record(input_path)
    try:
        supplied_validation = ValidationResult.model_validate_json(validation_path.read_bytes())
    except (OSError, ValueError):
        raise typer.BadParameter("external validation input is invalid") from None
    fresh_validation = validate_external_historical_evidence(record)
    if supplied_validation != fresh_validation:
        raise typer.BadParameter("validation does not match fresh external evidence bytes")
    report_spec = build_report_spec(
        report_kind="run",
        audience=audience,
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
        renderer_hash=offline_renderer_hash(),
        asset_hashes=offline_asset_hashes(),
        export_profile_selector_id=f"{audience}-run-v1",
        export_profile_selector_version=1,
    )
    report = reduce_external_historical_report(
        record,
        fresh_validation,
        report_spec=report_spec,
    )
    try:
        result = build_report_derivation(
            model=report,
            report_spec=report_spec,
            ordered_source_bindings=report.ordered_source_bindings,
            evidence_validations=(fresh_validation,),
            evidence_validation_targets=(record,),
            transform_spec_hash=canonical_sha256(
                ["amb-historical-report-transform-v1", report.reducer_binding]
            ),
            schema_versions=(
                "external_historical_evidence_report@1",
                "report_artifact_manifest@2",
            ),
            output_root=output_root,
            committed_at=record.imported_at,
        )
    except (OSError, ValueError):
        raise typer.BadParameter("external report publication failed") from None
    typer.echo(f"report: {result.report_path}")


def _load_external_record(input_path: Path | None) -> ExternalHistoricalEvidence:
    try:
        if input_path is None:
            return load_packaged_curated_evidence()
        return load_curated_evidence(input_path)
    except (OSError, ValueError):
        raise typer.BadParameter("external evidence input is invalid") from None


__all__ = ["external_app"]
