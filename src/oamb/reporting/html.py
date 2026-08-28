"""Deterministic single-file fake report and content-addressed publication."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from oamb.artifacts.atomic import atomic_write_bytes
from oamb.artifacts.capsule import (
    derive_publication_id,
    publish_with_last_marker,
    verify_published_directory,
)
from oamb.artifacts.validation.fake import EvidenceNotValidatedError, load_fake_capsule
from oamb.contracts.evidence import (
    CapsuleManifest,
    CapsuleManifestEntry,
    DerivationManifest,
    ValidationResult,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.reporting import ReportArtifactManifest, RunReportModelV2
from oamb.contracts.specifications import (
    DerivationSpec,
    SourceEvidenceBinding,
    SourceEvidenceKind,
)
from oamb.contracts.states import ValidationDisposition

from .claims import fake_report_id, fake_report_limitations
from .reduce import reduce_diagnostic_summary, reduce_run_summary_snapshot
from .renderer import (
    fake_report_renderer_hash,
    load_fake_report_css,
    render_fake_report_html,
)
from .validation import ExportValidationError, validate_fake_export

DERIVATION_MARKER = "derived-manifest.json"
FAKE_REPORT_COMMITTED_AT = datetime(2026, 1, 2, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class ReportBuildResult:
    derivation_id: str
    final_directory: Path
    report_path: Path
    attempt_directory: Path
    export_validation: ValidationResult


def build_fake_report(
    capsule_root: Path,
    evidence_validation: ValidationResult,
    *,
    output_root: Path,
    audience: Literal["local", "public"],
    diagnostic: bool = False,
) -> ReportBuildResult:
    capsule_root = Path(capsule_root)
    if evidence_validation.disposition != ValidationDisposition.VALIDATED and not diagnostic:
        raise EvidenceNotValidatedError("normal report requires evidence validation PASS")
    if diagnostic:
        manifest = CapsuleManifest.model_validate_json(
            (capsule_root / "capsule-manifest.json").read_bytes()
        )
        summary = reduce_diagnostic_summary(capsule_root)
    else:
        snapshot = load_fake_capsule(capsule_root)
        if snapshot.manifest is None:
            raise EvidenceNotValidatedError("normal report requires a valid capsule manifest")
        manifest = snapshot.manifest
        summary = reduce_run_summary_snapshot(snapshot, evidence_validation)
    evidence_hash = canonical_sha256(evidence_validation)
    limitations = fake_report_limitations(evidence_validation, diagnostic=diagnostic)
    report_id = fake_report_id(
        source_manifest_hash=manifest.source_manifest_hash,
        evidence_validation_hash=evidence_hash,
        summary=summary,
        audience=audience,
        diagnostic=diagnostic,
        limitations=limitations,
    )
    report_model = RunReportModelV2(
        report_id=report_id,
        source_manifest_hash=manifest.source_manifest_hash,
        evidence_validation_hash=evidence_hash,
        summary=summary,
        limitations=limitations,
    )
    css = load_fake_report_css()
    css_hash = hashlib.sha256(css).hexdigest()
    renderer_hash = fake_report_renderer_hash(css)
    report_model_bytes = canonical_json_bytes(report_model)
    artifact_manifest = ReportArtifactManifest(
        report_id=report_id,
        ordered_source_root_hashes=(manifest.source_manifest_hash,),
        evidence_validation_hash=evidence_hash,
        report_model_hash=hashlib.sha256(report_model_bytes).hexdigest(),
        renderer_hash=renderer_hash,
        asset_hashes=(css_hash,),
        audience=audience,
        limitations=limitations,
    )
    report_html = render_fake_report_html(report_model, css, diagnostic=diagnostic)
    source_binding = _source_binding(manifest, evidence_hash)
    report_spec = {
        "schema_name": "fake_report_spec",
        "schema_version": 1,
        "report_kind": "run",
        "audience": audience,
        "diagnostic": diagnostic,
        "renderer_hash": renderer_hash,
    }
    report_spec_hash = canonical_sha256(report_spec)
    ordered_source_hash = canonical_sha256(
        ["oamb-ordered-source-roots-v1", (manifest.source_manifest_hash,)]
    )
    transform_hash = canonical_sha256(
        ["oamb-fake-report-transform-v1", "diagnostic" if diagnostic else "normal"]
    )
    reducer_inputs = (
        canonical_sha256(summary),
        renderer_hash,
        css_hash,
    )
    derivation_fields = {
        "derivation_kind": "diagnostic_run_report" if diagnostic else "run_report",
        "ordered_source_bindings": (source_binding,),
        "ordered_source_root_hash": ordered_source_hash,
        "transform_spec_hash": transform_hash,
        "report_spec_hash": report_spec_hash,
        "reducer_and_renderer_input_hashes": reducer_inputs,
    }
    derivation_spec = DerivationSpec(
        derivation_kind=str(derivation_fields["derivation_kind"]),
        ordered_source_bindings=(source_binding,),
        ordered_source_root_hash=ordered_source_hash,
        transform_spec_hash=transform_hash,
        report_spec_hash=report_spec_hash,
        reducer_and_renderer_input_hashes=reducer_inputs,
        derivation_input_hash=canonical_sha256(["oamb-derivation-input-v1", derivation_fields]),
    )
    payloads = {
        "derivation-spec.json": canonical_json_bytes(derivation_spec),
        "source-roots.json": canonical_json_bytes((source_binding,)),
        "input-specs/report-spec.json": canonical_json_bytes(report_spec),
        "evidence-validation.json": canonical_json_bytes(evidence_validation),
        "outputs/summaries/run-summary.json": canonical_json_bytes(summary),
        "outputs/report-model.json": report_model_bytes,
        "outputs/report.html": report_html,
        "report-artifact-manifest.json": canonical_json_bytes(artifact_manifest),
    }
    attempt_id = canonical_sha256(
        [
            "oamb-fake-report-attempt-v1",
            derivation_spec.derivation_input_hash,
            evidence_hash,
        ]
    )
    attempt_directory = Path(output_root) / "derivation-attempts" / attempt_id
    for relative_path, content in sorted(payloads.items()):
        atomic_write_bytes(
            attempt_directory / relative_path,
            content,
            trusted_root=attempt_directory,
        )
    export_validation = validate_fake_export(payloads)
    export_bytes = canonical_json_bytes(export_validation)
    atomic_write_bytes(
        attempt_directory / "export-validation.json",
        export_bytes,
        trusted_root=attempt_directory,
    )
    if export_validation.disposition != ValidationDisposition.VALIDATED:
        raise ExportValidationError(export_validation)
    export_hash = canonical_sha256(export_validation)
    derivation_id = derive_publication_id(derivation_spec, evidence_hash, export_hash)
    committed_payloads = dict(payloads)
    committed_payloads["export-validation.json"] = export_bytes
    committed_entries = tuple(
        CapsuleManifestEntry(
            record_kind=_committed_record_kind(relative_path),
            record_id=hashlib.sha256(content).hexdigest(),
            relative_path=relative_path,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        for relative_path, content in sorted(committed_payloads.items())
    )
    derived_manifest = DerivationManifest(
        derivation_id=derivation_id,
        derivation_spec_hash=canonical_sha256(derivation_spec),
        evidence_validation_result_hash=evidence_hash,
        export_validation_result_hash=export_hash,
        committed_entries=committed_entries,
        committed_at=FAKE_REPORT_COMMITTED_AT,
    )
    final_directory = Path(output_root) / "derivations" / derivation_id
    publish_with_last_marker(
        final_directory,
        payloads=committed_payloads,
        marker_name=DERIVATION_MARKER,
        marker_bytes=canonical_json_bytes(derived_manifest),
    )
    verify_published_directory(final_directory, DERIVATION_MARKER)
    return ReportBuildResult(
        derivation_id=derivation_id,
        final_directory=final_directory,
        report_path=final_directory / "outputs" / "report.html",
        attempt_directory=attempt_directory,
        export_validation=export_validation,
    )


def _source_binding(manifest: CapsuleManifest, validation_hash: str) -> SourceEvidenceBinding:
    fields = {
        "source_kind": SourceEvidenceKind.RUN,
        "source_identity": manifest.run_id,
        "source_root_hash": manifest.source_manifest_hash,
        "validation_result_hash": validation_hash,
        "source_schema_versions": ("capsule_manifest@1",),
    }
    return SourceEvidenceBinding(
        binding_id=canonical_sha256(["oamb-source-evidence-binding-v1", fields]),
        source_kind=SourceEvidenceKind.RUN,
        source_identity=manifest.run_id,
        source_root_hash=manifest.source_manifest_hash,
        validation_result_hash=validation_hash,
        source_schema_versions=("capsule_manifest@1",),
    )


def _committed_record_kind(relative_path: str) -> str:
    if relative_path == "outputs/report.html":
        return "report_html"
    if relative_path == "source-roots.json":
        return "source_roots"
    return relative_path.rsplit("/", 1)[-1].removesuffix(".json").replace("-", "_")


__all__ = ["ReportBuildResult", "build_fake_report"]
