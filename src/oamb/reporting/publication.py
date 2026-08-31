"""Content-addressed generic report publication after closed export validation."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from oamb.artifacts.atomic import atomic_write_bytes
from oamb.artifacts.capsule import (
    PublicationFaultHook,
    publish_with_last_marker,
    verify_published_directory,
)
from oamb.artifacts.validation.profiles import exact_report_export_profile
from oamb.artifacts.validation.report_export import (
    ReportExportInput,
    validate_report_export,
)
from oamb.contracts.evidence import (
    CapsuleManifestEntry,
    DerivationManifest,
    ValidationResult,
)
from oamb.contracts.external import ExternalHistoricalEvidenceReport
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.reporting import (
    ComparisonReportModel,
    DiagnosticRunReportModel,
    EvaluationReportModel,
    ReleaseReportModel,
    ReportArtifactManifestV2,
    ReportArtifactManifestV3,
    RunReportModelV3,
)
from oamb.contracts.specifications import (
    DerivationSpecV2,
    DerivationSpecV3,
    ReportSpec,
    ReportSpecV2,
    SourceEvidenceBinding,
)
from oamb.contracts.states import ValidationDisposition
from oamb.external_evidence.limitations import historical_limitation_texts
from oamb.reporting.offline_renderer import (
    OfflineReportModel,
    offline_asset_hashes,
    offline_renderer_hash,
    render_offline_report,
)
from oamb.reporting.public import (
    build_evaluation_model_closure,
    build_report_artifact_manifest_v2,
    build_report_artifact_manifest_v3,
)
from oamb.reporting.roots import (
    build_derivation_spec_v2,
    build_derivation_spec_v3,
    build_report_identity_spec_binding,
)

DERIVATION_MARKER = "derived-manifest.json"
ReportIdentitySpec = ReportSpec | ReportSpecV2
ReportKind = Literal["run", "comparison", "evaluation", "release"]


class ReportExportError(ValueError):
    def __init__(self, result: ValidationResult, attempt_directory: Path) -> None:
        super().__init__("report export validation failed")
        self.result = result
        self.attempt_directory = attempt_directory


@dataclass(frozen=True, slots=True)
class ReportBuildResultV2:
    derivation_id: str
    final_directory: Path
    report_path: Path
    attempt_directory: Path
    report_html_hash: str
    export_validation: ValidationResult


def build_report_derivation(
    *,
    model: OfflineReportModel,
    report_spec: ReportIdentitySpec,
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...],
    evidence_validations: tuple[ValidationResult, ...],
    evidence_validation_targets: tuple[Any, ...],
    transform_spec_hash: str,
    schema_versions: tuple[str, ...],
    output_root: Path,
    committed_at: datetime,
    extra_publication_payloads: Mapping[str, bytes] | None = None,
    forbidden_public_values: tuple[bytes, ...] = (),
    omitted_scan_paths: frozenset[str] = frozenset(),
    fault_hook: PublicationFaultHook | None = None,
    diagnostic: bool = False,
) -> ReportBuildResultV2:
    report_kind = _report_kind(model, report_spec)
    if diagnostic != isinstance(model, DiagnosticRunReportModel):
        raise ValueError(
            "diagnostic publication requires both the restricted model and explicit opt-in"
        )
    report_model_bytes = canonical_json_bytes(model)
    report_html = render_offline_report(model)
    renderer_hash = offline_renderer_hash()
    asset_hashes = offline_asset_hashes()
    binding = build_report_identity_spec_binding(report_spec)
    validation_hashes = tuple(canonical_sha256(item) for item in evidence_validations)
    report_model_hash = hashlib.sha256(report_model_bytes).hexdigest()
    evidence_validation_hash = canonical_sha256(
        ["oamb-ordered-evidence-validations-v1", evidence_validations]
    )
    limitations = (
        historical_limitation_texts(model.limitation_codes)
        if isinstance(model, ExternalHistoricalEvidenceReport)
        else model.limitations
    )
    reducer_and_renderer_inputs = (
        report_model_hash,
        renderer_hash,
        *asset_hashes,
        report_spec.browser_contract_hash,
        report_spec.performance_contract_hash,
    )
    artifact: ReportArtifactManifestV2 | ReportArtifactManifestV3
    if isinstance(model, EvaluationReportModel):
        if not isinstance(report_spec, ReportSpecV2) or binding.schema_version != 2:
            raise ValueError("evaluation publication requires the v2 report identity chain")
        model_closure = build_evaluation_model_closure(model)
        export_profile_hash = canonical_sha256(
            exact_report_export_profile(
                report_kind="evaluation",
                audience=report_spec.audience,
            )
        )
        artifact = build_report_artifact_manifest_v3(
            report_id=model.report_id,
            report_identity_spec_binding=binding,
            ordered_source_bindings=ordered_source_bindings,
            ordered_evidence_validation_hashes=validation_hashes,
            report_model_hash=report_model_hash,
            evaluation_model_closure=model_closure,
            renderer_hash=renderer_hash,
            asset_hashes=asset_hashes,
            browser_contract_hash=report_spec.browser_contract_hash,
            performance_contract_hash=report_spec.performance_contract_hash,
            export_profile_selector_id=report_spec.export_profile_selector_id,
            export_profile_selector_version=report_spec.export_profile_selector_version,
            export_profile_hash=export_profile_hash,
            audience=report_spec.audience,
            schema_versions=schema_versions,
            limitations=limitations,
        )
        derivation: DerivationSpecV2 | DerivationSpecV3 = build_derivation_spec_v3(
            ordered_source_bindings=ordered_source_bindings,
            evidence_validation_result_hash=evidence_validation_hash,
            transform_spec_hash=transform_spec_hash,
            report_identity_spec_binding=binding,
            evaluation_model_closure=model_closure,
            reducer_and_renderer_input_hashes=reducer_and_renderer_inputs,
        )
    else:
        if isinstance(report_spec, ReportSpecV2) or binding.schema_version != 1:
            raise ValueError("non-evaluation publication requires the frozen v1/v2 chain")
        if report_kind == "evaluation":
            raise ValueError("evaluation publication requires an evaluation report model")
        artifact = build_report_artifact_manifest_v2(
            report_id=model.report_id,
            report_kind=report_kind,
            report_identity_spec_binding=binding,
            ordered_source_bindings=ordered_source_bindings,
            ordered_evidence_validation_hashes=validation_hashes,
            report_model_hash=report_model_hash,
            renderer_hash=renderer_hash,
            asset_hashes=asset_hashes,
            browser_contract_hash=report_spec.browser_contract_hash,
            performance_contract_hash=report_spec.performance_contract_hash,
            export_profile_selector_id=report_spec.export_profile_selector_id,
            export_profile_selector_version=report_spec.export_profile_selector_version,
            audience=report_spec.audience,
            schema_versions=schema_versions,
            limitations=limitations,
        )
        derivation = build_derivation_spec_v2(
            derivation_kind=_derivation_kind(report_kind),
            ordered_source_bindings=ordered_source_bindings,
            evidence_validation_result_hash=evidence_validation_hash,
            transform_spec_hash=transform_spec_hash,
            report_identity_spec_binding=binding,
            reducer_and_renderer_input_hashes=reducer_and_renderer_inputs,
        )
    payloads: dict[str, bytes] = {
        "derivation-spec.json": canonical_json_bytes(derivation),
        "source-roots.json": canonical_json_bytes(ordered_source_bindings),
        "input-specs/report-spec.json": canonical_json_bytes(report_spec),
        "evidence-validations.json": canonical_json_bytes(evidence_validations),
        "outputs/report-model.json": report_model_bytes,
        "outputs/report.html": report_html,
        "report-artifact-manifest.json": canonical_json_bytes(artifact),
    }
    for path, content in (extra_publication_payloads or {}).items():
        if path in payloads:
            raise ValueError(f"duplicate report publication payload: {path}")
        payloads[path] = content

    attempt_id = canonical_sha256(
        [
            (
                "oamb-report-derivation-attempt-v3"
                if isinstance(derivation, DerivationSpecV3)
                else "oamb-report-derivation-attempt-v2"
            ),
            derivation.derivation_input_hash,
            evidence_validation_hash,
        ]
    )
    output_root = Path(output_root)
    attempt_directory = output_root / "derivation-attempts" / attempt_id
    for path, content in sorted(payloads.items()):
        atomic_write_bytes(attempt_directory / path, content, trusted_root=attempt_directory)
    scan_paths = frozenset(payloads) - omitted_scan_paths
    export_validation = validate_report_export(
        ReportExportInput(
            payloads=payloads,
            scan_paths=scan_paths,
            model=model,
            artifact_manifest=artifact,
            report_spec=report_spec,
            derivation_spec=derivation,
            ordered_source_bindings=ordered_source_bindings,
            evidence_validations=evidence_validations,
            evidence_validation_targets=evidence_validation_targets,
            forbidden_values=forbidden_public_values,
        )
    )
    export_bytes = canonical_json_bytes(export_validation)
    atomic_write_bytes(
        attempt_directory / "export-validation.json",
        export_bytes,
        trusted_root=attempt_directory,
    )
    if export_validation.disposition != ValidationDisposition.VALIDATED:
        raise ReportExportError(export_validation, attempt_directory)

    export_validation_hash = canonical_sha256(export_validation)
    derivation_id = (
        derive_publication_id_v3(
            derivation,
            export_validation_result_hash=export_validation_hash,
        )
        if isinstance(derivation, DerivationSpecV3)
        else derive_publication_id_v2(
            derivation,
            export_validation_result_hash=export_validation_hash,
        )
    )
    committed_payloads = dict(payloads)
    committed_payloads["export-validation.json"] = export_bytes
    committed_entries = tuple(
        CapsuleManifestEntry(
            record_kind=_record_kind(path),
            record_id=hashlib.sha256(content).hexdigest(),
            relative_path=path,
            sha256=hashlib.sha256(content).hexdigest(),
        )
        for path, content in sorted(committed_payloads.items())
    )
    marker = DerivationManifest(
        derivation_id=derivation_id,
        derivation_spec_hash=canonical_sha256(derivation),
        evidence_validation_result_hash=evidence_validation_hash,
        export_validation_result_hash=export_validation_hash,
        committed_entries=committed_entries,
        committed_at=committed_at,
    )
    final_directory = output_root / "derivations" / derivation_id
    publish_with_last_marker(
        final_directory,
        payloads=committed_payloads,
        marker_name=DERIVATION_MARKER,
        marker_bytes=canonical_json_bytes(marker),
        fault_hook=fault_hook,
    )
    audit_report_envelope(final_directory)
    return ReportBuildResultV2(
        derivation_id=derivation_id,
        final_directory=final_directory,
        report_path=final_directory / "outputs" / "report.html",
        attempt_directory=attempt_directory,
        report_html_hash=hashlib.sha256(report_html).hexdigest(),
        export_validation=export_validation,
    )


def derive_publication_id_v2(
    derivation: DerivationSpecV2,
    *,
    export_validation_result_hash: str,
) -> str:
    return canonical_sha256(
        [
            "oamb-report-publication-v2",
            derivation.derivation_input_hash,
            export_validation_result_hash,
        ]
    )


def derive_publication_id_v3(
    derivation: DerivationSpecV3,
    *,
    export_validation_result_hash: str,
) -> str:
    return canonical_sha256(
        [
            "oamb-report-publication-v3",
            derivation.derivation_input_hash,
            export_validation_result_hash,
        ]
    )


def audit_report_envelope(final_directory: Path) -> None:
    final_directory = Path(final_directory)
    verify_published_directory(final_directory, DERIVATION_MARKER)
    marker_path = final_directory / DERIVATION_MARKER
    marker = DerivationManifest.model_validate_json(marker_path.read_bytes())
    if marker_path.read_bytes() != canonical_json_bytes(marker):
        raise ValueError("report derivation marker is not canonical")
    if marker.derivation_id != final_directory.name:
        raise ValueError("report derivation marker identity disagrees with its directory")
    indexed = {item.relative_path: item for item in marker.committed_entries}
    actual = {
        path.relative_to(final_directory).as_posix()
        for path in final_directory.rglob("*")
        if path.is_file() and path.name not in {"CHECKSUMS.sha256", DERIVATION_MARKER}
    }
    if set(indexed) != actual:
        raise ValueError("report derivation marker inventory does not close")
    for relative_path, entry in indexed.items():
        content = (final_directory / relative_path).read_bytes()
        if hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ValueError("report derivation marker hash does not close")


def _report_kind(
    model: OfflineReportModel,
    report_spec: ReportIdentitySpec,
) -> ReportKind:
    expected: ReportKind
    if isinstance(
        model,
        (RunReportModelV3, DiagnosticRunReportModel, ExternalHistoricalEvidenceReport),
    ):
        expected = "run"
    elif isinstance(model, ComparisonReportModel):
        expected = "comparison"
    elif isinstance(model, EvaluationReportModel):
        expected = "evaluation"
    elif isinstance(model, ReleaseReportModel):
        expected = "release"
    else:
        raise TypeError("unsupported offline report model")
    if report_spec.report_kind != expected:
        raise ValueError("report model and ReportSpec kind do not match")
    return expected


def _derivation_kind(report_kind: ReportKind) -> str:
    return {
        "run": "run_report",
        "comparison": "comparison_report",
        "evaluation": "evaluation_report",
        "release": "release_report",
    }[report_kind]


def _record_kind(path: str) -> str:
    return path.rsplit("/", 1)[-1].removesuffix(".json").replace("-", "_")


__all__ = [
    "ReportBuildResultV2",
    "ReportExportError",
    "audit_report_envelope",
    "build_report_derivation",
    "derive_publication_id_v2",
    "derive_publication_id_v3",
]
