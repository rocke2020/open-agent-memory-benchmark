"""Closed export validation over the complete report publication payload set."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from oamb.artifacts.atomic import read_regular_file
from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.artifacts.validation.engine import validate_closed_profile
from oamb.artifacts.validation.native import (
    NATIVE_EVIDENCE_PROFILE_ID,
    NATIVE_EVIDENCE_RULE_IDS,
    validate_native_capsule,
)
from oamb.artifacts.validation.phase import (
    T10_PHASE_GATE_RULE_IDS,
    PhaseGateValidationInput,
)
from oamb.artifacts.validation.profiles import (
    T8_REPORT_EXPORT_RULE_INVENTORY,
    exact_report_export_profile,
    validation_profile_catalog,
)
from oamb.artifacts.validation.registry import RuleRegistry, ValidationRule
from oamb.artifacts.validation.run_evidence import (
    NATIVE_RUN_EVIDENCE_PROFILE_ID,
    NATIVE_RUN_EVIDENCE_RULE_INVENTORY,
    NativeRunEvidenceValidationInput,
    validate_native_run_evidence,
)
from oamb.constants import REPORT_MAX_HTML_BYTES
from oamb.contracts.evidence import (
    CapsuleManifest,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.external import (
    ExternalHistoricalEvidence,
    ExternalHistoricalEvidenceReport,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.reporting import (
    ComparisonReportModel,
    DiagnosticRunReportModel,
    PhaseAcceptanceReport,
    ReleaseReportModel,
    ReportArtifactManifestV2,
    RunReportModelV3,
)
from oamb.contracts.specifications import (
    AcceptanceReportSpec,
    DerivationSpecV2,
    ReportSpec,
    SourceEvidenceBinding,
    ValidationStage,
)
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.acceptance_contracts import (
    browser_acceptance_contract_hash,
    performance_acceptance_contract_hash,
)
from oamb.reporting.offline_renderer import (
    OfflineReportModel,
    offline_asset_hashes,
    offline_renderer_hash,
    render_offline_report,
)

ReportIdentitySpec = ReportSpec | AcceptanceReportSpec

_EXPORT_SELECTOR_BY_KIND_AND_AUDIENCE = {
    (report_kind, audience): (f"{audience}-{report_kind.replace('_', '-')}-v1", 1)
    for report_kind in ("run", "comparison", "release", "phase_acceptance")
    for audience in ("public", "local")
}
_REPORT_SCHEMA_INVENTORY_BY_KIND = {
    "run": ("run_report_model@3", "report_artifact_manifest@2"),
    "comparison": ("comparison_report_model@1", "report_artifact_manifest@2"),
    "release": ("release_report_model@1", "report_artifact_manifest@2"),
    "phase_acceptance": ("phase_acceptance_report@1", "report_artifact_manifest@2"),
}


@dataclass(frozen=True, slots=True)
class ReportExportInput:
    payloads: Mapping[str, bytes]
    scan_paths: frozenset[str]
    model: OfflineReportModel
    artifact_manifest: ReportArtifactManifestV2
    report_spec: ReportIdentitySpec
    derivation_spec: DerivationSpecV2
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    evidence_validations: tuple[ValidationResult, ...]
    evidence_validation_targets: tuple[Any, ...]
    forbidden_values: tuple[bytes, ...]


def validate_report_export(
    target: ReportExportInput,
) -> ValidationResult:
    report_kind = (
        target.report_spec.report_kind
        if isinstance(target.report_spec, ReportSpec)
        else "phase_acceptance"
    )
    selected_profile = exact_report_export_profile(
        report_kind=report_kind,
        audience=target.report_spec.audience,
    )
    target_hash = canonical_sha256(
        [
            "oamb-report-publication-payload-v1",
            tuple(
                (path, hashlib.sha256(content).hexdigest())
                for path, content in sorted(target.payloads.items())
            ),
            tuple(sorted(target.scan_paths)),
        ]
    )
    return validate_closed_profile(
        target,
        target_hash=target_hash,
        profile=selected_profile,
        expected_stage=ValidationStage.EXPORT,
        expected_inventory=T8_REPORT_EXPORT_RULE_INVENTORY,
        registry=_report_export_registry(),
    )


def _report_export_registry() -> RuleRegistry:
    registry = RuleRegistry()
    for rule in REPORT_EXPORT_RULES:
        registry.register(rule)
    return registry


def _payload_closure_rule(target: ReportExportInput) -> tuple[ValidationIssue, ...]:
    rule_id = "report.export.payload-closure.v1"
    spec_name = (
        "acceptance-report-spec.json"
        if isinstance(target.report_spec, AcceptanceReportSpec)
        else "report-spec.json"
    )
    required = {
        "derivation-spec.json",
        "source-roots.json",
        f"input-specs/{spec_name}",
        "evidence-validations.json",
        "outputs/report-model.json",
        "outputs/report.html",
        "report-artifact-manifest.json",
    }
    paths = set(target.payloads)
    unsafe = any(
        not path
        or PurePosixPath(path).is_absolute()
        or ".." in PurePosixPath(path).parts
        or "\\" in path
        for path in paths
    )
    if (
        not required <= paths
        or any(not content for content in target.payloads.values())
        or target.scan_paths != frozenset(paths)
        or {"CHECKSUMS.sha256", "derived-manifest.json"} & paths
        or unsafe
    ):
        code = (
            "publication-scan-coverage-mismatch"
            if target.scan_paths != frozenset(paths)
            else "publication-payload-closure-mismatch"
        )
        return (_issue(rule_id, "publication-payload", code),)
    return ()


def _canonical_bindings_rule(target: ReportExportInput) -> tuple[ValidationIssue, ...]:
    rule_id = "report.export.canonical-bindings.v1"
    spec_path = (
        "input-specs/acceptance-report-spec.json"
        if isinstance(target.report_spec, AcceptanceReportSpec)
        else "input-specs/report-spec.json"
    )
    expected_validation_hashes = tuple(
        canonical_sha256(item) for item in target.evidence_validations
    )
    try:
        exact_bytes = (
            target.payloads["derivation-spec.json"] == canonical_json_bytes(target.derivation_spec)
            and target.payloads["source-roots.json"]
            == canonical_json_bytes(target.ordered_source_bindings)
            and target.payloads[spec_path] == canonical_json_bytes(target.report_spec)
            and target.payloads["evidence-validations.json"]
            == canonical_json_bytes(target.evidence_validations)
            and target.payloads["outputs/report-model.json"] == canonical_json_bytes(target.model)
            and target.payloads["report-artifact-manifest.json"]
            == canonical_json_bytes(target.artifact_manifest)
        )
    except KeyError:
        exact_bytes = False
    diagnostic = isinstance(target.model, DiagnosticRunReportModel)
    validation_closed = bool(target.evidence_validations) and all(
        (
            validation.disposition == ValidationDisposition.INVALID
            and _authorized_diagnostic_evidence_validation(validation)
            if diagnostic
            else validation.disposition == ValidationDisposition.VALIDATED
            and _authorized_evidence_validation(validation)
        )
        and validation.target_hash == source.source_root_hash
        and canonical_sha256(validation) == source.validation_result_hash
        and _fresh_evidence_validation(validation_target, validation) == validation
        for source, validation, validation_target in zip(
            target.ordered_source_bindings,
            target.evidence_validations,
            target.evidence_validation_targets,
            strict=False,
        )
    )
    model_sources = getattr(target.model, "ordered_source_bindings", None)
    spec_hash = canonical_sha256(target.report_spec)
    model_spec_hash = getattr(
        target.model,
        "report_spec_hash",
        getattr(target.model, "acceptance_report_spec_hash", None),
    )
    artifact = target.artifact_manifest
    model_validation_roots_close = _model_validation_roots_close(
        target.model,
        target.report_spec,
        target.evidence_validations,
        target.evidence_validation_targets,
        expected_validation_hashes,
    )
    report_kind = artifact.report_kind
    expected_selector = _EXPORT_SELECTOR_BY_KIND_AND_AUDIENCE.get(
        (report_kind, target.report_spec.audience)
    )
    if (
        not exact_bytes
        or len(target.ordered_source_bindings) != len(target.evidence_validations)
        or len(target.evidence_validations) != len(target.evidence_validation_targets)
        or not validation_closed
        or (model_sources is not None and model_sources != target.ordered_source_bindings)
        or model_spec_hash != spec_hash
        or not model_validation_roots_close
        or expected_selector is None
        or (
            target.report_spec.export_profile_selector_id,
            target.report_spec.export_profile_selector_version,
        )
        != expected_selector
        or (
            artifact.export_profile_selector_id,
            artifact.export_profile_selector_version,
        )
        != expected_selector
        or artifact.schema_versions != _report_schema_inventory(target.model, report_kind)
        or artifact.report_id != target.model.report_id
        or artifact.ordered_source_bindings != target.ordered_source_bindings
        or artifact.ordered_evidence_validation_hashes != expected_validation_hashes
        or artifact.report_model_hash
        != hashlib.sha256(target.payloads.get("outputs/report-model.json", b"")).hexdigest()
        or artifact.renderer_hash != offline_renderer_hash()
        or artifact.asset_hashes != offline_asset_hashes()
        or target.report_spec.renderer_hash != offline_renderer_hash()
        or target.report_spec.asset_hashes != offline_asset_hashes()
        or target.report_spec.browser_contract_hash != browser_acceptance_contract_hash()
        or target.report_spec.performance_contract_hash != performance_acceptance_contract_hash()
        or artifact.report_identity_spec_binding
        != target.derivation_spec.report_identity_spec_binding
    ):
        return (_issue(rule_id, "report-artifact-manifest.json", "report-binding-mismatch"),)
    return ()


def _authorized_evidence_validation(validation: ValidationResult) -> bool:
    expected_inventory = _expected_evidence_validation_inventory(validation)
    if expected_inventory is None:
        return False
    expected_ids = tuple(rule_id for rule_id, _version in expected_inventory)
    expected_versions = tuple(f"{rule_id}@{version}" for rule_id, version in expected_inventory)
    return bool(
        validation.required_rule_ids == expected_ids
        and validation.executed_rule_ids == expected_ids
        and validation.passed_rule_ids == expected_ids
        and validation.failed_rule_ids == ()
        and validation.not_applicable_rule_ids == ()
        and validation.missing_rule_ids == ()
        and validation.implementation_versions == expected_versions
        and validation.issues == ()
    )


def _authorized_diagnostic_evidence_validation(validation: ValidationResult) -> bool:
    expected_inventory = _expected_evidence_validation_inventory(validation)
    if expected_inventory is None:
        return False
    expected_ids = tuple(rule_id for rule_id, _version in expected_inventory)
    expected_versions = tuple(f"{rule_id}@{version}" for rule_id, version in expected_inventory)
    terminal_ids = (
        *validation.passed_rule_ids,
        *validation.failed_rule_ids,
        *validation.not_applicable_rule_ids,
    )
    return bool(
        validation.required_rule_ids == expected_ids
        and validation.executed_rule_ids == expected_ids
        and len(terminal_ids) == len(set(terminal_ids)) == len(expected_ids)
        and set(terminal_ids) == set(expected_ids)
        and validation.failed_rule_ids
        and validation.missing_rule_ids == ()
        and validation.implementation_versions == expected_versions
        and validation.issues
    )


def _expected_evidence_validation_inventory(
    validation: ValidationResult,
) -> tuple[tuple[str, int], ...] | None:
    if validation.validation_profile_id == NATIVE_EVIDENCE_PROFILE_ID:
        return tuple((rule_id, 1) for rule_id in NATIVE_EVIDENCE_RULE_IDS)
    if validation.validation_profile_id == NATIVE_RUN_EVIDENCE_PROFILE_ID:
        return NATIVE_RUN_EVIDENCE_RULE_INVENTORY
    definition = validation_profile_catalog().get(validation.validation_profile_id)
    if definition is None or definition.profile.stage != ValidationStage.EVIDENCE:
        return None
    return definition.rule_inventory


def _report_schema_inventory(
    model: OfflineReportModel,
    report_kind: str,
) -> tuple[str, ...] | None:
    if isinstance(model, DiagnosticRunReportModel):
        return ("diagnostic_run_report_model@1", "report_artifact_manifest@2")
    if isinstance(model, ExternalHistoricalEvidenceReport):
        return ("external_historical_evidence_report@1", "report_artifact_manifest@2")
    return _REPORT_SCHEMA_INVENTORY_BY_KIND.get(report_kind)


def _fresh_evidence_validation(
    target: Any,
    validation: ValidationResult,
) -> ValidationResult | None:
    try:
        if validation.validation_profile_id == NATIVE_EVIDENCE_PROFILE_ID:
            if not isinstance(target, Path):
                return None
            return validate_native_capsule(target)
        if validation.validation_profile_id == NATIVE_RUN_EVIDENCE_PROFILE_ID:
            if not isinstance(target, NativeRunEvidenceValidationInput):
                return None
            return validate_native_run_evidence(target)
        return validate_catalog_profile(validation.validation_profile_id, target)
    except (OSError, TypeError, ValueError):
        return None


def _model_validation_roots_close(
    model: OfflineReportModel,
    report_spec: ReportIdentitySpec,
    validations: tuple[ValidationResult, ...],
    validation_targets: tuple[Any, ...],
    validation_hashes: tuple[str, ...],
) -> bool:
    if isinstance(model, ExternalHistoricalEvidenceReport):
        if (
            not isinstance(report_spec, ReportSpec)
            or len(validations) != 1
            or len(validation_targets) != 1
            or not isinstance(validation_targets[0], ExternalHistoricalEvidence)
            or model.evidence_validation_result_hash != validation_hashes[0]
        ):
            return False
        validation = validations[0]
        expected_profile_hash = canonical_sha256(
            [
                "oamb-validation-profile-binding-v1",
                validation.validation_profile_id,
                validation.required_rule_ids,
                validation.implementation_versions,
            ]
        )
        if model.evidence_validation_profile_hash != expected_profile_hash:
            return False
        try:
            from oamb.external_evidence.reduce import reduce_external_historical_report

            external_fresh_model = reduce_external_historical_report(
                validation_targets[0],
                validation,
                report_spec=report_spec,
            )
        except (OSError, TypeError, ValueError):
            return False
        return external_fresh_model == model
    if isinstance(model, DiagnosticRunReportModel):
        if len(validations) != 1 or len(validation_targets) != 1:
            return False
        validation = validations[0]
        validation_target = validation_targets[0]
        if (
            model.origin_kind != "native"
            or not isinstance(validation_target, Path)
            or validation.validation_profile_id != NATIVE_EVIDENCE_PROFILE_ID
        ):
            return False
        try:
            manifest = CapsuleManifest.model_validate_json(
                read_regular_file(validation_target / "capsule-manifest.json")
            )
        except (OSError, TypeError, ValueError):
            return False
        expected_profile_hash = canonical_sha256(
            [
                "oamb-validation-profile-binding-v1",
                validation.validation_profile_id,
                validation.required_rule_ids,
                validation.implementation_versions,
            ]
        )
        return bool(
            validation.disposition == ValidationDisposition.INVALID
            and model.evidence_validation_result_hash == validation_hashes[0]
            and model.evidence_validation_profile_hash == expected_profile_hash
            and model.validation_issue_codes
            == tuple(dict.fromkeys(issue.code for issue in validation.issues))
            and model.run_id == manifest.run_id
        )
    if isinstance(model, (RunReportModelV3, ComparisonReportModel, ReleaseReportModel)):
        boundary = model.claim_boundary
        if (
            boundary.status != "pass"
            or boundary.applicable_rule_count
            != sum(len(validation.required_rule_ids) for validation in validations)
            or boundary.executed_rule_count
            != sum(len(validation.executed_rule_ids) for validation in validations)
            or boundary.passed_rule_count
            != sum(len(validation.passed_rule_ids) for validation in validations)
            or boundary.failed_rule_count
            != sum(len(validation.failed_rule_ids) for validation in validations)
            or boundary.not_applicable_rule_count
            != sum(len(validation.not_applicable_rule_ids) for validation in validations)
            or boundary.missing_rule_count
            != sum(len(validation.missing_rule_ids) for validation in validations)
        ):
            return False
    if isinstance(model, RunReportModelV3):
        previews = tuple(
            preview
            for projection in model.record_projections
            for preview in projection.display_previews
        )
        if (
            not isinstance(report_spec, ReportSpec)
            or any(
                preview.shown_bytes > report_spec.preview_max_field_bytes for preview in previews
            )
            or sum(preview.shown_bytes for preview in previews) > report_spec.preview_total_bytes
            or len(validations) != 1
            or model.evidence_validation_result_hash != validation_hashes[0]
        ):
            return False
        validation = validations[0]
        expected_profile_hash = canonical_sha256(
            [
                "oamb-validation-profile-binding-v1",
                validation.validation_profile_id,
                validation.required_rule_ids,
                validation.implementation_versions,
            ]
        )
        if model.evidence_validation_profile_hash != expected_profile_hash:
            return False
        if validation_targets:
            accounting = None
            if isinstance(validation_targets[0], NativeRunEvidenceValidationInput):
                accounting = validation_targets[0].accounting_target.reduction
            elif isinstance(validation_targets[0], Path):
                try:
                    from oamb.reporting.native_reduce import (
                        build_native_accounting_validation_input,
                    )

                    accounting = build_native_accounting_validation_input(
                        validation_targets[0]
                    ).reduction
                except (OSError, TypeError, ValueError):
                    return False
            if accounting is not None and (
                model.claim_boundary.billing_complete != accounting.billing_complete
                or model.claim_boundary.cost_complete != accounting.cost_complete
                or model.measurement_lines != accounting.lines
            ):
                return False
        if model.origin_kind == "native":
            exact_workload = model.workload_id in {
                "lme30-native-smoke-plus-v1",
                "mab65-v1",
            }
            expected_profile_id = (
                NATIVE_RUN_EVIDENCE_PROFILE_ID if exact_workload else NATIVE_EVIDENCE_PROFILE_ID
            )
            validation_target = validation_targets[0]
            capsule_root = (
                validation_target.capsule_root
                if isinstance(validation_target, NativeRunEvidenceValidationInput)
                else validation_target
            )
            if validation.validation_profile_id != expected_profile_id or not isinstance(
                capsule_root, Path
            ):
                return False
            try:
                from oamb.reporting.native_reduce import reduce_native_run_report

                native_fresh_model = reduce_native_run_report(
                    capsule_root,
                    validation,
                    report_spec=report_spec,
                    validation_target=(validation_target if exact_workload else None),
                )
            except (OSError, TypeError, ValueError):
                return False
            return native_fresh_model == model
        return False
    if isinstance(model, (ComparisonReportModel, ReleaseReportModel)):
        return False
    if isinstance(model, PhaseAcceptanceReport):
        if (
            not isinstance(report_spec, AcceptanceReportSpec)
            or len(validations) != 1
            or len(validation_targets) != 1
            or validations[0].validation_profile_id != "oamb-t8-t10-phase-gate-v1"
            or validations[0].required_rule_ids != T10_PHASE_GATE_RULE_IDS
            or not isinstance(validation_targets[0], PhaseGateValidationInput)
        ):
            return False
        target = validation_targets[0]
        gate = target.gate
        bundle = target.bundle
        human = gate.human_record
        return bool(
            human is not None
            and model.evaluation_report_hash
            == report_spec.evaluation_report_hash
            == bundle.report_model_hash
            and model.evaluation_export_validation_hash
            == report_spec.evaluation_export_validation_hash
            == bundle.export_validation_hash
            and model.review_bundle_hash == report_spec.review_bundle_hash == bundle.bundle_id
            and model.phase_gate_hash == report_spec.phase_gate_hash == gate.gate_id
            and model.ai_review_record_hash
            == report_spec.ai_review_record_hash
            == gate.canonical_ai_review_record_hash
            and model.human_review_record_hash
            == report_spec.human_review_record_hash
            == human.human_review_record_id
            and model.phase_id == bundle.phase_id == gate.phase_id
            and model.gate_review_bundle_hash == gate.review_bundle_hash
            and model.review_current == (bundle.bundle_id == gate.review_bundle_hash)
            and model.passed_by_ai == gate.passed_by_ai
            and model.passed_by_human == gate.passed_by_human
            and model.finding_codes == human.finding_codes
            and model.evidence_references == human.evidence_references
        )
    return True


def _safe_html_rule(target: ReportExportInput) -> tuple[ValidationIssue, ...]:
    rule_id = "report.export.safe-html.v1"
    for path in sorted(target.scan_paths):
        if path != "outputs/report.html" and _is_network_active_payload(
            path, target.payloads[path]
        ):
            return (_issue(rule_id, path, "network-active-publication-payload"),)
    html_bytes = target.payloads.get("outputs/report.html", b"")
    try:
        html = html_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return (_issue(rule_id, "outputs/report.html", "report-html-invalid"),)
    required = (
        '<meta name="color-scheme" content="light dark">',
        "default-src 'none'",
        "connect-src 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "font-src 'none'",
    )
    unsafe = bool(
        re.search(r"(?:src|href)=[\"'](?:https?:)?//", html, re.IGNORECASE)
        or re.search(r"\son[a-z]+\s*=", html, re.IGNORECASE)
        or "javascript:" in html.lower()
        or "@import" in html.lower()
    )
    if (
        html_bytes != render_offline_report(target.model)
        or any(item not in html for item in required)
        or unsafe
    ):
        return (_issue(rule_id, "outputs/report.html", "unsafe-or-drifted-report-html"),)
    return ()


def _is_network_active_payload(path: str, content: bytes) -> bool:
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in {".js", ".mjs"}:
        return True
    if suffix not in {".css", ".htm", ".html", ".svg", ".xml"}:
        return False
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        return True
    lowered = text.lower()
    if suffix == ".css":
        return bool(
            "@import" in lowered
            or "javascript:" in lowered
            or re.search(r"url\s*\(\s*['\"]?(?:https?:)?//", text, re.IGNORECASE)
        )
    return bool(
        re.search(r"(?:src|href|xlink:href)\s*=\s*['\"](?:https?:)?//", text, re.IGNORECASE)
        or re.search(r"\son[a-z]+\s*=", text, re.IGNORECASE)
        or "javascript:" in lowered
        or "@import" in lowered
        or re.search(r"<(?:base|embed|iframe|link|object|script)\b", text, re.IGNORECASE)
    )


def _audience_scan_rule(target: ReportExportInput) -> tuple[ValidationIssue, ...]:
    rule_id = "report.export.audience-scan.v1"
    public = target.artifact_manifest.audience == "public"
    generic_public_patterns = (
        rb"sk-[A-Za-z0-9_-]{8,}",
        rb"(?i)(?:api[_-]?key|authorization)\s*[:=]",
        rb"/Users/",
        rb"/home/",
        rb"(?i)(?:localhost|127\.0\.0\.1)",
        rb"OAMB-RESTRICTED",
    )
    bidi_controls = tuple(
        chr(codepoint).encode("utf-8")
        for codepoint in (
            0x200E,
            0x200F,
            0x202A,
            0x202B,
            0x202C,
            0x202D,
            0x202E,
            0x2066,
            0x2067,
            0x2068,
            0x2069,
        )
    )
    for path in sorted(target.scan_paths):
        content = target.payloads[path]
        if any(value and value in content for value in target.forbidden_values) or (
            public
            and (
                any(re.search(pattern, content) for pattern in generic_public_patterns)
                or any(control in content for control in bidi_controls)
            )
        ):
            return (_issue(rule_id, path, "public-sensitive-content"),)
    return ()


def _size_budget_rule(target: ReportExportInput) -> tuple[ValidationIssue, ...]:
    rule_id = "report.export.size-budget.v1"
    html = target.payloads.get("outputs/report.html", b"")
    if len(html) > REPORT_MAX_HTML_BYTES:
        return (_issue(rule_id, "outputs/report.html", "report-html-size-budget-exceeded"),)
    return ()


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


REPORT_EXPORT_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("report.export.payload-closure.v1", 1, _payload_closure_rule),
    ValidationRule("report.export.canonical-bindings.v1", 1, _canonical_bindings_rule),
    ValidationRule("report.export.safe-html.v1", 1, _safe_html_rule),
    ValidationRule("report.export.audience-scan.v1", 1, _audience_scan_rule),
    ValidationRule("report.export.size-budget.v1", 1, _size_budget_rule),
)


__all__ = [
    "ReportExportInput",
    "validate_report_export",
]
