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
from oamb.artifacts.validation.profiles import (
    T8_REPORT_EXPORT_RULE_INVENTORY,
    exact_report_export_profile,
    validation_profile_catalog,
)
from oamb.artifacts.validation.reduction import ComparisonValidationInput
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
from oamb.reporting.public import build_evaluation_model_closure

ReportIdentitySpec = ReportSpec | ReportSpecV2

_EXPORT_SELECTOR_BY_KIND_AND_AUDIENCE = {
    (report_kind, audience): (f"{audience}-{report_kind.replace('_', '-')}-v1", 1)
    for report_kind in ("run", "comparison", "evaluation", "release")
    for audience in ("public", "local")
}
_REPORT_SCHEMA_INVENTORY_BY_KIND = {
    "run": ("run_report_model@3", "report_artifact_manifest@2"),
    "comparison": ("comparison_report_model@1", "report_artifact_manifest@2"),
    "evaluation": ("evaluation_report_model@1", "report_artifact_manifest@3"),
    "release": ("release_report_model@1", "report_artifact_manifest@2"),
}


@dataclass(frozen=True, slots=True)
class ReportExportInput:
    payloads: Mapping[str, bytes]
    scan_paths: frozenset[str]
    model: OfflineReportModel
    artifact_manifest: ReportArtifactManifestV2 | ReportArtifactManifestV3
    report_spec: ReportIdentitySpec
    derivation_spec: DerivationSpecV2 | DerivationSpecV3
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    evidence_validations: tuple[ValidationResult, ...]
    evidence_validation_targets: tuple[Any, ...]
    forbidden_values: tuple[bytes, ...]


def validate_report_export(
    target: ReportExportInput,
) -> ValidationResult:
    report_kind = target.report_spec.report_kind
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
    required = {
        "derivation-spec.json",
        "source-roots.json",
        "input-specs/report-spec.json",
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
    spec_path = "input-specs/report-spec.json"
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
    model_spec_hash = target.model.report_spec_hash
    artifact = target.artifact_manifest
    evaluation = isinstance(target.model, EvaluationReportModel)
    expected_evaluation_closure = None
    if isinstance(target.model, EvaluationReportModel):
        expected_evaluation_closure = build_evaluation_model_closure(target.model)
    model_validation_roots_close = _model_validation_roots_close(
        target.model,
        target.report_spec,
        target.ordered_source_bindings,
        target.evidence_validations,
        target.evidence_validation_targets,
        expected_validation_hashes,
    )
    report_kind = artifact.report_kind
    expected_selector = _EXPORT_SELECTOR_BY_KIND_AND_AUDIENCE.get(
        (report_kind, target.report_spec.audience)
    )
    expected_spec_kind = "benchmark_report"
    expected_spec_id = target.report_spec.report_spec_id
    identity_binding = artifact.report_identity_spec_binding
    version_chain_closed = (
        isinstance(target.report_spec, ReportSpecV2)
        and isinstance(target.derivation_spec, DerivationSpecV3)
        and isinstance(artifact, ReportArtifactManifestV3)
        and identity_binding.schema_version == 2
        and target.derivation_spec.report_identity_spec_binding == identity_binding
        and target.derivation_spec.evaluation_model_closure == expected_evaluation_closure
        and artifact.evaluation_model_closure == expected_evaluation_closure
        and artifact.export_profile_hash
        == canonical_sha256(
            exact_report_export_profile(
                report_kind="evaluation",
                audience=target.report_spec.audience,
            )
        )
        if evaluation
        else isinstance(target.derivation_spec, DerivationSpecV2)
        and isinstance(artifact, ReportArtifactManifestV2)
        and not isinstance(target.report_spec, ReportSpecV2)
        and identity_binding.schema_version == 1
        and target.derivation_spec.report_identity_spec_binding == identity_binding
    )
    if (
        not exact_bytes
        or not version_chain_closed
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
        or identity_binding.spec_kind != expected_spec_kind
        or identity_binding.spec_schema_name != target.report_spec.schema_name
        or identity_binding.spec_schema_version != target.report_spec.schema_version
        or identity_binding.spec_id != expected_spec_id
        or identity_binding.spec_hash != spec_hash
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
    source_bindings: tuple[SourceEvidenceBinding, ...],
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
            # Native capsule paths are re-reduced below. Exact model equality
            # includes accounting lines and completeness without imposing the
            # older standalone accounting profile's one-usage-per-attempt rule.
            if accounting is not None and (
                model.claim_boundary.billing_complete != accounting.billing_complete
                or model.claim_boundary.cost_complete != accounting.cost_complete
                or model.measurement_lines != accounting.lines
            ):
                return False
        if model.origin_kind == "native":
            composite_validation = (
                validation.validation_profile_id == NATIVE_RUN_EVIDENCE_PROFILE_ID
            )
            expected_profile_id = (
                NATIVE_RUN_EVIDENCE_PROFILE_ID
                if composite_validation
                else NATIVE_EVIDENCE_PROFILE_ID
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
                    validation_target=(validation_target if composite_validation else None),
                )
            except (OSError, TypeError, ValueError):
                return False
            return native_fresh_model == model
        return False
    if isinstance(model, EvaluationReportModel):
        if not isinstance(report_spec, ReportSpecV2):
            return False
        try:
            closure = build_evaluation_model_closure(model)
        except (TypeError, ValueError):
            return False
        return bool(
            model.report_spec_hash == canonical_sha256(report_spec)
            and tuple(item.report_id for item in model.ordered_run_models)
            == tuple(item.report_model_id for item in closure.ordered_run_entries)
            and tuple(item.report_id for item in model.eligible_comparison_models)
            == tuple(item.report_model_id for item in closure.eligible_comparison_entries)
            and _evaluation_sources_and_coverage_close(
                model,
                source_bindings,
                validations,
                validation_targets,
                validation_hashes,
            )
        )
    if isinstance(model, (ComparisonReportModel, ReleaseReportModel)):
        return False
    return True


def _evaluation_sources_and_coverage_close(
    model: EvaluationReportModel,
    source_bindings: tuple[SourceEvidenceBinding, ...],
    validations: tuple[ValidationResult, ...],
    validation_targets: tuple[Any, ...],
    validation_hashes: tuple[str, ...],
) -> bool:
    run_count = len(model.ordered_run_models)
    comparisons = model.eligible_comparison_models
    expected_source_count = run_count + len(comparisons)
    if not (
        run_count == 4
        and len(source_bindings)
        == len(validations)
        == len(validation_targets)
        == len(validation_hashes)
        == expected_source_count
    ):
        return False

    run_sources = tuple(run.ordered_source_bindings[0] for run in model.ordered_run_models)
    if source_bindings[:run_count] != run_sources:
        return False
    for run, validation, validation_hash in zip(
        model.ordered_run_models,
        validations[:run_count],
        validation_hashes[:run_count],
        strict=True,
    ):
        expected_profile_hash = canonical_sha256(
            [
                "oamb-validation-profile-binding-v1",
                validation.validation_profile_id,
                validation.required_rule_ids,
                validation.implementation_versions,
            ]
        )
        if (
            run.evidence_validation_profile_hash != expected_profile_hash
            or run.evidence_validation_result_hash != validation_hash
        ):
            return False

    run_index_by_source = {source: index for index, source in enumerate(run_sources)}
    if len(run_index_by_source) != run_count:
        return False
    for offset, comparison in enumerate(comparisons, run_count):
        target = validation_targets[offset]
        source = source_bindings[offset]
        validation = validations[offset]
        try:
            left_index = run_index_by_source[comparison.ordered_source_bindings[0]]
            right_index = run_index_by_source[comparison.ordered_source_bindings[1]]
        except KeyError:
            return False
        left_run = model.ordered_run_models[left_index]
        right_run = model.ordered_run_models[right_index]
        if (
            left_index == right_index
            or comparison.ordered_evidence_validation_hashes
            != (
                left_run.evidence_validation_result_hash,
                right_run.evidence_validation_result_hash,
            )
            or comparison.left_run_report_hash != canonical_sha256(left_run)
            or comparison.right_run_report_hash != canonical_sha256(right_run)
            or not isinstance(target, ComparisonValidationInput)
            or target.report != comparison.comparison
            or target.left.run_id != left_run.summary.run_id
            or target.right.run_id != right_run.summary.run_id
            or source.source_root_hash != validation.target_hash
            or source.validation_result_hash != validation_hashes[offset]
        ):
            return False

    case_occurrence_ids: list[str] = []
    case_manifest_entry_ids: list[str] = []
    for run in model.ordered_run_models:
        case_projections = tuple(
            projection for projection in run.record_projections if projection.axis == "case"
        )
        if tuple(item.record_id for item in case_projections) != run.case_occurrence_ids:
            return False
        for projection in case_projections:
            case_manifest_entry_id = dict(projection.detail_items).get("case_manifest_entry_id")
            if (
                case_manifest_entry_id is None
                or len(case_manifest_entry_id) != 64
                or any(character not in "0123456789abcdef" for character in case_manifest_entry_id)
            ):
                return False
            case_manifest_entry_ids.append(case_manifest_entry_id)
        case_occurrence_ids.extend(run.case_occurrence_ids)
    return bool(
        len(case_occurrence_ids) == len(set(case_occurrence_ids))
        and model.system_result_count == len(case_occurrence_ids)
        and model.unique_case_count == len(set(case_manifest_entry_ids))
    )


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
    # This exact renderer escapes markup in its inert JSON data element.
    # Historical code examples there are data, not active HTML attributes.
    active_html = re.sub(
        r'<script type="application/json" id="oamb-report-data">.*?</script>',
        "",
        html,
        count=1,
        flags=re.DOTALL,
    )
    unsafe = bool(
        re.search(r"(?:src|href)=[\"'](?:https?:)?//", active_html, re.IGNORECASE)
        or re.search(r"\son[a-z]+\s*=", active_html, re.IGNORECASE)
        or "javascript:" in active_html.lower()
        or "@import" in active_html.lower()
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
