"""Closed post-render validation for deterministic offline fake reports."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

from pydantic import TypeAdapter

from oamb.artifacts.validation.profiles import fake_export_profile
from oamb.artifacts.validation.registry import RuleRegistry, ValidationRule
from oamb.contracts.evidence import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.reporting import (
    ReportArtifactManifest,
    RunReportModelV2,
    RunSummaryV2,
)
from oamb.contracts.specifications import (
    DerivationSpec,
    SourceEvidenceBinding,
    SourceEvidenceKind,
    ValidationProfile,
    ValidationStage,
)
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.renderer import (
    fake_report_renderer_hash,
    load_fake_report_css,
    render_fake_report_html,
)

from .claims import fake_report_id, fake_report_limitations

FAKE_EXPORT_REQUIRED_PAYLOADS = frozenset(
    {
        "derivation-spec.json",
        "source-roots.json",
        "input-specs/report-spec.json",
        "evidence-validation.json",
        "outputs/summaries/run-summary.json",
        "outputs/report-model.json",
        "outputs/report.html",
        "report-artifact-manifest.json",
    }
)
FAKE_EXPORT_RULE_IDS = (
    "fake.export.payload-closure.v1",
    "fake.export.safe-html.v1",
    "fake.export.report-binding.v1",
)


class ExportValidationError(ValueError):
    def __init__(self, result: ValidationResult) -> None:
        super().__init__("fake report export validation failed")
        self.result = result


def validate_fake_export(
    payloads: Mapping[str, bytes],
    *,
    profile: ValidationProfile | None = None,
    registry: RuleRegistry | None = None,
) -> ValidationResult:
    selected_profile = profile or fake_export_profile()
    selected_registry = registry or fake_export_registry()
    target_hash = canonical_sha256(
        [
            "oamb-fake-publication-payload-v1",
            tuple(
                (path, hashlib.sha256(content).hexdigest())
                for path, content in sorted(payloads.items())
            ),
        ]
    )
    required_ids = tuple(requirement.rule_id for requirement in selected_profile.required_rules)
    expected_inventory_hash = canonical_sha256(
        [
            "oamb-required-rule-inventory-v1",
            tuple(
                (requirement.rule_id, requirement.minimum_version)
                for requirement in selected_profile.required_rules
            ),
        ]
    )
    if selected_profile.stage != ValidationStage.EXPORT:
        return _profile_failure(
            selected_profile,
            target_hash,
            required_ids,
            "validation.stage.v1",
            "wrong-validation-stage",
        )
    if selected_profile.required_rule_inventory_hash != expected_inventory_hash:
        return _profile_failure(
            selected_profile,
            target_hash,
            required_ids,
            "validation.profile-inventory.v1",
            "profile-inventory-mismatch",
        )
    if required_ids != FAKE_EXPORT_RULE_IDS:
        return _profile_failure(
            selected_profile,
            target_hash,
            required_ids,
            "validation.profile-coverage.v1",
            "profile-rule-coverage-mismatch",
        )
    executed: list[str] = []
    passed: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    issues: list[ValidationIssue] = []
    implementation_versions: list[str] = []
    for requirement in selected_profile.required_rules:
        rule = selected_registry.compatible(
            requirement.rule_id,
            requirement.minimum_version,
        )
        if rule is None:
            missing.append(requirement.rule_id)
            continue
        executed.append(rule.rule_id)
        implementation_versions.append(f"{rule.rule_id}@{rule.version}")
        rule_issues = rule.evaluate(payloads)
        if rule_issues:
            failed.append(rule.rule_id)
            issues.extend(rule_issues)
        else:
            passed.append(rule.rule_id)
    return ValidationResult(
        validation_profile_id=selected_profile.profile_id,
        target_hash=target_hash,
        disposition=(
            ValidationDisposition.VALIDATED
            if not failed and not missing
            else ValidationDisposition.INVALID
        ),
        required_rule_ids=required_ids,
        executed_rule_ids=tuple(executed),
        passed_rule_ids=tuple(passed),
        failed_rule_ids=tuple(failed),
        not_applicable_rule_ids=(),
        missing_rule_ids=tuple(missing),
        implementation_versions=tuple(implementation_versions),
        issues=tuple(issues),
    )


def fake_export_registry(*, exclude: set[str] | None = None) -> RuleRegistry:
    omitted = exclude or set()
    registry = RuleRegistry()
    for rule in FAKE_EXPORT_RULES:
        if rule.rule_id not in omitted:
            registry.register(rule)
    return registry


def _issue(rule_id: str, path: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=path,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


def _profile_failure(
    profile: ValidationProfile,
    target_hash: str,
    required_ids: tuple[str, ...],
    rule_id: str,
    code: str,
) -> ValidationResult:
    return ValidationResult(
        validation_profile_id=profile.profile_id,
        target_hash=target_hash,
        disposition=ValidationDisposition.INVALID,
        required_rule_ids=required_ids,
        executed_rule_ids=(),
        passed_rule_ids=(),
        failed_rule_ids=(rule_id,),
        not_applicable_rule_ids=(),
        missing_rule_ids=required_ids,
        implementation_versions=(),
        issues=(_issue(rule_id, profile.profile_id, code),),
    )


def _payload_closure_rule(payloads: Mapping[str, bytes]) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.export.payload-closure.v1"
    if set(payloads) != FAKE_EXPORT_REQUIRED_PAYLOADS:
        return (_issue(rule_id, "publication-payload", "payload-inventory-mismatch"),)
    if any(not content for content in payloads.values()):
        return (_issue(rule_id, "publication-payload", "empty-publication-payload"),)
    return ()


def _safe_html_rule(payloads: Mapping[str, bytes]) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.export.safe-html.v1"
    try:
        html = payloads["outputs/report.html"].decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        return (_issue(rule_id, "outputs/report.html", "html-unreadable"),)
    required = (
        '<meta name="color-scheme" content="light dark">',
        "Content-Security-Policy",
        "connect-src 'none'",
        "default-src 'none'",
    )
    forbidden_patterns = (
        r"https?://",
        r"javascript:",
        r"\son[a-z]+=",
        r"<(?:img|link|iframe|object)\b",
        r"@import\b",
        r"url\(",
    )
    if any(item not in html for item in required) or any(
        re.search(pattern, html, re.IGNORECASE) for pattern in forbidden_patterns
    ):
        return (_issue(rule_id, "outputs/report.html", "unsafe-or-network-active-html"),)
    if html.count('<script type="application/json" id="oamb-report-data">') != 1:
        return (_issue(rule_id, "outputs/report.html", "report-data-script-mismatch"),)
    if html.lower().count("<script") != 1 or html.lower().count("</script>") != 1:
        return (_issue(rule_id, "outputs/report.html", "executable-or-injected-script"),)
    return ()


def _report_binding_rule(payloads: Mapping[str, bytes]) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.export.report-binding.v1"
    try:
        report_model = RunReportModelV2.model_validate_json(payloads["outputs/report-model.json"])
        summary = RunSummaryV2.model_validate_json(payloads["outputs/summaries/run-summary.json"])
        validation = ValidationResult.model_validate_json(payloads["evidence-validation.json"])
        artifact = ReportArtifactManifest.model_validate_json(
            payloads["report-artifact-manifest.json"]
        )
        derivation = DerivationSpec.model_validate_json(payloads["derivation-spec.json"])
        source_roots = TypeAdapter(tuple[SourceEvidenceBinding, ...]).validate_json(
            payloads["source-roots.json"]
        )
        report_spec = json.loads(payloads["input-specs/report-spec.json"])
        if not isinstance(report_spec, dict):
            raise ValueError("report source/spec payload has the wrong shape")
    except Exception:
        return (_issue(rule_id, "report-artifact-manifest.json", "report-contract-invalid"),)
    expected_report_spec_keys = {
        "schema_name",
        "schema_version",
        "report_kind",
        "audience",
        "diagnostic",
        "renderer_hash",
    }
    diagnostic = report_spec.get("diagnostic")
    if (
        set(report_spec) != expected_report_spec_keys
        or report_spec.get("schema_name") != "fake_report_spec"
        or report_spec.get("schema_version") != 1
        or report_spec.get("report_kind") != "run"
        or report_spec.get("audience") not in {"public", "local"}
        or type(diagnostic) is not bool
    ):
        return (_issue(rule_id, "input-specs/report-spec.json", "report-spec-mismatch"),)
    assert isinstance(diagnostic, bool)
    if (diagnostic and validation.disposition == ValidationDisposition.VALIDATED) or (
        not diagnostic and validation.disposition != ValidationDisposition.VALIDATED
    ):
        return (_issue(rule_id, "evidence-validation.json", "report-posture-mismatch"),)
    evidence_hash = canonical_sha256(validation)
    expected_limitations = fake_report_limitations(validation, diagnostic=diagnostic)
    expected_report_id = fake_report_id(
        source_manifest_hash=report_model.source_manifest_hash,
        evidence_validation_hash=evidence_hash,
        summary=summary,
        audience=artifact.audience,
        diagnostic=diagnostic,
        limitations=expected_limitations,
    )
    if (
        report_model.limitations != expected_limitations
        or report_model.report_id != expected_report_id
    ):
        return (_issue(rule_id, "outputs/report-model.json", "report-identity-mismatch"),)
    expected_model_hash = hashlib.sha256(payloads["outputs/report-model.json"]).hexdigest()
    css = load_fake_report_css()
    css_hash = hashlib.sha256(css).hexdigest()
    renderer_hash = fake_report_renderer_hash(css)
    if (
        payloads["outputs/report-model.json"] != canonical_json_bytes(report_model)
        or payloads["outputs/summaries/run-summary.json"] != canonical_json_bytes(summary)
        or payloads["evidence-validation.json"] != canonical_json_bytes(validation)
        or payloads["derivation-spec.json"] != canonical_json_bytes(derivation)
        or payloads["source-roots.json"] != canonical_json_bytes(source_roots)
        or payloads["input-specs/report-spec.json"] != canonical_json_bytes(report_spec)
        or payloads["report-artifact-manifest.json"] != canonical_json_bytes(artifact)
    ):
        return (_issue(rule_id, "publication-payload", "noncanonical-report-payload"),)
    if report_model.summary != summary:
        return (_issue(rule_id, "outputs/summaries/run-summary.json", "summary-binding-mismatch"),)
    if len(source_roots) != 1:
        return (_issue(rule_id, "source-roots.json", "source-root-count-mismatch"),)
    source_binding = source_roots[0]
    expected_binding_fields = {
        "source_kind": SourceEvidenceKind.RUN,
        "source_identity": summary.run_id,
        "source_root_hash": report_model.source_manifest_hash,
        "validation_result_hash": evidence_hash,
        "source_schema_versions": ("capsule_manifest@1",),
    }
    expected_binding_id = canonical_sha256(
        ["oamb-source-evidence-binding-v1", expected_binding_fields]
    )
    if (
        source_binding.binding_id != expected_binding_id
        or source_binding.source_kind != SourceEvidenceKind.RUN
        or source_binding.source_identity != summary.run_id
        or source_binding.source_root_hash != report_model.source_manifest_hash
        or source_binding.validation_result_hash != evidence_hash
        or source_binding.source_schema_versions != ("capsule_manifest@1",)
    ):
        return (_issue(rule_id, "source-roots.json", "source-root-binding-mismatch"),)
    expected_root_hash = canonical_sha256(
        ["oamb-ordered-source-roots-v1", (report_model.source_manifest_hash,)]
    )
    expected_reducer_inputs = (
        canonical_sha256(summary),
        renderer_hash,
        css_hash,
    )
    expected_transform_hash = canonical_sha256(
        [
            "oamb-fake-report-transform-v1",
            "diagnostic" if diagnostic else "normal",
        ]
    )
    expected_derivation_kind = "diagnostic_run_report" if diagnostic else "run_report"
    expected_derivation_fields = {
        "derivation_kind": expected_derivation_kind,
        "ordered_source_bindings": source_roots,
        "ordered_source_root_hash": expected_root_hash,
        "transform_spec_hash": expected_transform_hash,
        "report_spec_hash": canonical_sha256(report_spec),
        "reducer_and_renderer_input_hashes": expected_reducer_inputs,
    }
    if (
        derivation.derivation_kind != expected_derivation_kind
        or derivation.ordered_source_bindings != source_roots
        or derivation.ordered_source_root_hash != expected_root_hash
        or derivation.transform_spec_hash != expected_transform_hash
        or derivation.report_spec_hash != canonical_sha256(report_spec)
        or derivation.reducer_and_renderer_input_hashes != expected_reducer_inputs
        or derivation.derivation_input_hash
        != canonical_sha256(["oamb-derivation-input-v1", expected_derivation_fields])
    ):
        return (_issue(rule_id, "derivation-spec.json", "derivation-binding-mismatch"),)
    if (
        artifact.report_id != report_model.report_id
        or artifact.ordered_source_root_hashes != (report_model.source_manifest_hash,)
        or artifact.evidence_validation_hash != evidence_hash
        or artifact.report_model_hash != expected_model_hash
        or artifact.renderer_hash != renderer_hash
        or artifact.asset_hashes != (css_hash,)
        or artifact.audience != report_spec["audience"]
        or artifact.limitations != report_model.limitations
        or report_model.evidence_validation_hash != evidence_hash
        or validation.target_hash != report_model.source_manifest_hash
        or report_spec["renderer_hash"] != renderer_hash
    ):
        return (_issue(rule_id, "report-artifact-manifest.json", "report-binding-mismatch"),)
    expected_html = render_fake_report_html(report_model, css, diagnostic=diagnostic)
    if payloads["outputs/report.html"] != expected_html:
        return (_issue(rule_id, "outputs/report.html", "rendered-html-mismatch"),)
    return ()


FAKE_EXPORT_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("fake.export.payload-closure.v1", 1, _payload_closure_rule),
    ValidationRule("fake.export.safe-html.v1", 1, _safe_html_rule),
    ValidationRule("fake.export.report-binding.v1", 1, _report_binding_rule),
)


__all__ = [
    "ExportValidationError",
    "FAKE_EXPORT_REQUIRED_PAYLOADS",
    "FAKE_EXPORT_RULE_IDS",
    "fake_export_registry",
    "validate_fake_export",
]
