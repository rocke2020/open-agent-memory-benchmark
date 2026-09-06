"""Fail-closed structural and provider-service evidence validation."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict

from oamb.contracts.accounting import (
    CostMeasurementSpec,
    CostRecord,
    ResourceUsageRecord,
    TokenUsageRecordV2,
)
from oamb.contracts.base import NonEmptyStr, Sha256
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptReceiptRecord,
    AttemptRecordV2,
    BudgetReservationRecord,
    ModelReadinessOccurrenceRecord,
    ModelReadinessOccurrenceState,
    OccurrenceClaimRecord,
    ProviderServiceEvidenceManifest,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256, sha256_identity
from oamb.contracts.schema import parse_contract
from oamb.contracts.specifications import (
    BudgetScopeKindV2,
    BudgetSpecV2,
    ModelRoleBindingV2,
    ProviderGateStatus,
    ProviderRuntimeProfileAttestation,
    RoleBindingStatus,
    TransportProfile,
    ValidationProfile,
    ValidationStage,
)
from oamb.contracts.states import (
    AttemptOutcome,
    CaseState,
    EntityKind,
    IngestionPlanState,
    ResumeDisposition,
    RunState,
    StateValue,
    TransitionEvent,
    ValidationDisposition,
    apply_transition,
)

from .registry import RuleRegistry, ValidationRule


class StructuralModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _require_json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_json_value(item)
        return value
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON maps require string keys")
        for item in value.values():
            _require_json_value(item)
        return value
    raise ValueError("value is not JSON-compatible")


JsonContractValue = Annotated[Any, BeforeValidator(_require_json_value)]


class ArtifactDocument(StructuralModel):
    record_id: NonEmptyStr
    record_kind: NonEmptyStr
    document: dict[str, JsonContractValue]


class IdentityClaim(StructuralModel):
    evidence_ref: NonEmptyStr
    identity_prefix: NonEmptyStr
    identity_parts: tuple[JsonContractValue, ...]
    actual_id: Sha256


class ReferenceClaim(StructuralModel):
    evidence_ref: NonEmptyStr
    target_record_id: NonEmptyStr
    expected_record_kind: NonEmptyStr


class CountClaim(StructuralModel):
    evidence_ref: NonEmptyStr
    declared_count: int
    member_record_ids: tuple[NonEmptyStr, ...]


class HashKind(StrEnum):
    CANONICAL_JSON = "canonical_json"
    RAW_BYTES = "raw_bytes"


class HashClaim(StructuralModel):
    evidence_ref: NonEmptyStr
    hash_kind: HashKind = HashKind.CANONICAL_JSON
    value: bytes | JsonContractValue
    actual_sha256: Sha256


class TransitionClaim(StructuralModel):
    evidence_ref: NonEmptyStr
    entity_kind: EntityKind
    current_state: str
    event: TransitionEvent
    claimed_next_state: str
    resume_disposition: ResumeDisposition | None = None


class ControlFieldClaim(StructuralModel):
    evidence_ref: NonEmptyStr
    control_values: tuple[str, ...]
    forbidden_values: tuple[str, ...]


class RuleCoverageClaim(StructuralModel):
    rule_id: NonEmptyStr
    target_evidence_refs: tuple[NonEmptyStr, ...]
    not_applicable_reason: str | None


class StructuralValidationInput(StructuralModel):
    documents: tuple[ArtifactDocument, ...]
    identity_claims: tuple[IdentityClaim, ...]
    reference_claims: tuple[ReferenceClaim, ...]
    count_claims: tuple[CountClaim, ...]
    hash_claims: tuple[HashClaim, ...]
    transition_claims: tuple[TransitionClaim, ...]
    control_field_claims: tuple[ControlFieldClaim, ...]
    rule_coverage_claims: tuple[RuleCoverageClaim, ...]


class EvidenceNotValidatedError(ValueError):
    pass


def _issue(rule_id: str, evidence_ref: str, code: str, pointer: str | None) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=pointer,
        remediation_code=f"repair-{code}",
    )


def _preflight_failure(
    *,
    profile: ValidationProfile,
    target_hash: str,
    required_ids: tuple[str, ...],
    rule_id: str,
    code: str,
    pointer: str,
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
        issues=(_issue(rule_id, profile.profile_id, code, pointer),),
    )


def _schema_rule(value: StructuralValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "structural.schema.v1"
    issues: list[ValidationIssue] = []
    for document in value.documents:
        try:
            parsed = parse_contract(document.document)
        except Exception:
            issues.append(_issue(rule_id, document.record_id, "schema-invalid", "/document"))
            continue
        schema_name = type(parsed).model_fields["schema_name"].default
        if schema_name != document.record_kind:
            issues.append(
                _issue(rule_id, document.record_id, "schema-kind-mismatch", "/record_kind")
            )
    return tuple(issues)


def _identity_rule(value: StructuralValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "structural.identity.v1"
    documents = {document.record_id: document for document in value.documents}
    issues: list[ValidationIssue] = []
    claims_by_ref: dict[str, list[IdentityClaim]] = {}
    for claim in value.identity_claims:
        claims_by_ref.setdefault(claim.evidence_ref, []).append(claim)
        document = documents.get(claim.evidence_ref)
        expected_prefix = claim.identity_prefix
        expected_parts = claim.identity_parts
        expected_actual_id = claim.actual_id
        if document is not None and document.record_kind == "logical_context_manifest_entry":
            if claim.identity_prefix == "oamb-context-content-v1":
                expected_prefix = "oamb-context-content-v1"
                if len(claim.identity_parts) == 4:
                    expected_parts = (
                        claim.identity_parts[0],
                        claim.identity_parts[1],
                        claim.identity_parts[2],
                        document.document["context_bytes_sha256"],
                    )
                expected_actual_id = document.document["context_content_id"]
            elif claim.identity_prefix == "oamb-context-manifest-entry-v1":
                expected_prefix = "oamb-context-manifest-entry-v1"
                if len(claim.identity_parts) == 4:
                    expected_parts = (
                        claim.identity_parts[0],
                        document.document["source_file_sha256"],
                        document.document["source_row_number_1_indexed"],
                        document.document["context_content_id"],
                    )
                expected_actual_id = document.document["context_manifest_entry_id"]
            else:
                expected_prefix = ""
                expected_parts = ()
                expected_actual_id = document.document["context_manifest_entry_id"]
        elif document is not None and document.record_kind == "case_manifest_entry":
            expected_prefix = "oamb-case-manifest-entry-v1"
            expected_parts = (
                document.document["context_manifest_entry_id"],
                document.document["source_question_number_1_indexed"],
                document.document["question_bytes_sha256"],
                document.document["raw_question_id"],
            )
            expected_actual_id = document.document["case_manifest_entry_id"]
        elif document is not None and document.record_kind == "ingestion_plan_manifest":
            if claim.identity_prefix == "oamb-ingestion-plan-manifest-entry-v1":
                expected_prefix = "oamb-ingestion-plan-manifest-entry-v1"
                expected_parts = (
                    document.document["workload_id"],
                    tuple(document.document["ordered_member_context_manifest_entry_ids"]),
                )
                expected_actual_id = document.document["plan_manifest_entry_id"]
            elif claim.identity_prefix == "oamb-ingestion-payload-v1":
                expected_prefix = "oamb-ingestion-payload-v1"
                expected_parts = (tuple(document.document["ordered_source_unit_bytes_sha256"]),)
                expected_actual_id = document.document["ingestion_payload_hash"]
            elif claim.identity_prefix == "oamb-ingestion-plan-v1":
                expected_prefix = "oamb-ingestion-plan-v1"
                expected_parts = (
                    document.document["plan_manifest_entry_id"],
                    document.document["ingestion_payload_hash"],
                )
                expected_actual_id = document.document["ingestion_plan_id"]
            else:
                expected_prefix = ""
                expected_parts = ()
                expected_actual_id = document.document["ingestion_plan_id"]
        try:
            computed_identity = sha256_identity(expected_prefix, expected_parts)
        except (TypeError, ValueError):
            computed_identity = None
        valid = (
            claim.identity_prefix == expected_prefix
            and claim.identity_parts == expected_parts
            and claim.actual_id == expected_actual_id
            and computed_identity == expected_actual_id
        )
        if not valid:
            issues.append(_issue(rule_id, claim.evidence_ref, "identity-mismatch", "/actual_id"))
    required_prefixes_by_kind = {
        "logical_context_manifest_entry": {
            "oamb-context-content-v1",
            "oamb-context-manifest-entry-v1",
        },
        "case_manifest_entry": {"oamb-case-manifest-entry-v1"},
        "ingestion_plan_manifest": {
            "oamb-ingestion-plan-manifest-entry-v1",
            "oamb-ingestion-payload-v1",
            "oamb-ingestion-plan-v1",
        },
    }
    for document in value.documents:
        required_prefixes = required_prefixes_by_kind.get(document.record_kind)
        if required_prefixes is None:
            continue
        actual_prefixes = {
            claim.identity_prefix for claim in claims_by_ref.get(document.record_id, [])
        }
        if actual_prefixes != required_prefixes:
            issues.append(
                _issue(
                    rule_id, document.record_id, "identity-inventory-mismatch", "/identity_claims"
                )
            )
    return tuple(issues)


def _reference_rule(value: StructuralValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "structural.reference.v1"
    record_kinds = {document.record_id: document.record_kind for document in value.documents}
    issues: list[ValidationIssue] = []
    if len(record_kinds) != len(value.documents):
        issues.append(_issue(rule_id, "documents", "duplicate-record-id", "/documents"))
    for claim in value.reference_claims:
        if record_kinds.get(claim.target_record_id) != claim.expected_record_kind:
            issues.append(
                _issue(rule_id, claim.evidence_ref, "reference-unresolved", "/target_record_id")
            )
    claims_by_ref: dict[str, list[ReferenceClaim]] = {}
    for claim in value.reference_claims:
        claims_by_ref.setdefault(claim.evidence_ref, []).append(claim)
    identity_by_ref: dict[str, list[IdentityClaim]] = {}
    for identity_claim in value.identity_claims:
        identity_by_ref.setdefault(identity_claim.evidence_ref, []).append(identity_claim)
    documents_by_id = {document.record_id: document for document in value.documents}
    for document in value.documents:
        claims = claims_by_ref.get(document.record_id, [])
        if document.record_kind == "logical_context_manifest_entry":
            identities = identity_by_ref.get(document.record_id, [])
            content_identity = next(
                (
                    claim
                    for claim in identities
                    if claim.identity_prefix == "oamb-context-content-v1"
                ),
                None,
            )
            manifest_identity = next(
                (
                    claim
                    for claim in identities
                    if claim.identity_prefix == "oamb-context-manifest-entry-v1"
                ),
                None,
            )
            valid = len(claims) == 1 and claims[0].expected_record_kind == "dataset_manifest"
            target = documents_by_id.get(claims[0].target_record_id) if valid else None
            source_file_matches = False
            if (
                target is not None
                and content_identity is not None
                and len(content_identity.identity_parts) == 4
            ):
                source_file_matches = any(
                    source_file.get("relative_path") == content_identity.identity_parts[2]
                    and source_file.get("sha256") == document.document["source_file_sha256"]
                    for source_file in target.document.get("source_files", [])
                )
            valid = bool(
                valid
                and target is not None
                and content_identity is not None
                and manifest_identity is not None
                and len(content_identity.identity_parts) == 4
                and len(manifest_identity.identity_parts) == 4
                and target.document.get("manifest_hash") == manifest_identity.identity_parts[0]
                and target.document.get("revision") == content_identity.identity_parts[0]
                and target.document.get("split") == content_identity.identity_parts[1]
                and content_identity.identity_parts[3] == document.document["context_bytes_sha256"]
                and source_file_matches
            )
            if not valid:
                issues.append(
                    _issue(
                        rule_id,
                        document.record_id,
                        "reference-shape-invalid",
                        "/reference_claims",
                    )
                )
        elif document.record_kind == "case_manifest_entry":
            valid = (
                len(claims) == 1
                and claims[0].expected_record_kind == "logical_context_manifest_entry"
                and claims[0].target_record_id == document.document["context_manifest_entry_id"]
            )
            if not valid:
                issues.append(
                    _issue(
                        rule_id,
                        document.record_id,
                        "reference-shape-invalid",
                        "/reference_claims",
                    )
                )
        elif document.record_kind == "ingestion_plan_manifest":
            member_targets = tuple(document.document["ordered_member_context_manifest_entry_ids"])
            case_targets = tuple(document.document["ordered_case_manifest_entry_ids"])
            claims_by_target: dict[str, list[ReferenceClaim]] = {}
            for claim in claims:
                claims_by_target.setdefault(claim.target_record_id, []).append(claim)
            valid = (
                len(set(member_targets)) == len(member_targets)
                and len(set(case_targets)) == len(case_targets)
                and not set(member_targets) & set(case_targets)
                and len(claims) == len(member_targets) + len(case_targets)
                and all(
                    len(claims_by_target.get(target_id, [])) == 1
                    and claims_by_target[target_id][0].expected_record_kind
                    == "logical_context_manifest_entry"
                    for target_id in member_targets
                )
                and all(
                    len(claims_by_target.get(target_id, [])) == 1
                    and claims_by_target[target_id][0].expected_record_kind == "case_manifest_entry"
                    for target_id in case_targets
                )
            )
            if not valid:
                issues.append(
                    _issue(
                        rule_id,
                        document.record_id,
                        "reference-shape-invalid",
                        "/reference_claims",
                    )
                )
    return tuple(issues)


def _count_rule(value: StructuralValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "structural.count.v1"
    return tuple(
        _issue(rule_id, claim.evidence_ref, "count-mismatch", "/declared_count")
        for claim in value.count_claims
        if claim.declared_count < 0 or claim.declared_count != len(claim.member_record_ids)
    )


def _hash_rule(value: StructuralValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "structural.hash.v1"
    documents = {document.record_id: document for document in value.documents}
    issues: list[ValidationIssue] = []
    hash_field_by_kind = {
        "logical_context_manifest_entry": "context_bytes_sha256",
        "case_manifest_entry": "question_bytes_sha256",
        "ingestion_plan_manifest": "ingestion_payload_hash",
    }
    required_kind_by_record_kind = {
        "logical_context_manifest_entry": HashKind.RAW_BYTES,
        "case_manifest_entry": HashKind.RAW_BYTES,
        "ingestion_plan_manifest": HashKind.CANONICAL_JSON,
    }
    for claim in value.hash_claims:
        document = documents.get(claim.evidence_ref)
        hash_field = hash_field_by_kind.get(document.record_kind) if document else None
        required_kind = required_kind_by_record_kind.get(document.record_kind) if document else None
        document_hash_matches = (
            document is None
            or hash_field is None
            or document.document[hash_field] == claim.actual_sha256
        )
        try:
            if claim.hash_kind == HashKind.RAW_BYTES and isinstance(claim.value, bytes):
                computed_hash = hashlib.sha256(claim.value).hexdigest()
            elif claim.hash_kind == HashKind.CANONICAL_JSON:
                computed_hash = canonical_sha256(claim.value)
            else:
                computed_hash = None
        except (TypeError, ValueError):
            computed_hash = None
        if (
            computed_hash != claim.actual_sha256
            or not document_hash_matches
            or (required_kind is not None and claim.hash_kind != required_kind)
        ):
            issues.append(_issue(rule_id, claim.evidence_ref, "hash-mismatch", "/actual_sha256"))
    return tuple(issues)


def _parse_state(entity_kind: EntityKind, value: str) -> StateValue:
    if entity_kind == EntityKind.RUN:
        return RunState(value)
    if entity_kind == EntityKind.VALIDATION:
        return ValidationDisposition(value)
    if entity_kind == EntityKind.INGESTION_PLAN:
        return IngestionPlanState(value)
    return CaseState(value)


def _transition_rule(value: StructuralValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "structural.transition.v1"
    issues: list[ValidationIssue] = []
    for claim in value.transition_claims:
        try:
            current_state = _parse_state(claim.entity_kind, claim.current_state)
            expected = apply_transition(
                claim.entity_kind,
                current_state,
                claim.event,
                resume_disposition=claim.resume_disposition,
            )
        except ValueError:
            issues.append(_issue(rule_id, claim.evidence_ref, "transition-invalid", "/event"))
            continue
        if expected.next_state.value != claim.claimed_next_state:
            issues.append(
                _issue(rule_id, claim.evidence_ref, "transition-mismatch", "/claimed_next_state")
            )
    return tuple(issues)


def _control_value_rule(value: StructuralValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "structural.control-value.v1"
    issues: list[ValidationIssue] = []
    for claim in value.control_field_claims:
        has_forbidden = any(
            forbidden and forbidden in control_value
            for forbidden in claim.forbidden_values
            for control_value in claim.control_values
        )
        if has_forbidden:
            issues.append(
                _issue(rule_id, claim.evidence_ref, "configured-value-exposed", "/control_values")
            )
    return tuple(issues)


STRUCTURAL_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("structural.schema.v1", 1, _schema_rule),
    ValidationRule("structural.identity.v1", 1, _identity_rule),
    ValidationRule("structural.reference.v1", 1, _reference_rule),
    ValidationRule("structural.count.v1", 1, _count_rule),
    ValidationRule("structural.hash.v1", 1, _hash_rule),
    ValidationRule("structural.transition.v1", 1, _transition_rule),
    ValidationRule("structural.control-value.v1", 1, _control_value_rule),
)

STRUCTURAL_RULE_IDS: tuple[str, ...] = tuple(rule.rule_id for rule in STRUCTURAL_RULES)

ARTIFACT_RULE_APPLICABILITY: dict[str, frozenset[str]] = {
    "logical_context_manifest_entry": frozenset(
        {
            "structural.identity.v1",
            "structural.reference.v1",
            "structural.hash.v1",
        }
    ),
    "case_manifest_entry": frozenset(
        {
            "structural.identity.v1",
            "structural.reference.v1",
            "structural.hash.v1",
        }
    ),
    "ingestion_plan_manifest": frozenset(
        {
            "structural.identity.v1",
            "structural.reference.v1",
            "structural.hash.v1",
        }
    ),
}


def _claim_refs_by_rule(value: StructuralValidationInput) -> dict[str, tuple[str, ...]]:
    return {
        "structural.schema.v1": tuple(document.record_id for document in value.documents),
        "structural.identity.v1": tuple(claim.evidence_ref for claim in value.identity_claims),
        "structural.reference.v1": tuple(claim.evidence_ref for claim in value.reference_claims),
        "structural.count.v1": tuple(claim.evidence_ref for claim in value.count_claims),
        "structural.hash.v1": tuple(claim.evidence_ref for claim in value.hash_claims),
        "structural.transition.v1": tuple(claim.evidence_ref for claim in value.transition_claims),
        "structural.control-value.v1": tuple(
            claim.evidence_ref for claim in value.control_field_claims
        ),
    }


def _input_coverage_is_closed(value: StructuralValidationInput) -> bool:
    if not value.documents:
        return False
    coverage_by_rule = {claim.rule_id: claim for claim in value.rule_coverage_claims}
    if len(coverage_by_rule) != len(value.rule_coverage_claims):
        return False
    if set(coverage_by_rule) != set(STRUCTURAL_RULE_IDS):
        return False
    for rule_id, actual_refs in _claim_refs_by_rule(value).items():
        coverage = coverage_by_rule[rule_id]
        if coverage.not_applicable_reason is not None:
            return False
        if len(set(coverage.target_evidence_refs)) != len(coverage.target_evidence_refs):
            return False
        if not actual_refs or set(actual_refs) != set(coverage.target_evidence_refs):
            return False
    refs_by_rule = {rule_id: set(refs) for rule_id, refs in _claim_refs_by_rule(value).items()}
    for document in value.documents:
        for rule_id in ARTIFACT_RULE_APPLICABILITY.get(document.record_kind, frozenset()):
            if document.record_id not in refs_by_rule[rule_id]:
                return False
    return True


def _validation_input_hash(value: StructuralValidationInput) -> str:
    return canonical_sha256(
        [
            "oamb-structural-validation-input-v1",
            _validation_hash_value(value.model_dump(mode="python")),
        ]
    )


def _validation_hash_value(value: Any) -> Any:
    if isinstance(value, dict):
        encoded_items = [
            [_validation_hash_value(key), _validation_hash_value(item)]
            for key, item in value.items()
        ]
        encoded_items.sort(key=canonical_json_bytes)
        return ["map", encoded_items]
    if isinstance(value, (list, tuple)):
        return ["sequence", [_validation_hash_value(item) for item in value]]
    if isinstance(value, (bytes, bytearray)):
        raw_bytes = bytes(value)
        return ["bytes", hashlib.sha256(raw_bytes).hexdigest(), len(raw_bytes)]
    if isinstance(value, Enum):
        return [
            "enum",
            type(value).__module__,
            type(value).__qualname__,
            _validation_hash_value(value.value),
        ]
    if value is None:
        return ["null"]
    if isinstance(value, bool):
        return ["bool", value]
    if isinstance(value, int):
        return ["int", value]
    if isinstance(value, float):
        return ["float", value.hex()]
    if isinstance(value, str):
        return ["string", value]
    if isinstance(value, Decimal):
        return ["decimal", str(value)]
    if isinstance(value, datetime):
        return ["datetime", value.isoformat()]
    if isinstance(value, date):
        return ["date", value.isoformat()]
    return [
        "unsupported",
        type(value).__module__,
        type(value).__qualname__,
    ]


def validate_structural_input(
    value: StructuralValidationInput,
    profile: ValidationProfile,
    registry: RuleRegistry,
) -> ValidationResult:
    required_ids = tuple(requirement.rule_id for requirement in profile.required_rules)
    target_hash = _validation_input_hash(value)
    if profile.stage != ValidationStage.EVIDENCE:
        return _preflight_failure(
            profile=profile,
            target_hash=target_hash,
            required_ids=required_ids,
            rule_id="validation.stage.v1",
            code="wrong-validation-stage",
            pointer="/stage",
        )
    expected_inventory_hash = canonical_sha256(
        [
            "oamb-required-rule-inventory-v1",
            tuple(
                (requirement.rule_id, requirement.minimum_version)
                for requirement in profile.required_rules
            ),
        ]
    )
    if profile.required_rule_inventory_hash != expected_inventory_hash:
        return _preflight_failure(
            profile=profile,
            target_hash=target_hash,
            required_ids=required_ids,
            rule_id="validation.profile-inventory.v1",
            code="profile-inventory-mismatch",
            pointer="/required_rule_inventory_hash",
        )
    if required_ids != STRUCTURAL_RULE_IDS:
        return _preflight_failure(
            profile=profile,
            target_hash=target_hash,
            required_ids=required_ids,
            rule_id="validation.profile-coverage.v1",
            code="profile-rule-coverage-mismatch",
            pointer="/required_rules",
        )
    if not _input_coverage_is_closed(value):
        return _preflight_failure(
            profile=profile,
            target_hash=target_hash,
            required_ids=required_ids,
            rule_id="validation.input-coverage.v1",
            code="input-rule-coverage-mismatch",
            pointer="/rule_coverage_claims",
        )
    executed: list[str] = []
    passed: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    issues: list[ValidationIssue] = []
    implementation_versions: list[str] = []

    schema_verified = False
    for requirement in profile.required_rules:
        if requirement.rule_id != "structural.schema.v1" and not schema_verified:
            missing.append(requirement.rule_id)
            continue
        rule = registry.compatible(requirement.rule_id, requirement.minimum_version)
        if rule is None:
            missing.append(requirement.rule_id)
            continue
        executed.append(rule.rule_id)
        implementation_versions.append(f"{rule.rule_id}@{rule.version}")
        rule_issues = rule.evaluate(value)
        if rule_issues:
            failed.append(rule.rule_id)
            issues.extend(rule_issues)
        else:
            passed.append(rule.rule_id)
            if rule.rule_id == "structural.schema.v1":
                schema_verified = True

    disposition = (
        ValidationDisposition.VALIDATED
        if not missing and not failed
        else ValidationDisposition.INVALID
    )
    return ValidationResult(
        validation_profile_id=profile.profile_id,
        target_hash=target_hash,
        disposition=disposition,
        required_rule_ids=required_ids,
        executed_rule_ids=tuple(executed),
        passed_rule_ids=tuple(passed),
        failed_rule_ids=tuple(failed),
        not_applicable_rule_ids=(),
        missing_rule_ids=tuple(missing),
        implementation_versions=tuple(implementation_versions),
        issues=tuple(issues),
    )


class ProviderServiceEvidenceValidationInput(StructuralModel):
    root_directory: Path
    source_kind: Literal["provider_service"]
    operation: Literal["model_readiness"]
    expected_provider: NonEmptyStr
    expected_provider_project_id: NonEmptyStr
    expected_provider_profile_id: NonEmptyStr
    expected_transport_profile: TransportProfile
    expected_release_version: NonEmptyStr
    expected_source_revision: NonEmptyStr
    expected_build_artifact_sha256: Sha256
    forbidden_control_values: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ProviderServiceSnapshot:
    value: ProviderServiceEvidenceValidationInput
    manifest_bytes: bytes | None
    manifest_document: dict[str, Any] | None
    manifest: ProviderServiceEvidenceManifest | None
    actual_files: dict[str, bytes]
    parsed_documents: dict[str, BaseModel]
    schema_errors: tuple[tuple[str, str], ...]
    symlink_paths: tuple[str, ...]


ProviderDocumentT = TypeVar("ProviderDocumentT", bound=BaseModel)


def _load_provider_service_snapshot(
    value: ProviderServiceEvidenceValidationInput,
) -> _ProviderServiceSnapshot:
    root = value.root_directory
    actual_files: dict[str, bytes] = {}
    symlink_paths: list[str] = []
    errors: list[tuple[str, str]] = []
    manifest_bytes: bytes | None = None
    manifest_document: dict[str, Any] | None = None
    manifest: ProviderServiceEvidenceManifest | None = None
    root_descriptor: int | None = None
    try:
        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode):
            symlink_paths.append(".")
            raise OSError("provider evidence root is a symbolic link")
        if not stat.S_ISDIR(root_metadata.st_mode):
            raise OSError("provider evidence root is not a directory")
        root_descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened_root = os.fstat(root_descriptor)
        if (
            not stat.S_ISDIR(opened_root.st_mode)
            or opened_root.st_dev != root_metadata.st_dev
            or opened_root.st_ino != root_metadata.st_ino
        ):
            raise OSError("provider evidence root changed while opening")
        for directory, directory_names, file_names, directory_descriptor in os.fwalk(
            ".",
            topdown=True,
            follow_symlinks=False,
            dir_fd=root_descriptor,
        ):
            for directory_name in tuple(directory_names):
                metadata = os.stat(
                    directory_name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
                if stat.S_ISLNK(metadata.st_mode):
                    relative = (
                        PurePosixPath(directory, directory_name).as_posix().removeprefix("./")
                    )
                    symlink_paths.append(relative)
                    directory_names.remove(directory_name)
            for file_name in sorted(file_names):
                relative_path = PurePosixPath(directory, file_name).as_posix().removeprefix("./")
                try:
                    metadata = os.stat(
                        file_name,
                        dir_fd=directory_descriptor,
                        follow_symlinks=False,
                    )
                    if stat.S_ISLNK(metadata.st_mode):
                        symlink_paths.append(relative_path)
                        continue
                    if not stat.S_ISREG(metadata.st_mode):
                        raise OSError("provider evidence leaf is not a regular file")
                    descriptor = os.open(
                        file_name,
                        os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
                        dir_fd=directory_descriptor,
                    )
                    try:
                        opened = os.fstat(descriptor)
                        if (
                            not stat.S_ISREG(opened.st_mode)
                            or opened.st_dev != metadata.st_dev
                            or opened.st_ino != metadata.st_ino
                        ):
                            raise OSError("provider evidence leaf changed while opening")
                        chunks: list[bytes] = []
                        while chunk := os.read(descriptor, 1024 * 1024):
                            chunks.append(chunk)
                        content = b"".join(chunks)
                    finally:
                        os.close(descriptor)
                    if relative_path == MANIFEST_PATH:
                        manifest_bytes = content
                    else:
                        actual_files[relative_path] = content
                except OSError:
                    errors.append((relative_path, "source-unreadable"))
    except OSError:
        errors.append((MANIFEST_PATH, "root-missing-or-unsafe"))
    finally:
        if root_descriptor is not None:
            os.close(root_descriptor)

    if manifest_bytes is not None:
        try:
            loaded = json.loads(manifest_bytes)
            if not isinstance(loaded, dict):
                raise ValueError("manifest must be an object")
            manifest_document = loaded
            parsed_manifest = parse_contract(loaded)
            if not isinstance(parsed_manifest, ProviderServiceEvidenceManifest):
                raise ValueError("wrong manifest contract")
            manifest = parsed_manifest
        except (OSError, ValueError, TypeError):
            errors.append((MANIFEST_PATH, "manifest-schema-invalid"))
    else:
        errors.append((MANIFEST_PATH, "manifest-schema-invalid"))

    parsed_documents: dict[str, BaseModel] = {}
    if manifest is not None:
        for entry in manifest.source_entries:
            relative_path = entry.relative_path
            if entry.record_kind == "raw" or relative_path not in actual_files:
                continue
            try:
                loaded = json.loads(actual_files[relative_path])
                if not isinstance(loaded, dict):
                    raise ValueError("contract document must be an object")
                parsed = parse_contract(loaded)
                schema_name = type(parsed).model_fields["schema_name"].default
                if schema_name != entry.record_kind:
                    raise ValueError("manifest record kind does not match schema name")
                parsed_documents[relative_path] = parsed
            except (ValueError, TypeError, UnicodeDecodeError):
                errors.append((relative_path, "source-schema-invalid"))
    return _ProviderServiceSnapshot(
        value=value,
        manifest_bytes=manifest_bytes,
        manifest_document=manifest_document,
        manifest=manifest,
        actual_files=actual_files,
        parsed_documents=parsed_documents,
        schema_errors=tuple(errors),
        symlink_paths=tuple(symlink_paths),
    )


MANIFEST_PATH = "provider-service-evidence-manifest.json"

_PROVIDER_RECORD_ID_FIELDS = {
    "provider_runtime_profile_attestation": "attestation_hash",
    "model_role_binding": "binding_id",
    "budget_spec": "budget_id",
    "cost_measurement_spec": "measurement_spec_id",
    "model_readiness_occurrence_record": "occurrence_id",
    "occurrence_claim_record": "claim_id",
    "budget_reservation_record": "reservation_id",
    "attempt_intent_record": "attempt_id",
    "attempt_receipt_record": "attempt_id",
    "attempt_record": "attempt_id",
    "token_usage_record": "usage_record_id",
    "resource_usage_record": "resource_record_id",
    "cost_record": "cost_record_id",
    "run_spec": "run_id",
    "run_record": "run_id",
    "memory_system_runtime_binding": "runtime_binding_hash",
    "ingestion_plan_record": "ingestion_occurrence_id",
    "case_record": "case_occurrence_id",
    "capsule_manifest": "capsule_id",
}


def _provider_issue(
    rule_id: str,
    code: str,
    evidence_ref: str = MANIFEST_PATH,
    pointer: str | None = None,
) -> tuple[ValidationIssue, ...]:
    return (_issue(rule_id, evidence_ref, code, pointer),)


def _provider_documents(
    snapshot: _ProviderServiceSnapshot,
    model_type: type[ProviderDocumentT],
) -> tuple[ProviderDocumentT, ...]:
    return tuple(
        document
        for document in snapshot.parsed_documents.values()
        if isinstance(document, model_type)
    )


def _provider_paths(
    snapshot: _ProviderServiceSnapshot,
    model_type: type[BaseModel],
) -> tuple[str, ...]:
    return tuple(
        path
        for path, document in snapshot.parsed_documents.items()
        if isinstance(document, model_type)
    )


def _provider_schema_rule(snapshot: _ProviderServiceSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-schema"
    if snapshot.schema_errors or snapshot.manifest is None:
        evidence_ref = snapshot.schema_errors[0][0] if snapshot.schema_errors else MANIFEST_PATH
        return _provider_issue(rule_id, "provider-schema-invalid", evidence_ref, "/")
    if (
        snapshot.value.source_kind != "provider_service"
        or snapshot.value.operation != "model_readiness"
        or snapshot.manifest.operation != "model_readiness"
    ):
        return _provider_issue(rule_id, "provider-schema-applicability-mismatch", pointer="/")
    return ()


def _provider_manifest_closure_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-manifest-closure"
    if snapshot.manifest is None or snapshot.manifest_document is None:
        return _provider_issue(rule_id, "provider-manifest-missing")
    entries = snapshot.manifest.source_entries
    paths = tuple(entry.relative_path for entry in entries)
    if len(set(paths)) != len(paths):
        return _provider_issue(
            rule_id, "provider-manifest-duplicate-path", pointer="/source_entries"
        )
    if set(paths) != set(snapshot.actual_files):
        return _provider_issue(
            rule_id, "provider-manifest-file-set-mismatch", pointer="/source_entries"
        )
    for entry in entries:
        content = snapshot.actual_files[entry.relative_path]
        if hashlib.sha256(content).hexdigest() != entry.sha256:
            return _provider_issue(
                rule_id,
                "provider-manifest-hash-mismatch",
                entry.relative_path,
                "/sha256",
            )
        if entry.record_kind == "raw":
            try:
                payload = (
                    gzip.decompress(content) if entry.relative_path.endswith(".gz") else content
                )
            except (OSError, EOFError):
                return _provider_issue(
                    rule_id,
                    "provider-manifest-raw-compression-invalid",
                    entry.relative_path,
                )
            expected_record_id = hashlib.sha256(payload).hexdigest()
            allowed_names = {
                expected_record_id,
                f"{expected_record_id}.json",
                f"{expected_record_id}.json.gz",
            }
            if (
                entry.record_id != expected_record_id
                or PurePosixPath(entry.relative_path).name not in allowed_names
            ):
                return _provider_issue(
                    rule_id,
                    "provider-manifest-raw-identity-mismatch",
                    entry.relative_path,
                    "/record_id",
                )
            continue
        document = snapshot.parsed_documents.get(entry.relative_path)
        identity_field = _PROVIDER_RECORD_ID_FIELDS.get(entry.record_kind)
        if (
            document is None
            or identity_field is None
            or str(getattr(document, identity_field, "")) != entry.record_id
        ):
            return _provider_issue(
                rule_id,
                "provider-manifest-record-identity-mismatch",
                entry.relative_path,
                "/record_id",
            )
    without_hash = {
        key: value for key, value in snapshot.manifest_document.items() if key != "manifest_hash"
    }
    if canonical_sha256(without_hash) != snapshot.manifest.manifest_hash:
        return _provider_issue(
            rule_id, "provider-manifest-identity-mismatch", pointer="/manifest_hash"
        )
    return ()


def _provider_path_safety_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-path-safety"
    if snapshot.manifest is None:
        return _provider_issue(rule_id, "provider-path-manifest-missing")
    operational_components = {".runtime", "checkpoints", "derivation-attempts", "tmp"}
    for entry in snapshot.manifest.source_entries:
        path = entry.relative_path
        parts = PurePosixPath(path).parts
        unsafe = (
            not parts
            or parts[0] != "source"
            or path.startswith("/")
            or ".." in parts
            or bool(set(parts) & operational_components)
            or path.endswith((".tmp", ".partial", "~"))
        )
        if unsafe:
            return _provider_issue(rule_id, "provider-path-unsafe", path, "/relative_path")
    if snapshot.symlink_paths:
        return _provider_issue(rule_id, "provider-path-symlink", snapshot.symlink_paths[0])
    return ()


def _provider_occurrence_terminal_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-occurrence-terminal"
    occurrences = _provider_documents(snapshot, ModelReadinessOccurrenceRecord)
    if len(occurrences) != 1 or snapshot.manifest is None:
        return _provider_issue(rule_id, "provider-occurrence-cardinality")
    occurrence = occurrences[0]
    terminal = {
        ModelReadinessOccurrenceState.SEALED,
        ModelReadinessOccurrenceState.ERROR,
        ModelReadinessOccurrenceState.CANCELLED,
        ModelReadinessOccurrenceState.BUDGET_EXCEEDED,
        ModelReadinessOccurrenceState.INTERRUPTED_UNKNOWN_OUTCOME,
    }
    if occurrence.state not in terminal:
        return _provider_issue(rule_id, "provider-occurrence-nonterminal", pointer="/state")
    entries_by_kind: dict[str, list[str]] = {}
    for entry in snapshot.manifest.source_entries:
        entries_by_kind.setdefault(entry.record_kind, []).append(entry.record_id)
    expected = {
        "attempt_record": tuple(occurrence.attempt_ids),
        "token_usage_record": tuple(occurrence.usage_record_ids),
        "resource_usage_record": tuple(occurrence.resource_record_ids),
        "cost_record": tuple(occurrence.cost_record_ids),
    }
    for record_kind, record_ids in expected.items():
        if tuple(entries_by_kind.get(record_kind, ())) != record_ids:
            return _provider_issue(
                rule_id,
                "provider-occurrence-inventory-mismatch",
                pointer=f"/{record_kind}",
            )
    return ()


def _provider_parent_isolation_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-parent-isolation"
    occurrences = _provider_documents(snapshot, ModelReadinessOccurrenceRecord)
    if len(occurrences) != 1:
        return _provider_issue(rule_id, "provider-parent-occurrence-missing")
    occurrence_id = occurrences[0].occurrence_id
    parent_records = (
        *_provider_documents(snapshot, AttemptRecordV2),
        *_provider_documents(snapshot, AttemptIntentRecord),
        *_provider_documents(snapshot, TokenUsageRecordV2),
        *_provider_documents(snapshot, ResourceUsageRecord),
        *_provider_documents(snapshot, CostRecord),
    )
    for record in parent_records:
        if (
            getattr(record, "parent_kind", None) != "model_readiness"
            or getattr(record, "parent_id", None) != occurrence_id
        ):
            return _provider_issue(rule_id, "provider-parent-cross-reference", pointer="/parent_id")
    usage_record_ids = set(occurrences[0].usage_record_ids)
    resource_record_ids = set(occurrences[0].resource_record_ids)
    for cost in _provider_documents(snapshot, CostRecord):
        if (
            not set(cost.source_usage_record_ids) <= usage_record_ids
            or not set(cost.source_resource_record_ids) <= resource_record_ids
        ):
            return _provider_issue(
                rule_id,
                "provider-cost-source-cross-reference",
                pointer="/source_usage_record_ids",
            )
    for reservation in _provider_documents(snapshot, BudgetReservationRecord):
        if reservation.scope_kind != BudgetScopeKindV2.MODEL_READINESS:
            return _provider_issue(rule_id, "provider-budget-parent-kind", pointer="/scope_kind")
        if reservation.scope_id != occurrence_id:
            return _provider_issue(rule_id, "provider-budget-parent-id", pointer="/scope_id")
    return ()


def _ceiling_map(ceilings: tuple[Any, ...]) -> dict[str, Decimal]:
    return {ceiling.dimension_id: ceiling.maximum for ceiling in ceilings}


def _provider_budget_closure_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-budget-closure"
    budgets = _provider_documents(snapshot, BudgetSpecV2)
    occurrences = _provider_documents(snapshot, ModelReadinessOccurrenceRecord)
    roles = _provider_documents(snapshot, ModelRoleBindingV2)
    reservations = _provider_documents(snapshot, BudgetReservationRecord)
    if not (len(budgets) == len(occurrences) == 1):
        return _provider_issue(rule_id, "provider-budget-cardinality")
    budget = budgets[0]
    occurrence = occurrences[0]
    selected_role_ids = tuple(
        role.binding_id for role in roles if role.role_status == RoleBindingStatus.SELECTED
    )
    role_ceilings = {ceiling.role_binding_id: ceiling for ceiling in budget.role_ceilings}
    roles_by_id = {role.binding_id: role for role in roles}
    closed = (
        budget.scope_kind == BudgetScopeKindV2.MODEL_READINESS
        and budget.scope_id == occurrence.occurrence_id
        and tuple(role_ceilings) == selected_role_ids == tuple(occurrence.role_binding_ids)
    )
    if not closed:
        return _provider_issue(rule_id, "provider-budget-binding-mismatch")
    if any(
        role_id not in roles_by_id
        or ceiling.provider_budget_cap.provider != roles_by_id[role_id].provider
        for role_id, ceiling in role_ceilings.items()
    ):
        return _provider_issue(rule_id, "provider-budget-provider-cap-mismatch")

    parent_resources = _ceiling_map(budget.resource_ceilings)
    parent_attempts = parent_input = parent_output = 0
    parent_wall = Decimal("0")
    parent_cost = Decimal("0")
    parent_resources_used = {dimension_id: Decimal("0") for dimension_id in parent_resources}
    role_totals: dict[str, dict[str, Any]] = {}
    for role_id, ceiling in role_ceilings.items():
        role_totals[role_id] = {
            "attempts": 0,
            "input": 0,
            "output": 0,
            "wall": Decimal("0"),
            "provider": Decimal("0"),
            "cost": Decimal("0"),
            "resources": {
                dimension_id: Decimal("0")
                for dimension_id in _ceiling_map(ceiling.resource_ceilings)
            },
        }
    for reservation in reservations:
        role_ceiling = role_ceilings.get(reservation.role_binding_id)
        if role_ceiling is None or reservation.budget_id != budget.budget_id:
            return _provider_issue(rule_id, "provider-reservation-role-mismatch")
        if reservation.scope_id != budget.scope_id:
            return _provider_issue(rule_id, "provider-reservation-scope-mismatch")
        if role_ceiling.currency != budget.currency or reservation.currency != budget.currency:
            return _provider_issue(rule_id, "provider-reservation-currency-mismatch")
        role_resource_limits = _ceiling_map(role_ceiling.resource_ceilings)
        reserved_resources = _ceiling_map(reservation.reserved_resource_ceilings)
        if set(reserved_resources) != set(role_resource_limits):
            return _provider_issue(rule_id, "provider-reservation-resource-mismatch")
        if any(
            value > role_resource_limits[dimension_id] or dimension_id not in parent_resources
            for dimension_id, value in reserved_resources.items()
        ):
            return _provider_issue(rule_id, "provider-reservation-resource-over-cap")
        if (
            reservation.reserved_attempts > role_ceiling.max_attempts
            or reservation.reserved_input_tokens > role_ceiling.max_input_tokens
            or reservation.reserved_output_tokens > role_ceiling.max_output_tokens
            or reservation.reserved_dispatch_wall_seconds > role_ceiling.max_dispatch_wall_seconds
            or reservation.reserved_provider_units
            > role_ceiling.provider_budget_cap.maximum_accepted_units
            or (
                role_ceiling.max_cost is not None
                and (reservation.reserved_cost or Decimal("0")) > role_ceiling.max_cost
            )
        ):
            return _provider_issue(rule_id, "provider-reservation-over-cap")
        totals = role_totals[reservation.role_binding_id]
        totals["attempts"] += reservation.reserved_attempts
        totals["input"] += reservation.reserved_input_tokens
        totals["output"] += reservation.reserved_output_tokens
        totals["wall"] += reservation.reserved_dispatch_wall_seconds
        totals["provider"] += reservation.reserved_provider_units
        totals["cost"] += reservation.reserved_cost or Decimal("0")
        parent_attempts += reservation.reserved_attempts
        parent_input += reservation.reserved_input_tokens
        parent_output += reservation.reserved_output_tokens
        parent_wall += reservation.reserved_dispatch_wall_seconds
        parent_cost += reservation.reserved_cost or Decimal("0")
        for dimension_id, amount in reserved_resources.items():
            totals["resources"][dimension_id] += amount
            parent_resources_used[dimension_id] += amount
    if (
        parent_attempts > budget.max_attempts
        or parent_input > budget.max_input_tokens
        or parent_output > budget.max_output_tokens
        or parent_wall > budget.max_dispatch_wall_seconds
        or (budget.max_cost is not None and parent_cost > budget.max_cost)
        or any(parent_resources_used[key] > parent_resources[key] for key in parent_resources)
    ):
        return _provider_issue(rule_id, "provider-parent-budget-over-cap")
    for role_id, totals in role_totals.items():
        ceiling = role_ceilings[role_id]
        resource_limits = _ceiling_map(ceiling.resource_ceilings)
        if (
            totals["attempts"] > ceiling.max_attempts
            or totals["input"] > ceiling.max_input_tokens
            or totals["output"] > ceiling.max_output_tokens
            or totals["wall"] > ceiling.max_dispatch_wall_seconds
            or totals["provider"] > ceiling.provider_budget_cap.maximum_accepted_units
            or (ceiling.max_cost is not None and totals["cost"] > ceiling.max_cost)
            or any(totals["resources"][key] > resource_limits[key] for key in resource_limits)
        ):
            return _provider_issue(rule_id, "provider-role-budget-over-cap")
    return ()


def _unique_by(values: tuple[Any, ...], field: str) -> dict[str, Any] | None:
    keyed = {str(getattr(value, field)): value for value in values}
    return keyed if len(keyed) == len(values) else None


def _provider_attempt_ordering_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-attempt-ordering"
    attempts = _provider_documents(snapshot, AttemptRecordV2)
    claims = _provider_documents(snapshot, OccurrenceClaimRecord)
    reservations = _provider_documents(snapshot, BudgetReservationRecord)
    intents = _provider_documents(snapshot, AttemptIntentRecord)
    receipts = _provider_documents(snapshot, AttemptReceiptRecord)
    attempts_by_id = _unique_by(attempts, "attempt_id")
    claims_by_id = _unique_by(claims, "claim_id")
    reservations_by_attempt = _unique_by(reservations, "attempt_id")
    intents_by_attempt = _unique_by(intents, "attempt_id")
    receipts_by_attempt = _unique_by(receipts, "attempt_id")
    if any(
        index is None
        for index in (
            attempts_by_id,
            claims_by_id,
            reservations_by_attempt,
            intents_by_attempt,
            receipts_by_attempt,
        )
    ):
        return _provider_issue(rule_id, "provider-attempt-duplicate")
    assert attempts_by_id is not None
    assert claims_by_id is not None
    assert reservations_by_attempt is not None
    assert intents_by_attempt is not None
    assert receipts_by_attempt is not None
    if not (
        set(attempts_by_id) == set(reservations_by_attempt) == set(intents_by_attempt)
    ) or not set(receipts_by_attempt) <= set(attempts_by_id):
        return _provider_issue(rule_id, "provider-attempt-ledger-incomplete")
    if len(claims_by_id) != len(attempts_by_id) or {
        intent.claim_id for intent in intents_by_attempt.values()
    } != set(claims_by_id):
        return _provider_issue(rule_id, "provider-attempt-claim-ledger-incomplete")
    raw_ids = (
        {
            entry.record_id
            for entry in snapshot.manifest.source_entries
            if entry.record_kind == "raw"
        }
        if snapshot.manifest is not None
        else set()
    )
    selected_role_ids = {
        role.binding_id
        for role in _provider_documents(snapshot, ModelRoleBindingV2)
        if role.role_status == RoleBindingStatus.SELECTED
    }
    ordinals: set[int] = set()
    for attempt_id, attempt in attempts_by_id.items():
        reservation = reservations_by_attempt[attempt_id]
        intent = intents_by_attempt[attempt_id]
        claim = claims_by_id[intent.claim_id]
        if (
            intent.reservation_id != reservation.reservation_id
            or intent.request_fingerprint != attempt.request_fingerprint
            or intent.parent_id != attempt.parent_id
            or intent.parent_kind != attempt.parent_kind
            or intent.stage != attempt.stage
            or intent.stage != "model_readiness"
            or intent.role_binding_id != reservation.role_binding_id
            or intent.role_binding_id not in selected_role_ids
            or intent.reconciliation_capability != attempt.reconciliation_capability
            or intent.idempotency_key_hash != attempt.idempotency_key_hash
            or claim.occurrence_id != intent.parent_id
            or claim.stage != intent.stage
            or claim.request_fingerprint != intent.request_fingerprint
            or claim.reconciliation_capability != intent.reconciliation_capability
            or attempt.ordinal in ordinals
        ):
            return _provider_issue(rule_id, "provider-attempt-order-mismatch")
        ordinals.add(attempt.ordinal)
        receipt = receipts_by_attempt.get(attempt_id)
        if attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME:
            if (
                receipt is not None
                or attempt.raw_response_ref is not None
                or attempt.raw_error_ref is not None
            ):
                return _provider_issue(rule_id, "provider-unknown-attempt-has-receipt")
            if not (
                reservation.reserved_at >= claim.claimed_at
                and reservation.reserved_at
                <= intent.sealed_at
                <= attempt.started_at
                <= attempt.ended_at
            ):
                return _provider_issue(rule_id, "provider-attempt-durability-order-invalid")
        else:
            if receipt is None:
                return _provider_issue(rule_id, "provider-attempt-receipt-missing")
            if not (
                reservation.reserved_at >= claim.claimed_at
                and reservation.reserved_at
                <= intent.sealed_at
                <= receipt.dispatch_started_at
                <= receipt.receipt_observed_at
                <= attempt.ended_at
                and attempt.started_at == receipt.dispatch_started_at
            ):
                return _provider_issue(rule_id, "provider-attempt-durability-order-invalid")
            receipt_raw = receipt.raw_response_ref or receipt.raw_error_ref
            attempt_raw = attempt.raw_response_ref or attempt.raw_error_ref
            if receipt_raw != attempt_raw or (
                receipt_raw is not None and receipt_raw not in raw_ids
            ):
                return _provider_issue(rule_id, "provider-attempt-receipt-mismatch")
        if attempt.retry_of_attempt_id is not None:
            predecessor = attempts_by_id.get(attempt.retry_of_attempt_id)
            if predecessor is None or predecessor.ordinal >= attempt.ordinal:
                return _provider_issue(rule_id, "provider-attempt-retry-order-invalid")
    if ordinals != set(range(1, len(attempts) + 1)):
        return _provider_issue(rule_id, "provider-attempt-ordinal-gap")
    return ()


def _provider_unknown_outcome_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-unknown-outcome"
    attempts = _provider_documents(snapshot, AttemptRecordV2)
    occurrences = _provider_documents(snapshot, ModelReadinessOccurrenceRecord)
    reservations = _provider_documents(snapshot, BudgetReservationRecord)
    unknown_ids = {
        attempt.attempt_id for attempt in attempts if attempt.outcome.value == "unknown_outcome"
    }
    if not unknown_ids:
        return ()
    if len(occurrences) != 1 or occurrences[0].state != (
        ModelReadinessOccurrenceState.INTERRUPTED_UNKNOWN_OUTCOME
    ):
        return _provider_issue(rule_id, "provider-unknown-parent-disposition")
    if any(attempt.retry_of_attempt_id in unknown_ids for attempt in attempts):
        return _provider_issue(rule_id, "provider-unknown-attempt-replayed")
    reserved_attempt_ids = {reservation.attempt_id for reservation in reservations}
    if not unknown_ids <= reserved_attempt_ids:
        return _provider_issue(rule_id, "provider-unknown-reservation-missing")
    return ()


def _provider_runtime_binding_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-runtime-binding"
    attestations = _provider_documents(snapshot, ProviderRuntimeProfileAttestation)
    occurrences = _provider_documents(snapshot, ModelReadinessOccurrenceRecord)
    roles = _provider_documents(snapshot, ModelRoleBindingV2)
    measurements = _provider_documents(snapshot, CostMeasurementSpec)
    if not (len(attestations) == len(occurrences) == len(measurements) == 1):
        return _provider_issue(rule_id, "provider-runtime-edge-missing")
    attestation = attestations[0]
    occurrence = occurrences[0]
    role_ids = tuple(role.binding_id for role in roles)
    if (
        snapshot.manifest is None
        or snapshot.manifest.provider_project_id != occurrence.provider_project_id
        or snapshot.manifest.provider_profile_id != occurrence.provider_profile_id
        or snapshot.manifest.occurrence_id != occurrence.occurrence_id
        or attestation.provider != occurrence.provider
        or attestation.provider_project_id != occurrence.provider_project_id
        or attestation.provider_profile_id != occurrence.provider_profile_id
        or attestation.attestation_hash != occurrence.provider_runtime_profile_attestation_hash
        or role_ids != tuple(attestation.model_role_binding_ids)
        or role_ids != tuple(occurrence.role_binding_ids)
    ):
        return _provider_issue(rule_id, "provider-runtime-binding-mismatch")
    expected_identity = snapshot.value
    if (
        attestation.provider != expected_identity.expected_provider
        or attestation.provider_project_id != expected_identity.expected_provider_project_id
        or attestation.provider_profile_id != expected_identity.expected_provider_profile_id
        or attestation.transport_profile != expected_identity.expected_transport_profile
        or attestation.release_version != expected_identity.expected_release_version
        or attestation.source_revision != expected_identity.expected_source_revision
        or attestation.build_artifact_sha256 != expected_identity.expected_build_artifact_sha256
    ):
        return _provider_issue(rule_id, "provider-runtime-exact-identity-mismatch")
    if any(role.role_status != RoleBindingStatus.SELECTED or role.model is None for role in roles):
        return _provider_issue(rule_id, "provider-runtime-role-unattested")
    if (
        attestation.liveness_status != ProviderGateStatus.PASS
        or attestation.storage_configuration_status != ProviderGateStatus.PASS
        or attestation.runtime_identity_status != ProviderGateStatus.PASS
    ):
        return _provider_issue(rule_id, "provider-runtime-service-gate-not-passed")
    return ()


def _provider_gate_separation_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-gate-separation"
    forbidden_kinds = {
        "run_spec",
        "run_record",
        "memory_system_runtime_binding",
        "ingestion_plan_record",
        "case_record",
        "capsule_manifest",
    }
    if snapshot.manifest is None:
        return _provider_issue(rule_id, "provider-gate-manifest-missing")
    if any(entry.record_kind in forbidden_kinds for entry in snapshot.manifest.source_entries):
        return _provider_issue(rule_id, "provider-gate-promotion")
    attestations = _provider_documents(snapshot, ProviderRuntimeProfileAttestation)
    if len(attestations) != 1:
        return _provider_issue(rule_id, "provider-gate-attestation-missing")
    attestation = attestations[0]
    if (
        attestation.model_readiness_status != ProviderGateStatus.NOT_RUN
        or attestation.memory_conformance_status != ProviderGateStatus.NOT_RUN
    ):
        return _provider_issue(rule_id, "provider-gate-state-promoted")
    return ()


def _all_control_strings(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, dict):
        return tuple(item for nested in value.values() for item in _all_control_strings(nested))
    if isinstance(value, (list, tuple)):
        return tuple(item for nested in value for item in _all_control_strings(nested))
    return ()


def _provider_control_redaction_rule(
    snapshot: _ProviderServiceSnapshot,
) -> tuple[ValidationIssue, ...]:
    rule_id = "t4-provider-control-redaction"
    forbidden_values = tuple(value for value in snapshot.value.forbidden_control_values if value)
    private_prefixes = ("/Users/", "/home/", "/private/var/", "C:\\Users\\")
    for path, document in snapshot.parsed_documents.items():
        strings = _all_control_strings(document.model_dump(mode="json"))
        for candidate in strings:
            if any(forbidden in candidate for forbidden in forbidden_values):
                return _provider_issue(rule_id, "provider-control-secret-exposed", path)
            if candidate.startswith(private_prefixes):
                return _provider_issue(rule_id, "provider-control-private-path", path)
            if "://" in candidate or candidate.startswith(("localhost:", "127.0.0.1:")):
                return _provider_issue(rule_id, "provider-control-authority-exposed", path)
    if snapshot.manifest is not None:
        for entry in snapshot.manifest.source_entries:
            if entry.record_kind != "raw":
                continue
            content = snapshot.actual_files.get(entry.relative_path)
            if content is None:
                continue
            try:
                payload = (
                    gzip.decompress(content) if entry.relative_path.endswith(".gz") else content
                )
            except (OSError, EOFError):
                continue
            for forbidden in forbidden_values:
                if forbidden.encode("utf-8") in payload:
                    return _provider_issue(
                        rule_id,
                        "provider-control-secret-exposed",
                        entry.relative_path,
                    )
            if any(prefix.encode("utf-8") in payload for prefix in private_prefixes):
                return _provider_issue(
                    rule_id,
                    "provider-control-private-path",
                    entry.relative_path,
                )
            if b"://" in payload or b"localhost:" in payload or b"127.0.0.1:" in payload:
                return _provider_issue(
                    rule_id,
                    "provider-control-authority-exposed",
                    entry.relative_path,
                )
    return ()


PROVIDER_SERVICE_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("t4-provider-schema", 1, _provider_schema_rule),
    ValidationRule("t4-provider-manifest-closure", 1, _provider_manifest_closure_rule),
    ValidationRule("t4-provider-path-safety", 1, _provider_path_safety_rule),
    ValidationRule("t4-provider-occurrence-terminal", 1, _provider_occurrence_terminal_rule),
    ValidationRule("t4-provider-parent-isolation", 1, _provider_parent_isolation_rule),
    ValidationRule("t4-provider-budget-closure", 1, _provider_budget_closure_rule),
    ValidationRule("t4-provider-attempt-ordering", 1, _provider_attempt_ordering_rule),
    ValidationRule("t4-provider-unknown-outcome", 1, _provider_unknown_outcome_rule),
    ValidationRule("t4-provider-runtime-binding", 1, _provider_runtime_binding_rule),
    ValidationRule("t4-provider-gate-separation", 1, _provider_gate_separation_rule),
    ValidationRule("t4-provider-control-redaction", 1, _provider_control_redaction_rule),
)

PROVIDER_SERVICE_RULE_IDS: tuple[str, ...] = tuple(rule.rule_id for rule in PROVIDER_SERVICE_RULES)


def _provider_validation_target_hash(snapshot: _ProviderServiceSnapshot) -> str:
    return canonical_sha256(
        [
            "oamb-t4-provider-service-validation-input-v1",
            snapshot.value.source_kind,
            snapshot.value.operation,
            snapshot.value.expected_provider,
            snapshot.value.expected_provider_project_id,
            snapshot.value.expected_provider_profile_id,
            snapshot.value.expected_transport_profile.value,
            snapshot.value.expected_release_version,
            snapshot.value.expected_source_revision,
            snapshot.value.expected_build_artifact_sha256,
            hashlib.sha256(snapshot.manifest_bytes or b"").hexdigest(),
            tuple(
                (path, hashlib.sha256(content).hexdigest())
                for path, content in sorted(snapshot.actual_files.items())
            ),
            tuple(
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in snapshot.value.forbidden_control_values
            ),
        ]
    )


def validate_provider_service_evidence(
    value: ProviderServiceEvidenceValidationInput,
    profile: ValidationProfile,
    registry: RuleRegistry,
) -> ValidationResult:
    snapshot = _load_provider_service_snapshot(value)
    required_ids = tuple(requirement.rule_id for requirement in profile.required_rules)
    target_hash = _provider_validation_target_hash(snapshot)
    if profile.stage != ValidationStage.EVIDENCE:
        return _preflight_failure(
            profile=profile,
            target_hash=target_hash,
            required_ids=required_ids,
            rule_id="validation.stage.v1",
            code="wrong-validation-stage",
            pointer="/stage",
        )
    expected_inventory_hash = canonical_sha256(
        [
            "oamb-required-rule-inventory-v1",
            tuple(
                (requirement.rule_id, requirement.minimum_version)
                for requirement in profile.required_rules
            ),
        ]
    )
    if profile.required_rule_inventory_hash != expected_inventory_hash:
        return _preflight_failure(
            profile=profile,
            target_hash=target_hash,
            required_ids=required_ids,
            rule_id="validation.profile-inventory.v1",
            code="profile-inventory-mismatch",
            pointer="/required_rule_inventory_hash",
        )
    if required_ids != PROVIDER_SERVICE_RULE_IDS or profile.applicability != (
        "source_kind=provider_service",
        "operation=model_readiness",
    ):
        return _preflight_failure(
            profile=profile,
            target_hash=target_hash,
            required_ids=required_ids,
            rule_id="validation.profile-coverage.v1",
            code="profile-rule-coverage-mismatch",
            pointer="/required_rules",
        )

    executed: list[str] = []
    passed: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    issues: list[ValidationIssue] = []
    implementation_versions: list[str] = []
    schema_verified = False
    for requirement in profile.required_rules:
        if requirement.rule_id != "t4-provider-schema" and not schema_verified:
            missing.append(requirement.rule_id)
            continue
        rule = registry.compatible(requirement.rule_id, requirement.minimum_version)
        if rule is None:
            missing.append(requirement.rule_id)
            continue
        executed.append(rule.rule_id)
        implementation_versions.append(f"{rule.rule_id}@{rule.version}")
        rule_issues = rule.evaluate(snapshot)
        if rule_issues:
            failed.append(rule.rule_id)
            issues.extend(rule_issues)
        else:
            passed.append(rule.rule_id)
            if rule.rule_id == "t4-provider-schema":
                schema_verified = True
    return ValidationResult(
        validation_profile_id=profile.profile_id,
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


def require_validated(result: ValidationResult) -> None:
    if result.disposition != ValidationDisposition.VALIDATED:
        raise EvidenceNotValidatedError("normal reduction requires VALIDATED evidence")
