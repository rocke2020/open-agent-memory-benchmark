"""Early fail-closed schema, identity, reference, count, hash, and state checks."""

from __future__ import annotations

import hashlib
import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum, StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict

from oamb.contracts.base import NonEmptyStr, Sha256
from oamb.contracts.evidence import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256, sha256_identity
from oamb.contracts.schema import parse_contract
from oamb.contracts.specifications import ValidationProfile, ValidationStage
from oamb.contracts.states import (
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


def require_validated(result: ValidationResult) -> None:
    if result.disposition != ValidationDisposition.VALIDATED:
        raise EvidenceNotValidatedError("normal reduction requires VALIDATED evidence")
