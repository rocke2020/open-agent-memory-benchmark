"""Closed validation profile for the pinned external historical evidence pack."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from oamb.artifacts.validation.engine import validate_closed_profile
from oamb.artifacts.validation.profiles import T9_EXTERNAL_EVIDENCE_RULE_INVENTORY
from oamb.artifacts.validation.registry import RuleRegistry, ValidationRule
from oamb.contracts.evidence import ValidationIssue, ValidationResult, ValidationSeverity
from oamb.contracts.external import ExternalHistoricalEvidence
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    ValidationProfile,
    ValidationRuleRequirement,
    ValidationStage,
)

from .amb import importer_implementation_hash, require_curated_evidence
from .pins import (
    AMB_PRODUCER_BASE_REVISION,
    AMB_PRODUCER_CODE_REVISION,
    AMB_SOURCE_ATTESTATION_REVISION,
    AMB_SOURCE_ATTESTATION_SHA256,
    AMB_SOURCE_BYTE_COUNT,
    AMB_SOURCE_SHA256,
    CURATED_IMPORTED_AT,
)

EXTERNAL_HISTORICAL_PROFILE_ID = "oamb-t9-external-amb-historical-v1"


def validate_external_historical_evidence(
    record: ExternalHistoricalEvidence,
) -> ValidationResult:
    profile = ValidationProfile.create(
        profile_id=EXTERNAL_HISTORICAL_PROFILE_ID,
        stage=ValidationStage.EVIDENCE,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=version)
            for rule_id, version in T9_EXTERNAL_EVIDENCE_RULE_INVENTORY
        ),
        applicability=(
            "origin=external_amb_generated",
            "producer_protocol=amb-longmemeval-rag",
            "compatibility=unknown",
        ),
    )
    registry = RuleRegistry()
    for rule in EXTERNAL_HISTORICAL_RULES:
        registry.register(rule)
    return validate_closed_profile(
        record,
        target_hash=canonical_sha256(record),
        profile=profile,
        expected_stage=ValidationStage.EVIDENCE,
        expected_inventory=T9_EXTERNAL_EVIDENCE_RULE_INVENTORY,
        registry=registry,
    )


def _provenance_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "external.provenance.v1"
    if not isinstance(target, ExternalHistoricalEvidence):
        return _wrong_target(rule_id)
    try:
        require_curated_evidence(target)
        reparsed = ExternalHistoricalEvidence.model_validate(target.model_dump(mode="python"))
    except (TypeError, ValueError):
        return (_issue(rule_id, "external-evidence", "external-provenance-drift"),)
    exact = bool(
        reparsed == target
        and target.origin_class.value == "external_amb_generated"
        and target.producer_repository == "rocke2020/agent-memory-benchmark"
        and target.producer_code_revision == AMB_PRODUCER_CODE_REVISION
        and target.producer_base_revision == AMB_PRODUCER_BASE_REVISION
        and target.source_sha256 == AMB_SOURCE_SHA256
        and target.source_byte_count == AMB_SOURCE_BYTE_COUNT
        and target.source_attestation_sha256 == AMB_SOURCE_ATTESTATION_SHA256
        and target.source_attestation_revision == AMB_SOURCE_ATTESTATION_REVISION
        and target.imported_at == datetime.fromisoformat(CURATED_IMPORTED_AT)
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.external_evidence_id, "external-provenance-drift"),)
    )


def _transformation_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "external.transformation.v1"
    if not isinstance(target, ExternalHistoricalEvidence):
        return _wrong_target(rule_id)
    transformation = target.transformation
    exact = bool(
        transformation.algorithm == "amb-factual-projection-v1"
        and transformation.allowed_case_fields
        == ("case_id", "category", "verdict", "context_tokens", "retrieval_time_ms")
        and transformation.allowed_aggregate_fields
        == (
            "total_cases",
            "correct_cases",
            "context_tokens_total",
            "retrieval_time_ms_total",
        )
        and transformation.unchanged_source_root_sha256 == target.source_sha256
        and transformation.importer_implementation_hash == importer_implementation_hash()
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.external_evidence_id, "external-transformation-drift"),)
    )


def _case_aggregate_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "external.case-aggregate.v1"
    if not isinstance(target, ExternalHistoricalEvidence):
        return _wrong_target(rule_id)
    category_expected = {
        "knowledge-update": (78, 75, 3_841_877, Decimal("47391.5")),
        "multi-session": (133, 107, 6_606_764, Decimal("83130.6")),
        "single-session-assistant": (56, 54, 2_758_246, Decimal("40933.3")),
        "single-session-preference": (30, 24, 1_485_501, Decimal("19295.6")),
        "single-session-user": (70, 66, 3_439_798, Decimal("39371.8")),
        "temporal-reasoning": (133, 122, 6_680_430, Decimal("85353.3")),
    }
    categories = {
        item.category: (
            item.total_cases,
            item.correct_cases,
            item.context_tokens_total,
            item.retrieval_time_ms_total,
        )
        for item in target.category_aggregates
    }
    exact = bool(
        len(target.cases) == 500
        and len({case.case_id for case in target.cases}) == 500
        and target.aggregate.total_cases == 500
        and target.aggregate.correct_cases == 448
        and target.aggregate.context_tokens_total == 24_812_616
        and target.aggregate.retrieval_time_ms_total == Decimal("315476.1")
        and categories == category_expected
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.external_evidence_id, "external-case-aggregate-drift"),)
    )


def _compatibility_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "external.compatibility.v1"
    if not isinstance(target, ExternalHistoricalEvidence):
        return _wrong_target(rule_id)
    exact = bool(
        target.compatibility.status == "unknown"
        and target.compatibility.target_protocol == "oamb-longmemeval-protocol"
        and target.compatibility.assessment_code == "not-evaluated-by-oamb-comparison-predicate"
        and not target.comparison_eligible
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.external_evidence_id, "external-compatibility-drift"),)
    )


def _limitations_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "external.limitations.v1"
    if not isinstance(target, ExternalHistoricalEvidence):
        return _wrong_target(rule_id)
    exact = bool(
        not target.oamb_attempt_ledger_present
        and not target.billing_complete
        and not target.cost_complete
        and target.limitation_codes
        == (
            "no-oamb-attempt-ledger",
            "indexing-usage-unavailable",
            "external-llm-usage-unavailable",
            "prompt-compatibility-unknown",
            "amb-context-view-not-oamb-context-view",
            "no-causal-attribution",
        )
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.external_evidence_id, "external-limitation-drift"),)
    )


def _wrong_target(rule_id: str) -> tuple[ValidationIssue, ...]:
    return (_issue(rule_id, "external-evidence", "wrong-validation-target"),)


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


EXTERNAL_HISTORICAL_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("external.provenance.v1", 1, _provenance_rule),
    ValidationRule("external.transformation.v1", 1, _transformation_rule),
    ValidationRule("external.case-aggregate.v1", 1, _case_aggregate_rule),
    ValidationRule("external.compatibility.v1", 1, _compatibility_rule),
    ValidationRule("external.limitations.v1", 1, _limitations_rule),
)


__all__ = [
    "EXTERNAL_HISTORICAL_PROFILE_ID",
    "EXTERNAL_HISTORICAL_RULES",
    "validate_external_historical_evidence",
]
