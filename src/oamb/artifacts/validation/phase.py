"""Closed T10 phase-gate validation over immutable review evidence."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal

from oamb.artifacts.validation.engine import validate_closed_profile
from oamb.artifacts.validation.profiles import (
    T8_T10_PHASE_GATE_RULE_INVENTORY,
    t10_phase_gate_profile,
)
from oamb.artifacts.validation.registry import RuleRegistry, ValidationRule
from oamb.contracts.accounting import (
    CostRecord,
    ProofStatus,
    ResourceUsageRecord,
    TokenStageV2,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
)
from oamb.contracts.evidence import (
    AttemptRecordV2,
    PhaseReviewOccurrenceRecord,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.ids import attempt_id as derive_attempt_id
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    AIReviewBatchResult,
    AIReviewIntegrityResult,
    EvaluationPhaseGate,
    EvaluationReviewBundle,
    QualityReviewStatus,
    evaluation_review_bundle_id,
    evaluation_review_input_hash,
)
from oamb.contracts.specifications import (
    AIReviewPlan,
    BudgetScopeKindV2,
    BudgetSpecV2,
    ExternalCallApprovalRecord,
    HumanReviewKeyBinding,
    ValidationProfile,
    ValidationStage,
)
from oamb.contracts.states import AttemptOutcome
from oamb.reporting.human_review import (
    validate_ai_review_history,
    verify_human_review_record_evidence,
)

T10_PHASE_GATE_RULE_IDS = tuple(rule_id for rule_id, _version in T8_T10_PHASE_GATE_RULE_INVENTORY)
T10_PHASE_ID = "phase_1_smoke_acceptance"
T10_CAPSULE_COUNT = 4
T10_UNIQUE_CASE_COUNT = 11
T10_SYSTEM_RESULT_COUNT = 22


@dataclass(frozen=True, slots=True)
class PhaseGateValidationInput:
    bundle: EvaluationReviewBundle
    gate: EvaluationPhaseGate
    review_evidence: PhaseReviewEvidence | None


@dataclass(frozen=True, slots=True)
class PhaseReviewEvidence:
    plan: AIReviewPlan
    occurrence: PhaseReviewOccurrenceRecord
    approval: ExternalCallApprovalRecord
    budget: BudgetSpecV2
    batch_results: tuple[AIReviewBatchResult, ...]
    integrity_result: AIReviewIntegrityResult
    ordered_ai_history: tuple[AIQualityReviewRecord, ...]
    attempts: tuple[AttemptRecordV2, ...]
    usage_records: tuple[TokenUsageRecordV2 | TokenUsageRecordV3, ...]
    resource_records: tuple[ResourceUsageRecord, ...]
    cost_records: tuple[CostRecord, ...]
    human_key_binding: HumanReviewKeyBinding
    human_decision_bytes: bytes
    human_signature_base64: str


def validate_t10_phase_gate(
    bundle: EvaluationReviewBundle,
    gate: EvaluationPhaseGate,
    *,
    review_evidence: PhaseReviewEvidence | None = None,
) -> ValidationResult:
    return _validate_t10_phase_gate_for_test(
        bundle,
        gate,
        review_evidence=review_evidence,
        profile=None,
        registry=None,
    )


def _validate_t10_phase_gate_for_test(
    bundle: EvaluationReviewBundle,
    gate: EvaluationPhaseGate,
    *,
    review_evidence: PhaseReviewEvidence | None,
    profile: ValidationProfile | None = None,
    registry: RuleRegistry | None = None,
) -> ValidationResult:
    selected_profile = profile or t10_phase_gate_profile()
    selected_registry = registry or phase_gate_registry()
    target = PhaseGateValidationInput(
        bundle=bundle,
        gate=gate,
        review_evidence=review_evidence,
    )
    target_hash = _phase_gate_target_hash(target)
    return validate_closed_profile(
        target,
        target_hash=target_hash,
        profile=selected_profile,
        expected_stage=ValidationStage.EVIDENCE,
        expected_inventory=T8_T10_PHASE_GATE_RULE_INVENTORY,
        registry=selected_registry,
    )


def _phase_gate_target_hash(target: PhaseGateValidationInput) -> str:
    evidence = target.review_evidence
    evidence_binding: object
    if evidence is None:
        evidence_binding = None
    else:
        evidence_binding = (
            _hashable_evidence_value(evidence.plan),
            _hashable_evidence_value(evidence.occurrence),
            _hashable_evidence_value(evidence.approval),
            _hashable_evidence_value(evidence.budget),
            _hashable_evidence_value(evidence.batch_results),
            _hashable_evidence_value(evidence.integrity_result),
            _hashable_evidence_value(evidence.ordered_ai_history),
            _hashable_evidence_value(evidence.attempts),
            _hashable_evidence_value(evidence.usage_records),
            _hashable_evidence_value(evidence.resource_records),
            _hashable_evidence_value(evidence.cost_records),
            _hashable_evidence_value(evidence.human_key_binding),
            hashlib.sha256(evidence.human_decision_bytes).hexdigest(),
            evidence.human_signature_base64,
        )
    return canonical_sha256(
        [
            "oamb-t10-phase-gate-validation-target-v2",
            target.bundle,
            target.gate,
            evidence_binding,
        ]
    )


def _hashable_evidence_value(value: object) -> object:
    try:
        canonical_sha256(["oamb-phase-evidence-component-v1", value])
    except (TypeError, ValueError):
        return (
            "invalid-evidence-component",
            f"{type(value).__module__}.{type(value).__qualname__}",
        )
    return value


def phase_gate_registry(*, exclude: set[str] | None = None) -> RuleRegistry:
    omitted = exclude or set()
    registry = RuleRegistry()
    for rule in PHASE_GATE_RULES:
        if rule.rule_id not in omitted:
            registry.register(rule)
    return registry


def _schema_identity_rule(target: PhaseGateValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "phase.schema-identity.v1"
    issues: list[ValidationIssue] = []
    for label, model in (("review-bundle", target.bundle), ("phase-gate", target.gate)):
        try:
            type(model).model_validate(model.model_dump(mode="python"))
        except Exception:
            issues.append(_issue(rule_id, label, "phase-contract-invalid"))
    return tuple(issues)


def _root_coverage_rule(target: PhaseGateValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "phase.root-coverage.v1"
    bundle = target.bundle
    if (
        bundle.phase_id != T10_PHASE_ID
        or len(bundle.ordered_capsule_hashes) != T10_CAPSULE_COUNT
        or len(bundle.ordered_validation_hashes) != T10_CAPSULE_COUNT
        or len(set(bundle.ordered_capsule_hashes)) != T10_CAPSULE_COUNT
        or len(set(bundle.ordered_validation_hashes)) != T10_CAPSULE_COUNT
    ):
        return (_issue(rule_id, bundle.phase_id, "phase-root-coverage-mismatch"),)
    return ()


def _case_coverage_rule(target: PhaseGateValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "phase.case-coverage.v1"
    bundle = target.bundle
    if (
        len(bundle.ordered_case_occurrence_ids) != T10_SYSTEM_RESULT_COUNT
        or len(set(bundle.ordered_case_occurrence_ids)) != T10_SYSTEM_RESULT_COUNT
        or len(bundle.unique_case_manifest_entry_ids) != T10_UNIQUE_CASE_COUNT
        or len(set(bundle.unique_case_manifest_entry_ids)) != T10_UNIQUE_CASE_COUNT
    ):
        return (_issue(rule_id, bundle.bundle_id, "phase-case-coverage-mismatch"),)
    return ()


def _derivation_export_rule(target: PhaseGateValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "phase.derivation-export.v1"
    bundle = target.bundle
    expected_input_hash = evaluation_review_input_hash(
        phase_id=bundle.phase_id,
        ordered_capsule_hashes=bundle.ordered_capsule_hashes,
        ordered_validation_hashes=bundle.ordered_validation_hashes,
        ordered_case_occurrence_ids=bundle.ordered_case_occurrence_ids,
        unique_case_manifest_entry_ids=bundle.unique_case_manifest_entry_ids,
        ordinary_derivation_hashes=bundle.ordinary_derivation_hashes,
        report_model_hash=bundle.report_model_hash,
        report_html_hash=bundle.report_html_hash,
        export_validation_hash=bundle.export_validation_hash,
        limitations=bundle.limitations,
    )
    expected_bundle_id = evaluation_review_bundle_id(ordered_review_input_hash=expected_input_hash)
    if (
        not bundle.ordinary_derivation_hashes
        or len(set(bundle.ordinary_derivation_hashes)) != len(bundle.ordinary_derivation_hashes)
        or bundle.ordered_review_input_hash != expected_input_hash
        or bundle.bundle_id != expected_bundle_id
    ):
        return (_issue(rule_id, bundle.bundle_id, "phase-derivation-export-mismatch"),)
    return ()


def _review_order_rule(target: PhaseGateValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "phase.review-order.v1"
    bundle = target.bundle
    gate = target.gate
    if (
        gate.phase_id != bundle.phase_id
        or gate.review_bundle_hash != bundle.bundle_id
        or not gate.ordered_ai_review_record_hashes
        or gate.ordered_ai_review_record_hashes[-1] != gate.canonical_ai_review_record_hash
        or gate.ai_record.ai_review_record_id != gate.canonical_ai_review_record_hash
        or gate.human_record is None
        or gate.human_review_record_hash != gate.human_record.human_review_record_id
    ):
        return (_issue(rule_id, gate.gate_id, "phase-review-bundle-mismatch"),)
    if target.review_evidence is None:
        return (_issue(rule_id, gate.gate_id, "phase-review-evidence-missing"),)
    if not _review_source_evidence_closes(target):
        return (_issue(rule_id, gate.gate_id, "phase-review-source-evidence-mismatch"),)
    return ()


def _human_approval_rule(target: PhaseGateValidationInput) -> tuple[ValidationIssue, ...]:
    rule_id = "phase.human-approval.v1"
    bundle = target.bundle
    gate = target.gate
    expected_coverage = canonical_sha256(
        ["oamb-ai-review-case-coverage-v1", bundle.ordered_case_occurrence_ids]
    )
    evidence = target.review_evidence
    if (
        gate.ai_record.status != QualityReviewStatus.PASS
        or not gate.ai_record.accounting_closed
        or gate.ai_record.case_coverage_hash != expected_coverage
        or gate.human_record is None
        or gate.human_record.status != "pass"
        or not gate.passed_by_ai
        or not gate.passed_by_human
    ):
        return (_issue(rule_id, gate.gate_id, "phase-human-approval-mismatch"),)
    assert gate.human_record is not None
    if evidence is None or not verify_human_review_record_evidence(
        record=gate.human_record,
        decision_bytes=evidence.human_decision_bytes,
        signature_base64=evidence.human_signature_base64,
        key_binding=evidence.human_key_binding,
        canonical_ai_record=gate.ai_record,
    ):
        return (_issue(rule_id, gate.gate_id, "phase-human-signature-evidence-mismatch"),)
    return ()


def _review_source_evidence_closes(target: PhaseGateValidationInput) -> bool:
    evidence = target.review_evidence
    if evidence is None:
        return False
    try:
        plan = evidence.plan
        occurrence = evidence.occurrence
        approval = evidence.approval
        budget = evidence.budget
        ai_record = target.gate.ai_record
        ai_history = evidence.ordered_ai_history
        validate_ai_review_history(target.bundle.bundle_id, ai_history)
        ai_history_ids = tuple(record.ai_review_record_id for record in ai_history)
        attempt_ids = tuple(attempt.attempt_id for attempt in evidence.attempts)
        usage_ids = tuple(record.usage_record_id for record in evidence.usage_records)
        resource_ids = tuple(record.resource_record_id for record in evidence.resource_records)
        cost_ids = tuple(record.cost_record_id for record in evidence.cost_records)
        role_ceiling_ids = tuple(item.role_binding_id for item in budget.role_ceilings)
        batch_result_hashes = tuple(canonical_sha256(item) for item in evidence.batch_results)
        expected_request_fingerprints = tuple(
            batch.request_fingerprint for batch in plan.case_batches
        ) + (plan.phase_integrity_request_fingerprint,)
        attempts_closed = bool(
            attempt_ids == ai_record.attempt_ids
            and len(attempt_ids) == plan.expected_attempt_count
            and tuple(attempt.ordinal for attempt in evidence.attempts)
            == tuple(range(1, plan.expected_attempt_count + 1))
            and tuple(attempt.request_fingerprint for attempt in evidence.attempts)
            == expected_request_fingerprints
            and all(
                attempt.parent_kind == "phase_review"
                and attempt.parent_id == occurrence.phase_review_occurrence_id
                and attempt.stage == "quality_review"
                and attempt.attempt_id
                == derive_attempt_id(
                    occurrence.phase_review_occurrence_id,
                    "quality_review",
                    attempt.ordinal,
                    attempt.request_fingerprint,
                )
                and attempt.outcome == AttemptOutcome.SUCCEEDED
                and attempt.raw_response_ref is not None
                and attempt.raw_error_ref is None
                and _attempt_time_closes(attempt, occurrence, approval)
                for attempt in evidence.attempts
            )
        )
        usage_closed = bool(
            usage_ids == ai_record.usage_record_ids
            and {record.attempt_id for record in evidence.usage_records} == set(attempt_ids)
            and all(
                record.parent_kind == "phase_review"
                and record.parent_id == occurrence.phase_review_occurrence_id
                and record.stage == TokenStageV2.QUALITY_REVIEW
                and record.proof_status == ProofStatus.MEASURED_COMPLETE
                for record in evidence.usage_records
            )
        )
        resources_closed = bool(
            resource_ids == ai_record.resource_record_ids
            and resource_ids
            and all(
                record.parent_kind == "phase_review"
                and record.parent_id == occurrence.phase_review_occurrence_id
                and record.stage == "quality_review"
                and record.proof_status == ProofStatus.MEASURED_COMPLETE
                for record in evidence.resource_records
            )
        )
        costs_closed = bool(
            cost_ids == ai_record.cost_record_ids
            and cost_ids
            and all(
                record.parent_kind == "phase_review"
                and record.parent_id == occurrence.phase_review_occurrence_id
                and record.proof_status == ProofStatus.MEASURED_COMPLETE
                and set(record.source_usage_record_ids) <= set(usage_ids)
                and set(record.source_resource_record_ids) <= set(resource_ids)
                for record in evidence.cost_records
            )
        )
        return bool(
            plan.review_bundle_hash == target.bundle.bundle_id
            and ai_history_ids == target.gate.ordered_ai_review_record_hashes
            and ai_history[-1] == ai_record
            and ai_record.review_plan_hash == plan.plan_hash
            and ai_record.projection_spec_hash == plan.projection_spec_hash
            and ai_record.finding_registry_hash == plan.finding_registry_hash
            and ai_record.prompt_pack_hash == plan.prompt_pack_hash
            and ai_record.output_contract_hash == plan.output_contract_hash
            and ai_record.parser_hash == plan.parser_hash
            and ai_record.reviewer_role_binding_hash == plan.reviewer_role_binding_hash
            and ai_record.reviewer_model_hash == plan.reviewer_model_hash
            and ai_record.reviewer_runtime_hash == plan.reviewer_runtime_hash
            and ai_record.reviewer_configuration_hash == plan.reviewer_configuration_hash
            and ai_record.case_coverage_hash == plan.case_coverage_hash
            and ai_record.batch_result_hashes == batch_result_hashes
            and ai_record.integrity_result_hash == canonical_sha256(evidence.integrity_result)
            and tuple(result.batch_id for result in evidence.batch_results)
            == tuple(batch.batch_id for batch in plan.case_batches)
            and all(result.status == QualityReviewStatus.PASS for result in evidence.batch_results)
            and evidence.integrity_result.integrity_id == plan.phase_integrity_id
            and evidence.integrity_result.status == QualityReviewStatus.PASS
            and occurrence.phase_id == target.bundle.phase_id
            and occurrence.review_bundle_hash == target.bundle.bundle_id
            and occurrence.reviewer_role_binding_hash == plan.reviewer_role_binding_hash
            and occurrence.ordinal == ai_record.ordinal
            and occurrence.phase_review_occurrence_id == ai_record.occurrence_id
            and occurrence.approval_record_id == approval.approval_hash
            and occurrence.budget_id == budget.budget_id
            and occurrence.state == "sealed"
            and occurrence.started_at is not None
            and occurrence.ended_at is not None
            and occurrence.started_at <= occurrence.ended_at <= ai_record.created_at
            and approval.operation_kind == "phase_review"
            and approval.scope_kind == BudgetScopeKindV2.PHASE_REVIEW
            and approval.scope_id == occurrence.phase_review_occurrence_id
            and approval.role_binding_ids == (plan.reviewer_role_binding_hash,)
            and approval.budget_hash == canonical_sha256(budget)
            and approval.approved_at <= occurrence.started_at < approval.expires_at
            and budget.scope_kind == BudgetScopeKindV2.PHASE_REVIEW
            and budget.scope_id == occurrence.phase_review_occurrence_id
            and budget.approval_id == approval.approval_id
            and budget.max_attempts >= plan.expected_attempt_count
            and role_ceiling_ids == (plan.reviewer_role_binding_hash,)
            and budget.role_ceilings[0].max_attempts >= plan.expected_attempt_count
            and _review_budget_closes(evidence)
            and ai_record.accounting_closed
            and attempts_closed
            and usage_closed
            and resources_closed
            and costs_closed
        )
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


def _review_budget_closes(evidence: PhaseReviewEvidence) -> bool:
    plan = evidence.plan
    budget = evidence.budget
    if len(budget.role_ceilings) != 1:
        return False
    role = budget.role_ceilings[0]
    planned_inputs = tuple(batch.input_tokens for batch in plan.case_batches) + (
        plan.phase_integrity_input_tokens,
    )
    planned_outputs = tuple(batch.maximal_output_tokens for batch in plan.case_batches) + (
        plan.phase_integrity_maximal_output_tokens,
    )
    if (
        len(planned_inputs) != plan.expected_attempt_count
        or any(
            input_tokens + output_tokens > plan.model_context_window_tokens
            for input_tokens, output_tokens in zip(planned_inputs, planned_outputs, strict=True)
        )
        or sum(planned_inputs) > budget.max_input_tokens
        or sum(planned_inputs) > role.max_input_tokens
        or sum(planned_outputs) > budget.max_output_tokens
        or sum(planned_outputs) > role.max_output_tokens
    ):
        return False

    attempt_ids = tuple(item.attempt_id for item in evidence.attempts)
    raw_response_refs = tuple(item.raw_response_ref for item in evidence.attempts)
    usage_attempt_ids = tuple(item.attempt_id for item in evidence.usage_records)
    usage_ids = tuple(item.usage_record_id for item in evidence.usage_records)
    resource_ids = tuple(item.resource_record_id for item in evidence.resource_records)
    cost_ids = tuple(item.cost_record_id for item in evidence.cost_records)
    if (
        len(attempt_ids) != len(set(attempt_ids))
        or len(usage_ids) != len(set(usage_ids))
        or len(resource_ids) != len(set(resource_ids))
        or len(cost_ids) != len(set(cost_ids))
        or usage_attempt_ids != attempt_ids
        or tuple(item.raw_telemetry_ref for item in evidence.resource_records) != raw_response_refs
    ):
        return False
    if any(
        item.input_tokens is None
        or item.visible_output_tokens is None
        or item.supplier_reported_total_tokens is None
        or item.supplier_reported_total_tokens < item.input_tokens + item.visible_output_tokens
        for item in evidence.usage_records
    ):
        return False
    if len(evidence.usage_records) != len(planned_inputs) or any(
        record.input_tokens is None
        or record.visible_output_tokens is None
        or record.input_tokens > planned_input_tokens
        or record.visible_output_tokens > planned_output_tokens
        for record, planned_input_tokens, planned_output_tokens in zip(
            evidence.usage_records,
            planned_inputs,
            planned_outputs,
            strict=True,
        )
    ):
        return False

    input_tokens = sum(item.input_tokens or 0 for item in evidence.usage_records)
    output_tokens = sum(item.visible_output_tokens or 0 for item in evidence.usage_records)
    wall_seconds = sum(
        (_exact_seconds(item.ended_at - item.started_at) for item in evidence.attempts),
        Decimal("0"),
    )
    cost_amount = sum(
        (item.amount or Decimal("0") for item in evidence.cost_records),
        Decimal("0"),
    )
    if (
        len(attempt_ids) > budget.max_attempts
        or len(attempt_ids) > role.max_attempts
        or input_tokens > budget.max_input_tokens
        or input_tokens > role.max_input_tokens
        or output_tokens > budget.max_output_tokens
        or output_tokens > role.max_output_tokens
        or wall_seconds > budget.max_dispatch_wall_seconds
        or wall_seconds > role.max_dispatch_wall_seconds
        or budget.max_cost is None
        or role.max_cost is None
        or cost_amount > budget.max_cost
        or cost_amount > role.max_cost
        or any(item.currency != budget.currency for item in evidence.cost_records)
        or Decimal(len(attempt_ids)) > role.provider_budget_cap.maximum_accepted_units
        or role.provider_budget_cap.operation_kind != "quality_review"
        or role.provider_budget_cap.billing_unit != "request"
    ):
        return False

    parent_resource_ceilings = {
        item.dimension_id: (item.maximum, item.unit) for item in budget.resource_ceilings
    }
    role_resource_ceilings = {
        item.dimension_id: (item.maximum, item.unit) for item in role.resource_ceilings
    }
    resource_totals: dict[str, Decimal] = {}
    for record in evidence.resource_records:
        if (
            record.value is None
            or record.dimension_id not in parent_resource_ceilings
            or record.dimension_id not in role_resource_ceilings
            or record.unit != parent_resource_ceilings[record.dimension_id][1]
            or record.unit != role_resource_ceilings[record.dimension_id][1]
        ):
            return False
        resource_totals[record.dimension_id] = (
            resource_totals.get(record.dimension_id, Decimal("0")) + record.value
        )
    if (
        set(resource_totals) != set(parent_resource_ceilings)
        or set(resource_totals) != set(role_resource_ceilings)
        or any(
            value > parent_resource_ceilings[dimension_id][0]
            or value > role_resource_ceilings[dimension_id][0]
            for dimension_id, value in resource_totals.items()
        )
    ):
        return False

    attributed_usage_ids = tuple(
        source_id
        for record in evidence.cost_records
        for source_id in record.source_usage_record_ids
    )
    attributed_resource_ids = tuple(
        source_id
        for record in evidence.cost_records
        for source_id in record.source_resource_record_ids
    )
    return bool(
        sorted(attributed_usage_ids) == sorted(usage_ids)
        and sorted(attributed_resource_ids) == sorted(resource_ids)
        and len(attributed_usage_ids) == len(set(attributed_usage_ids))
        and len(attributed_resource_ids) == len(set(attributed_resource_ids))
    )


def _attempt_time_closes(
    attempt: AttemptRecordV2,
    occurrence: PhaseReviewOccurrenceRecord,
    approval: ExternalCallApprovalRecord,
) -> bool:
    return bool(
        occurrence.started_at is not None
        and occurrence.ended_at is not None
        and approval.approved_at <= occurrence.started_at
        and occurrence.ended_at <= approval.expires_at
        and occurrence.started_at <= attempt.started_at
        and attempt.ended_at <= occurrence.ended_at
    )


def _exact_seconds(duration: timedelta) -> Decimal:
    days = duration.days
    seconds = duration.seconds
    microseconds = duration.microseconds
    return Decimal(days * 86_400 + seconds) + Decimal(microseconds) / Decimal(1_000_000)


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


PHASE_GATE_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("phase.schema-identity.v1", 1, _schema_identity_rule),
    ValidationRule("phase.root-coverage.v1", 1, _root_coverage_rule),
    ValidationRule("phase.case-coverage.v1", 1, _case_coverage_rule),
    ValidationRule("phase.derivation-export.v1", 1, _derivation_export_rule),
    ValidationRule("phase.review-order.v1", 1, _review_order_rule),
    ValidationRule("phase.human-approval.v1", 1, _human_approval_rule),
)


__all__ = [
    "T10_PHASE_GATE_RULE_IDS",
    "phase_gate_registry",
    "validate_t10_phase_gate",
]
