from __future__ import annotations

import base64
import importlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import ModuleType
from typing import Any, cast

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from oamb.contracts.accounting import (
    CostBasis,
    CostRecord,
    IndexingView,
    ProofStatus,
    ResourceUsageRecord,
    TokenDomain,
    TokenMeasurementSource,
    TokenStageV2,
    TokenUsageRecordV2,
)
from oamb.contracts.evidence import (
    AttemptRecordV2,
    PhaseReviewOccurrenceRecord,
    phase_review_occurrence_id,
)
from oamb.contracts.ids import attempt_id, canonical_sha256
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    AIReviewBatchResult,
    AIReviewCaseResult,
    AIReviewIntegrityResult,
    EvaluationPhaseGate,
    EvaluationReviewBundle,
    HumanQualityReviewRecord,
    QualityReviewStatus,
    SignatureVerificationRecord,
    ai_quality_review_record_identity,
    evaluation_phase_gate_id,
    human_quality_review_record_id,
    signature_verification_record_id,
)
from oamb.contracts.specifications import (
    AIReviewBatch,
    AIReviewPlan,
    BudgetScopeKindV2,
    BudgetSpecV2,
    ExternalCallApprovalRecord,
    ProviderBudgetCap,
    ResourceBudgetCeiling,
    RoleBudgetCeiling,
    ai_review_plan_hash,
    external_call_approval_hash,
)
from oamb.contracts.states import (
    AttemptOutcome,
    IndexContribution,
    ValidationDisposition,
)
from oamb.reporting.human_review import (
    build_human_review_key_binding,
    derive_evaluation_phase_gate,
    import_human_review_decision,
    prepare_human_review_decision,
)
from oamb.reporting.review import build_evaluation_review_bundle

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
NOW = datetime(2026, 8, 28, tzinfo=UTC)


def _phase() -> ModuleType:
    try:
        return importlib.import_module("oamb.artifacts.validation.phase")
    except ModuleNotFoundError:
        pytest.fail("phase-gate validation is not implemented", pytrace=False)


def _bundle(
    ordered_case_occurrence_ids: tuple[str, ...] | None = None,
) -> EvaluationReviewBundle:
    return build_evaluation_review_bundle(
        phase_id="phase_1_smoke_acceptance",
        ordered_capsule_hashes=(SHA_A, SHA_B, SHA_C, SHA_D),
        ordered_validation_hashes=(SHA_D, SHA_C, SHA_B, SHA_A),
        ordered_case_occurrence_ids=(
            ordered_case_occurrence_ids
            if ordered_case_occurrence_ids is not None
            else tuple(f"{index:064x}" for index in range(1, 23))
        ),
        unique_case_manifest_entry_ids=tuple(f"{index:064x}" for index in range(101, 112)),
        ordinary_derivation_hashes=(SHA_A, SHA_B),
        report_model_hash=SHA_B,
        report_html_hash=SHA_C,
        export_validation_hash=SHA_D,
        limitations=("fixture-only phase",),
    )


def _accepted_gate(bundle: EvaluationReviewBundle) -> EvaluationPhaseGate:
    review_bundle_hash = bundle.bundle_id
    ai_fields = {
        "review_bundle_hash": review_bundle_hash,
        "review_plan_hash": SHA_C,
        "projection_spec_hash": SHA_A,
        "finding_registry_hash": SHA_B,
        "prompt_pack_hash": SHA_C,
        "output_contract_hash": SHA_D,
        "parser_hash": SHA_A,
        "aggregate_version": "ai-quality-aggregate-v1",
        "reviewer_role_binding_hash": SHA_B,
        "reviewer_model_hash": SHA_C,
        "reviewer_runtime_hash": SHA_D,
        "reviewer_configuration_hash": SHA_A,
        "occurrence_id": SHA_D,
        "ordinal": 1,
        "previous_ai_review_record_hash": None,
        "previous_history_root_hash": None,
        "case_coverage_hash": canonical_sha256(
            ["oamb-ai-review-case-coverage-v1", bundle.ordered_case_occurrence_ids]
        ),
        "batch_result_hashes": (SHA_A, SHA_B),
        "integrity_result_hash": SHA_C,
        "status": QualityReviewStatus.PASS,
        "review_outcome_kind": "complete",
        "finding_codes": (),
        "attempt_ids": (SHA_A, SHA_B, SHA_C),
        "usage_record_ids": (SHA_A,),
        "resource_record_ids": (SHA_B,),
        "cost_record_ids": (SHA_C,),
        "accounting_closed": True,
        "created_at": NOW,
    }
    ai_id = ai_quality_review_record_identity(ai_fields)
    ai_record = AIQualityReviewRecord.model_validate(
        {
            "ai_review_record_id": ai_id,
            "history_root_hash": canonical_sha256(["oamb-ai-review-history-v1", None, ai_id]),
            **ai_fields,
        }
    )
    verification_fields = {
        "key_binding_id": SHA_A,
        "trusted_key_fingerprint": f"SHA256:{SHA_B}",
        "public_key_sha256": SHA_B,
        "signed_payload_sha256": SHA_C,
        "signature_sha256": SHA_D,
        "verified_at": NOW,
    }
    verification = SignatureVerificationRecord.model_validate(
        {
            "verification_id": signature_verification_record_id(**verification_fields),
            **verification_fields,
        }
    )
    human_fields = {
        "review_bundle_hash": review_bundle_hash,
        "ai_review_record_hash": ai_id,
        "decision_hash": SHA_A,
        "signature_verification": verification,
        "status": "pass",
        "finding_codes": (),
        "evidence_references": (),
        "reviewer_label": "fixture-operator",
        "decision_nonce": "fixture-nonce",
        "trusted_key_fingerprint": f"SHA256:{SHA_B}",
        "created_at": NOW,
    }
    human_record = HumanQualityReviewRecord.model_validate(
        {
            "human_review_record_id": human_quality_review_record_id(**human_fields),
            **human_fields,
        }
    )
    return derive_evaluation_phase_gate(
        phase_id=bundle.phase_id,
        review_bundle_hash=review_bundle_hash,
        ordered_ai_history=(ai_record,),
        human_records=(human_record,),
    )


def _phase_review_plan(bundle: EvaluationReviewBundle) -> AIReviewPlan:
    case_groups = (
        bundle.ordered_case_occurrence_ids[:16],
        bundle.ordered_case_occurrence_ids[16:],
    )
    batches = tuple(
        AIReviewBatch(
            batch_id=canonical_sha256(
                [
                    "oamb-ai-review-batch-v1",
                    bundle.bundle_id,
                    case_ids,
                    payload_hash,
                ]
            ),
            ordered_case_occurrence_ids=case_ids,
            payload_hash=payload_hash,
            request_fingerprint=canonical_sha256(["fixture-ai-review-request-v1", payload_hash]),
            input_bytes=1_000,
            input_tokens=250,
            maximal_output_bytes=2_000,
            maximal_output_tokens=500,
        )
        for case_ids, payload_hash in (
            (case_groups[0], SHA_A),
            (case_groups[1], SHA_B),
        )
    )
    coverage_hash = canonical_sha256(
        ["oamb-ai-review-case-coverage-v1", bundle.ordered_case_occurrence_ids]
    )
    phase_integrity_payload_hash = SHA_C
    phase_integrity_id = canonical_sha256(
        ["oamb-ai-review-integrity-v1", bundle.bundle_id, phase_integrity_payload_hash]
    )
    phase_integrity_request_fingerprint = canonical_sha256(
        ["fixture-ai-review-integrity-request-v1", phase_integrity_payload_hash]
    )
    fields = {
        "review_bundle_hash": bundle.bundle_id,
        "projection_spec_hash": SHA_A,
        "finding_registry_hash": SHA_B,
        "prompt_pack_hash": SHA_C,
        "output_contract_hash": SHA_D,
        "parser_hash": SHA_A,
        "reviewer_role_binding_hash": SHA_B,
        "reviewer_model_hash": SHA_C,
        "reviewer_runtime_hash": SHA_D,
        "reviewer_configuration_hash": SHA_A,
        "reviewer_counter_fingerprint": SHA_D,
        "model_context_window_tokens": 32_768,
        "ordered_case_occurrence_ids": bundle.ordered_case_occurrence_ids,
        "case_batches": batches,
        "case_coverage_hash": coverage_hash,
        "phase_integrity_id": phase_integrity_id,
        "phase_integrity_payload_hash": phase_integrity_payload_hash,
        "phase_integrity_request_fingerprint": phase_integrity_request_fingerprint,
        "phase_integrity_input_bytes": 1_000,
        "phase_integrity_input_tokens": 250,
        "phase_integrity_maximal_output_bytes": 2_000,
        "phase_integrity_maximal_output_tokens": 500,
        "expected_attempt_count": len(batches) + 1,
        "aggregate_version": "ai-quality-aggregate-v1",
    }
    plan_hash = ai_review_plan_hash(
        review_bundle_hash=bundle.bundle_id,
        projection_spec_hash=SHA_A,
        finding_registry_hash=SHA_B,
        prompt_pack_hash=SHA_C,
        output_contract_hash=SHA_D,
        parser_hash=SHA_A,
        reviewer_role_binding_hash=SHA_B,
        reviewer_model_hash=SHA_C,
        reviewer_runtime_hash=SHA_D,
        reviewer_configuration_hash=SHA_A,
        reviewer_counter_fingerprint=SHA_D,
        model_context_window_tokens=32_768,
        ordered_case_occurrence_ids=bundle.ordered_case_occurrence_ids,
        case_batches=batches,
        case_coverage_hash=coverage_hash,
        phase_integrity_id=phase_integrity_id,
        phase_integrity_payload_hash=phase_integrity_payload_hash,
        phase_integrity_request_fingerprint=phase_integrity_request_fingerprint,
        phase_integrity_input_bytes=1_000,
        phase_integrity_input_tokens=250,
        phase_integrity_maximal_output_bytes=2_000,
        phase_integrity_maximal_output_tokens=500,
        expected_attempt_count=len(batches) + 1,
        aggregate_version="ai-quality-aggregate-v1",
    )
    return AIReviewPlan.model_validate({"plan_hash": plan_hash, **fields})


def _accepted_gate_with_evidence() -> tuple[EvaluationReviewBundle, EvaluationPhaseGate, Any]:
    review = importlib.import_module("oamb.reporting.review")
    bundle = _bundle()
    plan = _phase_review_plan(bundle)
    occurrence_id = phase_review_occurrence_id(
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        reviewer_role_binding_hash=plan.reviewer_role_binding_hash,
        ordinal=1,
    )
    resource_ceiling = ResourceBudgetCeiling(
        dimension_id="provider_request_wall_seconds_v1",
        maximum=Decimal("30"),
        unit="seconds",
    )
    role_ceiling = RoleBudgetCeiling(
        role_binding_id=plan.reviewer_role_binding_hash,
        max_attempts=plan.expected_attempt_count,
        max_input_tokens=1_000,
        max_output_tokens=2_000,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=Decimal("1"),
        currency="USD",
        price_snapshot_id=None,
        resource_ceilings=(resource_ceiling,),
        provider_budget_cap=ProviderBudgetCap(
            provider="fixture-reviewer",
            operation_kind="quality_review",
            billing_unit="request",
            maximum_accepted_units=Decimal(plan.expected_attempt_count),
        ),
    )
    budget = BudgetSpecV2(
        budget_id="phase-review-budget-v1",
        scope_kind=BudgetScopeKindV2.PHASE_REVIEW,
        scope_id=occurrence_id,
        approval_id="phase-review-approval-v1",
        max_attempts=plan.expected_attempt_count,
        max_input_tokens=1_000,
        max_output_tokens=2_000,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=Decimal("1"),
        currency="USD",
        resource_ceilings=(resource_ceiling,),
        role_ceilings=(role_ceiling,),
        stop_condition_ids=("budget_exhausted",),
    )
    approval_fields = {
        "approval_id": budget.approval_id,
        "operation_kind": "phase_review",
        "scope_kind": BudgetScopeKindV2.PHASE_REVIEW,
        "scope_id": occurrence_id,
        "runtime_binding_hash": None,
        "provider_runtime_profile_attestation_hash": None,
        "role_binding_ids": (plan.reviewer_role_binding_hash,),
        "budget_hash": canonical_sha256(budget),
        "approved_at": NOW,
        "expires_at": NOW + timedelta(hours=1),
        "unmetered_cost_acknowledged": False,
        "stop_condition_ids": budget.stop_condition_ids,
    }
    approval = ExternalCallApprovalRecord.model_validate(
        {
            "approval_hash": external_call_approval_hash(approval_fields),
            **approval_fields,
        }
    )
    occurrence = PhaseReviewOccurrenceRecord(
        phase_review_occurrence_id=occurrence_id,
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        reviewer_role_binding_hash=plan.reviewer_role_binding_hash,
        ordinal=1,
        approval_record_id=approval.approval_hash,
        budget_id=budget.budget_id,
        state="sealed",
        started_at=NOW + timedelta(minutes=1),
        ended_at=NOW + timedelta(minutes=2),
    )
    request_fingerprints = tuple(batch.request_fingerprint for batch in plan.case_batches) + (
        plan.phase_integrity_request_fingerprint,
    )
    attempt_ids = tuple(
        attempt_id(occurrence_id, "quality_review", index, request_fingerprint)
        for index, request_fingerprint in enumerate(request_fingerprints, 1)
    )
    attempts = tuple(
        AttemptRecordV2(
            attempt_id=attempt_id,
            parent_kind="phase_review",
            parent_id=occurrence_id,
            stage="quality_review",
            ordinal=index,
            request_fingerprint=request_fingerprint,
            started_at=NOW + timedelta(minutes=1, seconds=index),
            ended_at=NOW + timedelta(minutes=1, seconds=index + 1),
            outcome=AttemptOutcome.SUCCEEDED,
            retry_of_attempt_id=None,
            idempotency_key_hash=None,
            reconciliation_capability="none",
            raw_response_ref=f"{900 + index:064x}",
            raw_error_ref=None,
            index_contribution=IndexContribution.NONE,
            superseded_by_attempt_id=None,
        )
        for index, (attempt_id, request_fingerprint) in enumerate(
            zip(attempt_ids, request_fingerprints, strict=True),
            1,
        )
    )
    usage_records = tuple(
        TokenUsageRecordV2(
            usage_record_id=f"{1_000 + index:064x}",
            attempt_id=attempt.attempt_id,
            parent_kind="phase_review",
            parent_id=occurrence_id,
            stage=TokenStageV2.QUALITY_REVIEW,
            operation_kind="quality_review",
            token_domain=TokenDomain.EXTERNAL_LLM,
            measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
            input_tokens=10,
            visible_output_tokens=5,
            supplier_reported_total_tokens=15,
            context_view_tokens=None,
            proof_status=ProofStatus.MEASURED_COMPLETE,
            reason=None,
            raw_response_ref=attempt.raw_response_ref,
        )
        for index, attempt in enumerate(attempts, 1)
    )
    resource_records = tuple(
        ResourceUsageRecord(
            resource_record_id=f"{1_100 + index:064x}",
            parent_kind="phase_review",
            parent_id=occurrence_id,
            stage="quality_review",
            meter_boundary="provider_request_wall_seconds_v1",
            dimension_id="provider_request_wall_seconds_v1",
            value=Decimal("1"),
            unit="seconds",
            measurement_source="fixture-clock",
            measurement_spec_id="phase-review-measurement-v1",
            environment_hash=SHA_A,
            started_at=attempt.started_at,
            ended_at=attempt.ended_at,
            raw_telemetry_ref=attempt.raw_response_ref,
            proof_status=ProofStatus.MEASURED_COMPLETE,
            reason=None,
        )
        for index, attempt in enumerate(attempts, 1)
    )
    cost_records = tuple(
        CostRecord(
            cost_record_id=f"{1_200 + index:064x}",
            parent_kind="phase_review",
            parent_id=occurrence_id,
            basis=CostBasis.ACTUAL_SUPPLIER_CHARGE,
            indexing_view=IndexingView.NOT_APPLICABLE,
            amount=Decimal("0.01"),
            currency="USD",
            price_snapshot_id=None,
            source_usage_record_ids=(usage_records[index - 1].usage_record_id,),
            source_resource_record_ids=(resource_records[index - 1].resource_record_id,),
            proof_status=ProofStatus.MEASURED_COMPLETE,
            reason=None,
        )
        for index in range(1, len(attempts) + 1)
    )
    batch_results = tuple(
        AIReviewBatchResult(
            batch_id=batch.batch_id,
            results=tuple(
                AIReviewCaseResult(
                    case_occurrence_id=case_id,
                    status=QualityReviewStatus.PASS,
                    findings=(),
                )
                for case_id in batch.ordered_case_occurrence_ids
            ),
            status=QualityReviewStatus.PASS,
        )
        for batch in plan.case_batches
    )
    integrity_result = AIReviewIntegrityResult(
        integrity_id=plan.phase_integrity_id,
        status=QualityReviewStatus.PASS,
        findings=(),
    )
    ai_record = review.reduce_ai_quality_review(
        plan,
        batch_results=batch_results,
        integrity_result=integrity_result,
        occurrence_id=occurrence_id,
        ordinal=1,
        previous_ai_review_record_hash=None,
        attempt_ids=attempt_ids,
        usage_record_ids=tuple(record.usage_record_id for record in usage_records),
        resource_record_ids=tuple(record.resource_record_id for record in resource_records),
        cost_record_ids=tuple(record.cost_record_id for record in cost_records),
        accounting_closed=True,
        created_at=NOW + timedelta(minutes=3),
    )
    private_key = Ed25519PrivateKey.generate()
    public_key_base64 = base64.b64encode(
        private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    ).decode("ascii")
    key_binding = build_human_review_key_binding(
        source_kind="raw",
        source_reference="phase-fixture",
        public_key_base64=public_key_base64,
    )
    prepared = prepare_human_review_decision(
        review_bundle_hash=bundle.bundle_id,
        ai_review_record_hash=ai_record.ai_review_record_id,
        status="pass",
        decided_at=NOW + timedelta(minutes=4),
        nonce="phase-fixture-nonce",
        reviewer_label="fixture-operator",
        finding_codes=(),
        evidence_references=(),
    )
    signature_base64 = base64.b64encode(private_key.sign(prepared.canonical_bytes)).decode("ascii")
    human_record = import_human_review_decision(
        decision_bytes=prepared.canonical_bytes,
        signature_base64=signature_base64,
        key_binding=key_binding,
        canonical_ai_record=ai_record,
        used_nonces=(),
        imported_at=NOW + timedelta(minutes=5),
    )
    gate = derive_evaluation_phase_gate(
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        ordered_ai_history=(ai_record,),
        human_records=(human_record,),
    )
    evidence = _phase().PhaseReviewEvidence(
        plan=plan,
        occurrence=occurrence,
        approval=approval,
        budget=budget,
        batch_results=batch_results,
        integrity_result=integrity_result,
        ordered_ai_history=(ai_record,),
        attempts=attempts,
        usage_records=usage_records,
        resource_records=resource_records,
        cost_records=cost_records,
        human_key_binding=key_binding,
        human_decision_bytes=prepared.canonical_bytes,
        human_signature_base64=signature_base64,
    )
    return bundle, gate, evidence


def test_t10_phase_profile_executes_the_exact_closed_inventory() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()

    result = phase.validate_t10_phase_gate(bundle, gate, review_evidence=evidence)

    assert result.disposition == ValidationDisposition.VALIDATED, result.issues
    assert result.required_rule_ids == phase.T10_PHASE_GATE_RULE_IDS
    assert result.executed_rule_ids == phase.T10_PHASE_GATE_RULE_IDS
    assert result.missing_rule_ids == ()


def test_t10_phase_profile_rejects_shape_only_review_records_without_source_evidence() -> None:
    phase = _phase()
    bundle = _bundle()
    gate = _accepted_gate(bundle)

    result = phase.validate_t10_phase_gate(bundle, gate)

    assert result.disposition == ValidationDisposition.INVALID
    assert "phase-review-evidence-missing" in {issue.code for issue in result.issues}


def test_t10_phase_profile_rejects_unclosed_review_source_evidence() -> None:
    phase = _phase()
    bundle = _bundle()
    gate = _accepted_gate(bundle)
    placeholder = cast(Any, object())
    evidence = phase.PhaseReviewEvidence(
        plan=placeholder,
        occurrence=placeholder,
        approval=placeholder,
        budget=placeholder,
        batch_results=(),
        integrity_result=placeholder,
        ordered_ai_history=(),
        attempts=(),
        usage_records=(),
        resource_records=(),
        cost_records=(),
        human_key_binding=placeholder,
        human_decision_bytes=b"{}",
        human_signature_base64="invalid",
    )

    result = phase.validate_t10_phase_gate(bundle, gate, review_evidence=evidence)

    assert result.disposition == ValidationDisposition.INVALID
    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in result.issues}


def test_t10_phase_profile_rejects_missing_cost_and_invalid_human_signature() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()

    missing_cost = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(evidence, cost_records=()),
    )
    invalid_signature = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(evidence, human_signature_base64="invalid"),
    )

    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in missing_cost.issues}
    assert "phase-human-signature-evidence-mismatch" in {
        issue.code for issue in invalid_signature.issues
    }


def test_t10_phase_target_hash_binds_review_source_evidence() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()

    accepted = phase.validate_t10_phase_gate(bundle, gate, review_evidence=evidence)
    changed = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(evidence, human_signature_base64="changed"),
    )

    assert accepted.target_hash != changed.target_hash


def test_t10_phase_profile_rejects_gate_history_hidden_from_source_evidence() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    ordered_hashes = ("e" * 64, *gate.ordered_ai_review_record_hashes)
    history_root = canonical_sha256(
        ["oamb-review-history-v1", ordered_hashes, gate.human_review_record_hash]
    )
    gate_fields = {
        "phase_id": gate.phase_id,
        "review_bundle_hash": gate.review_bundle_hash,
        "review_history_root_hash": history_root,
        "ordered_ai_review_record_hashes": ordered_hashes,
        "canonical_ai_review_record_hash": gate.canonical_ai_review_record_hash,
        "human_review_record_hash": gate.human_review_record_hash,
        "passed_by_ai": gate.passed_by_ai,
        "passed_by_human": gate.passed_by_human,
    }
    hidden_history_gate = EvaluationPhaseGate.model_validate(
        {
            "gate_id": evaluation_phase_gate_id(**gate_fields),
            "ai_record": gate.ai_record,
            "human_record": gate.human_record,
            **gate_fields,
        }
    )

    result = phase.validate_t10_phase_gate(
        bundle,
        hidden_history_gate,
        review_evidence=evidence,
    )

    assert result.disposition == ValidationDisposition.INVALID
    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in result.issues}


def test_t10_phase_profile_rejects_forked_ai_history_evidence() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    forked_head = evidence.ordered_ai_history[-1].model_copy(
        update={"previous_ai_review_record_hash": "e" * 64, "ordinal": 2}
    )

    result = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(evidence, ordered_ai_history=(forked_head,)),
    )

    assert result.disposition == ValidationDisposition.INVALID
    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in result.issues}


def test_t10_phase_profile_rejects_actual_token_and_cost_budget_overruns() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    excessive_usage = evidence.usage_records[0].model_copy(
        update={"input_tokens": 1_001, "supplier_reported_total_tokens": 1_006}
    )
    excessive_cost = evidence.cost_records[0].model_copy(update={"amount": Decimal("2")})

    token_overrun = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            usage_records=(excessive_usage, *evidence.usage_records[1:]),
        ),
    )
    cost_overrun = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            cost_records=(excessive_cost, *evidence.cost_records[1:]),
        ),
    )

    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in token_overrun.issues}
    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in cost_overrun.issues}


def test_t10_phase_profile_rejects_resource_and_dispatch_wall_overruns() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    excessive_resource = evidence.resource_records[0].model_copy(update={"value": Decimal("31")})
    excessive_wall = evidence.attempts[0].model_copy(
        update={"ended_at": evidence.attempts[0].started_at + timedelta(seconds=31)}
    )

    resource_overrun = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            resource_records=(excessive_resource, *evidence.resource_records[1:]),
        ),
    )
    wall_overrun = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            attempts=(excessive_wall, *evidence.attempts[1:]),
        ),
    )

    assert "phase-review-source-evidence-mismatch" in {
        issue.code for issue in resource_overrun.issues
    }
    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in wall_overrun.issues}


def test_t10_phase_profile_binds_attempt_requests_and_approval_time_window() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    wrong_request = evidence.attempts[0].model_copy(update={"request_fingerprint": "f" * 64})
    before_approval = evidence.attempts[0].model_copy(
        update={
            "started_at": evidence.approval.approved_at - timedelta(minutes=2),
            "ended_at": evidence.approval.approved_at - timedelta(minutes=1),
        }
    )

    request_drift = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            attempts=(wrong_request, *evidence.attempts[1:]),
        ),
    )
    time_drift = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            attempts=(before_approval, *evidence.attempts[1:]),
        ),
    )

    assert request_drift.disposition == ValidationDisposition.INVALID
    assert time_drift.disposition == ValidationDisposition.INVALID


def test_t10_phase_profile_rejects_per_attempt_token_cap_borrowing() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    excessive_usage = evidence.usage_records[0].model_copy(
        update={
            "input_tokens": evidence.plan.case_batches[0].input_tokens + 1,
            "visible_output_tokens": evidence.plan.case_batches[0].maximal_output_tokens + 1,
            "supplier_reported_total_tokens": (
                evidence.plan.case_batches[0].input_tokens
                + evidence.plan.case_batches[0].maximal_output_tokens
                + 2
            ),
        }
    )

    result = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            usage_records=(excessive_usage, *evidence.usage_records[1:]),
        ),
    )

    assert result.disposition == ValidationDisposition.INVALID


def test_t10_phase_profile_rejects_wrong_coverage_stale_gate_and_missing_rule() -> None:
    phase = _phase()
    bundle = _bundle()
    gate = _accepted_gate(bundle)
    wrong_coverage = bundle.model_copy(
        update={"ordered_case_occurrence_ids": bundle.ordered_case_occurrence_ids[:-1]}
    )

    coverage = phase.validate_t10_phase_gate(wrong_coverage, gate)
    stale = phase.validate_t10_phase_gate(
        bundle,
        gate.model_copy(update={"review_bundle_hash": SHA_A}),
    )
    missing = phase._validate_t10_phase_gate_for_test(
        bundle,
        gate,
        review_evidence=None,
        registry=phase.phase_gate_registry(exclude={phase.T10_PHASE_GATE_RULE_IDS[-1]}),
    )

    assert "phase-case-coverage-mismatch" in {issue.code for issue in coverage.issues}
    assert "phase-review-bundle-mismatch" in {issue.code for issue in stale.issues}
    assert missing.disposition == ValidationDisposition.INVALID
    assert missing.missing_rule_ids == (phase.T10_PHASE_GATE_RULE_IDS[-1],)
