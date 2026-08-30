from __future__ import annotations

import importlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import ModuleType
from typing import Any, cast

import pytest

from oamb.contracts.accounting import (
    AggregationOperator,
    CostBasis,
    CostMeasurementSpec,
    CostRecord,
    IndexingView,
    MeasurementDimensionSpec,
    PriceSnapshot,
    ProofStatus,
    ResourceUsageRecord,
    TokenDomain,
    TokenMeasurementSource,
    TokenStageV2,
    TokenUsageRecordV2,
)
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptReceiptKind,
    AttemptReceiptRecord,
    AttemptRecordV2,
    BudgetReservationRecord,
    OccurrenceClaimRecord,
    PhaseReviewOccurrenceRecordV2,
    RunLeaseRecord,
    phase_review_occurrence_id_v2,
)
from oamb.contracts.ids import attempt_id, canonical_sha256
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    AIReviewBatchResult,
    AIReviewCaseResult,
    AIReviewIntegrityResult,
    EvaluationPhaseGate,
    EvaluationReviewBundle,
    QualityReviewStatus,
    ai_quality_review_record_identity,
    evaluation_phase_gate_id,
)
from oamb.contracts.specifications import (
    AIReviewBatch,
    AIReviewPlan,
    BindingKind,
    BudgetScopeKindV2,
    BudgetSpecV2,
    ComparabilityStatus,
    ExecutionEnvironmentBinding,
    ExecutionOwner,
    ExternalCallApprovalRecord,
    ModelRole,
    ModelRoleBindingV2,
    ProviderBudgetCap,
    ResourceBudgetCeiling,
    RoleBindingStatus,
    RoleBudgetCeiling,
    ai_review_plan_hash,
    external_call_approval_hash,
)
from oamb.contracts.states import (
    AttemptOutcome,
    IndexContribution,
    ValidationDisposition,
)
from oamb.phase_review_profiles import (
    FAKE_PHASE_REVIEW_CLIENT_KIND,
    FAKE_PHASE_REVIEW_CONFIGURED_MODEL,
    FAKE_PHASE_REVIEW_COST_REASON,
    FAKE_PHASE_REVIEW_CREDENTIAL_VARIABLE,
    FAKE_PHASE_REVIEW_ENDPOINT,
    FAKE_PHASE_REVIEW_PROVIDER,
    FAKE_PHASE_REVIEW_RESOLVED_MODEL,
    OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND,
    PHASE_REVIEW_INPUT_TOKEN_PRICE_CLASS,
    PHASE_REVIEW_OUTPUT_TOKEN_PRICE_CLASS,
    PHASE_REVIEW_TOKEN_PRICE_UNIT_SCALE,
    phase_review_runtime_hash,
)
from oamb.reporting.human_review import (
    create_human_quality_review_record,
    derive_evaluation_phase_gate,
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
    human_record = create_human_quality_review_record(
        canonical_ai_record=ai_record,
        status="pass",
        operator_id="fixture-operator",
        nonce="fixture-nonce",
        finding_codes=(),
        evidence_references=(),
        used_nonces=(),
        created_at=NOW,
    )
    return derive_evaluation_phase_gate(
        phase_id=bundle.phase_id,
        review_bundle_hash=review_bundle_hash,
        ordered_ai_history=(ai_record,),
        human_records=(human_record,),
    )


def _phase_review_plan(
    bundle: EvaluationReviewBundle,
    *,
    reviewer_runtime_hash: str = SHA_D,
) -> AIReviewPlan:
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
        "reviewer_runtime_hash": reviewer_runtime_hash,
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
        reviewer_runtime_hash=reviewer_runtime_hash,
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


def _accepted_gate_with_evidence(
    *, fake_reviewer: bool = False
) -> tuple[EvaluationReviewBundle, EvaluationPhaseGate, Any]:
    review = importlib.import_module("oamb.reporting.review")
    bundle = _bundle()
    execution_environment = ExecutionEnvironmentBinding(
        environment_hash=SHA_A,
        operating_system="fixture-os",
        architecture="fixture-architecture",
        python_version="3.12.0",
        cpu_description="fixture-cpu",
        memory_bytes=1_000_000,
        comparability_status=ComparabilityStatus.COMPARABLE,
    )
    client_kind = (
        FAKE_PHASE_REVIEW_CLIENT_KIND
        if fake_reviewer
        else OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND
    )
    reviewer_provider = FAKE_PHASE_REVIEW_PROVIDER if fake_reviewer else "fixture-reviewer"
    reviewer_endpoint = FAKE_PHASE_REVIEW_ENDPOINT if fake_reviewer else "https://models.example/v1"
    reviewer_credential = (
        FAKE_PHASE_REVIEW_CREDENTIAL_VARIABLE if fake_reviewer else "OAMB_QUALITY_REVIEW_API_KEY"
    )
    reviewer_configured_model = (
        FAKE_PHASE_REVIEW_CONFIGURED_MODEL if fake_reviewer else "review-model"
    )
    reviewer_resolved_model = (
        FAKE_PHASE_REVIEW_RESOLVED_MODEL if fake_reviewer else "review-model@runtime"
    )
    runtime_hash = phase_review_runtime_hash(
        SHA_B,
        execution_environment.environment_hash,
        client_kind=client_kind,
    )
    plan = _phase_review_plan(bundle, reviewer_runtime_hash=runtime_hash)
    reviewer_role_binding = ModelRoleBindingV2(
        binding_id=plan.reviewer_role_binding_hash,
        role=ModelRole.QUALITY_REVIEW,
        role_status=RoleBindingStatus.SELECTED,
        execution_owner=ExecutionOwner.HARNESS,
        binding_kind=BindingKind.MODEL_CLIENT,
        provider=reviewer_provider,
        endpoint_reference=reviewer_endpoint,
        credential_variable_name=reviewer_credential,
        configured_model=reviewer_configured_model,
        resolved_model=reviewer_resolved_model,
        parameters_fingerprint=SHA_A,
        retry_policy_id="no-retry-v1",
        configuration_fingerprint=plan.reviewer_configuration_hash,
        redacted_endpoint_fingerprint=SHA_B,
    )
    occurrence_id = phase_review_occurrence_id_v2(
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        reviewer_role_binding_hash=plan.reviewer_role_binding_hash,
        artifact_repository_fingerprint=SHA_C,
        ordinal=1,
    )
    resource_ceiling = ResourceBudgetCeiling(
        dimension_id="provider_request_wall_seconds_v1",
        maximum=Decimal("30"),
        unit="seconds",
    )
    cost_measurement_spec = CostMeasurementSpec(
        measurement_spec_id="phase-review-measurement-v1",
        measurement_spec_version="1",
        dimensions=(
            MeasurementDimensionSpec(
                dimension_id=resource_ceiling.dimension_id,
                stage="quality_review",
                operation_kind="quality_review",
                parent_kind="phase_review",
                unit=resource_ceiling.unit,
                allowed_meter_sources=("fixture_clock_v1" if fake_reviewer else "process_meter",),
                required=True,
                price_class=None,
                indexing_view_rule="not_applicable",
                aggregation_operator=AggregationOperator.INTERVAL_UNION,
            ),
        ),
    )
    price_snapshot = (
        None
        if fake_reviewer
        else PriceSnapshot(
            price_snapshot_id="fixture-price-snapshot-v1",
            provider=reviewer_provider,
            effective_at=NOW,
            currency="USD",
            price_class_ids=(
                PHASE_REVIEW_INPUT_TOKEN_PRICE_CLASS,
                PHASE_REVIEW_OUTPUT_TOKEN_PRICE_CLASS,
            ),
            unit_prices=(Decimal("1000"), Decimal("2000")),
        )
    )
    role_ceiling = RoleBudgetCeiling(
        role_binding_id=plan.reviewer_role_binding_hash,
        max_attempts=plan.expected_attempt_count,
        max_input_tokens=1_000,
        max_output_tokens=2_000,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=Decimal("1"),
        currency="USD",
        price_snapshot_id=(None if price_snapshot is None else price_snapshot.price_snapshot_id),
        resource_ceilings=(resource_ceiling,),
        provider_budget_cap=ProviderBudgetCap(
            provider=reviewer_provider,
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
    occurrence = PhaseReviewOccurrenceRecordV2(
        phase_review_occurrence_id=occurrence_id,
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        reviewer_role_binding_hash=plan.reviewer_role_binding_hash,
        artifact_repository_fingerprint=SHA_C,
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
            measurement_source=("fixture_clock_v1" if fake_reviewer else "process_meter"),
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
            basis=(
                CostBasis.ACTUAL_SUPPLIER_CHARGE
                if fake_reviewer
                else CostBasis.ESTIMATE_FROM_MEASURED_USAGE
            ),
            indexing_view=IndexingView.NOT_APPLICABLE,
            amount=None if fake_reviewer else Decimal("0.02"),
            currency=None if fake_reviewer else "USD",
            price_snapshot_id=(
                None if price_snapshot is None else price_snapshot.price_snapshot_id
            ),
            source_usage_record_ids=(usage_records[index - 1].usage_record_id,),
            source_resource_record_ids=(resource_records[index - 1].resource_record_id,),
            proof_status=(
                ProofStatus.NOT_APPLICABLE if fake_reviewer else ProofStatus.MEASURED_COMPLETE
            ),
            reason=FAKE_PHASE_REVIEW_COST_REASON if fake_reviewer else None,
        )
        for index in range(1, len(attempts) + 1)
    )
    lease_candidate = RunLeaseRecord(
        lease_record_hash="0" * 64,
        run_id=occurrence_id,
        provider_project_id=reviewer_provider,
        provider_profile_id=canonical_sha256(
            [
                "oamb-phase-review-provider-profile-v1",
                client_kind,
                reviewer_role_binding.binding_id,
            ]
        ),
        lease_epoch=1,
        owner_id="oamb-phase-review-runner-v1",
        host_fingerprint=SHA_A,
        process_id=1,
        predecessor_lease_record_hash=None,
        acquired_at=NOW + timedelta(seconds=30),
    )
    lease_record = lease_candidate.model_copy(
        update={
            "lease_record_hash": canonical_sha256(
                lease_candidate.model_dump(mode="python", exclude={"lease_record_hash"})
            )
        }
    )
    planned_inputs = tuple(batch.input_tokens for batch in plan.case_batches) + (
        plan.phase_integrity_input_tokens,
    )
    planned_outputs = tuple(batch.maximal_output_tokens for batch in plan.case_batches) + (
        plan.phase_integrity_maximal_output_tokens,
    )
    occurrence_claims: list[OccurrenceClaimRecord] = []
    budget_reservations: list[BudgetReservationRecord] = []
    attempt_intents: list[AttemptIntentRecord] = []
    attempt_receipts: list[AttemptReceiptRecord] = []
    for index, attempt in enumerate(attempts, 1):
        claimed_at = attempt.started_at - timedelta(seconds=1)
        claim_fields = {
            "occurrence_id": occurrence_id,
            "lease_record_hash": lease_record.lease_record_hash,
            "lease_epoch": 1,
            "owner_id": lease_record.owner_id,
            "stage": "quality_review",
            "request_fingerprint": attempt.request_fingerprint,
            "reconciliation_capability": "none",
            "claimed_at": claimed_at,
        }
        claim_id = canonical_sha256(["oamb-phase-review-occurrence-claim-v1", claim_fields])
        claim = OccurrenceClaimRecord.model_validate({"claim_id": claim_id, **claim_fields})
        reserved_cost = (
            Decimal("0")
            if price_snapshot is None
            else (
                Decimal(planned_inputs[index - 1]) * price_snapshot.unit_prices[0]
                + Decimal(planned_outputs[index - 1]) * price_snapshot.unit_prices[1]
            )
            / Decimal(PHASE_REVIEW_TOKEN_PRICE_UNIT_SCALE)
        )
        reservation_fields = {
            "budget_id": budget.budget_id,
            "scope_kind": BudgetScopeKindV2.PHASE_REVIEW,
            "scope_id": occurrence_id,
            "role_binding_id": reviewer_role_binding.binding_id,
            "attempt_id": attempt.attempt_id,
            "reserved_attempts": 1,
            "reserved_input_tokens": planned_inputs[index - 1],
            "reserved_output_tokens": planned_outputs[index - 1],
            "reserved_dispatch_wall_seconds": Decimal("30"),
            "reserved_cost": reserved_cost,
            "currency": budget.currency,
            "reserved_resource_ceilings": (resource_ceiling,),
            "reserved_provider_units": Decimal("1"),
            "reserved_at": claimed_at,
        }
        reservation_id = canonical_sha256(
            ["oamb-phase-review-budget-reservation-v1", reservation_fields]
        )
        reservation = BudgetReservationRecord.model_validate(
            {"reservation_id": reservation_id, **reservation_fields}
        )
        intent = AttemptIntentRecord(
            attempt_id=attempt.attempt_id,
            claim_id=claim.claim_id,
            reservation_id=reservation.reservation_id,
            parent_kind="phase_review",
            parent_id=occurrence_id,
            role_binding_id=reviewer_role_binding.binding_id,
            stage="quality_review",
            request_fingerprint=attempt.request_fingerprint,
            reconciliation_capability="none",
            idempotency_key_hash=None,
            sealed_at=claimed_at,
        )
        receipt = AttemptReceiptRecord(
            attempt_id=attempt.attempt_id,
            receipt_kind=AttemptReceiptKind.RESPONSE,
            raw_response_ref=attempt.raw_response_ref,
            raw_error_ref=None,
            dispatch_started_at=attempt.started_at,
            receipt_observed_at=attempt.ended_at,
            provider_request_wall_seconds=resource_records[index - 1].value or Decimal("0"),
        )
        occurrence_claims.append(claim)
        budget_reservations.append(reservation)
        attempt_intents.append(intent)
        attempt_receipts.append(receipt)
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
    human_record = create_human_quality_review_record(
        canonical_ai_record=ai_record,
        status="pass",
        nonce="phase-fixture-nonce",
        operator_id="fixture-operator",
        finding_codes=(),
        evidence_references=(),
        used_nonces=(),
        created_at=NOW + timedelta(minutes=5),
    )
    gate = derive_evaluation_phase_gate(
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        ordered_ai_history=(ai_record,),
        human_records=(human_record,),
    )
    evidence = _phase().PhaseReviewEvidence(
        plan=plan,
        reviewer_role_binding=reviewer_role_binding,
        cost_measurement_spec=cost_measurement_spec,
        execution_environment=execution_environment,
        price_snapshot=price_snapshot,
        lease_record=lease_record,
        occurrence_claims=tuple(occurrence_claims),
        budget_reservations=tuple(budget_reservations),
        attempt_intents=tuple(attempt_intents),
        attempt_receipts=tuple(attempt_receipts),
        close_errors=(),
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
        human_record=human_record,
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
        reviewer_role_binding=placeholder,
        cost_measurement_spec=placeholder,
        execution_environment=placeholder,
        price_snapshot=None,
        lease_record=placeholder,
        occurrence_claims=(),
        budget_reservations=(),
        attempt_intents=(),
        attempt_receipts=(),
        close_errors=(),
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
        human_record=gate.human_record,
    )

    result = phase.validate_t10_phase_gate(bundle, gate, review_evidence=evidence)

    assert result.disposition == ValidationDisposition.INVALID
    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in result.issues}


def test_t10_phase_profile_rejects_missing_cost() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()

    missing_cost = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(evidence, cost_records=()),
    )
    assert "phase-review-source-evidence-mismatch" in {issue.code for issue in missing_cost.issues}


def test_t10_phase_target_hash_binds_review_source_evidence() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()

    accepted = phase.validate_t10_phase_gate(bundle, gate, review_evidence=evidence)
    changed = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(evidence, cost_records=()),
    )

    assert accepted.target_hash != changed.target_hash


def test_t10_phase_target_hash_binds_reviewer_role() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    changed_role = evidence.reviewer_role_binding.model_copy(
        update={"provider": "different-reviewer"}
    )

    accepted = phase.validate_t10_phase_gate(bundle, gate, review_evidence=evidence)
    changed = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(evidence, reviewer_role_binding=changed_role),
    )

    assert accepted.target_hash != changed.target_hash


def test_t10_phase_target_hash_binds_measurement_environment_and_price() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    assert evidence.price_snapshot is not None

    accepted = phase.validate_t10_phase_gate(bundle, gate, review_evidence=evidence)
    changed_measurement = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            cost_measurement_spec=evidence.cost_measurement_spec.model_copy(
                update={"measurement_spec_version": "2"}
            ),
        ),
    )
    changed_environment = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            execution_environment=evidence.execution_environment.model_copy(
                update={"cpu_description": "different-cpu"}
            ),
        ),
    )
    changed_price = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            price_snapshot=evidence.price_snapshot.model_copy(
                update={"unit_prices": (Decimal("1001"), Decimal("2000"))}
            ),
        ),
    )

    assert (
        len(
            {
                accepted.target_hash,
                changed_measurement.target_hash,
                changed_environment.target_hash,
                changed_price.target_hash,
            }
        )
        == 4
    )


def test_t10_phase_profile_rejects_environment_and_client_kind_runtime_drift() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    assert evidence.reviewer_role_binding.redacted_endpoint_fingerprint is not None
    changed_environment = evidence.execution_environment.model_copy(
        update={"environment_hash": "f" * 64}
    )
    wrong_client_runtime = phase_review_runtime_hash(
        evidence.reviewer_role_binding.redacted_endpoint_fingerprint,
        evidence.execution_environment.environment_hash,
        client_kind=FAKE_PHASE_REVIEW_CLIENT_KIND,
    )

    environment_result = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(evidence, execution_environment=changed_environment),
    )
    client_kind_result = phase.validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            plan=evidence.plan.model_copy(update={"reviewer_runtime_hash": wrong_client_runtime}),
        ),
    )

    assert environment_result.disposition == ValidationDisposition.INVALID
    assert client_kind_result.disposition == ValidationDisposition.INVALID


def test_t10_phase_profile_rejects_measurement_spec_and_resource_binding_drift() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    dimension = evidence.cost_measurement_spec.dimensions[0]
    changed_spec = evidence.cost_measurement_spec.model_copy(
        update={"measurement_spec_id": "different-measurement-spec"}
    )
    changed_dimension = evidence.cost_measurement_spec.model_copy(
        update={
            "dimensions": (
                dimension.model_copy(update={"allowed_meter_sources": ("different-clock",)}),
            )
        }
    )
    changed_resource = evidence.resource_records[0].model_copy(
        update={"environment_hash": "f" * 64}
    )

    results = (
        phase.validate_t10_phase_gate(
            bundle,
            gate,
            review_evidence=replace(evidence, cost_measurement_spec=changed_spec),
        ),
        phase.validate_t10_phase_gate(
            bundle,
            gate,
            review_evidence=replace(evidence, cost_measurement_spec=changed_dimension),
        ),
        phase.validate_t10_phase_gate(
            bundle,
            gate,
            review_evidence=replace(
                evidence,
                resource_records=(changed_resource, *evidence.resource_records[1:]),
            ),
        ),
    )

    assert all(result.disposition == ValidationDisposition.INVALID for result in results)


def test_t10_phase_profile_rejects_incomplete_or_tampered_attempt_transaction() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    tampered_lease = evidence.lease_record.model_copy(update={"owner_id": "different-owner"})

    results = (
        phase.validate_t10_phase_gate(
            bundle,
            gate,
            review_evidence=replace(
                evidence,
                attempt_receipts=evidence.attempt_receipts[:-1],
            ),
        ),
        phase.validate_t10_phase_gate(
            bundle,
            gate,
            review_evidence=replace(evidence, lease_record=tampered_lease),
        ),
    )

    assert all(result.disposition == ValidationDisposition.INVALID for result in results)


def test_t10_phase_profile_rejects_live_price_snapshot_inventory_rate_and_currency_drift() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    assert evidence.price_snapshot is not None
    changed_inventory = evidence.price_snapshot.model_copy(
        update={"price_class_ids": tuple(reversed(evidence.price_snapshot.price_class_ids))}
    )
    changed_rate = evidence.price_snapshot.model_copy(
        update={"unit_prices": (Decimal("1001"), Decimal("2000"))}
    )
    changed_currency = evidence.price_snapshot.model_copy(update={"currency": "EUR"})
    future_effective = evidence.price_snapshot.model_copy(
        update={"effective_at": evidence.occurrence.ended_at + timedelta(days=30)}
    )

    results = tuple(
        phase.validate_t10_phase_gate(
            bundle,
            gate,
            review_evidence=replace(evidence, price_snapshot=price_snapshot),
        )
        for price_snapshot in (
            None,
            changed_inventory,
            changed_rate,
            changed_currency,
            future_effective,
        )
    )

    assert all(result.disposition == ValidationDisposition.INVALID for result in results)


def test_t10_phase_profile_rejects_gate_history_hidden_from_source_evidence() -> None:
    phase = _phase()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    ordered_hashes = ("e" * 64, *gate.ordered_ai_review_record_hashes)
    history_root = canonical_sha256(
        ["oamb-review-history-v2", ordered_hashes, gate.human_review_record_hash]
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
