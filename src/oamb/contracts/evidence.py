"""Immutable source evidence and structural validation records."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from .base import (
    NonEmptyStr,
    NonNegativeDecimal,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictContract,
    UtcDateTime,
)
from .ids import canonical_sha256
from .specifications import (
    BudgetScopeKindV2,
    BudgetScopeKindV3,
    DispatchBudgetOwnerKind,
    ResourceBudgetCeiling,
)
from .states import (
    AttemptOutcome,
    CaseState,
    IndexContribution,
    IngestionPlanState,
    ResumeDisposition,
    RunState,
    ValidationDisposition,
)


class OriginClass(StrEnum):
    OAMB_NATIVE = "oamb_native"
    EXTERNAL_AMB_GENERATED = "external_amb_generated"
    IMPORTED_THIRD_PARTY = "imported_third_party"


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class ModelReadinessOccurrenceState(StrEnum):
    PLANNED = "planned"
    BUDGET_RESERVED = "budget_reserved"
    RUNNING = "running"
    SEALED = "sealed"
    ERROR = "error"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERRUPTED_UNKNOWN_OUTCOME = "interrupted_unknown_outcome"


class MemoryConformanceOccurrenceState(StrEnum):
    PLANNED = "planned"
    BUDGET_RESERVED = "budget_reserved"
    RUNNING = "running"
    SEALED = "sealed"
    ERROR = "error"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERRUPTED_UNKNOWN_OUTCOME = "interrupted_unknown_outcome"


class AttemptReceiptKind(StrEnum):
    RESPONSE = "response"
    ERROR = "error"
    UNKNOWN_OUTCOME = "unknown_outcome"


class RecoveryDisposition(StrEnum):
    RESUME_SAFE = "resume_safe"
    REPLACEMENT_RUN_REQUIRED = "replacement_run_required"
    TERMINAL_UNKNOWN_OUTCOME = "terminal_unknown_outcome"


class CaseEvaluationDisposition(StrEnum):
    NOT_RUN = "not_run"
    DETERMINISTIC_EVALUATED = "deterministic_evaluated"
    JUDGED = "judged"
    UNJUDGED = "unjudged"


class RawReference(StrictContract):
    schema_name: Literal["raw_reference"] = "raw_reference"
    schema_version: Literal[1] = 1
    sha256: Sha256
    media_type: NonEmptyStr
    byte_count: NonNegativeInt
    compression: Literal["none", "gzip"]


class OriginRecord(StrictContract):
    schema_name: Literal["origin_record"] = "origin_record"
    schema_version: Literal[1] = 1
    origin_id: NonEmptyStr
    origin_class: OriginClass
    producer: NonEmptyStr
    source_sha256: Sha256
    license_id: NonEmptyStr


class RunRecord(StrictContract):
    schema_name: Literal["run_record"] = "run_record"
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    run_spec_hash: Sha256
    state: RunState
    resume_disposition: ResumeDisposition
    started_at: UtcDateTime | None
    ended_at: UtcDateTime | None
    ingestion_occurrence_ids: tuple[Sha256, ...]
    case_occurrence_ids: tuple[Sha256, ...]


class AttemptRecord(StrictContract):
    schema_name: Literal["attempt_record"] = "attempt_record"
    schema_version: Literal[1] = 1
    attempt_id: Sha256
    parent_kind: Literal["ingestion_plan", "case", "phase_review"]
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    ordinal: PositiveInt
    request_fingerprint: Sha256
    started_at: UtcDateTime
    ended_at: UtcDateTime
    outcome: AttemptOutcome
    retry_of_attempt_id: Sha256 | None
    idempotency_key_hash: Sha256 | None
    reconciliation_capability: Literal["none", "idempotency_key", "receipt_lookup"]
    raw_response_ref: Sha256 | None
    raw_error_ref: Sha256 | None
    index_contribution: IndexContribution
    superseded_by_attempt_id: Sha256 | None

    @model_validator(mode="after")
    def coherent_times_and_contribution(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("attempt end precedes start")
        if self.index_contribution == IndexContribution.SUPERSEDED:
            if self.superseded_by_attempt_id is None:
                raise ValueError("superseded contribution requires successor attempt")
        elif self.superseded_by_attempt_id is not None:
            raise ValueError("only superseded contribution names a successor")
        return self


class AttemptRecordV2(StrictContract):
    schema_name: Literal["attempt_record"] = "attempt_record"
    schema_version: Literal[2] = 2
    attempt_id: Sha256
    parent_kind: Literal["ingestion_plan", "case", "phase_review", "model_readiness"]
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    ordinal: PositiveInt
    request_fingerprint: Sha256
    started_at: UtcDateTime
    ended_at: UtcDateTime
    outcome: AttemptOutcome
    retry_of_attempt_id: Sha256 | None
    idempotency_key_hash: Sha256 | None
    reconciliation_capability: Literal["none", "idempotency_key", "receipt_lookup"]
    raw_response_ref: Sha256 | None
    raw_error_ref: Sha256 | None
    index_contribution: IndexContribution
    superseded_by_attempt_id: Sha256 | None

    @model_validator(mode="after")
    def coherent_times_and_contribution(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("attempt end precedes start")
        if self.index_contribution == IndexContribution.SUPERSEDED:
            if self.superseded_by_attempt_id is None:
                raise ValueError("superseded contribution requires successor attempt")
        elif self.superseded_by_attempt_id is not None:
            raise ValueError("only superseded contribution names a successor")
        return self


class AttemptRecordV3(StrictContract):
    schema_name: Literal["attempt_record"] = "attempt_record"
    schema_version: Literal[3] = 3
    attempt_id: Sha256
    parent_kind: Literal[
        "ingestion_plan", "case", "phase_review", "model_readiness", "memory_conformance"
    ]
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    ordinal: PositiveInt
    request_fingerprint: Sha256
    started_at: UtcDateTime
    ended_at: UtcDateTime
    outcome: AttemptOutcome
    retry_of_attempt_id: Sha256 | None
    idempotency_key_hash: Sha256 | None
    reconciliation_capability: Literal["none", "idempotency_key", "receipt_lookup"]
    raw_response_ref: Sha256 | None
    raw_error_ref: Sha256 | None
    index_contribution: IndexContribution
    superseded_by_attempt_id: Sha256 | None

    @model_validator(mode="after")
    def coherent_times_and_contribution(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("attempt end precedes start")
        if self.parent_kind == "memory_conformance" and self.stage not in {
            "scope_allocate",
            "memory_ingest",
            "memory_readiness",
            "memory_projection",
            "memory_query",
        }:
            raise ValueError("memory-conformance attempt uses an unsupported stage")
        if self.index_contribution == IndexContribution.SUPERSEDED:
            if self.superseded_by_attempt_id is None:
                raise ValueError("superseded contribution requires successor attempt")
        elif self.superseded_by_attempt_id is not None:
            raise ValueError("only superseded contribution names a successor")
        return self


class RunLeaseRecord(StrictContract):
    schema_name: Literal["run_lease_record"] = "run_lease_record"
    schema_version: Literal[1] = 1
    lease_record_hash: Sha256
    run_id: NonEmptyStr
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    lease_epoch: PositiveInt
    owner_id: NonEmptyStr
    host_fingerprint: Sha256
    process_id: PositiveInt
    predecessor_lease_record_hash: Sha256 | None
    acquired_at: UtcDateTime

    @model_validator(mode="after")
    def predecessor_matches_epoch(self) -> Self:
        if self.lease_epoch == 1 and self.predecessor_lease_record_hash is not None:
            raise ValueError("first lease epoch cannot have a predecessor")
        if self.lease_epoch > 1 and self.predecessor_lease_record_hash is None:
            raise ValueError("later lease epoch requires a predecessor")
        return self


class RunLeaseHeartbeatRecord(StrictContract):
    schema_name: Literal["run_lease_heartbeat_record"] = "run_lease_heartbeat_record"
    schema_version: Literal[1] = 1
    heartbeat_record_hash: Sha256
    lease_record_hash: Sha256
    heartbeat_sequence: PositiveInt
    predecessor_heartbeat_hash: Sha256 | None
    observed_at: UtcDateTime

    @model_validator(mode="after")
    def predecessor_matches_sequence(self) -> Self:
        if self.heartbeat_sequence == 1 and self.predecessor_heartbeat_hash is not None:
            raise ValueError("first heartbeat cannot have a predecessor")
        if self.heartbeat_sequence > 1 and self.predecessor_heartbeat_hash is None:
            raise ValueError("later heartbeat requires a predecessor")
        return self


class OccurrenceClaimRecord(StrictContract):
    schema_name: Literal["occurrence_claim_record"] = "occurrence_claim_record"
    schema_version: Literal[1] = 1
    claim_id: Sha256
    occurrence_id: NonEmptyStr
    lease_record_hash: Sha256
    lease_epoch: PositiveInt
    owner_id: NonEmptyStr
    stage: NonEmptyStr
    request_fingerprint: Sha256
    reconciliation_capability: Literal["none", "idempotency_key", "receipt_lookup"]
    claimed_at: UtcDateTime


class ModelReadinessOccurrenceRecord(StrictContract):
    schema_name: Literal["model_readiness_occurrence_record"] = "model_readiness_occurrence_record"
    schema_version: Literal[1] = 1
    occurrence_id: NonEmptyStr
    provider: NonEmptyStr
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    provider_runtime_profile_attestation_hash: Sha256
    approval_id: NonEmptyStr
    budget_id: NonEmptyStr
    state: ModelReadinessOccurrenceState
    role_binding_ids: tuple[NonEmptyStr, ...]
    attempt_ids: tuple[Sha256, ...]
    usage_record_ids: tuple[Sha256, ...]
    resource_record_ids: tuple[Sha256, ...]
    cost_record_ids: tuple[Sha256, ...]
    started_at: UtcDateTime | None
    ended_at: UtcDateTime | None

    @model_validator(mode="after")
    def lifecycle_and_inventory_close(self) -> Self:
        if not self.role_binding_ids:
            raise ValueError("model-readiness occurrence requires selected roles")
        inventories = (
            self.role_binding_ids,
            self.attempt_ids,
            self.usage_record_ids,
            self.resource_record_ids,
            self.cost_record_ids,
        )
        if any(len(set(items)) != len(items) for items in inventories):
            raise ValueError("model-readiness inventories cannot contain duplicates")
        terminal_states = {
            ModelReadinessOccurrenceState.SEALED,
            ModelReadinessOccurrenceState.ERROR,
            ModelReadinessOccurrenceState.CANCELLED,
            ModelReadinessOccurrenceState.BUDGET_EXCEEDED,
            ModelReadinessOccurrenceState.INTERRUPTED_UNKNOWN_OUTCOME,
        }
        if self.state in terminal_states:
            if self.started_at is None or self.ended_at is None:
                raise ValueError("terminal model-readiness occurrence requires timestamps")
        elif self.ended_at is not None:
            raise ValueError("non-terminal model-readiness occurrence cannot have an end time")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("model-readiness end precedes start")
        return self


class MemoryConformanceOccurrenceRecord(StrictContract):
    schema_name: Literal["memory_conformance_occurrence_record"] = (
        "memory_conformance_occurrence_record"
    )
    schema_version: Literal[1] = 1
    occurrence_id: NonEmptyStr
    provider: NonEmptyStr
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    provider_scope_id: NonEmptyStr
    runtime_binding_hash: Sha256
    approval_id: NonEmptyStr
    budget_id: NonEmptyStr
    state: MemoryConformanceOccurrenceState
    operation_claim_ids: tuple[Sha256, ...]
    dispatch_route_ids: tuple[NonEmptyStr, ...]
    attempt_ids: tuple[Sha256, ...]
    usage_record_ids: tuple[Sha256, ...]
    resource_record_ids: tuple[Sha256, ...]
    cost_record_ids: tuple[Sha256, ...]
    protected_state_before_hash: Sha256
    protected_state_after_hash: Sha256
    started_at: UtcDateTime | None
    ended_at: UtcDateTime | None

    @model_validator(mode="after")
    def lifecycle_inventory_and_projection_close(self) -> Self:
        if self.provider_project_id == self.provider_scope_id:
            raise ValueError("memory-conformance project and scope identities must differ")
        inventories = (
            self.operation_claim_ids,
            self.dispatch_route_ids,
            self.attempt_ids,
            self.usage_record_ids,
            self.resource_record_ids,
            self.cost_record_ids,
        )
        if any(len(set(items)) != len(items) for items in inventories):
            raise ValueError("memory-conformance inventories cannot contain duplicates")
        terminal = {
            MemoryConformanceOccurrenceState.SEALED,
            MemoryConformanceOccurrenceState.ERROR,
            MemoryConformanceOccurrenceState.CANCELLED,
            MemoryConformanceOccurrenceState.BUDGET_EXCEEDED,
            MemoryConformanceOccurrenceState.INTERRUPTED_UNKNOWN_OUTCOME,
        }
        if self.state in terminal:
            if self.started_at is None or self.ended_at is None:
                raise ValueError("terminal memory-conformance occurrence requires timestamps")
        elif self.ended_at is not None:
            raise ValueError("non-terminal memory-conformance occurrence cannot have an end time")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("memory-conformance end precedes start")
        if (
            self.state == MemoryConformanceOccurrenceState.SEALED
            and self.protected_state_before_hash != self.protected_state_after_hash
        ):
            raise ValueError("sealed memory-conformance protected state changed during query")
        return self


class BudgetReservationRecord(StrictContract):
    schema_name: Literal["budget_reservation_record"] = "budget_reservation_record"
    schema_version: Literal[1] = 1
    reservation_id: Sha256
    budget_id: NonEmptyStr
    scope_kind: BudgetScopeKindV2
    scope_id: NonEmptyStr
    role_binding_id: NonEmptyStr
    attempt_id: Sha256
    reserved_attempts: PositiveInt
    reserved_input_tokens: NonNegativeInt
    reserved_output_tokens: NonNegativeInt
    reserved_dispatch_wall_seconds: NonNegativeDecimal
    reserved_cost: NonNegativeDecimal | None
    currency: str | None
    reserved_resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    reserved_provider_units: NonNegativeDecimal
    reserved_at: UtcDateTime

    @model_validator(mode="after")
    def currency_and_resources_close(self) -> Self:
        if (self.reserved_cost is None) != (self.currency is None):
            raise ValueError("reserved cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        dimensions = tuple(item.dimension_id for item in self.reserved_resource_ceilings)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("reservation contains duplicate resource dimensions")
        return self


class BudgetReservationRecordV2(StrictContract):
    schema_name: Literal["budget_reservation_record"] = "budget_reservation_record"
    schema_version: Literal[2] = 2
    reservation_id: Sha256
    budget_id: NonEmptyStr
    scope_kind: BudgetScopeKindV3
    scope_id: NonEmptyStr
    dispatch_owner_kind: DispatchBudgetOwnerKind
    role_binding_id: NonEmptyStr | None
    provider_operation_ceiling_id: NonEmptyStr | None
    internal_usage_role_binding_ids: tuple[NonEmptyStr, ...]
    attempt_id: Sha256
    reserved_attempts: PositiveInt
    reserved_input_tokens: NonNegativeInt
    reserved_output_tokens: NonNegativeInt
    reserved_dispatch_wall_seconds: NonNegativeDecimal
    reserved_cost: NonNegativeDecimal | None
    currency: str | None
    reserved_resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    reserved_provider_units: NonNegativeDecimal
    reserved_at: UtcDateTime

    @model_validator(mode="after")
    def scope_owner_and_resources_close(self) -> Self:
        if self.scope_kind != BudgetScopeKindV3.MEMORY_CONFORMANCE:
            raise ValueError("version 2 reservation currently belongs to memory conformance")
        if self.dispatch_owner_kind == DispatchBudgetOwnerKind.MODEL_ROLE:
            if self.role_binding_id is None or self.provider_operation_ceiling_id is not None:
                raise ValueError("model-role reservation requires only its role binding")
            if self.internal_usage_role_binding_ids:
                raise ValueError("direct model reservation cannot carry internal usage owners")
        else:
            if self.role_binding_id is not None or self.provider_operation_ceiling_id is None:
                raise ValueError(
                    "provider-operation reservation requires only its operation ceiling"
                )
        if len(set(self.internal_usage_role_binding_ids)) != len(
            self.internal_usage_role_binding_ids
        ):
            raise ValueError("reservation contains duplicate internal usage owners")
        if (self.reserved_cost is None) != (self.currency is None):
            raise ValueError("reserved cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        dimensions = tuple(item.dimension_id for item in self.reserved_resource_ceilings)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("reservation contains duplicate resource dimensions")
        return self


class AttemptIntentRecord(StrictContract):
    schema_name: Literal["attempt_intent_record"] = "attempt_intent_record"
    schema_version: Literal[1] = 1
    attempt_id: Sha256
    claim_id: Sha256
    reservation_id: Sha256
    parent_kind: Literal["ingestion_plan", "case", "phase_review", "model_readiness"]
    parent_id: NonEmptyStr
    role_binding_id: NonEmptyStr
    stage: NonEmptyStr
    request_fingerprint: Sha256
    reconciliation_capability: Literal["none", "idempotency_key", "receipt_lookup"]
    idempotency_key_hash: Sha256 | None
    sealed_at: UtcDateTime

    @model_validator(mode="after")
    def idempotency_shape(self) -> Self:
        if self.reconciliation_capability == "idempotency_key":
            if self.idempotency_key_hash is None:
                raise ValueError("idempotency-key reconciliation requires its hash")
        elif self.idempotency_key_hash is not None:
            raise ValueError("only idempotency-key reconciliation names its hash")
        return self


class AttemptIntentRecordV2(StrictContract):
    schema_name: Literal["attempt_intent_record"] = "attempt_intent_record"
    schema_version: Literal[2] = 2
    attempt_id: Sha256
    claim_id: Sha256
    reservation_id: Sha256
    parent_kind: Literal[
        "ingestion_plan", "case", "phase_review", "model_readiness", "memory_conformance"
    ]
    parent_id: NonEmptyStr
    dispatch_route_id: NonEmptyStr
    dispatch_owner_kind: DispatchBudgetOwnerKind
    role_binding_id: NonEmptyStr | None
    provider_operation_ceiling_id: NonEmptyStr | None
    internal_usage_role_binding_ids: tuple[NonEmptyStr, ...]
    stage: NonEmptyStr
    request_fingerprint: Sha256
    reconciliation_capability: Literal["none", "idempotency_key", "receipt_lookup"]
    idempotency_key_hash: Sha256 | None
    sealed_at: UtcDateTime

    @model_validator(mode="after")
    def parent_owner_and_idempotency_close(self) -> Self:
        if self.parent_kind != "memory_conformance":
            raise ValueError("version 2 intent currently belongs to memory conformance")
        if self.dispatch_owner_kind == DispatchBudgetOwnerKind.MODEL_ROLE:
            if self.role_binding_id is None or self.provider_operation_ceiling_id is not None:
                raise ValueError("model-role intent requires only its role binding")
            if self.internal_usage_role_binding_ids:
                raise ValueError("direct model intent cannot carry internal usage owners")
        else:
            if self.role_binding_id is not None or self.provider_operation_ceiling_id is None:
                raise ValueError("provider-operation intent requires only its operation ceiling")
        if len(set(self.internal_usage_role_binding_ids)) != len(
            self.internal_usage_role_binding_ids
        ):
            raise ValueError("intent contains duplicate internal usage owners")
        if self.reconciliation_capability == "idempotency_key":
            if self.idempotency_key_hash is None:
                raise ValueError("idempotency-key reconciliation requires its hash")
        elif self.idempotency_key_hash is not None:
            raise ValueError("only idempotency-key reconciliation names its hash")
        return self


class AttemptReceiptRecord(StrictContract):
    schema_name: Literal["attempt_receipt_record"] = "attempt_receipt_record"
    schema_version: Literal[1] = 1
    attempt_id: Sha256
    receipt_kind: AttemptReceiptKind
    raw_response_ref: Sha256 | None
    raw_error_ref: Sha256 | None
    dispatch_started_at: UtcDateTime
    receipt_observed_at: UtcDateTime
    provider_request_wall_seconds: NonNegativeDecimal

    @model_validator(mode="after")
    def receipt_and_timing_shape(self) -> Self:
        if self.receipt_observed_at < self.dispatch_started_at:
            raise ValueError("receipt observation precedes dispatch")
        if self.receipt_kind == AttemptReceiptKind.RESPONSE:
            if self.raw_response_ref is None or self.raw_error_ref is not None:
                raise ValueError("response receipt requires only a raw response reference")
        elif self.receipt_kind == AttemptReceiptKind.ERROR:
            if self.raw_error_ref is None or self.raw_response_ref is not None:
                raise ValueError("error receipt requires only a raw error reference")
        elif self.raw_response_ref is not None or self.raw_error_ref is not None:
            raise ValueError("unknown-outcome receipt cannot contain raw evidence")
        return self


class LogicalContextRecord(StrictContract):
    schema_name: Literal["logical_context_record"] = "logical_context_record"
    schema_version: Literal[1] = 1
    context_manifest_entry_id: Sha256
    run_id: NonEmptyStr
    context_content_id: Sha256
    source_file_sha256: Sha256
    source_row_number_1_indexed: PositiveInt
    context_bytes_sha256: Sha256
    ordered_case_manifest_entry_ids: tuple[Sha256, ...]
    origin_id: NonEmptyStr


class IngestionPlanRecord(StrictContract):
    schema_name: Literal["ingestion_plan_record"] = "ingestion_plan_record"
    schema_version: Literal[1] = 1
    ingestion_occurrence_id: Sha256
    run_id: NonEmptyStr
    memory_system_id: NonEmptyStr
    ingestion_plan_id: Sha256
    ordered_member_context_manifest_entry_ids: tuple[Sha256, ...]
    ordered_case_occurrence_ids: tuple[Sha256, ...]
    state: IngestionPlanState
    intended_source_count: NonNegativeInt
    accepted_source_count: NonNegativeInt
    failed_source_count: NonNegativeInt
    readiness_evidence_refs: tuple[Sha256, ...]
    attempt_ids: tuple[Sha256, ...]
    usage_record_ids: tuple[Sha256, ...]
    resource_record_ids: tuple[Sha256, ...]
    cost_record_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def source_counts_close(self) -> Self:
        if self.accepted_source_count + self.failed_source_count > self.intended_source_count:
            raise ValueError("accepted plus failed sources exceeds intended sources")
        if self.state in {IngestionPlanState.READY, IngestionPlanState.SEALED}:
            if self.accepted_source_count + self.failed_source_count != self.intended_source_count:
                raise ValueError("READY ingestion plan requires closed source counts")
            if not self.readiness_evidence_refs:
                raise ValueError("READY ingestion plan requires readiness evidence")
        return self


class CaseRecord(StrictContract):
    schema_name: Literal["case_record"] = "case_record"
    schema_version: Literal[1] = 1
    case_occurrence_id: Sha256
    run_id: NonEmptyStr
    ingestion_occurrence_id: Sha256
    case_manifest_entry_id: Sha256
    state: CaseState
    retrieval_raw_ref: Sha256 | None
    prompt_sha256: Sha256 | None
    answer_raw_ref: Sha256 | None
    evaluation_raw_ref: Sha256 | None
    attempt_ids: tuple[Sha256, ...]
    error_stage: str | None


class CaseRecordV2(StrictContract):
    schema_name: Literal["case_record"] = "case_record"
    schema_version: Literal[2] = 2
    case_occurrence_id: Sha256
    run_id: NonEmptyStr
    ingestion_occurrence_id: Sha256
    case_manifest_entry_id: Sha256
    state: CaseState
    retrieval_raw_ref: Sha256 | None
    prompt_sha256: Sha256 | None
    answer_raw_ref: Sha256 | None
    parsed_answer_sha256: Sha256 | None
    evaluation_raw_ref: Sha256 | None
    evaluation_disposition: CaseEvaluationDisposition
    attempt_ids: tuple[Sha256, ...]
    error_stage: str | None

    @model_validator(mode="after")
    def evaluation_shape_is_explicit(self) -> Self:
        answered = (
            self.retrieval_raw_ref is not None
            and self.prompt_sha256 is not None
            and self.answer_raw_ref is not None
            and self.parsed_answer_sha256 is not None
        )
        if self.evaluation_disposition == CaseEvaluationDisposition.NOT_RUN:
            if self.parsed_answer_sha256 is not None or self.evaluation_raw_ref is not None:
                raise ValueError("not-run evaluation cannot contain parsed/evaluation output")
            if self.state == CaseState.COMPLETED:
                raise ValueError("completed case cannot have a not-run evaluation")
        elif not answered:
            raise ValueError("evaluated or unjudged case requires a parsed answer")
        if self.evaluation_disposition in {
            CaseEvaluationDisposition.DETERMINISTIC_EVALUATED,
            CaseEvaluationDisposition.JUDGED,
        }:
            if self.evaluation_raw_ref is None or self.error_stage is not None:
                raise ValueError("successful evaluation requires evidence and no error stage")
            if self.state != CaseState.COMPLETED:
                raise ValueError("successful evaluation requires a completed case")
        elif self.evaluation_disposition == CaseEvaluationDisposition.UNJUDGED:
            if (
                self.state != CaseState.ERROR
                or self.evaluation_raw_ref is not None
                or self.error_stage != "judge"
            ):
                raise ValueError("unjudged case requires a terminal judge error")
        return self


class _NativeIngestionPlanRecord(StrictContract):
    ingestion_occurrence_id: Sha256
    run_id: NonEmptyStr
    memory_system_id: NonEmptyStr
    adapter_profile_id: NonEmptyStr
    runtime_binding_hash: Sha256
    ingestion_plan_id: Sha256
    ordered_member_context_manifest_entry_ids: tuple[Sha256, ...]
    ordered_case_occurrence_ids: tuple[Sha256, ...]
    state: IngestionPlanState
    scope_id: NonEmptyStr | None
    scope_raw_refs: tuple[Sha256, ...]
    ordered_source_unit_ids: tuple[Sha256, ...]
    ordered_dispatch_attempt_ids: tuple[Sha256, ...]
    ordered_dispatch_source_unit_ids: tuple[tuple[Sha256, ...], ...]
    accepted_source_unit_ids: tuple[Sha256, ...]
    rejected_source_unit_ids: tuple[Sha256, ...]
    readiness_evidence_refs: tuple[Sha256, ...]
    inventory_raw_ref: Sha256 | None
    projected_source_unit_ids: tuple[Sha256, ...]
    projection_raw_refs: tuple[Sha256, ...]
    protected_state_sha256: Sha256 | None
    attempt_ids: tuple[Sha256, ...]
    usage_record_ids: tuple[Sha256, ...]
    resource_record_ids: tuple[Sha256, ...]
    cost_record_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def native_plan_scope_and_dispatch_close(self) -> Self:
        for label, values in (
            ("member", self.ordered_member_context_manifest_entry_ids),
            ("case", self.ordered_case_occurrence_ids),
            ("source", self.ordered_source_unit_ids),
            ("dispatch", self.ordered_dispatch_attempt_ids),
            ("accepted", self.accepted_source_unit_ids),
            ("rejected", self.rejected_source_unit_ids),
            ("attempt", self.attempt_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"native ingestion plan contains duplicate {label} identities")
        intended = set(self.ordered_source_unit_ids)
        accepted = set(self.accepted_source_unit_ids)
        rejected = set(self.rejected_source_unit_ids)
        if accepted & rejected or not (accepted | rejected) <= intended:
            raise ValueError("native ingestion accepted/rejected source partition is invalid")
        if not set(self.ordered_dispatch_attempt_ids) <= set(self.attempt_ids):
            raise ValueError(
                "native ingestion dispatch attempts are absent from the attempt ledger"
            )
        dispatch_sources = tuple(
            source_id
            for source_ids in self.ordered_dispatch_source_unit_ids
            for source_id in source_ids
        )
        if (
            len(self.ordered_dispatch_source_unit_ids) != len(self.ordered_dispatch_attempt_ids)
            or any(not source_ids for source_ids in self.ordered_dispatch_source_unit_ids)
            or dispatch_sources != self.ordered_source_unit_ids
        ):
            raise ValueError(
                "native ingestion dispatch source ledger does not close the source order"
            )
        if self.state in {IngestionPlanState.READY, IngestionPlanState.SEALED}:
            if accepted | rejected != intended:
                raise ValueError("ready native ingestion requires a closed source partition")
            if (
                self.scope_id is None
                or not self.scope_raw_refs
                or not self.readiness_evidence_refs
                or self.inventory_raw_ref is None
                or not self.projection_raw_refs
                or self.protected_state_sha256 is None
            ):
                raise ValueError(
                    "ready native ingestion requires complete scope/readiness/projection"
                )
        return self


class IngestionPlanRecordV2(_NativeIngestionPlanRecord):
    schema_name: Literal["ingestion_plan_record"] = "ingestion_plan_record"
    schema_version: Literal[2] = 2

    @model_validator(mode="after")
    def native_plan_projection_closes_accepted_sources(self) -> Self:
        if (
            self.state in {IngestionPlanState.READY, IngestionPlanState.SEALED}
            and self.projected_source_unit_ids != self.accepted_source_unit_ids
        ):
            raise ValueError("native ingestion projection does not match accepted source order")
        return self


class IngestionPlanRecordV3(_NativeIngestionPlanRecord):
    schema_name: Literal["ingestion_plan_record"] = "ingestion_plan_record"
    schema_version: Literal[3] = 3
    memory_system_id: Literal["mem0"] = "mem0"
    adapter_profile_id: Literal["mem0-rest-v1"] = "mem0-rest-v1"
    projection_semantics: Literal["retrieval_visible_subset"] = "retrieval_visible_subset"

    @model_validator(mode="after")
    def native_plan_projection_is_source_ordered_subset(self) -> Self:
        projected = self.projected_source_unit_ids
        if len(set(projected)) != len(projected):
            raise ValueError("Mem0 REST projected source identities must be unique")
        accepted = self.accepted_source_unit_ids
        accepted_positions = {source_id: index for index, source_id in enumerate(accepted)}
        if any(source_id not in accepted_positions for source_id in projected):
            raise ValueError("Mem0 REST projected source identity was not accepted")
        positions = tuple(accepted_positions[source_id] for source_id in projected)
        if positions != tuple(sorted(positions)):
            raise ValueError("Mem0 REST projected source identities changed source order")
        return self


class CaseRecordV3(StrictContract):
    schema_name: Literal["case_record"] = "case_record"
    schema_version: Literal[3] = 3
    case_occurrence_id: Sha256
    run_id: NonEmptyStr
    ingestion_occurrence_id: Sha256
    case_manifest_entry_id: Sha256
    adapter_profile_id: NonEmptyStr
    state: CaseState
    retrieval_raw_ref: Sha256 | None
    retrieval_supporting_raw_refs: tuple[Sha256, ...]
    ordered_native_candidate_ids: tuple[NonEmptyStr, ...]
    ordered_native_content_sha256: tuple[Sha256, ...]
    native_candidate_source_unit_ids: tuple[Sha256 | None, ...]
    visible_evidence_raw_ref: Sha256 | None
    visible_evidence_sha256: Sha256 | None
    visible_evidence_byte_count: NonNegativeInt | None
    visible_evidence_token_count: NonNegativeInt | None
    visible_evidence_tokenizer_fingerprint: Sha256 | None
    native_candidate_count: NonNegativeInt
    visible_kept_count: NonNegativeInt
    visible_dropped_count: NonNegativeInt
    visible_truncated_count: NonNegativeInt
    visible_decision_ledger_raw_ref: Sha256 | None
    pre_query_projection_raw_refs: tuple[Sha256, ...]
    pre_query_state_sha256: Sha256 | None
    post_query_projection_raw_refs: tuple[Sha256, ...]
    post_query_state_sha256: Sha256 | None
    query_mutation_status: Literal["unchanged", "changed", "unavailable"]
    prompt_raw_ref: Sha256 | None
    prompt_sha256: Sha256 | None
    judge_prompt_raw_ref: Sha256 | None
    answer_raw_ref: Sha256 | None
    parsed_answer_sha256: Sha256 | None
    metric_id: NonEmptyStr | None
    metric_numerator: NonNegativeInt | None
    metric_denominator: PositiveInt | None
    evaluation_raw_ref: Sha256 | None
    evaluation_disposition: CaseEvaluationDisposition
    attempt_ids: tuple[Sha256, ...]
    usage_record_ids: tuple[Sha256, ...]
    resource_record_ids: tuple[Sha256, ...]
    cost_record_ids: tuple[Sha256, ...]
    error_stage: NonEmptyStr | None

    @model_validator(mode="after")
    def native_case_evidence_and_metric_close(self) -> Self:
        candidate_lengths = {
            len(self.ordered_native_candidate_ids),
            len(self.ordered_native_content_sha256),
            len(self.native_candidate_source_unit_ids),
        }
        if len(candidate_lengths) != 1:
            raise ValueError(
                "native candidate identities, content, and attribution must be aligned"
            )
        if len(set(self.ordered_native_candidate_ids)) != len(self.ordered_native_candidate_ids):
            raise ValueError("native candidate identities must be unique within one response")
        if self.native_candidate_count != len(self.ordered_native_candidate_ids):
            raise ValueError("native candidate count does not match the ordered retrieval")
        if self.visible_kept_count + self.visible_dropped_count != self.native_candidate_count:
            raise ValueError("visible evidence kept/dropped counts do not close")
        if self.visible_truncated_count > self.visible_kept_count:
            raise ValueError("visible evidence truncated count exceeds kept evidence")
        if (self.metric_numerator is None) != (self.metric_denominator is None):
            raise ValueError("metric fraction requires both numerator and denominator")
        if (
            self.metric_numerator is not None
            and self.metric_denominator is not None
            and self.metric_numerator > self.metric_denominator
        ):
            raise ValueError("metric fraction numerator exceeds its denominator")
        if self.state == CaseState.COMPLETED:
            if self.prompt_raw_ref is None:
                raise ValueError("completed native case requires prompt evidence")
            required = (
                self.retrieval_raw_ref,
                self.visible_evidence_raw_ref,
                self.visible_evidence_sha256,
                self.visible_evidence_byte_count,
                self.visible_evidence_token_count,
                self.visible_evidence_tokenizer_fingerprint,
                self.visible_decision_ledger_raw_ref,
                self.pre_query_state_sha256,
                self.post_query_state_sha256,
                self.prompt_sha256,
                self.answer_raw_ref,
                self.parsed_answer_sha256,
                self.metric_id,
                self.metric_numerator,
                self.metric_denominator,
                self.evaluation_raw_ref,
            )
            if any(value is None for value in required):
                raise ValueError(
                    "completed native case requires closed retrieval/context/metric evidence"
                )
            if not self.pre_query_projection_raw_refs or not self.post_query_projection_raw_refs:
                raise ValueError("completed native case requires both protected projections")
            if (
                self.query_mutation_status != "unchanged"
                or self.pre_query_state_sha256 != self.post_query_state_sha256
            ):
                raise ValueError("completed read-only case requires unchanged protected state")
            if self.error_stage is not None:
                raise ValueError("completed native case cannot carry an error stage")
            if self.evaluation_disposition not in {
                CaseEvaluationDisposition.DETERMINISTIC_EVALUATED,
                CaseEvaluationDisposition.JUDGED,
            }:
                raise ValueError("completed native case requires a successful evaluation")
            if (
                self.evaluation_disposition == CaseEvaluationDisposition.JUDGED
                and self.judge_prompt_raw_ref is None
            ):
                raise ValueError("judged native case requires judge prompt evidence")
            if (
                self.evaluation_disposition == CaseEvaluationDisposition.DETERMINISTIC_EVALUATED
                and self.judge_prompt_raw_ref is not None
            ):
                raise ValueError("deterministic native case cannot carry judge prompt evidence")
        return self


class PhaseReviewOccurrenceRecord(StrictContract):
    schema_name: Literal["phase_review_occurrence_record"] = "phase_review_occurrence_record"
    schema_version: Literal[1] = 1
    phase_review_occurrence_id: Sha256
    phase_id: NonEmptyStr
    review_bundle_hash: Sha256
    reviewer_role_binding_hash: Sha256
    ordinal: PositiveInt
    approval_record_id: Sha256
    budget_id: NonEmptyStr
    state: Literal[
        "planned",
        "budget_reserved",
        "running",
        "sealed",
        "error",
        "cancelled",
        "budget_exceeded",
        "interrupted_unknown_outcome",
    ]
    started_at: UtcDateTime | None
    ended_at: UtcDateTime | None

    @model_validator(mode="after")
    def occurrence_identity_and_times_are_canonical(self) -> Self:
        expected = phase_review_occurrence_id(
            phase_id=self.phase_id,
            review_bundle_hash=self.review_bundle_hash,
            reviewer_role_binding_hash=self.reviewer_role_binding_hash,
            ordinal=self.ordinal,
        )
        if self.phase_review_occurrence_id != expected:
            raise ValueError("phase-review occurrence identity does not match its payload")
        if (self.started_at is None) != (self.ended_at is None):
            raise ValueError("phase-review occurrence timing requires both endpoints")
        if self.started_at is not None and self.ended_at is not None:
            if self.ended_at < self.started_at:
                raise ValueError("phase-review occurrence ends before it starts")
        return self


def phase_review_occurrence_id(
    *,
    phase_id: str,
    review_bundle_hash: str,
    reviewer_role_binding_hash: str,
    ordinal: int,
) -> str:
    return canonical_sha256(
        [
            "oamb-phase-review-occurrence-v1",
            phase_id,
            review_bundle_hash,
            reviewer_role_binding_hash,
            ordinal,
        ]
    )


class PhaseReviewOccurrenceRecordV2(StrictContract):
    schema_name: Literal["phase_review_occurrence_record"] = "phase_review_occurrence_record"
    schema_version: Literal[2] = 2
    phase_review_occurrence_id: Sha256
    phase_id: NonEmptyStr
    review_bundle_hash: Sha256
    reviewer_role_binding_hash: Sha256
    artifact_repository_fingerprint: Sha256
    ordinal: PositiveInt
    approval_record_id: Sha256
    budget_id: NonEmptyStr
    state: Literal[
        "planned",
        "budget_reserved",
        "running",
        "sealed",
        "error",
        "cancelled",
        "budget_exceeded",
        "evidence_inconclusive",
        "interrupted_unknown_outcome",
    ]
    started_at: UtcDateTime | None
    ended_at: UtcDateTime | None

    @model_validator(mode="after")
    def occurrence_identity_and_times_are_canonical(self) -> Self:
        expected = phase_review_occurrence_id_v2(
            phase_id=self.phase_id,
            review_bundle_hash=self.review_bundle_hash,
            reviewer_role_binding_hash=self.reviewer_role_binding_hash,
            artifact_repository_fingerprint=self.artifact_repository_fingerprint,
            ordinal=self.ordinal,
        )
        if self.phase_review_occurrence_id != expected:
            raise ValueError("phase-review occurrence identity does not match its payload")
        if (self.started_at is None) != (self.ended_at is None):
            raise ValueError("phase-review occurrence timing requires both endpoints")
        if self.started_at is not None and self.ended_at is not None:
            if self.ended_at < self.started_at:
                raise ValueError("phase-review occurrence ends before it starts")
        return self


def phase_review_occurrence_id_v2(
    *,
    phase_id: str,
    review_bundle_hash: str,
    reviewer_role_binding_hash: str,
    artifact_repository_fingerprint: str,
    ordinal: int,
) -> str:
    return canonical_sha256(
        [
            "oamb-phase-review-occurrence-v2",
            phase_id,
            review_bundle_hash,
            reviewer_role_binding_hash,
            artifact_repository_fingerprint,
            ordinal,
        ]
    )


class CapsuleManifestEntry(StrictContract):
    schema_name: Literal["capsule_manifest_entry"] = "capsule_manifest_entry"
    schema_version: Literal[1] = 1
    record_kind: NonEmptyStr
    record_id: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: Sha256

    @model_validator(mode="after")
    def relative_safe_path(self) -> Self:
        components = self.relative_path.split("/")
        if self.relative_path.startswith("/") or ".." in components:
            raise ValueError("capsule path must be relative and non-escaping")
        return self


class CapsuleManifest(StrictContract):
    schema_name: Literal["capsule_manifest"] = "capsule_manifest"
    schema_version: Literal[1] = 1
    capsule_id: Sha256
    run_id: NonEmptyStr
    run_spec_hash: Sha256
    source_entries: tuple[CapsuleManifestEntry, ...]
    source_manifest_hash: Sha256

    @model_validator(mode="after")
    def excludes_self_and_checkpoints(self) -> Self:
        paths = tuple(item.relative_path for item in self.source_entries)
        if len(set(paths)) != len(paths):
            raise ValueError("capsule manifest contains duplicate paths")
        if any(
            path == "capsule-manifest.json" or path.startswith("checkpoints/") for path in paths
        ):
            raise ValueError("capsule manifest cannot index itself or checkpoints")
        return self


class CheckpointManifest(StrictContract):
    schema_name: Literal["checkpoint_manifest"] = "checkpoint_manifest"
    schema_version: Literal[1] = 1
    checkpoint_manifest_hash: Sha256
    run_id: NonEmptyStr
    lease_record_hash: Sha256
    lease_epoch: PositiveInt
    sequence: PositiveInt
    predecessor_checkpoint_hash: Sha256 | None
    source_entries: tuple[CapsuleManifestEntry, ...]
    created_at: UtcDateTime

    @model_validator(mode="after")
    def predecessor_and_entries_close(self) -> Self:
        if self.sequence == 1 and self.predecessor_checkpoint_hash is not None:
            raise ValueError("first checkpoint cannot have a predecessor")
        if self.sequence > 1 and self.predecessor_checkpoint_hash is None:
            raise ValueError("later checkpoint requires a predecessor")
        paths = tuple(item.relative_path for item in self.source_entries)
        if len(set(paths)) != len(paths):
            raise ValueError("checkpoint contains duplicate source paths")
        return self


class ProviderServiceEvidenceManifest(StrictContract):
    schema_name: Literal["provider_service_evidence_manifest"] = (
        "provider_service_evidence_manifest"
    )
    schema_version: Literal[1] = 1
    manifest_hash: Sha256
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    occurrence_id: NonEmptyStr
    operation: Literal["model_readiness"]
    source_entries: tuple[CapsuleManifestEntry, ...]
    created_at: UtcDateTime

    @model_validator(mode="after")
    def source_entries_are_unique(self) -> Self:
        if not self.source_entries:
            raise ValueError("provider-service manifest requires source entries")
        paths = tuple(item.relative_path for item in self.source_entries)
        if len(set(paths)) != len(paths):
            raise ValueError("provider-service manifest contains duplicate source paths")
        if any(not path.startswith("source/") for path in paths):
            raise ValueError("provider-service manifest may index only source paths")
        return self


class RecoveryDecisionRecord(StrictContract):
    schema_name: Literal["recovery_decision_record"] = "recovery_decision_record"
    schema_version: Literal[1] = 1
    recovery_decision_id: Sha256
    run_id: NonEmptyStr
    previous_lease_record_hash: Sha256
    new_lease_record_hash: Sha256 | None
    previous_lease_epoch: PositiveInt
    new_lease_epoch: PositiveInt | None
    authorization_id: NonEmptyStr
    archived_checkpoint_hashes: tuple[Sha256, ...]
    disposition: RecoveryDisposition
    decided_at: UtcDateTime

    @model_validator(mode="after")
    def replacement_epoch_shape(self) -> Self:
        if (self.new_lease_record_hash is None) != (self.new_lease_epoch is None):
            raise ValueError("new lease record and epoch must be present together")
        if self.disposition == RecoveryDisposition.RESUME_SAFE:
            if self.new_lease_epoch is None or self.new_lease_epoch <= self.previous_lease_epoch:
                raise ValueError("resume-safe recovery requires a later lease epoch")
        elif self.new_lease_record_hash is not None:
            raise ValueError("terminal recovery disposition cannot create a new lease")
        if len(set(self.archived_checkpoint_hashes)) != len(self.archived_checkpoint_hashes):
            raise ValueError("recovery decision contains duplicate archived checkpoints")
        return self


class CloseErrorRecord(StrictContract):
    schema_name: Literal["close_error_record"] = "close_error_record"
    schema_version: Literal[1] = 1
    close_error_id: Sha256
    owner_kind: Literal["run", "phase_review", "model_readiness", "worker"]
    owner_id: NonEmptyStr
    client_profile_id: NonEmptyStr
    error_ref: Sha256
    shutdown_stage: NonEmptyStr
    occurred_at: UtcDateTime


class DerivationManifest(StrictContract):
    schema_name: Literal["derivation_manifest"] = "derivation_manifest"
    schema_version: Literal[1] = 1
    derivation_id: Sha256
    derivation_spec_hash: Sha256
    evidence_validation_result_hash: Sha256
    export_validation_result_hash: Sha256
    committed_entries: tuple[CapsuleManifestEntry, ...]
    committed_at: UtcDateTime

    @model_validator(mode="after")
    def committed_entries_are_unique(self) -> Self:
        if not self.committed_entries:
            raise ValueError("derivation manifest requires committed entries")
        paths = tuple(item.relative_path for item in self.committed_entries)
        if len(set(paths)) != len(paths):
            raise ValueError("derivation manifest contains duplicate committed paths")
        if any(path in {"CHECKSUMS.sha256", "derived-manifest.json"} for path in paths):
            raise ValueError("derivation manifest cannot index itself or checksums")
        return self


class ValidationIssue(StrictContract):
    schema_name: Literal["validation_issue"] = "validation_issue"
    schema_version: Literal[1] = 1
    rule_id: NonEmptyStr
    code: NonEmptyStr
    severity: ValidationSeverity
    evidence_ref: NonEmptyStr
    json_pointer: str | None
    remediation_code: NonEmptyStr


class ValidationResult(StrictContract):
    schema_name: Literal["validation_result"] = "validation_result"
    schema_version: Literal[1] = 1
    validation_profile_id: NonEmptyStr
    target_hash: Sha256
    disposition: ValidationDisposition
    required_rule_ids: tuple[NonEmptyStr, ...]
    executed_rule_ids: tuple[NonEmptyStr, ...]
    passed_rule_ids: tuple[NonEmptyStr, ...]
    failed_rule_ids: tuple[NonEmptyStr, ...]
    not_applicable_rule_ids: tuple[NonEmptyStr, ...]
    missing_rule_ids: tuple[NonEmptyStr, ...]
    implementation_versions: tuple[NonEmptyStr, ...]
    issues: tuple[ValidationIssue, ...]

    @model_validator(mode="after")
    def rule_partitions_and_disposition_close(self) -> Self:
        categories = {
            "required": self.required_rule_ids,
            "executed": self.executed_rule_ids,
            "passed": self.passed_rule_ids,
            "failed": self.failed_rule_ids,
            "not_applicable": self.not_applicable_rule_ids,
            "missing": self.missing_rule_ids,
        }
        if any(len(set(values)) != len(values) for values in categories.values()):
            raise ValueError("validation rule partitions cannot contain duplicates")
        required = set(self.required_rule_ids)
        executed = set(self.executed_rule_ids)
        passed = set(self.passed_rule_ids)
        failed = set(self.failed_rule_ids)
        failed_required = failed & required
        not_applicable = set(self.not_applicable_rule_ids)
        missing = set(self.missing_rule_ids)
        if executed != passed | failed_required or passed & failed_required:
            raise ValueError("executed rules must partition into passed and failed rules")
        if any(
            left & right
            for left, right in (
                (executed, not_applicable),
                (executed, missing),
                (not_applicable, missing),
            )
        ):
            raise ValueError("required rule partitions must be disjoint")
        if required != executed | not_applicable | missing:
            raise ValueError(
                "required rules must have an executed, not-applicable, or missing result"
            )
        implemented_rule_ids = tuple(
            item.rsplit("@", 1)[0] for item in self.implementation_versions
        )
        if implemented_rule_ids != self.executed_rule_ids:
            raise ValueError("implementation versions must align with executed rules")
        error_issue_rule_ids = {
            issue.rule_id for issue in self.issues if issue.severity == ValidationSeverity.ERROR
        }
        if error_issue_rule_ids != failed:
            raise ValueError("error issues must correspond exactly to failed rules")
        if any(issue.rule_id not in executed | failed for issue in self.issues):
            raise ValueError("validation issues must correspond to executed or failed rules")
        if self.disposition == ValidationDisposition.VALIDATED:
            if failed or missing:
                raise ValueError("validated result cannot contain failed or missing rules")
        elif not failed and not missing:
            raise ValueError("invalid result requires a failed or missing rule")
        return self
