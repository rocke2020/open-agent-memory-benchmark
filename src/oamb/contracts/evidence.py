"""Immutable source evidence and structural validation records."""

from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from typing import Any, Literal, Self

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
from .ids import canonical_sha256, ingestion_occurrence_id
from .ingestion_failures import (
    HINDSIGHT_SETTLEMENT_BASIS,
    MEM0_SETTLEMENT_BASIS,
    OPENVIKING_SETTLEMENT_BASIS,
)
from .specifications import (
    INFRASTRUCTURE_RETRY_BACKOFF_SECONDS,
    MEMORY_CONFORMANCE_ROUTE_STAGES,
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
    parent_kind: Literal["ingestion_plan", "case"]
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
    parent_kind: Literal["ingestion_plan", "case", "model_readiness"]
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    ordinal: PositiveInt
    request_fingerprint: Sha256
    request_messages_sha256: Sha256 | None = None
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
    parent_kind: Literal["ingestion_plan", "case", "model_readiness", "memory_conformance"]
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
        if (
            self.parent_kind == "memory_conformance"
            and self.stage not in MEMORY_CONFORMANCE_ROUTE_STAGES
        ):
            raise ValueError("memory-conformance attempt uses an unsupported stage")
        if self.index_contribution == IndexContribution.SUPERSEDED:
            if self.superseded_by_attempt_id is None:
                raise ValueError("superseded contribution requires successor attempt")
        elif self.superseded_by_attempt_id is not None:
            raise ValueError("only superseded contribution names a successor")
        return self


class AttemptRecordV4(StrictContract):
    schema_name: Literal["attempt_record"] = "attempt_record"
    schema_version: Literal[4] = 4
    attempt_id: Sha256
    attempt_record_hash: Sha256
    intent_hash: Sha256
    dispatch_route_id: NonEmptyStr
    dispatch_route_hash: Sha256
    receipt_record_hash: Sha256
    run_id: NonEmptyStr
    parent_kind: Literal["ingestion_plan", "case"]
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    ordinal: PositiveInt
    request_fingerprint: Sha256
    request_messages_sha256: Sha256 | None = None
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
    def live_run_attempt_closes(self) -> Self:
        expected_hash = attempt_record_v4_hash(
            self.model_dump(mode="python", exclude={"attempt_record_hash"})
        )
        if self.attempt_record_hash != expected_hash:
            raise ValueError("attempt record hash does not match its canonical fields")
        if self.ended_at < self.started_at:
            raise ValueError("attempt end precedes start")
        if self.reconciliation_capability == "idempotency_key":
            if self.idempotency_key_hash is None:
                raise ValueError("idempotency-key reconciliation requires its hash")
        elif self.idempotency_key_hash is not None:
            raise ValueError("only idempotency-key reconciliation names its hash")
        if self.index_contribution == IndexContribution.SUPERSEDED:
            if self.superseded_by_attempt_id is None:
                raise ValueError("superseded contribution requires successor attempt")
        elif self.superseded_by_attempt_id is not None:
            raise ValueError("only superseded contribution names a successor")
        return self


def infrastructure_retry_event_id(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-infrastructure-retry-event-v1", fields])


def infrastructure_supplier_call_id(
    logical_attempt_id: str,
    supplier_call_ordinal: int,
    raw_error_ref: str,
) -> str:
    return canonical_sha256(
        [
            "oamb-infrastructure-supplier-call-v1",
            logical_attempt_id,
            supplier_call_ordinal,
            raw_error_ref,
        ]
    )


class InfrastructureRetryEvent(StrictContract):
    schema_name: Literal["infrastructure_retry_event"] = "infrastructure_retry_event"
    schema_version: Literal[1] = 1
    retry_event_id: Sha256
    run_id: NonEmptyStr
    logical_attempt_id: Sha256
    parent_kind: Literal["ingestion_plan", "case"]
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    supplier_call_ordinal: PositiveInt
    supplier_call_id: Sha256
    origin: Literal["model_supplier"]
    failure_kind: Literal["rate_limited"]
    status: Literal[429]
    acceptance: Literal["not_accepted"]
    provider_mutation: Literal["none"]
    retryable: Literal[True]
    internal_retry_count: NonNegativeInt
    raw_error_ref: Sha256
    usage_record_ids: tuple[Sha256, ...]
    retry_policy_hash: Sha256
    retry_scheduled: bool
    backoff_seconds: NonNegativeInt | None
    observed_at: UtcDateTime

    @model_validator(mode="after")
    def event_identity_and_backoff_are_closed(self) -> Self:
        if self.retry_scheduled != (self.backoff_seconds is not None):
            raise ValueError("infrastructure retry schedule and backoff do not match")
        if self.retry_scheduled and self.backoff_seconds == 0:
            raise ValueError("scheduled infrastructure retry requires a positive backoff")
        if len(set(self.usage_record_ids)) != len(self.usage_record_ids):
            raise ValueError("infrastructure retry usage inventory contains duplicates")
        expected = infrastructure_retry_event_id(
            self.model_dump(mode="python", exclude={"retry_event_id"})
        )
        if self.retry_event_id != expected:
            raise ValueError("infrastructure retry event identity does not match its fields")
        return self


def attempt_record_v4_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "attempt_record")
    payload.setdefault("schema_version", 4)
    payload.pop("attempt_record_hash", None)
    return canonical_sha256(payload)


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


class BudgetOwnerAllocation(StrictContract):
    schema_name: Literal["budget_owner_allocation"] = "budget_owner_allocation"
    schema_version: Literal[1] = 1
    allocation_hash: Sha256
    owner_kind: DispatchBudgetOwnerKind
    owner_id: NonEmptyStr
    allocated_attempts: PositiveInt
    allocated_input_tokens: NonNegativeInt
    allocated_output_tokens: NonNegativeInt
    allocated_dispatch_wall_seconds: NonNegativeDecimal
    allocated_provider_units: NonNegativeDecimal
    allocated_resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    allocated_cost: NonNegativeDecimal | None
    currency: str | None

    @model_validator(mode="after")
    def identity_currency_and_resources_close(self) -> Self:
        expected_hash = budget_owner_allocation_hash(
            self.model_dump(mode="python", exclude={"allocation_hash"})
        )
        if self.allocation_hash != expected_hash:
            raise ValueError("budget-owner allocation hash does not match its canonical fields")
        if (self.allocated_cost is None) != (self.currency is None):
            raise ValueError("allocated cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        dimensions = tuple(item.dimension_id for item in self.allocated_resource_ceilings)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("allocation contains duplicate resource dimensions")
        return self


def budget_owner_allocation_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "budget_owner_allocation")
    payload.setdefault("schema_version", 1)
    payload.pop("allocation_hash", None)
    return canonical_sha256(payload)


class BudgetReservationRecordV3(StrictContract):
    schema_name: Literal["budget_reservation_record"] = "budget_reservation_record"
    schema_version: Literal[3] = 3
    reservation_id: Sha256
    reservation_hash: Sha256
    budget_id: NonEmptyStr
    budget_hash: Sha256
    scope_kind: Literal[BudgetScopeKindV3.RUN, BudgetScopeKindV3.MEMORY_CONFORMANCE]
    scope_id: NonEmptyStr
    attempt_id: Sha256
    dispatch_route_id: NonEmptyStr
    dispatch_route_hash: Sha256
    owner_allocations: tuple[BudgetOwnerAllocation, ...]
    reserved_attempts: PositiveInt
    reserved_input_tokens: NonNegativeInt
    reserved_output_tokens: NonNegativeInt
    reserved_dispatch_wall_seconds: NonNegativeDecimal
    reserved_provider_units: NonNegativeDecimal
    reserved_resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    reserved_cost: NonNegativeDecimal | None
    currency: str | None
    reserved_at: UtcDateTime

    @model_validator(mode="after")
    def identity_owner_order_and_aggregate_close(self) -> Self:
        identity_fields = self.model_dump(
            mode="python", exclude={"reservation_id", "reservation_hash"}
        )
        if self.reservation_id != budget_reservation_v3_id(identity_fields):
            raise ValueError("budget reservation identity does not match its canonical fields")
        expected_hash = budget_reservation_v3_hash(
            self.model_dump(mode="python", exclude={"reservation_hash"})
        )
        if self.reservation_hash != expected_hash:
            raise ValueError("budget reservation hash does not match its canonical fields")
        if not self.owner_allocations:
            raise ValueError("version 3 reservation requires owner allocations")
        owner_ids = tuple(item.owner_id for item in self.owner_allocations)
        if len(set(owner_ids)) != len(owner_ids):
            raise ValueError("reservation contains duplicate budget owners")
        first_kind = self.owner_allocations[0].owner_kind
        if first_kind == DispatchBudgetOwnerKind.MODEL_ROLE:
            if len(self.owner_allocations) != 1:
                raise ValueError("direct model reservation requires exactly one allocation")
        elif any(
            item.owner_kind != DispatchBudgetOwnerKind.MODEL_ROLE
            for item in self.owner_allocations[1:]
        ):
            raise ValueError("provider-operation allocation first, then model-role allocations")
        if (self.reserved_cost is None) != (self.currency is None):
            raise ValueError("reserved cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        if any(item.currency != self.currency for item in self.owner_allocations):
            raise ValueError("allocation currency must match the reservation currency")
        dimensions = tuple(item.dimension_id for item in self.reserved_resource_ceilings)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("reservation aggregate contains duplicate resource dimensions")

        expected_resources = _sum_allocation_resources(self.owner_allocations)
        expected_cost = (
            None
            if self.currency is None
            else sum(
                (item.allocated_cost or Decimal("0") for item in self.owner_allocations),
                Decimal("0"),
            )
        )
        aggregate_matches = (
            self.reserved_attempts
            == sum(item.allocated_attempts for item in self.owner_allocations)
            and self.reserved_input_tokens
            == sum(item.allocated_input_tokens for item in self.owner_allocations)
            and self.reserved_output_tokens
            == sum(item.allocated_output_tokens for item in self.owner_allocations)
            and self.reserved_dispatch_wall_seconds
            == sum(
                (item.allocated_dispatch_wall_seconds for item in self.owner_allocations),
                Decimal("0"),
            )
            and self.reserved_provider_units
            == sum(
                (item.allocated_provider_units for item in self.owner_allocations),
                Decimal("0"),
            )
            and self.reserved_resource_ceilings == expected_resources
            and self.reserved_cost == expected_cost
        )
        if not aggregate_matches:
            raise ValueError("reservation aggregate does not equal its owner allocation sum")
        return self


def _sum_allocation_resources(
    allocations: tuple[BudgetOwnerAllocation, ...],
) -> tuple[ResourceBudgetCeiling, ...]:
    amounts: dict[str, Decimal] = {}
    units: dict[str, str] = {}
    order: list[str] = []
    for allocation in allocations:
        for resource in allocation.allocated_resource_ceilings:
            dimension = resource.dimension_id
            if dimension in units and units[dimension] != resource.unit:
                raise ValueError("allocation resource units disagree")
            if dimension not in amounts:
                order.append(dimension)
                amounts[dimension] = Decimal("0")
                units[dimension] = resource.unit
            amounts[dimension] += resource.maximum
    return tuple(
        ResourceBudgetCeiling(
            dimension_id=dimension,
            maximum=amounts[dimension],
            unit=units[dimension],
        )
        for dimension in order
    )


def budget_reservation_v3_id(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "budget_reservation_record")
    payload.setdefault("schema_version", 3)
    payload.pop("reservation_id", None)
    payload.pop("reservation_hash", None)
    return canonical_sha256(["oamb-budget-reservation-v3", payload])


def budget_reservation_v3_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "budget_reservation_record")
    payload.setdefault("schema_version", 3)
    payload.pop("reservation_hash", None)
    return canonical_sha256(payload)


class AttemptIntentRecord(StrictContract):
    schema_name: Literal["attempt_intent_record"] = "attempt_intent_record"
    schema_version: Literal[1] = 1
    attempt_id: Sha256
    claim_id: Sha256
    reservation_id: Sha256
    parent_kind: Literal["ingestion_plan", "case", "model_readiness"]
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
    parent_kind: Literal["ingestion_plan", "case", "model_readiness", "memory_conformance"]
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


class AttemptIntentRecordV3(StrictContract):
    schema_name: Literal["attempt_intent_record"] = "attempt_intent_record"
    schema_version: Literal[3] = 3
    attempt_id: Sha256
    intent_hash: Sha256
    claim_id: Sha256
    reservation_id: Sha256
    reservation_hash: Sha256
    scope_kind: Literal[BudgetScopeKindV3.RUN, BudgetScopeKindV3.MEMORY_CONFORMANCE]
    scope_id: NonEmptyStr
    parent_kind: Literal["ingestion_plan", "case", "memory_conformance"]
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    preflight_record_hash: Sha256 | None
    budget_id: NonEmptyStr
    budget_hash: Sha256
    dispatch_route_id: NonEmptyStr
    dispatch_route_hash: Sha256
    request_fingerprint: Sha256
    reconciliation_capability: Literal["none", "idempotency_key", "receipt_lookup"]
    idempotency_key_hash: Sha256 | None
    sealed_at: UtcDateTime

    @model_validator(mode="after")
    def scope_identity_and_reconciliation_close(self) -> Self:
        expected_hash = attempt_intent_v3_hash(
            self.model_dump(mode="python", exclude={"intent_hash"})
        )
        if self.intent_hash != expected_hash:
            raise ValueError("attempt intent hash does not match its canonical fields")
        if self.scope_kind == BudgetScopeKindV3.RUN:
            if self.parent_kind not in {"ingestion_plan", "case"}:
                raise ValueError("run intent requires an ingestion-plan or case parent")
            if self.preflight_record_hash is None:
                raise ValueError("run intent requires its preflight record hash")
        else:
            if self.parent_kind != "memory_conformance" or self.parent_id != self.scope_id:
                raise ValueError("memory-conformance intent requires its occurrence parent")
            if self.preflight_record_hash is not None:
                raise ValueError("memory-conformance intent has no run preflight record")
        if self.reconciliation_capability == "idempotency_key":
            if self.idempotency_key_hash is None:
                raise ValueError("idempotency-key reconciliation requires its hash")
        elif self.idempotency_key_hash is not None:
            raise ValueError("only idempotency-key reconciliation names its hash")
        return self


def attempt_intent_v3_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "attempt_intent_record")
    payload.setdefault("schema_version", 3)
    payload.pop("intent_hash", None)
    return canonical_sha256(payload)


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

    failure_kind: NonEmptyStr | None = None
    supplier_status_code: int | None = None
    settlement_basis: NonEmptyStr | None = None
    internal_retry_count: NonNegativeInt | None = None
    settlement_task_id: NonEmptyStr | None = None
    settlement_session_id: NonEmptyStr | None = None

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
    skipped_source_unit_ids: tuple[Sha256, ...] = ()
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
            ("skipped", self.skipped_source_unit_ids),
            ("attempt", self.attempt_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"native ingestion plan contains duplicate {label} identities")
        intended = set(self.ordered_source_unit_ids)
        accepted = set(self.accepted_source_unit_ids)
        rejected = set(self.rejected_source_unit_ids)
        skipped = set(self.skipped_source_unit_ids)
        if (
            accepted & rejected
            or accepted & skipped
            or rejected & skipped
            or not (accepted | rejected | skipped) <= intended
        ):
            raise ValueError("native ingestion source partition is invalid")
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
            if accepted | rejected | skipped != intended:
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
    def native_plan_projection_is_source_ordered_observed_subset(self) -> Self:
        projected = self.projected_source_unit_ids
        intended_positions = {
            source_id: index for index, source_id in enumerate(self.ordered_source_unit_ids)
        }
        observable = set(self.accepted_source_unit_ids) | set(self.skipped_source_unit_ids)
        if len(set(projected)) != len(projected) or any(
            source_id not in observable for source_id in projected
        ):
            raise ValueError("native ingestion projection is not an observed source-ordered subset")
        positions = tuple(intended_positions[source_id] for source_id in projected)
        if positions != tuple(sorted(positions)):
            raise ValueError("native ingestion projection is not an observed source-ordered subset")
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
        observable = set(self.accepted_source_unit_ids) | set(self.skipped_source_unit_ids)
        intended_positions = {
            source_id: index for index, source_id in enumerate(self.ordered_source_unit_ids)
        }
        if any(source_id not in observable for source_id in projected):
            raise ValueError("Mem0 REST projected source identity was not accepted or skipped")
        positions = tuple(intended_positions[source_id] for source_id in projected)
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
    retrieval_request_raw_ref: Sha256 | None = None
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


def capsule_composition_part_binding_hash(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-capsule-composition-part-v1", fields])


class CapsuleCompositionPartBinding(StrictContract):
    schema_name: Literal["capsule_composition_part_binding"] = "capsule_composition_part_binding"
    schema_version: Literal[1] = 1
    part_binding_hash: Sha256
    capsule_id: Sha256
    run_id: NonEmptyStr
    manifest_sha256: Sha256
    partition_id: Sha256
    embedded_root: NonEmptyStr
    run_state: RunState

    @model_validator(mode="after")
    def binding_hash_matches_fields(self) -> Self:
        if self.embedded_root != f"source/parts/{self.capsule_id}":
            raise ValueError("composition part embedded root is not canonical")
        expected = capsule_composition_part_binding_hash(
            self.model_dump(mode="python", exclude={"part_binding_hash"})
        )
        if self.part_binding_hash != expected:
            raise ValueError("composition part binding hash does not match its fields")
        return self


def history_attempt_id(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-history-attempt-initial-v1", fields])


class HistoryAttemptRecord(StrictContract):
    schema_name: Literal["history_attempt_record"] = "history_attempt_record"
    schema_version: Literal[1] = 1
    history_attempt_id: Sha256
    run_id: NonEmptyStr
    execution_run_id: NonEmptyStr
    ingestion_plan_id: Sha256
    history_attempt_ordinal: PositiveInt
    ingestion_occurrence_id: Sha256
    memory_system_id: NonEmptyStr
    runtime_binding_hash: Sha256
    retry_policy_hash: Sha256
    max_retries_per_operation: NonNegativeInt
    history_input_hash: Sha256
    scope_id: NonEmptyStr | None
    scope_raw_refs: tuple[Sha256, ...]
    previous_retry_event_id: Sha256 | None
    admission_claim_raw_ref: Sha256 | None
    status: Literal["ready", "retryable_failed_settled", "failed", "unknown", "cancelled"]
    operation_attempt_ids: tuple[Sha256, ...]
    terminal_failure_attempt_id: Sha256 | None
    settlement_basis: NonEmptyStr | None
    settlement_evidence_refs: tuple[Sha256, ...]
    settlement_status_code: int | None
    settlement_task_id: NonEmptyStr | None
    settlement_session_id: NonEmptyStr | None
    internal_retry_count: NonNegativeInt | None
    failure_kind: NonEmptyStr | None
    ingestion_plan_record_hash: Sha256 | None
    started_at: UtcDateTime
    ended_at: UtcDateTime

    @model_validator(mode="after")
    def history_attempt_is_closed(self) -> Self:
        if self.max_retries_per_operation not in {0, 1, 2}:
            raise ValueError("history attempt retry limit must be 0, 1, or 2")
        if self.history_attempt_ordinal > self.max_retries_per_operation + 1:
            raise ValueError("history attempt ordinal exceeds its retry allowance")
        if self.history_attempt_ordinal == 1:
            if self.previous_retry_event_id is not None or self.admission_claim_raw_ref is not None:
                raise ValueError("first history attempt cannot name retry admission evidence")
        elif self.previous_retry_event_id is None or self.admission_claim_raw_ref is None:
            raise ValueError("successor history attempt requires retry event and admission claim")
        expected_occurrence = ingestion_occurrence_id(
            self.execution_run_id,
            self.memory_system_id,
            self.ingestion_plan_id,
            history_attempt_ordinal=self.history_attempt_ordinal,
        )
        if self.ingestion_occurrence_id != expected_occurrence:
            raise ValueError("history attempt ingestion occurrence identity does not match")
        if len(set(self.operation_attempt_ids)) != len(self.operation_attempt_ids):
            raise ValueError("history attempt operation IDs must be unique")
        for references in (self.scope_raw_refs, self.settlement_evidence_refs):
            if len(set(references)) != len(references):
                raise ValueError("history attempt evidence references must be unique")
        if self.ended_at < self.started_at:
            raise ValueError("history attempt end precedes its start")
        if self.status == "ready":
            if self.scope_id is None or self.ingestion_plan_record_hash is None:
                raise ValueError("ready history attempt requires scope and ingestion plan record")
            if (
                any(
                    value is not None
                    for value in (
                        self.terminal_failure_attempt_id,
                        self.settlement_basis,
                        self.settlement_status_code,
                        self.settlement_task_id,
                        self.settlement_session_id,
                        self.internal_retry_count,
                        self.failure_kind,
                    )
                )
                or self.settlement_evidence_refs
            ):
                raise ValueError("ready history attempt cannot retain terminal failure fields")
        if self.status == "retryable_failed_settled":
            if (
                self.terminal_failure_attempt_id not in self.operation_attempt_ids
                or self.settlement_basis is None
                or not self.settlement_evidence_refs
                or self.settlement_status_code is None
                or self.internal_retry_count != 10
                or self.failure_kind is None
                or self.ingestion_plan_record_hash is not None
            ):
                raise ValueError(
                    "retryable failed history lacks configured extraction retry evidence"
                )
            expected_basis = {
                "hindsight": HINDSIGHT_SETTLEMENT_BASIS,
                "mem0": MEM0_SETTLEMENT_BASIS,
                "openviking": OPENVIKING_SETTLEMENT_BASIS,
            }.get(self.memory_system_id)
            if self.settlement_basis != expected_basis:
                raise ValueError("retryable failed history has the wrong provider settlement basis")
            task_session = (self.settlement_task_id, self.settlement_session_id)
            if self.memory_system_id == "openviking":
                if any(value is None for value in task_session):
                    raise ValueError(
                        "OpenViking settled history requires task and session identity"
                    )
            elif any(value is not None for value in task_session):
                raise ValueError(
                    "non-OpenViking settled history cannot name task or session identity"
                )
        elif (
            any(
                value is not None
                for value in (
                    self.settlement_basis,
                    self.settlement_status_code,
                    self.settlement_task_id,
                    self.settlement_session_id,
                    self.internal_retry_count,
                    self.failure_kind,
                )
            )
            or self.settlement_evidence_refs
        ):
            raise ValueError("non-settled history cannot retain settlement fields")
        expected_id = history_attempt_id(
            self.model_dump(mode="python", exclude={"history_attempt_id"})
        )
        if self.history_attempt_id != expected_id:
            raise ValueError("history attempt identity does not match its fields")
        return self


def history_retry_event_id(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-history-retry-event-initial-v1", fields])


class HistoryRetryEvent(StrictContract):
    schema_name: Literal["history_retry_event"] = "history_retry_event"
    schema_version: Literal[1] = 1
    history_retry_event_id: Sha256
    run_id: NonEmptyStr
    ingestion_plan_id: Sha256
    failed_history_attempt_id: Sha256
    failed_history_attempt_ordinal: PositiveInt
    retry_ordinal: PositiveInt
    retry_scheduled: bool
    successor_ingestion_occurrence_id: Sha256 | None
    successor_execution_run_id: NonEmptyStr | None
    successor_history_attempt_ordinal: PositiveInt | None
    retry_policy_hash: Sha256
    max_retries_per_operation: NonNegativeInt
    backoff_seconds: PositiveInt | None
    observed_at: UtcDateTime

    @model_validator(mode="after")
    def retry_event_is_closed(self) -> Self:
        if self.max_retries_per_operation not in {0, 1, 2}:
            raise ValueError("history retry event limit must be 0, 1, or 2")
        if self.retry_ordinal != self.failed_history_attempt_ordinal:
            raise ValueError("history retry ordinal must equal the failed attempt ordinal")
        successor_values = (
            self.successor_ingestion_occurrence_id,
            self.successor_execution_run_id,
            self.successor_history_attempt_ordinal,
        )
        if self.retry_scheduled:
            if self.retry_ordinal > self.max_retries_per_operation:
                raise ValueError("scheduled history retry exceeds its allowance")
            if any(value is None for value in successor_values):
                raise ValueError("scheduled history retry requires successor identity")
            if self.successor_history_attempt_ordinal != self.failed_history_attempt_ordinal + 1:
                raise ValueError("scheduled history retry successor ordinal is invalid")
            expected_backoff = INFRASTRUCTURE_RETRY_BACKOFF_SECONDS[self.retry_ordinal - 1]
            if self.backoff_seconds != expected_backoff:
                raise ValueError("scheduled history retry backoff is invalid")
        elif (
            self.failed_history_attempt_ordinal != self.max_retries_per_operation + 1
            or any(value is not None for value in successor_values)
            or self.backoff_seconds is not None
        ):
            raise ValueError("exhausted history retry event must omit successor and wait")
        expected_id = history_retry_event_id(
            self.model_dump(mode="python", exclude={"history_retry_event_id"})
        )
        if self.history_retry_event_id != expected_id:
            raise ValueError("history retry event identity does not match its fields")
        return self


class HistoryRetryAllowance(StrictContract):
    schema_name: Literal["history_retry_allowance"] = "history_retry_allowance"
    schema_version: Literal[1] = 1
    ingestion_plan_id: Sha256
    next_history_attempt_ordinal: PositiveInt
    execution_run_id: NonEmptyStr
    previous_retry_event_id: Sha256 | None
    predecessor_history_attempt_id: Sha256 | None
    consumed_retries: NonNegativeInt

    @model_validator(mode="after")
    def allowance_is_closed(self) -> Self:
        if self.consumed_retries not in {0, 1, 2}:
            raise ValueError("history retry allowance consumed count must be 0, 1, or 2")
        if self.next_history_attempt_ordinal != self.consumed_retries + 1:
            raise ValueError("history retry allowance next ordinal does not match consumption")
        retry_references = (self.previous_retry_event_id, self.predecessor_history_attempt_id)
        if self.consumed_retries == 0 and any(value is not None for value in retry_references):
            raise ValueError("unused history retry allowance cannot name predecessor evidence")
        if self.consumed_retries > 0 and any(value is None for value in retry_references):
            raise ValueError("consumed history retry allowance requires predecessor evidence")
        return self


def history_retry_carry_id(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-history-retry-carry-initial-v1", fields])


class HistoryRetryCarryRecord(StrictContract):
    schema_name: Literal["history_retry_carry_record"] = "history_retry_carry_record"
    schema_version: Literal[1] = 1
    carry_record_id: Sha256
    run_id: NonEmptyStr
    source_part_bindings: tuple[CapsuleCompositionPartBinding, ...]
    allowances: tuple[HistoryRetryAllowance, ...]
    retry_policy_hash: Sha256
    max_retries_per_operation: NonNegativeInt

    @model_validator(mode="after")
    def carry_is_closed(self) -> Self:
        if self.max_retries_per_operation not in {0, 1, 2}:
            raise ValueError("history retry carry limit must be 0, 1, or 2")
        capsule_ids = tuple(item.capsule_id for item in self.source_part_bindings)
        plan_ids = tuple(item.ingestion_plan_id for item in self.allowances)
        if not capsule_ids or capsule_ids != tuple(sorted(capsule_ids)):
            raise ValueError("history retry carry source parts must be non-empty canonical order")
        if len(set(capsule_ids)) != len(capsule_ids):
            raise ValueError("history retry carry source parts must be unique")
        if (
            not plan_ids
            or plan_ids != tuple(sorted(plan_ids))
            or len(set(plan_ids)) != len(plan_ids)
        ):
            raise ValueError("history retry carry allowances must be unique canonical order")
        if any(item.consumed_retries > self.max_retries_per_operation for item in self.allowances):
            raise ValueError("history retry carry allowance exceeds policy")
        expected_id = history_retry_carry_id(
            self.model_dump(mode="python", exclude={"carry_record_id"})
        )
        if self.carry_record_id != expected_id:
            raise ValueError("history retry carry identity does not match its fields")
        return self


def capsule_composition_contribution_hash(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-capsule-composition-contribution-v1", fields])


class CapsuleCompositionContribution(StrictContract):
    schema_name: Literal["capsule_composition_contribution"] = "capsule_composition_contribution"
    schema_version: Literal[1] = 1
    contribution_hash: Sha256
    source_capsule_id: Sha256
    source_run_id: NonEmptyStr
    ingestion_plan_id: Sha256
    ingestion_occurrence_id: Sha256
    case_manifest_entry_ids: tuple[Sha256, ...]
    case_occurrence_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def contribution_is_closed(self) -> Self:
        if (
            not self.case_manifest_entry_ids
            or len(self.case_manifest_entry_ids) != len(self.case_occurrence_ids)
            or len(set(self.case_manifest_entry_ids)) != len(self.case_manifest_entry_ids)
            or len(set(self.case_occurrence_ids)) != len(self.case_occurrence_ids)
        ):
            raise ValueError("composition contribution requires paired unique cases")
        expected = capsule_composition_contribution_hash(
            self.model_dump(mode="python", exclude={"contribution_hash"})
        )
        if self.contribution_hash != expected:
            raise ValueError("composition contribution hash does not match its fields")
        return self


def capsule_composition_id(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-capsule-composition-v1", fields])


class CapsuleCompositionRecord(StrictContract):
    schema_name: Literal["capsule_composition_record"] = "capsule_composition_record"
    schema_version: Literal[1] = 1
    composition_id: Sha256
    resolved_plan_hash: Sha256
    cell_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    target_case_manifest_hash: Sha256
    target_case_execution_bindings_hash: Sha256
    budget_policy_hash: Sha256
    retry_policy_hash: Sha256
    execution_configuration_hash: Sha256
    ordered_parts: tuple[CapsuleCompositionPartBinding, ...]
    ordered_contributions: tuple[CapsuleCompositionContribution, ...]
    exact_union_hash: Sha256

    @model_validator(mode="after")
    def composition_identity_is_canonical(self) -> Self:
        part_ids = tuple(part.capsule_id for part in self.ordered_parts)
        plan_ids = tuple(item.ingestion_plan_id for item in self.ordered_contributions)
        if (
            not part_ids
            or part_ids != tuple(sorted(part_ids))
            or len(set(part_ids)) != len(part_ids)
            or not plan_ids
            or len(set(plan_ids)) != len(plan_ids)
        ):
            raise ValueError("composition parts and contributions must be unique and canonical")
        expected_union_hash = canonical_sha256(
            ["oamb-capsule-composition-exact-union-v1", self.ordered_contributions]
        )
        if self.exact_union_hash != expected_union_hash:
            raise ValueError("composition exact-union hash does not match its contributions")
        expected_id = capsule_composition_id(
            self.model_dump(mode="python", exclude={"composition_id"})
        )
        if self.composition_id != expected_id:
            raise ValueError("capsule composition identity does not match its fields")
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


_SEALED_CONFORMANCE_RECORD_KINDS = frozenset(
    {
        "memory_conformance_spec",
        "budget_spec",
        "memory_conformance_occurrence_record",
        "occurrence_claim_record",
        "budget_reservation_record",
        "attempt_intent_record",
        "attempt_receipt_record",
        "attempt_record",
        "token_usage_record",
        "resource_usage_record",
        "cost_record",
        "projection_before",
        "projection_after",
    }
)


class MemoryConformanceEvidenceManifest(StrictContract):
    schema_name: Literal["memory_conformance_evidence_manifest"] = (
        "memory_conformance_evidence_manifest"
    )
    schema_version: Literal[1] = 1
    manifest_hash: Sha256
    conformance_spec_id: NonEmptyStr
    conformance_spec_hash: Sha256
    occurrence_id: NonEmptyStr
    terminal_occurrence_hash: Sha256
    terminal_state: MemoryConformanceOccurrenceState
    source_entries: tuple[CapsuleManifestEntry, ...]
    raw_entries: tuple[CapsuleManifestEntry, ...]
    attempt_root_hash: Sha256
    token_usage_root_hash: Sha256
    resource_usage_root_hash: Sha256
    cost_root_hash: Sha256
    created_at: UtcDateTime

    @model_validator(mode="after")
    def commit_marker_hash_inventory_and_paths_close(self) -> Self:
        expected_hash = memory_conformance_evidence_manifest_hash(
            self.model_dump(mode="python", exclude={"manifest_hash"})
        )
        if self.manifest_hash != expected_hash:
            raise ValueError("memory-conformance manifest hash does not match its fields")
        terminal_states = {
            MemoryConformanceOccurrenceState.SEALED,
            MemoryConformanceOccurrenceState.ERROR,
            MemoryConformanceOccurrenceState.CANCELLED,
            MemoryConformanceOccurrenceState.BUDGET_EXCEEDED,
            MemoryConformanceOccurrenceState.INTERRUPTED_UNKNOWN_OUTCOME,
        }
        if self.terminal_state not in terminal_states:
            raise ValueError("memory-conformance manifest requires a terminal state")
        if not self.source_entries:
            raise ValueError("memory-conformance manifest requires source entries")
        all_paths = tuple(
            entry.relative_path for entry in (*self.source_entries, *self.raw_entries)
        )
        if len(set(all_paths)) != len(all_paths):
            raise ValueError("memory-conformance manifest contains duplicate paths")
        if any(not entry.relative_path.startswith("source/") for entry in self.source_entries):
            raise ValueError("memory-conformance source entries must use source paths")
        if any(not entry.relative_path.startswith("raw/") for entry in self.raw_entries):
            raise ValueError("memory-conformance raw entries must use raw paths")
        if any(
            path == "memory-conformance-manifest.json" or path.startswith(".runtime/")
            for path in all_paths
        ):
            raise ValueError("memory-conformance manifest cannot index itself or runtime pointers")
        if self.terminal_state == MemoryConformanceOccurrenceState.SEALED:
            if not self.raw_entries:
                raise ValueError("sealed memory-conformance manifest requires raw entries")
            record_kinds = {entry.record_kind for entry in self.source_entries}
            missing_kinds = _SEALED_CONFORMANCE_RECORD_KINDS - record_kinds
            if missing_kinds:
                raise ValueError(
                    "sealed memory-conformance manifest lacks its complete source inventory"
                )
        return self


def memory_conformance_evidence_manifest_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "memory_conformance_evidence_manifest")
    payload.setdefault("schema_version", 1)
    payload.pop("manifest_hash", None)
    return canonical_sha256(payload)


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
    owner_kind: Literal["run", "model_readiness", "worker"]
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
