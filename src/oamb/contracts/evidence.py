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
from .specifications import BudgetScopeKindV2, ResourceBudgetCeiling
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
