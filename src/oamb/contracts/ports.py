"""Dependency-free behavior ports composed by the future runtime."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, runtime_checkable

from .evidence import CaseRecord, IngestionPlanRecord, LogicalContextRecord
from .ids import canonical_sha256
from .specifications import CaseManifest, DatasetManifest


@dataclass(frozen=True, slots=True)
class RawReferenceHandle:
    sha256: str


@dataclass(frozen=True, slots=True)
class SourceUnit:
    source_unit_id: str
    context_manifest_entry_id: str
    ordinal_1_indexed: int
    payload_sha256: str
    payload_bytes: bytes
    source_reference: str | None = None
    occurred_at: str | None = None
    context_text: str | None = None
    source_metadata: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class IngestionPlan:
    ingestion_plan_id: str
    ordered_member_context_manifest_entry_ids: tuple[str, ...]
    shared_context_sha256: str
    intended_source_count: int
    ordered_source_units: tuple[SourceUnit, ...]
    ordered_case_manifest_entry_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CasePlan:
    case_manifest_entry_id: str
    context_manifest_entry_id: str
    source_question_number_1_indexed: int
    question_bytes: bytes
    reference_payload: bytes
    reference_payload_sha256: str
    prompt_binding_id: str
    output_contract_id: str
    metric_id: str
    judge_binding_id: str | None
    answer_max_output_tokens: int = 1
    query_timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class NativeEvidenceCandidate:
    native_id: str
    native_rank_1_indexed: int
    content: str
    native_score: str | None
    provider_evidence_identity: str | None = None
    source_unit_id: str | None = None
    evidence_kind: str = "native"
    occurred_start: str | None = None
    occurred_end: str | None = None
    mentioned_at: str | None = None
    native_reference: str | None = None
    native_truncated: bool = False


@dataclass(frozen=True, slots=True)
class NativeEvidenceBatch:
    raw_reference: RawReferenceHandle
    candidates: tuple[NativeEvidenceCandidate, ...]


@dataclass(frozen=True, slots=True)
class VisibleEvidencePolicy:
    max_items: int
    max_characters: int
    max_tokens: int


@dataclass(frozen=True, slots=True)
class VisibleEvidence:
    canonical_bytes: bytes
    sha256: str
    included_native_ids: tuple[str, ...]
    token_count: int = 0
    candidate_count: int = 0
    kept_count: int = 0
    dropped_count: int = 0
    truncated_count: int = 0
    first_exceeded_limit: str | None = None
    decisions: tuple[EvidenceDecision, ...] = ()
    payload_byte_count: int = 0
    character_count: int = 0
    tokenizer_fingerprint: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceDecision:
    native_id: str
    provider_evidence_identity: str
    normalized_text_sha256: str
    disposition: Literal["kept", "duplicate", "budget_dropped", "truncated"]
    reason: str | None


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    canonical_bytes: bytes
    sha256: str
    prompt_pack_id: str | None = None
    template_name: str | None = None
    variable_names: tuple[str, ...] = ()
    variable_values: tuple[str, ...] = ()
    variable_value_sha256: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AnswerValue:
    raw_reference: RawReferenceHandle
    raw_answer: bytes
    parsed_value: bytes
    parsed_value_sha256: str


@dataclass(frozen=True, slots=True)
class DeterministicEvaluation:
    metric_id: str
    result_sha256: str
    numerator: int | None = None
    denominator: int | None = None
    trace_bytes: bytes | None = None


@dataclass(frozen=True, slots=True)
class JudgeRequest:
    prompt: RenderedPrompt
    output_contract_id: str


@dataclass(frozen=True, slots=True)
class WorkloadRecordSet:
    logical_context_records: tuple[LogicalContextRecord, ...]
    ingestion_plan_records: tuple[IngestionPlanRecord, ...]
    case_records: tuple[CaseRecord, ...]


@dataclass(frozen=True, slots=True)
class WorkloadRuleResult:
    rule_id: str
    passed: bool
    evidence_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeResolution:
    memory_system_id: str
    runtime_binding_hash: str
    raw_reference: RawReferenceHandle


@dataclass(frozen=True, slots=True)
class CapabilitySet:
    capability_ids: tuple[str, ...]
    provider_order_preserved: bool
    native_reranking_disabled: bool


@dataclass(frozen=True, slots=True)
class ScopeAllocationRequest:
    ingestion_occurrence_id: str
    ingestion_plan_id: str


@dataclass(frozen=True, slots=True)
class ScopeReceipt:
    ingestion_occurrence_id: str
    scope_id: str
    raw_reference: RawReferenceHandle


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    scope: ScopeReceipt
    ordered_source_units: tuple[SourceUnit, ...]


@dataclass(frozen=True, slots=True)
class IngestionReceipt:
    ingestion_occurrence_id: str
    accepted_source_unit_ids: tuple[str, ...]
    rejected_source_unit_ids: tuple[str, ...]
    raw_references: tuple[RawReferenceHandle, ...]


@dataclass(frozen=True, slots=True)
class ReadinessRequest:
    scope: ScopeReceipt
    expected_source_unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReadinessReceipt:
    ingestion_occurrence_id: str
    ready: bool
    evidence_references: tuple[RawReferenceHandle, ...]


@dataclass(frozen=True, slots=True)
class InventoryReceipt:
    ingestion_occurrence_id: str
    ordered_source_unit_ids: tuple[str, ...]
    raw_reference: RawReferenceHandle


@dataclass(frozen=True, slots=True)
class StateDigestReceipt:
    ingestion_occurrence_id: str
    state_sha256: str
    raw_reference: RawReferenceHandle


@dataclass(frozen=True, slots=True)
class ProjectionReceipt:
    inventory: InventoryReceipt
    state_digest: StateDigestReceipt


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    scope: ScopeReceipt
    case_occurrence_id: str
    query_bytes: bytes
    top_k: int
    query_timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class ModelRequest:
    attempt_id: str
    parent_kind: Literal["ingestion_plan", "case", "phase_review", "model_readiness"]
    parent_id: str
    stage: str
    role_binding_id: str
    messages_sha256: str
    messages: tuple[tuple[str, str], ...]
    output_contract_id: str = "fake-output-v1"
    max_output_tokens: int = 1
    candidate_count: int = 1
    temperature: str = "0"
    top_p: str = "1"
    reasoning_disabled: bool = True
    stop: tuple[str, ...] | None = None

    @property
    def request_fingerprint(self) -> str:
        return canonical_sha256(
            [
                "oamb-model-request-v1",
                self.parent_kind,
                self.parent_id,
                self.stage,
                self.role_binding_id,
                self.messages_sha256,
                self.messages,
                self.output_contract_id,
                self.max_output_tokens,
                self.candidate_count,
                self.temperature,
                self.top_p,
                self.reasoning_disabled,
                self.stop,
            ]
        )


class FinishDisposition(StrEnum):
    NORMAL_STOP = "normal_stop"
    LENGTH_LIMIT = "length_limit"
    CONTENT_FILTERED = "content_filtered"
    TOOL_CALL = "tool_call"
    MISSING_FINISH_REASON = "missing_finish_reason"
    CANCELLED = "cancelled"
    INCOMPLETE_STREAM = "incomplete_stream"
    TRANSPORT_ERROR = "transport_error"


@dataclass(frozen=True, slots=True)
class ModelCandidate:
    content: str | None
    tool_call_present: bool
    complete: bool
    reasoning_content: str | None = None


@dataclass(frozen=True, slots=True)
class ModelCompletion:
    finish_disposition: FinishDisposition
    candidates: tuple[ModelCandidate, ...]


@dataclass(frozen=True, slots=True)
class ModelReceipt:
    raw_reference: RawReferenceHandle
    output_text: str
    usage_reference_ids: tuple[str, ...]
    raw_response_bytes: bytes = b""
    finish_disposition: FinishDisposition = FinishDisposition.NORMAL_STOP
    candidates: tuple[ModelCandidate, ...] = ()
    runtime_model: str | None = None
    runtime_identity_status: Literal["matched", "recorded", "unattested", "mismatch"] = "unattested"
    supplier_status_code: int | None = None


class ModelCallFailure(RuntimeError):
    """A model call with sealed error/usage evidence and explicit retryability."""

    def __init__(
        self,
        message: str,
        *,
        raw_reference: RawReferenceHandle,
        usage_reference_ids: tuple[str, ...],
        retryable: bool,
        failure_kind: str = "supplier_error",
        supplier_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_reference = raw_reference
        self.usage_reference_ids = usage_reference_ids
        self.retryable = retryable
        self.failure_kind = failure_kind
        self.supplier_status_code = supplier_status_code


class ModelCallUnknownOutcome(RuntimeError):
    """A dispatched call without a trustworthy supplier receipt."""

    def __init__(self, message: str, *, failure_kind: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class ModelCallCancelledUnknownOutcome(asyncio.CancelledError, ModelCallUnknownOutcome):
    """A cancelled in-flight call that remains an unknown supplier outcome."""

    def __init__(self, message: str) -> None:
        ModelCallUnknownOutcome.__init__(self, message, failure_kind="cancelled")


@dataclass(frozen=True, slots=True)
class AttemptSealRequest:
    attempt_id: str
    canonical_sha256: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class RawPayloadSealRequest:
    sha256: str
    media_type: str
    compression: Literal["none", "gzip"]
    payload_bytes: bytes


@dataclass(frozen=True, slots=True)
class ArtifactWriteRequest:
    record_id: str
    relative_path: str
    canonical_sha256: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class ArtifactReadRequest:
    relative_path: str
    expected_sha256: str


@dataclass(frozen=True, slots=True)
class ArtifactSealReceipt:
    record_id: str
    canonical_sha256: str
    created: bool = True


class WorkerOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNKNOWN_OUTCOME = "unknown_outcome"


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    request_id: str
    operation: str
    payload_sha256: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class WorkerReceipt:
    request_id: str
    outcome: WorkerOutcome
    raw_reference: RawReferenceHandle | None
    canonical_bytes: bytes


@runtime_checkable
class WorkloadPort(Protocol):
    def resolve_sources(self) -> DatasetManifest: ...

    def build_case_manifest(self, dataset_manifest: DatasetManifest) -> CaseManifest: ...

    def iter_ingestion_plans(self, case_manifest: CaseManifest) -> tuple[IngestionPlan, ...]: ...

    def iter_case_plans(self, case_manifest: CaseManifest) -> tuple[CasePlan, ...]: ...

    def render_retrieval_query(self, case_plan: CasePlan) -> bytes: ...

    def build_visible_evidence(
        self,
        native_batch: NativeEvidenceBatch,
        policy: VisibleEvidencePolicy,
    ) -> VisibleEvidence: ...

    def render_answer(
        self,
        case_plan: CasePlan,
        visible_evidence: VisibleEvidence,
    ) -> RenderedPrompt: ...

    def evaluate(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
    ) -> DeterministicEvaluation | JudgeRequest: ...

    def validate_records(self, records: WorkloadRecordSet) -> tuple[WorkloadRuleResult, ...]: ...


@runtime_checkable
class MemorySystemPort(Protocol):
    async def resolve(self) -> RuntimeResolution: ...

    async def capabilities(self) -> CapabilitySet: ...

    async def allocate_ingestion_scope(self, request: ScopeAllocationRequest) -> ScopeReceipt: ...

    async def ingest(self, request: IngestionRequest) -> IngestionReceipt: ...

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt: ...

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt: ...

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt: ...

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt: ...

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch: ...

    async def close(self) -> None: ...


@runtime_checkable
class ModelClientPort(Protocol):
    async def complete(self, request: ModelRequest) -> ModelReceipt: ...

    async def close(self) -> None: ...


@runtime_checkable
class AttemptStorePort(Protocol):
    def seal_intent(self, request: AttemptSealRequest) -> ArtifactSealReceipt: ...

    def seal_receipt(self, request: AttemptSealRequest) -> ArtifactSealReceipt: ...


@runtime_checkable
class ArtifactStorePort(Protocol):
    def seal_raw(self, request: RawPayloadSealRequest) -> RawReferenceHandle: ...

    def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt: ...

    def seal_checkpoint(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt: ...

    def seal_source_manifest(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt: ...

    def read_verified(self, request: ArtifactReadRequest) -> bytes: ...


@runtime_checkable
class WorkerIPCPort(Protocol):
    def request(self, request: WorkerRequest) -> WorkerReceipt: ...

    def close(self) -> None: ...
