"""Dependency-free behavior ports composed by the future runtime."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from .accounting import TokenUsageRecordV3
from .evidence import CaseRecord, IngestionPlanRecord, LogicalContextRecord
from .ids import attempt_id, canonical_sha256
from .ingestion_failures import classify_settled_ingestion_failure
from .specifications import CaseManifest, DatasetManifest, GenerativeThinkingEffort

ThinkingEffort = GenerativeThinkingEffort
_SUPPORTED_THINKING_EFFORTS: Final[frozenset[str]] = frozenset({"low", "high", "max"})


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
    supporting_raw_references: tuple[RawReferenceHandle, ...] = ()
    request_raw_reference: RawReferenceHandle | None = None


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
    max_output_tokens: int


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
    supporting_raw_references: tuple[RawReferenceHandle, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestionRequest:
    scope: ScopeReceipt
    ordered_source_units: tuple[SourceUnit, ...]


@dataclass(frozen=True, slots=True)
class IngestionDispatch:
    dispatch_ordinal_1_indexed: int
    operation_kind: str
    request_fingerprint: str
    ordered_source_units: tuple[SourceUnit, ...]


@dataclass(frozen=True, slots=True)
class IngestionDispatchRequest:
    scope: ScopeReceipt
    attempt_id: str
    dispatch: IngestionDispatch
    batch_attempt_ordinal: int = 1

    def __post_init__(self) -> None:
        if type(self.batch_attempt_ordinal) is not int or self.batch_attempt_ordinal not in {
            1,
            2,
            3,
        }:
            raise ValueError("ingestion batch attempt ordinal must be 1, 2, or 3")


@dataclass(frozen=True, slots=True)
class IngestionDispatchReceipt:
    attempt_id: str
    dispatch: IngestionDispatch
    accepted_source_unit_ids: tuple[str, ...]
    rejected_source_unit_ids: tuple[str, ...]
    raw_reference: RawReferenceHandle
    raw_response_bytes: bytes
    usage_records: tuple[TokenUsageRecordV3, ...]
    skipped_source_unit_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IngestionReceipt:
    ingestion_occurrence_id: str
    accepted_source_unit_ids: tuple[str, ...]
    rejected_source_unit_ids: tuple[str, ...]
    raw_references: tuple[RawReferenceHandle, ...]
    dispatch_receipts: tuple[IngestionDispatchReceipt, ...]
    skipped_source_unit_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReadinessRequest:
    scope: ScopeReceipt
    expected_source_unit_ids: tuple[str, ...]
    ingestion_receipt: IngestionReceipt


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
    supporting_raw_references: tuple[RawReferenceHandle, ...] = ()


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
    parent_kind: Literal["ingestion_plan", "case", "model_readiness"]
    parent_id: str
    stage: str
    role_binding_id: str
    messages_sha256: str
    messages: tuple[tuple[str, str], ...]
    thinking_effort: ThinkingEffort
    output_contract_id: str = "fake-output-v1"
    max_output_tokens: int | None = 1
    candidate_count: int = 1
    temperature: str = "0"
    top_p: str = "1"
    stop: tuple[str, ...] | None = None
    supplier_call_ordinal: int = 1

    def __post_init__(self) -> None:
        if self.thinking_effort not in _SUPPORTED_THINKING_EFFORTS:
            raise ValueError("model request thinking effort must be low, high, or max")
        if self.supplier_call_ordinal < 1:
            raise ValueError("model request supplier call ordinal must be positive")

    @classmethod
    def for_attempt(
        cls,
        *,
        ordinal: int,
        parent_kind: Literal["ingestion_plan", "case", "model_readiness"],
        parent_id: str,
        stage: str,
        role_binding_id: str,
        messages_sha256: str,
        messages: tuple[tuple[str, str], ...],
        thinking_effort: ThinkingEffort,
        output_contract_id: str = "fake-output-v1",
        max_output_tokens: int | None = 1,
        candidate_count: int = 1,
        temperature: str = "0",
        top_p: str = "1",
        stop: tuple[str, ...] | None = None,
    ) -> ModelRequest:
        if ordinal < 1:
            raise ValueError("model request attempt ordinal must be positive")
        unbound = cls(
            attempt_id="0" * 64,
            parent_kind=parent_kind,
            parent_id=parent_id,
            stage=stage,
            role_binding_id=role_binding_id,
            messages_sha256=messages_sha256,
            messages=messages,
            thinking_effort=thinking_effort,
            output_contract_id=output_contract_id,
            max_output_tokens=max_output_tokens,
            candidate_count=candidate_count,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
        )
        return replace(
            unbound,
            attempt_id=attempt_id(parent_id, stage, ordinal, unbound.request_fingerprint),
        )

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
                self.thinking_effort,
                self.output_contract_id,
                self.max_output_tokens,
                self.candidate_count,
                self.temperature,
                self.top_p,
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
    model: str
    raw_response_bytes: bytes = b""
    finish_disposition: FinishDisposition = FinishDisposition.NORMAL_STOP
    candidates: tuple[ModelCandidate, ...] = ()
    supplier_status_code: int | None = None


class ModelCallFailure(RuntimeError):
    """A model call with sealed error/usage evidence and explicit retryability."""

    def __init__(
        self,
        message: str,
        *,
        raw_reference: RawReferenceHandle,
        raw_response_bytes: bytes | None = None,
        usage_reference_ids: tuple[str, ...],
        retryable: bool,
        failure_kind: str = "supplier_error",
        supplier_status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_reference = raw_reference
        self.raw_response_bytes = raw_response_bytes
        self.usage_reference_ids = usage_reference_ids
        self.retryable = retryable
        self.failure_kind = failure_kind
        self.supplier_status_code = supplier_status_code


@dataclass(frozen=True, slots=True)
class ModelSupplierRejectionClassification:
    origin: Literal["model_supplier"]
    failure_kind: Literal["rate_limited"]
    status: Literal[429]
    acceptance: Literal["not_accepted"]
    provider_mutation: Literal["none"]
    retryable: Literal[True]
    internal_retry_count: int

    def __post_init__(self) -> None:
        if self.internal_retry_count < 0:
            raise ValueError("supplier internal retry count must be non-negative")


class ModelSupplierRateLimitRejection(ModelCallFailure):
    """Pinned structured proof of a retry-safe model-supplier 429 rejection."""

    def __init__(
        self,
        message: str,
        *,
        classification: ModelSupplierRejectionClassification,
        raw_reference: RawReferenceHandle,
        raw_response_bytes: bytes,
        usage_reference_ids: tuple[str, ...],
    ) -> None:
        super().__init__(
            message,
            raw_reference=raw_reference,
            raw_response_bytes=raw_response_bytes,
            usage_reference_ids=usage_reference_ids,
            retryable=True,
            failure_kind="rate_limited",
            supplier_status_code=429,
        )
        self.classification = classification


class ModelCallUnknownOutcome(RuntimeError):
    """A dispatched call without a trustworthy supplier receipt."""

    def __init__(self, message: str, *, failure_kind: str) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class ModelCallCancelledUnknownOutcome(asyncio.CancelledError, ModelCallUnknownOutcome):
    """A cancelled in-flight call that remains an unknown supplier outcome."""

    def __init__(self, message: str) -> None:
        ModelCallUnknownOutcome.__init__(self, message, failure_kind="cancelled")


class MemorySystemCallFailure(RuntimeError):
    """A memory-system call with a known non-success outcome."""

    def __init__(
        self,
        message: str,
        *,
        failure_kind: str,
        raw_reference: RawReferenceHandle | None = None,
        raw_response_bytes: bytes | None = None,
        supporting_raw_references: tuple[RawReferenceHandle, ...] = (),
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.failure_kind = failure_kind
        self.raw_reference = raw_reference
        self.raw_response_bytes = raw_response_bytes
        self.supporting_raw_references = supporting_raw_references
        self.status_code = status_code


class SettledTransientIngestionFailure(MemorySystemCallFailure):
    """A pinned ingestion failure whose related provider work is proven settled."""

    def __init__(
        self,
        message: str,
        *,
        settlement_basis: str,
        internal_retry_count: int,
        failure_kind: str,
        raw_reference: RawReferenceHandle,
        raw_response_bytes: bytes,
        status_code: int,
        expected_task_id: str | None = None,
        expected_session_id: str | None = None,
        supporting_raw_references: tuple[RawReferenceHandle, ...] = (),
    ) -> None:
        if raw_reference.sha256 != hashlib.sha256(raw_response_bytes).hexdigest():
            raise ValueError("settled ingestion failure raw reference does not match response")
        classified = classify_settled_ingestion_failure(
            settlement_basis=settlement_basis,
            status_code=status_code,
            raw_response_bytes=raw_response_bytes,
            internal_retry_count=internal_retry_count,
            expected_task_id=expected_task_id,
            expected_session_id=expected_session_id,
        )
        if classified is None or classified != failure_kind:
            raise ValueError("ingestion failure is not an exact settled transient failure")
        super().__init__(
            message,
            failure_kind=failure_kind,
            raw_reference=raw_reference,
            raw_response_bytes=raw_response_bytes,
            supporting_raw_references=supporting_raw_references,
            status_code=status_code,
        )
        self.settlement_basis = settlement_basis
        self.internal_retry_count = internal_retry_count
        self.expected_task_id = expected_task_id
        self.expected_session_id = expected_session_id


class MemorySystemCallUnknownOutcome(RuntimeError):
    """A dispatched memory write without a trustworthy provider receipt."""

    def __init__(self, message: str, *, failure_kind: str = "unknown_outcome") -> None:
        super().__init__(message)
        self.failure_kind = failure_kind


class MemorySystemCallCancelledUnknownOutcome(
    asyncio.CancelledError,
    MemorySystemCallUnknownOutcome,
):
    """A cancelled memory write that retains asyncio cancellation semantics."""

    def __init__(self, message: str) -> None:
        MemorySystemCallUnknownOutcome.__init__(self, message, failure_kind="cancelled")


class MemorySystemCallCancelledBeforeDispatch(asyncio.CancelledError):
    """A cancelled call proven not to have entered the provider dispatch boundary."""

    def __init__(
        self,
        message: str,
        *,
        supporting_raw_references: tuple[RawReferenceHandle, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failure_kind = "cancelled_before_dispatch"
        self.raw_reference: RawReferenceHandle | None = None
        self.raw_response_bytes: bytes | None = None
        self.supporting_raw_references = supporting_raw_references
        self.status_code: int | None = None


class MemorySystemReadCancelled(asyncio.CancelledError):
    """A cancelled read with any raw responses sealed earlier in the composite call."""

    def __init__(
        self,
        message: str,
        *,
        supporting_raw_references: tuple[RawReferenceHandle, ...] = (),
    ) -> None:
        super().__init__(message)
        self.failure_kind = "cancelled"
        self.raw_reference: RawReferenceHandle | None = None
        self.raw_response_bytes: bytes | None = None
        self.supporting_raw_references = supporting_raw_references
        self.status_code: int | None = None


class MemorySystemProfileUnsupported(RuntimeError):
    """An exact adapter profile that cannot satisfy its declared contract."""

    def __init__(self, profile_id: str, *, reason_codes: tuple[str, ...]) -> None:
        if not profile_id:
            raise ValueError("unsupported profile requires an identity")
        if not reason_codes or any(not reason for reason in reason_codes):
            raise ValueError("unsupported profile requires reason codes")
        if len(set(reason_codes)) != len(reason_codes):
            raise ValueError("unsupported profile reason codes must be unique")
        super().__init__(f"memory-system profile is unsupported: {profile_id}")
        self.profile_id = profile_id
        self.reason_codes = reason_codes


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

    def finalize_judge(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
        judge_answer: AnswerValue,
    ) -> DeterministicEvaluation: ...

    def validate_records(self, records: WorkloadRecordSet) -> tuple[WorkloadRuleResult, ...]: ...


@runtime_checkable
class MemorySystemPort(Protocol):
    async def resolve(self) -> RuntimeResolution: ...

    async def capabilities(self) -> CapabilitySet: ...

    async def allocate_ingestion_scope(self, request: ScopeAllocationRequest) -> ScopeReceipt: ...

    def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]: ...

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt: ...

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt: ...

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt: ...

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt: ...

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt: ...

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch: ...

    async def close(self) -> None: ...


@runtime_checkable
class ModelClientPort(Protocol):
    def thinking_effort_for(self, *, stage: str, role_binding_id: str) -> ThinkingEffort: ...

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
