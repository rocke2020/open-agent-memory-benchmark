"""Dependency-free behavior ports composed by the future runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from .evidence import CaseRecord, IngestionPlanRecord, LogicalContextRecord
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


@dataclass(frozen=True, slots=True)
class NativeEvidenceCandidate:
    native_id: str
    native_rank_1_indexed: int
    content: str
    native_score: str | None


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


@dataclass(frozen=True, slots=True)
class RenderedPrompt:
    canonical_bytes: bytes
    sha256: str


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


@dataclass(frozen=True, slots=True)
class ModelRequest:
    role_binding_id: str
    messages_sha256: str
    messages: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class ModelReceipt:
    raw_reference: RawReferenceHandle
    output_text: str
    usage_reference_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AttemptSealRequest:
    attempt_id: str
    canonical_sha256: str
    canonical_bytes: bytes


@dataclass(frozen=True, slots=True)
class ArtifactSealReceipt:
    record_id: str
    canonical_sha256: str


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
class WorkerIPCPort(Protocol):
    def request(self, request: WorkerRequest) -> WorkerReceipt: ...

    def close(self) -> None: ...
