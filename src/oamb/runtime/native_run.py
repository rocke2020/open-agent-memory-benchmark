"""Fixture-safe native capsule composition through dependency-free ports."""

from __future__ import annotations

import asyncio
import hashlib
import math
import multiprocessing
import os
import signal
import sys
import threading
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, TypeVar

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
from oamb.contracts.accounting import (
    CostBasis,
    CostRecord,
    CostRecordV2,
    IndexingView,
    ProofStatus,
    ResourceUsageRecord,
    ResourceUsageRecordV2,
    TokenDomain,
    TokenMeasurementSource,
    TokenStage,
    TokenStageV2,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
    TokenUsageRecordV5,
)
from oamb.contracts.base import StrictContract
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptIntentRecordV3,
    AttemptReceiptKind,
    AttemptReceiptRecord,
    AttemptRecordV2,
    AttemptRecordV4,
    BudgetReservationRecord,
    BudgetReservationRecordV3,
    CapsuleManifest,
    CaseEvaluationDisposition,
    CaseRecordV3,
    CloseErrorRecord,
    HistoryAttemptRecord,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
    OccurrenceClaimRecord,
    RunLeaseRecord,
    RunRecord,
    attempt_intent_v3_hash,
    attempt_record_v4_hash,
    budget_owner_allocation_hash,
    budget_reservation_v3_hash,
    budget_reservation_v3_id,
    history_attempt_id,
)
from oamb.contracts.evidence import (
    BudgetOwnerAllocation as EvidenceBudgetOwnerAllocation,
)
from oamb.contracts.ids import (
    attempt_id,
    canonical_json_bytes,
    canonical_sha256,
    case_occurrence_id,
    ingestion_occurrence_id,
    ingestion_payload_hash,
)
from oamb.contracts.ports import (
    AnswerValue,
    ArtifactReadRequest,
    ArtifactSealReceipt,
    ArtifactStorePort,
    ArtifactWriteRequest,
    CasePlan,
    DeterministicEvaluation,
    FinishDisposition,
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionPlan,
    IngestionReceipt,
    IngestionRequest,
    JudgeRequest,
    MemorySystemCallCancelledBeforeDispatch,
    MemorySystemCallCancelledUnknownOutcome,
    MemorySystemCallUnknownOutcome,
    MemorySystemPort,
    ModelCallCancelledUnknownOutcome,
    ModelCallFailure,
    ModelCallUnknownOutcome,
    ModelClientPort,
    ModelReceipt,
    ModelRequest,
    NativeEvidenceBatch,
    ProjectionReceipt,
    RawPayloadSealRequest,
    RawReferenceHandle,
    ReadinessReceipt,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
    ScopeReceipt,
    SettledTransientIngestionFailure,
    VisibleEvidencePolicy,
    WorkloadPort,
)
from oamb.contracts.reporting import (
    RunComparisonControlBasisRecord,
    RuntimeMeasurementControlRecord,
    WorkloadExecutionControlRecord,
    provider_native_profile_hash,
)
from oamb.contracts.specifications import (
    INFRASTRUCTURE_RETRY_BACKOFF_SECONDS,
    INFRASTRUCTURE_RETRY_POLICY_HASH,
    BudgetScopeKindV2,
    BudgetScopeKindV3,
    BudgetSpecV4,
    DispatchBudgetOwnerKind,
    DispatchBudgetRoute,
    MemorySystemRuntimeBindingV2,
    ModelRoleBindingV2,
    RunPreflightRecord,
    RunPreflightRecordV2,
    RunSpec,
)
from oamb.contracts.states import (
    AttemptOutcome,
    CaseState,
    IndexContribution,
    IngestionPlanState,
    ResumeDisposition,
    RunState,
)
from oamb.runtime.budget import (
    BudgetAmount,
    BudgetCeiling,
    BudgetExceededError,
    BudgetLedger,
    ReservationRequest,
)
from oamb.runtime.budget import (
    BudgetOwnerAllocation as RuntimeBudgetOwnerAllocation,
)
from oamb.runtime.case_selection import select_case_execution
from oamb.runtime.infrastructure_retry import (
    InfrastructureBackoffCancelled,
    InfrastructureRetryExhausted,
)
from oamb.runtime.memory_query import execute_read_only_retrieval
from oamb.runtime.model_completion_retry import execute_model_completion_retry
from oamb.runtime.provider_lifecycle import (
    ProviderLifecycleBridge,
    ProviderLifecycleError,
)
from oamb.runtime.source_records import seal_source_contract

NATIVE_FIXTURE_STARTED_AT = datetime(2026, 1, 1, tzinfo=UTC)
NATIVE_RETRIEVAL_TOP_K = 100
NATIVE_CLOSE_TIMEOUT_SECONDS = 5.0
NATIVE_PROCESS_POLL_SECONDS = 0.01
NATIVE_PROCESS_TERMINATION_SECONDS = 0.10
_SUPERVISOR_EXITED_REASON = "native run stopped after its supervisor exited"
NATIVE_OWNER_NAME = "native-run-owner.json"
NATIVE_OWNER_ID = "native-fixture-owner-v1"
COOPERATIVE_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

NativeIngestionPlanRecord: TypeAlias = IngestionPlanRecordV2 | IngestionPlanRecordV3
TerminalCasePublisher: TypeAlias = Callable[
    [
        NativeIngestionPlanRecord,
        CaseRecordV3,
        Mapping[str, AttemptRecordV2 | AttemptRecordV4],
        tuple[HistoryAttemptRecord, ...],
    ],
    None,
]
_LiveDispatchResult = TypeVar("_LiveDispatchResult")


class NativeRunOwnershipError(RuntimeError):
    """A create-only run root already has an owner and cannot be replayed."""


class NativePortCloseTimeout(TimeoutError):
    """A supervised native client exceeded its hard close deadline."""


class NativePortCloseSkipped(RuntimeError):
    """A later client could not close after the native process was retired."""


class NativeRunProcessError(RuntimeError):
    """The supervised native composition process exited without a typed result."""


class NativeRunProcessTerminationError(RuntimeError):
    """The supervised native composition survived terminate and kill bounds."""


class NativeRunInterrupted(RuntimeError):
    """A planned process signal stopped the native run."""


class NativeCancellationEvidenceError(asyncio.CancelledError):
    """Preserves cancellation while carrying a separate evidence-seal failure."""

    def __init__(self, primary_error: BaseException, evidence_error: BaseException) -> None:
        super().__init__("native cancellation and failure-evidence seal both failed")
        self.errors = (primary_error, evidence_error)


class CapsuleArtifactStorePort(ArtifactStorePort, Protocol):
    def finalize_capsule(self, *, run_id: str, run_spec_hash: str) -> CapsuleManifest: ...


class _LiveModelUsageCaptureStore:
    """Keep transient model V2 usage out of the final source inventory until V5 conversion."""

    def __init__(
        self,
        delegate: ArtifactStorePort,
        pending_usage: dict[str, TokenUsageRecordV2 | TokenUsageRecordV3],
    ) -> None:
        self._delegate = delegate
        self._pending_usage = pending_usage

    def seal_raw(self, request: RawPayloadSealRequest) -> RawReferenceHandle:
        return self._delegate.seal_raw(request)

    def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        if request.relative_path.startswith("source/usage/"):
            try:
                usage: TokenUsageRecordV2 | TokenUsageRecordV3 = (
                    TokenUsageRecordV3.model_validate_json(request.canonical_bytes)
                )
            except ValueError:
                usage = TokenUsageRecordV2.model_validate_json(request.canonical_bytes)
            expected_path = f"source/usage/{usage.usage_record_id}.json"
            if request.record_id != usage.usage_record_id or request.relative_path != expected_path:
                raise ValueError("live model usage record ID or path does not match its bytes")
            if canonical_json_bytes(usage) != request.canonical_bytes:
                raise ValueError("live model usage record is not canonical")
            if hashlib.sha256(request.canonical_bytes).hexdigest() != request.canonical_sha256:
                raise ValueError("live model usage request hash does not match its bytes")
            previous = self._pending_usage.get(usage.usage_record_id)
            if previous is not None and previous != usage:
                raise ValueError("live model usage record collides with different bytes")
            self._pending_usage[usage.usage_record_id] = usage
            return ArtifactSealReceipt(
                usage.usage_record_id,
                request.canonical_sha256,
                created=previous is None,
            )
        return self._delegate.seal_source_record(request)

    def seal_checkpoint(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        return self._delegate.seal_checkpoint(request)

    def seal_source_manifest(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        return self._delegate.seal_source_manifest(request)

    def read_verified(self, request: ArtifactReadRequest) -> bytes:
        return self._delegate.read_verified(request)


@dataclass(frozen=True, slots=True)
class NativeRunArtifacts:
    capsule_root: Path
    manifest: CapsuleManifest
    ingestion_plan_records: tuple[NativeIngestionPlanRecord, ...]
    case_records: tuple[CaseRecordV3, ...]


@dataclass(frozen=True, slots=True)
class NativeRunControl:
    """Exact durable budget and identity closure for one live native run."""

    run_spec: RunSpec
    preflight_record: RunPreflightRecord | RunPreflightRecordV2
    budget: BudgetSpecV4
    role_bindings: tuple[ModelRoleBindingV2, ...]
    owner_id: str
    host_fingerprint: str
    process_id: int
    provider_runtime_directory: Path
    wall_clock: Callable[[], datetime]
    monotonic_clock: Callable[[], float]
    runtime_binding: MemorySystemRuntimeBindingV2 | None = None
    workload_control: WorkloadExecutionControlRecord | None = None
    runtime_measurement_control: RuntimeMeasurementControlRecord | None = None
    comparison_control_basis: RunComparisonControlBasisRecord | None = None
    max_parallel_history_ingestions: int = 1
    max_parallel_questions: int = 1
    max_retries_per_operation: int = len(INFRASTRUCTURE_RETRY_BACKOFF_SECONDS)
    extraction_max_retries: int = 10
    model_max_attempts: int = 6
    model_transport_max_retries: int = 2
    provider_lifecycle_coordination_directory: Path | None = None
    shutdown_owner_process_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        run_id = self.run_spec.run_id
        if (
            self.preflight_record.run_id != run_id
            or self.budget.scope_kind != BudgetScopeKindV3.RUN
            or self.budget.scope_id != run_id
        ):
            raise ValueError("live native control run and budget scopes do not close")
        if self.preflight_record.run_spec_hash != canonical_sha256(self.run_spec):
            raise ValueError("live native control preflight does not bind its run spec")
        if self.preflight_record.budget_hash != canonical_sha256(self.budget):
            raise ValueError("live native control preflight does not bind its budget")
        if self.preflight_record.dispatch_routes != self.budget.dispatch_routes:
            raise ValueError("live native control preflight does not bind its dispatch routes")
        if self.preflight_record.runtime_binding_hash != self.run_spec.runtime_binding_hash:
            raise ValueError("live native control runtime binding does not close")
        role_ids = tuple(item.binding_id for item in self.role_bindings)
        if (
            not role_ids
            or len(set(role_ids)) != len(role_ids)
            or role_ids != self.run_spec.model_role_binding_ids
            or role_ids != self.preflight_record.role_binding_ids
            or set(item.role_binding_id for item in self.budget.role_ceilings) != set(role_ids)
        ):
            raise ValueError("live native control role inventory does not close")
        if not self.owner_id or not self.host_fingerprint or self.process_id <= 0:
            raise ValueError("live native control requires concrete owner and host identity")
        if not self.provider_runtime_directory.is_absolute():
            raise ValueError("live native control provider runtime directory must be absolute")
        if (
            self.provider_lifecycle_coordination_directory is not None
            and not self.provider_lifecycle_coordination_directory.is_absolute()
        ):
            raise ValueError("live native lifecycle coordination directory must be absolute")
        if not callable(self.wall_clock) or not callable(self.monotonic_clock):
            raise ValueError("live native control requires trusted wall and monotonic clocks")
        if any(
            type(process_id) is not int or process_id <= 0
            for process_id in self.shutdown_owner_process_ids
        ) or len(set(self.shutdown_owner_process_ids)) != len(self.shutdown_owner_process_ids):
            raise ValueError("live native shutdown owners must be unique positive process IDs")
        if (
            min(
                self.max_parallel_history_ingestions,
                self.max_parallel_questions,
            )
            < 1
        ):
            raise ValueError("live native concurrency limits must be positive")
        if type(self.max_retries_per_operation) is not int or self.max_retries_per_operation != 2:
            raise ValueError("live native retry limit must equal 2")
        if (
            type(self.extraction_max_retries) is not int
            or self.extraction_max_retries != 10
            or type(self.model_max_attempts) is not int
            or self.model_max_attempts != 6
            or type(self.model_transport_max_retries) is not int
            or self.model_transport_max_retries != 2
        ):
            raise ValueError("live native layered retry controls differ from the frozen profile")
        comparison_records = (
            self.runtime_binding,
            self.workload_control,
            self.runtime_measurement_control,
            self.comparison_control_basis,
        )
        if isinstance(self.preflight_record, RunPreflightRecordV2):
            if any(record is None for record in comparison_records):
                raise ValueError(
                    "version 2 live control requires concrete comparison policy and basis records"
                )
            runtime_binding = self.runtime_binding
            workload_control = self.workload_control
            runtime_control = self.runtime_measurement_control
            basis = self.comparison_control_basis
            assert runtime_binding is not None
            assert workload_control is not None
            assert runtime_control is not None
            assert basis is not None
            expected_native_profile = provider_native_profile_hash(
                provider_project_id=runtime_binding.provider_project_id,
                provider_profile_id=runtime_binding.provider_profile_id,
                adapter_profile_hash=self.preflight_record.adapter_profile_hash,
                memory_system_id=runtime_binding.memory_system_id,
                release_version=runtime_binding.release_version,
                source_revision=runtime_binding.source_revision,
                artifact_sha256=runtime_binding.artifact_sha256,
                deployment_configuration_sha256=(runtime_binding.deployment_configuration_sha256),
                storage_engine=runtime_binding.storage_engine,
                storage_engine_version=runtime_binding.storage_engine_version,
                schema_revision=runtime_binding.schema_revision,
                vector_index_type=runtime_binding.vector_index_type,
                distance_metric=runtime_binding.distance_metric,
                index_configuration_sha256=runtime_binding.index_configuration_sha256,
                native_feature_flags_fingerprint=(runtime_binding.native_feature_flags_fingerprint),
                native_reranking_status=runtime_binding.native_reranking_status,
            )
            if (
                runtime_binding.runtime_binding_hash != self.run_spec.runtime_binding_hash
                or workload_control.run_id != run_id
                or workload_control.workload_id != self.run_spec.workload_id
                or workload_control.dataset_manifest_hash != self.run_spec.dataset_manifest_hash
                or workload_control.case_manifest_hash != self.run_spec.case_manifest_hash
                or runtime_control.run_id != run_id
                or runtime_control.runtime_binding_hash != self.run_spec.runtime_binding_hash
                or runtime_control.execution_environment.environment_hash
                != self.run_spec.environment_hash
                or basis.run_id != run_id
                or basis.run_spec_hash != canonical_sha256(self.run_spec)
                or basis.workload_control_hash != workload_control.workload_control_hash
                or basis.runtime_measurement_control_hash != runtime_control.runtime_control_hash
                or basis.runtime_binding_hash != runtime_binding.runtime_binding_hash
                or basis.provider_native_profile_hash != expected_native_profile
                or self.preflight_record.comparison_control_basis_hash != basis.basis_record_hash
            ):
                raise ValueError("version 2 live comparison control hash closure drifted")
        elif any(record is not None for record in comparison_records):
            raise ValueError("legacy live preflight cannot carry version 2 comparison controls")

    def require_budget_route(self, *, stage: str) -> DispatchBudgetRoute:
        routes = tuple(route for route in self.budget.dispatch_routes if route.stage == stage)
        if len(routes) != 1:
            raise ValueError(f"live native dispatch requires exactly one budget route for {stage}")
        return routes[0]


def seal_and_verify_live_comparison_controls(
    store: ArtifactStorePort,
    control: NativeRunControl,
    *,
    failpoint: Callable[[str], None] | None = None,
) -> None:
    """Durably seal and re-read the v2 comparison chain before client construction."""

    if not isinstance(control.preflight_record, RunPreflightRecordV2):
        raise ValueError("pre-dispatch comparison sealing requires RunPreflightRecord version 2")
    runtime_binding = control.runtime_binding
    workload_control = control.workload_control
    runtime_control = control.runtime_measurement_control
    basis = control.comparison_control_basis
    if any(
        record is None for record in (runtime_binding, workload_control, runtime_control, basis)
    ):
        raise ValueError("pre-dispatch comparison sealing requires the concrete control chain")
    assert runtime_binding is not None
    assert workload_control is not None
    assert runtime_control is not None
    assert basis is not None
    records: tuple[tuple[str, str, StrictContract], ...] = (
        (
            "source/specs/memory-system-runtime-binding.json",
            runtime_binding.runtime_binding_hash,
            runtime_binding,
        ),
        (
            "source/specs/workload-execution-control.json",
            workload_control.workload_control_id,
            workload_control,
        ),
        (
            "source/specs/runtime-measurement-control.json",
            runtime_control.runtime_control_id,
            runtime_control,
        ),
        (
            "source/specs/run-comparison-control-basis.json",
            basis.basis_record_id,
            basis,
        ),
        (
            "source/specs/run-preflight.json",
            control.preflight_record.preflight_record_hash,
            control.preflight_record,
        ),
    )
    for relative_path, record_id, record in records:
        seal_source_contract(
            store,
            relative_path=relative_path,
            record_id=record_id,
            record=record,
        )
        if failpoint is not None:
            failpoint(f"sealed:{record.schema_name}")  # type: ignore[attr-defined]
    for relative_path, _record_id, record in records:
        expected = canonical_json_bytes(record)
        observed = store.read_verified(
            ArtifactReadRequest(
                relative_path=relative_path,
                expected_sha256=hashlib.sha256(expected).hexdigest(),
            )
        )
        if observed != expected:
            raise ValueError("pre-dispatch comparison control canonical bytes changed")
        if failpoint is not None:
            failpoint(f"verified:{record.schema_name}")  # type: ignore[attr-defined]


_NativeErrorCategory: TypeAlias = Literal[
    "assertion",
    "base",
    "cancelled",
    "memory_cancelled_before_dispatch",
    "memory_cancelled_unknown",
    "memory_unknown",
    "model_cancelled_unknown",
    "model_unknown",
    "interrupted",
    "runtime",
    "system_exit",
    "timeout",
    "value",
]


@dataclass(frozen=True, slots=True)
class _NativeErrorEnvelope:
    category: _NativeErrorCategory
    type_name: str
    message: str
    failure_kind: str | None = None
    children: tuple[_NativeErrorEnvelope, ...] = ()


@dataclass(frozen=True, slots=True)
class _NativeCloseStarted:
    run_spec_hash: str
    ingestion_occurrence_ids: tuple[str, ...]
    case_occurrence_ids: tuple[str, ...]
    sequence: int
    clients: tuple[tuple[str, str], ...]
    client_index: int


@dataclass(frozen=True, slots=True)
class _NativeCloseFinished:
    client_index: int


@dataclass(frozen=True, slots=True)
class _NativeRunProgress:
    run_spec_hash: str
    ingestion_occurrence_ids: tuple[str, ...]
    case_occurrence_ids: tuple[str, ...]
    sequence: int


@dataclass(frozen=True, slots=True)
class _NativeRunReady:
    capsule_root: Path
    run_spec_hash: str
    ingestion_occurrence_ids: tuple[str, ...]
    case_occurrence_ids: tuple[str, ...]
    ingestion_plan_records: tuple[NativeIngestionPlanRecord, ...]
    case_records: tuple[CaseRecordV3, ...]
    sequence: int


@dataclass(frozen=True, slots=True)
class _NativeRunFailed:
    error: _NativeErrorEnvelope


_NativeProcessMessage: TypeAlias = (
    _NativeCloseStarted
    | _NativeCloseFinished
    | _NativeRunProgress
    | _NativeRunReady
    | _NativeRunFailed
)


@dataclass(frozen=True, slots=True)
class _NativeRunRequest:
    output_root: Path
    run_id: str
    adapter_profile_id: str
    workload: WorkloadPort
    visible_evidence_policy: VisibleEvidencePolicy
    artifact_store_factory: Callable[[Path], CapsuleArtifactStorePort]
    memory_factory: Callable[[ArtifactStorePort, tuple[IngestionPlan, ...]], MemorySystemPort]
    model_factory: Callable[[ArtifactStorePort], ModelClientPort]
    answer_role_binding_id: str
    judge_model_factory: Callable[[ArtifactStorePort], ModelClientPort] | None
    judge_role_binding_id: str | None
    close_timeout_seconds: float
    control: NativeRunControl | None
    lease: RunLeaseRecord | None = None
    provider_lifecycle: ProviderLifecycleBridge | None = None
    requested_case_manifest_entry_ids: tuple[str, ...] = ()
    terminal_case_publisher: TerminalCasePublisher | None = None
    stop_event: _NativeStopSignal | None = None
    interrupt_event: _NativeStopSignal | None = None
    supervisor_process_id: int | None = None


class _NativeStopSignal(Protocol):
    def is_set(self) -> bool: ...

    def set(self) -> None: ...


class _StoppableNativeProcess(Protocol):
    @property
    def exitcode(self) -> int | None: ...

    def start(self) -> None: ...

    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class _NativeExecutionState:
    store: CapsuleArtifactStorePort
    run_id: str
    lease_record_hash: str
    close_timeout_seconds: float
    lease_epoch: int = 1
    owner_id: str = NATIVE_OWNER_ID
    budget_id: str = "native-fixture-budget-v1"
    budget_scope_id: str | None = None
    started_at: datetime = NATIVE_FIXTURE_STARTED_AT
    live_timing: bool = False
    monotonic_started: float | None = None
    wall_clock: Callable[[], datetime] | None = None
    monotonic_clock: Callable[[], float] | None = None
    control: NativeRunControl | None = None
    budget_ledger: BudgetLedger | None = None
    provider_lifecycle: ProviderLifecycleBridge | None = None
    pending_model_usage: dict[str, TokenUsageRecordV2 | TokenUsageRecordV3] | None = None
    stop_event: _NativeStopSignal | None = None
    shutdown_owner_process_ids: tuple[int, ...] = ()
    supervisor_process_id: int | None = None
    sequence: int = 0
    operation_records: dict[str, AttemptRecordV2 | AttemptRecordV4] = field(default_factory=dict)
    attempt_accounting: dict[str, tuple[tuple[str, ...], str, str]] = field(default_factory=dict)
    history_records: list[HistoryAttemptRecord] = field(default_factory=list)
    history_scopes: dict[str, ScopeReceipt] = field(default_factory=dict)
    history_occurrence_bindings: dict[str, tuple[str, int]] = field(default_factory=dict)
    case_occurrence_bindings: dict[str, str] = field(default_factory=dict)
    publish_history_progress: Callable[[], None] | None = None

    def timestamp(self) -> datetime:
        self.sequence += 1
        deterministic_floor = self.started_at + timedelta(microseconds=self.sequence)
        if not self.live_timing:
            return deterministic_floor
        if self.wall_clock is None:
            raise RuntimeError("live native timing has no trusted wall clock")
        return max(self.wall_clock(), deterministic_floor)

    def monotonic(self) -> Decimal:
        if self.live_timing:
            if self.monotonic_started is None:
                raise RuntimeError("live native timing has no monotonic origin")
            if self.monotonic_clock is None:
                raise RuntimeError("live native timing has no trusted monotonic clock")
            return Decimal(str(self.monotonic_clock() - self.monotonic_started))
        self.sequence += 1
        return Decimal(self.sequence) / Decimal("1000000")


@dataclass(frozen=True, slots=True)
class _PreparedNativeAttempt:
    attempt_id: str
    parent_kind: Literal["ingestion_plan", "case"]
    parent_id: str
    stage: str
    ordinal: int
    request_fingerprint: str
    request_messages_sha256: str | None
    role_binding_id: str
    retry_of_attempt_id: str | None = None
    route: DispatchBudgetRoute | None = None
    reservation: BudgetReservationRecordV3 | None = None
    intent: AttemptIntentRecordV3 | None = None
    maximum: BudgetAmount | None = None
    owner_maximums: tuple[RuntimeBudgetOwnerAllocation, ...] = ()


@dataclass(frozen=True, slots=True)
class _LiveDispatchArtifacts:
    attempt_id: str
    usage_record_ids: tuple[str, ...]
    resource_record_id: str
    cost_record_id: str


async def _execute_live_memory_dispatch(
    state: _NativeExecutionState,
    *,
    parent_kind: Literal["ingestion_plan", "case"],
    parent_id: str,
    stage: str,
    ordinal: int,
    request_fingerprint: str,
    role_binding_id: str,
    call: Callable[[], Awaitable[_LiveDispatchResult]],
    raw_reference: Callable[[_LiveDispatchResult], str | None],
) -> tuple[_LiveDispatchResult, _LiveDispatchArtifacts]:
    prepared = _prepare_native_attempt(
        state,
        attempt_identity=attempt_id(parent_id, stage, ordinal, request_fingerprint),
        parent_kind=parent_kind,
        parent_id=parent_id,
        stage=stage,
        ordinal=ordinal,
        request_fingerprint=request_fingerprint,
        role_binding_id=role_binding_id,
    )
    started_at = state.timestamp()
    try:
        result = await call()
    except BaseException as exc:
        _seal_native_failure_preserving(
            state,
            prepared,
            started_at=started_at,
            error=exc,
        )
        raise
    ended_at = state.timestamp()
    raw_response_ref = raw_reference(result)
    if raw_response_ref is None:
        raw_response_ref = _seal_raw(
            state.store,
            canonical_json_bytes(
                {
                    "operation": stage,
                    "request_fingerprint": request_fingerprint,
                    "result": repr(result),
                }
            ),
            media_type="application/json",
        )
    _seal_native_success(
        state,
        prepared,
        started_at=started_at,
        ended_at=ended_at,
        raw_response_ref=raw_response_ref,
        index_contribution=IndexContribution.NOT_APPLICABLE,
    )
    usage_ids, resource_id, cost_id = _seal_native_attempt_accounting(
        state,
        prepared,
        started_at=started_at,
        ended_at=ended_at,
        raw_response_ref=raw_response_ref,
        usage_record_ids=(),
        indexing_view=IndexingView.NOT_APPLICABLE,
    )
    return result, _LiveDispatchArtifacts(
        attempt_id=prepared.attempt_id,
        usage_record_ids=usage_ids,
        resource_record_id=resource_id,
        cost_record_id=cost_id,
    )


def _scope_raw_reference(result: object) -> str | None:
    raw_reference = getattr(result, "raw_reference", None)
    return getattr(raw_reference, "sha256", None)


async def _infrastructure_retry_sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def _readiness_raw_reference(result: object) -> str | None:
    references = getattr(result, "evidence_references", ())
    return getattr(references[0], "sha256", None) if references else None


def _projection_raw_reference(result: object) -> str | None:
    inventory = getattr(result, "inventory", None)
    return getattr(getattr(inventory, "raw_reference", None), "sha256", None)


def _live_budget_ledger(control: NativeRunControl) -> BudgetLedger:
    budget = control.budget

    def resources(items: tuple[object, ...]) -> tuple[tuple[str, Decimal, str], ...]:
        return tuple(
            (item.dimension_id, item.maximum, item.unit)  # type: ignore[attr-defined]
            for item in items
        )

    role_ceilings = {
        item.role_binding_id: BudgetCeiling(
            maximum=BudgetAmount(
                attempts=item.max_attempts,
                input_tokens=item.max_input_tokens,
                output_tokens=item.max_output_tokens,
                wall_seconds=item.max_dispatch_wall_seconds,
                cost=item.max_cost or Decimal("0"),
                resources=resources(item.resource_ceilings),
                provider_units=(
                    (
                        item.provider_budget_cap.provider,
                        item.provider_budget_cap.operation_kind,
                        item.provider_budget_cap.billing_unit,
                        int(item.provider_budget_cap.maximum_accepted_units),
                    ),
                ),
            ),
            currency=item.currency,
        )
        for item in budget.role_ceilings
    }
    provider_ceilings = {
        item.provider_operation_ceiling_id: BudgetCeiling(
            maximum=BudgetAmount(
                attempts=item.max_attempts,
                wall_seconds=item.max_dispatch_wall_seconds,
                resources=resources(item.resource_ceilings),
                provider_units=(
                    (
                        item.adapter_profile_id,
                        item.operation_kind,
                        item.billing_unit,
                        int(item.maximum_accepted_units),
                    ),
                ),
            ),
            currency=budget.currency,
        )
        for item in budget.provider_operation_ceilings
    }
    provider_units: dict[tuple[str, str, str], int] = {}
    for ceiling in (*role_ceilings.values(), *provider_ceilings.values()):
        for provider, operation, unit, maximum in ceiling.maximum.provider_units:
            key = (provider, operation, unit)
            provider_units[key] = provider_units.get(key, 0) + maximum
    return BudgetLedger(
        BudgetCeiling(
            maximum=BudgetAmount(
                attempts=budget.max_attempts,
                input_tokens=budget.max_input_tokens,
                output_tokens=budget.max_output_tokens,
                wall_seconds=budget.max_dispatch_wall_seconds,
                cost=budget.max_cost or Decimal("0"),
                resources=resources(budget.resource_ceilings),
                provider_units=tuple((*key, maximum) for key, maximum in provider_units.items()),
            ),
            currency=budget.currency,
        ),
        role_ceilings=role_ceilings,
        provider_operation_ceilings=provider_ceilings,
    )


def _new_execution_state(
    request: _NativeRunRequest,
    store: CapsuleArtifactStorePort,
    *,
    sequence: int = 0,
) -> _NativeExecutionState:
    control = request.control
    lease = request.lease or _native_run_lease(
        request.run_id,
        request.adapter_profile_id,
        control=control,
    )
    return _NativeExecutionState(
        store=store,
        run_id=request.run_id,
        lease_record_hash=lease.lease_record_hash,
        lease_epoch=lease.lease_epoch,
        close_timeout_seconds=request.close_timeout_seconds,
        owner_id=control.owner_id if control is not None else NATIVE_OWNER_ID,
        budget_id=(control.budget.budget_id if control is not None else "native-fixture-budget-v1"),
        budget_scope_id=control.run_spec.run_id if control is not None else None,
        started_at=lease.acquired_at,
        live_timing=control is not None,
        monotonic_started=control.monotonic_clock() if control is not None else None,
        wall_clock=control.wall_clock if control is not None else None,
        monotonic_clock=control.monotonic_clock if control is not None else None,
        control=control,
        budget_ledger=_live_budget_ledger(control) if control is not None else None,
        provider_lifecycle=request.provider_lifecycle,
        pending_model_usage={} if control is not None else None,
        stop_event=request.stop_event,
        shutdown_owner_process_ids=(
            control.shutdown_owner_process_ids if control is not None else ()
        ),
        supervisor_process_id=request.supervisor_process_id,
        sequence=sequence,
    )


def run_native_vertical_slice(
    *,
    output_root: Path,
    run_id: str,
    adapter_profile_id: str,
    workload: WorkloadPort,
    visible_evidence_policy: VisibleEvidencePolicy,
    artifact_store_factory: Callable[[Path], CapsuleArtifactStorePort],
    memory_factory: Callable[[ArtifactStorePort, tuple[IngestionPlan, ...]], MemorySystemPort],
    model_factory: Callable[[ArtifactStorePort], ModelClientPort],
    answer_role_binding_id: str,
    judge_model_factory: Callable[[ArtifactStorePort], ModelClientPort] | None = None,
    judge_role_binding_id: str | None = None,
    close_timeout_seconds: float = NATIVE_CLOSE_TIMEOUT_SECONDS,
    control: NativeRunControl | None = None,
    requested_case_manifest_entry_ids: tuple[str, ...] = (),
    terminal_case_publisher: TerminalCasePublisher | None = None,
    stop_event: _NativeStopSignal | None = None,
) -> NativeRunArtifacts:
    """Compose one deterministic native capsule without selecting any live transport."""

    _require_safe_component(run_id, "run ID")
    if not adapter_profile_id or not answer_role_binding_id:
        raise ValueError("native adapter profile and answer role identities are required")
    if (judge_model_factory is None) != (judge_role_binding_id is None):
        raise ValueError("native judge model factory and role identity must be configured together")
    if not math.isfinite(close_timeout_seconds) or close_timeout_seconds <= 0:
        raise ValueError("native close timeout must be finite positive")
    if control is not None:
        if control.run_spec.run_id != run_id:
            raise ValueError("live native control run ID does not match the requested run")
        if control.preflight_record.adapter_profile_id != adapter_profile_id:
            raise ValueError("live native control adapter profile does not match")
        if answer_role_binding_id not in control.preflight_record.role_binding_ids:
            raise ValueError("live native answer role is outside the preflight inventory")
        if judge_role_binding_id is not None and (
            judge_role_binding_id not in control.preflight_record.role_binding_ids
        ):
            raise ValueError("live native judge role is outside the preflight inventory")
    capsule_root = Path(output_root) / run_id
    _acquire_native_run_owner(capsule_root, run_id, control=control)
    lease: RunLeaseRecord | None = None
    lifecycle: ProviderLifecycleBridge | None = None
    authority = None
    live_run_spec_hash: str | None = None
    if control is not None:
        observed_at = control.wall_clock()
        if observed_at.tzinfo is None or observed_at.utcoffset() is None:
            raise ValueError("trusted wall clock requires an explicit timezone")
        lease = _native_run_lease(
            run_id,
            adapter_profile_id,
            control=control,
            acquired_at=observed_at,
        )
        store = artifact_store_factory(capsule_root)
        lifecycle = ProviderLifecycleBridge(
            control.provider_runtime_directory,
            coordination_directory=control.provider_lifecycle_coordination_directory,
        )

        def seal_live_lease() -> None:
            seal_source_contract(
                store,
                relative_path=f"source/run-leases/{lease.lease_epoch}.json",
                record_id=lease.lease_record_hash,
                record=lease,
            )

        authority = lifecycle.acquire_run(
            run_id=run_id,
            provider_project=control.preflight_record.provider_project_id,
            profile_id=adapter_profile_id,
            lease_epoch=lease.lease_epoch,
            lease_record_hash=lease.lease_record_hash,
            durable_lease=seal_live_lease,
        )
        live_run_spec_hash = canonical_sha256(control.run_spec)
    request = _NativeRunRequest(
        output_root=Path(output_root),
        run_id=run_id,
        adapter_profile_id=adapter_profile_id,
        workload=workload,
        visible_evidence_policy=visible_evidence_policy,
        artifact_store_factory=artifact_store_factory,
        memory_factory=memory_factory,
        model_factory=model_factory,
        answer_role_binding_id=answer_role_binding_id,
        judge_model_factory=judge_model_factory,
        judge_role_binding_id=judge_role_binding_id,
        close_timeout_seconds=close_timeout_seconds,
        control=control,
        lease=lease,
        provider_lifecycle=lifecycle,
        requested_case_manifest_entry_ids=requested_case_manifest_entry_ids,
        terminal_case_publisher=terminal_case_publisher,
        stop_event=stop_event,
    )
    try:
        completed = _run_native_supervised(request)
    except BaseException as error:
        if lifecycle is not None and authority is not None and not _contains_unknown_outcome(error):
            if live_run_spec_hash is None:
                raise AssertionError("live lifecycle has no run-spec hash") from error
            try:
                lifecycle.release_run(
                    authority,
                    seal_and_verify_terminal_manifest=lambda: _verify_terminal_capsule(
                        capsule_root,
                        run_id=run_id,
                        run_spec_hash=live_run_spec_hash,
                    ),
                )
            except ProviderLifecycleError as release_error:
                if "active provider attempt" not in str(release_error):
                    raise
                lifecycle.release_supervised_aborted_run(
                    authority,
                    verify_terminal_abort=lambda attempts: _verify_supervised_aborted_capsule(
                        capsule_root,
                        run_id=run_id,
                        run_spec_hash=live_run_spec_hash,
                        active_attempts=attempts,
                    ),
                )
        raise
    if lifecycle is not None and authority is not None:
        if live_run_spec_hash is None:
            raise AssertionError("live lifecycle has no run-spec hash")
        lifecycle.release_run(
            authority,
            seal_and_verify_terminal_manifest=lambda: _verify_terminal_capsule(
                capsule_root,
                run_id=run_id,
                run_spec_hash=live_run_spec_hash,
            ),
        )
    return completed


def _run_native_supervised(request: _NativeRunRequest) -> NativeRunArtifacts:
    try:
        context = multiprocessing.get_context("fork")
    except ValueError as exc:
        raise NativeRunProcessError(
            "native composition requires an operating-system fork boundary"
        ) from exc
    receiver, sender = context.Pipe(duplex=False)
    stop_event = request.stop_event if request.stop_event is not None else context.Event()
    interrupt_event = context.Event()
    supervised_request = replace(
        request,
        stop_event=stop_event,
        interrupt_event=interrupt_event,
        supervisor_process_id=os.getpid(),
    )
    process = context.Process(
        target=_native_process_entry,
        args=(supervised_request, sender),
        name=f"oamb-native-{request.run_id}",
        daemon=False,
    )
    started = False
    previous_signal_handlers: dict[signal.Signals, Any] = {}
    stop_signal_count = [0]
    active_close: _NativeCloseStarted | None = None
    latest_progress: _NativeRunProgress | None = None
    close_deadline: float | None = None
    try:
        previous_signal_handlers = _install_native_stop_handlers(
            stop_event,
            interrupt_event,
            stop_signal_count,
        )
        process.start()
        started = True
        sender.close()
        while True:
            if stop_signal_count[0] > 1:
                _retire_native_process(process)
                error = NativeRunProcessError(
                    "native run force-stopped after a repeated process signal"
                )
                _raise_after_process_abort(
                    supervised_request,
                    latest_progress,
                    active_close,
                    error,
                )
            poll_seconds = NATIVE_PROCESS_POLL_SECONDS
            if close_deadline is not None:
                poll_seconds = min(
                    poll_seconds,
                    max(0.0, close_deadline - time.monotonic()),
                )
            if receiver.poll(poll_seconds):
                try:
                    message: object = receiver.recv()
                except (EOFError, OSError):
                    process.join(NATIVE_PROCESS_TERMINATION_SECONDS)
                    error = NativeRunProcessError(
                        "native composition process closed its result channel "
                        f"with status {process.exitcode}"
                    )
                    _raise_after_process_abort(
                        supervised_request,
                        latest_progress,
                        active_close,
                        error,
                    )
                if isinstance(message, _NativeRunProgress):
                    latest_progress = message
                    continue
                if isinstance(message, _NativeCloseStarted):
                    if active_close is not None:
                        raise NativeRunProcessError("native close stages overlapped")
                    active_close = message
                    latest_progress = _NativeRunProgress(
                        run_spec_hash=message.run_spec_hash,
                        ingestion_occurrence_ids=message.ingestion_occurrence_ids,
                        case_occurrence_ids=message.case_occurrence_ids,
                        sequence=message.sequence,
                    )
                    close_deadline = time.monotonic() + request.close_timeout_seconds
                    continue
                if isinstance(message, _NativeCloseFinished):
                    if active_close is None or message.client_index != active_close.client_index:
                        raise NativeRunProcessError(
                            "native close stage completion was out of order"
                        )
                    active_close = None
                    close_deadline = None
                    continue
                if isinstance(message, _NativeRunReady):
                    latest_progress = _NativeRunProgress(
                        run_spec_hash=message.run_spec_hash,
                        ingestion_occurrence_ids=message.ingestion_occurrence_ids,
                        case_occurrence_ids=message.case_occurrence_ids,
                        sequence=message.sequence,
                    )
                    try:
                        _join_completed_native_process(process)
                    except NativeRunProcessError as join_error:
                        _raise_after_process_abort(
                            supervised_request,
                            latest_progress,
                            active_close,
                            join_error,
                        )
                    return _finalize_supervised_native_success(request, message)
                if isinstance(message, _NativeRunFailed):
                    try:
                        _join_completed_native_process(process)
                    except NativeRunProcessError as error:
                        _raise_after_process_abort(
                            supervised_request,
                            latest_progress,
                            active_close,
                            error,
                        )
                    raise _restore_native_error(message.error)
                raise NativeRunProcessError("native composition returned an invalid message")
            if close_deadline is not None and time.monotonic() >= close_deadline:
                if active_close is None:
                    raise AssertionError("native close deadline has no active client")
                _retire_native_process(process)
                timeout_errors = _seal_supervised_close_abort(
                    supervised_request,
                    active_close,
                )
                if len(timeout_errors) == 1:
                    raise timeout_errors[0]
                raise BaseExceptionGroup(
                    "native hard-close retirement sealed terminal diagnostics",
                    timeout_errors,
                )
            if not process.is_alive() and not receiver.poll(0):
                process.join(0)
                process_error = NativeRunProcessError(
                    f"native composition process exited with status {process.exitcode}"
                )
                _raise_after_process_abort(
                    supervised_request,
                    latest_progress,
                    active_close,
                    process_error,
                )
    finally:
        _restore_native_stop_handlers(previous_signal_handlers)
        try:
            receiver.close()
        finally:
            sender.close()
        if started:
            if process.is_alive():
                _retire_native_process(process)
            process.join(0)
            process.close()
        else:
            process.close()


def _install_native_stop_handlers(
    stop_event: _NativeStopSignal,
    interrupt_event: _NativeStopSignal,
    stop_signal_count: list[int],
) -> dict[signal.Signals, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}

    def request_stop(_signum: int, _frame: object) -> None:
        stop_signal_count[0] += 1
        stop_event.set()
        interrupt_event.set()

    previous: dict[signal.Signals, Any] = {}
    for signum in COOPERATIVE_STOP_SIGNALS:
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    return previous


def _restore_native_stop_handlers(previous: dict[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _native_process_entry(request: _NativeRunRequest, sender: Connection) -> None:
    for signum in COOPERATIVE_STOP_SIGNALS:
        signal.signal(signum, signal.SIG_IGN)
    try:
        ready = asyncio.run(_run_native_with_cell_deadline(request, sender))
    except BaseException as exc:
        sender.send(_NativeRunFailed(_native_error_envelope(exc)))
    else:
        sender.send(ready)
    finally:
        sender.close()


async def _run_native_with_cell_deadline(
    request: _NativeRunRequest,
    sender: Connection | None,
) -> _NativeRunReady:
    async def execute() -> _NativeRunReady:
        return await _run_native_vertical_slice(
            output_root=request.output_root,
            run_id=request.run_id,
            adapter_profile_id=request.adapter_profile_id,
            workload=request.workload,
            visible_evidence_policy=request.visible_evidence_policy,
            artifact_store_factory=request.artifact_store_factory,
            memory_factory=request.memory_factory,
            model_factory=request.model_factory,
            answer_role_binding_id=request.answer_role_binding_id,
            judge_model_factory=request.judge_model_factory,
            judge_role_binding_id=request.judge_role_binding_id,
            close_timeout_seconds=request.close_timeout_seconds,
            control=request.control,
            lease=request.lease,
            provider_lifecycle=request.provider_lifecycle,
            lifecycle_sender=sender,
            requested_case_manifest_entry_ids=getattr(
                request,
                "requested_case_manifest_entry_ids",
                (),
            ),
            terminal_case_publisher=getattr(request, "terminal_case_publisher", None),
            stop_event=getattr(request, "stop_event", None),
            supervisor_process_id=getattr(request, "supervisor_process_id", None),
        )

    async def execute_with_shutdown_watch() -> _NativeRunReady:
        task = asyncio.create_task(execute())
        orphan_shutdown_finished: threading.Event | None = None
        try:
            while not task.done():
                reason = _native_shutdown_reason(request)
                if reason is not None:
                    orphan_shutdown_finished = _arm_shutdown_deadline(
                        request.close_timeout_seconds + NATIVE_PROCESS_TERMINATION_SECONDS
                    )
                    stop_event = getattr(request, "stop_event", None)
                    if stop_event is not None:
                        stop_event.set()
                    task.cancel()
                    try:
                        return await task
                    except BaseException as error:
                        if _is_planned_stop_settlement(error):
                            raise NativeRunInterrupted(reason) from None
                        raise
                await asyncio.wait((task,), timeout=NATIVE_PROCESS_POLL_SECONDS)
            return await task
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            if orphan_shutdown_finished is not None:
                orphan_shutdown_finished.set()

    if request.control is None:
        return await execute_with_shutdown_watch()
    async with asyncio.timeout(float(request.control.budget.max_dispatch_wall_seconds)):
        return await execute_with_shutdown_watch()


def _native_shutdown_reason(request: _NativeRunRequest) -> str | None:
    interrupt_event = getattr(request, "interrupt_event", None)
    if interrupt_event is not None and interrupt_event.is_set():
        return "native run stopped after an operator signal"
    supervisor_process_id = getattr(request, "supervisor_process_id", None)
    if _native_supervisor_exited(supervisor_process_id):
        return _SUPERVISOR_EXITED_REASON
    if request.control is not None and any(
        not _process_exists(process_id)
        for process_id in getattr(request.control, "shutdown_owner_process_ids", ())
    ):
        return "native run stopped after an owning process exited"
    return None


def _native_supervisor_exited(supervisor_process_id: int | None) -> bool:
    return supervisor_process_id is not None and os.getppid() != supervisor_process_id


def _arm_shutdown_deadline(delay_seconds: float) -> threading.Event:
    finished = threading.Event()

    def stop_after_deadline() -> None:
        if not finished.wait(delay_seconds):
            os._exit(1)

    threading.Thread(
        target=stop_after_deadline,
        name="oamb-shutdown-deadline",
        daemon=True,
    ).start()
    return finished


def _process_exists(process_id: int) -> bool:
    try:
        os.kill(process_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _join_completed_native_process(process: _StoppableNativeProcess) -> None:
    process.join(NATIVE_PROCESS_TERMINATION_SECONDS)
    if process.is_alive():
        _retire_native_process(process)
    if process.exitcode != 0:
        raise NativeRunProcessError(
            f"native composition process exited with status {process.exitcode} after handoff"
        )


def _retire_native_process(process: _StoppableNativeProcess) -> None:
    process.terminate()
    process.join(NATIVE_PROCESS_TERMINATION_SECONDS)
    if not process.is_alive():
        return
    process.kill()
    process.join(NATIVE_PROCESS_TERMINATION_SECONDS)
    if process.is_alive():
        raise NativeRunProcessTerminationError(
            "native composition survived terminate and kill bounds"
        )


def _finalize_supervised_native_success(
    request: _NativeRunRequest,
    ready: _NativeRunReady,
) -> NativeRunArtifacts:
    store = request.artifact_store_factory(ready.capsule_root)
    state = _new_execution_state(request, store, sequence=ready.sequence)
    _seal_native_run_record(
        state,
        run_spec_hash=ready.run_spec_hash,
        run_state=RunState.FINALIZED,
        ingestion_occurrence_ids=ready.ingestion_occurrence_ids,
        case_occurrence_ids=ready.case_occurrence_ids,
    )
    manifest = store.finalize_capsule(
        run_id=request.run_id,
        run_spec_hash=ready.run_spec_hash,
    )
    return NativeRunArtifacts(
        capsule_root=ready.capsule_root,
        manifest=manifest,
        ingestion_plan_records=ready.ingestion_plan_records,
        case_records=ready.case_records,
    )


def _native_error_envelope(error: BaseException) -> _NativeErrorEnvelope:
    if isinstance(error, NativeCancellationEvidenceError):
        return _NativeErrorEnvelope(
            category="cancelled",
            type_name=type(error).__name__,
            message=str(error),
            children=tuple(_native_error_envelope(item) for item in error.errors),
        )
    if isinstance(error, BaseExceptionGroup):
        return _NativeErrorEnvelope(
            category="base",
            type_name=type(error).__name__,
            message=error.message,
            children=tuple(_native_error_envelope(item) for item in error.exceptions),
        )
    if isinstance(error, asyncio.CancelledError) or getattr(error, "receipt", None) is not None:
        wrapped_dispatch_error = _wrapped_dispatch_error(error)
        if wrapped_dispatch_error is not None and wrapped_dispatch_error is not error:
            return _native_error_envelope(wrapped_dispatch_error)
    category: _NativeErrorCategory
    if isinstance(error, NativeRunInterrupted):
        category = "interrupted"
    elif isinstance(error, MemorySystemCallCancelledBeforeDispatch):
        category = "memory_cancelled_before_dispatch"
    elif isinstance(error, MemorySystemCallCancelledUnknownOutcome):
        category = "memory_cancelled_unknown"
    elif isinstance(error, ModelCallCancelledUnknownOutcome):
        category = "model_cancelled_unknown"
    elif isinstance(error, MemorySystemCallUnknownOutcome):
        category = "memory_unknown"
    elif isinstance(error, ModelCallUnknownOutcome):
        category = "model_unknown"
    elif isinstance(error, asyncio.CancelledError):
        category = "cancelled"
    elif isinstance(error, AssertionError):
        category = "assertion"
    elif isinstance(error, TimeoutError):
        category = "timeout"
    elif isinstance(error, ValueError):
        category = "value"
    elif isinstance(error, RuntimeError):
        category = "runtime"
    elif isinstance(error, SystemExit):
        category = "system_exit"
    else:
        category = "base"
    return _NativeErrorEnvelope(
        category=category,
        type_name=type(error).__name__,
        message=str(error),
        failure_kind=getattr(error, "failure_kind", None),
    )


def _wrapped_dispatch_error(error: BaseException) -> BaseException | None:
    typed_errors = (
        MemorySystemCallCancelledBeforeDispatch,
        MemorySystemCallCancelledUnknownOutcome,
        ModelCallCancelledUnknownOutcome,
        MemorySystemCallUnknownOutcome,
        ModelCallUnknownOutcome,
    )
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if isinstance(candidate, typed_errors):
            return candidate
        for linked in (candidate.__cause__, candidate.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
        receipt = getattr(candidate, "receipt", None)
        for linked in (
            getattr(receipt, "provider_error", None),
            getattr(receipt, "post_projection_error", None),
        ):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return None


def _restore_native_error(envelope: _NativeErrorEnvelope) -> BaseException:
    if envelope.children:
        restored_children = tuple(_restore_native_error(item) for item in envelope.children)
        if envelope.category == "cancelled" and len(restored_children) == 2:
            return NativeCancellationEvidenceError(*restored_children)
        return BaseExceptionGroup(
            envelope.message,
            restored_children,
        )
    if envelope.category == "memory_cancelled_before_dispatch":
        return MemorySystemCallCancelledBeforeDispatch(envelope.message)
    if envelope.category == "memory_cancelled_unknown":
        return MemorySystemCallCancelledUnknownOutcome(envelope.message)
    if envelope.category == "model_cancelled_unknown":
        return ModelCallCancelledUnknownOutcome(envelope.message)
    if envelope.category == "memory_unknown":
        return MemorySystemCallUnknownOutcome(
            envelope.message,
            failure_kind=envelope.failure_kind or "unknown_outcome",
        )
    if envelope.category == "model_unknown":
        return ModelCallUnknownOutcome(
            envelope.message,
            failure_kind=envelope.failure_kind or "unknown_outcome",
        )
    if envelope.category == "interrupted":
        return NativeRunInterrupted(envelope.message)
    if envelope.category == "cancelled":
        return asyncio.CancelledError(envelope.message)
    if envelope.category == "assertion":
        return AssertionError(envelope.message)
    if envelope.category == "timeout":
        return TimeoutError(envelope.message)
    if envelope.category == "value":
        return ValueError(envelope.message)
    if envelope.category == "system_exit":
        return SystemExit(envelope.message)
    if envelope.category == "runtime":
        return RuntimeError(envelope.message)
    return NativeRunProcessError(f"native child raised {envelope.type_name}: {envelope.message}")


def _seal_supervised_close_abort(
    request: _NativeRunRequest,
    close: _NativeCloseStarted,
) -> tuple[BaseException, ...]:
    return _seal_supervised_close_failure(
        request,
        close,
        NativePortCloseTimeout(
            f"{close.clients[close.client_index][1]} exceeded "
            f"{request.close_timeout_seconds:g} seconds"
        ),
    )


def _seal_supervised_close_failure(
    request: _NativeRunRequest,
    close: _NativeCloseStarted,
    active_error: BaseException,
) -> tuple[BaseException, ...]:
    store = request.artifact_store_factory(request.output_root / request.run_id)
    state = _new_execution_state(request, store, sequence=close.sequence)
    errors: list[BaseException] = []
    for client_index in range(close.client_index, len(close.clients)):
        profile_id, shutdown_stage = close.clients[client_index]
        if client_index == close.client_index:
            error = active_error
        else:
            error = NativePortCloseSkipped(
                f"{shutdown_stage} skipped after native process hard retirement"
            )
        errors.append(error)
        try:
            _seal_close_error(
                state,
                client_profile_id=profile_id,
                shutdown_stage=shutdown_stage,
                error=error,
            )
        except BaseException as seal_error:
            errors.append(seal_error)
    try:
        _seal_native_run_record(
            state,
            run_spec_hash=close.run_spec_hash,
            run_state=RunState.ABORTED,
            ingestion_occurrence_ids=close.ingestion_occurrence_ids,
            case_occurrence_ids=close.case_occurrence_ids,
        )
        store.finalize_capsule(run_id=request.run_id, run_spec_hash=close.run_spec_hash)
    except BaseException as diagnostic_error:
        errors.append(diagnostic_error)
    return tuple(errors)


def _raise_after_process_abort(
    request: _NativeRunRequest,
    progress: _NativeRunProgress | None,
    active_close: _NativeCloseStarted | None,
    process_error: NativeRunProcessError,
) -> None:
    if active_close is not None:
        close_errors = _seal_supervised_close_failure(request, active_close, process_error)
        if len(close_errors) == 1:
            raise close_errors[0]
        raise BaseExceptionGroup(
            "native process crashed during supervised close",
            close_errors,
        )
    diagnostic_errors = _seal_supervised_process_abort(request, progress, process_error)
    if diagnostic_errors:
        raise BaseExceptionGroup(
            "native process crash and diagnostic seal failed",
            (process_error, *diagnostic_errors),
        )
    raise process_error


def _seal_supervised_process_abort(
    request: _NativeRunRequest,
    progress: _NativeRunProgress | None,
    process_error: NativeRunProcessError,
) -> tuple[BaseException, ...]:
    store = request.artifact_store_factory(request.output_root / request.run_id)
    run_path = request.output_root / request.run_id / "source" / "run" / f"{request.run_id}.json"
    try:
        existing_run = RunRecord.model_validate_json(read_regular_file(run_path))
    except (OSError, ValueError):
        existing_run = None
    run_spec_hash = (
        progress.run_spec_hash
        if progress is not None
        else (
            canonical_sha256(request.control.run_spec)
            if request.control is not None
            else canonical_sha256(
                [
                    "oamb-native-fixture-crash-run-spec-v1",
                    request.run_id,
                    request.adapter_profile_id,
                ]
            )
        )
    )
    state = _new_execution_state(
        request,
        store,
        sequence=progress.sequence if progress is not None else 0,
    )
    errors: list[BaseException] = []
    if existing_run is None:
        try:
            _seal_close_error(
                state,
                client_profile_id=request.adapter_profile_id,
                shutdown_stage="composition_process",
                error=process_error,
            )
        except BaseException as seal_error:
            errors.append(seal_error)
        try:
            _seal_native_run_record(
                state,
                run_spec_hash=run_spec_hash,
                run_state=RunState.ABORTED,
                ingestion_occurrence_ids=(
                    progress.ingestion_occurrence_ids if progress is not None else ()
                ),
                case_occurrence_ids=(progress.case_occurrence_ids if progress is not None else ()),
            )
        except BaseException as seal_error:
            errors.append(seal_error)
    else:
        run_spec_hash = existing_run.run_spec_hash
    try:
        store.finalize_capsule(run_id=request.run_id, run_spec_hash=run_spec_hash)
    except BaseException as diagnostic_error:
        errors.append(diagnostic_error)
    return tuple(errors)


async def _run_native_vertical_slice(
    *,
    output_root: Path,
    run_id: str,
    adapter_profile_id: str,
    workload: WorkloadPort,
    visible_evidence_policy: VisibleEvidencePolicy,
    artifact_store_factory: Callable[[Path], CapsuleArtifactStorePort],
    memory_factory: Callable[[ArtifactStorePort, tuple[IngestionPlan, ...]], MemorySystemPort],
    model_factory: Callable[[ArtifactStorePort], ModelClientPort],
    answer_role_binding_id: str,
    judge_model_factory: Callable[[ArtifactStorePort], ModelClientPort] | None,
    judge_role_binding_id: str | None,
    close_timeout_seconds: float,
    control: NativeRunControl | None = None,
    lease: RunLeaseRecord | None = None,
    provider_lifecycle: ProviderLifecycleBridge | None = None,
    lifecycle_sender: Connection | None = None,
    requested_case_manifest_entry_ids: tuple[str, ...] = (),
    terminal_case_publisher: TerminalCasePublisher | None = None,
    stop_event: _NativeStopSignal | None = None,
    supervisor_process_id: int | None = None,
) -> _NativeRunReady:
    capsule_root = output_root / run_id
    store = artifact_store_factory(capsule_root)
    dataset_manifest = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset_manifest)
    ingestion_plans = workload.iter_ingestion_plans(case_manifest)
    case_plans = workload.iter_case_plans(case_manifest)
    if requested_case_manifest_entry_ids:
        selected = select_case_execution(
            case_manifest=case_manifest,
            ingestion_plans=ingestion_plans,
            case_plans=case_plans,
            requested_case_manifest_entry_ids=requested_case_manifest_entry_ids,
        )
        ingestion_plans = selected.ingestion_plans
        case_plans = selected.case_plans
    if not ingestion_plans or not case_plans:
        raise ValueError("native capsule requires at least one ingestion plan and case")
    request_identity = _NativeRunRequest(
        output_root=output_root,
        run_id=run_id,
        adapter_profile_id=adapter_profile_id,
        workload=workload,
        visible_evidence_policy=visible_evidence_policy,
        artifact_store_factory=artifact_store_factory,
        memory_factory=memory_factory,
        model_factory=model_factory,
        answer_role_binding_id=answer_role_binding_id,
        judge_model_factory=judge_model_factory,
        judge_role_binding_id=judge_role_binding_id,
        close_timeout_seconds=close_timeout_seconds,
        control=control,
        lease=lease,
        provider_lifecycle=provider_lifecycle,
        requested_case_manifest_entry_ids=requested_case_manifest_entry_ids,
        terminal_case_publisher=terminal_case_publisher,
        stop_event=stop_event,
        supervisor_process_id=supervisor_process_id,
    )
    lease = lease or _native_run_lease(run_id, adapter_profile_id, control=control)
    state = _new_execution_state(request_identity, store)
    if control is None:
        _seal(state, "run-leases", lease.lease_record_hash, lease)
    memory: MemorySystemPort | None = None
    answer_model: ModelClientPort | None = None
    judge_model: ModelClientPort | None = None
    execution_error: BaseException | None = None
    run_spec_hash = (
        canonical_sha256(control.run_spec)
        if control is not None
        else canonical_sha256(
            [
                "oamb-native-fixture-diagnostic-run-spec-v1",
                run_id,
                dataset_manifest.manifest_hash,
                case_manifest.manifest_hash,
                adapter_profile_id,
            ]
        )
    )
    plan_records: tuple[NativeIngestionPlanRecord, ...] = ()
    case_records: tuple[CaseRecordV3, ...] = ()
    ingestion_occurrence_ids: tuple[str, ...] = ()
    case_occurrence_ids: tuple[str, ...] = ()
    _seal(state, "specs", "dataset-manifest", dataset_manifest)
    _seal(state, "specs", "case-manifest", case_manifest)
    if control is not None:
        _require_live_manifest_closure(control, dataset_manifest, case_manifest)
        _seal_live_control(state, control)
    if lifecycle_sender is not None:
        lifecycle_sender.send(
            _NativeRunProgress(
                run_spec_hash=run_spec_hash,
                ingestion_occurrence_ids=(),
                case_occurrence_ids=(),
                sequence=state.sequence,
            )
        )
    try:
        memory = memory_factory(store, ingestion_plans)
        model_store: ArtifactStorePort = store
        if state.pending_model_usage is not None:
            model_store = _LiveModelUsageCaptureStore(store, state.pending_model_usage)
        answer_model = model_factory(model_store)
        if judge_model_factory is not None:
            judge_model = judge_model_factory(model_store)
        setup_artifacts: tuple[_LiveDispatchArtifacts, ...] = ()
        if _native_stop_requested(state):
            raise NativeRunInterrupted("native run stopped admission before runtime resolution")
        if control is not None:
            runtime_parent_id = _initial_history_occurrence(
                state,
                control.run_spec.memory_system_id,
                ingestion_plans[0].ingestion_plan_id,
            )
            runtime_fingerprint = canonical_sha256(
                ["oamb-live-runtime-resolve-v1", run_id, adapter_profile_id]
            )
            runtime_result, runtime_artifacts = await _execute_live_memory_dispatch(
                state,
                parent_kind="ingestion_plan",
                parent_id=runtime_parent_id,
                stage="runtime_resolve",
                ordinal=1,
                request_fingerprint=runtime_fingerprint,
                role_binding_id=adapter_profile_id,
                call=memory.resolve,
                raw_reference=lambda _result: None,
            )
            runtime = runtime_result
            setup_artifacts = (runtime_artifacts,)
        else:
            runtime = await memory.resolve()
        capabilities = await memory.capabilities()
        if not capabilities.provider_order_preserved or not capabilities.native_reranking_disabled:
            raise ValueError("native adapter must preserve provider order with reranking disabled")
        if control is not None:
            if (
                runtime.memory_system_id != control.run_spec.memory_system_id
                or runtime.runtime_binding_hash != control.run_spec.runtime_binding_hash
            ):
                raise ValueError("live native runtime resolution differs from the run spec")
        else:
            run_spec_hash = canonical_sha256(
                [
                    "oamb-native-fixture-run-spec-v1",
                    run_id,
                    dataset_manifest.manifest_hash,
                    case_manifest.manifest_hash,
                    runtime.memory_system_id,
                    runtime.runtime_binding_hash,
                    adapter_profile_id,
                ]
            )
        ingestion_occurrence_ids = tuple(
            _initial_history_occurrence(state, runtime.memory_system_id, plan.ingestion_plan_id)
            for plan in ingestion_plans
        )
        ingestion_occurrence_by_plan = dict(
            zip(
                (plan.ingestion_plan_id for plan in ingestion_plans),
                ingestion_occurrence_ids,
                strict=True,
            )
        )
        plan_by_case = {
            case_id: plan.ingestion_plan_id
            for plan in ingestion_plans
            for case_id in plan.ordered_case_manifest_entry_ids
        }
        case_occurrence_ids = tuple(
            case_occurrence_id(
                ingestion_occurrence_by_plan[plan_by_case[case.case_manifest_entry_id]],
                case.case_manifest_entry_id,
            )
            for case in case_plans
            if case.case_manifest_entry_id in plan_by_case
        )
        if lifecycle_sender is not None:
            lifecycle_sender.send(
                _NativeRunProgress(
                    run_spec_hash=run_spec_hash,
                    ingestion_occurrence_ids=ingestion_occurrence_ids,
                    case_occurrence_ids=case_occurrence_ids,
                    sequence=state.sequence,
                )
            )

        def publish_history_progress() -> None:
            if lifecycle_sender is None:
                return
            history_ids, case_ids = _executed_occurrence_ids(state, ingestion_plans, case_plans)
            lifecycle_sender.send(
                _NativeRunProgress(
                    run_spec_hash=run_spec_hash,
                    ingestion_occurrence_ids=history_ids,
                    case_occurrence_ids=case_ids,
                    sequence=state.sequence,
                )
            )

        state.publish_history_progress = publish_history_progress
        plan_records, case_records = await _execute_history_question_pipeline(
            state=state,
            workload=workload,
            memory=memory,
            answer_model=answer_model,
            judge_model=judge_model,
            plans=ingestion_plans,
            case_plans=case_plans,
            memory_system_id=runtime.memory_system_id,
            runtime_binding_hash=runtime.runtime_binding_hash,
            adapter_profile_id=adapter_profile_id,
            visible_evidence_policy=visible_evidence_policy,
            answer_role_binding_id=answer_role_binding_id,
            judge_role_binding_id=judge_role_binding_id,
            setup_artifacts=setup_artifacts,
            terminal_case_publisher=terminal_case_publisher,
        )
    except BaseException as exc:
        if state.stop_event is not None:
            state.stop_event.set()
        execution_error = exc
    if state.history_occurrence_bindings:
        ingestion_occurrence_ids, case_occurrence_ids = _executed_occurrence_ids(
            state,
            ingestion_plans,
            case_plans,
        )
    close_errors = await _close_ports(
        state,
        answer_model=answer_model,
        judge_model=judge_model,
        memory=memory,
        answer_model_profile_id=answer_role_binding_id,
        judge_model_profile_id=judge_role_binding_id,
        memory_profile_id=adapter_profile_id,
        run_spec_hash=run_spec_hash,
        ingestion_occurrence_ids=ingestion_occurrence_ids,
        case_occurrence_ids=case_occurrence_ids,
        lifecycle_sender=lifecycle_sender,
    )
    if execution_error is None and _native_supervisor_exited(supervisor_process_id):
        execution_error = NativeRunInterrupted(_SUPERVISOR_EXITED_REASON)
    if execution_error is None and state.pending_model_usage:
        execution_error = ValueError(
            "live model usage contains unreferenced records at terminal seal"
        )
    terminal_errors = (
        *((execution_error,) if execution_error is not None else ()),
        *close_errors,
    )
    if terminal_errors:
        diagnostic_error: BaseException | None = None
        try:
            terminal_run_state: Literal[
                RunState.ABORTED,
                RunState.INFRASTRUCTURE_BLOCKED,
            ] = (
                RunState.INFRASTRUCTURE_BLOCKED
                if execution_error is not None
                and not close_errors
                and _is_pure_infrastructure_retry_exhaustion(execution_error)
                else RunState.ABORTED
            )
            _seal_native_run_record(
                state,
                run_spec_hash=run_spec_hash,
                run_state=terminal_run_state,
                ingestion_occurrence_ids=ingestion_occurrence_ids,
                case_occurrence_ids=case_occurrence_ids,
            )
            store.finalize_capsule(run_id=run_id, run_spec_hash=run_spec_hash)
        except BaseException as exc:
            diagnostic_error = exc
        if diagnostic_error is not None:
            terminal_errors = (*terminal_errors, diagnostic_error)
        elif (
            terminal_run_state == RunState.ABORTED
            and provider_lifecycle is not None
            and _native_supervisor_exited(supervisor_process_id)
        ):
            try:
                provider_lifecycle.release_current_supervised_aborted_run(
                    verify_terminal_abort=lambda attempts: _verify_supervised_aborted_capsule(
                        capsule_root,
                        run_id=run_id,
                        run_spec_hash=run_spec_hash,
                        active_attempts=attempts,
                    )
                )
            except BaseException as exc:
                terminal_errors = (*terminal_errors, exc)
    if execution_error is not None:
        if len(terminal_errors) > 1:
            raise BaseExceptionGroup(
                "native execution, shutdown, or diagnostic seal failed",
                terminal_errors,
            )
        raise execution_error
    if len(terminal_errors) == 1:
        raise terminal_errors[0]
    if terminal_errors:
        raise BaseExceptionGroup("native shutdown or diagnostic seal failed", terminal_errors)
    return _NativeRunReady(
        capsule_root=capsule_root,
        run_spec_hash=run_spec_hash,
        ingestion_occurrence_ids=ingestion_occurrence_ids,
        case_occurrence_ids=case_occurrence_ids,
        ingestion_plan_records=plan_records,
        case_records=case_records,
        sequence=state.sequence,
    )


def _initial_history_occurrence(
    state: _NativeExecutionState,
    memory_system_id: str,
    plan_id: str,
) -> str:
    return ingestion_occurrence_id(
        state.run_id,
        memory_system_id,
        plan_id,
        history_attempt_ordinal=1,
    )


def _executed_occurrence_ids(
    state: _NativeExecutionState,
    plans: tuple[IngestionPlan, ...],
    case_plans: tuple[CasePlan, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    plan_order = {plan.ingestion_plan_id: index for index, plan in enumerate(plans)}
    histories = tuple(
        sorted(
            state.history_occurrence_bindings,
            key=lambda occurrence: (
                plan_order[state.history_occurrence_bindings[occurrence][0]],
                state.history_occurrence_bindings[occurrence][1],
            ),
        )
    )
    cases = tuple(
        state.case_occurrence_bindings[case.case_manifest_entry_id]
        for case in case_plans
        if case.case_manifest_entry_id in state.case_occurrence_bindings
    )
    return histories, cases


async def _close_ports(
    state: _NativeExecutionState,
    *,
    answer_model: ModelClientPort | None,
    judge_model: ModelClientPort | None,
    memory: MemorySystemPort | None,
    answer_model_profile_id: str,
    judge_model_profile_id: str | None,
    memory_profile_id: str,
    run_spec_hash: str,
    ingestion_occurrence_ids: tuple[str, ...],
    case_occurrence_ids: tuple[str, ...],
    lifecycle_sender: Connection | None,
) -> tuple[BaseException, ...]:
    errors: list[BaseException] = []
    clients = (
        (judge_model, judge_model_profile_id, "judge_model_close"),
        (answer_model, answer_model_profile_id, "model_close"),
        (memory, memory_profile_id, "memory_close"),
    )
    active_clients = tuple(
        (client, profile_id, shutdown_stage)
        for client, profile_id, shutdown_stage in clients
        if client is not None and profile_id is not None
    )
    public_clients = tuple(
        (profile_id, shutdown_stage) for _client, profile_id, shutdown_stage in active_clients
    )
    for client_index, (client, profile_id, shutdown_stage) in enumerate(active_clients):
        if lifecycle_sender is not None:
            lifecycle_sender.send(
                _NativeCloseStarted(
                    run_spec_hash=run_spec_hash,
                    ingestion_occurrence_ids=ingestion_occurrence_ids,
                    case_occurrence_ids=case_occurrence_ids,
                    sequence=state.sequence,
                    clients=public_clients,
                    client_index=client_index,
                )
            )
        try:
            await client.close()
        except BaseException as exc:
            errors.append(exc)
            try:
                _seal_close_error(
                    state,
                    client_profile_id=profile_id,
                    shutdown_stage=shutdown_stage,
                    error=exc,
                )
            except BaseException as seal_error:
                errors.append(seal_error)
        finally:
            if lifecycle_sender is not None:
                lifecycle_sender.send(_NativeCloseFinished(client_index=client_index))
    return tuple(errors)


async def _execute_history_question_pipeline(
    *,
    state: _NativeExecutionState,
    workload: WorkloadPort,
    memory: MemorySystemPort,
    answer_model: ModelClientPort,
    judge_model: ModelClientPort | None,
    plans: tuple[IngestionPlan, ...],
    case_plans: tuple[CasePlan, ...],
    memory_system_id: str,
    runtime_binding_hash: str,
    adapter_profile_id: str,
    visible_evidence_policy: VisibleEvidencePolicy,
    answer_role_binding_id: str,
    judge_role_binding_id: str | None,
    setup_artifacts: tuple[_LiveDispatchArtifacts, ...] = (),
    terminal_case_publisher: TerminalCasePublisher | None = None,
) -> tuple[tuple[NativeIngestionPlanRecord, ...], tuple[CaseRecordV3, ...]]:
    history_limit = state.control.max_parallel_history_ingestions if state.control else 1
    question_limit = state.control.max_parallel_questions if state.control else 1
    history_permits = asyncio.Semaphore(history_limit)
    question_permits = asyncio.Semaphore(question_limit)
    admission_stopped = False
    plan_records: list[NativeIngestionPlanRecord | None] = [None] * len(plans)
    case_records: list[CaseRecordV3 | None] = [None] * len(case_plans)
    case_by_id = {
        case.case_manifest_entry_id: (index, case) for index, case in enumerate(case_plans)
    }
    if len(case_by_id) != len(case_plans):
        raise ValueError("native case manifest identities must be unique")
    cases_by_plan: dict[str, tuple[tuple[int, CasePlan], ...]] = {}
    assigned_case_ids: set[str] = set()
    for plan in plans:
        selected: list[tuple[int, CasePlan]] = []
        for case_id in plan.ordered_case_manifest_entry_ids:
            indexed = case_by_id.get(case_id)
            if indexed is None:
                raise ValueError("native ingestion plan references an unknown case")
            if case_id in assigned_case_ids:
                raise ValueError("native case belongs to multiple ingestion plans")
            assigned_case_ids.add(case_id)
            selected.append(indexed)
        cases_by_plan[plan.ingestion_plan_id] = tuple(selected)
    if assigned_case_ids != set(case_by_id):
        raise ValueError("native case has no ingestion plan")

    async def admitted(
        permits: asyncio.Semaphore,
        operation: Callable[[], Awaitable[_LiveDispatchResult]],
    ) -> tuple[bool, _LiveDispatchResult | None]:
        nonlocal admission_stopped
        if admission_stopped or _native_stop_requested(state):
            return False, None
        await permits.acquire()
        if admission_stopped or _native_stop_requested(state):
            permits.release()
            return False, None
        try:
            return True, await operation()
        except BaseException as error:
            if not isinstance(error, NativeRunInterrupted):
                admission_stopped = True
            stop_event = getattr(state, "stop_event", None)
            if stop_event is not None:
                stop_event.set()
            raise
        finally:
            permits.release()

    async def execute_question(
        *,
        plan: IngestionPlan,
        plan_record: NativeIngestionPlanRecord,
        case_index: int,
        case_plan: CasePlan,
        scope: ScopeReceipt,
    ) -> None:
        async def operation() -> CaseRecordV3:
            records = await _execute_cases_serial(
                state=state,
                workload=workload,
                memory=memory,
                answer_model=answer_model,
                judge_model=judge_model,
                plans=(plan,),
                case_plans=(case_plan,),
                scopes={plan.ingestion_plan_id: scope},
                memory_system_id=memory_system_id,
                adapter_profile_id=adapter_profile_id,
                visible_evidence_policy=visible_evidence_policy,
                answer_role_binding_id=answer_role_binding_id,
                judge_role_binding_id=judge_role_binding_id,
            )
            if len(records) != 1:
                raise AssertionError("admitted question did not produce exactly one result")
            result = records[0]
            if terminal_case_publisher is not None:
                terminal_case_publisher(
                    plan_record,
                    result,
                    dict(state.operation_records),
                    tuple(state.history_records),
                )
            return result

        was_admitted, result = await admitted(question_permits, operation)
        if not was_admitted:
            return
        if result is None:
            raise AssertionError("admitted question did not produce a result")
        case_records[case_index] = result

    async def execute_history(plan_index: int, plan: IngestionPlan) -> None:
        async def operation() -> tuple[NativeIngestionPlanRecord, ScopeReceipt]:
            records, scopes = await _execute_history(
                state=state,
                memory=memory,
                plan=plan,
                case_plans=case_plans,
                memory_system_id=memory_system_id,
                runtime_binding_hash=runtime_binding_hash,
                adapter_profile_id=adapter_profile_id,
                setup_artifacts=setup_artifacts if plan_index == 0 else (),
            )
            if len(records) != 1 or set(scopes) != {plan.ingestion_plan_id}:
                raise AssertionError("admitted history did not close one record and scope")
            return records[0], scopes[plan.ingestion_plan_id]

        try:
            was_admitted, result = await admitted(history_permits, operation)
        except BaseException as error:
            if not isinstance(error, (asyncio.CancelledError, NativeRunInterrupted)):
                print(
                    f"oamb: provider={memory_system_id} "
                    f"history={plan.ingestion_plan_id} status=failed, "
                    f"reason={type(error).__name__}: {error}; "
                    "draining admitted operations",
                    file=sys.stderr,
                    flush=True,
                )
            raise
        if not was_admitted:
            return
        if result is None:
            raise AssertionError("admitted history did not produce a result")
        plan_record, scope = result
        plan_records[plan_index] = plan_record
        question_results = await _settle_pipeline_operations(
            tuple(
                execute_question(
                    plan=plan,
                    plan_record=plan_record,
                    case_index=case_index,
                    case_plan=case_plan,
                    scope=scope,
                )
                for case_index, case_plan in cases_by_plan[plan.ingestion_plan_id]
            )
        )
        _raise_pipeline_errors(question_results)

    history_results = await _settle_pipeline_operations(
        tuple(execute_history(index, plan) for index, plan in enumerate(plans)),
        cancel_when=lambda: _native_stop_requested(state) and not admission_stopped,
    )
    _raise_pipeline_errors(history_results)
    if _native_stop_requested(state) and (
        any(record is None for record in plan_records)
        or any(record is None for record in case_records)
    ):
        raise NativeRunInterrupted(
            "native run stopped admission and drained every accepted operation"
        )
    if any(record is None for record in plan_records):
        raise AssertionError("pipeline did not produce every admitted history record")
    if any(record is None for record in case_records):
        raise AssertionError("pipeline did not produce every admitted question record")
    return (
        tuple(record for record in plan_records if record is not None),
        tuple(record for record in case_records if record is not None),
    )


def _native_stop_requested(state: object) -> bool:
    stop_event = getattr(state, "stop_event", None)
    if stop_event is not None and stop_event.is_set():
        return True
    supervisor_process_id = getattr(state, "supervisor_process_id", None)
    owner_process_ids = getattr(state, "shutdown_owner_process_ids", ())
    owner_lost = (
        supervisor_process_id is not None and os.getppid() != supervisor_process_id
    ) or any(not _process_exists(process_id) for process_id in owner_process_ids)
    if owner_lost and stop_event is not None:
        stop_event.set()
    return owner_lost


async def _capture_pipeline_outcome(
    operation: Awaitable[_LiveDispatchResult],
) -> _LiveDispatchResult | BaseException:
    try:
        return await operation
    except asyncio.CancelledError as error:
        return error
    except BaseException as error:
        return error


async def _settle_pipeline_operations(
    operations: Sequence[Awaitable[_LiveDispatchResult]],
    *,
    cancel_when: Callable[[], bool] | None = None,
) -> tuple[_LiveDispatchResult | BaseException, ...]:
    tasks = tuple(
        asyncio.create_task(_capture_pipeline_outcome(operation)) for operation in operations
    )
    try:
        while tasks and cancel_when is not None and not all(task.done() for task in tasks):
            if cancel_when():
                for task in tasks:
                    if not task.done():
                        task.cancel()
                settled = await asyncio.gather(*tasks, return_exceptions=True)
                evidence_errors = tuple(
                    outcome
                    for outcome in settled
                    if isinstance(outcome, BaseException)
                    and not _is_planned_stop_settlement(outcome)
                )
                interruption = NativeRunInterrupted(
                    "native run stopped admission and cancelled accepted operations"
                )
                if evidence_errors:
                    raise BaseExceptionGroup(
                        "operator stop retained cancellation evidence failures",
                        [interruption, *evidence_errors],
                    )
                raise interruption
            await asyncio.wait(tasks, timeout=NATIVE_PROCESS_POLL_SECONDS)
        return tuple(await asyncio.gather(*tasks))
    except asyncio.CancelledError as cancellation:
        for task in tasks:
            if not task.done():
                task.cancel()
        cancellation_settled = await asyncio.gather(*tasks, return_exceptions=True)
        drain_errors = tuple(
            outcome
            for outcome in cancellation_settled
            if isinstance(outcome, BaseException) and type(outcome) is not asyncio.CancelledError
        )
        if drain_errors:
            raise BaseExceptionGroup(
                "pipeline cancellation retained active task failures",
                [cancellation, *drain_errors],
            ) from None
        raise


def _is_planned_stop_settlement(error: BaseException) -> bool:
    if isinstance(error, NativeCancellationEvidenceError):
        return False
    if isinstance(error, (asyncio.CancelledError, NativeRunInterrupted)):
        return True
    return (
        isinstance(error, BaseExceptionGroup)
        and bool(error.exceptions)
        and all(_is_planned_stop_settlement(child) for child in error.exceptions)
    )


def _raise_pipeline_errors(results: Sequence[object]) -> None:
    errors = tuple(result for result in results if isinstance(result, BaseException))
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("multiple admitted pipeline tasks failed", list(errors))


def _seal_history_attempt(
    *,
    state: _NativeExecutionState,
    plan: IngestionPlan,
    memory_system_id: str,
    runtime_binding_hash: str,
    execution_run_id: str,
    ordinal: int,
    maximum_retries: int,
    started_at: datetime,
    ready_record: NativeIngestionPlanRecord | None = None,
    error: BaseException | None = None,
) -> HistoryAttemptRecord:
    occurrence = ingestion_occurrence_id(
        execution_run_id,
        memory_system_id,
        plan.ingestion_plan_id,
        history_attempt_ordinal=ordinal,
    )
    operations = tuple(
        record
        for record in state.operation_records.values()
        if record.parent_kind == "ingestion_plan" and record.parent_id == occurrence
    )
    failure = (
        None
        if ready_record is not None
        else next(
            (
                record
                for record in reversed(operations)
                if record.outcome != AttemptOutcome.SUCCEEDED
            ),
            None,
        )
    )
    eligible = bool(
        isinstance(error, SettledTransientIngestionFailure)
        and failure is not None
        and failure.outcome == AttemptOutcome.FAILED
        and failure.stage in {"memory_ingest", "memory_readiness"}
    )
    status = (
        "ready"
        if ready_record is not None
        else "retryable_failed_settled"
        if eligible
        else "unknown"
        if error is not None and _contains_unknown_outcome(error)
        else "cancelled"
        if isinstance(error, (asyncio.CancelledError, NativeRunInterrupted))
        else "failed"
    )
    scope = state.history_scopes.get(occurrence)
    settlement = error if eligible and isinstance(error, SettledTransientIngestionFailure) else None
    fields = {
        "schema_name": "history_attempt_record",
        "schema_version": 1,
        "run_id": state.run_id,
        "execution_run_id": execution_run_id,
        "ingestion_plan_id": plan.ingestion_plan_id,
        "history_attempt_ordinal": ordinal,
        "ingestion_occurrence_id": occurrence,
        "memory_system_id": memory_system_id,
        "runtime_binding_hash": runtime_binding_hash,
        "retry_policy_hash": INFRASTRUCTURE_RETRY_POLICY_HASH,
        "max_retries_per_operation": maximum_retries,
        "history_input_hash": ingestion_payload_hash(
            tuple(s.payload_sha256 for s in plan.ordered_source_units)
        ),
        "scope_id": scope.scope_id if scope is not None else None,
        "scope_raw_refs": (
            tuple(
                dict.fromkeys(
                    (
                        scope.raw_reference.sha256,
                        *(r.sha256 for r in scope.supporting_raw_references),
                    )
                )
            )
            if scope is not None
            else ()
        ),
        "previous_retry_event_id": None,
        "admission_claim_raw_ref": None,
        "status": status,
        "operation_attempt_ids": tuple(record.attempt_id for record in operations),
        "terminal_failure_attempt_id": failure.attempt_id if failure is not None else None,
        "settlement_basis": settlement.settlement_basis if settlement else None,
        "settlement_status_code": settlement.status_code if settlement else None,
        "settlement_task_id": settlement.expected_task_id if settlement else None,
        "settlement_session_id": settlement.expected_session_id if settlement else None,
        "settlement_evidence_refs": tuple(
            dict.fromkeys(
                (
                    settlement.raw_reference.sha256,
                    *(r.sha256 for r in settlement.supporting_raw_references),
                )
            )
        )
        if settlement and settlement.raw_reference
        else (),
        "internal_retry_count": settlement.internal_retry_count if settlement else None,
        "failure_kind": settlement.failure_kind if settlement else None,
        "ingestion_plan_record_hash": canonical_sha256(ready_record)
        if ready_record is not None
        else None,
        "started_at": started_at,
        "ended_at": state.timestamp(),
    }
    record = HistoryAttemptRecord.model_validate(
        {"history_attempt_id": history_attempt_id(fields), **fields}
    )
    _seal(state, "history-attempts", record.history_attempt_id, record)
    state.history_records.append(record)
    return record


async def _execute_history(
    *,
    state: _NativeExecutionState,
    memory: MemorySystemPort,
    plan: IngestionPlan,
    case_plans: tuple[CasePlan, ...],
    memory_system_id: str,
    runtime_binding_hash: str,
    adapter_profile_id: str,
    setup_artifacts: tuple[_LiveDispatchArtifacts, ...] = (),
) -> tuple[tuple[NativeIngestionPlanRecord, ...], dict[str, ScopeReceipt]]:
    _raise_if_native_stop_requested(state)
    started_at = state.timestamp()
    occurrence = ingestion_occurrence_id(state.run_id, memory_system_id, plan.ingestion_plan_id)
    if occurrence in state.history_occurrence_bindings:
        raise ValueError("history execution occurrence was already admitted")
    state.history_occurrence_bindings[occurrence] = (plan.ingestion_plan_id, 1)
    if state.publish_history_progress is not None:
        state.publish_history_progress()
    try:
        records, scopes = await _execute_ingestion_plans_serial(
            state=state,
            memory=memory,
            plans=(plan,),
            case_plans=case_plans,
            memory_system_id=memory_system_id,
            runtime_binding_hash=runtime_binding_hash,
            adapter_profile_id=adapter_profile_id,
            setup_artifacts=setup_artifacts,
            history_attempt_ordinal=1,
            execution_run_id=state.run_id,
        )
    except BaseException as error:
        _seal_history_attempt(
            state=state,
            plan=plan,
            memory_system_id=memory_system_id,
            runtime_binding_hash=runtime_binding_hash,
            execution_run_id=state.run_id,
            ordinal=1,
            maximum_retries=0,
            started_at=started_at,
            error=error,
        )
        raise
    _seal_history_attempt(
        state=state,
        plan=plan,
        memory_system_id=memory_system_id,
        runtime_binding_hash=runtime_binding_hash,
        execution_run_id=state.run_id,
        ordinal=1,
        maximum_retries=0,
        started_at=started_at,
        ready_record=records[0],
    )
    return records, scopes


async def _execute_ingestion_batch(
    state: _NativeExecutionState,
    *,
    memory: MemorySystemPort,
    scope: ScopeReceipt,
    dispatch: IngestionDispatch,
    adapter_profile_id: str,
) -> tuple[IngestionDispatchReceipt, tuple[str, ...]]:
    maximum_attempts = (state.control.max_retries_per_operation + 1) if state.control else 3
    physical_attempt_ids: list[str] = []
    base_attempt = attempt_id(
        scope.ingestion_occurrence_id,
        "memory_ingest",
        dispatch.dispatch_ordinal_1_indexed,
        dispatch.request_fingerprint,
    )
    for batch_ordinal in range(1, maximum_attempts + 1):
        _raise_if_native_stop_requested(state)
        previous = physical_attempt_ids[-1] if physical_attempt_ids else None
        current = (
            base_attempt
            if batch_ordinal == 1
            else canonical_sha256(
                [
                    "oamb-native-batch-attempt-v1",
                    base_attempt,
                    batch_ordinal,
                    dispatch.request_fingerprint,
                ]
            )
        )
        prepared = _prepare_native_attempt(
            state,
            attempt_identity=current,
            parent_kind="ingestion_plan",
            parent_id=scope.ingestion_occurrence_id,
            stage="memory_ingest",
            ordinal=dispatch.dispatch_ordinal_1_indexed,
            request_fingerprint=dispatch.request_fingerprint,
            role_binding_id=adapter_profile_id,
            retry_of_attempt_id=previous,
        )
        physical_attempt_ids.append(current)
        started_at = state.timestamp()
        try:
            receipt = await memory.ingest(
                IngestionDispatchRequest(
                    scope=scope,
                    attempt_id=current,
                    dispatch=dispatch,
                    batch_attempt_ordinal=batch_ordinal,
                )
            )
            if receipt.attempt_id != current or receipt.dispatch != dispatch:
                raise ValueError("native ingestion receipt does not bind its dispatch")
        except BaseException as error:
            terminal = _seal_native_failure_preserving(
                state,
                prepared,
                started_at=started_at,
                error=error,
            )
            if terminal.raw_error_ref is not None and current not in state.attempt_accounting:
                _seal_native_attempt_accounting(
                    state,
                    prepared,
                    started_at=started_at,
                    ended_at=terminal.ended_at,
                    raw_response_ref=terminal.raw_error_ref,
                    usage_record_ids=tuple(getattr(error, "usage_reference_ids", ())),
                    indexing_view=IndexingView.ATTEMPTED,
                )
            if (
                not isinstance(error, SettledTransientIngestionFailure)
                or error.raw_reference is None
            ):
                raise
            if batch_ordinal < maximum_attempts:
                await _infrastructure_retry_sleep(10)
                continue
            return IngestionDispatchReceipt(
                attempt_id=current,
                dispatch=dispatch,
                accepted_source_unit_ids=(),
                rejected_source_unit_ids=(),
                skipped_source_unit_ids=tuple(
                    s.source_unit_id for s in dispatch.ordered_source_units
                ),
                raw_reference=error.raw_reference,
                raw_response_bytes=error.raw_response_bytes or b"",
                usage_records=(),
            ), tuple(physical_attempt_ids)
        ended_at = state.timestamp()
        _seal_native_success(
            state,
            prepared,
            started_at=started_at,
            ended_at=ended_at,
            raw_response_ref=receipt.raw_reference.sha256,
            index_contribution=IndexContribution.FINAL,
        )
        if state.control is None:
            for usage in receipt.usage_records:
                _seal(state, "usage", usage.usage_record_id, usage)
        _seal_native_attempt_accounting(
            state,
            prepared,
            started_at=started_at,
            ended_at=ended_at,
            raw_response_ref=receipt.raw_reference.sha256,
            usage_record_ids=tuple(usage.usage_record_id for usage in receipt.usage_records),
            inline_usage_records=tuple(receipt.usage_records),
            indexing_view=IndexingView.FINAL_CONTRIBUTION,
        )
        return receipt, tuple(physical_attempt_ids)
    raise AssertionError("ingestion batch exceeded its attempt bound")


async def _execute_ingestion_plans_serial(
    *,
    state: _NativeExecutionState,
    memory: MemorySystemPort,
    plans: tuple[IngestionPlan, ...],
    case_plans: tuple[CasePlan, ...],
    memory_system_id: str,
    runtime_binding_hash: str,
    adapter_profile_id: str,
    setup_artifacts: tuple[_LiveDispatchArtifacts, ...] = (),
    history_attempt_ordinal: int = 1,
    execution_run_id: str | None = None,
) -> tuple[tuple[NativeIngestionPlanRecord, ...], dict[str, ScopeReceipt]]:
    case_ids = {case.case_manifest_entry_id for case in case_plans}
    records: list[NativeIngestionPlanRecord] = []
    scopes: dict[str, ScopeReceipt] = {}
    for plan in plans:
        if not set(plan.ordered_case_manifest_entry_ids) <= case_ids:
            raise ValueError("native ingestion plan references an unknown case")
        occurrence_id = ingestion_occurrence_id(
            execution_run_id or state.run_id,
            memory_system_id,
            plan.ingestion_plan_id,
            history_attempt_ordinal=history_attempt_ordinal,
        )
        scope_request = ScopeAllocationRequest(
            ingestion_occurrence_id=occurrence_id,
            ingestion_plan_id=plan.ingestion_plan_id,
        )
        plan_artifacts: list[_LiveDispatchArtifacts] = list(
            setup_artifacts if plan is plans[0] else ()
        )
        if state.control is not None:

            async def allocate_scope(
                request: ScopeAllocationRequest = scope_request,
            ) -> ScopeReceipt:
                return await memory.allocate_ingestion_scope(request)

            scope_result, scope_artifacts = await _execute_live_memory_dispatch(
                state,
                parent_kind="ingestion_plan",
                parent_id=occurrence_id,
                stage="scope_allocate",
                ordinal=1,
                request_fingerprint=canonical_sha256(
                    ["oamb-live-scope-allocation-v1", occurrence_id, plan.ingestion_plan_id]
                ),
                role_binding_id=adapter_profile_id,
                call=allocate_scope,
                raw_reference=_scope_raw_reference,
            )
            scope = scope_result
            plan_artifacts.append(scope_artifacts)
        else:
            scope = await memory.allocate_ingestion_scope(scope_request)
        if scope.ingestion_occurrence_id != occurrence_id:
            raise ValueError("allocated history scope has a foreign occurrence")
        if any(existing.scope_id == scope.scope_id for existing in state.history_scopes.values()):
            raise ValueError("allocated history scope was already used by another attempt")
        state.history_scopes[occurrence_id] = scope
        scopes[plan.ingestion_plan_id] = scope
        dispatches = memory.plan_ingestion(
            IngestionRequest(scope=scope, ordered_source_units=plan.ordered_source_units)
        )
        expected_ordinals = tuple(range(1, len(dispatches) + 1))
        if tuple(item.dispatch_ordinal_1_indexed for item in dispatches) != expected_ordinals:
            raise ValueError("native ingestion dispatch ordinals are not contiguous")
        expected_sources = tuple(source.source_unit_id for source in plan.ordered_source_units)
        dispatched_sources = tuple(
            source.source_unit_id
            for dispatch in dispatches
            for source in dispatch.ordered_source_units
        )
        if dispatched_sources != expected_sources:
            raise ValueError("native ingestion dispatches changed source order")

        dispatch_receipts: list[IngestionDispatchReceipt] = []
        dispatch_attempt_ids: list[str] = []
        usage_record_ids: list[str] = []
        resource_record_ids: list[str] = []
        cost_record_ids: list[str] = []
        all_dispatch_attempt_ids: list[str] = []
        for dispatch in dispatches:
            receipt, physical_ids = await _execute_ingestion_batch(
                state,
                memory=memory,
                scope=scope,
                dispatch=dispatch,
                adapter_profile_id=adapter_profile_id,
            )
            all_dispatch_attempt_ids.extend(physical_ids)
            for physical_id in physical_ids:
                usage_ids, resource_id, cost_id = state.attempt_accounting[physical_id]
                usage_record_ids.extend(usage_ids)
                resource_record_ids.append(resource_id)
                cost_record_ids.append(cost_id)
            dispatch_attempt_ids.append(receipt.attempt_id)
            dispatch_receipts.append(receipt)

        _raise_if_native_stop_requested(state)
        ingestion_receipt = IngestionReceipt(
            ingestion_occurrence_id=occurrence_id,
            accepted_source_unit_ids=tuple(
                source_id
                for receipt in dispatch_receipts
                for source_id in receipt.accepted_source_unit_ids
            ),
            rejected_source_unit_ids=tuple(
                source_id
                for receipt in dispatch_receipts
                for source_id in receipt.rejected_source_unit_ids
            ),
            raw_references=tuple(receipt.raw_reference for receipt in dispatch_receipts),
            dispatch_receipts=tuple(dispatch_receipts),
            skipped_source_unit_ids=tuple(
                source_id
                for receipt in dispatch_receipts
                for source_id in receipt.skipped_source_unit_ids
            ),
        )
        readiness_request = ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=ingestion_receipt.accepted_source_unit_ids,
            ingestion_receipt=ingestion_receipt,
        )
        if state.control is not None:

            async def wait_for_readiness(
                request: ReadinessRequest = readiness_request,
            ) -> ReadinessReceipt:
                return await memory.wait_ready(request)

            async def project_ready_scope(
                requested_scope: ScopeReceipt = scope,
            ) -> ProjectionReceipt:
                return await memory.project(requested_scope)

            readiness_result, readiness_artifacts = await _execute_live_memory_dispatch(
                state,
                parent_kind="ingestion_plan",
                parent_id=occurrence_id,
                stage="memory_readiness",
                ordinal=1,
                request_fingerprint=canonical_sha256(
                    [
                        "oamb-live-memory-readiness-v1",
                        occurrence_id,
                        ingestion_receipt.accepted_source_unit_ids,
                    ]
                ),
                role_binding_id=adapter_profile_id,
                call=wait_for_readiness,
                raw_reference=_readiness_raw_reference,
            )
            readiness = readiness_result
            plan_artifacts.append(readiness_artifacts)
            _raise_if_native_stop_requested(state)
            projection_result, projection_artifacts = await _execute_live_memory_dispatch(
                state,
                parent_kind="ingestion_plan",
                parent_id=occurrence_id,
                stage="memory_projection",
                ordinal=1,
                request_fingerprint=canonical_sha256(
                    ["oamb-live-memory-projection-v1", occurrence_id]
                ),
                role_binding_id=adapter_profile_id,
                call=project_ready_scope,
                raw_reference=_projection_raw_reference,
            )
            projection = projection_result
            plan_artifacts.append(projection_artifacts)
        else:
            readiness = await memory.wait_ready(readiness_request)
            _raise_if_native_stop_requested(state)
            projection = await memory.project(scope)
        if not readiness.ready:
            raise ValueError("native fixture ingestion did not reach readiness")
        _require_projection_occurrence(projection, occurrence_id)
        if (
            adapter_profile_id != "mem0-rest-v1"
            and not ingestion_receipt.skipped_source_unit_ids
            and projection.inventory.ordered_source_unit_ids
            != ingestion_receipt.accepted_source_unit_ids
        ):
            raise ValueError("native ready projection differs from accepted source order")
        case_occurrence_ids = tuple(
            case_occurrence_id(occurrence_id, case_id)
            for case_id in plan.ordered_case_manifest_entry_ids
        )
        record_values = dict(
            ingestion_occurrence_id=occurrence_id,
            run_id=state.run_id,
            memory_system_id=memory_system_id,
            adapter_profile_id=adapter_profile_id,
            runtime_binding_hash=runtime_binding_hash,
            ingestion_plan_id=plan.ingestion_plan_id,
            ordered_member_context_manifest_entry_ids=(
                plan.ordered_member_context_manifest_entry_ids
            ),
            ordered_case_occurrence_ids=case_occurrence_ids,
            state=IngestionPlanState.SEALED,
            scope_id=scope.scope_id,
            scope_raw_refs=(
                scope.raw_reference.sha256,
                *(item.sha256 for item in scope.supporting_raw_references),
            ),
            ordered_source_unit_ids=expected_sources,
            ordered_dispatch_attempt_ids=tuple(dispatch_attempt_ids),
            ordered_dispatch_source_unit_ids=tuple(
                tuple(source.source_unit_id for source in receipt.dispatch.ordered_source_units)
                for receipt in dispatch_receipts
            ),
            accepted_source_unit_ids=ingestion_receipt.accepted_source_unit_ids,
            rejected_source_unit_ids=ingestion_receipt.rejected_source_unit_ids,
            skipped_source_unit_ids=ingestion_receipt.skipped_source_unit_ids,
            readiness_evidence_refs=tuple(
                dict.fromkeys(item.sha256 for item in readiness.evidence_references)
            ),
            inventory_raw_ref=projection.inventory.raw_reference.sha256,
            projected_source_unit_ids=projection.inventory.ordered_source_unit_ids,
            projection_raw_refs=_projection_raw_refs(projection),
            protected_state_sha256=projection.state_digest.state_sha256,
            attempt_ids=(
                *(item.attempt_id for item in plan_artifacts),
                *all_dispatch_attempt_ids,
            ),
            usage_record_ids=(
                *(usage_id for item in plan_artifacts for usage_id in item.usage_record_ids),
                *usage_record_ids,
            ),
            resource_record_ids=(
                *(item.resource_record_id for item in plan_artifacts),
                *resource_record_ids,
            ),
            cost_record_ids=(
                *(item.cost_record_id for item in plan_artifacts),
                *cost_record_ids,
            ),
        )
        record: NativeIngestionPlanRecord
        if adapter_profile_id == "mem0-rest-v1":
            record = IngestionPlanRecordV3.model_validate(
                {
                    **record_values,
                    "projection_semantics": "retrieval_visible_subset",
                }
            )
        else:
            record = IngestionPlanRecordV2.model_validate(record_values)
        _seal(state, "ingestion-plans", occurrence_id, record)
        records.append(record)
    return tuple(records), scopes


def _raise_if_native_stop_requested(state: object) -> None:
    if _native_stop_requested(state):
        raise NativeRunInterrupted("native run stopped before admitting another provider operation")


@dataclass(slots=True)
class _LiveAttemptedQueryMemory:
    state: _NativeExecutionState
    memory: MemorySystemPort
    scope: ScopeReceipt
    case_occurrence_id: str
    adapter_profile_id: str
    projection_count: int = 0
    artifacts: list[_LiveDispatchArtifacts] | None = None

    def __post_init__(self) -> None:
        self.artifacts = []

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        if scope != self.scope:
            raise ValueError("live attempted projection scope differs from the case scope")
        self.projection_count += 1
        stage = "pre_query_projection" if self.projection_count == 1 else "post_query_projection"
        result, artifacts = await _execute_live_memory_dispatch(
            self.state,
            parent_kind="case",
            parent_id=self.case_occurrence_id,
            stage=stage,
            ordinal=1,
            request_fingerprint=canonical_sha256(
                ["oamb-live-query-projection-v1", self.case_occurrence_id, stage]
            ),
            role_binding_id=self.adapter_profile_id,
            call=lambda: self.memory.project(scope),
            raw_reference=_projection_raw_reference,
        )
        assert self.artifacts is not None
        self.artifacts.append(artifacts)
        return result

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        if request.scope != self.scope:
            raise ValueError("live attempted retrieval scope differs from the case scope")
        if request.case_occurrence_id != self.case_occurrence_id:
            raise ValueError("live attempted retrieval case identity differs from its owner")
        result, artifacts = await _execute_live_memory_dispatch(
            self.state,
            parent_kind="case",
            parent_id=self.case_occurrence_id,
            stage="memory_query",
            ordinal=1,
            request_fingerprint=canonical_sha256(
                [
                    "oamb-live-memory-query-v1",
                    self.case_occurrence_id,
                    hashlib.sha256(request.query_bytes).hexdigest(),
                ]
            ),
            role_binding_id=self.adapter_profile_id,
            call=lambda: self.memory.retrieve(request),
            raw_reference=_scope_raw_reference,
        )
        assert self.artifacts is not None
        self.artifacts.append(artifacts)
        return result


@dataclass(frozen=True, slots=True)
class _ModelStageResult:
    receipt: ModelReceipt | None
    error: BaseException | None
    attempt_ids: tuple[str, ...]
    usage_record_ids: tuple[str, ...]
    resource_record_ids: tuple[str, ...]
    cost_record_ids: tuple[str, ...]


def _validate_assistant_output(receipt: ModelReceipt) -> None:
    if receipt.finish_disposition == FinishDisposition.CONTENT_FILTERED:
        raise RuntimeError("model output was content filtered")
    if receipt.finish_disposition != FinishDisposition.NORMAL_STOP:
        raise ValueError(f"assistant output must finish normally, got {receipt.finish_disposition}")
    if not receipt.output_text.strip():
        raise ValueError("assistant output must contain nonblank text")
    if receipt.candidates and (
        len(receipt.candidates) != 1
        or receipt.candidates[0].tool_call_present
        or not receipt.candidates[0].complete
    ):
        raise ValueError("assistant output must contain one complete text candidate")


async def _execute_model_stage(
    state: _NativeExecutionState,
    *,
    model: ModelClientPort,
    request: ModelRequest,
    maximum_output_tokens: int,
    validate: Callable[[ModelReceipt], None] = _validate_assistant_output,
) -> _ModelStageResult:
    if request.parent_kind != "case" or request.stage not in {"answer", "judge"}:
        raise ValueError("layered model completion requires an answer or judge case request")
    prepared_calls: dict[int, tuple[_PreparedNativeAttempt, datetime]] = {}
    last_receipt: ModelReceipt | None = None
    output_error: BaseException | None = None

    async def dispatch(ordinal: int, messages: tuple[tuple[str, str], ...]) -> ModelReceipt:
        _raise_if_native_stop_requested(state)
        messages_hash = _seal_raw(
            state.store, canonical_json_bytes(messages), media_type="application/json"
        )
        outgoing = replace(request, messages=messages, messages_sha256=messages_hash)
        outgoing = replace(
            outgoing,
            attempt_id=attempt_id(
                request.parent_id,
                request.stage,
                ordinal,
                outgoing.request_fingerprint,
            ),
        )
        previous = prepared_calls[ordinal - 1][0].attempt_id if ordinal > 1 else None
        prepared = _prepare_native_attempt(
            state,
            attempt_identity=outgoing.attempt_id,
            parent_kind="case",
            parent_id=request.parent_id,
            stage=request.stage,
            ordinal=ordinal,
            request_fingerprint=outgoing.request_fingerprint,
            role_binding_id=request.role_binding_id,
            request_messages_sha256=messages_hash,
            maximum_output_tokens=maximum_output_tokens,
            retry_of_attempt_id=previous,
        )
        prepared_calls[ordinal] = prepared, state.timestamp()
        return await model.complete(outgoing)

    async def record(
        ordinal: int, receipt: ModelReceipt | None, error: BaseException | None
    ) -> None:
        nonlocal last_receipt, output_error
        if ordinal not in prepared_calls:
            assert error is not None
            raise error
        prepared, started_at = prepared_calls[ordinal]
        last_receipt = receipt
        if error is None:
            assert receipt is not None
            ended_at = state.timestamp()
            _seal_native_success(
                state,
                prepared,
                started_at=started_at,
                ended_at=ended_at,
                raw_response_ref=receipt.raw_reference.sha256,
                index_contribution=IndexContribution.NOT_APPLICABLE,
            )
            _seal_native_attempt_accounting(
                state,
                prepared,
                started_at=started_at,
                ended_at=ended_at,
                raw_response_ref=receipt.raw_reference.sha256,
                usage_record_ids=receipt.usage_reference_ids,
                indexing_view=IndexingView.NOT_APPLICABLE,
            )
            return
        recorded_error = error
        if receipt is not None:
            output_error = error
            recorded_error = ModelCallFailure(
                str(error),
                raw_reference=receipt.raw_reference,
                raw_response_bytes=receipt.raw_response_bytes,
                usage_reference_ids=receipt.usage_reference_ids,
                retryable=isinstance(error, ValueError),
                failure_kind="output_contract_error",
                supplier_status_code=receipt.supplier_status_code,
            )
        terminal = _seal_native_failure_preserving(
            state,
            prepared,
            started_at=started_at,
            error=recorded_error,
        )
        if (
            terminal.raw_error_ref is not None
            and prepared.attempt_id not in state.attempt_accounting
        ):
            _seal_native_attempt_accounting(
                state,
                prepared,
                started_at=started_at,
                ended_at=terminal.ended_at,
                raw_response_ref=terminal.raw_error_ref,
                usage_record_ids=tuple(getattr(recorded_error, "usage_reference_ids", ())),
                indexing_view=IndexingView.NOT_APPLICABLE,
            )

    failure: BaseException | None = None
    try:
        last_receipt = await execute_model_completion_retry(
            messages=request.messages,
            dispatch=dispatch,
            validate=validate,
            record=record,
            max_outer_attempts=state.control.model_max_attempts if state.control else 6,
            max_transport_retries=state.control.model_transport_max_retries if state.control else 2,
            sleep=_infrastructure_retry_sleep,
        )
    except BaseException as error:
        if isinstance(error, (asyncio.CancelledError, NativeRunInterrupted)):
            raise
        if error is not output_error and not isinstance(
            error, (ModelCallFailure, ModelCallUnknownOutcome)
        ):
            raise
        if isinstance(error, ModelCallFailure) and error.failure_kind == "usage_parse_error":
            raise
        failure = error
    attempt_ids = tuple(prepared.attempt_id for prepared, _started in prepared_calls.values())
    accounting = tuple(
        state.attempt_accounting[identity]
        for identity in attempt_ids
        if identity in state.attempt_accounting
    )
    return _ModelStageResult(
        receipt=last_receipt,
        error=failure,
        attempt_ids=attempt_ids,
        usage_record_ids=tuple(
            identity for usage, _resource, _cost in accounting for identity in usage
        ),
        resource_record_ids=tuple(resource for _usage, resource, _cost in accounting),
        cost_record_ids=tuple(cost for _usage, _resource, cost in accounting),
    )


def _seal_model_error_case(
    state: _NativeExecutionState,
    *,
    common: dict[str, Any],
    stage: str,
    attempt_ids: tuple[str, ...],
    usage_record_ids: tuple[str, ...],
    resource_record_ids: tuple[str, ...],
    cost_record_ids: tuple[str, ...],
    answer_receipt: ModelReceipt | None = None,
    answer_value: AnswerValue | None = None,
    judge_prompt_raw_ref: str | None = None,
) -> CaseRecordV3:
    record = CaseRecordV3(
        **common,
        state=CaseState.ERROR,
        judge_prompt_raw_ref=judge_prompt_raw_ref,
        answer_raw_ref=answer_receipt.raw_reference.sha256 if answer_receipt else None,
        parsed_answer_sha256=answer_value.parsed_value_sha256 if answer_value else None,
        metric_id=None,
        metric_numerator=None,
        metric_denominator=None,
        evaluation_raw_ref=None,
        evaluation_disposition=CaseEvaluationDisposition.UNJUDGED
        if stage == "judge"
        else CaseEvaluationDisposition.NOT_RUN,
        attempt_ids=attempt_ids,
        usage_record_ids=usage_record_ids,
        resource_record_ids=resource_record_ids,
        cost_record_ids=cost_record_ids,
        error_stage=stage,
    )
    _seal(state, "cases", record.case_occurrence_id, record)
    return record


async def _execute_cases_serial(
    *,
    state: _NativeExecutionState,
    workload: WorkloadPort,
    memory: MemorySystemPort,
    answer_model: ModelClientPort,
    judge_model: ModelClientPort | None,
    plans: tuple[IngestionPlan, ...],
    case_plans: tuple[CasePlan, ...],
    scopes: dict[str, ScopeReceipt],
    memory_system_id: str,
    adapter_profile_id: str,
    visible_evidence_policy: VisibleEvidencePolicy,
    answer_role_binding_id: str,
    judge_role_binding_id: str | None,
) -> tuple[CaseRecordV3, ...]:
    plan_by_case: dict[str, IngestionPlan] = {}
    for plan in plans:
        for case_id in plan.ordered_case_manifest_entry_ids:
            if case_id in plan_by_case:
                raise ValueError("native case belongs to multiple ingestion plans")
            plan_by_case[case_id] = plan
    records: list[CaseRecordV3] = []
    for case_plan in case_plans:
        selected_plan = plan_by_case.get(case_plan.case_manifest_entry_id)
        if selected_plan is None:
            raise ValueError("native case has no ingestion plan")
        scope = scopes[selected_plan.ingestion_plan_id]
        ingestion_occurrence = scope.ingestion_occurrence_id
        case_occurrence = case_occurrence_id(
            ingestion_occurrence,
            case_plan.case_manifest_entry_id,
        )
        state.case_occurrence_bindings[case_plan.case_manifest_entry_id] = case_occurrence
        if state.publish_history_progress is not None:
            state.publish_history_progress()
        query = workload.render_retrieval_query(case_plan)
        query_fingerprint = canonical_sha256(
            ["oamb-native-query-v1", case_occurrence, hashlib.sha256(query).hexdigest()]
        )
        retrieval_request = _retrieval_request(scope, case_occurrence, case_plan, query)
        if state.control is not None:
            attempted_memory = _LiveAttemptedQueryMemory(
                state,
                memory,
                scope,
                case_occurrence,
                adapter_profile_id,
            )
            query_receipt = await execute_read_only_retrieval(
                memory=attempted_memory,
                request=retrieval_request,
                clock=state.monotonic,
            )
            query_artifacts = tuple(attempted_memory.artifacts or ())
            if len(query_artifacts) != 3:
                raise ValueError("live query did not seal projection/query/projection attempts")
            query_attempt_ids = tuple(item.attempt_id for item in query_artifacts)
            query_usage_ids = tuple(
                usage_id for item in query_artifacts for usage_id in item.usage_record_ids
            )
            query_resource_ids = tuple(item.resource_record_id for item in query_artifacts)
            query_cost_ids = tuple(item.cost_record_id for item in query_artifacts)
        else:
            query_attempt_id = attempt_id(
                case_occurrence,
                "memory_query",
                1,
                query_fingerprint,
            )
            prepared_query = _prepare_native_attempt(
                state,
                attempt_identity=query_attempt_id,
                parent_kind="case",
                parent_id=case_occurrence,
                stage="memory_query",
                ordinal=1,
                request_fingerprint=query_fingerprint,
                role_binding_id=adapter_profile_id,
            )
            query_started_at = state.timestamp()
            try:
                query_receipt = await execute_read_only_retrieval(
                    memory=memory,
                    request=retrieval_request,
                    clock=state.monotonic,
                )
            except BaseException as exc:
                _seal_native_failure_preserving(
                    state,
                    prepared_query,
                    started_at=query_started_at,
                    error=exc,
                )
                raise
            query_ended_at = state.timestamp()
            _seal_native_success(
                state,
                prepared_query,
                started_at=query_started_at,
                ended_at=query_ended_at,
                raw_response_ref=query_receipt.native_batch.raw_reference.sha256,
                index_contribution=IndexContribution.NOT_APPLICABLE,
            )
            query_usage_ids, query_resource_id, query_cost_id = _seal_native_attempt_accounting(
                state,
                prepared_query,
                started_at=query_started_at,
                ended_at=query_ended_at,
                raw_response_ref=query_receipt.native_batch.raw_reference.sha256,
                usage_record_ids=(),
                indexing_view=IndexingView.NOT_APPLICABLE,
            )
            query_attempt_ids = (query_attempt_id,)
            query_resource_ids = (query_resource_id,)
            query_cost_ids = (query_cost_id,)

        visible = workload.build_visible_evidence(
            query_receipt.native_batch,
            visible_evidence_policy,
        )
        if hashlib.sha256(visible.canonical_bytes).hexdigest() != visible.sha256:
            raise ValueError("workload visible-evidence hash does not match its bytes")
        visible_raw_ref = _seal_raw(
            state.store,
            visible.canonical_bytes,
            media_type="application/vnd.oamb.visible-evidence",
        )
        decision_ledger = canonical_json_bytes(
            {
                "operation": "visible_evidence_decision",
                "case_occurrence_id": case_occurrence,
                "policy": asdict(visible_evidence_policy),
                "included_native_ids": visible.included_native_ids,
                "visible_evidence_sha256": visible.sha256,
                "visible_evidence_byte_count": len(visible.canonical_bytes),
                "visible_evidence_token_count": visible.token_count,
                "visible_evidence_tokenizer_fingerprint": visible.tokenizer_fingerprint,
                "candidate_count": visible.candidate_count,
                "kept_count": visible.kept_count,
                "dropped_count": visible.dropped_count,
                "truncated_count": visible.truncated_count,
                "first_exceeded_limit": visible.first_exceeded_limit,
                "decisions": tuple(asdict(decision) for decision in visible.decisions),
            }
        )
        decision_raw_ref = _seal_raw(
            state.store,
            decision_ledger,
            media_type="application/json",
        )
        prompt = workload.render_answer(case_plan, visible)
        if hashlib.sha256(prompt.canonical_bytes).hexdigest() != prompt.sha256:
            raise ValueError("workload prompt hash does not match its bytes")
        prompt_raw_ref = _seal_raw(
            state.store,
            prompt.canonical_bytes,
            media_type="text/plain; charset=utf-8",
        )
        messages = (("user", prompt.canonical_bytes.decode("utf-8", errors="strict")),)
        messages_sha256 = canonical_sha256(messages)
        common_case_values: dict[str, Any] = dict(
            case_occurrence_id=case_occurrence,
            run_id=state.run_id,
            ingestion_occurrence_id=ingestion_occurrence,
            case_manifest_entry_id=case_plan.case_manifest_entry_id,
            adapter_profile_id=adapter_profile_id,
            retrieval_raw_ref=query_receipt.native_batch.raw_reference.sha256,
            retrieval_supporting_raw_refs=tuple(
                item.sha256 for item in query_receipt.native_batch.supporting_raw_references
            ),
            retrieval_request_raw_ref=(
                None
                if query_receipt.native_batch.request_raw_reference is None
                else query_receipt.native_batch.request_raw_reference.sha256
            ),
            ordered_native_candidate_ids=tuple(
                candidate.native_id for candidate in query_receipt.native_batch.candidates
            ),
            ordered_native_content_sha256=tuple(
                hashlib.sha256(candidate.content.encode("utf-8")).hexdigest()
                for candidate in query_receipt.native_batch.candidates
            ),
            native_candidate_source_unit_ids=tuple(
                candidate.source_unit_id for candidate in query_receipt.native_batch.candidates
            ),
            visible_evidence_raw_ref=visible_raw_ref,
            visible_evidence_sha256=visible.sha256,
            visible_evidence_byte_count=len(visible.canonical_bytes),
            visible_evidence_token_count=visible.token_count,
            visible_evidence_tokenizer_fingerprint=visible.tokenizer_fingerprint,
            native_candidate_count=visible.candidate_count,
            visible_kept_count=visible.kept_count,
            visible_dropped_count=visible.dropped_count,
            visible_truncated_count=visible.truncated_count,
            visible_decision_ledger_raw_ref=decision_raw_ref,
            pre_query_projection_raw_refs=_projection_raw_refs(query_receipt.before_projection),
            pre_query_state_sha256=(query_receipt.before_projection.state_digest.state_sha256),
            post_query_projection_raw_refs=_projection_raw_refs(query_receipt.after_projection),
            post_query_state_sha256=(query_receipt.after_projection.state_digest.state_sha256),
            query_mutation_status="unchanged",
            prompt_raw_ref=prompt_raw_ref,
            prompt_sha256=prompt.sha256,
        )
        answer_request = ModelRequest.for_attempt(
            ordinal=1,
            parent_kind="case",
            parent_id=case_occurrence,
            stage="answer",
            role_binding_id=answer_role_binding_id,
            messages_sha256=messages_sha256,
            messages=messages,
            thinking_effort=answer_model.thinking_effort_for(
                stage="answer",
                role_binding_id=answer_role_binding_id,
            ),
            output_contract_id=case_plan.output_contract_id,
            max_output_tokens=None,
        )
        answer_result = await _execute_model_stage(
            state,
            model=answer_model,
            request=answer_request,
            maximum_output_tokens=case_plan.answer_max_output_tokens,
        )
        if answer_result.error is not None:
            records.append(
                _seal_model_error_case(
                    state,
                    common=common_case_values,
                    stage="answer",
                    attempt_ids=(*query_attempt_ids, *answer_result.attempt_ids),
                    usage_record_ids=(*query_usage_ids, *answer_result.usage_record_ids),
                    resource_record_ids=(*query_resource_ids, *answer_result.resource_record_ids),
                    cost_record_ids=(*query_cost_ids, *answer_result.cost_record_ids),
                )
            )
            continue
        answer_receipt = answer_result.receipt
        assert answer_receipt is not None
        parsed_answer = answer_receipt.output_text.encode("utf-8")
        answer_value = AnswerValue(
            raw_reference=answer_receipt.raw_reference,
            raw_answer=parsed_answer,
            parsed_value=parsed_answer,
            parsed_value_sha256=hashlib.sha256(parsed_answer).hexdigest(),
        )
        evaluation = workload.evaluate(case_plan, answer_value)
        case_attempt_ids = [*query_attempt_ids, *answer_result.attempt_ids]
        usage_record_ids = [*query_usage_ids, *answer_result.usage_record_ids]
        resource_record_ids = [*query_resource_ids, *answer_result.resource_record_ids]
        cost_record_ids = [*query_cost_ids, *answer_result.cost_record_ids]
        evaluation_disposition = CaseEvaluationDisposition.DETERMINISTIC_EVALUATED
        judge_prompt_raw_ref: str | None = None
        if isinstance(evaluation, JudgeRequest):
            if (
                case_plan.judge_binding_id is None
                or judge_model is None
                or judge_role_binding_id is None
            ):
                raise ValueError("native judge request requires a judge role binding")
            if case_plan.judge_binding_id != judge_role_binding_id:
                raise ValueError("native case judge binding differs from the selected judge role")
            judge_prompt = evaluation.prompt
            if hashlib.sha256(judge_prompt.canonical_bytes).hexdigest() != judge_prompt.sha256:
                raise ValueError("workload judge prompt hash does not match its bytes")
            judge_prompt_raw_ref = _seal_raw(
                state.store,
                judge_prompt.canonical_bytes,
                media_type="text/plain; charset=utf-8",
            )
            judge_messages = (
                ("user", judge_prompt.canonical_bytes.decode("utf-8", errors="strict")),
            )
            judge_messages_sha256 = canonical_sha256(judge_messages)
            judge_request = ModelRequest.for_attempt(
                ordinal=1,
                parent_kind="case",
                parent_id=case_occurrence,
                stage="judge",
                role_binding_id=judge_role_binding_id,
                messages_sha256=judge_messages_sha256,
                messages=judge_messages,
                thinking_effort=judge_model.thinking_effort_for(
                    stage="judge",
                    role_binding_id=judge_role_binding_id,
                ),
                output_contract_id=evaluation.output_contract_id,
                max_output_tokens=None,
            )
            judge_evaluations: list[DeterministicEvaluation] = []

            def validate_judge(
                receipt: ModelReceipt,
                *,
                bound_case: CasePlan = case_plan,
                bound_answer: AnswerValue = answer_value,
                results: list[DeterministicEvaluation] = judge_evaluations,
            ) -> None:
                _validate_assistant_output(receipt)
                raw_answer = receipt.output_text.encode("utf-8")
                value = AnswerValue(
                    raw_reference=receipt.raw_reference,
                    raw_answer=raw_answer,
                    parsed_value=raw_answer,
                    parsed_value_sha256=hashlib.sha256(raw_answer).hexdigest(),
                )
                results.append(workload.finalize_judge(bound_case, bound_answer, value))

            judge_result = await _execute_model_stage(
                state,
                model=judge_model,
                request=judge_request,
                maximum_output_tokens=evaluation.max_output_tokens,
                validate=validate_judge,
            )
            case_attempt_ids.extend(judge_result.attempt_ids)
            usage_record_ids.extend(judge_result.usage_record_ids)
            resource_record_ids.extend(judge_result.resource_record_ids)
            cost_record_ids.extend(judge_result.cost_record_ids)
            if judge_result.error is not None:
                records.append(
                    _seal_model_error_case(
                        state,
                        common=common_case_values,
                        stage="judge",
                        attempt_ids=tuple(case_attempt_ids),
                        usage_record_ids=tuple(usage_record_ids),
                        resource_record_ids=tuple(resource_record_ids),
                        cost_record_ids=tuple(cost_record_ids),
                        answer_receipt=answer_receipt,
                        answer_value=answer_value,
                        judge_prompt_raw_ref=judge_prompt_raw_ref,
                    )
                )
                continue
            evaluation = judge_evaluations[-1]
            evaluation_disposition = CaseEvaluationDisposition.JUDGED
        evaluation_raw_ref = _seal_deterministic_evaluation(
            state.store,
            evaluation=evaluation,
            parsed_answer_sha256=answer_value.parsed_value_sha256,
        )
        if evaluation.numerator is None or evaluation.denominator is None:
            raise ValueError("native deterministic evaluation requires an exact fraction")
        record = CaseRecordV3(
            **common_case_values,
            state=CaseState.COMPLETED,
            judge_prompt_raw_ref=judge_prompt_raw_ref,
            answer_raw_ref=answer_receipt.raw_reference.sha256,
            parsed_answer_sha256=answer_value.parsed_value_sha256,
            metric_id=evaluation.metric_id,
            metric_numerator=evaluation.numerator,
            metric_denominator=evaluation.denominator,
            evaluation_raw_ref=evaluation_raw_ref,
            evaluation_disposition=evaluation_disposition,
            attempt_ids=tuple(case_attempt_ids),
            usage_record_ids=tuple(dict.fromkeys(usage_record_ids)),
            resource_record_ids=tuple(resource_record_ids),
            cost_record_ids=tuple(cost_record_ids),
            error_stage=None,
        )
        _seal(state, "cases", case_occurrence, record)
        records.append(record)
    return tuple(records)


def _retrieval_request(
    scope: ScopeReceipt,
    case_occurrence: str,
    case_plan: CasePlan,
    query: bytes,
) -> RetrievalRequest:
    return RetrievalRequest(
        scope=scope,
        case_occurrence_id=case_occurrence,
        query_bytes=query,
        top_k=NATIVE_RETRIEVAL_TOP_K,
        query_timestamp=case_plan.query_timestamp,
    )


def _projection_raw_refs(projection: ProjectionReceipt) -> tuple[str, ...]:
    return (
        projection.inventory.raw_reference.sha256,
        projection.state_digest.raw_reference.sha256,
        *(item.sha256 for item in projection.supporting_raw_references),
    )


def _require_projection_occurrence(
    projection: ProjectionReceipt,
    ingestion_occurrence: str,
) -> None:
    if (
        projection.inventory.ingestion_occurrence_id != ingestion_occurrence
        or projection.state_digest.ingestion_occurrence_id != ingestion_occurrence
    ):
        raise ValueError("native projection belongs to a different ingestion occurrence")


def _seal_deterministic_evaluation(
    store: ArtifactStorePort,
    *,
    evaluation: DeterministicEvaluation,
    parsed_answer_sha256: str,
) -> str:
    trace = evaluation.trace_bytes or canonical_json_bytes(
        {
            "metric_id": evaluation.metric_id,
            "numerator": evaluation.numerator,
            "denominator": evaluation.denominator,
            "parsed_answer_sha256": parsed_answer_sha256,
        }
    )
    payload = canonical_json_bytes(
        {
            "operation": "deterministic_evaluation",
            "metric_id": evaluation.metric_id,
            "result_sha256": evaluation.result_sha256,
            "numerator": evaluation.numerator,
            "denominator": evaluation.denominator,
            "parsed_answer_sha256": parsed_answer_sha256,
            "trace_hex": trace.hex(),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
        }
    )
    return _seal_raw(store, payload, media_type="application/json")


def _require_live_manifest_closure(
    control: NativeRunControl,
    dataset_manifest: object,
    case_manifest: object,
) -> None:
    dataset_hash = getattr(dataset_manifest, "manifest_hash", None)
    case_hash = getattr(case_manifest, "manifest_hash", None)
    workload_id = getattr(case_manifest, "workload_id", None)
    if (
        dataset_hash != control.run_spec.dataset_manifest_hash
        or case_hash != control.run_spec.case_manifest_hash
        or workload_id != control.run_spec.workload_id
        or dataset_hash != control.preflight_record.dataset_manifest_hash
        or case_hash != control.preflight_record.subset_manifest_hash
    ):
        raise ValueError("live native workload manifests differ from the durable control")


def _seal_live_control(state: _NativeExecutionState, control: NativeRunControl) -> None:
    records: tuple[tuple[str, str, StrictContract], ...] = (
        ("source/specs/run-spec.json", control.run_spec.run_id, control.run_spec),
        (
            "source/specs/run-preflight.json",
            control.preflight_record.preflight_record_hash,
            control.preflight_record,
        ),
        ("source/specs/budget.json", control.budget.budget_id, control.budget),
    )
    for relative_path, record_id, record in records:
        seal_source_contract(
            state.store,
            relative_path=relative_path,
            record_id=record_id,
            record=record,
        )
    for binding in control.role_bindings:
        seal_source_contract(
            state.store,
            relative_path=f"source/model-role-bindings/{binding.binding_id}.json",
            record_id=binding.binding_id,
            record=binding,
        )


def _seal_raw(store: ArtifactStorePort, payload: bytes, *, media_type: str) -> str:
    raw_sha256 = hashlib.sha256(payload).hexdigest()
    return store.seal_raw(
        RawPayloadSealRequest(
            sha256=raw_sha256,
            media_type=media_type,
            compression="gzip",
            payload_bytes=payload,
        )
    ).sha256


def _prepare_native_attempt(
    state: _NativeExecutionState,
    *,
    attempt_identity: str,
    parent_kind: Literal["ingestion_plan", "case"],
    parent_id: str,
    stage: str,
    ordinal: int,
    request_fingerprint: str,
    role_binding_id: str,
    request_messages_sha256: str | None = None,
    maximum_output_tokens: int = 0,
    retry_of_attempt_id: str | None = None,
) -> _PreparedNativeAttempt:
    if state.control is not None:
        return _prepare_live_native_attempt(
            state,
            attempt_identity=attempt_identity,
            parent_kind=parent_kind,
            parent_id=parent_id,
            stage=stage,
            ordinal=ordinal,
            request_fingerprint=request_fingerprint,
            role_binding_id=role_binding_id,
            request_messages_sha256=request_messages_sha256,
            maximum_output_tokens=maximum_output_tokens,
            retry_of_attempt_id=retry_of_attempt_id,
        )
    claimed_at = state.timestamp()
    claim_fields = {
        "occurrence_id": parent_id,
        "lease_record_hash": state.lease_record_hash,
        "lease_epoch": state.lease_epoch,
        "owner_id": state.owner_id,
        "stage": stage,
        "request_fingerprint": request_fingerprint,
        "reconciliation_capability": "none",
        "claimed_at": claimed_at,
    }
    claim_id = canonical_sha256(["oamb-native-occurrence-claim-v1", claim_fields])
    claim = OccurrenceClaimRecord.model_validate({"claim_id": claim_id, **claim_fields})
    reserved_at = state.timestamp()
    reservation_fields = {
        "budget_id": state.budget_id,
        "scope_kind": BudgetScopeKindV2.RUN,
        "scope_id": state.budget_scope_id or parent_id,
        "role_binding_id": role_binding_id,
        "attempt_id": attempt_identity,
        "reserved_attempts": 1,
        "reserved_input_tokens": 0,
        "reserved_output_tokens": maximum_output_tokens,
        "reserved_dispatch_wall_seconds": Decimal("0"),
        "reserved_cost": None,
        "currency": None,
        "reserved_resource_ceilings": (),
        "reserved_provider_units": Decimal("1"),
        "reserved_at": reserved_at,
    }
    reservation_id = canonical_sha256(["oamb-native-budget-reservation-v1", reservation_fields])
    reservation = BudgetReservationRecord.model_validate(
        {"reservation_id": reservation_id, **reservation_fields}
    )
    intent = AttemptIntentRecord(
        attempt_id=attempt_identity,
        claim_id=claim_id,
        reservation_id=reservation_id,
        parent_kind=parent_kind,
        parent_id=parent_id,
        role_binding_id=role_binding_id,
        stage=stage,
        request_fingerprint=request_fingerprint,
        reconciliation_capability="none",
        idempotency_key_hash=None,
        sealed_at=state.timestamp(),
    )
    _seal(state, "occurrence-claims", claim_id, claim)
    _seal(state, "budget-reservations", reservation_id, reservation)
    _seal(state, "attempt-intents", attempt_identity, intent)
    return _PreparedNativeAttempt(
        attempt_id=attempt_identity,
        parent_kind=parent_kind,
        parent_id=parent_id,
        stage=stage,
        ordinal=ordinal,
        request_fingerprint=request_fingerprint,
        request_messages_sha256=request_messages_sha256,
        role_binding_id=role_binding_id,
        retry_of_attempt_id=retry_of_attempt_id,
    )


def _reservation_allocations_for_route(
    budget: BudgetSpecV4,
    route: DispatchBudgetRoute,
    *,
    maximum_output_tokens: int,
) -> tuple[
    tuple[EvidenceBudgetOwnerAllocation, ...],
    tuple[RuntimeBudgetOwnerAllocation, ...],
    BudgetAmount,
]:
    role_ceilings = {item.role_binding_id: item for item in budget.role_ceilings}
    operation_ceilings = {
        item.provider_operation_ceiling_id: item for item in budget.provider_operation_ceilings
    }
    descriptors: list[
        tuple[
            DispatchBudgetOwnerKind,
            str,
            str,
            str,
            str,
            bool,
            bool,
        ]
    ] = []
    if route.dispatch_owner_kind == DispatchBudgetOwnerKind.PROVIDER_OPERATION:
        operation_ceiling = operation_ceilings.get(route.provider_operation_ceiling_id or "")
        if operation_ceiling is None:
            raise BudgetExceededError("dispatch route has no provider-operation ceiling")
        descriptors.append(
            (
                DispatchBudgetOwnerKind.PROVIDER_OPERATION,
                operation_ceiling.provider_operation_ceiling_id,
                operation_ceiling.adapter_profile_id,
                operation_ceiling.operation_kind,
                operation_ceiling.billing_unit,
                True,
                False,
            )
        )
        for owner_id in route.internal_usage_role_binding_ids:
            role = role_ceilings.get(owner_id)
            if role is None:
                raise BudgetExceededError("dispatch route has no internal-role ceiling")
            descriptors.append(
                (
                    DispatchBudgetOwnerKind.MODEL_ROLE,
                    owner_id,
                    role.provider_budget_cap.provider,
                    role.provider_budget_cap.operation_kind,
                    role.provider_budget_cap.billing_unit,
                    False,
                    True,
                )
            )
    else:
        owner_id = route.dispatch_model_role_binding_id or ""
        role = role_ceilings.get(owner_id)
        if role is None:
            raise BudgetExceededError("dispatch route has no model-role ceiling")
        descriptors.append(
            (
                DispatchBudgetOwnerKind.MODEL_ROLE,
                owner_id,
                role.provider_budget_cap.provider,
                role.provider_budget_cap.operation_kind,
                role.provider_budget_cap.billing_unit,
                True,
                False,
            )
        )

    evidence_allocations: list[EvidenceBudgetOwnerAllocation] = []
    runtime_allocations: list[RuntimeBudgetOwnerAllocation] = []
    maximum = BudgetAmount.zero()
    for (
        owner_kind,
        owner_id,
        provider,
        operation_kind,
        billing_unit,
        owns_wall,
        internal,
    ) in descriptors:
        if owner_kind == DispatchBudgetOwnerKind.MODEL_ROLE:
            role = role_ceilings[owner_id]
            input_tokens = _integer_per_attempt(role.max_input_tokens, role.max_attempts)
            output_ceiling = _integer_per_attempt(role.max_output_tokens, role.max_attempts)
            output_tokens = output_ceiling if internal else maximum_output_tokens
            if output_tokens > output_ceiling:
                raise BudgetExceededError("model request output exceeds its per-attempt ceiling")
            wall_seconds = (
                role.max_dispatch_wall_seconds / role.max_attempts if owns_wall else Decimal("0")
            )
            cost = (
                (role.max_cost or Decimal("0")) / role.max_attempts
                if budget.currency is not None
                else Decimal("0")
            )
            resource_ceilings = (
                tuple(
                    item.model_copy(update={"maximum": item.maximum / role.max_attempts})
                    for item in role.resource_ceilings
                )
                if owns_wall
                else ()
            )
        else:
            operation_ceiling = operation_ceilings[owner_id]
            input_tokens = 0
            output_tokens = 0
            wall_seconds = (
                operation_ceiling.max_dispatch_wall_seconds / operation_ceiling.max_attempts
            )
            cost = Decimal("0")
            resource_ceilings = tuple(
                item.model_copy(update={"maximum": item.maximum / operation_ceiling.max_attempts})
                for item in operation_ceiling.resource_ceilings
            )
        allocation_fields = dict(
            owner_kind=owner_kind,
            owner_id=owner_id,
            allocated_attempts=1,
            allocated_input_tokens=input_tokens,
            allocated_output_tokens=output_tokens,
            allocated_dispatch_wall_seconds=wall_seconds,
            allocated_provider_units=Decimal("1"),
            allocated_resource_ceilings=resource_ceilings,
            allocated_cost=cost if budget.currency is not None else None,
            currency=budget.currency,
        )
        evidence = EvidenceBudgetOwnerAllocation.model_validate(
            {
                "allocation_hash": budget_owner_allocation_hash(allocation_fields),
                **allocation_fields,
            }
        )
        owner_maximum = BudgetAmount(
            attempts=1,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_seconds=wall_seconds,
            cost=cost,
            resources=tuple(
                (item.dimension_id, item.maximum, item.unit) for item in resource_ceilings
            ),
            provider_units=((provider, operation_kind, billing_unit, 1),),
        )
        evidence_allocations.append(evidence)
        runtime_allocations.append(RuntimeBudgetOwnerAllocation(owner_id, owner_maximum))
        maximum = maximum.add(owner_maximum)
    return tuple(evidence_allocations), tuple(runtime_allocations), maximum


def _integer_per_attempt(total: int, attempts: int) -> int:
    return total // attempts


def _prepare_live_native_attempt(
    state: _NativeExecutionState,
    *,
    attempt_identity: str,
    parent_kind: Literal["ingestion_plan", "case"],
    parent_id: str,
    stage: str,
    ordinal: int,
    request_fingerprint: str,
    role_binding_id: str,
    request_messages_sha256: str | None,
    maximum_output_tokens: int,
    retry_of_attempt_id: str | None,
) -> _PreparedNativeAttempt:
    control = state.control
    ledger = state.budget_ledger
    lifecycle = state.provider_lifecycle
    if control is None or ledger is None or lifecycle is None:
        raise ValueError("live native attempt requires budget and provider lifecycle ownership")
    route = control.require_budget_route(stage=stage)
    if route.dispatch_owner_kind == DispatchBudgetOwnerKind.PROVIDER_OPERATION:
        if route.adapter_profile_id != control.preflight_record.adapter_profile_id:
            raise ValueError("live dispatch route adapter profile differs from preflight")
    evidence_allocations, runtime_allocations, maximum = _reservation_allocations_for_route(
        control.budget,
        route,
        maximum_output_tokens=maximum_output_tokens,
    )

    claim_fields = {
        "occurrence_id": parent_id,
        "lease_record_hash": state.lease_record_hash,
        "lease_epoch": state.lease_epoch,
        "owner_id": state.owner_id,
        "stage": stage,
        "request_fingerprint": request_fingerprint,
        "reconciliation_capability": "none",
        "claimed_at": state.timestamp(),
    }
    claim_id = canonical_sha256(["oamb-native-occurrence-claim-v1", claim_fields])
    claim = OccurrenceClaimRecord.model_validate({"claim_id": claim_id, **claim_fields})
    reservation_fields = dict(
        budget_id=control.budget.budget_id,
        budget_hash=control.budget.budget_hash,
        scope_kind=BudgetScopeKindV3.RUN,
        scope_id=state.run_id,
        attempt_id=attempt_identity,
        dispatch_route_id=route.route_id,
        dispatch_route_hash=route.route_hash,
        owner_allocations=evidence_allocations,
        reserved_attempts=sum(item.allocated_attempts for item in evidence_allocations),
        reserved_input_tokens=sum(item.allocated_input_tokens for item in evidence_allocations),
        reserved_output_tokens=sum(item.allocated_output_tokens for item in evidence_allocations),
        reserved_dispatch_wall_seconds=sum(
            (item.allocated_dispatch_wall_seconds for item in evidence_allocations), Decimal("0")
        ),
        reserved_provider_units=sum(
            (item.allocated_provider_units for item in evidence_allocations), Decimal("0")
        ),
        reserved_resource_ceilings=tuple(
            ceiling
            for allocation in evidence_allocations
            for ceiling in allocation.allocated_resource_ceilings
        ),
        reserved_cost=(
            sum(
                (item.allocated_cost or Decimal("0") for item in evidence_allocations),
                Decimal("0"),
            )
            if control.budget.currency is not None
            else None
        ),
        currency=control.budget.currency,
        reserved_at=state.timestamp(),
    )
    reservation_id = budget_reservation_v3_id(reservation_fields)
    reservation = BudgetReservationRecordV3.model_validate(
        {
            "reservation_id": reservation_id,
            "reservation_hash": budget_reservation_v3_hash(
                {**reservation_fields, "reservation_id": reservation_id}
            ),
            **reservation_fields,
        }
    )
    intent_fields = dict(
        attempt_id=attempt_identity,
        claim_id=claim_id,
        reservation_id=reservation.reservation_id,
        reservation_hash=reservation.reservation_hash,
        scope_kind=BudgetScopeKindV3.RUN,
        scope_id=state.run_id,
        parent_kind=parent_kind,
        parent_id=parent_id,
        stage=stage,
        preflight_record_hash=control.preflight_record.preflight_record_hash,
        budget_id=control.budget.budget_id,
        budget_hash=control.budget.budget_hash,
        dispatch_route_id=route.route_id,
        dispatch_route_hash=route.route_hash,
        request_fingerprint=request_fingerprint,
        reconciliation_capability="none",
        idempotency_key_hash=None,
        sealed_at=state.timestamp(),
    )
    intent = AttemptIntentRecordV3.model_validate(
        {"intent_hash": attempt_intent_v3_hash(intent_fields), **intent_fields}
    )
    ledger.reserve(
        ReservationRequest(
            reservation_id=reservation.reservation_id,
            maximum=maximum,
            owner_allocations=runtime_allocations,
        )
    )
    _seal(state, "occurrence-claims", claim_id, claim)
    _seal(state, "budget-reservations", reservation.reservation_id, reservation)
    _seal(state, "attempt-intents", attempt_identity, intent)
    lifecycle.mark_attempt_dispatched(
        attempt_id=attempt_identity,
        intent_record_hash=intent.intent_hash,
    )
    return _PreparedNativeAttempt(
        attempt_id=attempt_identity,
        parent_kind=parent_kind,
        parent_id=parent_id,
        stage=stage,
        ordinal=ordinal,
        request_fingerprint=request_fingerprint,
        request_messages_sha256=request_messages_sha256,
        role_binding_id=role_binding_id,
        retry_of_attempt_id=retry_of_attempt_id,
        route=route,
        reservation=reservation,
        intent=intent,
        maximum=maximum,
        owner_maximums=runtime_allocations,
    )


def _seal_native_success(
    state: _NativeExecutionState,
    prepared: _PreparedNativeAttempt,
    *,
    started_at: datetime,
    ended_at: datetime,
    raw_response_ref: str,
    index_contribution: IndexContribution,
) -> AttemptRecordV2 | AttemptRecordV4:
    receipt = AttemptReceiptRecord(
        attempt_id=prepared.attempt_id,
        receipt_kind=AttemptReceiptKind.RESPONSE,
        raw_response_ref=raw_response_ref,
        raw_error_ref=None,
        dispatch_started_at=started_at,
        receipt_observed_at=ended_at,
        provider_request_wall_seconds=_elapsed_seconds(started_at, ended_at),
    )
    _seal(state, "attempt-receipts", prepared.attempt_id, receipt)
    if prepared.intent is not None and prepared.route is not None:
        receipt_hash = canonical_sha256(receipt)
        terminal_fields = dict(
            attempt_id=prepared.attempt_id,
            intent_hash=prepared.intent.intent_hash,
            dispatch_route_id=prepared.route.route_id,
            dispatch_route_hash=prepared.route.route_hash,
            receipt_record_hash=receipt_hash,
            run_id=state.run_id,
            parent_kind=prepared.parent_kind,
            parent_id=prepared.parent_id,
            stage=prepared.stage,
            ordinal=prepared.ordinal,
            request_fingerprint=prepared.request_fingerprint,
            request_messages_sha256=prepared.request_messages_sha256,
            started_at=started_at,
            ended_at=ended_at,
            outcome=AttemptOutcome.SUCCEEDED,
            retry_of_attempt_id=prepared.retry_of_attempt_id,
            idempotency_key_hash=None,
            reconciliation_capability="none",
            raw_response_ref=raw_response_ref,
            raw_error_ref=None,
            index_contribution=index_contribution,
            superseded_by_attempt_id=None,
        )
        live_terminal = AttemptRecordV4.model_validate(
            {
                "attempt_record_hash": attempt_record_v4_hash(terminal_fields),
                **terminal_fields,
            }
        )
        _seal(state, "attempts", prepared.attempt_id, live_terminal)
        return live_terminal
    terminal = AttemptRecordV2(
        attempt_id=prepared.attempt_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        stage=prepared.stage,
        ordinal=prepared.ordinal,
        request_fingerprint=prepared.request_fingerprint,
        request_messages_sha256=prepared.request_messages_sha256,
        started_at=started_at,
        ended_at=ended_at,
        outcome=AttemptOutcome.SUCCEEDED,
        retry_of_attempt_id=prepared.retry_of_attempt_id,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=raw_response_ref,
        raw_error_ref=None,
        index_contribution=index_contribution,
        superseded_by_attempt_id=None,
    )
    _seal(state, "attempts", prepared.attempt_id, terminal)
    return terminal


def _seal_native_attempt_accounting(
    state: _NativeExecutionState,
    prepared: _PreparedNativeAttempt,
    *,
    started_at: datetime,
    ended_at: datetime,
    raw_response_ref: str,
    usage_record_ids: tuple[str, ...],
    indexing_view: IndexingView,
    inline_usage_records: tuple[StrictContract, ...] = (),
) -> tuple[tuple[str, ...], str, str]:
    cached = state.attempt_accounting.get(prepared.attempt_id)
    if cached is not None:
        return cached
    if prepared.route is not None:
        result = _seal_live_native_attempt_accounting(
            state,
            prepared,
            started_at=started_at,
            ended_at=ended_at,
            raw_response_ref=raw_response_ref,
            usage_record_ids=usage_record_ids,
            inline_usage_records=inline_usage_records,
            indexing_view=indexing_view,
        )
        state.attempt_accounting[prepared.attempt_id] = result
        return result
    if len(usage_record_ids) > 1:
        raise ValueError("native attempt cannot bind multiple token-usage records")
    if usage_record_ids:
        closed_usage_ids = usage_record_ids
    else:
        usage_record_id = canonical_sha256(
            [
                "oamb-native-unavailable-token-usage-v1",
                prepared.attempt_id,
                raw_response_ref,
            ]
        )
        unavailable_usage = TokenUsageRecordV2(
            usage_record_id=usage_record_id,
            attempt_id=prepared.attempt_id,
            parent_kind=prepared.parent_kind,
            parent_id=prepared.parent_id,
            stage=TokenStageV2(prepared.stage),
            operation_kind=f"{prepared.stage}_provider_call",
            token_domain=TokenDomain.EXTERNAL_LLM,
            measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
            input_tokens=None,
            visible_output_tokens=None,
            supplier_reported_total_tokens=None,
            context_view_tokens=None,
            proof_status=ProofStatus.UNAVAILABLE,
            reason="supplier_token_usage_unavailable",
            raw_response_ref=raw_response_ref,
        )
        _seal(state, "usage", usage_record_id, unavailable_usage)
        closed_usage_ids = (usage_record_id,)

    resource_record_id = canonical_sha256(
        [
            "oamb-native-unavailable-resource-usage-v1",
            prepared.attempt_id,
            raw_response_ref,
        ]
    )
    resource = ResourceUsageRecord(
        resource_record_id=resource_record_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        stage=prepared.stage,
        meter_boundary="provider_request_resource_v1",
        dimension_id="provider_request_resource_v1",
        value=None,
        unit="provider_native_unit",
        measurement_source="provider_response_without_resource_meter",
        measurement_spec_id="oamb-native-provider-resource-v1",
        environment_hash=canonical_sha256(
            ["oamb-native-resource-environment-v1", prepared.role_binding_id]
        ),
        started_at=started_at,
        ended_at=ended_at,
        raw_telemetry_ref=raw_response_ref,
        proof_status=ProofStatus.UNAVAILABLE,
        reason="provider_resource_usage_unavailable",
    )
    _seal(state, "resources", resource_record_id, resource)

    cost_record_id = canonical_sha256(
        [
            "oamb-native-unavailable-cost-v1",
            prepared.attempt_id,
            closed_usage_ids,
            resource_record_id,
        ]
    )
    cost = CostRecord(
        cost_record_id=cost_record_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        basis=CostBasis.ACTUAL_SUPPLIER_CHARGE,
        indexing_view=indexing_view,
        amount=None,
        currency=None,
        price_snapshot_id=None,
        source_usage_record_ids=closed_usage_ids,
        source_resource_record_ids=(resource_record_id,),
        proof_status=ProofStatus.UNAVAILABLE,
        reason="supplier_cost_unavailable",
    )
    _seal(state, "costs", cost_record_id, cost)
    result = closed_usage_ids, resource_record_id, cost_record_id
    state.attempt_accounting[prepared.attempt_id] = result
    return result


def _seal_live_native_attempt_accounting(
    state: _NativeExecutionState,
    prepared: _PreparedNativeAttempt,
    *,
    started_at: datetime,
    ended_at: datetime,
    raw_response_ref: str,
    usage_record_ids: tuple[str, ...],
    inline_usage_records: tuple[StrictContract, ...],
    indexing_view: IndexingView,
) -> tuple[tuple[str, ...], str, str]:
    route = prepared.route
    if route is None or prepared.reservation is None or prepared.intent is None:
        raise ValueError("live native accounting requires closed route evidence")
    control = state.control
    if control is None:
        raise ValueError("live native accounting requires the durable run control")
    usage_ids: tuple[str, ...] = ()
    try:
        token_stage = TokenStage(prepared.stage)
    except ValueError:
        token_stage = None
    if token_stage is not None:
        usage_ids = _seal_live_token_usage(
            state,
            prepared,
            token_stage=token_stage,
            raw_response_ref=raw_response_ref,
            usage_record_ids=usage_record_ids,
            inline_usage_records=inline_usage_records,
        )

    resource_record_id = canonical_sha256(
        ["oamb-live-dispatch-wall-v1", prepared.attempt_id, route.route_hash]
    )
    resource = ResourceUsageRecordV2(
        resource_record_id=resource_record_id,
        attempt_id=prepared.attempt_id,
        dispatch_route_id=route.route_id,
        dispatch_route_hash=route.route_hash,
        budget_owner_kind=prepared.reservation.owner_allocations[0].owner_kind,
        budget_owner_id=prepared.reservation.owner_allocations[0].owner_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        stage=prepared.stage,
        meter_boundary="trusted_dispatch_wall_v1",
        dimension_id="provider_request_wall_seconds_v1",
        value=_elapsed_seconds(started_at, ended_at),
        unit="seconds",
        measurement_source="trusted_wall_clock",
        measurement_spec_id="oamb-trusted-dispatch-wall-v1",
        environment_hash=control.run_spec.environment_hash,
        started_at=started_at,
        ended_at=ended_at,
        raw_telemetry_ref=raw_response_ref,
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
    )
    _seal(state, "resources", resource_record_id, resource)
    cost_record_id = canonical_sha256(
        ["oamb-live-unavailable-cost-v1", prepared.attempt_id, usage_ids, resource_record_id]
    )
    cost = CostRecordV2(
        cost_record_id=cost_record_id,
        attempt_id=prepared.attempt_id,
        dispatch_route_id=route.route_id,
        dispatch_route_hash=route.route_hash,
        budget_owner_kind=prepared.reservation.owner_allocations[0].owner_kind,
        budget_owner_id=prepared.reservation.owner_allocations[0].owner_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        basis=CostBasis.ACTUAL_SUPPLIER_CHARGE,
        indexing_view=indexing_view,
        amount=None,
        currency=None,
        price_snapshot_id=None,
        source_usage_record_ids=usage_ids,
        source_resource_record_ids=(resource_record_id,),
        proof_status=ProofStatus.UNAVAILABLE,
        reason="supplier_cost_unavailable",
    )
    _seal(state, "costs", cost_record_id, cost)
    ledger = state.budget_ledger
    lifecycle = state.provider_lifecycle
    if ledger is None or lifecycle is None or prepared.maximum is None:
        raise ValueError("live native accounting lost budget or lifecycle state")
    ledger.commit(
        prepared.reservation.reservation_id,
        observed=prepared.maximum,
        owner_observed=prepared.owner_maximums,
    )
    lifecycle.clear_attempt_after_receipt(
        attempt_id=prepared.attempt_id,
        expected_intent_record_hash=prepared.intent.intent_hash,
    )
    return usage_ids, resource_record_id, cost_record_id


def _seal_live_token_usage(
    state: _NativeExecutionState,
    prepared: _PreparedNativeAttempt,
    *,
    token_stage: TokenStage,
    raw_response_ref: str,
    usage_record_ids: tuple[str, ...],
    inline_usage_records: tuple[StrictContract, ...],
) -> tuple[str, ...]:
    route = prepared.route
    reservation = prepared.reservation
    control = state.control
    if route is None or reservation is None or control is None:
        raise ValueError("live token usage requires route, reservation, and run control")
    if len(set(usage_record_ids)) != len(usage_record_ids):
        raise ValueError("live token usage contains duplicate source usage IDs")
    inline_by_id: dict[str, TokenUsageRecordV2 | TokenUsageRecordV3] = {}
    for record in inline_usage_records:
        usage: TokenUsageRecordV2 | TokenUsageRecordV3
        if isinstance(record, TokenUsageRecordV3):
            usage = record
        elif isinstance(record, TokenUsageRecordV2):
            usage = record
        else:
            raise ValueError("live token usage source requires version 2 or 3")
        inline_by_id[usage.usage_record_id] = usage
    if len(inline_by_id) != len(inline_usage_records):
        raise ValueError("live token usage contains duplicate inline records")
    pending = state.pending_model_usage
    if pending is None:
        raise ValueError("live token usage capture is unavailable")
    sources: list[TokenUsageRecordV2 | TokenUsageRecordV3] = []
    for usage_id in usage_record_ids:
        inline = inline_by_id.get(usage_id)
        captured = pending.get(usage_id)
        if (inline is None) == (captured is None):
            raise ValueError("live token usage ID is missing or ambiguously sourced")
        sources.append(inline if inline is not None else captured)  # type: ignore[arg-type]
    if set(inline_by_id) != set(usage_record_ids) & set(inline_by_id):
        raise ValueError("live token usage contains an unreferenced inline record")

    model_allocations = tuple(
        allocation
        for allocation in reservation.owner_allocations
        if allocation.owner_kind == DispatchBudgetOwnerKind.MODEL_ROLE
    )
    token_allocations = model_allocations or (reservation.owner_allocations[0],)
    roles_by_id = {binding.binding_id: binding for binding in control.role_bindings}
    source_by_owner: dict[str, TokenUsageRecordV2 | TokenUsageRecordV3] = {}
    if route.dispatch_owner_kind == DispatchBudgetOwnerKind.MODEL_ROLE:
        if len(token_allocations) != 1 or len(sources) > 1:
            raise ValueError("direct live model usage requires one owner and at most one source")
        if sources:
            source_by_owner[token_allocations[0].owner_id] = sources[0]
    else:
        if not model_allocations and len(token_allocations) == 1 and len(sources) == 1:
            source_by_owner[token_allocations[0].owner_id] = sources[0]
        for source in sources:
            if source in source_by_owner.values():
                continue
            if not isinstance(source, TokenUsageRecordV3):
                raise ValueError("provider-internal live usage requires version 3 model identity")
            matching = tuple(
                allocation
                for allocation in token_allocations
                if allocation.owner_id in roles_by_id
                and roles_by_id[allocation.owner_id].model == source.model
            )
            if len(matching) != 1 or matching[0].owner_id in source_by_owner:
                raise ValueError("provider-internal live usage does not map to one role owner")
            source_by_owner[matching[0].owner_id] = source

    sealed_ids: list[str] = []
    for allocation in token_allocations:
        assigned_source = source_by_owner.get(allocation.owner_id)
        role = roles_by_id.get(allocation.owner_id)
        usage = _live_token_usage_v5(
            prepared,
            route=route,
            owner=allocation,
            token_stage=token_stage,
            raw_response_ref=raw_response_ref,
            role=role,
            source=assigned_source,
        )
        _seal(state, "usage", usage.usage_record_id, usage)
        sealed_ids.append(usage.usage_record_id)
    for usage_id in usage_record_ids:
        pending.pop(usage_id, None)
    return tuple(sealed_ids)


def _live_token_usage_v5(
    prepared: _PreparedNativeAttempt,
    *,
    route: DispatchBudgetRoute,
    owner: EvidenceBudgetOwnerAllocation,
    token_stage: TokenStage,
    raw_response_ref: str,
    role: ModelRoleBindingV2 | None,
    source: TokenUsageRecordV2 | TokenUsageRecordV3 | None,
) -> TokenUsageRecordV5:
    if source is not None and (
        source.attempt_id != prepared.attempt_id
        or source.parent_kind != prepared.parent_kind
        or source.parent_id != prepared.parent_id
        or source.stage.value != prepared.stage
        or source.raw_response_ref != raw_response_ref
        or source.token_domain != TokenDomain.EXTERNAL_LLM
        or source.measurement_source != TokenMeasurementSource.SUPPLIER_RESPONSE
    ):
        raise ValueError("live token usage source does not bind its attempt and raw response")
    model = (
        source.model
        if isinstance(source, TokenUsageRecordV3)
        else role.model
        if role is not None
        else "provider-managed-model"
    )
    if model is None:
        raise ValueError("live token usage owner lacks a model")
    values: tuple[int | None, int | None, int | None, int | None, int | None]
    raw_paths: tuple[tuple[str, str], ...]
    covered: tuple[str, ...]
    unavailable: tuple[str, ...]
    proof: ProofStatus
    reason: str | None
    meter_schema_id: str
    relationships: tuple[tuple[str, str], ...]
    if source is None:
        values = (None, None, None, None, None)
        raw_paths = ()
        covered = ()
        unavailable = (
            "input_tokens",
            "visible_output_tokens",
            "supplier_reported_total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        )
        proof = ProofStatus.UNAVAILABLE
        reason = "supplier_token_usage_unavailable"
        meter_schema_id = "supplier-usage-unavailable-v1"
        relationships = ()
    elif isinstance(source, TokenUsageRecordV3):
        values = (
            source.input_tokens,
            source.visible_output_tokens,
            source.supplier_reported_total_tokens,
            source.cached_input_tokens,
            source.reasoning_tokens,
        )
        raw_paths = source.raw_field_paths
        covered = source.covered_dimensions
        unavailable = source.unavailable_dimensions
        proof = source.proof_status
        reason = source.reason
        meter_schema_id = source.meter_schema_id
        relationships = source.inclusion_relationships
    else:
        values = (
            source.input_tokens,
            source.visible_output_tokens,
            source.supplier_reported_total_tokens,
            None,
            None,
        )
        base_names = (
            "input_tokens",
            "visible_output_tokens",
            "supplier_reported_total_tokens",
        )
        raw_names = (
            "usage.prompt_tokens",
            "usage.completion_tokens",
            "usage.total_tokens",
        )
        raw_paths = tuple(
            (name, path)
            for name, path, value in zip(base_names, raw_names, values[:3], strict=True)
            if value is not None
        )
        covered = tuple(
            name for name, value in zip(base_names, values[:3], strict=True) if value is not None
        )
        unavailable = tuple(
            name
            for name, value in zip(
                (*base_names, "cached_input_tokens", "reasoning_tokens"),
                values,
                strict=True,
            )
            if value is None
        )
        proof = ProofStatus.MEASURED_PARTIAL if covered else ProofStatus.UNAVAILABLE
        reason = (
            "source_v2_omits_cached_and_reasoning_dimensions"
            if covered
            else source.reason or "supplier_token_usage_unavailable"
        )
        meter_schema_id = "openai-compatible-usage-v2-conversion-v1"
        relationships = ()
    usage_id = canonical_sha256(
        [
            "oamb-live-token-usage-v5",
            prepared.attempt_id,
            route.route_hash,
            owner.owner_kind,
            owner.owner_id,
            canonical_sha256(source) if source is not None else None,
        ]
    )
    return TokenUsageRecordV5(
        usage_record_id=usage_id,
        attempt_id=prepared.attempt_id,
        dispatch_route_id=route.route_id,
        dispatch_route_hash=route.route_hash,
        budget_owner_kind=owner.owner_kind,
        budget_owner_id=owner.owner_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        stage=token_stage,
        operation_kind=route.operation_kind,
        token_domain=TokenDomain.EXTERNAL_LLM,
        measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=values[0],
        visible_output_tokens=values[1],
        supplier_reported_total_tokens=values[2],
        context_view_tokens=None,
        cached_input_tokens=values[3],
        reasoning_tokens=values[4],
        model=model,
        meter_schema_id=meter_schema_id,
        raw_field_paths=raw_paths,
        covered_dimensions=covered,
        unavailable_dimensions=unavailable,
        not_applicable_dimensions=(),
        inclusion_relationships=relationships,
        token_measurement_complete=not unavailable,
        billing_complete=False,
        proof_status=proof,
        reason=reason,
        raw_response_ref=raw_response_ref,
    )


def _seal_native_failure_preserving(
    state: _NativeExecutionState,
    prepared: _PreparedNativeAttempt,
    *,
    started_at: datetime,
    error: BaseException,
) -> AttemptRecordV2 | AttemptRecordV4:
    try:
        return _seal_native_failure(
            state,
            prepared,
            started_at=started_at,
            error=error,
        )
    except BaseException as evidence_error:
        if isinstance(error, asyncio.CancelledError):
            raise NativeCancellationEvidenceError(error, evidence_error) from None
        raise BaseExceptionGroup(
            "native dispatch and failure-evidence seal both failed",
            (error, evidence_error),
        ) from None


def _seal_native_failure(
    state: _NativeExecutionState,
    prepared: _PreparedNativeAttempt,
    *,
    started_at: datetime,
    error: BaseException,
) -> AttemptRecordV2 | AttemptRecordV4:
    ended_at = state.timestamp()
    cancelled_before_dispatch = _contains_cancelled_before_dispatch(error)
    unknown = not cancelled_before_dispatch and (
        isinstance(
            error,
            (
                ModelCallCancelledUnknownOutcome,
                ModelCallUnknownOutcome,
                MemorySystemCallCancelledUnknownOutcome,
                MemorySystemCallUnknownOutcome,
            ),
        )
        or isinstance(error, asyncio.CancelledError)
        or "UnknownOutcome" in type(error).__name__
    )
    raw_error_ref: str | None = None
    if unknown:
        receipt_kind = AttemptReceiptKind.UNKNOWN_OUTCOME
        outcome = AttemptOutcome.UNKNOWN_OUTCOME
    else:
        receipt_kind = AttemptReceiptKind.ERROR
        outcome = AttemptOutcome.CANCELLED if cancelled_before_dispatch else AttemptOutcome.FAILED
        raw_handle = getattr(error, "raw_reference", None)
        raw_error_ref = getattr(raw_handle, "sha256", None)
        if raw_error_ref is None:
            error_payload = canonical_json_bytes(
                {
                    "operation": "dispatch_error",
                    "stage": prepared.stage,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
            raw_error_ref = _seal_raw(state.store, error_payload, media_type="application/json")
    receipt = AttemptReceiptRecord(
        attempt_id=prepared.attempt_id,
        receipt_kind=receipt_kind,
        failure_kind=getattr(error, "failure_kind", None),
        supplier_status_code=getattr(
            error, "supplier_status_code", getattr(error, "status_code", None)
        ),
        settlement_basis=getattr(error, "settlement_basis", None),
        internal_retry_count=getattr(error, "internal_retry_count", None),
        settlement_task_id=getattr(error, "expected_task_id", None),
        settlement_session_id=getattr(error, "expected_session_id", None),
        raw_response_ref=None,
        raw_error_ref=raw_error_ref,
        dispatch_started_at=started_at,
        receipt_observed_at=ended_at,
        provider_request_wall_seconds=_elapsed_seconds(started_at, ended_at),
    )
    _seal(state, "attempt-receipts", prepared.attempt_id, receipt)
    if prepared.intent is not None and prepared.route is not None:
        terminal_fields = dict(
            attempt_id=prepared.attempt_id,
            intent_hash=prepared.intent.intent_hash,
            dispatch_route_id=prepared.route.route_id,
            dispatch_route_hash=prepared.route.route_hash,
            receipt_record_hash=canonical_sha256(receipt),
            run_id=state.run_id,
            parent_kind=prepared.parent_kind,
            parent_id=prepared.parent_id,
            stage=prepared.stage,
            ordinal=prepared.ordinal,
            request_fingerprint=prepared.request_fingerprint,
            request_messages_sha256=prepared.request_messages_sha256,
            started_at=started_at,
            ended_at=ended_at,
            outcome=outcome,
            retry_of_attempt_id=prepared.retry_of_attempt_id,
            idempotency_key_hash=None,
            reconciliation_capability="none",
            raw_response_ref=None,
            raw_error_ref=raw_error_ref,
            index_contribution=IndexContribution.NONE,
            superseded_by_attempt_id=None,
        )
        live_terminal = AttemptRecordV4.model_validate(
            {
                "attempt_record_hash": attempt_record_v4_hash(terminal_fields),
                **terminal_fields,
            }
        )
        _seal(state, "attempts", prepared.attempt_id, live_terminal)
        ledger = state.budget_ledger
        lifecycle = state.provider_lifecycle
        reservation = prepared.reservation
        if ledger is None or lifecycle is None or reservation is None:
            raise ValueError("live failed attempt lost budget or lifecycle state")
        if unknown:
            ledger.mark_unknown(reservation.reservation_id)
            if prepared.stage in {"answer", "judge"} and not isinstance(
                error, asyncio.CancelledError
            ):
                lifecycle.clear_attempt_after_receipt(
                    attempt_id=prepared.attempt_id,
                    expected_intent_record_hash=prepared.intent.intent_hash,
                )
            return live_terminal
        if cancelled_before_dispatch:
            ledger.cancel_before_dispatch(reservation.reservation_id)
            if not isinstance(error, (InfrastructureBackoffCancelled, NativeRunInterrupted)):
                lifecycle.clear_attempt_after_receipt(
                    attempt_id=prepared.attempt_id,
                    expected_intent_record_hash=prepared.intent.intent_hash,
                )
            return live_terminal
        if raw_error_ref is None:
            raise ValueError("failed live attempt requires durable raw error evidence")
        if _contains_infrastructure_retry_exhausted(error):
            owner_observed = (
                tuple(
                    RuntimeBudgetOwnerAllocation(
                        owner_id=allocation.owner_id,
                        maximum=BudgetAmount.zero(),
                    )
                    for allocation in prepared.owner_maximums
                )
                if prepared.owner_maximums
                else None
            )
            ledger.commit(
                reservation.reservation_id,
                observed=BudgetAmount.zero(),
                owner_observed=owner_observed,
            )
            return live_terminal
        _seal_native_attempt_accounting(
            state,
            prepared,
            started_at=started_at,
            ended_at=ended_at,
            raw_response_ref=raw_error_ref,
            usage_record_ids=tuple(getattr(error, "usage_reference_ids", ())),
            inline_usage_records=(),
            indexing_view=IndexingView.ATTEMPTED
            if prepared.stage == "memory_ingest"
            else IndexingView.NOT_APPLICABLE,
        )
        return live_terminal
    terminal = AttemptRecordV2(
        attempt_id=prepared.attempt_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        stage=prepared.stage,
        ordinal=prepared.ordinal,
        request_fingerprint=prepared.request_fingerprint,
        request_messages_sha256=prepared.request_messages_sha256,
        started_at=started_at,
        ended_at=ended_at,
        outcome=outcome,
        retry_of_attempt_id=prepared.retry_of_attempt_id,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=None,
        raw_error_ref=raw_error_ref,
        index_contribution=IndexContribution.NONE,
        superseded_by_attempt_id=None,
    )
    _seal(state, "attempts", prepared.attempt_id, terminal)
    return terminal


def _contains_cancelled_before_dispatch(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(
            current,
            (
                MemorySystemCallCancelledBeforeDispatch,
                InfrastructureBackoffCancelled,
                NativeRunInterrupted,
            ),
        ):
            return True
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
        receipt = getattr(current, "receipt", None)
        for linked in (
            getattr(receipt, "provider_error", None),
            getattr(receipt, "post_projection_error", None),
        ):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return False


def _seal_native_run_record(
    state: _NativeExecutionState,
    *,
    run_spec_hash: str,
    run_state: Literal[
        RunState.FINALIZED,
        RunState.ABORTED,
        RunState.INFRASTRUCTURE_BLOCKED,
    ],
    ingestion_occurrence_ids: tuple[str, ...],
    case_occurrence_ids: tuple[str, ...],
) -> None:
    record = RunRecord(
        run_id=state.run_id,
        run_spec_hash=run_spec_hash,
        state=run_state,
        resume_disposition=ResumeDisposition.NOT_APPLICABLE,
        started_at=state.started_at,
        ended_at=state.timestamp(),
        ingestion_occurrence_ids=ingestion_occurrence_ids,
        case_occurrence_ids=case_occurrence_ids,
    )
    _seal(state, "run", state.run_id, record)


def _contains_infrastructure_retry_exhausted(error: BaseException) -> bool:
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if isinstance(current, InfrastructureRetryExhausted):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return False


def _is_pure_infrastructure_retry_exhaustion(error: BaseException) -> bool:
    """Return true only when every terminal root is one safe retry exhaustion."""

    if isinstance(error, InfrastructureRetryExhausted):
        return True
    if isinstance(error, BaseExceptionGroup):
        return bool(error.exceptions) and all(
            _is_pure_infrastructure_retry_exhaustion(child) for child in error.exceptions
        )
    return False


def _elapsed_seconds(started_at: datetime, ended_at: datetime) -> Decimal:
    microseconds = int((ended_at - started_at).total_seconds() * 1_000_000)
    return Decimal(microseconds) / Decimal("1000000")


def _acquire_native_run_owner(
    capsule_root: Path,
    run_id: str,
    *,
    control: NativeRunControl | None = None,
) -> None:
    owner_bytes = canonical_json_bytes(
        {
            "schema_name": "native_run_owner",
            "schema_version": 1,
            "run_id": run_id,
            "owner_id": control.owner_id if control is not None else NATIVE_OWNER_ID,
        }
    )
    result = atomic_write_bytes(
        capsule_root / NATIVE_OWNER_NAME,
        owner_bytes,
        trusted_root=capsule_root,
    )
    if not result.created:
        raise NativeRunOwnershipError(f"native run {run_id!r} is already owned")


def _native_run_lease(
    run_id: str,
    adapter_profile_id: str,
    *,
    control: NativeRunControl | None = None,
    acquired_at: datetime | None = None,
) -> RunLeaseRecord:
    provider_project_id = (
        control.preflight_record.provider_project_id if control is not None else "native-fixture"
    )
    candidate = RunLeaseRecord(
        lease_record_hash="0" * 64,
        run_id=run_id,
        provider_project_id=provider_project_id,
        provider_profile_id=adapter_profile_id,
        lease_epoch=1,
        owner_id=control.owner_id if control is not None else NATIVE_OWNER_ID,
        host_fingerprint=(
            control.host_fingerprint
            if control is not None
            else canonical_sha256(["oamb-native-fixture-host-v1"])
        ),
        process_id=control.process_id if control is not None else 1,
        predecessor_lease_record_hash=None,
        acquired_at=(
            acquired_at
            if acquired_at is not None
            else control.wall_clock()
            if control is not None
            else NATIVE_FIXTURE_STARTED_AT
        ),
    )
    return candidate.model_copy(
        update={
            "lease_record_hash": canonical_sha256(
                candidate.model_dump(mode="python", exclude={"lease_record_hash"})
            )
        }
    )


def _contains_unknown_outcome(error: BaseException) -> bool:
    pending: list[BaseException] = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if (
            isinstance(
                current,
                (
                    ModelCallCancelledUnknownOutcome,
                    ModelCallUnknownOutcome,
                    MemorySystemCallCancelledUnknownOutcome,
                    MemorySystemCallUnknownOutcome,
                ),
            )
            or "UnknownOutcome" in type(current).__name__
        ):
            return True
        if isinstance(current, BaseExceptionGroup):
            pending.extend(current.exceptions)
        for linked in (current.__cause__, current.__context__):
            if isinstance(linked, BaseException):
                pending.append(linked)
    return False


def _verify_terminal_capsule(
    capsule_root: Path,
    *,
    run_id: str,
    run_spec_hash: str,
) -> None:
    payload = read_regular_file(capsule_root / "capsule-manifest.json")
    manifest = CapsuleManifest.model_validate_json(payload)
    if payload != canonical_json_bytes(manifest):
        raise ValueError("terminal capsule manifest is not canonical")
    if manifest.run_id != run_id or manifest.run_spec_hash != run_spec_hash:
        raise ValueError("terminal capsule manifest does not bind the live run")
    run_path = capsule_root / "source" / "run" / f"{run_id}.json"
    run_payload = read_regular_file(run_path)
    run_record = RunRecord.model_validate_json(run_payload)
    if run_record.state not in {
        RunState.FINALIZED,
        RunState.ABORTED,
        RunState.INFRASTRUCTURE_BLOCKED,
    }:
        raise ValueError("terminal capsule does not contain a terminal run record")
    relative_run_path = run_path.relative_to(capsule_root).as_posix()
    expected_entry = next(
        (entry for entry in manifest.source_entries if entry.relative_path == relative_run_path),
        None,
    )
    if expected_entry is None or expected_entry.sha256 != hashlib.sha256(run_payload).hexdigest():
        raise ValueError("terminal capsule manifest does not bind its run record")


def _verify_supervised_aborted_capsule(
    capsule_root: Path,
    *,
    run_id: str,
    run_spec_hash: str,
    active_attempts: tuple[dict[str, object], ...],
) -> None:
    _verify_terminal_capsule(
        capsule_root,
        run_id=run_id,
        run_spec_hash=run_spec_hash,
    )
    run_payload = read_regular_file(capsule_root / "source" / "run" / f"{run_id}.json")
    if RunRecord.model_validate_json(run_payload).state != RunState.ABORTED:
        raise ValueError("supervised abort release requires an aborted run record")
    manifest = CapsuleManifest.model_validate_json(
        read_regular_file(capsule_root / "capsule-manifest.json")
    )
    entries = {entry.relative_path: entry for entry in manifest.source_entries}
    for pointer in active_attempts:
        attempt_id_value = pointer["attempt_id"]
        intent_hash = pointer["intent_record_sha256"]
        if not isinstance(attempt_id_value, str) or not isinstance(intent_hash, str):
            raise ValueError("active attempt identity is malformed")
        relative_path = f"source/attempt-intents/{attempt_id_value}.json"
        intent_payload = read_regular_file(capsule_root / relative_path)
        intent = AttemptIntentRecordV3.model_validate_json(intent_payload)
        if intent_payload != canonical_json_bytes(intent):
            raise ValueError("active attempt intent is not canonical")
        if (
            intent.attempt_id != attempt_id_value
            or intent.intent_hash != intent_hash
            or intent.scope_kind != BudgetScopeKindV3.RUN
            or intent.scope_id != run_id
        ):
            raise ValueError("active attempt does not bind the aborted run")
        entry = entries.get(relative_path)
        if (
            entry is None
            or entry.record_kind != "attempt_intent_record"
            or entry.record_id != attempt_id_value
            or entry.sha256 != hashlib.sha256(intent_payload).hexdigest()
        ):
            raise ValueError("terminal capsule manifest does not bind active attempt intent")


def _seal_close_error(
    state: _NativeExecutionState,
    *,
    client_profile_id: str,
    shutdown_stage: str,
    error: BaseException,
) -> None:
    occurred_at = state.timestamp()
    error_payload = canonical_json_bytes(
        {
            "operation": "close",
            "shutdown_stage": shutdown_stage,
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )
    error_ref = _seal_raw(state.store, error_payload, media_type="application/json")
    fields = {
        "owner_kind": "run",
        "owner_id": state.run_id,
        "client_profile_id": client_profile_id,
        "error_ref": error_ref,
        "shutdown_stage": shutdown_stage,
        "occurred_at": occurred_at,
    }
    close_error_id = canonical_sha256(["oamb-close-error-v1", fields])
    record = CloseErrorRecord.model_validate({"close_error_id": close_error_id, **fields})
    _seal(state, "close-errors", close_error_id, record)


def _seal(
    state: _NativeExecutionState,
    collection: str,
    record_id: str,
    record: object,
) -> None:
    from oamb.contracts.base import StrictContract

    if not isinstance(record, StrictContract):
        raise TypeError("native source records must be strict public contracts")
    seal_source_contract(
        state.store,
        relative_path=f"source/{collection}/{record_id}.json",
        record_id=record_id,
        record=record,
    )
    if isinstance(record, (AttemptRecordV2, AttemptRecordV4)):
        state.operation_records[record.attempt_id] = record


def _require_safe_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be one safe path component")
