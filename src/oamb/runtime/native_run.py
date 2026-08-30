"""Fixture-safe native capsule composition through dependency-free ports."""

from __future__ import annotations

import asyncio
import hashlib
import math
import multiprocessing
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Literal, Protocol, TypeAlias

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
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
from oamb.contracts.base import StrictContract
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptReceiptKind,
    AttemptReceiptRecord,
    AttemptRecordV2,
    BudgetReservationRecord,
    CapsuleManifest,
    CaseEvaluationDisposition,
    CaseRecordV3,
    CloseErrorRecord,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
    OccurrenceClaimRecord,
    RunLeaseRecord,
    RunRecord,
)
from oamb.contracts.ids import (
    attempt_id,
    canonical_json_bytes,
    canonical_sha256,
    case_occurrence_id,
    ingestion_occurrence_id,
)
from oamb.contracts.ports import (
    AnswerValue,
    ArtifactStorePort,
    CasePlan,
    DeterministicEvaluation,
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
    ModelCallUnknownOutcome,
    ModelClientPort,
    ModelRequest,
    ProjectionReceipt,
    RawPayloadSealRequest,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
    ScopeReceipt,
    VisibleEvidencePolicy,
    WorkloadPort,
)
from oamb.contracts.specifications import (
    BudgetScopeKindV2,
    BudgetSpecV2,
    ExternalCallApprovalRecord,
    ModelRoleBindingV2,
    RunPreflightRecord,
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
from oamb.runtime.memory_query import execute_read_only_retrieval
from oamb.runtime.source_records import seal_source_contract

NATIVE_FIXTURE_STARTED_AT = datetime(2026, 1, 1, tzinfo=UTC)
NATIVE_RETRIEVAL_TOP_K = 100
NATIVE_JUDGE_MAX_OUTPUT_TOKENS = 1
NATIVE_CLOSE_TIMEOUT_SECONDS = 5.0
NATIVE_PROCESS_POLL_SECONDS = 0.01
NATIVE_PROCESS_TERMINATION_SECONDS = 0.10
NATIVE_OWNER_NAME = "native-run-owner.json"
NATIVE_OWNER_ID = "native-fixture-owner-v1"

NativeIngestionPlanRecord: TypeAlias = IngestionPlanRecordV2 | IngestionPlanRecordV3


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


class NativeCancellationEvidenceError(asyncio.CancelledError):
    """Preserves cancellation while carrying a separate evidence-seal failure."""

    def __init__(self, primary_error: BaseException, evidence_error: BaseException) -> None:
        super().__init__("native cancellation and failure-evidence seal both failed")
        self.errors = (primary_error, evidence_error)


class CapsuleArtifactStorePort(ArtifactStorePort, Protocol):
    def finalize_capsule(self, *, run_id: str, run_spec_hash: str) -> CapsuleManifest: ...


@dataclass(frozen=True, slots=True)
class NativeRunArtifacts:
    capsule_root: Path
    manifest: CapsuleManifest
    ingestion_plan_records: tuple[NativeIngestionPlanRecord, ...]
    case_records: tuple[CaseRecordV3, ...]


@dataclass(frozen=True, slots=True)
class NativeRunControl:
    """Exact durable authorization and identity closure for one live native run."""

    run_spec: RunSpec
    preflight_record: RunPreflightRecord
    approval: ExternalCallApprovalRecord
    budget: BudgetSpecV2
    role_bindings: tuple[ModelRoleBindingV2, ...]
    owner_id: str
    host_fingerprint: str
    process_id: int
    started_at: datetime

    def __post_init__(self) -> None:
        run_id = self.run_spec.run_id
        if (
            self.preflight_record.run_id != run_id
            or self.approval.scope_kind != BudgetScopeKindV2.RUN
            or self.approval.scope_id != run_id
            or self.budget.scope_kind != BudgetScopeKindV2.RUN
            or self.budget.scope_id != run_id
            or self.budget.approval_id != self.approval.approval_id
        ):
            raise ValueError("live native control run, approval, and budget scopes do not close")
        if self.preflight_record.run_spec_hash != canonical_sha256(self.run_spec):
            raise ValueError("live native control preflight does not bind its run spec")
        if self.preflight_record.approval_hash != self.approval.approval_hash:
            raise ValueError("live native control preflight does not bind its approval")
        if self.preflight_record.budget_hash != canonical_sha256(self.budget):
            raise ValueError("live native control preflight does not bind its budget")
        if self.preflight_record.runtime_binding_hash != self.run_spec.runtime_binding_hash:
            raise ValueError("live native control runtime binding does not close")
        role_ids = tuple(item.binding_id for item in self.role_bindings)
        if (
            not role_ids
            or len(set(role_ids)) != len(role_ids)
            or role_ids != self.run_spec.model_role_binding_ids
            or role_ids != self.preflight_record.role_binding_ids
            or role_ids != self.approval.role_binding_ids
            or set(item.role_binding_id for item in self.budget.role_ceilings) != set(role_ids)
        ):
            raise ValueError("live native control role inventory does not close")
        if not self.owner_id or not self.host_fingerprint or self.process_id <= 0:
            raise ValueError("live native control requires concrete owner and host identity")
        if self.started_at.tzinfo is None or self.started_at.utcoffset() is None:
            raise ValueError("live native control start time requires an explicit timezone")
        if not (
            self.approval.approved_at <= self.started_at < self.approval.expires_at
            and self.preflight_record.observed_at <= self.started_at
        ):
            raise ValueError("live native control is outside its approval/preflight time window")


_NativeErrorCategory: TypeAlias = Literal[
    "assertion",
    "base",
    "cancelled",
    "memory_cancelled_before_dispatch",
    "memory_cancelled_unknown",
    "memory_unknown",
    "model_cancelled_unknown",
    "model_unknown",
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
    owner_id: str = NATIVE_OWNER_ID
    budget_id: str = "native-fixture-budget-v1"
    budget_scope_id: str | None = None
    started_at: datetime = NATIVE_FIXTURE_STARTED_AT
    live_timing: bool = False
    monotonic_started: float | None = None
    sequence: int = 0

    def timestamp(self) -> datetime:
        self.sequence += 1
        deterministic_floor = self.started_at + timedelta(microseconds=self.sequence)
        return (
            max(datetime.now(UTC), deterministic_floor) if self.live_timing else deterministic_floor
        )

    def monotonic(self) -> Decimal:
        if self.live_timing:
            if self.monotonic_started is None:
                raise RuntimeError("live native timing has no monotonic origin")
            return Decimal(str(time.monotonic() - self.monotonic_started))
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
    role_binding_id: str


def _new_execution_state(
    request: _NativeRunRequest,
    store: CapsuleArtifactStorePort,
    *,
    sequence: int = 0,
) -> _NativeExecutionState:
    control = request.control
    return _NativeExecutionState(
        store=store,
        run_id=request.run_id,
        lease_record_hash=_native_run_lease(
            request.run_id,
            request.adapter_profile_id,
            control=control,
        ).lease_record_hash,
        close_timeout_seconds=request.close_timeout_seconds,
        owner_id=control.owner_id if control is not None else NATIVE_OWNER_ID,
        budget_id=(control.budget.budget_id if control is not None else "native-fixture-budget-v1"),
        budget_scope_id=control.run_spec.run_id if control is not None else None,
        started_at=control.started_at if control is not None else NATIVE_FIXTURE_STARTED_AT,
        live_timing=control is not None,
        monotonic_started=time.monotonic() if control is not None else None,
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
) -> NativeRunArtifacts:
    """Compose one deterministic native capsule without selecting any live transport."""

    if control is not None:
        raise ValueError("live native execution requires the lifecycle-aware composition root")
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
    return _run_native_supervised(
        _NativeRunRequest(
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
        )
    )


def _run_native_supervised(request: _NativeRunRequest) -> NativeRunArtifacts:
    try:
        context = multiprocessing.get_context("fork")
    except ValueError as exc:
        raise NativeRunProcessError(
            "native composition requires an operating-system fork boundary"
        ) from exc
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_native_process_entry,
        args=(request, sender),
        name=f"oamb-native-{request.run_id}",
        daemon=False,
    )
    started = False
    active_close: _NativeCloseStarted | None = None
    latest_progress: _NativeRunProgress | None = None
    close_deadline: float | None = None
    try:
        process.start()
        started = True
        sender.close()
        while True:
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
                    _raise_after_process_abort(request, latest_progress, active_close, error)
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
                            request,
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
                            request,
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
                timeout_errors = _seal_supervised_close_abort(request, active_close)
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
                    request,
                    latest_progress,
                    active_close,
                    process_error,
                )
    finally:
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


def _native_process_entry(request: _NativeRunRequest, sender: Connection) -> None:
    try:
        ready = asyncio.run(
            _run_native_vertical_slice(
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
                lifecycle_sender=sender,
            )
        )
    except BaseException as exc:
        sender.send(_NativeRunFailed(_native_error_envelope(exc)))
    else:
        sender.send(ready)
    finally:
        sender.close()


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
    if isinstance(error, MemorySystemCallCancelledBeforeDispatch):
        category: _NativeErrorCategory = "memory_cancelled_before_dispatch"
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
    lifecycle_sender: Connection | None = None,
) -> _NativeRunReady:
    capsule_root = output_root / run_id
    store = artifact_store_factory(capsule_root)
    dataset_manifest = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset_manifest)
    ingestion_plans = workload.iter_ingestion_plans(case_manifest)
    case_plans = workload.iter_case_plans(case_manifest)
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
    )
    lease = _native_run_lease(run_id, adapter_profile_id, control=control)
    state = _new_execution_state(request_identity, store)
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
        answer_model = model_factory(store)
        if judge_model_factory is not None:
            judge_model = judge_model_factory(store)
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
            ingestion_occurrence_id(run_id, runtime.memory_system_id, plan.ingestion_plan_id)
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
        plan_records, scopes = await _execute_ingestion_plans(
            state=state,
            memory=memory,
            plans=ingestion_plans,
            case_plans=case_plans,
            memory_system_id=runtime.memory_system_id,
            runtime_binding_hash=runtime.runtime_binding_hash,
            adapter_profile_id=adapter_profile_id,
        )
        case_records = await _execute_cases(
            state=state,
            workload=workload,
            memory=memory,
            answer_model=answer_model,
            judge_model=judge_model,
            plans=ingestion_plans,
            case_plans=case_plans,
            scopes=scopes,
            memory_system_id=runtime.memory_system_id,
            adapter_profile_id=adapter_profile_id,
            visible_evidence_policy=visible_evidence_policy,
            answer_role_binding_id=answer_role_binding_id,
            judge_role_binding_id=judge_role_binding_id,
        )
    except BaseException as exc:
        execution_error = exc
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
    terminal_errors = (
        *((execution_error,) if execution_error is not None else ()),
        *close_errors,
    )
    if terminal_errors:
        diagnostic_error: BaseException | None = None
        try:
            _seal_native_run_record(
                state,
                run_spec_hash=run_spec_hash,
                run_state=RunState.ABORTED,
                ingestion_occurrence_ids=ingestion_occurrence_ids,
                case_occurrence_ids=case_occurrence_ids,
            )
            store.finalize_capsule(run_id=run_id, run_spec_hash=run_spec_hash)
        except BaseException as exc:
            diagnostic_error = exc
        if diagnostic_error is not None:
            terminal_errors = (*terminal_errors, diagnostic_error)
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


async def _execute_ingestion_plans(
    *,
    state: _NativeExecutionState,
    memory: MemorySystemPort,
    plans: tuple[IngestionPlan, ...],
    case_plans: tuple[CasePlan, ...],
    memory_system_id: str,
    runtime_binding_hash: str,
    adapter_profile_id: str,
) -> tuple[tuple[NativeIngestionPlanRecord, ...], dict[str, ScopeReceipt]]:
    case_ids = {case.case_manifest_entry_id for case in case_plans}
    records: list[NativeIngestionPlanRecord] = []
    scopes: dict[str, ScopeReceipt] = {}
    for plan in plans:
        if not set(plan.ordered_case_manifest_entry_ids) <= case_ids:
            raise ValueError("native ingestion plan references an unknown case")
        occurrence_id = ingestion_occurrence_id(
            state.run_id,
            memory_system_id,
            plan.ingestion_plan_id,
        )
        scope = await memory.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id=occurrence_id,
                ingestion_plan_id=plan.ingestion_plan_id,
            )
        )
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

        dispatch_receipts = []
        dispatch_attempt_ids: list[str] = []
        usage_record_ids: list[str] = []
        resource_record_ids: list[str] = []
        cost_record_ids: list[str] = []
        for dispatch in dispatches:
            dispatch_attempt_id = attempt_id(
                occurrence_id,
                "memory_ingest",
                dispatch.dispatch_ordinal_1_indexed,
                dispatch.request_fingerprint,
            )
            prepared = _prepare_native_attempt(
                state,
                attempt_identity=dispatch_attempt_id,
                parent_kind="ingestion_plan",
                parent_id=occurrence_id,
                stage="memory_ingest",
                ordinal=dispatch.dispatch_ordinal_1_indexed,
                request_fingerprint=dispatch.request_fingerprint,
                role_binding_id=adapter_profile_id,
            )
            started_at = state.timestamp()
            try:
                receipt = await memory.ingest(
                    IngestionDispatchRequest(
                        scope=scope,
                        attempt_id=dispatch_attempt_id,
                        dispatch=dispatch,
                    )
                )
                if receipt.attempt_id != dispatch_attempt_id or receipt.dispatch != dispatch:
                    raise ValueError("native ingestion receipt does not bind its dispatch")
            except BaseException as exc:
                _seal_native_failure_preserving(
                    state,
                    prepared,
                    started_at=started_at,
                    error=exc,
                )
                raise
            ended_at = state.timestamp()
            _seal_native_success(
                state,
                prepared,
                started_at=started_at,
                ended_at=ended_at,
                raw_response_ref=receipt.raw_reference.sha256,
                index_contribution=IndexContribution.FINAL,
            )
            for usage in receipt.usage_records:
                _seal(state, "usage", usage.usage_record_id, usage)
                usage_record_ids.append(usage.usage_record_id)
            attempt_usage_ids, attempt_resource_id, attempt_cost_id = (
                _seal_native_attempt_accounting(
                    state,
                    prepared,
                    started_at=started_at,
                    ended_at=ended_at,
                    raw_response_ref=receipt.raw_reference.sha256,
                    usage_record_ids=tuple(
                        usage.usage_record_id for usage in receipt.usage_records
                    ),
                    indexing_view=IndexingView.FINAL_CONTRIBUTION,
                )
            )
            if not receipt.usage_records:
                usage_record_ids.extend(attempt_usage_ids)
            resource_record_ids.append(attempt_resource_id)
            cost_record_ids.append(attempt_cost_id)
            dispatch_attempt_ids.append(dispatch_attempt_id)
            dispatch_receipts.append(receipt)

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
        )
        readiness = await memory.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=ingestion_receipt.accepted_source_unit_ids,
                ingestion_receipt=ingestion_receipt,
            )
        )
        if not readiness.ready:
            raise ValueError("native fixture ingestion did not reach readiness")
        projection = await memory.project(scope)
        _require_projection_occurrence(projection, occurrence_id)
        if (
            adapter_profile_id != "mem0-rest-v1"
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
            readiness_evidence_refs=tuple(item.sha256 for item in readiness.evidence_references),
            inventory_raw_ref=projection.inventory.raw_reference.sha256,
            projected_source_unit_ids=projection.inventory.ordered_source_unit_ids,
            projection_raw_refs=_projection_raw_refs(projection),
            protected_state_sha256=projection.state_digest.state_sha256,
            attempt_ids=tuple(dispatch_attempt_ids),
            usage_record_ids=tuple(usage_record_ids),
            resource_record_ids=tuple(resource_record_ids),
            cost_record_ids=tuple(cost_record_ids),
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


async def _execute_cases(
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
        ingestion_occurrence = ingestion_occurrence_id(
            state.run_id,
            memory_system_id,
            selected_plan.ingestion_plan_id,
        )
        case_occurrence = case_occurrence_id(
            ingestion_occurrence,
            case_plan.case_manifest_entry_id,
        )
        scope = scopes[selected_plan.ingestion_plan_id]
        query = workload.render_retrieval_query(case_plan)
        query_fingerprint = canonical_sha256(
            ["oamb-native-query-v1", case_occurrence, hashlib.sha256(query).hexdigest()]
        )
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
                request=_retrieval_request(scope, case_occurrence, case_plan, query),
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
        answer_attempt_id = attempt_id(case_occurrence, "answer", 1, messages_sha256)
        prepared_answer = _prepare_native_attempt(
            state,
            attempt_identity=answer_attempt_id,
            parent_kind="case",
            parent_id=case_occurrence,
            stage="answer",
            ordinal=1,
            request_fingerprint=messages_sha256,
            role_binding_id=answer_role_binding_id,
            maximum_output_tokens=case_plan.answer_max_output_tokens,
        )
        answer_started_at = state.timestamp()
        try:
            answer_receipt = await answer_model.complete(
                ModelRequest(
                    attempt_id=answer_attempt_id,
                    parent_kind="case",
                    parent_id=case_occurrence,
                    stage="answer",
                    role_binding_id=answer_role_binding_id,
                    messages_sha256=messages_sha256,
                    messages=messages,
                    output_contract_id=case_plan.output_contract_id,
                    max_output_tokens=case_plan.answer_max_output_tokens,
                )
            )
        except BaseException as exc:
            _seal_native_failure_preserving(
                state,
                prepared_answer,
                started_at=answer_started_at,
                error=exc,
            )
            raise
        answer_ended_at = state.timestamp()
        _seal_native_success(
            state,
            prepared_answer,
            started_at=answer_started_at,
            ended_at=answer_ended_at,
            raw_response_ref=answer_receipt.raw_reference.sha256,
            index_contribution=IndexContribution.NOT_APPLICABLE,
        )
        answer_usage_ids, answer_resource_id, answer_cost_id = _seal_native_attempt_accounting(
            state,
            prepared_answer,
            started_at=answer_started_at,
            ended_at=answer_ended_at,
            raw_response_ref=answer_receipt.raw_reference.sha256,
            usage_record_ids=answer_receipt.usage_reference_ids,
            indexing_view=IndexingView.NOT_APPLICABLE,
        )
        parsed_answer = answer_receipt.output_text.encode("utf-8")
        answer_value = AnswerValue(
            raw_reference=answer_receipt.raw_reference,
            raw_answer=parsed_answer,
            parsed_value=parsed_answer,
            parsed_value_sha256=hashlib.sha256(parsed_answer).hexdigest(),
        )
        evaluation = workload.evaluate(case_plan, answer_value)
        case_attempt_ids = [query_attempt_id, answer_attempt_id]
        usage_record_ids = [*query_usage_ids, *answer_usage_ids]
        resource_record_ids = [query_resource_id, answer_resource_id]
        cost_record_ids = [query_cost_id, answer_cost_id]
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
            judge_attempt_id = attempt_id(
                case_occurrence,
                "judge",
                1,
                judge_messages_sha256,
            )
            prepared_judge = _prepare_native_attempt(
                state,
                attempt_identity=judge_attempt_id,
                parent_kind="case",
                parent_id=case_occurrence,
                stage="judge",
                ordinal=1,
                request_fingerprint=judge_messages_sha256,
                role_binding_id=case_plan.judge_binding_id,
                maximum_output_tokens=NATIVE_JUDGE_MAX_OUTPUT_TOKENS,
            )
            judge_started_at = state.timestamp()
            try:
                judge_receipt = await judge_model.complete(
                    ModelRequest(
                        attempt_id=judge_attempt_id,
                        parent_kind="case",
                        parent_id=case_occurrence,
                        stage="judge",
                        role_binding_id=judge_role_binding_id,
                        messages_sha256=judge_messages_sha256,
                        messages=judge_messages,
                        output_contract_id=evaluation.output_contract_id,
                        max_output_tokens=NATIVE_JUDGE_MAX_OUTPUT_TOKENS,
                    )
                )
            except BaseException as exc:
                _seal_native_failure_preserving(
                    state,
                    prepared_judge,
                    started_at=judge_started_at,
                    error=exc,
                )
                raise
            judge_ended_at = state.timestamp()
            _seal_native_success(
                state,
                prepared_judge,
                started_at=judge_started_at,
                ended_at=judge_ended_at,
                raw_response_ref=judge_receipt.raw_reference.sha256,
                index_contribution=IndexContribution.NOT_APPLICABLE,
            )
            judge_usage_ids, judge_resource_id, judge_cost_id = _seal_native_attempt_accounting(
                state,
                prepared_judge,
                started_at=judge_started_at,
                ended_at=judge_ended_at,
                raw_response_ref=judge_receipt.raw_reference.sha256,
                usage_record_ids=judge_receipt.usage_reference_ids,
                indexing_view=IndexingView.NOT_APPLICABLE,
            )
            judge_bytes = judge_receipt.output_text.encode("utf-8")
            judge_answer = AnswerValue(
                raw_reference=judge_receipt.raw_reference,
                raw_answer=judge_bytes,
                parsed_value=judge_bytes,
                parsed_value_sha256=hashlib.sha256(judge_bytes).hexdigest(),
            )
            evaluation = workload.finalize_judge(case_plan, answer_value, judge_answer)
            case_attempt_ids.append(judge_attempt_id)
            usage_record_ids.extend(judge_usage_ids)
            resource_record_ids.append(judge_resource_id)
            cost_record_ids.append(judge_cost_id)
            evaluation_disposition = CaseEvaluationDisposition.JUDGED
        evaluation_raw_ref = _seal_deterministic_evaluation(
            state.store,
            evaluation=evaluation,
            parsed_answer_sha256=answer_value.parsed_value_sha256,
        )
        if evaluation.numerator is None or evaluation.denominator is None:
            raise ValueError("native deterministic evaluation requires an exact fraction")
        record = CaseRecordV3(
            case_occurrence_id=case_occurrence,
            run_id=state.run_id,
            ingestion_occurrence_id=ingestion_occurrence,
            case_manifest_entry_id=case_plan.case_manifest_entry_id,
            adapter_profile_id=adapter_profile_id,
            state=CaseState.COMPLETED,
            retrieval_raw_ref=query_receipt.native_batch.raw_reference.sha256,
            retrieval_supporting_raw_refs=tuple(
                item.sha256 for item in query_receipt.native_batch.supporting_raw_references
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
        (
            "source/specs/external-call-approval.json",
            control.approval.approval_hash,
            control.approval,
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
    maximum_output_tokens: int = 0,
) -> _PreparedNativeAttempt:
    claimed_at = state.timestamp()
    claim_fields = {
        "occurrence_id": parent_id,
        "lease_record_hash": state.lease_record_hash,
        "lease_epoch": 1,
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
        role_binding_id=role_binding_id,
    )


def _seal_native_success(
    state: _NativeExecutionState,
    prepared: _PreparedNativeAttempt,
    *,
    started_at: datetime,
    ended_at: datetime,
    raw_response_ref: str,
    index_contribution: IndexContribution,
) -> AttemptRecordV2:
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
    terminal = AttemptRecordV2(
        attempt_id=prepared.attempt_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        stage=prepared.stage,
        ordinal=prepared.ordinal,
        request_fingerprint=prepared.request_fingerprint,
        started_at=started_at,
        ended_at=ended_at,
        outcome=AttemptOutcome.SUCCEEDED,
        retry_of_attempt_id=None,
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
) -> tuple[tuple[str, ...], str, str]:
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
    return closed_usage_ids, resource_record_id, cost_record_id


def _seal_native_failure_preserving(
    state: _NativeExecutionState,
    prepared: _PreparedNativeAttempt,
    *,
    started_at: datetime,
    error: BaseException,
) -> AttemptRecordV2:
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
) -> AttemptRecordV2:
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
        raw_response_ref=None,
        raw_error_ref=raw_error_ref,
        dispatch_started_at=started_at,
        receipt_observed_at=ended_at,
        provider_request_wall_seconds=_elapsed_seconds(started_at, ended_at),
    )
    _seal(state, "attempt-receipts", prepared.attempt_id, receipt)
    terminal = AttemptRecordV2(
        attempt_id=prepared.attempt_id,
        parent_kind=prepared.parent_kind,
        parent_id=prepared.parent_id,
        stage=prepared.stage,
        ordinal=prepared.ordinal,
        request_fingerprint=prepared.request_fingerprint,
        started_at=started_at,
        ended_at=ended_at,
        outcome=outcome,
        retry_of_attempt_id=None,
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
        if isinstance(current, MemorySystemCallCancelledBeforeDispatch):
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
    run_state: Literal[RunState.FINALIZED, RunState.ABORTED],
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
        acquired_at=control.started_at if control is not None else NATIVE_FIXTURE_STARTED_AT,
    )
    return candidate.model_copy(
        update={
            "lease_record_hash": canonical_sha256(
                candidate.model_dump(mode="python", exclude={"lease_record_hash"})
            )
        }
    )


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


def _require_safe_component(value: str, label: str) -> None:
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{label} must be one safe path component")
