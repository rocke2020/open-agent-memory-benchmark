"""Credential-free end-to-end execution through the real runtime and artifact store."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol

from oamb.contracts.accounting import (
    ProofStatus,
    TokenDomain,
    TokenMeasurementSource,
    TokenStage,
    TokenUsageRecord,
)
from oamb.contracts.base import StrictContract
from oamb.contracts.evidence import (
    AttemptRecord,
    CapsuleManifest,
    CaseEvaluationDisposition,
    CaseRecordV2,
    CloseErrorRecord,
    IngestionPlanRecord,
    LogicalContextRecord,
    OriginClass,
    OriginRecord,
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
    IngestionPlan,
    IngestionRequest,
    JudgeRequest,
    MemorySystemPort,
    ModelCallFailure,
    ModelClientPort,
    ModelRequest,
    RawPayloadSealRequest,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
    ScopeReceipt,
    VisibleEvidencePolicy,
    WorkloadPort,
)
from oamb.contracts.specifications import BudgetScopeKind, BudgetSpec, RunSpec
from oamb.contracts.states import (
    AttemptOutcome,
    CaseState,
    IndexContribution,
    IngestionPlanState,
    ResumeDisposition,
    RunState,
)
from oamb.runtime.preflight import (
    AdapterProfileDescriptor,
    ArtifactDurabilityPreflight,
    GateStatus,
    OperationKind,
    PreflightRejected,
    ProviderGateClosure,
    ResolutionStatus,
    ResolvedRunPlan,
    RoleSlot,
    RoleSlotName,
    RunPreflightRequest,
    TransportKind,
    resolve_run_plan,
)
from oamb.runtime.runner import (
    CaseTask,
    IngestionPlanUnavailable,
    IngestionTask,
    SerialRunner,
    SerialRunResult,
    UnknownExternalOutcome,
)
from oamb.runtime.shutdown import ShutdownCoordinator, ShutdownHooks
from oamb.runtime.source_records import seal_source_contract

FAKE_MEMORY_SYSTEM_ID = "fake-memory"
FAKE_RUNTIME_BINDING_HASH = canonical_sha256(["oamb-fake-runtime-v1"])
FAKE_MAX_READINESS_CHECKS = 2
FAKE_MAX_MODEL_ATTEMPTS = 2
FAKE_STARTED_AT = datetime(2026, 1, 1, tzinfo=UTC)


class FakeRunScenario(StrEnum):
    STANDARD = "standard"
    CANCELLED = "cancelled"
    UNKNOWN_OUTCOME = "unknown_outcome"


class CapsuleArtifactStorePort(ArtifactStorePort, Protocol):
    def finalize_capsule(self, *, run_id: str, run_spec_hash: str) -> CapsuleManifest: ...


@dataclass(frozen=True, slots=True)
class FakeRunArtifacts:
    capsule_root: Path
    manifest: CapsuleManifest
    run_record: RunRecord


@dataclass(slots=True)
class _ExecutionState:
    store: CapsuleArtifactStorePort
    workload: WorkloadPort
    memory: MemorySystemPort
    model: ModelClientPort
    run_id: str
    scenario: FakeRunScenario
    sequence: int = 0

    def timestamp(self) -> datetime:
        self.sequence += 1
        return FAKE_STARTED_AT + timedelta(seconds=self.sequence)


@dataclass(slots=True)
class _ClientLifecycle:
    memory: MemorySystemPort | None = None
    model: ModelClientPort | None = None
    shutdown_started: bool = False

    async def close_before_shutdown(self) -> tuple[BaseException, ...]:
        errors: list[BaseException] = []
        for client in (self.model, self.memory):
            if client is None:
                continue
            try:
                await client.close()
            except BaseException as exc:
                errors.append(exc)
        return tuple(errors)


def run_fake_vertical_slice(
    *,
    output_root: Path,
    run_id: str,
    workload: WorkloadPort,
    artifact_store_factory: Callable[[Path], CapsuleArtifactStorePort],
    memory_factory: Callable[[ArtifactStorePort, tuple[IngestionPlan, ...]], MemorySystemPort],
    model_factory: Callable[[ArtifactStorePort], ModelClientPort],
    scenario: FakeRunScenario = FakeRunScenario.STANDARD,
) -> FakeRunArtifacts:
    """Run deterministic fakes and seal one immutable source capsule."""

    _require_safe_run_id(run_id)
    return asyncio.run(
        _run_fake_vertical_slice(
            output_root=Path(output_root),
            run_id=run_id,
            workload=workload,
            artifact_store_factory=artifact_store_factory,
            memory_factory=memory_factory,
            model_factory=model_factory,
            scenario=scenario,
        )
    )


async def _run_fake_vertical_slice(
    *,
    output_root: Path,
    run_id: str,
    workload: WorkloadPort,
    artifact_store_factory: Callable[[Path], CapsuleArtifactStorePort],
    memory_factory: Callable[[ArtifactStorePort, tuple[IngestionPlan, ...]], MemorySystemPort],
    model_factory: Callable[[ArtifactStorePort], ModelClientPort],
    scenario: FakeRunScenario,
) -> FakeRunArtifacts:
    lifecycle = _ClientLifecycle()

    def tracked_memory_factory(
        store: ArtifactStorePort,
        plans: tuple[IngestionPlan, ...],
    ) -> MemorySystemPort:
        lifecycle.memory = memory_factory(store, plans)
        return lifecycle.memory

    def tracked_model_factory(store: ArtifactStorePort) -> ModelClientPort:
        lifecycle.model = model_factory(store)
        return lifecycle.model

    try:
        return await _run_fake_vertical_slice_impl(
            output_root=output_root,
            run_id=run_id,
            workload=workload,
            artifact_store_factory=artifact_store_factory,
            memory_factory=tracked_memory_factory,
            model_factory=tracked_model_factory,
            scenario=scenario,
            lifecycle=lifecycle,
        )
    except BaseException as primary_error:
        if lifecycle.shutdown_started:
            raise
        close_errors = await lifecycle.close_before_shutdown()
        if close_errors:
            raise BaseExceptionGroup(
                "fake execution and early client close failed",
                (primary_error, *close_errors),
            ) from None
        raise


async def _run_fake_vertical_slice_impl(
    *,
    output_root: Path,
    run_id: str,
    workload: WorkloadPort,
    artifact_store_factory: Callable[[Path], CapsuleArtifactStorePort],
    memory_factory: Callable[[ArtifactStorePort, tuple[IngestionPlan, ...]], MemorySystemPort],
    model_factory: Callable[[ArtifactStorePort], ModelClientPort],
    scenario: FakeRunScenario,
    lifecycle: _ClientLifecycle,
) -> FakeRunArtifacts:
    _require_safe_run_id(run_id)
    capsule_root = output_root / run_id
    store = artifact_store_factory(capsule_root)
    resolve_fake_preflight(capsule_root, run_id)
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    ingestion_plans = workload.iter_ingestion_plans(case_manifest)
    case_plans = workload.iter_case_plans(case_manifest)
    memory = memory_factory(store, ingestion_plans)
    model = model_factory(store)
    state = _ExecutionState(store, workload, memory, model, run_id, scenario)

    runtime = await memory.resolve()
    run_spec = build_fake_run_spec(
        run_id=run_id,
        dataset_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        runtime_binding_hash=runtime.runtime_binding_hash,
        workload_id=case_manifest.workload_id,
    )
    run_spec_hash = canonical_sha256(run_spec)
    budget = build_fake_budget(run_id)
    origin = OriginRecord(
        origin_id=f"{run_id}-generated-origin",
        origin_class=OriginClass.OAMB_NATIVE,
        producer="oamb-generated-fake-v1",
        source_sha256=dataset.source_files[0].sha256,
        license_id="CC0-1.0",
    )
    _seal(state, "specs", "dataset-manifest", dataset)
    _seal(state, "specs", "case-manifest", case_manifest)
    _seal(state, "specs", "run-spec", run_spec)
    _seal(state, "specs", "budget", budget)
    _seal(state, "origins", origin.origin_id, origin)

    cases_by_context: dict[str, list[str]] = {}
    for case in case_manifest.cases:
        cases_by_context.setdefault(case.context_manifest_entry_id, []).append(
            case.case_manifest_entry_id
        )
    logical_records = tuple(
        LogicalContextRecord(
            context_manifest_entry_id=context.context_manifest_entry_id,
            run_id=run_id,
            context_content_id=context.context_content_id,
            source_file_sha256=context.source_file_sha256,
            source_row_number_1_indexed=context.source_row_number_1_indexed,
            context_bytes_sha256=context.context_bytes_sha256,
            ordered_case_manifest_entry_ids=tuple(
                cases_by_context[context.context_manifest_entry_id]
            ),
            origin_id=origin.origin_id,
        )
        for context in case_manifest.logical_contexts
    )
    for logical_record in logical_records:
        _seal(
            state,
            "logical-contexts",
            logical_record.context_manifest_entry_id,
            logical_record,
        )

    ingestion_occurrences = {
        plan.ingestion_plan_id: ingestion_occurrence_id(
            run_id,
            FAKE_MEMORY_SYSTEM_ID,
            plan.ingestion_plan_id,
        )
        for plan in ingestion_plans
    }
    plan_by_case = {
        case_id: plan.ingestion_plan_id
        for plan in ingestion_plans
        for case_id in plan.ordered_case_manifest_entry_ids
    }
    case_occurrences = {
        case.case_manifest_entry_id: case_occurrence_id(
            ingestion_occurrences[plan_by_case[case.case_manifest_entry_id]],
            case.case_manifest_entry_id,
        )
        for case in case_plans
    }
    scopes: dict[str, ScopeReceipt] = {}
    plan_records: dict[str, IngestionPlanRecord] = {}
    case_records: dict[str, CaseRecordV2] = {}
    runner = SerialRunner()

    async def execute_plan(plan_index: int) -> None:
        plan = ingestion_plans[plan_index]
        occurrence_id = ingestion_occurrences[plan.ingestion_plan_id]
        if scenario == FakeRunScenario.UNKNOWN_OUTCOME and plan_index == 0:
            request_fingerprint = canonical_sha256(
                ["oamb-fake-ingest-request-v1", plan.ingestion_plan_id]
            )
            unknown_attempt_id = attempt_id(
                occurrence_id,
                "memory_ingest",
                1,
                request_fingerprint,
            )
            unknown_attempt = AttemptRecord(
                attempt_id=unknown_attempt_id,
                parent_kind="ingestion_plan",
                parent_id=occurrence_id,
                stage="memory_ingest",
                ordinal=1,
                request_fingerprint=request_fingerprint,
                started_at=state.timestamp(),
                ended_at=state.timestamp(),
                outcome=AttemptOutcome.UNKNOWN_OUTCOME,
                retry_of_attempt_id=None,
                idempotency_key_hash=None,
                reconciliation_capability="none",
                raw_response_ref=None,
                raw_error_ref=None,
                index_contribution=IndexContribution.NONE,
                superseded_by_attempt_id=None,
            )
            _seal(state, "attempts", unknown_attempt_id, unknown_attempt)
            unknown_plan = _terminal_plan_record(
                run_id=run_id,
                plan=plan,
                occurrence_id=occurrence_id,
                case_occurrences=case_occurrences,
                state=IngestionPlanState.INTERRUPTED_UNKNOWN_OUTCOME,
            ).model_copy(update={"attempt_ids": (unknown_attempt_id,)})
            plan_records[plan.ingestion_plan_id] = unknown_plan
            _seal(state, "ingestion-plans", occurrence_id, unknown_plan)
            raise UnknownExternalOutcome("scripted lost ingest receipt")
        scope = await memory.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id=occurrence_id,
                ingestion_plan_id=plan.ingestion_plan_id,
            )
        )
        scopes[plan.ingestion_plan_id] = scope
        request_fingerprint = canonical_sha256(
            ["oamb-fake-ingest-request-v1", plan.ingestion_plan_id]
        )
        ingest_attempt_id = attempt_id(occurrence_id, "memory_ingest", 1, request_fingerprint)
        started = state.timestamp()
        receipt = await memory.ingest(
            IngestionRequest(scope=scope, ordered_source_units=plan.ordered_source_units)
        )
        ended = state.timestamp()
        index_contribution = (
            IndexContribution.FINAL
            if not receipt.rejected_source_unit_ids
            else IndexContribution.NONE
        )
        attempt = AttemptRecord(
            attempt_id=ingest_attempt_id,
            parent_kind="ingestion_plan",
            parent_id=occurrence_id,
            stage="memory_ingest",
            ordinal=1,
            request_fingerprint=request_fingerprint,
            started_at=started,
            ended_at=ended,
            outcome=AttemptOutcome.SUCCEEDED,
            retry_of_attempt_id=None,
            idempotency_key_hash=None,
            reconciliation_capability="none",
            raw_response_ref=receipt.raw_references[0].sha256,
            raw_error_ref=None,
            index_contribution=index_contribution,
            superseded_by_attempt_id=None,
        )
        _seal(state, "attempts", ingest_attempt_id, attempt)
        usage_id = _seal_fake_usage(
            state,
            attempt_id_value=ingest_attempt_id,
            parent_kind="ingestion_plan",
            parent_id=occurrence_id,
            stage=TokenStage.MEMORY_INGEST,
            operation_kind="fake_memory_ingest",
            input_tokens=sum(len(item.payload_bytes.split()) for item in plan.ordered_source_units),
            output_tokens=0,
            raw_response_ref=receipt.raw_references[0].sha256,
        )
        readiness_refs: list[str] = []
        ready = False
        for _check in range(FAKE_MAX_READINESS_CHECKS):
            readiness = await memory.wait_ready(
                ReadinessRequest(
                    scope=scope,
                    expected_source_unit_ids=receipt.accepted_source_unit_ids,
                )
            )
            readiness_refs.extend(item.sha256 for item in readiness.evidence_references)
            if readiness.ready:
                ready = True
                break
        record = IngestionPlanRecord(
            ingestion_occurrence_id=occurrence_id,
            run_id=run_id,
            memory_system_id=FAKE_MEMORY_SYSTEM_ID,
            ingestion_plan_id=plan.ingestion_plan_id,
            ordered_member_context_manifest_entry_ids=(
                plan.ordered_member_context_manifest_entry_ids
            ),
            ordered_case_occurrence_ids=tuple(
                case_occurrences[case_id] for case_id in plan.ordered_case_manifest_entry_ids
            ),
            state=IngestionPlanState.SEALED if ready else IngestionPlanState.ERROR,
            intended_source_count=plan.intended_source_count,
            accepted_source_count=len(receipt.accepted_source_unit_ids),
            failed_source_count=len(receipt.rejected_source_unit_ids),
            readiness_evidence_refs=tuple(readiness_refs),
            attempt_ids=(ingest_attempt_id,),
            usage_record_ids=(usage_id,),
            resource_record_ids=(),
            cost_record_ids=(),
        )
        plan_records[plan.ingestion_plan_id] = record
        _seal(state, "ingestion-plans", occurrence_id, record)
        if scenario == FakeRunScenario.CANCELLED and plan_index == 0:
            runner.request_cancel()
        if not ready:
            raise IngestionPlanUnavailable("fake readiness did not close")

    async def execute_case(case_plan: CasePlan) -> None:
        plan_id = plan_by_case[case_plan.case_manifest_entry_id]
        occurrence_id = case_occurrences[case_plan.case_manifest_entry_id]
        ingestion_occurrence = ingestion_occurrences[plan_id]
        scope = scopes[plan_id]
        query = workload.render_retrieval_query(case_plan)
        query_request_hash = canonical_sha256(
            ["oamb-fake-query-request-v1", occurrence_id, hashlib.sha256(query).hexdigest()]
        )
        query_attempt_id = attempt_id(occurrence_id, "memory_query", 1, query_request_hash)
        query_started = state.timestamp()
        native = await memory.retrieve(
            RetrievalRequest(
                scope=scope,
                case_occurrence_id=occurrence_id,
                query_bytes=query,
                top_k=3,
            )
        )
        query_ended = state.timestamp()
        query_attempt = AttemptRecord(
            attempt_id=query_attempt_id,
            parent_kind="case",
            parent_id=occurrence_id,
            stage="memory_query",
            ordinal=1,
            request_fingerprint=query_request_hash,
            started_at=query_started,
            ended_at=query_ended,
            outcome=AttemptOutcome.SUCCEEDED,
            retry_of_attempt_id=None,
            idempotency_key_hash=None,
            reconciliation_capability="none",
            raw_response_ref=native.raw_reference.sha256,
            raw_error_ref=None,
            index_contribution=IndexContribution.NOT_APPLICABLE,
            superseded_by_attempt_id=None,
        )
        _seal(state, "attempts", query_attempt_id, query_attempt)
        _seal_not_applicable_query_usage(
            state,
            query_attempt_id,
            occurrence_id,
            native.raw_reference.sha256,
        )
        visible = workload.build_visible_evidence(
            native,
            VisibleEvidencePolicy(max_items=3, max_characters=256, max_tokens=64),
        )
        _seal_context_usage(state, query_attempt_id, occurrence_id, visible.canonical_bytes)
        prompt = workload.render_answer(case_plan, visible)
        messages = (("user", prompt.canonical_bytes.decode("utf-8")),)
        messages_hash = canonical_sha256(messages)
        answer_attempt_ids: list[str] = []
        answer_raw_ref: str | None = None
        answer_value: AnswerValue | None = None
        retry_of: str | None = None
        for ordinal in range(1, FAKE_MAX_MODEL_ATTEMPTS + 1):
            current_attempt_id = attempt_id(occurrence_id, "answer", ordinal, messages_hash)
            answer_attempt_ids.append(current_attempt_id)
            started = state.timestamp()
            try:
                model_receipt = await model.complete(
                    ModelRequest(
                        attempt_id=current_attempt_id,
                        parent_kind="case",
                        parent_id=occurrence_id,
                        stage="answer",
                        role_binding_id="fake-answer-v1",
                        messages_sha256=messages_hash,
                        messages=messages,
                    )
                )
            except ModelCallFailure as exc:
                failed = AttemptRecord(
                    attempt_id=current_attempt_id,
                    parent_kind="case",
                    parent_id=occurrence_id,
                    stage="answer",
                    ordinal=ordinal,
                    request_fingerprint=messages_hash,
                    started_at=started,
                    ended_at=state.timestamp(),
                    outcome=AttemptOutcome.FAILED,
                    retry_of_attempt_id=retry_of,
                    idempotency_key_hash=None,
                    reconciliation_capability="none",
                    raw_response_ref=None,
                    raw_error_ref=exc.raw_reference.sha256,
                    index_contribution=IndexContribution.NOT_APPLICABLE,
                    superseded_by_attempt_id=None,
                )
                _seal(state, "attempts", current_attempt_id, failed)
                if not exc.retryable or ordinal == FAKE_MAX_MODEL_ATTEMPTS:
                    raise
                retry_of = current_attempt_id
                continue
            succeeded = AttemptRecord(
                attempt_id=current_attempt_id,
                parent_kind="case",
                parent_id=occurrence_id,
                stage="answer",
                ordinal=ordinal,
                request_fingerprint=messages_hash,
                started_at=started,
                ended_at=state.timestamp(),
                outcome=AttemptOutcome.SUCCEEDED,
                retry_of_attempt_id=retry_of,
                idempotency_key_hash=None,
                reconciliation_capability="none",
                raw_response_ref=model_receipt.raw_reference.sha256,
                raw_error_ref=None,
                index_contribution=IndexContribution.NOT_APPLICABLE,
                superseded_by_attempt_id=None,
            )
            _seal(state, "attempts", current_attempt_id, succeeded)
            parsed = model_receipt.output_text.encode("utf-8")
            answer_raw_ref = model_receipt.raw_reference.sha256
            answer_value = AnswerValue(
                raw_reference=model_receipt.raw_reference,
                raw_answer=parsed,
                parsed_value=parsed,
                parsed_value_sha256=hashlib.sha256(parsed).hexdigest(),
            )
            break
        if answer_value is None or answer_raw_ref is None:
            raise RuntimeError("bounded fake answer retry ended without terminal evidence")
        evaluation = workload.evaluate(case_plan, answer_value)
        evaluation_ref: str | None
        error_stage: str | None = None
        case_attempt_ids = [query_attempt_id, *answer_attempt_ids]
        if isinstance(evaluation, JudgeRequest):
            judge_messages = (("user", evaluation.prompt.canonical_bytes.decode("utf-8")),)
            judge_request_hash = canonical_sha256(judge_messages)
            judge_attempt_id = attempt_id(occurrence_id, "judge", 1, judge_request_hash)
            case_attempt_ids.append(judge_attempt_id)
            judge_started = state.timestamp()
            try:
                await model.complete(
                    ModelRequest(
                        attempt_id=judge_attempt_id,
                        parent_kind="case",
                        parent_id=occurrence_id,
                        stage="judge",
                        role_binding_id=case_plan.judge_binding_id or "fake-judge-v1",
                        messages_sha256=judge_request_hash,
                        messages=judge_messages,
                    )
                )
            except ModelCallFailure as exc:
                judge_attempt = AttemptRecord(
                    attempt_id=judge_attempt_id,
                    parent_kind="case",
                    parent_id=occurrence_id,
                    stage="judge",
                    ordinal=1,
                    request_fingerprint=judge_request_hash,
                    started_at=judge_started,
                    ended_at=state.timestamp(),
                    outcome=AttemptOutcome.FAILED,
                    retry_of_attempt_id=None,
                    idempotency_key_hash=None,
                    reconciliation_capability="none",
                    raw_response_ref=None,
                    raw_error_ref=exc.raw_reference.sha256,
                    index_contribution=IndexContribution.NOT_APPLICABLE,
                    superseded_by_attempt_id=None,
                )
                _seal(state, "attempts", judge_attempt_id, judge_attempt)
                evaluation_ref = None
                error_stage = "judge"
            else:
                raise RuntimeError("scripted fake judge unexpectedly succeeded")
        else:
            evaluation_payload = canonical_json_bytes(
                {
                    "metric_id": evaluation.metric_id,
                    "result_sha256": evaluation.result_sha256,
                }
            )
            evaluation_ref = state.store.seal_raw(
                RawPayloadSealRequest(
                    sha256=hashlib.sha256(evaluation_payload).hexdigest(),
                    media_type="application/json",
                    compression="gzip",
                    payload_bytes=evaluation_payload,
                )
            ).sha256
        evaluation_disposition = (
            CaseEvaluationDisposition.UNJUDGED
            if error_stage == "judge"
            else CaseEvaluationDisposition.DETERMINISTIC_EVALUATED
        )
        record = CaseRecordV2(
            case_occurrence_id=occurrence_id,
            run_id=run_id,
            ingestion_occurrence_id=ingestion_occurrence,
            case_manifest_entry_id=case_plan.case_manifest_entry_id,
            state=CaseState.ERROR if error_stage == "judge" else CaseState.COMPLETED,
            retrieval_raw_ref=native.raw_reference.sha256,
            prompt_sha256=prompt.sha256,
            answer_raw_ref=answer_raw_ref,
            parsed_answer_sha256=answer_value.parsed_value_sha256,
            evaluation_raw_ref=evaluation_ref,
            evaluation_disposition=evaluation_disposition,
            attempt_ids=tuple(case_attempt_ids),
            error_stage=error_stage,
        )
        case_records[case_plan.case_manifest_entry_id] = record
        _seal(state, "cases", occurrence_id, record)

    def plan_operation(index: int) -> Callable[[], Awaitable[None]]:
        async def operation() -> None:
            await execute_plan(index)

        return operation

    def case_operation(case_plan: CasePlan) -> Callable[[], Awaitable[None]]:
        async def operation() -> None:
            await execute_case(case_plan)

        return operation

    execution_error: BaseException | None = None
    try:
        result = await runner.run(
            ingestion_tasks=tuple(
                IngestionTask(plan.ingestion_plan_id, plan_operation(index))
                for index, plan in enumerate(ingestion_plans)
            ),
            case_tasks=tuple(
                CaseTask(
                    case_plan.case_manifest_entry_id,
                    plan_by_case[case_plan.case_manifest_entry_id],
                    case_operation(case_plan),
                )
                for case_plan in case_plans
            ),
        )
    except BaseException as exc:
        execution_error = exc
        result = SerialRunResult(
            completed_plan_ids=tuple(
                plan_id
                for plan_id, record in plan_records.items()
                if record.state == IngestionPlanState.SEALED
            ),
            unavailable_plan_ids=tuple(
                plan_id
                for plan_id, record in plan_records.items()
                if record.state == IngestionPlanState.ERROR
            ),
            completed_case_ids=tuple(case_records),
            cancelled=False,
            unknown_outcome_task_id=None,
        )

    sealed_run: list[RunRecord] = []
    sealed_manifest: list[CapsuleManifest] = []

    def seal_terminal_records(_unknown_active_outcome: bool) -> None:
        for plan in ingestion_plans:
            if plan.ingestion_plan_id in plan_records:
                continue
            if result.unknown_outcome_task_id == plan.ingestion_plan_id:
                terminal_state = IngestionPlanState.INTERRUPTED_UNKNOWN_OUTCOME
            elif result.cancelled:
                terminal_state = IngestionPlanState.CANCELLED
            else:
                terminal_state = IngestionPlanState.ERROR
            plan_record = _terminal_plan_record(
                run_id=run_id,
                plan=plan,
                occurrence_id=ingestion_occurrences[plan.ingestion_plan_id],
                case_occurrences=case_occurrences,
                state=terminal_state,
            )
            plan_records[plan.ingestion_plan_id] = plan_record
            _seal(
                state,
                "ingestion-plans",
                plan_record.ingestion_occurrence_id,
                plan_record,
            )
        for case_plan in case_plans:
            if case_plan.case_manifest_entry_id in case_records:
                continue
            plan_id = plan_by_case[case_plan.case_manifest_entry_id]
            if result.cancelled:
                case_state = CaseState.CANCELLED
                error_stage = "cancelled"
            elif result.unknown_outcome_task_id is not None:
                case_state = CaseState.ERROR
                error_stage = "unknown_outcome"
            elif execution_error is not None:
                case_state = CaseState.ERROR
                error_stage = "runtime"
            else:
                case_state = CaseState.ERROR
                error_stage = "ingestion_readiness"
            case_record = CaseRecordV2(
                case_occurrence_id=case_occurrences[case_plan.case_manifest_entry_id],
                run_id=run_id,
                ingestion_occurrence_id=ingestion_occurrences[plan_id],
                case_manifest_entry_id=case_plan.case_manifest_entry_id,
                state=case_state,
                retrieval_raw_ref=None,
                prompt_sha256=None,
                answer_raw_ref=None,
                parsed_answer_sha256=None,
                evaluation_raw_ref=None,
                evaluation_disposition=CaseEvaluationDisposition.NOT_RUN,
                attempt_ids=(),
                error_stage=error_stage,
            )
            case_records[case_plan.case_manifest_entry_id] = case_record
            _seal(state, "cases", case_record.case_occurrence_id, case_record)

        if result.unknown_outcome_task_id is not None:
            run_state = RunState.INTERRUPTED
            resume = ResumeDisposition.REPLACEMENT_RUN_REQUIRED
        elif result.cancelled or execution_error is not None:
            run_state = RunState.ABORTED
            resume = ResumeDisposition.NOT_APPLICABLE
        else:
            run_state = RunState.FINALIZED
            resume = ResumeDisposition.NOT_APPLICABLE
        run_record = RunRecord(
            run_id=run_id,
            run_spec_hash=run_spec_hash,
            state=run_state,
            resume_disposition=resume,
            started_at=FAKE_STARTED_AT,
            ended_at=state.timestamp(),
            ingestion_occurrence_ids=tuple(
                ingestion_occurrences[plan.ingestion_plan_id] for plan in ingestion_plans
            ),
            case_occurrence_ids=tuple(
                case_occurrences[case.case_manifest_entry_id] for case in case_plans
            ),
        )
        _seal(state, "run", run_id, run_record)
        sealed_run.append(run_record)

    def record_close_error(owner: str, error: BaseException) -> None:
        payload = canonical_json_bytes({"error_type": type(error).__name__, "message": str(error)})
        error_hash = hashlib.sha256(payload).hexdigest()
        store.seal_raw(
            RawPayloadSealRequest(
                sha256=error_hash,
                media_type="application/json",
                compression="gzip",
                payload_bytes=payload,
            )
        )
        close_error_id = canonical_sha256(["oamb-fake-close-error-v1", run_id, owner, error_hash])
        close_error = CloseErrorRecord(
            close_error_id=close_error_id,
            owner_kind="run",
            owner_id=run_id,
            client_profile_id=owner,
            error_ref=error_hash,
            shutdown_stage="client_close",
            occurred_at=state.timestamp(),
        )
        _seal(state, "close-errors", close_error_id, close_error)

    def write_manifest() -> None:
        sealed_manifest.append(store.finalize_capsule(run_id=run_id, run_spec_hash=run_spec_hash))

    shutdown_error: BaseException | None = None
    lifecycle.shutdown_started = True
    try:
        await ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=seal_terminal_records,
                flush_records=lambda: None,
                write_manifest=write_manifest,
                record_close_error=record_close_error,
            )
        ).shutdown(
            clients=(("fake-model", model), ("fake-memory", memory)),
            grace_seconds=0,
        )
    except BaseException as exc:
        shutdown_error = exc
    if execution_error is not None and shutdown_error is not None:
        raise BaseExceptionGroup(
            "fake execution and shutdown both failed",
            (execution_error, shutdown_error),
        )
    if execution_error is not None:
        raise execution_error
    if shutdown_error is not None:
        raise shutdown_error
    return FakeRunArtifacts(capsule_root, sealed_manifest[0], sealed_run[0])


def build_fake_budget(run_id: str) -> BudgetSpec:
    return BudgetSpec(
        budget_id=f"{run_id}-zero-external",
        scope_kind=BudgetScopeKind.RUN,
        scope_id=run_id,
        approval_id=None,
        max_attempts=0,
        max_input_tokens=0,
        max_output_tokens=0,
        max_wall_seconds=Decimal("0"),
        max_cost=None,
        currency=None,
    )


def resolve_fake_preflight(
    capsule_root: Path,
    run_id: str,
    *,
    artifact_durability: ArtifactDurabilityPreflight | None = None,
) -> ResolvedRunPlan:
    _require_safe_run_id(run_id)
    budget = build_fake_budget(run_id)
    selected_artifact_durability = artifact_durability or _probe_artifact_durability(capsule_root)
    slots = tuple(
        RoleSlot(
            role=role,
            status=ResolutionStatus.NOT_APPLICABLE,
            binding=None,
            credential_reference=None,
            evidence_reference="credential-free-fake",
        )
        for role in RoleSlotName
    )
    request = RunPreflightRequest(
        operation_kind=OperationKind.FAKE_RUN,
        budget_spec=budget,
        role_slots=slots,
        adapter_profile=AdapterProfileDescriptor(
            profile_id="fake-memory-v1",
            memory_system_id=FAKE_MEMORY_SYSTEM_ID,
            transport_kind=TransportKind.FAKE,
            controlled_embedding=None,
            native_reranking_disabled=True,
            oamb_reranker_configured=False,
        ),
        provider_gates=ProviderGateClosure(
            liveness=GateStatus.NOT_APPLICABLE,
            storage_configuration=GateStatus.NOT_APPLICABLE,
            runtime_identity=GateStatus.NOT_APPLICABLE,
            model_readiness=GateStatus.NOT_APPLICABLE,
            memory_conformance=GateStatus.NOT_APPLICABLE,
        ),
        artifact_durability=selected_artifact_durability,
        runtime_binding=None,
        provider_runtime_attestation=None,
        approval=None,
        cost_measurement_spec=None,
        price_snapshot=None,
    )
    return resolve_run_plan(request)


def _probe_artifact_durability(capsule_root: Path) -> ArtifactDurabilityPreflight:
    root = Path(capsule_root).resolve(strict=False)
    try:
        root.mkdir(parents=True, exist_ok=True)
        if not root.is_dir():
            raise OSError("artifact root is not a directory")
        root_metadata = root.stat()
        probe_directory = Path(tempfile.mkdtemp(prefix=".oamb-preflight-", dir=root))
    except OSError as exc:
        raise PreflightRejected("artifact durability probe could not create its root") from exc

    file_fsync_supported = False
    directory_fsync_supported = False
    lease_supported = False
    no_replace_supported = False
    try:
        source = probe_directory / "source"
        source_descriptor = os.open(source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(source_descriptor, b"oamb-artifact-probe")
            os.fsync(source_descriptor)
            file_fsync_supported = True
        except OSError:
            pass
        finally:
            os.close(source_descriptor)

        directory_descriptor = os.open(
            probe_directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
            directory_fsync_supported = True
        except OSError:
            pass
        finally:
            os.close(directory_descriptor)

        lease = probe_directory / "lease"
        lease_descriptor = os.open(lease, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        os.close(lease_descriptor)
        try:
            duplicate_lease = os.open(
                lease,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
        except FileExistsError:
            lease_supported = True
        else:
            os.close(duplicate_lease)

        published = probe_directory / "published"
        os.link(source, published)
        try:
            os.link(source, published)
        except FileExistsError:
            no_replace_supported = True
    except OSError:
        pass
    finally:
        shutil.rmtree(probe_directory)

    return ArtifactDurabilityPreflight(
        artifact_root_fingerprint=canonical_sha256(
            [
                "oamb-artifact-root-v1",
                root.as_posix(),
                root_metadata.st_dev,
                root_metadata.st_ino,
            ]
        ),
        available_bytes=shutil.disk_usage(root).free,
        lease_supported=lease_supported,
        file_fsync_supported=file_fsync_supported,
        directory_fsync_supported=directory_fsync_supported,
        no_replace_supported=no_replace_supported,
    )


def build_fake_run_spec(
    *,
    run_id: str,
    dataset_hash: str,
    case_manifest_hash: str,
    runtime_binding_hash: str,
    workload_id: str,
) -> RunSpec:
    _require_safe_run_id(run_id)
    return RunSpec(
        run_id=run_id,
        protocol_id="oamb-fake-v1",
        dataset_manifest_hash=dataset_hash,
        case_manifest_hash=case_manifest_hash,
        workload_id=workload_id,
        memory_system_id=FAKE_MEMORY_SYSTEM_ID,
        runtime_binding_hash=runtime_binding_hash,
        environment_hash=canonical_sha256(["oamb-fake-environment-v1"]),
        model_role_binding_ids=("fake-answer-v1", "fake-judge-v1"),
        budget_id=f"{run_id}-zero-external",
        code_revision="generated-fake-v1",
        normalizer_fingerprint=canonical_sha256(["oamb-fake-normalizer-v1"]),
    )


def _terminal_plan_record(
    *,
    run_id: str,
    plan: IngestionPlan,
    occurrence_id: str,
    case_occurrences: dict[str, str],
    state: IngestionPlanState,
) -> IngestionPlanRecord:
    return IngestionPlanRecord(
        ingestion_occurrence_id=occurrence_id,
        run_id=run_id,
        memory_system_id=FAKE_MEMORY_SYSTEM_ID,
        ingestion_plan_id=plan.ingestion_plan_id,
        ordered_member_context_manifest_entry_ids=plan.ordered_member_context_manifest_entry_ids,
        ordered_case_occurrence_ids=tuple(
            case_occurrences[case_id] for case_id in plan.ordered_case_manifest_entry_ids
        ),
        state=state,
        intended_source_count=plan.intended_source_count,
        accepted_source_count=0,
        failed_source_count=0,
        readiness_evidence_refs=(),
        attempt_ids=(),
        usage_record_ids=(),
        resource_record_ids=(),
        cost_record_ids=(),
    )


def _seal_fake_usage(
    state: _ExecutionState,
    *,
    attempt_id_value: str,
    parent_kind: Literal["ingestion_plan", "case", "phase_review"],
    parent_id: str,
    stage: TokenStage,
    operation_kind: str,
    input_tokens: int,
    output_tokens: int,
    raw_response_ref: str,
) -> str:
    usage_id = canonical_sha256(
        [
            "oamb-fake-runtime-usage-v1",
            attempt_id_value,
            stage.value,
            input_tokens,
            output_tokens,
        ]
    )
    record = TokenUsageRecord(
        usage_record_id=usage_id,
        attempt_id=attempt_id_value,
        parent_kind=parent_kind,
        parent_id=parent_id,
        stage=stage,
        operation_kind=operation_kind,
        token_domain=TokenDomain.EXTERNAL_LLM,
        measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=input_tokens,
        visible_output_tokens=output_tokens,
        supplier_reported_total_tokens=input_tokens + output_tokens,
        context_view_tokens=None,
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
        raw_response_ref=raw_response_ref,
    )
    _seal(state, "usage", usage_id, record)
    return usage_id


def _seal_not_applicable_query_usage(
    state: _ExecutionState,
    attempt_id_value: str,
    parent_id: str,
    raw_response_ref: str,
) -> str:
    usage_id = canonical_sha256(["oamb-fake-query-usage-v1", attempt_id_value, "not-applicable"])
    record = TokenUsageRecord(
        usage_record_id=usage_id,
        attempt_id=attempt_id_value,
        parent_kind="case",
        parent_id=parent_id,
        stage=TokenStage.MEMORY_QUERY,
        operation_kind="fake_memory_query",
        token_domain=TokenDomain.EXTERNAL_LLM,
        measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=None,
        visible_output_tokens=None,
        supplier_reported_total_tokens=None,
        context_view_tokens=None,
        proof_status=ProofStatus.NOT_APPLICABLE,
        reason="fake-memory-query-has-no-model-meter",
        raw_response_ref=raw_response_ref,
    )
    _seal(state, "usage", usage_id, record)
    return usage_id


def _seal_context_usage(
    state: _ExecutionState,
    attempt_id_value: str,
    parent_id: str,
    visible_bytes: bytes,
) -> str:
    token_count = len(visible_bytes.decode("utf-8").split())
    usage_id = canonical_sha256(["oamb-fake-context-usage-v1", attempt_id_value, token_count])
    record = TokenUsageRecord(
        usage_record_id=usage_id,
        attempt_id=attempt_id_value,
        parent_kind="case",
        parent_id=parent_id,
        stage=TokenStage.CONTEXT_VIEW,
        operation_kind="fake_visible_context",
        token_domain=TokenDomain.LOCAL_CONTEXT_VIEW,
        measurement_source=TokenMeasurementSource.LOCAL_TOKENIZER,
        input_tokens=None,
        visible_output_tokens=None,
        supplier_reported_total_tokens=None,
        context_view_tokens=token_count,
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
        raw_response_ref=None,
    )
    _seal(state, "usage", usage_id, record)
    return usage_id


def _seal(
    state: _ExecutionState,
    collection: str,
    record_id: str,
    record: StrictContract,
) -> None:
    seal_source_contract(
        state.store,
        relative_path=f"source/{collection}/{record_id}.json",
        record_id=record_id,
        record=record,
    )


def _require_safe_run_id(run_id: str) -> None:
    if not run_id or run_id in {".", ".."} or "/" in run_id or "\\" in run_id or "\x00" in run_id:
        raise ValueError("run ID must be one safe path component")
