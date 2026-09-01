from __future__ import annotations

import asyncio
import hashlib
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import (
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionPlan,
    IngestionRequest,
    RawReferenceHandle,
    RetrievalRequest,
    ScopeReceipt,
    SourceUnit,
)
from oamb.memory_systems.fake import ScriptedFakeMemorySystem


def test_ready_history_releases_questions_while_other_histories_keep_ingesting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    history_started: list[str] = []
    question_started: list[str] = []
    history_releases = {name: asyncio.Event() for name in ("plan-a", "plan-b", "plan-c")}
    question_releases = {name: asyncio.Event() for name in ("case-a", "case-b", "case-c")}
    history_events = {name: asyncio.Event() for name in history_releases}
    question_events = {name: asyncio.Event() for name in question_releases}
    scopes = {name: object() for name in history_releases}

    async def ingest_one(**kwargs: Any) -> tuple[tuple[str], dict[str, object]]:
        plan = kwargs["plans"][0]
        plan_id = plan.ingestion_plan_id
        history_started.append(plan_id)
        history_events[plan_id].set()
        await history_releases[plan_id].wait()
        return (f"record-{plan_id}",), {plan_id: scopes[plan_id]}

    async def execute_one_question(**kwargs: Any) -> tuple[str]:
        plan = kwargs["plans"][0]
        case = kwargs["case_plans"][0]
        plan_id = plan.ingestion_plan_id
        case_id = case.case_manifest_entry_id
        assert kwargs["scopes"] == {plan_id: scopes[plan_id]}
        question_started.append(case_id)
        question_events[case_id].set()
        await question_releases[case_id].wait()
        return (f"record-{case_id}",)

    monkeypatch.setattr(native_run, "_execute_ingestion_plans_serial", ingest_one)
    monkeypatch.setattr(native_run, "_execute_cases_serial", execute_one_question)

    async def scenario() -> None:
        cases = tuple(
            SimpleNamespace(case_manifest_entry_id=f"case-{suffix}") for suffix in ("a", "b", "c")
        )
        plans = tuple(
            SimpleNamespace(
                ingestion_plan_id=f"plan-{suffix}",
                ordered_case_manifest_entry_ids=(f"case-{suffix}",),
            )
            for suffix in ("a", "b", "c")
        )
        state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=2,
                max_parallel_questions=2,
            )
        )
        run = asyncio.create_task(
            native_run._execute_history_question_pipeline(
                state=cast(Any, state),
                workload=cast(Any, object()),
                memory=cast(Any, object()),
                answer_model=cast(Any, object()),
                judge_model=cast(Any, object()),
                plans=cast(Any, plans),
                case_plans=cast(Any, cases),
                memory_system_id="memory",
                runtime_binding_hash="runtime",
                adapter_profile_id="adapter",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id="judge",
            )
        )

        await asyncio.wait_for(history_events["plan-a"].wait(), timeout=1)
        await asyncio.wait_for(history_events["plan-b"].wait(), timeout=1)
        assert history_started == ["plan-a", "plan-b"]
        assert not history_events["plan-c"].is_set()

        history_releases["plan-a"].set()
        await asyncio.wait_for(question_events["case-a"].wait(), timeout=1)
        await asyncio.wait_for(history_events["plan-c"].wait(), timeout=1)
        assert history_started == ["plan-a", "plan-b", "plan-c"]

        history_releases["plan-b"].set()
        await asyncio.wait_for(question_events["case-b"].wait(), timeout=1)
        history_releases["plan-c"].set()
        await asyncio.sleep(0)
        assert not question_events["case-c"].is_set()

        question_releases["case-a"].set()
        await asyncio.wait_for(question_events["case-c"].wait(), timeout=1)
        question_releases["case-b"].set()
        question_releases["case-c"].set()
        plan_records, case_records = await run

        assert cast(Any, plan_records) == (
            "record-plan-a",
            "record-plan-b",
            "record-plan-c",
        )
        assert cast(Any, case_records) == (
            "record-case-a",
            "record-case-b",
            "record-case-c",
        )
        assert question_started == ["case-a", "case-b", "case-c"]

    asyncio.run(scenario())


def test_questions_sharing_one_ready_history_overlap_without_reingestion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    ingestion_count = 0
    admitted_questions: list[tuple[str, object]] = []
    both_questions_started = asyncio.Event()
    release = asyncio.Event()
    shared_scope = object()

    async def ingest_one(**kwargs: Any) -> tuple[tuple[str], dict[str, object]]:
        nonlocal ingestion_count
        ingestion_count += 1
        plan = kwargs["plans"][0]
        return ("history-record",), {plan.ingestion_plan_id: shared_scope}

    async def execute_one_question(**kwargs: Any) -> tuple[str]:
        case = kwargs["case_plans"][0]
        plan = kwargs["plans"][0]
        assert kwargs["scopes"] == {plan.ingestion_plan_id: shared_scope}
        admitted_questions.append((case.case_manifest_entry_id, shared_scope))
        if len(admitted_questions) == 2:
            both_questions_started.set()
        await release.wait()
        return (case.case_manifest_entry_id,)

    monkeypatch.setattr(native_run, "_execute_ingestion_plans_serial", ingest_one)
    monkeypatch.setattr(native_run, "_execute_cases_serial", execute_one_question)

    async def scenario() -> None:
        cases = tuple(SimpleNamespace(case_manifest_entry_id=value) for value in ("a-1", "a-2"))
        plans = (
            SimpleNamespace(
                ingestion_plan_id="plan-a",
                ordered_case_manifest_entry_ids=("a-1", "a-2"),
            ),
        )
        state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=2,
                max_parallel_questions=2,
            )
        )
        run = asyncio.create_task(
            native_run._execute_history_question_pipeline(
                state=cast(Any, state),
                workload=cast(Any, object()),
                memory=cast(Any, object()),
                answer_model=cast(Any, object()),
                judge_model=cast(Any, object()),
                plans=cast(Any, plans),
                case_plans=cast(Any, cases),
                memory_system_id="memory",
                runtime_binding_hash="runtime",
                adapter_profile_id="adapter",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id="judge",
            )
        )
        await asyncio.wait_for(both_questions_started.wait(), timeout=1)
        assert admitted_questions == [("a-1", shared_scope), ("a-2", shared_scope)]
        assert ingestion_count == 1
        release.set()
        plan_records, case_records = await run
        assert cast(Any, plan_records) == ("history-record",)
        assert cast(Any, case_records) == ("a-1", "a-2")

    asyncio.run(scenario())


def test_fatal_history_stops_queued_admission_but_active_sibling_settles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    started: list[str] = []
    both_started = asyncio.Event()
    fatal_raised = asyncio.Event()
    release_sibling = asyncio.Event()
    sibling_settled = asyncio.Event()

    async def ingest_one(**kwargs: Any) -> tuple[tuple[str], dict[str, object]]:
        plan = kwargs["plans"][0]
        plan_id = plan.ingestion_plan_id
        started.append(plan_id)
        if len(started) == 2:
            both_started.set()
        if plan_id == "plan-a":
            await both_started.wait()
            fatal_raised.set()
            raise RuntimeError("fatal-plan-a")
        if plan_id == "plan-b":
            await release_sibling.wait()
            sibling_settled.set()
        return (f"record-{plan_id}",), {plan_id: object()}

    monkeypatch.setattr(native_run, "_execute_ingestion_plans_serial", ingest_one)

    async def scenario() -> None:
        plans = tuple(
            SimpleNamespace(
                ingestion_plan_id=f"plan-{suffix}",
                ordered_case_manifest_entry_ids=(),
            )
            for suffix in ("a", "b", "c")
        )
        state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=2,
                max_parallel_questions=2,
            )
        )
        run = asyncio.create_task(
            native_run._execute_history_question_pipeline(
                state=cast(Any, state),
                workload=cast(Any, object()),
                memory=cast(Any, object()),
                answer_model=cast(Any, object()),
                judge_model=None,
                plans=cast(Any, plans),
                case_plans=(),
                memory_system_id="memory",
                runtime_binding_hash="runtime",
                adapter_profile_id="adapter",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id=None,
            )
        )

        await asyncio.wait_for(fatal_raised.wait(), timeout=1)
        await asyncio.sleep(0)
        assert started == ["plan-a", "plan-b"]
        assert not sibling_settled.is_set()
        assert not run.done()

        release_sibling.set()
        with pytest.raises(RuntimeError, match="fatal-plan-a"):
            await run
        assert sibling_settled.is_set()
        assert started == ["plan-a", "plan-b"]

    asyncio.run(scenario())


def test_malformed_history_result_stops_queued_admission_before_releasing_the_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    started: list[str] = []

    async def ingest_one(**kwargs: Any) -> tuple[tuple[str, ...], dict[str, object]]:
        plan = kwargs["plans"][0]
        plan_id = plan.ingestion_plan_id
        started.append(plan_id)
        if plan_id == "plan-a":
            return (), {plan_id: object()}
        return (f"record-{plan_id}",), {plan_id: object()}

    monkeypatch.setattr(native_run, "_execute_ingestion_plans_serial", ingest_one)

    async def scenario() -> None:
        plans = tuple(
            SimpleNamespace(
                ingestion_plan_id=f"plan-{suffix}",
                ordered_case_manifest_entry_ids=(),
            )
            for suffix in ("a", "b", "c")
        )
        state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=1,
                max_parallel_questions=1,
            )
        )
        with pytest.raises(AssertionError, match="one record and scope"):
            await native_run._execute_history_question_pipeline(
                state=cast(Any, state),
                workload=cast(Any, object()),
                memory=cast(Any, object()),
                answer_model=cast(Any, object()),
                judge_model=None,
                plans=cast(Any, plans),
                case_plans=(),
                memory_system_id="memory",
                runtime_binding_hash="runtime",
                adapter_profile_id="adapter",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id=None,
            )

    asyncio.run(scenario())
    assert started == ["plan-a"]


def test_malformed_question_result_stops_queued_admission_before_releasing_the_permit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    started: list[str] = []
    scope = object()

    async def ingest_one(**kwargs: Any) -> tuple[tuple[str], dict[str, object]]:
        plan = kwargs["plans"][0]
        return ("history-record",), {plan.ingestion_plan_id: scope}

    async def execute_one_question(**kwargs: Any) -> tuple[str, ...]:
        case = kwargs["case_plans"][0]
        case_id = case.case_manifest_entry_id
        started.append(case_id)
        if case_id == "case-a":
            return ()
        return (f"record-{case_id}",)

    monkeypatch.setattr(native_run, "_execute_ingestion_plans_serial", ingest_one)
    monkeypatch.setattr(native_run, "_execute_cases_serial", execute_one_question)

    async def scenario() -> None:
        cases = tuple(
            SimpleNamespace(case_manifest_entry_id=f"case-{suffix}") for suffix in ("a", "b", "c")
        )
        plans = (
            SimpleNamespace(
                ingestion_plan_id="plan-a",
                ordered_case_manifest_entry_ids=tuple(
                    case.case_manifest_entry_id for case in cases
                ),
            ),
        )
        state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=1,
                max_parallel_questions=1,
            )
        )
        with pytest.raises(AssertionError, match="exactly one result"):
            await native_run._execute_history_question_pipeline(
                state=cast(Any, state),
                workload=cast(Any, object()),
                memory=cast(Any, object()),
                answer_model=cast(Any, object()),
                judge_model=None,
                plans=cast(Any, plans),
                case_plans=cast(Any, cases),
                memory_system_id="memory",
                runtime_binding_hash="runtime",
                adapter_profile_id="adapter",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id=None,
            )

    asyncio.run(scenario())
    assert started == ["case-a"]


def test_parent_cancellation_drains_active_histories_and_preserves_cleanup_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    both_started = asyncio.Event()
    started: list[str] = []

    async def ingest_one(**kwargs: Any) -> tuple[tuple[str], dict[str, object]]:
        plan = kwargs["plans"][0]
        plan_id = plan.ingestion_plan_id
        started.append(plan_id)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if plan_id == "plan-a":
                raise RuntimeError("cleanup-evidence-failed") from None
            raise
        raise AssertionError("unreachable")

    monkeypatch.setattr(native_run, "_execute_ingestion_plans_serial", ingest_one)

    async def scenario() -> None:
        plans = tuple(
            SimpleNamespace(
                ingestion_plan_id=f"plan-{suffix}",
                ordered_case_manifest_entry_ids=(),
            )
            for suffix in ("a", "b")
        )
        state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=2,
                max_parallel_questions=1,
            )
        )
        run = asyncio.create_task(
            native_run._execute_history_question_pipeline(
                state=cast(Any, state),
                workload=cast(Any, object()),
                memory=cast(Any, object()),
                answer_model=cast(Any, object()),
                judge_model=None,
                plans=cast(Any, plans),
                case_plans=(),
                memory_system_id="memory",
                runtime_binding_hash="runtime",
                adapter_profile_id="adapter",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id=None,
            )
        )
        await asyncio.wait_for(both_started.wait(), timeout=1)
        run.cancel()
        with pytest.raises(BaseExceptionGroup) as captured:
            await run
        assert any(isinstance(error, asyncio.CancelledError) for error in captured.value.exceptions)
        assert any(
            isinstance(error, RuntimeError) and str(error) == "cleanup-evidence-failed"
            for error in captured.value.exceptions
        )

    asyncio.run(scenario())


def test_parent_cancellation_preserves_every_sibling_cancellation_evidence_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    both_started = asyncio.Event()
    started: list[str] = []

    async def ingest_one(**kwargs: Any) -> tuple[tuple[str], dict[str, object]]:
        plan = kwargs["plans"][0]
        plan_id = plan.ingestion_plan_id
        started.append(plan_id)
        if len(started) == 2:
            both_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            suffix = plan_id.removeprefix("plan-")
            raise native_run.NativeCancellationEvidenceError(
                asyncio.CancelledError(f"cancel-{suffix}"),
                RuntimeError(f"seal-{suffix}"),
            ) from None
        raise AssertionError("unreachable")

    monkeypatch.setattr(native_run, "_execute_ingestion_plans_serial", ingest_one)

    async def scenario() -> None:
        plans = tuple(
            SimpleNamespace(
                ingestion_plan_id=f"plan-{suffix}",
                ordered_case_manifest_entry_ids=(),
            )
            for suffix in ("a", "b")
        )
        state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=2,
                max_parallel_questions=1,
            )
        )
        run = asyncio.create_task(
            native_run._execute_history_question_pipeline(
                state=cast(Any, state),
                workload=cast(Any, object()),
                memory=cast(Any, object()),
                answer_model=cast(Any, object()),
                judge_model=None,
                plans=cast(Any, plans),
                case_plans=(),
                memory_system_id="memory",
                runtime_binding_hash="runtime",
                adapter_profile_id="adapter",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id=None,
            )
        )
        await asyncio.wait_for(both_started.wait(), timeout=1)
        run.cancel()
        with pytest.raises(BaseExceptionGroup) as captured:
            await run
        sibling_errors = tuple(
            error
            for error in captured.value.exceptions
            if isinstance(error, native_run.NativeCancellationEvidenceError)
        )
        assert {str(error.errors[1]) for error in sibling_errors} == {"seal-a", "seal-b"}

    asyncio.run(scenario())


def test_source_writes_are_serial_within_history_and_overlap_across_histories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    class ObservedSourceMemory(ScriptedFakeMemorySystem):
        def __init__(self, store: ArtifactStore) -> None:
            super().__init__(store)
            self.calls: dict[str, list[int]] = {}
            self.active_by_plan: dict[str, int] = {}
            self.max_active_by_plan: dict[str, int] = {}
            self.active_total = 0
            self.max_active_total = 0
            self.first_started: dict[str, asyncio.Event] = {}
            self.both_first_started = asyncio.Event()
            self.release_first = asyncio.Event()

        def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]:
            plan_id = self._plan_by_scope[request.scope.scope_id]
            self.calls.setdefault(plan_id, [])
            self.active_by_plan.setdefault(plan_id, 0)
            self.max_active_by_plan.setdefault(plan_id, 0)
            self.first_started.setdefault(plan_id, asyncio.Event())
            return tuple(
                IngestionDispatch(
                    dispatch_ordinal_1_indexed=index,
                    operation_kind="observed_source_ingest",
                    request_fingerprint=canonical_sha256(
                        ["observed-source-ingest-v1", request.scope.scope_id, index]
                    ),
                    ordered_source_units=(source,),
                )
                for index, source in enumerate(request.ordered_source_units, start=1)
            )

        async def ingest(
            self,
            request: IngestionDispatchRequest,
        ) -> IngestionDispatchReceipt:
            plan_id = self._plan_by_scope[request.scope.scope_id]
            ordinal = request.dispatch.dispatch_ordinal_1_indexed
            self.calls[plan_id].append(ordinal)
            self.active_by_plan[plan_id] += 1
            self.active_total += 1
            self.max_active_by_plan[plan_id] = max(
                self.max_active_by_plan[plan_id], self.active_by_plan[plan_id]
            )
            self.max_active_total = max(self.max_active_total, self.active_total)
            if ordinal == 1:
                self.first_started[plan_id].set()
                if all(event.is_set() for event in self.first_started.values()):
                    self.both_first_started.set()
                await self.release_first.wait()
            self.active_by_plan[plan_id] -= 1
            self.active_total -= 1

            source_ids = tuple(
                source.source_unit_id for source in request.dispatch.ordered_source_units
            )
            accepted = (*self._accepted_by_scope.get(request.scope.scope_id, ()), *source_ids)
            self._accepted_by_scope[request.scope.scope_id] = accepted
            raw_bytes = f"{plan_id}:{ordinal}".encode()
            return IngestionDispatchReceipt(
                attempt_id=request.attempt_id,
                dispatch=request.dispatch,
                accepted_source_unit_ids=source_ids,
                rejected_source_unit_ids=(),
                raw_reference=self._seal_raw(raw_bytes),
                raw_response_bytes=raw_bytes,
                usage_records=(),
            )

    def make_plan(label: str) -> IngestionPlan:
        context_id = canonical_sha256(["context", label])
        source_units = tuple(
            SourceUnit(
                source_unit_id=canonical_sha256(["source", label, ordinal]),
                context_manifest_entry_id=context_id,
                ordinal_1_indexed=ordinal,
                payload_sha256=hashlib.sha256(f"{label}-{ordinal}".encode()).hexdigest(),
                payload_bytes=f"{label}-{ordinal}".encode(),
            )
            for ordinal in (1, 2)
        )
        return IngestionPlan(
            ingestion_plan_id=canonical_sha256(["plan", label]),
            ordered_member_context_manifest_entry_ids=(context_id,),
            shared_context_sha256=canonical_sha256(["shared-context", label]),
            intended_source_count=len(source_units),
            ordered_source_units=source_units,
            ordered_case_manifest_entry_ids=(),
        )

    plans = (make_plan("a"), make_plan("b"))
    memory = ObservedSourceMemory(ArtifactStore(tmp_path / "raw"))
    serial_ingest = native_run._execute_ingestion_plans_serial
    serial_states = {
        plan.ingestion_plan_id: native_run._NativeExecutionState(
            store=ArtifactStore(tmp_path / plan.ingestion_plan_id),
            run_id=canonical_sha256(["run"]),
            lease_record_hash=canonical_sha256(["lease"]),
            close_timeout_seconds=1,
        )
        for plan in plans
    }

    async def ingest_one_with_isolated_state(**kwargs: Any) -> Any:
        plan = kwargs["plans"][0]
        delegated = dict(kwargs)
        delegated["state"] = serial_states[plan.ingestion_plan_id]
        return await serial_ingest(**delegated)

    monkeypatch.setattr(
        native_run,
        "_execute_ingestion_plans_serial",
        ingest_one_with_isolated_state,
    )

    async def scenario() -> None:
        pipeline_state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=2,
                max_parallel_questions=1,
            )
        )
        run = asyncio.create_task(
            native_run._execute_history_question_pipeline(
                state=cast(Any, pipeline_state),
                workload=cast(Any, object()),
                memory=cast(Any, memory),
                answer_model=cast(Any, object()),
                judge_model=None,
                plans=plans,
                case_plans=(),
                memory_system_id="fake-memory",
                runtime_binding_hash=canonical_sha256(["runtime"]),
                adapter_profile_id="fake-memory-v1",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id=None,
            )
        )

        await asyncio.wait_for(memory.both_first_started.wait(), timeout=1)
        assert tuple(memory.calls[plan.ingestion_plan_id] for plan in plans) == ([1], [1])
        assert memory.max_active_total == 2
        assert tuple(memory.max_active_by_plan[plan.ingestion_plan_id] for plan in plans) == (1, 1)

        memory.release_first.set()
        plan_records, case_records = await run
        assert tuple(memory.calls[plan.ingestion_plan_id] for plan in plans) == ([1, 2], [1, 2])
        assert tuple(record.ingestion_plan_id for record in plan_records) == tuple(
            plan.ingestion_plan_id for plan in plans
        )
        assert case_records == ()

    asyncio.run(scenario())

    async def stop_between_dispatches() -> None:
        plan = make_plan("stop")
        stopped_memory = ObservedSourceMemory(ArtifactStore(tmp_path / "stopped-raw"))
        stop_event = asyncio.Event()
        stopped_state = native_run._NativeExecutionState(
            store=ArtifactStore(tmp_path / "stopped-state"),
            run_id=canonical_sha256(["stopped-run"]),
            lease_record_hash=canonical_sha256(["stopped-lease"]),
            close_timeout_seconds=1,
            stop_event=cast(Any, stop_event),
        )
        run = asyncio.create_task(
            serial_ingest(
                state=stopped_state,
                memory=stopped_memory,
                plans=(plan,),
                case_plans=(),
                memory_system_id="fake-memory",
                runtime_binding_hash=canonical_sha256(["runtime"]),
                adapter_profile_id="fake-memory-v1",
            )
        )
        await asyncio.wait_for(stopped_memory.both_first_started.wait(), timeout=1)
        stop_event.set()
        stopped_memory.release_first.set()
        with pytest.raises(native_run.NativeRunInterrupted):
            await run
        assert stopped_memory.calls[plan.ingestion_plan_id] == [1]

    asyncio.run(stop_between_dispatches())


def test_native_stop_handlers_are_installed_before_child_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    call_order: list[str] = []

    class FakeConnection:
        def close(self) -> None:
            pass

    class FakeEvent:
        def is_set(self) -> bool:
            return False

        def set(self) -> None:
            pass

    class FakeProcess:
        pid = None

        def start(self) -> None:
            call_order.append("start")
            assert call_order == ["install", "start"]
            raise RuntimeError("planted start failure")

        def close(self) -> None:
            pass

    class FakeContext:
        def Pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is False
            return FakeConnection(), FakeConnection()

        def Event(self) -> FakeEvent:
            return FakeEvent()

        def Process(self, **_kwargs: Any) -> FakeProcess:
            return FakeProcess()

    def install(_event: object, _count: object) -> dict[object, object]:
        call_order.append("install")
        return {}

    monkeypatch.setattr(multiprocessing, "get_context", lambda _method: FakeContext())
    monkeypatch.setattr(native_run, "_install_native_stop_handlers", install)
    monkeypatch.setattr(native_run, "_restore_native_stop_handlers", lambda _previous: None)
    request = native_run._NativeRunRequest(
        output_root=tmp_path,
        run_id="handler-install-order",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=cast(Any, object()),
        visible_evidence_policy=cast(Any, object()),
        artifact_store_factory=cast(Any, object()),
        memory_factory=cast(Any, object()),
        model_factory=cast(Any, object()),
        answer_role_binding_id="answer",
        judge_model_factory=None,
        judge_role_binding_id=None,
        close_timeout_seconds=1,
        control=None,
    )

    with pytest.raises(RuntimeError, match="planted start failure"):
        native_run._run_native_supervised(request)

    assert call_order == ["install", "start"]


def test_limits_one_and_two_return_the_same_canonical_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    plans = tuple(
        SimpleNamespace(
            ingestion_plan_id=f"plan-{suffix}",
            ordered_case_manifest_entry_ids=(f"case-{suffix}",),
        )
        for suffix in ("a", "b", "c")
    )
    cases = tuple(
        SimpleNamespace(case_manifest_entry_id=f"case-{suffix}") for suffix in ("a", "b", "c")
    )

    async def run_with_limit(limit: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
        history_b_started = asyncio.Event()
        case_b_started = asyncio.Event()
        scopes = {plan.ingestion_plan_id: object() for plan in plans}

        async def ingest_one(**kwargs: Any) -> tuple[tuple[str], dict[str, object]]:
            plan = kwargs["plans"][0]
            if limit == 2 and plan.ingestion_plan_id == "plan-a":
                await history_b_started.wait()
            if plan.ingestion_plan_id == "plan-b":
                history_b_started.set()
            return (f"record-{plan.ingestion_plan_id}",), {
                plan.ingestion_plan_id: scopes[plan.ingestion_plan_id]
            }

        async def execute_one_question(**kwargs: Any) -> tuple[str]:
            case = kwargs["case_plans"][0]
            if limit == 2 and case.case_manifest_entry_id == "case-a":
                await case_b_started.wait()
            if case.case_manifest_entry_id == "case-b":
                case_b_started.set()
            return (f"record-{case.case_manifest_entry_id}",)

        monkeypatch.setattr(native_run, "_execute_ingestion_plans_serial", ingest_one)
        monkeypatch.setattr(native_run, "_execute_cases_serial", execute_one_question)
        state = SimpleNamespace(
            control=SimpleNamespace(
                max_parallel_history_ingestions=limit,
                max_parallel_questions=limit,
            )
        )
        return cast(
            tuple[tuple[str, ...], tuple[str, ...]],
            await native_run._execute_history_question_pipeline(
                state=cast(Any, state),
                workload=cast(Any, object()),
                memory=cast(Any, object()),
                answer_model=cast(Any, object()),
                judge_model=None,
                plans=cast(Any, plans),
                case_plans=cast(Any, cases),
                memory_system_id="memory",
                runtime_binding_hash="runtime",
                adapter_profile_id="adapter",
                visible_evidence_policy=cast(Any, object()),
                answer_role_binding_id="answer",
                judge_role_binding_id=None,
            ),
        )

    serial = asyncio.run(run_with_limit(1))
    parallel = asyncio.run(run_with_limit(2))

    assert (
        serial
        == parallel
        == (
            ("record-plan-a", "record-plan-b", "record-plan-c"),
            ("record-case-a", "record-case-b", "record-case-c"),
        )
    )


def test_live_question_rejects_a_retrieval_request_for_another_history_before_dispatch() -> None:
    from oamb.runtime.native_run import _LiveAttemptedQueryMemory

    expected_scope = ScopeReceipt("a" * 64, "scope-a", RawReferenceHandle("b" * 64))
    wrong_scope = ScopeReceipt("c" * 64, "scope-b", RawReferenceHandle("d" * 64))

    class NeverCalledMemory:
        def __init__(self) -> None:
            self.calls = 0

        async def retrieve(self, _request: RetrievalRequest) -> object:
            self.calls += 1
            raise AssertionError("cross-history request reached the memory provider")

    memory = NeverCalledMemory()
    attempted = _LiveAttemptedQueryMemory(
        state=cast(Any, object()),
        memory=cast(Any, memory),
        scope=expected_scope,
        case_occurrence_id="e" * 64,
        adapter_profile_id="adapter",
    )

    async def scenario() -> None:
        with pytest.raises(ValueError, match="scope differs"):
            await attempted.retrieve(
                RetrievalRequest(
                    scope=wrong_scope,
                    case_occurrence_id="e" * 64,
                    query_bytes=b"question",
                    top_k=100,
                )
            )
        with pytest.raises(ValueError, match="case identity differs"):
            await attempted.retrieve(
                RetrievalRequest(
                    scope=expected_scope,
                    case_occurrence_id="f" * 64,
                    query_bytes=b"question",
                    top_k=100,
                )
            )

    asyncio.run(scenario())
    assert memory.calls == 0
