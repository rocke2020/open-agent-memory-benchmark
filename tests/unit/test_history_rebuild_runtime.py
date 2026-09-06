from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Coroutine
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import oamb.runtime.native_run as native
from oamb.artifacts.store import ArtifactStore
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256, case_occurrence_id
from oamb.contracts.ports import (
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionRequest,
    MemorySystemCallFailure,
    MemorySystemCallUnknownOutcome,
    NativeEvidenceBatch,
    RetrievalRequest,
    RuntimeResolution,
    ScopeAllocationRequest,
    ScopeReceipt,
    SettledTransientIngestionFailure,
)
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY
from tests.e2e.test_native_fixture_vertical_slice import (
    _NativeFixtureWorkload,
    _RecordedNativeMemory,
    _RecordedNativeModel,
)


class _PartialHistoryMemory(_RecordedNativeMemory):
    """An in-memory boundary retaining writes made before a terminal failure."""

    def __init__(self, store: ArtifactStore, *, failures: int, failure_kind: str = "settled"):
        super().__init__(store)
        self.failures = failures
        self.failure_kind = failure_kind
        self.scopes: list[Any] = []
        self.dispatch_log: list[tuple[str, tuple[bytes, ...]]] = []
        self.sentinels: dict[str, str] = {}
        self.retrieval_scopes: list[str] = []
        self.case_manifest_id = ""

    async def allocate_ingestion_scope(self, request: ScopeAllocationRequest) -> ScopeReceipt:
        scope = await super().allocate_ingestion_scope(request)
        assert scope.scope_id not in self.sentinels
        self.scopes.append(scope)
        self._accepted_by_scope[scope.scope_id] = ()
        return scope

    def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]:
        return tuple(
            replace(
                super(_PartialHistoryMemory, self).plan_ingestion(
                    replace(request, ordered_source_units=(source,))
                )[0],
                dispatch_ordinal_1_indexed=ordinal,
            )
            for ordinal, source in enumerate(request.ordered_source_units, 1)
        )

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt:
        sources = request.dispatch.ordered_source_units
        self.dispatch_log.append((request.scope.scope_id, tuple(s.payload_bytes for s in sources)))
        self._accepted_by_scope[request.scope.scope_id] += tuple(s.source_unit_id for s in sources)
        if len(self.scopes) <= self.failures and request.dispatch.dispatch_ordinal_1_indexed == 2:
            self.sentinels[request.scope.scope_id] = "OLD_SCOPE_ONLY_SENTINEL"
            if self.failure_kind == "unknown":
                raise MemorySystemCallUnknownOutcome("simulated unknown write")
            payload = canonical_json_bytes(
                {
                    "detail": "Fact extraction failed: 1/1 chunks failed. First failures: "
                    "chunk 0: APIConnectionError: Connection error."
                }
            )
            reference = self._seal_raw(payload)
            if self.failure_kind == "ordinary":
                raise MemorySystemCallFailure(
                    "unclassified failure", failure_kind="provider_error", raw_reference=reference
                )
            raise SettledTransientIngestionFailure(
                "known terminal extraction failure",
                settlement_basis="hindsight_sync_extraction_drained_v1",
                internal_retry_count=0,
                failure_kind="supplier_connection",
                status_code=500,
                raw_reference=reference,
                raw_response_bytes=payload,
            )
        payload = canonical_json_bytes(
            {
                "operation": "ingest",
                "attempt_id": request.attempt_id,
                "ingestion_occurrence_id": request.scope.ingestion_occurrence_id,
                "scope_id": request.scope.scope_id,
                "source_units": tuple(
                    {"source_unit_id": s.source_unit_id, "payload_sha256": s.payload_sha256}
                    for s in sources
                ),
                "accepted_source_unit_ids": tuple(s.source_unit_id for s in sources),
                "rejected_source_unit_ids": (),
            }
        )
        return IngestionDispatchReceipt(
            attempt_id=request.attempt_id,
            dispatch=request.dispatch,
            accepted_source_unit_ids=tuple(s.source_unit_id for s in sources),
            rejected_source_unit_ids=(),
            raw_reference=self._seal_raw(payload),
            raw_response_bytes=payload,
            usage_records=(),
        )

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        assert request.scope == self.scopes[-1], "retrieval returned to a failed scope"
        assert request.scope.scope_id not in self.sentinels, "old partial writes leaked"
        assert request.case_occurrence_id == case_occurrence_id(
            request.scope.ingestion_occurrence_id, self.case_manifest_id
        ), "case identity returned to the original history occurrence"
        self.retrieval_scopes.append(request.scope.scope_id)
        return await super().retrieve(request)


def _pipeline(
    tmp_path: Path,
    memory: _PartialHistoryMemory,
    stop: threading.Event | None = None,
) -> tuple[native._NativeExecutionState, Coroutine[Any, Any, Any]]:
    workload = _NativeFixtureWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    plan = workload.iter_ingestion_plans(manifest)[1]
    cases = tuple(
        case
        for case in workload.iter_case_plans(manifest)
        if case.case_manifest_entry_id in plan.ordered_case_manifest_entry_ids
    )
    memory.case_manifest_id = cases[0].case_manifest_entry_id
    state = native._NativeExecutionState(
        store=ArtifactStore(tmp_path),
        run_id="history-runtime-test",
        lease_record_hash="a" * 64,
        close_timeout_seconds=1,
        stop_event=stop,
    )
    return state, native._execute_history_question_pipeline(
        state=state,
        workload=workload,
        memory=memory,
        answer_model=_RecordedNativeModel(ArtifactStore(tmp_path)),
        judge_model=None,
        plans=(plan,),
        case_plans=cases,
        memory_system_id="hindsight",
        runtime_binding_hash=canonical_sha256(["fixture-runtime"]),
        adapter_profile_id="recorded-native-fixture-v1",
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        answer_role_binding_id="recorded-answer-v1",
        judge_role_binding_id=None,
    )


@pytest.mark.parametrize("retries", (0, 1, 2))
def test_whole_history_limit_counts_all_attempts_and_preserves_partial_scopes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retries: int,
) -> None:
    monkeypatch.setattr(native, "INFRASTRUCTURE_RETRY_BACKOFF_SECONDS", (1, 2)[:retries])
    waits: list[int] = []

    async def wait(seconds: int) -> None:
        waits.append(seconds)

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", wait)
    memory = _PartialHistoryMemory(ArtifactStore(tmp_path), failures=3)
    _state, operation = _pipeline(tmp_path, memory)
    with pytest.raises(RuntimeError):
        asyncio.run(operation)
    assert len(memory.scopes) == retries + 1
    assert len(memory.sentinels) == retries + 1
    assert waits == [1, 2][:retries]
    assert memory.retrieval_scopes == []
    records = [
        json.loads(p.read_bytes()) for p in (tmp_path / "source/history-attempts").glob("*.json")
    ]
    assert sorted(r["history_attempt_ordinal"] for r in records) == list(range(1, retries + 2))
    assert all(r["status"] == "retryable_failed_settled" for r in records)


def test_partial_history_failure_rebuilds_from_first_source_and_selects_only_new_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    waits: list[int] = []

    async def wait(seconds: int) -> None:
        events = list((tmp_path / "source/history-retries").glob("*.json"))
        assert len(events) == 1, "retry must be durable before waiting"
        assert json.loads(events[0].read_bytes())["retry_scheduled"] is True
        waits.append(seconds)

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", wait)
    memory = _PartialHistoryMemory(ArtifactStore(tmp_path), failures=1)
    _state, operation = _pipeline(tmp_path, memory)
    plans, cases = asyncio.run(operation)
    assert len(plans) == len(cases) == 1
    assert len(memory.scopes) == 2
    old, new = memory.scopes
    expected = [(b"partial accepted",), (b"partial rejected",)]
    assert [payload for scope, payload in memory.dispatch_log if scope == old.scope_id] == expected
    assert [payload for scope, payload in memory.dispatch_log if scope == new.scope_id] == expected
    assert memory.sentinels == {old.scope_id: "OLD_SCOPE_ONLY_SENTINEL"}
    assert memory.retrieval_scopes == [new.scope_id]
    assert (
        plans[0].ingestion_occurrence_id
        == cases[0].ingestion_occurrence_id
        == new.ingestion_occurrence_id
    )
    assert waits == [1]


@pytest.mark.parametrize("failure_kind", ("ordinary", "unknown"))
def test_unclassified_or_unknown_failure_never_allocates_a_successor(
    tmp_path: Path,
    failure_kind: str,
) -> None:
    memory = _PartialHistoryMemory(ArtifactStore(tmp_path), failures=3, failure_kind=failure_kind)
    _state, operation = _pipeline(tmp_path, memory)
    with pytest.raises(RuntimeError):
        asyncio.run(operation)
    assert len(memory.scopes) == 1
    assert not list((tmp_path / "source/history-retries").glob("*.json"))


def test_stop_during_backoff_preserves_scheduled_slot_without_allocating_successor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stop = threading.Event()

    async def wait(_seconds: int) -> None:
        stop.set()

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", wait)
    memory = _PartialHistoryMemory(ArtifactStore(tmp_path), failures=1)
    _state, operation = _pipeline(tmp_path, memory, stop)
    with pytest.raises(native.NativeRunInterrupted):
        asyncio.run(operation)
    assert len(memory.scopes) == 1
    events = [
        json.loads(p.read_bytes()) for p in (tmp_path / "source/history-retries").glob("*.json")
    ]
    assert len(events) == 1
    assert events[0]["retry_scheduled"] is True


@pytest.mark.parametrize("value", (True, -1, 3))
def test_native_control_cannot_bypass_frozen_retry_limit(value: int) -> None:
    from tests.unit.test_t10_native_run_control import _control

    with pytest.raises(ValueError, match="retry limit"):
        replace(_control(), max_retries_per_operation=value)


def test_final_inventory_keeps_both_histories_and_only_selected_case(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.contracts.specifications import INFRASTRUCTURE_RETRY_POLICY_HASH
    from oamb.runtime.case_partition import build_case_partition_spec

    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    plan = manifest.ingestion_plans[1]
    partition = build_case_partition_spec(
        run_id="history-final-inventory",
        resolved_plan_hash="1" * 64,
        cell_spec_hash="2" * 64,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=plan.ordered_case_manifest_entry_ids,
        budget_policy_hash="3" * 64,
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )

    class Memory(_PartialHistoryMemory):
        async def resolve(self) -> RuntimeResolution:
            return replace(await super().resolve(), memory_system_id="hindsight")

    memory = Memory(ArtifactStore(tmp_path / partition.run_id), failures=1)
    memory.case_manifest_id = plan.ordered_case_manifest_entry_ids[0]

    async def wait(_seconds: int) -> None:
        pass

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", wait)
    ready = asyncio.run(
        native._run_native_vertical_slice(
            output_root=tmp_path,
            run_id=partition.run_id,
            adapter_profile_id="recorded-native-fixture-v1",
            workload=workload,
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=lambda _store, _plans: memory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
            judge_model_factory=None,
            judge_role_binding_id=None,
            close_timeout_seconds=1,
            partition=partition,
        )
    )
    assert len(memory.scopes) == 2
    assert ready.ingestion_occurrence_ids == tuple(s.ingestion_occurrence_id for s in memory.scopes)
    assert ready.case_occurrence_ids == (
        case_occurrence_id(
            memory.scopes[-1].ingestion_occurrence_id,
            memory.case_manifest_id,
        ),
    )


def test_recovery_preparation_failure_constructs_no_provider_or_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import oamb.artifacts.history_recovery as recovery
    from oamb.contracts.specifications import INFRASTRUCTURE_RETRY_POLICY_HASH
    from oamb.runtime.case_partition import build_case_partition_spec

    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    partition = build_case_partition_spec(
        run_id="recovery-preparation-rejected",
        resolved_plan_hash="1" * 64,
        cell_spec_hash="2" * 64,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=manifest.ingestion_plans[
            1
        ].ordered_case_manifest_entry_ids,
        budget_policy_hash="3" * 64,
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    constructed: list[str] = []

    def reject(**_kwargs: Any) -> Any:
        raise ValueError("recovery preparation rejected")

    def factory(*_args: Any) -> Any:
        constructed.append("client")
        raise AssertionError("client construction must follow recovery closure")

    monkeypatch.setattr(recovery, "prepare_history_recovery", reject)
    with pytest.raises(ValueError, match="recovery preparation rejected"):
        asyncio.run(
            native._run_native_vertical_slice(
                output_root=tmp_path,
                run_id=partition.run_id,
                adapter_profile_id="recorded-native-fixture-v1",
                workload=workload,
                visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
                artifact_store_factory=ArtifactStore,
                memory_factory=factory,
                model_factory=factory,
                answer_role_binding_id="recorded-answer-v1",
                judge_model_factory=None,
                judge_role_binding_id=None,
                close_timeout_seconds=1,
                partition=partition,
                recovery_parts=(tmp_path / "predecessor",),
            )
        )
    assert constructed == []
