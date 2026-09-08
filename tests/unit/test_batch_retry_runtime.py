from __future__ import annotations

import asyncio
import json
from collections.abc import Coroutine
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import oamb.runtime.native_run as native
from oamb.artifacts.store import ArtifactStore
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionRequest,
    NativeEvidenceBatch,
    ReadinessReceipt,
    ReadinessRequest,
    RetrievalRequest,
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
    def __init__(self, store: ArtifactStore, *, failures: int) -> None:
        super().__init__(store)
        self.failures = failures
        self.scopes: list[Any] = []
        self.retrieval_scopes: list[str] = []

    async def allocate_ingestion_scope(self, request: ScopeAllocationRequest) -> ScopeReceipt:
        scope = await super().allocate_ingestion_scope(request)
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


def _pipeline(
    tmp_path: Path,
    memory: _PartialHistoryMemory,
) -> tuple[native._NativeExecutionState, Coroutine[Any, Any, Any]]:
    workload = _NativeFixtureWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    plan = workload.iter_ingestion_plans(manifest)[1]
    cases = tuple(
        case
        for case in workload.iter_case_plans(manifest)
        if case.case_manifest_entry_id in plan.ordered_case_manifest_entry_ids
    )
    state = native._NativeExecutionState(
        store=ArtifactStore(tmp_path),
        run_id="history-runtime-test",
        lease_record_hash="a" * 64,
        close_timeout_seconds=1,
    )
    operation = native._execute_history_question_pipeline(
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
    return state, operation


class BatchMemory(_PartialHistoryMemory):
    def __init__(self, store: ArtifactStore, failures: dict[int, int]) -> None:
        super().__init__(store, failures=0)
        self.remaining = dict(failures)
        self.submissions: list[tuple[int, int]] = []

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt:
        ordinal = request.dispatch.dispatch_ordinal_1_indexed
        self.submissions.append((ordinal, request.batch_attempt_ordinal))
        scope = request.scope.scope_id
        previous = self._accepted_by_scope.get(scope, ())
        source_ids = tuple(
            source.source_unit_id for source in request.dispatch.ordered_source_units
        )
        # A failed batch really leaves memory in this boundary; it is not rolled back.
        self._accepted_by_scope[scope] = tuple(dict.fromkeys((*previous, *source_ids)))
        if self.remaining.get(ordinal, 0):
            self.remaining[ordinal] -= 1
            payload = canonical_json_bytes(
                {
                    "detail": "Fact extraction failed: 1/1 chunks failed. First failures: "
                    "chunk 0: APIConnectionError: Connection error."
                }
            )
            raise SettledTransientIngestionFailure(
                "settled extraction failure after partial persistence",
                settlement_basis="hindsight_sync_extraction_drained_v1",
                internal_retry_count=10,
                failure_kind="supplier_connection",
                status_code=500,
                raw_reference=self._seal_raw(payload),
                raw_response_bytes=payload,
            )
        payload = canonical_json_bytes(
            {
                "operation": "ingest",
                "attempt_id": request.attempt_id,
                "ingestion_occurrence_id": request.scope.ingestion_occurrence_id,
                "scope_id": scope,
                "accepted_source_unit_ids": source_ids,
                "rejected_source_unit_ids": (),
                "source_units": [
                    {
                        "source_unit_id": source.source_unit_id,
                        "payload_sha256": source.payload_sha256,
                    }
                    for source in request.dispatch.ordered_source_units
                ],
            }
        )
        return IngestionDispatchReceipt(
            attempt_id=request.attempt_id,
            dispatch=request.dispatch,
            accepted_source_unit_ids=source_ids,
            rejected_source_unit_ids=(),
            raw_reference=self._seal_raw(payload),
            raw_response_bytes=payload,
            usage_records=(),
        )

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        assert set(request.expected_source_unit_ids) <= set(
            self._accepted_by_scope[request.scope.scope_id]
        )
        return ReadinessReceipt(
            ingestion_occurrence_id=request.scope.ingestion_occurrence_id,
            ready=True,
            evidence_references=(self._raw_reference({"operation": "wait_ready", "ready": True}),),
        )

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        self.retrieval_scopes.append(request.scope.scope_id)
        return await _RecordedNativeMemory.retrieve(self, request)


def test_exhausted_batch_keeps_partial_memory_and_continues_next_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    waits: list[int] = []

    async def sleep(seconds: int) -> None:
        waits.append(seconds)

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", sleep)
    memory = BatchMemory(ArtifactStore(tmp_path), failures={1: 3})
    _state, operation = _pipeline(tmp_path, memory)
    plans, cases = asyncio.run(operation)

    assert memory.submissions == [(1, 1), (1, 2), (1, 3), (2, 1)]
    assert len(memory.scopes) == 1
    assert waits == [10, 10]
    assert len(plans) == len(cases) == 1
    plan = plans[0]
    assert plan.skipped_source_unit_ids == (plan.ordered_source_unit_ids[0],)
    assert plan.accepted_source_unit_ids == (plan.ordered_source_unit_ids[1],)
    assert plan.projected_source_unit_ids == plan.ordered_source_unit_ids
    assert memory.retrieval_scopes == [memory.scopes[0].scope_id]
    assert not list((tmp_path / "source/history-retries").glob("*.json"))
    attempts = [json.loads(p.read_bytes()) for p in (tmp_path / "source/attempts").glob("*.json")]
    ingests = [attempt for attempt in attempts if attempt["stage"] == "memory_ingest"]
    assert len(ingests) == 4
    assert sum(attempt["outcome"] == "failed" for attempt in ingests) == 3
    assert len(plan.ordered_dispatch_attempt_ids) == 2
    assert {attempt["attempt_id"] for attempt in ingests} <= set(plan.attempt_ids)


def test_each_batch_receives_its_own_three_submission_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def sleep(_seconds: int) -> None:
        pass

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", sleep)
    memory = BatchMemory(ArtifactStore(tmp_path), failures={1: 2, 2: 2})
    _state, operation = _pipeline(tmp_path, memory)
    plans, cases = asyncio.run(operation)
    assert memory.submissions == [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
    assert len(memory.scopes) == 1
    assert not plans[0].skipped_source_unit_ids
    assert len(cases) == 1
