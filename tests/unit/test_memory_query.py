from __future__ import annotations

import asyncio
from decimal import Decimal
from importlib import import_module
from typing import Any

import pytest

from oamb.contracts.ports import (
    InventoryReceipt,
    NativeEvidenceBatch,
    ProjectionReceipt,
    RawReferenceHandle,
    RetrievalRequest,
    ScopeReceipt,
    StateDigestReceipt,
)


class ManualClock:
    def __init__(self) -> None:
        self.value = Decimal("0")

    def __call__(self) -> Decimal:
        return self.value

    def advance(self, seconds: str) -> None:
        self.value += Decimal(seconds)


class ProjectingMemory:
    def __init__(
        self,
        clock: ManualClock,
        *,
        mutate: bool = False,
        wrong_before_occurrence: bool = False,
        provider_failure: bool = False,
        provider_cancelled: bool = False,
        post_projection_failure: bool = False,
    ) -> None:
        self.clock = clock
        self.mutate = mutate
        self.wrong_before_occurrence = wrong_before_occurrence
        self.provider_failure = provider_failure
        self.provider_cancelled = provider_cancelled
        self.post_projection_failure = post_projection_failure
        self.projection_calls = 0
        self.retrieve_calls = 0

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        self.clock.advance("2" if self.projection_calls == 0 else "4")
        self.projection_calls += 1
        if self.post_projection_failure and self.projection_calls == 2:
            raise RuntimeError("fixture post-projection failed")
        digest = "b" * 64 if self.mutate and self.projection_calls == 2 else "a" * 64
        occurrence_id = (
            "9" * 64
            if self.wrong_before_occurrence and self.projection_calls == 1
            else scope.ingestion_occurrence_id
        )
        return ProjectionReceipt(
            inventory=InventoryReceipt(
                ingestion_occurrence_id=occurrence_id,
                ordered_source_unit_ids=("source-1",),
                raw_reference=RawReferenceHandle(f"{self.projection_calls:064x}"),
            ),
            state_digest=StateDigestReceipt(
                ingestion_occurrence_id=occurrence_id,
                state_sha256=digest,
                raw_reference=RawReferenceHandle(f"{self.projection_calls + 2:064x}"),
            ),
        )

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        self.clock.advance("3")
        self.retrieve_calls += 1
        if self.provider_cancelled:
            raise asyncio.CancelledError("fixture provider cancelled")
        if self.provider_failure:
            raise RuntimeError("fixture provider failed")
        return NativeEvidenceBatch(
            raw_reference=RawReferenceHandle("f" * 64),
            candidates=(),
        )


class BlockingProjectingMemory(ProjectingMemory):
    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock)
        self.provider_entered = asyncio.Event()

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        del request
        self.clock.advance("3")
        self.retrieve_calls += 1
        self.provider_entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")


def _query_module() -> Any:
    try:
        return import_module("oamb.runtime.memory_query")
    except ModuleNotFoundError:
        pytest.fail("read-only memory query coordinator is not implemented")


def _request() -> RetrievalRequest:
    scope = ScopeReceipt(
        ingestion_occurrence_id="1" * 64,
        scope_id="scope-1",
        raw_reference=RawReferenceHandle("2" * 64),
    )
    return RetrievalRequest(
        scope=scope,
        case_occurrence_id="3" * 64,
        query_bytes=b"question",
        top_k=100,
    )


@pytest.mark.asyncio
async def test_read_only_query_separates_timing_and_preserves_both_projections() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = ProjectingMemory(clock)

    receipt = await query.execute_read_only_retrieval(
        memory=memory,
        request=_request(),
        clock=clock,
    )

    assert memory.projection_calls == 2
    assert memory.retrieve_calls == 1
    assert receipt.before_projection.state_digest.state_sha256 == "a" * 64
    assert receipt.after_projection.state_digest.state_sha256 == "a" * 64
    assert receipt.timing.provider_request.seconds == Decimal("3")
    assert receipt.timing.projection_verification.seconds == Decimal("6")
    assert receipt.timing.case_total.seconds == Decimal("9")


@pytest.mark.asyncio
async def test_read_only_query_preserves_evidence_when_projection_mutates() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = ProjectingMemory(clock, mutate=True)

    with pytest.raises(query.MemorySystemProjectionMutation) as mutation:
        await query.execute_read_only_retrieval(
            memory=memory,
            request=_request(),
            clock=clock,
        )

    assert mutation.value.receipt.native_batch.raw_reference.sha256 == "f" * 64
    assert mutation.value.receipt.before_projection.state_digest.state_sha256 == "a" * 64
    assert mutation.value.receipt.after_projection.state_digest.state_sha256 == "b" * 64
    assert memory.retrieve_calls == 1


@pytest.mark.asyncio
async def test_read_only_query_rejects_wrong_before_projection_without_dispatch() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = ProjectingMemory(clock, wrong_before_occurrence=True)

    with pytest.raises(ValueError, match="different ingestion occurrence"):
        await query.execute_read_only_retrieval(
            memory=memory,
            request=_request(),
            clock=clock,
        )

    assert memory.projection_calls == 1
    assert memory.retrieve_calls == 0


@pytest.mark.asyncio
async def test_provider_failure_retains_post_projection_and_complete_timing() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = ProjectingMemory(clock, provider_failure=True)

    with pytest.raises(query.ReadOnlyRetrievalFailure) as failure:
        await query.execute_read_only_retrieval(
            memory=memory,
            request=_request(),
            clock=clock,
        )

    receipt = failure.value.receipt
    assert memory.projection_calls == 2
    assert memory.retrieve_calls == 1
    assert receipt.native_batch is None
    assert receipt.after_projection is not None
    assert isinstance(receipt.provider_error, RuntimeError)
    assert receipt.post_projection_error is None
    assert receipt.projection_mutated is False
    assert receipt.timing.provider_request.seconds == Decimal("3")
    assert receipt.timing.projection_verification.seconds == Decimal("6")
    assert receipt.timing.case_total.seconds == Decimal("9")


@pytest.mark.asyncio
async def test_provider_cancellation_retains_cancel_semantics_projection_and_timing() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = ProjectingMemory(clock, provider_cancelled=True)

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await query.execute_read_only_retrieval(
            memory=memory,
            request=_request(),
            clock=clock,
        )

    assert isinstance(cancelled.value, query.ReadOnlyRetrievalCancelled)
    receipt = cancelled.value.receipt
    assert memory.projection_calls == 2
    assert memory.retrieve_calls == 1
    assert receipt.after_projection is not None
    assert isinstance(receipt.provider_error, asyncio.CancelledError)
    assert receipt.post_projection_error is None
    assert receipt.timing.provider_request.seconds == Decimal("3")
    assert receipt.timing.projection_verification.seconds == Decimal("6")


@pytest.mark.asyncio
async def test_external_task_cancellation_still_captures_post_projection() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = BlockingProjectingMemory(clock)
    task = asyncio.create_task(
        query.execute_read_only_retrieval(
            memory=memory,
            request=_request(),
            clock=clock,
        )
    )
    await memory.provider_entered.wait()

    task.cancel()
    with pytest.raises(query.ReadOnlyRetrievalCancelled) as cancelled:
        await task

    receipt = cancelled.value.receipt
    assert memory.projection_calls == 2
    assert memory.retrieve_calls == 1
    assert receipt.after_projection is not None
    assert isinstance(receipt.provider_error, asyncio.CancelledError)
    assert receipt.timing.provider_request.seconds == Decimal("3")
    assert receipt.timing.projection_verification.seconds == Decimal("6")


@pytest.mark.asyncio
async def test_provider_and_post_projection_failures_are_both_preserved() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = ProjectingMemory(
        clock,
        provider_failure=True,
        post_projection_failure=True,
    )

    with pytest.raises(query.ReadOnlyRetrievalFailure) as failure:
        await query.execute_read_only_retrieval(
            memory=memory,
            request=_request(),
            clock=clock,
        )

    receipt = failure.value.receipt
    assert receipt.native_batch is None
    assert receipt.after_projection is None
    assert isinstance(receipt.provider_error, RuntimeError)
    assert isinstance(receipt.post_projection_error, RuntimeError)
    assert receipt.projection_mutated is None
    assert receipt.timing.case_total.seconds == Decimal("9")


@pytest.mark.asyncio
async def test_provider_failure_also_records_post_projection_mutation() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = ProjectingMemory(clock, provider_failure=True, mutate=True)

    with pytest.raises(query.ReadOnlyRetrievalFailure) as failure:
        await query.execute_read_only_retrieval(
            memory=memory,
            request=_request(),
            clock=clock,
        )

    receipt = failure.value.receipt
    assert receipt.after_projection is not None
    assert receipt.projection_mutated is True
    assert receipt.before_projection.state_digest.state_sha256 == "a" * 64
    assert receipt.after_projection.state_digest.state_sha256 == "b" * 64


@pytest.mark.asyncio
async def test_post_projection_failure_retains_native_response_and_timing() -> None:
    query = _query_module()
    clock = ManualClock()
    memory = ProjectingMemory(clock, post_projection_failure=True)

    with pytest.raises(query.ReadOnlyRetrievalFailure) as failure:
        await query.execute_read_only_retrieval(
            memory=memory,
            request=_request(),
            clock=clock,
        )

    receipt = failure.value.receipt
    assert receipt.native_batch is not None
    assert receipt.native_batch.raw_reference.sha256 == "f" * 64
    assert receipt.provider_error is None
    assert isinstance(receipt.post_projection_error, RuntimeError)
    assert receipt.timing.provider_request.seconds == Decimal("3")
    assert receipt.timing.projection_verification.seconds == Decimal("6")
