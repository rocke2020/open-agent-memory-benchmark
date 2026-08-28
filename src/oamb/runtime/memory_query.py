"""Read-only retrieval orchestration with separate projection timing."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, cast

from oamb.contracts.ports import (
    NativeEvidenceBatch,
    ProjectionReceipt,
    RetrievalRequest,
    ScopeReceipt,
)

from .timing import CapturedQueryTiming, capture_async_query_outcome


class ReadOnlyMemoryPort(Protocol):
    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt: ...

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch: ...


@dataclass(frozen=True, slots=True)
class ReadOnlyRetrievalReceipt:
    before_projection: ProjectionReceipt
    native_batch: NativeEvidenceBatch
    after_projection: ProjectionReceipt
    timing: CapturedQueryTiming[NativeEvidenceBatch]


@dataclass(frozen=True, slots=True)
class FailedReadOnlyRetrievalReceipt:
    before_projection: ProjectionReceipt
    native_batch: NativeEvidenceBatch | None
    after_projection: ProjectionReceipt | None
    provider_error: BaseException | None
    post_projection_error: BaseException | None
    projection_mutated: bool | None
    timing: CapturedQueryTiming[NativeEvidenceBatch | None]


class ReadOnlyRetrievalFailure(RuntimeError):
    """A dispatched retrieval or its required post-projection failed."""

    def __init__(self, receipt: FailedReadOnlyRetrievalReceipt) -> None:
        super().__init__("read-only retrieval failed after provider dispatch")
        self.receipt = receipt


class ReadOnlyRetrievalCancelled(asyncio.CancelledError):
    """A cancelled retrieval that retains its projections and timing."""

    def __init__(self, receipt: FailedReadOnlyRetrievalReceipt) -> None:
        super().__init__("read-only retrieval was cancelled after provider dispatch")
        self.receipt = receipt


class MemorySystemProjectionMutation(RuntimeError):
    """A provider query changed its protected native state."""

    def __init__(self, receipt: ReadOnlyRetrievalReceipt) -> None:
        super().__init__("memory-system query changed the protected projection")
        self.receipt = receipt


async def execute_read_only_retrieval(
    *,
    memory: ReadOnlyMemoryPort,
    request: RetrievalRequest,
    clock: Callable[[], Decimal],
) -> ReadOnlyRetrievalReceipt:
    before_projection: ProjectionReceipt | None = None
    after_projection: ProjectionReceipt | None = None
    expected_occurrence = request.scope.ingestion_occurrence_id

    def validate_projection_occurrence(projection: ProjectionReceipt) -> None:
        if (
            projection.inventory.ingestion_occurrence_id != expected_occurrence
            or projection.state_digest.ingestion_occurrence_id != expected_occurrence
        ):
            raise ValueError("projection belongs to a different ingestion occurrence")

    async def capture_before() -> ProjectionReceipt:
        nonlocal before_projection
        before_projection = await memory.project(request.scope)
        validate_projection_occurrence(before_projection)
        return before_projection

    async def capture_after() -> ProjectionReceipt:
        nonlocal after_projection
        after_projection = await memory.project(request.scope)
        validate_projection_occurrence(after_projection)
        return after_projection

    outcome = await capture_async_query_outcome(
        before_projection=capture_before,
        provider_request=lambda: memory.retrieve(request),
        after_projection=capture_after,
        clock=clock,
    )
    if before_projection is None:
        raise AssertionError("query timing returned without a before projection")
    if outcome.provider_error is not None or outcome.post_projection_error is not None:
        projection_mutated = (
            None
            if after_projection is None
            else _projection_changed(before_projection, after_projection)
        )
        failure_receipt = FailedReadOnlyRetrievalReceipt(
            before_projection=before_projection,
            native_batch=outcome.timing.result,
            after_projection=after_projection,
            provider_error=outcome.provider_error,
            post_projection_error=outcome.post_projection_error,
            projection_mutated=projection_mutated,
            timing=outcome.timing,
        )
        errors = (outcome.provider_error, outcome.post_projection_error)
        cancellation = next(
            (error for error in errors if isinstance(error, asyncio.CancelledError)),
            None,
        )
        if cancellation is not None:
            raise ReadOnlyRetrievalCancelled(failure_receipt) from cancellation
        primary_error = outcome.provider_error or outcome.post_projection_error
        if primary_error is None:
            raise AssertionError("failed retrieval outcome has no error")
        raise ReadOnlyRetrievalFailure(failure_receipt) from primary_error
    if after_projection is None or outcome.timing.result is None:
        raise AssertionError("successful query timing returned incomplete evidence")
    success_timing = cast(CapturedQueryTiming[NativeEvidenceBatch], outcome.timing)
    receipt = ReadOnlyRetrievalReceipt(
        before_projection=before_projection,
        native_batch=outcome.timing.result,
        after_projection=after_projection,
        timing=success_timing,
    )
    if _projection_changed(before_projection, after_projection):
        raise MemorySystemProjectionMutation(receipt)
    return receipt


def _projection_changed(
    before_projection: ProjectionReceipt,
    after_projection: ProjectionReceipt,
) -> bool:
    return (
        before_projection.inventory.ordered_source_unit_ids
        != after_projection.inventory.ordered_source_unit_ids
        or before_projection.state_digest.state_sha256 != after_projection.state_digest.state_sha256
    )


__all__ = [
    "FailedReadOnlyRetrievalReceipt",
    "MemorySystemProjectionMutation",
    "ReadOnlyMemoryPort",
    "ReadOnlyRetrievalCancelled",
    "ReadOnlyRetrievalFailure",
    "ReadOnlyRetrievalReceipt",
    "execute_read_only_retrieval",
]
