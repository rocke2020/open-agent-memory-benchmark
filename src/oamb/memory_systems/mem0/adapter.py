"""Zero-dispatch Mem0 v2.0.19 REST adapter boundary."""

from __future__ import annotations

import asyncio
from typing import Never, Protocol

from oamb.contracts.ports import (
    CapabilitySet,
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionRequest,
    InventoryReceipt,
    NativeEvidenceBatch,
    ProjectionReceipt,
    ReadinessReceipt,
    ReadinessRequest,
    RetrievalRequest,
    RuntimeResolution,
    ScopeAllocationRequest,
    ScopeReceipt,
    StateDigestReceipt,
)

from .profiles import (
    MEM0_REST_PROFILE,
    Mem0ExactProfile,
    Mem0OperationAuditEvent,
    Mem0ZeroDispatchAudit,
    _seal_mem0_zero_dispatch_audit,
    build_mem0_operation_audit_event,
    reference_profile_unsupported,
)


class Mem0RestDispatcher(Protocol):
    async def request(self, *args: object, **kwargs: object) -> object: ...

    async def close(self) -> None: ...


class _ZeroDispatchMem0Adapter:
    def __init__(self, *, profile: Mem0ExactProfile) -> None:
        self._profile = profile
        self._dispatched_operation_count = 0
        self._audit_events: list[Mem0OperationAuditEvent] = []

    async def resolve(self) -> RuntimeResolution:
        self._raise_unsupported("resolve")

    async def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            capability_ids=("exact-profile-negative-fixtures", "zero-dispatch-unsupported"),
            provider_order_preserved=True,
            native_reranking_disabled=True,
        )

    async def allocate_ingestion_scope(self, request: ScopeAllocationRequest) -> ScopeReceipt:
        del request
        self._raise_unsupported("allocate_ingestion_scope")

    def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]:
        del request
        self._raise_unsupported("plan_ingestion")

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt:
        del request
        self._raise_unsupported("ingest")

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        del request
        self._raise_unsupported("wait_ready")

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt:
        del scope
        self._raise_unsupported("inventory")

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt:
        del scope
        self._raise_unsupported("state_digest")

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        del scope
        self._raise_unsupported("project")

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        del request
        self._raise_unsupported("retrieve")

    def _raise_unsupported(self, operation_kind: str) -> Never:
        self._audit_events.append(
            build_mem0_operation_audit_event(
                profile=self._profile,
                sequence=len(self._audit_events) + 1,
                operation_kind=operation_kind,
                dispatcher_count_before=self._dispatched_operation_count,
                dispatcher_count_after=self._dispatched_operation_count,
            )
        )
        raise reference_profile_unsupported(self._profile)

    def validation_audit(self) -> Mem0ZeroDispatchAudit:
        return _seal_mem0_zero_dispatch_audit(
            profile=self._profile,
            ordered_events=tuple(self._audit_events),
        )


class Mem0RestAdapter(_ZeroDispatchMem0Adapter):
    """Expose exact negative fixtures while refusing all v2.0.19 REST dispatch."""

    def __init__(self, *, dispatcher: Mem0RestDispatcher | None = None) -> None:
        super().__init__(profile=MEM0_REST_PROFILE)
        self._dispatcher = dispatcher
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            if self._dispatcher is not None:
                await self._dispatcher.close()
            self._closed = True


__all__ = ["Mem0RestAdapter", "Mem0RestDispatcher"]
