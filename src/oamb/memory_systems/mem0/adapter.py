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

from .profiles import MEM0_REST_PROFILE, Mem0ExactProfile, reference_profile_unsupported


class Mem0RestDispatcher(Protocol):
    async def request(self, *args: object, **kwargs: object) -> object: ...

    async def close(self) -> None: ...


class _ZeroDispatchMem0Adapter:
    def __init__(self, *, profile: Mem0ExactProfile) -> None:
        self._profile = profile

    async def resolve(self) -> RuntimeResolution:
        self._raise_unsupported()

    async def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            capability_ids=("exact-profile-negative-fixtures", "zero-dispatch-unsupported"),
            provider_order_preserved=True,
            native_reranking_disabled=True,
        )

    async def allocate_ingestion_scope(self, request: ScopeAllocationRequest) -> ScopeReceipt:
        del request
        self._raise_unsupported()

    def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]:
        del request
        self._raise_unsupported()

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt:
        del request
        self._raise_unsupported()

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        del request
        self._raise_unsupported()

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt:
        del scope
        self._raise_unsupported()

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt:
        del scope
        self._raise_unsupported()

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        del scope
        self._raise_unsupported()

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        del request
        self._raise_unsupported()

    def _raise_unsupported(self) -> Never:
        raise reference_profile_unsupported(self._profile)


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
