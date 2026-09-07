"""MemorySystemPort implementation for pinned Hindsight v0.9.2 REST."""

from __future__ import annotations

import asyncio
import hashlib
import re

import httpx

from oamb.config.provider_services import HINDSIGHT_RETAIN_BATCH_LIMIT
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ingestion_failures import (
    HINDSIGHT_SETTLEMENT_BASIS,
    classify_settled_ingestion_failure,
)
from oamb.contracts.ports import (
    ArtifactStorePort,
    CapabilitySet,
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionRequest,
    InventoryReceipt,
    MemorySystemCallCancelledBeforeDispatch,
    MemorySystemCallFailure,
    MemorySystemReadCancelled,
    NativeEvidenceBatch,
    ProjectionReceipt,
    RawPayloadSealRequest,
    ReadinessReceipt,
    ReadinessRequest,
    RetrievalRequest,
    RuntimeResolution,
    ScopeAllocationRequest,
    ScopeReceipt,
    SettledTransientIngestionFailure,
    SourceUnit,
    StateDigestReceipt,
)
from oamb.memory_systems.rest import (
    DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS,
    DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS,
    SealedRestResponse,
    bind_sealed_response_validation,
    link_preceding_before_dispatch_cancellation,
    link_preceding_raw_references,
    link_preceding_read_cancellation,
    sealed_response_validation_failure,
)

from .client import HindsightClient
from .normalize import normalize_recall
from .profiles import (
    MEMORY_SYSTEM_ID,
    parse_bank_config,
    parse_bank_page,
    parse_bank_profile,
    parse_retain_response,
    parse_version_response,
    retain_usage_record,
)
from .projection import ProjectionSnapshot, build_projection

_BANK_PAGE_LIMIT = 1000
_RECALL_MAX_TOKENS = 32768


class HindsightAdapter:
    def __init__(
        self,
        *,
        store: ArtifactStorePort,
        base_url: str,
        extraction_model: str,
        runtime_binding_hash: str,
        authorization: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        read_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS,
        total_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS,
        internal_retry_count: int | None = None,
    ) -> None:
        if not extraction_model:
            raise ValueError("Hindsight extraction model is required")
        if re.fullmatch(r"[0-9a-f]{64}", runtime_binding_hash) is None:
            raise ValueError("Hindsight runtime binding hash must be lowercase SHA-256")
        self._store = store
        self._extraction_model = extraction_model
        self._runtime_binding_hash = runtime_binding_hash
        self._internal_retry_count = internal_retry_count
        self._resolved = False
        self._known_bank_ids: set[str] = set()
        self._bank_inventory_baseline_initialized = False
        self._attempted_allocations: set[str] = set()
        self._allocation_lock = asyncio.Lock()
        self._allocated_occurrences: dict[str, str] = {}
        self._planned_dispatches_by_scope: dict[str, tuple[IngestionDispatch, ...]] = {}
        self._attempted_dispatches: set[tuple[str, str, int]] = set()
        self._next_batch_attempt_ordinals: dict[tuple[str, str], int] = {}
        self._ready_sources: dict[str, tuple[SourceUnit, ...]] = {}
        self._ready_document_sources: dict[str, dict[str, str]] = {}
        self._ready_occurrences: dict[str, str] = {}
        self._client = HindsightClient(
            store=store,
            base_url=base_url,
            authorization=authorization,
            transport=transport,
            read_timeout_seconds=read_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
        )

    async def resolve(self) -> RuntimeResolution:
        response = await self._client.get_version()
        with bind_sealed_response_validation(
            response,
            message="Hindsight version response failed exact-profile validation",
        ):
            parse_version_response(response.raw_bytes)
        resolution = RuntimeResolution(
            memory_system_id=MEMORY_SYSTEM_ID,
            runtime_binding_hash=self._runtime_binding_hash,
            raw_reference=response.raw_reference,
        )
        self._resolved = True
        return resolution

    async def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            capability_ids=(
                "create-only-bank",
                "complete-projection",
                "explicit-ingestion-batching",
            ),
            provider_order_preserved=True,
            native_reranking_disabled=True,
        )

    async def allocate_ingestion_scope(
        self,
        request: ScopeAllocationRequest,
    ) -> ScopeReceipt:
        if not self._resolved:
            raise RuntimeError("Hindsight resolve must pass before scope allocation")
        async with self._allocation_lock:
            return await self._allocate_ingestion_scope_once(request)

    async def _allocate_ingestion_scope_once(
        self,
        request: ScopeAllocationRequest,
    ) -> ScopeReceipt:
        bank_id = hashlib.sha256(
            f"oamb-hindsight-bank-v1\0{request.ingestion_occurrence_id}".encode()
        ).hexdigest()
        if bank_id in self._attempted_allocations:
            raise ValueError("Hindsight bank allocation was already attempted and cannot replay")
        visible_bank_ids, inventory_responses = await self._complete_bank_inventory()
        terminal_inventory_response = inventory_responses[-1]
        inventory_references = tuple(response.raw_reference for response in inventory_responses)
        if bank_id in visible_bank_ids:
            raise sealed_response_validation_failure(
                terminal_inventory_response,
                message="Hindsight ingestion bank already exists",
                supporting_raw_references=inventory_references[:-1],
            )
        if not self._bank_inventory_baseline_initialized:
            self._known_bank_ids.update(visible_bank_ids)
            self._bank_inventory_baseline_initialized = True
        elif visible_bank_ids != self._known_bank_ids:
            raise sealed_response_validation_failure(
                terminal_inventory_response,
                message="Hindsight bank inventory contains an unexpected scope",
                supporting_raw_references=inventory_references[:-1],
            )
        self._attempted_allocations.add(bank_id)
        try:
            created = await self._client.create_bank(bank_id)
        except MemorySystemCallCancelledBeforeDispatch as exc:
            self._attempted_allocations.remove(bank_id)
            raise link_preceding_before_dispatch_cancellation(
                exc,
                preceding_raw_references=inventory_references,
            ) from exc
        except MemorySystemCallFailure as exc:
            raise link_preceding_raw_references(
                exc,
                preceding_raw_references=inventory_references,
            ) from exc
        with bind_sealed_response_validation(
            created,
            message="Hindsight bank-create response failed exact-profile validation",
            supporting_raw_references=inventory_references,
        ):
            parse_bank_profile(created.raw_bytes, expected_bank_id=bank_id)
        try:
            config = await self._client.get_bank_config(bank_id)
        except MemorySystemReadCancelled as exc:
            raise link_preceding_read_cancellation(
                exc,
                preceding_raw_references=(*inventory_references, created.raw_reference),
            ) from exc
        except MemorySystemCallFailure as exc:
            raise link_preceding_raw_references(
                exc,
                preceding_raw_references=(*inventory_references, created.raw_reference),
            ) from exc
        with bind_sealed_response_validation(
            config,
            message="Hindsight bank-config response failed exact-profile validation",
            supporting_raw_references=(*inventory_references, created.raw_reference),
        ):
            parse_bank_config(config.raw_bytes, expected_bank_id=bank_id)
        try:
            profile = await self._client.get_bank_profile(bank_id)
        except MemorySystemReadCancelled as exc:
            raise link_preceding_read_cancellation(
                exc,
                preceding_raw_references=(
                    *inventory_references,
                    created.raw_reference,
                    config.raw_reference,
                ),
            ) from exc
        except MemorySystemCallFailure as exc:
            raise link_preceding_raw_references(
                exc,
                preceding_raw_references=(
                    *inventory_references,
                    created.raw_reference,
                    config.raw_reference,
                ),
            ) from exc
        with bind_sealed_response_validation(
            profile,
            message="Hindsight bank-profile response failed exact-profile validation",
            supporting_raw_references=(
                *inventory_references,
                created.raw_reference,
                config.raw_reference,
            ),
        ):
            parse_bank_profile(profile.raw_bytes, expected_bank_id=bank_id)
        self._known_bank_ids.add(bank_id)
        self._allocated_occurrences[bank_id] = request.ingestion_occurrence_id
        return ScopeReceipt(
            ingestion_occurrence_id=request.ingestion_occurrence_id,
            scope_id=bank_id,
            raw_reference=created.raw_reference,
            supporting_raw_references=(
                *inventory_references,
                config.raw_reference,
                profile.raw_reference,
            ),
        )

    async def _complete_bank_inventory(
        self,
    ) -> tuple[set[str], tuple[SealedRestResponse, ...]]:
        bank_ids: list[str] = []
        responses: list[SealedRestResponse] = []
        offset = 0
        expected_total: int | None = None
        while True:
            try:
                response = await self._client.list_banks(
                    limit=_BANK_PAGE_LIMIT,
                    offset=offset,
                )
            except MemorySystemReadCancelled as exc:
                if not responses:
                    raise
                raise link_preceding_read_cancellation(
                    exc,
                    preceding_raw_references=tuple(
                        previous.raw_reference for previous in responses
                    ),
                ) from exc
            except MemorySystemCallFailure as exc:
                if not responses:
                    raise
                raise link_preceding_raw_references(
                    exc,
                    preceding_raw_references=tuple(
                        previous.raw_reference for previous in responses
                    ),
                ) from exc
            responses.append(response)
            with bind_sealed_response_validation(
                response,
                message="Hindsight bank-list response failed exact-profile validation",
                supporting_raw_references=tuple(
                    previous.raw_reference for previous in responses[:-1]
                ),
            ):
                page = parse_bank_page(
                    response.raw_bytes,
                    expected_limit=_BANK_PAGE_LIMIT,
                    expected_offset=offset,
                )
                if expected_total is None:
                    expected_total = page.total
                elif page.total != expected_total:
                    raise ValueError("Hindsight bank total changed during pagination")
                if any(bank_id in bank_ids for bank_id in page.bank_ids):
                    raise ValueError("Hindsight bank pagination returned a duplicate ID")
                bank_ids.extend(page.bank_ids)
                if len(bank_ids) > expected_total:
                    raise ValueError("Hindsight bank page exceeds its declared total")
                if len(bank_ids) == expected_total:
                    if page.bank_ids:
                        offset = expected_total
                        continue
                    break
                if not page.bank_ids:
                    raise ValueError("Hindsight bank pagination ended before its declared total")
                offset += len(page.bank_ids)
        return set(bank_ids), tuple(responses)

    def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]:
        self._require_allocated_scope(request.scope)
        if request.scope.scope_id in self._planned_dispatches_by_scope:
            raise ValueError("Hindsight ingestion dispatches are already planned and frozen")
        dispatches: list[IngestionDispatch] = []
        for start in range(
            0,
            len(request.ordered_source_units),
            HINDSIGHT_RETAIN_BATCH_LIMIT,
        ):
            source_units = request.ordered_source_units[
                start : start + HINDSIGHT_RETAIN_BATCH_LIMIT
            ]
            for source in source_units:
                self._retain_item(source)
            ordinal = len(dispatches) + 1
            dispatches.append(
                IngestionDispatch(
                    dispatch_ordinal_1_indexed=ordinal,
                    operation_kind="retain_extraction",
                    request_fingerprint=self._dispatch_fingerprint(
                        scope_id=request.scope.scope_id,
                        ordinal=ordinal,
                        source_units=source_units,
                    ),
                    ordered_source_units=source_units,
                )
            )
        planned_dispatches = tuple(dispatches)
        self._planned_dispatches_by_scope[request.scope.scope_id] = planned_dispatches
        return planned_dispatches

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt:
        self._require_allocated_scope(request.scope)
        dispatch = request.dispatch
        planned_dispatches = self._planned_dispatches_by_scope.get(request.scope.scope_id)
        ordinal_index = dispatch.dispatch_ordinal_1_indexed - 1
        if (
            planned_dispatches is None
            or ordinal_index < 0
            or ordinal_index >= len(planned_dispatches)
            or dispatch != planned_dispatches[ordinal_index]
        ):
            raise ValueError("Hindsight dispatch does not match the frozen plan")
        if (
            dispatch.operation_kind != "retain_extraction"
            or dispatch.dispatch_ordinal_1_indexed < 1
            or not dispatch.ordered_source_units
            or len(dispatch.ordered_source_units) > HINDSIGHT_RETAIN_BATCH_LIMIT
        ):
            raise ValueError("Hindsight retain dispatch does not match the exact profile")
        expected_fingerprint = self._dispatch_fingerprint(
            scope_id=request.scope.scope_id,
            ordinal=dispatch.dispatch_ordinal_1_indexed,
            source_units=dispatch.ordered_source_units,
        )
        if dispatch.request_fingerprint != expected_fingerprint:
            raise ValueError("Hindsight retain dispatch fingerprint does not match its request")
        dispatch_key = (request.scope.scope_id, request.dispatch.request_fingerprint)
        expected_batch_attempt = self._next_batch_attempt_ordinals.get(dispatch_key, 1)
        if request.batch_attempt_ordinal != expected_batch_attempt:
            raise ValueError("Hindsight retain batch attempt is not the next allowed retry")
        attempt_key = (*dispatch_key, request.batch_attempt_ordinal)
        if attempt_key in self._attempted_dispatches:
            raise ValueError("Hindsight retain dispatch was already attempted and cannot replay")
        self._attempted_dispatches.add(attempt_key)
        items = [self._retain_item(source) for source in request.dispatch.ordered_source_units]
        try:
            response = await self._client.retain(request.scope.scope_id, items)
        except MemorySystemCallCancelledBeforeDispatch:
            self._attempted_dispatches.remove(attempt_key)
            raise
        except MemorySystemCallFailure as exc:
            if (
                exc.status_code is not None
                and exc.raw_response_bytes is not None
                and exc.raw_reference is not None
                and self._internal_retry_count is not None
            ):
                reason = classify_settled_ingestion_failure(
                    settlement_basis=HINDSIGHT_SETTLEMENT_BASIS,
                    status_code=exc.status_code,
                    raw_response_bytes=exc.raw_response_bytes,
                    internal_retry_count=self._internal_retry_count,
                )
                if reason is not None:
                    self._next_batch_attempt_ordinals[dispatch_key] = (
                        request.batch_attempt_ordinal + 1
                    )
                    raise SettledTransientIngestionFailure(
                        "Hindsight synchronous extraction settled with a supplier failure",
                        failure_kind=reason,
                        raw_reference=exc.raw_reference,
                        raw_response_bytes=exc.raw_response_bytes,
                        supporting_raw_references=exc.supporting_raw_references,
                        status_code=exc.status_code,
                        settlement_basis=HINDSIGHT_SETTLEMENT_BASIS,
                        internal_retry_count=self._internal_retry_count,
                    ) from exc
            raise
        with bind_sealed_response_validation(
            response,
            message="Hindsight retain response failed exact-profile validation",
        ):
            result = parse_retain_response(
                response.raw_bytes,
                expected_bank_id=request.scope.scope_id,
                expected_items_count=len(items),
            )
            usage = retain_usage_record(
                result,
                attempt_id=request.attempt_id,
                parent_id=request.scope.ingestion_occurrence_id,
                model=self._extraction_model,
                raw_reference=response.raw_reference,
            )
        source_ids = tuple(
            source.source_unit_id for source in request.dispatch.ordered_source_units
        )
        self._next_batch_attempt_ordinals[dispatch_key] = 0
        return IngestionDispatchReceipt(
            attempt_id=request.attempt_id,
            dispatch=request.dispatch,
            accepted_source_unit_ids=source_ids,
            rejected_source_unit_ids=(),
            raw_reference=response.raw_reference,
            raw_response_bytes=response.raw_bytes,
            usage_records=(usage,),
        )

    @staticmethod
    def _retain_item(source: SourceUnit) -> dict[str, object]:
        if hashlib.sha256(source.payload_bytes).hexdigest() != source.payload_sha256:
            raise ValueError("Hindsight source payload hash does not match its bytes")
        if source.occurred_at is None or source.context_text is None:
            raise ValueError("Hindsight retain source requires timestamp and context")
        return {
            "content": source.payload_bytes.decode("utf-8", errors="strict"),
            "timestamp": source.occurred_at,
            "context": source.context_text,
            "metadata": dict(source.source_metadata),
            "document_id": source.source_unit_id,
        }

    def _dispatch_fingerprint(
        self,
        *,
        scope_id: str,
        ordinal: int,
        source_units: tuple[SourceUnit, ...],
    ) -> str:
        return canonical_sha256(
            [
                "oamb-hindsight-retain-dispatch-v1",
                scope_id,
                ordinal,
                tuple(self._retain_item(source) for source in source_units),
            ]
        )

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        sources = self._validate_readiness_request(request)
        snapshot = await build_projection(
            self._client,
            bank_id=request.scope.scope_id,
            ordered_sources=sources,
        )
        self._ready_sources[request.scope.scope_id] = sources
        self._ready_document_sources[request.scope.scope_id] = dict(
            snapshot.document_to_source_unit
        )
        self._ready_occurrences[request.scope.scope_id] = request.scope.ingestion_occurrence_id
        return ReadinessReceipt(
            ingestion_occurrence_id=request.scope.ingestion_occurrence_id,
            ready=True,
            evidence_references=(
                *request.ingestion_receipt.raw_references,
                *snapshot.raw_references,
            ),
        )

    def _validate_readiness_request(
        self,
        request: ReadinessRequest,
    ) -> tuple[SourceUnit, ...]:
        self._require_allocated_scope(request.scope)
        receipt = request.ingestion_receipt
        planned_dispatches = self._planned_dispatches_by_scope.get(request.scope.scope_id)
        if planned_dispatches is None:
            raise ValueError("Hindsight readiness has no frozen ingestion plan")
        planned_source_ids = tuple(
            source.source_unit_id
            for dispatch in planned_dispatches
            for source in dispatch.ordered_source_units
        )
        if receipt.ingestion_occurrence_id != request.scope.ingestion_occurrence_id:
            raise ValueError("Hindsight readiness receipt names another ingestion occurrence")
        if not planned_source_ids:
            raise ValueError("Hindsight readiness requires expected source IDs")
        if request.expected_source_unit_ids != receipt.accepted_source_unit_ids:
            raise ValueError("Hindsight readiness expected inventory differs from accepted sources")
        if tuple(item.dispatch for item in receipt.dispatch_receipts) != planned_dispatches:
            raise ValueError("Hindsight readiness receipts do not match the frozen dispatch set")
        if (
            receipt.accepted_source_unit_ids
            != tuple(
                source_id
                for item in receipt.dispatch_receipts
                for source_id in item.accepted_source_unit_ids
            )
            or receipt.rejected_source_unit_ids
            != tuple(
                source_id
                for item in receipt.dispatch_receipts
                for source_id in item.rejected_source_unit_ids
            )
            or receipt.skipped_source_unit_ids
            != tuple(
                source_id
                for item in receipt.dispatch_receipts
                for source_id in item.skipped_source_unit_ids
            )
        ):
            raise ValueError("Hindsight readiness aggregate source partition is invalid")
        expected_raw_references = tuple(
            dispatch_receipt.raw_reference for dispatch_receipt in receipt.dispatch_receipts
        )
        if receipt.raw_references != expected_raw_references:
            raise ValueError("Hindsight readiness receipt raw references are not dispatch-bound")
        sources: list[SourceUnit] = []
        for ordinal, dispatch_receipt in enumerate(receipt.dispatch_receipts, start=1):
            dispatch = dispatch_receipt.dispatch
            if dispatch.dispatch_ordinal_1_indexed != ordinal:
                raise ValueError("Hindsight readiness dispatch order is not contiguous")
            dispatch_source_ids = tuple(
                source.source_unit_id for source in dispatch.ordered_source_units
            )
            partitions = (
                set(dispatch_receipt.accepted_source_unit_ids),
                set(dispatch_receipt.rejected_source_unit_ids),
                set(dispatch_receipt.skipped_source_unit_ids),
            )
            if any(
                left & right
                for index, left in enumerate(partitions)
                for right in partitions[index + 1 :]
            ) or set().union(*partitions) != set(dispatch_source_ids):
                raise ValueError("Hindsight readiness dispatch receipt is not source-bound")
            if (
                hashlib.sha256(dispatch_receipt.raw_response_bytes).hexdigest()
                != dispatch_receipt.raw_reference.sha256
            ):
                raise ValueError("Hindsight readiness raw receipt hash does not match")
            if dispatch_receipt.skipped_source_unit_ids:
                sources.extend(
                    source
                    for source in dispatch.ordered_source_units
                    if source.source_unit_id not in dispatch_receipt.rejected_source_unit_ids
                )
                continue
            retain_result = parse_retain_response(
                dispatch_receipt.raw_response_bytes,
                expected_bank_id=request.scope.scope_id,
                expected_items_count=len(dispatch.ordered_source_units),
            )
            expected_usage = retain_usage_record(
                retain_result,
                attempt_id=dispatch_receipt.attempt_id,
                parent_id=request.scope.ingestion_occurrence_id,
                model=self._extraction_model,
                raw_reference=dispatch_receipt.raw_reference,
            )
            if dispatch_receipt.usage_records != (expected_usage,):
                raise ValueError("Hindsight readiness usage is not bound to the raw receipt")
            sources.extend(dispatch.ordered_source_units)
        source_ids = tuple(source.source_unit_id for source in sources)
        observable_source_ids = tuple(
            source_id
            for source_id in planned_source_ids
            if source_id not in receipt.rejected_source_unit_ids
        )
        if source_ids != observable_source_ids:
            raise ValueError("Hindsight readiness dispatch sources do not match observable order")
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Hindsight readiness source IDs are not unique")
        return tuple(sources)

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt:
        return (await self.project(scope)).inventory

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt:
        return (await self.project(scope)).state_digest

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        snapshot = await self._snapshot(scope)
        raw_reference = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=snapshot.state_sha256,
                media_type="application/json",
                compression="gzip",
                payload_bytes=snapshot.canonical_bytes,
            )
        )
        return ProjectionReceipt(
            inventory=InventoryReceipt(
                ingestion_occurrence_id=scope.ingestion_occurrence_id,
                ordered_source_unit_ids=snapshot.ordered_source_unit_ids,
                raw_reference=raw_reference,
            ),
            state_digest=StateDigestReceipt(
                ingestion_occurrence_id=scope.ingestion_occurrence_id,
                state_sha256=snapshot.state_sha256,
                raw_reference=raw_reference,
            ),
            supporting_raw_references=snapshot.raw_references,
        )

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        self._require_ready_scope(request.scope)
        try:
            query = request.query_bytes.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ValueError("Hindsight recall query must be UTF-8") from exc
        if not query.strip():
            raise ValueError("Hindsight recall query must be non-empty")
        response = await self._client.recall(
            request.scope.scope_id,
            {
                "query": query,
                "types": ["world", "experience"],
                "budget": "high",
                "max_tokens": _RECALL_MAX_TOKENS,
                "query_timestamp": request.query_timestamp,
                "trace": True,
                "include": {"entities": None, "chunks": {}},
            },
        )
        try:
            candidates = normalize_recall(
                response.raw_bytes,
                document_to_source_unit=self._ready_document_sources[request.scope.scope_id],
            )
        except ValueError as exc:
            raise sealed_response_validation_failure(
                response,
                message="Hindsight recall response failed exact-profile validation",
            ) from exc
        return NativeEvidenceBatch(
            raw_reference=response.raw_reference,
            candidates=candidates,
            request_raw_reference=response.request_reference,
        )

    async def _snapshot(self, scope: ScopeReceipt) -> ProjectionSnapshot:
        self._require_ready_scope(scope)
        sources = self._ready_sources[scope.scope_id]
        return await build_projection(
            self._client,
            bank_id=scope.scope_id,
            ordered_sources=sources,
        )

    def _require_ready_scope(self, scope: ScopeReceipt) -> None:
        try:
            occurrence_id = self._ready_occurrences[scope.scope_id]
        except KeyError as exc:
            raise ValueError("Hindsight scope is not ready for projection") from exc
        if occurrence_id != scope.ingestion_occurrence_id:
            raise ValueError("Hindsight projection scope binding changed")

    def _require_allocated_scope(self, scope: ScopeReceipt) -> None:
        if self._allocated_occurrences.get(scope.scope_id) != scope.ingestion_occurrence_id:
            raise ValueError("Hindsight scope was not allocated by this adapter")

    async def close(self) -> None:
        await self._client.close()
