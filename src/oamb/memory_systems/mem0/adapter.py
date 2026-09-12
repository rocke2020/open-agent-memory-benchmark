"""Mem0 v2.0.19 black-box REST adapter and historical negative fixture."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Never, Protocol

import httpx

from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ingestion_failures import (
    MEM0_SETTLEMENT_BASIS,
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
    NativeEvidenceCandidate,
    ProjectionReceipt,
    RawPayloadSealRequest,
    RawReferenceHandle,
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
    SealedRestClient,
    SealedRestResponse,
    bind_sealed_response_validation,
    link_preceding_raw_references,
    link_preceding_read_cancellation,
    parse_exact_json_object,
    sealed_response_validation_failure,
)

from .profiles import (
    MEM0_RELEASE_VERSION,
    MEM0_REST_PROFILE,
    Mem0ExactProfile,
    Mem0OperationAuditEvent,
    Mem0ZeroDispatchAudit,
    _seal_mem0_zero_dispatch_audit,
    build_mem0_operation_audit_event,
    reference_profile_unsupported,
)
from .projection import (
    MEM0_PROJECTION_MAX_PAGES,
    Mem0Projection,
    parse_projection_pages,
    projection_source_unit_ids,
    projection_state_sha256,
)
from .wire import (
    Mem0RestRequest,
    Mem0SourceMetadata,
    build_add_http_request,
    build_search_http_request,
    parse_add_response,
    parse_search_response,
)

MEM0_MEMORY_SYSTEM_ID = "mem0"
MEM0_COLLECTION = "oamb_memories"
MEM0_ADD_OPERATION = "mem0_add"
MEM0_OPENAPI_TITLE = "Mem0 REST APIs"
MEM0_OPENAPI_VERSION = "1.0.0"
MEM0_INSPECTOR_MODE = "read_only_projection"

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_PROJECTION_PAGE_FIELDS = frozenset({"collection", "run_id", "count", "points", "next_cursor"})


class Mem0RestDispatcher(Protocol):
    async def request(self, *args: object, **kwargs: object) -> object: ...

    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class _ScopeBinding:
    ingestion_occurrence_id: str
    ingestion_plan_id: str


@dataclass(frozen=True, slots=True)
class _PlannedAdd:
    dispatch: IngestionDispatch
    requests: tuple[Mem0RestRequest, ...]


@dataclass(frozen=True, slots=True)
class _CapturedProjection:
    native: Mem0Projection
    receipt: ProjectionReceipt


class _ZeroDispatchMem0Adapter:
    """Historical exact-profile negative producer retained for SDK fixtures."""

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


class Mem0ReferenceNegativeAdapter(_ZeroDispatchMem0Adapter):
    """Preserve the T8 v2.0.19 negative fixture without gating live REST."""

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


class Mem0RestAdapter:
    """Original REST adapter for the exact Mem0 v2.0.19 black-box profile."""

    def __init__(
        self,
        *,
        store: ArtifactStorePort,
        base_url: str,
        api_key: str,
        inspector_base_url: str,
        inspector_api_key: str,
        runtime_binding_hash: str,
        transport: httpx.AsyncBaseTransport | None = None,
        inspector_transport: httpx.AsyncBaseTransport | None = None,
        read_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS,
        total_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS,
        internal_retry_count: int | None = None,
    ) -> None:
        for label, value in (
            ("api_key", api_key),
            ("inspector_api_key", inspector_api_key),
            ("runtime_binding_hash", runtime_binding_hash),
        ):
            if not value:
                raise ValueError(f"Mem0 {label} is required")
        if _RUN_ID_PATTERN.fullmatch(runtime_binding_hash) is None:
            raise ValueError("Mem0 runtime binding hash must be lowercase SHA-256")
        self._store = store
        self._runtime_binding_hash = runtime_binding_hash
        self._internal_retry_count = internal_retry_count
        self._public_client = SealedRestClient(
            store=store,
            base_url=base_url,
            headers={"X-API-Key": api_key},
            transport=transport,
            read_timeout_seconds=read_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
        )
        self._inspector_client = SealedRestClient(
            store=store,
            base_url=inspector_base_url,
            headers={"Authorization": f"Bearer {inspector_api_key}"},
            transport=inspector_transport,
            read_timeout_seconds=read_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
        )
        self._resolved = False
        self._scopes: dict[str, _ScopeBinding] = {}
        self._allocation_lock = asyncio.Lock()
        self._ingestion_plan_locks: dict[str, asyncio.Lock] = {}
        self._planned_adds: dict[str, tuple[_PlannedAdd, ...]] = {}
        self._next_ingestion_dispatch_ordinal: dict[str, int] = {}
        self._attempted_dispatches: set[tuple[str, str, int]] = set()
        self._next_batch_attempt_ordinals: dict[tuple[str, str], int] = {}
        self._completed_add_response_refs: dict[tuple[str, str], tuple[str, ...]] = {}
        self._ready_projections: dict[str, _CapturedProjection] = {}
        self._projection_capture_sequences: dict[str, int] = {}
        self._close_lock = asyncio.Lock()
        self._accepting_operations = True
        self._public_closed = False
        self._inspector_closed = False

    async def resolve(self) -> RuntimeResolution:
        self._require_open()
        openapi = await self._public_client.request("GET", "/openapi.json")
        with bind_sealed_response_validation(
            openapi,
            message="Mem0 OpenAPI response failed exact-profile validation",
        ):
            _validate_openapi(openapi.raw_bytes)
        health = await self._inspector_client.request("GET", "/health")
        with bind_sealed_response_validation(
            health,
            message="Mem0 inspector health failed exact-profile validation",
            supporting_raw_references=(openapi.raw_reference,),
        ):
            document = parse_exact_json_object(
                health.raw_bytes,
                expected_fields=frozenset({"status", "mode"}),
            )
            if document != {"status": "ok", "mode": MEM0_INSPECTOR_MODE}:
                raise ValueError("Mem0 inspector is not the read-only projection profile")
        binding_bytes = canonical_json_bytes(
            {
                "schema": "oamb-mem0-runtime-resolution-v1",
                "release_version": MEM0_RELEASE_VERSION,
                "runtime_binding_hash": self._runtime_binding_hash,
                "openapi_raw_ref": openapi.raw_reference.sha256,
                "inspector_health_raw_ref": health.raw_reference.sha256,
            }
        )
        self._resolved = True
        return RuntimeResolution(
            memory_system_id=MEM0_MEMORY_SYSTEM_ID,
            runtime_binding_hash=self._runtime_binding_hash,
            raw_reference=self._seal_raw(binding_bytes),
        )

    async def capabilities(self) -> CapabilitySet:
        self._require_open()
        return CapabilitySet(
            capability_ids=(
                "create-only-run-scope",
                "retrieval-visible-main-projection",
                "opaque-empty-provider-outcome",
            ),
            provider_order_preserved=True,
            native_reranking_disabled=True,
        )

    async def allocate_ingestion_scope(
        self,
        request: ScopeAllocationRequest,
    ) -> ScopeReceipt:
        self._require_open()
        if not self._resolved:
            raise RuntimeError("Mem0 resolve must pass before scope allocation")
        _require_run_id(request.ingestion_occurrence_id)
        async with self._allocation_lock:
            if request.ingestion_occurrence_id in self._scopes:
                raise ValueError("Mem0 run scope was already allocated and cannot replay")
            captured = await self._capture_native_projection(request.ingestion_occurrence_id)
            if captured.native.declared_count != 0 or captured.native.points:
                raise ValueError("Mem0 run scope already contains retrieval-visible memory")
            self._scopes[request.ingestion_occurrence_id] = _ScopeBinding(
                ingestion_occurrence_id=request.ingestion_occurrence_id,
                ingestion_plan_id=request.ingestion_plan_id,
            )
            self._ingestion_plan_locks[request.ingestion_occurrence_id] = asyncio.Lock()
            return ScopeReceipt(
                ingestion_occurrence_id=request.ingestion_occurrence_id,
                scope_id=request.ingestion_occurrence_id,
                raw_reference=captured.receipt.inventory.raw_reference,
                supporting_raw_references=captured.receipt.supporting_raw_references,
            )

    def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]:
        self._require_open()
        binding = self._require_scope(request.scope)
        if request.scope.scope_id in self._planned_adds:
            raise ValueError("Mem0 ingestion dispatches are already planned and frozen")
        sources = request.ordered_source_units
        if not sources:
            raise ValueError("Mem0 ingestion requires at least one source unit")
        if tuple(source.ordinal_1_indexed for source in sources) != tuple(
            range(1, len(sources) + 1)
        ):
            raise ValueError("Mem0 source ordinals must be contiguous and ordered")
        source_ids = tuple(source.source_unit_id for source in sources)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("Mem0 source-unit IDs must be unique")

        planned: list[_PlannedAdd] = []
        for ordinal, source in enumerate(sources, start=1):
            metadata = Mem0SourceMetadata(
                ingestion_occurrence_id=binding.ingestion_occurrence_id,
                ingestion_plan_id=binding.ingestion_plan_id,
                source_unit_id=source.source_unit_id,
                source_ordinal=source.ordinal_1_indexed,
            )
            wire_requests = tuple(
                build_add_http_request(
                    messages=messages,
                    run_id=request.scope.scope_id,
                    metadata=metadata,
                    occurred_at=source.occurred_at,
                    context_text=source.context_text,
                )
                for messages in _source_message_pairs(source)
            )
            dispatch = IngestionDispatch(
                dispatch_ordinal_1_indexed=ordinal,
                operation_kind=MEM0_ADD_OPERATION,
                request_fingerprint=canonical_sha256(
                    [
                        "oamb-mem0-add-dispatch-v1",
                        request.scope.scope_id,
                        ordinal,
                        source.source_unit_id,
                        tuple(item.body.decode("utf-8") for item in wire_requests),
                    ]
                ),
                ordered_source_units=(source,),
            )
            planned.append(_PlannedAdd(dispatch=dispatch, requests=wire_requests))
        result = tuple(planned)
        self._planned_adds[request.scope.scope_id] = result
        self._next_ingestion_dispatch_ordinal[request.scope.scope_id] = 1
        return tuple(item.dispatch for item in result)

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt:
        self._require_open()
        self._require_scope(request.scope)
        planned = self._planned_add(request.scope.scope_id, request.dispatch)
        dispatch_key = (request.scope.scope_id, request.dispatch.request_fingerprint)
        ingestion_plan_lock = self._ingestion_plan_locks[request.scope.scope_id]
        try:
            await ingestion_plan_lock.acquire()
        except asyncio.CancelledError as exc:
            raise MemorySystemCallCancelledBeforeDispatch(
                "Mem0 add was cancelled before its ingestion-plan dispatch turn"
            ) from exc
        try:
            expected_ordinal = self._next_ingestion_dispatch_ordinal[request.scope.scope_id]
            if request.dispatch.dispatch_ordinal_1_indexed != expected_ordinal:
                raise ValueError("Mem0 add dispatch is not the next ordered source")
            expected_batch_attempt = self._next_batch_attempt_ordinals.get(dispatch_key, 1)
            if request.batch_attempt_ordinal != expected_batch_attempt:
                raise ValueError("Mem0 add batch attempt is not the next allowed retry")
            attempt_key = (*dispatch_key, request.batch_attempt_ordinal)
            if attempt_key in self._attempted_dispatches:
                raise ValueError("Mem0 add dispatch was already attempted and cannot replay")
            self._attempted_dispatches.add(attempt_key)
            responses: list[SealedRestResponse] = []
            partial_failure: MemorySystemCallFailure | None = None
            for wire_request in planned.requests:
                try:
                    response = await self._public_client.request(
                        wire_request.method,
                        wire_request.path,
                        json_payload=json.loads(wire_request.body),
                        write_intent=True,
                    )
                except MemorySystemCallCancelledBeforeDispatch:
                    if not responses:
                        self._attempted_dispatches.remove(attempt_key)
                        raise
                    last_response = responses[-1]
                    partial_failure = MemorySystemCallFailure(
                        "Mem0 pair add was cancelled after an earlier pair completed",
                        failure_kind="partial_write_cancelled",
                        raw_reference=last_response.raw_reference,
                        raw_response_bytes=last_response.raw_bytes,
                        supporting_raw_references=tuple(
                            item.raw_reference for item in responses[:-1]
                        ),
                        status_code=last_response.status_code,
                    )
                    break
                except MemorySystemCallFailure as exc:
                    if responses:
                        raise link_preceding_raw_references(
                            exc,
                            preceding_raw_references=tuple(
                                item.raw_reference for item in responses
                            ),
                        ) from exc
                    if (
                        exc.status_code is not None
                        and exc.raw_response_bytes is not None
                        and exc.raw_reference is not None
                        and self._internal_retry_count is not None
                    ):
                        reason = classify_settled_ingestion_failure(
                            settlement_basis=MEM0_SETTLEMENT_BASIS,
                            status_code=exc.status_code,
                            raw_response_bytes=exc.raw_response_bytes,
                            internal_retry_count=self._internal_retry_count,
                        )
                        if reason is not None:
                            self._next_batch_attempt_ordinals[dispatch_key] = (
                                request.batch_attempt_ordinal + 1
                            )
                            if request.batch_attempt_ordinal == 3:
                                self._next_ingestion_dispatch_ordinal[request.scope.scope_id] = (
                                    expected_ordinal + 1
                                )
                            raise SettledTransientIngestionFailure(
                                "Mem0 synchronous add settled with a supplier failure",
                                failure_kind=reason,
                                raw_reference=exc.raw_reference,
                                raw_response_bytes=exc.raw_response_bytes,
                                supporting_raw_references=exc.supporting_raw_references,
                                status_code=exc.status_code,
                                settlement_basis=MEM0_SETTLEMENT_BASIS,
                                internal_retry_count=self._internal_retry_count,
                            ) from exc
                    raise
                with bind_sealed_response_validation(
                    response,
                    message="Mem0 add response failed exact-profile validation",
                    supporting_raw_references=tuple(item.raw_reference for item in responses),
                ):
                    parse_add_response(response.raw_bytes)
                responses.append(response)
            if partial_failure is not None:
                raise partial_failure
            self._next_ingestion_dispatch_ordinal[request.scope.scope_id] = expected_ordinal + 1
            self._next_batch_attempt_ordinals[dispatch_key] = 0
            self._completed_add_response_refs[dispatch_key] = tuple(
                item.raw_reference.sha256 for item in responses
            )
            source_id = request.dispatch.ordered_source_units[0].source_unit_id
            if len(responses) == 1:
                receipt_bytes = responses[0].raw_bytes
                receipt_reference = responses[0].raw_reference
            else:
                receipt_bytes = canonical_json_bytes(
                    {
                        "schema_name": "oamb_mem0_pair_add_receipt",
                        "schema_version": 1,
                        "source_unit_id": source_id,
                        "ordered_response_sha256": tuple(
                            item.raw_reference.sha256 for item in responses
                        ),
                    }
                )
                receipt_reference = self._seal_raw(receipt_bytes)
            return IngestionDispatchReceipt(
                attempt_id=request.attempt_id,
                dispatch=request.dispatch,
                accepted_source_unit_ids=(source_id,),
                rejected_source_unit_ids=(),
                raw_reference=receipt_reference,
                raw_response_bytes=receipt_bytes,
                usage_records=(),
            )
        finally:
            ingestion_plan_lock.release()

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        self._require_open()
        binding = self._require_scope(request.scope)
        planned = self._planned_adds.get(request.scope.scope_id)
        if planned is None:
            raise ValueError("Mem0 readiness has no frozen ingestion plan")
        receipt = request.ingestion_receipt
        if (
            request.expected_source_unit_ids != receipt.accepted_source_unit_ids
            or receipt.ingestion_occurrence_id != binding.ingestion_occurrence_id
            or tuple(item.dispatch for item in receipt.dispatch_receipts)
            != tuple(item.dispatch for item in planned)
            or receipt.raw_references
            != tuple(item.raw_reference for item in receipt.dispatch_receipts)
        ):
            raise ValueError("Mem0 readiness receipt does not bind the completed source ledger")
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
            raise ValueError("Mem0 readiness aggregate source partition is invalid")
        for dispatch_receipt in receipt.dispatch_receipts:
            if (
                hashlib.sha256(dispatch_receipt.raw_response_bytes).hexdigest()
                != dispatch_receipt.raw_reference.sha256
            ):
                raise ValueError("Mem0 readiness raw add receipt hash does not match")
            source_id = dispatch_receipt.dispatch.ordered_source_units[0].source_unit_id
            partitions = (
                set(dispatch_receipt.accepted_source_unit_ids),
                set(dispatch_receipt.rejected_source_unit_ids),
                set(dispatch_receipt.skipped_source_unit_ids),
            )
            if any(
                left & right
                for index, left in enumerate(partitions)
                for right in partitions[index + 1 :]
            ) or set().union(*partitions) != {source_id}:
                raise ValueError("Mem0 readiness dispatch source partition is invalid")
            if not dispatch_receipt.skipped_source_unit_ids:
                planned_add = self._planned_add(
                    request.scope.scope_id,
                    dispatch_receipt.dispatch,
                )
                if len(planned_add.requests) == 1:
                    parse_add_response(dispatch_receipt.raw_response_bytes)
                else:
                    document = parse_exact_json_object(
                        dispatch_receipt.raw_response_bytes,
                        expected_fields=frozenset(
                            {
                                "schema_name",
                                "schema_version",
                                "source_unit_id",
                                "ordered_response_sha256",
                            }
                        ),
                    )
                    expected_response_refs = self._completed_add_response_refs.get(
                        (
                            request.scope.scope_id,
                            dispatch_receipt.dispatch.request_fingerprint,
                        )
                    )
                    if (
                        document["schema_name"] != "oamb_mem0_pair_add_receipt"
                        or document["schema_version"] != 1
                        or document["source_unit_id"] != source_id
                        or not isinstance(document["ordered_response_sha256"], list)
                        or tuple(document["ordered_response_sha256"]) != expected_response_refs
                    ):
                        raise ValueError("Mem0 pair-add receipt does not match its frozen plan")

        first = await self._capture_native_projection(request.scope.scope_id)
        second = await self._capture_native_projection(request.scope.scope_id)
        if (
            first.receipt.inventory.ordered_source_unit_ids
            != second.receipt.inventory.ordered_source_unit_ids
            or first.receipt.state_digest.state_sha256 != second.receipt.state_digest.state_sha256
        ):
            raise ValueError("Mem0 retrieval-visible projection did not reach a stable boundary")
        self._validate_projection_sources(
            binding=binding,
            projection=second.native,
            completed_sources=tuple(
                item.dispatch.ordered_source_units[0]
                for item in planned
                if item.dispatch.ordered_source_units[0].source_unit_id
                not in receipt.rejected_source_unit_ids
            ),
        )
        self._ready_projections[request.scope.scope_id] = second
        return ReadinessReceipt(
            ingestion_occurrence_id=binding.ingestion_occurrence_id,
            ready=True,
            evidence_references=(
                *receipt.raw_references,
                first.receipt.inventory.raw_reference,
                *first.receipt.supporting_raw_references,
                second.receipt.inventory.raw_reference,
                *second.receipt.supporting_raw_references,
            ),
        )

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt:
        self._require_open()
        return (await self.project(scope)).inventory

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt:
        self._require_open()
        return (await self.project(scope)).state_digest

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        self._require_open()
        self._require_scope(scope)
        ready = self._ready_projections.get(scope.scope_id)
        if ready is None:
            raise ValueError("Mem0 scope has not passed readiness")
        captured = await self._capture_native_projection(scope.scope_id)
        if (
            captured.receipt.inventory.ordered_source_unit_ids
            != ready.receipt.inventory.ordered_source_unit_ids
            or captured.receipt.state_digest.state_sha256 != ready.receipt.state_digest.state_sha256
        ):
            raise ValueError("Mem0 projection differs from the readiness-sealed state")
        return captured.receipt

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        self._require_open()
        binding = self._require_scope(request.scope)
        ready = self._ready_projections.get(request.scope.scope_id)
        if ready is None:
            raise ValueError("Mem0 scope has not passed readiness")
        try:
            query = request.query_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("Mem0 search query must be UTF-8") from exc
        if not query:
            raise ValueError("Mem0 search query must not be empty")
        wire_request = build_search_http_request(
            query=query,
            run_id=request.scope.scope_id,
            top_k=request.top_k,
        )
        response = await self._public_client.request(
            wire_request.method,
            wire_request.path,
            json_payload=json.loads(wire_request.body),
            request_evidence=True,
        )
        with bind_sealed_response_validation(
            response,
            message="Mem0 search response failed exact-profile validation",
        ):
            items = parse_search_response(
                response.raw_bytes,
                expected_run_id=request.scope.scope_id,
            )
            ready_points = {point.native_id: point for point in ready.native.points}
            candidates: list[NativeEvidenceCandidate] = []
            for item in items:
                point = ready_points.get(item.native_id)
                if (
                    point is None
                    or item.memory != point.memory
                    or item.run_id != point.run_id
                    or item.created_at != point.created_at
                    or item.updated_at != point.updated_at
                ):
                    raise ValueError("Mem0 search candidate does not match the sealed projection")
                if item.memory_hash is not None and item.memory_hash != point.memory_hash:
                    raise ValueError("Mem0 search candidate hash does not match sealed projection")
                if item.metadata != point.metadata:
                    raise ValueError(
                        "Mem0 search candidate source does not match sealed projection"
                    )
                if item.metadata is not None and (
                    item.metadata.ingestion_occurrence_id != binding.ingestion_occurrence_id
                    or item.metadata.ingestion_plan_id != binding.ingestion_plan_id
                ):
                    raise ValueError("Mem0 search candidate belongs to a foreign scope")
                if item.attributed_to != point.attributed_to:
                    raise ValueError(
                        "Mem0 search candidate attribution does not match sealed projection"
                    )
                candidates.append(
                    NativeEvidenceCandidate(
                        native_id=item.native_id,
                        native_rank_1_indexed=item.native_rank_1_indexed,
                        content=item.memory,
                        native_score=_canonical_score(item.native_score),
                        provider_evidence_identity=item.native_id,
                        source_unit_id=(
                            None if item.metadata is None else item.metadata.source_unit_id
                        ),
                        evidence_kind="native_memory",
                        native_reference=item.native_id,
                        native_truncated=False,
                    )
                )
        return NativeEvidenceBatch(
            raw_reference=response.raw_reference,
            candidates=tuple(candidates),
            request_raw_reference=response.request_reference,
        )

    async def close(self) -> None:
        self._accepting_operations = False
        self._public_client.stop_accepting()
        self._inspector_client.stop_accepting()
        async with self._close_lock:
            errors: list[BaseException] = []
            if not self._public_closed:
                try:
                    await self._public_client.close()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    self._public_closed = True
            if not self._inspector_closed:
                try:
                    await self._inspector_client.close()
                except BaseException as exc:
                    errors.append(exc)
                else:
                    self._inspector_closed = True
            if errors:
                if len(errors) == 1:
                    raise errors[0]
                raise BaseExceptionGroup("Mem0 REST clients failed to close", errors)

    def _require_open(self) -> None:
        if not self._accepting_operations:
            raise RuntimeError("Mem0 REST adapter is closed")

    def _require_scope(self, scope: ScopeReceipt) -> _ScopeBinding:
        binding = self._scopes.get(scope.scope_id)
        if (
            binding is None
            or scope.scope_id != scope.ingestion_occurrence_id
            or binding.ingestion_occurrence_id != scope.ingestion_occurrence_id
        ):
            raise ValueError("Mem0 scope receipt is unknown or inconsistent")
        return binding

    def _planned_add(self, scope_id: str, dispatch: IngestionDispatch) -> _PlannedAdd:
        planned = self._planned_adds.get(scope_id)
        ordinal_index = dispatch.dispatch_ordinal_1_indexed - 1
        if (
            planned is None
            or ordinal_index < 0
            or ordinal_index >= len(planned)
            or dispatch != planned[ordinal_index].dispatch
            or dispatch.operation_kind != MEM0_ADD_OPERATION
            or len(dispatch.ordered_source_units) != 1
        ):
            raise ValueError("Mem0 add dispatch does not match the frozen plan")
        return planned[ordinal_index]

    async def _capture_native_projection(self, run_id: str) -> _CapturedProjection:
        raw_pages: list[bytes] = []
        responses: list[SealedRestResponse] = []
        cursor: str | None = None
        for _ in range(MEM0_PROJECTION_MAX_PAGES):
            params: dict[str, str] = {"run_id": run_id}
            if cursor is not None:
                params["cursor"] = cursor
            preceding_raw_references = tuple(previous.raw_reference for previous in responses)
            try:
                response = await self._inspector_client.request(
                    "GET",
                    "/v1/projection",
                    params=params,
                )
            except MemorySystemReadCancelled as exc:
                if preceding_raw_references:
                    raise link_preceding_read_cancellation(
                        exc,
                        preceding_raw_references=preceding_raw_references,
                    ) from exc
                raise
            except MemorySystemCallFailure as exc:
                if preceding_raw_references:
                    raise link_preceding_raw_references(
                        exc,
                        preceding_raw_references=preceding_raw_references,
                    ) from exc
                raise
            responses.append(response)
            with bind_sealed_response_validation(
                response,
                message="Mem0 inspector page failed exact-profile validation",
                supporting_raw_references=tuple(
                    previous.raw_reference for previous in responses[:-1]
                ),
            ):
                page = parse_exact_json_object(
                    response.raw_bytes,
                    expected_fields=_PROJECTION_PAGE_FIELDS,
                )
                next_cursor = page["next_cursor"]
                if next_cursor is not None and not isinstance(next_cursor, str):
                    raise ValueError("Mem0 inspector cursor must be a string or null")
            raw_pages.append(response.raw_bytes)
            cursor = next_cursor
            if cursor is None:
                break
        else:
            terminal = responses[-1]
            raise sealed_response_validation_failure(
                terminal,
                message="Mem0 inspector projection page limit exceeded",
                supporting_raw_references=tuple(
                    previous.raw_reference for previous in responses[:-1]
                ),
            )
        terminal = responses[-1]
        with bind_sealed_response_validation(
            terminal,
            message="Mem0 projection failed exact-profile validation",
            supporting_raw_references=tuple(previous.raw_reference for previous in responses[:-1]),
        ):
            native = parse_projection_pages(
                tuple(raw_pages),
                expected_collection=MEM0_COLLECTION,
                expected_run_id=run_id,
            )
            source_ids = projection_source_unit_ids(native)
            state_sha256 = projection_state_sha256(native)
        raw_references = tuple(response.raw_reference for response in responses)
        capture_sequence = self._projection_capture_sequences.get(run_id, 0) + 1
        self._projection_capture_sequences[run_id] = capture_sequence
        summary_reference = self._seal_raw(
            canonical_json_bytes(
                {
                    "schema": "oamb-mem0-projection-receipt-v1",
                    "collection": native.collection,
                    "run_id": native.run_id,
                    "ordered_source_unit_ids": source_ids,
                    "state_sha256": state_sha256,
                    "capture_sequence": capture_sequence,
                    "page_raw_refs": tuple(item.sha256 for item in raw_references),
                }
            )
        )
        receipt = ProjectionReceipt(
            inventory=InventoryReceipt(
                ingestion_occurrence_id=run_id,
                ordered_source_unit_ids=source_ids,
                raw_reference=summary_reference,
            ),
            state_digest=StateDigestReceipt(
                ingestion_occurrence_id=run_id,
                state_sha256=state_sha256,
                raw_reference=summary_reference,
            ),
            supporting_raw_references=raw_references,
        )
        return _CapturedProjection(native=native, receipt=receipt)

    @staticmethod
    def _validate_projection_sources(
        *,
        binding: _ScopeBinding,
        projection: Mem0Projection,
        completed_sources: tuple[SourceUnit, ...],
    ) -> None:
        completed_by_id = {source.source_unit_id: source for source in completed_sources}
        for point in projection.points:
            metadata = point.metadata
            if metadata is None:
                continue
            if (
                metadata.ingestion_occurrence_id != binding.ingestion_occurrence_id
                or metadata.ingestion_plan_id != binding.ingestion_plan_id
            ):
                raise ValueError("Mem0 projection contains foreign scope metadata")
            source = completed_by_id.get(metadata.source_unit_id)
            if source is None:
                raise ValueError("Mem0 projection contains an unplanned source identity")
            if metadata.source_ordinal != source.ordinal_1_indexed:
                raise ValueError("Mem0 projection source ordinal differs from the frozen plan")
            if source.occurred_at is not None and point.created_at != source.occurred_at:
                raise ValueError(
                    "Mem0 projection created_at differs from the source observation time"
                )

    def _seal_raw(self, raw_bytes: bytes) -> RawReferenceHandle:
        return self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=hashlib.sha256(raw_bytes).hexdigest(),
                media_type="application/json",
                compression="gzip",
                payload_bytes=raw_bytes,
            )
        )


def _source_messages(source: SourceUnit) -> tuple[tuple[str, str], ...]:
    if hashlib.sha256(source.payload_bytes).hexdigest() != source.payload_sha256:
        raise ValueError("Mem0 source payload hash does not match its bytes")
    lme_fields = (source.source_reference, source.occurred_at, source.context_text)
    if any(value is not None for value in lme_fields):
        if not all(value is not None for value in lme_fields):
            raise ValueError("Mem0 LongMemEval source metadata must be all present or absent")
        document = _parse_json(source.payload_bytes)
        if not isinstance(document, list) or not document:
            raise ValueError("Mem0 LongMemEval source payload must be a non-empty message list")
        messages: list[tuple[str, str]] = []
        for item in document:
            if not isinstance(item, dict) or frozenset(item) != frozenset({"role", "content"}):
                raise ValueError("Mem0 LongMemEval message must contain role and content")
            role = item["role"]
            content = item["content"]
            if not isinstance(role, str) or not role or not isinstance(content, str):
                raise ValueError("Mem0 LongMemEval requires a non-empty role and string content")
            messages.append((role, content))
        return tuple(messages)
    try:
        content = source.payload_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Mem0 source payload must be UTF-8") from exc
    if not content:
        raise ValueError("Mem0 source payload must not be empty")
    return (("user", content),)


def _source_message_pairs(source: SourceUnit) -> tuple[tuple[tuple[str, str], ...], ...]:
    messages = _source_messages(source)
    return tuple(messages[index : index + 2] for index in range(0, len(messages), 2))


def _validate_openapi(raw_bytes: bytes) -> None:
    document = _parse_json(raw_bytes)
    if not isinstance(document, dict):
        raise ValueError("Mem0 OpenAPI root must be an object")
    info = document.get("info")
    paths = document.get("paths")
    if (
        document.get("openapi") != "3.1.0"
        or not isinstance(info, dict)
        or info.get("title") != MEM0_OPENAPI_TITLE
        or info.get("version") != MEM0_OPENAPI_VERSION
        or not isinstance(paths, dict)
    ):
        raise ValueError("Mem0 OpenAPI identity does not match the exact server profile")
    for path in ("/memories", "/search"):
        operation = paths.get(path)
        if not isinstance(operation, dict) or "post" not in operation:
            raise ValueError(f"Mem0 OpenAPI is missing POST {path}")


def _parse_json(raw_bytes: bytes) -> object:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value: str) -> Never:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        return json.loads(
            raw_bytes,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("response is not valid JSON") from exc


def _canonical_score(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Mem0 native score must be finite")
    return str(value)


def _require_run_id(value: str) -> None:
    if _RUN_ID_PATTERN.fullmatch(value) is None:
        raise ValueError("Mem0 run_id must be a full lowercase SHA-256 identifier")


__all__ = [
    "Mem0ReferenceNegativeAdapter",
    "Mem0RestAdapter",
    "Mem0RestDispatcher",
]
