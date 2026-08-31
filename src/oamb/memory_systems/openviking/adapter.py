"""OpenViking v0.4.16 native resource/find profile."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import httpx

from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
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
    StateDigestReceipt,
)
from oamb.memory_systems.rest import (
    SealedRestClient,
    SealedRestResponse,
    bind_sealed_response_validation,
    link_preceding_before_dispatch_cancellation,
    link_preceding_raw_references,
    link_preceding_read_cancellation,
    parse_exact_json_object,
    sealed_response_validation_failure,
)

OPENVIKING_VERSION = "v0.4.16"
OPENVIKING_MEMORY_SYSTEM_ID = "openviking"
OPENVIKING_AUTH_MODE = "api_key"
OPENVIKING_USER_ROLE = "admin"
ACTOR_PEER_HEADER = "X-OpenViking-Actor-Peer"
OPENVIKING_BATCH_OPERATION = "openviking_batch_write"
MAX_BATCH_OPERATIONS = 256
MAX_BATCH_FILE_BYTES = 8 * 1024 * 1024
MAX_BATCH_TOTAL_BYTES = 16 * 1024 * 1024
NATIVE_RETRIEVAL_TOP_K = 100


class OpenVikingProfileError(ValueError):
    """The exact OpenViking runtime or wire profile did not match v0.4.16."""


@dataclass(frozen=True, slots=True)
class _ScopeBinding:
    ingestion_occurrence_id: str
    ingestion_plan_id: str
    actor_peer_id: str
    peer_root: str
    root_uri: str


@dataclass(frozen=True, slots=True)
class _PlannedBatch:
    dispatch: IngestionDispatch
    chunk_uris: tuple[str, ...]
    decoded_contents: tuple[str, ...]


class OpenVikingRestAdapter:
    """Original REST adapter for the exact OpenViking v0.4.16 profile."""

    def __init__(
        self,
        *,
        store: ArtifactStorePort,
        base_url: str,
        api_key: str,
        benchmark_account: str,
        benchmark_user: str,
        runtime_binding_hash: str,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        for label, value in (
            ("api_key", api_key),
            ("benchmark_account", benchmark_account),
            ("benchmark_user", benchmark_user),
            ("runtime_binding_hash", runtime_binding_hash),
        ):
            if not value:
                raise ValueError(f"OpenViking {label} is required")
        if re.fullmatch(r"[0-9a-f]{64}", runtime_binding_hash) is None:
            raise ValueError("OpenViking runtime binding hash must be lowercase SHA-256")
        self._store = store
        self._benchmark_account = benchmark_account
        self._benchmark_user = benchmark_user
        self._runtime_binding_hash = runtime_binding_hash
        self._client = SealedRestClient(
            store=store,
            base_url=base_url,
            headers={"X-API-Key": api_key},
            transport=transport,
        )
        self._scopes: dict[str, _ScopeBinding] = {}
        self._attempted_allocations: set[str] = set()
        self._allocation_lock = asyncio.Lock()
        self._planned_batches: dict[str, _PlannedBatch] = {}
        self._attempted_scopes: set[str] = set()
        self._attempted_wrong_peer_write_probes: set[tuple[str, str, str]] = set()
        self._wrong_peer_probe_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._accepting_wrong_peer_probes = True
        self._active_wrong_peer_probe_count = 0
        self._wrong_peer_probes_drained = asyncio.Event()
        self._wrong_peer_probes_drained.set()
        self._ready_scopes: set[str] = set()
        self._resolved = False

    async def resolve(self) -> RuntimeResolution:
        response = await self._client.request("GET", "/health")
        with bind_sealed_response_validation(
            response,
            message="OpenViking health response failed exact-profile validation",
        ):
            document = _parse_exact_object(
                response.raw_bytes,
                frozenset(
                    {
                        "status",
                        "healthy",
                        "version",
                        "auth_mode",
                        "account_id",
                        "user_id",
                        "role",
                    }
                ),
                "health",
            )
            expected = {
                "status": "ok",
                "healthy": True,
                "version": OPENVIKING_VERSION,
                "auth_mode": OPENVIKING_AUTH_MODE,
                "account_id": self._benchmark_account,
                "user_id": self._benchmark_user,
                "role": OPENVIKING_USER_ROLE,
            }
            if document != expected:
                raise OpenVikingProfileError(
                    "OpenViking health identity does not match the profile"
                )
        self._resolved = True
        return RuntimeResolution(
            memory_system_id=OPENVIKING_MEMORY_SYSTEM_ID,
            runtime_binding_hash=self._runtime_binding_hash,
            raw_reference=response.raw_reference,
        )

    async def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            capability_ids=(
                "create-only-peer-root",
                "complete-hidden-projection",
                "native-find-level-hydration",
            ),
            provider_order_preserved=True,
            native_reranking_disabled=True,
        )

    async def allocate_ingestion_scope(
        self,
        request: ScopeAllocationRequest,
    ) -> ScopeReceipt:
        if not self._resolved:
            raise RuntimeError("OpenViking resolve must pass before scope allocation")
        async with self._allocation_lock:
            return await self._allocate_ingestion_scope_once(request)

    async def _allocate_ingestion_scope_once(
        self,
        request: ScopeAllocationRequest,
    ) -> ScopeReceipt:
        binding = self._scope_binding(request)
        if binding.root_uri in self._attempted_allocations:
            raise OpenVikingProfileError(
                "OpenViking scope allocation was already attempted and cannot replay"
            )
        headers = {ACTOR_PEER_HEADER: binding.actor_peer_id}
        peer_absence = await self._require_absent(binding.peer_root, headers=headers)
        try:
            root_absence = await self._require_absent(binding.root_uri, headers=headers)
        except MemorySystemReadCancelled as exc:
            raise link_preceding_read_cancellation(
                exc,
                preceding_raw_references=(peer_absence,),
            ) from exc
        except MemorySystemCallFailure as exc:
            raise link_preceding_raw_references(
                exc,
                preceding_raw_references=(peer_absence,),
            ) from exc
        self._attempted_allocations.add(binding.root_uri)
        try:
            response = await self._client.request(
                "POST",
                "/api/v1/fs/mkdir",
                json_payload={"uri": binding.root_uri},
                request_headers=headers,
                write_intent=True,
            )
        except MemorySystemCallCancelledBeforeDispatch as exc:
            self._attempted_allocations.remove(binding.root_uri)
            raise link_preceding_before_dispatch_cancellation(
                exc,
                preceding_raw_references=(peer_absence, root_absence),
            ) from exc
        except MemorySystemCallFailure as exc:
            raise link_preceding_raw_references(
                exc,
                preceding_raw_references=(peer_absence, root_absence),
            ) from exc
        with bind_sealed_response_validation(
            response,
            message="OpenViking mkdir response failed exact-profile validation",
            supporting_raw_references=(peer_absence, root_absence),
        ):
            document = _parse_standard_success(response.raw_bytes, "mkdir")
            result = _require_exact_dict(document["result"], frozenset({"uri"}), "mkdir.result")
            if result["uri"] != binding.root_uri:
                raise OpenVikingProfileError("OpenViking mkdir returned the wrong root URI")
        if binding.root_uri in self._scopes:
            raise OpenVikingProfileError("OpenViking ingestion scope is already allocated")
        self._scopes[binding.root_uri] = binding
        return ScopeReceipt(
            ingestion_occurrence_id=request.ingestion_occurrence_id,
            scope_id=binding.root_uri,
            raw_reference=response.raw_reference,
            supporting_raw_references=(peer_absence, root_absence),
        )

    def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]:
        binding = self._require_scope(request.scope)
        if binding.root_uri in self._planned_batches:
            raise OpenVikingProfileError("OpenViking ingestion is already planned for this scope")
        sources = request.ordered_source_units
        if not sources or len(sources) > MAX_BATCH_OPERATIONS:
            raise OpenVikingProfileError("OpenViking batch operation count exceeds the profile")
        expected_ordinals = tuple(range(1, len(sources) + 1))
        if tuple(source.ordinal_1_indexed for source in sources) != expected_ordinals:
            raise OpenVikingProfileError(
                "OpenViking source ordinals must be contiguous and ordered"
            )
        source_ids = tuple(source.source_unit_id for source in sources)
        if len(set(source_ids)) != len(source_ids):
            raise OpenVikingProfileError("OpenViking source-unit IDs must be unique")

        decoded_contents: list[str] = []
        total_bytes = 0
        for source in sources:
            if hashlib.sha256(source.payload_bytes).hexdigest() != source.payload_sha256:
                raise OpenVikingProfileError("OpenViking source payload hash does not match bytes")
            if len(source.payload_bytes) > MAX_BATCH_FILE_BYTES:
                raise OpenVikingProfileError("OpenViking batch file exceeds the profile limit")
            total_bytes += len(source.payload_bytes)
            if total_bytes > MAX_BATCH_TOTAL_BYTES:
                raise OpenVikingProfileError("OpenViking batch content exceeds the profile limit")
            try:
                decoded_contents.append(source.payload_bytes.decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise OpenVikingProfileError(
                    "OpenViking source payload must be exact UTF-8 text"
                ) from exc

        chunk_uris = tuple(
            f"{binding.root_uri}/chunk-{ordinal:04d}.txt" for ordinal in expected_ordinals
        )
        dispatch = IngestionDispatch(
            dispatch_ordinal_1_indexed=1,
            operation_kind=OPENVIKING_BATCH_OPERATION,
            request_fingerprint=canonical_sha256(
                [
                    "oamb-openviking-batch-write-v1",
                    binding.root_uri,
                    tuple(
                        (source.source_unit_id, source.payload_sha256, chunk_uri)
                        for source, chunk_uri in zip(sources, chunk_uris, strict=True)
                    ),
                ]
            ),
            ordered_source_units=sources,
        )
        self._planned_batches[binding.root_uri] = _PlannedBatch(
            dispatch=dispatch,
            chunk_uris=chunk_uris,
            decoded_contents=tuple(decoded_contents),
        )
        return (dispatch,)

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt:
        binding = self._require_scope(request.scope)
        planned = self._planned_batches.get(binding.root_uri)
        if planned is None or request.dispatch != planned.dispatch:
            raise OpenVikingProfileError("OpenViking dispatch does not match the frozen plan")
        if binding.root_uri in self._attempted_scopes:
            raise OpenVikingProfileError("OpenViking create-only batch cannot be replayed")
        self._attempted_scopes.add(binding.root_uri)
        try:
            response = await self._client.request(
                "POST",
                "/api/v1/content/batch-write",
                json_payload={
                    "root_uri": binding.root_uri,
                    "operations": [
                        {"uri": uri, "content": content, "mode": "create"}
                        for uri, content in zip(
                            planned.chunk_uris,
                            planned.decoded_contents,
                            strict=True,
                        )
                    ],
                    "wait": True,
                },
                request_headers={ACTOR_PEER_HEADER: binding.actor_peer_id},
                write_intent=True,
            )
        except MemorySystemCallCancelledBeforeDispatch:
            self._attempted_scopes.remove(binding.root_uri)
            raise
        with bind_sealed_response_validation(
            response,
            message="OpenViking batch-write response failed exact-profile validation",
        ):
            created, _queue_ready = _parse_batch_response(
                response.raw_bytes,
                expected_root=binding.root_uri,
                expected_chunk_uris=planned.chunk_uris,
            )
        source_by_uri = dict(
            zip(planned.chunk_uris, planned.dispatch.ordered_source_units, strict=True)
        )
        accepted = tuple(source_by_uri[uri].source_unit_id for uri in created)
        accepted_set = frozenset(accepted)
        rejected = tuple(
            source.source_unit_id
            for source in planned.dispatch.ordered_source_units
            if source.source_unit_id not in accepted_set
        )
        return IngestionDispatchReceipt(
            attempt_id=request.attempt_id,
            dispatch=request.dispatch,
            accepted_source_unit_ids=accepted,
            rejected_source_unit_ids=rejected,
            raw_reference=response.raw_reference,
            raw_response_bytes=response.raw_bytes,
            usage_records=(),
        )

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        binding = self._require_scope(request.scope)
        planned = self._planned_batches.get(binding.root_uri)
        if planned is None:
            raise OpenVikingProfileError("OpenViking readiness has no frozen batch plan")
        expected_source_ids = tuple(
            source.source_unit_id for source in planned.dispatch.ordered_source_units
        )
        receipt = request.ingestion_receipt
        if (
            request.expected_source_unit_ids != expected_source_ids
            or receipt.ingestion_occurrence_id != request.scope.ingestion_occurrence_id
            or len(receipt.dispatch_receipts) != 1
            or len(receipt.raw_references) != 1
        ):
            raise OpenVikingProfileError("OpenViking readiness receipt binding is invalid")
        dispatch_receipt = receipt.dispatch_receipts[0]
        if (
            dispatch_receipt.dispatch != planned.dispatch
            or dispatch_receipt.raw_reference != receipt.raw_references[0]
            or receipt.accepted_source_unit_ids != dispatch_receipt.accepted_source_unit_ids
            or receipt.rejected_source_unit_ids != dispatch_receipt.rejected_source_unit_ids
        ):
            raise OpenVikingProfileError("OpenViking readiness dispatch receipt is invalid")
        if (
            hashlib.sha256(dispatch_receipt.raw_response_bytes).hexdigest()
            != dispatch_receipt.raw_reference.sha256
        ):
            raise OpenVikingProfileError(
                "OpenViking immutable readiness raw receipt hash does not match"
            )
        created, queue_ready = _parse_batch_response(
            dispatch_receipt.raw_response_bytes,
            expected_root=binding.root_uri,
            expected_chunk_uris=planned.chunk_uris,
        )
        ready = (
            created == planned.chunk_uris
            and queue_ready
            and dispatch_receipt.accepted_source_unit_ids == expected_source_ids
            and not dispatch_receipt.rejected_source_unit_ids
        )
        if not ready:
            return ReadinessReceipt(
                ingestion_occurrence_id=request.scope.ingestion_occurrence_id,
                ready=False,
                evidence_references=(dispatch_receipt.raw_reference,),
            )
        projection = await self._capture_projection(request.scope)
        if projection.inventory.ordered_source_unit_ids != expected_source_ids:
            raise OpenVikingProfileError(
                "OpenViking readiness projection does not match expected sources"
            )
        self._ready_scopes.add(binding.root_uri)
        return ReadinessReceipt(
            ingestion_occurrence_id=request.scope.ingestion_occurrence_id,
            ready=True,
            evidence_references=(
                dispatch_receipt.raw_reference,
                projection.inventory.raw_reference,
                projection.state_digest.raw_reference,
            ),
        )

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt:
        return (await self._capture_projection(scope)).inventory

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt:
        return (await self._capture_projection(scope)).state_digest

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        return await self._capture_projection(scope)

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        binding = self._require_scope(request.scope)
        planned = self._planned_batches.get(binding.root_uri)
        if planned is None:
            raise OpenVikingProfileError("OpenViking retrieval has no frozen batch plan")
        if binding.root_uri not in self._ready_scopes:
            raise OpenVikingProfileError("OpenViking scope has not passed readiness")
        if request.top_k != NATIVE_RETRIEVAL_TOP_K:
            raise OpenVikingProfileError("OpenViking retrieval top_k must equal 100")
        try:
            query = request.query_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OpenVikingProfileError("OpenViking find query must be UTF-8") from exc
        headers = {ACTOR_PEER_HEADER: binding.actor_peer_id}
        find_response = await self._client.request(
            "POST",
            "/api/v1/search/find",
            json_payload={
                "query": query,
                "target_uri": binding.root_uri,
                "context_type": "resource",
                "limit": NATIVE_RETRIEVAL_TOP_K,
            },
            request_headers=headers,
        )
        try:
            hits = _parse_find_response(
                find_response.raw_bytes,
                root_uri=binding.root_uri,
                chunk_uris=planned.chunk_uris,
            )
        except ValueError as exc:
            raise sealed_response_validation_failure(
                find_response,
                message="OpenViking find response failed exact-profile validation",
            ) from exc
        source_by_uri = dict(
            zip(planned.chunk_uris, planned.dispatch.ordered_source_units, strict=True)
        )
        candidates: list[NativeEvidenceCandidate] = []
        supporting_references: list[RawReferenceHandle] = []
        for rank, hit in enumerate(hits, start=1):
            uri = hit["uri"]
            level = hit["level"]
            source_unit_id: str | None = None
            if level == 0:
                content = hit["abstract"]
                evidence_kind = "native_abstract"
            elif level == 1:
                content = hit["abstract"]
                evidence_kind = "native_overview"
            else:
                preceding_references = (
                    find_response.raw_reference,
                    *supporting_references,
                )
                try:
                    read_response = await self._client.request(
                        "GET",
                        "/api/v1/content/read",
                        params={"uri": uri, "offset": 0, "limit": -1, "raw": True},
                        request_headers=headers,
                    )
                except MemorySystemReadCancelled as exc:
                    raise link_preceding_read_cancellation(
                        exc,
                        preceding_raw_references=preceding_references,
                    ) from exc
                except MemorySystemCallFailure as exc:
                    raise link_preceding_raw_references(
                        exc,
                        preceding_raw_references=preceding_references,
                    ) from exc
                try:
                    read_document = _parse_standard_success(
                        read_response.raw_bytes, "find L2 content.read"
                    )
                    content = read_document["result"]
                    if not isinstance(content, str):
                        raise OpenVikingProfileError("OpenViking L2 hydration is not full text")
                    source = source_by_uri[uri]
                    if content.encode("utf-8") != source.payload_bytes:
                        raise OpenVikingProfileError(
                            "OpenViking L2 hydration does not match the frozen source"
                        )
                except ValueError as exc:
                    raise sealed_response_validation_failure(
                        read_response,
                        message="OpenViking L2 response failed exact-profile validation",
                        supporting_raw_references=preceding_references,
                    ) from exc
                source_unit_id = source.source_unit_id
                evidence_kind = "source_content"
                supporting_references.append(read_response.raw_reference)
            evidence_identity = f"{uri}#level={level}"
            candidates.append(
                NativeEvidenceCandidate(
                    native_id=evidence_identity,
                    native_rank_1_indexed=rank,
                    content=content,
                    native_score=_canonical_score(hit["score"]),
                    provider_evidence_identity=evidence_identity,
                    source_unit_id=source_unit_id,
                    evidence_kind=evidence_kind,
                    native_reference=uri,
                    native_truncated=False,
                )
            )
        return NativeEvidenceBatch(
            raw_reference=find_response.raw_reference,
            candidates=tuple(candidates),
            supporting_raw_references=tuple(supporting_references),
        )

    async def verify_wrong_peer_find_isolation(
        self,
        *,
        scope: ScopeReceipt,
        wrong_actor_peer_id: str,
    ) -> RawReferenceHandle:
        binding = self._require_wrong_peer(scope, wrong_actor_peer_id)
        return await self._expect_wrong_peer_denial(
            method="POST",
            path="/api/v1/search/find",
            wrong_actor_peer_id=wrong_actor_peer_id,
            expected_uri=binding.root_uri,
            json_payload={
                "query": "oamb wrong-peer isolation probe",
                "target_uri": binding.root_uri,
                "context_type": "resource",
                "limit": NATIVE_RETRIEVAL_TOP_K,
            },
            probe_lock_key=(binding.root_uri, wrong_actor_peer_id),
        )

    async def verify_wrong_peer_stat_isolation(
        self,
        *,
        scope: ScopeReceipt,
        wrong_actor_peer_id: str,
    ) -> RawReferenceHandle:
        binding = self._require_wrong_peer(scope, wrong_actor_peer_id)
        return await self._expect_wrong_peer_denial(
            method="GET",
            path="/api/v1/fs/stat",
            wrong_actor_peer_id=wrong_actor_peer_id,
            expected_uri=binding.root_uri,
            params={"uri": binding.root_uri},
            probe_lock_key=(binding.root_uri, wrong_actor_peer_id),
        )

    async def verify_wrong_peer_create_isolation(
        self,
        *,
        scope: ScopeReceipt,
        wrong_actor_peer_id: str,
    ) -> RawReferenceHandle:
        binding = self._require_wrong_peer(scope, wrong_actor_peer_id)
        probe_uri = f"{binding.root_uri}/.oamb-wrong-peer-create-probe"
        return await self._expect_wrong_peer_denial(
            method="POST",
            path="/api/v1/fs/mkdir",
            wrong_actor_peer_id=wrong_actor_peer_id,
            expected_uri=probe_uri,
            json_payload={"uri": probe_uri},
            write_probe_key=(binding.root_uri, wrong_actor_peer_id, "create"),
            probe_lock_key=(binding.root_uri, wrong_actor_peer_id),
        )

    async def verify_wrong_peer_write_isolation(
        self,
        *,
        scope: ScopeReceipt,
        wrong_actor_peer_id: str,
    ) -> RawReferenceHandle:
        binding = self._require_wrong_peer(scope, wrong_actor_peer_id)
        return await self._expect_wrong_peer_denial(
            method="POST",
            path="/api/v1/content/batch-write",
            wrong_actor_peer_id=wrong_actor_peer_id,
            expected_uri=binding.root_uri,
            json_payload={
                "root_uri": binding.root_uri,
                "operations": [
                    {
                        "uri": f"{binding.root_uri}/.oamb-wrong-peer-write-probe.txt",
                        "content": "oamb wrong-peer isolation probe",
                        "mode": "create",
                    }
                ],
                "wait": True,
            },
            write_probe_key=(binding.root_uri, wrong_actor_peer_id, "batch-write"),
            probe_lock_key=(binding.root_uri, wrong_actor_peer_id),
        )

    def _require_wrong_peer(
        self,
        scope: ScopeReceipt,
        wrong_actor_peer_id: str,
    ) -> _ScopeBinding:
        binding = self._require_scope(scope)
        if (
            wrong_actor_peer_id == binding.actor_peer_id
            or re.fullmatch(r"oamb-[0-9a-f]{64}", wrong_actor_peer_id) is None
        ):
            raise ValueError("wrong OpenViking actor peer must be a distinct deterministic ID")
        return binding

    async def _expect_wrong_peer_denial(
        self,
        *,
        method: str,
        path: str,
        wrong_actor_peer_id: str,
        expected_uri: str,
        params: dict[str, str] | None = None,
        json_payload: object | None = None,
        write_probe_key: tuple[str, str, str] | None = None,
        probe_lock_key: tuple[str, str],
    ) -> RawReferenceHandle:
        self._admit_wrong_peer_probe()
        try:
            return await self._expect_wrong_peer_denial_admitted(
                method=method,
                path=path,
                wrong_actor_peer_id=wrong_actor_peer_id,
                expected_uri=expected_uri,
                params=params,
                json_payload=json_payload,
                write_probe_key=write_probe_key,
                probe_lock_key=probe_lock_key,
            )
        finally:
            self._release_wrong_peer_probe()

    async def _expect_wrong_peer_denial_admitted(
        self,
        *,
        method: str,
        path: str,
        wrong_actor_peer_id: str,
        expected_uri: str,
        params: dict[str, str] | None,
        json_payload: object | None,
        write_probe_key: tuple[str, str, str] | None,
        probe_lock_key: tuple[str, str],
    ) -> RawReferenceHandle:
        probe_lock = self._wrong_peer_probe_locks.setdefault(probe_lock_key, asyncio.Lock())
        try:
            await probe_lock.acquire()
        except asyncio.CancelledError as exc:
            if write_probe_key is not None:
                raise MemorySystemCallCancelledBeforeDispatch(
                    "OpenViking wrong-peer write probe was cancelled before dispatch"
                ) from exc
            raise MemorySystemReadCancelled(
                "OpenViking wrong-peer read probe was cancelled before dispatch"
            ) from exc
        try:
            if write_probe_key is not None:
                if write_probe_key in self._attempted_wrong_peer_write_probes:
                    raise OpenVikingProfileError(
                        "OpenViking wrong-peer write probe was already attempted and cannot replay"
                    )
                self._attempted_wrong_peer_write_probes.add(write_probe_key)
            try:
                response = await self._client.request(
                    method,
                    path,
                    params=params,
                    json_payload=json_payload,
                    request_headers={ACTOR_PEER_HEADER: wrong_actor_peer_id},
                    write_intent=write_probe_key is not None,
                )
            except MemorySystemCallCancelledBeforeDispatch:
                if write_probe_key is not None:
                    self._attempted_wrong_peer_write_probes.remove(write_probe_key)
                raise
            except MemorySystemCallFailure as exc:
                if (
                    exc.failure_kind != "http_status"
                    or exc.status_code != 403
                    or exc.raw_response_bytes is None
                    or exc.raw_reference is None
                ):
                    raise
                denial_response = SealedRestResponse(
                    status_code=exc.status_code,
                    raw_bytes=exc.raw_response_bytes,
                    raw_reference=exc.raw_reference,
                )
                with bind_sealed_response_validation(
                    denial_response,
                    message="OpenViking permission-denied response failed exact-profile validation",
                ):
                    _parse_permission_denied_response(
                        exc.raw_response_bytes,
                        expected_uri=expected_uri,
                    )
                return exc.raw_reference
            raise sealed_response_validation_failure(
                response,
                message=f"OpenViking wrong-peer {path} was not denied",
            )
        finally:
            probe_lock.release()

    async def close(self) -> None:
        self._accepting_wrong_peer_probes = False
        self._client.stop_accepting()
        await self._wrong_peer_probes_drained.wait()
        await self._client.close()

    def _admit_wrong_peer_probe(self) -> None:
        if not self._accepting_wrong_peer_probes:
            raise RuntimeError("OpenViking REST adapter is closed")
        self._active_wrong_peer_probe_count += 1
        if self._active_wrong_peer_probe_count == 1:
            self._wrong_peer_probes_drained.clear()

    def _release_wrong_peer_probe(self) -> None:
        self._active_wrong_peer_probe_count -= 1
        if self._active_wrong_peer_probe_count == 0:
            self._wrong_peer_probes_drained.set()

    async def _require_absent(
        self,
        uri: str,
        *,
        headers: dict[str, str],
    ) -> RawReferenceHandle:
        try:
            response = await self._client.request(
                "GET",
                "/api/v1/fs/stat",
                params={"uri": uri},
                request_headers=headers,
            )
        except MemorySystemCallFailure as exc:
            if exc.failure_kind == "http_status" and exc.status_code == 404:
                if exc.raw_response_bytes is None or exc.raw_reference is None:
                    raise OpenVikingProfileError(
                        "OpenViking absence response has no raw envelope"
                    ) from exc
                absence_response = SealedRestResponse(
                    status_code=exc.status_code,
                    raw_bytes=exc.raw_response_bytes,
                    raw_reference=exc.raw_reference,
                )
                with bind_sealed_response_validation(
                    absence_response,
                    message="OpenViking not-found response failed exact-profile validation",
                ):
                    _parse_not_found_response(exc.raw_response_bytes, expected_uri=uri)
                return exc.raw_reference
            raise
        raise sealed_response_validation_failure(
            response,
            message=f"OpenViking create-only scope already exists: {uri}",
        )

    async def _capture_projection(self, scope: ScopeReceipt) -> ProjectionReceipt:
        responses: list[SealedRestResponse] = []
        try:
            return await self._capture_projection_exact(scope, responses=responses)
        except MemorySystemReadCancelled as exc:
            if not responses:
                raise
            raise link_preceding_read_cancellation(
                exc,
                preceding_raw_references=tuple(response.raw_reference for response in responses),
            ) from exc
        except MemorySystemCallFailure as exc:
            if not responses:
                raise
            raise link_preceding_raw_references(
                exc,
                preceding_raw_references=tuple(response.raw_reference for response in responses),
            ) from exc
        except ValueError as exc:
            if not responses:
                raise
            current_response = responses[-1]
            raise sealed_response_validation_failure(
                current_response,
                message="OpenViking projection response failed exact-profile validation",
                supporting_raw_references=tuple(
                    response.raw_reference for response in responses[:-1]
                ),
            ) from exc

    async def _capture_projection_exact(
        self,
        scope: ScopeReceipt,
        *,
        responses: list[SealedRestResponse],
    ) -> ProjectionReceipt:
        binding = self._require_scope(scope)
        planned = self._planned_batches.get(binding.root_uri)
        if planned is None:
            raise OpenVikingProfileError("OpenViking projection has no frozen batch plan")
        inventory_uris = tuple(
            sorted(
                (
                    f"{binding.root_uri}/.abstract.md",
                    f"{binding.root_uri}/.overview.md",
                    *planned.chunk_uris,
                )
            )
        )
        node_limit = len(inventory_uris) + 1
        headers = {ACTOR_PEER_HEADER: binding.actor_peer_id}
        ls_response = await self._client.request(
            "GET",
            "/api/v1/fs/ls",
            params={
                "uri": binding.root_uri,
                "simple": True,
                "recursive": True,
                "output": "original",
                "show_all_hidden": True,
                "node_limit": node_limit,
            },
            request_headers=headers,
        )
        responses.append(ls_response)
        ls_document = _parse_standard_success(ls_response.raw_bytes, "ls")
        returned_uris = _require_string_list(ls_document["result"], "ls.result")
        if (
            len(returned_uris) >= node_limit
            or len(set(returned_uris)) != len(returned_uris)
            or frozenset(returned_uris) != frozenset(inventory_uris)
        ):
            raise OpenVikingProfileError(
                "OpenViking recursive hidden inventory is incomplete or unexpected"
            )

        tags_by_uri: dict[str, tuple[str, ...]] = {}
        attrs_references: dict[str, RawReferenceHandle] = {}
        for uri in (binding.root_uri, *inventory_uris):
            response = await self._client.request(
                "GET",
                "/api/v1/fs/attrs",
                params={"uri": uri},
                request_headers=headers,
            )
            responses.append(response)
            tags_by_uri[uri] = _parse_attrs_response(response.raw_bytes, expected_uri=uri)
            attrs_references[uri] = response.raw_reference

        contents_by_uri: dict[str, str] = {}
        content_references: dict[str, RawReferenceHandle] = {}
        for uri in inventory_uris:
            response = await self._client.request(
                "GET",
                "/api/v1/content/read",
                params={"uri": uri, "offset": 0, "limit": -1, "raw": True},
                request_headers=headers,
            )
            responses.append(response)
            document = _parse_standard_success(response.raw_bytes, "content.read")
            content = document["result"]
            if not isinstance(content, str):
                raise OpenVikingProfileError("OpenViking raw content read is not text")
            contents_by_uri[uri] = content
            content_references[uri] = response.raw_reference

        source_by_uri = dict(
            zip(planned.chunk_uris, planned.dispatch.ordered_source_units, strict=True)
        )
        for uri, source in source_by_uri.items():
            if contents_by_uri[uri].encode("utf-8") != source.payload_bytes:
                raise OpenVikingProfileError(
                    "OpenViking projected chunk bytes do not match the frozen source"
                )

        state_entries = (
            (binding.root_uri, None, tags_by_uri[binding.root_uri]),
            *((uri, contents_by_uri[uri], tags_by_uri[uri]) for uri in inventory_uris),
        )
        state_sha256 = canonical_sha256(
            ["oamb-openviking-projection-v1", binding.root_uri, state_entries]
        )
        evidence_bytes = canonical_json_bytes(
            {
                "schema": "oamb-openviking-projection-evidence-v1",
                "root_uri": binding.root_uri,
                "inventory_raw_reference": ls_response.raw_reference.sha256,
                "state_sha256": state_sha256,
                "entries": tuple(
                    {
                        "uri": uri,
                        "content": content,
                        "tags": tags,
                        "attrs_raw_reference": attrs_references[uri].sha256,
                        "content_raw_reference": (
                            content_references[uri].sha256 if content is not None else None
                        ),
                    }
                    for uri, content, tags in state_entries
                ),
            }
        )
        evidence_reference = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=hashlib.sha256(evidence_bytes).hexdigest(),
                media_type="application/json",
                compression="gzip",
                payload_bytes=evidence_bytes,
            )
        )
        source_ids = tuple(
            source.source_unit_id for source in planned.dispatch.ordered_source_units
        )
        return ProjectionReceipt(
            inventory=InventoryReceipt(
                ingestion_occurrence_id=scope.ingestion_occurrence_id,
                ordered_source_unit_ids=source_ids,
                raw_reference=ls_response.raw_reference,
            ),
            state_digest=StateDigestReceipt(
                ingestion_occurrence_id=scope.ingestion_occurrence_id,
                state_sha256=state_sha256,
                raw_reference=evidence_reference,
            ),
            supporting_raw_references=tuple(response.raw_reference for response in responses[1:]),
        )

    def _scope_binding(self, request: ScopeAllocationRequest) -> _ScopeBinding:
        actor_peer_id = (
            "oamb-"
            + hashlib.sha256(
                b"peer\0" + request.ingestion_occurrence_id.encode("utf-8")
            ).hexdigest()
        )
        plan_key = hashlib.sha256(b"plan\0" + request.ingestion_plan_id.encode("utf-8")).hexdigest()
        peer_root = f"viking://user/{self._benchmark_user}/peers/{actor_peer_id}"
        root_uri = f"{peer_root}/resources/oamb/mab65-v1/{plan_key}"
        return _ScopeBinding(
            ingestion_occurrence_id=request.ingestion_occurrence_id,
            ingestion_plan_id=request.ingestion_plan_id,
            actor_peer_id=actor_peer_id,
            peer_root=peer_root,
            root_uri=root_uri,
        )

    def _require_scope(self, scope: ScopeReceipt) -> _ScopeBinding:
        binding = self._scopes.get(scope.scope_id)
        if binding is None or binding.ingestion_occurrence_id != scope.ingestion_occurrence_id:
            raise OpenVikingProfileError("OpenViking scope is not allocated by this adapter")
        return binding


def _parse_exact_object(raw_bytes: bytes, fields: frozenset[str], label: str) -> dict[str, Any]:
    try:
        return parse_exact_json_object(raw_bytes, expected_fields=fields)
    except ValueError as exc:
        raise OpenVikingProfileError(f"invalid OpenViking {label} response") from exc


def _parse_standard_success(raw_bytes: bytes, label: str) -> dict[str, Any]:
    document = _parse_exact_object(
        raw_bytes,
        frozenset({"status", "result", "error", "telemetry", "profile"}),
        label,
    )
    if (
        document["status"] != "ok"
        or document["error"] is not None
        or document["telemetry"] is not None
        or document["profile"] is not None
    ):
        raise OpenVikingProfileError(f"OpenViking {label} did not return a clean success")
    return document


def _require_exact_dict(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != fields:
        raise OpenVikingProfileError(f"OpenViking {label} fields do not match the profile")
    return value


def _parse_batch_response(
    raw_bytes: bytes,
    *,
    expected_root: str,
    expected_chunk_uris: tuple[str, ...],
) -> tuple[tuple[str, ...], bool]:
    document = _parse_exact_object(raw_bytes, frozenset({"status", "result"}), "batch-write")
    if document["status"] != "ok":
        raise OpenVikingProfileError("OpenViking batch-write status is not ok")
    result = _require_exact_dict(
        document["result"],
        frozenset({"root_uri", "created", "updated", "unchanged", "queue_status"}),
        "batch-write.result",
    )
    if result["root_uri"] != expected_root:
        raise OpenVikingProfileError("OpenViking batch-write returned the wrong root")
    created = _require_string_list(result["created"], "batch-write.created")
    updated = _require_string_list(result["updated"], "batch-write.updated")
    unchanged = _require_string_list(result["unchanged"], "batch-write.unchanged")
    if updated or unchanged:
        raise OpenVikingProfileError("OpenViking create-only batch reported existing content")
    if len(set(created)) != len(created) or any(uri not in expected_chunk_uris for uri in created):
        raise OpenVikingProfileError("OpenViking batch-write created URI set is invalid")
    if created != tuple(uri for uri in expected_chunk_uris if uri in frozenset(created)):
        raise OpenVikingProfileError("OpenViking batch-write created URI order is invalid")

    queue_status = _require_exact_dict(
        result["queue_status"],
        frozenset({"Semantic", "Embedding"}),
        "batch-write.queue_status",
    )
    queue_ready = True
    for name in ("Semantic", "Embedding"):
        status = _require_exact_dict(
            queue_status[name],
            frozenset({"processed", "requeue_count", "error_count", "errors"}),
            f"batch-write.queue_status.{name}",
        )
        for field in ("processed", "requeue_count", "error_count"):
            value = status[field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise OpenVikingProfileError(f"OpenViking batch-write {name}.{field} is invalid")
        errors = status["errors"]
        if not isinstance(errors, list):
            raise OpenVikingProfileError(f"OpenViking batch-write {name}.errors is invalid")
        for error in errors:
            error_object = _require_exact_dict(
                error,
                frozenset({"message"}),
                f"batch-write.queue_status.{name}.errors[]",
            )
            if not isinstance(error_object["message"], str) or not error_object["message"]:
                raise OpenVikingProfileError(
                    f"OpenViking batch-write {name}.errors message is invalid"
                )
        if status["error_count"] != len(errors):
            raise OpenVikingProfileError(
                f"OpenViking batch-write {name} error count does not match errors"
            )
        queue_ready = queue_ready and status["error_count"] == 0 and not errors
    return created, queue_ready


def _require_string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise OpenVikingProfileError(f"OpenViking {label} is not a string list")
    return tuple(value)


def _parse_not_found_response(raw_bytes: bytes, *, expected_uri: str) -> None:
    document = _parse_exact_object(
        raw_bytes,
        frozenset({"status", "result", "error", "telemetry", "profile"}),
        "not-found",
    )
    if (
        document["status"] != "error"
        or document["result"] is not None
        or document["telemetry"] is not None
        or document["profile"] is not None
    ):
        raise OpenVikingProfileError("OpenViking not-found envelope is invalid")
    error = _require_exact_dict(
        document["error"], frozenset({"code", "message", "details"}), "not-found.error"
    )
    details = _require_exact_dict(
        error["details"], frozenset({"type", "resource"}), "not-found.error.details"
    )
    if (
        error["code"] != "NOT_FOUND"
        or not isinstance(error["message"], str)
        or not error["message"]
        or details != {"type": "file", "resource": expected_uri}
    ):
        raise OpenVikingProfileError("OpenViking absence is not an exact file-not-found")


def _parse_permission_denied_response(raw_bytes: bytes, *, expected_uri: str) -> None:
    document = _parse_exact_object(
        raw_bytes,
        frozenset({"status", "result", "error", "telemetry", "profile"}),
        "permission-denied",
    )
    if (
        document["status"] != "error"
        or document["result"] is not None
        or document["telemetry"] is not None
        or document["profile"] is not None
    ):
        raise OpenVikingProfileError("OpenViking permission-denied envelope is invalid")
    error = _require_exact_dict(
        document["error"],
        frozenset({"code", "message", "details"}),
        "permission-denied.error",
    )
    details = _require_exact_dict(
        error["details"],
        frozenset({"resource"}),
        "permission-denied.error.details",
    )
    if (
        error["code"] != "PERMISSION_DENIED"
        or not isinstance(error["message"], str)
        or not error["message"]
        or details["resource"] != expected_uri
    ):
        raise OpenVikingProfileError("OpenViking wrong-peer denial is not exact")


def _parse_attrs_response(raw_bytes: bytes, *, expected_uri: str) -> tuple[str, ...]:
    document = _parse_standard_success(raw_bytes, "attrs")
    result = _require_exact_dict(
        document["result"], frozenset({"uri", "context_type", "attrs"}), "attrs.result"
    )
    attrs = _require_exact_dict(result["attrs"], frozenset({"tags"}), "attrs.result.attrs")
    tags = _require_string_list(attrs["tags"], "attrs.result.attrs.tags")
    if (
        result["uri"] != expected_uri
        or result["context_type"] != "resource"
        or len(set(tags)) != len(tags)
    ):
        raise OpenVikingProfileError("OpenViking attrs response does not match the resource")
    return tuple(sorted(tags))


def _parse_find_response(
    raw_bytes: bytes,
    *,
    root_uri: str,
    chunk_uris: tuple[str, ...],
) -> tuple[dict[str, Any], ...]:
    document = _parse_exact_object(raw_bytes, frozenset({"status", "result"}), "find")
    if document["status"] != "ok":
        raise OpenVikingProfileError("OpenViking find status is not ok")
    result = _require_exact_dict(
        document["result"],
        frozenset({"memories", "resources", "skills", "total"}),
        "find.result",
    )
    memories = result["memories"]
    resources = result["resources"]
    skills = result["skills"]
    if memories != [] or skills != [] or not isinstance(resources, list):
        raise OpenVikingProfileError("OpenViking find returned a non-resource bucket")
    total = result["total"]
    if isinstance(total, bool) or not isinstance(total, int) or total != len(resources):
        raise OpenVikingProfileError("OpenViking find total does not match resource hits")

    expected_uri_by_level = {
        0: frozenset({f"{root_uri}/.abstract.md"}),
        1: frozenset({f"{root_uri}/.overview.md"}),
        2: frozenset(chunk_uris),
    }
    hits: list[dict[str, Any]] = []
    seen_identities: set[tuple[str, int]] = set()
    for value in resources:
        hit = _require_exact_dict(
            value,
            frozenset({"context_type", "uri", "level", "score", "abstract", "tags"}),
            "find.result.resources[]",
        )
        level = hit["level"]
        if isinstance(level, bool) or not isinstance(level, int) or level not in (0, 1, 2):
            raise OpenVikingProfileError("OpenViking find returned an unknown native level")
        uri = hit["uri"]
        if not isinstance(uri, str) or uri not in expected_uri_by_level[level]:
            raise OpenVikingProfileError(
                "OpenViking find returned a URI outside the peer root inventory"
            )
        identity = (uri, level)
        if identity in seen_identities:
            raise OpenVikingProfileError("OpenViking find returned a duplicate identity")
        seen_identities.add(identity)
        score = hit["score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
        ):
            raise OpenVikingProfileError("OpenViking find returned a non-finite score")
        abstract = hit["abstract"]
        if not isinstance(abstract, str) or (level in (0, 1) and not abstract):
            raise OpenVikingProfileError("OpenViking find preview text is missing")
        tags = _require_string_list(hit["tags"], "find.result.resources[].tags")
        if hit["context_type"] != "resource" or len(set(tags)) != len(tags):
            raise OpenVikingProfileError("OpenViking find resource metadata is invalid")
        hits.append(hit)
    return tuple(hits)


def _canonical_score(value: int | float) -> str:
    decimal = Decimal(str(value))
    if not decimal.is_finite():
        raise OpenVikingProfileError("OpenViking find score is not finite")
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text
