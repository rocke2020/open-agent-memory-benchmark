from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from decimal import Decimal
from importlib import import_module
from itertools import count
from pathlib import Path
from typing import Any

import httpx
import pytest

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import (
    ArtifactReadRequest,
    ArtifactSealReceipt,
    ArtifactWriteRequest,
    IngestionDispatchRequest,
    IngestionReceipt,
    IngestionRequest,
    MemorySystemCallCancelledBeforeDispatch,
    MemorySystemCallFailure,
    MemorySystemCallUnknownOutcome,
    MemorySystemReadCancelled,
    RawPayloadSealRequest,
    RawReferenceHandle,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
    SourceUnit,
)
from oamb.runtime.memory_query import (
    MemorySystemProjectionMutation,
    ReadOnlyRetrievalFailure,
    execute_read_only_retrieval,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "adapters" / "openviking"
INGESTION_OCCURRENCE_ID = "a" * 64
INGESTION_PLAN_ID = "mab-plan-1"
EXPECTED_PEER_ID = "oamb-91e91f25bf2d1461b3ae8fd972a3dbdd6e825a03eda5710025933c5994b7b6c5"
EXPECTED_PLAN_KEY = "f5f0891b8782a6ad22be1157d296347bad72eb802b519d324c51cef1dfcb62fb"
EXPECTED_PEER_ROOT = f"viking://user/oamb-admin/peers/{EXPECTED_PEER_ID}"
EXPECTED_ROOT = f"{EXPECTED_PEER_ROOT}/resources/oamb/mab65-v1/{EXPECTED_PLAN_KEY}"


class CapturingStore:
    def __init__(self) -> None:
        self.raw: list[RawPayloadSealRequest] = []

    def seal_raw(self, request: RawPayloadSealRequest) -> RawReferenceHandle:
        self.raw.append(request)
        return RawReferenceHandle(request.sha256)

    def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        raise AssertionError(f"unexpected source record: {request.record_id}")

    def seal_checkpoint(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        raise AssertionError(f"unexpected checkpoint: {request.record_id}")

    def seal_source_manifest(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        raise AssertionError(f"unexpected source manifest: {request.record_id}")

    def read_verified(self, request: ArtifactReadRequest) -> bytes:
        raise AssertionError(f"unexpected read: {request.relative_path}")


def _fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes().rstrip(b"\n")


def _replace_fixture_strings(name: str, replacements: dict[str, str]) -> bytes:
    payload = _fixture(name)
    for old, new in replacements.items():
        payload = payload.replace(json.dumps(old).encode(), json.dumps(new).encode())
    return payload


def _source(source_id: str, ordinal: int, payload: bytes) -> SourceUnit:
    return SourceUnit(
        source_unit_id=source_id,
        context_manifest_entry_id="context-1",
        ordinal_1_indexed=ordinal,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=payload,
    )


def _standard_response(result: object) -> bytes:
    return json.dumps(
        {
            "status": "ok",
            "result": result,
            "error": None,
            "telemetry": None,
            "profile": None,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _batch_success(root_uri: str, chunk_uris: tuple[str, ...]) -> bytes:
    return json.dumps(
        {
            "status": "ok",
            "result": {
                "root_uri": root_uri,
                "created": list(chunk_uris),
                "updated": [],
                "unchanged": [],
                "queue_status": {
                    "Semantic": {
                        "processed": len(chunk_uris),
                        "requeue_count": 0,
                        "error_count": 0,
                        "errors": [],
                    },
                    "Embedding": {
                        "processed": len(chunk_uris),
                        "requeue_count": 0,
                        "error_count": 0,
                        "errors": [],
                    },
                },
            },
        },
        separators=(",", ":"),
    ).encode()


def _find_success(result: dict[str, object]) -> bytes:
    return json.dumps(
        {"status": "ok", "result": result},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()


def _permission_denied(resource: str) -> bytes:
    return json.dumps(
        {
            "status": "error",
            "result": None,
            "error": {
                "code": "PERMISSION_DENIED",
                "message": "Permission denied",
                "details": {"resource": resource},
            },
            "telemetry": None,
            "profile": None,
        },
        separators=(",", ":"),
    ).encode()


def _projection_response(
    request: httpx.Request,
    *,
    chunk_contents: dict[str, str],
) -> httpx.Response | None:
    abstract_uri = f"{EXPECTED_ROOT}/.abstract.md"
    overview_uri = f"{EXPECTED_ROOT}/.overview.md"
    inventory_uris = (abstract_uri, overview_uri, *chunk_contents)
    path = request.url.path
    if path == "/api/v1/fs/ls":
        return httpx.Response(200, content=_standard_response(list(inventory_uris)))
    if path == "/api/v1/fs/attrs":
        uri = request.url.params["uri"]
        return httpx.Response(
            200,
            content=_standard_response(
                {"uri": uri, "context_type": "resource", "attrs": {"tags": []}}
            ),
        )
    if path == "/api/v1/content/read":
        uri = request.url.params["uri"]
        contents = {
            abstract_uri: "fixture abstract",
            overview_uri: "fixture overview",
            **chunk_contents,
        }
        return httpx.Response(200, content=_standard_response(contents[uri]))
    return None


def _openviking_module() -> Any:
    try:
        return import_module("oamb.memory_systems.openviking")
    except ModuleNotFoundError:
        pytest.fail("the OpenViking REST adapter is not implemented")


def _adapter(
    handler: httpx.MockTransport,
    *,
    store: CapturingStore | None = None,
) -> Any:
    module = _openviking_module()
    return module.OpenVikingRestAdapter(
        store=store or CapturingStore(),
        base_url="https://openviking.example",
        api_key="user-secret",
        benchmark_account="oamb-benchmark",
        benchmark_user="oamb-admin",
        runtime_binding_hash="f" * 64,
        transport=handler,
    )


def test_runtime_binding_hash_must_be_lowercase_sha256() -> None:
    module = _openviking_module()

    with pytest.raises(ValueError, match="runtime binding hash|SHA-256"):
        module.OpenVikingRestAdapter(
            store=CapturingStore(),
            base_url="https://openviking.example",
            api_key="user-secret",
            benchmark_account="oamb-benchmark",
            benchmark_user="oamb-admin",
            runtime_binding_hash="not-a-runtime-binding-hash",
            transport=httpx.MockTransport(
                lambda request: pytest.fail(
                    f"constructor validation dispatched {request.method} {request.url}"
                )
            ),
        )


async def _resolved_adapter(
    transport: httpx.MockTransport,
    *,
    store: CapturingStore | None = None,
) -> Any:
    async def profile_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=_fixture("health-user.json"))
        return await transport.handle_async_request(request)

    adapter = _adapter(httpx.MockTransport(profile_handler), store=store)
    await adapter.resolve()
    return adapter


async def _mark_ready(adapter: Any, scope: Any, dispatch_receipt: Any) -> None:
    ingestion_receipt = IngestionReceipt(
        ingestion_occurrence_id=scope.ingestion_occurrence_id,
        accepted_source_unit_ids=dispatch_receipt.accepted_source_unit_ids,
        rejected_source_unit_ids=dispatch_receipt.rejected_source_unit_ids,
        raw_references=(dispatch_receipt.raw_reference,),
        dispatch_receipts=(dispatch_receipt,),
    )
    readiness = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=dispatch_receipt.accepted_source_unit_ids,
            ingestion_receipt=ingestion_receipt,
        )
    )
    assert readiness.ready is True


@pytest.mark.asyncio
async def test_resolve_requires_exact_user_bound_non_root_health() -> None:
    calls: list[httpx.Request] = []
    store = CapturingStore()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=_fixture("health-user.json"))

    adapter = _adapter(httpx.MockTransport(handler), store=store)
    resolution = await adapter.resolve()

    assert len(calls) == 1
    assert calls[0].method == "GET"
    assert calls[0].url == httpx.URL("https://openviking.example/health")
    assert calls[0].headers["X-API-Key"] == "user-secret"
    assert "X-OpenViking-Actor-Peer" not in calls[0].headers
    assert resolution.memory_system_id == "openviking"
    assert resolution.runtime_binding_hash == "f" * 64
    assert (
        resolution.raw_reference.sha256 == hashlib.sha256(_fixture("health-user.json")).hexdigest()
    )
    assert store.raw[0].payload_bytes == _fixture("health-user.json")
    await adapter.close()


@pytest.mark.asyncio
async def test_health_validation_failure_preserves_sealed_response() -> None:
    malformed_response = b'{"unexpected":true}'
    adapter = _adapter(
        httpx.MockTransport(lambda request: httpx.Response(200, content=malformed_response))
    )

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.resolve()

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_response_bytes == malformed_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(malformed_response).hexdigest()
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_scope_allocation_requires_resolution_before_any_request() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("scope allocation must not dispatch before resolve")

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="resolve"):
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
        )
    assert calls == []
    await adapter.close()


@pytest.mark.asyncio
async def test_allocate_scope_proves_peer_and_root_absent_before_create_only_mkdir() -> None:
    calls: list[httpx.Request] = []
    store = CapturingStore()
    mkdir_response = _fixture("mkdir-ok.json").replace(
        b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        body = json.loads(request.content)
        assert body == {"uri": EXPECTED_ROOT}
        return httpx.Response(200, content=mkdir_response)

    adapter = await _resolved_adapter(httpx.MockTransport(handler), store=store)
    receipt = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(
            ingestion_occurrence_id=INGESTION_OCCURRENCE_ID,
            ingestion_plan_id=INGESTION_PLAN_ID,
        )
    )

    assert receipt.ingestion_occurrence_id == INGESTION_OCCURRENCE_ID
    assert receipt.scope_id == EXPECTED_ROOT
    assert [request.method for request in calls] == ["GET", "GET", "POST"]
    assert [request.url.params["uri"] for request in calls[:2]] == [
        EXPECTED_PEER_ROOT,
        EXPECTED_ROOT,
    ]
    assert all(request.headers["X-OpenViking-Actor-Peer"] == EXPECTED_PEER_ID for request in calls)
    assert json.loads(calls[2].content) == {"uri": EXPECTED_ROOT}
    assert calls[2].url.path == "/api/v1/fs/mkdir"
    assert receipt.raw_reference.sha256 == hashlib.sha256(mkdir_response).hexdigest()
    assert receipt.supporting_raw_references == tuple(
        RawReferenceHandle(item.sha256) for item in store.raw[1:3]
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_mkdir_validation_failure_preserves_sealed_response() -> None:
    malformed_response = b'{"unexpected":true}'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        return httpx.Response(200, content=malformed_response)

    adapter = await _resolved_adapter(httpx.MockTransport(handler))

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
        )

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_response_bytes == malformed_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(malformed_response).hexdigest()
    )
    await adapter.close()


def test_fixture_identity_literals_are_independent_of_adapter_helpers() -> None:
    assert (
        EXPECTED_PEER_ID
        == "oamb-" + hashlib.sha256(b"peer\0" + INGESTION_OCCURRENCE_ID.encode()).hexdigest()
    )
    assert EXPECTED_PLAN_KEY == hashlib.sha256(b"plan\0" + INGESTION_PLAN_ID.encode()).hexdigest()


@pytest.mark.asyncio
async def test_single_batch_plan_ingest_and_projection_bound_readiness_are_exact() -> None:
    calls: list[httpx.Request] = []
    sources = (
        _source("source-1", 1, b"alpha"),
        _source("source-2", 2, "beta \u03b2".encode()),
    )
    chunk_uris = (f"{EXPECTED_ROOT}/chunk-0001.txt", f"{EXPECTED_ROOT}/chunk-0002.txt")
    batch_response = _replace_fixture_strings(
        "batch-write-success.json",
        {"ROOT": EXPECTED_ROOT, "CHUNK1": chunk_uris[0], "CHUNK2": chunk_uris[1]},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        projection_response = _projection_response(
            request,
            chunk_contents={chunk_uris[0]: "alpha", chunk_uris[1]: "beta β"},
        )
        if projection_response is not None:
            return projection_response
        if request.url.path == "/api/v1/fs/mkdir":
            return httpx.Response(
                200,
                content=_fixture("mkdir-ok.json").replace(
                    b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
                ),
            )
        return httpx.Response(200, content=batch_response)

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    capabilities = await adapter.capabilities()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatches = adapter.plan_ingestion(IngestionRequest(scope, sources))

    assert capabilities.provider_order_preserved is True
    assert capabilities.native_reranking_disabled is True
    assert len(dispatches) == 1
    dispatch = dispatches[0]
    assert dispatch.dispatch_ordinal_1_indexed == 1
    assert dispatch.operation_kind == "openviking_batch_write"
    assert dispatch.ordered_source_units == sources
    assert len(dispatch.request_fingerprint) == 64

    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )

    batch_call = calls[-1]
    assert batch_call.method == "POST"
    assert batch_call.url.path == "/api/v1/content/batch-write"
    assert batch_call.headers["X-OpenViking-Actor-Peer"] == EXPECTED_PEER_ID
    assert json.loads(batch_call.content) == {
        "root_uri": EXPECTED_ROOT,
        "operations": [
            {"uri": chunk_uris[0], "content": "alpha", "mode": "create"},
            {"uri": chunk_uris[1], "content": "beta \u03b2", "mode": "create"},
        ],
        "wait": True,
    }
    assert receipt.attempt_id == "c" * 64
    assert receipt.accepted_source_unit_ids == ("source-1", "source-2")
    assert receipt.rejected_source_unit_ids == ()
    assert receipt.raw_response_bytes == batch_response
    assert receipt.usage_records == ()

    with pytest.raises(Exception, match="ready|readiness"):
        await adapter.retrieve(RetrievalRequest(scope, "d" * 64, b"query before readiness", 100))
    assert not any(request.url.path == "/api/v1/search/find" for request in calls)

    ingestion = IngestionReceipt(
        ingestion_occurrence_id=INGESTION_OCCURRENCE_ID,
        accepted_source_unit_ids=receipt.accepted_source_unit_ids,
        rejected_source_unit_ids=receipt.rejected_source_unit_ids,
        raw_references=(receipt.raw_reference,),
        dispatch_receipts=(receipt,),
    )
    readiness = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=("source-1", "source-2"),
            ingestion_receipt=ingestion,
        )
    )

    assert readiness.ready is True
    assert len(readiness.evidence_references) == 3
    assert readiness.evidence_references[0] == receipt.raw_reference
    assert [request.url.path for request in calls].count("/api/v1/fs/ls") == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_complete_batch_receipt_cannot_bypass_incomplete_readiness_projection() -> None:
    calls: list[httpx.Request] = []
    source = _source("source-1", 1, b"alpha")
    chunk_uri = f"{EXPECTED_ROOT}/chunk-0001.txt"
    batch_response = _batch_success(EXPECTED_ROOT, (chunk_uri,))

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        if request.url.path == "/api/v1/fs/mkdir":
            return httpx.Response(
                200,
                content=_fixture("mkdir-ok.json").replace(
                    b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
                ),
            )
        if request.url.path == "/api/v1/content/batch-write":
            return httpx.Response(200, content=batch_response)
        if request.url.path == "/api/v1/fs/ls":
            return httpx.Response(
                200,
                content=_standard_response(
                    [f"{EXPECTED_ROOT}/.abstract.md", f"{EXPECTED_ROOT}/.overview.md"]
                ),
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )
    ingestion = IngestionReceipt(
        ingestion_occurrence_id=INGESTION_OCCURRENCE_ID,
        accepted_source_unit_ids=receipt.accepted_source_unit_ids,
        rejected_source_unit_ids=receipt.rejected_source_unit_ids,
        raw_references=(receipt.raw_reference,),
        dispatch_receipts=(receipt,),
    )

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=("source-1",),
                ingestion_receipt=ingestion,
            )
        )
    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_reference is not None
    assert failure.value.raw_response_bytes is not None
    with pytest.raises(Exception, match="ready|readiness"):
        await adapter.retrieve(RetrievalRequest(scope, "d" * 64, b"query", 100))

    assert [request.url.path for request in calls].count("/api/v1/fs/ls") == 1
    assert not any(request.url.path == "/api/v1/search/find" for request in calls)
    await adapter.close()


@pytest.mark.asyncio
async def test_lost_batch_receipt_is_unknown_and_never_retried_or_deleted() -> None:
    calls: list[httpx.Request] = []
    source = _source("source-1", 1, b"alpha")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.method == "GET":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        if request.url.path == "/api/v1/fs/mkdir":
            return httpx.Response(
                200,
                content=_fixture("mkdir-ok.json").replace(
                    b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
                ),
            )
        raise httpx.ReadTimeout("lost batch receipt", request=request)

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]

    with pytest.raises(MemorySystemCallUnknownOutcome):
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
        )
    with pytest.raises(Exception, match="ready|readiness"):
        await adapter.retrieve(RetrievalRequest(scope, "d" * 64, b"query", 100))
    await adapter.close()
    await adapter.close()

    assert [request.url.path for request in calls].count("/api/v1/content/batch-write") == 1
    assert all(request.method != "DELETE" for request in calls)


@pytest.mark.asyncio
async def test_batch_validation_failure_preserves_sealed_response() -> None:
    source = _source("source-1", 1, b"alpha")
    malformed_response = b'{"unexpected":true}'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        if request.url.path == "/api/v1/fs/mkdir":
            return httpx.Response(
                200,
                content=_fixture("mkdir-ok.json").replace(
                    b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
                ),
            )
        return httpx.Response(200, content=malformed_response)

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
        )

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_response_bytes == malformed_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(malformed_response).hexdigest()
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_scope_absence_requires_the_exact_not_found_error_envelope() -> None:
    calls: list[httpx.Request] = []
    wrong_error = _fixture("stat-not-found.json").replace(b'"NOT_FOUND"', b'"PERMISSION_DENIED"')

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(404, content=wrong_error)

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
        )
    assert failure.value.failure_kind == "response_validation"
    assert failure.value.status_code == 404
    assert failure.value.raw_response_bytes == wrong_error
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(wrong_error).hexdigest()
    )
    assert len(calls) == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_scope_collision_preserves_successful_stat_response() -> None:
    existing_response = _standard_response(
        {"uri": EXPECTED_PEER_ROOT, "context_type": "resource", "attrs": {"tags": []}}
    )
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=existing_response)

    adapter = await _resolved_adapter(httpx.MockTransport(handler))

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
        )

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.status_code == 200
    assert failure.value.raw_response_bytes == existing_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(existing_response).hexdigest()
    )
    assert [request.url.path for request in calls] == ["/api/v1/fs/stat"]
    await adapter.close()


@pytest.mark.asyncio
async def test_lost_mkdir_receipt_retires_allocation_without_replay() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        raise httpx.ReadTimeout("lost mkdir receipt", request=request)

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    allocation = ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)

    with pytest.raises(MemorySystemCallUnknownOutcome):
        await adapter.allocate_ingestion_scope(allocation)
    call_count = len(calls)
    with pytest.raises(Exception, match="attempted|replay"):
        await adapter.allocate_ingestion_scope(allocation)

    assert len(calls) == call_count == 3
    await adapter.close()


@pytest.mark.asyncio
async def test_concurrent_scope_allocation_dispatches_mkdir_once() -> None:
    calls: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        await asyncio.sleep(0)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        return httpx.Response(
            200,
            content=_fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            ),
        )

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    allocation = ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)

    results = await asyncio.gather(
        adapter.allocate_ingestion_scope(allocation),
        adapter.allocate_ingestion_scope(allocation),
        return_exceptions=True,
    )

    assert sum(not isinstance(result, BaseException) for result in results) == 1
    failures = [result for result in results if isinstance(result, BaseException)]
    assert len(failures) == 1
    assert any(term in str(failures[0]) for term in ("attempted", "replay"))
    mkdir_calls = [request for request in calls if request.url.path == "/api/v1/fs/mkdir"]
    assert len(mkdir_calls) == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_cancelled_allocation_preflight_releases_owner_for_waiter() -> None:
    calls: list[httpx.Request] = []
    first_stat_started = asyncio.Event()
    block_first_stat = True

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal block_first_stat
        calls.append(request)
        if request.url.path == "/api/v1/fs/stat" and block_first_stat:
            block_first_stat = False
            first_stat_started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled stat request unexpectedly resumed")
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        return httpx.Response(
            200,
            content=_fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            ),
        )

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    allocation = ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    owner = asyncio.create_task(adapter.allocate_ingestion_scope(allocation))
    await first_stat_started.wait()
    waiter = asyncio.create_task(adapter.allocate_ingestion_scope(allocation))
    await asyncio.sleep(0)

    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    scope = await waiter

    assert scope.scope_id == EXPECTED_ROOT
    assert [request.url.path for request in calls].count("/api/v1/fs/mkdir") == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_cancelled_second_absence_probe_preserves_first_raw_reference() -> None:
    second_stat_started = asyncio.Event()
    stat_count = 0
    first_response_bytes: bytes | None = None

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal first_response_bytes, stat_count
        assert request.url.path == "/api/v1/fs/stat"
        stat_count += 1
        if stat_count == 1:
            first_response_bytes = _fixture("stat-not-found.json").replace(
                b'"fixture"', json.dumps(request.url.params["uri"]).encode()
            )
            return httpx.Response(404, content=first_response_bytes)
        second_stat_started.set()
        await asyncio.Event().wait()
        raise AssertionError("cancelled second stat unexpectedly resumed")

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    allocation_task = asyncio.create_task(
        adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
        )
    )
    await second_stat_started.wait()
    allocation_task.cancel()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await allocation_task

    assert isinstance(cancelled.value, MemorySystemReadCancelled)
    assert first_response_bytes is not None
    assert cancelled.value.supporting_raw_references == (
        RawReferenceHandle(hashlib.sha256(first_response_bytes).hexdigest()),
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_project_reads_exact_hidden_inventory_attrs_tags_and_full_raw_content() -> None:
    calls: list[httpx.Request] = []
    store = CapturingStore()
    sources = (
        _source("source-1", 1, b"alpha"),
        _source("source-2", 2, "beta \u03b2".encode()),
    )
    chunk_uris = (f"{EXPECTED_ROOT}/chunk-0001.txt", f"{EXPECTED_ROOT}/chunk-0002.txt")
    abstract_uri = f"{EXPECTED_ROOT}/.abstract.md"
    overview_uri = f"{EXPECTED_ROOT}/.overview.md"
    inventory_uris = (abstract_uri, overview_uri, *chunk_uris)
    contents = {
        abstract_uri: "abstract body",
        overview_uri: "overview body",
        chunk_uris[0]: "alpha",
        chunk_uris[1]: "beta \u03b2",
    }
    tags = {uri: ["z=2", "a=1"] if uri == EXPECTED_ROOT else [] for uri in inventory_uris}
    tags[EXPECTED_ROOT] = ["z=2", "a=1"]
    batch_response = _replace_fixture_strings(
        "batch-write-success.json",
        {"ROOT": EXPECTED_ROOT, "CHUNK1": chunk_uris[0], "CHUNK2": chunk_uris[1]},
    )
    ls_response = _replace_fixture_strings(
        "projection-ls.json",
        {
            "OVERVIEW": overview_uri,
            "CHUNK2": chunk_uris[1],
            "ABSTRACT": abstract_uri,
            "CHUNK1": chunk_uris[0],
        },
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        path = request.url.path
        if path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        if path == "/api/v1/fs/mkdir":
            return httpx.Response(
                200,
                content=_fixture("mkdir-ok.json").replace(
                    b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
                ),
            )
        if path == "/api/v1/content/batch-write":
            return httpx.Response(200, content=batch_response)
        if path == "/api/v1/fs/ls":
            return httpx.Response(200, content=ls_response)
        uri = request.url.params["uri"]
        if path == "/api/v1/fs/attrs":
            return httpx.Response(
                200,
                content=_standard_response(
                    {"uri": uri, "context_type": "resource", "attrs": {"tags": tags[uri]}}
                ),
            )
        if path == "/api/v1/content/read":
            return httpx.Response(200, content=_standard_response(contents[uri]))
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = await _resolved_adapter(httpx.MockTransport(handler), store=store)
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, sources))[0]
    await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )
    projection = await adapter.project(scope)

    ls_call = next(request for request in calls if request.url.path == "/api/v1/fs/ls")
    assert dict(ls_call.url.params) == {
        "uri": EXPECTED_ROOT,
        "simple": "true",
        "recursive": "true",
        "output": "original",
        "show_all_hidden": "true",
        "node_limit": "5",
    }
    assert ls_call.headers["X-OpenViking-Actor-Peer"] == EXPECTED_PEER_ID
    attrs_calls = [request for request in calls if request.url.path == "/api/v1/fs/attrs"]
    read_calls = [request for request in calls if request.url.path == "/api/v1/content/read"]
    assert {request.url.params["uri"] for request in attrs_calls} == {
        EXPECTED_ROOT,
        *inventory_uris,
    }
    assert {request.url.params["uri"] for request in read_calls} == set(inventory_uris)
    assert all(
        dict(request.url.params)
        == {"uri": request.url.params["uri"], "offset": "0", "limit": "-1", "raw": "true"}
        for request in read_calls
    )
    expected_state = canonical_sha256(
        [
            "oamb-openviking-projection-v1",
            EXPECTED_ROOT,
            (
                (EXPECTED_ROOT, None, ("a=1", "z=2")),
                *((uri, contents[uri], ()) for uri in sorted(inventory_uris)),
            ),
        ]
    )
    assert projection.inventory.ingestion_occurrence_id == INGESTION_OCCURRENCE_ID
    assert projection.inventory.ordered_source_unit_ids == ("source-1", "source-2")
    assert projection.inventory.raw_reference.sha256 == hashlib.sha256(ls_response).hexdigest()
    assert projection.state_digest.state_sha256 == expected_state
    assert len(projection.supporting_raw_references) == 9
    manifest = next(
        request.payload_bytes
        for request in store.raw
        if request.sha256 == projection.state_digest.raw_reference.sha256
    )
    assert json.loads(manifest)["state_sha256"] == expected_state
    await adapter.close()


@pytest.mark.asyncio
async def test_find_hydrates_l0_l1_l2_in_provider_order_with_ordered_raw_references() -> None:
    calls: list[httpx.Request] = []
    store = CapturingStore()
    sources = (
        _source("source-1", 1, b"alpha"),
        _source("source-2", 2, b"beta"),
    )
    chunk_uris = (f"{EXPECTED_ROOT}/chunk-0001.txt", f"{EXPECTED_ROOT}/chunk-0002.txt")
    abstract_uri = f"{EXPECTED_ROOT}/.abstract.md"
    overview_uri = f"{EXPECTED_ROOT}/.overview.md"
    batch_response = _replace_fixture_strings(
        "batch-write-success.json",
        {"ROOT": EXPECTED_ROOT, "CHUNK1": chunk_uris[0], "CHUNK2": chunk_uris[1]},
    )
    find_response = _replace_fixture_strings(
        "find-levels.json",
        {"OVERVIEW": overview_uri, "CHUNK1": chunk_uris[0], "ABSTRACT": abstract_uri},
    )
    read_response = _standard_response("alpha")

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        if request.url.path == "/api/v1/fs/mkdir":
            return httpx.Response(
                200,
                content=_fixture("mkdir-ok.json").replace(
                    b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
                ),
            )
        if request.url.path == "/api/v1/content/batch-write":
            return httpx.Response(200, content=batch_response)
        projection_response = _projection_response(
            request,
            chunk_contents={chunk_uris[0]: "alpha", chunk_uris[1]: "beta"},
        )
        if projection_response is not None:
            return projection_response
        if request.url.path == "/api/v1/search/find":
            return httpx.Response(200, content=find_response)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = await _resolved_adapter(httpx.MockTransport(handler), store=store)
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, sources))[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )
    await _mark_ready(adapter, scope, receipt)
    projection_read_count = sum(request.url.path == "/api/v1/content/read" for request in calls)
    batch = await adapter.retrieve(
        RetrievalRequest(
            scope=scope,
            case_occurrence_id="d" * 64,
            query_bytes=b"where is alpha?",
            top_k=100,
        )
    )

    find_call = next(request for request in calls if request.url.path == "/api/v1/search/find")
    assert find_call.method == "POST"
    assert find_call.headers["X-OpenViking-Actor-Peer"] == EXPECTED_PEER_ID
    assert json.loads(find_call.content) == {
        "query": "where is alpha?",
        "target_uri": EXPECTED_ROOT,
        "context_type": "resource",
        "limit": 100,
    }
    read_calls = [request for request in calls if request.url.path == "/api/v1/content/read"][
        projection_read_count:
    ]
    assert len(read_calls) == 1
    assert dict(read_calls[0].url.params) == {
        "uri": chunk_uris[0],
        "offset": "0",
        "limit": "-1",
        "raw": "true",
    }
    assert [candidate.evidence_kind for candidate in batch.candidates] == [
        "native_overview",
        "source_content",
        "native_abstract",
    ]
    assert [candidate.content for candidate in batch.candidates] == [
        "overview match",
        "alpha",
        "abstract match",
    ]
    assert [candidate.native_rank_1_indexed for candidate in batch.candidates] == [1, 2, 3]
    assert [candidate.native_score for candidate in batch.candidates] == ["0.91", "0.8", "0.7"]
    assert [candidate.native_id for candidate in batch.candidates] == [
        f"{overview_uri}#level=1",
        f"{chunk_uris[0]}#level=2",
        f"{abstract_uri}#level=0",
    ]
    assert [candidate.source_unit_id for candidate in batch.candidates] == [
        None,
        "source-1",
        None,
    ]
    assert all(candidate.native_truncated is False for candidate in batch.candidates)
    assert batch.raw_reference.sha256 == hashlib.sha256(find_response).hexdigest()
    assert batch.supporting_raw_references == (
        RawReferenceHandle(hashlib.sha256(read_response).hexdigest()),
    )
    assert not any(request.url.path.endswith("/overview") for request in calls)
    assert not any(request.url.path.endswith("/abstract") for request in calls)
    with pytest.raises(Exception, match="top_k|100"):
        await adapter.retrieve(
            RetrievalRequest(
                scope=scope,
                case_occurrence_id="e" * 64,
                query_bytes=b"where is beta?",
                top_k=3,
            )
        )
    assert [request.url.path for request in calls].count("/api/v1/search/find") == 1
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_mode", "expected_failure_kind"),
    [
        ("response_validation", "response_validation"),
        ("http_status", "http_status"),
        ("transport_error", "transport_error"),
        ("cancelled", "cancelled"),
    ],
)
async def test_late_l2_failure_preserves_find_and_prior_hydration_raws(
    failure_mode: str,
    expected_failure_kind: str,
) -> None:
    sources = (
        _source("source-1", 1, b"alpha"),
        _source("source-2", 2, b"beta"),
    )
    chunk_uris = (f"{EXPECTED_ROOT}/chunk-0001.txt", f"{EXPECTED_ROOT}/chunk-0002.txt")
    batch_response = _batch_success(EXPECTED_ROOT, chunk_uris)
    find_response = _find_success(
        {
            "memories": [],
            "resources": [
                {
                    "context_type": "resource",
                    "uri": uri,
                    "level": 2,
                    "score": score,
                    "abstract": f"match {ordinal}",
                    "tags": [],
                }
                for ordinal, (uri, score) in enumerate(
                    zip(chunk_uris, (0.9, 0.8), strict=True),
                    start=1,
                )
            ],
            "skills": [],
            "total": 2,
        }
    )
    first_read_response = _standard_response("alpha")
    failure_response = b'{"unexpected":true}'
    find_seen = False
    second_hydration_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal find_seen
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        if request.url.path == "/api/v1/fs/mkdir":
            return httpx.Response(
                200,
                content=_fixture("mkdir-ok.json").replace(
                    b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
                ),
            )
        if request.url.path == "/api/v1/content/batch-write":
            return httpx.Response(200, content=batch_response)
        if request.url.path == "/api/v1/search/find":
            find_seen = True
            return httpx.Response(200, content=find_response)
        if not find_seen:
            projection_response = _projection_response(
                request,
                chunk_contents={chunk_uris[0]: "alpha", chunk_uris[1]: "beta"},
            )
            if projection_response is not None:
                return projection_response
        if request.url.path == "/api/v1/content/read":
            if request.url.params["uri"] == chunk_uris[0]:
                return httpx.Response(200, content=first_read_response)
            if failure_mode == "response_validation":
                return httpx.Response(200, content=failure_response)
            if failure_mode == "http_status":
                return httpx.Response(500, content=failure_response)
            if failure_mode == "transport_error":
                raise httpx.ConnectError("L2 transport failed", request=request)
            if failure_mode == "cancelled":
                second_hydration_started.set()
                await asyncio.Event().wait()
                raise AssertionError("cancelled L2 hydration unexpectedly resumed")
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, sources))[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )
    await _mark_ready(adapter, scope, receipt)

    request = RetrievalRequest(scope, "d" * 64, b"query", 100)
    call_failure: MemorySystemCallFailure | MemorySystemReadCancelled
    if failure_mode == "cancelled":
        retrieval_task = asyncio.create_task(adapter.retrieve(request))
        await second_hydration_started.wait()
        retrieval_task.cancel()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await retrieval_task
        assert isinstance(cancelled.value, MemorySystemReadCancelled)
        call_failure = cancelled.value
    else:
        with pytest.raises(MemorySystemCallFailure) as failure:
            await adapter.retrieve(request)
        call_failure = failure.value

    assert call_failure.failure_kind == expected_failure_kind
    assert call_failure.raw_response_bytes == (
        failure_response if failure_mode in {"response_validation", "http_status"} else None
    )
    assert call_failure.supporting_raw_references == (
        RawReferenceHandle(hashlib.sha256(find_response).hexdigest()),
        RawReferenceHandle(hashlib.sha256(first_read_response).hexdigest()),
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_find_rejects_cross_peer_evidence_before_any_hydration() -> None:
    calls: list[httpx.Request] = []
    source = _source("source-1", 1, b"alpha")
    chunk_uri = f"{EXPECTED_ROOT}/chunk-0001.txt"
    wrong_uri = "viking://user/oamb-admin/peers/oamb-wrong/resources/oamb/mab65-v1/x/chunk-0001.txt"
    batch_response = _batch_success(EXPECTED_ROOT, (chunk_uri,))
    find_response = _find_success(
        {
            "memories": [],
            "resources": [
                {
                    "context_type": "resource",
                    "uri": wrong_uri,
                    "level": 2,
                    "score": 0.5,
                    "abstract": "wrong peer",
                    "tags": [],
                }
            ],
            "skills": [],
            "total": 1,
        }
    )

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/fs/stat":
            content = _fixture("stat-not-found.json").replace(
                b'"fixture"', json.dumps(request.url.params["uri"]).encode()
            )
            return httpx.Response(404, content=content)
        if request.url.path == "/api/v1/fs/mkdir":
            content = _fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            )
            return httpx.Response(200, content=content)
        if request.url.path == "/api/v1/content/batch-write":
            return httpx.Response(200, content=batch_response)
        projection_response = _projection_response(
            request,
            chunk_contents={chunk_uri: "alpha"},
        )
        if projection_response is not None:
            return projection_response
        if request.url.path == "/api/v1/search/find":
            return httpx.Response(200, content=find_response)
        raise AssertionError("cross-peer evidence must be rejected before hydration")

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )
    await _mark_ready(adapter, scope, receipt)
    projection_read_count = sum(request.url.path == "/api/v1/content/read" for request in calls)

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.retrieve(RetrievalRequest(scope, "d" * 64, b"query", 100))
    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_reference is not None
    assert failure.value.raw_response_bytes == find_response
    assert failure.value.raw_reference.sha256 == hashlib.sha256(find_response).hexdigest()
    assert (
        sum(request.url.path == "/api/v1/content/read" for request in calls)
        == projection_read_count
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_wrong_peer_find_probe_requires_exact_permission_denial() -> None:
    calls: list[httpx.Request] = []
    wrong_peer = "oamb-" + "e" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers.get("X-OpenViking-Actor-Peer") == wrong_peer:
            if request.url.path == "/api/v1/fs/stat":
                resource = request.url.params["uri"]
            else:
                body = json.loads(request.content)
                resource = body.get("uri") or body.get("root_uri") or body.get("target_uri")
            return httpx.Response(403, content=_permission_denied(resource))
        if request.url.path == "/api/v1/fs/stat":
            content = _fixture("stat-not-found.json").replace(
                b'"fixture"', json.dumps(request.url.params["uri"]).encode()
            )
            return httpx.Response(404, content=content)
        if request.url.path == "/api/v1/fs/mkdir":
            content = _fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            )
            return httpx.Response(200, content=content)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    references = (
        await adapter.verify_wrong_peer_stat_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        ),
        await adapter.verify_wrong_peer_create_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        ),
        await adapter.verify_wrong_peer_write_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        ),
        await adapter.verify_wrong_peer_find_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        ),
    )

    probes = calls[-4:]
    assert [probe.url.path for probe in probes] == [
        "/api/v1/fs/stat",
        "/api/v1/fs/mkdir",
        "/api/v1/content/batch-write",
        "/api/v1/search/find",
    ]
    assert all(probe.headers["X-OpenViking-Actor-Peer"] == wrong_peer for probe in probes)
    assert references[0] == references[2] == references[3]
    assert references[1] != references[0]
    assert json.loads(probes[-1].content) == {
        "query": "oamb wrong-peer isolation probe",
        "target_uri": EXPECTED_ROOT,
        "context_type": "resource",
        "limit": 100,
    }
    assert references[-1].sha256 == hashlib.sha256(_permission_denied(EXPECTED_ROOT)).hexdigest()
    assert all(request.method != "DELETE" for request in calls)
    await adapter.close()


@pytest.mark.asyncio
async def test_wrong_peer_denial_validation_failure_preserves_error_response() -> None:
    wrong_peer = "oamb-" + "e" * 64
    malformed_response = b'{"unexpected":true}'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-OpenViking-Actor-Peer") == wrong_peer:
            return httpx.Response(403, content=malformed_response)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        return httpx.Response(
            200,
            content=_fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            ),
        )

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.verify_wrong_peer_find_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        )

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.status_code == 403
    assert failure.value.raw_response_bytes == malformed_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(malformed_response).hexdigest()
    )
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe_method_name",
    ["verify_wrong_peer_create_isolation", "verify_wrong_peer_write_isolation"],
)
async def test_wrong_peer_allowed_write_preserves_success_response(
    probe_method_name: str,
) -> None:
    wrong_peer = "oamb-" + "e" * 64
    allowed_response = b'{"status":"ok"}'
    wrong_peer_calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("X-OpenViking-Actor-Peer") == wrong_peer:
            wrong_peer_calls.append(request)
            return httpx.Response(200, content=allowed_response)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        return httpx.Response(
            200,
            content=_fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            ),
        )

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    probe = getattr(adapter, probe_method_name)

    with pytest.raises(MemorySystemCallFailure) as failure:
        await probe(scope=scope, wrong_actor_peer_id=wrong_peer)

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.status_code == 200
    assert failure.value.raw_response_bytes == allowed_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(allowed_response).hexdigest()
    )
    assert len(wrong_peer_calls) == 1
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("probe_method_name", "probe_path"),
    [
        ("verify_wrong_peer_create_isolation", "/api/v1/fs/mkdir"),
        ("verify_wrong_peer_write_isolation", "/api/v1/content/batch-write"),
    ],
)
async def test_wrong_peer_write_probe_unknown_outcome_is_never_replayed(
    probe_method_name: str,
    probe_path: str,
) -> None:
    calls: list[httpx.Request] = []
    wrong_peer = "oamb-" + "e" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.headers.get("X-OpenViking-Actor-Peer") == wrong_peer:
            raise httpx.ReadTimeout("lost wrong-peer probe receipt", request=request)
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        return httpx.Response(
            200,
            content=_fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            ),
        )

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    probe = getattr(adapter, probe_method_name)

    with pytest.raises(MemorySystemCallUnknownOutcome):
        await probe(scope=scope, wrong_actor_peer_id=wrong_peer)
    with pytest.raises(Exception, match="attempted|replay"):
        await probe(scope=scope, wrong_actor_peer_id=wrong_peer)

    wrong_peer_probe_calls = [
        request for request in calls if request.headers.get("X-OpenViking-Actor-Peer") == wrong_peer
    ]
    assert [request.url.path for request in wrong_peer_probe_calls] == [probe_path]
    await adapter.close()


@pytest.mark.asyncio
async def test_cancelled_queued_wrong_peer_write_is_retryable_before_dispatch() -> None:
    wrong_peer = "oamb-" + "e" * 64
    active_stat_started = asyncio.Event()
    release_active_stat = asyncio.Event()
    wrong_peer_write_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal wrong_peer_write_calls
        if request.headers.get("X-OpenViking-Actor-Peer") == wrong_peer:
            if request.url.path == "/api/v1/fs/stat":
                active_stat_started.set()
                await release_active_stat.wait()
                return httpx.Response(403, content=_permission_denied(EXPECTED_ROOT))
            wrong_peer_write_calls += 1
            return httpx.Response(403, content=_permission_denied(EXPECTED_ROOT))
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        return httpx.Response(
            200,
            content=_fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            ),
        )

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    active_stat = asyncio.create_task(
        adapter.verify_wrong_peer_stat_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        )
    )
    await active_stat_started.wait()
    queued_write = asyncio.create_task(
        adapter.verify_wrong_peer_write_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        )
    )
    await asyncio.sleep(0)
    queued_write.cancel()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await queued_write

    assert isinstance(cancelled.value, MemorySystemCallCancelledBeforeDispatch)
    assert not isinstance(cancelled.value, MemorySystemCallUnknownOutcome)
    assert wrong_peer_write_calls == 0
    release_active_stat.set()
    await active_stat
    await adapter.verify_wrong_peer_write_isolation(
        scope=scope,
        wrong_actor_peer_id=wrong_peer,
    )
    assert wrong_peer_write_calls == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_close_drains_admitted_wrong_peer_probes_and_rejects_new_probe() -> None:
    wrong_peer = "oamb-" + "e" * 64
    late_wrong_peer = "oamb-" + "d" * 64
    active_stat_started = asyncio.Event()
    release_active_stat = asyncio.Event()
    wrong_peer_write_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal wrong_peer_write_calls
        actor_peer = request.headers.get("X-OpenViking-Actor-Peer")
        if actor_peer == wrong_peer:
            if request.url.path == "/api/v1/fs/stat":
                active_stat_started.set()
                await release_active_stat.wait()
                return httpx.Response(403, content=_permission_denied(EXPECTED_ROOT))
            wrong_peer_write_calls += 1
            return httpx.Response(403, content=_permission_denied(EXPECTED_ROOT))
        if actor_peer == late_wrong_peer:
            raise AssertionError("late wrong-peer probe unexpectedly dispatched")
        if request.url.path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        return httpx.Response(
            200,
            content=_fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            ),
        )

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    active_stat = asyncio.create_task(
        adapter.verify_wrong_peer_stat_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        )
    )
    await active_stat_started.wait()
    admitted_write = asyncio.create_task(
        adapter.verify_wrong_peer_write_isolation(
            scope=scope,
            wrong_actor_peer_id=wrong_peer,
        )
    )
    await asyncio.sleep(0)
    closing = asyncio.create_task(adapter.close())
    await asyncio.sleep(0)

    assert not closing.done()
    with pytest.raises(RuntimeError, match="closed"):
        await adapter.verify_wrong_peer_stat_isolation(
            scope=scope,
            wrong_actor_peer_id=late_wrong_peer,
        )

    release_active_stat.set()
    await active_stat
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await admitted_write
    assert isinstance(cancelled.value, MemorySystemCallCancelledBeforeDispatch)
    await closing
    assert wrong_peer_write_calls == 0


@pytest.mark.asyncio
async def test_partial_batch_is_preserved_unready_and_receipt_bytes_cannot_be_forged() -> None:
    calls: list[httpx.Request] = []
    sources = (
        _source("source-1", 1, b"alpha"),
        _source("source-2", 2, b"beta"),
    )
    chunk_uris = (f"{EXPECTED_ROOT}/chunk-0001.txt", f"{EXPECTED_ROOT}/chunk-0002.txt")
    partial_response = _batch_success(EXPECTED_ROOT, chunk_uris[:1])
    full_response = _batch_success(EXPECTED_ROOT, chunk_uris)

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/api/v1/fs/stat":
            content = _fixture("stat-not-found.json").replace(
                b'"fixture"', json.dumps(request.url.params["uri"]).encode()
            )
            return httpx.Response(404, content=content)
        if request.url.path == "/api/v1/fs/mkdir":
            content = _fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            )
            return httpx.Response(200, content=content)
        return httpx.Response(200, content=partial_response)

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, sources))[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )
    assert receipt.accepted_source_unit_ids == ("source-1",)
    assert receipt.rejected_source_unit_ids == ("source-2",)
    ingestion = IngestionReceipt(
        INGESTION_OCCURRENCE_ID,
        receipt.accepted_source_unit_ids,
        receipt.rejected_source_unit_ids,
        (receipt.raw_reference,),
        (receipt,),
    )
    readiness = await adapter.wait_ready(
        ReadinessRequest(scope, ("source-1", "source-2"), ingestion)
    )
    assert readiness.ready is False
    with pytest.raises(Exception, match="ready|readiness"):
        await adapter.retrieve(RetrievalRequest(scope, "d" * 64, b"query", 100))
    assert not any(request.url.path == "/api/v1/search/find" for request in calls)

    forged_receipt = replace(
        receipt,
        accepted_source_unit_ids=("source-1", "source-2"),
        rejected_source_unit_ids=(),
        raw_response_bytes=full_response,
    )
    forged_ingestion = IngestionReceipt(
        INGESTION_OCCURRENCE_ID,
        forged_receipt.accepted_source_unit_ids,
        forged_receipt.rejected_source_unit_ids,
        (forged_receipt.raw_reference,),
        (forged_receipt,),
    )
    with pytest.raises(Exception, match="raw|receipt|immutable|hash"):
        await adapter.wait_ready(
            ReadinessRequest(scope, ("source-1", "source-2"), forged_ingestion)
        )
    with pytest.raises(Exception, match="replay|create-only"):
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
        )
    assert [request.url.path for request in calls].count("/api/v1/content/batch-write") == 1
    assert all(request.method != "DELETE" for request in calls)
    await adapter.close()


@pytest.mark.parametrize(
    ("hit_change", "result_change", "message"),
    [
        ({"level": 3}, {}, "level"),
        ({"level": 1, "abstract": ""}, {}, "preview"),
        ({"future": True}, {}, "fields"),
        ({}, {"total": 2}, "total"),
    ],
)
def test_find_parser_planted_failures_are_terminal(
    hit_change: dict[str, object],
    result_change: dict[str, object],
    message: str,
) -> None:
    parser_module = import_module("oamb.memory_systems.openviking.adapter")
    chunk_uri = f"{EXPECTED_ROOT}/chunk-0001.txt"
    hit_uri = f"{EXPECTED_ROOT}/.overview.md" if hit_change.get("level") == 1 else chunk_uri
    hit = {
        "context_type": "resource",
        "uri": hit_uri,
        "level": 2,
        "score": 0.5,
        "abstract": "preview",
        "tags": [],
    } | hit_change
    result = {
        "memories": [],
        "resources": [hit],
        "skills": [],
        "total": 1,
    } | result_change

    with pytest.raises(Exception, match=message):
        parser_module._parse_find_response(
            _find_success(result),
            root_uri=EXPECTED_ROOT,
            chunk_uris=(chunk_uri,),
        )


def test_find_parser_rejects_duplicate_native_identity() -> None:
    parser_module = import_module("oamb.memory_systems.openviking.adapter")
    chunk_uri = f"{EXPECTED_ROOT}/chunk-0001.txt"
    hit = {
        "context_type": "resource",
        "uri": chunk_uri,
        "level": 2,
        "score": 0.5,
        "abstract": "preview",
        "tags": [],
    }

    with pytest.raises(Exception, match="duplicate|identity"):
        parser_module._parse_find_response(
            _find_success(
                {
                    "memories": [],
                    "resources": [hit, hit],
                    "skills": [],
                    "total": 2,
                }
            ),
            root_uri=EXPECTED_ROOT,
            chunk_uris=(chunk_uri,),
        )


@pytest.mark.asyncio
async def test_read_only_query_detects_projection_mutation_after_find() -> None:
    calls: list[httpx.Request] = []
    find_seen = False
    source = _source("source-1", 1, b"alpha")
    chunk_uri = f"{EXPECTED_ROOT}/chunk-0001.txt"
    abstract_uri = f"{EXPECTED_ROOT}/.abstract.md"
    overview_uri = f"{EXPECTED_ROOT}/.overview.md"
    inventory_uris = (abstract_uri, overview_uri, chunk_uri)
    batch_response = _batch_success(EXPECTED_ROOT, (chunk_uri,))
    ls_response = _standard_response(list(inventory_uris))
    empty_find = _find_success({"memories": [], "resources": [], "skills": [], "total": 0})

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal find_seen
        calls.append(request)
        path = request.url.path
        if path == "/api/v1/fs/stat":
            content = _fixture("stat-not-found.json").replace(
                b'"fixture"', json.dumps(request.url.params["uri"]).encode()
            )
            return httpx.Response(404, content=content)
        if path == "/api/v1/fs/mkdir":
            content = _fixture("mkdir-ok.json").replace(
                b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
            )
            return httpx.Response(200, content=content)
        if path == "/api/v1/content/batch-write":
            return httpx.Response(200, content=batch_response)
        if path == "/api/v1/fs/ls":
            return httpx.Response(200, content=ls_response)
        if path == "/api/v1/fs/attrs":
            uri = request.url.params["uri"]
            return httpx.Response(
                200,
                content=_standard_response(
                    {"uri": uri, "context_type": "resource", "attrs": {"tags": []}}
                ),
            )
        if path == "/api/v1/content/read":
            uri = request.url.params["uri"]
            read_content = {
                abstract_uri: "abstract after" if find_seen else "abstract before",
                overview_uri: "overview",
                chunk_uri: "alpha",
            }[uri]
            return httpx.Response(200, content=_standard_response(read_content))
        if path == "/api/v1/search/find":
            find_seen = True
            return httpx.Response(200, content=empty_find)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )
    await _mark_ready(adapter, scope, receipt)
    ticks = count()
    with pytest.raises(MemorySystemProjectionMutation) as mutation:
        await execute_read_only_retrieval(
            memory=adapter,
            request=RetrievalRequest(scope, "d" * 64, b"query", 100),
            clock=lambda: Decimal(next(ticks)),
        )

    receipt = mutation.value.receipt
    assert (
        receipt.before_projection.state_digest.state_sha256
        != receipt.after_projection.state_digest.state_sha256
    )
    assert [request.url.path for request in calls].count("/api/v1/search/find") == 1
    assert all(request.method != "DELETE" for request in calls)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("drift_kind", ["inventory", "chunk_bytes"])
async def test_post_projection_failure_preserves_native_and_projection_raw_evidence(
    drift_kind: str,
) -> None:
    find_seen = False
    source = _source("source-1", 1, b"alpha")
    chunk_uri = f"{EXPECTED_ROOT}/chunk-0001.txt"
    abstract_uri = f"{EXPECTED_ROOT}/.abstract.md"
    overview_uri = f"{EXPECTED_ROOT}/.overview.md"
    inventory_uris = (abstract_uri, overview_uri, chunk_uri)
    batch_response = _batch_success(EXPECTED_ROOT, (chunk_uri,))
    empty_find = _find_success({"memories": [], "resources": [], "skills": [], "total": 0})

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal find_seen
        path = request.url.path
        if path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                content=_fixture("stat-not-found.json").replace(
                    b'"fixture"', json.dumps(request.url.params["uri"]).encode()
                ),
            )
        if path == "/api/v1/fs/mkdir":
            return httpx.Response(
                200,
                content=_fixture("mkdir-ok.json").replace(
                    b'"fixture"', json.dumps(EXPECTED_ROOT).encode()
                ),
            )
        if path == "/api/v1/content/batch-write":
            return httpx.Response(200, content=batch_response)
        if path == "/api/v1/fs/ls":
            returned_uris = (
                inventory_uris[:-1] if find_seen and drift_kind == "inventory" else inventory_uris
            )
            return httpx.Response(200, content=_standard_response(list(returned_uris)))
        if path == "/api/v1/fs/attrs":
            uri = request.url.params["uri"]
            return httpx.Response(
                200,
                content=_standard_response(
                    {"uri": uri, "context_type": "resource", "attrs": {"tags": []}}
                ),
            )
        if path == "/api/v1/content/read":
            uri = request.url.params["uri"]
            contents = {
                abstract_uri: "abstract",
                overview_uri: "overview",
                chunk_uri: ("changed" if find_seen and drift_kind == "chunk_bytes" else "alpha"),
            }
            return httpx.Response(200, content=_standard_response(contents[uri]))
        if path == "/api/v1/search/find":
            find_seen = True
            return httpx.Response(200, content=empty_find)
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = await _resolved_adapter(httpx.MockTransport(handler))
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]
    dispatch_receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
    )
    await _mark_ready(adapter, scope, dispatch_receipt)
    ticks = count()

    with pytest.raises(ReadOnlyRetrievalFailure) as failure:
        await execute_read_only_retrieval(
            memory=adapter,
            request=RetrievalRequest(scope, "d" * 64, b"query", 100),
            clock=lambda: Decimal(next(ticks)),
        )

    receipt = failure.value.receipt
    assert receipt.native_batch is not None
    assert receipt.native_batch.raw_reference.sha256 == hashlib.sha256(empty_find).hexdigest()
    assert isinstance(receipt.post_projection_error, MemorySystemCallFailure)
    projection_error = receipt.post_projection_error
    assert projection_error.failure_kind == "response_validation"
    assert projection_error.raw_reference is not None
    assert projection_error.raw_response_bytes is not None
    assert (
        projection_error.raw_reference.sha256
        == hashlib.sha256(projection_error.raw_response_bytes).hexdigest()
    )
    if drift_kind == "chunk_bytes":
        assert projection_error.supporting_raw_references
    assert receipt.timing.provider_request.seconds == Decimal("1")
    assert receipt.timing.projection_verification.seconds == Decimal("2")
    await adapter.close()
