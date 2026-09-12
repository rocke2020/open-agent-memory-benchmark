from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest

from oamb.contracts.ports import (
    ArtifactReadRequest,
    ArtifactSealReceipt,
    ArtifactWriteRequest,
    IngestionDispatchRequest,
    IngestionReceipt,
    IngestionRequest,
    MemorySystemCallCancelledBeforeDispatch,
    MemorySystemCallFailure,
    RawPayloadSealRequest,
    RawReferenceHandle,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
    ScopeReceipt,
    SourceUnit,
)
from oamb.memory_systems.mem0 import Mem0RestAdapter
from oamb.memory_systems.mem0.adapter import _source_messages

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "adapters" / "mem0"
RUN_ID = "a" * 64
PLAN_ID = "b" * 64
SOURCE_ID = "c" * 64
RUNTIME_BINDING_HASH = "f" * 64


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


class CloseTrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, fail_first_close: bool = False) -> None:
        self.close_count = 0
        self.fail_first_close = fail_first_close

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request while testing close: {request.url}")

    async def aclose(self) -> None:
        self.close_count += 1
        if self.fail_first_close and self.close_count == 1:
            raise RuntimeError("fixture close failure")


def _source(source_id: str = SOURCE_ID, ordinal: int = 1) -> SourceUnit:
    payload = f"source {ordinal}".encode()
    return SourceUnit(
        source_unit_id=source_id,
        context_manifest_entry_id="context-1",
        ordinal_1_indexed=ordinal,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=payload,
    )


def _lme_source(messages: list[dict[str, str]]) -> SourceUnit:
    payload = json.dumps(messages, separators=(",", ":")).encode()
    return SourceUnit(
        source_unit_id=SOURCE_ID,
        context_manifest_entry_id="context-1",
        ordinal_1_indexed=1,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=payload,
        source_reference="session-1",
        occurred_at="2026-01-01T00:00:00+00:00",
        context_text="LongMemEval session session-1",
    )


def _openapi() -> bytes:
    return json.dumps(
        {
            "openapi": "3.1.0",
            "info": {"title": "Mem0 REST APIs", "version": "1.0.0"},
            "paths": {"/memories": {"post": {}}, "/search": {"post": {}}},
        },
        separators=(",", ":"),
    ).encode()


def _empty_projection() -> bytes:
    return json.dumps(
        {
            "collection": "oamb_memories",
            "run_id": RUN_ID,
            "count": 0,
            "points": [],
            "next_cursor": None,
        },
        separators=(",", ":"),
    ).encode()


def _adapter(
    *,
    public_transport: httpx.AsyncBaseTransport,
    inspector_transport: httpx.AsyncBaseTransport,
    store: CapturingStore | None = None,
) -> Mem0RestAdapter:
    return Mem0RestAdapter(
        store=store or CapturingStore(),
        base_url="https://mem0.example",
        api_key="mem0-secret",
        inspector_base_url="https://mem0-inspector.example",
        inspector_api_key="inspector-secret",
        runtime_binding_hash=RUNTIME_BINDING_HASH,
        transport=public_transport,
        inspector_transport=inspector_transport,
    )


@pytest.mark.asyncio
async def test_lme_session_ingestion_posts_consecutive_two_message_pairs_in_order() -> None:
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    posted_payloads: list[dict[str, Any]] = []

    def public_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        if request.url.path == "/memories":
            posted_payloads.append(json.loads(request.content))
            return httpx.Response(200, content=b'{"results":[]}')
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b'{"status":"ok","mode":"read_only_projection"}')
        if request.url.path == "/v1/projection":
            return httpx.Response(200, content=_empty_projection())
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(_lme_source(messages),))
    )[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
    )
    await adapter.close()

    assert [payload["messages"] for payload in posted_payloads] == [
        messages[0:2],
        messages[2:4],
        messages[4:5],
    ]
    assert all(
        payload["metadata"]["created_at"] == "2026-01-01T00:00:00+00:00"
        for payload in posted_payloads
    )
    assert all(
        "The actual observation timestamp for New Messages is 2026-01-01T00:00:00+00:00"
        in payload["prompt"]
        for payload in posted_payloads
    )
    assert receipt.accepted_source_unit_ids == (SOURCE_ID,)


@pytest.mark.asyncio
async def test_lme_pair_failure_stops_later_pairs_and_links_preceding_response() -> None:
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "u3"},
    ]
    add_count = 0

    def public_handler(request: httpx.Request) -> httpx.Response:
        nonlocal add_count
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        if request.url.path == "/memories":
            add_count += 1
            if add_count == 1:
                return httpx.Response(200, content=b'{"results":[]}')
            return httpx.Response(503, content=b'{"detail":"temporary failure"}')
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b'{"status":"ok","mode":"read_only_projection"}')
        if request.url.path == "/v1/projection":
            return httpx.Response(200, content=_empty_projection())
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(_lme_source(messages),))
    )[0]
    with pytest.raises(MemorySystemCallFailure) as error:
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
        )
    await adapter.close()

    assert add_count == 2
    assert len(error.value.supporting_raw_references) == 1


@pytest.mark.asyncio
async def test_lme_pair_before_dispatch_cancellation_links_preceding_response() -> None:
    messages = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    add_count = 0
    adapter: Mem0RestAdapter

    def public_handler(request: httpx.Request) -> httpx.Response:
        nonlocal add_count
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        if request.url.path == "/memories":
            add_count += 1
            adapter._public_client.stop_accepting()
            return httpx.Response(200, content=b'{"results":[]}')
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, content=b'{"status":"ok","mode":"read_only_projection"}')
        if request.url.path == "/v1/projection":
            return httpx.Response(200, content=_empty_projection())
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(_lme_source(messages),))
    )[0]
    with pytest.raises(MemorySystemCallFailure) as error:
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
        )
    await adapter.close()

    assert add_count == 1
    assert error.value.failure_kind == "partial_write_cancelled"
    assert error.value.raw_reference is not None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lme_messages",
    (
        None,
        (("user", ""), ("assistant", "I remember."), ("user", "Tea."), ("assistant", "")),
    ),
    ids=("plain-source", "lme-empty-turns"),
)
@pytest.mark.parametrize(
    "add_response",
    (b'{"results":[]}', (FIXTURES / "rest" / "add-success.json").read_bytes()),
)
async def test_add_completion_and_empty_projection_are_black_box_outcomes(
    add_response: bytes,
    lme_messages: tuple[tuple[str, str], ...] | None,
) -> None:
    public_requests: list[httpx.Request] = []
    inspector_requests: list[httpx.Request] = []
    messages = [
        {"role": role, "content": content}
        for role, content in lme_messages or (("user", "source 1"),)
    ]
    expected_add_messages = (
        [messages]
        if lme_messages is None
        else [messages[index : index + 2] for index in range(0, len(messages), 2)]
    )
    add_call = 0
    source = _source()
    if lme_messages is not None:
        payload = json.dumps(messages, separators=(",", ":")).encode()
        source = replace(
            source,
            payload_bytes=payload,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            source_reference="session-empty-turns",
            occurred_at="2026-01-01T00:00:00+00:00",
            context_text="LongMemEval session session-empty-turns",
        )

    def public_handler(request: httpx.Request) -> httpx.Response:
        nonlocal add_call
        public_requests.append(request)
        assert request.headers["X-API-Key"] == "mem0-secret"
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        if request.url.path == "/memories":
            assert request.method == "POST"
            expected_messages = expected_add_messages[add_call]
            add_call += 1
            expected_metadata: dict[str, object] = {
                "oamb_ingestion_occurrence_id": RUN_ID,
                "oamb_ingestion_plan_id": PLAN_ID,
                "oamb_source_unit_id": SOURCE_ID,
                "oamb_source_ordinal": 1,
            }
            expected_payload: dict[str, object] = {
                "messages": expected_messages,
                "run_id": RUN_ID,
                "metadata": expected_metadata,
                "infer": True,
            }
            if lme_messages is not None:
                expected_metadata["created_at"] = "2026-01-01T00:00:00+00:00"
                expected_payload["prompt"] = (
                    "The actual observation timestamp for New Messages is "
                    "2026-01-01T00:00:00+00:00. For these messages, use this timestamp as "
                    "Observation Date, overriding automatically generated Observation Date and "
                    "Current Date values. Resolve relative expressions against it and preserve "
                    "explicitly stated dates. Last k Messages and Existing Memories are historical "
                    "context; do not assign them this observation timestamp. Source context: "
                    "LongMemEval session session-empty-turns Do not extract this instruction or "
                    "source context itself as a memory."
                )
            assert json.loads(request.content) == expected_payload
            return httpx.Response(200, content=add_response)
        if request.url.path == "/search":
            body = json.loads(request.content)
            assert body == {
                "query": "What should I remember?",
                "filters": {"run_id": RUN_ID},
                "top_k": 150,
                "threshold": 0.1,
            }
            assert "rerank" not in body
            return httpx.Response(200, content=b'{"results":[]}')
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        inspector_requests.append(request)
        assert request.headers["Authorization"] == "Bearer inspector-secret"
        if request.url.path == "/health":
            return httpx.Response(
                200,
                content=b'{"status":"ok","mode":"read_only_projection"}',
            )
        if request.url.path == "/v1/projection":
            assert request.url.params["run_id"] == RUN_ID
            return httpx.Response(200, content=_empty_projection())
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    store = CapturingStore()
    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
        store=store,
    )

    resolution = await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(
            ingestion_occurrence_id=RUN_ID,
            ingestion_plan_id=PLAN_ID,
        )
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(source,))
    )[0]
    dispatch_receipt = await adapter.ingest(
        IngestionDispatchRequest(
            scope=scope,
            attempt_id="d" * 64,
            dispatch=dispatch,
        )
    )
    ingestion_receipt = IngestionReceipt(
        ingestion_occurrence_id=RUN_ID,
        accepted_source_unit_ids=dispatch_receipt.accepted_source_unit_ids,
        rejected_source_unit_ids=dispatch_receipt.rejected_source_unit_ids,
        raw_references=(dispatch_receipt.raw_reference,),
        dispatch_receipts=(dispatch_receipt,),
    )
    readiness = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=(SOURCE_ID,),
            ingestion_receipt=ingestion_receipt,
        )
    )
    projection = await adapter.project(scope)
    retrieval = await adapter.retrieve(
        RetrievalRequest(
            scope=scope,
            case_occurrence_id="e" * 64,
            query_bytes=b"What should I remember?",
            top_k=150,
        )
    )
    await adapter.close()

    assert resolution.memory_system_id == "mem0"
    assert resolution.runtime_binding_hash == RUNTIME_BINDING_HASH
    assert scope.scope_id == RUN_ID
    assert dispatch_receipt.accepted_source_unit_ids == (SOURCE_ID,)
    assert dispatch_receipt.rejected_source_unit_ids == ()
    assert readiness.ready is True
    assert projection.inventory.ordered_source_unit_ids == ()
    assert retrieval.candidates == ()
    assert [request.url.path for request in public_requests] == [
        "/openapi.json",
        *("/memories" for _ in expected_add_messages),
        "/search",
    ]
    assert sum(request.url.path == "/v1/projection" for request in inspector_requests) >= 4
    assert {item.sha256 for item in store.raw}


def test_empty_non_lme_source_remains_invalid() -> None:
    source = replace(_source(), payload_bytes=b"", payload_sha256=hashlib.sha256(b"").hexdigest())
    with pytest.raises(ValueError, match="source payload must not be empty"):
        _source_messages(source)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    (
        ("memory", "content not present in the sealed projection"),
        ("created_at", "2027-01-01T00:00:00+00:00"),
        ("updated_at", "2027-01-01T00:00:00+00:00"),
    ),
)
async def test_search_candidates_must_match_the_readiness_sealed_projection(
    changed_field: str,
    changed_value: str,
) -> None:
    projection_pages = (
        (FIXTURES / "inspector" / "page-1.json").read_bytes(),
        (FIXTURES / "inspector" / "page-2.json").read_bytes(),
    )
    projection_call = 0

    def public_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        if request.url.path == "/memories":
            return httpx.Response(
                200, content=(FIXTURES / "rest" / "add-success.json").read_bytes()
            )
        if request.url.path == "/search":
            foreign = json.loads((FIXTURES / "rest" / "search-success.json").read_bytes())
            foreign["results"][0][changed_field] = changed_value
            return httpx.Response(
                200,
                content=json.dumps(foreign, separators=(",", ":")).encode(),
            )
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        nonlocal projection_call
        if request.url.path == "/health":
            return httpx.Response(
                200,
                content=b'{"status":"ok","mode":"read_only_projection"}',
            )
        if request.url.path == "/v1/projection":
            if projection_call == 0:
                projection_call += 1
                return httpx.Response(200, content=_empty_projection())
            page = projection_pages[1 if "cursor" in request.url.params else 0]
            projection_call += 1
            return httpx.Response(200, content=page)
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(_source(),))
    )[0]
    dispatch_receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
    )
    ingestion_receipt = IngestionReceipt(
        ingestion_occurrence_id=RUN_ID,
        accepted_source_unit_ids=(SOURCE_ID,),
        rejected_source_unit_ids=(),
        raw_references=(dispatch_receipt.raw_reference,),
        dispatch_receipts=(dispatch_receipt,),
    )
    await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=(SOURCE_ID,),
            ingestion_receipt=ingestion_receipt,
        )
    )

    with pytest.raises(MemorySystemCallFailure, match="search response"):
        await adapter.retrieve(
            RetrievalRequest(
                scope=scope,
                case_occurrence_id="e" * 64,
                query_bytes=b"Which tea?",
                top_k=150,
            )
        )
    await adapter.close()


@pytest.mark.asyncio
async def test_close_attempts_both_clients_and_retries_failed_transport_close() -> None:
    public_transport = CloseTrackingTransport(fail_first_close=True)
    inspector_transport = CloseTrackingTransport()
    adapter = _adapter(
        public_transport=public_transport,
        inspector_transport=inspector_transport,
    )

    with pytest.raises(RuntimeError, match="fixture close failure"):
        await adapter.close()
    await adapter.close()
    await adapter.close()

    assert public_transport.close_count == 2
    assert inspector_transport.close_count == 1


@pytest.mark.asyncio
async def test_cancelled_queued_add_releases_the_dispatch_guard_for_retry() -> None:
    active_started = asyncio.Event()
    release_active = asyncio.Event()
    add_source_ids: list[str] = []

    async def public_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        if request.url.path == "/memories":
            source_id = json.loads(request.content)["metadata"]["oamb_source_unit_id"]
            add_source_ids.append(source_id)
            if len(add_source_ids) == 1:
                active_started.set()
                await release_active.wait()
            return httpx.Response(200, content=b'{"results":[]}')
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                content=b'{"status":"ok","mode":"read_only_projection"}',
            )
        if request.url.path == "/v1/projection":
            return httpx.Response(200, content=_empty_projection())
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
    )
    second_source_id = "e" * 64
    first_dispatch, second_dispatch = adapter.plan_ingestion(
        IngestionRequest(
            scope=scope,
            ordered_source_units=(
                _source(),
                _source(second_source_id, 2),
            ),
        )
    )
    active = asyncio.create_task(
        adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="1" * 64,
                dispatch=first_dispatch,
            )
        )
    )
    await active_started.wait()
    queued = asyncio.create_task(
        adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="2" * 64,
                dispatch=second_dispatch,
            )
        )
    )
    await asyncio.sleep(0)
    queued.cancel()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await queued
    assert isinstance(cancelled.value, MemorySystemCallCancelledBeforeDispatch)

    release_active.set()
    await active
    retried = await adapter.ingest(
        IngestionDispatchRequest(
            scope=scope,
            attempt_id="3" * 64,
            dispatch=second_dispatch,
        )
    )

    assert retried.accepted_source_unit_ids == (second_source_id,)
    assert add_source_ids == [SOURCE_ID, second_source_id]
    await adapter.close()


@pytest.mark.asyncio
async def test_add_rejects_out_of_order_source_before_dispatch() -> None:
    add_source_ids: list[str] = []

    async def public_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        if request.url.path == "/memories":
            add_source_ids.append(json.loads(request.content)["metadata"]["oamb_source_unit_id"])
            return httpx.Response(200, content=b'{"results":[]}')
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                content=b'{"status":"ok","mode":"read_only_projection"}',
            )
        if request.url.path == "/v1/projection":
            return httpx.Response(200, content=_empty_projection())
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
    )
    second_source_id = "e" * 64
    first_dispatch, second_dispatch = adapter.plan_ingestion(
        IngestionRequest(
            scope=scope,
            ordered_source_units=(_source(), _source(second_source_id, 2)),
        )
    )

    with pytest.raises(ValueError, match="next ordered source"):
        await adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="2" * 64,
                dispatch=second_dispatch,
            )
        )
    assert add_source_ids == []

    await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="1" * 64, dispatch=first_dispatch)
    )
    await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="3" * 64, dispatch=second_dispatch)
    )
    assert add_source_ids == [SOURCE_ID, second_source_id]
    await adapter.close()


@pytest.mark.asyncio
async def test_adds_from_independent_ingestion_plans_can_dispatch_in_parallel() -> None:
    both_adds_started = asyncio.Event()
    release_adds = asyncio.Event()
    add_run_ids: list[str] = []

    async def public_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        if request.url.path == "/memories":
            add_run_ids.append(json.loads(request.content)["run_id"])
            if len(add_run_ids) == 2:
                both_adds_started.set()
            await release_adds.wait()
            return httpx.Response(200, content=b'{"results":[]}')
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                content=b'{"status":"ok","mode":"read_only_projection"}',
            )
        if request.url.path == "/v1/projection":
            return httpx.Response(
                200,
                json={
                    "collection": "oamb_memories",
                    "run_id": request.url.params["run_id"],
                    "count": 0,
                    "points": [],
                    "next_cursor": None,
                },
            )
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
    )
    await adapter.resolve()
    second_run_id = "e" * 64
    scopes = [
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
        ),
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id=second_run_id,
                ingestion_plan_id="d" * 64,
            )
        ),
    ]
    dispatches = [
        adapter.plan_ingestion(IngestionRequest(scope=scope, ordered_source_units=(_source(),)))[0]
        for scope in scopes
    ]
    tasks = [
        asyncio.create_task(
            adapter.ingest(
                IngestionDispatchRequest(
                    scope=scope,
                    attempt_id=str(index) * 64,
                    dispatch=dispatch,
                )
            )
        )
        for index, (scope, dispatch) in enumerate(zip(scopes, dispatches, strict=True), start=1)
    ]

    try:
        await asyncio.wait_for(both_adds_started.wait(), timeout=1.0)
    finally:
        release_adds.set()

    receipts = await asyncio.gather(*tasks)
    assert [receipt.accepted_source_unit_ids for receipt in receipts] == [
        (SOURCE_ID,),
        (SOURCE_ID,),
    ]
    assert set(add_run_ids) == {RUN_ID, second_run_id}
    await adapter.close()


@pytest.mark.asyncio
async def test_projection_capture_sequences_are_independent_per_scope() -> None:
    store = CapturingStore()

    def public_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                content=b'{"status":"ok","mode":"read_only_projection"}',
            )
        if request.url.path == "/v1/projection":
            return httpx.Response(
                200,
                json={
                    "collection": "oamb_memories",
                    "run_id": request.url.params["run_id"],
                    "count": 0,
                    "points": [],
                    "next_cursor": None,
                },
            )
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
        store=store,
    )
    await adapter.resolve()
    second_run_id = "e" * 64
    await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
    )
    await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=second_run_id, ingestion_plan_id="d" * 64)
    )

    summaries = [
        document
        for request in store.raw
        if (document := json.loads(request.payload_bytes)).get("schema")
        == "oamb-mem0-projection-receipt-v1"
    ]
    assert [(item["run_id"], item["capture_sequence"]) for item in summaries] == [
        (RUN_ID, 1),
        (second_run_id, 1),
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_close_arms_adapter_gate_before_waiting_for_client_drain() -> None:
    class BlockingCloseTransport(CloseTrackingTransport):
        def __init__(self) -> None:
            super().__init__()
            self.close_started = asyncio.Event()
            self.release_close = asyncio.Event()

        async def aclose(self) -> None:
            self.close_count += 1
            self.close_started.set()
            await self.release_close.wait()

    public_transport = BlockingCloseTransport()
    inspector_transport = CloseTrackingTransport()
    adapter = _adapter(
        public_transport=public_transport,
        inspector_transport=inspector_transport,
    )
    closing = asyncio.create_task(adapter.close())
    await public_transport.close_started.wait()

    with pytest.raises(RuntimeError, match="closed"):
        await adapter.resolve()
    with pytest.raises(RuntimeError, match="closed"):
        adapter.plan_ingestion(
            IngestionRequest(
                scope=ScopeReceipt(
                    ingestion_occurrence_id=RUN_ID,
                    scope_id=RUN_ID,
                    raw_reference=RawReferenceHandle("d" * 64),
                ),
                ordered_source_units=(_source(),),
            )
        )

    public_transport.release_close.set()
    await closing
    assert public_transport.close_count == 1
    assert inspector_transport.close_count == 1


@pytest.mark.asyncio
async def test_projection_page_failure_links_preceding_sealed_pages() -> None:
    page_1 = (FIXTURES / "inspector" / "page-1.json").read_bytes()
    page_1_sha256 = hashlib.sha256(page_1).hexdigest()

    def public_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        raise AssertionError(f"unexpected public request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                content=b'{"status":"ok","mode":"read_only_projection"}',
            )
        if request.url.path == "/v1/projection" and "cursor" not in request.url.params:
            return httpx.Response(200, content=page_1)
        if request.url.path == "/v1/projection":
            return httpx.Response(502, content=b'{"error":"fixture page failure"}')
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    adapter = _adapter(
        public_transport=httpx.MockTransport(public_handler),
        inspector_transport=httpx.MockTransport(inspector_handler),
    )
    await adapter.resolve()

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(ingestion_occurrence_id=RUN_ID, ingestion_plan_id=PLAN_ID)
        )

    assert tuple(item.sha256 for item in failure.value.supporting_raw_references) == (
        page_1_sha256,
    )
    await adapter.close()
