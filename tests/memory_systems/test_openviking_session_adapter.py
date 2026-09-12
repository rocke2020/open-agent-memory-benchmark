from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from oamb.contracts.ports import (
    ArtifactReadRequest,
    ArtifactSealReceipt,
    ArtifactWriteRequest,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionReceipt,
    IngestionRequest,
    MemorySystemCallFailure,
    MemorySystemCallUnknownOutcome,
    RawPayloadSealRequest,
    RawReferenceHandle,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
    SourceUnit,
)

INGESTION_OCCURRENCE_ID = "a" * 64
INGESTION_PLAN_ID = "lme-plan-1"
BENCHMARK_USER = "oamb-admin"
EXPECTED_OCCURRENCE_USER = "oamb-df9280ae264f68b9ea2866df3312890168543ddb41eb5496fbcea95e64536818"
EXPECTED_OCCURRENCE_USER_KEY = (
    "b2FtYi1iZW5jaG1hcms."
    "b2FtYi1kZjkyODBhZTI2NGY2OGI5ZWEyODY2ZGYzMzEyODkwMTY4NTQzZGRiNDFlYjU0OTZmYmNlYTk1ZTY0NTM2ODE4."
    "ODRhY2FlYjJjY2I5MGU0NDRjZTE5YTFlYzUxMmU2YzZiNDNmM2IzM2UyZDhhZjBhYzAwOWQ1Y2E0NDQzZjExMQ"
)
EXPECTED_OCCURRENCE_USER_SEED = "68bc2689a37d0a19749291d845372ef46dcbf6af9f39644e716634eb31f8ed09"
EXPECTED_USER_ROOT = f"viking://user/{EXPECTED_OCCURRENCE_USER}"
EXPECTED_USER_MEMORY_ROOT = f"viking://user/{EXPECTED_OCCURRENCE_USER}/memories"
EXPECTED_MEMORY_ROOT = EXPECTED_USER_MEMORY_ROOT
EXPECTED_PRISTINE_MEMORY_URIS = (
    f"{EXPECTED_MEMORY_ROOT}/.abstract.md",
    f"{EXPECTED_MEMORY_ROOT}/.overview.md",
)
CANONICAL_TIME = "2024-02-28T01:02:00+00:00"


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


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def _ok(result: object, *, compact: bool = False) -> bytes:
    if compact:
        return _json_bytes({"status": "ok", "result": result})
    return _json_bytes(
        {
            "status": "ok",
            "result": result,
            "error": None,
            "telemetry": None,
            "profile": None,
        }
    )


def _not_found(resource: str, resource_type: str | None) -> bytes:
    error: dict[str, object] = {
        "code": "NOT_FOUND",
        "message": (
            f"Session {resource} not found"
            if resource_type is None
            else f"{resource_type.capitalize()} not found: {resource}"
        ),
    }
    if resource_type is not None:
        error["details"] = {"resource": resource, "type": resource_type}
    return _json_bytes(
        {
            "status": "error",
            "error": error,
        }
    )


def _health() -> bytes:
    return _json_bytes(
        {
            "status": "ok",
            "healthy": True,
            "version": "v0.4.19",
            "auth_mode": "api_key",
            "account_id": "oamb-benchmark",
            "user_id": BENCHMARK_USER,
            "role": "admin",
        }
    )


def _messages(count: int) -> list[dict[str, str]]:
    return [
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": f"message-{index + 1}",
        }
        for index in range(count)
    ]


def _source(source_id: str, ordinal: int, messages: list[dict[str, str]]) -> SourceUnit:
    payload = _json_bytes(messages)
    return SourceUnit(
        source_unit_id=source_id,
        context_manifest_entry_id="context-1",
        ordinal_1_indexed=ordinal,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=payload,
        source_reference=f"source-session-{ordinal}",
        occurred_at=CANONICAL_TIME,
        source_metadata=(("session_id", f"source-session-{ordinal}"),),
    )


def _session_id(source: SourceUnit) -> str:
    from oamb.contracts.ids import openviking_session_id

    return openviking_session_id(INGESTION_OCCURRENCE_ID, source.source_unit_id)


def test_session_id_is_namespaced_by_ingestion_occurrence() -> None:
    from oamb.contracts.ids import openviking_session_id

    assert openviking_session_id("run-a", "source-1") != openviking_session_id("run-b", "source-1")


@dataclass
class SessionService:
    sources: tuple[SourceUnit, ...]
    question_user_exists: bool = False
    memory_populated: bool = False
    question_health_user: str = EXPECTED_OCCURRENCE_USER
    existing_session_ids: frozenset[str] = frozenset()
    empty_projection: bool = False
    task_never_completes: bool = False
    continuation_tasks: dict[str, dict[str, Any]] | None = None
    existing_session_details: dict[str, dict[str, Any]] | None = None

    def __post_init__(self) -> None:
        self.calls: list[httpx.Request] = []
        self.task_polls: dict[str, int] = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        path = request.url.path
        if path == "/health":
            if request.headers.get("X-API-Key") == "user-secret":
                return httpx.Response(200, content=_health())
            return httpx.Response(
                200,
                content=_json_bytes(
                    {
                        "status": "ok",
                        "healthy": True,
                        "version": "v0.4.19",
                        "auth_mode": "api_key",
                        "account_id": "oamb-benchmark",
                        "user_id": self.question_health_user,
                        "role": "user",
                    }
                ),
            )
        if path == "/api/v1/admin/accounts/oamb-benchmark/users":
            if request.method == "GET":
                users = (
                    [
                        {
                            "user_id": EXPECTED_OCCURRENCE_USER,
                            "role": "user",
                            "api_key": "listed-secret",
                        }
                    ]
                    if self.question_user_exists
                    else []
                )
                return httpx.Response(200, content=_ok(users))
            self.question_user_exists = True
            return httpx.Response(
                200,
                content=_ok(
                    {
                        "account_id": "oamb-benchmark",
                        "user_id": EXPECTED_OCCURRENCE_USER,
                        "user_key": "registered-secret",
                    }
                ),
            )
        if path.startswith("/api/v1/sessions/") and path.count("/") == 4:
            session_id = path.rsplit("/", 1)[-1]
            if session_id in self.existing_session_ids:
                details = (self.existing_session_details or {}).get(
                    session_id,
                    {"session_id": session_id},
                )
                return httpx.Response(200, content=_ok(details))
            return httpx.Response(404, content=_not_found(session_id, None))
        if path == "/api/v1/sessions" and request.method == "POST":
            body = json.loads(request.content)
            session_id = body["session_id"]
            return httpx.Response(
                200,
                content=_ok(
                    {
                        "session_id": session_id,
                        "uri": f"{EXPECTED_USER_ROOT}/sessions/{session_id}",
                        "user": {"user_id": EXPECTED_OCCURRENCE_USER},
                        "auto_commit_policy": None,
                        "memory_extraction_config": {},
                    }
                ),
            )
        if path.endswith("/messages/batch"):
            session_id = path.split("/")[4]
            body = json.loads(request.content)
            return httpx.Response(
                200,
                content=_ok(
                    {
                        "session_id": session_id,
                        "message_count": len(body["messages"]),
                        "pending_tokens": len(body["messages"]),
                    }
                ),
            )
        if path.endswith("/commit"):
            session_id = path.split("/")[4]
            task_id = f"task-{session_id}"
            archive_uri = f"{EXPECTED_USER_ROOT}/sessions/{session_id}/history/archive_001"
            return httpx.Response(
                200,
                content=_ok(
                    {
                        "session_id": session_id,
                        "status": "accepted",
                        "task_id": task_id,
                        "archive_uri": archive_uri,
                        "archived": True,
                        "trace_id": f"trace-{session_id}",
                        "estimated_active_tokens": 0,
                        "budget_exceeded": False,
                    },
                    compact=True,
                ),
            )
        if path.startswith("/api/v1/tasks/"):
            task_id = path.rsplit("/", 1)[-1]
            if task_id in (self.continuation_tasks or {}):
                return httpx.Response(
                    200,
                    content=_ok((self.continuation_tasks or {})[task_id]),
                )
            poll_count = self.task_polls.get(task_id, 0) + 1
            self.task_polls[task_id] = poll_count
            session_id = task_id.removeprefix("task-")
            status = "running" if self.task_never_completes or poll_count == 1 else "completed"
            result = None
            if status == "completed":
                self.memory_populated = True
                result = {
                    "session_id": session_id,
                    "archive_uri": (
                        f"{EXPECTED_USER_ROOT}/sessions/{session_id}/history/archive_001"
                    ),
                    "memories_extracted": 1,
                    "usage_events_extracted": 0,
                    "token_usage": {
                        "llm": {"prompt_tokens": 11, "completion_tokens": 3},
                        "embedding": {"total_tokens": 7},
                        "total": {
                            "total_tokens": 21,
                            "cached_tokens": 0,
                            "reasoning_tokens": 0,
                        },
                    },
                }
            return httpx.Response(
                200,
                content=_ok(
                    {
                        "task_id": task_id,
                        "task_type": "session_commit",
                        "status": status,
                        "resource_id": session_id,
                        "meta": {},
                        "stage": "memory_extraction",
                        "result": result,
                        "error": None,
                    }
                ),
            )
        if "/archives/" in path:
            session_id = path.split("/")[4]
            return httpx.Response(
                200,
                content=_ok(
                    {
                        "archive_id": "archive_001",
                        "session_id": session_id,
                        "uri": (f"{EXPECTED_USER_ROOT}/sessions/{session_id}/history/archive_001"),
                        "message_count": len(self.sources[0].payload_bytes),
                    }
                ),
            )
        if path == "/api/v1/fs/ls":
            if self.empty_projection or not self.memory_populated:
                return httpx.Response(200, content=_ok(list(EXPECTED_PRISTINE_MEMORY_URIS)))
            return httpx.Response(
                200,
                content=_ok(
                    [
                        *EXPECTED_PRISTINE_MEMORY_URIS,
                        f"{EXPECTED_MEMORY_ROOT}/events/event-1.md",
                        f"{EXPECTED_MEMORY_ROOT}/preferences/preference-1.md",
                    ]
                ),
            )
        if path == "/api/v1/search/find":
            return httpx.Response(
                200,
                content=_ok(
                    {
                        "memories": [
                            {
                                "context_type": "memory",
                                "uri": f"{EXPECTED_MEMORY_ROOT}/events/event-1.md",
                                "level": 2,
                                "score": 0.9,
                                "abstract": "The user likes coffee.",
                                "tags": [],
                            }
                        ],
                        "resources": [],
                        "skills": [],
                        "total": 1,
                    },
                    compact=True,
                ),
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")


def _adapter(service: SessionService, store: CapturingStore | None = None) -> Any:
    from oamb.memory_systems.openviking import OpenVikingSessionAdapter

    return OpenVikingSessionAdapter(
        store=store or CapturingStore(),
        base_url="https://openviking.example",
        api_key="user-secret",
        benchmark_account="oamb-benchmark",
        benchmark_user=BENCHMARK_USER,
        runtime_binding_hash="f" * 64,
        transport=httpx.MockTransport(service),
        task_poll_interval_seconds=0,
        maximum_task_polls=3,
    )


def test_task_poll_limit_matches_the_outer_call_timeout() -> None:
    from oamb.memory_systems.openviking.session_adapter import (
        maximum_task_polls_for_timeout,
    )

    assert maximum_task_polls_for_timeout(900, poll_interval_seconds=0.1) == 9_000


@pytest.mark.asyncio
async def test_scope_provisions_one_occurrence_user_and_uses_its_memory_root() -> None:
    from oamb.memory_systems.openviking import OpenVikingSessionAdapter
    from oamb.memory_systems.openviking.adapter import OPENVIKING_VERSION

    calls: list[httpx.Request] = []
    store = CapturingStore()

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        api_key = request.headers.get("X-API-Key")
        if request.url.path == "/health":
            user_id = BENCHMARK_USER if api_key == "user-secret" else EXPECTED_OCCURRENCE_USER
            role = "admin" if api_key == "user-secret" else "user"
            return httpx.Response(
                200,
                content=_json_bytes(
                    {
                        "status": "ok",
                        "healthy": True,
                        "version": OPENVIKING_VERSION,
                        "auth_mode": "api_key",
                        "account_id": "oamb-benchmark",
                        "user_id": user_id,
                        "role": role,
                    }
                ),
            )
        if request.url.path == "/api/v1/admin/accounts/oamb-benchmark/users":
            if request.method == "GET":
                return httpx.Response(200, content=_ok([]))
            return httpx.Response(
                200,
                content=_ok(
                    {
                        "account_id": "oamb-benchmark",
                        "user_id": EXPECTED_OCCURRENCE_USER,
                        "user_key": "server-returned-secret",
                        "seed": "server-echoed-secret",
                    }
                ),
            )
        if request.url.path == "/api/v1/fs/ls":
            return httpx.Response(
                200,
                content=_ok(list(EXPECTED_PRISTINE_MEMORY_URIS)),
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = OpenVikingSessionAdapter(
        store=store,
        base_url="https://openviking.example",
        api_key="user-secret",
        benchmark_account="oamb-benchmark",
        benchmark_user=BENCHMARK_USER,
        runtime_binding_hash="f" * 64,
        transport=httpx.MockTransport(handler),
        task_poll_interval_seconds=0,
        maximum_task_polls=3,
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )

    assert scope.scope_id == EXPECTED_USER_MEMORY_ROOT
    list_call, register_call = calls[1:3]
    assert list_call.method == "GET"
    assert list_call.url.params["name"] == EXPECTED_OCCURRENCE_USER
    assert _request_payload(register_call) == {
        "user_id": EXPECTED_OCCURRENCE_USER,
        "role": "user",
        "seed": EXPECTED_OCCURRENCE_USER_SEED,
    }
    assert calls[3].headers["X-API-Key"] == EXPECTED_OCCURRENCE_USER_KEY
    assert calls[4].headers["X-API-Key"] == EXPECTED_OCCURRENCE_USER_KEY
    assert all("X-OpenViking-Actor-Peer" not in request.headers for request in calls)
    assert scope.raw_reference == RawReferenceHandle(store.raw[4].sha256)
    assert scope.supporting_raw_references == tuple(
        RawReferenceHandle(item.sha256) for item in store.raw[1:4]
    )
    sealed = b"\n".join(item.payload_bytes for item in store.raw)
    assert b"server-returned-secret" not in sealed
    assert b"server-echoed-secret" not in sealed
    assert b"user_key" not in sealed
    assert b'"seed"' not in sealed
    await adapter.close()


@pytest.mark.asyncio
async def test_existing_question_user_with_memory_is_not_reused_as_a_fresh_scope() -> None:
    from oamb.memory_systems.openviking import OpenVikingSessionProfileError

    source = _source("source-1", 1, _messages(1))
    service = SessionService(
        (source,),
        question_user_exists=True,
        memory_populated=True,
    )
    adapter = _adapter(service)
    await adapter.resolve()

    with pytest.raises(OpenVikingSessionProfileError, match="pristine"):
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
        )

    assert not any(request.method == "POST" for request in service.calls)
    await adapter.close()


@pytest.mark.asyncio
async def test_existing_question_user_with_the_wrong_derived_key_fails_closed() -> None:
    service = SessionService(
        (),
        question_user_exists=True,
        question_health_user="collision-user",
    )
    adapter = _adapter(service)
    await adapter.resolve()

    with pytest.raises(
        MemorySystemCallFailure,
        match="question-user identity probe failed validation",
    ):
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
        )

    assert not any(request.method == "POST" for request in service.calls)
    assert not any(request.url.path == "/api/v1/fs/ls" for request in service.calls)
    await adapter.close()


@pytest.mark.asyncio
async def test_task_polling_has_one_total_deadline_instead_of_per_poll_timeout() -> None:
    from oamb.memory_systems.openviking import OpenVikingSessionAdapter

    source = _source("source-timeout", 1, _messages(1))
    service = SessionService((source,), task_never_completes=True)
    adapter = OpenVikingSessionAdapter(
        store=CapturingStore(),
        base_url="https://openviking.example",
        api_key="user-secret",
        benchmark_account="oamb-benchmark",
        benchmark_user=BENCHMARK_USER,
        runtime_binding_hash="f" * 64,
        transport=httpx.MockTransport(service),
        task_poll_interval_seconds=0.001,
        maximum_task_polls=10_000,
        task_poll_timeout_seconds=0.01,
    )
    scope = await _resolved_scope(adapter)
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]

    with pytest.raises(
        MemorySystemCallUnknownOutcome,
        match="task did not become terminal",
    ):
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="b" * 64, dispatch=dispatch)
        )

    assert 1 <= sum(service.task_polls.values()) < 10_000
    await adapter.close()


async def _resolved_scope(adapter: Any) -> Any:
    await adapter.resolve()
    return await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
    )


def _ingestion_receipt(
    scope: Any,
    receipts: tuple[IngestionDispatchReceipt, ...],
) -> IngestionReceipt:
    return IngestionReceipt(
        ingestion_occurrence_id=scope.ingestion_occurrence_id,
        accepted_source_unit_ids=tuple(
            source_id for receipt in receipts for source_id in receipt.accepted_source_unit_ids
        ),
        rejected_source_unit_ids=tuple(
            source_id for receipt in receipts for source_id in receipt.rejected_source_unit_ids
        ),
        raw_references=tuple(receipt.raw_reference for receipt in receipts),
        dispatch_receipts=receipts,
    )


def _request_payload(request: httpx.Request) -> dict[str, Any]:
    value = json.loads(request.content)
    if not isinstance(value, dict):
        raise AssertionError("request payload is not an object")
    return value


@pytest.mark.asyncio
async def test_plan_creates_one_ordered_native_session_dispatch_per_source_unit() -> None:
    sources = (
        _source("source-1", 1, _messages(2)),
        _source("source-2", 2, _messages(3)),
    )
    adapter = _adapter(SessionService(sources))
    scope = await _resolved_scope(adapter)

    dispatches = adapter.plan_ingestion(IngestionRequest(scope, sources))

    assert tuple(dispatch.dispatch_ordinal_1_indexed for dispatch in dispatches) == (1, 2)
    assert tuple(dispatch.operation_kind for dispatch in dispatches) == (
        "openviking_session_commit",
        "openviking_session_commit",
    )
    assert tuple(dispatch.ordered_source_units for dispatch in dispatches) == (
        (sources[0],),
        (sources[1],),
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_ingest_disables_auto_commit_preserves_messages_and_closes_commit_evidence() -> None:
    source = _source("source-1", 1, _messages(2))
    service = SessionService((source,))
    store = CapturingStore()
    adapter = _adapter(service, store)
    scope = await _resolved_scope(adapter)
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]

    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="b" * 64, dispatch=dispatch)
    )

    session_id = _session_id(source)
    paths = [request.url.path for request in service.calls]
    session_get = paths.index(f"/api/v1/sessions/{session_id}")
    session_create = paths.index("/api/v1/sessions")
    messages = paths.index(f"/api/v1/sessions/{session_id}/messages/batch")
    commit = paths.index(f"/api/v1/sessions/{session_id}/commit")
    first_task = paths.index(f"/api/v1/tasks/task-{session_id}")
    last_task = len(paths) - 1 - paths[::-1].index(f"/api/v1/tasks/task-{session_id}")
    archive = paths.index(f"/api/v1/sessions/{session_id}/archives/archive_001")
    projection = len(paths) - 1 - paths[::-1].index("/api/v1/fs/ls")
    assert (
        session_get
        < session_create
        < messages
        < commit
        < first_task
        < last_task
        < archive
        < projection
    )

    create_call = service.calls[session_create]
    assert _request_payload(create_call) == {
        "session_id": session_id,
        "auto_commit_policy": None,
    }
    message_call = service.calls[messages]
    assert _request_payload(message_call) == {
        "messages": [
            {
                "role": "user",
                "content": "message-1",
                "created_at": CANONICAL_TIME,
            },
            {
                "role": "assistant",
                "content": "message-2",
                "created_at": CANONICAL_TIME,
            },
        ]
    }
    assert _request_payload(service.calls[commit]) == {"keep_recent_count": 0}
    assert service.task_polls[f"task-{session_id}"] == 2
    assert receipt.accepted_source_unit_ids == (source.source_unit_id,)
    assert receipt.rejected_source_unit_ids == ()
    assert receipt.raw_response_bytes == store.raw[-1].payload_bytes
    assert all(request.method != "DELETE" for request in service.calls)
    message_request_proofs = [
        item.payload_bytes
        for item in store.raw
        if b'"path":"/api/v1/sessions/' in item.payload_bytes
        and b'/messages/batch"' in item.payload_bytes
    ]
    assert len(message_request_proofs) == 1
    assert b'"peer_id"' not in message_request_proofs[0]
    assert b'"request_header_names":[]' in message_request_proofs[0]
    assert len(store.raw) == len(service.calls) + 1
    await adapter.close()


@pytest.mark.asyncio
async def test_message_batches_split_only_above_provider_limit_without_reordering() -> None:
    source = _source("source-101", 1, _messages(101))
    service = SessionService((source,))
    adapter = _adapter(service)
    scope = await _resolved_scope(adapter)
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]

    await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="b" * 64, dispatch=dispatch)
    )

    message_calls = [
        request for request in service.calls if request.url.path.endswith("/messages/batch")
    ]
    payloads = [_request_payload(request)["messages"] for request in message_calls]
    assert tuple(len(payload) for payload in payloads) == (100, 1)
    assert [item["content"] for payload in payloads for item in payload] == [
        f"message-{index}" for index in range(1, 102)
    ]
    assert all(item["created_at"] == CANONICAL_TIME for payload in payloads for item in payload)
    await adapter.close()


@pytest.mark.asyncio
async def test_ready_requires_every_planned_session_commit_and_retrieval_never_writes_question() -> (
    None
):
    sources = (
        _source("source-1", 1, _messages(1)),
        _source("source-2", 2, _messages(1)),
    )
    service = SessionService(sources)
    adapter = _adapter(service)
    scope = await _resolved_scope(adapter)
    dispatches = adapter.plan_ingestion(IngestionRequest(scope, sources))
    first = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="b" * 64, dispatch=dispatches[0])
    )

    partial = _ingestion_receipt(scope, (first,))
    readiness = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=tuple(source.source_unit_id for source in sources),
            ingestion_receipt=partial,
        )
    )
    assert readiness.ready is False
    with pytest.raises(Exception, match="ready"):
        await adapter.retrieve(RetrievalRequest(scope, "c" * 64, b"What coffee do I like?", 150))

    second = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatches[1])
    )
    complete = _ingestion_receipt(scope, (first, second))
    readiness = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=tuple(source.source_unit_id for source in sources),
            ingestion_receipt=complete,
        )
    )
    assert readiness.ready is True

    history_write_count = sum(
        request.url.path.endswith("/messages/batch") for request in service.calls
    )
    evidence = await adapter.retrieve(
        RetrievalRequest(scope, "d" * 64, b"What coffee do I like?", 150)
    )
    capabilities = await adapter.capabilities()
    find_call = service.calls[-1]
    assert find_call.url.path == "/api/v1/search/find"
    assert _request_payload(find_call) == {
        "query": "What coffee do I like?",
        "target_uri": EXPECTED_MEMORY_ROOT,
        "context_type": "memory",
        "limit": 150,
    }
    assert "intent-free-user-memory-find" in capabilities.capability_ids
    assert (
        sum(request.url.path.endswith("/messages/batch") for request in service.calls)
        == history_write_count
    )
    assert tuple(candidate.content for candidate in evidence.candidates) == (
        "The user likes coffee.",
    )
    assert all(request.method != "DELETE" for request in service.calls)
    await adapter.close()


@pytest.mark.asyncio
async def test_existing_question_user_is_reused_without_rotation_and_session_is_never_reused() -> (
    None
):
    source = _source("source-1", 1, _messages(1))
    user_service = SessionService((source,), question_user_exists=True)
    user_adapter = _adapter(user_service)
    scope = await _resolved_scope(user_adapter)
    calls_after_first_allocation = len(user_service.calls)
    with pytest.raises(Exception, match="already exists|create-only"):
        await user_adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID)
        )
    assert len(user_service.calls) == calls_after_first_allocation
    assert not any(
        request.method == "POST"
        and request.url.path == "/api/v1/admin/accounts/oamb-benchmark/users"
        for request in user_service.calls
    )
    assert scope.scope_id == EXPECTED_MEMORY_ROOT
    await user_adapter.close()

    session_id = _session_id(source)
    session_service = SessionService((source,), existing_session_ids=frozenset({session_id}))
    session_adapter = _adapter(session_service)
    scope = await _resolved_scope(session_adapter)
    dispatch = session_adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]
    with pytest.raises(Exception, match="already exists|create-only"):
        await session_adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="b" * 64, dispatch=dispatch)
        )
    calls_after_first_attempt = len(session_service.calls)
    with pytest.raises(Exception, match="already.*attempted|cannot be reused"):
        await session_adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="c" * 64, dispatch=dispatch)
        )
    assert len(session_service.calls) == calls_after_first_attempt
    assert not any(
        request.method == "POST" and request.url.path.startswith("/api/v1/sessions")
        for request in session_service.calls
    )
    assert all(request.method != "DELETE" for request in session_service.calls)
    await session_adapter.close()


@pytest.mark.asyncio
async def test_adoption_rejects_zero_counters_without_task_no_mutation_proof() -> None:
    sources = tuple(_source(f"source-{ordinal}", ordinal, _messages(1)) for ordinal in range(1, 4))
    session_ids = tuple(_session_id(source) for source in sources)
    completed_task_ids = ("completed-task-1", "completed-task-2")
    failed_task_id = "failed-task-3"
    continuation_tasks = {
        task_id: {
            "task_id": task_id,
            "task_type": "session_commit",
            "status": "completed",
            "resource_id": session_id,
            "meta": {},
            "stage": "completed",
            "result": {
                "session_id": session_id,
                "archive_uri": (f"{EXPECTED_USER_ROOT}/sessions/{session_id}/history/archive_001"),
                "memories_extracted": 1,
                "usage_events_extracted": 0,
                "token_usage": {
                    "llm": {"prompt_tokens": 11, "completion_tokens": 3},
                    "embedding": {"total_tokens": 7},
                    "total": {
                        "total_tokens": 21,
                        "cached_tokens": 0,
                        "reasoning_tokens": 0,
                    },
                },
            },
            "error": None,
        }
        for task_id, session_id in zip(completed_task_ids, session_ids[:2], strict=True)
    }
    continuation_tasks[failed_task_id] = {
        "task_id": failed_task_id,
        "task_type": "session_commit",
        "status": "failed",
        "resource_id": session_ids[2],
        "meta": {},
        "stage": "memory_extraction",
        "result": None,
        "error": "embedding server stopped before extraction",
    }
    service = SessionService(
        sources,
        question_user_exists=True,
        existing_session_ids=frozenset(session_ids),
        continuation_tasks=continuation_tasks,
        existing_session_details={
            session_ids[2]: {
                "session_id": session_ids[2],
                "message_count": 0,
                "total_message_count": 1,
                "commit_count": 1,
                "pending_tokens": 0,
                "memories_extracted": {"total": 0},
                "llm_token_usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                },
                "embedding_token_usage": {"total_tokens": 0},
            }
        },
    )
    adapter = _adapter(service)
    await adapter.resolve()

    with pytest.raises(Exception, match="no-mutation.*proof"):
        await adapter.adopt_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID),
            completed_session_task_ids=completed_task_ids,
            failed_session_task_id=failed_task_id,
        )
    assert all(request.method in {"GET", "HEAD"} for request in service.calls)
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_status", "memory_total"),
    (("running", 0), ("failed", 1)),
)
async def test_adoption_never_replays_unknown_or_partial_failed_session(
    failed_status: str,
    memory_total: int,
) -> None:
    source = _source("source-1", 1, _messages(1))
    session_id = _session_id(source)
    failed_task_id = "failed-task-1"
    service = SessionService(
        (source,),
        question_user_exists=True,
        existing_session_ids=frozenset({session_id}),
        continuation_tasks={
            failed_task_id: {
                "task_id": failed_task_id,
                "task_type": "session_commit",
                "status": failed_status,
                "resource_id": session_id,
                "meta": {},
                "stage": "memory_extraction",
                "result": None,
                "error": "provider failed" if failed_status == "failed" else None,
            }
        },
        existing_session_details={
            session_id: {
                "session_id": session_id,
                "message_count": 0,
                "total_message_count": 1,
                "commit_count": 1,
                "pending_tokens": 0,
                "memories_extracted": {"total": memory_total},
                "llm_token_usage": {"total_tokens": 0},
                "embedding_token_usage": {"total_tokens": 0},
            }
        },
    )
    adapter = _adapter(service)
    await adapter.resolve()

    with pytest.raises(Exception, match="terminal failed|zero memory and usage"):
        await adapter.adopt_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID),
            completed_session_task_ids=(),
            failed_session_task_id=failed_task_id,
        )

    assert all(request.method in {"GET", "HEAD"} for request in service.calls)
    await adapter.close()


@pytest.mark.asyncio
async def test_adoption_rejects_partial_memory_even_when_session_counters_are_zero() -> None:
    source = _source("source-1", 1, _messages(1))
    session_id = _session_id(source)
    failed_task_id = "failed-task-1"
    session_details = {
        session_id: {
            "session_id": session_id,
            "message_count": 0,
            "total_message_count": 1,
            "commit_count": 1,
            "pending_tokens": 0,
            "memories_extracted": {"total": 0},
            "llm_token_usage": {"total_tokens": 0},
            "embedding_token_usage": {"total_tokens": 0},
        }
    }
    service = SessionService(
        (source,),
        question_user_exists=True,
        existing_session_ids=frozenset({session_id}),
        continuation_tasks={
            failed_task_id: {
                "task_id": failed_task_id,
                "task_type": "session_commit",
                "status": "failed",
                "resource_id": session_id,
                "meta": {},
                "stage": "memory_diff_publication",
                "result": None,
                "error": "provider failed",
            }
        },
        existing_session_details=session_details,
    )
    adapter = _adapter(service)
    await adapter.resolve()
    # The peer has memory, but failed-task counters have not merged into the session.
    assert service.question_user_exists and not service.empty_projection
    with pytest.raises(Exception, match="no-mutation.*proof"):
        await adapter.adopt_ingestion_scope(
            ScopeAllocationRequest(INGESTION_OCCURRENCE_ID, INGESTION_PLAN_ID),
            completed_session_task_ids=(),
            failed_session_task_id=failed_task_id,
        )

    assert all(request.method in {"GET", "HEAD"} for request in service.calls)
    await adapter.close()


@pytest.mark.asyncio
async def test_completed_empty_peer_memory_projection_is_ready_quality_evidence() -> None:
    source = _source("source-empty", 1, _messages(1))
    service = SessionService((source,), empty_projection=True)
    adapter = _adapter(service)
    scope = await _resolved_scope(adapter)
    dispatch = adapter.plan_ingestion(IngestionRequest(scope, (source,)))[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="b" * 64, dispatch=dispatch)
    )

    readiness = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=(source.source_unit_id,),
            ingestion_receipt=_ingestion_receipt(scope, (receipt,)),
        )
    )

    assert readiness.ready is True
    assert (await adapter.inventory(scope)).ordered_source_unit_ids == (source.source_unit_id,)
    await adapter.close()
