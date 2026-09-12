from __future__ import annotations

import hashlib
import json
from typing import Any

import httpx
import pytest

from oamb.contracts.ports import (
    IngestionDispatchRequest,
    IngestionReceipt,
    IngestionRequest,
    MemorySystemCallFailure,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
)
from tests.memory_systems.test_hindsight_adapter import (
    _HindsightFixtureService,
)
from tests.memory_systems.test_hindsight_adapter import (
    _source as hindsight_source,
)
from tests.memory_systems.test_mem0_rest_blackbox_adapter import (
    CapturingStore,
    _openapi,
)
from tests.memory_systems.test_mem0_rest_blackbox_adapter import (
    _source as mem0_source,
)
from tests.memory_systems.test_openviking_session_adapter import (
    BENCHMARK_USER,
    INGESTION_OCCURRENCE_ID,
    SessionService,
    _messages,
    _ok,
)
from tests.memory_systems.test_openviking_session_adapter import (
    _source as openviking_source,
)

HINDSIGHT_BASIS = "hindsight_sync_extraction_drained_v1"
MEM0_BASIS = "mem0_sync_add_returned_v1"
OPENVIKING_BASIS = "openviking_task_work_drained_v1"
HINDSIGHT_DETAIL = (
    "Fact extraction failed: 1/1 chunks failed. "
    "First failures: chunk 0: APIConnectionError: Connection error."
)
HINDSIGHT_INVALID_JSON_DETAIL = (
    "Fact extraction failed: 1/1 chunks failed. "
    "First failures: chunk 0: JSONDecodeError: Invalid control character at: "
    "line 8 column 36 (char 243)"
)
HINDSIGHT_EMBEDDING_CONNECTION_DETAIL = "Failed to generate batch embeddings: Connection error."
TASK_ID = "11111111-1111-4111-8111-111111111111"
SESSION_ID = "oamb-native-session-fixture"


def _raw(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode()


def _task(**changes: object) -> dict[str, object]:
    result: dict[str, object] = {
        "task_id": TASK_ID,
        "task_type": "session_commit",
        "resource_id": SESSION_ID,
        "status": "failed",
        "stage": "failed",
        "error": "Connection error.",
        "result": None,
        "meta": {},
        "created_at": 1788664234.0,
        "updated_at": 1788664235.0,
        "created_at_iso": "2026-09-06T01:10:34+00:00",
        "updated_at_iso": "2026-09-06T01:10:35+00:00",
    }
    result.update(changes)
    return result


def _classify(basis: str, body: bytes, status: int, **options: Any) -> str | None:
    from oamb.contracts.ingestion_failures import classify_settled_ingestion_failure

    arguments = {
        "settlement_basis": basis,
        "raw_response_bytes": body,
        "status_code": status,
        "internal_retry_count": 10,
        "expected_task_id": TASK_ID,
        "expected_session_id": SESSION_ID,
        **options,
    }
    return classify_settled_ingestion_failure(**arguments)


def test_live_native_retry_proof_count_ten_is_accepted() -> None:
    assert (
        _classify(
            MEM0_BASIS,
            _raw(
                {
                    "detail": "Provider memory extraction failed after retries.",
                    "code": "provider_extraction_failed",
                    "request_id": "1234abcd",
                }
            ),
            502,
            internal_retry_count=10,
        )
        == "provider_ingestion_error"
    )


@pytest.mark.parametrize(
    "detail",
    (
        "Fact extraction failed: 2/3 chunks failed. First failures: "
        "chunk 0: JSONDecodeError: Expecting value: line 1 column 1 (char 0), "
        "chunk 2: RuntimeError: schema returned non-dict facts",
        "Fact extraction failed: 6/8 chunks failed. First failures: "
        "chunk 0: JSONDecodeError: bad, chunk 1: ValidationError: schema, "
        "chunk 2: RuntimeError: non-dict, chunk 3: ValueError: missing facts, "
        "chunk 4: RuntimeError: unusable facts",
    ),
)
def test_hindsight_terminal_mixed_extraction_aggregate_is_settled(detail: str) -> None:
    assert _classify(HINDSIGHT_BASIS, _raw({"detail": detail}), 500) == "provider_ingestion_error"


@pytest.mark.parametrize(
    ("detail", "failure_kind"),
    (
        (HINDSIGHT_EMBEDDING_CONNECTION_DETAIL, "supplier_connection"),
        ("Failed to generate batch embeddings: Request timed out.", "supplier_connection"),
        (
            "Failed to generate batch embeddings: Error code: 429 - {'error': 'rate limited'}",
            "supplier_rate_limit",
        ),
        (
            "Failed to generate batch embeddings: Error code: 503 - {'detail': 'unavailable'}",
            "supplier_connection",
        ),
    ),
)
def test_hindsight_terminal_transient_embedding_failure_is_settled(
    detail: str, failure_kind: str
) -> None:
    assert _classify(HINDSIGHT_BASIS, _raw({"detail": detail}), 500) == failure_kind


@pytest.mark.parametrize(
    "detail",
    (
        "Failed to generate batch embeddings: Error code: 401 - {'error': 'unauthorized'}",
        "Failed to generate batch embeddings: embedding dimension mismatch",
        "Failed to generate batch embeddings: unexpected backend failure",
    ),
)
def test_hindsight_terminal_permanent_or_unknown_embedding_failure_is_not_settled(
    detail: str,
) -> None:
    assert _classify(HINDSIGHT_BASIS, _raw({"detail": detail}), 500) is None


def test_openviking_bound_failed_task_accepts_nonempty_extraction_error() -> None:
    assert (
        _classify(
            OPENVIKING_BASIS,
            _ok(_task(error="ValidationError: extracted memory schema mismatch")),
            200,
            internal_retry_count=10,
        )
        == "provider_ingestion_error"
    )


@pytest.mark.parametrize(
    ("basis", "status", "body", "expected"),
    (
        (HINDSIGHT_BASIS, 500, _raw({"detail": HINDSIGHT_DETAIL}), "supplier_connection"),
        (
            HINDSIGHT_BASIS,
            500,
            _raw({"detail": HINDSIGHT_INVALID_JSON_DETAIL}),
            "supplier_invalid_json_output",
        ),
        (
            HINDSIGHT_BASIS,
            500,
            _raw(
                {
                    "detail": "Fact extraction failed: 2/3 chunks failed. First failures: "
                    "chunk 0: JSONDecodeError: Invalid control character at: line 1 column 1 (char 0), "
                    "chunk 2: JSONDecodeError: Invalid control character at: line 8 column 36 (char 243)"
                }
            ),
            "supplier_invalid_json_output",
        ),
        (
            HINDSIGHT_BASIS,
            500,
            _raw(
                {
                    "detail": "Fact extraction failed: 2/3 chunks failed. First failures: "
                    "chunk 0: APIConnectionError: Connection error., "
                    "chunk 2: APIConnectionError: Connection error."
                }
            ),
            "supplier_connection",
        ),
        (
            MEM0_BASIS,
            502,
            _raw(
                {
                    "detail": "Provider is unreachable or returned a server error.",
                    "code": "provider_unavailable",
                    "request_id": "1234abcd",
                }
            ),
            "supplier_connection",
        ),
        (
            MEM0_BASIS,
            502,
            _raw(
                {
                    "detail": "Provider rate limit hit. Retry shortly.",
                    "code": "provider_rate_limited",
                    "request_id": "1234abcd",
                }
            ),
            "supplier_rate_limit",
        ),
        (OPENVIKING_BASIS, 200, _ok(_task()), "supplier_connection"),
    ),
)
def test_exact_pinned_terminal_receipts_have_one_normalized_reason(
    basis: str, status: int, body: bytes, expected: str
) -> None:
    assert _classify(basis, body, status) == expected


@pytest.mark.parametrize("proof_count", (None, 0, 1, 2, -1, False, True))
@pytest.mark.parametrize(
    ("basis", "status", "body"),
    (
        (HINDSIGHT_BASIS, 500, _raw({"detail": HINDSIGHT_DETAIL})),
        (HINDSIGHT_BASIS, 500, _raw({"detail": HINDSIGHT_INVALID_JSON_DETAIL})),
        (
            MEM0_BASIS,
            502,
            _raw(
                {"detail": "unavailable", "code": "provider_unavailable", "request_id": "1234abcd"}
            ),
        ),
        (OPENVIKING_BASIS, 200, _ok(_task())),
    ),
)
def test_missing_or_nonzero_internal_retry_proof_cannot_qualify(
    proof_count: object, basis: str, status: int, body: bytes
) -> None:
    assert _classify(basis, body, status, internal_retry_count=proof_count) is None


@pytest.mark.parametrize(
    "detail",
    (
        "Connection error.",
        HINDSIGHT_DETAIL.replace("1/1", "2/2"),
        HINDSIGHT_DETAIL.replace("1/1", "6/6"),
        HINDSIGHT_DETAIL.replace("chunk 0", "chunk 1"),
        "Fact extraction failed: 2/2 chunks failed. First failures: "
        "chunk 0: APIConnectionError: Connection error., chunk 0: APIConnectionError: Connection error.",
    ),
)
def test_hindsight_requires_complete_unambiguous_connection_failures(detail: str) -> None:
    assert _classify(HINDSIGHT_BASIS, _raw({"detail": detail}), 500) is None


@pytest.mark.parametrize(
    "detail",
    (
        HINDSIGHT_INVALID_JSON_DETAIL.replace("1/1", "2/2"),
        HINDSIGHT_INVALID_JSON_DETAIL.replace("1/1", "6/6"),
        HINDSIGHT_INVALID_JSON_DETAIL.replace("chunk 0", "chunk 1"),
        "Fact extraction failed: 2/2 chunks failed. First failures: "
        "chunk 0: JSONDecodeError: Invalid control character at: line 8 column 36 (char 243), "
        "chunk 0: JSONDecodeError: Invalid control character at: line 8 column 36 (char 243)",
    ),
)
def test_hindsight_invalid_json_rebuild_requires_exact_uniform_diagnostics(detail: str) -> None:
    assert _classify(HINDSIGHT_BASIS, _raw({"detail": detail}), 500) is None


@pytest.mark.parametrize(
    "code",
    (
        "provider_auth_failed",
        "provider_bad_request",
        "provider_timeout",
        "unknown",
        "datastore_unavailable",
        "vector_store_unavailable",
        "APIConnectionError",
    ),
)
def test_mem0_does_not_retry_auth_configuration_timeout_or_opaque_errors(code: str) -> None:
    body = _raw({"detail": "Connection error.", "code": code, "request_id": "1234abcd"})
    assert _classify(MEM0_BASIS, body, 502) is None


@pytest.mark.parametrize(
    "changes",
    (
        {"status": "running"},
        {"status": "cancelled"},
        {"stage": "running"},
        {"task_id": "other-task"},
        {"resource_id": "other-session"},
        {"task_type": "add_resource"},
        {"error": None},
    ),
)
def test_openviking_requires_the_owned_terminal_task(changes: dict[str, object]) -> None:
    assert _classify(OPENVIKING_BASIS, _ok(_task(**changes)), 200) is None


@pytest.mark.parametrize(
    "options",
    (
        {"expected_task_id": None},
        {"expected_session_id": None},
        {"expected_task_id": "other-task"},
        {"expected_session_id": "other-session"},
    ),
)
def test_openviking_identity_must_come_from_independent_commit_evidence(
    options: dict[str, object],
) -> None:
    assert _classify(OPENVIKING_BASIS, _ok(_task()), 200, **options) is None


@pytest.mark.parametrize(
    "body", (b"not json", b"[]", b"null", b'{"detail":"Connection error."}', b"\xff")
)
def test_malformed_or_generic_http_errors_are_not_settlement_proof(body: bytes) -> None:
    for basis, status in ((HINDSIGHT_BASIS, 500), (MEM0_BASIS, 502), (OPENVIKING_BASIS, 200)):
        assert _classify(basis, body, status) is None


@pytest.mark.parametrize("status", (200, 400, 401, 429, 502, 503))
@pytest.mark.parametrize("detail", (HINDSIGHT_DETAIL, HINDSIGHT_INVALID_JSON_DETAIL))
def test_hindsight_status_must_match_the_pinned_sync_route(status: int, detail: str) -> None:
    assert _classify(HINDSIGHT_BASIS, _raw({"detail": detail}), status) is None


def test_duplicate_failure_fields_are_rejected() -> None:
    body = (
        b'{"detail":"opaque","code":"provider_auth_failed","code":'
        b'"provider_unavailable","request_id":"1234abcd"}'
    )
    assert _classify(MEM0_BASIS, body, 502) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("proof_count", (None, 0, 10))
@pytest.mark.parametrize(
    ("detail", "failure_kind"),
    (
        (HINDSIGHT_DETAIL, "supplier_connection"),
        (HINDSIGHT_INVALID_JSON_DETAIL, "supplier_invalid_json_output"),
        (HINDSIGHT_EMBEDDING_CONNECTION_DETAIL, "supplier_connection"),
    ),
)
async def test_hindsight_adapter_preserves_exact_terminal_receipt_and_proof_gate(
    proof_count: int | None,
    detail: str,
    failure_kind: str,
) -> None:
    from oamb.memory_systems.hindsight import HindsightAdapter

    service = _HindsightFixtureService("occurrence-1")
    raw = _raw({"detail": detail})

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/memories"):
            return httpx.Response(500, content=raw)
        return service(request)

    adapter = HindsightAdapter(
        store=CapturingStore(),
        base_url="https://hindsight.example",
        extraction_model="fixture",
        runtime_binding_hash="f" * 64,
        transport=httpx.MockTransport(handler),
        internal_retry_count=proof_count,
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id="occurrence-1", ingestion_plan_id="plan-1")
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(hindsight_source("source-1", 1),))
    )[0]
    with pytest.raises(MemorySystemCallFailure) as caught:
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
        )
    error = caught.value
    assert type(error).__name__ == (
        "SettledTransientIngestionFailure" if proof_count == 10 else "MemorySystemCallFailure"
    )
    assert error.raw_response_bytes == raw
    assert error.raw_reference is not None
    assert error.raw_reference.sha256 == hashlib.sha256(raw).hexdigest()
    assert error.status_code == 500
    if proof_count == 10:
        assert error.failure_kind == failure_kind
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("proof_count", (None, 0, 10))
async def test_mem0_adapter_promotes_only_proven_terminal_add_failure(
    proof_count: int | None,
) -> None:
    from oamb.memory_systems.mem0 import Mem0RestAdapter

    run_id = "a" * 64
    raw = _raw(
        {
            "detail": "Provider is unreachable or returned a server error.",
            "code": "provider_unavailable",
            "request_id": "1234abcd",
        }
    )

    def public(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        assert request.url.path == "/memories"
        assert json.loads(request.content)["run_id"] == run_id
        return httpx.Response(502, content=raw)

    def inspector(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "mode": "read_only_projection"})
        assert request.url.params["run_id"] == run_id
        return httpx.Response(
            200,
            json={
                "collection": "oamb_memories",
                "run_id": run_id,
                "count": 0,
                "points": [],
                "next_cursor": None,
            },
        )

    adapter = Mem0RestAdapter(
        store=CapturingStore(),
        base_url="https://mem0.example",
        api_key="fixture",
        inspector_base_url="https://inspector.example",
        inspector_api_key="fixture",
        runtime_binding_hash="f" * 64,
        transport=httpx.MockTransport(public),
        inspector_transport=httpx.MockTransport(inspector),
        internal_retry_count=proof_count,
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=run_id, ingestion_plan_id="b" * 64)
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(mem0_source(),))
    )[0]
    with pytest.raises(MemorySystemCallFailure) as caught:
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
        )
    assert type(caught.value).__name__ == (
        "SettledTransientIngestionFailure" if proof_count == 10 else "MemorySystemCallFailure"
    )
    assert caught.value.raw_response_bytes == raw
    await adapter.close()


@pytest.mark.asyncio
async def test_openviking_terminal_failure_retains_commit_and_all_poll_evidence() -> None:
    from oamb.memory_systems.openviking import OpenVikingSessionAdapter

    source = openviking_source("source-1", 1, _messages(2))
    service = SessionService((source,))
    store = CapturingStore()
    polls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal polls
        if request.url.path.startswith("/api/v1/tasks/"):
            polls += 1
            task_id = request.url.path.rsplit("/", 1)[-1]
            session_id = next(
                json.loads(call.content)["session_id"]
                for call in reversed(service.calls)
                if call.url.path == "/api/v1/sessions" and call.method == "POST"
            )
            return httpx.Response(
                200,
                content=_ok(
                    _task(
                        task_id=task_id,
                        resource_id=session_id,
                        status="running" if polls == 1 else "failed",
                        stage="running" if polls == 1 else "failed",
                    )
                ),
            )
        return service(request)

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
        internal_retry_count=10,
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(
            ingestion_occurrence_id=INGESTION_OCCURRENCE_ID, ingestion_plan_id="lme-plan-1"
        )
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(source,))
    )[0]
    with pytest.raises(MemorySystemCallFailure) as caught:
        await adapter.ingest(
            IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
        )
    error = caught.value
    assert type(error).__name__ == "SettledTransientIngestionFailure"
    assert polls == 2
    refs = (*error.supporting_raw_references, error.raw_reference)
    assert len(refs) >= 6
    retained = {item.sha256: item.payload_bytes for item in store.raw}
    assert all(ref is not None and ref.sha256 in retained for ref in refs)
    assert any(b'"status":"running"' in retained[ref.sha256] for ref in refs if ref is not None)
    assert any(
        b'"task_id"' in retained[ref.sha256] and b'"archive_uri"' in retained[ref.sha256]
        for ref in refs
        if ref is not None
    )
    with pytest.raises(MemorySystemCallFailure):
        await adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="e" * 64,
                dispatch=dispatch,
                batch_attempt_ordinal=2,
            )
        )
    created_session_ids = [
        json.loads(call.content)["session_id"]
        for call in service.calls
        if call.url.path == "/api/v1/sessions" and call.method == "POST"
    ]
    assert len(created_session_ids) == 2
    assert len(set(created_session_ids)) == 2
    await adapter.close()


@pytest.mark.asyncio
async def test_mem0_fresh_occurrence_binds_add_projection_and_search_requests() -> None:
    """Recorded REST proves scope propagation; native entity/message code is source-audited."""
    from oamb.memory_systems.mem0 import Mem0RestAdapter

    fresh_run, plan_id = "b" * 64, "c" * 64
    source = mem0_source()
    metadata = {
        "oamb_ingestion_occurrence_id": fresh_run,
        "oamb_ingestion_plan_id": plan_id,
        "oamb_source_unit_id": source.source_unit_id,
        "oamb_source_ordinal": 1,
    }
    point = {
        "id": "fresh-memory",
        "payload": {
            "data": "fresh occurrence only",
            "text_lemmatized": "fresh occurrence only",
            "hash": "f" * 32,
            "created_at": "2026-09-06T01:10:34+00:00",
            "updated_at": "2026-09-06T01:10:34+00:00",
            "run_id": fresh_run,
            **metadata,
        },
    }
    added = False
    projections = 0

    def public(request: httpx.Request) -> httpx.Response:
        nonlocal added
        if request.url.path == "/openapi.json":
            return httpx.Response(200, content=_openapi())
        body = json.loads(request.content)
        if request.url.path == "/memories":
            assert body["run_id"] == fresh_run
            assert body["metadata"] == metadata
            assert body["messages"] == [{"role": "user", "content": "source 1"}]
            assert body["infer"] is True
            added = True
            return httpx.Response(
                200,
                json={
                    "results": [
                        {"id": point["id"], "memory": "fresh occurrence only", "event": "ADD"}
                    ]
                },
            )
        assert request.url.path == "/search"
        assert body["filters"] == {"run_id": fresh_run}
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "id": point["id"],
                        "memory": "fresh occurrence only",
                        "hash": "f" * 32,
                        "metadata": metadata,
                        "run_id": fresh_run,
                        "score": 0.91,
                        "created_at": "2026-09-06T01:10:34+00:00",
                        "updated_at": "2026-09-06T01:10:34+00:00",
                    }
                ]
            },
        )

    def inspector(request: httpx.Request) -> httpx.Response:
        nonlocal projections
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "mode": "read_only_projection"})
        projections += 1
        assert request.url.params["run_id"] == fresh_run
        return httpx.Response(
            200,
            json={
                "collection": "oamb_memories",
                "run_id": fresh_run,
                "count": int(added),
                "points": [point] if added else [],
                "next_cursor": None,
            },
        )

    adapter = Mem0RestAdapter(
        store=CapturingStore(),
        base_url="https://mem0.example",
        api_key="fixture",
        inspector_base_url="https://inspector.example",
        inspector_api_key="fixture",
        runtime_binding_hash="f" * 64,
        transport=httpx.MockTransport(public),
        inspector_transport=httpx.MockTransport(inspector),
        internal_retry_count=10,
    )
    await adapter.resolve()
    scope = await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(ingestion_occurrence_id=fresh_run, ingestion_plan_id=plan_id)
    )
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(source,))
    )[0]
    receipt = await adapter.ingest(
        IngestionDispatchRequest(scope=scope, attempt_id="d" * 64, dispatch=dispatch)
    )
    ready = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=(source.source_unit_id,),
            ingestion_receipt=IngestionReceipt(
                ingestion_occurrence_id=fresh_run,
                accepted_source_unit_ids=(source.source_unit_id,),
                rejected_source_unit_ids=(),
                raw_references=(receipt.raw_reference,),
                dispatch_receipts=(receipt,),
            ),
        )
    )
    result = await adapter.retrieve(
        RetrievalRequest(
            scope=scope, case_occurrence_id="e" * 64, query_bytes=b"shared name", top_k=150
        )
    )
    assert ready.ready and projections == 3
    assert [(item.content, item.native_score) for item in result.candidates] == [
        ("fresh occurrence only", "0.91")
    ]
    await adapter.close()
