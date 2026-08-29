from __future__ import annotations

import asyncio
import hashlib
from importlib import import_module
from typing import Any

import httpx
import pytest

from oamb.contracts.ports import (
    ArtifactReadRequest,
    ArtifactSealReceipt,
    ArtifactWriteRequest,
    MemorySystemCallFailure,
    MemorySystemReadCancelled,
    RawPayloadSealRequest,
    RawReferenceHandle,
)


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


class FailingRawStore(CapturingStore):
    def seal_raw(self, request: RawPayloadSealRequest) -> RawReferenceHandle:
        raise OSError(f"fixture cannot seal {request.sha256}")


def _rest_module() -> Any:
    try:
        return import_module("oamb.memory_systems.rest")
    except ModuleNotFoundError:
        pytest.fail("the shared memory-system REST transport is not implemented")


@pytest.mark.asyncio
async def test_rest_transport_dispatches_once_and_seals_exact_raw_response() -> None:
    rest = _rest_module()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            content=b'{"status":"ok"}',
            headers={"Content-Type": "application/json; charset=utf-8"},
        )

    store = CapturingStore()
    client = rest.SealedRestClient(
        store=store,
        base_url="https://memory.example/api",
        headers={"X-API-Key": "secret"},
        transport=httpx.MockTransport(handler),
    )

    response = await client.request(
        "POST",
        "/v1/write",
        json_payload={"value": "exact"},
        write_intent=True,
    )

    assert len(calls) == 1
    assert calls[0].url == httpx.URL("https://memory.example/api/v1/write")
    assert calls[0].headers["X-API-Key"] == "secret"
    assert response.raw_bytes == b'{"status":"ok"}'
    assert response.status_code == 200
    assert response.raw_reference.sha256 == hashlib.sha256(response.raw_bytes).hexdigest()
    assert store.raw == [
        RawPayloadSealRequest(
            sha256=response.raw_reference.sha256,
            media_type="application/json",
            compression="gzip",
            payload_bytes=response.raw_bytes,
        )
    ]
    await client.close()


@pytest.mark.asyncio
async def test_request_headers_add_scope_without_replacing_client_authority() -> None:
    rest = _rest_module()
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={"X-API-Key": "user-key"},
        transport=httpx.MockTransport(handler),
    )

    await client.request(
        "GET",
        "/scope",
        request_headers={"X-OpenViking-Actor-Peer": "oamb-peer"},
    )
    assert calls[0].headers["X-API-Key"] == "user-key"
    assert calls[0].headers["X-OpenViking-Actor-Peer"] == "oamb-peer"

    with pytest.raises(ValueError, match="override"):
        await client.request(
            "GET",
            "/scope",
            request_headers={"x-api-key": "root-key"},
        )
    assert len(calls) == 1
    await client.close()


@pytest.mark.asyncio
async def test_rest_transport_preserves_http_failure_receipt() -> None:
    rest = _rest_module()
    store = CapturingStore()
    client = rest.SealedRestClient(
        store=store,
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(
            lambda request: httpx.Response(409, json={"error": "scope exists"})
        ),
    )

    with pytest.raises(rest.MemorySystemCallFailure) as failure:
        await client.request("PUT", "/scope", json_payload={}, write_intent=True)

    assert failure.value.failure_kind == "http_status"
    assert failure.value.status_code == 409
    assert failure.value.raw_reference == RawReferenceHandle(store.raw[0].sha256)
    assert failure.value.raw_response_bytes == store.raw[0].payload_bytes
    await client.close()


@pytest.mark.asyncio
async def test_write_response_seal_failure_is_unknown_outcome_without_retry() -> None:
    rest = _rest_module()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content=b'{"status":"ok"}')

    client = rest.SealedRestClient(
        store=FailingRawStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(rest.MemorySystemCallUnknownOutcome) as failure:
        await client.request("POST", "/write", json_payload={}, write_intent=True)

    assert failure.value.failure_kind == "receipt_seal_error"
    assert calls == 1
    await client.close()


@pytest.mark.asyncio
async def test_read_response_seal_failure_is_a_known_transport_failure() -> None:
    rest = _rest_module()
    client = rest.SealedRestClient(
        store=FailingRawStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=b'{"status":"ok"}')
        ),
    )

    with pytest.raises(rest.MemorySystemCallFailure) as failure:
        await client.request("GET", "/projection")

    assert not isinstance(failure.value, rest.MemorySystemCallUnknownOutcome)
    assert failure.value.failure_kind == "receipt_seal_error"
    await client.close()


@pytest.mark.asyncio
async def test_write_transport_timeout_is_unknown_outcome_without_retry() -> None:
    rest = _rest_module()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("lost response", request=request)

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(rest.MemorySystemCallUnknownOutcome) as failure:
        await client.request("POST", "/write", json_payload={}, write_intent=True)

    assert failure.value.failure_kind == "timeout"
    assert calls == 1
    await client.close()


@pytest.mark.asyncio
async def test_read_transport_timeout_is_not_misclassified_as_unknown_write() -> None:
    rest = _rest_module()

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timeout", request=request)

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(rest.MemorySystemCallFailure) as failure:
        await client.request("GET", "/projection")

    assert not isinstance(failure.value, rest.MemorySystemCallUnknownOutcome)
    assert failure.value.failure_kind == "timeout"
    await client.close()


@pytest.mark.asyncio
async def test_cancelled_write_retains_asyncio_cancellation_and_unknown_outcome() -> None:
    rest = _rest_module()
    entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )
    task = asyncio.create_task(client.request("POST", "/write", json_payload={}, write_intent=True))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task

    assert isinstance(cancelled.value, rest.MemorySystemCallUnknownOutcome)
    assert cancelled.value.failure_kind == "cancelled"
    await client.close()


@pytest.mark.asyncio
async def test_cancelled_read_retains_asyncio_semantics_without_unknown_write() -> None:
    rest = _rest_module()
    entered = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )
    task = asyncio.create_task(client.request("GET", "/projection"))
    await entered.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await task

    assert isinstance(cancelled.value, MemorySystemReadCancelled)
    assert not isinstance(cancelled.value, rest.MemorySystemCallUnknownOutcome)
    assert cancelled.value.failure_kind == "cancelled"
    assert cancelled.value.supporting_raw_references == ()
    await client.close()


@pytest.mark.asyncio
async def test_cancelled_queued_read_is_typed_without_dispatch() -> None:
    rest = _rest_module()
    active_started = asyncio.Event()
    release_active = asyncio.Event()
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        active_started.set()
        await release_active.wait()
        return httpx.Response(200, json={"status": "ok"})

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )
    active = asyncio.create_task(client.request("GET", "/active"))
    await active_started.wait()
    queued = asyncio.create_task(client.request("GET", "/queued"))
    await asyncio.sleep(0)
    queued.cancel()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await queued

    assert isinstance(cancelled.value, MemorySystemReadCancelled)
    assert calls == ["/active"]
    release_active.set()
    await active
    await client.close()


@pytest.mark.asyncio
async def test_close_rejects_new_calls_and_waits_for_the_active_call() -> None:
    rest = _rest_module()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"status": "ok"})

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )
    active = asyncio.create_task(client.request("GET", "/projection"))
    await entered.wait()
    closing = asyncio.create_task(client.close())
    await asyncio.sleep(0)

    assert not closing.done()
    with pytest.raises(RuntimeError, match="closed"):
        await client.request("GET", "/late")

    release.set()
    assert (await active).status_code == 200
    await closing


@pytest.mark.asyncio
async def test_close_classifies_a_queued_write_as_cancelled_before_dispatch() -> None:
    rest = _rest_module()
    active_started = asyncio.Event()
    release_active = asyncio.Event()
    calls: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        if request.url.path == "/active":
            active_started.set()
            await release_active.wait()
        return httpx.Response(200, json={"status": "ok"})

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )
    active = asyncio.create_task(client.request("GET", "/active"))
    await active_started.wait()
    queued_write = asyncio.create_task(
        client.request("POST", "/queued", json_payload={}, write_intent=True)
    )
    await asyncio.sleep(0)
    closing = asyncio.create_task(client.close())
    await asyncio.sleep(0)
    release_active.set()

    assert (await active).status_code == 200
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await queued_write

    assert isinstance(cancelled.value, rest.MemorySystemCallCancelledBeforeDispatch)
    assert calls == ["/active"]
    await closing


def test_exact_json_object_rejects_duplicate_unknown_and_non_finite_fields() -> None:
    rest = _rest_module()

    assert rest.parse_exact_json_object(
        b'{"items":[],"total":0}',
        expected_fields=frozenset({"items", "total"}),
    ) == {"items": [], "total": 0}
    for payload in (
        b'{"items":[],"total":0,"extra":true}',
        b'{"items":[],"items":[],"total":0}',
        b'{"items":[],"total":NaN}',
    ):
        with pytest.raises(ValueError):
            rest.parse_exact_json_object(
                payload,
                expected_fields=frozenset({"items", "total"}),
            )


def test_linked_failure_preserves_ordered_duplicate_response_occurrences() -> None:
    rest = _rest_module()
    repeated_reference = RawReferenceHandle("a" * 64)
    later_reference = RawReferenceHandle("b" * 64)
    failure = MemorySystemCallFailure(
        "fixture failure",
        failure_kind="transport_error",
        supporting_raw_references=(later_reference,),
    )

    linked = rest.link_preceding_raw_references(
        failure,
        preceding_raw_references=(repeated_reference, repeated_reference),
    )

    assert linked.supporting_raw_references == (
        repeated_reference,
        repeated_reference,
        later_reference,
    )
