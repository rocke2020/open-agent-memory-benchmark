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


class BarrierTransport(httpx.AsyncBaseTransport):
    def __init__(self, *, expected_calls: int) -> None:
        self._expected_calls = expected_calls
        self.paths: list[str] = []
        self.all_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.close_started = asyncio.Event()
        self.close_count = 0

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if len(self.paths) == self._expected_calls:
            self.all_entered.set()
        await self.release.wait()
        return httpx.Response(200, json={"status": "ok"})

    async def aclose(self) -> None:
        self.close_count += 1
        self.close_started.set()


class FailFirstCloseTransport(httpx.AsyncBaseTransport):
    def __init__(self) -> None:
        self.close_count = 0
        self.first_close_started = asyncio.Event()
        self.release_first_close = asyncio.Event()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request while testing close: {request.url}")

    async def aclose(self) -> None:
        self.close_count += 1
        if self.close_count == 1:
            self.first_close_started.set()
            await self.release_first_close.wait()
            raise RuntimeError("fixture close failure")


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
        request_evidence=True,
    )

    assert len(calls) == 1
    assert calls[0].url == httpx.URL("https://memory.example/api/v1/write")
    assert calls[0].headers["X-API-Key"] == "secret"
    assert response.raw_bytes == b'{"status":"ok"}'
    assert response.status_code == 200
    assert response.raw_reference.sha256 == hashlib.sha256(response.raw_bytes).hexdigest()
    assert response.request_reference is not None
    assert store.raw == [
        RawPayloadSealRequest(
            sha256=response.request_reference.sha256,
            media_type="application/vnd.oamb.request+json",
            compression="gzip",
            payload_bytes=(
                b'{"json_payload":{"value":"exact"},"method":"POST","params":{},'
                b'"path":"/v1/write","request_header_names":[],"schema_name":'
                b'"oamb_rest_request_proof","schema_version":1,"write_intent":true}'
            ),
        ),
        RawPayloadSealRequest(
            sha256=response.raw_reference.sha256,
            media_type="application/json",
            compression="gzip",
            payload_bytes=response.raw_bytes,
        ),
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
async def test_seven_requests_enter_transport_in_parallel_without_an_internal_limit() -> None:
    rest = _rest_module()
    transport = BarrierTransport(expected_calls=7)
    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=transport,
    )
    tasks = [asyncio.create_task(client.request("GET", f"/call-{index}")) for index in range(7)]

    try:
        await asyncio.wait_for(transport.all_entered.wait(), timeout=1.0)
        assert sorted(transport.paths) == [f"/call-{index}" for index in range(7)]
    finally:
        transport.release.set()
        responses = await asyncio.gather(*tasks)

    assert [response.status_code for response in responses] == [200] * 7
    await client.close()


@pytest.mark.asyncio
async def test_stop_accepting_classifies_a_write_as_cancelled_before_dispatch() -> None:
    rest = _rest_module()
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200)

    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(handler),
    )

    client.stop_accepting()

    with pytest.raises(asyncio.CancelledError) as cancelled:
        await client.request("POST", "/late", json_payload={}, write_intent=True)
    assert isinstance(cancelled.value, rest.MemorySystemCallCancelledBeforeDispatch)
    assert calls == []
    await client.close()


@pytest.mark.asyncio
async def test_fully_closed_client_rejects_a_call_as_closed() -> None:
    rest = _rest_module()
    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=httpx.MockTransport(lambda request: httpx.Response(200)),
    )

    await client.close()

    with pytest.raises(RuntimeError, match="closed"):
        await client.request("GET", "/late")


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
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await client.request("GET", "/late")
    assert isinstance(cancelled.value, MemorySystemReadCancelled)

    release.set()
    assert (await active).status_code == 200
    await closing


@pytest.mark.asyncio
async def test_close_drains_every_admitted_call_before_closing_transport() -> None:
    rest = _rest_module()
    transport = BarrierTransport(expected_calls=3)
    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=transport,
    )
    active = [asyncio.create_task(client.request("GET", f"/active-{index}")) for index in range(3)]
    await asyncio.wait_for(transport.all_entered.wait(), timeout=1.0)
    await asyncio.sleep(0)
    closing = asyncio.create_task(client.close())
    await asyncio.sleep(0)

    assert not closing.done()
    assert not transport.close_started.is_set()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await client.request("GET", "/late")
    assert isinstance(cancelled.value, MemorySystemReadCancelled)

    transport.release.set()
    assert [response.status_code for response in await asyncio.gather(*active)] == [200] * 3
    await closing
    assert transport.close_count == 1


@pytest.mark.asyncio
async def test_cancelled_admitted_read_releases_close_drain() -> None:
    rest = _rest_module()
    transport = BarrierTransport(expected_calls=1)
    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=transport,
    )
    active = asyncio.create_task(client.request("GET", "/active"))
    await transport.all_entered.wait()

    active.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await active
    assert isinstance(cancelled.value, MemorySystemReadCancelled)

    await asyncio.wait_for(client.close(), timeout=1.0)
    assert transport.close_count == 1


@pytest.mark.asyncio
async def test_concurrent_close_propagates_first_error_and_allows_waiter_to_retry() -> None:
    rest = _rest_module()
    transport = FailFirstCloseTransport()
    client = rest.SealedRestClient(
        store=CapturingStore(),
        base_url="https://memory.example",
        headers={},
        transport=transport,
    )

    first_close = asyncio.create_task(client.close())
    await transport.first_close_started.wait()
    waiting_close = asyncio.create_task(client.close())
    await asyncio.sleep(0)
    assert not waiting_close.done()

    transport.release_first_close.set()
    with pytest.raises(RuntimeError, match="fixture close failure"):
        await first_close
    await waiting_close
    await client.close()

    assert transport.close_count == 2


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
