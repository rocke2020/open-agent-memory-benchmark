from __future__ import annotations

import asyncio
import json

import pytest

from oamb.contracts.ports import (
    MemorySystemCallFailure,
    ModelCallFailure,
    ModelCallUnknownOutcome,
    ModelSupplierRateLimitRejection,
    RawReferenceHandle,
)
from oamb.runtime.infrastructure_retry import (
    InfrastructureRetryController,
    InfrastructureRetryExhausted,
    execute_with_infrastructure_retry,
    parse_structured_supplier_rejection,
)
from oamb.runtime.native_run import _is_pure_infrastructure_retry_exhaustion


def _rejection(
    call_number: int,
    *,
    internal_retry_count: int = 0,
) -> ModelSupplierRateLimitRejection:
    classification = parse_structured_supplier_rejection(
        json.dumps(
            {
                "error": {
                    "origin": "model_supplier",
                    "failure_kind": "rate_limited",
                    "status": 429,
                    "acceptance": "not_accepted",
                    "provider_mutation": "none",
                    "retryable": True,
                    "internal_retry_count": internal_retry_count,
                }
            }
        ).encode(),
        status_code=429,
    )
    assert classification is not None
    return ModelSupplierRateLimitRejection(
        "structured supplier rejection",
        classification=classification,
        raw_reference=RawReferenceHandle(f"{call_number:064x}"),
        raw_response_bytes=b"{}",
        usage_reference_ids=(f"{call_number + 100:064x}",),
    )


def test_infrastructure_blocked_requires_only_safe_exhaustion_roots() -> None:
    exhausted = InfrastructureRetryExhausted(
        "retry policy exhausted",
        last_rejection=_rejection(1),
    )

    assert _is_pure_infrastructure_retry_exhaustion(exhausted)
    assert not _is_pure_infrastructure_retry_exhaustion(
        BaseExceptionGroup(
            "mixed terminal failures",
            (
                exhausted,
                ModelCallUnknownOutcome(
                    "supplier acceptance is unknown",
                    failure_kind="transport_error",
                ),
            ),
        )
    )


def test_classifier_requires_the_complete_exact_supplier_schema() -> None:
    exact = {
        "error": {
            "origin": "model_supplier",
            "failure_kind": "rate_limited",
            "status": 429,
            "acceptance": "not_accepted",
            "provider_mutation": "none",
            "retryable": True,
            "internal_retry_count": 0,
        }
    }
    assert (
        parse_structured_supplier_rejection(json.dumps(exact).encode(), status_code=429) is not None
    )

    invalid_documents = (
        {"error": {"message": "rate limit"}},
        {**exact, "extra": True},
        {"error": {**exact["error"], "acceptance": "unknown"}},
        {"error": {**exact["error"], "provider_mutation": "unknown"}},
        {"error": {**exact["error"], "internal_retry_count": -1}},
    )
    for document in invalid_documents:
        assert (
            parse_structured_supplier_rejection(json.dumps(document).encode(), status_code=429)
            is None
        )

    duplicate_key_documents = (
        b'{"error":{"origin":"unsafe"},"error":{"origin":"model_supplier",'
        b'"failure_kind":"rate_limited","status":429,"acceptance":"not_accepted",'
        b'"provider_mutation":"none","retryable":true,"internal_retry_count":0}}',
        b'{"error":{"origin":"unsafe","origin":"model_supplier",'
        b'"failure_kind":"rate_limited","status":429,"acceptance":"not_accepted",'
        b'"provider_mutation":"none","retryable":true,"internal_retry_count":0}}',
    )
    for raw_document in duplicate_key_documents:
        assert parse_structured_supplier_rejection(raw_document, status_code=429) is None
    assert parse_structured_supplier_rejection(json.dumps(exact).encode(), status_code=500) is None


@pytest.mark.asyncio
async def test_retry_events_are_durable_before_one_two_four_backoffs() -> None:
    calls = 0
    events: list[tuple[int, int | None]] = []
    sleeps: list[int] = []

    async def call(_supplier_call_ordinal: int) -> str:
        nonlocal calls
        calls += 1
        if calls <= 3:
            raise _rejection(calls)
        return "accepted"

    async def persist(
        rejection: ModelSupplierRateLimitRejection,
        call_ordinal: int,
        backoff_seconds: int | None,
    ) -> None:
        assert rejection.classification.acceptance == "not_accepted"
        events.append((call_ordinal, backoff_seconds))

    async def sleep(seconds: int) -> None:
        assert events[-1] == (len(events), seconds)
        sleeps.append(seconds)

    result = await execute_with_infrastructure_retry(
        call,
        persist_rejection=persist,
        sleep=sleep,
        controller=InfrastructureRetryController(maximum_total_retries=3),
    )

    assert result == "accepted"
    assert calls == 4
    assert events == [(1, 1), (2, 2), (3, 4)]
    assert sleeps == [1, 2, 4]


@pytest.mark.asyncio
async def test_retry_exhaustion_records_final_rejection_and_honors_global_budget() -> None:
    calls = 0
    events: list[tuple[int, int | None]] = []

    async def call(_supplier_call_ordinal: int) -> None:
        nonlocal calls
        calls += 1
        raise _rejection(calls)

    async def persist(
        _rejection_value: ModelSupplierRateLimitRejection,
        call_ordinal: int,
        backoff_seconds: int | None,
    ) -> None:
        events.append((call_ordinal, backoff_seconds))

    with pytest.raises(InfrastructureRetryExhausted, match="global retry budget"):
        await execute_with_infrastructure_retry(
            call,
            persist_rejection=persist,
            sleep=lambda _seconds: asyncio.sleep(0),
            controller=InfrastructureRetryController(maximum_total_retries=2),
        )

    assert calls == 3
    assert events == [(1, 1), (2, 2), (3, None)]


@pytest.mark.asyncio
async def test_supplier_internal_retries_consume_the_global_retry_budget() -> None:
    calls = 0
    events: list[tuple[int, int | None, int]] = []

    async def call(_supplier_call_ordinal: int) -> None:
        nonlocal calls
        calls += 1
        raise _rejection(calls, internal_retry_count=4)

    async def persist(
        rejection: ModelSupplierRateLimitRejection,
        call_ordinal: int,
        backoff_seconds: int | None,
    ) -> None:
        events.append(
            (
                call_ordinal,
                backoff_seconds,
                rejection.classification.internal_retry_count,
            )
        )

    with pytest.raises(InfrastructureRetryExhausted, match="global retry budget"):
        await execute_with_infrastructure_retry(
            call,
            persist_rejection=persist,
            sleep=lambda _seconds: asyncio.sleep(0),
            controller=InfrastructureRetryController(maximum_total_retries=3),
        )

    assert calls == 1
    assert events == [(1, None, 4)]


@pytest.mark.asyncio
async def test_cancellation_during_backoff_starts_no_later_call() -> None:
    calls = 0
    persisted = asyncio.Event()

    async def call(_supplier_call_ordinal: int) -> None:
        nonlocal calls
        calls += 1
        raise _rejection(calls)

    async def persist(
        _rejection_value: ModelSupplierRateLimitRejection,
        _call_ordinal: int,
        _backoff_seconds: int | None,
    ) -> None:
        persisted.set()

    async def blocked_sleep(_seconds: int) -> None:
        await asyncio.Event().wait()

    task = asyncio.create_task(
        execute_with_infrastructure_retry(
            call,
            persist_rejection=persist,
            sleep=blocked_sleep,
            controller=InfrastructureRetryController(maximum_total_retries=3),
        )
    )
    await persisted.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("proxy_kind", ["status-only", "hindsight-outer-500"])
async def test_proxy_failures_never_enter_the_infrastructure_retry_loop(
    proxy_kind: str,
) -> None:
    calls = 0
    persisted = 0

    async def call(_supplier_call_ordinal: int) -> None:
        nonlocal calls
        calls += 1
        if proxy_kind == "status-only":
            raise ModelCallFailure(
                "busy",
                raw_reference=RawReferenceHandle("f" * 64),
                raw_response_bytes=b'{"error":{"message":"busy"}}',
                usage_reference_ids=(),
                retryable=True,
                supplier_status_code=429,
            )
        raise MemorySystemCallFailure(
            "internal supplier failure text",
            failure_kind="http_status",
            raw_reference=RawReferenceHandle("e" * 64),
            raw_response_bytes=b'{"detail":"rate limited"}',
            status_code=500,
        )

    async def persist(
        _rejection_value: ModelSupplierRateLimitRejection,
        _call_ordinal: int,
        _backoff_seconds: int | None,
    ) -> None:
        nonlocal persisted
        persisted += 1

    expected = ModelCallFailure if proxy_kind == "status-only" else MemorySystemCallFailure
    with pytest.raises(expected):
        await execute_with_infrastructure_retry(
            call,
            persist_rejection=persist,
            sleep=lambda _seconds: asyncio.sleep(0),
            controller=InfrastructureRetryController(maximum_total_retries=3),
        )

    assert calls == 1
    assert persisted == 0
