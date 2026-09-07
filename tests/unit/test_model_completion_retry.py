from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest

from oamb.contracts.ports import (
    ModelCallFailure,
    ModelCallUnknownOutcome,
    ModelReceipt,
    RawReferenceHandle,
)
from oamb.runtime.model_completion_retry import execute_model_completion_retry

Messages = tuple[tuple[str, str], ...]


def _receipt(output: str) -> ModelReceipt:
    return ModelReceipt(
        raw_reference=RawReferenceHandle("a" * 64),
        output_text=output,
        usage_reference_ids=("usage",),
        model="model",
    )


def _http_failure(status: int, *, failure_kind: str = "supplier_error") -> ModelCallFailure:
    return ModelCallFailure(
        f"HTTP {status}",
        raw_reference=RawReferenceHandle("b" * 64),
        raw_response_bytes=b"failure",
        usage_reference_ids=("usage",),
        retryable=True,
        failure_kind=failure_kind,
        supplier_status_code=status,
    )


async def _record(
    records: list[tuple[int, ModelReceipt | None, BaseException | None]],
    ordinal: int,
    receipt: ModelReceipt | None,
    error: BaseException | None,
) -> None:
    records.append((ordinal, receipt, error))


@pytest.mark.asyncio
async def test_rate_limit_uses_six_outer_attempts_and_three_transport_calls_each() -> None:
    ordinals: list[int] = []
    records: list[tuple[int, ModelReceipt | None, BaseException | None]] = []
    sleeps: list[float] = []

    async def dispatch(ordinal: int, _messages: Messages) -> ModelReceipt:
        ordinals.append(ordinal)
        raise _http_failure(429, failure_kind="rate_limited")

    with pytest.raises(ModelCallFailure):
        await execute_model_completion_retry(
            messages=(("user", "answer"),),
            dispatch=dispatch,
            validate=lambda _receipt: None,
            record=lambda ordinal, receipt, error: _record(records, ordinal, receipt, error),
            sleep=lambda seconds: _append_sleep(sleeps, seconds),
        )

    assert ordinals == list(range(1, 19))
    assert [ordinal for ordinal, _, _ in records] == ordinals
    assert sleeps == [
        0.5,
        1.0,
        5.0,
        0.5,
        1.0,
        10.0,
        0.5,
        1.0,
        20.0,
        0.5,
        1.0,
        40.0,
        0.5,
        1.0,
        80.0,
        0.5,
        1.0,
    ]


async def _append_sleep(sleeps: list[float], seconds: float) -> None:
    sleeps.append(seconds)


@pytest.mark.asyncio
async def test_six_format_failures_feed_back_exact_output_and_error() -> None:
    calls: list[tuple[int, Messages]] = []
    records: list[tuple[int, ModelReceipt | None, BaseException | None]] = []

    async def dispatch(ordinal: int, messages: Messages) -> ModelReceipt:
        calls.append((ordinal, messages))
        return _receipt(f"bad output {ordinal}")

    def validate(receipt: ModelReceipt) -> None:
        raise ValueError(f"expected yes/no; received {receipt.output_text!r}")

    with pytest.raises(ValueError, match="bad output 6"):
        await execute_model_completion_retry(
            messages=(("system", "judge"), ("user", "score")),
            dispatch=dispatch,
            validate=validate,
            record=lambda ordinal, receipt, error: _record(records, ordinal, receipt, error),
            sleep=_no_sleep,
        )

    assert [ordinal for ordinal, _ in calls] == list(range(1, 7))
    for next_call, failed_ordinal in zip(calls[1:], range(1, 6), strict=True):
        messages = next_call[1]
        assert messages[-2] == ("assistant", f"bad output {failed_ordinal}")
        assert f"expected yes/no; received 'bad output {failed_ordinal}'" in messages[-1][1]
    assert all(
        receipt is not None and isinstance(error, ValueError) for _, receipt, error in records
    )


async def _no_sleep(_seconds: float) -> None:
    return None


@pytest.mark.asyncio
async def test_non_rate_transport_exhaustion_raises_after_three_calls() -> None:
    calls: list[int] = []

    async def dispatch(ordinal: int, _messages: Messages) -> ModelReceipt:
        calls.append(ordinal)
        raise ModelCallUnknownOutcome("timeout", failure_kind="timeout")

    with pytest.raises(ModelCallUnknownOutcome):
        await execute_model_completion_retry(
            messages=(("user", "answer"),),
            dispatch=dispatch,
            validate=lambda _receipt: None,
            record=_discard_record,
            sleep=_no_sleep,
        )

    assert calls == [1, 2, 3]


async def _discard_record(
    _ordinal: int, _receipt: ModelReceipt | None, _error: BaseException | None
) -> None:
    return None


@pytest.mark.asyncio
async def test_invocations_have_independent_physical_call_quotas() -> None:
    async def run_once() -> list[int]:
        calls: list[int] = []

        async def dispatch(ordinal: int, _messages: Messages) -> ModelReceipt:
            calls.append(ordinal)
            if ordinal == 1:
                raise _http_failure(408)
            return _receipt("yes")

        await execute_model_completion_retry(
            messages=(("user", "score"),),
            dispatch=dispatch,
            validate=lambda _receipt: None,
            record=_discard_record,
            sleep=_no_sleep,
        )
        return calls

    assert await run_once() == [1, 2]
    assert await run_once() == [1, 2]


@pytest.mark.asyncio
async def test_valid_no_is_returned_without_retry() -> None:
    calls: list[int] = []

    async def dispatch(ordinal: int, _messages: Messages) -> ModelReceipt:
        calls.append(ordinal)
        return _receipt("no")

    receipt = await execute_model_completion_retry(
        messages=(("user", "score"),),
        dispatch=dispatch,
        validate=lambda _receipt: None,
        record=_discard_record,
        sleep=_no_sleep,
    )

    assert receipt.output_text == "no"
    assert calls == [1]


@pytest.mark.asyncio
async def test_cancellation_is_recorded_and_starts_no_successor() -> None:
    calls: list[int] = []
    records: list[tuple[int, ModelReceipt | None, BaseException | None]] = []

    async def dispatch(ordinal: int, _messages: Messages) -> ModelReceipt:
        calls.append(ordinal)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await execute_model_completion_retry(
            messages=(("user", "answer"),),
            dispatch=dispatch,
            validate=lambda _receipt: None,
            record=lambda ordinal, receipt, error: _record(records, ordinal, receipt, error),
            sleep=_no_sleep,
        )

    assert calls == [1]
    assert len(records) == 1 and isinstance(records[0][2], asyncio.CancelledError)


@pytest.mark.asyncio
async def test_record_completes_before_retry_dispatch() -> None:
    events: list[str] = []
    failures: Sequence[BaseException | None] = (_http_failure(500), None)

    async def dispatch(ordinal: int, _messages: Messages) -> ModelReceipt:
        events.append(f"dispatch-{ordinal}")
        failure = failures[ordinal - 1]
        if failure is not None:
            raise failure
        return _receipt("answer")

    async def record(
        ordinal: int, _receipt: ModelReceipt | None, _error: BaseException | None
    ) -> None:
        events.append(f"record-{ordinal}")

    await execute_model_completion_retry(
        messages=(("user", "answer"),),
        dispatch=dispatch,
        validate=lambda _receipt: None,
        record=record,
        sleep=_no_sleep,
    )

    assert events == ["dispatch-1", "record-1", "dispatch-2", "record-2"]


@pytest.mark.asyncio
async def test_non_value_validation_failure_is_not_retried() -> None:
    calls: list[int] = []

    async def dispatch(ordinal: int, _messages: Messages) -> ModelReceipt:
        calls.append(ordinal)
        return _receipt("filtered")

    def validate(_receipt: ModelReceipt) -> None:
        raise RuntimeError("content filtered")

    with pytest.raises(RuntimeError, match="content filtered"):
        await execute_model_completion_retry(
            messages=(("user", "answer"),),
            dispatch=dispatch,
            validate=validate,
            record=_discard_record,
            sleep=_no_sleep,
        )

    assert calls == [1]
