"""Bounded layered retries for one answer or judge model completion."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from oamb.contracts.ports import (
    ModelCallFailure,
    ModelCallUnknownOutcome,
    ModelReceipt,
    ModelSupplierRateLimitRejection,
)

Messages = tuple[tuple[str, str], ...]
Dispatch = Callable[[int, Messages], Awaitable[ModelReceipt]]
Validate = Callable[[ModelReceipt], None]
Record = Callable[[int, ModelReceipt | None, BaseException | None], Awaitable[None]]
Sleep = Callable[[float], Awaitable[None]]

_TRANSPORT_RETRY_STATUS_CODES = frozenset({408, 409, 429})


async def execute_model_completion_retry(
    *,
    messages: Messages,
    dispatch: Dispatch,
    validate: Validate,
    record: Record,
    max_outer_attempts: int = 6,
    max_transport_retries: int = 2,
    sleep: Sleep = asyncio.sleep,
) -> ModelReceipt:
    """Execute one role-local completion with transport and format retry layers."""

    if type(max_outer_attempts) is not int or max_outer_attempts < 1:
        raise ValueError("max outer attempts must be a positive integer")
    if type(max_transport_retries) is not int or max_transport_retries < 0:
        raise ValueError("max transport retries must be a non-negative integer")

    physical_call_ordinal = 0
    request_messages = messages
    for outer_attempt_index in range(max_outer_attempts):
        format_error: ValueError | None = None
        terminal_error: BaseException | None = None
        for transport_retry_index in range(max_transport_retries + 1):
            physical_call_ordinal += 1
            try:
                receipt = await dispatch(physical_call_ordinal, request_messages)
            except BaseException as error:
                await record(physical_call_ordinal, None, error)
                if (
                    isinstance(error, ModelSupplierRateLimitRejection)
                    and error.classification.internal_retry_count != 0
                ):
                    raise RuntimeError("supplier internal retry configuration drift") from error
                if isinstance(error, asyncio.CancelledError):
                    raise
                if _is_transport_retryable(error) and transport_retry_index < max_transport_retries:
                    await sleep(0.5 * (2**transport_retry_index))
                    continue
                terminal_error = error
                break

            try:
                validate(receipt)
            except BaseException as error:
                await record(physical_call_ordinal, receipt, error)
                if not isinstance(error, ValueError):
                    raise
                format_error = error
                if outer_attempt_index + 1 < max_outer_attempts:
                    request_messages = _append_format_correction(
                        request_messages,
                        output_text=receipt.output_text,
                        error=error,
                    )
                break
            else:
                await record(physical_call_ordinal, receipt, None)
                return receipt

        if format_error is not None:
            if outer_attempt_index + 1 == max_outer_attempts:
                raise format_error
            continue
        if terminal_error is None:
            raise AssertionError("model completion attempt ended without a result")
        if _is_outer_rate_limit(terminal_error) and outer_attempt_index + 1 < max_outer_attempts:
            await sleep(5.0 * (2**outer_attempt_index))
            continue
        raise terminal_error

    raise AssertionError("model completion retry loop exceeded its bound")


def _is_transport_retryable(error: BaseException) -> bool:
    if isinstance(error, ModelCallUnknownOutcome):
        return error.failure_kind in {"timeout", "transport_error"}
    if not isinstance(error, ModelCallFailure):
        return False
    status = error.supplier_status_code
    return status in _TRANSPORT_RETRY_STATUS_CODES or (status is not None and 500 <= status <= 599)


def _is_outer_rate_limit(error: BaseException) -> bool:
    return isinstance(error, ModelCallFailure) and (
        error.supplier_status_code == 429 or error.failure_kind == "rate_limited"
    )


def _append_format_correction(
    messages: Messages,
    *,
    output_text: str,
    error: ValueError,
) -> Messages:
    correction = (
        "The previous response failed output validation: "
        f"{error}\nReturn a corrected response that satisfies the required output format."
    )
    return (*messages, ("assistant", output_text), ("user", correction))
