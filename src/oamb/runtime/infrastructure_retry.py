"""Fail-closed structured model-supplier rejection and bounded retry ownership."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from oamb.contracts.ports import (
    ModelSupplierRateLimitRejection,
)
from oamb.contracts.specifications import INFRASTRUCTURE_RETRY_BACKOFF_SECONDS
from oamb.contracts.supplier_rejection import parse_structured_supplier_rejection

_Result = TypeVar("_Result")


class InfrastructureRetryExhausted(RuntimeError):
    """A safe rejection could not be retried within its operation limit."""

    def __init__(
        self,
        message: str,
        *,
        last_rejection: ModelSupplierRateLimitRejection,
    ) -> None:
        super().__init__(message)
        self.last_rejection = last_rejection
        self.raw_reference = last_rejection.raw_reference
        self.raw_response_bytes = last_rejection.raw_response_bytes
        self.usage_reference_ids = last_rejection.usage_reference_ids


class InfrastructureBackoffCancelled(asyncio.CancelledError):
    """Backoff ended before another supplier call could be dispatched."""


class InfrastructureRetryController:
    """One logical operation's independent infrastructure retry counter."""

    def __init__(self, *, maximum_total_retries: int) -> None:
        if maximum_total_retries < 0:
            raise ValueError("maximum total infrastructure retries must be non-negative")
        self._maximum = maximum_total_retries
        self._consumed = 0
        self._lock = asyncio.Lock()

    async def reserve(self, units: int) -> bool:
        if units < 1:
            raise ValueError("infrastructure retry reservation must be positive")
        async with self._lock:
            if self._consumed + units > self._maximum:
                return False
            self._consumed += units
            return True

    @property
    def consumed(self) -> int:
        return self._consumed


def _backoff_seconds(retry_ordinal: int) -> int:
    seed_index = retry_ordinal - 1
    if seed_index < len(INFRASTRUCTURE_RETRY_BACKOFF_SECONDS):
        return INFRASTRUCTURE_RETRY_BACKOFF_SECONDS[seed_index]
    additional_doublings = seed_index - len(INFRASTRUCTURE_RETRY_BACKOFF_SECONDS) + 1
    return INFRASTRUCTURE_RETRY_BACKOFF_SECONDS[-1] << additional_doublings


async def execute_with_infrastructure_retry(
    call: Callable[[int], Awaitable[_Result]],
    *,
    persist_rejection: Callable[
        [ModelSupplierRateLimitRejection, int, int | None], Awaitable[None]
    ],
    sleep: Callable[[int], Awaitable[None]],
    controller: InfrastructureRetryController,
) -> _Result:
    """Run one logical call; persist every rejection before any bounded backoff."""

    call_ordinal = 0
    while True:
        call_ordinal += 1
        try:
            return await call(call_ordinal)
        except ModelSupplierRateLimitRejection as rejection:
            backoff: int | None = None
            units = 1 + rejection.classification.internal_retry_count
            if await controller.reserve(units):
                backoff = _backoff_seconds(call_ordinal)
            await persist_rejection(rejection, call_ordinal, backoff)
            if backoff is None:
                raise InfrastructureRetryExhausted(
                    "operation retry limit exhausted",
                    last_rejection=rejection,
                ) from rejection
            try:
                await sleep(backoff)
            except asyncio.CancelledError as exc:
                raise InfrastructureBackoffCancelled(
                    "infrastructure retry backoff was cancelled before dispatch"
                ) from exc


__all__ = [
    "InfrastructureRetryController",
    "InfrastructureBackoffCancelled",
    "InfrastructureRetryExhausted",
    "execute_with_infrastructure_retry",
    "parse_structured_supplier_rejection",
]
