"""Monotonic timing boundaries that keep projection overhead out of provider latency."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Generic, TypeVar, cast

ResultValue = TypeVar("ResultValue")

PROVIDER_REQUEST_WALL_SECONDS = "provider_request_wall_seconds_v1"
PROJECTION_VERIFICATION_WALL_SECONDS = "projection_verification_wall_seconds_v1"
CASE_WALL_SECONDS = "case_wall_seconds_v1"


@dataclass(frozen=True, slots=True)
class MonotonicInterval:
    started_at: Decimal
    ended_at: Decimal

    @property
    def seconds(self) -> Decimal:
        return self.ended_at - self.started_at


@dataclass(frozen=True, slots=True)
class TimingMeasurement:
    dimension_id: str
    seconds: Decimal


@dataclass(frozen=True, slots=True)
class CapturedQueryTiming(Generic[ResultValue]):
    result: ResultValue
    provider_request: TimingMeasurement
    projection_verification: TimingMeasurement
    case_total: TimingMeasurement
    provider_interval: MonotonicInterval
    projection_intervals: tuple[MonotonicInterval, MonotonicInterval]


@dataclass(frozen=True, slots=True)
class CapturedQueryOutcome(Generic[ResultValue]):
    timing: CapturedQueryTiming[ResultValue | None]
    provider_error: BaseException | None
    post_projection_error: BaseException | None


def _capture_interval(
    operation: Callable[[], ResultValue], clock: Callable[[], Decimal]
) -> tuple[ResultValue, MonotonicInterval]:
    started_at = clock()
    result = operation()
    ended_at = clock()
    if ended_at < started_at:
        raise ValueError("monotonic clock moved backwards")
    return result, MonotonicInterval(started_at=started_at, ended_at=ended_at)


async def _capture_async_interval(
    operation: Callable[[], Awaitable[ResultValue]],
    clock: Callable[[], Decimal],
) -> tuple[ResultValue, MonotonicInterval]:
    started_at = clock()
    result = await operation()
    ended_at = clock()
    if ended_at < started_at:
        raise ValueError("monotonic clock moved backwards")
    return result, MonotonicInterval(started_at=started_at, ended_at=ended_at)


async def _capture_async_outcome_interval(
    operation: Callable[[], Awaitable[ResultValue]],
    clock: Callable[[], Decimal],
) -> tuple[ResultValue | None, BaseException | None, MonotonicInterval]:
    started_at = clock()
    try:
        result = await operation()
    except (Exception, asyncio.CancelledError) as error:
        ended_at = clock()
        if ended_at < started_at:
            raise ValueError("monotonic clock moved backwards") from error
        return None, error, MonotonicInterval(started_at=started_at, ended_at=ended_at)
    ended_at = clock()
    if ended_at < started_at:
        raise ValueError("monotonic clock moved backwards")
    return result, None, MonotonicInterval(started_at=started_at, ended_at=ended_at)


def _build_query_timing(
    *,
    case_started_at: Decimal,
    before_interval: MonotonicInterval,
    result: ResultValue,
    provider_interval: MonotonicInterval,
    after_interval: MonotonicInterval,
    case_ended_at: Decimal,
) -> CapturedQueryTiming[ResultValue]:
    ordered_boundaries = (
        case_started_at,
        before_interval.started_at,
        before_interval.ended_at,
        provider_interval.started_at,
        provider_interval.ended_at,
        after_interval.started_at,
        after_interval.ended_at,
        case_ended_at,
    )
    if any(
        right < left
        for left, right in zip(ordered_boundaries, ordered_boundaries[1:], strict=False)
    ):
        raise ValueError("monotonic timing boundaries are globally out of order")
    case_interval = MonotonicInterval(case_started_at, case_ended_at)
    projection_seconds = before_interval.seconds + after_interval.seconds
    return CapturedQueryTiming(
        result=result,
        provider_request=TimingMeasurement(
            PROVIDER_REQUEST_WALL_SECONDS, provider_interval.seconds
        ),
        projection_verification=TimingMeasurement(
            PROJECTION_VERIFICATION_WALL_SECONDS, projection_seconds
        ),
        case_total=TimingMeasurement(CASE_WALL_SECONDS, case_interval.seconds),
        provider_interval=provider_interval,
        projection_intervals=(before_interval, after_interval),
    )


def capture_query_timing(
    *,
    before_projection: Callable[[], object],
    provider_request: Callable[[], ResultValue],
    after_projection: Callable[[], object],
    clock: Callable[[], Decimal],
) -> CapturedQueryTiming[ResultValue]:
    case_started_at = clock()
    _, before_interval = _capture_interval(before_projection, clock)
    result, provider_interval = _capture_interval(provider_request, clock)
    _, after_interval = _capture_interval(after_projection, clock)
    case_ended_at = clock()
    return _build_query_timing(
        case_started_at=case_started_at,
        result=result,
        before_interval=before_interval,
        provider_interval=provider_interval,
        after_interval=after_interval,
        case_ended_at=case_ended_at,
    )


async def capture_async_query_timing(
    *,
    before_projection: Callable[[], Awaitable[object]],
    provider_request: Callable[[], Awaitable[ResultValue]],
    after_projection: Callable[[], Awaitable[object]],
    clock: Callable[[], Decimal],
) -> CapturedQueryTiming[ResultValue]:
    outcome = await capture_async_query_outcome(
        before_projection=before_projection,
        provider_request=provider_request,
        after_projection=after_projection,
        clock=clock,
    )
    if outcome.provider_error is not None:
        raise outcome.provider_error
    if outcome.post_projection_error is not None:
        raise outcome.post_projection_error
    return cast(CapturedQueryTiming[ResultValue], outcome.timing)


async def capture_async_query_outcome(
    *,
    before_projection: Callable[[], Awaitable[object]],
    provider_request: Callable[[], Awaitable[ResultValue]],
    after_projection: Callable[[], Awaitable[object]],
    clock: Callable[[], Decimal],
) -> CapturedQueryOutcome[ResultValue]:
    case_started_at = clock()
    _, before_interval = await _capture_async_interval(before_projection, clock)
    result, provider_error, provider_interval = await _capture_async_outcome_interval(
        provider_request,
        clock,
    )
    _, post_projection_error, after_interval = await _capture_async_outcome_interval(
        after_projection,
        clock,
    )
    case_ended_at = clock()
    return CapturedQueryOutcome(
        timing=_build_query_timing(
            case_started_at=case_started_at,
            before_interval=before_interval,
            result=result,
            provider_interval=provider_interval,
            after_interval=after_interval,
            case_ended_at=case_ended_at,
        ),
        provider_error=provider_error,
        post_projection_error=post_projection_error,
    )
