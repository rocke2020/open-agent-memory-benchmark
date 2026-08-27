from __future__ import annotations

from decimal import Decimal

import pytest


class ManualClock:
    def __init__(self) -> None:
        self.value = Decimal("0")

    def __call__(self) -> Decimal:
        return self.value

    def advance(self, seconds: str) -> None:
        self.value += Decimal(seconds)


def test_projection_intervals_do_not_inflate_provider_request_latency() -> None:
    from oamb.runtime.timing import capture_query_timing

    clock = ManualClock()

    def before_projection() -> str:
        clock.advance("2")
        return "before"

    def provider_request() -> str:
        clock.advance("3")
        return "provider-result"

    def after_projection() -> str:
        clock.advance("4")
        return "after"

    captured = capture_query_timing(
        before_projection=before_projection,
        provider_request=provider_request,
        after_projection=after_projection,
        clock=clock,
    )

    assert captured.result == "provider-result"
    assert captured.provider_request.dimension_id == "provider_request_wall_seconds_v1"
    assert captured.provider_request.seconds == Decimal("3")
    assert captured.projection_verification.dimension_id == (
        "projection_verification_wall_seconds_v1"
    )
    assert captured.projection_verification.seconds == Decimal("6")
    assert captured.case_total.dimension_id == "case_wall_seconds_v1"
    assert captured.case_total.seconds == Decimal("9")
    assert captured.projection_intervals[0].ended_at <= captured.provider_interval.started_at
    assert captured.provider_interval.ended_at <= captured.projection_intervals[1].started_at


def test_slow_projection_changes_projection_and_case_but_not_provider_interval() -> None:
    from oamb.runtime.timing import capture_query_timing

    def run(projection_seconds: str) -> tuple[Decimal, Decimal, Decimal]:
        clock = ManualClock()

        def projection() -> None:
            clock.advance(projection_seconds)

        def provider() -> None:
            clock.advance("3")

        captured = capture_query_timing(
            before_projection=projection,
            provider_request=provider,
            after_projection=projection,
            clock=clock,
        )
        return (
            captured.provider_request.seconds,
            captured.projection_verification.seconds,
            captured.case_total.seconds,
        )

    fast = run("2")
    slow = run("20")

    assert fast == (Decimal("3"), Decimal("4"), Decimal("7"))
    assert slow == (Decimal("3"), Decimal("40"), Decimal("43"))


def test_case_timing_rejects_globally_backwards_monotonic_boundaries() -> None:
    from oamb.runtime.timing import capture_query_timing

    values = iter(Decimal(value) for value in ("10", "11", "12", "13", "14", "15", "16", "9"))

    with pytest.raises(ValueError, match="globally out of order"):
        capture_query_timing(
            before_projection=lambda: None,
            provider_request=lambda: None,
            after_projection=lambda: None,
            clock=lambda: next(values),
        )
