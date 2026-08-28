from __future__ import annotations

import asyncio

import pytest

from oamb.runtime.runner import (
    CaseTask,
    IngestionPlanUnavailable,
    IngestionTask,
    SerialRunner,
    UnknownExternalOutcome,
)


def test_serial_runner_completes_all_plans_before_cases_with_one_active_task() -> None:
    events: list[str] = []
    active = 0
    maximum_active = 0

    async def record(name: str) -> None:
        nonlocal active, maximum_active
        active += 1
        maximum_active = max(maximum_active, active)
        await asyncio.sleep(0)
        events.append(name)
        active -= 1

    runner = SerialRunner()
    result = asyncio.run(
        runner.run(
            ingestion_tasks=(
                IngestionTask("plan-a", lambda: record("plan-a")),
                IngestionTask("plan-b", lambda: record("plan-b")),
            ),
            case_tasks=(
                CaseTask("case-a", "plan-a", lambda: record("case-a")),
                CaseTask("case-b", "plan-b", lambda: record("case-b")),
            ),
        )
    )

    assert events == ["plan-a", "plan-b", "case-a", "case-b"]
    assert maximum_active == 1
    assert result.completed_plan_ids == ("plan-a", "plan-b")
    assert result.completed_case_ids == ("case-a", "case-b")


def test_serial_runner_rejects_duplicate_or_unknown_plan_parent_before_execution() -> None:
    calls = 0

    async def called() -> None:
        nonlocal calls
        calls += 1

    runner = SerialRunner()
    with pytest.raises(ValueError, match="duplicate ingestion plan"):
        asyncio.run(
            runner.run(
                ingestion_tasks=(
                    IngestionTask("plan-a", called),
                    IngestionTask("plan-a", called),
                ),
                case_tasks=(),
            )
        )
    with pytest.raises(ValueError, match="unknown ingestion plan"):
        asyncio.run(
            runner.run(
                ingestion_tasks=(IngestionTask("plan-a", called),),
                case_tasks=(CaseTask("case-a", "plan-b", called),),
            )
        )
    assert calls == 0


def test_unknown_external_outcome_stops_scheduling_and_is_never_retried() -> None:
    calls: list[str] = []

    async def unknown() -> None:
        calls.append("unknown")
        raise UnknownExternalOutcome("receipt lost")

    async def later() -> None:
        calls.append("later")

    result = asyncio.run(
        SerialRunner().run(
            ingestion_tasks=(
                IngestionTask("plan-a", unknown),
                IngestionTask("plan-b", later),
            ),
            case_tasks=(),
        )
    )

    assert calls == ["unknown"]
    assert result.unknown_outcome_task_id == "plan-a"
    assert result.completed_plan_ids == ()


def test_cancellation_stops_new_scheduling_after_active_task_returns() -> None:
    calls: list[str] = []
    runner = SerialRunner()

    async def first() -> None:
        calls.append("first")
        runner.request_cancel()

    async def second() -> None:
        calls.append("second")

    result = asyncio.run(
        runner.run(
            ingestion_tasks=(
                IngestionTask("plan-a", first),
                IngestionTask("plan-b", second),
            ),
            case_tasks=(),
        )
    )

    assert calls == ["first"]
    assert result.cancelled is True


def test_unavailable_plan_skips_only_its_cases_and_continues_ready_sibling() -> None:
    calls: list[str] = []

    async def unavailable() -> None:
        calls.append("plan-unavailable")
        raise IngestionPlanUnavailable("readiness did not close")

    async def complete(name: str) -> None:
        calls.append(name)

    result = asyncio.run(
        SerialRunner().run(
            ingestion_tasks=(
                IngestionTask("plan-unavailable", unavailable),
                IngestionTask("plan-ready", lambda: complete("plan-ready")),
            ),
            case_tasks=(
                CaseTask("case-unavailable", "plan-unavailable", lambda: complete("bad")),
                CaseTask("case-ready", "plan-ready", lambda: complete("case-ready")),
            ),
        )
    )

    assert calls == ["plan-unavailable", "plan-ready", "case-ready"]
    assert result.completed_plan_ids == ("plan-ready",)
    assert result.unavailable_plan_ids == ("plan-unavailable",)
    assert result.completed_case_ids == ("case-ready",)
