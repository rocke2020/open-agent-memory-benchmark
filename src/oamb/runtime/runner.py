"""Single-owner ingestion-plan-before-case scheduler for v0.1."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass


class UnknownExternalOutcome(RuntimeError):
    """Signals a dispatched operation whose terminal receipt cannot be proven."""


Operation = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class IngestionTask:
    ingestion_plan_id: str
    execute: Operation


@dataclass(frozen=True, slots=True)
class CaseTask:
    case_id: str
    ingestion_plan_id: str
    execute: Operation


@dataclass(frozen=True, slots=True)
class SerialRunResult:
    completed_plan_ids: tuple[str, ...]
    completed_case_ids: tuple[str, ...]
    cancelled: bool
    unknown_outcome_task_id: str | None


class SerialRunner:
    """Runs one task at a time and never schedules cases before plan completion."""

    def __init__(self) -> None:
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    async def run(
        self,
        *,
        ingestion_tasks: tuple[IngestionTask, ...],
        case_tasks: tuple[CaseTask, ...],
    ) -> SerialRunResult:
        self._validate_schedule(ingestion_tasks, case_tasks)
        completed_plans: list[str] = []
        completed_cases: list[str] = []

        for ingestion_task in ingestion_tasks:
            if self._cancel_requested:
                return SerialRunResult(tuple(completed_plans), tuple(completed_cases), True, None)
            try:
                await ingestion_task.execute()
            except UnknownExternalOutcome:
                return SerialRunResult(
                    tuple(completed_plans),
                    tuple(completed_cases),
                    False,
                    ingestion_task.ingestion_plan_id,
                )
            completed_plans.append(ingestion_task.ingestion_plan_id)

        completed_plan_set = set(completed_plans)
        for case_task in case_tasks:
            if self._cancel_requested:
                return SerialRunResult(tuple(completed_plans), tuple(completed_cases), True, None)
            if case_task.ingestion_plan_id not in completed_plan_set:
                continue
            try:
                await case_task.execute()
            except UnknownExternalOutcome:
                return SerialRunResult(
                    tuple(completed_plans), tuple(completed_cases), False, case_task.case_id
                )
            completed_cases.append(case_task.case_id)

        return SerialRunResult(
            tuple(completed_plans),
            tuple(completed_cases),
            self._cancel_requested,
            None,
        )

    @staticmethod
    def _validate_schedule(
        ingestion_tasks: tuple[IngestionTask, ...],
        case_tasks: tuple[CaseTask, ...],
    ) -> None:
        plan_ids = tuple(task.ingestion_plan_id for task in ingestion_tasks)
        if any(not plan_id for plan_id in plan_ids):
            raise ValueError("ingestion plan identity must not be empty")
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("duplicate ingestion plan identity")
        plan_id_set = set(plan_ids)
        case_ids = tuple(task.case_id for task in case_tasks)
        if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
            raise ValueError("case identities must be non-empty and unique")
        for task in case_tasks:
            if task.ingestion_plan_id not in plan_id_set:
                raise ValueError(
                    f"case {task.case_id} references unknown ingestion plan "
                    f"{task.ingestion_plan_id}"
                )
