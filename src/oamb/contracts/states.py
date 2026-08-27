"""Central execution and validation transition tables."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias


class EntityKind(StrEnum):
    RUN = "run"
    VALIDATION = "validation"
    INGESTION_PLAN = "ingestion_plan"
    CASE = "case"


class RunState(StrEnum):
    PLANNED = "planned"
    PREFLIGHTING = "preflighting"
    READY = "ready"
    RUNNING = "running"
    FINALIZING = "finalizing"
    FINALIZED = "finalized"
    REJECTED = "rejected"
    INTERRUPTED = "interrupted"
    ABORTED = "aborted"


class ValidationDisposition(StrEnum):
    NOT_RUN = "not_run"
    VALIDATED = "validated"
    INVALID = "invalid"


class IngestionPlanState(StrEnum):
    PENDING = "pending"
    SCOPE_ALLOCATED = "scope_allocated"
    INGESTING = "ingesting"
    WAITING_READY = "waiting_ready"
    READY = "ready"
    SEALED = "sealed"
    ERROR = "error"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    INTERRUPTED_UNKNOWN_OUTCOME = "interrupted_unknown_outcome"


class CaseState(StrEnum):
    PENDING = "pending"
    RETRIEVING = "retrieving"
    NORMALIZING = "normalizing"
    ANSWERING = "answering"
    EVALUATING = "evaluating"
    SEALING = "sealing"
    COMPLETED = "completed"
    ERROR = "error"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"


class AttemptOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"
    UNKNOWN_OUTCOME = "unknown_outcome"


class ResumeDisposition(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    RESUME_SAFE = "resume_safe"
    REPLACEMENT_RUN_REQUIRED = "replacement_run_required"


class IndexContribution(StrEnum):
    FINAL = "final"
    SUPERSEDED = "superseded"
    NONE = "none"
    NOT_APPLICABLE = "not_applicable"


class TransitionEvent(StrEnum):
    START_PREFLIGHT = "start_preflight"
    PREFLIGHT_READY = "preflight_ready"
    REJECT = "reject"
    START_RUN = "start_run"
    FINALIZE = "finalize"
    SEAL_RUN = "seal_run"
    INTERRUPT = "interrupt"
    ABORT = "abort"
    RESUME = "resume"
    VALIDATION_PASS = "validation_pass"
    VALIDATION_FAIL = "validation_fail"
    ALLOCATE_SCOPE = "allocate_scope"
    START_INGEST = "start_ingest"
    WAIT_READY = "wait_ready"
    MARK_READY = "mark_ready"
    SEAL_PLAN = "seal_plan"
    START_RETRIEVE = "start_retrieve"
    NORMALIZE = "normalize"
    START_ANSWER = "start_answer"
    START_EVALUATE = "start_evaluate"
    START_SEAL = "start_seal"
    CASE_COMPLETE = "case_complete"
    FAIL = "fail"
    MARK_UNSUPPORTED = "mark_unsupported"
    CANCEL = "cancel"
    EXHAUST_BUDGET = "exhaust_budget"
    MARK_UNKNOWN_OUTCOME = "mark_unknown_outcome"


StateValue: TypeAlias = RunState | ValidationDisposition | IngestionPlanState | CaseState


@dataclass(frozen=True, slots=True)
class StateTransitionRecord:
    entity_kind: EntityKind
    previous_state: StateValue
    event: TransitionEvent
    next_state: StateValue


class IllegalTransitionError(ValueError):
    pass


_TRANSITIONS: dict[tuple[EntityKind, StateValue, TransitionEvent], StateValue] = {
    (EntityKind.RUN, RunState.PLANNED, TransitionEvent.START_PREFLIGHT): RunState.PREFLIGHTING,
    (EntityKind.RUN, RunState.PREFLIGHTING, TransitionEvent.PREFLIGHT_READY): RunState.READY,
    (EntityKind.RUN, RunState.PREFLIGHTING, TransitionEvent.REJECT): RunState.REJECTED,
    (EntityKind.RUN, RunState.READY, TransitionEvent.START_RUN): RunState.RUNNING,
    (EntityKind.RUN, RunState.RUNNING, TransitionEvent.FINALIZE): RunState.FINALIZING,
    (EntityKind.RUN, RunState.FINALIZING, TransitionEvent.SEAL_RUN): RunState.FINALIZED,
    (EntityKind.RUN, RunState.RUNNING, TransitionEvent.INTERRUPT): RunState.INTERRUPTED,
    (EntityKind.RUN, RunState.RUNNING, TransitionEvent.ABORT): RunState.ABORTED,
    (EntityKind.RUN, RunState.INTERRUPTED, TransitionEvent.RESUME): RunState.RUNNING,
    (
        EntityKind.VALIDATION,
        ValidationDisposition.NOT_RUN,
        TransitionEvent.VALIDATION_PASS,
    ): ValidationDisposition.VALIDATED,
    (
        EntityKind.VALIDATION,
        ValidationDisposition.NOT_RUN,
        TransitionEvent.VALIDATION_FAIL,
    ): ValidationDisposition.INVALID,
    (
        EntityKind.INGESTION_PLAN,
        IngestionPlanState.PENDING,
        TransitionEvent.ALLOCATE_SCOPE,
    ): IngestionPlanState.SCOPE_ALLOCATED,
    (
        EntityKind.INGESTION_PLAN,
        IngestionPlanState.SCOPE_ALLOCATED,
        TransitionEvent.START_INGEST,
    ): IngestionPlanState.INGESTING,
    (
        EntityKind.INGESTION_PLAN,
        IngestionPlanState.INGESTING,
        TransitionEvent.WAIT_READY,
    ): IngestionPlanState.WAITING_READY,
    (
        EntityKind.INGESTION_PLAN,
        IngestionPlanState.WAITING_READY,
        TransitionEvent.MARK_READY,
    ): IngestionPlanState.READY,
    (
        EntityKind.INGESTION_PLAN,
        IngestionPlanState.READY,
        TransitionEvent.SEAL_PLAN,
    ): IngestionPlanState.SEALED,
    (EntityKind.CASE, CaseState.PENDING, TransitionEvent.START_RETRIEVE): CaseState.RETRIEVING,
    (EntityKind.CASE, CaseState.RETRIEVING, TransitionEvent.NORMALIZE): CaseState.NORMALIZING,
    (EntityKind.CASE, CaseState.NORMALIZING, TransitionEvent.START_ANSWER): CaseState.ANSWERING,
    (EntityKind.CASE, CaseState.ANSWERING, TransitionEvent.START_EVALUATE): CaseState.EVALUATING,
    (EntityKind.CASE, CaseState.EVALUATING, TransitionEvent.START_SEAL): CaseState.SEALING,
    (EntityKind.CASE, CaseState.SEALING, TransitionEvent.CASE_COMPLETE): CaseState.COMPLETED,
}


def _add_terminal_transitions() -> None:
    ingestion_active = (
        IngestionPlanState.SCOPE_ALLOCATED,
        IngestionPlanState.INGESTING,
        IngestionPlanState.WAITING_READY,
        IngestionPlanState.READY,
    )
    for ingestion_state in ingestion_active:
        _TRANSITIONS[(EntityKind.INGESTION_PLAN, ingestion_state, TransitionEvent.FAIL)] = (
            IngestionPlanState.ERROR
        )
        _TRANSITIONS[
            (EntityKind.INGESTION_PLAN, ingestion_state, TransitionEvent.MARK_UNSUPPORTED)
        ] = IngestionPlanState.UNSUPPORTED
        _TRANSITIONS[(EntityKind.INGESTION_PLAN, ingestion_state, TransitionEvent.CANCEL)] = (
            IngestionPlanState.CANCELLED
        )
        _TRANSITIONS[
            (EntityKind.INGESTION_PLAN, ingestion_state, TransitionEvent.EXHAUST_BUDGET)
        ] = IngestionPlanState.BUDGET_EXCEEDED
        _TRANSITIONS[
            (EntityKind.INGESTION_PLAN, ingestion_state, TransitionEvent.MARK_UNKNOWN_OUTCOME)
        ] = IngestionPlanState.INTERRUPTED_UNKNOWN_OUTCOME

    case_active = (
        CaseState.PENDING,
        CaseState.RETRIEVING,
        CaseState.NORMALIZING,
        CaseState.ANSWERING,
        CaseState.EVALUATING,
        CaseState.SEALING,
    )
    for case_state in case_active:
        _TRANSITIONS[(EntityKind.CASE, case_state, TransitionEvent.FAIL)] = CaseState.ERROR
        _TRANSITIONS[(EntityKind.CASE, case_state, TransitionEvent.MARK_UNSUPPORTED)] = (
            CaseState.UNSUPPORTED
        )
        _TRANSITIONS[(EntityKind.CASE, case_state, TransitionEvent.CANCEL)] = CaseState.CANCELLED
        _TRANSITIONS[(EntityKind.CASE, case_state, TransitionEvent.EXHAUST_BUDGET)] = (
            CaseState.BUDGET_EXCEEDED
        )


_add_terminal_transitions()


def apply_transition(
    entity_kind: EntityKind,
    current_state: StateValue,
    event: TransitionEvent,
    *,
    resume_disposition: ResumeDisposition | None = None,
) -> StateTransitionRecord:
    if (
        entity_kind == EntityKind.RUN
        and current_state == RunState.INTERRUPTED
        and event == TransitionEvent.RESUME
        and resume_disposition != ResumeDisposition.RESUME_SAFE
    ):
        raise IllegalTransitionError("interrupted run resume requires resume_safe disposition")
    key = (entity_kind, current_state, event)
    try:
        next_state = _TRANSITIONS[key]
    except KeyError as exc:
        raise IllegalTransitionError(
            f"illegal {entity_kind.value} transition: {current_state.value} + {event.value}"
        ) from exc
    return StateTransitionRecord(entity_kind, current_state, event, next_state)
