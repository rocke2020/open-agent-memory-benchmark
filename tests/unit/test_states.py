from __future__ import annotations

import importlib
from dataclasses import FrozenInstanceError
from types import ModuleType

import pytest


def require_states() -> ModuleType:
    try:
        return importlib.import_module("oamb.contracts.states")
    except ModuleNotFoundError:
        pytest.fail("oamb.contracts.states is not implemented", pytrace=False)


def test_run_and_validation_transitions_are_separate() -> None:
    states = require_states()

    transition = states.apply_transition(
        states.EntityKind.RUN,
        states.RunState.PLANNED,
        states.TransitionEvent.START_PREFLIGHT,
    )
    validation = states.apply_transition(
        states.EntityKind.VALIDATION,
        states.ValidationDisposition.NOT_RUN,
        states.TransitionEvent.VALIDATION_PASS,
    )

    assert transition.next_state == states.RunState.PREFLIGHTING
    assert validation.next_state == states.ValidationDisposition.VALIDATED
    assert states.ValidationDisposition.VALIDATED not in states.RunState


def test_illegal_and_terminal_rewrites_fail() -> None:
    states = require_states()

    with pytest.raises(states.IllegalTransitionError):
        states.apply_transition(
            states.EntityKind.CASE,
            states.CaseState.PENDING,
            states.TransitionEvent.CASE_COMPLETE,
        )
    with pytest.raises(states.IllegalTransitionError):
        states.apply_transition(
            states.EntityKind.RUN,
            states.RunState.FINALIZED,
            states.TransitionEvent.START_RUN,
        )


def test_transition_records_are_immutable() -> None:
    states = require_states()
    transition = states.apply_transition(
        states.EntityKind.INGESTION_PLAN,
        states.IngestionPlanState.PENDING,
        states.TransitionEvent.ALLOCATE_SCOPE,
    )

    with pytest.raises(FrozenInstanceError):
        transition.next_state = states.IngestionPlanState.ERROR


def test_unknown_outcome_terminal_states_are_explicit() -> None:
    states = require_states()

    assert states.AttemptOutcome.UNKNOWN_OUTCOME.value == "unknown_outcome"
    assert (
        states.IngestionPlanState.INTERRUPTED_UNKNOWN_OUTCOME.value == "interrupted_unknown_outcome"
    )
    assert states.ResumeDisposition.REPLACEMENT_RUN_REQUIRED.value == ("replacement_run_required")


def test_interrupted_run_resumes_only_with_resume_safe_disposition() -> None:
    states = require_states()

    with pytest.raises(states.IllegalTransitionError, match="resume_safe"):
        states.apply_transition(
            states.EntityKind.RUN,
            states.RunState.INTERRUPTED,
            states.TransitionEvent.RESUME,
        )
    with pytest.raises(states.IllegalTransitionError, match="resume_safe"):
        states.apply_transition(
            states.EntityKind.RUN,
            states.RunState.INTERRUPTED,
            states.TransitionEvent.RESUME,
            resume_disposition=states.ResumeDisposition.REPLACEMENT_RUN_REQUIRED,
        )

    resumed = states.apply_transition(
        states.EntityKind.RUN,
        states.RunState.INTERRUPTED,
        states.TransitionEvent.RESUME,
        resume_disposition=states.ResumeDisposition.RESUME_SAFE,
    )
    assert resumed.next_state == states.RunState.RUNNING
