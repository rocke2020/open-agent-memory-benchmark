from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import cast

import pytest
from pydantic import ValidationError

from oamb.contracts.evidence import (
    CapsuleCompositionPartBinding,
    HistoryAttemptRecord,
    HistoryRetryAllowance,
    HistoryRetryCarryRecord,
    HistoryRetryEvent,
    capsule_composition_part_binding_hash,
    history_attempt_id,
    history_retry_carry_id,
    history_retry_event_id,
)
from oamb.contracts.ids import canonical_sha256, ingestion_occurrence_id
from oamb.contracts.ports import RawReferenceHandle, SettledTransientIngestionFailure
from oamb.contracts.schema import parse_contract
from oamb.contracts.states import RunState

NOW = datetime(2026, 9, 6, tzinfo=UTC)
HASHES = tuple(canonical_sha256(["history-rebuild-contract", index]) for index in range(20))
HINDSIGHT_FAILURE = (
    b'{"detail":"Fact extraction failed: 1/1 chunks failed. First failures: '
    b'chunk 0: APIConnectionError: Connection error."}'
)


def _attempt_fields(*, ordinal: int = 1, status: str = "ready") -> dict[str, object]:
    execution_run_id = "execution-run"
    plan_id = HASHES[0]
    occurrence_id = ingestion_occurrence_id(
        execution_run_id,
        "hindsight",
        plan_id,
        history_attempt_ordinal=ordinal,
    )
    fields: dict[str, object] = {
        "schema_name": "history_attempt_record",
        "schema_version": 1,
        "run_id": "capsule-run",
        "execution_run_id": execution_run_id,
        "ingestion_plan_id": plan_id,
        "history_attempt_ordinal": ordinal,
        "ingestion_occurrence_id": occurrence_id,
        "memory_system_id": "hindsight",
        "runtime_binding_hash": HASHES[1],
        "retry_policy_hash": HASHES[2],
        "max_retries_per_operation": 2,
        "history_input_hash": HASHES[3],
        "scope_id": "bank-new",
        "scope_raw_refs": (HASHES[4],),
        "previous_retry_event_id": None if ordinal == 1 else HASHES[5],
        "admission_claim_raw_ref": None if ordinal == 1 else HASHES[9],
        "status": status,
        "operation_attempt_ids": (HASHES[6],),
        "terminal_failure_attempt_id": None,
        "settlement_basis": None,
        "settlement_evidence_refs": (),
        "settlement_status_code": None,
        "settlement_task_id": None,
        "settlement_session_id": None,
        "internal_retry_count": None,
        "failure_kind": None,
        "ingestion_plan_record_hash": HASHES[7],
        "started_at": NOW,
        "ended_at": NOW + timedelta(seconds=1),
    }
    if status == "retryable_failed_settled":
        fields.update(
            scope_id="bank-failed",
            terminal_failure_attempt_id=HASHES[6],
            settlement_basis="hindsight_sync_extraction_drained_v1",
            settlement_evidence_refs=(HASHES[8],),
            settlement_status_code=500,
            internal_retry_count=10,
            failure_kind="transient_upstream_failure",
            ingestion_plan_record_hash=None,
        )
    return fields


def _attempt(**updates: object) -> HistoryAttemptRecord:
    fields = _attempt_fields(
        ordinal=cast(int, updates.pop("history_attempt_ordinal", 1)),
        status=str(updates.pop("status", "ready")),
    )
    fields.update(updates)
    return HistoryAttemptRecord.model_validate(
        {"history_attempt_id": history_attempt_id(fields), **fields}
    )


def _part(index: int) -> CapsuleCompositionPartBinding:
    capsule_id = HASHES[10 + index]
    fields = {
        "schema_name": "capsule_composition_part_binding",
        "schema_version": 1,
        "capsule_id": capsule_id,
        "run_id": f"part-{index}",
        "manifest_sha256": HASHES[12 + index],
        "partition_id": HASHES[14 + index],
        "embedded_root": f"source/parts/{capsule_id}",
        "run_state": RunState.ABORTED,
    }
    return CapsuleCompositionPartBinding.model_validate(
        {"part_binding_hash": capsule_composition_part_binding_hash(fields), **fields}
    )


def test_ingestion_occurrence_first_identity_is_stable_and_successors_are_distinct() -> None:
    current = ingestion_occurrence_id("run", "hindsight", HASHES[0])
    explicit_first = ingestion_occurrence_id(
        "run", "hindsight", HASHES[0], history_attempt_ordinal=1
    )
    successor = ingestion_occurrence_id("run", "hindsight", HASHES[0], history_attempt_ordinal=2)

    assert explicit_first == current
    assert successor == canonical_sha256(
        ["oamb-ingestion-occurrence-v1", "run", "hindsight", HASHES[0], 2]
    )
    assert successor != current
    for invalid in (True, 0, -1):
        with pytest.raises(ValueError):
            ingestion_occurrence_id("run", "hindsight", HASHES[0], history_attempt_ordinal=invalid)


def test_settled_transient_failure_requires_exact_raw_classification_and_native_retries() -> None:
    raw_reference = RawReferenceHandle(sha256=hashlib.sha256(HINDSIGHT_FAILURE).hexdigest())
    failure = SettledTransientIngestionFailure(
        "retain failed after work drained",
        settlement_basis="hindsight_sync_extraction_drained_v1",
        internal_retry_count=10,
        failure_kind="supplier_connection",
        raw_reference=raw_reference,
        raw_response_bytes=HINDSIGHT_FAILURE,
        status_code=500,
    )
    assert failure.failure_kind == "supplier_connection"
    assert failure.internal_retry_count == 10

    for update in (
        {"internal_retry_count": 1},
        {"internal_retry_count": True},
        {"failure_kind": "supplier_rate_limit"},
        {"raw_reference": RawReferenceHandle(sha256=HASHES[0])},
    ):
        arguments = {
            "settlement_basis": "hindsight_sync_extraction_drained_v1",
            "internal_retry_count": 10,
            "failure_kind": "supplier_connection",
            "raw_reference": raw_reference,
            "raw_response_bytes": HINDSIGHT_FAILURE,
            "status_code": 500,
        }
        arguments.update(update)
        with pytest.raises(ValueError):
            SettledTransientIngestionFailure("invalid settled failure", **arguments)  # type: ignore[arg-type]


def test_ready_history_attempt_requires_complete_ready_record_and_fresh_occurrence() -> None:
    record = _attempt()
    assert parse_contract(record.model_dump(mode="json")) == record

    for update in (
        {"scope_id": None},
        {"ingestion_plan_record_hash": None},
        {"terminal_failure_attempt_id": HASHES[6]},
        {"ingestion_occurrence_id": HASHES[9]},
    ):
        fields = _attempt_fields()
        fields.update(update)
        with pytest.raises(ValidationError):
            HistoryAttemptRecord.model_validate(
                {"history_attempt_id": history_attempt_id(fields), **fields}
            )


def test_retryable_failed_history_requires_configured_extraction_retry_evidence() -> None:
    record = _attempt(status="retryable_failed_settled")
    assert record.terminal_failure_attempt_id in record.operation_attempt_ids

    for update in (
        {"settlement_basis": None},
        {"settlement_evidence_refs": ()},
        {"settlement_status_code": None},
        {"internal_retry_count": 1},
        {"terminal_failure_attempt_id": HASHES[9]},
        {"ingestion_plan_record_hash": HASHES[7]},
    ):
        fields = _attempt_fields(status="retryable_failed_settled")
        fields.update(update)
        with pytest.raises(ValidationError):
            HistoryAttemptRecord.model_validate(
                {"history_attempt_id": history_attempt_id(fields), **fields}
            )


def test_openviking_settled_history_requires_bound_task_and_session() -> None:
    fields = _attempt_fields(status="retryable_failed_settled")
    fields.update(
        memory_system_id="openviking",
        settlement_basis="openviking_task_work_drained_v1",
        settlement_status_code=200,
        settlement_task_id="task-id",
        settlement_session_id="session-id",
    )
    fields["ingestion_occurrence_id"] = ingestion_occurrence_id(
        str(fields["execution_run_id"]),
        "openviking",
        str(fields["ingestion_plan_id"]),
    )
    record = HistoryAttemptRecord.model_validate(
        {"history_attempt_id": history_attempt_id(fields), **fields}
    )
    assert record.settlement_task_id == "task-id"

    for missing in ("settlement_task_id", "settlement_session_id"):
        invalid = dict(fields)
        invalid[missing] = None
        with pytest.raises(ValidationError):
            HistoryAttemptRecord.model_validate(
                {"history_attempt_id": history_attempt_id(invalid), **invalid}
            )


def test_non_openviking_and_nonsettled_history_reject_task_session_classifier_fields() -> None:
    for update in (
        {"settlement_task_id": "task-id", "settlement_session_id": "session-id"},
        {"status": "ready", "settlement_status_code": 200},
    ):
        fields = _attempt_fields(
            status="retryable_failed_settled" if "status" not in update else "ready"
        )
        fields.update(update)
        with pytest.raises(ValidationError):
            HistoryAttemptRecord.model_validate(
                {"history_attempt_id": history_attempt_id(fields), **fields}
            )


def test_history_attempt_bounds_ordinals_events_uniqueness_and_time() -> None:
    invalid_updates = (
        {"history_attempt_ordinal": 2, "previous_retry_event_id": None},
        {"history_attempt_ordinal": 4},
        {"operation_attempt_ids": (HASHES[6], HASHES[6])},
        {"scope_raw_refs": (HASHES[4], HASHES[4])},
        {"ended_at": NOW - timedelta(seconds=1)},
    )
    for update in invalid_updates:
        fields = _attempt_fields(ordinal=cast(int, update.get("history_attempt_ordinal", 1)))
        fields.update(update)
        with pytest.raises(ValidationError):
            HistoryAttemptRecord.model_validate(
                {"history_attempt_id": history_attempt_id(fields), **fields}
            )


def test_scheduled_and_exhausted_history_retry_events_are_exact() -> None:
    scheduled_fields = {
        "schema_name": "history_retry_event",
        "schema_version": 1,
        "run_id": "capsule-run",
        "ingestion_plan_id": HASHES[0],
        "failed_history_attempt_id": HASHES[1],
        "failed_history_attempt_ordinal": 1,
        "retry_ordinal": 1,
        "retry_scheduled": True,
        "successor_ingestion_occurrence_id": ingestion_occurrence_id(
            "next-execution", "hindsight", HASHES[0], history_attempt_ordinal=2
        ),
        "successor_execution_run_id": "next-execution",
        "successor_history_attempt_ordinal": 2,
        "retry_policy_hash": HASHES[2],
        "max_retries_per_operation": 2,
        "backoff_seconds": 1,
        "observed_at": NOW,
    }
    scheduled = HistoryRetryEvent.model_validate(
        {
            "history_retry_event_id": history_retry_event_id(scheduled_fields),
            **scheduled_fields,
        }
    )
    assert parse_contract(scheduled.model_dump(mode="json")) == scheduled

    exhausted_fields = dict(scheduled_fields)
    exhausted_fields.update(
        failed_history_attempt_ordinal=3,
        retry_ordinal=3,
        retry_scheduled=False,
        successor_ingestion_occurrence_id=None,
        successor_execution_run_id=None,
        successor_history_attempt_ordinal=None,
        backoff_seconds=None,
    )
    exhausted = HistoryRetryEvent.model_validate(
        {
            "history_retry_event_id": history_retry_event_id(exhausted_fields),
            **exhausted_fields,
        }
    )
    assert exhausted.retry_scheduled is False

    for update in (
        {"backoff_seconds": 2},
        {"successor_history_attempt_ordinal": 3},
        {"successor_ingestion_occurrence_id": None},
    ):
        fields = dict(scheduled_fields)
        fields.update(update)
        with pytest.raises(ValidationError):
            HistoryRetryEvent.model_validate(
                {"history_retry_event_id": history_retry_event_id(fields), **fields}
            )


def test_history_retry_carry_requires_canonical_unique_sources_and_allowances() -> None:
    parts = tuple(sorted((_part(0), _part(1)), key=lambda item: item.capsule_id))
    allowances = (
        HistoryRetryAllowance(
            schema_name="history_retry_allowance",
            schema_version=1,
            ingestion_plan_id=HASHES[0],
            next_history_attempt_ordinal=2,
            execution_run_id="execution-run",
            previous_retry_event_id=HASHES[1],
            predecessor_history_attempt_id=HASHES[2],
            consumed_retries=1,
        ),
    )
    fields = {
        "schema_name": "history_retry_carry_record",
        "schema_version": 1,
        "run_id": "carry-run",
        "source_part_bindings": parts,
        "allowances": allowances,
        "retry_policy_hash": HASHES[3],
        "max_retries_per_operation": 2,
    }
    carry = HistoryRetryCarryRecord.model_validate(
        {"carry_record_id": history_retry_carry_id(fields), **fields}
    )
    assert parse_contract(carry.model_dump(mode="json")) == carry

    for update in (
        {"source_part_bindings": ()},
        {"source_part_bindings": tuple(reversed(parts))},
        {"allowances": allowances + allowances},
    ):
        invalid = dict(fields)
        invalid.update(update)
        with pytest.raises(ValidationError):
            HistoryRetryCarryRecord.model_validate(
                {"carry_record_id": history_retry_carry_id(invalid), **invalid}
            )


def test_history_retry_allowance_closes_consumed_and_next_ordinals() -> None:
    with pytest.raises(ValidationError):
        HistoryRetryAllowance(
            schema_name="history_retry_allowance",
            schema_version=1,
            ingestion_plan_id=HASHES[0],
            next_history_attempt_ordinal=3,
            execution_run_id="execution-run",
            previous_retry_event_id=HASHES[1],
            predecessor_history_attempt_id=HASHES[2],
            consumed_retries=1,
        )
