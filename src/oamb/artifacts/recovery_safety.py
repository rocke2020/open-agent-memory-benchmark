"""Shared fail-closed checks for fresh-scope whole-history recovery."""

from __future__ import annotations

from collections.abc import Sequence

from oamb.contracts.evidence import (
    AttemptReceiptKind,
    AttemptReceiptRecord,
    AttemptRecordV2,
    AttemptRecordV4,
    HistoryAttemptRecord,
)
from oamb.contracts.states import AttemptOutcome


def history_group_is_terminal_known(
    history_attempts: Sequence[HistoryAttemptRecord],
    operation_attempts: Sequence[AttemptRecordV2 | AttemptRecordV4],
    receipts: Sequence[AttemptReceiptRecord],
) -> bool:
    """Prove every admitted history operation ended without an unknown write."""

    if not history_attempts or any(
        item.status not in {"ready", "failed", "cancelled"} for item in history_attempts
    ):
        return False
    for history in history_attempts:
        matching_attempts = tuple(
            attempt
            for attempt in operation_attempts
            if attempt.parent_kind == "ingestion_plan"
            and attempt.parent_id == history.ingestion_occurrence_id
        )
        matching_receipts = tuple(
            receipt for receipt in receipts if receipt.attempt_id in history.operation_attempt_ids
        )
        if (
            len(matching_attempts) != len(history.operation_attempt_ids)
            or {item.attempt_id for item in matching_attempts} != set(history.operation_attempt_ids)
            or len(matching_receipts) != len(history.operation_attempt_ids)
            or {item.attempt_id for item in matching_receipts} != set(history.operation_attempt_ids)
        ):
            return False
        receipt_by_attempt = {item.attempt_id: item for item in matching_receipts}
        for attempt in matching_attempts:
            receipt = receipt_by_attempt[attempt.attempt_id]
            if (
                attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME
                or receipt.receipt_kind == AttemptReceiptKind.UNKNOWN_OUTCOME
                or attempt.outcome == AttemptOutcome.CANCELLED
                and receipt.failure_kind != "cancelled_before_dispatch"
            ):
                return False
    return True


__all__ = ["history_group_is_terminal_known"]
