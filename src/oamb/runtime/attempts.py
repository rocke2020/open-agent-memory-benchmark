"""Durable reservation, intent, receipt, and terminal-attempt ordering."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from oamb.contracts.base import StrictContract
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptIntentRecordV2,
    AttemptReceiptRecord,
    AttemptRecordV2,
    AttemptRecordV3,
    BudgetReservationRecord,
    BudgetReservationRecordV2,
    OccurrenceClaimRecord,
)
from oamb.contracts.ids import canonical_json_bytes
from oamb.contracts.ports import ArtifactStorePort, RawPayloadSealRequest
from oamb.contracts.specifications import BudgetScopeKindV2, BudgetScopeKindV3
from oamb.contracts.states import AttemptOutcome

from .budget import (
    BudgetAmount,
    BudgetLedger,
    ReservationRequest,
)
from .source_records import seal_source_contract


class AttemptOrderingError(ValueError):
    """Raised before a write when transaction records do not form one attempt."""


class ProviderLifecyclePort(Protocol):
    def mark_attempt_dispatched(self, *, attempt_id: str, intent_record_hash: str) -> None: ...

    def clear_attempt_after_receipt(self, expected_intent_record_hash: str) -> None: ...


@dataclass(frozen=True, slots=True)
class _PreparedAttempt:
    intent: AttemptIntentRecord | AttemptIntentRecordV2
    reservation_id: str
    intent_record_hash: str
    dispatched: bool = False
    receipt_sealed: bool = False


class AttemptCoordinator:
    """Uses only the artifact port and never dispatches before durable intent."""

    def __init__(
        self,
        store: ArtifactStorePort,
        budget: BudgetLedger,
        *,
        provider_lifecycle: ProviderLifecyclePort | None = None,
    ) -> None:
        self._store = store
        self._budget = budget
        self._provider_lifecycle = provider_lifecycle
        self._prepared: dict[str, _PreparedAttempt] = {}

    def prepare(
        self,
        *,
        claim: OccurrenceClaimRecord,
        reservation: BudgetReservationRecord | BudgetReservationRecordV2,
        intent: AttemptIntentRecord | AttemptIntentRecordV2,
        maximum: BudgetAmount,
    ) -> None:
        if isinstance(reservation, BudgetReservationRecordV2) or isinstance(
            intent, AttemptIntentRecordV2
        ):
            raise AttemptOrderingError(
                "memory conformance requires explicit owner allocations from the closed "
                "composition root"
            )
        self._validate_prepare(claim, reservation, intent, maximum)
        assert isinstance(reservation, BudgetReservationRecord)
        reservation_request = ReservationRequest(
            reservation_id=reservation.reservation_id,
            maximum=maximum,
            role_binding_id=reservation.role_binding_id,
        )
        self._budget.reserve(reservation_request)
        self._seal("occurrence-claims", claim.claim_id, claim)
        self._seal("budget-reservations", reservation.reservation_id, reservation)
        self._seal("attempt-intents", intent.attempt_id, intent)
        intent_record_hash = hashlib.sha256(canonical_json_bytes(intent)).hexdigest()
        self._prepared[intent.attempt_id] = _PreparedAttempt(
            intent=intent,
            reservation_id=reservation.reservation_id,
            intent_record_hash=intent_record_hash,
        )

    def mark_dispatched(self, attempt_id: str) -> None:
        prepared = self._require_prepared(attempt_id)
        if prepared.dispatched or prepared.receipt_sealed:
            raise AttemptOrderingError("attempt is already dispatched or received")
        if self._provider_lifecycle is not None:
            self._provider_lifecycle.mark_attempt_dispatched(
                attempt_id=attempt_id,
                intent_record_hash=prepared.intent_record_hash,
            )
        self._prepared[attempt_id] = _PreparedAttempt(
            intent=prepared.intent,
            reservation_id=prepared.reservation_id,
            intent_record_hash=prepared.intent_record_hash,
            dispatched=True,
        )

    def record_receipt(
        self,
        *,
        receipt: AttemptReceiptRecord,
        raw_payload: bytes,
        media_type: str,
        observed: BudgetAmount,
    ) -> None:
        prepared = self._require_prepared(receipt.attempt_id)
        if prepared.receipt_sealed:
            raise AttemptOrderingError("attempt receipt is already sealed")
        if self._provider_lifecycle is not None and not prepared.dispatched:
            raise AttemptOrderingError("provider attempt receipt precedes dispatch guard")
        expected_raw_hash = receipt.raw_response_ref or receipt.raw_error_ref
        actual_raw_hash = hashlib.sha256(raw_payload).hexdigest()
        if expected_raw_hash != actual_raw_hash:
            raise AttemptOrderingError("attempt receipt raw hash does not match payload")
        self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=actual_raw_hash,
                media_type=media_type,
                compression="gzip",
                payload_bytes=raw_payload,
            )
        )
        self._seal("attempt-receipts", receipt.attempt_id, receipt)
        if self._provider_lifecycle is not None:
            self._provider_lifecycle.clear_attempt_after_receipt(prepared.intent_record_hash)
        self._budget.commit(prepared.reservation_id, observed=observed)
        self._prepared[receipt.attempt_id] = _PreparedAttempt(
            intent=prepared.intent,
            reservation_id=prepared.reservation_id,
            intent_record_hash=prepared.intent_record_hash,
            dispatched=prepared.dispatched,
            receipt_sealed=True,
        )

    def seal_terminal(self, attempt: AttemptRecordV2 | AttemptRecordV3) -> None:
        prepared = self._require_prepared(attempt.attempt_id)
        self._validate_terminal(prepared, attempt)
        self._seal("attempts", attempt.attempt_id, attempt)

    def mark_unknown(
        self,
        *,
        reservation_id: str,
        attempt: AttemptRecordV2 | AttemptRecordV3,
    ) -> None:
        prepared = self._require_prepared(attempt.attempt_id)
        if reservation_id != prepared.reservation_id:
            raise AttemptOrderingError("unknown outcome names a different reservation")
        if prepared.receipt_sealed:
            raise AttemptOrderingError("received attempt cannot become unknown outcome")
        if self._provider_lifecycle is not None and not prepared.dispatched:
            raise AttemptOrderingError("provider attempt was not dispatched")
        if attempt.outcome != AttemptOutcome.UNKNOWN_OUTCOME:
            raise AttemptOrderingError("unknown outcome requires the matching terminal outcome")
        if attempt.raw_response_ref is not None or attempt.raw_error_ref is not None:
            raise AttemptOrderingError("unknown outcome cannot name a raw receipt")
        self._validate_terminal(prepared, attempt)
        self._budget.mark_unknown(reservation_id)
        self._seal("attempts", attempt.attempt_id, attempt)

    def _seal(self, collection: str, record_id: str, record: StrictContract) -> None:
        seal_source_contract(
            self._store,
            relative_path=f"source/{collection}/{record_id}.json",
            record_id=record_id,
            record=record,
        )

    def _require_prepared(self, attempt_id: str) -> _PreparedAttempt:
        try:
            return self._prepared[attempt_id]
        except KeyError as exc:
            raise AttemptOrderingError("attempt was not durably prepared") from exc

    @staticmethod
    def _validate_prepare(
        claim: OccurrenceClaimRecord,
        reservation: BudgetReservationRecord | BudgetReservationRecordV2,
        intent: AttemptIntentRecord | AttemptIntentRecordV2,
        maximum: BudgetAmount,
    ) -> None:
        legacy_pair = isinstance(reservation, BudgetReservationRecord) and isinstance(
            intent, AttemptIntentRecord
        )
        conformance_pair = isinstance(reservation, BudgetReservationRecordV2) and isinstance(
            intent, AttemptIntentRecordV2
        )
        if not (legacy_pair or conformance_pair):
            raise AttemptOrderingError("reservation and intent contract versions differ")
        if intent.attempt_id != reservation.attempt_id:
            raise AttemptOrderingError("intent and reservation attempt IDs differ")
        if intent.claim_id != claim.claim_id or intent.reservation_id != reservation.reservation_id:
            raise AttemptOrderingError("intent references a different claim or reservation")
        if intent.parent_id != claim.occurrence_id:
            raise AttemptOrderingError("intent parent differs from claimed occurrence")
        if intent.stage != claim.stage:
            raise AttemptOrderingError("intent stage differs from claimed occurrence")
        if intent.reconciliation_capability != claim.reconciliation_capability:
            raise AttemptOrderingError("intent reconciliation differs from claimed occurrence")
        if intent.request_fingerprint != claim.request_fingerprint:
            raise AttemptOrderingError("intent and claim request fingerprints differ")
        expected_scope_kind = (
            BudgetScopeKindV3.MEMORY_CONFORMANCE
            if conformance_pair
            else {
                "ingestion_plan": BudgetScopeKindV2.RUN,
                "case": BudgetScopeKindV2.RUN,
                "phase_review": BudgetScopeKindV2.PHASE_REVIEW,
                "model_readiness": BudgetScopeKindV2.MODEL_READINESS,
            }[intent.parent_kind]
        )
        if (
            reservation.scope_kind != expected_scope_kind
            or reservation.scope_id != intent.parent_id
        ):
            raise AttemptOrderingError("reservation scope differs from the attempt parent")
        if intent.role_binding_id != reservation.role_binding_id:
            raise AttemptOrderingError("intent and reservation roles differ")
        if conformance_pair:
            assert isinstance(reservation, BudgetReservationRecordV2)
            assert isinstance(intent, AttemptIntentRecordV2)
            if (
                intent.dispatch_owner_kind != reservation.dispatch_owner_kind
                or intent.provider_operation_ceiling_id != reservation.provider_operation_ceiling_id
                or intent.internal_usage_role_binding_ids
                != reservation.internal_usage_role_binding_ids
            ):
                raise AttemptOrderingError(
                    "intent and reservation discriminated budget owners differ"
                )
        expected_resources = tuple(
            (item.dimension_id, item.maximum, item.unit)
            for item in reservation.reserved_resource_ceilings
        )
        if maximum.resources != expected_resources:
            raise AttemptOrderingError("reservation resource ceilings differ from runtime maximum")
        if maximum.attempts != reservation.reserved_attempts:
            raise AttemptOrderingError("reservation attempt count differs from runtime maximum")
        if maximum.input_tokens != reservation.reserved_input_tokens:
            raise AttemptOrderingError("reservation input tokens differ from runtime maximum")
        if maximum.output_tokens != reservation.reserved_output_tokens:
            raise AttemptOrderingError("reservation output tokens differ from runtime maximum")
        if maximum.wall_seconds != reservation.reserved_dispatch_wall_seconds:
            raise AttemptOrderingError("reservation wall time differs from runtime maximum")
        if maximum.cost != (reservation.reserved_cost or 0):
            raise AttemptOrderingError("reservation cost differs from runtime maximum")
        if sum(item[3] for item in maximum.provider_units) != reservation.reserved_provider_units:
            raise AttemptOrderingError("reservation provider units differ from runtime maximum")

    @staticmethod
    def _validate_terminal(
        prepared: _PreparedAttempt,
        attempt: AttemptRecordV2 | AttemptRecordV3,
    ) -> None:
        intent = prepared.intent
        if isinstance(intent, AttemptIntentRecordV2) != isinstance(attempt, AttemptRecordV3):
            raise AttemptOrderingError("terminal attempt contract version differs from intent")
        if (
            attempt.parent_kind != intent.parent_kind
            or attempt.parent_id != intent.parent_id
            or attempt.stage != intent.stage
            or attempt.request_fingerprint != intent.request_fingerprint
            or attempt.reconciliation_capability != intent.reconciliation_capability
            or attempt.idempotency_key_hash != intent.idempotency_key_hash
        ):
            raise AttemptOrderingError("terminal attempt differs from its durable intent")
        if attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME:
            if prepared.receipt_sealed:
                raise AttemptOrderingError("received attempt cannot be unknown")
        elif not prepared.receipt_sealed:
            raise AttemptOrderingError("terminal attempt requires a durable receipt")
