from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptReceiptKind,
    AttemptReceiptRecord,
    AttemptRecordV2,
    BudgetReservationRecord,
    OccurrenceClaimRecord,
)
from oamb.contracts.ports import (
    ArtifactReadRequest,
    ArtifactSealReceipt,
    ArtifactStorePort,
    ArtifactWriteRequest,
    RawPayloadSealRequest,
    RawReferenceHandle,
)
from oamb.contracts.specifications import BudgetScopeKindV2, ResourceBudgetCeiling
from oamb.contracts.states import AttemptOutcome, IndexContribution
from oamb.runtime.attempts import AttemptCoordinator, AttemptOrderingError
from oamb.runtime.budget import (
    BudgetAmount,
    BudgetCeiling,
    BudgetLedger,
)
from oamb.runtime.resume import ProviderLifecycleBridge

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
ATTEMPT_ID = "a" * 64
CLAIM_ID = "b" * 64
RESERVATION_ID = "c" * 64
REQUEST_HASH = "d" * 64


class RecordingStore:
    def __init__(self, root: Path) -> None:
        self.delegate = ArtifactStore(root)
        self.events: list[str] = []

    def seal_raw(self, request: RawPayloadSealRequest) -> RawReferenceHandle:
        self.events.append(f"raw:{request.sha256}")
        return self.delegate.seal_raw(request)

    def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        self.events.append(request.relative_path)
        return self.delegate.seal_source_record(request)

    def seal_checkpoint(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        return self.delegate.seal_checkpoint(request)

    def seal_source_manifest(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        return self.delegate.seal_source_manifest(request)

    def read_verified(self, request: ArtifactReadRequest) -> bytes:
        return self.delegate.read_verified(request)


def claim() -> OccurrenceClaimRecord:
    return OccurrenceClaimRecord(
        claim_id=CLAIM_ID,
        occurrence_id="case-1",
        lease_record_hash="e" * 64,
        lease_epoch=1,
        owner_id="runner-1",
        stage="answer",
        request_fingerprint=REQUEST_HASH,
        reconciliation_capability="none",
        claimed_at=NOW,
    )


def reservation() -> BudgetReservationRecord:
    return BudgetReservationRecord(
        reservation_id=RESERVATION_ID,
        budget_id="budget-1",
        scope_kind=BudgetScopeKindV2.RUN,
        scope_id="case-1",
        role_binding_id="answer-role",
        attempt_id=ATTEMPT_ID,
        reserved_attempts=1,
        reserved_input_tokens=10,
        reserved_output_tokens=5,
        reserved_dispatch_wall_seconds=Decimal("2"),
        reserved_cost=Decimal("0.1"),
        currency="USD",
        reserved_resource_ceilings=(
            ResourceBudgetCeiling(
                dimension_id="cpu_seconds",
                maximum=Decimal("1"),
                unit="seconds",
            ),
        ),
        reserved_provider_units=Decimal("1"),
        reserved_at=NOW,
    )


def intent() -> AttemptIntentRecord:
    return AttemptIntentRecord(
        attempt_id=ATTEMPT_ID,
        claim_id=CLAIM_ID,
        reservation_id=RESERVATION_ID,
        parent_kind="case",
        parent_id="case-1",
        role_binding_id="answer-role",
        stage="answer",
        request_fingerprint=REQUEST_HASH,
        reconciliation_capability="none",
        idempotency_key_hash=None,
        sealed_at=NOW,
    )


def maximum() -> BudgetAmount:
    return BudgetAmount(
        attempts=1,
        input_tokens=10,
        output_tokens=5,
        wall_seconds=reservation().reserved_dispatch_wall_seconds,
        cost=reservation().reserved_cost or Decimal("0"),
        resources=(
            ("cpu_seconds", reservation().reserved_resource_ceilings[0].maximum, "seconds"),
        ),
        provider_units=(("openai", "chat_completion", "call", 1),),
    )


def ledger() -> BudgetLedger:
    ceiling = BudgetCeiling(maximum=maximum(), currency="USD")
    return BudgetLedger(ceiling, role_ceilings={"answer-role": ceiling})


def terminal(outcome: AttemptOutcome, *, raw_response_ref: str | None) -> AttemptRecordV2:
    return AttemptRecordV2(
        attempt_id=ATTEMPT_ID,
        parent_kind="case",
        parent_id="case-1",
        stage="answer",
        ordinal=1,
        request_fingerprint=REQUEST_HASH,
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
        outcome=outcome,
        retry_of_attempt_id=None,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=raw_response_ref,
        raw_error_ref=None,
        index_contribution=IndexContribution.NOT_APPLICABLE,
        superseded_by_attempt_id=None,
    )


def test_prepare_seals_claim_reservation_and_intent_before_dispatch(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "capsule")
    budget = ledger()
    coordinator = AttemptCoordinator(store, budget)

    coordinator.prepare(
        claim=claim(),
        reservation=reservation(),
        intent=intent(),
        maximum=maximum(),
    )

    assert isinstance(store, ArtifactStorePort)
    assert store.events == [
        f"source/occurrence-claims/{CLAIM_ID}.json",
        f"source/budget-reservations/{RESERVATION_ID}.json",
        f"source/attempt-intents/{ATTEMPT_ID}.json",
    ]
    assert budget.snapshot().reserved == maximum()


@pytest.mark.parametrize(
    ("claim_update", "reservation_update", "intent_update", "maximum_update", "match"),
    (
        ({"stage": "different"}, {}, {}, {}, "stage"),
        ({"reconciliation_capability": "receipt_lookup"}, {}, {}, {}, "reconciliation"),
        ({}, {"scope_id": "different"}, {}, {}, "scope"),
        (
            {},
            {},
            {},
            {"provider_units": (("openai", "chat_completion", "call", 2),)},
            "provider units",
        ),
    ),
)
def test_prepare_rejects_a_chain_that_would_fail_validation_before_writing(
    tmp_path: Path,
    claim_update: dict[str, object],
    reservation_update: dict[str, object],
    intent_update: dict[str, object],
    maximum_update: dict[str, object],
    match: str,
) -> None:
    store = RecordingStore(tmp_path / "capsule")
    coordinator = AttemptCoordinator(store, ledger())

    with pytest.raises(AttemptOrderingError, match=match):
        coordinator.prepare(
            claim=claim().model_copy(update=claim_update),
            reservation=reservation().model_copy(update=reservation_update),
            intent=intent().model_copy(update=intent_update),
            maximum=replace(maximum(), **maximum_update),
        )

    assert store.events == []


def test_response_seals_raw_then_receipt_then_terminal_and_commits_usage(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "capsule")
    budget = ledger()
    coordinator = AttemptCoordinator(store, budget)
    coordinator.prepare(
        claim=claim(), reservation=reservation(), intent=intent(), maximum=maximum()
    )
    raw = b'{"answer":"ok"}'
    raw_hash = hashlib.sha256(raw).hexdigest()
    receipt = AttemptReceiptRecord(
        attempt_id=ATTEMPT_ID,
        receipt_kind=AttemptReceiptKind.RESPONSE,
        raw_response_ref=raw_hash,
        raw_error_ref=None,
        dispatch_started_at=NOW,
        receipt_observed_at=NOW + timedelta(seconds=1),
        provider_request_wall_seconds=Decimal("1"),
    )
    observed = BudgetAmount(
        attempts=1,
        input_tokens=4,
        output_tokens=2,
        wall_seconds=receipt.provider_request_wall_seconds,
        cost=reservation().reserved_cost or Decimal("0"),
        resources=(
            ("cpu_seconds", reservation().reserved_resource_ceilings[0].maximum, "seconds"),
        ),
        provider_units=(("openai", "chat_completion", "call", 1),),
    )

    coordinator.record_receipt(
        receipt=receipt,
        raw_payload=raw,
        media_type="application/json",
        observed=observed,
    )
    coordinator.seal_terminal(terminal(AttemptOutcome.SUCCEEDED, raw_response_ref=raw_hash))

    assert store.events[-3:] == [
        f"raw:{raw_hash}",
        f"source/attempt-receipts/{ATTEMPT_ID}.json",
        f"source/attempts/{ATTEMPT_ID}.json",
    ]
    assert budget.snapshot().reserved == BudgetAmount.zero()
    assert budget.snapshot().committed == observed


def test_unknown_outcome_keeps_reservation_and_seals_no_receipt(tmp_path: Path) -> None:
    store = RecordingStore(tmp_path / "capsule")
    budget = ledger()
    coordinator = AttemptCoordinator(store, budget)
    coordinator.prepare(
        claim=claim(), reservation=reservation(), intent=intent(), maximum=maximum()
    )

    coordinator.mark_unknown(
        reservation_id=RESERVATION_ID,
        attempt=terminal(AttemptOutcome.UNKNOWN_OUTCOME, raw_response_ref=None),
    )

    assert budget.snapshot().reserved == maximum()
    assert not any("attempt-receipts" in event for event in store.events)
    assert store.events[-1] == f"source/attempts/{ATTEMPT_ID}.json"


def test_invalid_unknown_terminal_does_not_consume_the_reservation(tmp_path: Path) -> None:
    budget = ledger()
    coordinator = AttemptCoordinator(RecordingStore(tmp_path / "capsule"), budget)
    coordinator.prepare(
        claim=claim(), reservation=reservation(), intent=intent(), maximum=maximum()
    )
    invalid = terminal(AttemptOutcome.UNKNOWN_OUTCOME, raw_response_ref=None).model_copy(
        update={"parent_id": "different-case"}
    )

    with pytest.raises(AttemptOrderingError, match="durable intent"):
        coordinator.mark_unknown(reservation_id=RESERVATION_ID, attempt=invalid)

    budget.cancel_before_dispatch(RESERVATION_ID)
    assert budget.snapshot().reserved == BudgetAmount.zero()


def test_provider_attempt_guard_exists_only_from_dispatch_until_durable_receipt(
    tmp_path: Path,
) -> None:
    provider_runtime = tmp_path / "provider-runtime"
    bridge = ProviderLifecycleBridge(provider_runtime)
    bridge.acquire_run(
        run_id="run-1",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="e" * 64,
    )
    coordinator = AttemptCoordinator(
        RecordingStore(tmp_path / "capsule"),
        ledger(),
        provider_lifecycle=bridge,
    )
    coordinator.prepare(
        claim=claim(), reservation=reservation(), intent=intent(), maximum=maximum()
    )

    coordinator.mark_dispatched(ATTEMPT_ID)

    assert (provider_runtime / "active-provider-attempt").is_file()
    raw = b'{"answer":"ok"}'
    raw_hash = hashlib.sha256(raw).hexdigest()
    coordinator.record_receipt(
        receipt=AttemptReceiptRecord(
            attempt_id=ATTEMPT_ID,
            receipt_kind=AttemptReceiptKind.RESPONSE,
            raw_response_ref=raw_hash,
            raw_error_ref=None,
            dispatch_started_at=NOW,
            receipt_observed_at=NOW + timedelta(seconds=1),
            provider_request_wall_seconds=Decimal("1"),
        ),
        raw_payload=raw,
        media_type="application/json",
        observed=maximum(),
    )

    assert not (provider_runtime / "active-provider-attempt").exists()
