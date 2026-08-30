from __future__ import annotations

from decimal import Decimal

import pytest

from oamb.runtime import budget

BudgetAmount = budget.BudgetAmount
BudgetCeiling = budget.BudgetCeiling
BudgetExceededError = budget.BudgetExceededError
BudgetLedger = budget.BudgetLedger
ReservationRequest = budget.ReservationRequest


def _amount(*, attempts: int = 1) -> BudgetAmount:
    return BudgetAmount(
        attempts=attempts,
        wall_seconds=Decimal(attempts),
        provider_units=(("provider", "operation", "request", attempts),),
    )


def _ceiling(attempts: int) -> BudgetCeiling:
    return BudgetCeiling(maximum=_amount(attempts=attempts), currency=None)


def test_provider_reservation_checks_every_owner_atomically() -> None:
    assert hasattr(budget, "BudgetOwnerAllocation"), "multi-owner allocation is missing"
    BudgetOwnerAllocation = budget.BudgetOwnerAllocation
    ledger = BudgetLedger(
        _ceiling(6),
        role_ceilings={"extract": _ceiling(1), "embed": _ceiling(2)},
        provider_operation_ceilings={"ingest": _ceiling(2)},
    )
    allocations = (
        BudgetOwnerAllocation("ingest", _amount()),
        BudgetOwnerAllocation("extract", _amount()),
        BudgetOwnerAllocation("embed", _amount()),
    )

    ledger.reserve(
        ReservationRequest(
            reservation_id="first",
            maximum=_amount(attempts=3),
            owner_allocations=allocations,
        )
    )
    before = ledger.snapshot()

    with pytest.raises(BudgetExceededError, match="extract"):
        ledger.reserve(
            ReservationRequest(
                reservation_id="second",
                maximum=_amount(attempts=3),
                owner_allocations=allocations,
            )
        )

    assert ledger.snapshot() == before


def test_multi_owner_cancel_and_commit_reconcile_the_same_owner_set() -> None:
    assert hasattr(budget, "BudgetOwnerAllocation"), "multi-owner allocation is missing"
    BudgetOwnerAllocation = budget.BudgetOwnerAllocation
    ledger = BudgetLedger(
        _ceiling(4),
        role_ceilings={"extract": _ceiling(2)},
        provider_operation_ceilings={"ingest": _ceiling(2)},
    )
    request = ReservationRequest(
        reservation_id="reservation",
        maximum=_amount(attempts=2),
        owner_allocations=(
            BudgetOwnerAllocation("ingest", _amount()),
            BudgetOwnerAllocation("extract", _amount()),
        ),
    )

    ledger.reserve(request)
    ledger.cancel_before_dispatch(request.reservation_id)
    assert ledger.snapshot().reserved == BudgetAmount.zero()

    committed_request = ReservationRequest(
        reservation_id="committed",
        maximum=_amount(attempts=2),
        owner_allocations=request.owner_allocations,
    )
    ledger.reserve(committed_request)
    ledger.commit(committed_request.reservation_id, observed=_amount(attempts=2))

    assert ledger.snapshot().reserved == BudgetAmount.zero()
    assert ledger.snapshot().committed == _amount(attempts=2)
