from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace
from typing import cast

import pytest

from oamb.contracts.evidence import BudgetReservationRecordV3
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


def test_multi_owner_commit_accounts_each_observed_owner_independently() -> None:
    """Catches charging every owner its reservation maximum after a smaller receipt."""

    BudgetOwnerAllocation = budget.BudgetOwnerAllocation
    ledger = BudgetLedger(
        _ceiling(6),
        role_ceilings={"extract": _ceiling(2), "embed": _ceiling(2)},
        provider_operation_ceilings={"ingest": _ceiling(2)},
    )
    reservation = ReservationRequest(
        reservation_id="first",
        maximum=_amount(attempts=3),
        owner_allocations=(
            BudgetOwnerAllocation("ingest", _amount()),
            BudgetOwnerAllocation("extract", _amount()),
            BudgetOwnerAllocation("embed", _amount()),
        ),
    )
    ledger.reserve(reservation)

    ledger.commit(
        reservation.reservation_id,
        observed=_amount(),
        owner_observed=(
            BudgetOwnerAllocation("ingest", _amount()),
            BudgetOwnerAllocation("extract", BudgetAmount.zero()),
            BudgetOwnerAllocation("embed", BudgetAmount.zero()),
        ),
    )

    # The extraction owner observed zero, so its complete two-attempt ceiling remains available.
    ledger.reserve(
        ReservationRequest(
            reservation_id="second",
            maximum=_amount(attempts=2),
            owner_allocations=(BudgetOwnerAllocation("extract", _amount(attempts=2)),),
        )
    )


def test_multi_owner_commit_rejects_incomplete_observed_owner_inventory_atomically() -> None:
    """Catches a partial receipt silently omitting an owner and corrupting ledger state."""

    BudgetOwnerAllocation = budget.BudgetOwnerAllocation
    ledger = BudgetLedger(
        _ceiling(4),
        role_ceilings={"extract": _ceiling(2)},
        provider_operation_ceilings={"ingest": _ceiling(2)},
    )
    reservation = ReservationRequest(
        reservation_id="reservation",
        maximum=_amount(attempts=2),
        owner_allocations=(
            BudgetOwnerAllocation("ingest", _amount()),
            BudgetOwnerAllocation("extract", _amount()),
        ),
    )
    ledger.reserve(reservation)
    before = ledger.snapshot()

    with pytest.raises(budget.ReservationStateError, match="owner inventory"):
        ledger.commit(
            reservation.reservation_id,
            observed=_amount(),
            owner_observed=(BudgetOwnerAllocation("ingest", _amount()),),
        )

    assert ledger.snapshot() == before


def test_runtime_owner_provider_units_bind_by_owner_not_sorted_position() -> None:
    from oamb.runtime.attempts import _runtime_owner_allocations

    operation_owner = "z-operation-owner"
    model_owner = "a-model-owner"

    def allocation(owner_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            owner_id=owner_id,
            allocated_attempts=1,
            allocated_input_tokens=0,
            allocated_output_tokens=0,
            allocated_dispatch_wall_seconds=Decimal("1"),
            allocated_cost=None,
            allocated_resource_ceilings=(),
            allocated_provider_units=Decimal("1"),
        )

    reservation = cast(
        BudgetReservationRecordV3,
        SimpleNamespace(owner_allocations=(allocation(operation_owner), allocation(model_owner))),
    )
    maximum = BudgetAmount(
        attempts=2,
        wall_seconds=Decimal("2"),
        provider_units=(
            (operation_owner, "memory_ingest", "request", 1),
            (model_owner, "memory_ingest", "request", 1),
        ),
    )

    runtime_allocations = _runtime_owner_allocations(reservation, maximum)

    assert tuple(item.maximum.provider_units[0][0] for item in runtime_allocations) == (
        operation_owner,
        model_owner,
    )
