from __future__ import annotations

from decimal import Decimal

import pytest

from oamb.runtime.budget import (
    BudgetAmount,
    BudgetCeiling,
    BudgetExceededError,
    BudgetLedger,
    ReservationRequest,
)


def amount(
    *,
    attempts: int = 0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    wall_seconds: str = "0",
    cost: str = "0",
    cpu_seconds: str = "0",
    provider_calls: int = 0,
) -> BudgetAmount:
    return BudgetAmount(
        attempts=attempts,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        wall_seconds=Decimal(wall_seconds),
        cost=Decimal(cost),
        resources=(("cpu_seconds", Decimal(cpu_seconds), "seconds"),),
        provider_units=(("openai", "chat_completion", "call", provider_calls),),
    )


def test_zero_external_allowance_rejects_before_reservation() -> None:
    ledger = BudgetLedger(BudgetCeiling.zero(currency="USD"), role_ceilings={})

    with pytest.raises(BudgetExceededError, match="parent attempts"):
        ledger.reserve(
            ReservationRequest(
                reservation_id="reservation-1",
                role_binding_id="answer-role",
                maximum=amount(attempts=1, provider_calls=1),
            )
        )

    assert ledger.snapshot().reserved == amount()


def test_reservation_checks_parent_role_provider_and_resource_atomically() -> None:
    parent = BudgetCeiling(
        maximum=amount(attempts=2, cpu_seconds="2", provider_calls=2), currency="USD"
    )
    role = BudgetCeiling(
        maximum=amount(attempts=1, cpu_seconds="1", provider_calls=1), currency="USD"
    )
    ledger = BudgetLedger(parent, role_ceilings={"answer-role": role})
    request = ReservationRequest(
        reservation_id="reservation-1",
        role_binding_id="answer-role",
        maximum=amount(attempts=1, cpu_seconds="1.5", provider_calls=1),
    )

    with pytest.raises(BudgetExceededError, match="role resource cpu_seconds"):
        ledger.reserve(request)

    assert ledger.snapshot().reserved == amount()


def test_unknown_outcome_retains_the_full_reservation() -> None:
    ceiling = BudgetCeiling(maximum=amount(attempts=1, provider_calls=1), currency="USD")
    ledger = BudgetLedger(ceiling, role_ceilings={"answer-role": ceiling})
    request = ReservationRequest(
        reservation_id="reservation-1",
        role_binding_id="answer-role",
        maximum=amount(attempts=1, provider_calls=1),
    )
    ledger.reserve(request)

    ledger.mark_unknown("reservation-1")

    assert ledger.snapshot().reserved == amount(attempts=1, provider_calls=1)
    with pytest.raises(BudgetExceededError, match="parent attempts"):
        ledger.reserve(
            ReservationRequest(
                reservation_id="reservation-2",
                role_binding_id="answer-role",
                maximum=amount(attempts=1, provider_calls=1),
            )
        )


def test_receipt_commits_observed_usage_and_releases_only_known_unused_capacity() -> None:
    ceiling = BudgetCeiling(
        maximum=amount(
            attempts=2,
            input_tokens=100,
            output_tokens=20,
            wall_seconds="10",
            cost="1",
            cpu_seconds="4",
            provider_calls=2,
        ),
        currency="USD",
    )
    ledger = BudgetLedger(ceiling, role_ceilings={"answer-role": ceiling})
    ledger.reserve(
        ReservationRequest(
            reservation_id="reservation-1",
            role_binding_id="answer-role",
            maximum=amount(
                attempts=1,
                input_tokens=50,
                output_tokens=10,
                wall_seconds="5",
                cost="0.5",
                cpu_seconds="2",
                provider_calls=1,
            ),
        )
    )

    ledger.commit(
        "reservation-1",
        observed=amount(
            attempts=1,
            input_tokens=20,
            output_tokens=4,
            wall_seconds="2",
            cost="0.2",
            cpu_seconds="0.5",
            provider_calls=1,
        ),
    )

    snapshot = ledger.snapshot()
    assert snapshot.reserved == amount()
    assert snapshot.committed == amount(
        attempts=1,
        input_tokens=20,
        output_tokens=4,
        wall_seconds="2",
        cost="0.2",
        cpu_seconds="0.5",
        provider_calls=1,
    )


def test_ledger_rejects_mixed_parent_and_role_currencies() -> None:
    parent = BudgetCeiling(maximum=amount(cost="1"), currency="USD")
    role = BudgetCeiling(maximum=amount(cost="1"), currency="EUR")

    with pytest.raises(ValueError, match="currency"):
        BudgetLedger(parent, role_ceilings={"answer-role": role})
