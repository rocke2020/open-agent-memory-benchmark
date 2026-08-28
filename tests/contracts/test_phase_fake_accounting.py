from __future__ import annotations

import runpy
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from oamb.artifacts.validation.phase import validate_t10_phase_gate
from oamb.contracts.accounting import ProofStatus
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.states import ValidationDisposition
from oamb.phase_review_profiles import (
    FAKE_PHASE_REVIEW_CLIENT_KIND,
    OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND,
)
from oamb.phase_review_run import _build_costs

_PHASE_FIXTURES = runpy.run_path(str(Path(__file__).with_name("test_t8_phase_validation.py")))
_accepted_gate_with_evidence = cast(
    Callable[..., tuple[Any, Any, Any]],
    _PHASE_FIXTURES["_accepted_gate_with_evidence"],
)


def _fake_not_applicable_evidence(evidence: Any) -> Any:
    runner_costs = _build_costs(
        evidence.usage_records,
        evidence.resource_records,
        client_kind=FAKE_PHASE_REVIEW_CLIENT_KIND,
        price_snapshot=None,
    )
    costs = tuple(
        generated.model_copy(update={"cost_record_id": original.cost_record_id})
        for generated, original in zip(
            runner_costs,
            evidence.cost_records,
            strict=True,
        )
    )
    return replace(
        evidence,
        cost_records=costs,
    )


def _issue_codes(result: Any) -> set[str]:
    return {issue.code for issue in result.issues}


def test_fake_role_accepts_runner_not_applicable_cost_evidence() -> None:
    bundle, gate, evidence = _accepted_gate_with_evidence(fake_reviewer=True)
    fake_evidence = _fake_not_applicable_evidence(evidence)

    result = validate_t10_phase_gate(bundle, gate, review_evidence=fake_evidence)

    assert result.disposition == ValidationDisposition.VALIDATED, result.issues


def test_phase_review_cost_ids_use_shared_evidence_domain() -> None:
    _bundle, _gate, evidence = _accepted_gate_with_evidence(fake_reviewer=True)

    costs = _build_costs(
        evidence.usage_records,
        evidence.resource_records,
        client_kind=FAKE_PHASE_REVIEW_CLIENT_KIND,
        price_snapshot=None,
    )

    assert costs[0].cost_record_id == canonical_sha256(
        [
            "oamb-phase-review-cost-v1",
            evidence.usage_records[0].usage_record_id,
            evidence.resource_records[0].resource_record_id,
        ]
    )


def test_live_unavailable_usage_produces_unavailable_cost_evidence() -> None:
    _bundle, _gate, evidence = _accepted_gate_with_evidence()
    assert evidence.price_snapshot is not None
    unavailable_usage = evidence.usage_records[0].model_copy(
        update={
            "input_tokens": None,
            "visible_output_tokens": None,
            "supplier_reported_total_tokens": None,
            "proof_status": ProofStatus.UNAVAILABLE,
            "reason": "supplier_usage_missing",
        }
    )

    costs = _build_costs(
        (unavailable_usage,),
        (evidence.resource_records[0],),
        client_kind=OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND,
        price_snapshot=evidence.price_snapshot,
    )

    assert len(costs) == 1
    assert costs[0].proof_status == ProofStatus.UNAVAILABLE
    assert costs[0].amount is None
    assert costs[0].currency is None
    assert costs[0].price_snapshot_id == evidence.price_snapshot.price_snapshot_id
    assert costs[0].reason == "supplier token usage unavailable for price estimation"


def test_live_failed_attempt_preserves_measured_usage_cost_estimate() -> None:
    _bundle, _gate, evidence = _accepted_gate_with_evidence()
    assert evidence.price_snapshot is not None

    costs = _build_costs(
        (evidence.usage_records[0],),
        (evidence.resource_records[0],),
        client_kind=OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND,
        price_snapshot=evidence.price_snapshot,
        failure=True,
    )

    assert costs[0].proof_status == ProofStatus.MEASURED_COMPLETE
    assert costs[0].amount == Decimal("0.02")
    assert costs[0].currency == "USD"
    assert costs[0].price_snapshot_id == evidence.price_snapshot.price_snapshot_id
    assert costs[0].reason is None


def test_fake_not_applicable_cost_rejects_wrong_role_or_reason() -> None:
    bundle, gate, evidence = _accepted_gate_with_evidence(fake_reviewer=True)
    fake_evidence = _fake_not_applicable_evidence(evidence)
    wrong_role = fake_evidence.reviewer_role_binding.model_copy(
        update={"provider": "different-fake-reviewer"}
    )
    wrong_reason = fake_evidence.cost_records[0].model_copy(update={"reason": "not billed"})

    role_result = validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(fake_evidence, reviewer_role_binding=wrong_role),
    )
    reason_result = validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            fake_evidence,
            cost_records=(wrong_reason, *fake_evidence.cost_records[1:]),
        ),
    )

    assert role_result.disposition == ValidationDisposition.INVALID
    assert reason_result.disposition == ValidationDisposition.INVALID
    assert "phase-review-source-evidence-mismatch" in _issue_codes(role_result)
    assert "phase-review-source-evidence-mismatch" in _issue_codes(reason_result)


def test_live_role_rejects_not_applicable_or_unavailable_cost() -> None:
    bundle, gate, evidence = _accepted_gate_with_evidence()
    not_applicable = evidence.cost_records[0].model_copy(
        update={
            "amount": None,
            "currency": None,
            "proof_status": ProofStatus.NOT_APPLICABLE,
            "reason": "credential-free fake review has no supplier billing",
        }
    )
    unavailable = evidence.cost_records[0].model_copy(
        update={
            "amount": None,
            "currency": None,
            "proof_status": ProofStatus.UNAVAILABLE,
            "reason": "supplier billing evidence unavailable",
        }
    )

    not_applicable_result = validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            cost_records=(not_applicable, *evidence.cost_records[1:]),
        ),
    )
    unavailable_result = validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            evidence,
            cost_records=(unavailable, *evidence.cost_records[1:]),
        ),
    )

    assert not_applicable_result.disposition == ValidationDisposition.INVALID
    assert unavailable_result.disposition == ValidationDisposition.INVALID
    assert "phase-review-source-evidence-mismatch" in _issue_codes(not_applicable_result)
    assert "phase-review-source-evidence-mismatch" in _issue_codes(unavailable_result)


def test_fake_role_rejects_supplier_price_snapshot() -> None:
    bundle, gate, fake_evidence = _accepted_gate_with_evidence(fake_reviewer=True)
    _live_bundle, _live_gate, live_evidence = _accepted_gate_with_evidence()
    assert live_evidence.price_snapshot is not None

    result = validate_t10_phase_gate(
        bundle,
        gate,
        review_evidence=replace(
            fake_evidence,
            price_snapshot=live_evidence.price_snapshot,
        ),
    )

    assert result.disposition == ValidationDisposition.INVALID
    assert "phase-review-source-evidence-mismatch" in _issue_codes(result)
