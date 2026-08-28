from __future__ import annotations

import importlib
from types import ModuleType
from typing import cast

import pytest

from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.artifacts.validation.reduction import comparison_validation_input
from oamb.contracts.reporting import ComparisonControlSnapshot, PairedMetricDelta
from oamb.contracts.specifications import ComparisonSpec
from oamb.contracts.states import ValidationDisposition

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    reporting = importlib.import_module("oamb.contracts.reporting")
    specifications = importlib.import_module("oamb.contracts.specifications")
    try:
        comparison = importlib.import_module("oamb.reporting.compare")
    except ModuleNotFoundError:
        pytest.fail("oamb.reporting.compare is not implemented", pytrace=False)
    reporting_required = (
        "ComparisonControlBinding",
        "ComparisonControlSnapshot",
    )
    specification_required = (
        "ComparisonCostView",
        "ComparisonPairBinding",
        "ComparisonSpec",
    )
    missing = tuple(name for name in reporting_required if not hasattr(reporting, name)) + tuple(
        name for name in specification_required if not hasattr(specifications, name)
    )
    if missing:
        pytest.fail(f"missing comparison contracts: {', '.join(missing)}", pytrace=False)
    return reporting, specifications, comparison


def _snapshot(
    reporting: ModuleType,
    required_control_ids: tuple[str, ...],
    *,
    run_id: str,
    root_hash: str,
    memory_system_id: str,
) -> ComparisonControlSnapshot:
    return cast(
        ComparisonControlSnapshot,
        reporting.ComparisonControlSnapshot(
            run_id=run_id,
            source_root_hash=root_hash,
            memory_system_id=memory_system_id,
            provider_native_profile_hash=("d" if memory_system_id == "left-memory" else "e") * 64,
            controls=tuple(
                reporting.ComparisonControlBinding(control_id=control_id, value_hash=SHA_A)
                for control_id in required_control_ids
            ),
        ),
    )


def _spec(
    specifications: ModuleType,
    comparison: ModuleType,
    required_control_ids: tuple[str, ...],
    *,
    cost_view: str = "none",
    cost_control: object | None = None,
) -> ComparisonSpec:
    return cast(
        ComparisonSpec,
        comparison.build_comparison_spec(
            left_run_id="run-left",
            right_run_id="run-right",
            cost_view=specifications.ComparisonCostView(cost_view),
            ordered_pair_bindings=(
                comparison.build_comparison_pair_binding(
                    case_manifest_entry_id=SHA_A,
                    left_case_occurrence_id=SHA_B,
                    right_case_occurrence_id=SHA_C,
                    metric_id="metric-v1",
                ),
            ),
            required_control_ids=required_control_ids,
            comparison_policy_hash=SHA_B,
            winner_reducer=None,
            cost_control=cost_control,
        ),
    )


def _delta(reporting: ModuleType) -> PairedMetricDelta:
    return cast(
        PairedMetricDelta,
        reporting.PairedMetricDelta(
            case_manifest_entry_id=SHA_A,
            left_case_occurrence_id=SHA_B,
            right_case_occurrence_id=SHA_C,
            metric_id="metric-v1",
            left=reporting.ExactRational(numerator=3, denominator=4),
            right=reporting.ExactRational(numerator=1, denominator=2),
            signed_delta=reporting.ExactRational(numerator=1, denominator=4),
            absolute_delta=reporting.ExactRational(numerator=1, denominator=4),
        ),
    )


def test_comparison_allows_native_provider_differences_when_controls_match() -> None:
    reporting, specifications, comparison = _modules()
    required_control_ids = comparison.REQUIRED_COMPARISON_CONTROL_IDS
    delta = _delta(reporting)
    report = comparison.evaluate_comparison(
        _spec(specifications, comparison, required_control_ids),
        _snapshot(
            reporting,
            required_control_ids,
            run_id="run-left",
            root_hash=SHA_A,
            memory_system_id="left-memory",
        ),
        _snapshot(
            reporting,
            required_control_ids,
            run_id="run-right",
            root_hash=SHA_B,
            memory_system_id="right-memory",
        ),
        paired_metric_deltas=(delta,),
    )

    assert report.comparable is True
    assert all(predicate.passed for predicate in report.predicates)
    assert report.paired_metric_deltas == (delta,)
    assert report.winner is None

    profile_id = "oamb-t8-comparison-paired-native-v1"
    target = comparison_validation_input(
        spec=_spec(specifications, comparison, required_control_ids),
        left=_snapshot(
            reporting,
            required_control_ids,
            run_id="run-left",
            root_hash=SHA_A,
            memory_system_id="left-memory",
        ),
        right=_snapshot(
            reporting,
            required_control_ids,
            run_id="run-right",
            root_hash=SHA_B,
            memory_system_id="right-memory",
        ),
        paired_metric_deltas=(delta,),
    )
    validation = validate_catalog_profile(profile_id, target)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues


def test_comparison_suppresses_all_claims_when_one_control_differs() -> None:
    reporting, specifications, comparison = _modules()
    required_control_ids = comparison.REQUIRED_COMPARISON_CONTROL_IDS
    left = _snapshot(
        reporting,
        required_control_ids,
        run_id="run-left",
        root_hash=SHA_A,
        memory_system_id="left-memory",
    )
    right = _snapshot(
        reporting,
        required_control_ids,
        run_id="run-right",
        root_hash=SHA_B,
        memory_system_id="right-memory",
    )
    controls = list(right.controls)
    controls[3] = controls[3].model_copy(update={"value_hash": SHA_C})
    right = right.model_copy(update={"controls": tuple(controls)})

    report = comparison.evaluate_comparison(
        _spec(specifications, comparison, required_control_ids),
        left,
        right,
        paired_metric_deltas=(_delta(reporting),),
    )

    assert report.comparable is False
    assert [item.rule_id for item in report.predicates if not item.passed] == [
        required_control_ids[3]
    ]
    assert "paired_metric_deltas" not in report.model_dump(mode="json")
    assert "winner" not in report.model_dump(mode="json")
    assert report.limitations == (f"comparison control mismatch: {required_control_ids[3]}",)


def test_comparison_rejects_spec_or_snapshot_inventory_drift() -> None:
    reporting, specifications, comparison = _modules()
    required_control_ids = comparison.REQUIRED_COMPARISON_CONTROL_IDS
    left = _snapshot(
        reporting,
        required_control_ids,
        run_id="run-left",
        root_hash=SHA_A,
        memory_system_id="left-memory",
    )
    right = _snapshot(
        reporting,
        required_control_ids,
        run_id="run-right",
        root_hash=SHA_B,
        memory_system_id="right-memory",
    )

    try:
        comparison.evaluate_comparison(
            _spec(specifications, comparison, required_control_ids).model_copy(
                update={"required_control_ids": required_control_ids[:-1]}
            ),
            left,
            right,
        )
    except ValueError as exc:
        assert "closed comparison control inventory" in str(exc)
    else:
        raise AssertionError("comparison accepted a shortened control inventory")

    try:
        comparison.evaluate_comparison(
            _spec(specifications, comparison, required_control_ids),
            left.model_copy(update={"controls": left.controls[:-1]}),
            right,
            paired_metric_deltas=(_delta(reporting),),
        )
    except ValueError as exc:
        assert "closed comparison control inventory" in str(exc)
    else:
        raise AssertionError("comparison accepted a missing control binding")


def test_comparison_spec_recomputes_identity_and_rejects_duplicate_pairs() -> None:
    _reporting, specifications, comparison = _modules()
    required_control_ids = comparison.REQUIRED_COMPARISON_CONTROL_IDS
    spec = _spec(specifications, comparison, required_control_ids)

    with pytest.raises(ValueError, match="comparison spec identity"):
        specifications.ComparisonSpec.model_validate(
            {**spec.model_dump(mode="python"), "comparison_spec_id": SHA_A}
        )

    pair = spec.ordered_pair_bindings[0]
    with pytest.raises(ValueError, match="duplicate comparison pair"):
        comparison.build_comparison_spec(
            left_run_id="run-left",
            right_run_id="run-right",
            cost_view=specifications.ComparisonCostView.NONE,
            ordered_pair_bindings=(pair, pair),
            required_control_ids=required_control_ids,
            comparison_policy_hash=SHA_B,
            winner_reducer=None,
            cost_control=None,
        )


def test_cost_comparison_requires_a_typed_matching_cost_contract_and_delta() -> None:
    reporting, specifications, comparison = _modules()
    required_control_ids = comparison.REQUIRED_COMPARISON_CONTROL_IDS

    with pytest.raises(ValueError, match="cost control"):
        _spec(
            specifications,
            comparison,
            required_control_ids,
            cost_view="actual_charge",
            cost_control=None,
        )

    control = specifications.ComparisonCostControl(
        view=specifications.ComparisonCostView.ACTUAL_CHARGE,
        dimension_id="supplier-charge",
        unit="currency_minor_unit",
        basis="actual_supplier_charge",
        currency="USD",
        environment_hash=SHA_A,
        measurement_spec_hash=SHA_B,
        price_policy_hash=SHA_C,
        fx_policy_hash=None,
    )
    spec = _spec(
        specifications,
        comparison,
        required_control_ids,
        cost_view="actual_charge",
        cost_control=control,
    )
    report = comparison.evaluate_comparison(
        spec,
        _snapshot(
            reporting,
            required_control_ids,
            run_id="run-left",
            root_hash=SHA_A,
            memory_system_id="left-memory",
        ),
        _snapshot(
            reporting,
            required_control_ids,
            run_id="run-right",
            root_hash=SHA_B,
            memory_system_id="right-memory",
        ),
        paired_metric_deltas=(_delta(reporting),),
    )

    assert report.comparable is False
    assert "cost_delta" not in report.model_dump(mode="json")
    assert report.limitations == ("selected cost comparison evidence is unavailable",)
