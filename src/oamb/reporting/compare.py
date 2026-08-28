"""Fail-closed pairwise comparison over validated control snapshots."""

from __future__ import annotations

from typing import Literal

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import (
    AggregateMetricDelta,
    ComparableComparisonReport,
    ComparisonControlSnapshot,
    ComparisonCostDelta,
    ComparisonPredicateResult,
    ComparisonReport,
    IncomparableComparisonReport,
    PairedMetricDelta,
)
from oamb.contracts.specifications import (
    ComparisonCostControl,
    ComparisonCostView,
    ComparisonPairBinding,
    ComparisonSpec,
    ComparisonWinnerReducer,
    comparison_pair_id,
    comparison_spec_id,
)

REQUIRED_COMPARISON_CONTROL_IDS = (
    "comparison.protocol-version.v1",
    "comparison.workload-version.v1",
    "comparison.dataset-revision.v1",
    "comparison.ordered-case-manifest.v1",
    "comparison.prompt-packs.v1",
    "comparison.output-contracts.v1",
    "comparison.metric-evaluation-policy.v1",
    "comparison.answer-role-binding.v1",
    "comparison.judge-role-binding.v1",
    "comparison.controlled-embedding.v1",
    "comparison.native-reranking-disabled.v1",
    "comparison.visible-context-policy.v1",
    "comparison.token-measurement-contract.v1",
    "comparison.resource-measurement-contract.v1",
    "comparison.cost-measurement-contract.v1",
    "comparison.query-effect-policy.v1",
    "comparison.retry-policy.v1",
    "comparison.failure-denominator-policy.v1",
)

COST_EVIDENCE_PREDICATE_ID = "comparison.selected-cost-evidence.v1"
WINNER_EVIDENCE_PREDICATE_ID = "comparison.selected-winner-evidence.v1"


def build_comparison_pair_binding(
    *,
    case_manifest_entry_id: str,
    left_case_occurrence_id: str,
    right_case_occurrence_id: str,
    metric_id: str,
) -> ComparisonPairBinding:
    return ComparisonPairBinding(
        pair_id=comparison_pair_id(
            case_manifest_entry_id=case_manifest_entry_id,
            left_case_occurrence_id=left_case_occurrence_id,
            right_case_occurrence_id=right_case_occurrence_id,
            metric_id=metric_id,
        ),
        case_manifest_entry_id=case_manifest_entry_id,
        left_case_occurrence_id=left_case_occurrence_id,
        right_case_occurrence_id=right_case_occurrence_id,
        metric_id=metric_id,
    )


def build_comparison_spec(
    *,
    left_run_id: str,
    right_run_id: str,
    cost_view: ComparisonCostView,
    ordered_pair_bindings: tuple[ComparisonPairBinding, ...],
    required_control_ids: tuple[str, ...],
    comparison_policy_hash: str,
    winner_reducer: ComparisonWinnerReducer | None,
    cost_control: ComparisonCostControl | None,
) -> ComparisonSpec:
    identity = comparison_spec_id(
        left_run_id=left_run_id,
        right_run_id=right_run_id,
        cost_view=cost_view,
        ordered_pair_bindings=ordered_pair_bindings,
        required_control_ids=required_control_ids,
        comparison_policy_hash=comparison_policy_hash,
        winner_reducer=winner_reducer,
        cost_control=cost_control,
    )
    return ComparisonSpec(
        comparison_spec_id=identity,
        left_run_id=left_run_id,
        right_run_id=right_run_id,
        cost_view=cost_view,
        ordered_pair_bindings=ordered_pair_bindings,
        required_control_ids=required_control_ids,
        comparison_policy_hash=comparison_policy_hash,
        winner_reducer=winner_reducer,
        cost_control=cost_control,
    )


def evaluate_comparison(
    spec: ComparisonSpec,
    left: ComparisonControlSnapshot,
    right: ComparisonControlSnapshot,
    *,
    paired_metric_deltas: tuple[PairedMetricDelta, ...] = (),
    aggregate_metric_delta: AggregateMetricDelta | None = None,
    cost_delta: ComparisonCostDelta | None = None,
) -> ComparisonReport:
    """Evaluate every frozen predicate before admitting any paired claim."""

    if spec.required_control_ids != REQUIRED_COMPARISON_CONTROL_IDS:
        raise ValueError("comparison spec does not contain the closed comparison control inventory")
    if left.run_id != spec.left_run_id or right.run_id != spec.right_run_id:
        raise ValueError("comparison snapshots do not bind the selected run order")
    _require_pair_inventory(spec, paired_metric_deltas)
    left_controls = _closed_control_map(left)
    right_controls = _closed_control_map(right)
    predicate_list = [
        ComparisonPredicateResult(
            rule_id=control_id,
            expected_hash=left_controls[control_id],
            left_hash=left_controls[control_id],
            right_hash=right_controls[control_id],
            passed=left_controls[control_id] == right_controls[control_id],
        )
        for control_id in REQUIRED_COMPARISON_CONTROL_IDS
    ]
    limitations: list[str] = [
        f"comparison control mismatch: {item.rule_id}" for item in predicate_list if not item.passed
    ]
    admitted_cost_delta = _admit_cost_delta(spec, cost_delta)
    if spec.cost_view != ComparisonCostView.NONE and admitted_cost_delta is None:
        predicate_list.append(
            _missing_evidence_predicate(COST_EVIDENCE_PREDICATE_ID, spec.comparison_spec_id)
        )
        limitations.append("selected cost comparison evidence is unavailable")
    admitted_aggregate = _admit_aggregate_delta(spec, aggregate_metric_delta)
    if spec.winner_reducer is not None and admitted_aggregate is None:
        predicate_list.append(
            _missing_evidence_predicate(WINNER_EVIDENCE_PREDICATE_ID, spec.comparison_spec_id)
        )
        limitations.append("selected winner aggregate evidence is unavailable")
    predicates = tuple(predicate_list)
    comparable = all(item.passed for item in predicates)
    admitted_deltas = paired_metric_deltas if comparable else ()
    admitted_aggregate = admitted_aggregate if comparable else None
    admitted_cost_delta = admitted_cost_delta if comparable else None
    winner = _winner_from_aggregate(admitted_aggregate)
    frozen_limitations = tuple(limitations)
    comparison_fields = {
        "comparison_spec_hash": spec.comparison_spec_id,
        "ordered_source_root_hashes": (left.source_root_hash, right.source_root_hash),
        "predicates": predicates,
        "paired_metric_deltas": admitted_deltas,
        "aggregate_metric_delta": admitted_aggregate,
        "cost_delta": admitted_cost_delta,
        "winner": winner,
        "limitations": frozen_limitations,
    }
    common = {
        "comparison_id": canonical_sha256(["oamb-comparison-report-v1", comparison_fields]),
        "comparison_spec_hash": spec.comparison_spec_id,
        "ordered_source_root_hashes": (left.source_root_hash, right.source_root_hash),
        "predicates": predicates,
        "limitations": frozen_limitations,
    }
    if comparable:
        return ComparableComparisonReport.model_validate(
            {
                **common,
                "paired_metric_deltas": admitted_deltas,
                "aggregate_metric_delta": admitted_aggregate,
                "cost_delta": admitted_cost_delta,
                "winner": winner,
            }
        )
    return IncomparableComparisonReport.model_validate(common)


def _closed_control_map(snapshot: ComparisonControlSnapshot) -> dict[str, str]:
    control_ids = tuple(item.control_id for item in snapshot.controls)
    if control_ids != REQUIRED_COMPARISON_CONTROL_IDS:
        raise ValueError("snapshot does not contain the closed comparison control inventory")
    return {item.control_id: item.value_hash for item in snapshot.controls}


def _require_pair_inventory(
    spec: ComparisonSpec,
    deltas: tuple[PairedMetricDelta, ...],
) -> None:
    actual = tuple(
        (
            item.case_manifest_entry_id,
            item.left_case_occurrence_id,
            item.right_case_occurrence_id,
            item.metric_id,
        )
        for item in deltas
    )
    expected = tuple(
        (
            item.case_manifest_entry_id,
            item.left_case_occurrence_id,
            item.right_case_occurrence_id,
            item.metric_id,
        )
        for item in spec.ordered_pair_bindings
    )
    if actual != expected:
        raise ValueError("paired metric delta inventory is missing, duplicated, or reordered")


def _admit_cost_delta(
    spec: ComparisonSpec,
    delta: ComparisonCostDelta | None,
) -> ComparisonCostDelta | None:
    if spec.cost_view == ComparisonCostView.NONE:
        if delta is not None:
            raise ValueError("cost delta is forbidden when the selected cost view is none")
        return None
    if delta is None:
        return None
    assert spec.cost_control is not None
    control = spec.cost_control
    if (
        delta.cost_control_hash != canonical_sha256(control)
        or delta.view != control.view
        or delta.dimension_id != control.dimension_id
        or delta.unit != control.unit
        or delta.basis != control.basis
        or delta.currency != control.currency
    ):
        raise ValueError("cost delta does not bind the selected cost comparison contract")
    return delta


def _admit_aggregate_delta(
    spec: ComparisonSpec,
    delta: AggregateMetricDelta | None,
) -> AggregateMetricDelta | None:
    if spec.winner_reducer is None:
        if delta is not None:
            raise ValueError("aggregate winner evidence requires a winner reducer")
        return None
    if delta is None:
        return None
    reducer = spec.winner_reducer
    if (
        delta.reducer_id != reducer.reducer_id
        or delta.reducer_version != reducer.reducer_version
        or delta.metric_id != reducer.metric_id
        or delta.denominator != reducer.denominator
        or delta.aggregate_contract_hash != reducer.aggregate_contract_hash
    ):
        raise ValueError("aggregate metric delta does not bind the selected winner reducer")
    return delta


def _winner_from_aggregate(
    delta: AggregateMetricDelta | None,
) -> Literal["left", "right", "tie"] | None:
    if delta is None:
        return None
    if delta.signed_delta.numerator > 0:
        return "left"
    if delta.signed_delta.numerator < 0:
        return "right"
    return "tie"


def _missing_evidence_predicate(rule_id: str, expected_hash: str) -> ComparisonPredicateResult:
    return ComparisonPredicateResult(
        rule_id=rule_id,
        expected_hash=expected_hash,
        left_hash=canonical_sha256([rule_id, "left", "missing"]),
        right_hash=canonical_sha256([rule_id, "right", "missing"]),
        passed=False,
    )


__all__ = [
    "REQUIRED_COMPARISON_CONTROL_IDS",
    "build_comparison_pair_binding",
    "build_comparison_spec",
    "evaluate_comparison",
]
