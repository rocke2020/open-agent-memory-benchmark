"""Production semantic rules for accounting and paired comparison reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oamb.artifacts.validation.registry import ValidationRule
from oamb.contracts.accounting import (
    CostRecord,
    ResourceUsageRecord,
    TokenUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
)
from oamb.contracts.evidence import AttemptRecordV2, ValidationIssue, ValidationSeverity
from oamb.contracts.reporting import (
    AggregateMetricDelta,
    ComparisonControlSnapshot,
    ComparisonCostDelta,
    ComparisonReport,
    PairedMetricDelta,
)
from oamb.contracts.specifications import ComparisonCostView, ComparisonSpec
from oamb.reporting.accounting import (
    AccountingRecordView,
    AccountingReduction,
    AccountingReductionError,
    reduce_accounting_records,
)
from oamb.reporting.compare import REQUIRED_COMPARISON_CONTROL_IDS, evaluate_comparison

TokenRecord = TokenUsageRecord | TokenUsageRecordV2 | TokenUsageRecordV3


@dataclass(frozen=True, slots=True)
class AccountingValidationInput:
    token_records: tuple[TokenRecord, ...]
    resource_records: tuple[ResourceUsageRecord, ...]
    cost_records: tuple[CostRecord, ...]
    record_views: tuple[AccountingRecordView, ...]
    expected_plan_ids: frozenset[str]
    expected_attempt_ids: frozenset[str]
    attempts: tuple[AttemptRecordV2, ...]
    require_attempt_accounting_closure: bool
    reduction: AccountingReduction


def accounting_validation_input(
    *,
    token_records: tuple[TokenRecord, ...],
    resource_records: tuple[ResourceUsageRecord, ...],
    cost_records: tuple[CostRecord, ...],
    record_views: tuple[AccountingRecordView, ...],
    expected_plan_ids: frozenset[str],
    expected_attempt_ids: frozenset[str],
    attempts: tuple[AttemptRecordV2, ...] = (),
    require_attempt_accounting_closure: bool = False,
) -> AccountingValidationInput:
    reduction = reduce_accounting_records(
        token_records=token_records,
        resource_records=resource_records,
        cost_records=cost_records,
        record_views=record_views,
        expected_plan_ids=expected_plan_ids,
    )
    return AccountingValidationInput(
        token_records=token_records,
        resource_records=resource_records,
        cost_records=cost_records,
        record_views=record_views,
        expected_plan_ids=expected_plan_ids,
        expected_attempt_ids=expected_attempt_ids,
        attempts=attempts,
        require_attempt_accounting_closure=require_attempt_accounting_closure,
        reduction=reduction,
    )


@dataclass(frozen=True, slots=True)
class ComparisonValidationInput:
    spec: ComparisonSpec
    left: ComparisonControlSnapshot
    right: ComparisonControlSnapshot
    paired_metric_deltas: tuple[PairedMetricDelta, ...]
    aggregate_metric_delta: AggregateMetricDelta | None
    cost_delta: ComparisonCostDelta | None
    report: ComparisonReport


def comparison_validation_input(
    *,
    spec: ComparisonSpec,
    left: ComparisonControlSnapshot,
    right: ComparisonControlSnapshot,
    paired_metric_deltas: tuple[PairedMetricDelta, ...] = (),
    aggregate_metric_delta: AggregateMetricDelta | None = None,
    cost_delta: ComparisonCostDelta | None = None,
) -> ComparisonValidationInput:
    report = evaluate_comparison(
        spec,
        left,
        right,
        paired_metric_deltas=paired_metric_deltas,
        aggregate_metric_delta=aggregate_metric_delta,
        cost_delta=cost_delta,
    )
    return ComparisonValidationInput(
        spec=spec,
        left=left,
        right=right,
        paired_metric_deltas=paired_metric_deltas,
        aggregate_metric_delta=aggregate_metric_delta,
        cost_delta=cost_delta,
        report=report,
    )


def _reduced_accounting(target: AccountingValidationInput) -> AccountingReduction | None:
    try:
        return reduce_accounting_records(
            token_records=target.token_records,
            resource_records=target.resource_records,
            cost_records=target.cost_records,
            record_views=target.record_views,
            expected_plan_ids=target.expected_plan_ids,
        )
    except AccountingReductionError:
        return None


def _accounting_attempt_coverage_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "accounting.attempt-coverage.v1"
    if not isinstance(target, AccountingValidationInput):
        return _wrong_target(rule_id)
    ordered_attempt_ids = tuple(record.attempt_id for record in target.token_records)
    actual_attempt_ids = frozenset(ordered_attempt_ids)
    exact = bool(
        actual_attempt_ids == target.expected_attempt_ids
        and len(ordered_attempt_ids) == len(actual_attempt_ids)
    )
    return () if exact else (_issue(rule_id, "accounting", "accounting-attempt-coverage-drift"),)


def _accounting_token_meter_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "accounting.token-meter-coverage.v1"
    if not isinstance(target, AccountingValidationInput):
        return _wrong_target(rule_id)
    reduced = _reduced_accounting(target)
    exact = bool(
        reduced is not None
        and all(
            not isinstance(record, TokenUsageRecordV3)
            or (
                set(record.covered_dimensions).isdisjoint(record.unavailable_dimensions)
                and set(record.covered_dimensions).isdisjoint(record.not_applicable_dimensions)
                and set(record.unavailable_dimensions).isdisjoint(record.not_applicable_dimensions)
            )
            for record in target.token_records
        )
    )
    return () if exact else (_issue(rule_id, "accounting", "token-meter-coverage-drift"),)


def _accounting_ownership_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "accounting.resource-cost-ownership.v1"
    if not isinstance(target, AccountingValidationInput):
        return _wrong_target(rule_id)
    reduced = _reduced_accounting(target)
    resource_ids = {record.resource_record_id for record in target.resource_records}
    token_ids = {record.usage_record_id for record in target.token_records}
    attempt_ids = tuple(attempt.attempt_id for attempt in target.attempts)
    token_by_id = {record.usage_record_id: record for record in target.token_records}
    resource_by_id = {record.resource_record_id: record for record in target.resource_records}
    cost_attempt_ids: list[str] = []
    cost_resource_ids: list[str] = []
    attempts_by_id = {attempt.attempt_id: attempt for attempt in target.attempts}
    attempt_resources_close = True
    for cost in target.cost_records:
        if len(cost.source_usage_record_ids) != 1 or len(cost.source_resource_record_ids) != 1:
            attempt_resources_close = False
            continue
        usage = token_by_id.get(cost.source_usage_record_ids[0])
        resource = resource_by_id.get(cost.source_resource_record_ids[0])
        if usage is None or resource is None:
            attempt_resources_close = False
            continue
        attempt = attempts_by_id.get(usage.attempt_id)
        if (
            attempt is None
            or resource.parent_kind != attempt.parent_kind
            or resource.parent_id != attempt.parent_id
            or resource.stage != attempt.stage
            or resource.raw_telemetry_ref != attempt.raw_response_ref
            or cost.parent_kind != attempt.parent_kind
            or cost.parent_id != attempt.parent_id
        ):
            attempt_resources_close = False
            continue
        cost_attempt_ids.append(attempt.attempt_id)
        cost_resource_ids.append(resource.resource_record_id)
    required_attempt_closure = bool(
        not target.require_attempt_accounting_closure
        or (
            attempt_ids
            and len(attempt_ids) == len(set(attempt_ids))
            and frozenset(attempt_ids) == target.expected_attempt_ids
            and len(target.resource_records) == len(attempt_ids)
            and len(target.cost_records) == len(attempt_ids)
            and len(cost_attempt_ids) == len(set(cost_attempt_ids))
            and frozenset(cost_attempt_ids) == target.expected_attempt_ids
            and len(cost_resource_ids) == len(set(cost_resource_ids))
            and set(cost_resource_ids) == resource_ids
            and attempt_resources_close
        )
    )
    exact = bool(
        reduced is not None
        and required_attempt_closure
        and all(
            set(record.source_resource_record_ids) <= resource_ids
            and set(record.source_usage_record_ids) <= token_ids
            for record in target.cost_records
        )
    )
    return () if exact else (_issue(rule_id, "accounting", "resource-cost-ownership-drift"),)


def _accounting_completeness_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "accounting.completeness-claims.v1"
    if not isinstance(target, AccountingValidationInput):
        return _wrong_target(rule_id)
    reduced = _reduced_accounting(target)
    exact = reduced is not None and reduced == target.reduction
    return () if exact else (_issue(rule_id, "accounting", "accounting-completeness-drift"),)


def _reduced_comparison(target: ComparisonValidationInput) -> ComparisonReport | None:
    try:
        return evaluate_comparison(
            target.spec,
            target.left,
            target.right,
            paired_metric_deltas=target.paired_metric_deltas,
            aggregate_metric_delta=target.aggregate_metric_delta,
            cost_delta=target.cost_delta,
        )
    except ValueError:
        return None


def _comparison_control_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "comparison.control-equality.v1"
    if not isinstance(target, ComparisonValidationInput):
        return _wrong_target(rule_id)
    reduced = _reduced_comparison(target)
    exact = bool(
        reduced is not None
        and target.spec.required_control_ids == REQUIRED_COMPARISON_CONTROL_IDS
        and tuple(item.control_id for item in target.left.controls)
        == REQUIRED_COMPARISON_CONTROL_IDS
        and tuple(item.control_id for item in target.right.controls)
        == REQUIRED_COMPARISON_CONTROL_IDS
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.spec.comparison_spec_id, "comparison-control-drift"),)
    )


def _comparison_pairing_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "comparison.metric-pairing.v1"
    if not isinstance(target, ComparisonValidationInput):
        return _wrong_target(rule_id)
    reduced = _reduced_comparison(target)
    exact = bool(
        reduced is not None
        and len(target.paired_metric_deltas) == len(target.spec.ordered_pair_bindings)
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.spec.comparison_spec_id, "comparison-pairing-drift"),)
    )


def _comparison_cost_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "comparison.cost-basis.v1"
    if not isinstance(target, ComparisonValidationInput):
        return _wrong_target(rule_id)
    reduced = _reduced_comparison(target)
    cost_shape = (
        target.cost_delta is None
        if target.spec.cost_view == ComparisonCostView.NONE
        else target.spec.cost_control is not None
    )
    exact = reduced is not None and cost_shape
    return (
        ()
        if exact
        else (_issue(rule_id, target.spec.comparison_spec_id, "comparison-cost-basis-drift"),)
    )


def _comparison_claim_suppression_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "comparison.claim-suppression.v1"
    if not isinstance(target, ComparisonValidationInput):
        return _wrong_target(rule_id)
    reduced = _reduced_comparison(target)
    exact = reduced is not None and reduced == target.report
    return (
        ()
        if exact
        else (
            _issue(rule_id, target.spec.comparison_spec_id, "comparison-claim-suppression-drift"),
        )
    )


def _wrong_target(rule_id: str) -> tuple[ValidationIssue, ...]:
    return (_issue(rule_id, "target", "validation-target-type-mismatch"),)


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


REDUCTION_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("accounting.attempt-coverage.v1", 1, _accounting_attempt_coverage_rule),
    ValidationRule("accounting.token-meter-coverage.v1", 1, _accounting_token_meter_rule),
    ValidationRule("accounting.resource-cost-ownership.v1", 1, _accounting_ownership_rule),
    ValidationRule("accounting.completeness-claims.v1", 1, _accounting_completeness_rule),
    ValidationRule("comparison.control-equality.v1", 1, _comparison_control_rule),
    ValidationRule("comparison.metric-pairing.v1", 1, _comparison_pairing_rule),
    ValidationRule("comparison.cost-basis.v1", 1, _comparison_cost_rule),
    ValidationRule("comparison.claim-suppression.v1", 1, _comparison_claim_suppression_rule),
)


__all__ = [
    "AccountingValidationInput",
    "ComparisonValidationInput",
    "REDUCTION_RULES",
    "accounting_validation_input",
    "comparison_validation_input",
]
