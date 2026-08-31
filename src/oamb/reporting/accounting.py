"""Deterministic accounting reduction without implicit zeros or ownership fan-out."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from oamb.contracts.accounting import (
    CostRecord,
    ProofStatus,
    ResourceUsageRecord,
    TokenUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
)
from oamb.contracts.reporting import ExactRational, MeasurementSummaryLine

AccountingOwner = Literal[
    "ingestion_plan",
    "case",
    "model_readiness",
    "run",
]
AccountingIndexingView = Literal["attempted", "final_contribution", "not_applicable"]
MeasurementBasis = Literal[
    "raw_resource",
    "actual_supplier_charge",
    "estimate_from_measured_usage",
    "token_count",
]
_MeasurementGroupKey = tuple[
    str,
    str,
    AccountingOwner,
    AccountingIndexingView,
    str,
    str,
    str | None,
    MeasurementBasis,
    str | None,
]
TokenRecord = TokenUsageRecord | TokenUsageRecordV2 | TokenUsageRecordV3


class AccountingReductionError(ValueError):
    """The sealed accounting inventory cannot be reduced without ambiguity."""


@dataclass(frozen=True, slots=True)
class AccountingRecordView:
    record_id: str
    owner_kind: AccountingOwner
    indexing_view: AccountingIndexingView


@dataclass(frozen=True, slots=True)
class AccountingReduction:
    lines: tuple[MeasurementSummaryLine, ...]
    billing_complete: bool
    cost_complete: bool


@dataclass(frozen=True, slots=True)
class _Measurement:
    dimension_id: str
    stage: str
    owner_kind: AccountingOwner
    indexing_view: AccountingIndexingView
    value: Fraction | None
    unit: str
    proof_status: ProofStatus
    basis: MeasurementBasis
    currency: str | None
    record_id: str
    reason: str | None


def reduce_accounting_records(
    *,
    token_records: tuple[TokenRecord, ...],
    resource_records: tuple[ResourceUsageRecord, ...],
    cost_records: tuple[CostRecord, ...],
    record_views: tuple[AccountingRecordView, ...],
    expected_plan_ids: frozenset[str],
) -> AccountingReduction:
    records: tuple[TokenRecord | ResourceUsageRecord | CostRecord, ...] = (
        *token_records,
        *resource_records,
        *cost_records,
    )
    record_ids = tuple(_record_id(record) for record in records)
    if len(set(record_ids)) != len(record_ids):
        raise AccountingReductionError("duplicate accounting record identity")
    view_ids = tuple(item.record_id for item in record_views)
    if len(set(view_ids)) != len(view_ids):
        raise AccountingReductionError("duplicate accounting record view")
    if set(record_ids) != set(view_ids):
        raise AccountingReductionError("accounting record views do not close the inventory")
    views = {item.record_id: item for item in record_views}

    measurements: list[_Measurement] = []
    for record in records:
        record_id = _record_id(record)
        view = views[record_id]
        parent_kind = str(record.parent_kind)
        if parent_kind != view.owner_kind:
            raise AccountingReductionError("accounting owner kind disagrees with its record")
        if parent_kind == "ingestion_plan" and record.parent_id not in expected_plan_ids:
            raise AccountingReductionError("accounting record names an unowned plan")
        if isinstance(record, CostRecord):
            if record.indexing_view.value != view.indexing_view:
                raise AccountingReductionError("cost indexing view disagrees with its selection")
            measurements.append(_cost_measurement(record, view))
        elif isinstance(record, ResourceUsageRecord):
            measurements.append(_resource_measurement(record, view))
        else:
            measurements.extend(_token_measurements(record, view))

    lines = _aggregate_measurements(tuple(measurements))
    billing_records = tuple(
        record
        for record in token_records
        if isinstance(record, TokenUsageRecordV3) and record.token_domain.value == "external_llm"
    )
    billing_complete = bool(billing_records) and all(
        record.billing_complete for record in billing_records
    )
    cost_complete = bool(cost_records) and all(
        record.proof_status == ProofStatus.MEASURED_COMPLETE for record in cost_records
    )
    return AccountingReduction(
        lines=lines,
        billing_complete=billing_complete,
        cost_complete=cost_complete,
    )


def _record_id(record: TokenRecord | ResourceUsageRecord | CostRecord) -> str:
    if isinstance(record, ResourceUsageRecord):
        return record.resource_record_id
    if isinstance(record, CostRecord):
        return record.cost_record_id
    return record.usage_record_id


def _resource_measurement(
    record: ResourceUsageRecord,
    view: AccountingRecordView,
) -> _Measurement:
    value = Fraction(record.value) if record.value is not None else None
    return _Measurement(
        dimension_id=record.dimension_id,
        stage=record.stage,
        owner_kind=view.owner_kind,
        indexing_view=view.indexing_view,
        value=value,
        unit=record.unit,
        proof_status=record.proof_status,
        basis="raw_resource",
        currency=None,
        record_id=record.resource_record_id,
        reason=record.reason,
    )


def _cost_measurement(record: CostRecord, view: AccountingRecordView) -> _Measurement:
    value = Fraction(record.amount) if record.amount is not None else None
    return _Measurement(
        dimension_id="supplier_cost",
        stage="memory_ingest" if view.owner_kind == "ingestion_plan" else view.owner_kind,
        owner_kind=view.owner_kind,
        indexing_view=view.indexing_view,
        value=value,
        unit="currency",
        proof_status=record.proof_status,
        basis=record.basis.value,
        currency=record.currency,
        record_id=record.cost_record_id,
        reason=record.reason,
    )


def _token_measurements(
    record: TokenRecord,
    view: AccountingRecordView,
) -> tuple[_Measurement, ...]:
    stage = record.stage.value
    values = {
        "input_tokens": record.input_tokens,
        "visible_output_tokens": record.visible_output_tokens,
        "supplier_reported_total_tokens": record.supplier_reported_total_tokens,
        "context_view_tokens": record.context_view_tokens,
    }
    if isinstance(record, TokenUsageRecordV3):
        values["cached_input_tokens"] = record.cached_input_tokens
        values["reasoning_tokens"] = record.reasoning_tokens
        dimensions = (
            *record.covered_dimensions,
            *record.unavailable_dimensions,
            *record.not_applicable_dimensions,
        )
    else:
        measured = tuple(name for name, value in values.items() if value is not None)
        dimensions = measured or ("token_usage",)
    output: list[_Measurement] = []
    for dimension in dimensions:
        raw_value = values.get(dimension)
        if raw_value is not None:
            status = record.proof_status
            reason = record.reason
        elif (
            isinstance(record, TokenUsageRecordV3) and dimension in record.not_applicable_dimensions
        ):
            status = ProofStatus.NOT_APPLICABLE
            reason = record.reason or "dimension not applicable"
        else:
            status = ProofStatus.UNAVAILABLE
            reason = record.reason or "dimension unavailable"
        output.append(
            _Measurement(
                dimension_id=dimension,
                stage=stage,
                owner_kind=view.owner_kind,
                indexing_view=view.indexing_view,
                value=Fraction(raw_value) if raw_value is not None else None,
                unit="token",
                proof_status=status,
                basis="token_count",
                currency=None,
                record_id=record.usage_record_id,
                reason=reason,
            )
        )
    return tuple(output)


def _aggregate_measurements(
    measurements: tuple[_Measurement, ...],
) -> tuple[MeasurementSummaryLine, ...]:
    groups: dict[_MeasurementGroupKey, list[_Measurement]] = {}
    for item in measurements:
        key: _MeasurementGroupKey = (
            item.dimension_id,
            item.stage,
            item.owner_kind,
            item.indexing_view,
            item.unit,
            item.proof_status.value,
            item.reason,
            item.basis,
            item.currency,
        )
        groups.setdefault(key, []).append(item)
    lines: list[MeasurementSummaryLine] = []
    for key, members in sorted(groups.items()):
        values = tuple(item.value for item in members)
        if all(value is not None for value in values):
            total = sum((value for value in values if value is not None), Fraction())
            exact = ExactRational(numerator=total.numerator, denominator=total.denominator)
        else:
            exact = None
        dimension, stage, owner, indexing_view, unit, proof, reason, basis, currency = key
        lines.append(
            MeasurementSummaryLine(
                dimension_id=dimension,
                stage=stage,
                owner_kind=owner,
                indexing_view=indexing_view,
                value=exact,
                unit=unit,
                proof_status=ProofStatus(proof),
                basis=basis,
                currency=currency,
                source_record_ids=tuple(item.record_id for item in members),
                reason=reason,
            )
        )
    return tuple(lines)


__all__ = [
    "AccountingRecordView",
    "AccountingReduction",
    "AccountingReductionError",
    "reduce_accounting_records",
]
