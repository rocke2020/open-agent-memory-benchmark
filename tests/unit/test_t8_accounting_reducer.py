from __future__ import annotations

import importlib
from decimal import Decimal
from types import ModuleType

import pytest

from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.artifacts.validation.reduction import accounting_validation_input
from oamb.contracts.accounting import (
    CostBasis,
    CostRecord,
    IndexingView,
    ProofStatus,
    ResourceUsageRecord,
    TokenDomain,
    TokenMeasurementSource,
    TokenStageV2,
    TokenUsageRecordV2,
)
from oamb.contracts.states import ValidationDisposition

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _accounting() -> ModuleType:
    try:
        return importlib.import_module("oamb.reporting.accounting")
    except ModuleNotFoundError:
        pytest.fail("oamb.reporting.accounting is not implemented", pytrace=False)


def _resource(
    record_id: str,
    *,
    dimension_id: str,
    value: Decimal | None,
    proof_status: ProofStatus,
    reason: str | None,
) -> ResourceUsageRecord:
    return ResourceUsageRecord(
        resource_record_id=record_id,
        parent_kind="ingestion_plan",
        parent_id="plan-a",
        stage="memory_ingest",
        meter_boundary="fixture-meter",
        dimension_id=dimension_id,
        value=value,
        unit="byte",
        measurement_source="fixture",
        measurement_spec_id="fixture-v1",
        environment_hash=SHA_D,
        started_at=None,
        ended_at=None,
        raw_telemetry_ref=None,
        proof_status=proof_status,
        reason=reason,
    )


def _cost(
    record_id: str,
    *,
    basis: CostBasis,
    indexing_view: IndexingView,
    amount: Decimal,
) -> CostRecord:
    return CostRecord(
        cost_record_id=record_id,
        parent_kind="ingestion_plan",
        parent_id="plan-a",
        basis=basis,
        indexing_view=indexing_view,
        amount=amount,
        currency="USD",
        price_snapshot_id="fixture-price-v1",
        source_usage_record_ids=(),
        source_resource_record_ids=(SHA_A,),
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
    )


def _token(record_id: str, attempt_id: str) -> TokenUsageRecordV2:
    return TokenUsageRecordV2(
        usage_record_id=record_id,
        attempt_id=attempt_id,
        parent_kind="case",
        parent_id="case-a",
        stage=TokenStageV2.ANSWER,
        operation_kind="answer",
        token_domain=TokenDomain.EXTERNAL_LLM,
        measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=10,
        visible_output_tokens=5,
        supplier_reported_total_tokens=15,
        context_view_tokens=None,
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
        raw_response_ref=SHA_D,
    )


def test_accounting_reducer_keeps_attempted_final_actual_estimate_and_unavailable_distinct() -> (
    None
):
    accounting = _accounting()
    storage = _resource(
        SHA_A,
        dimension_id="storage_bytes",
        value=Decimal("3"),
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
    )
    unavailable = _resource(
        SHA_B,
        dimension_id="peak_rss_bytes",
        value=None,
        proof_status=ProofStatus.UNAVAILABLE,
        reason="fixture meter unavailable",
    )
    actual = _cost(
        SHA_C,
        basis=CostBasis.ACTUAL_SUPPLIER_CHARGE,
        indexing_view=IndexingView.ATTEMPTED,
        amount=Decimal("1.25"),
    )
    estimate = _cost(
        SHA_D,
        basis=CostBasis.ESTIMATE_FROM_MEASURED_USAGE,
        indexing_view=IndexingView.FINAL_CONTRIBUTION,
        amount=Decimal("0.50"),
    )
    views = (
        accounting.AccountingRecordView(SHA_A, "ingestion_plan", "final_contribution"),
        accounting.AccountingRecordView(SHA_B, "ingestion_plan", "attempted"),
        accounting.AccountingRecordView(SHA_C, "ingestion_plan", "attempted"),
        accounting.AccountingRecordView(SHA_D, "ingestion_plan", "final_contribution"),
    )

    reduced = accounting.reduce_accounting_records(
        token_records=(),
        resource_records=(storage, unavailable),
        cost_records=(actual, estimate),
        record_views=views,
        expected_plan_ids=frozenset({"plan-a"}),
    )

    actual_line = next(line for line in reduced.lines if line.basis == "actual_supplier_charge")
    estimate_line = next(
        line for line in reduced.lines if line.basis == "estimate_from_measured_usage"
    )
    unavailable_line = next(line for line in reduced.lines if line.dimension_id == "peak_rss_bytes")
    assert (actual_line.value.numerator, actual_line.value.denominator) == (5, 4)
    assert actual_line.indexing_view == "attempted"
    assert (estimate_line.value.numerator, estimate_line.value.denominator) == (1, 2)
    assert estimate_line.indexing_view == "final_contribution"
    assert unavailable_line.value is None
    assert unavailable_line.reason == "fixture meter unavailable"
    assert not reduced.billing_complete
    assert reduced.cost_complete

    profile_id = "oamb-t8-accounting-native-v1"
    target = accounting_validation_input(
        token_records=(),
        resource_records=(storage, unavailable),
        cost_records=(actual, estimate),
        record_views=views,
        expected_plan_ids=frozenset({"plan-a"}),
        expected_attempt_ids=frozenset(),
    )
    validation = validate_catalog_profile(profile_id, target)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues


def test_accounting_reducer_rejects_duplicate_or_unowned_plan_records() -> None:
    accounting = _accounting()
    storage = _resource(
        SHA_A,
        dimension_id="storage_bytes",
        value=Decimal("3"),
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
    )
    view = accounting.AccountingRecordView(SHA_A, "ingestion_plan", "final_contribution")

    with pytest.raises(accounting.AccountingReductionError, match="duplicate"):
        accounting.reduce_accounting_records(
            token_records=(),
            resource_records=(storage, storage),
            cost_records=(),
            record_views=(view,),
            expected_plan_ids=frozenset({"plan-a"}),
        )
    with pytest.raises(accounting.AccountingReductionError, match="unowned plan"):
        accounting.reduce_accounting_records(
            token_records=(),
            resource_records=(storage,),
            cost_records=(),
            record_views=(view,),
            expected_plan_ids=frozenset({"plan-b"}),
        )


def test_accounting_profile_rejects_two_usage_records_for_one_attempt() -> None:
    accounting = _accounting()
    first = _token(SHA_A, SHA_C)
    duplicate_attempt = _token(SHA_B, SHA_C)
    views = (
        accounting.AccountingRecordView(SHA_A, "case", "not_applicable"),
        accounting.AccountingRecordView(SHA_B, "case", "not_applicable"),
    )
    target = accounting_validation_input(
        token_records=(first, duplicate_attempt),
        resource_records=(),
        cost_records=(),
        record_views=views,
        expected_plan_ids=frozenset(),
        expected_attempt_ids=frozenset({SHA_C}),
    )

    validation = validate_catalog_profile("oamb-t8-accounting-native-v1", target)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "accounting.attempt-coverage.v1" in validation.failed_rule_ids
