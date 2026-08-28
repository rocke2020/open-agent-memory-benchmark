"""Deterministic reduction of validated external evidence into an external-only report."""

from __future__ import annotations

import hashlib
from fractions import Fraction
from pathlib import Path

from oamb.contracts.evidence import ValidationResult
from oamb.contracts.external import (
    ExternalHistoricalEvidence,
    ExternalHistoricalEvidenceReport,
    external_historical_case_report_projections,
    external_historical_evidence_report_id,
)
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import ExactRational, ReducerBinding
from oamb.contracts.specifications import ReportSpec, SourceEvidenceBinding, SourceEvidenceKind
from oamb.contracts.states import ValidationDisposition

from .validation import EXTERNAL_HISTORICAL_PROFILE_ID, validate_external_historical_evidence

EXTERNAL_REPORT_REDUCER_CONTRACT = {
    "reducer_id": "amb-historical-factual-report-v1",
    "version": 1,
    "input_schema": "external_historical_evidence@1",
    "output_schema": "external_historical_evidence_report@1",
    "case_order": "producer source order",
    "accuracy": "exact correct cases divided by total cases",
    "context_measurement": "AMB formatted retrieval view; not OAMB context_view",
    "forbidden_native_shapes": ("capsule", "ingestion plan", "attempt", "indexing usage"),
}


def external_report_reducer_implementation_hash() -> str:
    return canonical_sha256(
        [
            EXTERNAL_REPORT_REDUCER_CONTRACT,
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ]
    )


def reduce_external_historical_report(
    record: ExternalHistoricalEvidence,
    validation_result: ValidationResult,
    *,
    report_spec: ReportSpec,
) -> ExternalHistoricalEvidenceReport:
    fresh = validate_external_historical_evidence(record)
    if (
        validation_result.disposition != ValidationDisposition.VALIDATED
        or validation_result.validation_profile_id != EXTERNAL_HISTORICAL_PROFILE_ID
        or fresh != validation_result
    ):
        raise ValueError("external historical report requires fresh validation")
    if report_spec.report_kind != "run":
        raise ValueError("external historical evidence requires a run report spec")
    validation_hash = canonical_sha256(validation_result)
    source_fields = {
        "source_kind": SourceEvidenceKind.EXTERNAL,
        "source_identity": record.external_evidence_id,
        "source_root_hash": validation_result.target_hash,
        "validation_result_hash": validation_hash,
        "source_schema_versions": (
            "external_historical_evidence@1",
            "external_historical_case@1",
        ),
    }
    source_binding = SourceEvidenceBinding.model_validate(
        {
            "binding_id": canonical_sha256(["oamb-source-evidence-binding-v1", source_fields]),
            **source_fields,
        }
    )
    reducer_binding = ReducerBinding(
        reducer_id="amb-historical-factual-report-v1",
        reducer_version=1,
        implementation_hash=external_report_reducer_implementation_hash(),
    )
    accuracy = _exact(Fraction(record.aggregate.correct_cases, record.aggregate.total_cases))
    context_mean = _exact(
        Fraction(record.aggregate.context_tokens_total, record.aggregate.total_cases)
    )
    fields = {
        "report_spec_hash": canonical_sha256(report_spec),
        "ordered_source_bindings": (source_binding,),
        "evidence_validation_profile_hash": canonical_sha256(
            [
                "oamb-validation-profile-binding-v1",
                validation_result.validation_profile_id,
                validation_result.required_rule_ids,
                validation_result.implementation_versions,
            ]
        ),
        "evidence_validation_result_hash": validation_hash,
        "reducer_binding": reducer_binding,
        "audience": report_spec.audience,
        "origin_class": "external_amb_generated",
        "external_evidence_id": record.external_evidence_id,
        "producer_repository": record.producer_repository,
        "producer_code_revision": record.producer_code_revision,
        "producer_base_revision": record.producer_base_revision,
        "producer_protocol": record.producer_protocol,
        "source_sha256": record.source_sha256,
        "source_byte_count": record.source_byte_count,
        "source_attestation_sha256": record.source_attestation_sha256,
        "source_attestation_revision": record.source_attestation_revision,
        "importer_implementation_hash": record.transformation.importer_implementation_hash,
        "compatibility_status": "unknown",
        "comparison_eligible": False,
        "billing_complete": False,
        "cost_complete": False,
        "total_cases": record.aggregate.total_cases,
        "correct_cases": record.aggregate.correct_cases,
        "accuracy": accuracy,
        "category_aggregates": record.category_aggregates,
        "amb_formatted_view_context_tokens_total": record.aggregate.context_tokens_total,
        "amb_formatted_view_context_tokens_mean": context_mean,
        "retrieval_time_ms_total": record.aggregate.retrieval_time_ms_total,
        "retrieval_time_ms_mean": (
            record.aggregate.retrieval_time_ms_total / record.aggregate.total_cases
        ),
        "cases": record.cases,
        "record_projections": external_historical_case_report_projections(
            record.external_evidence_id,
            record.cases,
        ),
        "limitation_codes": record.limitation_codes,
    }
    return ExternalHistoricalEvidenceReport.model_validate(
        {
            "report_id": external_historical_evidence_report_id(**fields),
            **fields,
        }
    )


def _exact(value: Fraction) -> ExactRational:
    return ExactRational(numerator=value.numerator, denominator=value.denominator)


__all__ = [
    "EXTERNAL_REPORT_REDUCER_CONTRACT",
    "external_report_reducer_implementation_hash",
    "reduce_external_historical_report",
]
