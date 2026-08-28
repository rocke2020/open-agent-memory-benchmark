"""Strict contracts for factual external historical evidence and reports."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .base import (
    NonEmptyStr,
    NonNegativeDecimal,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictContract,
    UtcDateTime,
    canonical_decimal_text,
)
from .evidence import OriginClass
from .ids import canonical_sha256
from .reporting import ExactRational, ReducerBinding, ReportRecordProjection
from .specifications import SourceEvidenceBinding, SourceEvidenceKind

GitRevision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
HistoricalCategory = Literal[
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
]
HistoricalLimitationCode = Literal[
    "no-oamb-attempt-ledger",
    "indexing-usage-unavailable",
    "external-llm-usage-unavailable",
    "prompt-compatibility-unknown",
    "amb-context-view-not-oamb-context-view",
    "no-causal-attribution",
]

_BIDI_CONTROLS = frozenset(
    {
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)
_ALLOWED_CASE_FIELDS = (
    "case_id",
    "category",
    "verdict",
    "context_tokens",
    "retrieval_time_ms",
)
_ALLOWED_AGGREGATE_FIELDS = (
    "total_cases",
    "correct_cases",
    "context_tokens_total",
    "retrieval_time_ms_total",
)


def _reject_bidi(value: str) -> None:
    if any(character in value for character in _BIDI_CONTROLS):
        raise ValueError("external evidence text contains a bidirectional control")


class ExternalHistoricalCase(StrictContract):
    schema_name: Literal["external_historical_case"] = "external_historical_case"
    schema_version: Literal[1] = 1
    case_id: Annotated[
        str,
        Field(
            min_length=8,
            max_length=17,
            pattern=r"^(?:gpt4_)?[0-9a-z]{8}(?:_abs)?$",
        ),
    ]
    category: HistoricalCategory
    verdict: Literal["correct", "incorrect"]
    context_tokens: NonNegativeInt
    retrieval_time_ms: NonNegativeDecimal

    @model_validator(mode="after")
    def safe_case_identity(self) -> Self:
        _reject_bidi(self.case_id)
        return self


class ExternalHistoricalCategoryAggregate(StrictContract):
    schema_name: Literal["external_historical_category_aggregate"] = (
        "external_historical_category_aggregate"
    )
    schema_version: Literal[1] = 1
    category: HistoricalCategory
    total_cases: PositiveInt
    correct_cases: NonNegativeInt
    context_tokens_total: NonNegativeInt
    retrieval_time_ms_total: NonNegativeDecimal

    @model_validator(mode="after")
    def correct_count_is_bounded(self) -> Self:
        if self.correct_cases > self.total_cases:
            raise ValueError("external category correct count exceeds total cases")
        return self


class ExternalHistoricalAggregate(StrictContract):
    schema_name: Literal["external_historical_aggregate"] = "external_historical_aggregate"
    schema_version: Literal[1] = 1
    total_cases: PositiveInt
    correct_cases: NonNegativeInt
    context_tokens_total: NonNegativeInt
    retrieval_time_ms_total: NonNegativeDecimal

    @model_validator(mode="after")
    def correct_count_is_bounded(self) -> Self:
        if self.correct_cases > self.total_cases:
            raise ValueError("external correct count exceeds total cases")
        return self


class ExternalCompatibilityAssessment(StrictContract):
    schema_name: Literal["external_compatibility_assessment"] = "external_compatibility_assessment"
    schema_version: Literal[1] = 1
    target_protocol: Literal["oamb-longmemeval-protocol"]
    status: Literal["unknown"]
    assessment_code: Literal["not-evaluated-by-oamb-comparison-predicate"]


class ExternalTransformationRecord(StrictContract):
    schema_name: Literal["external_transformation_record"] = "external_transformation_record"
    schema_version: Literal[1] = 1
    algorithm: Literal["amb-factual-projection-v1"]
    allowed_case_fields: tuple[NonEmptyStr, ...]
    allowed_aggregate_fields: tuple[NonEmptyStr, ...]
    unchanged_source_root_sha256: Sha256
    importer_implementation_hash: Sha256

    @model_validator(mode="after")
    def allowlist_is_exact(self) -> Self:
        if self.allowed_case_fields != _ALLOWED_CASE_FIELDS:
            raise ValueError("external transformation case allowlist is not exact")
        if self.allowed_aggregate_fields != _ALLOWED_AGGREGATE_FIELDS:
            raise ValueError("external transformation aggregate allowlist is not exact")
        return self


class ExternalHistoricalEvidence(StrictContract):
    schema_name: Literal["external_historical_evidence"] = "external_historical_evidence"
    schema_version: Literal[1] = 1
    external_evidence_id: Sha256
    origin_class: Literal[OriginClass.EXTERNAL_AMB_GENERATED]
    producer_repository: Literal["rocke2020/agent-memory-benchmark"]
    producer_code_revision: GitRevision
    producer_base_revision: GitRevision
    producer_protocol: Literal["amb-longmemeval-rag"]
    imported_at: UtcDateTime
    source_path: Literal["outputs/longmemeval/hindsight-deepseek/rag/s.json"]
    source_sha256: Sha256
    source_byte_count: PositiveInt
    source_attestation_path: Literal["eval_analysis/evidence/official-comparison.json"]
    source_attestation_sha256: Sha256
    source_attestation_revision: GitRevision
    run_name: Literal["hindsight-deepseek"]
    dataset: Literal["longmemeval"]
    split: Literal["s"]
    memory_provider: Literal["hindsight"]
    mode: Literal["rag"]
    oracle: Literal[False]
    answer_model: Literal["openai:deepseek-v4-pro"]
    judge_model: Literal["openai:deepseek-v4-flash"]
    compatibility: ExternalCompatibilityAssessment
    transformation: ExternalTransformationRecord
    aggregate: ExternalHistoricalAggregate
    category_aggregates: tuple[ExternalHistoricalCategoryAggregate, ...]
    cases: tuple[ExternalHistoricalCase, ...]
    oamb_attempt_ledger_present: Literal[False]
    comparison_eligible: Literal[False]
    billing_complete: Literal[False]
    cost_complete: Literal[False]
    limitation_codes: tuple[HistoricalLimitationCode, ...]

    @model_validator(mode="after")
    def exact_identity_and_factual_aggregates_close(self) -> Self:
        source_path = PurePosixPath(self.source_path)
        attestation_path = PurePosixPath(self.source_attestation_path)
        if source_path.is_absolute() or attestation_path.is_absolute():
            raise ValueError("external source paths must be producer-relative")
        if self.transformation.unchanged_source_root_sha256 != self.source_sha256:
            raise ValueError("external transformation changed its source root")
        if len(self.cases) != len({case.case_id for case in self.cases}):
            raise ValueError("external historical cases must be unique")
        category_order = tuple(item.category for item in self.category_aggregates)
        expected_category_order: tuple[HistoricalCategory, ...] = (
            "knowledge-update",
            "multi-session",
            "single-session-assistant",
            "single-session-preference",
            "single-session-user",
            "temporal-reasoning",
        )
        if category_order != expected_category_order:
            raise ValueError("external category aggregate inventory is not exact")
        expected_aggregate = ExternalHistoricalAggregate(
            total_cases=len(self.cases),
            correct_cases=sum(case.verdict == "correct" for case in self.cases),
            context_tokens_total=sum(case.context_tokens for case in self.cases),
            retrieval_time_ms_total=sum((case.retrieval_time_ms for case in self.cases), Decimal()),
        )
        if self.aggregate != expected_aggregate:
            raise ValueError("external aggregate does not close from cases")
        rebuilt_categories = tuple(
            ExternalHistoricalCategoryAggregate(
                category=category,
                total_cases=len(category_cases),
                correct_cases=sum(case.verdict == "correct" for case in category_cases),
                context_tokens_total=sum(case.context_tokens for case in category_cases),
                retrieval_time_ms_total=sum(
                    (case.retrieval_time_ms for case in category_cases), Decimal()
                ),
            )
            for category in expected_category_order
            for category_cases in (tuple(case for case in self.cases if case.category == category),)
        )
        if self.category_aggregates != rebuilt_categories:
            raise ValueError("external category aggregates do not close from cases")
        expected_limitations: tuple[HistoricalLimitationCode, ...] = (
            "no-oamb-attempt-ledger",
            "indexing-usage-unavailable",
            "external-llm-usage-unavailable",
            "prompt-compatibility-unknown",
            "amb-context-view-not-oamb-context-view",
            "no-causal-attribution",
        )
        if self.limitation_codes != expected_limitations:
            raise ValueError("external limitation-code inventory is not exact")
        fields = self.model_dump(
            mode="python",
            exclude={"schema_name", "schema_version", "external_evidence_id"},
        )
        if self.external_evidence_id != external_historical_evidence_id(**fields):
            raise ValueError("external historical evidence identity does not match its payload")
        return self


class ExternalHistoricalEvidenceReport(StrictContract):
    schema_name: Literal["external_historical_evidence_report"] = (
        "external_historical_evidence_report"
    )
    schema_version: Literal[1] = 1
    report_id: Sha256
    report_spec_hash: Sha256
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    evidence_validation_profile_hash: Sha256
    evidence_validation_result_hash: Sha256
    reducer_binding: ReducerBinding
    audience: Literal["local", "public"]
    origin_class: Literal[OriginClass.EXTERNAL_AMB_GENERATED]
    external_evidence_id: Sha256
    producer_repository: Literal["rocke2020/agent-memory-benchmark"]
    producer_code_revision: GitRevision
    producer_base_revision: GitRevision
    producer_protocol: Literal["amb-longmemeval-rag"]
    source_sha256: Sha256
    source_byte_count: PositiveInt
    source_attestation_sha256: Sha256
    source_attestation_revision: GitRevision
    importer_implementation_hash: Sha256
    compatibility_status: Literal["unknown"]
    comparison_eligible: Literal[False]
    billing_complete: Literal[False]
    cost_complete: Literal[False]
    total_cases: PositiveInt
    correct_cases: NonNegativeInt
    accuracy: ExactRational
    category_aggregates: tuple[ExternalHistoricalCategoryAggregate, ...]
    amb_formatted_view_context_tokens_total: NonNegativeInt
    amb_formatted_view_context_tokens_mean: ExactRational
    retrieval_time_ms_total: NonNegativeDecimal
    retrieval_time_ms_mean: NonNegativeDecimal
    cases: tuple[ExternalHistoricalCase, ...]
    record_projections: tuple[ReportRecordProjection, ...]
    limitation_codes: tuple[HistoricalLimitationCode, ...]

    @model_validator(mode="after")
    def external_only_identity_and_aggregates_close(self) -> Self:
        if len(self.ordered_source_bindings) != 1:
            raise ValueError("external historical report requires exactly one source binding")
        source = self.ordered_source_bindings[0]
        if source.source_kind != SourceEvidenceKind.EXTERNAL:
            raise ValueError("external historical report requires an external source binding")
        if source.source_identity != self.external_evidence_id:
            raise ValueError("external report source identity does not match evidence")
        if self.total_cases != len(self.cases) or self.correct_cases != sum(
            case.verdict == "correct" for case in self.cases
        ):
            raise ValueError("external report case counts do not close")
        if Fraction(self.accuracy.numerator, self.accuracy.denominator) != Fraction(
            self.correct_cases, self.total_cases
        ):
            raise ValueError("external report accuracy does not close")
        context_total = sum(case.context_tokens for case in self.cases)
        if self.amb_formatted_view_context_tokens_total != context_total or Fraction(
            self.amb_formatted_view_context_tokens_mean.numerator,
            self.amb_formatted_view_context_tokens_mean.denominator,
        ) != Fraction(context_total, self.total_cases):
            raise ValueError("external AMB formatted-view token measurement does not close")
        retrieval_total = sum((case.retrieval_time_ms for case in self.cases), Decimal())
        if (
            self.retrieval_time_ms_total != retrieval_total
            or self.retrieval_time_ms_mean != retrieval_total / self.total_cases
        ):
            raise ValueError("external retrieval timing does not close")
        expected_projections = external_historical_case_report_projections(
            self.external_evidence_id,
            self.cases,
        )
        if self.record_projections != expected_projections:
            raise ValueError("external historical case projections do not close")
        fields = self.model_dump(
            mode="python", exclude={"schema_name", "schema_version", "report_id"}
        )
        if self.report_id != external_historical_evidence_report_id(**fields):
            raise ValueError("external historical report identity does not match its payload")
        return self


def external_historical_evidence_id(**fields: object) -> str:
    return canonical_sha256(["oamb-external-historical-evidence-v1", fields])


def external_historical_evidence_report_id(**fields: object) -> str:
    return canonical_sha256(["oamb-external-historical-evidence-report-v1", fields])


def external_historical_case_report_record_id(
    external_evidence_id: str,
    case_id: str,
) -> str:
    return canonical_sha256(
        [
            "oamb-external-historical-case-report-record-v1",
            external_evidence_id,
            case_id,
        ]
    )


def external_historical_case_report_projections(
    external_evidence_id: str,
    cases: tuple[ExternalHistoricalCase, ...],
) -> tuple[ReportRecordProjection, ...]:
    projections: list[ReportRecordProjection] = []
    for source_order, case in enumerate(cases, 1):
        latency_microseconds = case.retrieval_time_ms * Decimal(1_000)
        if latency_microseconds != latency_microseconds.to_integral_value():
            raise ValueError("external retrieval timing is not an exact microsecond value")
        projections.append(
            ReportRecordProjection(
                record_id=external_historical_case_report_record_id(
                    external_evidence_id,
                    case.case_id,
                ),
                axis="case",
                label=f"AMB historical case {source_order}: {case.case_id}",
                status="judged",
                failure_stage=None,
                evaluation_status="judged",
                verdict=case.verdict,
                capabilities_or_types=(case.category,),
                metric_ids=("amb-historical-judge-verdict-v1",),
                proof_statuses=(),
                raw_evidence_present=False,
                latency_microseconds=int(latency_microseconds),
                context_view_tokens=None,
                declared_usage=None,
                detail_items=(
                    ("source_order", str(source_order)),
                    ("producer_case_id", case.case_id),
                    ("category", case.category),
                    ("verdict", case.verdict),
                    ("amb_formatted_view_context_tokens", str(case.context_tokens)),
                    ("retrieval_time_ms", canonical_decimal_text(case.retrieval_time_ms)),
                    ("compatibility_status", "unknown"),
                ),
            )
        )
    return tuple(projections)


__all__ = [
    "ExternalCompatibilityAssessment",
    "ExternalHistoricalAggregate",
    "ExternalHistoricalCase",
    "ExternalHistoricalCategoryAggregate",
    "ExternalHistoricalEvidence",
    "ExternalHistoricalEvidenceReport",
    "ExternalTransformationRecord",
    "HistoricalCategory",
    "HistoricalLimitationCode",
    "external_historical_evidence_id",
    "external_historical_case_report_projections",
    "external_historical_case_report_record_id",
    "external_historical_evidence_report_id",
]
