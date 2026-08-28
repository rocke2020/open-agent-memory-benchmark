"""Strict report, comparison, and quality-review contracts."""

from __future__ import annotations

from enum import StrEnum
from fractions import Fraction
from math import gcd
from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from .accounting import ProofStatus
from .base import (
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictContract,
    UtcDateTime,
)
from .ids import canonical_sha256
from .specifications import (
    ComparisonCostView,
    ReportIdentitySpecBinding,
    SourceEvidenceBinding,
)


class QualityReviewStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class ComparisonControlBinding(StrictContract):
    schema_name: Literal["comparison_control_binding"] = "comparison_control_binding"
    schema_version: Literal[1] = 1
    control_id: NonEmptyStr
    value_hash: Sha256


class ComparisonControlSnapshot(StrictContract):
    schema_name: Literal["comparison_control_snapshot"] = "comparison_control_snapshot"
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    source_root_hash: Sha256
    memory_system_id: NonEmptyStr
    provider_native_profile_hash: Sha256
    controls: tuple[ComparisonControlBinding, ...]

    @model_validator(mode="after")
    def control_ids_are_unique(self) -> Self:
        control_ids = tuple(item.control_id for item in self.controls)
        if not control_ids or len(set(control_ids)) != len(control_ids):
            raise ValueError("comparison control snapshot requires a unique closed inventory")
        return self


class DisplayPreview(StrictContract):
    schema_name: Literal["display_preview"] = "display_preview"
    schema_version: Literal[1] = 1
    text: str
    shown_bytes: NonNegativeInt
    total_bytes: NonNegativeInt
    sha256: Sha256
    media_type: NonEmptyStr
    truncated: bool
    source_reference: NonEmptyStr
    limitation: NonEmptyStr | None

    @model_validator(mode="after")
    def preview_shape_is_exact(self) -> Self:
        if len(self.text.encode("utf-8")) != self.shown_bytes:
            raise ValueError("display preview shown bytes do not match UTF-8 content")
        if self.shown_bytes > self.total_bytes:
            raise ValueError("display preview shown bytes exceed total bytes")
        if self.truncated != (self.shown_bytes < self.total_bytes):
            raise ValueError("display preview truncation does not match byte counts")
        if self.truncated and self.limitation is None:
            raise ValueError("truncated display preview requires a limitation")
        if not self.truncated and self.limitation is not None:
            raise ValueError("complete display preview cannot claim a truncation limitation")
        source_path = PurePosixPath(self.source_reference)
        if (
            source_path.is_absolute()
            or ".." in source_path.parts
            or "\\" in self.source_reference
            or (source_path.parts and source_path.parts[0].endswith(":"))
        ):
            raise ValueError("display preview source reference must be capsule-relative")
        return self


class ValidationClaimBoundary(StrictContract):
    schema_name: Literal["validation_claim_boundary"] = "validation_claim_boundary"
    schema_version: Literal[1] = 1
    status: Literal["pass", "diagnostic"]
    applicable_rule_count: NonNegativeInt
    executed_rule_count: NonNegativeInt
    passed_rule_count: NonNegativeInt
    failed_rule_count: NonNegativeInt
    not_applicable_rule_count: NonNegativeInt
    missing_rule_count: NonNegativeInt
    comparison_eligible: bool
    billing_complete: bool
    cost_complete: bool

    @model_validator(mode="after")
    def rule_counts_and_claims_close(self) -> Self:
        if self.executed_rule_count != (
            self.passed_rule_count + self.failed_rule_count + self.not_applicable_rule_count
        ):
            raise ValueError("claim-boundary executed rule counts do not close")
        if self.applicable_rule_count != self.executed_rule_count + self.missing_rule_count:
            raise ValueError("claim-boundary applicable rule counts do not close")
        if self.status == "pass" and (
            self.failed_rule_count
            or self.missing_rule_count
            or self.executed_rule_count != self.applicable_rule_count
        ):
            raise ValueError("PASS claim boundary requires complete successful validation")
        return self


class ReportRecordProjection(StrictContract):
    schema_name: Literal["report_record_projection"] = "report_record_projection"
    schema_version: Literal[1] = 1
    record_id: Sha256
    axis: Literal["logical-context", "plan", "case", "attempt"]
    label: NonEmptyStr
    status: NonEmptyStr
    failure_stage: str | None = None
    evaluation_status: str | None = None
    verdict: str | None = None
    capabilities_or_types: tuple[NonEmptyStr, ...]
    metric_ids: tuple[NonEmptyStr, ...]
    proof_statuses: tuple[ProofStatus, ...]
    raw_evidence_present: bool
    display_previews: tuple[DisplayPreview, ...] = ()
    latency_microseconds: NonNegativeInt | None = None
    context_view_tokens: NonNegativeInt | None = None
    declared_usage: NonNegativeInt | None = None
    detail_items: tuple[tuple[NonEmptyStr, str], ...]

    @model_validator(mode="after")
    def display_projection_is_bounded_and_unambiguous(self) -> Self:
        for label, values in (
            ("capability/type", self.capabilities_or_types),
            ("metric", self.metric_ids),
            ("proof status", self.proof_statuses),
            ("detail", tuple(item[0] for item in self.detail_items)),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"report projection contains duplicate {label} values")
        if not self.detail_items:
            raise ValueError("report projection requires precomputed display details")
        preview_references = tuple(item.source_reference for item in self.display_previews)
        if len(set(preview_references)) != len(preview_references):
            raise ValueError("report projection contains duplicate display preview references")
        return self


class ExactRational(StrictContract):
    schema_name: Literal["exact_rational"] = "exact_rational"
    schema_version: Literal[1] = 1
    numerator: int
    denominator: PositiveInt

    @model_validator(mode="after")
    def value_is_in_lowest_terms(self) -> Self:
        if gcd(abs(self.numerator), self.denominator) != 1:
            raise ValueError("exact rational must be in lowest terms")
        return self


class PairedMetricDelta(StrictContract):
    schema_name: Literal["paired_metric_delta"] = "paired_metric_delta"
    schema_version: Literal[1] = 1
    case_manifest_entry_id: Sha256
    left_case_occurrence_id: Sha256
    right_case_occurrence_id: Sha256
    metric_id: NonEmptyStr
    left: ExactRational
    right: ExactRational
    signed_delta: ExactRational
    absolute_delta: ExactRational

    @model_validator(mode="after")
    def delta_is_exact(self) -> Self:
        _require_exact_delta(
            self.left,
            self.right,
            self.signed_delta,
            self.absolute_delta,
            label="paired metric delta",
        )
        return self


class ComparisonPredicateResult(StrictContract):
    schema_name: Literal["comparison_predicate_result"] = "comparison_predicate_result"
    schema_version: Literal[1] = 1
    rule_id: NonEmptyStr
    expected_hash: Sha256
    left_hash: Sha256
    right_hash: Sha256
    passed: bool


class ComparisonCostDelta(StrictContract):
    schema_name: Literal["comparison_cost_delta"] = "comparison_cost_delta"
    schema_version: Literal[1] = 1
    cost_control_hash: Sha256
    view: ComparisonCostView
    dimension_id: NonEmptyStr
    unit: NonEmptyStr
    basis: Literal["raw_resource", "actual_supplier_charge", "estimate_from_measured_usage"]
    currency: str | None
    left: ExactRational
    right: ExactRational
    signed_delta: ExactRational
    absolute_delta: ExactRational

    @model_validator(mode="after")
    def cost_delta_is_exact(self) -> Self:
        _require_exact_delta(
            self.left,
            self.right,
            self.signed_delta,
            self.absolute_delta,
            label="comparison cost delta",
        )
        if self.view == ComparisonCostView.RAW_RESOURCE:
            if self.basis != "raw_resource" or self.currency is not None:
                raise ValueError("raw-resource delta cannot contain monetary fields")
        elif self.currency is None or len(self.currency) != 3:
            raise ValueError("monetary cost delta requires an ISO currency")
        return self


class AggregateMetricDelta(StrictContract):
    schema_name: Literal["aggregate_metric_delta"] = "aggregate_metric_delta"
    schema_version: Literal[1] = 1
    reducer_id: NonEmptyStr
    reducer_version: PositiveInt
    metric_id: NonEmptyStr
    denominator: PositiveInt
    aggregate_contract_hash: Sha256
    left: ExactRational
    right: ExactRational
    signed_delta: ExactRational
    absolute_delta: ExactRational

    @model_validator(mode="after")
    def aggregate_delta_is_exact(self) -> Self:
        _require_exact_delta(
            self.left,
            self.right,
            self.signed_delta,
            self.absolute_delta,
            label="aggregate metric delta",
        )
        return self


class ComparableComparisonReport(StrictContract):
    schema_name: Literal["comparison_report"] = "comparison_report"
    schema_version: Literal[1] = 1
    comparison_id: Sha256
    comparison_spec_hash: Sha256
    ordered_source_root_hashes: tuple[Sha256, Sha256]
    predicates: tuple[ComparisonPredicateResult, ...]
    comparable: Literal[True] = True
    paired_metric_deltas: tuple[PairedMetricDelta, ...]
    aggregate_metric_delta: AggregateMetricDelta | None
    cost_delta: ComparisonCostDelta | None
    winner: Literal["left", "right", "tie"] | None
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def every_predicate_passes(self) -> Self:
        if not self.predicates:
            raise ValueError("comparison requires a closed predicate inventory")
        rule_ids = tuple(item.rule_id for item in self.predicates)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("comparison predicate inventory contains duplicates")
        if not all(item.passed for item in self.predicates):
            raise ValueError("comparable comparison requires every predicate to pass")
        if (self.aggregate_metric_delta is None) != (self.winner is None):
            raise ValueError("winner requires exactly one validated aggregate metric delta")
        if self.aggregate_metric_delta is not None:
            signed = Fraction(
                self.aggregate_metric_delta.signed_delta.numerator,
                self.aggregate_metric_delta.signed_delta.denominator,
            )
            expected = "left" if signed > 0 else "right" if signed < 0 else "tie"
            if self.winner != expected:
                raise ValueError("winner does not match the validated aggregate metric delta")
        return self


class IncomparableComparisonReport(StrictContract):
    schema_name: Literal["comparison_report"] = "comparison_report"
    schema_version: Literal[1] = 1
    comparison_id: Sha256
    comparison_spec_hash: Sha256
    ordered_source_root_hashes: tuple[Sha256, Sha256]
    predicates: tuple[ComparisonPredicateResult, ...]
    comparable: Literal[False] = False
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def at_least_one_predicate_fails(self) -> Self:
        if not self.predicates:
            raise ValueError("comparison requires a closed predicate inventory")
        rule_ids = tuple(item.rule_id for item in self.predicates)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("comparison predicate inventory contains duplicates")
        if all(item.passed for item in self.predicates):
            raise ValueError("incomparable comparison requires a failed predicate")
        if not self.limitations:
            raise ValueError("incomparable comparison requires explicit limitations")
        return self


ComparisonReport = ComparableComparisonReport | IncomparableComparisonReport


def _require_exact_delta(
    left_value: ExactRational,
    right_value: ExactRational,
    signed_value: ExactRational,
    absolute_value: ExactRational,
    *,
    label: str,
) -> None:
    left = Fraction(left_value.numerator, left_value.denominator)
    right = Fraction(right_value.numerator, right_value.denominator)
    signed = Fraction(signed_value.numerator, signed_value.denominator)
    absolute = Fraction(absolute_value.numerator, absolute_value.denominator)
    if signed != left - right or absolute != abs(left - right):
        raise ValueError(f"{label} arithmetic does not close")


class EvaluationReviewBundle(StrictContract):
    schema_name: Literal["evaluation_review_bundle"] = "evaluation_review_bundle"
    schema_version: Literal[1] = 1
    bundle_id: Sha256
    phase_id: NonEmptyStr
    ordered_capsule_hashes: tuple[Sha256, ...]
    ordered_validation_hashes: tuple[Sha256, ...]
    ordered_case_occurrence_ids: tuple[Sha256, ...]
    unique_case_manifest_entry_ids: tuple[Sha256, ...]
    ordinary_derivation_hashes: tuple[Sha256, ...]
    report_model_hash: Sha256
    report_html_hash: Sha256
    export_validation_hash: Sha256
    ordered_review_input_hash: Sha256
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def generic_coverage_is_internally_closed(self) -> Self:
        if not self.ordered_capsule_hashes or len(self.ordered_capsule_hashes) != len(
            self.ordered_validation_hashes
        ):
            raise ValueError("review bundle capsule and validation roots must align")
        if not self.ordered_case_occurrence_ids or not self.unique_case_manifest_entry_ids:
            raise ValueError("review bundle requires case occurrence and manifest coverage")
        for label, values in (
            ("capsule", self.ordered_capsule_hashes),
            ("validation", self.ordered_validation_hashes),
            ("case occurrence", self.ordered_case_occurrence_ids),
            ("case manifest", self.unique_case_manifest_entry_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"review bundle contains duplicate {label} identities")
        if not self.ordinary_derivation_hashes:
            raise ValueError("review bundle requires ordinary derived artifacts")
        expected_input_hash = evaluation_review_input_hash(
            phase_id=self.phase_id,
            ordered_capsule_hashes=self.ordered_capsule_hashes,
            ordered_validation_hashes=self.ordered_validation_hashes,
            ordered_case_occurrence_ids=self.ordered_case_occurrence_ids,
            unique_case_manifest_entry_ids=self.unique_case_manifest_entry_ids,
            ordinary_derivation_hashes=self.ordinary_derivation_hashes,
            report_model_hash=self.report_model_hash,
            report_html_hash=self.report_html_hash,
            export_validation_hash=self.export_validation_hash,
            limitations=self.limitations,
        )
        if self.ordered_review_input_hash != expected_input_hash:
            raise ValueError("review bundle ordered review input hash does not match its payload")
        expected_bundle_id = evaluation_review_bundle_id(
            ordered_review_input_hash=self.ordered_review_input_hash
        )
        if self.bundle_id != expected_bundle_id:
            raise ValueError("review bundle identity does not match its ordered review input")
        return self


def evaluation_review_input_hash(**fields: object) -> str:
    return canonical_sha256(["oamb-evaluation-review-input-v1", fields])


def evaluation_review_bundle_id(**fields: object) -> str:
    return canonical_sha256(["oamb-evaluation-review-bundle-v1", fields])


class ReviewFindingSeverity(StrEnum):
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    ADVISORY = "advisory"


class AIReviewCaseProjection(StrictContract):
    schema_name: Literal["ai_review_case_projection"] = "ai_review_case_projection"
    schema_version: Literal[1] = 1
    projection_id: Sha256
    case_occurrence_id: Sha256
    case_manifest_entry_id: Sha256
    memory_system_id: NonEmptyStr
    workload_id: NonEmptyStr
    stratum: NonEmptyStr
    manifest_ordinal: PositiveInt
    completion_state: NonEmptyStr
    question: DisplayPreview
    accepted_answers: tuple[DisplayPreview, ...]
    retrieved_evidence: tuple[DisplayPreview, ...]
    metric_or_judge_hash: Sha256
    usage_proof_statuses: tuple[NonEmptyStr, ...]
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def projection_identity_and_required_evidence_close(self) -> Self:
        if not self.accepted_answers:
            raise ValueError("AI review case projection requires accepted answer evidence")
        if not self.usage_proof_statuses:
            raise ValueError("AI review case projection requires usage proof statuses")
        expected = ai_review_case_projection_id(
            case_occurrence_id=self.case_occurrence_id,
            case_manifest_entry_id=self.case_manifest_entry_id,
            memory_system_id=self.memory_system_id,
            workload_id=self.workload_id,
            stratum=self.stratum,
            manifest_ordinal=self.manifest_ordinal,
            completion_state=self.completion_state,
            question=self.question,
            accepted_answers=self.accepted_answers,
            retrieved_evidence=self.retrieved_evidence,
            metric_or_judge_hash=self.metric_or_judge_hash,
            usage_proof_statuses=self.usage_proof_statuses,
            limitations=self.limitations,
        )
        if self.projection_id != expected:
            raise ValueError("AI review case projection identity does not match its payload")
        return self


class AIReviewIntegrityProjection(StrictContract):
    schema_name: Literal["ai_review_integrity_projection"] = "ai_review_integrity_projection"
    schema_version: Literal[1] = 1
    integrity_id: Sha256
    phase_id: NonEmptyStr
    ordered_manifest_hashes: tuple[Sha256, ...]
    validation_inventory_hashes: tuple[Sha256, ...]
    reducer_hashes: tuple[Sha256, ...]
    comparison_hashes: tuple[Sha256, ...]
    accounting_hash: Sha256
    report_model_hash: Sha256
    report_html_hash: Sha256
    export_validation_hash: Sha256
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def integrity_identity_and_roots_close(self) -> Self:
        if not self.ordered_manifest_hashes or len(self.ordered_manifest_hashes) != len(
            self.validation_inventory_hashes
        ):
            raise ValueError("AI review integrity manifests and validations must align")
        expected = ai_review_integrity_projection_id(
            phase_id=self.phase_id,
            ordered_manifest_hashes=self.ordered_manifest_hashes,
            validation_inventory_hashes=self.validation_inventory_hashes,
            reducer_hashes=self.reducer_hashes,
            comparison_hashes=self.comparison_hashes,
            accounting_hash=self.accounting_hash,
            report_model_hash=self.report_model_hash,
            report_html_hash=self.report_html_hash,
            export_validation_hash=self.export_validation_hash,
            limitations=self.limitations,
        )
        if self.integrity_id != expected:
            raise ValueError("AI review integrity projection identity does not match its payload")
        return self


class AIReviewFinding(StrictContract):
    schema_name: Literal["ai_review_finding"] = "ai_review_finding"
    schema_version: Literal[1] = 1
    code: NonEmptyStr
    severity: ReviewFindingSeverity
    explanation: NonEmptyStr
    evidence_references: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def finding_is_bounded(self) -> Self:
        from oamb.constants import (
            AI_REVIEW_MAX_EVIDENCE_REFERENCES_PER_ITEM,
            AI_REVIEW_MAX_EXPLANATION_BYTES,
        )

        if len(self.explanation.encode("utf-8")) > AI_REVIEW_MAX_EXPLANATION_BYTES:
            raise ValueError("AI review finding explanation exceeds its byte limit")
        if len(self.evidence_references) > AI_REVIEW_MAX_EVIDENCE_REFERENCES_PER_ITEM:
            raise ValueError("AI review finding has too many evidence references")
        return self


def _require_bounded_review_findings(
    findings: tuple[AIReviewFinding, ...],
    *,
    result_kind: str,
) -> None:
    from oamb.constants import (
        AI_REVIEW_MAX_EVIDENCE_REFERENCES_PER_ITEM,
        AI_REVIEW_MAX_EXPLANATION_BYTES,
        AI_REVIEW_MAX_FINDINGS_PER_ITEM,
    )

    if len(findings) > AI_REVIEW_MAX_FINDINGS_PER_ITEM:
        raise ValueError(f"AI review {result_kind} result has too many findings")
    codes = tuple(item.code for item in findings)
    if len(set(codes)) != len(codes):
        raise ValueError(f"AI review {result_kind} result contains duplicate finding codes")
    if (
        sum(len(item.explanation.encode("utf-8")) for item in findings)
        > AI_REVIEW_MAX_EXPLANATION_BYTES
    ):
        raise ValueError(f"AI review {result_kind} explanation budget is exceeded")
    if (
        sum(len(item.evidence_references) for item in findings)
        > AI_REVIEW_MAX_EVIDENCE_REFERENCES_PER_ITEM
    ):
        raise ValueError(f"AI review {result_kind} evidence-reference budget is exceeded")


class AIReviewCaseResult(StrictContract):
    schema_name: Literal["ai_review_case_result"] = "ai_review_case_result"
    schema_version: Literal[1] = 1
    case_occurrence_id: Sha256
    status: QualityReviewStatus
    findings: tuple[AIReviewFinding, ...]

    @model_validator(mode="after")
    def result_findings_are_bounded_and_unique(self) -> Self:
        _require_bounded_review_findings(self.findings, result_kind="case")
        return self


class AIReviewBatchResult(StrictContract):
    schema_name: Literal["ai_review_batch_result"] = "ai_review_batch_result"
    schema_version: Literal[1] = 1
    batch_id: Sha256
    results: tuple[AIReviewCaseResult, ...]
    status: QualityReviewStatus

    @model_validator(mode="after")
    def result_ids_are_nonempty_and_unique(self) -> Self:
        result_ids = tuple(item.case_occurrence_id for item in self.results)
        if not result_ids or len(set(result_ids)) != len(result_ids):
            raise ValueError("AI review batch result coverage must be nonempty and unique")
        return self


class AIReviewIntegrityResult(StrictContract):
    schema_name: Literal["ai_review_integrity_result"] = "ai_review_integrity_result"
    schema_version: Literal[1] = 1
    integrity_id: Sha256
    status: QualityReviewStatus
    findings: tuple[AIReviewFinding, ...]

    @model_validator(mode="after")
    def integrity_findings_are_bounded_and_unique(self) -> Self:
        _require_bounded_review_findings(self.findings, result_kind="integrity")
        return self


class AIQualityReviewRecord(StrictContract):
    schema_name: Literal["ai_quality_review_record"] = "ai_quality_review_record"
    schema_version: Literal[1] = 1
    ai_review_record_id: Sha256
    review_bundle_hash: Sha256
    review_plan_hash: Sha256
    projection_spec_hash: Sha256
    finding_registry_hash: Sha256
    prompt_pack_hash: Sha256
    output_contract_hash: Sha256
    parser_hash: Sha256
    aggregate_version: NonEmptyStr
    reviewer_role_binding_hash: Sha256
    reviewer_model_hash: Sha256
    reviewer_runtime_hash: Sha256
    reviewer_configuration_hash: Sha256
    occurrence_id: Sha256
    ordinal: PositiveInt
    previous_ai_review_record_hash: Sha256 | None
    previous_history_root_hash: Sha256 | None
    history_root_hash: Sha256
    case_coverage_hash: Sha256
    batch_result_hashes: tuple[Sha256, ...]
    integrity_result_hash: Sha256
    status: QualityReviewStatus
    review_outcome_kind: Literal[
        "complete",
        "semantic_failure",
        "operational_inconclusive",
        "evidence_inconclusive",
    ]
    finding_codes: tuple[NonEmptyStr, ...]
    attempt_ids: tuple[Sha256, ...]
    usage_record_ids: tuple[Sha256, ...]
    resource_record_ids: tuple[Sha256, ...]
    cost_record_ids: tuple[Sha256, ...]
    accounting_closed: bool
    created_at: UtcDateTime

    @model_validator(mode="after")
    def review_record_is_coherent(self) -> Self:
        if self.ordinal == 1 and self.previous_ai_review_record_hash is not None:
            raise ValueError("first AI review cannot name a predecessor")
        if self.ordinal == 1 and self.previous_history_root_hash is not None:
            raise ValueError("first AI review cannot name a predecessor history")
        if self.ordinal > 1 and (
            self.previous_ai_review_record_hash is None or self.previous_history_root_hash is None
        ):
            raise ValueError("superseding AI review requires a predecessor")
        if self.status == QualityReviewStatus.PASS and not self.accounting_closed:
            raise ValueError("AI PASS requires closed review accounting")
        expected_kind = {
            QualityReviewStatus.PASS: "complete",
            QualityReviewStatus.FAIL: "semantic_failure",
        }.get(self.status)
        if expected_kind is not None and self.review_outcome_kind != expected_kind:
            raise ValueError("AI review outcome kind does not match its terminal status")
        if self.status == QualityReviewStatus.INCONCLUSIVE and self.review_outcome_kind not in {
            "operational_inconclusive",
            "evidence_inconclusive",
        }:
            raise ValueError("AI INCONCLUSIVE requires an explicit inconclusive kind")
        if len(set(self.finding_codes)) != len(self.finding_codes):
            raise ValueError("AI review contains duplicate finding codes")
        if not self.attempt_ids or len(set(self.attempt_ids)) != len(self.attempt_ids):
            raise ValueError("AI review record requires attempt evidence")
        for label, values in (
            ("usage", self.usage_record_ids),
            ("resource", self.resource_record_ids),
            ("cost", self.cost_record_ids),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"AI review contains duplicate {label} record references")
        expected_record_id = ai_quality_review_record_id(self)
        if self.ai_review_record_id != expected_record_id:
            raise ValueError("AI quality review record identity does not match its payload")
        expected_history_root = canonical_sha256(
            [
                "oamb-ai-review-history-v1",
                self.previous_history_root_hash,
                self.ai_review_record_id,
            ]
        )
        if self.history_root_hash != expected_history_root:
            raise ValueError("AI review history root does not match its predecessor chain")
        return self


def ai_review_case_projection_id(**fields: object) -> str:
    return canonical_sha256(["oamb-ai-review-case-projection-v1", fields])


def ai_review_integrity_projection_id(**fields: object) -> str:
    return canonical_sha256(["oamb-ai-review-integrity-projection-v1", fields])


def ai_quality_review_record_id(record: AIQualityReviewRecord) -> str:
    fields = record.model_dump(
        mode="python",
        exclude={
            "schema_name",
            "schema_version",
            "ai_review_record_id",
            "history_root_hash",
        },
    )
    return ai_quality_review_record_identity(fields)


def ai_quality_review_record_identity(fields: object) -> str:
    return canonical_sha256(["oamb-ai-quality-review-record-v1", fields])


class HumanReviewDecision(StrictContract):
    schema_name: Literal["human_review_decision"] = "human_review_decision"
    schema_version: Literal[1] = 1
    decision_id: Sha256
    review_bundle_hash: Sha256
    ai_review_record_hash: Sha256
    status: Literal["pass", "fail"]
    decided_at: UtcDateTime
    nonce: NonEmptyStr
    reviewer_label: NonEmptyStr
    finding_codes: tuple[NonEmptyStr, ...]
    evidence_references: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def decision_identity_and_findings_are_canonical(self) -> Self:
        if len(set(self.finding_codes)) != len(self.finding_codes):
            raise ValueError("human review decision contains duplicate finding codes")
        expected = human_review_decision_id(
            review_bundle_hash=self.review_bundle_hash,
            ai_review_record_hash=self.ai_review_record_hash,
            status=self.status,
            decided_at=self.decided_at,
            nonce=self.nonce,
            reviewer_label=self.reviewer_label,
            finding_codes=self.finding_codes,
            evidence_references=self.evidence_references,
        )
        if self.decision_id != expected:
            raise ValueError("human review decision identity does not match its payload")
        return self


class SignatureVerificationRecord(StrictContract):
    schema_name: Literal["signature_verification_record"] = "signature_verification_record"
    schema_version: Literal[1] = 1
    verification_id: Sha256
    algorithm: Literal["ed25519"] = "ed25519"
    key_binding_id: Sha256
    trusted_key_fingerprint: NonEmptyStr
    public_key_sha256: Sha256
    signed_payload_sha256: Sha256
    signature_sha256: Sha256
    verified_at: UtcDateTime

    @model_validator(mode="after")
    def verification_identity_is_canonical(self) -> Self:
        expected = signature_verification_record_id(
            key_binding_id=self.key_binding_id,
            trusted_key_fingerprint=self.trusted_key_fingerprint,
            public_key_sha256=self.public_key_sha256,
            signed_payload_sha256=self.signed_payload_sha256,
            signature_sha256=self.signature_sha256,
            verified_at=self.verified_at,
        )
        if self.verification_id != expected:
            raise ValueError("signature verification identity does not match its payload")
        return self


class HumanQualityReviewRecord(StrictContract):
    schema_name: Literal["human_quality_review_record"] = "human_quality_review_record"
    schema_version: Literal[1] = 1
    human_review_record_id: Sha256
    review_bundle_hash: Sha256
    ai_review_record_hash: Sha256
    decision_hash: Sha256
    signature_verification: SignatureVerificationRecord
    status: Literal["pass", "fail"]
    finding_codes: tuple[NonEmptyStr, ...]
    evidence_references: tuple[NonEmptyStr, ...]
    reviewer_label: NonEmptyStr
    decision_nonce: NonEmptyStr
    trusted_key_fingerprint: NonEmptyStr
    created_at: UtcDateTime

    @model_validator(mode="after")
    def human_record_identity_and_signature_bind(self) -> Self:
        if self.trusted_key_fingerprint != self.signature_verification.trusted_key_fingerprint:
            raise ValueError("human review record does not bind its trusted key")
        expected = human_quality_review_record_id(
            review_bundle_hash=self.review_bundle_hash,
            ai_review_record_hash=self.ai_review_record_hash,
            decision_hash=self.decision_hash,
            signature_verification=self.signature_verification,
            status=self.status,
            finding_codes=self.finding_codes,
            evidence_references=self.evidence_references,
            reviewer_label=self.reviewer_label,
            decision_nonce=self.decision_nonce,
            trusted_key_fingerprint=self.trusted_key_fingerprint,
            created_at=self.created_at,
        )
        if self.human_review_record_id != expected:
            raise ValueError("human review record identity does not match its payload")
        return self


class EvaluationPhaseGate(StrictContract):
    schema_name: Literal["evaluation_phase_gate"] = "evaluation_phase_gate"
    schema_version: Literal[1] = 1
    gate_id: Sha256
    phase_id: NonEmptyStr
    review_bundle_hash: Sha256
    review_history_root_hash: Sha256
    ordered_ai_review_record_hashes: tuple[Sha256, ...]
    canonical_ai_review_record_hash: Sha256
    human_review_record_hash: Sha256 | None
    passed_by_ai: bool
    passed_by_human: bool
    ai_record: AIQualityReviewRecord
    human_record: HumanQualityReviewRecord | None

    @model_validator(mode="after")
    def acceptance_flags_are_derived_from_current_records(self) -> Self:
        if (
            not self.ordered_ai_review_record_hashes
            or self.ordered_ai_review_record_hashes[-1] != self.canonical_ai_review_record_hash
            or len(set(self.ordered_ai_review_record_hashes))
            != len(self.ordered_ai_review_record_hashes)
        ):
            raise ValueError(
                "phase gate AI review history is empty, duplicated, or selects a non-head"
            )
        if self.ai_record.review_bundle_hash != self.review_bundle_hash:
            raise ValueError("AI record must bind the same review bundle")
        if self.canonical_ai_review_record_hash != self.ai_record.ai_review_record_id:
            raise ValueError("phase gate must bind the canonical AI record")
        expected_history_root = canonical_sha256(
            [
                "oamb-review-history-v1",
                self.ordered_ai_review_record_hashes,
                self.human_review_record_hash,
            ]
        )
        if self.review_history_root_hash != expected_history_root:
            raise ValueError("phase gate must bind the complete AI and human review history")
        expected_ai = self.ai_record.status == QualityReviewStatus.PASS
        if self.passed_by_ai != expected_ai:
            raise ValueError("passed_by_ai must derive from the current AI record")
        if self.human_record is None:
            if self.human_review_record_hash is not None or self.passed_by_human:
                raise ValueError("human acceptance requires a human review record")
        else:
            if not expected_ai:
                raise ValueError("human review requires the canonical AI PASS head")
            if (
                self.human_record.review_bundle_hash != self.review_bundle_hash
                or self.human_record.ai_review_record_hash != self.canonical_ai_review_record_hash
            ):
                raise ValueError(
                    "human record must bind the same review bundle and canonical AI record"
                )
            if self.human_review_record_hash != self.human_record.human_review_record_id:
                raise ValueError("phase gate must bind the human review record")
            expected_human = self.human_record.status == "pass"
            if self.passed_by_human != expected_human:
                raise ValueError("passed_by_human must derive after current AI PASS")
        expected_gate_id = evaluation_phase_gate_id(
            phase_id=self.phase_id,
            review_bundle_hash=self.review_bundle_hash,
            review_history_root_hash=self.review_history_root_hash,
            ordered_ai_review_record_hashes=self.ordered_ai_review_record_hashes,
            canonical_ai_review_record_hash=self.canonical_ai_review_record_hash,
            human_review_record_hash=self.human_review_record_hash,
            passed_by_ai=self.passed_by_ai,
            passed_by_human=self.passed_by_human,
        )
        if self.gate_id != expected_gate_id:
            raise ValueError("evaluation phase gate identity does not match its payload")
        return self


def human_review_decision_id(**fields: object) -> str:
    return canonical_sha256(["oamb-human-review-decision-v1", fields])


def signature_verification_record_id(**fields: object) -> str:
    return canonical_sha256(["oamb-signature-verification-record-v1", fields])


def human_quality_review_record_id(**fields: object) -> str:
    return canonical_sha256(["oamb-human-quality-review-record-v1", fields])


def evaluation_phase_gate_id(**fields: object) -> str:
    return canonical_sha256(["oamb-evaluation-phase-gate-v1", fields])


class RunSummary(StrictContract):
    schema_name: Literal["run_summary"] = "run_summary"
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    intended_logical_contexts: NonNegativeInt
    intended_ingestion_plans: NonNegativeInt
    ready_ingestion_plans: NonNegativeInt
    intended_cases: NonNegativeInt
    terminal_cases: NonNegativeInt
    completed_cases: NonNegativeInt
    errored_cases: NonNegativeInt
    unsupported_cases: NonNegativeInt
    cancelled_cases: NonNegativeInt
    budget_exceeded_cases: NonNegativeInt
    billing_complete: bool
    cost_complete: bool


class RunSummaryV2(StrictContract):
    schema_name: Literal["run_summary"] = "run_summary"
    schema_version: Literal[2] = 2
    run_id: NonEmptyStr
    intended_logical_contexts: NonNegativeInt
    intended_ingestion_plans: NonNegativeInt
    ready_ingestion_plans: NonNegativeInt
    intended_cases: NonNegativeInt
    terminal_cases: NonNegativeInt
    completed_cases: NonNegativeInt
    errored_cases: NonNegativeInt
    unsupported_cases: NonNegativeInt
    cancelled_cases: NonNegativeInt
    budget_exceeded_cases: NonNegativeInt
    parsed_cases: NonNegativeInt
    evaluated_cases: NonNegativeInt
    judged_cases: NonNegativeInt
    unjudged_cases: NonNegativeInt
    billing_complete: bool
    cost_complete: bool

    @model_validator(mode="after")
    def completion_counts_are_bounded(self) -> RunSummaryV2:
        terminal_parts = (
            self.completed_cases
            + self.errored_cases
            + self.unsupported_cases
            + self.cancelled_cases
            + self.budget_exceeded_cases
        )
        if self.ready_ingestion_plans > self.intended_ingestion_plans:
            raise ValueError("ready ingestion plans exceed intended plans")
        if self.terminal_cases > self.intended_cases or terminal_parts != self.terminal_cases:
            raise ValueError("terminal case counts do not close")
        if self.parsed_cases > self.terminal_cases:
            raise ValueError("parsed cases exceed terminal cases")
        if self.judged_cases > self.evaluated_cases:
            raise ValueError("judged cases exceed evaluated cases")
        if self.evaluated_cases + self.unjudged_cases > self.parsed_cases:
            raise ValueError("evaluation dispositions exceed parsed cases")
        if self.unjudged_cases > self.errored_cases:
            raise ValueError("unjudged cases require terminal errors")
        return self


class RunReportModel(StrictContract):
    schema_name: Literal["run_report_model"] = "run_report_model"
    schema_version: Literal[1] = 1
    report_id: Sha256
    source_manifest_hash: Sha256
    evidence_validation_hash: Sha256
    summary: RunSummary
    limitations: tuple[NonEmptyStr, ...]


class RunReportModelV2(StrictContract):
    schema_name: Literal["run_report_model"] = "run_report_model"
    schema_version: Literal[2] = 2
    report_id: Sha256
    source_manifest_hash: Sha256
    evidence_validation_hash: Sha256
    summary: RunSummaryV2
    limitations: tuple[NonEmptyStr, ...]


class ReportArtifactManifest(StrictContract):
    schema_name: Literal["report_artifact_manifest"] = "report_artifact_manifest"
    schema_version: Literal[1] = 1
    report_id: Sha256
    ordered_source_root_hashes: tuple[Sha256, ...]
    evidence_validation_hash: Sha256
    report_model_hash: Sha256
    renderer_hash: Sha256
    asset_hashes: tuple[Sha256, ...]
    audience: Literal["local", "public"]
    limitations: tuple[NonEmptyStr, ...]


class CompletionSummaryV3(StrictContract):
    schema_name: Literal["completion_summary"] = "completion_summary"
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    intended_logical_contexts: NonNegativeInt
    intended_ingestion_plans: NonNegativeInt
    ready_ingestion_plans: NonNegativeInt
    intended_cases: NonNegativeInt
    terminal_cases: NonNegativeInt
    completed_cases: NonNegativeInt
    errored_cases: NonNegativeInt
    unsupported_cases: NonNegativeInt
    cancelled_cases: NonNegativeInt
    budget_exceeded_cases: NonNegativeInt
    parsed_cases: NonNegativeInt
    evaluated_cases: NonNegativeInt
    judged_cases: NonNegativeInt
    unjudged_cases: NonNegativeInt
    metric_eligible_cases: NonNegativeInt

    @model_validator(mode="after")
    def completion_counts_are_closed(self) -> Self:
        terminal_parts = (
            self.completed_cases
            + self.errored_cases
            + self.unsupported_cases
            + self.cancelled_cases
            + self.budget_exceeded_cases
        )
        if self.ready_ingestion_plans > self.intended_ingestion_plans:
            raise ValueError("ready ingestion plans exceed intended plans")
        if self.terminal_cases > self.intended_cases or terminal_parts != self.terminal_cases:
            raise ValueError("terminal case counts do not close")
        if self.parsed_cases > self.terminal_cases:
            raise ValueError("parsed cases exceed terminal cases")
        if self.evaluated_cases > self.parsed_cases:
            raise ValueError("evaluated cases exceed parsed cases")
        if self.judged_cases > self.evaluated_cases:
            raise ValueError("judged cases exceed evaluated cases")
        if self.unjudged_cases > self.errored_cases:
            raise ValueError("unjudged cases require terminal errors")
        if self.metric_eligible_cases > self.evaluated_cases:
            raise ValueError("metric-eligible cases exceed evaluated cases")
        return self


class MetricSummary(StrictContract):
    schema_name: Literal["metric_summary"] = "metric_summary"
    schema_version: Literal[1] = 1
    metric_summary_id: Sha256
    metric_id: NonEmptyStr
    metric_version: PositiveInt
    stratum_id: NonEmptyStr
    score_numerator: int
    score_denominator: PositiveInt
    input_count: PositiveInt
    case_occurrence_ids: tuple[Sha256, ...]
    value: ExactRational
    claim_note: NonEmptyStr

    @model_validator(mode="after")
    def metric_identity_and_value_are_exact(self) -> Self:
        if len(self.case_occurrence_ids) != self.input_count or len(
            set(self.case_occurrence_ids)
        ) != len(self.case_occurrence_ids):
            raise ValueError("metric summary case coverage must be exact and unique")
        expected_value = Fraction(self.score_numerator, self.score_denominator)
        actual_value = Fraction(self.value.numerator, self.value.denominator)
        if actual_value != expected_value:
            raise ValueError("metric summary exact value does not match its fraction")
        fields = {
            "metric_id": self.metric_id,
            "metric_version": self.metric_version,
            "stratum_id": self.stratum_id,
            "score_numerator": self.score_numerator,
            "score_denominator": self.score_denominator,
            "input_count": self.input_count,
            "case_occurrence_ids": self.case_occurrence_ids,
            "value": self.value,
            "claim_note": self.claim_note,
        }
        if self.metric_summary_id != metric_summary_id(**fields):
            raise ValueError("metric summary identity does not match its payload")
        return self


class ReducerBinding(StrictContract):
    schema_name: Literal["reducer_binding"] = "reducer_binding"
    schema_version: Literal[1] = 1
    reducer_id: NonEmptyStr
    reducer_version: PositiveInt
    implementation_hash: Sha256


class MeasurementSummaryLine(StrictContract):
    schema_name: Literal["measurement_summary_line"] = "measurement_summary_line"
    schema_version: Literal[1] = 1
    dimension_id: NonEmptyStr
    stage: NonEmptyStr
    owner_kind: Literal[
        "ingestion_plan",
        "case",
        "phase_review",
        "model_readiness",
        "run",
    ]
    indexing_view: Literal["attempted", "final_contribution", "not_applicable"]
    value: ExactRational | None
    unit: NonEmptyStr
    proof_status: ProofStatus
    basis: Literal[
        "raw_resource",
        "actual_supplier_charge",
        "estimate_from_measured_usage",
        "token_count",
    ]
    currency: str | None
    source_record_ids: tuple[Sha256, ...]
    reason: NonEmptyStr | None

    @model_validator(mode="after")
    def proof_and_money_shape_are_explicit(self) -> Self:
        unavailable = self.proof_status in {
            ProofStatus.UNAVAILABLE,
            ProofStatus.NOT_APPLICABLE,
        }
        if unavailable:
            if self.value is not None or self.reason is None:
                raise ValueError("unavailable measurement requires no value and an explicit reason")
        elif self.value is None:
            raise ValueError("measured measurement requires an exact value")
        monetary = self.basis in {
            "actual_supplier_charge",
            "estimate_from_measured_usage",
        }
        if (monetary and not unavailable and self.currency is None) or (
            not monetary and self.currency is not None
        ):
            raise ValueError("measurement currency must appear only for monetary values")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("measurement currency must be an ISO code")
        if not self.source_record_ids or len(set(self.source_record_ids)) != len(
            self.source_record_ids
        ):
            raise ValueError("measurement summary requires unique source records")
        return self


class MabPlanEvidenceBinding(StrictContract):
    schema_name: Literal["mab_plan_evidence_binding"] = "mab_plan_evidence_binding"
    schema_version: Literal[1] = 1
    plan_manifest_entry_id: NonEmptyStr
    ingestion_plan_id: Sha256
    ingestion_occurrence_id: Sha256
    ordered_logical_context_ids: tuple[Sha256, ...]
    member_labels: tuple[NonEmptyStr, ...]
    ordered_case_occurrence_ids: tuple[Sha256, ...]
    metric_id: NonEmptyStr
    resolution_evidence_references: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def plan_evidence_inventory_is_closed(self) -> Self:
        if (
            not self.ordered_logical_context_ids
            or len(set(self.ordered_logical_context_ids)) != len(self.ordered_logical_context_ids)
            or len(self.member_labels) != len(self.ordered_logical_context_ids)
            or len(set(self.member_labels)) != len(self.member_labels)
        ):
            raise ValueError("MAB plan logical-member evidence inventory is incomplete")
        if not self.ordered_case_occurrence_ids or len(
            set(self.ordered_case_occurrence_ids)
        ) != len(self.ordered_case_occurrence_ids):
            raise ValueError("MAB plan case evidence inventory is incomplete")
        if len(set(self.resolution_evidence_references)) != len(
            self.resolution_evidence_references
        ):
            raise ValueError("MAB plan resolution evidence contains duplicates")
        if any(
            reference.startswith(("/", "../")) or "/../" in reference
            for reference in self.resolution_evidence_references
        ):
            raise ValueError("MAB plan resolution evidence must be capsule-relative")
        return self


class MabPlanMetricSummary(StrictContract):
    schema_name: Literal["mab_plan_metric_summary"] = "mab_plan_metric_summary"
    schema_version: Literal[1] = 1
    plan_manifest_entry_id: NonEmptyStr
    component: Literal["ar", "icl", "recsys", "lru", "cr_sf"]
    case_count: PositiveInt
    score: ExactRational
    evidence_binding: MabPlanEvidenceBinding

    @model_validator(mode="after")
    def plan_metric_evidence_is_exact(self) -> Self:
        binding = self.evidence_binding
        if (
            binding.plan_manifest_entry_id != self.plan_manifest_entry_id
            or len(binding.ordered_case_occurrence_ids) != self.case_count
        ):
            raise ValueError("MAB plan metric does not close from its evidence binding")
        expected_metric = {
            "ar": "mab-substring-em-v1",
            "icl": "mab-exact-v1",
            "recsys": "mab-redial-recall-at-5-v1",
            "lru": "mab-exact-v1",
            "cr_sf": "mab-substring-em-v1",
        }[self.component]
        if binding.metric_id != expected_metric:
            raise ValueError("MAB plan metric ID does not match its component")
        if self.component == "cr_sf":
            if len(binding.ordered_logical_context_ids) != 2 or not (
                "factconsolidation_sh_" in binding.member_labels[0].lower()
                and "factconsolidation_mh_" in binding.member_labels[1].lower()
            ):
                raise ValueError("FactConsolidation plan requires ordered SH/MH membership")
        elif len(binding.ordered_logical_context_ids) != 1:
            raise ValueError("non-grouped MAB plan requires exactly one logical member")
        if self.component == "recsys":
            if not binding.resolution_evidence_references:
                raise ValueError("ReDial plan requires resolution evidence")
        elif binding.resolution_evidence_references:
            raise ValueError("resolution evidence is owned only by the ReDial plan")
        return self


class MabComponentMetricSummary(StrictContract):
    schema_name: Literal["mab_component_metric_summary"] = "mab_component_metric_summary"
    schema_version: Literal[1] = 1
    component: Literal["ar", "icl", "recsys", "lru", "cr_sf"]
    plan_count: PositiveInt
    case_count: PositiveInt
    score: ExactRational


class MabCapabilityMetricSummary(StrictContract):
    schema_name: Literal["mab_capability_metric_summary"] = "mab_capability_metric_summary"
    schema_version: Literal[1] = 1
    capability: Literal["ar", "ttl", "lru", "cr_sf"]
    score: ExactRational
    weight: ExactRational
    weighted_contribution: ExactRational

    @model_validator(mode="after")
    def weighted_contribution_is_exact(self) -> Self:
        score = Fraction(self.score.numerator, self.score.denominator)
        weight = Fraction(self.weight.numerator, self.weight.denominator)
        contribution = Fraction(
            self.weighted_contribution.numerator,
            self.weighted_contribution.denominator,
        )
        if score * weight != contribution:
            raise ValueError("MAB capability weighted contribution does not close")
        return self


class Mab65ReportReduction(StrictContract):
    schema_name: Literal["mab65_report_reduction"] = "mab65_report_reduction"
    schema_version: Literal[1] = 1
    reducer_id: Literal["mab65-capability-balanced-index-v1"]
    available: bool
    unavailable_reason: NonEmptyStr | None = None
    plans: tuple[MabPlanMetricSummary, ...]
    components: tuple[MabComponentMetricSummary, ...]
    ttl_score: ExactRational | None = None
    capabilities: tuple[MabCapabilityMetricSummary, ...]
    index_value: ExactRational | None = None

    @model_validator(mode="after")
    def complete_only_inventory_and_waterfall_close(self) -> Self:
        if not self.available:
            if self.unavailable_reason is None:
                raise ValueError("unavailable MAB-65 index requires a reason")
            if (
                self.plans
                or self.components
                or self.ttl_score is not None
                or self.capabilities
                or self.index_value is not None
            ):
                raise ValueError("unavailable MAB-65 index cannot expose partial index fields")
            return self
        if (
            self.unavailable_reason is not None
            or self.ttl_score is None
            or self.index_value is None
        ):
            raise ValueError("available MAB-65 index requires complete exact values")
        expected_components = ("ar", "icl", "recsys", "lru", "cr_sf")
        expected_capabilities = ("ar", "ttl", "lru", "cr_sf")
        expected_plan_counts = {"ar": 5, "icl": 5, "recsys": 1, "lru": 10, "cr_sf": 4}
        expected_case_counts = {"ar": 15, "icl": 10, "recsys": 10, "lru": 15, "cr_sf": 15}
        expected_plan_components = tuple(
            component
            for component in expected_components
            for _ in range(expected_plan_counts[component])
        )
        if len(self.plans) != 25 or tuple(item.component for item in self.components) != (
            expected_components
        ):
            raise ValueError("available MAB-65 report requires exact component-first inventory")
        if tuple(item.component for item in self.plans) != expected_plan_components:
            raise ValueError("MAB-65 plan summaries require component-first order")
        if tuple(item.capability for item in self.capabilities) != expected_capabilities:
            raise ValueError("available MAB-65 report requires exact capability inventory")
        plan_manifest_ids = tuple(item.plan_manifest_entry_id for item in self.plans)
        ingestion_plan_ids = tuple(item.evidence_binding.ingestion_plan_id for item in self.plans)
        ingestion_occurrence_ids = tuple(
            item.evidence_binding.ingestion_occurrence_id for item in self.plans
        )
        logical_context_ids = tuple(
            context_id
            for item in self.plans
            for context_id in item.evidence_binding.ordered_logical_context_ids
        )
        case_occurrence_ids = tuple(
            case_id
            for item in self.plans
            for case_id in item.evidence_binding.ordered_case_occurrence_ids
        )
        if (
            len(set(plan_manifest_ids)) != 25
            or len(set(ingestion_plan_ids)) != 25
            or len(set(ingestion_occurrence_ids)) != 25
            or len(logical_context_ids) != 29
            or len(set(logical_context_ids)) != 29
            or len(case_occurrence_ids) != 65
            or len(set(case_occurrence_ids)) != 65
        ):
            raise ValueError("MAB-65 29/25/65 evidence inventory does not close")
        for component in self.components:
            component_plans = tuple(
                item for item in self.plans if item.component == component.component
            )
            if (
                component.plan_count != expected_plan_counts[component.component]
                or component.case_count != expected_case_counts[component.component]
                or len(component_plans) != component.plan_count
                or sum(item.case_count for item in component_plans) != component.case_count
            ):
                raise ValueError("MAB-65 component plan/case inventory does not close")
            plan_macro = sum(
                (
                    Fraction(item.score.numerator, item.score.denominator)
                    for item in component_plans
                ),
                Fraction(),
            ) / len(component_plans)
            if plan_macro != Fraction(component.score.numerator, component.score.denominator):
                raise ValueError("MAB-65 component macro score does not close")
        scores = {
            item.component: Fraction(item.score.numerator, item.score.denominator)
            for item in self.components
        }
        ttl = Fraction(self.ttl_score.numerator, self.ttl_score.denominator)
        if ttl != (scores["icl"] + scores["recsys"]) / 2:
            raise ValueError("MAB-65 TTL score does not close from ICL and ReDial")
        capability_score = {
            item.capability: Fraction(item.score.numerator, item.score.denominator)
            for item in self.capabilities
        }
        if capability_score != {
            "ar": scores["ar"],
            "ttl": ttl,
            "lru": scores["lru"],
            "cr_sf": scores["cr_sf"],
        }:
            raise ValueError("MAB-65 capability scores do not close from components")
        index_value = Fraction(self.index_value.numerator, self.index_value.denominator)
        contributions = sum(
            (
                Fraction(
                    item.weighted_contribution.numerator, item.weighted_contribution.denominator
                )
                for item in self.capabilities
            ),
            Fraction(),
        )
        if index_value != 100 * contributions:
            raise ValueError("MAB-65 capability-balanced index does not close")
        return self


class RunReportModelV3(StrictContract):
    schema_name: Literal["run_report_model"] = "run_report_model"
    schema_version: Literal[3] = 3
    report_id: Sha256
    report_spec_hash: Sha256
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    evidence_validation_profile_hash: Sha256
    evidence_validation_result_hash: Sha256
    reducer_bindings: tuple[ReducerBinding, ...]
    audience: Literal["local", "public"]
    origin_kind: Literal["native", "external"]
    capsule_id: Sha256 | None
    protocol_id: NonEmptyStr
    workload_id: NonEmptyStr
    memory_system_id: NonEmptyStr
    claim_boundary: ValidationClaimBoundary
    summary: CompletionSummaryV3
    metric_summaries: tuple[MetricSummary, ...]
    mab65_reduction: Mab65ReportReduction | None = None
    measurement_lines: tuple[MeasurementSummaryLine, ...]
    logical_context_ids: tuple[Sha256, ...]
    ingestion_occurrence_ids: tuple[Sha256, ...]
    case_occurrence_ids: tuple[Sha256, ...]
    attempt_ids: tuple[Sha256, ...]
    record_projections: tuple[ReportRecordProjection, ...]
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def single_source_inventory_and_identity_close(self) -> Self:
        if self.origin_kind == "native" and self.capsule_id is None:
            raise ValueError("native run report requires its capsule identity")
        if len(self.ordered_source_bindings) != 1:
            raise ValueError("run report requires exactly one ordered source binding")
        inventories = (
            ("logical context", self.logical_context_ids, self.summary.intended_logical_contexts),
            (
                "ingestion plan",
                self.ingestion_occurrence_ids,
                self.summary.intended_ingestion_plans,
            ),
            ("case occurrence", self.case_occurrence_ids, self.summary.intended_cases),
        )
        for label, values, count in inventories:
            if len(values) != count or len(set(values)) != len(values):
                raise ValueError(f"run report {label} inventory is incomplete or duplicated")
        if not self.attempt_ids or len(set(self.attempt_ids)) != len(self.attempt_ids):
            raise ValueError("run report attempt inventory must be nonempty and unique")
        reducer_ids = tuple(item.reducer_id for item in self.reducer_bindings)
        if not reducer_ids or len(set(reducer_ids)) != len(reducer_ids):
            raise ValueError("run report reducer inventory must be nonempty and unique")
        metric_ids = tuple(item.metric_summary_id for item in self.metric_summaries)
        if len(set(metric_ids)) != len(metric_ids):
            raise ValueError("run report metric summaries contain duplicate identities")
        projected_by_axis = {
            axis: tuple(
                projection.record_id
                for projection in self.record_projections
                if projection.axis == axis
            )
            for axis in ("logical-context", "plan", "case", "attempt")
        }
        if projected_by_axis != {
            "logical-context": self.logical_context_ids,
            "plan": self.ingestion_occurrence_ids,
            "case": self.case_occurrence_ids,
            "attempt": self.attempt_ids,
        }:
            raise ValueError("run report display projections do not close every record axis")
        if self.workload_id == "mab65-v1":
            if self.mab65_reduction is None:
                raise ValueError("MAB-65 run report requires an explicit reduction result")
            if self.mab65_reduction.available:
                bindings = tuple(item.evidence_binding for item in self.mab65_reduction.plans)
                expected_logical_context_ids = tuple(
                    context_id
                    for binding in bindings
                    for context_id in binding.ordered_logical_context_ids
                )
                expected_ingestion_occurrence_ids = tuple(
                    binding.ingestion_occurrence_id for binding in bindings
                )
                expected_case_occurrence_ids = tuple(
                    case_id
                    for binding in bindings
                    for case_id in binding.ordered_case_occurrence_ids
                )
                if (
                    self.logical_context_ids != expected_logical_context_ids
                    or self.ingestion_occurrence_ids != expected_ingestion_occurrence_ids
                    or self.case_occurrence_ids != expected_case_occurrence_ids
                    or self.summary.ready_ingestion_plans != 25
                    or self.summary.terminal_cases != 65
                    or self.summary.metric_eligible_cases != 65
                ):
                    raise ValueError("MAB-65 report topology does not close at 29/25/65")
            final_plan_cost_ids = tuple(
                source_record_id
                for line in self.measurement_lines
                if line.dimension_id == "supplier_cost"
                and line.owner_kind == "ingestion_plan"
                and line.indexing_view == "final_contribution"
                and line.basis in {"actual_supplier_charge", "estimate_from_measured_usage"}
                for source_record_id in line.source_record_ids
            )
            if len(final_plan_cost_ids) != 25 or len(set(final_plan_cost_ids)) != 25:
                raise ValueError(
                    "MAB-65 plan-owned final cost inventory must contain one unique record "
                    "per physical plan"
                )
        elif self.mab65_reduction is not None:
            raise ValueError("MAB-65 reduction cannot appear on another workload")
        fields = self.model_dump(
            mode="python",
            exclude={"schema_name", "schema_version", "report_id"},
        )
        if self.report_id != run_report_model_v3_id(**fields):
            raise ValueError("run report identity does not match its payload")
        return self


class DiagnosticRunReportModel(StrictContract):
    schema_name: Literal["diagnostic_run_report_model"] = "diagnostic_run_report_model"
    schema_version: Literal[1] = 1
    report_id: Sha256
    report_spec_hash: Sha256
    ordered_source_bindings: tuple[SourceEvidenceBinding]
    evidence_validation_profile_hash: Sha256
    evidence_validation_result_hash: Sha256
    origin_kind: Literal["native", "external"]
    run_id: NonEmptyStr
    validation_issue_codes: tuple[NonEmptyStr, ...]
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def restricted_diagnostic_identity_closes(self) -> Self:
        if self.run_id != self.ordered_source_bindings[0].source_identity:
            raise ValueError("diagnostic run identity does not match its source binding")
        if not self.validation_issue_codes or len(set(self.validation_issue_codes)) != len(
            self.validation_issue_codes
        ):
            raise ValueError("diagnostic report requires unique validation issue codes")
        if not self.limitations:
            raise ValueError("diagnostic report requires an explicit limitation")
        fields = self.model_dump(
            mode="python", exclude={"schema_name", "schema_version", "report_id"}
        )
        if self.report_id != diagnostic_run_report_model_id(**fields):
            raise ValueError("diagnostic report identity does not match its payload")
        return self


ComparisonReportPayload = Annotated[
    ComparableComparisonReport | IncomparableComparisonReport,
    Field(discriminator="comparable"),
]


class ComparisonReportModel(StrictContract):
    schema_name: Literal["comparison_report_model"] = "comparison_report_model"
    schema_version: Literal[1] = 1
    report_id: Sha256
    report_spec_hash: Sha256
    ordered_source_bindings: tuple[SourceEvidenceBinding, SourceEvidenceBinding]
    ordered_evidence_validation_hashes: tuple[Sha256, Sha256]
    left_run_report_hash: Sha256
    right_run_report_hash: Sha256
    claim_boundary: ValidationClaimBoundary
    comparison: ComparisonReportPayload
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def source_roots_and_identity_close(self) -> Self:
        roots = tuple(item.source_root_hash for item in self.ordered_source_bindings)
        if roots != self.comparison.ordered_source_root_hashes or roots[0] == roots[1]:
            raise ValueError("comparison report ordered source roots do not close")
        fields = self.model_dump(
            mode="python", exclude={"schema_name", "schema_version", "report_id"}
        )
        if self.report_id != comparison_report_model_id(**fields):
            raise ValueError("comparison report-model identity does not match its payload")
        return self


class ReleaseReportModel(StrictContract):
    schema_name: Literal["release_report_model"] = "release_report_model"
    schema_version: Literal[1] = 1
    report_id: Sha256
    report_spec_hash: Sha256
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    ordered_evidence_validation_hashes: tuple[Sha256, ...]
    run_report_hashes: tuple[Sha256, ...]
    comparison_report_hashes: tuple[Sha256, ...]
    claim_boundary: ValidationClaimBoundary
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def release_inventory_and_identity_close(self) -> Self:
        if len(self.ordered_source_bindings) < 2 or len(self.ordered_source_bindings) != len(
            self.ordered_evidence_validation_hashes
        ):
            raise ValueError("release report requires aligned multi-root evidence")
        if not self.run_report_hashes or len(set(self.run_report_hashes)) != len(
            self.run_report_hashes
        ):
            raise ValueError("release report run inventory must be nonempty and unique")
        if len(set(self.comparison_report_hashes)) != len(self.comparison_report_hashes):
            raise ValueError("release report comparison inventory contains duplicates")
        fields = self.model_dump(
            mode="python", exclude={"schema_name", "schema_version", "report_id"}
        )
        if self.report_id != release_report_model_id(**fields):
            raise ValueError("release report-model identity does not match its payload")
        return self


class PhaseAcceptanceReport(StrictContract):
    schema_name: Literal["phase_acceptance_report"] = "phase_acceptance_report"
    schema_version: Literal[1] = 1
    report_id: Sha256
    acceptance_report_spec_hash: Sha256
    phase_id: NonEmptyStr
    evaluation_report_hash: Sha256
    evaluation_export_validation_hash: Sha256
    review_bundle_hash: Sha256
    phase_gate_hash: Sha256
    ai_review_record_hash: Sha256
    human_review_record_hash: Sha256
    gate_review_bundle_hash: Sha256
    review_current: bool
    passed_by_ai: bool
    passed_by_human: bool
    finding_codes: tuple[NonEmptyStr, ...]
    evidence_references: tuple[NonEmptyStr, ...]
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def acceptance_is_derived_and_identity_is_canonical(self) -> Self:
        expected_current = self.review_bundle_hash == self.gate_review_bundle_hash
        if self.review_current != expected_current:
            raise ValueError("acceptance report review currency does not match its bundle roots")
        if not self.review_current and (self.passed_by_ai or self.passed_by_human):
            raise ValueError("stale acceptance report cannot carry passing flags")
        if self.passed_by_human and not self.passed_by_ai:
            raise ValueError("human acceptance requires current AI acceptance")
        if len(set(self.finding_codes)) != len(self.finding_codes):
            raise ValueError("acceptance report contains duplicate finding codes")
        fields = self.model_dump(
            mode="python", exclude={"schema_name", "schema_version", "report_id"}
        )
        if self.report_id != phase_acceptance_report_id(**fields):
            raise ValueError("phase acceptance report identity does not match its payload")
        return self


class ReportArtifactManifestV2(StrictContract):
    schema_name: Literal["report_artifact_manifest"] = "report_artifact_manifest"
    schema_version: Literal[2] = 2
    artifact_manifest_id: Sha256
    report_id: Sha256
    report_kind: Literal["run", "comparison", "release", "phase_acceptance"]
    report_identity_spec_binding: ReportIdentitySpecBinding
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    ordered_evidence_validation_hashes: tuple[Sha256, ...]
    report_model_hash: Sha256
    renderer_hash: Sha256
    asset_hashes: tuple[Sha256, ...]
    browser_contract_hash: Sha256
    performance_contract_hash: Sha256
    export_profile_selector_id: NonEmptyStr
    export_profile_selector_version: PositiveInt
    audience: Literal["local", "public"]
    schema_versions: tuple[NonEmptyStr, ...]
    limitations: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def artifact_inventory_and_identity_close(self) -> Self:
        if not self.ordered_source_bindings or len(self.ordered_source_bindings) != len(
            self.ordered_evidence_validation_hashes
        ):
            raise ValueError("report artifact source and validation roots must align")
        for label, values in (
            ("source binding", tuple(item.binding_id for item in self.ordered_source_bindings)),
            ("validation", self.ordered_evidence_validation_hashes),
            ("asset", self.asset_hashes),
            ("schema", self.schema_versions),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"report artifact contains duplicate {label} identities")
        fields = self.model_dump(
            mode="python",
            exclude={"schema_name", "schema_version", "artifact_manifest_id"},
        )
        if self.artifact_manifest_id != report_artifact_manifest_v2_id(**fields):
            raise ValueError("report artifact manifest identity does not match its payload")
        return self


def metric_summary_id(**fields: object) -> str:
    return canonical_sha256(["oamb-metric-summary-v1", fields])


def run_report_model_v3_id(**fields: object) -> str:
    return canonical_sha256(["oamb-run-report-model-v3", fields])


def diagnostic_run_report_model_id(**fields: object) -> str:
    return canonical_sha256(["oamb-diagnostic-run-report-model-v1", fields])


def comparison_report_model_id(**fields: object) -> str:
    return canonical_sha256(["oamb-comparison-report-model-v1", fields])


def release_report_model_id(**fields: object) -> str:
    return canonical_sha256(["oamb-release-report-model-v1", fields])


def phase_acceptance_report_id(**fields: object) -> str:
    return canonical_sha256(["oamb-phase-acceptance-report-v1", fields])


def report_artifact_manifest_v2_id(**fields: object) -> str:
    return canonical_sha256(["oamb-report-artifact-manifest-v2", fields])
