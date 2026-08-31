"""Pure builders for public/redacted report models."""

from __future__ import annotations

from fractions import Fraction
from typing import Literal

from oamb.contracts.accounting import ProofStatus
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import (
    ComparisonReport,
    ComparisonReportModel,
    CompletionSummaryV3,
    DiagnosticRunReportModel,
    DisplayPreview,
    EvaluationReportModel,
    ExactRational,
    Mab65ReportReduction,
    MabCapabilityMetricSummary,
    MabComponentMetricSummary,
    MabPlanEvidenceBinding,
    MabPlanMetricSummary,
    MeasurementSummaryLine,
    MetricSummary,
    ReducerBinding,
    ReleaseReportModel,
    ReportArtifactManifestV2,
    ReportArtifactManifestV3,
    ReportRecordProjection,
    RunReportModelV3,
    ValidationClaimBoundary,
    comparison_report_model_id,
    diagnostic_run_report_model_id,
    evaluation_report_model_id,
    metric_summary_id,
    release_report_model_id,
    report_artifact_manifest_v2_id,
    report_artifact_manifest_v3_id,
    report_artifact_manifest_v3_render_input_hash,
    run_report_model_v3_id,
)
from oamb.contracts.specifications import (
    EvaluationModelClosure,
    EvaluationModelClosureEntry,
    ReportIdentitySpecBinding,
    ReportIdentitySpecBindingV2,
    SourceEvidenceBinding,
    evaluation_model_closure_hash,
)
from oamb.workloads.mab65_reduction import Mab65Reduction


def _exact(value: Fraction) -> ExactRational:
    return ExactRational(numerator=value.numerator, denominator=value.denominator)


def build_validation_claim_boundary(
    *,
    status: Literal["pass", "diagnostic"],
    applicable_rule_count: int,
    executed_rule_count: int,
    passed_rule_count: int,
    failed_rule_count: int,
    not_applicable_rule_count: int,
    missing_rule_count: int,
    comparison_eligible: bool,
    billing_complete: bool,
    cost_complete: bool,
) -> ValidationClaimBoundary:
    return ValidationClaimBoundary(
        status=status,
        applicable_rule_count=applicable_rule_count,
        executed_rule_count=executed_rule_count,
        passed_rule_count=passed_rule_count,
        failed_rule_count=failed_rule_count,
        not_applicable_rule_count=not_applicable_rule_count,
        missing_rule_count=missing_rule_count,
        comparison_eligible=comparison_eligible,
        billing_complete=billing_complete,
        cost_complete=cost_complete,
    )


def build_report_record_projection(
    *,
    record_id: str,
    axis: Literal["logical-context", "plan", "case", "attempt"],
    label: str,
    status: str,
    failure_stage: str | None,
    evaluation_status: str | None,
    verdict: str | None,
    capabilities_or_types: tuple[str, ...],
    metric_ids: tuple[str, ...],
    proof_statuses: tuple[ProofStatus, ...],
    raw_evidence_present: bool,
    latency_microseconds: int | None,
    context_view_tokens: int | None,
    declared_usage: int | None,
    detail_items: tuple[tuple[str, str], ...],
    display_previews: tuple[DisplayPreview, ...] = (),
) -> ReportRecordProjection:
    return ReportRecordProjection(
        record_id=record_id,
        axis=axis,
        label=label,
        status=status,
        failure_stage=failure_stage,
        evaluation_status=evaluation_status,
        verdict=verdict,
        capabilities_or_types=capabilities_or_types,
        metric_ids=metric_ids,
        proof_statuses=proof_statuses,
        raw_evidence_present=raw_evidence_present,
        display_previews=display_previews,
        latency_microseconds=latency_microseconds,
        context_view_tokens=context_view_tokens,
        declared_usage=declared_usage,
        detail_items=detail_items,
    )


def build_metric_summary(
    *,
    metric_id: str,
    metric_version: int,
    stratum_id: str,
    score_numerator: int,
    score_denominator: int,
    input_count: int,
    case_occurrence_ids: tuple[str, ...],
    claim_note: str,
) -> MetricSummary:
    value = _exact(Fraction(score_numerator, score_denominator))
    fields = {
        "metric_id": metric_id,
        "metric_version": metric_version,
        "stratum_id": stratum_id,
        "score_numerator": score_numerator,
        "score_denominator": score_denominator,
        "input_count": input_count,
        "case_occurrence_ids": case_occurrence_ids,
        "value": value,
        "claim_note": claim_note,
    }
    return MetricSummary.model_validate(
        {"metric_summary_id": metric_summary_id(**fields), **fields}
    )


def build_run_report_model(
    *,
    report_spec_hash: str,
    source_binding: SourceEvidenceBinding,
    evidence_validation_profile_hash: str,
    evidence_validation_result_hash: str,
    reducer_bindings: tuple[tuple[str, int, str], ...],
    audience: Literal["local", "public"],
    origin_kind: Literal["native", "external"],
    capsule_id: str | None,
    protocol_id: str,
    workload_id: str,
    memory_system_id: str,
    claim_boundary: ValidationClaimBoundary,
    summary: CompletionSummaryV3,
    metric_summaries: tuple[MetricSummary, ...],
    mab65_reduction: Mab65ReportReduction | None = None,
    measurement_lines: tuple[MeasurementSummaryLine, ...],
    logical_context_ids: tuple[str, ...],
    ingestion_occurrence_ids: tuple[str, ...],
    case_occurrence_ids: tuple[str, ...],
    attempt_ids: tuple[str, ...],
    record_projections: tuple[ReportRecordProjection, ...],
    limitations: tuple[str, ...],
) -> RunReportModelV3:
    typed_reducers = tuple(
        ReducerBinding(
            reducer_id=reducer_id,
            reducer_version=reducer_version,
            implementation_hash=implementation_hash,
        )
        for reducer_id, reducer_version, implementation_hash in reducer_bindings
    )
    provisional = RunReportModelV3.model_construct(
        report_id="0" * 64,
        report_spec_hash=report_spec_hash,
        ordered_source_bindings=(source_binding,),
        evidence_validation_profile_hash=evidence_validation_profile_hash,
        evidence_validation_result_hash=evidence_validation_result_hash,
        reducer_bindings=typed_reducers,
        audience=audience,
        origin_kind=origin_kind,
        capsule_id=capsule_id,
        protocol_id=protocol_id,
        workload_id=workload_id,
        memory_system_id=memory_system_id,
        claim_boundary=claim_boundary,
        summary=summary,
        metric_summaries=metric_summaries,
        mab65_reduction=mab65_reduction,
        measurement_lines=measurement_lines,
        logical_context_ids=logical_context_ids,
        ingestion_occurrence_ids=ingestion_occurrence_ids,
        case_occurrence_ids=case_occurrence_ids,
        attempt_ids=attempt_ids,
        record_projections=record_projections,
        limitations=limitations,
    )
    identity_fields = provisional.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "report_id"},
    )
    return RunReportModelV3.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "report_id": run_report_model_v3_id(**identity_fields),
        }
    )


def build_diagnostic_run_report_model(
    *,
    report_spec_hash: str,
    source_binding: SourceEvidenceBinding,
    evidence_validation_profile_hash: str,
    evidence_validation_result_hash: str,
    origin_kind: Literal["native", "external"],
    run_id: str,
    validation_issue_codes: tuple[str, ...],
    limitations: tuple[str, ...],
) -> DiagnosticRunReportModel:
    fields = {
        "report_spec_hash": report_spec_hash,
        "ordered_source_bindings": (source_binding,),
        "evidence_validation_profile_hash": evidence_validation_profile_hash,
        "evidence_validation_result_hash": evidence_validation_result_hash,
        "origin_kind": origin_kind,
        "run_id": run_id,
        "validation_issue_codes": validation_issue_codes,
        "limitations": limitations,
    }
    return DiagnosticRunReportModel.model_validate(
        {"report_id": diagnostic_run_report_model_id(**fields), **fields}
    )


def build_mab65_report_reduction(
    reduction: Mab65Reduction,
    *,
    plan_evidence_bindings: tuple[MabPlanEvidenceBinding, ...] = (),
) -> Mab65ReportReduction:
    if not reduction.available:
        if plan_evidence_bindings:
            raise ValueError("unavailable MAB-65 reduction cannot bind partial plan evidence")
        return Mab65ReportReduction(
            reducer_id="mab65-capability-balanced-index-v1",
            available=False,
            unavailable_reason=reduction.unavailable_reason,
            plans=(),
            components=(),
            ttl_score=None,
            capabilities=(),
            index_value=None,
        )
    if reduction.ttl_score is None or reduction.index_value is None:
        raise ValueError("available MAB-65 reduction is missing exact values")
    bindings_by_plan = {item.plan_manifest_entry_id: item for item in plan_evidence_bindings}
    reduction_plan_ids = tuple(item.plan_manifest_entry_id for item in reduction.plans)
    if (
        len(bindings_by_plan) != len(plan_evidence_bindings)
        or len(set(reduction_plan_ids)) != len(reduction_plan_ids)
        or set(reduction_plan_ids) != set(bindings_by_plan)
    ):
        raise ValueError("available MAB-65 plan evidence inventory is incomplete")
    component_order = {
        component: index for index, component in enumerate(("ar", "icl", "recsys", "lru", "cr_sf"))
    }
    ordered_plans = tuple(sorted(reduction.plans, key=lambda item: component_order[item.component]))
    return Mab65ReportReduction(
        reducer_id="mab65-capability-balanced-index-v1",
        available=True,
        unavailable_reason=None,
        plans=tuple(
            MabPlanMetricSummary.model_validate(
                {
                    "plan_manifest_entry_id": item.plan_manifest_entry_id,
                    "component": item.component,
                    "case_count": item.case_count,
                    "score": _exact(item.score),
                    "evidence_binding": bindings_by_plan[item.plan_manifest_entry_id],
                }
            )
            for item in ordered_plans
        ),
        components=tuple(
            MabComponentMetricSummary.model_validate(
                {
                    "component": item.component,
                    "plan_count": item.plan_count,
                    "case_count": item.case_count,
                    "score": _exact(item.score),
                }
            )
            for item in reduction.components
        ),
        ttl_score=_exact(reduction.ttl_score),
        capabilities=tuple(
            MabCapabilityMetricSummary.model_validate(
                {
                    "capability": item.capability,
                    "score": _exact(item.score),
                    "weight": _exact(item.weight),
                    "weighted_contribution": _exact(item.weighted_contribution),
                }
            )
            for item in reduction.capabilities
        ),
        index_value=_exact(reduction.index_value),
    )


def build_comparison_report_model(
    *,
    report_spec_hash: str,
    ordered_source_bindings: tuple[SourceEvidenceBinding, SourceEvidenceBinding],
    ordered_evidence_validation_hashes: tuple[str, str],
    left_run_report_hash: str,
    right_run_report_hash: str,
    claim_boundary: ValidationClaimBoundary,
    comparison: ComparisonReport,
    limitations: tuple[str, ...],
) -> ComparisonReportModel:
    fields = {
        "report_spec_hash": report_spec_hash,
        "ordered_source_bindings": ordered_source_bindings,
        "ordered_evidence_validation_hashes": ordered_evidence_validation_hashes,
        "left_run_report_hash": left_run_report_hash,
        "right_run_report_hash": right_run_report_hash,
        "claim_boundary": claim_boundary,
        "comparison": comparison,
        "limitations": limitations,
    }
    return ComparisonReportModel.model_validate(
        {"report_id": comparison_report_model_id(**fields), **fields}
    )


def build_evaluation_report_model(
    *,
    phase_id: str,
    report_spec_hash: str,
    ordered_run_models: tuple[
        RunReportModelV3,
        RunReportModelV3,
        RunReportModelV3,
        RunReportModelV3,
    ],
    eligible_comparison_models: tuple[ComparisonReportModel, ...],
    unique_case_count: int,
    limitations: tuple[str, ...],
) -> EvaluationReportModel:
    fields = {
        "phase_id": phase_id,
        "report_spec_hash": report_spec_hash,
        "ordered_run_models": ordered_run_models,
        "ordered_run_model_hashes": tuple(
            canonical_sha256(item.model_dump(mode="python")) for item in ordered_run_models
        ),
        "eligible_comparison_models": eligible_comparison_models,
        "eligible_comparison_model_hashes": tuple(
            canonical_sha256(item.model_dump(mode="python")) for item in eligible_comparison_models
        ),
        "unique_case_count": unique_case_count,
        "system_result_count": sum(item.summary.intended_cases for item in ordered_run_models),
        "limitations": limitations,
    }
    return EvaluationReportModel.model_validate(
        {"report_id": evaluation_report_model_id(**fields), **fields}
    )


def build_evaluation_model_closure(
    model: EvaluationReportModel,
) -> EvaluationModelClosure:
    run_entries = tuple(
        EvaluationModelClosureEntry(
            model_kind="run",
            report_model_id=run.report_id,
            canonical_bytes_hash=canonical_sha256(run),
            ordered_source_root_hash=_ordered_model_source_root_hash(run.ordered_source_bindings),
            evidence_validation_result_hash=_ordered_model_validation_hash(
                (run.evidence_validation_result_hash,)
            ),
            coverage_hash=canonical_sha256(
                [
                    "oamb-evaluation-run-model-coverage-v1",
                    run.logical_context_ids,
                    run.ingestion_occurrence_ids,
                    run.case_occurrence_ids,
                    tuple(item.record_id for item in run.record_projections),
                ]
            ),
        )
        for run in model.ordered_run_models
    )
    comparison_entries = tuple(
        EvaluationModelClosureEntry(
            model_kind="comparison",
            report_model_id=comparison.report_id,
            canonical_bytes_hash=canonical_sha256(comparison),
            ordered_source_root_hash=_ordered_model_source_root_hash(
                comparison.ordered_source_bindings
            ),
            evidence_validation_result_hash=_ordered_model_validation_hash(
                comparison.ordered_evidence_validation_hashes
            ),
            coverage_hash=canonical_sha256(
                [
                    "oamb-evaluation-comparison-model-coverage-v1",
                    comparison.left_run_report_hash,
                    comparison.right_run_report_hash,
                    comparison.comparison,
                ]
            ),
        )
        for comparison in model.eligible_comparison_models
    )
    fields: dict[str, object] = {
        "ordered_run_entries": run_entries,
        "eligible_comparison_entries": comparison_entries,
    }
    return EvaluationModelClosure.model_validate(
        {
            **fields,
            "model_closure_hash": evaluation_model_closure_hash(**fields),
        }
    )


def _ordered_model_source_root_hash(
    bindings: tuple[SourceEvidenceBinding, ...],
) -> str:
    return canonical_sha256(
        ["oamb-evaluation-model-source-roots-v1", tuple(item.source_root_hash for item in bindings)]
    )


def _ordered_model_validation_hash(validation_hashes: tuple[str, ...]) -> str:
    return canonical_sha256(["oamb-evaluation-model-validations-v1", validation_hashes])


def build_release_report_model(
    *,
    report_spec_hash: str,
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...],
    ordered_evidence_validation_hashes: tuple[str, ...],
    run_report_hashes: tuple[str, ...],
    comparison_report_hashes: tuple[str, ...],
    claim_boundary: ValidationClaimBoundary,
    limitations: tuple[str, ...],
) -> ReleaseReportModel:
    fields = {
        "report_spec_hash": report_spec_hash,
        "ordered_source_bindings": ordered_source_bindings,
        "ordered_evidence_validation_hashes": ordered_evidence_validation_hashes,
        "run_report_hashes": run_report_hashes,
        "comparison_report_hashes": comparison_report_hashes,
        "claim_boundary": claim_boundary,
        "limitations": limitations,
    }
    return ReleaseReportModel.model_validate(
        {"report_id": release_report_model_id(**fields), **fields}
    )


def build_report_artifact_manifest_v2(
    *,
    report_id: str,
    report_kind: Literal["run", "comparison", "release"],
    report_identity_spec_binding: ReportIdentitySpecBinding,
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...],
    ordered_evidence_validation_hashes: tuple[str, ...],
    report_model_hash: str,
    renderer_hash: str,
    asset_hashes: tuple[str, ...],
    browser_contract_hash: str,
    performance_contract_hash: str,
    export_profile_selector_id: str,
    export_profile_selector_version: int,
    audience: Literal["local", "public"],
    schema_versions: tuple[str, ...],
    limitations: tuple[str, ...],
) -> ReportArtifactManifestV2:
    fields = {
        "report_id": report_id,
        "report_kind": report_kind,
        "report_identity_spec_binding": report_identity_spec_binding,
        "ordered_source_bindings": ordered_source_bindings,
        "ordered_evidence_validation_hashes": ordered_evidence_validation_hashes,
        "report_model_hash": report_model_hash,
        "renderer_hash": renderer_hash,
        "asset_hashes": asset_hashes,
        "browser_contract_hash": browser_contract_hash,
        "performance_contract_hash": performance_contract_hash,
        "export_profile_selector_id": export_profile_selector_id,
        "export_profile_selector_version": export_profile_selector_version,
        "audience": audience,
        "schema_versions": schema_versions,
        "limitations": limitations,
    }
    return ReportArtifactManifestV2.model_validate(
        {
            "artifact_manifest_id": report_artifact_manifest_v2_id(**fields),
            **fields,
        }
    )


def build_report_artifact_manifest_v3(
    *,
    report_id: str,
    report_identity_spec_binding: ReportIdentitySpecBindingV2,
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...],
    ordered_evidence_validation_hashes: tuple[str, ...],
    report_model_hash: str,
    evaluation_model_closure: EvaluationModelClosure,
    renderer_hash: str,
    asset_hashes: tuple[str, ...],
    browser_contract_hash: str,
    performance_contract_hash: str,
    export_profile_selector_id: str,
    export_profile_selector_version: int,
    export_profile_hash: str,
    audience: Literal["local", "public"],
    schema_versions: tuple[str, ...],
    limitations: tuple[str, ...],
) -> ReportArtifactManifestV3:
    fields = {
        "report_id": report_id,
        "report_kind": "evaluation",
        "report_identity_spec_binding": report_identity_spec_binding,
        "ordered_source_bindings": ordered_source_bindings,
        "ordered_evidence_validation_hashes": ordered_evidence_validation_hashes,
        "report_model_hash": report_model_hash,
        "evaluation_model_closure": evaluation_model_closure,
        "renderer_hash": renderer_hash,
        "asset_hashes": asset_hashes,
        "browser_contract_hash": browser_contract_hash,
        "performance_contract_hash": performance_contract_hash,
        "export_profile_selector_id": export_profile_selector_id,
        "export_profile_selector_version": export_profile_selector_version,
        "export_profile_hash": export_profile_hash,
        "audience": audience,
        "schema_versions": schema_versions,
        "limitations": limitations,
    }
    render_input_hash = report_artifact_manifest_v3_render_input_hash(**fields)
    return ReportArtifactManifestV3.model_validate(
        {
            "artifact_manifest_id": report_artifact_manifest_v3_id(
                **fields, render_input_hash=render_input_hash
            ),
            "render_input_hash": render_input_hash,
            **fields,
        }
    )


def report_model_hash(model: object) -> str:
    return canonical_sha256(model)


__all__ = [
    "build_comparison_report_model",
    "build_diagnostic_run_report_model",
    "build_evaluation_model_closure",
    "build_evaluation_report_model",
    "build_mab65_report_reduction",
    "build_metric_summary",
    "build_release_report_model",
    "build_report_artifact_manifest_v2",
    "build_report_artifact_manifest_v3",
    "build_run_report_model",
    "report_model_hash",
]
