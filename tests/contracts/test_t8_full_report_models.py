from __future__ import annotations

import importlib
from types import ModuleType

import pytest
from pydantic import ValidationError

from oamb.contracts.specifications import SourceEvidenceBinding, SourceEvidenceKind

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _reporting() -> ModuleType:
    module = importlib.import_module("oamb.contracts.reporting")
    required = (
        "CompletionSummaryV3",
        "MetricSummary",
        "RunReportModelV3",
        "ComparisonReportModel",
        "ReleaseReportModel",
        "ReportArtifactManifestV2",
    )
    missing = tuple(name for name in required if not hasattr(module, name))
    if missing:
        pytest.fail(f"full T8 report contracts are missing: {missing}", pytrace=False)
    return module


def _source(root: str, identity: str) -> SourceEvidenceBinding:
    return SourceEvidenceBinding(
        binding_id=root,
        source_kind=SourceEvidenceKind.RUN,
        source_identity=identity,
        source_root_hash=root,
        validation_result_hash=SHA_D,
        source_schema_versions=("capsule_manifest@1",),
    )


def test_completion_summary_keeps_logical_rows_physical_plans_and_cases_distinct() -> None:
    reporting = _reporting()
    summary = reporting.CompletionSummaryV3(
        run_id="mab65-fixture",
        intended_logical_contexts=29,
        intended_ingestion_plans=25,
        ready_ingestion_plans=25,
        intended_cases=65,
        terminal_cases=65,
        completed_cases=64,
        errored_cases=1,
        unsupported_cases=0,
        cancelled_cases=0,
        budget_exceeded_cases=0,
        parsed_cases=64,
        evaluated_cases=64,
        judged_cases=0,
        unjudged_cases=0,
        metric_eligible_cases=64,
    )

    assert (summary.intended_logical_contexts, summary.intended_ingestion_plans) == (29, 25)
    with pytest.raises(ValidationError, match="terminal case counts"):
        reporting.CompletionSummaryV3.model_validate(
            {**summary.model_dump(mode="python"), "terminal_cases": 64}
        )


def test_pre_review_run_model_has_no_acceptance_flags() -> None:
    reporting = _reporting()
    public = importlib.import_module("oamb.reporting.public")
    if not hasattr(public, "build_run_report_model"):
        pytest.fail("T8 run report model builder is not implemented", pytrace=False)
    summary = reporting.CompletionSummaryV3(
        run_id="native-fixture",
        intended_logical_contexts=1,
        intended_ingestion_plans=1,
        ready_ingestion_plans=1,
        intended_cases=1,
        terminal_cases=1,
        completed_cases=1,
        errored_cases=0,
        unsupported_cases=0,
        cancelled_cases=0,
        budget_exceeded_cases=0,
        parsed_cases=1,
        evaluated_cases=1,
        judged_cases=1,
        unjudged_cases=0,
        metric_eligible_cases=1,
    )
    metric = public.build_metric_summary(
        metric_id="lme-judge-v1",
        metric_version=1,
        stratum_id="all",
        score_numerator=1,
        score_denominator=1,
        input_count=1,
        case_occurrence_ids=(SHA_A,),
        claim_note="fixture-only exact judge result",
    )
    model = public.build_run_report_model(
        report_spec_hash=SHA_A,
        source_binding=_source(SHA_B, "native-fixture"),
        evidence_validation_profile_hash=SHA_C,
        evidence_validation_result_hash=SHA_D,
        reducer_bindings=(("completion-v1", 1, SHA_A),),
        audience="public",
        origin_kind="native",
        capsule_id=SHA_A,
        protocol_id="lme-v1",
        workload_id="lme6-live-smoke-v1",
        memory_system_id="fixture-memory",
        claim_boundary=public.build_validation_claim_boundary(
            status="pass",
            applicable_rule_count=1,
            executed_rule_count=1,
            passed_rule_count=1,
            failed_rule_count=0,
            not_applicable_rule_count=0,
            missing_rule_count=0,
            comparison_eligible=True,
            billing_complete=False,
            cost_complete=False,
        ),
        summary=summary,
        metric_summaries=(metric,),
        measurement_lines=(),
        logical_context_ids=(SHA_A,),
        ingestion_occurrence_ids=(SHA_B,),
        case_occurrence_ids=(SHA_C,),
        attempt_ids=(SHA_D,),
        record_projections=tuple(
            public.build_report_record_projection(
                record_id=record_id,
                axis=axis,
                label=f"Fixture {axis}",
                status=status,
                failure_stage=None,
                evaluation_status=("judged" if axis == "case" else None),
                verdict=("pass" if axis == "case" else None),
                capabilities_or_types=(),
                metric_ids=(("lme-judge-v1",) if axis == "case" else ()),
                proof_statuses=(),
                raw_evidence_present=axis in {"case", "attempt"},
                latency_microseconds=None,
                context_view_tokens=None,
                declared_usage=None,
                detail_items=(("identity", record_id),),
            )
            for record_id, axis, status in (
                (SHA_A, "logical-context", "included"),
                (SHA_B, "plan", "sealed"),
                (SHA_C, "case", "completed"),
                (SHA_D, "attempt", "succeeded"),
            )
        ),
        limitations=("fixture only",),
    )

    payload = model.model_dump(mode="json")
    assert "passed_by_ai" not in payload
    assert "passed_by_human" not in payload
    assert payload["summary"]["intended_logical_contexts"] == 1


def test_incomparable_comparison_model_cannot_smuggle_delta_or_winner() -> None:
    reporting = _reporting()
    public = importlib.import_module("oamb.reporting.public")
    failed = reporting.ComparisonPredicateResult(
        rule_id="comparison.dataset-revision.v1",
        expected_hash=SHA_A,
        left_hash=SHA_A,
        right_hash=SHA_B,
        passed=False,
    )
    comparison = reporting.IncomparableComparisonReport(
        comparison_id=SHA_D,
        comparison_spec_hash=SHA_A,
        ordered_source_root_hashes=(SHA_B, SHA_C),
        predicates=(failed,),
        limitations=("dataset revisions differ",),
    )
    model = public.build_comparison_report_model(
        report_spec_hash=SHA_A,
        ordered_source_bindings=(_source(SHA_B, "left"), _source(SHA_C, "right")),
        ordered_evidence_validation_hashes=(SHA_A, SHA_D),
        left_run_report_hash=SHA_B,
        right_run_report_hash=SHA_C,
        claim_boundary=public.build_validation_claim_boundary(
            status="pass",
            applicable_rule_count=2,
            executed_rule_count=2,
            passed_rule_count=2,
            failed_rule_count=0,
            not_applicable_rule_count=0,
            missing_rule_count=0,
            comparison_eligible=False,
            billing_complete=False,
            cost_complete=False,
        ),
        comparison=comparison,
        limitations=("not comparable",),
    )

    payload = model.model_dump(mode="json")
    assert "winner" not in payload["comparison"]
    with pytest.raises(ValidationError):
        reporting.ComparisonReportModel.model_validate(
            {
                **payload,
                "comparison": {**payload["comparison"], "winner": "left"},
            }
        )
