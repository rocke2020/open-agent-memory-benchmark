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
