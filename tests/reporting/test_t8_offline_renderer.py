from __future__ import annotations

import base64
import hashlib
import importlib
import re
from types import ModuleType

import pytest

from oamb.contracts.accounting import ProofStatus
from oamb.contracts.ids import canonical_json_bytes
from oamb.contracts.reporting import (
    AggregateMetricDelta,
    ComparableComparisonReport,
    ComparisonPredicateResult,
    ComparisonReportModel,
    CompletionSummaryV3,
    DisplayPreview,
    ExactRational,
    IncomparableComparisonReport,
    PairedMetricDelta,
    ReportRecordProjection,
    RunReportModelV3,
    ValidationClaimBoundary,
    comparison_report_model_id,
)
from oamb.contracts.specifications import (
    SourceEvidenceBinding,
    SourceEvidenceKind,
)
from oamb.reporting.public import (
    build_report_record_projection,
    build_run_report_model,
    build_validation_claim_boundary,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
MALICIOUS_LIMITATION = "</script><script>alert(1)</script>&\u2028\u2029"


def build_claim_boundary(
    *,
    rule_count: int = 4,
    comparison_eligible: bool = True,
    billing_complete: bool = False,
    cost_complete: bool = False,
) -> ValidationClaimBoundary:
    return build_validation_claim_boundary(
        status="pass",
        applicable_rule_count=rule_count,
        executed_rule_count=rule_count,
        passed_rule_count=rule_count,
        failed_rule_count=0,
        not_applicable_rule_count=0,
        missing_rule_count=0,
        comparison_eligible=comparison_eligible,
        billing_complete=billing_complete,
        cost_complete=cost_complete,
    )


def build_record_projections(
    *,
    logical_ids: tuple[str, ...],
    plan_ids: tuple[str, ...],
    case_ids: tuple[str, ...],
    attempt_ids: tuple[str, ...],
    evidence_preview_bytes: int = 0,
) -> tuple[ReportRecordProjection, ...]:
    if evidence_preview_bytes < 0:
        raise ValueError("evidence preview byte count cannot be negative")
    evidence_preview = "e" * evidence_preview_bytes

    def display_previews(record_id: str) -> tuple[DisplayPreview, ...]:
        if not evidence_preview:
            return ()
        encoded = evidence_preview.encode()
        return (
            DisplayPreview(
                text=evidence_preview,
                shown_bytes=len(encoded),
                total_bytes=len(encoded),
                sha256=hashlib.sha256(encoded).hexdigest(),
                media_type="application/json",
                truncated=False,
                source_reference=f"source/raw/{record_id}.json",
                limitation=None,
            ),
        )

    projections: list[ReportRecordProjection] = []
    for index, record_id in enumerate(logical_ids, 1):
        projections.append(
            build_report_record_projection(
                record_id=record_id,
                axis="logical-context",
                label=f"Logical context {index}",
                status="included",
                failure_stage=None,
                evaluation_status=None,
                verdict=None,
                capabilities_or_types=(),
                metric_ids=(),
                proof_statuses=(),
                raw_evidence_present=False,
                latency_microseconds=None,
                context_view_tokens=None,
                declared_usage=None,
                detail_items=(
                    ("source_order", str(index)),
                    (
                        "member_label",
                        "fixture-logical-context",
                    ),
                ),
            )
        )
    for index, record_id in enumerate(plan_ids, 1):
        projections.append(
            build_report_record_projection(
                record_id=record_id,
                axis="plan",
                label=f"Ingestion plan {index}",
                status="sealed",
                failure_stage=None,
                evaluation_status=None,
                verdict=None,
                capabilities_or_types=(),
                metric_ids=(),
                proof_statuses=(ProofStatus.UNAVAILABLE,),
                raw_evidence_present=False,
                latency_microseconds=100 + index,
                context_view_tokens=None,
                declared_usage=index,
                detail_items=(
                    (
                        "logical_context_ids",
                        logical_ids[(index - 1) % len(logical_ids)],
                    ),
                    (
                        "resolution_evidence_references",
                        "unavailable",
                    ),
                ),
            )
        )
    for index, record_id in enumerate(case_ids, 1):
        projections.append(
            build_report_record_projection(
                record_id=record_id,
                axis="case",
                label=f"Case occurrence {index}",
                status="completed",
                failure_stage=None,
                evaluation_status="judged",
                verdict="pass" if index % 2 else "fail",
                capabilities_or_types=("fixture",),
                metric_ids=("fixture-exact-v1",),
                proof_statuses=(ProofStatus.MEASURED_COMPLETE,),
                raw_evidence_present=True,
                display_previews=display_previews(record_id),
                latency_microseconds=200 + index,
                context_view_tokens=10 + index,
                declared_usage=20 + index,
                detail_items=(
                    (
                        "ingestion_plan_id",
                        plan_ids[0],
                    ),
                    ("evaluation", "precomputed fixture verdict"),
                ),
            )
        )
    for index, record_id in enumerate(attempt_ids, 1):
        projections.append(
            build_report_record_projection(
                record_id=record_id,
                axis="attempt",
                label=f"Attempt {index}",
                status="succeeded",
                failure_stage=None,
                evaluation_status=None,
                verdict=None,
                capabilities_or_types=(),
                metric_ids=(),
                proof_statuses=(ProofStatus.MEASURED_COMPLETE,),
                raw_evidence_present=True,
                display_previews=display_previews(record_id),
                latency_microseconds=50 + index,
                context_view_tokens=None,
                declared_usage=5 + index,
                detail_items=(
                    ("stage", "answer"),
                    ("parent", case_ids[0]),
                ),
            )
        )
    return tuple(projections)


def _renderer() -> ModuleType:
    try:
        return importlib.import_module("oamb.reporting.offline_renderer")
    except ModuleNotFoundError:
        pytest.fail("oamb.reporting.offline_renderer is not implemented", pytrace=False)


def _source_binding(identity: str, root_hash: str, validation_hash: str) -> SourceEvidenceBinding:
    return SourceEvidenceBinding(
        binding_id=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
        source_kind=SourceEvidenceKind.RUN,
        source_identity=identity,
        source_root_hash=root_hash,
        validation_result_hash=validation_hash,
        source_schema_versions=("capsule_manifest@1",),
    )


def _run_report() -> RunReportModelV3:
    summary = CompletionSummaryV3(
        run_id="run-a",
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
    projections = build_record_projections(
        logical_ids=(SHA_A,),
        plan_ids=(SHA_B,),
        case_ids=(SHA_C,),
        attempt_ids=(SHA_D,),
    )
    return build_run_report_model(
        report_spec_hash=SHA_A,
        source_binding=_source_binding("run-a", SHA_B, SHA_C),
        evidence_validation_profile_hash=SHA_D,
        evidence_validation_result_hash=SHA_C,
        reducer_bindings=(("completion-first-v1", 1, SHA_A),),
        audience="public",
        origin_kind="native",
        capsule_id=SHA_A,
        protocol_id="fixture-protocol-v1",
        workload_id="fixture-workload-v1",
        memory_system_id="fixture-memory-v1",
        claim_boundary=build_claim_boundary(),
        summary=summary,
        metric_summaries=(),
        measurement_lines=(),
        logical_context_ids=(SHA_A,),
        ingestion_occurrence_ids=(SHA_B,),
        case_occurrence_ids=(SHA_C,),
        attempt_ids=(SHA_D,),
        record_projections=projections,
        limitations=(MALICIOUS_LIMITATION,),
    )


def _incomparable_report() -> ComparisonReportModel:
    left = _source_binding("run-left", SHA_A, SHA_C)
    right = _source_binding("run-right", SHA_B, SHA_D)
    comparison = IncomparableComparisonReport(
        comparison_id=SHA_A,
        comparison_spec_hash=SHA_B,
        ordered_source_root_hashes=(SHA_A, SHA_B),
        predicates=(
            ComparisonPredicateResult(
                rule_id="comparison.dataset-revision.v1",
                expected_hash=SHA_A,
                left_hash=SHA_A,
                right_hash=SHA_B,
                passed=False,
            ),
        ),
        limitations=("dataset revisions differ",),
    )
    fields = {
        "report_spec_hash": SHA_C,
        "ordered_source_bindings": (left, right),
        "ordered_evidence_validation_hashes": (SHA_C, SHA_D),
        "left_run_report_hash": SHA_A,
        "right_run_report_hash": SHA_B,
        "claim_boundary": build_claim_boundary(comparison_eligible=False),
        "comparison": comparison,
        "limitations": ("side-by-side evidence only",),
    }
    return ComparisonReportModel.model_validate(
        {"report_id": comparison_report_model_id(**fields), **fields}
    )


def build_comparable_report_fixture() -> ComparisonReportModel:
    left = _source_binding("run-left", SHA_A, SHA_C)
    right = _source_binding("run-right", SHA_B, SHA_D)
    left_value = ExactRational(numerator=3, denominator=4)
    right_value = ExactRational(numerator=1, denominator=2)
    signed = ExactRational(numerator=1, denominator=4)
    paired = PairedMetricDelta(
        case_manifest_entry_id=SHA_A,
        left_case_occurrence_id=SHA_C,
        right_case_occurrence_id=SHA_D,
        metric_id="fixture-exact-v1",
        left=left_value,
        right=right_value,
        signed_delta=signed,
        absolute_delta=signed,
    )
    aggregate = AggregateMetricDelta(
        reducer_id="fixture-aggregate-v1",
        reducer_version=1,
        metric_id="fixture-exact-v1",
        denominator=1,
        aggregate_contract_hash=SHA_D,
        left=left_value,
        right=right_value,
        signed_delta=signed,
        absolute_delta=signed,
    )
    comparison = ComparableComparisonReport(
        comparison_id=SHA_D,
        comparison_spec_hash=SHA_C,
        ordered_source_root_hashes=(SHA_A, SHA_B),
        predicates=(
            ComparisonPredicateResult(
                rule_id="comparison.dataset-revision.v1",
                expected_hash=SHA_C,
                left_hash=SHA_C,
                right_hash=SHA_C,
                passed=True,
            ),
        ),
        paired_metric_deltas=(paired,),
        aggregate_metric_delta=aggregate,
        cost_delta=None,
        winner="left",
        limitations=("fixture compatible comparison",),
    )
    fields = {
        "report_spec_hash": SHA_C,
        "ordered_source_bindings": (left, right),
        "ordered_evidence_validation_hashes": (SHA_C, SHA_D),
        "left_run_report_hash": SHA_A,
        "right_run_report_hash": SHA_B,
        "claim_boundary": build_claim_boundary(),
        "comparison": comparison,
        "limitations": ("compatible fixture",),
    }
    return ComparisonReportModel.model_validate(
        {"report_id": comparison_report_model_id(**fields), **fields}
    )


def build_external_report_fixture() -> RunReportModelV3:
    native = _run_report()
    return build_run_report_model(
        report_spec_hash=native.report_spec_hash,
        source_binding=SourceEvidenceBinding(
            binding_id=SHA_A,
            source_kind=SourceEvidenceKind.EXTERNAL,
            source_identity="synthetic-external-fixture",
            source_root_hash=SHA_B,
            validation_result_hash=SHA_C,
            source_schema_versions=("external_evidence_record@1",),
        ),
        evidence_validation_profile_hash=native.evidence_validation_profile_hash,
        evidence_validation_result_hash=native.evidence_validation_result_hash,
        reducer_bindings=(("external-fixture-v1", 1, SHA_A),),
        audience="public",
        origin_kind="external",
        capsule_id=None,
        protocol_id=native.protocol_id,
        workload_id=native.workload_id,
        memory_system_id=native.memory_system_id,
        claim_boundary=native.claim_boundary,
        summary=native.summary,
        metric_summaries=native.metric_summaries,
        measurement_lines=native.measurement_lines,
        logical_context_ids=native.logical_context_ids,
        ingestion_occurrence_ids=native.ingestion_occurrence_ids,
        case_occurrence_ids=native.case_occurrence_ids,
        attempt_ids=native.attempt_ids,
        record_projections=native.record_projections,
        limitations=("synthetic renderer discrimination only",),
    )


def _inline_asset(html: str, tag: str, element_id: str) -> str:
    match = re.search(
        rf'<{tag}[^>]*\bid="{element_id}"[^>]*>(.*?)</{tag}>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None
    return match.group(1)


def test_renderer_is_deterministic_and_hashes_the_exact_inline_assets() -> None:
    renderer = _renderer()

    first = renderer.render_offline_report(_run_report())
    second = renderer.render_offline_report(_run_report())

    assert first == second
    model = _run_report()
    assert renderer._canonical_display_model_bytes(model) == canonical_json_bytes(
        model.model_dump(mode="python", exclude_none=True)
    )
    html = first.decode("utf-8")
    css = _inline_asset(html, "style", "oamb-report-style")
    script = _inline_asset(html, "script", "oamb-report-script")
    css_hash = base64.b64encode(hashlib.sha256(css.encode("utf-8")).digest()).decode("ascii")
    script_hash = base64.b64encode(hashlib.sha256(script.encode("utf-8")).digest()).decode("ascii")
    assert f"style-src 'sha256-{css_hash}'" in html
    assert f"script-src 'sha256-{script_hash}'" in html
    for directive in (
        "default-src 'none'",
        "connect-src 'none'",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "font-src 'none'",
    ):
        assert directive in html


def test_renderer_safely_embeds_canonical_json_and_has_no_network_or_code_sinks() -> None:
    html = _renderer().render_offline_report(_run_report()).decode("utf-8")
    script = _inline_asset(html, "script", "oamb-report-script")

    assert MALICIOUS_LIMITATION not in html
    assert "\\u003c/script\\u003e\\u003cscript\\u003ealert(1)\\u003c/script\\u003e" in html
    assert "\\u0026" in html
    assert "\\u2028" in html
    assert "\\u2029" in html
    assert not re.search(r"https?://|(?:src|href)=[\"']//", html, flags=re.IGNORECASE)
    assert not re.search(r"\son[a-z]+\s*=", html, flags=re.IGNORECASE)
    assert "innerHTML" not in script
    assert "eval(" not in script
    assert "new Function" not in script
    assert ".textContent" in script


def test_incomparable_html_source_contains_no_winner_or_delta_claims() -> None:
    html = _renderer().render_offline_report(_incomparable_report()).decode("utf-8")

    assert "winner" not in html.lower()
    assert "delta" not in html.lower()
    assert "dataset revisions differ" in html


def test_native_external_and_comparison_report_snapshots_are_distinct_and_stable() -> None:
    renderer = _renderer()
    models = (
        _run_report(),
        build_external_report_fixture(),
        build_comparable_report_fixture(),
        _incomparable_report(),
    )
    first = tuple(renderer.render_offline_report(model) for model in models)
    second = tuple(renderer.render_offline_report(model) for model in models)

    assert first == second
    assert len({hashlib.sha256(item).hexdigest() for item in first}) == 4
    assert b'"origin_kind":"external"' in first[1]
    assert b'"source_kind":"external"' in first[1]
    assert b'"winner":"left"' in first[2]
    assert b'"signed_delta"' in first[2]
    assert b'"winner"' not in first[3]
    assert b'"signed_delta"' not in first[3]
