from __future__ import annotations

import base64
import hashlib
import importlib
import json
import re
from fractions import Fraction
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
    MabPlanEvidenceBinding,
    MeasurementSummaryLine,
    PairedMetricDelta,
    PhaseAcceptanceReport,
    ReportRecordProjection,
    RunReportModelV3,
    ValidationClaimBoundary,
    comparison_report_model_id,
    phase_acceptance_report_id,
    run_report_model_v3_id,
)
from oamb.contracts.specifications import (
    SourceEvidenceBinding,
    SourceEvidenceKind,
)
from oamb.reporting.public import (
    build_evaluation_report_model,
    build_mab65_report_reduction,
    build_report_record_projection,
    build_run_report_model,
    build_validation_claim_boundary,
)
from oamb.workloads.mab65_reduction import (
    MAB65_INDEX_ID,
    Mab65Reduction,
    MabCapabilityContribution,
    MabComponentScore,
    MabPlanScore,
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
    mab_plan_bindings: tuple[MabPlanEvidenceBinding, ...] = (),
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

    mab_plan_by_id = {item.ingestion_occurrence_id: item for item in mab_plan_bindings}
    mab_case_by_id = {
        case_id: item for item in mab_plan_bindings for case_id in item.ordered_case_occurrence_ids
    }
    mab_context_by_id = {
        context_id: item
        for item in mab_plan_bindings
        for context_id in item.ordered_logical_context_ids
    }
    projections: list[ReportRecordProjection] = []
    for index, record_id in enumerate(logical_ids, 1):
        binding = mab_context_by_id.get(record_id)
        projections.append(
            build_report_record_projection(
                record_id=record_id,
                axis="logical-context",
                label=f"Logical context {index}",
                status="included",
                failure_stage=None,
                evaluation_status=None,
                verdict=None,
                capabilities_or_types=(
                    (_mab_binding_component(binding),) if binding is not None else ()
                ),
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
                        (
                            binding.member_labels[
                                binding.ordered_logical_context_ids.index(record_id)
                            ]
                            if binding is not None
                            else "fixture-logical-context"
                        ),
                    ),
                ),
            )
        )
    for index, record_id in enumerate(plan_ids, 1):
        binding = mab_plan_by_id.get(record_id)
        projections.append(
            build_report_record_projection(
                record_id=record_id,
                axis="plan",
                label=f"Ingestion plan {index}",
                status="sealed",
                failure_stage=None,
                evaluation_status=None,
                verdict=None,
                capabilities_or_types=(
                    (_mab_binding_component(binding),) if binding is not None else ()
                ),
                metric_ids=((binding.metric_id,) if binding is not None else ()),
                proof_statuses=(ProofStatus.UNAVAILABLE,),
                raw_evidence_present=bool(
                    binding is not None and binding.resolution_evidence_references
                ),
                latency_microseconds=100 + index,
                context_view_tokens=None,
                declared_usage=index,
                detail_items=(
                    (
                        "logical_context_ids",
                        ",".join(binding.ordered_logical_context_ids)
                        if binding is not None
                        else logical_ids[(index - 1) % len(logical_ids)],
                    ),
                    (
                        "resolution_evidence_references",
                        ",".join(binding.resolution_evidence_references)
                        if binding is not None
                        else "unavailable",
                    ),
                ),
            )
        )
    for index, record_id in enumerate(case_ids, 1):
        binding = mab_case_by_id.get(record_id)
        metric_id = binding.metric_id if binding is not None else "fixture-exact-v1"
        projections.append(
            build_report_record_projection(
                record_id=record_id,
                axis="case",
                label=f"Case occurrence {index}",
                status="completed",
                failure_stage=None,
                evaluation_status="judged",
                verdict="pass" if index % 2 else "fail",
                capabilities_or_types=(
                    (_mab_binding_component(binding),) if binding is not None else ("fixture",)
                ),
                metric_ids=(metric_id,),
                proof_statuses=(ProofStatus.MEASURED_COMPLETE,),
                raw_evidence_present=True,
                display_previews=display_previews(record_id),
                latency_microseconds=200 + index,
                context_view_tokens=10 + index,
                declared_usage=20 + index,
                detail_items=(
                    (
                        "ingestion_plan_id",
                        binding.ingestion_plan_id if binding is not None else plan_ids[0],
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


def _mab_binding_component(binding: MabPlanEvidenceBinding) -> str:
    label = " ".join(binding.member_labels).lower()
    if binding.metric_id == "mab-redial-recall-at-5-v1":
        return "recsys"
    if "factconsolidation" in label:
        return "cr_sf"
    if label.startswith("ar_"):
        return "ar"
    if "icl_" in label:
        return "icl"
    return "lru"


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


def _acceptance_report() -> PhaseAcceptanceReport:
    fields = {
        "acceptance_report_spec_hash": SHA_A,
        "phase_id": "fixture-phase",
        "evaluation_report_hash": SHA_B,
        "evaluation_export_validation_hash": SHA_C,
        "review_bundle_hash": SHA_D,
        "phase_gate_hash": SHA_A,
        "ai_review_record_hash": SHA_B,
        "human_review_record_hash": SHA_C,
        "gate_review_bundle_hash": SHA_D,
        "review_current": True,
        "passed_by_ai": True,
        "passed_by_human": True,
        "finding_codes": (),
        "evidence_references": ("review:fixture",),
        "limitations": ("fixture acceptance",),
    }
    return PhaseAcceptanceReport.model_validate(
        {"report_id": phase_acceptance_report_id(**fields), **fields}
    )


def build_mab65_report_fixture(*, available: bool = True) -> RunReportModelV3:
    case_count_vectors = {
        "ar": (3, 3, 3, 3, 3),
        "icl": (2, 2, 2, 2, 2),
        "recsys": (10,),
        "lru": (1, 1, 1, 1, 1, 2, 2, 2, 2, 2),
        "cr_sf": (3, 4, 4, 4),
    }
    plan_counts = {component: len(values) for component, values in case_count_vectors.items()}
    metric_ids = {
        "ar": "mab-substring-em-v1",
        "icl": "mab-exact-v1",
        "recsys": "mab-redial-recall-at-5-v1",
        "lru": "mab-exact-v1",
        "cr_sf": "mab-substring-em-v1",
    }
    plans: list[MabPlanScore] = []
    bindings: list[MabPlanEvidenceBinding] = []
    logical_ordinal = 1
    case_ordinal = 1
    plan_ordinal = 1
    for component in ("ar", "icl", "recsys", "lru", "cr_sf"):
        for case_count in case_count_vectors[component]:
            plan_manifest_id = f"mab65-{component}-{plan_ordinal}"
            plans.append(
                MabPlanScore(
                    plan_manifest_entry_id=plan_manifest_id,
                    component=component,
                    case_count=case_count,
                    score=Fraction(1, 2),
                )
            )
            member_count = 2 if component == "cr_sf" else 1
            logical_ids = tuple(
                f"{ordinal:064x}"
                for ordinal in range(logical_ordinal, logical_ordinal + member_count)
            )
            logical_ordinal += member_count
            case_ids = tuple(
                f"{ordinal + 1_000:064x}"
                for ordinal in range(case_ordinal, case_ordinal + case_count)
            )
            case_ordinal += case_count
            bindings.append(
                MabPlanEvidenceBinding(
                    plan_manifest_entry_id=plan_manifest_id,
                    ingestion_plan_id=f"{plan_ordinal + 2_000:064x}",
                    ingestion_occurrence_id=f"{plan_ordinal + 3_000:064x}",
                    ordered_logical_context_ids=logical_ids,
                    member_labels=(
                        (
                            f"factconsolidation_sh_{plan_ordinal}",
                            f"factconsolidation_mh_{plan_ordinal}",
                        )
                        if component == "cr_sf"
                        else (f"{component}_{plan_ordinal}",)
                    ),
                    ordered_case_occurrence_ids=case_ids,
                    metric_id=metric_ids[component],
                    resolution_evidence_references=(
                        ("source/redial-resolution.json.gz",) if component == "recsys" else ()
                    ),
                )
            )
            plan_ordinal += 1
    components = tuple(
        MabComponentScore(
            component=component,
            plan_count=plan_counts[component],
            case_count=sum(plan.case_count for plan in plans if plan.component == component),
            score=Fraction(1, 2),
        )
        for component in ("ar", "icl", "recsys", "lru", "cr_sf")
    )
    capabilities = tuple(
        MabCapabilityContribution(
            capability=capability,
            score=Fraction(1, 2),
            weight=Fraction(1, 4),
            weighted_contribution=Fraction(1, 8),
        )
        for capability in ("ar", "ttl", "lru", "cr_sf")
    )
    reduction = Mab65Reduction(
        reducer_id=MAB65_INDEX_ID,
        available=True,
        unavailable_reason=None,
        plans=tuple(plans),
        components=components,
        ttl_score=Fraction(1, 2),
        capabilities=capabilities,
        index_value=Fraction(50),
    )
    mab65 = (
        build_mab65_report_reduction(
            reduction,
            plan_evidence_bindings=tuple(bindings),
        )
        if available
        else build_mab65_report_reduction(
            Mab65Reduction(
                reducer_id=MAB65_INDEX_ID,
                available=False,
                unavailable_reason="case_metric_inventory",
                plans=(),
                components=(),
                ttl_score=None,
                capabilities=(),
                index_value=None,
            )
        )
    )
    logical_ids = tuple(
        context_id for binding in bindings for context_id in binding.ordered_logical_context_ids
    )
    plan_ids = tuple(binding.ingestion_occurrence_id for binding in bindings)
    case_ids = tuple(
        case_id for binding in bindings for case_id in binding.ordered_case_occurrence_ids
    )
    cost_record_ids = tuple(f"{ordinal + 3_000:064x}" for ordinal in range(1, 26))
    return build_run_report_model(
        report_spec_hash=SHA_A,
        source_binding=_source_binding("mab65-run", SHA_B, SHA_C),
        evidence_validation_profile_hash=SHA_D,
        evidence_validation_result_hash=SHA_C,
        reducer_bindings=((MAB65_INDEX_ID, 1, SHA_A),),
        audience="public",
        origin_kind="native",
        capsule_id=SHA_A,
        protocol_id="mab65-v1",
        workload_id="mab65-v1",
        memory_system_id="fixture-memory-v1",
        claim_boundary=build_claim_boundary(),
        summary=CompletionSummaryV3(
            run_id="mab65-run",
            intended_logical_contexts=29,
            intended_ingestion_plans=25,
            ready_ingestion_plans=25,
            intended_cases=65,
            terminal_cases=65,
            completed_cases=65,
            errored_cases=0,
            unsupported_cases=0,
            cancelled_cases=0,
            budget_exceeded_cases=0,
            parsed_cases=65,
            evaluated_cases=65,
            judged_cases=0,
            unjudged_cases=0,
            metric_eligible_cases=65,
        ),
        metric_summaries=(),
        mab65_reduction=mab65,
        measurement_lines=(
            MeasurementSummaryLine(
                dimension_id="supplier_cost",
                stage="memory_ingest",
                owner_kind="ingestion_plan",
                indexing_view="final_contribution",
                value=None,
                unit="currency",
                proof_status=ProofStatus.UNAVAILABLE,
                basis="actual_supplier_charge",
                currency=None,
                source_record_ids=cost_record_ids,
                reason="fixture supplier charge is unavailable",
            ),
        ),
        logical_context_ids=logical_ids,
        ingestion_occurrence_ids=plan_ids,
        case_occurrence_ids=case_ids,
        attempt_ids=(SHA_D,),
        record_projections=build_record_projections(
            logical_ids=logical_ids,
            plan_ids=plan_ids,
            case_ids=case_ids,
            attempt_ids=(SHA_D,),
            mab_plan_bindings=tuple(bindings),
        ),
        limitations=(
            "plan-owned indexing and cost appear once per physical plan",
            *(
                ("MAB-65 secondary index unavailable: case_metric_inventory",)
                if not available
                else ()
            ),
        ),
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


def test_renderer_freezes_information_order_theme_and_interaction_contract() -> None:
    html = _renderer().render_offline_report(_run_report()).decode("utf-8")
    css = _inline_asset(html, "style", "oamb-report-style")
    script = _inline_asset(html, "script", "oamb-report-script")

    assert '<meta name="color-scheme" content="light dark">' in html
    section_ids = (
        "identity",
        "validation",
        "completion",
        "quality",
        "usage-cost",
        "limitations",
    )
    positions = tuple(html.index(f'id="{section_id}"') for section_id in section_ids)
    assert positions == tuple(sorted(positions))
    assert "@media (prefers-color-scheme: dark)" in css
    assert "@media print" in css
    assert "@media (max-width: 320px)" in css
    assert ":focus-visible" in css
    for element_id in (
        "oamb-filter",
        "oamb-axis-filter",
        "oamb-sort",
        "oamb-copy",
        "oamb-copy-status",
        "oamb-records",
    ):
        assert f'id="{element_id}"' in html
    for behavior_marker in (
        "stableIndex",
        "location.hash",
        'event.key === "j"',
        'event.key === "k"',
        "navigator.clipboard",
        "Select and copy manually",
        '"logical-context"',
        '"plan"',
        '"case"',
        '"attempt"',
        "dataset.axis",
    ):
        assert behavior_marker in script
    for forbidden_derivation in (
        "score_numerator",
        "score_denominator",
        "metric_numerator",
        "metric_denominator",
        "passed_by_ai",
        "passed_by_human",
        "signed_delta",
        "absolute_delta",
    ):
        assert forbidden_derivation not in script


def test_incomparable_html_source_contains_no_winner_or_delta_claims() -> None:
    html = _renderer().render_offline_report(_incomparable_report()).decode("utf-8")

    assert "winner" not in html.lower()
    assert "delta" not in html.lower()
    assert "dataset revisions differ" in html


def test_acceptance_report_embeds_only_precomputed_gate_values() -> None:
    html = _renderer().render_offline_report(_acceptance_report()).decode("utf-8")
    script = _inline_asset(html, "script", "oamb-report-script")

    assert '"passed_by_ai":true' in html
    assert '"passed_by_human":true' in html
    assert "passed_by_ai" not in script
    assert "passed_by_human" not in script
    assert "fixture acceptance" in html


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


def test_evaluation_report_embeds_exact_four_run_models_and_eligible_comparison() -> None:
    base = _run_report()
    runs_list = []
    for character in "abcd":
        payload = base.model_dump(mode="python")
        payload["summary"]["run_id"] = f"run-{character}"
        fields = {
            key: value
            for key, value in payload.items()
            if key not in {"schema_name", "schema_version", "report_id"}
        }
        payload["report_id"] = run_report_model_v3_id(**fields)
        runs_list.append(RunReportModelV3.model_validate(payload))
    runs = (runs_list[0], runs_list[1], runs_list[2], runs_list[3])
    comparison = _incomparable_report()
    model = build_evaluation_report_model(
        phase_id="phase_1_smoke_acceptance",
        report_spec_hash=SHA_A,
        ordered_run_models=runs,
        eligible_comparison_models=(comparison,),
        unique_case_count=2,
        limitations=("fixture only",),
    )

    rendered = _renderer().render_offline_report(model)

    assert model.system_result_count == 4
    for run in runs:
        assert run.report_id.encode() in rendered
    assert comparison.report_id.encode() in rendered


def test_mab65_html_golden_preserves_29_25_65_topology_and_component_waterfall() -> None:
    html = _renderer().render_offline_report(build_mab65_report_fixture()).decode("utf-8")
    payload_text = _inline_asset(html, "script", "oamb-report-data")
    payload = json.loads(payload_text)
    mab65 = payload["mab65_reduction"]

    assert len(payload["logical_context_ids"]) == 29
    assert len(payload["ingestion_occurrence_ids"]) == 25
    assert len(payload["case_occurrence_ids"]) == 65
    plan_cost_ids = tuple(
        record_id
        for line in payload["measurement_lines"]
        if line["dimension_id"] == "supplier_cost"
        and line["owner_kind"] == "ingestion_plan"
        and line["indexing_view"] == "final_contribution"
        for record_id in line["source_record_ids"]
    )
    assert len(plan_cost_ids) == len(set(plan_cost_ids)) == 25
    assert len(mab65["plans"]) == 25
    assert (
        sum(len(item["evidence_binding"]["ordered_logical_context_ids"]) for item in mab65["plans"])
        == 29
    )
    assert (
        sum(len(item["evidence_binding"]["ordered_case_occurrence_ids"]) for item in mab65["plans"])
        == 65
    )
    assert [item["component"] for item in mab65["components"]] == [
        "ar",
        "icl",
        "recsys",
        "lru",
        "cr_sf",
    ]
    assert [item["capability"] for item in mab65["capabilities"]] == [
        "ar",
        "ttl",
        "lru",
        "cr_sf",
    ]
    grouped = [item for item in mab65["plans"] if item["component"] == "cr_sf"]
    assert len(grouped) == 4
    assert all(
        len(item["evidence_binding"]["ordered_logical_context_ids"]) == 2
        and "factconsolidation_sh_" in item["evidence_binding"]["member_labels"][0]
        and "factconsolidation_mh_" in item["evidence_binding"]["member_labels"][1]
        for item in grouped
    )
    redial = next(item for item in mab65["plans"] if item["component"] == "recsys")
    assert redial["case_count"] == 10
    assert redial["evidence_binding"]["metric_id"] == "mab-redial-recall-at-5-v1"
    assert redial["evidence_binding"]["resolution_evidence_references"]
    assert "MAB-65 Capability-Balanced Index" in html
    assert "ReDial Recall@5" in html
    assert "accuracy" not in html.lower()
    assert html.index("<h4>Primary components</h4>") < html.index("<h4>Capabilities</h4>")
    assert html.index("<h4>Capabilities</h4>") < html.index('id="oamb-mab65-index"')


def test_mab65_run_report_rejects_reordered_topology_inventory() -> None:
    model = build_mab65_report_fixture()
    fields = model.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "report_id"},
    )
    fields["logical_context_ids"] = tuple(reversed(model.logical_context_ids))
    fields["record_projections"] = tuple(
        projection
        for axis in ("logical-context", "plan", "case", "attempt")
        for projection in (
            tuple(reversed(tuple(item for item in model.record_projections if item.axis == axis)))
            if axis == "logical-context"
            else tuple(item for item in model.record_projections if item.axis == axis)
        )
    )

    with pytest.raises(ValueError, match="MAB-65 report topology"):
        RunReportModelV3.model_validate({"report_id": run_report_model_v3_id(**fields), **fields})


def test_mab65_run_report_rejects_case_fanout_cost_inventory() -> None:
    model = build_mab65_report_fixture()
    fields = model.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "report_id"},
    )
    fields["measurement_lines"][0]["source_record_ids"] = tuple(
        f"{ordinal + 4_000:064x}" for ordinal in range(1, 66)
    )

    with pytest.raises(ValueError, match="one unique record per physical plan"):
        RunReportModelV3.model_validate({"report_id": run_report_model_v3_id(**fields), **fields})
