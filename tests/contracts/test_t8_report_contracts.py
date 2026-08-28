from __future__ import annotations

import importlib
from fractions import Fraction

import pytest
from pydantic import ValidationError

from oamb.contracts.reporting import (
    ComparisonPredicateResult,
    DisplayPreview,
    EvaluationReviewBundle,
    ExactRational,
    IncomparableComparisonReport,
    PairedMetricDelta,
)
from oamb.contracts.specifications import AIReviewPlan, SourceEvidenceBinding, SourceEvidenceKind

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def test_display_preview_binds_exact_utf8_bytes_and_truncation() -> None:
    preview = DisplayPreview(
        text="é",
        shown_bytes=2,
        total_bytes=3,
        sha256=SHA_A,
        media_type="text/plain",
        truncated=True,
        source_reference="source/raw/aa.json.gz",
        limitation="one byte is omitted",
    )

    assert preview.shown_bytes == len(preview.text.encode("utf-8"))
    with pytest.raises(ValidationError, match="shown bytes"):
        DisplayPreview(
            text="é",
            shown_bytes=1,
            total_bytes=3,
            sha256=SHA_A,
            media_type="text/plain",
            truncated=True,
            source_reference="source/raw/aa.json.gz",
            limitation="one byte is omitted",
        )
    with pytest.raises(ValidationError, match="truncation"):
        DisplayPreview(
            text="complete",
            shown_bytes=8,
            total_bytes=8,
            sha256=SHA_A,
            media_type="text/plain",
            truncated=True,
            source_reference="source/raw/aa.json.gz",
            limitation="not actually truncated",
        )
    for unsafe_reference in (
        "source/raw/x/..",
        r"C:\Users\operator\secret.json",
        "C:/Users/operator/secret.json",
    ):
        with pytest.raises(ValidationError, match="capsule-relative"):
            DisplayPreview(
                text="safe",
                shown_bytes=4,
                total_bytes=4,
                sha256=SHA_A,
                media_type="application/json",
                truncated=False,
                source_reference=unsafe_reference,
                limitation=None,
            )


def test_exact_rational_is_reduced_and_delta_arithmetic_is_closed() -> None:
    left = ExactRational(numerator=1, denominator=2)
    right = ExactRational(numerator=3, denominator=4)
    delta = PairedMetricDelta(
        case_manifest_entry_id=SHA_A,
        left_case_occurrence_id=SHA_B,
        right_case_occurrence_id=SHA_C,
        metric_id="metric-v1",
        left=left,
        right=right,
        signed_delta=ExactRational(numerator=-1, denominator=4),
        absolute_delta=ExactRational(numerator=1, denominator=4),
    )

    assert Fraction(delta.right.numerator, delta.right.denominator) == Fraction(3, 4)
    with pytest.raises(ValidationError, match="lowest terms"):
        ExactRational(numerator=2, denominator=4)
    with pytest.raises(ValidationError, match="delta arithmetic"):
        PairedMetricDelta(
            case_manifest_entry_id=SHA_A,
            left_case_occurrence_id=SHA_B,
            right_case_occurrence_id=SHA_C,
            metric_id="metric-v1",
            left=left,
            right=right,
            signed_delta=ExactRational(numerator=1, denominator=4),
            absolute_delta=ExactRational(numerator=1, denominator=4),
        )


def test_incomparable_comparison_json_has_no_delta_or_winner_fields() -> None:
    failed = ComparisonPredicateResult(
        rule_id="comparison.dataset-revision.v1",
        expected_hash=SHA_A,
        left_hash=SHA_A,
        right_hash=SHA_B,
        passed=False,
    )
    report = IncomparableComparisonReport(
        comparison_id=SHA_D,
        comparison_spec_hash=SHA_A,
        ordered_source_root_hashes=(SHA_B, SHA_C),
        predicates=(failed,),
        limitations=("dataset revisions differ",),
    )

    payload = report.model_dump(mode="json")
    assert "paired_metric_deltas" not in payload
    assert "winner" not in payload


def test_review_bundle_v1_is_phase_generic_with_internal_closure_only() -> None:
    case_ids = tuple(f"{index:064x}" for index in range(1, 4))
    unique_case_ids = tuple(f"{index:064x}" for index in range(101, 103))
    review = importlib.import_module("oamb.reporting.review")

    bundle = review.build_evaluation_review_bundle(
        phase_id="fixture_phase",
        ordered_capsule_hashes=(SHA_A, SHA_B),
        ordered_validation_hashes=(SHA_D, SHA_C),
        ordered_case_occurrence_ids=case_ids,
        unique_case_manifest_entry_ids=unique_case_ids,
        ordinary_derivation_hashes=(SHA_A,),
        report_model_hash=SHA_B,
        report_html_hash=SHA_C,
        export_validation_hash=SHA_D,
        limitations=("offline fixture",),
    )

    assert len(bundle.ordered_case_occurrence_ids) == 3
    with pytest.raises(ValidationError, match="capsule.*validation"):
        EvaluationReviewBundle(
            **{
                **bundle.model_dump(mode="python"),
                "ordered_validation_hashes": (SHA_D,),
            }
        )


def test_review_bundle_identity_binds_every_ordered_review_input() -> None:
    review = importlib.import_module("oamb.reporting.review")
    if not hasattr(review, "build_evaluation_review_bundle"):
        pytest.fail("review bundle canonical builder is not implemented", pytrace=False)

    bundle = review.build_evaluation_review_bundle(
        phase_id="fixture_phase",
        ordered_capsule_hashes=(SHA_A, SHA_B),
        ordered_validation_hashes=(SHA_C, SHA_D),
        ordered_case_occurrence_ids=(SHA_A, SHA_B),
        unique_case_manifest_entry_ids=(SHA_C,),
        ordinary_derivation_hashes=(SHA_D,),
        report_model_hash=SHA_A,
        report_html_hash=SHA_B,
        export_validation_hash=SHA_C,
        limitations=("offline fixture",),
    )

    assert bundle.bundle_id != bundle.ordered_review_input_hash
    with pytest.raises(ValidationError, match="ordered review input"):
        EvaluationReviewBundle.model_validate(
            {
                **bundle.model_dump(mode="python"),
                "report_html_hash": SHA_D,
            }
        )


def test_ai_review_plan_is_owned_by_the_control_plane_contract_module() -> None:
    assert AIReviewPlan.__module__ == "oamb.contracts.specifications"


def test_report_identity_binding_is_required_exactly_for_rendering_derivations() -> None:
    specifications = importlib.import_module("oamb.contracts.specifications")
    try:
        roots = importlib.import_module("oamb.reporting.roots")
        renderer = importlib.import_module("oamb.reporting.offline_renderer")
        acceptance_contracts = importlib.import_module("oamb.reporting.acceptance_contracts")
    except ModuleNotFoundError:
        pytest.fail("oamb.reporting.roots is not implemented", pytrace=False)
    for name in (
        "ReportSpec",
        "AcceptanceReportSpec",
        "ReportIdentitySpecBinding",
        "DerivationSpecV2",
    ):
        if not hasattr(specifications, name):
            pytest.fail(f"oamb.contracts.specifications.{name} is not implemented", pytrace=False)

    report_spec = roots.build_report_spec(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "completion", "quality", "cost"),
        renderer_hash=renderer.offline_renderer_hash(),
        asset_hashes=renderer.offline_asset_hashes(),
        browser_contract_hash=acceptance_contracts.browser_acceptance_contract_hash(),
        performance_contract_hash=acceptance_contracts.performance_acceptance_contract_hash(),
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )
    binding = roots.build_report_identity_spec_binding(report_spec)
    source_binding = SourceEvidenceBinding(
        binding_id=SHA_A,
        source_kind=SourceEvidenceKind.RUN,
        source_identity="run-a",
        source_root_hash=SHA_B,
        validation_result_hash=SHA_C,
        source_schema_versions=("capsule_manifest@1",),
    )
    derivation = roots.build_derivation_spec_v2(
        derivation_kind="run_report",
        ordered_source_bindings=(source_binding,),
        evidence_validation_result_hash=SHA_C,
        transform_spec_hash=SHA_D,
        report_identity_spec_binding=binding,
        reducer_and_renderer_input_hashes=(SHA_A, SHA_B),
    )

    assert derivation.report_identity_spec_binding == binding
    with pytest.raises(ValueError, match="exactly one report identity"):
        roots.build_derivation_spec_v2(
            derivation_kind="run_report",
            ordered_source_bindings=(source_binding,),
            evidence_validation_result_hash=SHA_C,
            transform_spec_hash=SHA_D,
            report_identity_spec_binding=None,
            reducer_and_renderer_input_hashes=(SHA_A,),
        )
    with pytest.raises(ValueError, match="non-rendering"):
        roots.build_derivation_spec_v2(
            derivation_kind="run_summary",
            ordered_source_bindings=(source_binding,),
            evidence_validation_result_hash=SHA_C,
            transform_spec_hash=SHA_D,
            report_identity_spec_binding=binding,
            reducer_and_renderer_input_hashes=(SHA_A,),
        )


def test_acceptance_report_spec_uses_an_independent_export_selector_without_a_hash_cycle() -> None:
    roots = importlib.import_module("oamb.reporting.roots")
    renderer = importlib.import_module("oamb.reporting.offline_renderer")
    acceptance_contracts = importlib.import_module("oamb.reporting.acceptance_contracts")
    acceptance = roots.build_acceptance_report_spec(
        audience="public",
        evaluation_report_hash=SHA_A,
        evaluation_export_validation_hash=SHA_B,
        review_bundle_hash=SHA_C,
        ai_review_record_hash=SHA_D,
        human_review_record_hash=SHA_A,
        phase_gate_hash=SHA_B,
        renderer_hash=renderer.offline_renderer_hash(),
        asset_hashes=renderer.offline_asset_hashes(),
        browser_contract_hash=acceptance_contracts.browser_acceptance_contract_hash(),
        performance_contract_hash=acceptance_contracts.performance_acceptance_contract_hash(),
        export_profile_selector_id="public-phase-acceptance-v1",
        export_profile_selector_version=1,
    )

    payload = acceptance.model_dump(mode="json")
    assert payload["export_profile_selector_id"] == "public-phase-acceptance-v1"
    assert "export_profile_hash" not in payload
    with pytest.raises(ValidationError, match="acceptance report spec identity"):
        type(acceptance).model_validate(
            {**acceptance.model_dump(mode="python"), "acceptance_report_spec_id": SHA_D}
        )
