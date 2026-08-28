from __future__ import annotations

from oamb.contracts.reporting import (
    ComparableComparisonReport,
    IncomparableComparisonReport,
)
from oamb.contracts.schema import CONTRACT_REGISTRY, parse_contract

T8_VERSIONED_CONTRACT_KEYS = {
    ("acceptance_report_spec", 1),
    ("aggregate_metric_delta", 1),
    ("ai_quality_review_record", 1),
    ("ai_review_batch", 1),
    ("ai_review_batch_result", 1),
    ("ai_review_case_projection", 1),
    ("ai_review_case_result", 1),
    ("ai_review_finding", 1),
    ("ai_review_integrity_projection", 1),
    ("ai_review_integrity_result", 1),
    ("ai_review_plan", 1),
    ("case_record", 3),
    ("comparison_control_binding", 1),
    ("comparison_control_snapshot", 1),
    ("comparison_cost_control", 1),
    ("comparison_cost_delta", 1),
    ("comparison_pair_binding", 1),
    ("comparison_predicate_result", 1),
    ("comparison_report", 1),
    ("comparison_report_model", 1),
    ("comparison_spec", 1),
    ("comparison_winner_reducer", 1),
    ("completion_summary", 1),
    ("derivation_spec", 2),
    ("diagnostic_run_report_model", 1),
    ("display_preview", 1),
    ("evaluation_phase_gate", 1),
    ("evaluation_review_bundle", 1),
    ("exact_rational", 1),
    ("human_quality_review_record", 1),
    ("human_review_decision", 1),
    ("human_review_key_binding", 1),
    ("ingestion_plan_record", 2),
    ("mab65_report_reduction", 1),
    ("mab_capability_metric_summary", 1),
    ("mab_component_metric_summary", 1),
    ("mab_plan_evidence_binding", 1),
    ("mab_plan_metric_summary", 1),
    ("measurement_summary_line", 1),
    ("metric_summary", 1),
    ("paired_metric_delta", 1),
    ("phase_acceptance_report", 1),
    ("phase_review_occurrence_record", 1),
    ("reducer_binding", 1),
    ("release_report_model", 1),
    ("report_artifact_manifest", 2),
    ("report_identity_spec_binding", 1),
    ("report_record_projection", 1),
    ("report_spec", 1),
    ("run_report_model", 3),
    ("signature_verification_record", 1),
    ("validation_claim_boundary", 1),
}


def test_t8_versioned_contract_inventory_is_complete_and_unique() -> None:
    assert T8_VERSIONED_CONTRACT_KEYS <= set(CONTRACT_REGISTRY)
    registered = [key for key in CONTRACT_REGISTRY if key in T8_VERSIONED_CONTRACT_KEYS]
    assert len(registered) == len(T8_VERSIONED_CONTRACT_KEYS)


def test_comparison_report_parser_selects_both_public_branches() -> None:
    common = {
        "schema_name": "comparison_report",
        "schema_version": 1,
        "comparison_id": "a" * 64,
        "comparison_spec_hash": "b" * 64,
        "ordered_source_root_hashes": ["c" * 64, "d" * 64],
    }
    comparable = {
        **common,
        "predicates": [
            {
                "schema_name": "comparison_predicate_result",
                "schema_version": 1,
                "rule_id": "same-protocol",
                "expected_hash": "e" * 64,
                "left_hash": "e" * 64,
                "right_hash": "e" * 64,
                "passed": True,
            }
        ],
        "comparable": True,
        "paired_metric_deltas": [],
        "aggregate_metric_delta": None,
        "cost_delta": None,
        "winner": None,
        "limitations": [],
    }
    incomparable = {
        **common,
        "predicates": [
            {
                "schema_name": "comparison_predicate_result",
                "schema_version": 1,
                "rule_id": "same-protocol",
                "expected_hash": "e" * 64,
                "left_hash": "e" * 64,
                "right_hash": "f" * 64,
                "passed": False,
            }
        ],
        "comparable": False,
        "limitations": ["protocol controls differ"],
    }

    assert isinstance(parse_contract(comparable), ComparableComparisonReport)
    assert isinstance(parse_contract(incomparable), IncomparableComparisonReport)
