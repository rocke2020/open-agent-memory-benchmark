from __future__ import annotations

import importlib
from types import ModuleType

import pytest
from pydantic import ValidationError

EXPECTED_VERSIONED_CONTRACT_NAMES = {
    "aggregate_metric_delta",
    "attempt_intent_record",
    "attempt_receipt_record",
    "attempt_record",
    "budget_owner_allocation",
    "budget_reservation_record",
    "budget_spec",
    "capsule_manifest",
    "capsule_manifest_entry",
    "case_manifest",
    "case_manifest_entry",
    "case_record",
    "checkpoint_manifest",
    "close_error_record",
    "comparison_control_binding",
    "comparison_control_provenance_binding",
    "comparison_control_source_reference",
    "comparison_control_snapshot",
    "comparison_cost_control",
    "comparison_cost_delta",
    "comparison_pair_binding",
    "comparison_predicate_result",
    "comparison_report",
    "comparison_report_model",
    "comparison_spec",
    "comparison_winner_reducer",
    "completion_summary",
    "controlled_embedding_comparison_projection",
    "cost_measurement_spec",
    "cost_record",
    "dataset_file",
    "dataset_manifest",
    "derivation_manifest",
    "derivation_spec",
    "diagnostic_run_report_model",
    "display_preview",
    "evaluation_report_model",
    "exact_rational",
    "execution_environment_binding",
    "external_compatibility_assessment",
    "external_historical_aggregate",
    "external_historical_case",
    "external_historical_category_aggregate",
    "external_historical_evidence",
    "external_historical_evidence_report",
    "external_transformation_record",
    "ingestion_plan_manifest",
    "ingestion_plan_record",
    "interaction_spec",
    "logical_context_manifest_entry",
    "logical_context_record",
    "mab65_report_reduction",
    "mab_capability_metric_summary",
    "mab_component_metric_summary",
    "mab_plan_evidence_binding",
    "mab_plan_metric_summary",
    "measurement_dimension_spec",
    "measurement_summary_line",
    "memory_system_runtime_binding",
    "memory_system_spec",
    "memory_conformance_evidence_manifest",
    "memory_conformance_occurrence_record",
    "memory_conformance_spec",
    "model_readiness_occurrence_record",
    "model_role_binding",
    "metric_spec",
    "metric_summary",
    "occurrence_claim_record",
    "origin_record",
    "output_contract",
    "paired_metric_delta",
    "report_spec",
    "price_snapshot",
    "prompt_pack_manifest",
    "protocol_spec",
    "provider_budget_cap",
    "provider_operation_budget_ceiling",
    "provider_runtime_profile_attestation",
    "provider_service_evidence_manifest",
    "raw_reference",
    "recovery_decision_record",
    "reducer_binding",
    "release_report_model",
    "report_artifact_manifest",
    "report_identity_spec_binding",
    "report_record_projection",
    "resource_usage_record",
    "resource_budget_ceiling",
    "role_budget_ceiling",
    "run_lease_heartbeat_record",
    "run_lease_record",
    "run_record",
    "run_preflight_record",
    "run_comparison_control_basis_record",
    "run_report_model",
    "run_spec",
    "run_summary",
    "runtime_measurement_control_record",
    "source_evidence_binding",
    "dispatch_budget_route",
    "token_usage_record",
    "validation_issue",
    "validation_claim_boundary",
    "validation_profile",
    "validation_result",
    "validation_rule_requirement",
    "workload_spec",
    "workload_execution_control_record",
}

EXPECTED_V2_SCHEMAS = {
    "attempt_record",
    "attempt_intent_record",
    "budget_reservation_record",
    "budget_spec",
    "case_record",
    "comparison_control_snapshot",
    "cost_record",
    "derivation_spec",
    "ingestion_plan_record",
    "memory_system_runtime_binding",
    "model_role_binding",
    "report_artifact_manifest",
    "report_identity_spec_binding",
    "report_spec",
    "resource_usage_record",
    "run_report_model",
    "run_preflight_record",
    "run_summary",
    "token_usage_record",
}

EXPECTED_V3_SCHEMAS = {
    "attempt_intent_record",
    "attempt_record",
    "budget_reservation_record",
    "budget_spec",
    "case_record",
    "derivation_spec",
    "ingestion_plan_record",
    "report_artifact_manifest",
    "run_report_model",
    "token_usage_record",
}

EXPECTED_V4_SCHEMAS = {"attempt_record", "budget_spec", "token_usage_record"}

EXPECTED_V5_SCHEMAS = {"token_usage_record"}


def require_schema() -> ModuleType:
    try:
        return importlib.import_module("oamb.contracts.schema")
    except ModuleNotFoundError:
        pytest.fail("oamb.contracts.schema is not implemented", pytrace=False)


def test_versioned_contract_inventory_is_explicit_and_unique() -> None:
    schema = require_schema()

    actual = {schema._registry_key(model)[0] for model in schema.VERSIONED_CONTRACTS}

    assert actual == EXPECTED_VERSIONED_CONTRACT_NAMES
    assert len(schema.CONTRACT_REGISTRY) == len(schema.VERSIONED_CONTRACTS)
    assert set(schema.CONTRACT_REGISTRY) == {
        *((name, 1) for name in EXPECTED_VERSIONED_CONTRACT_NAMES),
        *((name, 2) for name in EXPECTED_V2_SCHEMAS),
        *((name, 3) for name in EXPECTED_V3_SCHEMAS),
        *((name, 4) for name in EXPECTED_V4_SCHEMAS),
        *((name, 5) for name in EXPECTED_V5_SCHEMAS),
    }


def test_contract_registry_exposes_no_generated_schema_maintenance_surface() -> None:
    schema = require_schema()
    removed_names = {
        "PUBLIC_CONTRACTS",
        "schema_filename",
        "schema_bytes",
        "expected_schema_files",
        "generate_schemas",
        "schema_drift",
        "packaged_schema_names",
    }

    assert not any(hasattr(schema, name) for name in removed_names)


@pytest.mark.parametrize("maximum", [1.5, "1.0", "01"])
def test_runtime_parser_rejects_noncanonical_decimal_json(maximum: object) -> None:
    schema = require_schema()
    document = {
        "schema_name": "budget_spec",
        "schema_version": 1,
        "budget_id": "budget-1",
        "scope_kind": "run",
        "scope_id": "run-1",
        "max_attempts": 1,
        "max_input_tokens": 1,
        "max_output_tokens": 1,
        "max_wall_seconds": maximum,
        "max_cost": None,
        "currency": None,
    }

    with pytest.raises(ValidationError, match="decimal"):
        schema.parse_contract(document)


def test_unknown_schema_or_version_fails_closed() -> None:
    schema = require_schema()

    with pytest.raises(schema.UnknownContractError):
        schema.parse_contract({"schema_name": "missing", "schema_version": 1})
    with pytest.raises(schema.UnknownContractError):
        schema.parse_contract({"schema_name": "protocol_spec", "schema_version": 99})
    with pytest.raises(schema.UnknownContractError):
        schema.parse_contract({"schema_name": "protocol_spec", "schema_version": True})


def test_parser_rejects_missing_nested_discriminators() -> None:
    schema = require_schema()
    manifest = {
        "schema_name": "dataset_manifest",
        "schema_version": 1,
        "dataset_id": "dataset-1",
        "revision": "r1",
        "split": "test",
        "manifest_hash": "a" * 64,
        "source_files": [
            {
                "relative_path": "fixture.json",
                "sha256": "b" * 64,
                "byte_count": 1,
                "license_id": "Apache-2.0",
            }
        ],
        "payload_policy": "fixture-only",
    }

    with pytest.raises(ValidationError, match="schema_name"):
        schema.parse_contract(manifest)
