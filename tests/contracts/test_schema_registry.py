from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest
from pydantic import ValidationError

EXPECTED_PUBLIC_SCHEMAS = {
    "attempt_intent_record",
    "attempt_receipt_record",
    "attempt_record",
    "budget_reservation_record",
    "budget_spec",
    "capsule_manifest",
    "capsule_manifest_entry",
    "case_manifest",
    "case_manifest_entry",
    "case_record",
    "checkpoint_manifest",
    "close_error_record",
    "cost_measurement_spec",
    "cost_record",
    "dataset_file",
    "dataset_manifest",
    "derivation_manifest",
    "derivation_spec",
    "execution_environment_binding",
    "external_call_approval_record",
    "ingestion_plan_manifest",
    "ingestion_plan_record",
    "interaction_spec",
    "logical_context_manifest_entry",
    "logical_context_record",
    "measurement_dimension_spec",
    "memory_system_runtime_binding",
    "memory_system_spec",
    "model_readiness_occurrence_record",
    "model_role_binding",
    "occurrence_claim_record",
    "origin_record",
    "price_snapshot",
    "protocol_spec",
    "provider_budget_cap",
    "provider_runtime_profile_attestation",
    "provider_service_evidence_manifest",
    "raw_reference",
    "recovery_decision_record",
    "report_artifact_manifest",
    "resource_usage_record",
    "resource_budget_ceiling",
    "role_budget_ceiling",
    "run_lease_heartbeat_record",
    "run_lease_record",
    "run_record",
    "run_report_model",
    "run_spec",
    "run_summary",
    "source_evidence_binding",
    "token_usage_record",
    "validation_issue",
    "validation_profile",
    "validation_result",
    "validation_rule_requirement",
    "workload_spec",
}

EXPECTED_V2_SCHEMAS = {
    "attempt_record",
    "budget_spec",
    "case_record",
    "memory_system_runtime_binding",
    "model_role_binding",
    "run_report_model",
    "run_summary",
    "token_usage_record",
}


def require_schema() -> ModuleType:
    try:
        return importlib.import_module("oamb.contracts.schema")
    except ModuleNotFoundError:
        pytest.fail("oamb.contracts.schema is not implemented", pytrace=False)


def test_public_contract_inventory_is_explicit_and_unique() -> None:
    schema = require_schema()

    actual = {model.model_fields["schema_name"].default for model in schema.PUBLIC_CONTRACTS}

    assert actual == EXPECTED_PUBLIC_SCHEMAS
    assert len(schema.CONTRACT_REGISTRY) == len(schema.PUBLIC_CONTRACTS)
    assert set(schema.CONTRACT_REGISTRY) == {
        *((name, 1) for name in EXPECTED_PUBLIC_SCHEMAS),
        *((name, 2) for name in EXPECTED_V2_SCHEMAS),
    }


def test_schema_generation_is_byte_reproducible(tmp_path: Path) -> None:
    schema = require_schema()
    first = tmp_path / "first"
    second = tmp_path / "second"

    schema.generate_schemas(first)
    schema.generate_schemas(second)

    first_files = sorted(path.name for path in first.iterdir())
    second_files = sorted(path.name for path in second.iterdir())
    assert first_files == second_files
    assert [path.read_bytes() for path in sorted(first.iterdir())] == [
        path.read_bytes() for path in sorted(second.iterdir())
    ]
    for path in first.iterdir():
        document = json.loads(path.read_text())
        assert document
        assert {"schema_name", "schema_version"} <= set(document["required"])
        _assert_nested_discriminators_are_required(document)


def _assert_nested_discriminators_are_required(value: object) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and {"schema_name", "schema_version"} <= set(properties):
            assert {"schema_name", "schema_version"} <= set(value["required"])
        for nested in value.values():
            _assert_nested_discriminators_are_required(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_nested_discriminators_are_required(nested)


def test_decimal_schemas_accept_only_canonical_strings() -> None:
    schema = require_schema()

    cost_schema = json.loads(schema.schema_bytes(schema.CONTRACT_REGISTRY[("cost_record", 1)]))
    amount_schema = cost_schema["properties"]["amount"]["anyOf"][0]

    assert amount_schema["type"] == "string"
    assert "number" not in json.dumps(amount_schema)


def test_schema_check_detects_changed_and_extra_files(tmp_path: Path) -> None:
    schema = require_schema()
    schema.generate_schemas(tmp_path)
    changed = next(tmp_path.iterdir())
    changed.write_text("{}\n", encoding="utf-8")
    (tmp_path / "extra.schema.json").write_text("{}\n", encoding="utf-8")

    drift = schema.schema_drift(tmp_path)

    assert any(item.startswith("changed:") for item in drift)
    assert "extra:extra.schema.json" in drift


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
