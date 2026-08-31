from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from oamb.artifacts.validation import core, profiles, registry
from oamb.contracts.specifications import TransportProfile

HASHES = {str(index): str(index) * 64 for index in range(1, 10)}
HASHES.update({letter: letter * 64 for letter in "abcdef"})
EXPECTED_INVENTORY = (
    "t4-provider-schema",
    "t4-provider-manifest-closure",
    "t4-provider-path-safety",
    "t4-provider-occurrence-terminal",
    "t4-provider-parent-isolation",
    "t4-provider-budget-closure",
    "t4-provider-attempt-ordering",
    "t4-provider-unknown-outcome",
    "t4-provider-runtime-binding",
    "t4-provider-gate-separation",
    "t4-provider-control-redaction",
)
EXPECTED_INVENTORY_HASH = "73fc43ce2152e4ccf4e8c103c5c41bc0302eaa68bac54a0731ae6085dbd47d15"
MANIFEST_NAME = "provider-service-evidence-manifest.json"


def _json_bytes(document: dict[str, Any]) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _attestation_hash(document: dict[str, Any]) -> str:
    from oamb.contracts.specifications import provider_runtime_profile_attestation_hash

    return provider_runtime_profile_attestation_hash(document)


def _write_document(root: Path, relative_path: str, document: dict[str, Any]) -> None:
    target = root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_json_bytes(document))


def _resource_ceiling(maximum: str = "60") -> dict[str, Any]:
    return {
        "schema_name": "resource_budget_ceiling",
        "schema_version": 1,
        "dimension_id": "provider_request_wall_seconds_v1",
        "maximum": maximum,
        "unit": "seconds",
    }


def _valid_documents() -> dict[str, tuple[str, dict[str, Any] | bytes]]:
    role = {
        "schema_name": "model_role_binding",
        "schema_version": 2,
        "binding_id": "embedding-binding",
        "role": "embedding",
        "role_status": "selected",
        "execution_owner": "memory_system",
        "binding_kind": "native",
        "provider": "ollama",
        "endpoint_reference": "controlled-embedding-endpoint",
        "credential_variable_name": None,
        "configured_model": "qwen3-embedding:0.6b",
        "resolved_model": "qwen3-embedding:0.6b@sha256:manifest",
        "thinking_effort": "not_applicable",
        "parameters_fingerprint": HASHES["1"],
        "retry_policy_id": "no-retry-v1",
        "configuration_fingerprint": HASHES["2"],
        "redacted_endpoint_fingerprint": HASHES["3"],
    }
    role_ceiling = {
        "schema_name": "role_budget_ceiling",
        "schema_version": 1,
        "role_binding_id": "embedding-binding",
        "max_attempts": 2,
        "max_input_tokens": 0,
        "max_output_tokens": 0,
        "max_dispatch_wall_seconds": "60",
        "max_cost": None,
        "currency": None,
        "price_snapshot_id": None,
        "resource_ceilings": [_resource_ceiling()],
        "provider_budget_cap": {
            "schema_name": "provider_budget_cap",
            "schema_version": 1,
            "provider": "ollama",
            "operation_kind": "embedding_readiness",
            "billing_unit": "request",
            "maximum_accepted_units": "2",
        },
    }
    budget = {
        "schema_name": "budget_spec",
        "schema_version": 2,
        "budget_id": "readiness-budget",
        "scope_kind": "model_readiness",
        "scope_id": "readiness-occurrence",
        "max_attempts": 2,
        "max_input_tokens": 0,
        "max_output_tokens": 0,
        "max_dispatch_wall_seconds": "60",
        "max_cost": None,
        "currency": None,
        "resource_ceilings": [_resource_ceiling()],
        "role_ceilings": [role_ceiling],
        "stop_condition_ids": ["identity_drift", "budget_exhausted"],
    }
    attestation = {
        "schema_name": "provider_runtime_profile_attestation",
        "schema_version": 1,
        "provider": "hindsight",
        "provider_project_id": "provider-project-1",
        "provider_profile_id": "hindsight-rest-v0.9.2",
        "attestation_hash": HASHES["5"],
        "transport_profile": "rest",
        "release_version": "0.9.2",
        "source_revision": "source-revision",
        "build_artifact_sha256": HASHES["6"],
        "redacted_configuration_sha256": HASHES["7"],
        "redacted_endpoint_fingerprint": HASHES["8"],
        "auth_configuration_sha256": HASHES["9"],
        "storage_configuration_sha256": HASHES["a"],
        "model_role_binding_ids": ["embedding-binding"],
        "native_reranking_status": "disabled",
        "liveness_status": "pass",
        "storage_configuration_status": "pass",
        "runtime_identity_status": "pass",
        "model_readiness_status": "not_run",
        "memory_conformance_status": "not_run",
        "raw_proof_refs": [HASHES["b"]],
    }
    attestation["attestation_hash"] = _attestation_hash(attestation)
    measurement = {
        "schema_name": "cost_measurement_spec",
        "schema_version": 1,
        "measurement_spec_id": "readiness-measurement-v1",
        "measurement_spec_version": "1",
        "dimensions": [
            {
                "schema_name": "measurement_dimension_spec",
                "schema_version": 1,
                "dimension_id": "provider_request_wall_seconds_v1",
                "stage": "model_readiness",
                "operation_kind": "embedding_readiness",
                "parent_kind": "model_readiness",
                "unit": "seconds",
                "allowed_meter_sources": ["process_meter"],
                "required": True,
                "price_class": None,
                "indexing_view_rule": "not_applicable",
                "aggregation_operator": "interval_union",
            }
        ],
    }
    occurrence = {
        "schema_name": "model_readiness_occurrence_record",
        "schema_version": 1,
        "occurrence_id": "readiness-occurrence",
        "provider": "hindsight",
        "provider_project_id": "provider-project-1",
        "provider_profile_id": "hindsight-rest-v0.9.2",
        "provider_runtime_profile_attestation_hash": attestation["attestation_hash"],
        "budget_id": "readiness-budget",
        "state": "sealed",
        "role_binding_ids": ["embedding-binding"],
        "attempt_ids": [HASHES["1"], HASHES["2"]],
        "usage_record_ids": [HASHES["3"], HASHES["4"]],
        "resource_record_ids": [],
        "cost_record_ids": [],
        "started_at": "2026-08-27T12:00:00Z",
        "ended_at": "2026-08-27T12:00:04Z",
    }
    documents: dict[str, tuple[str, dict[str, Any] | bytes]] = {
        "source/specs/provider-runtime-profile-attestation.json": (
            "provider_runtime_profile_attestation",
            attestation,
        ),
        "source/specs/model-role-bindings/embedding-binding.json": (
            "model_role_binding",
            role,
        ),
        "source/specs/budget.json": ("budget_spec", budget),
        "source/specs/cost-measurement-spec.json": ("cost_measurement_spec", measurement),
        "source/occurrence.json": ("model_readiness_occurrence_record", occurrence),
    }
    for ordinal, attempt_id in enumerate((HASHES["1"], HASHES["2"]), start=1):
        reservation_id = HASHES[str(ordinal + 4)]
        request_fingerprint = HASHES[str(ordinal + 6)]
        claim_id = HASHES[str(ordinal + 8)] if ordinal == 1 else HASHES["b"]
        raw_payload = _json_bytes({"attempt": ordinal, "ok": ordinal == 2})
        raw_hash = _sha256(raw_payload)
        documents[f"source/occurrence-claims/{claim_id}.json"] = (
            "occurrence_claim_record",
            {
                "schema_name": "occurrence_claim_record",
                "schema_version": 1,
                "claim_id": claim_id,
                "occurrence_id": "readiness-occurrence",
                "lease_record_hash": HASHES["c"],
                "lease_epoch": 1,
                "owner_id": "readiness-runner",
                "stage": "model_readiness",
                "request_fingerprint": request_fingerprint,
                "reconciliation_capability": "none",
                "claimed_at": f"2026-08-27T12:00:0{ordinal - 1}Z",
            },
        )
        documents[f"source/budget-reservations/{reservation_id}.json"] = (
            "budget_reservation_record",
            {
                "schema_name": "budget_reservation_record",
                "schema_version": 1,
                "reservation_id": reservation_id,
                "budget_id": "readiness-budget",
                "scope_kind": "model_readiness",
                "scope_id": "readiness-occurrence",
                "role_binding_id": "embedding-binding",
                "attempt_id": attempt_id,
                "reserved_attempts": 1,
                "reserved_input_tokens": 0,
                "reserved_output_tokens": 0,
                "reserved_dispatch_wall_seconds": "30",
                "reserved_cost": None,
                "currency": None,
                "reserved_resource_ceilings": [_resource_ceiling("30")],
                "reserved_provider_units": "1",
                "reserved_at": f"2026-08-27T12:00:0{ordinal - 1}Z",
            },
        )
        documents[f"source/attempt-intents/{attempt_id}.json"] = (
            "attempt_intent_record",
            {
                "schema_name": "attempt_intent_record",
                "schema_version": 1,
                "attempt_id": attempt_id,
                "claim_id": claim_id,
                "reservation_id": reservation_id,
                "parent_kind": "model_readiness",
                "parent_id": "readiness-occurrence",
                "role_binding_id": "embedding-binding",
                "stage": "model_readiness",
                "request_fingerprint": request_fingerprint,
                "reconciliation_capability": "none",
                "idempotency_key_hash": None,
                "sealed_at": f"2026-08-27T12:00:0{ordinal - 1}Z",
            },
        )
        documents[f"source/attempt-receipts/{attempt_id}.json"] = (
            "attempt_receipt_record",
            {
                "schema_name": "attempt_receipt_record",
                "schema_version": 1,
                "attempt_id": attempt_id,
                "receipt_kind": "response",
                "raw_response_ref": raw_hash,
                "raw_error_ref": None,
                "dispatch_started_at": f"2026-08-27T12:00:0{ordinal - 1}Z",
                "receipt_observed_at": f"2026-08-27T12:00:0{ordinal}Z",
                "provider_request_wall_seconds": "1",
            },
        )
        documents[f"source/attempts/{attempt_id}.json"] = (
            "attempt_record",
            {
                "schema_name": "attempt_record",
                "schema_version": 2,
                "attempt_id": attempt_id,
                "parent_kind": "model_readiness",
                "parent_id": "readiness-occurrence",
                "stage": "model_readiness",
                "ordinal": ordinal,
                "request_fingerprint": request_fingerprint,
                "started_at": f"2026-08-27T12:00:0{ordinal - 1}Z",
                "ended_at": f"2026-08-27T12:00:0{ordinal}Z",
                "outcome": "failed" if ordinal == 1 else "succeeded",
                "retry_of_attempt_id": None if ordinal == 1 else HASHES["1"],
                "idempotency_key_hash": None,
                "reconciliation_capability": "none",
                "raw_response_ref": raw_hash,
                "raw_error_ref": None,
                "index_contribution": "not_applicable",
                "superseded_by_attempt_id": None,
            },
        )
        usage_id = HASHES[str(ordinal + 2)]
        documents[f"source/usage/{usage_id}.json"] = (
            "token_usage_record",
            {
                "schema_name": "token_usage_record",
                "schema_version": 2,
                "usage_record_id": usage_id,
                "attempt_id": attempt_id,
                "parent_kind": "model_readiness",
                "parent_id": "readiness-occurrence",
                "stage": "model_readiness",
                "operation_kind": "embedding_readiness",
                "token_domain": "external_llm",
                "measurement_source": "supplier_response",
                "input_tokens": None,
                "visible_output_tokens": None,
                "supplier_reported_total_tokens": None,
                "context_view_tokens": None,
                "proof_status": "unavailable",
                "reason": "provider_has_no_portable_billing_receipt",
                "raw_response_ref": raw_hash,
            },
        )
        documents[f"source/raw/{raw_hash}.json"] = ("raw", raw_payload)
    return documents


def _record_id(record_kind: str, document: dict[str, Any] | bytes, path: str) -> str:
    if isinstance(document, bytes):
        return Path(path).stem
    fields = {
        "provider_runtime_profile_attestation": "attestation_hash",
        "model_role_binding": "binding_id",
        "budget_spec": "budget_id",
        "cost_measurement_spec": "measurement_spec_id",
        "model_readiness_occurrence_record": "occurrence_id",
        "occurrence_claim_record": "claim_id",
        "budget_reservation_record": "reservation_id",
        "attempt_intent_record": "attempt_id",
        "attempt_receipt_record": "attempt_id",
        "attempt_record": "attempt_id",
        "token_usage_record": "usage_record_id",
        "cost_record": "cost_record_id",
        "run_spec": "run_id",
    }
    return str(document[fields[record_kind]])


def _refresh_manifest(root: Path) -> None:
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["source_entries"]:
        entry["sha256"] = _sha256((root / entry["relative_path"]).read_bytes())
    manifest["source_entries"].sort(key=lambda entry: entry["relative_path"])
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = _sha256(_json_bytes(without_hash))
    manifest_path.write_bytes(_json_bytes(manifest))


def _write_valid_root(root: Path) -> core.ProviderServiceEvidenceValidationInput:
    documents = _valid_documents()
    entries = []
    for relative_path, (record_kind, document) in documents.items():
        content = document if isinstance(document, bytes) else _json_bytes(document)
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        entries.append(
            {
                "schema_name": "capsule_manifest_entry",
                "schema_version": 1,
                "record_kind": record_kind,
                "record_id": _record_id(record_kind, document, relative_path),
                "relative_path": relative_path,
                "sha256": _sha256(content),
            }
        )
    entries.sort(key=lambda entry: str(entry["relative_path"]))
    manifest = {
        "schema_name": "provider_service_evidence_manifest",
        "schema_version": 1,
        "manifest_hash": "0" * 64,
        "provider_project_id": "provider-project-1",
        "provider_profile_id": "hindsight-rest-v0.9.2",
        "occurrence_id": "readiness-occurrence",
        "operation": "model_readiness",
        "source_entries": entries,
        "created_at": "2026-08-27T12:00:05Z",
    }
    without_hash = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    manifest["manifest_hash"] = _sha256(_json_bytes(without_hash))
    (root / MANIFEST_NAME).write_bytes(_json_bytes(manifest))
    return core.ProviderServiceEvidenceValidationInput(
        root_directory=root,
        source_kind="provider_service",
        operation="model_readiness",
        expected_provider="hindsight",
        expected_provider_project_id="provider-project-1",
        expected_provider_profile_id="hindsight-rest-v0.9.2",
        expected_transport_profile=TransportProfile.REST,
        expected_release_version="0.9.2",
        expected_source_revision="source-revision",
        expected_build_artifact_sha256=HASHES["6"],
        forbidden_control_values=(),
    )


def _rewrite(
    root: Path,
    relative_path: str,
    mutate: Any,
) -> None:
    path = root / relative_path
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_bytes(_json_bytes(document))
    _refresh_manifest(root)


def _rewrite_attestation(root: Path, mutate: Any) -> None:
    attestation_path = root / "source/specs/provider-runtime-profile-attestation.json"
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    mutate(attestation)
    attestation["attestation_hash"] = _attestation_hash(attestation)
    attestation_path.write_bytes(_json_bytes(attestation))

    occurrence_path = root / "source/occurrence.json"
    occurrence = json.loads(occurrence_path.read_text(encoding="utf-8"))
    occurrence["provider_runtime_profile_attestation_hash"] = attestation["attestation_hash"]
    occurrence_path.write_bytes(_json_bytes(occurrence))

    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = next(
        item
        for item in manifest["source_entries"]
        if item["record_kind"] == "provider_runtime_profile_attestation"
    )
    entry["record_id"] = attestation["attestation_hash"]
    manifest_path.write_bytes(_json_bytes(manifest))
    _refresh_manifest(root)


def _add_indexed_document(
    root: Path,
    relative_path: str,
    record_kind: str,
    document: dict[str, Any],
) -> None:
    _write_document(root, relative_path, document)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_entries"].append(
        {
            "schema_name": "capsule_manifest_entry",
            "schema_version": 1,
            "record_kind": record_kind,
            "record_id": _record_id(record_kind, document, relative_path),
            "relative_path": relative_path,
            "sha256": _sha256((root / relative_path).read_bytes()),
        }
    )
    manifest_path.write_bytes(_json_bytes(manifest))
    _refresh_manifest(root)


def _remove_indexed_file(root: Path, relative_path: str) -> None:
    (root / relative_path).unlink()
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_entries"] = [
        entry for entry in manifest["source_entries"] if entry["relative_path"] != relative_path
    ]
    manifest_path.write_bytes(_json_bytes(manifest))
    _refresh_manifest(root)


def _mutate_for_rule(
    rule_id: str,
    root: Path,
    value: core.ProviderServiceEvidenceValidationInput,
) -> core.ProviderServiceEvidenceValidationInput:
    if rule_id == "t4-provider-schema":
        _rewrite(
            root,
            "source/specs/cost-measurement-spec.json",
            lambda document: document.update({"unknown_field": True}),
        )
    elif rule_id == "t4-provider-manifest-closure":
        _write_document(root, "source/unindexed.json", {"unindexed": True})
    elif rule_id == "t4-provider-path-safety":
        manifest_path = root / MANIFEST_NAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw_entry = next(
            entry for entry in manifest["source_entries"] if entry["record_kind"] == "raw"
        )
        old_path = root / raw_entry["relative_path"]
        new_relative_path = f"source/.runtime/{old_path.name}"
        new_path = root / new_relative_path
        new_path.parent.mkdir(parents=True, exist_ok=True)
        old_path.rename(new_path)
        raw_entry["relative_path"] = new_relative_path
        manifest_path.write_bytes(_json_bytes(manifest))
        _refresh_manifest(root)
    elif rule_id == "t4-provider-occurrence-terminal":
        _rewrite(
            root,
            "source/occurrence.json",
            lambda document: document.update({"state": "running", "ended_at": None}),
        )
    elif rule_id == "t4-provider-parent-isolation":
        _rewrite(
            root,
            f"source/attempts/{HASHES['1']}.json",
            lambda document: document.update({"parent_kind": "case", "parent_id": "case-1"}),
        )
        _rewrite(
            root,
            f"source/attempt-intents/{HASHES['1']}.json",
            lambda document: document.update({"parent_kind": "case", "parent_id": "case-1"}),
        )
        _rewrite(
            root,
            f"source/occurrence-claims/{HASHES['9']}.json",
            lambda document: document.update({"occurrence_id": "case-1"}),
        )
    elif rule_id == "t4-provider-budget-closure":
        _rewrite(
            root,
            f"source/budget-reservations/{HASHES['5']}.json",
            lambda document: document.update({"reserved_provider_units": "3"}),
        )
    elif rule_id == "t4-provider-attempt-ordering":
        _rewrite(
            root,
            f"source/attempt-intents/{HASHES['1']}.json",
            lambda document: document.update({"request_fingerprint": HASHES["f"]}),
        )
    elif rule_id == "t4-provider-unknown-outcome":
        attempt_id = HASHES["2"]
        receipt_path = f"source/attempt-receipts/{attempt_id}.json"
        receipt = json.loads((root / receipt_path).read_text(encoding="utf-8"))
        raw_path = f"source/raw/{receipt['raw_response_ref']}.json"
        _rewrite(
            root,
            f"source/attempts/{attempt_id}.json",
            lambda document: document.update(
                {"outcome": "unknown_outcome", "raw_response_ref": None}
            ),
        )
        _rewrite(
            root,
            f"source/usage/{HASHES['4']}.json",
            lambda document: document.update({"raw_response_ref": None}),
        )
        _remove_indexed_file(root, receipt_path)
        _remove_indexed_file(root, raw_path)
    elif rule_id == "t4-provider-runtime-binding":
        _rewrite(
            root,
            "source/occurrence.json",
            lambda document: document.update({"provider_profile_id": "different-profile"}),
        )
    elif rule_id == "t4-provider-gate-separation":
        _add_indexed_document(
            root,
            "source/specs/run-spec.json",
            "run_spec",
            {
                "schema_name": "run_spec",
                "schema_version": 1,
                "run_id": "promoted-run",
                "protocol_id": "oamb-v0.1",
                "dataset_manifest_hash": HASHES["1"],
                "case_manifest_hash": HASHES["2"],
                "workload_id": "lme30-v1",
                "memory_system_id": "hindsight",
                "runtime_binding_hash": HASHES["3"],
                "environment_hash": HASHES["4"],
                "model_role_binding_ids": ["embedding-binding"],
                "budget_id": "readiness-budget",
                "code_revision": "revision",
                "normalizer_fingerprint": HASHES["5"],
            },
        )
    elif rule_id == "t4-provider-control-redaction":
        secret_endpoint = "https://secret.internal:8443/v1"
        _rewrite(
            root,
            "source/specs/model-role-bindings/embedding-binding.json",
            lambda document: document.update({"endpoint_reference": secret_endpoint}),
        )
        return value.model_copy(update={"forbidden_control_values": (secret_endpoint,)})
    else:
        raise AssertionError(f"unknown rule mutation: {rule_id}")
    return value


def _validate(value: core.ProviderServiceEvidenceValidationInput) -> Any:
    return core.validate_provider_service_evidence(
        value,
        profiles.provider_service_evidence_profile(),
        registry.provider_service_registry(),
    )


def test_t4_provider_profile_freezes_exact_order_version_and_hash() -> None:
    profile = profiles.provider_service_evidence_profile()

    assert (
        tuple(requirement.rule_id for requirement in profile.required_rules) == EXPECTED_INVENTORY
    )
    assert tuple(requirement.minimum_version for requirement in profile.required_rules) == (1,) * 11
    assert profile.required_rule_inventory_hash == EXPECTED_INVENTORY_HASH
    assert profile.applicability == (
        "source_kind=provider_service",
        "operation=model_readiness",
    )


def test_valid_provider_root_executes_every_closed_rule(tmp_path: Path) -> None:
    result = _validate(_write_valid_root(tmp_path / "provider-root"))

    assert result.disposition.value == "validated"
    assert result.required_rule_ids == EXPECTED_INVENTORY
    assert result.executed_rule_ids == EXPECTED_INVENTORY
    assert result.passed_rule_ids == EXPECTED_INVENTORY
    assert result.failed_rule_ids == ()
    assert result.missing_rule_ids == ()


def test_unavailable_cost_record_is_closed_by_occurrence_inventory_not_parent_fields(
    tmp_path: Path,
) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    cost_record_id = HASHES["d"]
    _add_indexed_document(
        root,
        f"source/cost/{cost_record_id}.json",
        "cost_record",
        {
            "schema_name": "cost_record",
            "schema_version": 1,
            "cost_record_id": cost_record_id,
            "parent_kind": "model_readiness",
            "parent_id": "readiness-occurrence",
            "basis": "actual_supplier_charge",
            "indexing_view": "not_applicable",
            "amount": None,
            "currency": None,
            "price_snapshot_id": None,
            "source_usage_record_ids": [HASHES["3"]],
            "source_resource_record_ids": [],
            "proof_status": "unavailable",
            "reason": "supplier_billing_receipt_is_unavailable",
        },
    )
    _rewrite(
        root,
        "source/occurrence.json",
        lambda document: document.update({"cost_record_ids": [cost_record_id]}),
    )

    result = _validate(value)

    assert result.disposition.value == "validated", result.issues


def test_cost_record_sources_must_belong_to_the_occurrence_inventory(tmp_path: Path) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    cost_record_id = HASHES["d"]
    _add_indexed_document(
        root,
        f"source/cost/{cost_record_id}.json",
        "cost_record",
        {
            "schema_name": "cost_record",
            "schema_version": 1,
            "cost_record_id": cost_record_id,
            "parent_kind": "model_readiness",
            "parent_id": "readiness-occurrence",
            "basis": "actual_supplier_charge",
            "indexing_view": "not_applicable",
            "amount": None,
            "currency": None,
            "price_snapshot_id": None,
            "source_usage_record_ids": [HASHES["e"]],
            "source_resource_record_ids": [],
            "proof_status": "unavailable",
            "reason": "supplier_billing_receipt_is_unavailable",
        },
    )
    _rewrite(
        root,
        "source/occurrence.json",
        lambda document: document.update({"cost_record_ids": [cost_record_id]}),
    )

    result = _validate(value)

    assert result.disposition.value == "invalid"
    assert result.failed_rule_ids == ("t4-provider-parent-isolation",)


def test_unknown_outcome_without_provider_receipt_is_valid_and_not_replayed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    attempt_id = HASHES["2"]
    receipt_path = f"source/attempt-receipts/{attempt_id}.json"
    receipt = json.loads((root / receipt_path).read_text(encoding="utf-8"))
    raw_path = f"source/raw/{receipt['raw_response_ref']}.json"

    _rewrite(
        root,
        f"source/attempts/{attempt_id}.json",
        lambda document: document.update({"outcome": "unknown_outcome", "raw_response_ref": None}),
    )
    _rewrite(
        root,
        f"source/usage/{HASHES['4']}.json",
        lambda document: document.update({"raw_response_ref": None}),
    )
    _rewrite(
        root,
        "source/occurrence.json",
        lambda document: document.update({"state": "interrupted_unknown_outcome"}),
    )
    _remove_indexed_file(root, receipt_path)
    _remove_indexed_file(root, raw_path)

    result = _validate(value)

    assert result.disposition.value == "validated", result.issues
    assert result.failed_rule_ids == ()


def test_manifest_rejects_raw_record_id_that_is_not_the_payload_hash(tmp_path: Path) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    manifest = json.loads((root / MANIFEST_NAME).read_text(encoding="utf-8"))
    raw_entry = next(entry for entry in manifest["source_entries"] if entry["record_kind"] == "raw")
    (root / raw_entry["relative_path"]).write_bytes(b"different raw payload")
    _refresh_manifest(root)

    result = _validate(value)

    assert result.disposition.value == "invalid"
    assert "t4-provider-manifest-closure" in result.failed_rule_ids


def test_manifest_rejects_typed_record_id_that_differs_from_contract_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attempt_entry = next(
        entry
        for entry in manifest["source_entries"]
        if entry["record_kind"] == "attempt_record" and entry["record_id"] == HASHES["1"]
    )
    attempt_entry["record_id"] = "0" * 64
    manifest_path.write_bytes(_json_bytes(manifest))
    _rewrite(
        root,
        "source/occurrence.json",
        lambda document: document.update({"attempt_ids": ["0" * 64, HASHES["2"]]}),
    )

    result = _validate(value)

    assert result.disposition.value == "invalid"
    assert "t4-provider-manifest-closure" in result.failed_rule_ids


def test_budget_closure_rejects_reservation_currency_drift(tmp_path: Path) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    budget_path = "source/specs/budget.json"
    _rewrite(
        root,
        budget_path,
        lambda document: (
            document.update({"max_cost": "1", "currency": "USD"}),
            document["role_ceilings"][0].update({"max_cost": "1", "currency": "USD"}),
        ),
    )
    _rewrite(
        root,
        f"source/budget-reservations/{HASHES['5']}.json",
        lambda document: document.update({"reserved_cost": "0.1", "currency": "EUR"}),
    )

    result = _validate(value)

    assert result.disposition.value == "invalid"
    assert "t4-provider-budget-closure" in result.failed_rule_ids


def test_runtime_binding_rejects_resealed_exact_release_identity_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    attestation_path = "source/specs/provider-runtime-profile-attestation.json"
    attestation = json.loads((root / attestation_path).read_text(encoding="utf-8"))
    attestation["release_version"] = "9.9.9"
    attestation["source_revision"] = "different-source"
    attestation["attestation_hash"] = _attestation_hash(attestation)
    (root / attestation_path).write_bytes(_json_bytes(attestation))
    _rewrite(
        root,
        "source/occurrence.json",
        lambda document: document.update(
            {"provider_runtime_profile_attestation_hash": attestation["attestation_hash"]}
        ),
    )
    manifest_path = root / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attestation_entry = next(
        entry
        for entry in manifest["source_entries"]
        if entry["record_kind"] == "provider_runtime_profile_attestation"
    )
    attestation_entry["record_id"] = attestation["attestation_hash"]
    manifest_path.write_bytes(_json_bytes(manifest))
    _refresh_manifest(root)

    result = _validate(value)

    assert result.disposition.value == "invalid"
    assert "t4-provider-runtime-binding" in result.failed_rule_ids


@pytest.mark.parametrize(
    ("relative_path", "updates"),
    [
        (f"source/attempts/{HASHES['1']}.json", {"stage": "answer"}),
        (
            f"source/attempt-intents/{HASHES['1']}.json",
            {"role_binding_id": "missing-role"},
        ),
        (
            f"source/attempt-intents/{HASHES['1']}.json",
            {
                "reconciliation_capability": "idempotency_key",
                "idempotency_key_hash": HASHES["f"],
            },
        ),
    ],
)
def test_attempt_ordering_closes_stage_role_and_reconciliation_edges(
    tmp_path: Path,
    relative_path: str,
    updates: dict[str, Any],
) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    _rewrite(root, relative_path, lambda document: document.update(updates))

    result = _validate(value)

    assert result.disposition.value == "invalid"
    assert "t4-provider-attempt-ordering" in result.failed_rule_ids


def test_attempt_ordering_requires_each_durable_occurrence_claim(tmp_path: Path) -> None:
    root = tmp_path / "provider-root"
    value = _write_valid_root(root)
    _remove_indexed_file(root, f"source/occurrence-claims/{HASHES['9']}.json")

    result = _validate(value)

    assert result.disposition.value == "invalid"
    assert "t4-provider-attempt-ordering" in result.failed_rule_ids


@pytest.mark.parametrize("rule_id", EXPECTED_INVENTORY)
def test_each_t4_provider_rule_rejects_an_independent_planted_failure(
    tmp_path: Path,
    rule_id: str,
) -> None:
    root = tmp_path / rule_id
    value = _write_valid_root(root)
    mutated = _mutate_for_rule(rule_id, root, value)

    result = _validate(mutated)

    assert result.disposition.value == "invalid"
    assert result.failed_rule_ids == (rule_id,)


def test_provider_profile_fails_closed_when_one_required_rule_is_unregistered(
    tmp_path: Path,
) -> None:
    value = _write_valid_root(tmp_path / "provider-root")

    result = core.validate_provider_service_evidence(
        value,
        profiles.provider_service_evidence_profile(),
        registry.provider_service_registry(exclude={"t4-provider-budget-closure"}),
    )

    assert result.disposition.value == "invalid"
    assert result.missing_rule_ids == ("t4-provider-budget-closure",)


def test_attempt_ordering_rejects_intent_sealed_after_dispatch(tmp_path: Path) -> None:
    root = tmp_path / "late-intent"
    value = _write_valid_root(root)
    _rewrite(
        root,
        f"source/attempt-intents/{HASHES['1']}.json",
        lambda document: document.update({"sealed_at": "2026-08-27T12:00:02Z"}),
    )

    result = _validate(value)

    assert result.failed_rule_ids == ("t4-provider-attempt-ordering",)


def test_runtime_binding_requires_all_pre_readiness_service_gates(tmp_path: Path) -> None:
    root = tmp_path / "failed-liveness"
    value = _write_valid_root(root)
    _rewrite_attestation(
        root,
        lambda document: document.update({"liveness_status": "fail"}),
    )

    result = _validate(value)

    assert result.failed_rule_ids == ("t4-provider-runtime-binding",)


def test_budget_closure_rejects_provider_cap_for_a_different_provider(tmp_path: Path) -> None:
    root = tmp_path / "wrong-provider-cap"
    value = _write_valid_root(root)
    _rewrite(
        root,
        "source/specs/budget.json",
        lambda document: document["role_ceilings"][0]["provider_budget_cap"].update(
            {"provider": "different-provider"}
        ),
    )
    result = _validate(value)

    assert result.failed_rule_ids == ("t4-provider-budget-closure",)


def test_path_safety_rejects_a_symlinked_manifest(tmp_path: Path) -> None:
    root = tmp_path / "symlinked-manifest"
    value = _write_valid_root(root)
    manifest_path = root / MANIFEST_NAME
    external_manifest = tmp_path / "external-manifest.json"
    manifest_path.rename(external_manifest)
    manifest_path.symlink_to(external_manifest)

    result = _validate(value)

    assert result.failed_rule_ids == ("t4-provider-schema",)


def test_provider_validation_rejects_a_symlinked_root_without_following_it(
    tmp_path: Path,
) -> None:
    real_root = tmp_path / "real-provider-root"
    value = _write_valid_root(real_root)
    linked_root = tmp_path / "linked-provider-root"
    linked_root.symlink_to(real_root, target_is_directory=True)

    result = _validate(value.model_copy(update={"root_directory": linked_root}))

    assert result.disposition.value == "invalid"
    assert result.failed_rule_ids == ("t4-provider-schema",)


def test_control_redaction_scans_raw_provider_proof_payloads(tmp_path: Path) -> None:
    value = _write_valid_root(tmp_path / "provider-root")

    result = _validate(value.model_copy(update={"forbidden_control_values": ("attempt",)}))

    assert result.failed_rule_ids == ("t4-provider-control-redaction",)
