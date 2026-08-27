from __future__ import annotations

import hashlib
import importlib
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

HASH = "a" * 64
OTHER_HASH = "b" * 64
NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def require(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"{module_name} T4 contracts are not implemented: {exc}", pytrace=False)


def resource_ceiling(specifications: ModuleType) -> Any:
    return specifications.ResourceBudgetCeiling(
        dimension_id="provider_request_wall_seconds_v1",
        maximum=Decimal("30"),
        unit="seconds",
    )


def provider_cap(specifications: ModuleType) -> Any:
    return specifications.ProviderBudgetCap(
        provider="ollama",
        operation_kind="embedding_readiness",
        billing_unit="request",
        maximum_accepted_units=Decimal("1"),
    )


def role_ceiling(specifications: ModuleType) -> Any:
    return specifications.RoleBudgetCeiling(
        role_binding_id="embedding-binding",
        max_attempts=1,
        max_input_tokens=0,
        max_output_tokens=0,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=None,
        currency=None,
        price_snapshot_id=None,
        resource_ceilings=(resource_ceiling(specifications),),
        provider_budget_cap=provider_cap(specifications),
    )


def test_v2_external_contracts_close_roles_budgets_and_readiness_evidence() -> None:
    specifications = require("oamb.contracts.specifications")

    role = specifications.ModelRoleBindingV2(
        binding_id="embedding-binding",
        role=specifications.ModelRole.EMBEDDING,
        role_status=specifications.RoleBindingStatus.SELECTED,
        execution_owner=specifications.ExecutionOwner.MEMORY_SYSTEM,
        binding_kind=specifications.BindingKind.NATIVE,
        provider="ollama",
        endpoint_reference="controlled-embedding-endpoint",
        credential_variable_name=None,
        configured_model="qwen3-embedding:0.6b",
        resolved_model="qwen3-embedding:0.6b@sha256:manifest",
        parameters_fingerprint=HASH,
        retry_policy_id="no-retry-v1",
        configuration_fingerprint=OTHER_HASH,
        redacted_endpoint_fingerprint=HASH,
    )
    budget = specifications.BudgetSpecV2(
        budget_id="readiness-budget",
        scope_kind=specifications.BudgetScopeKindV2.MODEL_READINESS,
        scope_id="readiness-occurrence",
        approval_id="approval-1",
        max_attempts=1,
        max_input_tokens=0,
        max_output_tokens=0,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=None,
        currency=None,
        resource_ceilings=(resource_ceiling(specifications),),
        role_ceilings=(role_ceiling(specifications),),
        stop_condition_ids=("identity_drift", "budget_exhausted"),
    )
    source_binding = specifications.SourceEvidenceBinding(
        binding_id=HASH,
        source_kind=specifications.SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity="readiness-occurrence",
        source_root_hash=OTHER_HASH,
        validation_result_hash=HASH,
        source_schema_versions=(
            "model_readiness_occurrence_record@1",
            "attempt_record@2",
        ),
    )
    runtime_values = dict(
        memory_system_id="hindsight",
        provider_project_id="oamb-hindsight-1",
        provider_profile_id="hindsight-rest-v0.9.2",
        edition="oss",
        distribution_channel="official_container",
        api_version="v1",
        release_version="0.9.2",
        source_revision="commit-1",
        artifact_kind="container_image",
        artifact_sha256=OTHER_HASH,
        endpoint_fingerprint=HASH,
        deployment_configuration_sha256=OTHER_HASH,
        storage_engine="pg0",
        storage_engine_version="1",
        schema_revision="schema-1",
        vector_index_type="native",
        distance_metric="cosine",
        vector_dimension=1024,
        index_configuration_sha256=HASH,
        model_role_binding_ids=(role.binding_id,),
        native_feature_flags_fingerprint=OTHER_HASH,
        native_reranking_status="disabled",
        attestation_method="runtime_and_build_provenance",
        attestation_status=specifications.RuntimeAttestationStatus.RUNTIME_VERIFIED,
        raw_proof_refs=(HASH,),
        model_readiness_required=True,
        model_readiness_evidence=source_binding,
    )
    runtime = specifications.MemorySystemRuntimeBindingV2(
        runtime_binding_hash=specifications.memory_system_runtime_binding_hash(runtime_values),
        **runtime_values,
    )

    assert budget.scope_kind == specifications.BudgetScopeKindV2.MODEL_READINESS
    assert runtime.model_readiness_evidence == source_binding

    with pytest.raises(ValidationError, match="runtime binding hash"):
        specifications.MemorySystemRuntimeBindingV2(
            **(runtime.model_dump() | {"release_version": "99.0.0"})
        )

    with pytest.raises(ValidationError, match="selected binding"):
        specifications.ModelRoleBindingV2(**(role.model_dump() | {"configured_model": None}))
    with pytest.raises(ValidationError, match="duplicate role ceiling"):
        specifications.BudgetSpecV2(
            **(budget.model_dump() | {"role_ceilings": (role_ceiling(specifications),) * 2})
        )
    mixed_currency_role = role_ceiling(specifications).model_copy(
        update={"max_cost": Decimal("1"), "currency": "EUR"}
    )
    with pytest.raises(ValidationError, match="currency"):
        specifications.BudgetSpecV2(
            **(
                budget.model_dump()
                | {
                    "max_cost": Decimal("1"),
                    "currency": "USD",
                    "role_ceilings": (mixed_currency_role,),
                }
            )
        )
    with pytest.raises(ValidationError, match="readiness evidence"):
        specifications.MemorySystemRuntimeBindingV2(
            **(runtime.model_dump() | {"model_readiness_evidence": None})
        )


def test_pre_readiness_attestation_and_approval_keep_gates_separate() -> None:
    specifications = require("oamb.contracts.specifications")
    attestation_values = dict(
        provider="hindsight",
        provider_project_id="oamb-hindsight-1",
        provider_profile_id="hindsight-rest-v0.9.2",
        transport_profile=specifications.TransportProfile.REST,
        release_version="0.9.2",
        source_revision="commit-1",
        build_artifact_sha256=OTHER_HASH,
        redacted_configuration_sha256=HASH,
        redacted_endpoint_fingerprint=OTHER_HASH,
        auth_configuration_sha256=HASH,
        storage_configuration_sha256=OTHER_HASH,
        model_role_binding_ids=("embedding-binding",),
        native_reranking_status="disabled",
        liveness_status=specifications.ProviderGateStatus.PASS,
        storage_configuration_status=specifications.ProviderGateStatus.PASS,
        runtime_identity_status=specifications.ProviderGateStatus.PASS,
        model_readiness_status=specifications.ProviderGateStatus.NOT_RUN,
        memory_conformance_status=specifications.ProviderGateStatus.NOT_RUN,
        raw_proof_refs=(HASH,),
    )
    attestation = specifications.ProviderRuntimeProfileAttestation(
        attestation_hash=specifications.provider_runtime_profile_attestation_hash(
            attestation_values
        ),
        **attestation_values,
    )
    approval_values = dict(
        approval_id="approval-1",
        operation_kind="model_readiness",
        scope_kind=specifications.BudgetScopeKindV2.MODEL_READINESS,
        scope_id="readiness-occurrence",
        runtime_binding_hash=None,
        provider_runtime_profile_attestation_hash=attestation.attestation_hash,
        role_binding_ids=("embedding-binding",),
        budget_hash=HASH,
        approved_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        unmetered_cost_acknowledged=True,
        stop_condition_ids=("identity_drift",),
    )
    approval = specifications.ExternalCallApprovalRecord(
        approval_hash=specifications.external_call_approval_hash(approval_values),
        **approval_values,
    )

    assert approval.provider_runtime_profile_attestation_hash == attestation.attestation_hash

    promoted_attestation = attestation.model_dump()
    promoted_attestation["model_readiness_status"] = specifications.ProviderGateStatus.PASS
    promoted_attestation["attestation_hash"] = (
        specifications.provider_runtime_profile_attestation_hash(promoted_attestation)
    )
    with pytest.raises(ValidationError, match="pre-readiness"):
        specifications.ProviderRuntimeProfileAttestation(**promoted_attestation)
    with pytest.raises(ValidationError, match="attestation hash"):
        specifications.ProviderRuntimeProfileAttestation(
            **(attestation.model_dump() | {"release_version": "9.9.9"})
        )
    missing_attestation_approval = approval.model_dump()
    missing_attestation_approval["provider_runtime_profile_attestation_hash"] = None
    missing_attestation_approval["approval_hash"] = specifications.external_call_approval_hash(
        missing_attestation_approval
    )
    with pytest.raises(ValidationError, match="attestation"):
        specifications.ExternalCallApprovalRecord(**missing_attestation_approval)
    with pytest.raises(ValidationError, match="approval hash"):
        specifications.ExternalCallApprovalRecord(
            **(approval.model_dump() | {"scope_id": "different-occurrence"})
        )


def test_unknown_outcome_receipt_cannot_name_raw_evidence() -> None:
    evidence = require("oamb.contracts.evidence")

    with pytest.raises(ValidationError, match="cannot contain raw evidence"):
        evidence.AttemptReceiptRecord(
            attempt_id=HASH,
            receipt_kind=evidence.AttemptReceiptKind.UNKNOWN_OUTCOME,
            raw_response_ref=None,
            raw_error_ref=OTHER_HASH,
            dispatch_started_at=NOW,
            receipt_observed_at=NOW,
            provider_request_wall_seconds=Decimal("0"),
        )


def test_v2_attempt_and_usage_add_only_model_readiness_parent_and_stage() -> None:
    accounting = require("oamb.contracts.accounting")
    evidence = require("oamb.contracts.evidence")
    states = require("oamb.contracts.states")

    attempt = evidence.AttemptRecordV2(
        attempt_id=HASH,
        parent_kind="model_readiness",
        parent_id="readiness-occurrence",
        stage="model_readiness",
        ordinal=1,
        request_fingerprint=OTHER_HASH,
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
        outcome=states.AttemptOutcome.SUCCEEDED,
        retry_of_attempt_id=None,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=HASH,
        raw_error_ref=None,
        index_contribution=states.IndexContribution.NOT_APPLICABLE,
        superseded_by_attempt_id=None,
    )
    usage = accounting.TokenUsageRecordV2(
        usage_record_id=OTHER_HASH,
        attempt_id=attempt.attempt_id,
        parent_kind="model_readiness",
        parent_id=attempt.parent_id,
        stage=accounting.TokenStageV2.MODEL_READINESS,
        operation_kind="embedding_readiness",
        token_domain=accounting.TokenDomain.EXTERNAL_LLM,
        measurement_source=accounting.TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=None,
        visible_output_tokens=None,
        supplier_reported_total_tokens=None,
        context_view_tokens=None,
        proof_status=accounting.ProofStatus.UNAVAILABLE,
        reason="provider_does_not_expose_portable_billing_receipt",
        raw_response_ref=HASH,
    )

    assert usage.parent_id == attempt.parent_id
    with pytest.raises(ValidationError):
        evidence.AttemptRecord(**attempt.model_dump(exclude={"schema_version"}))
    with pytest.raises(ValidationError):
        accounting.TokenUsageRecord(**usage.model_dump(exclude={"schema_version"}))


def test_transaction_records_enforce_append_only_chain_shapes() -> None:
    evidence = require("oamb.contracts.evidence")
    specifications = require("oamb.contracts.specifications")

    lease = evidence.RunLeaseRecord(
        lease_record_hash=HASH,
        run_id="run-1",
        provider_project_id="provider-project-1",
        provider_profile_id="profile-1",
        lease_epoch=1,
        owner_id="owner-1",
        host_fingerprint=OTHER_HASH,
        process_id=123,
        predecessor_lease_record_hash=None,
        acquired_at=NOW,
    )
    heartbeat = evidence.RunLeaseHeartbeatRecord(
        heartbeat_record_hash=OTHER_HASH,
        lease_record_hash=lease.lease_record_hash,
        heartbeat_sequence=1,
        predecessor_heartbeat_hash=None,
        observed_at=NOW + timedelta(seconds=1),
    )
    claim = evidence.OccurrenceClaimRecord(
        claim_id=HASH,
        occurrence_id="readiness-occurrence",
        lease_record_hash=lease.lease_record_hash,
        lease_epoch=lease.lease_epoch,
        owner_id=lease.owner_id,
        stage="model_readiness",
        request_fingerprint=OTHER_HASH,
        reconciliation_capability="none",
        claimed_at=NOW,
    )
    reservation = evidence.BudgetReservationRecord(
        reservation_id=OTHER_HASH,
        budget_id="readiness-budget",
        scope_kind=specifications.BudgetScopeKindV2.MODEL_READINESS,
        scope_id="readiness-occurrence",
        role_binding_id="embedding-binding",
        attempt_id=HASH,
        reserved_attempts=1,
        reserved_input_tokens=0,
        reserved_output_tokens=0,
        reserved_dispatch_wall_seconds=Decimal("30"),
        reserved_cost=None,
        currency=None,
        reserved_resource_ceilings=(resource_ceiling(specifications),),
        reserved_provider_units=Decimal("1"),
        reserved_at=NOW,
    )
    intent = evidence.AttemptIntentRecord(
        attempt_id=HASH,
        claim_id=claim.claim_id,
        reservation_id=reservation.reservation_id,
        parent_kind="model_readiness",
        parent_id="readiness-occurrence",
        role_binding_id="embedding-binding",
        stage="model_readiness",
        request_fingerprint=claim.request_fingerprint,
        reconciliation_capability="none",
        idempotency_key_hash=None,
        sealed_at=NOW,
    )
    receipt = evidence.AttemptReceiptRecord(
        attempt_id=intent.attempt_id,
        receipt_kind=evidence.AttemptReceiptKind.RESPONSE,
        raw_response_ref=HASH,
        raw_error_ref=None,
        dispatch_started_at=NOW,
        receipt_observed_at=NOW + timedelta(seconds=1),
        provider_request_wall_seconds=Decimal("1"),
    )
    occurrence = evidence.ModelReadinessOccurrenceRecord(
        occurrence_id="readiness-occurrence",
        provider="hindsight",
        provider_project_id="provider-project-1",
        provider_profile_id="profile-1",
        provider_runtime_profile_attestation_hash=HASH,
        approval_id="approval-1",
        budget_id="readiness-budget",
        state=evidence.ModelReadinessOccurrenceState.SEALED,
        role_binding_ids=("embedding-binding",),
        attempt_ids=(intent.attempt_id,),
        usage_record_ids=(OTHER_HASH,),
        resource_record_ids=(),
        cost_record_ids=(),
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
    )

    assert heartbeat.heartbeat_sequence == 1
    assert receipt.raw_response_ref == HASH
    assert occurrence.state == evidence.ModelReadinessOccurrenceState.SEALED

    with pytest.raises(ValidationError, match="predecessor"):
        evidence.RunLeaseRecord(**(lease.model_dump() | {"lease_epoch": 2}))
    with pytest.raises(ValidationError, match="response receipt"):
        evidence.AttemptReceiptRecord(**(receipt.model_dump() | {"raw_response_ref": None}))


def test_checkpoint_provider_manifest_and_derivation_envelope_are_strict() -> None:
    evidence = require("oamb.contracts.evidence")
    specifications = require("oamb.contracts.specifications")
    entry = evidence.CapsuleManifestEntry(
        record_kind="attempt",
        record_id="attempt-1",
        relative_path="source/attempts/attempt-1.json",
        sha256=HASH,
    )
    checkpoint = evidence.CheckpointManifest(
        checkpoint_manifest_hash=HASH,
        run_id="run-1",
        lease_record_hash=OTHER_HASH,
        lease_epoch=1,
        sequence=1,
        predecessor_checkpoint_hash=None,
        source_entries=(entry,),
        created_at=NOW,
    )
    provider_manifest = evidence.ProviderServiceEvidenceManifest(
        manifest_hash=OTHER_HASH,
        provider_project_id="provider-project-1",
        provider_profile_id="profile-1",
        occurrence_id="readiness-occurrence",
        operation="model_readiness",
        source_entries=(entry,),
        created_at=NOW,
    )
    source = specifications.SourceEvidenceBinding(
        binding_id=HASH,
        source_kind=specifications.SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity="readiness-occurrence",
        source_root_hash=provider_manifest.manifest_hash,
        validation_result_hash=OTHER_HASH,
        source_schema_versions=("provider_service_evidence_manifest@1",),
    )
    derivation = specifications.DerivationSpec(
        derivation_kind="diagnostic",
        ordered_source_bindings=(source,),
        ordered_source_root_hash=HASH,
        transform_spec_hash=OTHER_HASH,
        report_spec_hash=None,
        reducer_and_renderer_input_hashes=(HASH,),
        derivation_input_hash=OTHER_HASH,
    )
    manifest = evidence.DerivationManifest(
        derivation_id=HASH,
        derivation_spec_hash=OTHER_HASH,
        evidence_validation_result_hash=HASH,
        export_validation_result_hash=OTHER_HASH,
        committed_entries=(entry,),
        committed_at=NOW,
    )

    assert checkpoint.source_entries == (entry,)
    assert derivation.ordered_source_bindings == (source,)
    assert manifest.committed_entries == (entry,)

    with pytest.raises(ValidationError, match="duplicate"):
        evidence.ProviderServiceEvidenceManifest(
            **(provider_manifest.model_dump() | {"source_entries": (entry, entry)})
        )
    with pytest.raises(ValidationError, match="predecessor"):
        evidence.CheckpointManifest(**(checkpoint.model_dump() | {"sequence": 2}))
    with pytest.raises(ValidationError, match="checksums_sha256"):
        evidence.DerivationManifest(**(manifest.model_dump() | {"checksums_sha256": HASH}))


def test_artifact_store_port_is_a_complete_runtime_dependency() -> None:
    ports = require("oamb.contracts.ports")

    class FakeArtifactStore:
        def seal_raw(self, request: Any) -> Any:
            return ports.RawReferenceHandle(sha256=request.sha256)

        def seal_source_record(self, request: Any) -> Any:
            return ports.ArtifactSealReceipt(request.record_id, request.canonical_sha256)

        def seal_checkpoint(self, request: Any) -> Any:
            return ports.ArtifactSealReceipt(request.record_id, request.canonical_sha256)

        def seal_source_manifest(self, request: Any) -> Any:
            return ports.ArtifactSealReceipt(request.record_id, request.canonical_sha256)

        def read_verified(self, request: Any) -> bytes:
            return b"{}"

    assert isinstance(FakeArtifactStore(), ports.ArtifactStorePort)


def test_t4_contract_versions_are_registered_explicitly() -> None:
    schema = require("oamb.contracts.schema")
    expected_keys = {
        ("model_role_binding", 2),
        ("memory_system_runtime_binding", 2),
        ("budget_spec", 2),
        ("attempt_record", 2),
        ("token_usage_record", 2),
        ("resource_budget_ceiling", 1),
        ("provider_budget_cap", 1),
        ("role_budget_ceiling", 1),
        ("external_call_approval_record", 1),
        ("provider_runtime_profile_attestation", 1),
        ("run_lease_record", 1),
        ("run_lease_heartbeat_record", 1),
        ("occurrence_claim_record", 1),
        ("model_readiness_occurrence_record", 1),
        ("provider_service_evidence_manifest", 1),
        ("budget_reservation_record", 1),
        ("attempt_intent_record", 1),
        ("attempt_receipt_record", 1),
        ("checkpoint_manifest", 1),
        ("recovery_decision_record", 1),
        ("close_error_record", 1),
        ("source_evidence_binding", 1),
        ("derivation_spec", 1),
        ("derivation_manifest", 1),
    }

    assert expected_keys <= set(schema.CONTRACT_REGISTRY)


def test_legacy_v1_external_contract_documents_remain_parseable() -> None:
    schema = require("oamb.contracts.schema")
    documents = (
        {
            "schema_name": "model_role_binding",
            "schema_version": 1,
            "binding_id": "answer-binding",
            "role": "answer",
            "owner": "harness",
            "endpoint_fingerprint": HASH,
            "configured_model": "fixture-model",
            "resolved_model": "fixture-model",
            "parameters_fingerprint": OTHER_HASH,
            "budget_role": "answer",
        },
        {
            "schema_name": "memory_system_runtime_binding",
            "schema_version": 1,
            "memory_system_id": "fake-memory",
            "runtime_binding_hash": HASH,
            "release_version": "fixture-1",
            "artifact_sha256": OTHER_HASH,
            "storage_engine": "fixture-store",
            "model_role_binding_ids": ["answer-binding"],
            "attestation_status": "attested",
        },
        {
            "schema_name": "budget_spec",
            "schema_version": 1,
            "budget_id": "fake-budget",
            "scope_kind": "run",
            "scope_id": "fake-run",
            "approval_id": None,
            "max_attempts": 0,
            "max_input_tokens": 0,
            "max_output_tokens": 0,
            "max_wall_seconds": "0",
            "max_cost": None,
            "currency": None,
        },
        {
            "schema_name": "attempt_record",
            "schema_version": 1,
            "attempt_id": HASH,
            "parent_kind": "case",
            "parent_id": "case-1",
            "stage": "answer",
            "ordinal": 1,
            "request_fingerprint": OTHER_HASH,
            "started_at": "2026-08-27T12:00:00Z",
            "ended_at": "2026-08-27T12:00:01Z",
            "outcome": "succeeded",
            "retry_of_attempt_id": None,
            "idempotency_key_hash": None,
            "reconciliation_capability": "none",
            "raw_response_ref": HASH,
            "raw_error_ref": None,
            "index_contribution": "not_applicable",
            "superseded_by_attempt_id": None,
        },
        {
            "schema_name": "token_usage_record",
            "schema_version": 1,
            "usage_record_id": OTHER_HASH,
            "attempt_id": HASH,
            "parent_kind": "case",
            "parent_id": "case-1",
            "stage": "answer",
            "operation_kind": "answer_completion",
            "token_domain": "external_llm",
            "measurement_source": "supplier_response",
            "input_tokens": None,
            "visible_output_tokens": None,
            "supplier_reported_total_tokens": None,
            "context_view_tokens": None,
            "proof_status": "unavailable",
            "reason": "supplier_usage_missing",
            "raw_response_ref": HASH,
        },
    )

    assert [item.schema_version for item in map(schema.parse_contract, documents)] == [1] * 5


def test_all_pre_t4_v1_schema_bytes_remain_unchanged() -> None:
    expected = {
        "attempt_record.v1.schema.json": "39ff94a93d5e34cd7f216b481f65ab993cabb40c9b1abcc7042436d7a38d82e9",
        "budget_spec.v1.schema.json": "d090f9830b766e6cae3daddb8142ab7350a6e726dc34228a31bd8cd4c05ae5c1",
        "capsule_manifest.v1.schema.json": "5b5b76410295e8b08c9b8a98caddef24ef16dab6753b9021d7ede6cef056553b",
        "capsule_manifest_entry.v1.schema.json": "819575c44fa58c1cf6b8ad46ed5fb26ab8739de4b562c794aa94e73f2d886868",
        "case_manifest.v1.schema.json": "de7c0ebf66de6557b1dd5e2864561bfd1ee843d432bdefa4078d886eeb4067e9",
        "case_manifest_entry.v1.schema.json": "3e058a25cd0c6c44c6050c66a43e68e1f48140e0fbfecfd4845e326e9a465c79",
        "case_record.v1.schema.json": "057adbe9a4b5f6e170de33afb31adf30e10c175a9333ec84dc11f7b2926c98c2",
        "cost_measurement_spec.v1.schema.json": "57e4a752ad4aae8693c325e4d06014aa5931a5e2dff26b23ba4818fd49a0313d",
        "cost_record.v1.schema.json": "1eebcfe4c7e9d2c42524f41e8b28030429a09d8e28b70324b534dbe459dcbc27",
        "dataset_file.v1.schema.json": "8a69235ec067acd72ab03b9f2ca99bc8b5f25366e1c274f26325f3890da32f1c",
        "dataset_manifest.v1.schema.json": "0f0a619eaf2eae2f1f6d6177afcdabc0b1ec39c874a5476875e3292ff3a0e185",
        "execution_environment_binding.v1.schema.json": "142e5047324f97e9dd893d1f9697f0e1b2d3915708f901b41575592a40631ce9",
        "ingestion_plan_manifest.v1.schema.json": "10af804b2181cb9bb648e8e0425e1b1728641d22c8b4e4b7f51ed3d9e7a2b2c9",
        "ingestion_plan_record.v1.schema.json": "920beed0ac49df899b763e6c00adb48dd72ab3c3daf36803ef15438a25164bdf",
        "interaction_spec.v1.schema.json": "533c2c22da2864e2119ee6949225ffcb58d49967352e8082d76a57291f412cc9",
        "logical_context_manifest_entry.v1.schema.json": "517fde256e807c08b0eff8461a26eee2663e0046f570d6ba542bf9773d37aa06",
        "logical_context_record.v1.schema.json": "ca781bbd59686ae0106cc3f3e9f72835a45a0324896623afad4cb2a78a23723a",
        "measurement_dimension_spec.v1.schema.json": "33837288ba832487d05a7985c644de89b51237430ec5efc2331dd3a4b8d75b82",
        "memory_system_runtime_binding.v1.schema.json": "437d1756ff9c7e28dbef29b73a52a34da705f4f7e0e0431728778cc7fea47746",
        "memory_system_spec.v1.schema.json": "a7006c43eca673c85652eb5effc00a3512adc7c21834afdcb549dc77568756e1",
        "model_role_binding.v1.schema.json": "fd46bbaacfba0dbb32612b9ca0fb3c67425525b73bdc125132d0d64688283a55",
        "origin_record.v1.schema.json": "9358b3707107f6be7edba33f2c863696e3ddcaa92ec6b501afb5bc54d80e1c6f",
        "price_snapshot.v1.schema.json": "5e5a3cff152b3ae81ab3f001cd682a99832926ac7da56351b8a4f2c5aae2b200",
        "protocol_spec.v1.schema.json": "bfff97c8b2fad85f9671ae84c159c9c665eb5fb7ed14aacb8364bf4d918e81bf",
        "raw_reference.v1.schema.json": "c88b010a0de1bec2d33e5195d33883a91f2169912d88b19bf450847974dd007d",
        "report_artifact_manifest.v1.schema.json": "38fe62b3426c9515e9e82b45641c4f7297e08e4d78a9fe77c379130430fa8845",
        "resource_usage_record.v1.schema.json": "9af1f13c45e4246705281db502f73a067c67020b6232d3c710a109fa5466ccee",
        "run_record.v1.schema.json": "0a98e2aa90d8bad3f936b37c32bb9f05101cafb163c81124f6ba087f367c8a2f",
        "run_report_model.v1.schema.json": "5388933b33e7c38cf9ae5892a3123d1aa48249b9703a7de711545b7b4cefbadd",
        "run_spec.v1.schema.json": "216e59e452ff7f411a0d935da935393651a879daf7cd1d0a98cfc660950b1e0e",
        "run_summary.v1.schema.json": "3da2f46fc284b766c8671094697b5594d528c8544951d7ef36fe27474f237e4f",
        "token_usage_record.v1.schema.json": "4f6b5fea666b252e88dbb8b693d6bde2f505848bf9e1617fa5063a418bfa9f9e",
        "validation_issue.v1.schema.json": "d4d42cb944270028ce849b7bce189ff2b6d2db22f0c874ccb74b499072ab75ac",
        "validation_profile.v1.schema.json": "1450ab8b15cc92be8f0ed0f6c5beeeb1dd5838f9a2fca79a89e37dca1b62a245",
        "validation_result.v1.schema.json": "52ccfb2fc44951d8337e5ea954d721fbb8e012c920d81245b64a3ee03d9092e1",
        "validation_rule_requirement.v1.schema.json": "1a9d19dc309dbab9497f6bc1e94801d8d8155978de21d81a68928f036be52907",
        "workload_spec.v1.schema.json": "d8da526d00b37374b41b0bbad5799c4bf45f842234176da43a9fbc1c69d7cbe8",
    }
    schema_directory = Path(__file__).resolve().parents[2] / "schemas"
    actual = {
        name: hashlib.sha256((schema_directory / name).read_bytes()).hexdigest()
        for name in expected
    }

    assert actual == expected
