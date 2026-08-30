from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

from oamb.contracts.ids import canonical_sha256

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


def _external_legacy_fixtures() -> tuple[tuple[dict[str, Any], type[Any]], ...]:
    specifications = require("oamb.contracts.specifications")
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

    expected_types = (
        specifications.ModelRoleBinding,
        specifications.MemorySystemRuntimeBinding,
        specifications.BudgetSpec,
        require("oamb.contracts.evidence").AttemptRecord,
        require("oamb.contracts.accounting").TokenUsageRecord,
    )

    return tuple(zip(documents, expected_types, strict=True))


def test_every_remaining_legacy_contract_version_parses_from_strict_json_bytes() -> None:
    schema = require("oamb.contracts.schema")
    evidence = require("oamb.contracts.evidence")
    reporting = require("oamb.contracts.reporting")
    specifications = require("oamb.contracts.specifications")
    accounting = require("oamb.contracts.accounting")
    states = require("oamb.contracts.states")
    source_binding = {
        "schema_name": "source_evidence_binding",
        "schema_version": 1,
        "binding_id": HASH,
        "source_kind": "run",
        "source_identity": "legacy-run",
        "source_root_hash": OTHER_HASH,
        "validation_result_hash": HASH,
        "source_schema_versions": ["capsule_manifest@1"],
    }
    run_summary_v1 = {
        "schema_name": "run_summary",
        "schema_version": 1,
        "run_id": "legacy-run",
        "intended_logical_contexts": 0,
        "intended_ingestion_plans": 0,
        "ready_ingestion_plans": 0,
        "intended_cases": 0,
        "terminal_cases": 0,
        "completed_cases": 0,
        "errored_cases": 0,
        "unsupported_cases": 0,
        "cancelled_cases": 0,
        "budget_exceeded_cases": 0,
        "billing_complete": False,
        "cost_complete": False,
    }
    run_summary_v2 = {
        **run_summary_v1,
        "schema_version": 2,
        "parsed_cases": 0,
        "evaluated_cases": 0,
        "judged_cases": 0,
        "unjudged_cases": 0,
    }
    legacy_budget = specifications.BudgetSpecV2(
        budget_id="legacy-budget-v2",
        scope_kind=specifications.BudgetScopeKindV2.RUN,
        scope_id="legacy-run",
        approval_id="legacy-approval-v1",
        max_attempts=1,
        max_input_tokens=0,
        max_output_tokens=0,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=None,
        currency=None,
        resource_ceilings=(resource_ceiling(specifications),),
        role_ceilings=(role_ceiling(specifications),),
        stop_condition_ids=("budget_exhausted",),
    )
    legacy_approval_values = dict(
        approval_id="legacy-approval-v1",
        operation_kind="benchmark_run",
        scope_kind=specifications.BudgetScopeKindV2.RUN,
        scope_id="legacy-run",
        runtime_binding_hash=HASH,
        provider_runtime_profile_attestation_hash=None,
        role_binding_ids=("embedding-binding",),
        budget_hash=OTHER_HASH,
        approved_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        unmetered_cost_acknowledged=True,
        stop_condition_ids=("budget_exhausted",),
    )
    legacy_approval = specifications.ExternalCallApprovalRecord(
        approval_hash=specifications.external_call_approval_hash(legacy_approval_values),
        **legacy_approval_values,
    )
    legacy_reservation = evidence.BudgetReservationRecord(
        reservation_id=HASH,
        budget_id=legacy_budget.budget_id,
        scope_kind=specifications.BudgetScopeKindV2.RUN,
        scope_id="legacy-run",
        role_binding_id="embedding-binding",
        attempt_id=OTHER_HASH,
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
    legacy_intent = evidence.AttemptIntentRecord(
        attempt_id=OTHER_HASH,
        claim_id=HASH,
        reservation_id=legacy_reservation.reservation_id,
        parent_kind="case",
        parent_id="legacy-case",
        role_binding_id="embedding-binding",
        stage="answer",
        request_fingerprint=HASH,
        reconciliation_capability="none",
        idempotency_key_hash=None,
        sealed_at=NOW,
    )
    legacy_attempt_v2 = evidence.AttemptRecordV2(
        attempt_id=OTHER_HASH,
        parent_kind="case",
        parent_id="legacy-case",
        stage="answer",
        ordinal=1,
        request_fingerprint=HASH,
        started_at=NOW,
        ended_at=NOW,
        outcome=states.AttemptOutcome.SUCCEEDED,
        retry_of_attempt_id=None,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=HASH,
        raw_error_ref=None,
        index_contribution=states.IndexContribution.NOT_APPLICABLE,
        superseded_by_attempt_id=None,
    )
    legacy_usage_v3 = accounting.TokenUsageRecordV3(
        usage_record_id=OTHER_HASH,
        attempt_id=HASH,
        parent_kind="case",
        parent_id="legacy-case",
        stage=accounting.TokenStageV2.ANSWER,
        operation_kind="answer_completion",
        token_domain=accounting.TokenDomain.EXTERNAL_LLM,
        measurement_source=accounting.TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=None,
        visible_output_tokens=None,
        supplier_reported_total_tokens=None,
        context_view_tokens=None,
        cached_input_tokens=None,
        reasoning_tokens=None,
        configured_model="legacy-model",
        runtime_model="legacy-model",
        meter_schema_id="legacy-meter-v1",
        raw_field_paths=(),
        covered_dimensions=(),
        unavailable_dimensions=(
            "input_tokens",
            "visible_output_tokens",
            "supplier_reported_total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ),
        not_applicable_dimensions=(),
        inclusion_relationships=(),
        token_measurement_complete=False,
        billing_complete=False,
        proof_status=accounting.ProofStatus.UNAVAILABLE,
        reason="legacy usage unavailable",
        raw_response_ref=HASH,
    )
    legacy_report_spec_fields = dict(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=1024,
        preview_total_bytes=4096,
        display_field_ids=("summary",),
        renderer_hash=HASH,
        asset_hashes=(OTHER_HASH,),
        browser_contract_hash=HASH,
        performance_contract_hash=OTHER_HASH,
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )
    legacy_report_spec = specifications.ReportSpec(
        report_spec_id=specifications.report_spec_id(**legacy_report_spec_fields),
        **legacy_report_spec_fields,
    )
    legacy_binding_fields = dict(
        spec_kind="benchmark_report",
        spec_schema_name="report_spec",
        spec_schema_version=1,
        spec_id=legacy_report_spec.report_spec_id,
        spec_hash=canonical_sha256(legacy_report_spec),
    )
    legacy_binding = specifications.ReportIdentitySpecBinding(
        binding_id=specifications.report_identity_spec_binding_id(**legacy_binding_fields),
        **legacy_binding_fields,
    )
    legacy_source = specifications.SourceEvidenceBinding.model_validate_json(
        json.dumps(source_binding)
    )
    ordered_source_root_hash = canonical_sha256(
        ["oamb-ordered-source-roots-v1", (legacy_source.source_root_hash,)]
    )
    legacy_derivation_fields = dict(
        derivation_kind="run_report",
        ordered_source_bindings=(legacy_source,),
        ordered_source_root_hash=ordered_source_root_hash,
        evidence_validation_result_hash=HASH,
        transform_spec_hash=OTHER_HASH,
        report_identity_spec_binding=legacy_binding,
        reducer_and_renderer_input_hashes=(HASH,),
    )
    legacy_derivation = specifications.DerivationSpecV2(
        derivation_input_hash=specifications.derivation_spec_v2_input_hash(
            **legacy_derivation_fields
        ),
        **legacy_derivation_fields,
    )
    legacy_artifact_fields = dict(
        report_id=HASH,
        report_kind="run",
        report_identity_spec_binding=legacy_binding,
        ordered_source_bindings=(legacy_source,),
        ordered_evidence_validation_hashes=(HASH,),
        report_model_hash=OTHER_HASH,
        renderer_hash=HASH,
        asset_hashes=(OTHER_HASH,),
        browser_contract_hash=HASH,
        performance_contract_hash=OTHER_HASH,
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
        audience="public",
        schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
        limitations=("legacy fixture",),
    )
    legacy_artifact = reporting.ReportArtifactManifestV2(
        artifact_manifest_id=reporting.report_artifact_manifest_v2_id(**legacy_artifact_fields),
        **legacy_artifact_fields,
    )
    from tests.unit.test_t8_human_review import _ai_record

    legacy_ai_record = _ai_record()
    signature_fields = dict(
        key_binding_id=HASH,
        trusted_key_fingerprint="legacy-key",
        public_key_sha256=OTHER_HASH,
        signed_payload_sha256=HASH,
        signature_sha256=OTHER_HASH,
        verified_at=NOW,
    )
    legacy_signature = reporting.SignatureVerificationRecord(
        verification_id=reporting.signature_verification_record_id(**signature_fields),
        **signature_fields,
    )
    legacy_human_fields = dict(
        review_bundle_hash=legacy_ai_record.review_bundle_hash,
        ai_review_record_hash=legacy_ai_record.ai_review_record_id,
        decision_hash=HASH,
        signature_verification=legacy_signature,
        status="pass",
        finding_codes=(),
        evidence_references=(),
        reviewer_label="legacy-reviewer",
        decision_nonce="legacy-nonce",
        trusted_key_fingerprint="legacy-key",
        created_at=NOW,
    )
    legacy_human = reporting.HumanQualityReviewRecordV1(
        human_review_record_id=reporting.human_quality_review_record_v1_id(**legacy_human_fields),
        **legacy_human_fields,
    )
    legacy_history = (legacy_ai_record.ai_review_record_id,)
    legacy_history_root = canonical_sha256(
        ["oamb-review-history-v1", legacy_history, legacy_human.human_review_record_id]
    )
    legacy_gate_fields = dict(
        phase_id="legacy-phase",
        review_bundle_hash=legacy_ai_record.review_bundle_hash,
        review_history_root_hash=legacy_history_root,
        ordered_ai_review_record_hashes=legacy_history,
        canonical_ai_review_record_hash=legacy_ai_record.ai_review_record_id,
        human_review_record_hash=legacy_human.human_review_record_id,
        passed_by_ai=True,
        passed_by_human=True,
        ai_record=legacy_ai_record,
        human_record=legacy_human,
    )
    legacy_gate = reporting.EvaluationPhaseGateV1(
        gate_id=reporting.evaluation_phase_gate_v1_id(
            **{
                key: value
                for key, value in legacy_gate_fields.items()
                if key not in {"ai_record", "human_record"}
            }
        ),
        **legacy_gate_fields,
    )
    fixtures = (
        *_external_legacy_fixtures(),
        (legacy_approval.model_dump(mode="json"), specifications.ExternalCallApprovalRecord),
        (legacy_budget.model_dump(mode="json"), specifications.BudgetSpecV2),
        (legacy_reservation.model_dump(mode="json"), evidence.BudgetReservationRecord),
        (legacy_intent.model_dump(mode="json"), evidence.AttemptIntentRecord),
        (legacy_attempt_v2.model_dump(mode="json"), evidence.AttemptRecordV2),
        (legacy_usage_v3.model_dump(mode="json"), accounting.TokenUsageRecordV3),
        (legacy_report_spec.model_dump(mode="json"), specifications.ReportSpec),
        (
            legacy_binding.model_dump(mode="json"),
            specifications.ReportIdentitySpecBinding,
        ),
        (legacy_derivation.model_dump(mode="json"), specifications.DerivationSpecV2),
        (
            legacy_artifact.model_dump(mode="json"),
            reporting.ReportArtifactManifestV2,
        ),
        (
            legacy_human.model_dump(mode="json"),
            reporting.HumanQualityReviewRecordV1,
        ),
        (
            legacy_gate.model_dump(mode="json"),
            reporting.EvaluationPhaseGateV1,
        ),
        (
            {
                "schema_name": "case_record",
                "schema_version": 1,
                "case_occurrence_id": HASH,
                "run_id": "legacy-run",
                "ingestion_occurrence_id": OTHER_HASH,
                "case_manifest_entry_id": HASH,
                "state": "error",
                "retrieval_raw_ref": None,
                "prompt_sha256": None,
                "answer_raw_ref": None,
                "evaluation_raw_ref": None,
                "attempt_ids": [],
                "error_stage": "answer",
            },
            evidence.CaseRecord,
        ),
        (
            {
                "schema_name": "case_record",
                "schema_version": 2,
                "case_occurrence_id": HASH,
                "run_id": "legacy-run",
                "ingestion_occurrence_id": OTHER_HASH,
                "case_manifest_entry_id": HASH,
                "state": "error",
                "retrieval_raw_ref": None,
                "prompt_sha256": None,
                "answer_raw_ref": None,
                "parsed_answer_sha256": None,
                "evaluation_raw_ref": None,
                "evaluation_disposition": "not_run",
                "attempt_ids": [],
                "error_stage": "answer",
            },
            evidence.CaseRecordV2,
        ),
        (
            {
                "schema_name": "derivation_spec",
                "schema_version": 1,
                "derivation_kind": "legacy-report",
                "ordered_source_bindings": [source_binding],
                "ordered_source_root_hash": HASH,
                "transform_spec_hash": OTHER_HASH,
                "report_spec_hash": None,
                "reducer_and_renderer_input_hashes": [HASH],
                "derivation_input_hash": OTHER_HASH,
            },
            specifications.DerivationSpec,
        ),
        (
            {
                "schema_name": "ingestion_plan_record",
                "schema_version": 1,
                "ingestion_occurrence_id": HASH,
                "run_id": "legacy-run",
                "memory_system_id": "legacy-memory",
                "ingestion_plan_id": OTHER_HASH,
                "ordered_member_context_manifest_entry_ids": [HASH],
                "ordered_case_occurrence_ids": [OTHER_HASH],
                "state": "pending",
                "intended_source_count": 1,
                "accepted_source_count": 0,
                "failed_source_count": 0,
                "readiness_evidence_refs": [],
                "attempt_ids": [],
                "usage_record_ids": [],
                "resource_record_ids": [],
                "cost_record_ids": [],
            },
            evidence.IngestionPlanRecord,
        ),
        (
            {
                "schema_name": "ingestion_plan_record",
                "schema_version": 2,
                "ingestion_occurrence_id": HASH,
                "run_id": "legacy-run",
                "memory_system_id": "legacy-memory",
                "adapter_profile_id": "legacy-memory-v1",
                "runtime_binding_hash": OTHER_HASH,
                "ingestion_plan_id": OTHER_HASH,
                "ordered_member_context_manifest_entry_ids": [HASH],
                "ordered_case_occurrence_ids": [OTHER_HASH],
                "state": "pending",
                "scope_id": None,
                "scope_raw_refs": [],
                "ordered_source_unit_ids": [HASH],
                "ordered_dispatch_attempt_ids": [HASH],
                "ordered_dispatch_source_unit_ids": [[HASH]],
                "accepted_source_unit_ids": [],
                "rejected_source_unit_ids": [],
                "readiness_evidence_refs": [],
                "inventory_raw_ref": None,
                "projected_source_unit_ids": [],
                "projection_raw_refs": [],
                "protected_state_sha256": None,
                "attempt_ids": [HASH],
                "usage_record_ids": [],
                "resource_record_ids": [],
                "cost_record_ids": [],
            },
            evidence.IngestionPlanRecordV2,
        ),
        (
            {
                "schema_name": "report_artifact_manifest",
                "schema_version": 1,
                "report_id": HASH,
                "ordered_source_root_hashes": [OTHER_HASH],
                "evidence_validation_hash": HASH,
                "report_model_hash": OTHER_HASH,
                "renderer_hash": HASH,
                "asset_hashes": [OTHER_HASH],
                "audience": "public",
                "limitations": ["legacy fixture"],
            },
            reporting.ReportArtifactManifest,
        ),
        (
            {
                "schema_name": "phase_review_occurrence_record",
                "schema_version": 1,
                "phase_review_occurrence_id": evidence.phase_review_occurrence_id(
                    phase_id="legacy-phase",
                    review_bundle_hash=HASH,
                    reviewer_role_binding_hash=OTHER_HASH,
                    ordinal=1,
                ),
                "phase_id": "legacy-phase",
                "review_bundle_hash": HASH,
                "reviewer_role_binding_hash": OTHER_HASH,
                "ordinal": 1,
                "approval_record_id": HASH,
                "budget_id": "legacy-phase-budget",
                "state": "planned",
                "started_at": None,
                "ended_at": None,
            },
            evidence.PhaseReviewOccurrenceRecord,
        ),
        (run_summary_v1, reporting.RunSummary),
        (
            {
                "schema_name": "run_report_model",
                "schema_version": 1,
                "report_id": HASH,
                "source_manifest_hash": OTHER_HASH,
                "evidence_validation_hash": HASH,
                "summary": run_summary_v1,
                "limitations": ["legacy fixture"],
            },
            reporting.RunReportModel,
        ),
        (
            {
                "schema_name": "run_report_model",
                "schema_version": 2,
                "report_id": HASH,
                "source_manifest_hash": OTHER_HASH,
                "evidence_validation_hash": HASH,
                "summary": run_summary_v2,
                "limitations": ["legacy fixture"],
            },
            reporting.RunReportModelV2,
        ),
        (
            {
                "schema_name": "token_usage_record",
                "schema_version": 2,
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
                "raw_response_ref": None,
            },
            accounting.TokenUsageRecordV2,
        ),
    )

    covered_versions = {
        (document["schema_name"], document["schema_version"]) for document, _ in fixtures
    }
    latest_versions: dict[str, int] = {}
    for name, version in schema.CONTRACT_REGISTRY:
        latest_versions[name] = max(version, latest_versions.get(name, 0))
    expected_legacy_versions = {
        (name, version)
        for name, version in schema.CONTRACT_REGISTRY
        if version < latest_versions[name]
    }
    assert covered_versions == expected_legacy_versions

    for document, expected_type in fixtures:
        encoded = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode()
        parsed = schema.parse_contract(json.loads(encoded))
        assert type(parsed) is expected_type
        with pytest.raises(ValidationError, match="extra_forbidden"):
            schema.parse_contract(json.loads(encoded) | {"unknown_field": True})
