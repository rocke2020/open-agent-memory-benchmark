from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    BindingKind,
    BudgetScopeKindV2,
    BudgetSpecV2,
    ExecutionOwner,
    ExternalCallApprovalRecord,
    ModelRole,
    ModelRoleBindingV2,
    ProviderBudgetCap,
    ResourceBudgetCeiling,
    RoleBindingStatus,
    RoleBudgetCeiling,
    RunPreflightRecord,
    RunSpec,
    SourceEvidenceBinding,
    SourceEvidenceKind,
    external_call_approval_hash,
    run_preflight_record_hash,
)
from oamb.runtime.native_run import NativeRunControl

NOW = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)


def _sha(label: str) -> str:
    return canonical_sha256([label])


def _source(identity: str) -> SourceEvidenceBinding:
    return SourceEvidenceBinding(
        binding_id=_sha(f"binding:{identity}"),
        source_kind=SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity=identity,
        source_root_hash=_sha(f"root:{identity}"),
        validation_result_hash=_sha(f"validation:{identity}"),
        source_schema_versions=("provider_service_evidence_manifest@1",),
    )


def _control(
    *,
    run_id: str = "live-run",
    dataset_manifest_hash: str | None = None,
    case_manifest_hash: str | None = None,
    workload_id: str = "fixture-workload-v1",
    memory_system_id: str = "fixture-memory-v1",
    runtime_binding_hash: str | None = None,
    adapter_profile_id: str = "fixture-rest-v1",
    answer_role_binding_id: str = "answer-binding",
) -> NativeRunControl:
    role = ModelRoleBindingV2(
        binding_id=answer_role_binding_id,
        role=ModelRole.ANSWER,
        role_status=RoleBindingStatus.SELECTED,
        execution_owner=ExecutionOwner.HARNESS,
        binding_kind=BindingKind.MODEL_CLIENT,
        provider="deepseek",
        endpoint_reference="DEEPSEEK_BASE_URL",
        credential_variable_name="DEEPSEEK_API_KEY",
        configured_model="deepseek-chat",
        resolved_model="deepseek-chat",
        parameters_fingerprint=_sha("parameters"),
        retry_policy_id="no-retry-v1",
        configuration_fingerprint=_sha("role-configuration"),
        redacted_endpoint_fingerprint=_sha("endpoint"),
    )
    budget = BudgetSpecV2(
        budget_id="run-budget",
        scope_kind=BudgetScopeKindV2.RUN,
        scope_id=run_id,
        approval_id="run-approval",
        max_attempts=10,
        max_input_tokens=100_000,
        max_output_tokens=10_000,
        max_dispatch_wall_seconds=Decimal("600"),
        max_cost=None,
        currency=None,
        resource_ceilings=(),
        role_ceilings=(
            RoleBudgetCeiling(
                role_binding_id=role.binding_id,
                max_attempts=10,
                max_input_tokens=100_000,
                max_output_tokens=10_000,
                max_dispatch_wall_seconds=Decimal("600"),
                max_cost=None,
                currency=None,
                price_snapshot_id=None,
                resource_ceilings=(
                    ResourceBudgetCeiling(
                        dimension_id="provider_request_wall_seconds_v1",
                        maximum=Decimal("600"),
                        unit="seconds",
                    ),
                ),
                provider_budget_cap=ProviderBudgetCap(
                    provider="deepseek",
                    operation_kind="chat_completion",
                    billing_unit="request",
                    maximum_accepted_units=Decimal("10"),
                ),
            ),
        ),
        stop_condition_ids=("identity_drift", "budget_exhausted"),
    )
    run_spec = RunSpec(
        run_id=run_id,
        protocol_id="oamb-native-v1",
        dataset_manifest_hash=dataset_manifest_hash or _sha("dataset"),
        case_manifest_hash=case_manifest_hash or _sha("subset"),
        workload_id=workload_id,
        memory_system_id=memory_system_id,
        runtime_binding_hash=runtime_binding_hash or _sha("runtime"),
        environment_hash=_sha("environment"),
        model_role_binding_ids=(role.binding_id,),
        budget_id=budget.budget_id,
        code_revision="fixture-revision",
        normalizer_fingerprint=_sha("normalizer"),
    )
    approval_fields = dict(
        approval_id=budget.approval_id,
        operation_kind="benchmark_run",
        scope_kind=BudgetScopeKindV2.RUN,
        scope_id=run_spec.run_id,
        runtime_binding_hash=run_spec.runtime_binding_hash,
        provider_runtime_profile_attestation_hash=None,
        role_binding_ids=(role.binding_id,),
        budget_hash=canonical_sha256(budget),
        approved_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        unmetered_cost_acknowledged=True,
        stop_condition_ids=budget.stop_condition_ids,
    )
    approval = ExternalCallApprovalRecord.model_validate(
        {
            "approval_hash": external_call_approval_hash(approval_fields),
            **approval_fields,
        }
    )
    preflight_fields = dict(
        run_id=run_spec.run_id,
        observed_at=NOW + timedelta(minutes=1),
        resolved_plan_hash=_sha("resolved-plan"),
        run_spec_hash=canonical_sha256(run_spec),
        dataset_manifest_hash=run_spec.dataset_manifest_hash,
        subset_manifest_hash=run_spec.case_manifest_hash,
        adapter_profile_id=adapter_profile_id,
        adapter_profile_hash=_sha("adapter-profile"),
        provider_project_id="oamb-providers-fixture1",
        provider_profile_id=adapter_profile_id,
        runtime_binding_hash=run_spec.runtime_binding_hash,
        provider_service_evidence=_source("fixture-model-readiness"),
        memory_conformance_evidence=_source("fixture-memory-conformance"),
        role_binding_ids=(role.binding_id,),
        dispatch_routes=(),
        approval_hash=approval.approval_hash,
        budget_hash=canonical_sha256(budget),
        redacted_endpoint_fingerprints=(role.redacted_endpoint_fingerprint,),
        credential_reference_fingerprints=(_sha("DEEPSEEK_API_KEY"),),
        artifact_repository_fingerprint=_sha("artifact-root"),
        artifact_durability_proof_hash=_sha("durability"),
    )
    preflight = RunPreflightRecord.model_validate(
        {
            "preflight_record_hash": run_preflight_record_hash(preflight_fields),
            **preflight_fields,
        }
    )
    return NativeRunControl(
        run_spec=run_spec,
        preflight_record=preflight,
        approval=approval,
        budget=budget,
        role_bindings=(role,),
        owner_id="operator-process-1",
        host_fingerprint=_sha("host"),
        process_id=1234,
        started_at=NOW + timedelta(minutes=2),
    )


def test_live_native_control_closes_all_durable_identities() -> None:
    control = _control()

    assert control.preflight_record.run_spec_hash == canonical_sha256(control.run_spec)
    assert control.preflight_record.budget_hash == canonical_sha256(control.budget)
    assert control.preflight_record.approval_hash == control.approval.approval_hash


def test_live_native_control_rejects_role_or_scope_drift() -> None:
    control = _control()

    with pytest.raises(ValueError, match="role inventory"):
        replace(control, role_bindings=())
