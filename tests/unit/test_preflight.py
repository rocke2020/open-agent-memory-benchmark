from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from oamb.contracts.specifications import (
    BudgetSpec,
    BudgetSpecV2,
    ExecutionEnvironmentBinding,
    ExecutionOwner,
    ExternalCallApprovalRecord,
    MemorySystemRuntimeBindingV2,
    ModelRole,
    ModelRoleBindingV2,
    ProviderRuntimeProfileAttestation,
    RoleBudgetCeiling,
)
from oamb.runtime.preflight import (
    AdapterProfileDescriptor,
    ArtifactDurabilityPreflight,
    ProviderGateClosure,
    RoleSlot,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def _zero_fake_budget() -> BudgetSpec:
    from oamb.contracts.specifications import BudgetScopeKind, BudgetSpec

    return BudgetSpec(
        budget_id="fake-zero-budget",
        scope_kind=BudgetScopeKind.RUN,
        scope_id="fake-run",
        approval_id=None,
        max_attempts=0,
        max_input_tokens=0,
        max_output_tokens=0,
        max_wall_seconds=Decimal("0"),
        max_cost=None,
        currency=None,
    )


def _durable_artifact_preflight() -> ArtifactDurabilityPreflight:
    return ArtifactDurabilityPreflight(
        artifact_root_fingerprint="a" * 64,
        available_bytes=1_000_000,
        lease_supported=True,
        file_fsync_supported=True,
        directory_fsync_supported=True,
        no_replace_supported=True,
    )


def _fake_role_slots() -> tuple[object, ...]:
    from oamb.runtime.preflight import ResolutionStatus, RoleSlot, RoleSlotName

    return tuple(
        RoleSlot(
            role=role,
            status=(
                ResolutionStatus.UNSELECTED
                if role == RoleSlotName.QUALITY_REVIEW
                else ResolutionStatus.NOT_APPLICABLE
            ),
            binding=None,
            credential_reference=None,
            evidence_reference="fake-local-role",
        )
        for role in RoleSlotName
    )


def _fake_profile() -> object:
    from oamb.runtime.preflight import AdapterProfileDescriptor, TransportKind

    return AdapterProfileDescriptor(
        profile_id="fake-v1",
        memory_system_id="fake-memory",
        transport_kind=TransportKind.FAKE,
        controlled_embedding=None,
        native_reranking_disabled=True,
        oamb_reranker_configured=False,
    )


def _not_applicable_gates() -> object:
    from oamb.runtime.preflight import GateStatus, ProviderGateClosure

    return ProviderGateClosure(
        liveness=GateStatus.NOT_APPLICABLE,
        storage_configuration=GateStatus.NOT_APPLICABLE,
        runtime_identity=GateStatus.NOT_APPLICABLE,
        model_readiness=GateStatus.NOT_APPLICABLE,
        memory_conformance=GateStatus.NOT_APPLICABLE,
    )


def test_v1_zero_external_fake_resolves_to_an_immutable_plan() -> None:
    from oamb.runtime.preflight import OperationKind, RunPreflightRequest, resolve_run_plan

    plan = resolve_run_plan(
        RunPreflightRequest(
            operation_kind=OperationKind.FAKE_RUN,
            budget_spec=_zero_fake_budget(),
            role_slots=_fake_role_slots(),
            adapter_profile=_fake_profile(),
            provider_gates=_not_applicable_gates(),
            artifact_durability=_durable_artifact_preflight(),
            runtime_binding=None,
            provider_runtime_attestation=None,
            approval=None,
            cost_measurement_spec=None,
            price_snapshot=None,
        )
    )

    assert plan.operation_kind == OperationKind.FAKE_RUN
    assert plan.schema_versions == (1,)
    assert len(plan.role_slots) == 5
    assert plan.role_slots[-1].status.value == "unselected"
    assert len(plan.plan_hash) == 64
    with pytest.raises(FrozenInstanceError):
        plan.plan_hash = "b" * 64  # type: ignore[misc]


def test_v1_budget_cannot_authorize_nonzero_external_allowance() -> None:
    from oamb.runtime.preflight import (
        OperationKind,
        PreflightRejected,
        RunPreflightRequest,
        resolve_run_plan,
    )

    nonzero = _zero_fake_budget().model_copy(update={"max_attempts": 1})

    with pytest.raises(PreflightRejected, match="version 1.*zero-external"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.FAKE_RUN,
                budget_spec=nonzero,
                role_slots=_fake_role_slots(),
                adapter_profile=_fake_profile(),
                provider_gates=_not_applicable_gates(),
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=None,
                provider_runtime_attestation=None,
                approval=None,
                cost_measurement_spec=None,
                price_snapshot=None,
            )
        )


def test_fake_plan_rejects_a_quality_review_credential_reference() -> None:
    from oamb.config.load import EnvironmentReference
    from oamb.runtime.preflight import (
        OperationKind,
        PreflightRejected,
        ResolutionStatus,
        RoleSlot,
        RoleSlotName,
        RunPreflightRequest,
        resolve_run_plan,
    )

    role_slots = _fake_role_slots()[:-1] + (
        RoleSlot(
            role=RoleSlotName.QUALITY_REVIEW,
            status=ResolutionStatus.UNSELECTED,
            binding=None,
            credential_reference=EnvironmentReference("OAMB_QUALITY_REVIEW_API_KEY"),
            evidence_reference="must-remain-unread",
        ),
    )

    with pytest.raises(PreflightRejected, match="unselected.*credential"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.FAKE_RUN,
                budget_spec=_zero_fake_budget(),
                role_slots=role_slots,
                adapter_profile=_fake_profile(),
                provider_gates=_not_applicable_gates(),
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=None,
                provider_runtime_attestation=None,
                approval=None,
                cost_measurement_spec=None,
                price_snapshot=None,
            )
        )


@pytest.mark.parametrize(
    ("model", "dimension", "reranking_disabled", "oamb_reranker", "message"),
    (
        ("wrong-model", 1024, True, False, "qwen3-embedding:0.6b"),
        ("qwen3-embedding:0.6b", 768, True, False, "1,024"),
        ("qwen3-embedding:0.6b", 1024, False, False, "native reranking"),
        ("qwen3-embedding:0.6b", 1024, True, True, "OAMB reranker"),
    ),
)
def test_external_profile_rejects_embedding_or_reranking_drift(
    model: str,
    dimension: int,
    reranking_disabled: bool,
    oamb_reranker: bool,
    message: str,
) -> None:
    from oamb.runtime.preflight import (
        AdapterProfileDescriptor,
        ControlledEmbeddingDescriptor,
        PreflightRejected,
        TransportKind,
        validate_adapter_profile,
    )

    profile = AdapterProfileDescriptor(
        profile_id="mem0-rest-v1",
        memory_system_id="mem0",
        transport_kind=TransportKind.REST_API,
        controlled_embedding=ControlledEmbeddingDescriptor(
            endpoint_fingerprint="b" * 64,
            model=model,
            artifact_fingerprint="c" * 64,
            dimension=dimension,
            input_adaptation_fingerprint="d" * 64,
        ),
        native_reranking_disabled=reranking_disabled,
        oamb_reranker_configured=oamb_reranker,
        provider_project_id="provider-project-1",
        release_version="2.0.19",
        source_revision="source-revision",
        build_artifact_sha256=HASH_B,
    )

    with pytest.raises(PreflightRejected, match=message):
        validate_adapter_profile(profile)


def test_rest_and_sdk_profiles_have_distinct_default_comparison_eligibility() -> None:
    from oamb.runtime.preflight import (
        AdapterProfileDescriptor,
        ControlledEmbeddingDescriptor,
        TransportKind,
        validate_adapter_profile,
    )

    embedding = ControlledEmbeddingDescriptor(
        endpoint_fingerprint="b" * 64,
        model="qwen3-embedding:0.6b",
        artifact_fingerprint="c" * 64,
        dimension=1024,
        input_adaptation_fingerprint="d" * 64,
    )
    rest = validate_adapter_profile(
        AdapterProfileDescriptor(
            profile_id="mem0-rest-v1",
            memory_system_id="mem0",
            transport_kind=TransportKind.REST_API,
            controlled_embedding=embedding,
            native_reranking_disabled=True,
            oamb_reranker_configured=False,
            provider_project_id="provider-project-1",
            release_version="2.0.19",
            source_revision="source-revision",
            build_artifact_sha256=HASH_B,
        )
    )
    sdk = validate_adapter_profile(
        AdapterProfileDescriptor(
            profile_id="mem0-sdk-v1",
            memory_system_id="mem0",
            transport_kind=TransportKind.PYTHON_SDK,
            controlled_embedding=embedding,
            native_reranking_disabled=True,
            oamb_reranker_configured=False,
            provider_project_id="provider-project-1",
            release_version="2.0.19",
            source_revision="source-revision",
            build_artifact_sha256=HASH_B,
        )
    )

    assert rest.default_comparison_eligible is True
    assert sdk.default_comparison_eligible is False


def _selected_role(
    role: ModelRole,
    binding_id: str,
    *,
    owner: ExecutionOwner,
    credential_variable_name: str | None = None,
) -> ModelRoleBindingV2:
    from oamb.contracts.specifications import BindingKind, ModelRoleBindingV2, RoleBindingStatus

    return ModelRoleBindingV2(
        binding_id=binding_id,
        role=role,
        role_status=RoleBindingStatus.SELECTED,
        execution_owner=owner,
        binding_kind=BindingKind.NATIVE,
        provider="fixture-provider",
        endpoint_reference="fixture-endpoint",
        credential_variable_name=credential_variable_name,
        configured_model=("qwen3-embedding:0.6b" if str(role) == "embedding" else "fixture-model"),
        resolved_model="fixture-model@sha256:resolved",
        parameters_fingerprint=HASH_A,
        retry_policy_id="no-retry-v1",
        configuration_fingerprint=HASH_B,
        redacted_endpoint_fingerprint=HASH_C,
    )


def _external_role_slots() -> tuple[RoleSlot, ...]:
    from oamb.config.load import EnvironmentReference
    from oamb.contracts.specifications import ExecutionOwner, ModelRole
    from oamb.runtime.preflight import ResolutionStatus, RoleSlot, RoleSlotName

    extraction = _selected_role(
        ModelRole.MEMORY_EXTRACTION,
        "extraction-binding",
        owner=ExecutionOwner.MEMORY_SYSTEM,
    )
    embedding = _selected_role(
        ModelRole.EMBEDDING,
        "embedding-binding",
        owner=ExecutionOwner.MEMORY_SYSTEM,
    )
    answer = _selected_role(
        ModelRole.ANSWER,
        "answer-binding",
        owner=ExecutionOwner.HARNESS,
        credential_variable_name="OAMB_ANSWER_API_KEY",
    )
    return (
        RoleSlot(
            role=RoleSlotName.EXTRACTION,
            status=ResolutionStatus.RESOLVED,
            binding=extraction,
            credential_reference=None,
            evidence_reference="extraction-proof",
        ),
        RoleSlot(
            role=RoleSlotName.EMBEDDING,
            status=ResolutionStatus.RESOLVED,
            binding=embedding,
            credential_reference=None,
            evidence_reference="embedding-proof",
        ),
        RoleSlot(
            role=RoleSlotName.ANSWER,
            status=ResolutionStatus.RESOLVED,
            binding=answer,
            credential_reference=EnvironmentReference("OAMB_ANSWER_API_KEY"),
            evidence_reference="answer-proof",
        ),
        RoleSlot(
            role=RoleSlotName.JUDGE,
            status=ResolutionStatus.NOT_APPLICABLE,
            binding=None,
            credential_reference=None,
            evidence_reference="deterministic-metric",
        ),
        RoleSlot(
            role=RoleSlotName.QUALITY_REVIEW,
            status=ResolutionStatus.UNSELECTED,
            binding=None,
            credential_reference=None,
            evidence_reference="post-export-only",
        ),
    )


def _role_ceiling(binding_id: str, *, provider: str = "fixture-provider") -> RoleBudgetCeiling:
    from oamb.contracts.specifications import (
        ProviderBudgetCap,
        ResourceBudgetCeiling,
        RoleBudgetCeiling,
    )

    resource = ResourceBudgetCeiling(
        dimension_id="provider_request_wall_seconds_v1",
        maximum=Decimal("30"),
        unit="seconds",
    )
    return RoleBudgetCeiling(
        role_binding_id=binding_id,
        max_attempts=1,
        max_input_tokens=100,
        max_output_tokens=100,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=None,
        currency=None,
        price_snapshot_id=None,
        resource_ceilings=(resource,),
        provider_budget_cap=ProviderBudgetCap(
            provider=provider,
            operation_kind=f"{binding_id}-operation",
            billing_unit="request",
            maximum_accepted_units=Decimal("1"),
        ),
    )


def _external_budget(*, role_ids: tuple[str, ...] | None = None) -> BudgetSpecV2:
    from oamb.contracts.specifications import (
        BudgetScopeKindV2,
        BudgetSpecV2,
        ResourceBudgetCeiling,
    )

    selected_role_ids = role_ids or (
        "extraction-binding",
        "embedding-binding",
        "answer-binding",
    )
    return BudgetSpecV2(
        budget_id="external-budget",
        scope_kind=BudgetScopeKindV2.RUN,
        scope_id="run-1",
        approval_id="approval-1",
        max_attempts=3,
        max_input_tokens=300,
        max_output_tokens=300,
        max_dispatch_wall_seconds=Decimal("90"),
        max_cost=None,
        currency=None,
        resource_ceilings=(
            ResourceBudgetCeiling(
                dimension_id="provider_request_wall_seconds_v1",
                maximum=Decimal("90"),
                unit="seconds",
            ),
        ),
        role_ceilings=tuple(_role_ceiling(role_id) for role_id in selected_role_ids),
        stop_condition_ids=("identity_drift", "budget_exhausted"),
    )


def _runtime_binding() -> MemorySystemRuntimeBindingV2:
    from oamb.contracts.specifications import (
        MemorySystemRuntimeBindingV2,
        RuntimeAttestationStatus,
        SourceEvidenceBinding,
        SourceEvidenceKind,
        memory_system_runtime_binding_hash,
    )

    readiness = SourceEvidenceBinding(
        binding_id=HASH_A,
        source_kind=SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity="readiness-occurrence",
        source_root_hash=HASH_B,
        validation_result_hash=HASH_C,
        source_schema_versions=("model_readiness_occurrence_record@1",),
    )
    values: dict[str, Any] = dict(
        memory_system_id="hindsight",
        provider_project_id="provider-project-1",
        provider_profile_id="hindsight-rest-v0.9.2",
        edition="oss",
        distribution_channel="official-container",
        api_version="v1",
        release_version="0.9.2",
        source_revision="source-revision",
        artifact_kind="container-image",
        artifact_sha256=HASH_B,
        endpoint_fingerprint=HASH_C,
        deployment_configuration_sha256=HASH_A,
        storage_engine="pg0",
        storage_engine_version="1",
        schema_revision="schema-1",
        vector_index_type="native",
        distance_metric="cosine",
        vector_dimension=1024,
        index_configuration_sha256=HASH_B,
        model_role_binding_ids=("extraction-binding", "embedding-binding"),
        native_feature_flags_fingerprint=HASH_C,
        native_reranking_status="disabled",
        attestation_method="runtime-and-build",
        attestation_status=RuntimeAttestationStatus.RUNTIME_VERIFIED,
        raw_proof_refs=(HASH_A,),
        model_readiness_required=True,
        model_readiness_evidence=readiness,
    )
    return MemorySystemRuntimeBindingV2(
        runtime_binding_hash=memory_system_runtime_binding_hash(values),
        **values,
    )


def _external_profile() -> AdapterProfileDescriptor:
    from oamb.runtime.preflight import (
        AdapterProfileDescriptor,
        ControlledEmbeddingDescriptor,
        TransportKind,
    )

    return AdapterProfileDescriptor(
        profile_id="hindsight-rest-v0.9.2",
        memory_system_id="hindsight",
        transport_kind=TransportKind.REST_API,
        controlled_embedding=ControlledEmbeddingDescriptor(
            endpoint_fingerprint=HASH_C,
            model="qwen3-embedding:0.6b",
            artifact_fingerprint=HASH_B,
            dimension=1024,
            input_adaptation_fingerprint=HASH_A,
        ),
        native_reranking_disabled=True,
        oamb_reranker_configured=False,
        provider_project_id="provider-project-1",
        release_version="0.9.2",
        source_revision="source-revision",
        build_artifact_sha256=HASH_B,
    )


def _passing_external_gates() -> ProviderGateClosure:
    from oamb.runtime.preflight import GateStatus, ProviderGateClosure

    return ProviderGateClosure(
        liveness=GateStatus.PASS,
        storage_configuration=GateStatus.PASS,
        runtime_identity=GateStatus.PASS,
        model_readiness=GateStatus.PASS,
        memory_conformance=GateStatus.PASS,
    )


def _external_approval(
    budget: BudgetSpecV2, runtime: MemorySystemRuntimeBindingV2
) -> ExternalCallApprovalRecord:
    from oamb.contracts.ids import canonical_sha256
    from oamb.contracts.specifications import (
        BudgetScopeKindV2,
        ExternalCallApprovalRecord,
        external_call_approval_hash,
    )

    values: dict[str, Any] = dict(
        approval_id="approval-1",
        operation_kind="benchmark_run",
        scope_kind=BudgetScopeKindV2.RUN,
        scope_id="run-1",
        runtime_binding_hash=runtime.runtime_binding_hash,
        provider_runtime_profile_attestation_hash=None,
        role_binding_ids=("extraction-binding", "embedding-binding", "answer-binding"),
        budget_hash=canonical_sha256(budget.model_dump(mode="python")),
        approved_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        unmetered_cost_acknowledged=True,
        stop_condition_ids=("identity_drift", "budget_exhausted"),
    )
    return ExternalCallApprovalRecord(
        approval_hash=external_call_approval_hash(values),
        **values,
    )


def _execution_environment() -> ExecutionEnvironmentBinding:
    from oamb.contracts.specifications import ComparabilityStatus, ExecutionEnvironmentBinding

    return ExecutionEnvironmentBinding(
        environment_hash=HASH_B,
        operating_system="linux",
        architecture="x86_64",
        python_version="3.11",
        cpu_description="fixture-cpu",
        memory_bytes=1_000_000,
        comparability_status=ComparabilityStatus.COMPARABLE,
    )


def test_external_benchmark_v2_closes_roles_budget_runtime_approval_meter_and_gates() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.runtime.preflight import OperationKind, RunPreflightRequest, resolve_run_plan

    budget = _external_budget()
    runtime = _runtime_binding()
    approval = _external_approval(budget, runtime)

    plan = resolve_run_plan(
        RunPreflightRequest(
            operation_kind=OperationKind.BENCHMARK_RUN,
            budget_spec=budget,
            role_slots=_external_role_slots(),
            adapter_profile=_external_profile(),
            provider_gates=_passing_external_gates(),
            artifact_durability=_durable_artifact_preflight(),
            runtime_binding=runtime,
            provider_runtime_attestation=None,
            approval=approval,
            cost_measurement_spec=CostMeasurementSpec(
                measurement_spec_id="measurement-v1",
                measurement_spec_version="1",
                dimensions=(),
            ),
            price_snapshot=None,
            environment={"OAMB_ANSWER_API_KEY": "must-not-enter-plan"},
            observed_at=NOW + timedelta(minutes=1),
            execution_environment_binding=_execution_environment(),
        )
    )

    assert plan.schema_versions == (1, 2)
    assert plan.runtime_binding is runtime
    assert plan.approval is approval
    assert plan.role_slots[-1].status.value == "unselected"
    assert "must-not-enter-plan" not in repr(plan)


def test_external_benchmark_rejects_self_consistent_runtime_release_drift() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.contracts.specifications import (
        MemorySystemRuntimeBindingV2,
        memory_system_runtime_binding_hash,
    )
    from oamb.runtime.preflight import (
        OperationKind,
        PreflightRejected,
        RunPreflightRequest,
        resolve_run_plan,
    )

    budget = _external_budget()
    runtime_values = _runtime_binding().model_dump(
        mode="python",
        exclude={"runtime_binding_hash"},
    )
    runtime_values["release_version"] = "99.0.0"
    drifted_runtime = MemorySystemRuntimeBindingV2(
        runtime_binding_hash=memory_system_runtime_binding_hash(runtime_values),
        **runtime_values,
    )

    with pytest.raises(PreflightRejected, match="exact identities"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.BENCHMARK_RUN,
                budget_spec=budget,
                role_slots=_external_role_slots(),
                adapter_profile=_external_profile(),
                provider_gates=_passing_external_gates(),
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=drifted_runtime,
                provider_runtime_attestation=None,
                approval=_external_approval(budget, drifted_runtime),
                cost_measurement_spec=CostMeasurementSpec(
                    measurement_spec_id="measurement-v1",
                    measurement_spec_version="1",
                    dimensions=(),
                ),
                price_snapshot=None,
                environment={"OAMB_ANSWER_API_KEY": "must-not-enter-plan"},
                observed_at=NOW + timedelta(minutes=1),
                execution_environment_binding=_execution_environment(),
            )
        )


def test_external_benchmark_rejects_missing_role_ceiling_before_provider_construction() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.runtime.preflight import (
        OperationKind,
        PreflightRejected,
        RunPreflightRequest,
        resolve_run_plan,
    )

    budget = _external_budget(role_ids=("extraction-binding", "embedding-binding"))
    runtime = _runtime_binding()
    approval = _external_approval(budget, runtime)

    with pytest.raises(PreflightRejected, match="exactly one.*selected external role"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.BENCHMARK_RUN,
                budget_spec=budget,
                role_slots=_external_role_slots(),
                adapter_profile=_external_profile(),
                provider_gates=_passing_external_gates(),
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=runtime,
                provider_runtime_attestation=None,
                approval=approval,
                cost_measurement_spec=CostMeasurementSpec(
                    measurement_spec_id="measurement-v1",
                    measurement_spec_version="1",
                    dimensions=(),
                ),
                price_snapshot=None,
                environment={"OAMB_ANSWER_API_KEY": "secret"},
                observed_at=NOW + timedelta(minutes=1),
                execution_environment_binding=_execution_environment(),
            )
        )


def test_external_benchmark_rejects_missing_execution_environment_binding() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.runtime.preflight import (
        OperationKind,
        PreflightRejected,
        RunPreflightRequest,
        resolve_run_plan,
    )

    budget = _external_budget()
    runtime = _runtime_binding()
    approval = _external_approval(budget, runtime)

    with pytest.raises(PreflightRejected, match="execution environment"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.BENCHMARK_RUN,
                budget_spec=budget,
                role_slots=_external_role_slots(),
                adapter_profile=_external_profile(),
                provider_gates=_passing_external_gates(),
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=runtime,
                provider_runtime_attestation=None,
                approval=approval,
                cost_measurement_spec=CostMeasurementSpec(
                    measurement_spec_id="measurement-v1",
                    measurement_spec_version="1",
                    dimensions=(),
                ),
                price_snapshot=None,
                environment={"OAMB_ANSWER_API_KEY": "secret"},
                observed_at=NOW + timedelta(minutes=1),
                execution_environment_binding=None,
            )
        )


def test_external_benchmark_rejects_controlled_endpoint_runtime_drift() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.runtime.preflight import (
        OperationKind,
        PreflightRejected,
        RunPreflightRequest,
        resolve_run_plan,
    )

    budget = _external_budget()
    runtime = _runtime_binding()
    approval = _external_approval(budget, runtime)
    slots = list(_external_role_slots())
    embedding_slot = slots[1]
    assert isinstance(embedding_slot.binding, ModelRoleBindingV2)
    slots[1] = replace(
        embedding_slot,
        binding=embedding_slot.binding.model_copy(update={"redacted_endpoint_fingerprint": HASH_A}),
    )

    with pytest.raises(PreflightRejected, match="endpoint fingerprint"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.BENCHMARK_RUN,
                budget_spec=budget,
                role_slots=tuple(slots),
                adapter_profile=_external_profile(),
                provider_gates=_passing_external_gates(),
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=runtime,
                provider_runtime_attestation=None,
                approval=approval,
                cost_measurement_spec=CostMeasurementSpec(
                    measurement_spec_id="measurement-v1",
                    measurement_spec_version="1",
                    dimensions=(),
                ),
                price_snapshot=None,
                environment={"OAMB_ANSWER_API_KEY": "secret"},
                observed_at=NOW + timedelta(minutes=1),
                execution_environment_binding=_execution_environment(),
            )
        )


def _model_readiness_role_slots() -> tuple[RoleSlot, ...]:
    from oamb.contracts.specifications import ExecutionOwner, ModelRole
    from oamb.runtime.preflight import ResolutionStatus, RoleSlot, RoleSlotName

    extraction = _selected_role(
        ModelRole.MEMORY_EXTRACTION,
        "extraction-binding",
        owner=ExecutionOwner.MEMORY_SYSTEM,
    )
    embedding = _selected_role(
        ModelRole.EMBEDDING,
        "embedding-binding",
        owner=ExecutionOwner.MEMORY_SYSTEM,
    )
    selected = {
        RoleSlotName.EXTRACTION: extraction,
        RoleSlotName.EMBEDDING: embedding,
    }
    return tuple(
        RoleSlot(
            role=role,
            status=(ResolutionStatus.RESOLVED if role in selected else ResolutionStatus.UNSELECTED),
            binding=selected.get(role),
            credential_reference=None,
            evidence_reference=("selected-readiness-role" if role in selected else "unselected"),
        )
        for role in RoleSlotName
    )


def _model_readiness_budget() -> BudgetSpecV2:
    from oamb.contracts.specifications import (
        BudgetScopeKindV2,
        BudgetSpecV2,
        ResourceBudgetCeiling,
    )

    return BudgetSpecV2(
        budget_id="readiness-budget",
        scope_kind=BudgetScopeKindV2.MODEL_READINESS,
        scope_id="readiness-occurrence",
        approval_id="readiness-approval",
        max_attempts=2,
        max_input_tokens=0,
        max_output_tokens=0,
        max_dispatch_wall_seconds=Decimal("60"),
        max_cost=None,
        currency=None,
        resource_ceilings=(
            ResourceBudgetCeiling(
                dimension_id="provider_request_wall_seconds_v1",
                maximum=Decimal("60"),
                unit="seconds",
            ),
        ),
        role_ceilings=(
            _role_ceiling("extraction-binding"),
            _role_ceiling("embedding-binding"),
        ),
        stop_condition_ids=("identity_drift", "budget_exhausted"),
    )


def _provider_runtime_attestation() -> ProviderRuntimeProfileAttestation:
    from oamb.contracts.specifications import (
        ProviderGateStatus,
        ProviderRuntimeProfileAttestation,
        TransportProfile,
        provider_runtime_profile_attestation_hash,
    )

    values: dict[str, Any] = dict(
        provider="hindsight",
        provider_project_id="provider-project-1",
        provider_profile_id="hindsight-rest-v0.9.2",
        transport_profile=TransportProfile.REST,
        release_version="0.9.2",
        source_revision="source-revision",
        build_artifact_sha256=HASH_B,
        redacted_configuration_sha256=HASH_C,
        redacted_endpoint_fingerprint=HASH_A,
        auth_configuration_sha256=HASH_B,
        storage_configuration_sha256=HASH_C,
        model_role_binding_ids=("extraction-binding", "embedding-binding"),
        native_reranking_status="disabled",
        liveness_status=ProviderGateStatus.PASS,
        storage_configuration_status=ProviderGateStatus.PASS,
        runtime_identity_status=ProviderGateStatus.PASS,
        model_readiness_status=ProviderGateStatus.NOT_RUN,
        memory_conformance_status=ProviderGateStatus.NOT_RUN,
        raw_proof_refs=(HASH_A,),
    )
    return ProviderRuntimeProfileAttestation(
        attestation_hash=provider_runtime_profile_attestation_hash(values),
        **values,
    )


def _model_readiness_approval(
    budget: BudgetSpecV2, attestation: ProviderRuntimeProfileAttestation
) -> ExternalCallApprovalRecord:
    from oamb.contracts.ids import canonical_sha256
    from oamb.contracts.specifications import (
        BudgetScopeKindV2,
        ExternalCallApprovalRecord,
        external_call_approval_hash,
    )

    values: dict[str, Any] = dict(
        approval_id="readiness-approval",
        operation_kind="model_readiness",
        scope_kind=BudgetScopeKindV2.MODEL_READINESS,
        scope_id="readiness-occurrence",
        runtime_binding_hash=None,
        provider_runtime_profile_attestation_hash=attestation.attestation_hash,
        role_binding_ids=("extraction-binding", "embedding-binding"),
        budget_hash=canonical_sha256(budget.model_dump(mode="python")),
        approved_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        unmetered_cost_acknowledged=True,
        stop_condition_ids=("identity_drift", "budget_exhausted"),
    )
    return ExternalCallApprovalRecord(
        approval_hash=external_call_approval_hash(values),
        **values,
    )


def _pre_readiness_gates() -> ProviderGateClosure:
    from oamb.runtime.preflight import GateStatus, ProviderGateClosure

    return ProviderGateClosure(
        liveness=GateStatus.PASS,
        storage_configuration=GateStatus.PASS,
        runtime_identity=GateStatus.PASS,
        model_readiness=GateStatus.NOT_RUN,
        memory_conformance=GateStatus.NOT_RUN,
    )


def test_model_readiness_uses_attestation_and_keeps_readiness_and_conformance_not_run() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.runtime.preflight import OperationKind, RunPreflightRequest, resolve_run_plan

    budget = _model_readiness_budget()
    attestation = _provider_runtime_attestation()
    approval = _model_readiness_approval(budget, attestation)

    plan = resolve_run_plan(
        RunPreflightRequest(
            operation_kind=OperationKind.MODEL_READINESS,
            budget_spec=budget,
            role_slots=_model_readiness_role_slots(),
            adapter_profile=_external_profile(),
            provider_gates=_pre_readiness_gates(),
            artifact_durability=_durable_artifact_preflight(),
            runtime_binding=None,
            provider_runtime_attestation=attestation,
            approval=approval,
            cost_measurement_spec=CostMeasurementSpec(
                measurement_spec_id="model-readiness-measurement-v1",
                measurement_spec_version="1",
                dimensions=(),
            ),
            price_snapshot=None,
            observed_at=NOW + timedelta(minutes=1),
        )
    )

    assert plan.runtime_binding is None
    assert plan.provider_runtime_attestation is attestation
    assert plan.provider_gates is not None
    assert plan.provider_gates.model_readiness.value == "not_run"
    assert plan.provider_gates.memory_conformance.value == "not_run"


def test_model_readiness_rejects_exact_release_drift_and_preapproval_observation() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.contracts.specifications import (
        ProviderRuntimeProfileAttestation,
        provider_runtime_profile_attestation_hash,
    )
    from oamb.runtime.preflight import (
        OperationKind,
        PreflightRejected,
        RunPreflightRequest,
        resolve_run_plan,
    )

    budget = _model_readiness_budget()
    attestation = _provider_runtime_attestation()
    attestation_values = attestation.model_dump(mode="python", exclude={"attestation_hash"})
    attestation_values["release_version"] = "99.0.0"
    drifted_attestation = ProviderRuntimeProfileAttestation(
        attestation_hash=provider_runtime_profile_attestation_hash(attestation_values),
        **attestation_values,
    )
    measurement = CostMeasurementSpec(
        measurement_spec_id="model-readiness-measurement-v1",
        measurement_spec_version="1",
        dimensions=(),
    )

    with pytest.raises(PreflightRejected, match="exact runtime identities"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.MODEL_READINESS,
                budget_spec=budget,
                role_slots=_model_readiness_role_slots(),
                adapter_profile=_external_profile(),
                provider_gates=_pre_readiness_gates(),
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=None,
                provider_runtime_attestation=drifted_attestation,
                approval=_model_readiness_approval(budget, drifted_attestation),
                cost_measurement_spec=measurement,
                price_snapshot=None,
                observed_at=NOW + timedelta(minutes=1),
            )
        )

    with pytest.raises(PreflightRejected, match="active interval"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.MODEL_READINESS,
                budget_spec=budget,
                role_slots=_model_readiness_role_slots(),
                adapter_profile=_external_profile(),
                provider_gates=_pre_readiness_gates(),
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=None,
                provider_runtime_attestation=attestation,
                approval=_model_readiness_approval(budget, attestation),
                cost_measurement_spec=measurement,
                price_snapshot=None,
                observed_at=NOW - timedelta(minutes=1),
            )
        )


def test_model_readiness_rejects_a_promoted_memory_conformance_gate() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.runtime.preflight import (
        GateStatus,
        OperationKind,
        PreflightRejected,
        ProviderGateClosure,
        RunPreflightRequest,
        resolve_run_plan,
    )

    budget = _model_readiness_budget()
    attestation = _provider_runtime_attestation()
    approval = _model_readiness_approval(budget, attestation)
    promoted = ProviderGateClosure(
        liveness=GateStatus.PASS,
        storage_configuration=GateStatus.PASS,
        runtime_identity=GateStatus.PASS,
        model_readiness=GateStatus.NOT_RUN,
        memory_conformance=GateStatus.PASS,
    )

    with pytest.raises(PreflightRejected, match="memory conformance.*NOT_RUN"):
        resolve_run_plan(
            RunPreflightRequest(
                operation_kind=OperationKind.MODEL_READINESS,
                budget_spec=budget,
                role_slots=_model_readiness_role_slots(),
                adapter_profile=_external_profile(),
                provider_gates=promoted,
                artifact_durability=_durable_artifact_preflight(),
                runtime_binding=None,
                provider_runtime_attestation=attestation,
                approval=approval,
                cost_measurement_spec=CostMeasurementSpec(
                    measurement_spec_id="model-readiness-measurement-v1",
                    measurement_spec_version="1",
                    dimensions=(),
                ),
                price_snapshot=None,
                observed_at=NOW + timedelta(minutes=1),
            )
        )


def _phase_review_role_slots() -> tuple[RoleSlot, ...]:
    from oamb.config.load import EnvironmentReference
    from oamb.contracts.specifications import (
        BindingKind,
        ExecutionOwner,
        ModelRole,
        ModelRoleBindingV2,
        RoleBindingStatus,
    )
    from oamb.runtime.preflight import ResolutionStatus, RoleSlotName

    quality = ModelRoleBindingV2(
        binding_id="quality-review-binding",
        role=ModelRole.QUALITY_REVIEW,
        role_status=RoleBindingStatus.SELECTED,
        execution_owner=ExecutionOwner.HARNESS,
        binding_kind=BindingKind.MODEL_CLIENT,
        provider="review-provider",
        endpoint_reference="review-endpoint",
        credential_variable_name="OAMB_QUALITY_REVIEW_API_KEY",
        configured_model="review-model",
        resolved_model="review-model@sha256:resolved",
        parameters_fingerprint=HASH_A,
        retry_policy_id="no-retry-v1",
        configuration_fingerprint=HASH_B,
        redacted_endpoint_fingerprint=HASH_C,
    )
    return tuple(
        RoleSlot(
            role=role,
            status=(
                ResolutionStatus.RESOLVED
                if role == RoleSlotName.QUALITY_REVIEW
                else ResolutionStatus.UNSELECTED
            ),
            binding=quality if role == RoleSlotName.QUALITY_REVIEW else None,
            credential_reference=(
                EnvironmentReference("OAMB_QUALITY_REVIEW_API_KEY")
                if role == RoleSlotName.QUALITY_REVIEW
                else None
            ),
            evidence_reference=(
                "phase-review-role" if role == RoleSlotName.QUALITY_REVIEW else "unselected"
            ),
        )
        for role in RoleSlotName
    )


def _phase_review_budget() -> BudgetSpecV2:
    from oamb.contracts.specifications import (
        BudgetScopeKindV2,
        BudgetSpecV2,
        ResourceBudgetCeiling,
    )

    return BudgetSpecV2(
        budget_id="phase-review-budget",
        scope_kind=BudgetScopeKindV2.PHASE_REVIEW,
        scope_id="phase-review-occurrence",
        approval_id="phase-review-approval",
        max_attempts=1,
        max_input_tokens=100,
        max_output_tokens=100,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=None,
        currency=None,
        resource_ceilings=(
            ResourceBudgetCeiling(
                dimension_id="provider_request_wall_seconds_v1",
                maximum=Decimal("30"),
                unit="seconds",
            ),
        ),
        role_ceilings=(_role_ceiling("quality-review-binding", provider="review-provider"),),
        stop_condition_ids=("budget_exhausted",),
    )


def _phase_review_approval(budget: BudgetSpecV2) -> ExternalCallApprovalRecord:
    from oamb.contracts.ids import canonical_sha256
    from oamb.contracts.specifications import (
        BudgetScopeKindV2,
        ExternalCallApprovalRecord,
        external_call_approval_hash,
    )

    values: dict[str, Any] = dict(
        approval_id="phase-review-approval",
        operation_kind="phase_review",
        scope_kind=BudgetScopeKindV2.PHASE_REVIEW,
        scope_id="phase-review-occurrence",
        runtime_binding_hash=None,
        provider_runtime_profile_attestation_hash=None,
        role_binding_ids=("quality-review-binding",),
        budget_hash=canonical_sha256(budget.model_dump(mode="python")),
        approved_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        unmetered_cost_acknowledged=False,
        stop_condition_ids=("budget_exhausted",),
    )
    return ExternalCallApprovalRecord(
        approval_hash=external_call_approval_hash(values),
        **values,
    )


def test_phase_review_selects_only_quality_role_and_has_no_memory_runtime_binding() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.runtime.preflight import OperationKind, RunPreflightRequest, resolve_run_plan

    budget = _phase_review_budget()
    approval = _phase_review_approval(budget)
    plan = resolve_run_plan(
        RunPreflightRequest(
            operation_kind=OperationKind.PHASE_REVIEW,
            budget_spec=budget,
            role_slots=_phase_review_role_slots(),
            adapter_profile=None,
            provider_gates=None,
            artifact_durability=_durable_artifact_preflight(),
            runtime_binding=None,
            provider_runtime_attestation=None,
            approval=approval,
            cost_measurement_spec=CostMeasurementSpec(
                measurement_spec_id="phase-review-measurement-v1",
                measurement_spec_version="1",
                dimensions=(),
            ),
            price_snapshot=None,
            environment={"OAMB_QUALITY_REVIEW_API_KEY": "must-not-enter-plan"},
            observed_at=NOW + timedelta(minutes=1),
            execution_environment_binding=_execution_environment(),
        )
    )

    assert plan.adapter_profile is None
    assert plan.provider_gates is None
    assert plan.runtime_binding is None
    assert [slot.status.value for slot in plan.role_slots] == [
        "unselected",
        "unselected",
        "unselected",
        "unselected",
        "resolved",
    ]
    assert "must-not-enter-plan" not in repr(plan)
