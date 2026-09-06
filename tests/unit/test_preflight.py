from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Any

import pytest

from oamb.contracts.specifications import (
    BudgetSpec,
    BudgetSpecV2,
    BudgetSpecV4,
    ExecutionEnvironmentBinding,
    ExecutionOwner,
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
from tests.benchmark_configuration import load_canonical_configuration

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
BENCHMARK_EMBEDDING_MODEL = load_canonical_configuration().models.embedding.model


def _zero_fake_budget() -> BudgetSpec:
    from oamb.contracts.specifications import BudgetScopeKind, BudgetSpec

    return BudgetSpec(
        budget_id="fake-zero-budget",
        scope_kind=BudgetScopeKind.RUN,
        scope_id="fake-run",
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


def test_durability_capability_proof_is_stable_when_free_space_drifts() -> None:
    from oamb.runtime.preflight import artifact_durability_capability_proof_hash

    initial = _durable_artifact_preflight()

    assert artifact_durability_capability_proof_hash(initial) == (
        artifact_durability_capability_proof_hash(
            replace(initial, available_bytes=initial.available_bytes - 1)
        )
    )
    assert artifact_durability_capability_proof_hash(initial) != (
        artifact_durability_capability_proof_hash(replace(initial, no_replace_supported=False))
    )


def _fake_role_slots() -> tuple[object, ...]:
    from oamb.runtime.preflight import ResolutionStatus, RoleSlot, RoleSlotName

    return tuple(
        RoleSlot(
            role=role,
            status=ResolutionStatus.NOT_APPLICABLE,
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
            cost_measurement_spec=None,
            price_snapshot=None,
        )
    )

    assert plan.operation_kind == OperationKind.FAKE_RUN
    assert plan.schema_versions == (1,)
    assert len(plan.role_slots) == 4
    assert plan.role_slots[-1].status.value == "not_applicable"
    assert len(plan.plan_hash) == 64
    with pytest.raises(FrozenInstanceError):
        plan.plan_hash = "b" * 64  # type: ignore[misc]


def test_v1_budget_cannot_allow_nonzero_external_allowance() -> None:
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
                cost_measurement_spec=None,
                price_snapshot=None,
            )
        )


@pytest.mark.parametrize(
    ("model", "dimension", "reranking_disabled", "oamb_reranker", "message"),
    (
        ("wrong-model", 1024, True, False, BENCHMARK_EMBEDDING_MODEL),
        (BENCHMARK_EMBEDDING_MODEL, 768, True, False, "1,024"),
        (BENCHMARK_EMBEDDING_MODEL, 1024, False, False, "native reranking"),
        (BENCHMARK_EMBEDDING_MODEL, 1024, True, True, "OAMB reranker"),
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
        model=BENCHMARK_EMBEDDING_MODEL,
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
        binding_kind=(
            BindingKind.MODEL_CLIENT if owner == ExecutionOwner.HARNESS else BindingKind.NATIVE
        ),
        provider="fixture-provider",
        endpoint_reference="fixture-endpoint",
        credential_variable_name=credential_variable_name,
        model=(BENCHMARK_EMBEDDING_MODEL if str(role) == "embedding" else "fixture-model"),
        thinking_effort=(
            "not_applicable"
            if role == ModelRole.EMBEDDING
            else "low"
            if role == ModelRole.ANSWER
            else "high"
        ),
        parameters_fingerprint=HASH_A,
        retry_policy_id="no-retry-v1",
        configuration_fingerprint=HASH_B,
        redacted_endpoint_fingerprint=HASH_C,
    )


def test_external_role_owner_and_transport_kind_matrix_fails_closed() -> None:
    from oamb.contracts.specifications import (
        BindingKind,
        ExecutionOwner,
        ModelRole,
        ModelRoleBindingV2,
    )
    from oamb.runtime.preflight import PreflightRejected, _validate_external_roles

    slots = list(_external_role_slots())
    answer = slots[2].binding
    assert isinstance(answer, ModelRoleBindingV2)
    slots[2] = slots[2].__class__(
        role=slots[2].role,
        status=slots[2].status,
        binding=answer.model_copy(update={"binding_kind": BindingKind.NATIVE}),
        credential_reference=slots[2].credential_reference,
        evidence_reference=slots[2].evidence_reference,
    )
    with pytest.raises(PreflightRejected, match="answer.*harness.*model_client"):
        _validate_external_roles(tuple(slots), {"OAMB_ANSWER_API_KEY": "secret"})

    embedding = _selected_role(
        ModelRole.EMBEDDING,
        "bad-embedding",
        owner=ExecutionOwner.HARNESS,
    ).model_copy(update={"binding_kind": BindingKind.MODEL_CLIENT})
    slots = list(_external_role_slots())
    slots[1] = slots[1].__class__(
        role=slots[1].role,
        status=slots[1].status,
        binding=embedding,
        credential_reference=slots[1].credential_reference,
        evidence_reference=slots[1].evidence_reference,
    )
    with pytest.raises(PreflightRejected, match="embedding.*memory_system.*native"):
        _validate_external_roles(tuple(slots), {"OAMB_ANSWER_API_KEY": "secret"})


def test_selected_harness_model_client_requires_zero_opaque_retry_profile() -> None:
    from oamb.contracts.specifications import ModelRoleBindingV2
    from oamb.runtime.preflight import PreflightRejected, _validate_external_roles

    slots = list(_external_role_slots())
    answer = slots[2].binding
    assert isinstance(answer, ModelRoleBindingV2)
    slots[2] = slots[2].__class__(
        role=slots[2].role,
        status=slots[2].status,
        binding=answer.model_copy(update={"retry_policy_id": "sdk-default-retries"}),
        credential_reference=slots[2].credential_reference,
        evidence_reference=slots[2].evidence_reference,
    )

    with pytest.raises(PreflightRejected, match="no-retry-v1"):
        _validate_external_roles(tuple(slots), {"OAMB_ANSWER_API_KEY": "secret"})


def test_selected_harness_model_client_requires_a_credential_reference() -> None:
    from oamb.contracts.specifications import ModelRoleBindingV2
    from oamb.runtime.preflight import PreflightRejected, _validate_external_roles

    slots = list(_external_role_slots())
    answer = slots[2].binding
    assert isinstance(answer, ModelRoleBindingV2)
    slots[2] = slots[2].__class__(
        role=slots[2].role,
        status=slots[2].status,
        binding=answer.model_copy(update={"credential_variable_name": None}),
        credential_reference=None,
        evidence_reference=slots[2].evidence_reference,
    )

    with pytest.raises(PreflightRejected, match="credential"):
        _validate_external_roles(tuple(slots), {})


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


def _external_budget(*, role_ids: tuple[str, ...] | None = None) -> BudgetSpecV4:
    from oamb.contracts.specifications import (
        BudgetScopeKindV3,
        BudgetSpecV4,
        DispatchBudgetOwnerKind,
        DispatchBudgetRoute,
        ProviderOperationBudgetCeiling,
        ResourceBudgetCeiling,
        budget_spec_v4_hash,
        dispatch_budget_route_hash,
        provider_operation_budget_ceiling_hash,
    )

    selected_role_ids = role_ids or (
        "extraction-binding",
        "embedding-binding",
        "answer-binding",
    )
    resource = ResourceBudgetCeiling(
        dimension_id="provider_request_wall_seconds_v1",
        maximum=Decimal("90"),
        unit="seconds",
    )
    provider_fields = dict(
        provider_operation_ceiling_id="hindsight-memory-ingest",
        adapter_profile_id="hindsight-rest-v0.9.2",
        operation_kind="memory_ingest",
        billing_unit="request",
        maximum_accepted_units=Decimal("1"),
        max_attempts=1,
        max_dispatch_wall_seconds=Decimal("30"),
        resource_ceilings=(resource,),
    )
    provider_ceiling = ProviderOperationBudgetCeiling.model_validate(
        {
            "provider_operation_ceiling_hash": provider_operation_budget_ceiling_hash(
                provider_fields
            ),
            **provider_fields,
        }
    )
    provider_route_fields = dict(
        route_id="memory-ingest",
        stage="memory_ingest",
        dispatch_owner_kind=DispatchBudgetOwnerKind.PROVIDER_OPERATION,
        dispatch_model_role_binding_id=None,
        provider_operation_ceiling_id=provider_ceiling.provider_operation_ceiling_id,
        adapter_profile_id=provider_ceiling.adapter_profile_id,
        operation_kind=provider_ceiling.operation_kind,
        billing_unit=provider_ceiling.billing_unit,
        internal_usage_role_binding_ids=tuple(
            role_id
            for role_id in ("extraction-binding", "embedding-binding")
            if role_id in selected_role_ids
        ),
    )
    answer_route_fields = dict(
        route_id="answer",
        stage="answer",
        dispatch_owner_kind=DispatchBudgetOwnerKind.MODEL_ROLE,
        dispatch_model_role_binding_id="answer-binding",
        provider_operation_ceiling_id=None,
        adapter_profile_id=None,
        operation_kind="answer-binding-operation",
        billing_unit="request",
        internal_usage_role_binding_ids=(),
    )
    fields = dict(
        budget_id="external-budget",
        scope_kind=BudgetScopeKindV3.RUN,
        scope_id="run-1",
        max_attempts=3,
        max_input_tokens=300,
        max_output_tokens=300,
        max_dispatch_wall_seconds=Decimal("90"),
        max_cost=None,
        currency=None,
        resource_ceilings=(resource,),
        role_ceilings=tuple(_role_ceiling(role_id) for role_id in selected_role_ids),
        provider_operation_ceilings=(provider_ceiling,),
        dispatch_routes=(
            DispatchBudgetRoute.model_validate(
                {
                    "route_hash": dispatch_budget_route_hash(provider_route_fields),
                    **provider_route_fields,
                }
            ),
            DispatchBudgetRoute.model_validate(
                {
                    "route_hash": dispatch_budget_route_hash(answer_route_fields),
                    **answer_route_fields,
                }
            ),
        ),
        stop_condition_ids=("identity_drift", "budget_exhausted"),
    )
    return BudgetSpecV4.model_validate({"budget_hash": budget_spec_v4_hash(fields), **fields})


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
            model=BENCHMARK_EMBEDDING_MODEL,
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


def test_external_benchmark_v4_closes_routes_roles_runtime_meter_and_gates() -> None:
    from oamb.contracts.accounting import CostMeasurementSpec
    from oamb.runtime.preflight import OperationKind, RunPreflightRequest, resolve_run_plan

    budget = _external_budget()
    runtime = _runtime_binding()

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

    assert plan.schema_versions == (1, 2, 4)
    assert plan.runtime_binding is runtime
    assert plan.role_slots[-1].status.value == "not_applicable"
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

    complete_budget = _external_budget()
    budget = complete_budget.model_copy(update={"role_ceilings": complete_budget.role_ceilings[:2]})
    runtime = _runtime_binding()

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


def test_model_readiness_rejects_exact_release_drift() -> None:
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
                cost_measurement_spec=measurement,
                price_snapshot=None,
                observed_at=NOW + timedelta(minutes=1),
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
                cost_measurement_spec=CostMeasurementSpec(
                    measurement_spec_id="model-readiness-measurement-v1",
                    measurement_spec_version="1",
                    dimensions=(),
                ),
                price_snapshot=None,
                observed_at=NOW + timedelta(minutes=1),
            )
        )
