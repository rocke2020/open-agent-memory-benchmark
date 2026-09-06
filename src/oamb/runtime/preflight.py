"""Fail-closed configuration closure before any provider construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from oamb.config.load import EnvironmentReference
from oamb.config.resolve import RoleSelection, validate_selected_environment
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import ArtifactStorePort
from oamb.contracts.specifications import (
    BindingKind,
    BudgetSpecV2,
    BudgetSpecV4,
    DispatchBudgetRoute,
    ExecutionEnvironmentBinding,
    ExecutionOwner,
    MemorySystemRuntimeBindingV2,
    ModelRole,
    ModelRoleBindingV2,
    ProviderGateStatus,
    ProviderRuntimeProfileAttestation,
    RoleBindingStatus,
    RunPreflightRecord,
    RunPreflightRecordV2,
    RuntimeAttestationStatus,
    SourceEvidenceBinding,
    TransportProfile,
    memory_system_runtime_binding_hash,
    provider_runtime_profile_attestation_hash,
    run_preflight_record_hash,
    run_preflight_record_v2_hash,
)

from .source_records import seal_source_contract

CONTROLLED_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CONTROLLED_EMBEDDING_DIMENSION = 1024


class PreflightRejected(ValueError):
    pass


class OperationKind(StrEnum):
    FAKE_RUN = "fake_run"
    BENCHMARK_RUN = "benchmark_run"
    MODEL_READINESS = "model_readiness"


class TransportKind(StrEnum):
    REST_API = "rest_api"
    PYTHON_SDK = "python_sdk"
    FAKE = "fake"


class ResolutionStatus(StrEnum):
    RESOLVED = "resolved"
    UNSUPPORTED = "unsupported"
    UNATTESTED = "unattested"
    NOT_APPLICABLE = "not_applicable"
    UNSELECTED = "unselected"


class RoleSlotName(StrEnum):
    EXTRACTION = "extraction"
    EMBEDDING = "embedding"
    ANSWER = "answer"
    JUDGE = "judge"


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_RUN = "not_run"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class RoleSlot:
    role: RoleSlotName
    status: ResolutionStatus
    binding: object | None
    credential_reference: EnvironmentReference | None
    evidence_reference: str


@dataclass(frozen=True, slots=True)
class ControlledEmbeddingDescriptor:
    endpoint_fingerprint: str
    model: str
    artifact_fingerprint: str
    dimension: int
    input_adaptation_fingerprint: str


@dataclass(frozen=True, slots=True)
class AdapterProfileDescriptor:
    profile_id: str
    memory_system_id: str
    transport_kind: TransportKind
    controlled_embedding: ControlledEmbeddingDescriptor | None
    native_reranking_disabled: bool
    oamb_reranker_configured: bool
    provider_project_id: str | None = None
    release_version: str | None = None
    source_revision: str | None = None
    build_artifact_sha256: str | None = None

    @property
    def default_comparison_eligible(self) -> bool:
        return self.transport_kind == TransportKind.REST_API


@dataclass(frozen=True, slots=True)
class ProviderGateClosure:
    liveness: GateStatus
    storage_configuration: GateStatus
    runtime_identity: GateStatus
    model_readiness: GateStatus
    memory_conformance: GateStatus


@dataclass(frozen=True, slots=True)
class ArtifactDurabilityPreflight:
    artifact_root_fingerprint: str
    available_bytes: int
    lease_supported: bool
    file_fsync_supported: bool
    directory_fsync_supported: bool
    no_replace_supported: bool


def artifact_durability_capability_proof_hash(
    value: ArtifactDurabilityPreflight,
) -> str:
    """Bind stable root identity and atomic-publication capabilities, not free space."""

    return canonical_sha256(
        [
            "oamb-artifact-durability-capability-proof-v1",
            value.artifact_root_fingerprint,
            value.lease_supported,
            value.file_fsync_supported,
            value.directory_fsync_supported,
            value.no_replace_supported,
        ]
    )


@dataclass(frozen=True, slots=True)
class RunPreflightRequest:
    operation_kind: OperationKind
    budget_spec: object
    role_slots: tuple[object, ...]
    adapter_profile: object | None
    provider_gates: object | None
    artifact_durability: ArtifactDurabilityPreflight
    runtime_binding: object | None
    provider_runtime_attestation: object | None
    cost_measurement_spec: object | None
    price_snapshot: object | None
    environment: Mapping[str, str] | None = None
    observed_at: datetime | None = None
    execution_environment_binding: object | None = None


@dataclass(frozen=True, slots=True)
class ResolvedRunPlan:
    operation_kind: OperationKind
    budget_spec: object
    role_slots: tuple[RoleSlot, ...]
    adapter_profile: AdapterProfileDescriptor | None
    provider_gates: ProviderGateClosure | None
    artifact_durability: ArtifactDurabilityPreflight
    runtime_binding: object | None
    provider_runtime_attestation: object | None
    cost_measurement_spec: object | None
    price_snapshot: object | None
    execution_environment_binding: object | None
    schema_versions: tuple[int, ...]
    plan_hash: str


def validate_adapter_profile(profile: AdapterProfileDescriptor) -> AdapterProfileDescriptor:
    if profile.oamb_reranker_configured:
        raise PreflightRejected("an OAMB reranker is forbidden")
    if not profile.native_reranking_disabled:
        raise PreflightRejected("native reranking must be disabled and attested")
    if profile.transport_kind == TransportKind.FAKE:
        if profile.controlled_embedding is not None or any(
            value is not None
            for value in (
                profile.provider_project_id,
                profile.release_version,
                profile.source_revision,
                profile.build_artifact_sha256,
            )
        ):
            raise PreflightRejected("fake profile cannot claim an external runtime identity")
        return profile
    if any(
        not isinstance(value, str) or not value
        for value in (
            profile.provider_project_id,
            profile.release_version,
            profile.source_revision,
            profile.build_artifact_sha256,
        )
    ):
        raise PreflightRejected("external profile requires an exact runtime identity")
    embedding = profile.controlled_embedding
    if embedding is None:
        raise PreflightRejected("external profile requires the controlled embedding")
    if embedding.model != CONTROLLED_EMBEDDING_MODEL:
        raise PreflightRejected(f"controlled embedding model must be {CONTROLLED_EMBEDDING_MODEL}")
    if embedding.dimension != CONTROLLED_EMBEDDING_DIMENSION:
        raise PreflightRejected("controlled embedding dimension must be exactly 1,024")
    return profile


def _validate_artifact_durability(value: ArtifactDurabilityPreflight) -> None:
    if value.available_bytes <= 0:
        raise PreflightRejected("artifact root has no available space")
    if not all(
        (
            value.lease_supported,
            value.file_fsync_supported,
            value.directory_fsync_supported,
            value.no_replace_supported,
        )
    ):
        raise PreflightRejected("artifact root does not attest lease/fsync/no-replace durability")


def _validated_role_slots(values: tuple[object, ...]) -> tuple[RoleSlot, ...]:
    if not all(isinstance(value, RoleSlot) for value in values):
        raise PreflightRejected("role slots must use typed RoleSlot values")
    slots = tuple(value for value in values if isinstance(value, RoleSlot))
    if tuple(slot.role for slot in slots) != tuple(RoleSlotName):
        raise PreflightRejected("configuration must contain the five canonical role slots in order")
    for slot in slots:
        if (
            slot.status
            in {
                ResolutionStatus.UNSELECTED,
                ResolutionStatus.NOT_APPLICABLE,
                ResolutionStatus.UNSUPPORTED,
            }
            and slot.credential_reference is not None
        ):
            raise PreflightRejected(
                f"{slot.status.value} role {slot.role.value} cannot carry a credential reference"
            )
    return slots


def _budget_has_zero_external_allowance(budget: object) -> bool:
    return (
        getattr(budget, "max_attempts", None) == 0
        and getattr(budget, "max_input_tokens", None) == 0
        and getattr(budget, "max_output_tokens", None) == 0
        and getattr(budget, "max_wall_seconds", None) == 0
        and getattr(budget, "max_cost", None) is None
    )


def _plan_hash_payload(
    request: RunPreflightRequest,
    slots: tuple[RoleSlot, ...],
    profile: AdapterProfileDescriptor | None,
    gates: ProviderGateClosure | None,
) -> list[object]:
    budget = request.budget_spec
    if hasattr(budget, "model_dump"):
        budget_value: Any = budget.model_dump(mode="python")
    else:
        budget_value = repr(budget)
    return [
        "oamb-resolved-run-plan-v1",
        request.operation_kind.value,
        budget_value,
        [
            [
                slot.role.value,
                slot.status.value,
                slot.credential_reference.name if slot.credential_reference else None,
                slot.evidence_reference,
                _contract_value(slot.binding),
            ]
            for slot in slots
        ],
        (
            None
            if profile is None
            else [
                profile.profile_id,
                profile.memory_system_id,
                profile.transport_kind.value,
                (
                    None
                    if profile.controlled_embedding is None
                    else [
                        profile.controlled_embedding.endpoint_fingerprint,
                        profile.controlled_embedding.model,
                        profile.controlled_embedding.artifact_fingerprint,
                        profile.controlled_embedding.dimension,
                        profile.controlled_embedding.input_adaptation_fingerprint,
                    ]
                ),
                profile.native_reranking_disabled,
                profile.oamb_reranker_configured,
                profile.provider_project_id,
                profile.release_version,
                profile.source_revision,
                profile.build_artifact_sha256,
            ]
        ),
        (
            None
            if gates is None
            else [
                gates.liveness.value,
                gates.storage_configuration.value,
                gates.runtime_identity.value,
                gates.model_readiness.value,
                gates.memory_conformance.value,
            ]
        ),
        [
            request.artifact_durability.artifact_root_fingerprint,
            request.artifact_durability.available_bytes,
            request.artifact_durability.lease_supported,
            request.artifact_durability.file_fsync_supported,
            request.artifact_durability.directory_fsync_supported,
            request.artifact_durability.no_replace_supported,
        ],
        _contract_value(request.runtime_binding),
        _contract_value(request.provider_runtime_attestation),
        _contract_value(request.cost_measurement_spec),
        _contract_value(request.price_snapshot),
        _contract_value(request.execution_environment_binding),
    ]


def _contract_value(value: object | None) -> object:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    raise PreflightRejected(f"preflight input is not a versioned contract: {type(value).__name__}")


_MODEL_ROLE_BY_SLOT = {
    RoleSlotName.EXTRACTION: ModelRole.MEMORY_EXTRACTION,
    RoleSlotName.EMBEDDING: ModelRole.EMBEDDING,
    RoleSlotName.ANSWER: ModelRole.ANSWER,
    RoleSlotName.JUDGE: ModelRole.JUDGE,
}

_SELECTED_ROLE_OWNERSHIP = {
    ModelRole.MEMORY_EXTRACTION: {
        (ExecutionOwner.MEMORY_SYSTEM, BindingKind.NATIVE),
        (ExecutionOwner.HARNESS, BindingKind.MODEL_CLIENT),
    },
    ModelRole.EMBEDDING: {(ExecutionOwner.MEMORY_SYSTEM, BindingKind.NATIVE)},
    ModelRole.ANSWER: {(ExecutionOwner.HARNESS, BindingKind.MODEL_CLIENT)},
    ModelRole.JUDGE: {(ExecutionOwner.HARNESS, BindingKind.MODEL_CLIENT)},
}


def _validate_selected_role_ownership(binding: ModelRoleBindingV2) -> None:
    actual = (binding.execution_owner, binding.binding_kind)
    if actual not in _SELECTED_ROLE_OWNERSHIP[binding.role]:
        expected = " or ".join(
            f"{owner.value}/{kind.value}"
            for owner, kind in sorted(
                _SELECTED_ROLE_OWNERSHIP[binding.role],
                key=lambda item: (item[0].value, item[1].value),
            )
        )
        raise PreflightRejected(
            f"{binding.role.value} role requires {expected}; received "
            f"{binding.execution_owner.value}/{binding.binding_kind.value}"
        )
    if (
        binding.binding_kind == BindingKind.MODEL_CLIENT
        and binding.retry_policy_id != "no-retry-v1"
    ):
        raise PreflightRejected("harness model clients require no-retry-v1")
    if (
        binding.binding_kind == BindingKind.MODEL_CLIENT
        and binding.credential_variable_name is None
    ):
        raise PreflightRejected("harness model clients require a credential reference")


def _validate_external_roles(
    slots: tuple[RoleSlot, ...], environment: Mapping[str, str] | None
) -> tuple[ModelRoleBindingV2, ...]:
    selected: list[ModelRoleBindingV2] = []
    environment_roles: list[RoleSelection] = []
    for slot in slots:
        if slot.status == ResolutionStatus.RESOLVED:
            if not isinstance(slot.binding, ModelRoleBindingV2):
                raise PreflightRejected(
                    "external selected roles require ModelRoleBinding version 2"
                )
            if slot.binding.role_status != RoleBindingStatus.SELECTED:
                raise PreflightRejected("resolved role slot requires a selected role binding")
            if slot.binding.role != _MODEL_ROLE_BY_SLOT[slot.role]:
                raise PreflightRejected("role binding does not match its canonical role slot")
            _validate_selected_role_ownership(slot.binding)
            expected_reference = slot.binding.credential_variable_name
            actual_reference = (
                slot.credential_reference.name if slot.credential_reference is not None else None
            )
            if actual_reference != expected_reference:
                raise PreflightRejected("role credential reference does not match its binding")
            selected.append(slot.binding)
            environment_roles.append(
                RoleSelection(
                    role=slot.role.value,
                    selected=True,
                    credential_reference=slot.credential_reference,
                )
            )
        elif slot.binding is not None:
            if not isinstance(slot.binding, ModelRoleBindingV2):
                raise PreflightRejected("external role slots require version 2 bindings")
            if slot.binding.role != _MODEL_ROLE_BY_SLOT[slot.role]:
                raise PreflightRejected("role binding does not match its canonical role slot")
            if slot.binding.role_status == RoleBindingStatus.SELECTED:
                raise PreflightRejected("unselected role slot cannot carry a selected binding")
    validate_selected_environment(tuple(environment_roles), environment or {})
    return tuple(selected)


def _validate_budget_role_closure(
    budget: BudgetSpecV2 | BudgetSpecV4,
    selected_roles: tuple[ModelRoleBindingV2, ...],
) -> None:
    selected_ids = tuple(role.binding_id for role in selected_roles)
    ceiling_ids = tuple(ceiling.role_binding_id for ceiling in budget.role_ceilings)
    if ceiling_ids != selected_ids:
        raise PreflightRejected(
            "budget requires exactly one ordered ceiling for each selected external role"
        )
    parent_resource_ids = {ceiling.dimension_id for ceiling in budget.resource_ceilings}
    for role, ceiling in zip(selected_roles, budget.role_ceilings, strict=True):
        if ceiling.provider_budget_cap.provider != role.provider:
            raise PreflightRejected("provider budget cap does not match its selected role")
        if not {item.dimension_id for item in ceiling.resource_ceilings} <= parent_resource_ids:
            raise PreflightRejected("role resource ceiling is absent from the parent budget")


def _validate_controlled_embedding_binding(
    profile: AdapterProfileDescriptor,
    selected_roles: tuple[ModelRoleBindingV2, ...],
) -> None:
    embedding_roles = tuple(role for role in selected_roles if role.role == ModelRole.EMBEDDING)
    if len(embedding_roles) != 1 or profile.controlled_embedding is None:
        raise PreflightRejected("external profile requires one selected controlled embedding role")
    binding = embedding_roles[0]
    if binding.model != profile.controlled_embedding.model:
        raise PreflightRejected("controlled embedding model does not match its role binding")
    if binding.redacted_endpoint_fingerprint != profile.controlled_embedding.endpoint_fingerprint:
        raise PreflightRejected("controlled embedding endpoint fingerprint does not match its role")


def _validate_price_closure(
    budget: BudgetSpecV2 | BudgetSpecV4,
    price_snapshot: object | None,
) -> None:
    price_ids = {
        ceiling.price_snapshot_id
        for ceiling in budget.role_ceilings
        if ceiling.price_snapshot_id is not None
    }
    if not price_ids:
        if price_snapshot is not None:
            raise PreflightRejected("unused price snapshot is not part of budget closure")
        return
    if len(price_ids) != 1 or price_snapshot is None:
        raise PreflightRejected("priced roles require one matching price snapshot")
    if getattr(price_snapshot, "price_snapshot_id", None) not in price_ids:
        raise PreflightRejected("price snapshot does not match the role budget ceiling")


def _validate_benchmark_runtime(
    runtime: object | None,
    profile: AdapterProfileDescriptor,
    selected_roles: tuple[ModelRoleBindingV2, ...],
) -> MemorySystemRuntimeBindingV2:
    if not isinstance(runtime, MemorySystemRuntimeBindingV2):
        raise PreflightRejected("external benchmark requires a validated runtime binding version 2")
    expected_hash = memory_system_runtime_binding_hash(
        runtime.model_dump(mode="python", exclude={"runtime_binding_hash"})
    )
    if runtime.runtime_binding_hash != expected_hash:
        raise PreflightRejected("runtime binding hash does not match its fields")
    if runtime.attestation_status not in {
        RuntimeAttestationStatus.RUNTIME_VERIFIED,
        RuntimeAttestationStatus.BUILD_PROVENANCE_VERIFIED,
    }:
        raise PreflightRejected("external benchmark runtime is unattested or unsupported")
    if runtime.memory_system_id != profile.memory_system_id:
        raise PreflightRejected("adapter profile and runtime memory system do not match")
    if runtime.provider_profile_id != profile.profile_id:
        raise PreflightRejected("adapter and runtime profile identities do not match")
    if (
        runtime.provider_project_id != profile.provider_project_id
        or runtime.release_version != profile.release_version
        or runtime.source_revision != profile.source_revision
        or runtime.artifact_sha256 != profile.build_artifact_sha256
    ):
        raise PreflightRejected("adapter and runtime exact identities do not match")
    if runtime.vector_dimension != CONTROLLED_EMBEDDING_DIMENSION:
        raise PreflightRejected("runtime controlled embedding dimension must be exactly 1,024")
    if runtime.native_reranking_status != "disabled":
        raise PreflightRejected("runtime native reranking must be disabled")
    memory_owned_ids = tuple(
        role.binding_id
        for role in selected_roles
        if role.execution_owner == ExecutionOwner.MEMORY_SYSTEM
    )
    if runtime.model_role_binding_ids != memory_owned_ids:
        raise PreflightRejected("model roles do not match selected memory-owned roles")
    return runtime


def _require_all_provider_gates_pass(gates: ProviderGateClosure) -> None:
    gate_values = (
        gates.liveness,
        gates.storage_configuration,
        gates.runtime_identity,
        gates.model_readiness,
        gates.memory_conformance,
    )
    if any(status != GateStatus.PASS for status in gate_values):
        raise PreflightRejected("benchmark run requires all five provider gates to PASS")


def _validate_model_readiness_attestation(
    attestation: object | None,
    profile: AdapterProfileDescriptor,
    selected_roles: tuple[ModelRoleBindingV2, ...],
    gates: ProviderGateClosure,
) -> ProviderRuntimeProfileAttestation:
    if not isinstance(attestation, ProviderRuntimeProfileAttestation):
        raise PreflightRejected("model readiness requires a provider runtime profile attestation")
    expected_hash = provider_runtime_profile_attestation_hash(
        attestation.model_dump(mode="python", exclude={"attestation_hash"})
    )
    if attestation.attestation_hash != expected_hash:
        raise PreflightRejected("provider runtime attestation hash does not match its fields")
    if attestation.provider_profile_id != profile.profile_id:
        raise PreflightRejected("adapter profile and pre-readiness attestation do not match")
    if (
        attestation.provider != profile.memory_system_id
        or attestation.provider_project_id != profile.provider_project_id
        or attestation.release_version != profile.release_version
        or attestation.source_revision != profile.source_revision
        or attestation.build_artifact_sha256 != profile.build_artifact_sha256
    ):
        raise PreflightRejected("adapter and attested exact runtime identities do not match")
    expected_transport = {
        TransportKind.REST_API: TransportProfile.REST,
        TransportKind.PYTHON_SDK: TransportProfile.SDK,
    }.get(profile.transport_kind)
    if expected_transport is None or attestation.transport_profile != expected_transport:
        raise PreflightRejected("adapter and attested transport profiles do not match")
    selected_ids = tuple(role.binding_id for role in selected_roles)
    if attestation.model_role_binding_ids != selected_ids:
        raise PreflightRejected("attestation does not bind the selected readiness roles")
    expected_gate_pairs = (
        (gates.liveness, attestation.liveness_status),
        (gates.storage_configuration, attestation.storage_configuration_status),
        (gates.runtime_identity, attestation.runtime_identity_status),
    )
    if any(
        internal != GateStatus.PASS or public != ProviderGateStatus.PASS
        for internal, public in expected_gate_pairs
    ):
        raise PreflightRejected("model readiness requires liveness/storage/runtime identity PASS")
    if gates.model_readiness != GateStatus.NOT_RUN:
        raise PreflightRejected("model readiness gate must remain NOT_RUN before dispatch")
    if gates.memory_conformance != GateStatus.NOT_RUN:
        raise PreflightRejected("memory conformance gate must remain NOT_RUN")
    return attestation


def _schema_versions(request: RunPreflightRequest) -> tuple[int, ...]:
    records = (
        request.budget_spec,
        *(slot.binding for slot in request.role_slots if isinstance(slot, RoleSlot)),
        request.runtime_binding,
        request.provider_runtime_attestation,
        request.cost_measurement_spec,
        request.price_snapshot,
        request.execution_environment_binding,
    )
    versions = {
        version
        for record in records
        if record is not None
        for version in (getattr(record, "schema_version", None),)
        if isinstance(version, int)
    }
    return tuple(sorted(versions))


def resolve_run_plan(request: RunPreflightRequest) -> ResolvedRunPlan:
    _validate_artifact_durability(request.artifact_durability)
    slots = _validated_role_slots(request.role_slots)
    profile: AdapterProfileDescriptor | None = None
    gates: ProviderGateClosure | None = None
    if not isinstance(request.adapter_profile, AdapterProfileDescriptor):
        raise PreflightRejected("run preflight requires an adapter profile descriptor")
    profile = validate_adapter_profile(request.adapter_profile)
    if not isinstance(request.provider_gates, ProviderGateClosure):
        raise PreflightRejected("run preflight requires provider gate closure")
    gates = request.provider_gates

    budget_version = getattr(request.budget_spec, "schema_version", None)
    if request.operation_kind == OperationKind.FAKE_RUN:
        assert profile is not None and gates is not None
        if budget_version != 1 or not _budget_has_zero_external_allowance(request.budget_spec):
            raise PreflightRejected("version 1 budgets are zero-external fake-only")
        if profile.transport_kind != TransportKind.FAKE:
            raise PreflightRejected("fake run requires the fake adapter profile")
        gate_values = (
            gates.liveness,
            gates.storage_configuration,
            gates.runtime_identity,
            gates.model_readiness,
            gates.memory_conformance,
        )
        if any(status != GateStatus.NOT_APPLICABLE for status in gate_values):
            raise PreflightRejected("fake run provider gates must be not_applicable")
    elif request.operation_kind == OperationKind.BENCHMARK_RUN:
        assert profile is not None and gates is not None
        if not isinstance(request.budget_spec, BudgetSpecV4):
            raise PreflightRejected("live benchmark runs require BudgetSpec version 4")
        selected_roles = _validate_external_roles(slots, request.environment)
        _validate_controlled_embedding_binding(profile, selected_roles)
        _validate_budget_role_closure(request.budget_spec, selected_roles)
        _validate_benchmark_runtime(
            request.runtime_binding,
            profile,
            selected_roles,
        )
        if request.provider_runtime_attestation is not None:
            raise PreflightRejected(
                "benchmark run consumes a runtime binding, not pre-readiness attestation"
            )
        if request.cost_measurement_spec is None:
            raise PreflightRejected("external benchmark requires a cost measurement specification")
        if not isinstance(request.execution_environment_binding, ExecutionEnvironmentBinding):
            raise PreflightRejected("external benchmark requires an execution environment binding")
        _validate_price_closure(request.budget_spec, request.price_snapshot)
        _require_all_provider_gates_pass(gates)
    elif request.operation_kind == OperationKind.MODEL_READINESS:
        assert profile is not None and gates is not None
        if not isinstance(request.budget_spec, BudgetSpecV2):
            raise PreflightRejected("external operations require BudgetSpec version 2")
        selected_roles = _validate_external_roles(slots, request.environment)
        _validate_controlled_embedding_binding(profile, selected_roles)
        _validate_budget_role_closure(request.budget_spec, selected_roles)
        if request.runtime_binding is not None:
            raise PreflightRejected(
                "model readiness uses pre-readiness attestation, not runtime binding"
            )
        _validate_model_readiness_attestation(
            request.provider_runtime_attestation,
            profile,
            selected_roles,
            gates,
        )
        if request.cost_measurement_spec is None:
            raise PreflightRejected("model readiness requires a cost measurement specification")
        _validate_price_closure(request.budget_spec, request.price_snapshot)
    else:
        raise PreflightRejected("external operations require version 2 control contracts")

    return ResolvedRunPlan(
        operation_kind=request.operation_kind,
        budget_spec=request.budget_spec,
        role_slots=slots,
        adapter_profile=profile,
        provider_gates=gates,
        artifact_durability=request.artifact_durability,
        runtime_binding=request.runtime_binding,
        provider_runtime_attestation=request.provider_runtime_attestation,
        cost_measurement_spec=request.cost_measurement_spec,
        price_snapshot=request.price_snapshot,
        execution_environment_binding=request.execution_environment_binding,
        schema_versions=_schema_versions(request),
        plan_hash=canonical_sha256(_plan_hash_payload(request, slots, profile, gates)),
    )


def build_run_preflight_record(
    *,
    plan: ResolvedRunPlan,
    run_id: str,
    observed_at: datetime,
    run_spec_hash: str,
    dataset_manifest_hash: str,
    subset_manifest_hash: str,
    adapter_profile_hash: str,
    provider_service_evidence: SourceEvidenceBinding,
    provider_profile_evidence: SourceEvidenceBinding,
    dispatch_routes: tuple[DispatchBudgetRoute, ...],
    redacted_endpoint_fingerprints: tuple[str, ...],
    credential_reference_fingerprints: tuple[str, ...],
    comparison_control_basis_hash: str | None = None,
) -> RunPreflightRecord | RunPreflightRecordV2:
    """Build the public durable closure from one already-resolved live plan."""

    if plan.operation_kind != OperationKind.BENCHMARK_RUN:
        raise PreflightRejected("run preflight record requires a benchmark-run plan")
    if plan.adapter_profile is None or plan.runtime_binding is None:
        raise PreflightRejected("run preflight record requires live profile and runtime")
    if not isinstance(plan.budget_spec, BudgetSpecV4):
        raise PreflightRejected("run preflight record requires BudgetSpec version 4")
    profile = plan.adapter_profile
    if profile.provider_project_id is None:
        raise PreflightRejected("run preflight record requires a provider project")
    runtime_binding_hash = getattr(plan.runtime_binding, "runtime_binding_hash", None)
    if not isinstance(runtime_binding_hash, str):
        raise PreflightRejected("run preflight record requires a hashed runtime")
    role_binding_ids_list: list[str] = []
    for slot in plan.role_slots:
        if slot.status != ResolutionStatus.RESOLVED:
            continue
        binding_value = _contract_value(slot.binding)
        if not isinstance(binding_value, Mapping):
            raise PreflightRejected("resolved run role requires a versioned binding")
        binding_id = binding_value.get("binding_id")
        if not isinstance(binding_id, str) or not binding_id:
            raise PreflightRejected("resolved run role requires a binding identity")
        role_binding_ids_list.append(binding_id)
    role_binding_ids = tuple(role_binding_ids_list)
    if dispatch_routes != plan.budget_spec.dispatch_routes:
        raise PreflightRejected("run preflight routes differ from the exact version 4 budget")
    budget_hash = canonical_sha256(_contract_value(plan.budget_spec))
    durability_proof_hash = artifact_durability_capability_proof_hash(plan.artifact_durability)
    fields = {
        "run_id": run_id,
        "observed_at": observed_at,
        "resolved_plan_hash": plan.plan_hash,
        "run_spec_hash": run_spec_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "subset_manifest_hash": subset_manifest_hash,
        "adapter_profile_id": profile.profile_id,
        "adapter_profile_hash": adapter_profile_hash,
        "provider_project_id": profile.provider_project_id,
        "provider_profile_id": profile.profile_id,
        "runtime_binding_hash": runtime_binding_hash,
        "provider_service_evidence": provider_service_evidence,
        "provider_profile_evidence": provider_profile_evidence,
        "role_binding_ids": role_binding_ids,
        "dispatch_routes": dispatch_routes,
        "budget_hash": budget_hash,
        "redacted_endpoint_fingerprints": redacted_endpoint_fingerprints,
        "credential_reference_fingerprints": credential_reference_fingerprints,
        "artifact_repository_fingerprint": (plan.artifact_durability.artifact_root_fingerprint),
        "artifact_durability_proof_hash": durability_proof_hash,
    }
    if comparison_control_basis_hash is None:
        return RunPreflightRecord.model_validate(
            {
                "preflight_record_hash": run_preflight_record_hash(fields),
                **fields,
            }
        )
    version_two_fields = {
        **fields,
        "comparison_control_basis_hash": comparison_control_basis_hash,
    }
    return RunPreflightRecordV2.model_validate(
        {
            "preflight_record_hash": run_preflight_record_v2_hash(version_two_fields),
            **version_two_fields,
        }
    )


def seal_run_preflight_record(
    store: ArtifactStorePort,
    record: RunPreflightRecord | RunPreflightRecordV2,
) -> None:
    seal_source_contract(
        store,
        relative_path="source/specs/run-preflight.json",
        record_id=record.preflight_record_hash,
        record=record,
    )
