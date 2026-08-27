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
from oamb.contracts.specifications import (
    BudgetScopeKindV2,
    BudgetSpecV2,
    ExecutionEnvironmentBinding,
    ExecutionOwner,
    ExternalCallApprovalRecord,
    MemorySystemRuntimeBindingV2,
    ModelRole,
    ModelRoleBindingV2,
    ProviderGateStatus,
    ProviderRuntimeProfileAttestation,
    RoleBindingStatus,
    RuntimeAttestationStatus,
    TransportProfile,
    external_call_approval_hash,
    memory_system_runtime_binding_hash,
    provider_runtime_profile_attestation_hash,
)

CONTROLLED_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CONTROLLED_EMBEDDING_DIMENSION = 1024


class PreflightRejected(ValueError):
    pass


class OperationKind(StrEnum):
    FAKE_RUN = "fake_run"
    BENCHMARK_RUN = "benchmark_run"
    MODEL_READINESS = "model_readiness"
    PHASE_REVIEW = "phase_review"


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
    QUALITY_REVIEW = "quality_review"


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
    approval: object | None
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
    approval: object | None
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
        and getattr(budget, "approval_id", None) is None
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
        _contract_value(request.approval),
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
    RoleSlotName.QUALITY_REVIEW: ModelRole.QUALITY_REVIEW,
}


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
    budget: BudgetSpecV2, selected_roles: tuple[ModelRoleBindingV2, ...]
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
    if binding.configured_model != profile.controlled_embedding.model:
        raise PreflightRejected("controlled embedding model does not match its role binding")
    if binding.redacted_endpoint_fingerprint != profile.controlled_embedding.endpoint_fingerprint:
        raise PreflightRejected("controlled embedding endpoint fingerprint does not match its role")


def _validate_price_closure(budget: BudgetSpecV2, price_snapshot: object | None) -> None:
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


def _require_active_approval(
    request: RunPreflightRequest,
    approval: ExternalCallApprovalRecord,
) -> None:
    expected_hash = external_call_approval_hash(
        approval.model_dump(mode="python", exclude={"approval_hash"})
    )
    if approval.approval_hash != expected_hash:
        raise PreflightRejected("external-call approval hash does not match its fields")
    if (
        request.observed_at is None
        or request.observed_at < approval.approved_at
        or request.observed_at >= approval.expires_at
    ):
        raise PreflightRejected("external-call approval is outside its active interval")


def _validate_benchmark_approval(
    request: RunPreflightRequest,
    budget: BudgetSpecV2,
    runtime: MemorySystemRuntimeBindingV2,
    selected_roles: tuple[ModelRoleBindingV2, ...],
) -> ExternalCallApprovalRecord:
    approval = request.approval
    if not isinstance(approval, ExternalCallApprovalRecord):
        raise PreflightRejected("external benchmark requires an approval record")
    _require_active_approval(request, approval)
    if approval.operation_kind != OperationKind.BENCHMARK_RUN.value:
        raise PreflightRejected("approval operation does not match benchmark_run")
    if approval.scope_kind != BudgetScopeKindV2.RUN or budget.scope_kind != BudgetScopeKindV2.RUN:
        raise PreflightRejected("benchmark run requires run-scoped approval and budget")
    if approval.scope_id != budget.scope_id or approval.approval_id != budget.approval_id:
        raise PreflightRejected("approval and budget scope do not match")
    if approval.runtime_binding_hash != runtime.runtime_binding_hash:
        raise PreflightRejected("approval does not bind the selected runtime")
    selected_ids = tuple(role.binding_id for role in selected_roles)
    if approval.role_binding_ids != selected_ids:
        raise PreflightRejected("approval does not bind the selected role set")
    if approval.stop_condition_ids != budget.stop_condition_ids:
        raise PreflightRejected("approval and budget stop conditions do not match")
    expected_budget_hash = canonical_sha256(budget.model_dump(mode="python"))
    if approval.budget_hash != expected_budget_hash:
        raise PreflightRejected("approval does not bind the exact budget bytes")
    return approval


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
        raise PreflightRejected("runtime model roles do not match selected memory-owned roles")
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


def _validate_model_readiness_approval(
    request: RunPreflightRequest,
    budget: BudgetSpecV2,
    attestation: ProviderRuntimeProfileAttestation,
    selected_roles: tuple[ModelRoleBindingV2, ...],
) -> ExternalCallApprovalRecord:
    approval = request.approval
    if not isinstance(approval, ExternalCallApprovalRecord):
        raise PreflightRejected("model readiness requires an approval record")
    _require_active_approval(request, approval)
    if approval.operation_kind != OperationKind.MODEL_READINESS.value:
        raise PreflightRejected("approval operation does not match model_readiness")
    if (
        approval.scope_kind != BudgetScopeKindV2.MODEL_READINESS
        or budget.scope_kind != BudgetScopeKindV2.MODEL_READINESS
    ):
        raise PreflightRejected("model readiness requires its own occurrence-scoped budget")
    if approval.scope_id != budget.scope_id or approval.approval_id != budget.approval_id:
        raise PreflightRejected("approval and budget scope do not match")
    if approval.provider_runtime_profile_attestation_hash != attestation.attestation_hash:
        raise PreflightRejected("approval does not bind the pre-readiness attestation")
    selected_ids = tuple(role.binding_id for role in selected_roles)
    if approval.role_binding_ids != selected_ids:
        raise PreflightRejected("approval does not bind the selected readiness roles")
    if approval.stop_condition_ids != budget.stop_condition_ids:
        raise PreflightRejected("approval and budget stop conditions do not match")
    expected_budget_hash = canonical_sha256(budget.model_dump(mode="python"))
    if approval.budget_hash != expected_budget_hash:
        raise PreflightRejected("approval does not bind the exact budget bytes")
    if not approval.unmetered_cost_acknowledged:
        raise PreflightRejected("model-readiness approval must acknowledge unavailable billing")
    return approval


def _validate_phase_review_approval(
    request: RunPreflightRequest,
    budget: BudgetSpecV2,
    selected_roles: tuple[ModelRoleBindingV2, ...],
) -> ExternalCallApprovalRecord:
    approval = request.approval
    if not isinstance(approval, ExternalCallApprovalRecord):
        raise PreflightRejected("phase review requires an approval record")
    _require_active_approval(request, approval)
    if approval.operation_kind != OperationKind.PHASE_REVIEW.value:
        raise PreflightRejected("approval operation does not match phase_review")
    if (
        approval.scope_kind != BudgetScopeKindV2.PHASE_REVIEW
        or budget.scope_kind != BudgetScopeKindV2.PHASE_REVIEW
    ):
        raise PreflightRejected("phase review requires its own occurrence-scoped budget")
    if approval.scope_id != budget.scope_id or approval.approval_id != budget.approval_id:
        raise PreflightRejected("approval and budget scope do not match")
    selected_ids = tuple(role.binding_id for role in selected_roles)
    if approval.role_binding_ids != selected_ids:
        raise PreflightRejected("approval does not bind the selected phase-review role")
    if approval.stop_condition_ids != budget.stop_condition_ids:
        raise PreflightRejected("approval and budget stop conditions do not match")
    expected_budget_hash = canonical_sha256(budget.model_dump(mode="python"))
    if approval.budget_hash != expected_budget_hash:
        raise PreflightRejected("approval does not bind the exact budget bytes")
    return approval


def _schema_versions(request: RunPreflightRequest) -> tuple[int, ...]:
    records = (
        request.budget_spec,
        *(slot.binding for slot in request.role_slots if isinstance(slot, RoleSlot)),
        request.runtime_binding,
        request.provider_runtime_attestation,
        request.approval,
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
    if request.operation_kind != OperationKind.PHASE_REVIEW:
        if not isinstance(request.adapter_profile, AdapterProfileDescriptor):
            raise PreflightRejected("run preflight requires an adapter profile descriptor")
        profile = validate_adapter_profile(request.adapter_profile)
        if not isinstance(request.provider_gates, ProviderGateClosure):
            raise PreflightRejected("run preflight requires provider gate closure")
        gates = request.provider_gates
    elif request.adapter_profile is not None or request.provider_gates is not None:
        raise PreflightRejected("phase review has no memory adapter profile or provider gates")

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
        if not isinstance(request.budget_spec, BudgetSpecV2):
            raise PreflightRejected("external operations require BudgetSpec version 2")
        selected_roles = _validate_external_roles(slots, request.environment)
        _validate_controlled_embedding_binding(profile, selected_roles)
        if slots[-1].status != ResolutionStatus.UNSELECTED:
            raise PreflightRejected("benchmark quality-review role must remain unselected")
        _validate_budget_role_closure(request.budget_spec, selected_roles)
        runtime = _validate_benchmark_runtime(
            request.runtime_binding,
            profile,
            selected_roles,
        )
        _validate_benchmark_approval(request, request.budget_spec, runtime, selected_roles)
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
        attestation = _validate_model_readiness_attestation(
            request.provider_runtime_attestation,
            profile,
            selected_roles,
            gates,
        )
        _validate_model_readiness_approval(
            request,
            request.budget_spec,
            attestation,
            selected_roles,
        )
        if request.cost_measurement_spec is None:
            raise PreflightRejected("model readiness requires a cost measurement specification")
        _validate_price_closure(request.budget_spec, request.price_snapshot)
    elif request.operation_kind == OperationKind.PHASE_REVIEW:
        if not isinstance(request.budget_spec, BudgetSpecV2):
            raise PreflightRejected("external operations require BudgetSpec version 2")
        selected_roles = _validate_external_roles(slots, request.environment)
        if tuple(role.role for role in selected_roles) != (ModelRole.QUALITY_REVIEW,):
            raise PreflightRejected("phase review selects only the quality-review role")
        _validate_budget_role_closure(request.budget_spec, selected_roles)
        if request.runtime_binding is not None or request.provider_runtime_attestation is not None:
            raise PreflightRejected("phase review has no memory-system runtime binding")
        _validate_phase_review_approval(request, request.budget_spec, selected_roles)
        if request.cost_measurement_spec is None:
            raise PreflightRejected("phase review requires a cost measurement specification")
        if not isinstance(request.execution_environment_binding, ExecutionEnvironmentBinding):
            raise PreflightRejected("phase review requires an execution environment binding")
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
        approval=request.approval,
        cost_measurement_spec=request.cost_measurement_spec,
        price_snapshot=request.price_snapshot,
        execution_environment_binding=request.execution_environment_binding,
        schema_versions=_schema_versions(request),
        plan_hash=canonical_sha256(_plan_hash_payload(request, slots, profile, gates)),
    )
