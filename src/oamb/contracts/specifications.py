"""Strict control-plane and manifest contracts used by the offline kernel."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import model_validator

from .base import (
    NonEmptyStr,
    NonNegativeDecimal,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictContract,
    UtcDateTime,
)
from .ids import (
    canonical_sha256,
    case_manifest_entry_id,
    ingestion_payload_hash,
    ingestion_plan_id,
    plan_manifest_entry_id,
)


class ValidationStage(StrEnum):
    EVIDENCE = "evidence"
    EXPORT = "export"


class BudgetScopeKind(StrEnum):
    RUN = "run"
    PHASE_REVIEW = "phase_review"


class BudgetScopeKindV2(StrEnum):
    RUN = "run"
    PHASE_REVIEW = "phase_review"
    MODEL_READINESS = "model_readiness"


class TransportProfile(StrEnum):
    REST = "rest"
    SDK = "sdk"
    FAKE = "fake"


class AttestationStatus(StrEnum):
    ATTESTED = "attested"
    UNATTESTED = "unattested"
    UNSUPPORTED = "unsupported"


class ComparabilityStatus(StrEnum):
    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"
    UNKNOWN = "unknown"


class ModelRole(StrEnum):
    MEMORY_EXTRACTION = "memory_extraction"
    EMBEDDING = "embedding"
    ANSWER = "answer"
    JUDGE = "judge"
    QUALITY_REVIEW = "quality_review"


class RoleBindingStatus(StrEnum):
    SELECTED = "selected"
    DISABLED = "disabled"
    NOT_APPLICABLE = "not_applicable"


class ExecutionOwner(StrEnum):
    MEMORY_SYSTEM = "memory_system"
    HARNESS = "harness"
    DETERMINISTIC_METRIC = "deterministic_metric"


class BindingKind(StrEnum):
    NATIVE = "native"
    MODEL_CLIENT = "model_client"
    DISABLED = "disabled"
    NOT_APPLICABLE = "not_applicable"


class RuntimeAttestationStatus(StrEnum):
    RUNTIME_VERIFIED = "runtime_verified"
    BUILD_PROVENANCE_VERIFIED = "build_provenance_verified"
    UNATTESTED = "unattested"
    UNSUPPORTED = "unsupported"


class ProviderGateStatus(StrEnum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"


class SourceEvidenceKind(StrEnum):
    RUN = "run"
    EXTERNAL = "external"
    DERIVATION = "derivation"
    PROVIDER_SERVICE = "provider_service"


class SourceEvidenceBinding(StrictContract):
    schema_name: Literal["source_evidence_binding"] = "source_evidence_binding"
    schema_version: Literal[1] = 1
    binding_id: Sha256
    source_kind: SourceEvidenceKind
    source_identity: NonEmptyStr
    source_root_hash: Sha256
    validation_result_hash: Sha256
    source_schema_versions: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def schema_versions_are_unique(self) -> Self:
        if not self.source_schema_versions:
            raise ValueError("source evidence binding requires schema versions")
        if len(set(self.source_schema_versions)) != len(self.source_schema_versions):
            raise ValueError("source evidence binding contains duplicate schema versions")
        return self


class ProtocolSpec(StrictContract):
    schema_name: Literal["protocol_spec"] = "protocol_spec"
    schema_version: Literal[1] = 1
    protocol_id: NonEmptyStr
    protocol_version: NonEmptyStr
    stages: tuple[NonEmptyStr, ...]
    lifecycle_version: NonEmptyStr
    context_policy: NonEmptyStr
    output_contract_ids: tuple[NonEmptyStr, ...]
    metric_ids: tuple[NonEmptyStr, ...]
    comparison_rule_ids: tuple[NonEmptyStr, ...]


class DatasetFile(StrictContract):
    schema_name: Literal["dataset_file"] = "dataset_file"
    schema_version: Literal[1] = 1
    relative_path: NonEmptyStr
    sha256: Sha256
    byte_count: NonNegativeInt
    license_id: NonEmptyStr


class DatasetManifest(StrictContract):
    schema_name: Literal["dataset_manifest"] = "dataset_manifest"
    schema_version: Literal[1] = 1
    dataset_id: NonEmptyStr
    revision: NonEmptyStr
    split: NonEmptyStr
    manifest_hash: Sha256
    source_files: tuple[DatasetFile, ...]
    payload_policy: NonEmptyStr


class LogicalContextManifestEntry(StrictContract):
    schema_name: Literal["logical_context_manifest_entry"] = "logical_context_manifest_entry"
    schema_version: Literal[1] = 1
    context_content_id: Sha256
    context_manifest_entry_id: Sha256
    source_file_sha256: Sha256
    source_row_number_1_indexed: PositiveInt
    context_bytes_sha256: Sha256


class CaseManifestEntry(StrictContract):
    schema_name: Literal["case_manifest_entry"] = "case_manifest_entry"
    schema_version: Literal[1] = 1
    case_manifest_entry_id: Sha256
    context_manifest_entry_id: Sha256
    source_question_number_1_indexed: PositiveInt
    question_bytes_sha256: Sha256
    raw_question_id: NonEmptyStr
    answer_value_sha256: tuple[Sha256, ...]

    @model_validator(mode="after")
    def identity_matches_content(self) -> Self:
        expected = case_manifest_entry_id(
            self.context_manifest_entry_id,
            self.source_question_number_1_indexed,
            self.question_bytes_sha256,
            self.raw_question_id,
        )
        if self.case_manifest_entry_id != expected:
            raise ValueError("case manifest entry identity does not match its content")
        return self


class IngestionPlanManifest(StrictContract):
    schema_name: Literal["ingestion_plan_manifest"] = "ingestion_plan_manifest"
    schema_version: Literal[1] = 1
    plan_manifest_entry_id: Sha256
    ingestion_payload_hash: Sha256
    ingestion_plan_id: Sha256
    workload_id: NonEmptyStr
    ordered_member_context_manifest_entry_ids: tuple[Sha256, ...]
    ordered_source_unit_bytes_sha256: tuple[Sha256, ...]
    ordered_case_manifest_entry_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def non_empty_members(self) -> Self:
        if not self.ordered_member_context_manifest_entry_ids:
            raise ValueError("ingestion plan requires a logical member")
        if not self.ordered_source_unit_bytes_sha256:
            raise ValueError("ingestion plan requires a source unit")
        expected_manifest_id = plan_manifest_entry_id(
            self.workload_id, self.ordered_member_context_manifest_entry_ids
        )
        expected_payload_hash = ingestion_payload_hash(self.ordered_source_unit_bytes_sha256)
        expected_plan_id = ingestion_plan_id(expected_manifest_id, expected_payload_hash)
        if self.plan_manifest_entry_id != expected_manifest_id:
            raise ValueError("plan manifest entry identity does not match its members")
        if self.ingestion_payload_hash != expected_payload_hash:
            raise ValueError("ingestion payload hash does not match its source units")
        if self.ingestion_plan_id != expected_plan_id:
            raise ValueError("ingestion plan identity does not match its manifest and payload")
        return self


class CaseManifest(StrictContract):
    schema_name: Literal["case_manifest"] = "case_manifest"
    schema_version: Literal[1] = 1
    manifest_id: NonEmptyStr
    manifest_hash: Sha256
    workload_id: NonEmptyStr
    logical_contexts: tuple[LogicalContextManifestEntry, ...]
    ingestion_plans: tuple[IngestionPlanManifest, ...]
    cases: tuple[CaseManifestEntry, ...]

    @model_validator(mode="after")
    def unique_and_closed_parentage(self) -> Self:
        context_ids = tuple(item.context_manifest_entry_id for item in self.logical_contexts)
        case_ids = tuple(item.case_manifest_entry_id for item in self.cases)
        plan_ids = tuple(item.ingestion_plan_id for item in self.ingestion_plans)
        if len(set(context_ids)) != len(context_ids):
            raise ValueError("duplicate logical context identity")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("duplicate case identity")
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("duplicate ingestion plan identity")
        context_set = set(context_ids)
        case_set = set(case_ids)
        flattened_context_ids: list[str] = []
        flattened_case_ids: list[str] = []
        cases_by_id = {item.case_manifest_entry_id: item for item in self.cases}
        for plan in self.ingestion_plans:
            if plan.workload_id != self.workload_id:
                raise ValueError("ingestion plan belongs to a different workload")
            if not set(plan.ordered_member_context_manifest_entry_ids) <= context_set:
                raise ValueError("ingestion plan references an unknown logical context")
            if not set(plan.ordered_case_manifest_entry_ids) <= case_set:
                raise ValueError("ingestion plan references an unknown case")
            member_ids = set(plan.ordered_member_context_manifest_entry_ids)
            for referenced_case_id in plan.ordered_case_manifest_entry_ids:
                if cases_by_id[referenced_case_id].context_manifest_entry_id not in member_ids:
                    raise ValueError("case context does not belong to its ingestion plan")
            flattened_context_ids.extend(plan.ordered_member_context_manifest_entry_ids)
            flattened_case_ids.extend(plan.ordered_case_manifest_entry_ids)
        if (
            len(flattened_context_ids) != len(context_ids)
            or set(flattened_context_ids) != context_set
        ):
            raise ValueError("every logical context must belong to exactly once ingestion plan")
        if len(flattened_case_ids) != len(case_ids) or set(flattened_case_ids) != case_set:
            raise ValueError("every case must belong to exactly once ingestion plan")
        return self


class InteractionSpec(StrictContract):
    schema_name: Literal["interaction_spec"] = "interaction_spec"
    schema_version: Literal[1] = 1
    interaction_id: NonEmptyStr
    interaction_version: NonEmptyStr
    chunking_id: NonEmptyStr
    ingestion_mapping_id: NonEmptyStr
    retrieval_top_k: PositiveInt
    normalizer_id: NonEmptyStr
    query_effect_policy: NonEmptyStr
    prompt_pack_ids: tuple[NonEmptyStr, ...]
    output_contract_ids: tuple[NonEmptyStr, ...]


class WorkloadSpec(StrictContract):
    schema_name: Literal["workload_spec"] = "workload_spec"
    schema_version: Literal[1] = 1
    workload_id: NonEmptyStr
    workload_version: NonEmptyStr
    dataset_manifest_hash: Sha256
    case_manifest_hash: Sha256
    interaction_spec_id: NonEmptyStr
    prompt_pack_ids: tuple[NonEmptyStr, ...]
    metric_ids: tuple[NonEmptyStr, ...]


class MemorySystemSpec(StrictContract):
    schema_name: Literal["memory_system_spec"] = "memory_system_spec"
    schema_version: Literal[1] = 1
    memory_system_id: NonEmptyStr
    adapter_id: NonEmptyStr
    adapter_revision: NonEmptyStr
    transport_profile: TransportProfile
    capabilities: tuple[NonEmptyStr, ...]
    native_configuration_fingerprint: Sha256


class MemorySystemRuntimeBinding(StrictContract):
    schema_name: Literal["memory_system_runtime_binding"] = "memory_system_runtime_binding"
    schema_version: Literal[1] = 1
    memory_system_id: NonEmptyStr
    runtime_binding_hash: Sha256
    release_version: NonEmptyStr
    artifact_sha256: Sha256
    storage_engine: NonEmptyStr
    model_role_binding_ids: tuple[NonEmptyStr, ...]
    attestation_status: AttestationStatus


class MemorySystemRuntimeBindingV2(StrictContract):
    schema_name: Literal["memory_system_runtime_binding"] = "memory_system_runtime_binding"
    schema_version: Literal[2] = 2
    memory_system_id: NonEmptyStr
    runtime_binding_hash: Sha256
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    edition: NonEmptyStr
    distribution_channel: NonEmptyStr
    api_version: NonEmptyStr
    release_version: NonEmptyStr
    source_revision: NonEmptyStr
    artifact_kind: NonEmptyStr
    artifact_sha256: Sha256
    endpoint_fingerprint: Sha256
    deployment_configuration_sha256: Sha256
    storage_engine: NonEmptyStr
    storage_engine_version: NonEmptyStr
    schema_revision: NonEmptyStr
    vector_index_type: NonEmptyStr
    distance_metric: NonEmptyStr
    vector_dimension: PositiveInt
    index_configuration_sha256: Sha256
    model_role_binding_ids: tuple[NonEmptyStr, ...]
    native_feature_flags_fingerprint: Sha256
    native_reranking_status: Literal["disabled"]
    attestation_method: NonEmptyStr
    attestation_status: RuntimeAttestationStatus
    raw_proof_refs: tuple[Sha256, ...]
    model_readiness_required: bool
    model_readiness_evidence: SourceEvidenceBinding | None

    @model_validator(mode="after")
    def model_readiness_binding_is_closed(self) -> Self:
        if not self.model_role_binding_ids:
            raise ValueError("runtime binding requires model role bindings")
        if len(set(self.model_role_binding_ids)) != len(self.model_role_binding_ids):
            raise ValueError("runtime binding contains duplicate model role bindings")
        if not self.raw_proof_refs:
            raise ValueError("runtime binding requires raw proof references")
        if self.model_readiness_required:
            if self.model_readiness_evidence is None:
                raise ValueError("required model readiness evidence is missing")
            if self.model_readiness_evidence.source_kind != SourceEvidenceKind.PROVIDER_SERVICE:
                raise ValueError("model readiness evidence must bind provider-service evidence")
        elif self.model_readiness_evidence is not None:
            raise ValueError("optional model readiness cannot carry an evidence binding")
        expected_hash = memory_system_runtime_binding_hash(
            self.model_dump(mode="python", exclude={"runtime_binding_hash"})
        )
        if self.runtime_binding_hash != expected_hash:
            raise ValueError("runtime binding hash does not match its canonical fields")
        return self


def memory_system_runtime_binding_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "memory_system_runtime_binding")
    payload.setdefault("schema_version", 2)
    payload.pop("runtime_binding_hash", None)
    return canonical_sha256(payload)


class ExecutionEnvironmentBinding(StrictContract):
    schema_name: Literal["execution_environment_binding"] = "execution_environment_binding"
    schema_version: Literal[1] = 1
    environment_hash: Sha256
    operating_system: NonEmptyStr
    architecture: NonEmptyStr
    python_version: NonEmptyStr
    cpu_description: NonEmptyStr
    memory_bytes: PositiveInt
    comparability_status: ComparabilityStatus


class ModelRoleBinding(StrictContract):
    schema_name: Literal["model_role_binding"] = "model_role_binding"
    schema_version: Literal[1] = 1
    binding_id: NonEmptyStr
    role: NonEmptyStr
    owner: NonEmptyStr
    endpoint_fingerprint: Sha256
    configured_model: NonEmptyStr
    resolved_model: NonEmptyStr
    parameters_fingerprint: Sha256
    budget_role: NonEmptyStr


class ModelRoleBindingV2(StrictContract):
    schema_name: Literal["model_role_binding"] = "model_role_binding"
    schema_version: Literal[2] = 2
    binding_id: NonEmptyStr
    role: ModelRole
    role_status: RoleBindingStatus
    execution_owner: ExecutionOwner
    binding_kind: BindingKind
    provider: NonEmptyStr | None
    endpoint_reference: NonEmptyStr | None
    credential_variable_name: NonEmptyStr | None
    configured_model: NonEmptyStr | None
    resolved_model: NonEmptyStr | None
    parameters_fingerprint: Sha256 | None
    retry_policy_id: NonEmptyStr | None
    configuration_fingerprint: Sha256
    redacted_endpoint_fingerprint: Sha256 | None

    @model_validator(mode="after")
    def binding_status_matches_shape(self) -> Self:
        selected_fields = (
            self.provider,
            self.endpoint_reference,
            self.configured_model,
            self.resolved_model,
            self.parameters_fingerprint,
            self.retry_policy_id,
            self.redacted_endpoint_fingerprint,
        )
        if self.role_status == RoleBindingStatus.SELECTED:
            if self.binding_kind not in {BindingKind.NATIVE, BindingKind.MODEL_CLIENT}:
                raise ValueError("selected binding requires native or model_client kind")
            if any(value is None for value in selected_fields):
                raise ValueError("selected binding requires resolved provider and model fields")
        else:
            expected_kind = (
                BindingKind.DISABLED
                if self.role_status == RoleBindingStatus.DISABLED
                else BindingKind.NOT_APPLICABLE
            )
            if self.binding_kind != expected_kind:
                raise ValueError("unselected binding status and kind must match")
            if any(value is not None for value in selected_fields):
                raise ValueError("unselected binding cannot carry provider or model fields")
            if self.credential_variable_name is not None:
                raise ValueError("unselected binding cannot name a credential variable")
        if self.execution_owner == ExecutionOwner.DETERMINISTIC_METRIC:
            if self.role != ModelRole.JUDGE or self.role_status != RoleBindingStatus.NOT_APPLICABLE:
                raise ValueError("deterministic metric is only valid for a not-applicable judge")
        return self


class ResourceBudgetCeiling(StrictContract):
    schema_name: Literal["resource_budget_ceiling"] = "resource_budget_ceiling"
    schema_version: Literal[1] = 1
    dimension_id: NonEmptyStr
    maximum: NonNegativeDecimal
    unit: NonEmptyStr


class ProviderBudgetCap(StrictContract):
    schema_name: Literal["provider_budget_cap"] = "provider_budget_cap"
    schema_version: Literal[1] = 1
    provider: NonEmptyStr
    operation_kind: NonEmptyStr
    billing_unit: NonEmptyStr
    maximum_accepted_units: NonNegativeDecimal


class RoleBudgetCeiling(StrictContract):
    schema_name: Literal["role_budget_ceiling"] = "role_budget_ceiling"
    schema_version: Literal[1] = 1
    role_binding_id: NonEmptyStr
    max_attempts: PositiveInt
    max_input_tokens: NonNegativeInt
    max_output_tokens: NonNegativeInt
    max_dispatch_wall_seconds: NonNegativeDecimal
    max_cost: NonNegativeDecimal | None
    currency: str | None
    price_snapshot_id: NonEmptyStr | None
    resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    provider_budget_cap: ProviderBudgetCap

    @model_validator(mode="after")
    def currency_and_resources_close(self) -> Self:
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        if self.price_snapshot_id is not None and self.max_cost is None:
            raise ValueError("price snapshot requires a cost ceiling")
        resource_ids = tuple(item.dimension_id for item in self.resource_ceilings)
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("role ceiling contains duplicate resource dimensions")
        return self


class BudgetSpec(StrictContract):
    schema_name: Literal["budget_spec"] = "budget_spec"
    schema_version: Literal[1] = 1
    budget_id: NonEmptyStr
    scope_kind: BudgetScopeKind
    scope_id: NonEmptyStr
    approval_id: NonEmptyStr | None
    max_attempts: NonNegativeInt
    max_input_tokens: NonNegativeInt
    max_output_tokens: NonNegativeInt
    max_wall_seconds: NonNegativeDecimal
    max_cost: NonNegativeDecimal | None
    currency: str | None

    @model_validator(mode="after")
    def currency_matches_cost(self) -> Self:
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        return self


class BudgetSpecV2(StrictContract):
    schema_name: Literal["budget_spec"] = "budget_spec"
    schema_version: Literal[2] = 2
    budget_id: NonEmptyStr
    scope_kind: BudgetScopeKindV2
    scope_id: NonEmptyStr
    approval_id: NonEmptyStr
    max_attempts: PositiveInt
    max_input_tokens: NonNegativeInt
    max_output_tokens: NonNegativeInt
    max_dispatch_wall_seconds: NonNegativeDecimal
    max_cost: NonNegativeDecimal | None
    currency: str | None
    resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    role_ceilings: tuple[RoleBudgetCeiling, ...]
    stop_condition_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def parent_and_role_ceilings_close(self) -> Self:
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        if not self.role_ceilings:
            raise ValueError("version 2 budget requires at least one role ceiling")
        role_ids = tuple(item.role_binding_id for item in self.role_ceilings)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("duplicate role ceiling")
        resource_ids = tuple(item.dimension_id for item in self.resource_ceilings)
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("budget contains duplicate resource dimensions")
        if len(set(self.stop_condition_ids)) != len(self.stop_condition_ids):
            raise ValueError("budget contains duplicate stop conditions")
        if any(ceiling.currency != self.currency for ceiling in self.role_ceilings):
            raise ValueError("role budget currency must match the parent currency")
        if self.max_cost is not None and any(
            ceiling.max_cost is None or ceiling.max_cost > self.max_cost
            for ceiling in self.role_ceilings
        ):
            raise ValueError("role cost ceiling must fit the parent cost ceiling")
        return self


class ProviderRuntimeProfileAttestation(StrictContract):
    schema_name: Literal["provider_runtime_profile_attestation"] = (
        "provider_runtime_profile_attestation"
    )
    schema_version: Literal[1] = 1
    provider: NonEmptyStr
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    attestation_hash: Sha256
    transport_profile: TransportProfile
    release_version: NonEmptyStr
    source_revision: NonEmptyStr
    build_artifact_sha256: Sha256
    redacted_configuration_sha256: Sha256
    redacted_endpoint_fingerprint: Sha256
    auth_configuration_sha256: Sha256
    storage_configuration_sha256: Sha256
    model_role_binding_ids: tuple[NonEmptyStr, ...]
    native_reranking_status: Literal["disabled"]
    liveness_status: ProviderGateStatus
    storage_configuration_status: ProviderGateStatus
    runtime_identity_status: ProviderGateStatus
    model_readiness_status: ProviderGateStatus
    memory_conformance_status: ProviderGateStatus
    raw_proof_refs: tuple[Sha256, ...]

    @model_validator(mode="after")
    def remains_pre_readiness(self) -> Self:
        expected_hash = provider_runtime_profile_attestation_hash(
            self.model_dump(mode="python", exclude={"attestation_hash"})
        )
        if self.attestation_hash != expected_hash:
            raise ValueError("attestation hash does not match its canonical fields")
        if self.model_readiness_status != ProviderGateStatus.NOT_RUN:
            raise ValueError("pre-readiness attestation requires model readiness NOT_RUN")
        if self.memory_conformance_status != ProviderGateStatus.NOT_RUN:
            raise ValueError("pre-readiness attestation requires memory conformance NOT_RUN")
        if not self.model_role_binding_ids:
            raise ValueError("pre-readiness attestation requires selected model roles")
        if len(set(self.model_role_binding_ids)) != len(self.model_role_binding_ids):
            raise ValueError("pre-readiness attestation contains duplicate model roles")
        if not self.raw_proof_refs:
            raise ValueError("pre-readiness attestation requires raw proof references")
        return self


class ExternalCallApprovalRecord(StrictContract):
    schema_name: Literal["external_call_approval_record"] = "external_call_approval_record"
    schema_version: Literal[1] = 1
    approval_id: NonEmptyStr
    approval_hash: Sha256
    operation_kind: NonEmptyStr
    scope_kind: BudgetScopeKindV2
    scope_id: NonEmptyStr
    runtime_binding_hash: Sha256 | None
    provider_runtime_profile_attestation_hash: Sha256 | None
    role_binding_ids: tuple[NonEmptyStr, ...]
    budget_hash: Sha256
    approved_at: UtcDateTime
    expires_at: UtcDateTime
    unmetered_cost_acknowledged: bool
    stop_condition_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def scope_runtime_and_expiry_close(self) -> Self:
        expected_hash = external_call_approval_hash(
            self.model_dump(mode="python", exclude={"approval_hash"})
        )
        if self.approval_hash != expected_hash:
            raise ValueError("approval hash does not match its canonical fields")
        if self.expires_at <= self.approved_at:
            raise ValueError("approval expiry must follow approval time")
        if not self.role_binding_ids:
            raise ValueError("approval requires at least one role binding")
        if len(set(self.role_binding_ids)) != len(self.role_binding_ids):
            raise ValueError("approval contains duplicate role bindings")
        if len(set(self.stop_condition_ids)) != len(self.stop_condition_ids):
            raise ValueError("approval contains duplicate stop conditions")
        if self.scope_kind == BudgetScopeKindV2.RUN:
            if self.runtime_binding_hash is None:
                raise ValueError("run approval requires a runtime binding")
            if self.provider_runtime_profile_attestation_hash is not None:
                raise ValueError("run approval cannot use a pre-readiness attestation")
        elif self.scope_kind == BudgetScopeKindV2.MODEL_READINESS:
            if self.provider_runtime_profile_attestation_hash is None:
                raise ValueError("model-readiness approval requires an attestation")
            if self.runtime_binding_hash is not None:
                raise ValueError("model-readiness approval cannot use a runtime binding")
        elif (
            self.runtime_binding_hash is not None
            or self.provider_runtime_profile_attestation_hash is not None
        ):
            raise ValueError("phase-review approval has no memory-system runtime binding")
        return self


def provider_runtime_profile_attestation_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "provider_runtime_profile_attestation")
    payload.setdefault("schema_version", 1)
    payload.pop("attestation_hash", None)
    return canonical_sha256(payload)


def external_call_approval_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "external_call_approval_record")
    payload.setdefault("schema_version", 1)
    payload.pop("approval_hash", None)
    return canonical_sha256(payload)


class DerivationSpec(StrictContract):
    schema_name: Literal["derivation_spec"] = "derivation_spec"
    schema_version: Literal[1] = 1
    derivation_kind: NonEmptyStr
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    ordered_source_root_hash: Sha256
    transform_spec_hash: Sha256
    report_spec_hash: Sha256 | None
    reducer_and_renderer_input_hashes: tuple[Sha256, ...]
    derivation_input_hash: Sha256

    @model_validator(mode="after")
    def source_bindings_are_non_empty_and_unique(self) -> Self:
        if not self.ordered_source_bindings:
            raise ValueError("derivation requires at least one source binding")
        binding_ids = tuple(item.binding_id for item in self.ordered_source_bindings)
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("derivation contains duplicate source bindings")
        return self


class RunSpec(StrictContract):
    schema_name: Literal["run_spec"] = "run_spec"
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    protocol_id: NonEmptyStr
    dataset_manifest_hash: Sha256
    case_manifest_hash: Sha256
    workload_id: NonEmptyStr
    memory_system_id: NonEmptyStr
    runtime_binding_hash: Sha256
    environment_hash: Sha256
    model_role_binding_ids: tuple[NonEmptyStr, ...]
    budget_id: NonEmptyStr
    code_revision: NonEmptyStr
    normalizer_fingerprint: Sha256


class ValidationRuleRequirement(StrictContract):
    schema_name: Literal["validation_rule_requirement"] = "validation_rule_requirement"
    schema_version: Literal[1] = 1
    rule_id: NonEmptyStr
    minimum_version: PositiveInt


class ValidationProfile(StrictContract):
    schema_name: Literal["validation_profile"] = "validation_profile"
    schema_version: Literal[1] = 1
    profile_id: NonEmptyStr
    stage: ValidationStage
    required_rules: tuple[ValidationRuleRequirement, ...]
    applicability: tuple[NonEmptyStr, ...]
    required_rule_inventory_hash: Sha256

    @classmethod
    def create(
        cls,
        *,
        profile_id: str,
        stage: ValidationStage,
        required_rules: tuple[ValidationRuleRequirement, ...],
        applicability: tuple[str, ...],
    ) -> Self:
        inventory = tuple(
            (requirement.rule_id, requirement.minimum_version) for requirement in required_rules
        )
        return cls(
            profile_id=profile_id,
            stage=stage,
            required_rules=required_rules,
            applicability=applicability,
            required_rule_inventory_hash=canonical_sha256(
                ["oamb-required-rule-inventory-v1", inventory]
            ),
        )

    @model_validator(mode="after")
    def unique_requirements(self) -> Self:
        rule_ids = tuple(item.rule_id for item in self.required_rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("validation profile contains duplicate rule IDs")
        return self
