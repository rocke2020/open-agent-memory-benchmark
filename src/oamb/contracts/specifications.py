"""Strict control-plane and manifest contracts used by the offline kernel."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from .base import (
    NonEmptyStr,
    NonNegativeDecimal,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictContract,
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
