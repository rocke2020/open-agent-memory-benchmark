"""Strict control-plane and manifest contracts used by the offline kernel."""

from __future__ import annotations

import re
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

INGESTION_RETRY_UNIT = "batch_dispatch"
INGESTION_RECOVERY_STRATEGY = "same_scope_batch_retry_skip_v1"
INFRASTRUCTURE_RETRY_BACKOFF_SECONDS = (1, 2)
INFRASTRUCTURE_MAX_TOTAL_RETRIES = len(INFRASTRUCTURE_RETRY_BACKOFF_SECONDS)
INFRASTRUCTURE_RETRY_POLICY_HASH = canonical_sha256(
    [
        "oamb-infrastructure-retry-policy-v2",
        INFRASTRUCTURE_RETRY_BACKOFF_SECONDS,
        INFRASTRUCTURE_MAX_TOTAL_RETRIES,
        INGESTION_RETRY_UNIT,
        INGESTION_RECOVERY_STRATEGY,
    ]
)


class ValidationStage(StrEnum):
    EVIDENCE = "evidence"
    EXPORT = "export"


class BudgetScopeKind(StrEnum):
    RUN = "run"


class BudgetScopeKindV2(StrEnum):
    RUN = "run"
    MODEL_READINESS = "model_readiness"


class BudgetScopeKindV3(StrEnum):
    RUN = "run"
    MODEL_READINESS = "model_readiness"
    MEMORY_CONFORMANCE = "memory_conformance"


class DispatchBudgetOwnerKind(StrEnum):
    MODEL_ROLE = "model_role"
    PROVIDER_OPERATION = "provider_operation"


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


GenerativeThinkingEffort = Literal["low", "high", "max"]
ThinkingEffort = Literal["low", "high", "max", "not_applicable"]


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


class ComparisonCostView(StrEnum):
    NONE = "none"
    RAW_RESOURCE = "raw_resource"
    ACTUAL_CHARGE = "actual_charge"
    DECLARED_RATE_ESTIMATE = "declared_rate_estimate"


class ComparisonPairBinding(StrictContract):
    schema_name: Literal["comparison_pair_binding"] = "comparison_pair_binding"
    schema_version: Literal[1] = 1
    pair_id: Sha256
    case_manifest_entry_id: Sha256
    left_case_occurrence_id: Sha256
    right_case_occurrence_id: Sha256
    metric_id: NonEmptyStr

    @model_validator(mode="after")
    def pair_identity_is_canonical(self) -> Self:
        expected = comparison_pair_id(
            case_manifest_entry_id=self.case_manifest_entry_id,
            left_case_occurrence_id=self.left_case_occurrence_id,
            right_case_occurrence_id=self.right_case_occurrence_id,
            metric_id=self.metric_id,
        )
        if self.pair_id != expected:
            raise ValueError("comparison pair identity does not match its canonical payload")
        return self


class ComparisonCostControl(StrictContract):
    schema_name: Literal["comparison_cost_control"] = "comparison_cost_control"
    schema_version: Literal[1] = 1
    view: ComparisonCostView
    dimension_id: NonEmptyStr
    unit: NonEmptyStr
    basis: Literal["raw_resource", "actual_supplier_charge", "estimate_from_measured_usage"]
    currency: str | None
    environment_hash: Sha256
    measurement_spec_hash: Sha256
    price_policy_hash: Sha256 | None
    fx_policy_hash: Sha256 | None

    @model_validator(mode="after")
    def selected_cost_view_has_one_compatible_basis(self) -> Self:
        if self.view == ComparisonCostView.NONE:
            raise ValueError("none cost view cannot contain a cost control")
        if self.view == ComparisonCostView.RAW_RESOURCE:
            if (
                self.basis != "raw_resource"
                or self.currency is not None
                or self.price_policy_hash is not None
                or self.fx_policy_hash is not None
            ):
                raise ValueError("raw-resource comparison cannot contain monetary policy")
        else:
            expected_basis = (
                "actual_supplier_charge"
                if self.view == ComparisonCostView.ACTUAL_CHARGE
                else "estimate_from_measured_usage"
            )
            if self.basis != expected_basis:
                raise ValueError("cost view and monetary basis do not match")
            if self.currency is None or len(self.currency) != 3 or self.price_policy_hash is None:
                raise ValueError("monetary comparison requires currency and price policy")
        return self


class ComparisonWinnerReducer(StrictContract):
    schema_name: Literal["comparison_winner_reducer"] = "comparison_winner_reducer"
    schema_version: Literal[1] = 1
    reducer_id: NonEmptyStr
    reducer_version: PositiveInt
    metric_id: NonEmptyStr
    denominator: PositiveInt
    aggregate_contract_hash: Sha256


class ComparisonSpec(StrictContract):
    schema_name: Literal["comparison_spec"] = "comparison_spec"
    schema_version: Literal[1] = 1
    comparison_spec_id: Sha256
    left_run_id: NonEmptyStr
    right_run_id: NonEmptyStr
    cost_view: ComparisonCostView
    ordered_pair_bindings: tuple[ComparisonPairBinding, ...]
    required_control_ids: tuple[NonEmptyStr, ...]
    comparison_policy_hash: Sha256
    winner_reducer: ComparisonWinnerReducer | None
    cost_control: ComparisonCostControl | None

    @model_validator(mode="after")
    def comparison_scope_and_identity_are_closed(self) -> Self:
        if self.left_run_id == self.right_run_id:
            raise ValueError("comparison requires two distinct runs")
        if not self.ordered_pair_bindings:
            raise ValueError("comparison spec requires an ordered pair inventory")
        pair_ids = tuple(item.pair_id for item in self.ordered_pair_bindings)
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("duplicate comparison pair identity")
        if not self.required_control_ids or len(set(self.required_control_ids)) != len(
            self.required_control_ids
        ):
            raise ValueError("comparison spec requires a unique closed control inventory")
        if (self.cost_view == ComparisonCostView.NONE) != (self.cost_control is None):
            raise ValueError("selected comparison cost view requires exactly one cost control")
        if self.cost_control is not None and self.cost_control.view != self.cost_view:
            raise ValueError("comparison cost control does not match the selected view")
        expected = comparison_spec_id(
            left_run_id=self.left_run_id,
            right_run_id=self.right_run_id,
            cost_view=self.cost_view,
            ordered_pair_bindings=self.ordered_pair_bindings,
            required_control_ids=self.required_control_ids,
            comparison_policy_hash=self.comparison_policy_hash,
            winner_reducer=self.winner_reducer,
            cost_control=self.cost_control,
        )
        if self.comparison_spec_id != expected:
            raise ValueError("comparison spec identity does not match its canonical payload")
        return self


def comparison_pair_id(
    *,
    case_manifest_entry_id: str,
    left_case_occurrence_id: str,
    right_case_occurrence_id: str,
    metric_id: str,
) -> str:
    return canonical_sha256(
        [
            "oamb-comparison-pair-v1",
            case_manifest_entry_id,
            left_case_occurrence_id,
            right_case_occurrence_id,
            metric_id,
        ]
    )


def comparison_spec_id(
    *,
    left_run_id: str,
    right_run_id: str,
    cost_view: ComparisonCostView,
    ordered_pair_bindings: tuple[ComparisonPairBinding, ...],
    required_control_ids: tuple[str, ...],
    comparison_policy_hash: str,
    winner_reducer: ComparisonWinnerReducer | None,
    cost_control: ComparisonCostControl | None,
) -> str:
    return canonical_sha256(
        [
            "oamb-comparison-spec-v1",
            left_run_id,
            right_run_id,
            cost_view,
            ordered_pair_bindings,
            required_control_ids,
            comparison_policy_hash,
            winner_reducer,
            cost_control,
        ]
    )


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


class PromptTemplateManifest(StrictContract):
    schema_name: Literal["prompt_template_manifest"] = "prompt_template_manifest"
    schema_version: Literal[1] = 1
    template_name: NonEmptyStr
    relative_path: NonEmptyStr
    content_sha256: Sha256
    byte_count: PositiveInt
    source_extracted_sha256: Sha256 | None = None
    adaptation_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def source_extraction_and_adaptation_are_paired(self) -> Self:
        if (self.source_extracted_sha256 is None) != (self.adaptation_id is None):
            raise ValueError("source-extracted prompt hash and adaptation ID must be paired")
        return self


def prompt_pack_manifest_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "prompt_pack_manifest")
    payload.setdefault("schema_version", 1)
    payload.pop("manifest_sha256", None)
    return canonical_sha256(payload)


class PromptPackManifest(StrictContract):
    schema_name: Literal["prompt_pack_manifest"] = "prompt_pack_manifest"
    schema_version: Literal[1] = 1
    prompt_pack_id: NonEmptyStr
    prompt_pack_version: NonEmptyStr
    workload_id: NonEmptyStr
    origin: Literal["oamb_authored", "attributed_source", "user_supplied", "generated"]
    source_repository: NonEmptyStr | None
    source_revision: NonEmptyStr | None
    source_file_sha256: Sha256 | None
    license_expression: NonEmptyStr
    rights_attestation: NonEmptyStr
    redistribution_allowed: bool
    templates: tuple[PromptTemplateManifest, ...]
    variables: tuple[NonEmptyStr, ...]
    output_contract_id: NonEmptyStr
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def identity_and_members_are_closed(self) -> Self:
        if not self.templates:
            raise ValueError("prompt pack requires at least one template")
        names = tuple(template.template_name for template in self.templates)
        paths = tuple(template.relative_path for template in self.templates)
        if len(set(names)) != len(names) or len(set(paths)) != len(paths):
            raise ValueError("prompt pack contains duplicate template names or paths")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("prompt pack contains a duplicate variable")
        if not self.variables:
            raise ValueError("prompt pack requires declared variables")
        for variable in self.variables:
            if re.fullmatch(r"[a-z][a-z0-9_]*", variable) is None:
                raise ValueError("prompt variable names require lower snake case")
            if any(
                sensitive in variable
                for sensitive in ("api_key", "secret", "password", "credential", "access_token")
            ):
                raise ValueError("secret values are not valid prompt variables")
        source_values = (
            self.source_repository,
            self.source_revision,
            self.source_file_sha256,
        )
        if self.origin == "attributed_source" and any(value is None for value in source_values):
            raise ValueError("attributed PromptPack requires complete source provenance")
        if self.origin == "attributed_source" and any(
            template.source_extracted_sha256 is None for template in self.templates
        ):
            raise ValueError(
                "attributed PromptPack templates require source-extracted hashes and adaptations"
            )
        if any(value is not None for value in source_values) and any(
            value is None for value in source_values
        ):
            raise ValueError("prompt source repository, revision, and hash must be complete")
        expected = prompt_pack_manifest_hash(
            self.model_dump(mode="python", exclude={"manifest_sha256"})
        )
        if self.manifest_sha256 != expected:
            raise ValueError("prompt pack manifest hash does not match its content")
        return self


class OutputContract(StrictContract):
    schema_name: Literal["output_contract"] = "output_contract"
    schema_version: Literal[1] = 1
    output_contract_id: NonEmptyStr
    representation: Literal["text", "boolean", "ranked_text_list"]
    parser_id: NonEmptyStr
    max_output_tokens: PositiveInt
    required_candidate_count: PositiveInt
    accepted_finish_dispositions: tuple[
        Literal[
            "normal_stop",
            "length_limit",
            "content_filtered",
            "tool_call",
            "missing_finish_reason",
            "cancelled",
            "incomplete_stream",
            "transport_error",
        ],
        ...,
    ]
    parse_failure_policy: Literal["terminal_error", "unjudged"]

    @model_validator(mode="after")
    def finish_dispositions_are_closed(self) -> Self:
        if not self.accepted_finish_dispositions:
            raise ValueError("output contract requires a finish disposition")
        if len(set(self.accepted_finish_dispositions)) != len(self.accepted_finish_dispositions):
            raise ValueError("output contract contains duplicate finish dispositions")
        return self


class MetricSpec(StrictContract):
    schema_name: Literal["metric_spec"] = "metric_spec"
    schema_version: Literal[1] = 1
    metric_id: NonEmptyStr
    output_contract_id: NonEmptyStr
    compatible_output_contract_ids: tuple[NonEmptyStr, ...] = ()
    normalizer_id: NonEmptyStr
    scorer_id: NonEmptyStr
    answer_set_policy: NonEmptyStr
    input_fields: tuple[NonEmptyStr, ...]
    judge_prompt_pack_id: NonEmptyStr | None
    failure_semantics: Literal["unavailable", "terminal_error"]
    unjudged_semantics: Literal["unavailable", "not_applicable", "unjudged"]

    @model_validator(mode="after")
    def evaluation_ownership_is_explicit(self) -> Self:
        if not self.input_fields or len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError("metric input fields must be non-empty and unique")
        if self.judge_prompt_pack_id is not None and "judge" not in self.scorer_id:
            raise ValueError("judge prompt requires a judge scorer")
        output_ids = (self.output_contract_id, *self.compatible_output_contract_ids)
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("metric contains duplicate output contract IDs")
        return self


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


def case_manifest_hash(fields: Mapping[str, Any]) -> str:
    payload = {
        field: fields[field]
        for field in (
            "manifest_id",
            "workload_id",
            "logical_contexts",
            "ingestion_plans",
            "cases",
        )
    }
    return canonical_sha256(["oamb-case-manifest-v1", payload])


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
        expected_hash = case_manifest_hash(self.model_dump(mode="python"))
        if self.manifest_hash != expected_hash:
            raise ValueError("case manifest hash does not match its exact members")
        return self


def case_execution_binding_hash(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-case-execution-binding-v1", fields])


class CaseExecutionBinding(StrictContract):
    schema_name: Literal["case_execution_binding"] = "case_execution_binding"
    schema_version: Literal[1] = 1
    binding_hash: Sha256
    case_manifest_entry_id: Sha256
    prompt_binding_id: NonEmptyStr
    output_contract_id: NonEmptyStr
    metric_id: NonEmptyStr
    judge_binding_id: NonEmptyStr | None
    answer_max_output_tokens: PositiveInt
    query_timestamp: NonEmptyStr | None

    @model_validator(mode="after")
    def identity_matches_execution_fields(self) -> Self:
        expected = case_execution_binding_hash(
            self.model_dump(mode="python", exclude={"binding_hash"})
        )
        if self.binding_hash != expected:
            raise ValueError("case execution binding hash does not match its canonical fields")
        return self


def case_partition_id(fields: Mapping[str, Any]) -> str:
    return canonical_sha256(["oamb-case-partition-v1", fields])


class CasePartitionSpec(StrictContract):
    schema_name: Literal["case_partition_spec"] = "case_partition_spec"
    schema_version: Literal[1] = 1
    partition_id: Sha256
    run_id: NonEmptyStr
    resolved_plan_hash: Sha256
    cell_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    target_case_manifest_hash: Sha256
    budget_policy_hash: Sha256
    retry_policy_hash: Sha256
    requested_case_manifest_entry_ids: tuple[Sha256, ...]
    selected_ingestion_plan_ids: tuple[Sha256, ...]
    selected_case_manifest_entry_ids: tuple[Sha256, ...]
    target_case_execution_bindings: tuple[CaseExecutionBinding, ...]
    target_case_execution_bindings_hash: Sha256

    @model_validator(mode="after")
    def selection_and_identity_are_closed(self) -> Self:
        inventories = (
            self.requested_case_manifest_entry_ids,
            self.selected_ingestion_plan_ids,
            self.selected_case_manifest_entry_ids,
        )
        if any(not items or len(set(items)) != len(items) for items in inventories):
            raise ValueError("case partition inventories must be non-empty and unique")
        target_case_ids = tuple(
            binding.case_manifest_entry_id for binding in self.target_case_execution_bindings
        )
        if not target_case_ids or len(set(target_case_ids)) != len(target_case_ids):
            raise ValueError("target case execution bindings must be non-empty and unique")
        if not set(self.requested_case_manifest_entry_ids) <= set(
            self.selected_case_manifest_entry_ids
        ):
            raise ValueError("requested cases must belong to the selected whole groups")
        if not set(self.selected_case_manifest_entry_ids) <= set(target_case_ids):
            raise ValueError("selected cases must belong to the target execution inventory")
        expected_bindings_hash = canonical_sha256(
            ["oamb-target-case-execution-bindings-v1", self.target_case_execution_bindings]
        )
        if self.target_case_execution_bindings_hash != expected_bindings_hash:
            raise ValueError("target case execution binding hash does not match its inventory")
        expected_partition_id = case_partition_id(
            self.model_dump(mode="python", exclude={"partition_id"})
        )
        if self.partition_id != expected_partition_id:
            raise ValueError("case partition identity does not match its canonical fields")
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
    model: NonEmptyStr
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
    model: NonEmptyStr | None
    thinking_effort: ThinkingEffort | None
    parameters_fingerprint: Sha256 | None
    retry_policy_id: NonEmptyStr | None
    configuration_fingerprint: Sha256
    redacted_endpoint_fingerprint: Sha256 | None

    @model_validator(mode="after")
    def binding_status_matches_shape(self) -> Self:
        selected_fields = (
            self.provider,
            self.endpoint_reference,
            self.model,
            self.thinking_effort,
            self.parameters_fingerprint,
            self.retry_policy_id,
            self.redacted_endpoint_fingerprint,
        )
        if self.role_status == RoleBindingStatus.SELECTED:
            if self.binding_kind not in {BindingKind.NATIVE, BindingKind.MODEL_CLIENT}:
                raise ValueError("selected binding requires native or model_client kind")
            if any(value is None for value in selected_fields):
                raise ValueError("selected binding requires provider and model fields")
            if self.role == ModelRole.EMBEDDING:
                if self.thinking_effort != "not_applicable":
                    raise ValueError("selected embedding binding requires not_applicable effort")
            elif self.thinking_effort == "not_applicable":
                raise ValueError("selected generative binding requires low, high, or max effort")
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


class ProviderOperationBudgetCeiling(StrictContract):
    schema_name: Literal["provider_operation_budget_ceiling"] = "provider_operation_budget_ceiling"
    schema_version: Literal[1] = 1
    provider_operation_ceiling_id: NonEmptyStr
    provider_operation_ceiling_hash: Sha256
    adapter_profile_id: NonEmptyStr
    operation_kind: NonEmptyStr
    billing_unit: NonEmptyStr
    maximum_accepted_units: NonNegativeDecimal
    max_attempts: PositiveInt
    max_dispatch_wall_seconds: NonNegativeDecimal
    resource_ceilings: tuple[ResourceBudgetCeiling, ...]

    @model_validator(mode="after")
    def identity_and_resources_close(self) -> Self:
        expected = provider_operation_budget_ceiling_hash(
            self.model_dump(mode="python", exclude={"provider_operation_ceiling_hash"})
        )
        if self.provider_operation_ceiling_hash != expected:
            raise ValueError("provider-operation ceiling hash does not match its fields")
        dimensions = tuple(item.dimension_id for item in self.resource_ceilings)
        if len(set(dimensions)) != len(dimensions):
            raise ValueError("provider-operation ceiling contains duplicate resources")
        return self


def provider_operation_budget_ceiling_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "provider_operation_budget_ceiling")
    payload.setdefault("schema_version", 1)
    payload.pop("provider_operation_ceiling_hash", None)
    return canonical_sha256(payload)


class DispatchBudgetRoute(StrictContract):
    schema_name: Literal["dispatch_budget_route"] = "dispatch_budget_route"
    schema_version: Literal[1] = 1
    route_id: NonEmptyStr
    route_hash: Sha256
    stage: NonEmptyStr
    dispatch_owner_kind: DispatchBudgetOwnerKind
    dispatch_model_role_binding_id: NonEmptyStr | None
    provider_operation_ceiling_id: NonEmptyStr | None
    adapter_profile_id: NonEmptyStr | None
    operation_kind: NonEmptyStr
    billing_unit: NonEmptyStr
    internal_usage_role_binding_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def owner_and_identity_close(self) -> Self:
        if self.dispatch_owner_kind == DispatchBudgetOwnerKind.MODEL_ROLE:
            if self.dispatch_model_role_binding_id is None:
                raise ValueError("model-role dispatch requires its role binding")
            if (
                self.provider_operation_ceiling_id is not None
                or self.adapter_profile_id is not None
            ):
                raise ValueError("model-role dispatch cannot name a provider operation")
            if self.internal_usage_role_binding_ids:
                raise ValueError("direct model dispatch cannot carry internal usage owners")
        else:
            if self.dispatch_model_role_binding_id is not None:
                raise ValueError("provider-operation dispatch cannot name a direct model role")
            if self.provider_operation_ceiling_id is None or self.adapter_profile_id is None:
                raise ValueError("provider-operation dispatch requires its ceiling and profile")
        if len(set(self.internal_usage_role_binding_ids)) != len(
            self.internal_usage_role_binding_ids
        ):
            raise ValueError("dispatch route contains duplicate internal usage owners")
        expected = dispatch_budget_route_hash(
            self.model_dump(mode="python", exclude={"route_hash"})
        )
        if self.route_hash != expected:
            raise ValueError("dispatch route hash does not match its fields")
        return self


def dispatch_budget_route_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "dispatch_budget_route")
    payload.setdefault("schema_version", 1)
    payload.pop("route_hash", None)
    return canonical_sha256(payload)


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


class BudgetSpecV3(StrictContract):
    schema_name: Literal["budget_spec"] = "budget_spec"
    schema_version: Literal[3] = 3
    budget_id: NonEmptyStr
    scope_kind: BudgetScopeKindV3
    scope_id: NonEmptyStr
    max_attempts: PositiveInt
    max_input_tokens: NonNegativeInt
    max_output_tokens: NonNegativeInt
    max_dispatch_wall_seconds: NonNegativeDecimal
    max_cost: NonNegativeDecimal | None
    currency: str | None
    resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    role_ceilings: tuple[RoleBudgetCeiling, ...]
    provider_operation_ceilings: tuple[ProviderOperationBudgetCeiling, ...]
    dispatch_routes: tuple[DispatchBudgetRoute, ...]
    stop_condition_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def parent_and_routes_close(self) -> Self:
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        if self.scope_kind != BudgetScopeKindV3.MEMORY_CONFORMANCE:
            raise ValueError("version 3 budget currently applies to memory conformance only")
        if not self.role_ceilings:
            raise ValueError("version 3 budget requires at least one role ceiling")
        if not self.provider_operation_ceilings:
            raise ValueError("version 3 budget requires a provider-operation ceiling")
        if not self.dispatch_routes:
            raise ValueError("version 3 budget requires a dispatch route")
        role_ids = tuple(item.role_binding_id for item in self.role_ceilings)
        operation_ids = tuple(
            item.provider_operation_ceiling_id for item in self.provider_operation_ceilings
        )
        route_ids = tuple(item.route_id for item in self.dispatch_routes)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("duplicate role ceiling")
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("duplicate provider-operation ceiling")
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("duplicate dispatch route")
        if len(set(self.stop_condition_ids)) != len(self.stop_condition_ids):
            raise ValueError("budget contains duplicate stop conditions")
        if any(ceiling.currency != self.currency for ceiling in self.role_ceilings):
            raise ValueError("role budget currency must match the parent currency")
        if self.max_cost is not None and any(
            ceiling.max_cost is None or ceiling.max_cost > self.max_cost
            for ceiling in self.role_ceilings
        ):
            raise ValueError("role cost ceiling must fit the parent cost ceiling")
        role_set = set(role_ids)
        operation_set = set(operation_ids)
        operations_by_id = {
            item.provider_operation_ceiling_id: item for item in self.provider_operation_ceilings
        }
        for route in self.dispatch_routes:
            if (
                route.dispatch_owner_kind == DispatchBudgetOwnerKind.MODEL_ROLE
                and route.dispatch_model_role_binding_id not in role_set
            ):
                raise ValueError("dispatch route names an unknown model-role ceiling")
            if (
                route.dispatch_owner_kind == DispatchBudgetOwnerKind.PROVIDER_OPERATION
                and route.provider_operation_ceiling_id not in operation_set
            ):
                raise ValueError("dispatch route names an unknown provider-operation ceiling")
            if not set(route.internal_usage_role_binding_ids) <= role_set:
                raise ValueError("dispatch route names an unknown internal usage owner")
            if route.dispatch_owner_kind == DispatchBudgetOwnerKind.PROVIDER_OPERATION:
                operation = operations_by_id.get(route.provider_operation_ceiling_id or "")
                if operation is None or (
                    route.adapter_profile_id != operation.adapter_profile_id
                    or route.operation_kind != operation.operation_kind
                    or route.stage != operation.operation_kind
                    or route.billing_unit != operation.billing_unit
                ):
                    raise ValueError(
                        "dispatch route and provider-operation ceiling do not exactly match"
                    )
        return self


class BudgetSpecV4(StrictContract):
    schema_name: Literal["budget_spec"] = "budget_spec"
    schema_version: Literal[4] = 4
    budget_id: NonEmptyStr
    budget_hash: Sha256
    scope_kind: BudgetScopeKindV3
    scope_id: NonEmptyStr
    max_attempts: PositiveInt
    max_input_tokens: NonNegativeInt
    max_output_tokens: NonNegativeInt
    max_dispatch_wall_seconds: NonNegativeDecimal
    max_cost: NonNegativeDecimal | None
    currency: str | None
    resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    role_ceilings: tuple[RoleBudgetCeiling, ...]
    provider_operation_ceilings: tuple[ProviderOperationBudgetCeiling, ...]
    dispatch_routes: tuple[DispatchBudgetRoute, ...]
    stop_condition_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def live_run_budget_and_routes_close(self) -> Self:
        expected_hash = budget_spec_v4_hash(self.model_dump(mode="python", exclude={"budget_hash"}))
        if self.budget_hash != expected_hash:
            raise ValueError("budget hash does not match its canonical fields")
        if self.scope_kind != BudgetScopeKindV3.RUN:
            raise ValueError("version 4 budget applies to a benchmark run only")
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        if not self.role_ceilings:
            raise ValueError("version 4 budget requires at least one role ceiling")
        if not self.provider_operation_ceilings:
            raise ValueError("version 4 budget requires a provider-operation ceiling")
        if not self.dispatch_routes:
            raise ValueError("version 4 budget requires a dispatch route")

        role_ids = tuple(item.role_binding_id for item in self.role_ceilings)
        operation_ids = tuple(
            item.provider_operation_ceiling_id for item in self.provider_operation_ceilings
        )
        route_ids = tuple(item.route_id for item in self.dispatch_routes)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("duplicate role ceiling")
        if len(set(operation_ids)) != len(operation_ids):
            raise ValueError("duplicate provider-operation ceiling")
        if set(role_ids) & set(operation_ids):
            raise ValueError("budget owner identities must be unique across owner kinds")
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("duplicate dispatch route")
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

        role_set = set(role_ids)
        operation_set = set(operation_ids)
        operations_by_id = {
            item.provider_operation_ceiling_id: item for item in self.provider_operation_ceilings
        }
        referenced_roles: set[str] = set()
        referenced_operations: set[str] = set()
        for route in self.dispatch_routes:
            if route.dispatch_owner_kind == DispatchBudgetOwnerKind.MODEL_ROLE:
                role_id = route.dispatch_model_role_binding_id
                if role_id not in role_set:
                    raise ValueError("dispatch route names an unknown model-role ceiling")
                referenced_roles.add(role_id or "")
            else:
                operation_id = route.provider_operation_ceiling_id
                if operation_id not in operation_set:
                    raise ValueError("dispatch route names an unknown provider-operation ceiling")
                operation = operations_by_id.get(operation_id or "")
                if operation is None or (
                    route.adapter_profile_id != operation.adapter_profile_id
                    or route.operation_kind != operation.operation_kind
                    or route.stage != operation.operation_kind
                    or route.billing_unit != operation.billing_unit
                ):
                    raise ValueError(
                        "dispatch route and provider-operation ceiling do not exactly match"
                    )
                referenced_operations.add(operation_id or "")
            if not set(route.internal_usage_role_binding_ids) <= role_set:
                raise ValueError("dispatch route names an unknown internal usage owner")
            referenced_roles.update(route.internal_usage_role_binding_ids)

        if referenced_roles != role_set or referenced_operations != operation_set:
            raise ValueError("live budget contains an orphan budget owner ceiling")
        return self


def budget_spec_v4_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "budget_spec")
    payload.setdefault("schema_version", 4)
    payload.pop("budget_hash", None)
    return canonical_sha256(payload)


MEMORY_CONFORMANCE_ROUTE_STAGES = (
    "runtime_resolve",
    "scope_allocate",
    "memory_ingest",
    "memory_readiness",
    "pre_query_projection",
    "memory_query",
    "post_query_projection",
)


class MemoryConformanceSpec(StrictContract):
    schema_name: Literal["memory_conformance_spec"] = "memory_conformance_spec"
    schema_version: Literal[1] = 1
    conformance_spec_id: NonEmptyStr
    conformance_spec_hash: Sha256
    occurrence_id: NonEmptyStr
    provider: NonEmptyStr
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    runtime_binding_hash: Sha256
    budget_id: NonEmptyStr
    budget_hash: Sha256
    dispatch_routes: tuple[DispatchBudgetRoute, ...]
    minimal_source_sha256: Sha256
    minimal_query_sha256: Sha256
    expected_marker_sha256: Sha256
    stop_condition_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def immutable_authority_and_routes_close(self) -> Self:
        expected_hash = memory_conformance_spec_hash(
            self.model_dump(mode="python", exclude={"conformance_spec_hash"})
        )
        if self.conformance_spec_hash != expected_hash:
            raise ValueError("memory-conformance spec hash does not match its canonical fields")
        if tuple(route.stage for route in self.dispatch_routes) != (
            MEMORY_CONFORMANCE_ROUTE_STAGES
        ):
            raise ValueError("memory-conformance spec requires the exact seven-stage route order")
        route_ids = tuple(route.route_id for route in self.dispatch_routes)
        if len(set(route_ids)) != len(route_ids):
            raise ValueError("memory-conformance spec contains duplicate routes")
        if len(set(self.stop_condition_ids)) != len(self.stop_condition_ids):
            raise ValueError("memory-conformance spec contains duplicate stop conditions")
        for route in self.dispatch_routes:
            if route.dispatch_owner_kind != DispatchBudgetOwnerKind.PROVIDER_OPERATION:
                raise ValueError("memory-conformance stages require provider-operation routes")
            if route.adapter_profile_id != self.provider_profile_id:
                raise ValueError("memory-conformance route profile does not match the spec")
            if route.operation_kind != route.stage:
                raise ValueError("memory-conformance route stage and operation must match")
        return self


def memory_conformance_spec_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "memory_conformance_spec")
    payload.setdefault("schema_version", 1)
    payload.pop("conformance_spec_hash", None)
    return canonical_sha256(payload)


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


def provider_runtime_profile_attestation_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "provider_runtime_profile_attestation")
    payload.setdefault("schema_version", 1)
    payload.pop("attestation_hash", None)
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


class ReportSpec(StrictContract):
    schema_name: Literal["report_spec"] = "report_spec"
    schema_version: Literal[1] = 1
    report_spec_id: Sha256
    report_kind: Literal["run", "comparison", "release"]
    audience: Literal["local", "public"]
    preview_max_field_bytes: PositiveInt
    preview_total_bytes: PositiveInt
    display_field_ids: tuple[NonEmptyStr, ...]
    renderer_hash: Sha256
    asset_hashes: tuple[Sha256, ...]
    browser_contract_hash: Sha256
    performance_contract_hash: Sha256
    export_profile_selector_id: NonEmptyStr
    export_profile_selector_version: PositiveInt

    @model_validator(mode="after")
    def report_spec_identity_and_limits_are_canonical(self) -> Self:
        if self.preview_max_field_bytes > self.preview_total_bytes:
            raise ValueError("report field preview exceeds the total preview budget")
        if not self.display_field_ids or len(set(self.display_field_ids)) != len(
            self.display_field_ids
        ):
            raise ValueError("report spec display fields must be nonempty and unique")
        if len(set(self.asset_hashes)) != len(self.asset_hashes):
            raise ValueError("report spec contains duplicate asset hashes")
        expected = report_spec_id(
            report_kind=self.report_kind,
            audience=self.audience,
            preview_max_field_bytes=self.preview_max_field_bytes,
            preview_total_bytes=self.preview_total_bytes,
            display_field_ids=self.display_field_ids,
            renderer_hash=self.renderer_hash,
            asset_hashes=self.asset_hashes,
            browser_contract_hash=self.browser_contract_hash,
            performance_contract_hash=self.performance_contract_hash,
            export_profile_selector_id=self.export_profile_selector_id,
            export_profile_selector_version=self.export_profile_selector_version,
        )
        if self.report_spec_id != expected:
            raise ValueError("report spec identity does not match its canonical payload")
        return self


class ReportSpecV2(StrictContract):
    schema_name: Literal["report_spec"] = "report_spec"
    schema_version: Literal[2] = 2
    report_spec_id: Sha256
    report_kind: Literal["evaluation"]
    audience: Literal["local", "public"]
    preview_max_field_bytes: PositiveInt
    preview_total_bytes: PositiveInt
    display_field_ids: tuple[NonEmptyStr, ...]
    renderer_hash: Sha256
    asset_hashes: tuple[Sha256, ...]
    browser_contract_hash: Sha256
    performance_contract_hash: Sha256
    export_profile_selector_id: NonEmptyStr
    export_profile_selector_version: PositiveInt

    @model_validator(mode="after")
    def report_spec_identity_and_limits_are_canonical(self) -> Self:
        if self.preview_max_field_bytes > self.preview_total_bytes:
            raise ValueError("report field preview exceeds the total preview budget")
        if not self.display_field_ids or len(set(self.display_field_ids)) != len(
            self.display_field_ids
        ):
            raise ValueError("report spec display fields must be nonempty and unique")
        if len(set(self.asset_hashes)) != len(self.asset_hashes):
            raise ValueError("report spec contains duplicate asset hashes")
        expected = report_spec_v2_id(
            report_kind=self.report_kind,
            audience=self.audience,
            preview_max_field_bytes=self.preview_max_field_bytes,
            preview_total_bytes=self.preview_total_bytes,
            display_field_ids=self.display_field_ids,
            renderer_hash=self.renderer_hash,
            asset_hashes=self.asset_hashes,
            browser_contract_hash=self.browser_contract_hash,
            performance_contract_hash=self.performance_contract_hash,
            export_profile_selector_id=self.export_profile_selector_id,
            export_profile_selector_version=self.export_profile_selector_version,
        )
        if self.report_spec_id != expected:
            raise ValueError("report spec identity does not match its canonical payload")
        return self


class ReportIdentitySpecBinding(StrictContract):
    schema_name: Literal["report_identity_spec_binding"] = "report_identity_spec_binding"
    schema_version: Literal[1] = 1
    binding_id: Sha256
    spec_kind: Literal["benchmark_report"] = "benchmark_report"
    spec_schema_name: Literal["report_spec"] = "report_spec"
    spec_schema_version: Literal[1] = 1
    spec_id: Sha256
    spec_hash: Sha256

    @model_validator(mode="after")
    def binding_discriminator_and_identity_are_canonical(self) -> Self:
        expected = report_identity_spec_binding_id(
            spec_kind=self.spec_kind,
            spec_schema_name=self.spec_schema_name,
            spec_schema_version=self.spec_schema_version,
            spec_id=self.spec_id,
            spec_hash=self.spec_hash,
        )
        if self.binding_id != expected:
            raise ValueError("report identity binding does not match its canonical payload")
        return self


class ReportIdentitySpecBindingV2(StrictContract):
    schema_name: Literal["report_identity_spec_binding"] = "report_identity_spec_binding"
    schema_version: Literal[2] = 2
    binding_id: Sha256
    spec_kind: Literal["benchmark_report"] = "benchmark_report"
    spec_schema_name: Literal["report_spec"] = "report_spec"
    spec_schema_version: Literal[2] = 2
    spec_id: Sha256
    spec_hash: Sha256

    @model_validator(mode="after")
    def binding_identity_is_canonical(self) -> Self:
        expected = report_identity_spec_binding_v2_id(
            spec_kind=self.spec_kind,
            spec_schema_name=self.spec_schema_name,
            spec_schema_version=self.spec_schema_version,
            spec_id=self.spec_id,
            spec_hash=self.spec_hash,
        )
        if self.binding_id != expected:
            raise ValueError("evaluation report identity binding is not canonical")
        return self


class EvaluationModelClosureEntry(StrictContract):
    schema_name: Literal["evaluation_model_closure_entry"] = "evaluation_model_closure_entry"
    schema_version: Literal[1] = 1
    model_kind: Literal["run", "comparison"]
    report_model_id: Sha256
    canonical_bytes_hash: Sha256
    ordered_source_root_hash: Sha256
    evidence_validation_result_hash: Sha256
    coverage_hash: Sha256


class EvaluationModelClosure(StrictContract):
    schema_name: Literal["evaluation_model_closure"] = "evaluation_model_closure"
    schema_version: Literal[1] = 1
    ordered_run_entries: tuple[
        EvaluationModelClosureEntry,
        EvaluationModelClosureEntry,
        EvaluationModelClosureEntry,
        EvaluationModelClosureEntry,
    ]
    eligible_comparison_entries: tuple[EvaluationModelClosureEntry, ...]
    model_closure_hash: Sha256

    @model_validator(mode="after")
    def kinds_cardinality_and_identity_close(self) -> Self:
        if any(item.model_kind != "run" for item in self.ordered_run_entries):
            raise ValueError("evaluation run closure contains a non-run model")
        if not self.eligible_comparison_entries or any(
            item.model_kind != "comparison" for item in self.eligible_comparison_entries
        ):
            raise ValueError("evaluation comparison closure must be nonempty and typed")
        model_ids = tuple(
            item.report_model_id
            for item in (*self.ordered_run_entries, *self.eligible_comparison_entries)
        )
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("evaluation model closure contains duplicate model identities")
        expected = evaluation_model_closure_hash(
            ordered_run_entries=self.ordered_run_entries,
            eligible_comparison_entries=self.eligible_comparison_entries,
        )
        if self.model_closure_hash != expected:
            raise ValueError("evaluation model closure hash does not match its entries")
        return self


RENDERING_DERIVATION_KINDS = frozenset(
    {
        "run_report",
        "diagnostic_run_report",
        "comparison_report",
        "release_report",
    }
)


class DerivationSpecV2(StrictContract):
    schema_name: Literal["derivation_spec"] = "derivation_spec"
    schema_version: Literal[2] = 2
    derivation_kind: NonEmptyStr
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    ordered_source_root_hash: Sha256
    evidence_validation_result_hash: Sha256
    transform_spec_hash: Sha256
    report_identity_spec_binding: ReportIdentitySpecBinding | None
    reducer_and_renderer_input_hashes: tuple[Sha256, ...]
    derivation_input_hash: Sha256

    @model_validator(mode="after")
    def source_report_cardinality_and_identity_close(self) -> Self:
        if not self.ordered_source_bindings:
            raise ValueError("derivation requires at least one source binding")
        binding_ids = tuple(item.binding_id for item in self.ordered_source_bindings)
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("derivation contains duplicate source bindings")
        expected_root_hash = canonical_sha256(
            [
                "oamb-ordered-source-roots-v1",
                tuple(item.source_root_hash for item in self.ordered_source_bindings),
            ]
        )
        if self.ordered_source_root_hash != expected_root_hash:
            raise ValueError("derivation ordered source-root hash does not match its bindings")
        rendering = self.derivation_kind in RENDERING_DERIVATION_KINDS
        if rendering and self.report_identity_spec_binding is None:
            raise ValueError("rendering derivation requires exactly one report identity binding")
        if not rendering and self.report_identity_spec_binding is not None:
            raise ValueError("non-rendering derivation cannot carry a report identity binding")
        expected_input_hash = derivation_spec_v2_input_hash(
            derivation_kind=self.derivation_kind,
            ordered_source_bindings=self.ordered_source_bindings,
            ordered_source_root_hash=self.ordered_source_root_hash,
            evidence_validation_result_hash=self.evidence_validation_result_hash,
            transform_spec_hash=self.transform_spec_hash,
            report_identity_spec_binding=self.report_identity_spec_binding,
            reducer_and_renderer_input_hashes=self.reducer_and_renderer_input_hashes,
        )
        if self.derivation_input_hash != expected_input_hash:
            raise ValueError("derivation input hash does not match its canonical payload")
        return self


class DerivationSpecV3(StrictContract):
    schema_name: Literal["derivation_spec"] = "derivation_spec"
    schema_version: Literal[3] = 3
    derivation_kind: Literal["evaluation_report"] = "evaluation_report"
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    ordered_source_root_hash: Sha256
    evidence_validation_result_hash: Sha256
    transform_spec_hash: Sha256
    report_identity_spec_binding: ReportIdentitySpecBindingV2
    evaluation_model_closure: EvaluationModelClosure
    reducer_and_renderer_input_hashes: tuple[Sha256, ...]
    derivation_input_hash: Sha256

    @model_validator(mode="after")
    def evaluation_source_and_identity_close(self) -> Self:
        if not self.ordered_source_bindings:
            raise ValueError("evaluation derivation requires source bindings")
        binding_ids = tuple(item.binding_id for item in self.ordered_source_bindings)
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("evaluation derivation contains duplicate source bindings")
        expected_root_hash = canonical_sha256(
            [
                "oamb-ordered-source-roots-v1",
                tuple(item.source_root_hash for item in self.ordered_source_bindings),
            ]
        )
        if self.ordered_source_root_hash != expected_root_hash:
            raise ValueError("evaluation ordered source-root hash does not match its bindings")
        expected_input_hash = derivation_spec_v3_input_hash(
            derivation_kind=self.derivation_kind,
            ordered_source_bindings=self.ordered_source_bindings,
            ordered_source_root_hash=self.ordered_source_root_hash,
            evidence_validation_result_hash=self.evidence_validation_result_hash,
            transform_spec_hash=self.transform_spec_hash,
            report_identity_spec_binding=self.report_identity_spec_binding,
            evaluation_model_closure=self.evaluation_model_closure,
            reducer_and_renderer_input_hashes=self.reducer_and_renderer_input_hashes,
        )
        if self.derivation_input_hash != expected_input_hash:
            raise ValueError("evaluation derivation input hash does not match its payload")
        return self


def report_spec_id(**fields: object) -> str:
    return canonical_sha256(["oamb-report-spec-v1", fields])


def report_spec_v2_id(**fields: object) -> str:
    return canonical_sha256(["oamb-report-spec-v2", fields])


def report_identity_spec_binding_id(**fields: object) -> str:
    return canonical_sha256(["oamb-report-identity-spec-binding-v1", fields])


def report_identity_spec_binding_v2_id(**fields: object) -> str:
    return canonical_sha256(["oamb-report-identity-spec-binding-v2", fields])


def evaluation_model_closure_hash(**fields: object) -> str:
    return canonical_sha256(["oamb-evaluation-model-closure-v1", fields])


def derivation_spec_v2_input_hash(**fields: object) -> str:
    return canonical_sha256(["oamb-derivation-input-v2", fields])


def derivation_spec_v3_input_hash(**fields: object) -> str:
    return canonical_sha256(["oamb-derivation-input-v3", fields])


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


class RunPreflightRecord(StrictContract):
    schema_name: Literal["run_preflight_record"] = "run_preflight_record"
    schema_version: Literal[1] = 1
    preflight_record_hash: Sha256
    run_id: NonEmptyStr
    observed_at: UtcDateTime
    resolved_plan_hash: Sha256
    run_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    subset_manifest_hash: Sha256
    adapter_profile_id: NonEmptyStr
    adapter_profile_hash: Sha256
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    runtime_binding_hash: Sha256
    provider_service_evidence: SourceEvidenceBinding
    provider_profile_evidence: SourceEvidenceBinding
    role_binding_ids: tuple[NonEmptyStr, ...]
    dispatch_routes: tuple[DispatchBudgetRoute, ...]
    budget_hash: Sha256
    redacted_endpoint_fingerprints: tuple[Sha256, ...]
    credential_reference_fingerprints: tuple[Sha256, ...]
    artifact_repository_fingerprint: Sha256
    artifact_durability_proof_hash: Sha256

    @model_validator(mode="after")
    def closure_and_identity_match(self) -> Self:
        if self.provider_service_evidence.source_kind != SourceEvidenceKind.PROVIDER_SERVICE:
            raise ValueError("preflight provider-service evidence has the wrong source kind")
        if self.provider_profile_evidence.source_kind != SourceEvidenceKind.PROVIDER_SERVICE:
            raise ValueError("preflight provider-profile evidence has the wrong source kind")
        if "provider_service_evidence_manifest@1" not in (
            self.provider_profile_evidence.source_schema_versions
        ):
            raise ValueError("preflight requires an explicit provider-profile source binding")
        inventories = (
            self.role_binding_ids,
            tuple(item.route_id for item in self.dispatch_routes),
            self.redacted_endpoint_fingerprints,
            self.credential_reference_fingerprints,
        )
        if any(len(set(items)) != len(items) for items in inventories):
            raise ValueError("preflight inventories cannot contain duplicates")
        role_ids = set(self.role_binding_ids)
        for route in self.dispatch_routes:
            if (
                route.dispatch_model_role_binding_id is not None
                and route.dispatch_model_role_binding_id not in role_ids
            ):
                raise ValueError("preflight route names an unknown dispatch role")
            if not set(route.internal_usage_role_binding_ids) <= role_ids:
                raise ValueError("preflight route names an unknown internal usage role")
            if (
                route.dispatch_owner_kind == DispatchBudgetOwnerKind.PROVIDER_OPERATION
                and route.adapter_profile_id != self.adapter_profile_id
            ):
                raise ValueError("preflight route adapter profile does not match the run")
        expected = run_preflight_record_hash(
            self.model_dump(mode="python", exclude={"preflight_record_hash"})
        )
        if self.preflight_record_hash != expected:
            raise ValueError("preflight record hash does not match its canonical fields")
        return self


def run_preflight_record_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "run_preflight_record")
    payload.setdefault("schema_version", 1)
    payload.pop("preflight_record_hash", None)
    return canonical_sha256(payload)


class RunPreflightRecordV2(StrictContract):
    schema_name: Literal["run_preflight_record"] = "run_preflight_record"
    schema_version: Literal[2] = 2
    preflight_record_hash: Sha256
    run_id: NonEmptyStr
    observed_at: UtcDateTime
    resolved_plan_hash: Sha256
    run_spec_hash: Sha256
    dataset_manifest_hash: Sha256
    subset_manifest_hash: Sha256
    adapter_profile_id: NonEmptyStr
    adapter_profile_hash: Sha256
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    runtime_binding_hash: Sha256
    provider_service_evidence: SourceEvidenceBinding
    memory_conformance_evidence: SourceEvidenceBinding
    role_binding_ids: tuple[NonEmptyStr, ...]
    dispatch_routes: tuple[DispatchBudgetRoute, ...]
    budget_hash: Sha256
    redacted_endpoint_fingerprints: tuple[Sha256, ...]
    credential_reference_fingerprints: tuple[Sha256, ...]
    artifact_repository_fingerprint: Sha256
    artifact_durability_proof_hash: Sha256
    comparison_control_basis_hash: Sha256

    @model_validator(mode="after")
    def closure_and_identity_match(self) -> Self:
        if self.provider_service_evidence.source_kind != SourceEvidenceKind.PROVIDER_SERVICE:
            raise ValueError("preflight provider-service evidence has the wrong source kind")
        if self.memory_conformance_evidence.source_kind != SourceEvidenceKind.PROVIDER_SERVICE:
            raise ValueError("preflight memory-conformance evidence has the wrong source kind")
        if (
            "memory_conformance_evidence_manifest@1"
            not in self.memory_conformance_evidence.source_schema_versions
        ):
            raise ValueError("preflight requires an explicit memory-conformance source binding")
        inventories = (
            self.role_binding_ids,
            tuple(item.route_id for item in self.dispatch_routes),
            self.redacted_endpoint_fingerprints,
            self.credential_reference_fingerprints,
        )
        if any(len(set(items)) != len(items) for items in inventories):
            raise ValueError("preflight inventories cannot contain duplicates")
        role_ids = set(self.role_binding_ids)
        for route in self.dispatch_routes:
            if (
                route.dispatch_model_role_binding_id is not None
                and route.dispatch_model_role_binding_id not in role_ids
            ):
                raise ValueError("preflight route names an unknown dispatch role")
            if not set(route.internal_usage_role_binding_ids) <= role_ids:
                raise ValueError("preflight route names an unknown internal usage role")
            if (
                route.dispatch_owner_kind == DispatchBudgetOwnerKind.PROVIDER_OPERATION
                and route.adapter_profile_id != self.adapter_profile_id
            ):
                raise ValueError("preflight route adapter profile does not match the run")
        expected = run_preflight_record_v2_hash(
            self.model_dump(mode="python", exclude={"preflight_record_hash"})
        )
        if self.preflight_record_hash != expected:
            raise ValueError("preflight record hash does not match its canonical fields")
        return self


def run_preflight_record_v2_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "run_preflight_record")
    payload.setdefault("schema_version", 2)
    payload.pop("preflight_record_hash", None)
    return canonical_sha256(["oamb-run-preflight-record-v2", payload])


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
