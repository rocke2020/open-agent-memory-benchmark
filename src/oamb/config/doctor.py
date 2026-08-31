"""Build and load the immutable three-cell LongMemEval execution plan."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from oamb.artifacts.atomic import read_regular_file
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256

from .benchmark import (
    DEEPSEEK_THINKING_EFFORT_SCALE,
    MODEL_ROLE_IDS,
    T10_CELL_IDS,
    T10_RETRIEVAL_BINDING_IDS,
    BenchmarkConfiguration,
    CellConfiguration,
    ExecutionLimits,
    GenerativeThinkingEffort,
    ModelRoleConfiguration,
    ModelRoleId,
    RetrievalBindingConfiguration,
    RunLimits,
    ThinkingEffort,
)

_SCHEMA_NAME = "resolved_plan"
_SCHEMA_VERSION = 1
_PLAN_HASH_DOMAIN = "oamb-resolved-plan-initial-v1"
_MODEL_BINDING_HASH_DOMAIN = "oamb-model-execution-binding-initial-v1"
_RETRIEVAL_BINDING_HASH_DOMAIN = "oamb-retrieval-binding-initial-v1"
_CELL_SPEC_HASH_DOMAIN = "oamb-cell-spec-initial-v1"
_LIMITS_HASH_DOMAIN = "oamb-run-limits-initial-v1"
_EXECUTION_HASH_DOMAIN = "oamb-execution-limits-initial-v1"

_ROOT_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "resolved_plan_hash",
        "comparison_id",
        "dataset",
        "model_roles",
        "retrieval",
        "limits",
        "execution",
        "cells",
    }
)
_DATASET_KEYS = frozenset(
    {
        "dataset_id",
        "workload_id",
        "selection",
        "path",
        "revision",
        "source_sha256",
        "case_manifest_hash",
    }
)
_MODEL_ROLE_KEYS = frozenset(
    {
        "binding_hash",
        "role_id",
        "configured_model",
        "runtime_model",
        "thinking_effort",
        "thinking_effort_scale",
        "thinking_effort_rank_1_indexed",
        "endpoint_variable",
        "credential_variable",
        "recipient",
        "execution_owner",
        "proof_kind",
        "proof_reference",
        "usage_coverage",
        "max_input_tokens",
        "max_output_tokens",
        "temperature",
        "top_p",
    }
)
_RETRIEVAL_KEYS = frozenset({"generation", "bindings"})
_RETRIEVAL_BINDING_KEYS = frozenset(
    {
        "binding_hash",
        "binding_id",
        "provider_id",
        "route",
        "query_embedding_role",
        "disabled_feature",
        "disabled_value",
        "request_constraint",
        "proof_kind",
    }
)
_LIMIT_KEYS = (
    "max_attempts_per_case",
    "max_budgeted_attempts",
    "memory_operation_timeout_seconds",
    "model_call_timeout_seconds",
    "total_wall_time_seconds",
    "max_input_tokens",
    "max_output_tokens",
    "max_recall_context_tokens_per_case",
    "max_storage_bytes",
    "max_peak_memory_bytes",
    "max_cost",
    "currency",
)
_EXECUTION_KEYS = (
    "max_parallel_datasets",
    "max_parallel_providers_per_dataset",
    "max_parallel_history_ingestions_per_provider",
    "max_parallel_questions_per_provider",
)
_CELL_KEYS = frozenset(
    {
        "cell_spec_hash",
        "ordinal_1_indexed",
        "cell_id",
        "provider_id",
        "adapter_profile_id",
        "endpoint_variable",
        "credential_variable",
        "recipient",
        "dataset_id",
        "workload_id",
        "selection",
        "source_sha256",
        "case_manifest_hash",
        "producer_role_id",
        "embedding_role_id",
        "answer_role_id",
        "judge_role_id",
        "model_role_binding_hashes",
        "retrieval_binding_id",
        "retrieval_binding_hash",
        "limits_hash",
        "execution_hash",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class ResolvedPlanError(ValueError):
    """The resolved plan is malformed, mutable, or has lost hash closure."""


@dataclass(frozen=True, slots=True)
class ResolvedDataset:
    dataset_id: str
    workload_id: str
    selection: str
    path: str
    revision: str
    source_sha256: str
    case_manifest_hash: str


@dataclass(frozen=True, slots=True)
class ModelExecutionBinding:
    binding_hash: str
    role_id: ModelRoleId
    configured_model: str
    runtime_model: str
    thinking_effort: ThinkingEffort
    thinking_effort_scale: tuple[GenerativeThinkingEffort, ...]
    thinking_effort_rank_1_indexed: int | None
    endpoint_variable: str
    credential_variable: str
    recipient: str
    execution_owner: str
    proof_kind: str
    proof_reference: str
    usage_coverage: str
    max_input_tokens: int
    max_output_tokens: int
    temperature: str
    top_p: str

    @property
    def model(self) -> str:
        """Compatibility spelling for callers migrating to configured_model."""

        return self.configured_model


ResolvedModelRole = ModelExecutionBinding


@dataclass(frozen=True, slots=True)
class ResolvedRetrievalBinding:
    binding_hash: str
    binding_id: str
    provider_id: str
    route: str
    query_embedding_role: ModelRoleId
    disabled_feature: str
    disabled_value: str
    request_constraint: str
    proof_kind: str


@dataclass(frozen=True, slots=True)
class ResolvedRetrieval:
    generation: str
    bindings: tuple[ResolvedRetrievalBinding, ...]


@dataclass(frozen=True, slots=True)
class CellSpec:
    cell_spec_hash: str
    ordinal_1_indexed: int
    cell_id: str
    provider_id: str
    adapter_profile_id: str
    endpoint_variable: str
    credential_variable: str
    recipient: str
    dataset_id: str
    workload_id: str
    selection: str
    source_sha256: str
    case_manifest_hash: str
    producer_role_id: ModelRoleId
    embedding_role_id: ModelRoleId
    answer_role_id: ModelRoleId
    judge_role_id: ModelRoleId
    model_role_binding_hashes: tuple[tuple[ModelRoleId, str], ...]
    retrieval_binding_id: str
    retrieval_binding_hash: str
    limits_hash: str
    execution_hash: str


@dataclass(frozen=True, slots=True)
class ResolvedPlan:
    schema_name: str
    schema_version: int
    resolved_plan_hash: str
    comparison_id: str
    dataset: ResolvedDataset
    model_roles: tuple[ModelExecutionBinding, ...]
    retrieval: ResolvedRetrieval
    limits: RunLimits
    execution: ExecutionLimits
    cells: tuple[CellSpec, ...]


def build_resolved_plan(configuration: BenchmarkConfiguration) -> ResolvedPlan:
    """Freeze the complete comparison into three ordered executable cells."""

    dataset = ResolvedDataset(
        dataset_id=configuration.dataset.dataset_id,
        workload_id=configuration.dataset.workload_id,
        selection=configuration.dataset.selection,
        path=configuration.dataset.path,
        revision=configuration.dataset.revision,
        source_sha256=configuration.dataset.source_sha256,
        case_manifest_hash=configuration.dataset.case_manifest_hash,
    )
    model_roles = tuple(
        _build_model_binding(role_id, role, configuration.limits)
        for role_id, role in configuration.models.ordered_items()
    )
    retrieval_bindings = tuple(
        _build_retrieval_binding(binding) for binding in configuration.retrieval.bindings
    )
    retrieval = ResolvedRetrieval(
        generation=configuration.retrieval.generation,
        bindings=retrieval_bindings,
    )
    roles_by_id = {role.role_id: role for role in model_roles}
    retrieval_by_id = {binding.binding_id: binding for binding in retrieval_bindings}
    limits_hash = canonical_sha256([_LIMITS_HASH_DOMAIN, _limits_document(configuration.limits)])
    execution_hash = canonical_sha256(
        [_EXECUTION_HASH_DOMAIN, _execution_document(configuration.execution)]
    )
    cells = tuple(
        _build_cell_spec(
            ordinal=ordinal,
            configuration=cell,
            dataset=dataset,
            roles_by_id=roles_by_id,
            retrieval_by_id=retrieval_by_id,
            limits_hash=limits_hash,
            execution_hash=execution_hash,
        )
        for ordinal, cell in enumerate(configuration.cells, start=1)
    )
    payload = _payload(
        comparison_id=configuration.comparison_id,
        dataset=dataset,
        model_roles=model_roles,
        retrieval=retrieval,
        limits=configuration.limits,
        execution=configuration.execution,
        cells=cells,
    )
    return ResolvedPlan(
        schema_name=_SCHEMA_NAME,
        schema_version=_SCHEMA_VERSION,
        resolved_plan_hash=_plan_hash(payload),
        comparison_id=configuration.comparison_id,
        dataset=dataset,
        model_roles=model_roles,
        retrieval=retrieval,
        limits=configuration.limits,
        execution=configuration.execution,
        cells=cells,
    )


def resolved_plan_bytes(plan: ResolvedPlan) -> bytes:
    """Serialize a resolved plan in its sole accepted byte representation."""

    document = _payload(
        comparison_id=plan.comparison_id,
        dataset=plan.dataset,
        model_roles=plan.model_roles,
        retrieval=plan.retrieval,
        limits=plan.limits,
        execution=plan.execution,
        cells=plan.cells,
    )
    document["resolved_plan_hash"] = plan.resolved_plan_hash
    return canonical_json_bytes(document)


def load_resolved_plan_for_run(path: Path) -> ResolvedPlan:
    """Load only canonical frozen bytes; never reopen source YAML or environment."""

    try:
        content = read_regular_file(path)
        document = json.loads(
            content,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ResolvedPlanError(f"cannot load resolved plan: {path}") from exc
    if canonical_json_bytes(document) != content:
        raise ResolvedPlanError("resolved plan bytes are not canonical JSON")
    root = _require_exact_mapping(document, _ROOT_KEYS, "resolved plan")
    if root["schema_name"] != _SCHEMA_NAME or type(root["schema_name"]) is not str:
        raise ResolvedPlanError("resolved plan schema_name is invalid")
    if root["schema_version"] != _SCHEMA_VERSION or type(root["schema_version"]) is not int:
        raise ResolvedPlanError("resolved plan schema_version is invalid")
    comparison_id = _require_text(root["comparison_id"], "comparison ID")
    dataset = _parse_dataset(root["dataset"])
    model_roles = _parse_model_roles(root["model_roles"])
    retrieval = _parse_retrieval(root["retrieval"])
    limits = _parse_limits(root["limits"])
    execution = _parse_execution(root["execution"])
    cells = _parse_cells(
        root["cells"],
        dataset=dataset,
        model_roles=model_roles,
        retrieval=retrieval,
        limits=limits,
        execution=execution,
    )
    payload = _payload(
        comparison_id=comparison_id,
        dataset=dataset,
        model_roles=model_roles,
        retrieval=retrieval,
        limits=limits,
        execution=execution,
        cells=cells,
    )
    plan_hash = root["resolved_plan_hash"]
    if type(plan_hash) is not str or plan_hash != _plan_hash(payload):
        raise ResolvedPlanError("resolved plan hash does not match its frozen content")
    return ResolvedPlan(
        schema_name=_SCHEMA_NAME,
        schema_version=_SCHEMA_VERSION,
        resolved_plan_hash=plan_hash,
        comparison_id=comparison_id,
        dataset=dataset,
        model_roles=model_roles,
        retrieval=retrieval,
        limits=limits,
        execution=execution,
        cells=cells,
    )


def _build_model_binding(
    role_id: ModelRoleId, role: ModelRoleConfiguration, limits: RunLimits
) -> ModelExecutionBinding:
    values = {
        "role_id": role_id,
        "configured_model": role.model,
        "runtime_model": role.runtime_model,
        "thinking_effort": role.thinking_effort,
        "thinking_effort_scale": role.thinking_effort_scale,
        "thinking_effort_rank_1_indexed": role.thinking_effort_rank_1_indexed,
        "endpoint_variable": role.endpoint_variable,
        "credential_variable": role.credential_variable,
        "recipient": role.recipient,
        "execution_owner": role.execution_owner,
        "proof_kind": role.proof_kind,
        "proof_reference": role.proof_reference,
        "usage_coverage": role.usage_coverage,
        "max_input_tokens": limits.max_input_tokens,
        "max_output_tokens": limits.max_output_tokens,
        "temperature": "not_applicable" if role_id == "embedding" else "0",
        "top_p": "not_applicable" if role_id == "embedding" else "1",
    }
    return ModelExecutionBinding(
        binding_hash=canonical_sha256([_MODEL_BINDING_HASH_DOMAIN, values]),
        **values,  # type: ignore[arg-type]
    )


def _build_retrieval_binding(
    binding: RetrievalBindingConfiguration,
) -> ResolvedRetrievalBinding:
    values = {
        "binding_id": binding.binding_id,
        "provider_id": binding.provider_id,
        "route": binding.route,
        "query_embedding_role": binding.query_embedding_role,
        "disabled_feature": binding.disabled_feature,
        "disabled_value": binding.disabled_value,
        "request_constraint": binding.request_constraint,
        "proof_kind": binding.proof_kind,
    }
    return ResolvedRetrievalBinding(
        binding_hash=canonical_sha256([_RETRIEVAL_BINDING_HASH_DOMAIN, values]),
        **values,  # type: ignore[arg-type]
    )


def _build_cell_spec(
    *,
    ordinal: int,
    configuration: CellConfiguration,
    dataset: ResolvedDataset,
    roles_by_id: Mapping[ModelRoleId, ModelExecutionBinding],
    retrieval_by_id: Mapping[str, ResolvedRetrievalBinding],
    limits_hash: str,
    execution_hash: str,
) -> CellSpec:
    role_ids = (
        configuration.producer_role,
        configuration.embedding_role,
        configuration.answer_role,
        configuration.judge_role,
    )
    retrieval = retrieval_by_id[configuration.retrieval_binding]
    values = {
        "ordinal_1_indexed": ordinal,
        "cell_id": configuration.cell_id,
        "provider_id": configuration.provider_id,
        "adapter_profile_id": configuration.adapter_profile_id,
        "endpoint_variable": configuration.endpoint_variable,
        "credential_variable": configuration.credential_variable,
        "recipient": configuration.recipient,
        "dataset_id": dataset.dataset_id,
        "workload_id": dataset.workload_id,
        "selection": dataset.selection,
        "source_sha256": dataset.source_sha256,
        "case_manifest_hash": dataset.case_manifest_hash,
        "producer_role_id": role_ids[0],
        "embedding_role_id": role_ids[1],
        "answer_role_id": role_ids[2],
        "judge_role_id": role_ids[3],
        "model_role_binding_hashes": tuple(
            (role_id, roles_by_id[role_id].binding_hash) for role_id in role_ids
        ),
        "retrieval_binding_id": retrieval.binding_id,
        "retrieval_binding_hash": retrieval.binding_hash,
        "limits_hash": limits_hash,
        "execution_hash": execution_hash,
    }
    return CellSpec(
        cell_spec_hash=canonical_sha256([_CELL_SPEC_HASH_DOMAIN, values]),
        **values,  # type: ignore[arg-type]
    )


def _payload(
    *,
    comparison_id: str,
    dataset: ResolvedDataset,
    model_roles: tuple[ModelExecutionBinding, ...],
    retrieval: ResolvedRetrieval,
    limits: RunLimits,
    execution: ExecutionLimits,
    cells: tuple[CellSpec, ...],
) -> dict[str, object]:
    return {
        "schema_name": _SCHEMA_NAME,
        "schema_version": _SCHEMA_VERSION,
        "comparison_id": comparison_id,
        "dataset": _dataset_document(dataset),
        "model_roles": [_model_binding_document(role) for role in model_roles],
        "retrieval": {
            "generation": retrieval.generation,
            "bindings": [_retrieval_binding_document(binding) for binding in retrieval.bindings],
        },
        "limits": _limits_document(limits),
        "execution": _execution_document(execution),
        "cells": [_cell_document(cell) for cell in cells],
    }


def _dataset_document(dataset: ResolvedDataset) -> dict[str, object]:
    return {key: getattr(dataset, key) for key in sorted(_DATASET_KEYS)}


def _model_binding_document(binding: ModelExecutionBinding) -> dict[str, object]:
    return {key: getattr(binding, key) for key in sorted(_MODEL_ROLE_KEYS)}


def _retrieval_binding_document(binding: ResolvedRetrievalBinding) -> dict[str, object]:
    return {key: getattr(binding, key) for key in sorted(_RETRIEVAL_BINDING_KEYS)}


def _cell_document(cell: CellSpec) -> dict[str, object]:
    return {key: getattr(cell, key) for key in sorted(_CELL_KEYS)}


def _limits_document(limits: RunLimits) -> dict[str, object]:
    return {key: getattr(limits, key) for key in _LIMIT_KEYS}


def _execution_document(execution: ExecutionLimits) -> dict[str, object]:
    return {key: getattr(execution, key) for key in _EXECUTION_KEYS}


def _plan_hash(payload: Mapping[str, object]) -> str:
    return canonical_sha256([_PLAN_HASH_DOMAIN, payload])


def _parse_dataset(value: object) -> ResolvedDataset:
    document = _require_exact_mapping(value, _DATASET_KEYS, "resolved dataset")
    values = {key: _require_text(document[key], f"resolved dataset {key}") for key in _DATASET_KEYS}
    return ResolvedDataset(**values)


def _parse_model_roles(value: object) -> tuple[ModelExecutionBinding, ...]:
    if not isinstance(value, list):
        raise ResolvedPlanError("resolved model_roles must be a list")
    roles = tuple(_parse_model_role(item) for item in value)
    if tuple(role.role_id for role in roles) != MODEL_ROLE_IDS:
        raise ResolvedPlanError("resolved model role inventory or order is invalid")
    return roles


def _parse_model_role(value: object) -> ModelExecutionBinding:
    document = _require_exact_mapping(value, _MODEL_ROLE_KEYS, "resolved model role")
    role_id = document["role_id"]
    if type(role_id) is not str or role_id not in MODEL_ROLE_IDS:
        raise ResolvedPlanError("resolved model role ID is invalid")
    effort = document["thinking_effort"]
    if type(effort) is not str or effort not in (*DEEPSEEK_THINKING_EFFORT_SCALE, "not_applicable"):
        raise ResolvedPlanError(f"resolved model role {role_id} effort is invalid")
    expected_scale: tuple[GenerativeThinkingEffort, ...]
    expected_rank: int | None
    if role_id == "embedding":
        expected_scale = ()
        expected_rank = None
        if effort != "not_applicable":
            raise ResolvedPlanError("resolved embedding effort must be not_applicable")
    else:
        expected_scale = DEEPSEEK_THINKING_EFFORT_SCALE
        expected_rank = expected_scale.index(effort) + 1
    if (
        document["thinking_effort_scale"] != list(expected_scale)
        or document["thinking_effort_rank_1_indexed"] != expected_rank
    ):
        raise ResolvedPlanError(f"resolved model role {role_id} effort closure is invalid")
    values = {
        "role_id": role_id,
        "configured_model": _require_text(document["configured_model"], "configured model"),
        "runtime_model": _require_text(document["runtime_model"], "runtime model"),
        "thinking_effort": cast(ThinkingEffort, effort),
        "thinking_effort_scale": expected_scale,
        "thinking_effort_rank_1_indexed": expected_rank,
        "endpoint_variable": _require_text(document["endpoint_variable"], "endpoint variable"),
        "credential_variable": _require_text(
            document["credential_variable"], "credential variable"
        ),
        "recipient": _require_text(document["recipient"], "model recipient"),
        "execution_owner": _require_text(document["execution_owner"], "execution owner"),
        "proof_kind": _require_text(document["proof_kind"], "model proof kind"),
        "proof_reference": _require_text(document["proof_reference"], "model proof reference"),
        "usage_coverage": _require_text(document["usage_coverage"], "usage coverage"),
        "max_input_tokens": _require_positive_integer(
            document["max_input_tokens"], "model input ceiling"
        ),
        "max_output_tokens": _require_positive_integer(
            document["max_output_tokens"], "model output ceiling"
        ),
        "temperature": _require_text(document["temperature"], "model temperature"),
        "top_p": _require_text(document["top_p"], "model top_p"),
    }
    binding_hash = document["binding_hash"]
    if type(binding_hash) is not str or binding_hash != canonical_sha256(
        [_MODEL_BINDING_HASH_DOMAIN, values]
    ):
        raise ResolvedPlanError(f"resolved model role {role_id} binding hash is invalid")
    return ModelExecutionBinding(binding_hash=binding_hash, **values)  # type: ignore[arg-type]


def _parse_retrieval(value: object) -> ResolvedRetrieval:
    document = _require_exact_mapping(value, _RETRIEVAL_KEYS, "resolved retrieval")
    if document["generation"] != "disabled":
        raise ResolvedPlanError("resolved retrieval generation must be disabled")
    raw_bindings = document["bindings"]
    if not isinstance(raw_bindings, list):
        raise ResolvedPlanError("resolved retrieval bindings must be a list")
    bindings = tuple(_parse_retrieval_binding(item) for item in raw_bindings)
    if tuple(binding.binding_id for binding in bindings) != T10_RETRIEVAL_BINDING_IDS:
        raise ResolvedPlanError("resolved retrieval binding inventory or order is invalid")
    return ResolvedRetrieval(generation="disabled", bindings=bindings)


def _parse_retrieval_binding(value: object) -> ResolvedRetrievalBinding:
    document = _require_exact_mapping(value, _RETRIEVAL_BINDING_KEYS, "resolved retrieval binding")
    query_embedding_role = document["query_embedding_role"]
    if query_embedding_role != "embedding":
        raise ResolvedPlanError("resolved retrieval query role must be embedding")
    values = {
        key: _require_text(document[key], f"resolved retrieval {key}")
        for key in _RETRIEVAL_BINDING_KEYS
        if key != "binding_hash"
    }
    binding_hash = document["binding_hash"]
    if type(binding_hash) is not str or binding_hash != canonical_sha256(
        [_RETRIEVAL_BINDING_HASH_DOMAIN, values]
    ):
        raise ResolvedPlanError("resolved retrieval binding hash is invalid")
    return ResolvedRetrievalBinding(
        binding_hash=binding_hash,
        binding_id=values["binding_id"],
        provider_id=values["provider_id"],
        route=values["route"],
        query_embedding_role="embedding",
        disabled_feature=values["disabled_feature"],
        disabled_value=values["disabled_value"],
        request_constraint=values["request_constraint"],
        proof_kind=values["proof_kind"],
    )


def _parse_limits(value: object) -> RunLimits:
    document = _require_exact_mapping(value, frozenset(_LIMIT_KEYS), "resolved limits")
    integer_values = {
        key: _require_positive_integer(document[key], f"resolved limit {key}")
        for key in _LIMIT_KEYS
        if key not in {"max_cost", "currency"}
    }
    return RunLimits(
        **integer_values,
        max_cost=_require_text(document["max_cost"], "resolved max cost"),
        currency=_require_text(document["currency"], "resolved currency"),
    )


def _parse_execution(value: object) -> ExecutionLimits:
    document = _require_exact_mapping(value, frozenset(_EXECUTION_KEYS), "resolved execution")
    return ExecutionLimits(
        **{
            key: _require_positive_integer(document[key], f"resolved execution {key}")
            for key in _EXECUTION_KEYS
        }
    )


def _parse_cells(
    value: object,
    *,
    dataset: ResolvedDataset,
    model_roles: tuple[ModelExecutionBinding, ...],
    retrieval: ResolvedRetrieval,
    limits: RunLimits,
    execution: ExecutionLimits,
) -> tuple[CellSpec, ...]:
    if not isinstance(value, list):
        raise ResolvedPlanError("resolved cells must be a list")
    roles_by_id = {role.role_id: role for role in model_roles}
    retrieval_by_id = {binding.binding_id: binding for binding in retrieval.bindings}
    limits_hash = canonical_sha256([_LIMITS_HASH_DOMAIN, _limits_document(limits)])
    execution_hash = canonical_sha256([_EXECUTION_HASH_DOMAIN, _execution_document(execution)])
    cells = tuple(_parse_cell(item) for item in value)
    if tuple(cell.cell_id for cell in cells) != T10_CELL_IDS or tuple(
        cell.ordinal_1_indexed for cell in cells
    ) != (1, 2, 3):
        raise ResolvedPlanError("resolved cell order is invalid")
    for cell in cells:
        expected_roles = (
            cell.producer_role_id,
            cell.embedding_role_id,
            cell.answer_role_id,
            cell.judge_role_id,
        )
        if (
            cell.dataset_id != dataset.dataset_id
            or cell.workload_id != dataset.workload_id
            or cell.selection != dataset.selection
            or cell.source_sha256 != dataset.source_sha256
            or cell.case_manifest_hash != dataset.case_manifest_hash
            or cell.model_role_binding_hashes
            != tuple((role_id, roles_by_id[role_id].binding_hash) for role_id in expected_roles)
            or cell.retrieval_binding_id not in retrieval_by_id
            or cell.retrieval_binding_hash
            != retrieval_by_id[cell.retrieval_binding_id].binding_hash
            or cell.limits_hash != limits_hash
            or cell.execution_hash != execution_hash
        ):
            raise ResolvedPlanError(f"resolved cell {cell.cell_id} cross-binding is invalid")
    return cells


def _parse_cell(value: object) -> CellSpec:
    document = _require_exact_mapping(value, _CELL_KEYS, "resolved cell")
    role_fields = (
        "producer_role_id",
        "embedding_role_id",
        "answer_role_id",
        "judge_role_id",
    )
    parsed_roles: dict[str, ModelRoleId] = {}
    for field in role_fields:
        role_id = document[field]
        if type(role_id) is not str or role_id not in MODEL_ROLE_IDS:
            raise ResolvedPlanError(f"resolved cell {field} is invalid")
        parsed_roles[field] = role_id
    raw_role_hashes = document["model_role_binding_hashes"]
    if not isinstance(raw_role_hashes, list):
        raise ResolvedPlanError("resolved cell model role hashes must be a list")
    role_hashes: list[tuple[ModelRoleId, str]] = []
    for item in raw_role_hashes:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or type(item[0]) is not str
            or item[0] not in MODEL_ROLE_IDS
            or type(item[1]) is not str
            or _SHA256.fullmatch(item[1]) is None
        ):
            raise ResolvedPlanError("resolved cell model role hash binding is invalid")
        role_hashes.append((item[0], item[1]))
    values = {
        "ordinal_1_indexed": _require_positive_integer(
            document["ordinal_1_indexed"], "resolved cell ordinal"
        ),
        "cell_id": _require_text(document["cell_id"], "resolved cell ID"),
        "provider_id": _require_text(document["provider_id"], "resolved provider ID"),
        "adapter_profile_id": _require_text(
            document["adapter_profile_id"], "resolved adapter profile"
        ),
        "endpoint_variable": _require_text(
            document["endpoint_variable"], "resolved provider endpoint variable"
        ),
        "credential_variable": _require_text(
            document["credential_variable"], "resolved provider credential variable"
        ),
        "recipient": _require_text(document["recipient"], "resolved provider recipient"),
        "dataset_id": _require_text(document["dataset_id"], "resolved dataset ID"),
        "workload_id": _require_text(document["workload_id"], "resolved workload ID"),
        "selection": _require_text(document["selection"], "resolved dataset selection"),
        "source_sha256": _require_sha256(document["source_sha256"], "resolved source hash"),
        "case_manifest_hash": _require_sha256(
            document["case_manifest_hash"], "resolved case manifest hash"
        ),
        **parsed_roles,
        "model_role_binding_hashes": tuple(role_hashes),
        "retrieval_binding_id": _require_text(
            document["retrieval_binding_id"], "resolved retrieval binding ID"
        ),
        "retrieval_binding_hash": _require_sha256(
            document["retrieval_binding_hash"], "resolved retrieval binding hash"
        ),
        "limits_hash": _require_sha256(document["limits_hash"], "resolved limits hash"),
        "execution_hash": _require_sha256(document["execution_hash"], "resolved execution hash"),
    }
    cell_hash = document["cell_spec_hash"]
    if type(cell_hash) is not str or cell_hash != canonical_sha256(
        [_CELL_SPEC_HASH_DOMAIN, values]
    ):
        raise ResolvedPlanError("resolved cell spec hash is invalid")
    return CellSpec(cell_spec_hash=cell_hash, **values)  # type: ignore[arg-type]


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise ResolvedPlanError(f"{label} must be non-empty text")
    return value


def _require_positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ResolvedPlanError(f"{label} must be a positive integer")
    return value


def _require_sha256(value: object, label: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ResolvedPlanError(f"{label} must be lowercase SHA-256")
    return value


def _require_exact_mapping(
    value: object,
    expected_keys: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise ResolvedPlanError(f"{label} must be a JSON object with text keys")
    actual_keys = frozenset(cast(dict[str, object], value))
    if actual_keys != expected_keys:
        raise ResolvedPlanError(f"{label} keys do not match the closed schema")
    return cast(Mapping[str, object], value)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ResolvedPlanError(f"duplicate resolved plan key: {key}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> object:
    raise ResolvedPlanError(f"non-finite resolved-plan constant: {value}")


__all__ = [
    "CellSpec",
    "ModelExecutionBinding",
    "ResolvedDataset",
    "ResolvedModelRole",
    "ResolvedPlan",
    "ResolvedPlanError",
    "ResolvedRetrieval",
    "ResolvedRetrievalBinding",
    "build_resolved_plan",
    "load_resolved_plan_for_run",
    "resolved_plan_bytes",
]
