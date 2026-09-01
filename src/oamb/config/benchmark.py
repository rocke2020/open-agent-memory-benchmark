"""Strict loading for the complete T10 LongMemEval comparison configuration."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Literal, cast

import yaml  # type: ignore[import-untyped]
from yaml.constructor import ConstructorError  # type: ignore[import-untyped]
from yaml.nodes import MappingNode  # type: ignore[import-untyped]
from yaml.tokens import TagToken  # type: ignore[import-untyped]

ModelRoleId = Literal[
    "hindsight_extraction",
    "mem0_extraction",
    "openviking_semantic_understanding",
    "answer",
    "judge",
    "embedding",
]
GenerativeThinkingEffort = Literal["low", "high", "max"]
ThinkingEffort = Literal["low", "high", "max", "not_applicable"]
ExecutionOwner = Literal["provider_internal", "harness"]

MODEL_ROLE_IDS: tuple[ModelRoleId, ...] = (
    "hindsight_extraction",
    "mem0_extraction",
    "openviking_semantic_understanding",
    "answer",
    "judge",
    "embedding",
)
DEEPSEEK_THINKING_EFFORT_SCALE: tuple[GenerativeThinkingEffort, ...] = (
    "low",
    "high",
    "max",
)
T10_CELL_IDS = ("hindsight-lme6", "mem0-lme6", "openviking-lme6")
T10_RETRIEVAL_BINDING_IDS = ("hindsight", "mem0", "openviking")

_EXPECTED_DATASET = {
    "dataset_id": "longmemeval-s-cleaned",
    "workload_id": "lme30-native-smoke-plus-v1",
    "selection": "lme6",
    "path": "datasets/longmemeval-cleaned/longmemeval_s_cleaned.json",
    "revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
    "source_sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
    "case_manifest_hash": "3c0bc0e2e539b3f7ceca81569531c6a0fb5823ccc55426cb2b2cf9d295fa33e4",
}
_EXPECTED_CELLS = (
    ("hindsight-lme6", "hindsight", "hindsight-rest-v1", "hindsight_extraction", "hindsight"),
    ("mem0-lme6", "mem0", "mem0-rest-v1", "mem0_extraction", "mem0"),
    (
        "openviking-lme6",
        "openviking",
        "openviking-session-rest-v1",
        "openviking_semantic_understanding",
        "openviking",
    ),
)
_EXPECTED_RETRIEVAL_BINDINGS = (
    (
        "hindsight",
        "hindsight",
        "recall",
        "embedding",
        "reflect",
        "absent",
        "omit_reflect",
        "complete_outbound_trace",
    ),
    (
        "mem0",
        "mem0",
        "/search",
        "embedding",
        "native_reranking",
        "disabled",
        "rerank_field_absent_by_schema",
        "effective_config_and_complete_outbound_trace",
    ),
    (
        "openviking",
        "openviking",
        "/api/v1/search/find",
        "embedding",
        "session_intent",
        "not_applicable",
        "no_session_id_on_find",
        "source_verified_endpoint_and_complete_outbound_trace",
    ),
)
_EXPECTED_EFFORTS: dict[ModelRoleId, ThinkingEffort] = {
    "hindsight_extraction": "low",
    "mem0_extraction": "low",
    "openviking_semantic_understanding": "low",
    "answer": "low",
    "judge": "high",
    "embedding": "not_applicable",
}
_EXPECTED_MODELS: dict[ModelRoleId, str] = {
    "hindsight_extraction": "deepseek-v4-flash",
    "mem0_extraction": "deepseek-v4-flash",
    "openviking_semantic_understanding": "deepseek-v4-flash",
    "answer": "deepseek-v4-pro",
    "judge": "deepseek-v4-flash",
    "embedding": "qwen3-embedding:0.6b",
}
MODEL_EXECUTION_OWNER_BY_ROLE: dict[ModelRoleId, ExecutionOwner] = {
    "hindsight_extraction": "provider_internal",
    "mem0_extraction": "provider_internal",
    "openviking_semantic_understanding": "provider_internal",
    "answer": "harness",
    "judge": "harness",
    "embedding": "provider_internal",
}

_TOP_LEVEL_KEYS = frozenset(
    {"comparison", "dataset", "cells", "models", "retrieval", "limits", "execution"}
)
_DATASET_KEYS = frozenset(_EXPECTED_DATASET)
_CELL_KEYS = frozenset(
    {
        "cell_id",
        "provider_id",
        "adapter_profile_id",
        "endpoint_variable",
        "credential_variable",
        "recipient",
        "producer_role",
        "embedding_role",
        "answer_role",
        "judge_role",
        "retrieval_binding",
    }
)
_MODEL_KEYS = frozenset(
    {
        "model",
        "runtime_model",
        "thinking_effort",
        "endpoint_variable",
        "credential_variable",
        "recipient",
        "execution_owner",
        "proof_kind",
        "proof_reference",
        "usage_coverage",
    }
)
_RETRIEVAL_KEYS = frozenset({"generation", "bindings"})
_RETRIEVAL_BINDING_KEYS = frozenset(
    {
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
_LIMIT_KEYS = frozenset(
    {
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
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "max_parallel_datasets",
        "max_parallel_providers_per_dataset",
        "max_parallel_history_ingestions_per_provider",
        "max_parallel_questions_per_provider",
    }
)
_ENVIRONMENT_REFERENCE = re.compile(r"OAMB_[A-Z0-9_]+")
_CURRENCY = re.compile(r"[A-Z]{3}")
_COST = re.compile(r"(?:0|[1-9][0-9]*)\.[0-9]{2}")


class BenchmarkConfigurationError(ValueError):
    """The benchmark YAML is absent, malformed, or outside its closed schema."""


class _DuplicateRejectingSafeLoader(yaml.SafeLoader):  # type: ignore[misc]
    pass


def _construct_unique_mapping(
    loader: _DuplicateRejectingSafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    mapping: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping key is not hashable",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate mapping key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_DuplicateRejectingSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclass(frozen=True, slots=True)
class DatasetConfiguration:
    dataset_id: str
    workload_id: str
    selection: str
    path: str
    revision: str
    source_sha256: str
    case_manifest_hash: str


@dataclass(frozen=True, slots=True)
class CellConfiguration:
    cell_id: str
    provider_id: str
    adapter_profile_id: str
    endpoint_variable: str
    credential_variable: str
    recipient: str
    producer_role: ModelRoleId
    embedding_role: ModelRoleId
    answer_role: ModelRoleId
    judge_role: ModelRoleId
    retrieval_binding: str


@dataclass(frozen=True, slots=True)
class ModelRoleConfiguration:
    model: str
    runtime_model: str
    thinking_effort: ThinkingEffort
    endpoint_variable: str
    credential_variable: str
    recipient: str
    execution_owner: ExecutionOwner
    proof_kind: str
    proof_reference: str
    usage_coverage: str

    @property
    def thinking_effort_scale(self) -> tuple[GenerativeThinkingEffort, ...]:
        if self.thinking_effort == "not_applicable":
            return ()
        return DEEPSEEK_THINKING_EFFORT_SCALE

    @property
    def thinking_effort_rank_1_indexed(self) -> int | None:
        if self.thinking_effort == "not_applicable":
            return None
        return DEEPSEEK_THINKING_EFFORT_SCALE.index(self.thinking_effort) + 1


@dataclass(frozen=True, slots=True)
class ModelRoleConfigurations:
    hindsight_extraction: ModelRoleConfiguration
    mem0_extraction: ModelRoleConfiguration
    openviking_semantic_understanding: ModelRoleConfiguration
    answer: ModelRoleConfiguration
    judge: ModelRoleConfiguration
    embedding: ModelRoleConfiguration

    @property
    def ordered_role_ids(self) -> tuple[ModelRoleId, ...]:
        return MODEL_ROLE_IDS

    def ordered_items(self) -> tuple[tuple[ModelRoleId, ModelRoleConfiguration], ...]:
        return tuple((role_id, getattr(self, role_id)) for role_id in MODEL_ROLE_IDS)


@dataclass(frozen=True, slots=True)
class RetrievalBindingConfiguration:
    binding_id: str
    provider_id: str
    route: str
    query_embedding_role: ModelRoleId
    disabled_feature: str
    disabled_value: str
    request_constraint: str
    proof_kind: str


@dataclass(frozen=True, slots=True)
class RetrievalConfiguration:
    generation: Literal["disabled"]
    bindings: tuple[RetrievalBindingConfiguration, ...]


@dataclass(frozen=True, slots=True)
class RunLimits:
    max_attempts_per_case: int
    max_budgeted_attempts: int
    memory_operation_timeout_seconds: int
    model_call_timeout_seconds: int
    total_wall_time_seconds: int
    max_input_tokens: int
    max_output_tokens: int
    max_recall_context_tokens_per_case: int
    max_storage_bytes: int
    max_peak_memory_bytes: int
    max_cost: str
    currency: str

    def as_tuple(
        self,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int, str, str]:
        return (
            self.max_attempts_per_case,
            self.max_budgeted_attempts,
            self.memory_operation_timeout_seconds,
            self.model_call_timeout_seconds,
            self.total_wall_time_seconds,
            self.max_input_tokens,
            self.max_output_tokens,
            self.max_recall_context_tokens_per_case,
            self.max_storage_bytes,
            self.max_peak_memory_bytes,
            self.max_cost,
            self.currency,
        )


@dataclass(frozen=True, slots=True)
class ExecutionLimits:
    max_parallel_datasets: int
    max_parallel_providers_per_dataset: int
    max_parallel_history_ingestions_per_provider: int
    max_parallel_questions_per_provider: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            self.max_parallel_datasets,
            self.max_parallel_providers_per_dataset,
            self.max_parallel_history_ingestions_per_provider,
            self.max_parallel_questions_per_provider,
        )


@dataclass(frozen=True, slots=True)
class BenchmarkConfiguration:
    comparison_id: str
    dataset: DatasetConfiguration
    cells: tuple[CellConfiguration, ...]
    models: ModelRoleConfigurations
    retrieval: RetrievalConfiguration
    limits: RunLimits
    execution: ExecutionLimits


def load_benchmark_configuration(path: Path) -> BenchmarkConfiguration:
    """Load one strict comparison without reading endpoints or credential values."""

    try:
        content = Path(path).read_text(encoding="utf-8")
        if any(isinstance(token, TagToken) for token in yaml.scan(content)):
            raise BenchmarkConfigurationError("explicit YAML tags are not permitted")
        document = yaml.load(content, Loader=_DuplicateRejectingSafeLoader)
    except BenchmarkConfigurationError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise BenchmarkConfigurationError(f"cannot load benchmark configuration: {path}") from exc
    return _parse_benchmark_configuration(document)


def _parse_benchmark_configuration(document: object) -> BenchmarkConfiguration:
    root = _require_exact_mapping(document, _TOP_LEVEL_KEYS, "benchmark configuration")
    comparison_id = _require_text(root["comparison"], "comparison")
    dataset = _parse_dataset(root["dataset"])
    models = _parse_model_roles(root["models"])
    retrieval = _parse_retrieval(root["retrieval"])
    cells = _parse_cells(root["cells"], models=models, retrieval=retrieval)
    limits = _parse_limits(root["limits"])
    execution = _parse_execution_limits(root["execution"])
    return BenchmarkConfiguration(
        comparison_id=comparison_id,
        dataset=dataset,
        cells=cells,
        models=models,
        retrieval=retrieval,
        limits=limits,
        execution=execution,
    )


def _parse_dataset(value: object) -> DatasetConfiguration:
    document = _require_exact_mapping(value, _DATASET_KEYS, "dataset")
    parsed = {key: _require_text(document[key], f"dataset {key}") for key in _DATASET_KEYS}
    if parsed != _EXPECTED_DATASET:
        raise BenchmarkConfigurationError(
            "v0.1 T10 selects only the pinned LongMemEval LME-6 dataset"
        )
    return DatasetConfiguration(**parsed)


def _parse_cells(
    value: object,
    *,
    models: ModelRoleConfigurations,
    retrieval: RetrievalConfiguration,
) -> tuple[CellConfiguration, ...]:
    items = _require_sequence(value, "cells")
    cells: list[CellConfiguration] = []
    for ordinal, item in enumerate(items, start=1):
        document = _require_exact_mapping(item, _CELL_KEYS, f"cell {ordinal}")
        provider_id = _require_text(document["provider_id"], f"cell {ordinal} provider")
        credential_value = document["credential_variable"]
        credential_variable = (
            "not_applicable"
            if provider_id == "hindsight" and credential_value == "not_applicable"
            else _require_environment_reference(credential_value, f"cell {ordinal} credential")
        )
        producer_role = _require_role_id(document["producer_role"], f"cell {ordinal} producer")
        embedding_role = _require_role_id(document["embedding_role"], f"cell {ordinal} embedding")
        answer_role = _require_role_id(document["answer_role"], f"cell {ordinal} answer")
        judge_role = _require_role_id(document["judge_role"], f"cell {ordinal} judge")
        cells.append(
            CellConfiguration(
                cell_id=_require_text(document["cell_id"], f"cell {ordinal} ID"),
                provider_id=provider_id,
                adapter_profile_id=_require_text(
                    document["adapter_profile_id"], f"cell {ordinal} adapter profile"
                ),
                endpoint_variable=_require_environment_reference(
                    document["endpoint_variable"], f"cell {ordinal} endpoint"
                ),
                credential_variable=credential_variable,
                recipient=_require_text(document["recipient"], f"cell {ordinal} recipient"),
                producer_role=producer_role,
                embedding_role=embedding_role,
                answer_role=answer_role,
                judge_role=judge_role,
                retrieval_binding=_require_text(
                    document["retrieval_binding"], f"cell {ordinal} retrieval binding"
                ),
            )
        )
    actual = tuple(
        (
            cell.cell_id,
            cell.provider_id,
            cell.adapter_profile_id,
            cell.producer_role,
            cell.retrieval_binding,
        )
        for cell in cells
    )
    if actual != _EXPECTED_CELLS:
        raise BenchmarkConfigurationError(
            "cells must be the ordered Hindsight, Mem0, and OpenViking-session LME-6 profiles"
        )
    retrieval_by_id = {binding.binding_id: binding for binding in retrieval.bindings}
    for cell in cells:
        if (
            cell.embedding_role != "embedding"
            or cell.answer_role != "answer"
            or cell.judge_role != "judge"
            or cell.retrieval_binding not in retrieval_by_id
            or retrieval_by_id[cell.retrieval_binding].provider_id != cell.provider_id
        ):
            raise BenchmarkConfigurationError(f"cell {cell.cell_id} role route is incomplete")
        for role_id in (
            cell.producer_role,
            cell.embedding_role,
            cell.answer_role,
            cell.judge_role,
        ):
            getattr(models, role_id)
    return tuple(cells)


def _parse_model_roles(value: object) -> ModelRoleConfigurations:
    document = _require_exact_mapping(value, frozenset(MODEL_ROLE_IDS), "models")
    parsed = {role_id: _parse_model_role(role_id, document[role_id]) for role_id in MODEL_ROLE_IDS}
    return ModelRoleConfigurations(**parsed)


def _parse_model_role(role_id: ModelRoleId, value: object) -> ModelRoleConfiguration:
    document = _require_exact_mapping(value, _MODEL_KEYS, f"model role {role_id}")
    model = _require_text(document["model"], f"model role {role_id} configured model")
    runtime_model = _require_text(document["runtime_model"], f"model role {role_id} runtime model")
    effort = document["thinking_effort"]
    if type(effort) is not str or effort not in (*DEEPSEEK_THINKING_EFFORT_SCALE, "not_applicable"):
        raise BenchmarkConfigurationError(f"model role {role_id} has an invalid thinking effort")
    if effort != _EXPECTED_EFFORTS[role_id]:
        raise BenchmarkConfigurationError(
            f"model role {role_id} does not match the T10 effort profile"
        )
    expected_model = _EXPECTED_MODELS[role_id]
    if model != expected_model:
        raise BenchmarkConfigurationError(
            f"model role {role_id} does not match the T10 model profile"
        )
    endpoint_variable = _require_environment_reference(
        document["endpoint_variable"], f"model role {role_id} endpoint"
    )
    credential_variable = _require_credential_reference(
        document["credential_variable"], role_id=role_id
    )
    owner = document["execution_owner"]
    if owner != MODEL_EXECUTION_OWNER_BY_ROLE[role_id]:
        raise BenchmarkConfigurationError(f"model role {role_id} has the wrong execution owner")
    if owner == "provider_internal" and runtime_model != model:
        raise BenchmarkConfigurationError(
            f"model role {role_id} requires the same configured and runtime model "
            "because provider proof exposes one model identity"
        )
    if runtime_model != expected_model:
        raise BenchmarkConfigurationError(
            f"model role {role_id} does not match the T10 model profile"
        )
    proof_kind = _require_text(document["proof_kind"], f"model role {role_id} proof kind")
    expected_proof_kind = (
        "model_dimension_probe"
        if role_id == "embedding"
        else "sealed_outbound_request_and_response"
        if owner == "harness"
        else "effective_config_and_outbound_request"
    )
    if proof_kind != expected_proof_kind:
        raise BenchmarkConfigurationError(f"model role {role_id} has the wrong proof kind")
    proof_reference = _require_text(
        document["proof_reference"], f"model role {role_id} proof reference"
    )
    expected_proof_reference = (
        "not_applicable" if role_id == "embedding" else f"reasoning_effort={effort}"
    )
    if proof_reference != expected_proof_reference:
        raise BenchmarkConfigurationError(f"model role {role_id} effort proof is incomplete")
    if document["usage_coverage"] != "classified":
        raise BenchmarkConfigurationError(f"model role {role_id} usage coverage is incomplete")
    return ModelRoleConfiguration(
        model=model,
        runtime_model=runtime_model,
        thinking_effort=effort,
        endpoint_variable=endpoint_variable,
        credential_variable=credential_variable,
        recipient=_require_text(document["recipient"], f"model role {role_id} recipient"),
        execution_owner=owner,
        proof_kind=proof_kind,
        proof_reference=proof_reference,
        usage_coverage="classified",
    )


def _parse_retrieval(value: object) -> RetrievalConfiguration:
    document = _require_exact_mapping(value, _RETRIEVAL_KEYS, "retrieval")
    if document["generation"] != "disabled":
        raise BenchmarkConfigurationError("retrieval generation must be disabled")
    raw_bindings = _require_sequence(document["bindings"], "retrieval bindings")
    bindings: list[RetrievalBindingConfiguration] = []
    for ordinal, item in enumerate(raw_bindings, start=1):
        raw = _require_exact_mapping(item, _RETRIEVAL_BINDING_KEYS, f"retrieval binding {ordinal}")
        bindings.append(
            RetrievalBindingConfiguration(
                binding_id=_require_text(raw["binding_id"], "retrieval binding ID"),
                provider_id=_require_text(raw["provider_id"], "retrieval provider"),
                route=_require_text(raw["route"], "retrieval route"),
                query_embedding_role=_require_role_id(
                    raw["query_embedding_role"], "retrieval query embedding role"
                ),
                disabled_feature=_require_text(
                    raw["disabled_feature"], "retrieval disabled feature"
                ),
                disabled_value=_require_text(raw["disabled_value"], "retrieval disabled value"),
                request_constraint=_require_text(
                    raw["request_constraint"], "retrieval request constraint"
                ),
                proof_kind=_require_text(raw["proof_kind"], "retrieval proof kind"),
            )
        )
    actual = tuple(
        (
            item.binding_id,
            item.provider_id,
            item.route,
            item.query_embedding_role,
            item.disabled_feature,
            item.disabled_value,
            item.request_constraint,
            item.proof_kind,
        )
        for item in bindings
    )
    if actual != _EXPECTED_RETRIEVAL_BINDINGS:
        raise BenchmarkConfigurationError("retrieval bindings do not match the exact T10 profile")
    return RetrievalConfiguration(generation="disabled", bindings=tuple(bindings))


def _parse_limits(value: object) -> RunLimits:
    document = _require_exact_mapping(value, _LIMIT_KEYS, "limits")
    integers = {
        key: _require_positive_integer(document[key], f"limit {key}")
        for key in _LIMIT_KEYS
        if key not in {"max_cost", "currency"}
    }
    max_cost = document["max_cost"]
    if type(max_cost) is not str or _COST.fullmatch(max_cost) is None:
        raise BenchmarkConfigurationError("limit max_cost must be a finite decimal string")
    try:
        parsed_cost = Decimal(max_cost)
    except InvalidOperation as exc:
        raise BenchmarkConfigurationError("limit max_cost is invalid") from exc
    if not parsed_cost.is_finite() or parsed_cost <= 0:
        raise BenchmarkConfigurationError("limit max_cost must be finite and positive")
    currency = document["currency"]
    if type(currency) is not str or _CURRENCY.fullmatch(currency) is None:
        raise BenchmarkConfigurationError("limit currency must be a three-letter code")
    return RunLimits(**integers, max_cost=max_cost, currency=currency)


def _parse_execution_limits(value: object) -> ExecutionLimits:
    document = _require_exact_mapping(value, _EXECUTION_KEYS, "execution")
    limits = {
        key: _require_positive_integer(document[key], f"execution limit {key}")
        for key in _EXECUTION_KEYS
    }
    return ExecutionLimits(**limits)


def _require_role_id(value: object, label: str) -> ModelRoleId:
    if type(value) is not str or value not in MODEL_ROLE_IDS:
        raise BenchmarkConfigurationError(f"{label} is not a configured model role")
    return value


def _require_environment_reference(value: object, label: str) -> str:
    if type(value) is not str or _ENVIRONMENT_REFERENCE.fullmatch(value) is None:
        raise BenchmarkConfigurationError(f"{label} must name an OAMB environment variable")
    return value


def _require_credential_reference(value: object, *, role_id: ModelRoleId) -> str:
    if role_id == "embedding":
        if value != "not_applicable":
            raise BenchmarkConfigurationError("embedding credential must be not_applicable")
        return "not_applicable"
    return _require_environment_reference(value, f"model role {role_id} credential")


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise BenchmarkConfigurationError(f"{label} must be non-empty text")
    return value


def _require_positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise BenchmarkConfigurationError(f"{label} must be a positive finite integer")
    return value


def _require_sequence(value: object, label: str) -> list[object]:
    if not isinstance(value, list) or not value:
        raise BenchmarkConfigurationError(f"{label} must be a non-empty sequence")
    return cast(list[object], value)


def _require_exact_mapping(
    value: object,
    expected_keys: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise BenchmarkConfigurationError(f"{label} must be a mapping with text keys")
    actual_keys = frozenset(cast(dict[str, object], value))
    if actual_keys != expected_keys:
        missing = ", ".join(sorted(expected_keys - actual_keys)) or "none"
        extra = ", ".join(sorted(actual_keys - expected_keys)) or "none"
        raise BenchmarkConfigurationError(
            f"{label} keys do not match (missing: {missing}; extra: {extra})"
        )
    return cast(Mapping[str, object], value)


__all__ = [
    "BenchmarkConfiguration",
    "BenchmarkConfigurationError",
    "CellConfiguration",
    "DEEPSEEK_THINKING_EFFORT_SCALE",
    "DatasetConfiguration",
    "ExecutionLimits",
    "GenerativeThinkingEffort",
    "MODEL_ROLE_IDS",
    "MODEL_EXECUTION_OWNER_BY_ROLE",
    "ModelRoleConfiguration",
    "ModelRoleConfigurations",
    "ModelRoleId",
    "RetrievalBindingConfiguration",
    "RetrievalConfiguration",
    "RunLimits",
    "T10_CELL_IDS",
    "T10_RETRIEVAL_BINDING_IDS",
    "ThinkingEffort",
    "load_benchmark_configuration",
]
