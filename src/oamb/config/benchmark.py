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

_DATASET_COMMON = {
    "dataset_id": "longmemeval-s-cleaned",
    "path": "datasets/longmemeval-cleaned/longmemeval_s_cleaned.json",
    "revision": "98d7416c24c778c2fee6e6f3006e7a073259d48f",
    "source_sha256": "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
}
_EXPECTED_DATASETS = {
    "lme6": {
        **_DATASET_COMMON,
        "workload_id": "lme30-native-smoke-plus-v1",
        "selection": "lme6",
        "case_manifest_hash": "3c0bc0e2e539b3f7ceca81569531c6a0fb5823ccc55426cb2b2cf9d295fa33e4",
    },
    "lme60": {
        **_DATASET_COMMON,
        "workload_id": "lme60-balanced-v1",
        "selection": "lme60",
        "case_manifest_hash": "90b2669f7b893e59d404549f5803882bcd6640ce82520a9bf09672cc79464c80",
    },
}


def _expected_cells(selection: str) -> tuple[tuple[str, str, str, str, str], ...]:
    suffix = "lme6" if selection == "lme6" else "lme60"
    return (
        (
            f"hindsight-{suffix}",
            "hindsight",
            "hindsight-rest-v1",
            "hindsight_extraction",
            "hindsight",
        ),
        (f"mem0-{suffix}", "mem0", "mem0-rest-v1", "mem0_extraction", "mem0"),
        (
            f"openviking-{suffix}",
            "openviking",
            "openviking-session-rest-v1",
            "openviking_semantic_understanding",
            "openviking",
        ),
    )


def expected_cell_ids_for_selection(selection: str) -> tuple[str, ...]:
    """Return the closed cell inventory for a supported workload selection."""

    if selection not in _EXPECTED_DATASETS:
        raise BenchmarkConfigurationError(f"unsupported dataset selection: {selection}")
    return tuple(cell[0] for cell in _expected_cells(selection))


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
MODEL_EXECUTION_OWNER_BY_ROLE: dict[ModelRoleId, ExecutionOwner] = {
    "hindsight_extraction": "provider_internal",
    "mem0_extraction": "provider_internal",
    "openviking_semantic_understanding": "provider_internal",
    "answer": "harness",
    "judge": "harness",
    "embedding": "provider_internal",
}
_REQUIRED_TOP_LEVEL_KEYS = frozenset(
    {
        "comparison",
        "dataset",
        "embedding",
        "cells",
        "llm_profiles",
        "models",
        "retrieval",
        "execution",
    }
)
_OPTIONAL_TOP_LEVEL_KEYS = frozenset({"decision"})
_DATASET_KEYS = frozenset(_DATASET_COMMON) | frozenset(
    {"workload_id", "selection", "case_manifest_hash"}
)
_EMBEDDING_KEYS = frozenset({"managed_local_endpoint"})
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
_LLM_PROFILE_KEYS = frozenset({"light_model", "deep_model"})
_DECISION_KEYS = frozenset({"minimum_accuracy_delta", "maximum_exact_mcnemar_p_value"})
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
_EXECUTION_KEYS = frozenset(
    {
        "max_retries_per_operation",
        "extraction_max_retries",
        "model_max_attempts",
        "model_transport_max_retries",
        "operation_timeout_seconds",
    }
)
_ENVIRONMENT_REFERENCE = re.compile(r"(?:LLM|OAMB)_[A-Z0-9_]+")


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
class EmbeddingConfiguration:
    managed_local_endpoint: str


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
class LlmProfilesConfiguration:
    light_model: str
    deep_model: str


@dataclass(frozen=True, slots=True)
class ModelRoleConfiguration:
    model: str
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
class EvaluationControls:
    max_retries_per_operation: int
    operation_timeout_seconds: int
    extraction_max_retries: int = 10
    model_max_attempts: int = 6
    model_transport_max_retries: int = 2

    def as_tuple(self) -> tuple[int, int, int, int, int]:
        return (
            self.max_retries_per_operation,
            self.extraction_max_retries,
            self.model_max_attempts,
            self.model_transport_max_retries,
            self.operation_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class DecisionConfiguration:
    minimum_accuracy_delta: str
    maximum_exact_mcnemar_p_value: str


@dataclass(frozen=True, slots=True)
class BenchmarkConfiguration:
    comparison_id: str
    dataset: DatasetConfiguration
    embedding: EmbeddingConfiguration
    cells: tuple[CellConfiguration, ...]
    llm_profiles: LlmProfilesConfiguration
    models: ModelRoleConfigurations
    retrieval: RetrievalConfiguration
    evaluation_controls: EvaluationControls
    decision: DecisionConfiguration | None


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
    root = _require_mapping_with_optional_keys(
        document,
        required_keys=_REQUIRED_TOP_LEVEL_KEYS,
        optional_keys=_OPTIONAL_TOP_LEVEL_KEYS,
        label="benchmark configuration",
    )
    comparison_id = _require_text(root["comparison"], "comparison")
    dataset = _parse_dataset(root["dataset"])
    embedding = _parse_embedding(root["embedding"])
    llm_profiles = _parse_llm_profiles(root["llm_profiles"])
    models = _parse_model_roles(root["models"], llm_profiles=llm_profiles)
    retrieval = _parse_retrieval(root["retrieval"])
    cells = _parse_cells(
        root["cells"],
        selection=dataset.selection,
        models=models,
        retrieval=retrieval,
    )
    evaluation_controls = _parse_evaluation_controls(root["execution"])
    decision = _parse_decision(root.get("decision"), selection=dataset.selection)
    return BenchmarkConfiguration(
        comparison_id=comparison_id,
        dataset=dataset,
        embedding=embedding,
        cells=cells,
        llm_profiles=llm_profiles,
        models=models,
        retrieval=retrieval,
        evaluation_controls=evaluation_controls,
        decision=decision,
    )


def _parse_dataset(value: object) -> DatasetConfiguration:
    document = _require_exact_mapping(value, _DATASET_KEYS, "dataset")
    parsed = {key: _require_text(document[key], f"dataset {key}") for key in _DATASET_KEYS}
    selection = parsed["selection"]
    if selection not in _EXPECTED_DATASETS or parsed != _EXPECTED_DATASETS[selection]:
        raise BenchmarkConfigurationError(
            "dataset does not match a pinned v0.1 LongMemEval profile"
        )
    return DatasetConfiguration(**parsed)


def _parse_embedding(value: object) -> EmbeddingConfiguration:
    document = _require_exact_mapping(value, _EMBEDDING_KEYS, "embedding")
    endpoint = _require_text(document["managed_local_endpoint"], "managed local endpoint")
    if endpoint != "http://127.0.0.1:18000/v1":
        raise BenchmarkConfigurationError(
            "managed local embedding endpoint does not match the v0.1 profile"
        )
    return EmbeddingConfiguration(managed_local_endpoint=endpoint)


def _parse_cells(
    value: object,
    *,
    selection: str,
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
    if actual != _expected_cells(selection):
        raise BenchmarkConfigurationError(
            "cells must be the ordered Hindsight, Mem0, and OpenViking-session profiles "
            f"for {selection}"
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


def _parse_llm_profiles(value: object) -> LlmProfilesConfiguration:
    document = _require_exact_mapping(value, _LLM_PROFILE_KEYS, "LLM profiles")
    return LlmProfilesConfiguration(
        light_model=_require_text(document["light_model"], "light model profile"),
        deep_model=_require_text(document["deep_model"], "deep model profile"),
    )


def _parse_model_roles(
    value: object,
    *,
    llm_profiles: LlmProfilesConfiguration,
) -> ModelRoleConfigurations:
    document = _require_exact_mapping(value, frozenset(MODEL_ROLE_IDS), "models")
    parsed = {role_id: _parse_model_role(role_id, document[role_id]) for role_id in MODEL_ROLE_IDS}
    result = ModelRoleConfigurations(**parsed)
    for role_id in (
        "hindsight_extraction",
        "mem0_extraction",
        "openviking_semantic_understanding",
        "judge",
    ):
        if getattr(result, role_id).model != llm_profiles.light_model:
            raise BenchmarkConfigurationError(
                f"model role {role_id} must use the light_model profile"
            )
    if result.answer.model != llm_profiles.deep_model:
        raise BenchmarkConfigurationError("model role answer must use the deep_model profile")
    return result


def _parse_model_role(role_id: ModelRoleId, value: object) -> ModelRoleConfiguration:
    document = _require_exact_mapping(value, _MODEL_KEYS, f"model role {role_id}")
    model = _require_text(document["model"], f"model role {role_id} model")
    effort = document["thinking_effort"]
    if type(effort) is not str or effort not in (*DEEPSEEK_THINKING_EFFORT_SCALE, "not_applicable"):
        raise BenchmarkConfigurationError(f"model role {role_id} has an invalid thinking effort")
    effort = cast(ThinkingEffort, effort)
    if role_id == "embedding" and effort != "not_applicable":
        raise BenchmarkConfigurationError(
            "model role embedding requires not_applicable thinking effort"
        )
    if role_id != "embedding" and effort == "not_applicable":
        raise BenchmarkConfigurationError(
            f"generative model role {role_id} requires a DeepSeek thinking effort"
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


def _parse_evaluation_controls(value: object) -> EvaluationControls:
    document = _require_exact_mapping(value, _EXECUTION_KEYS, "execution")
    retries = _require_non_negative_integer(
        document["max_retries_per_operation"],
        "execution max_retries_per_operation",
    )
    if retries != 2:
        raise BenchmarkConfigurationError("execution batch retry count must equal 2")
    timeout = _require_positive_integer(
        document["operation_timeout_seconds"],
        "execution operation_timeout_seconds",
    )
    extraction_retries = _require_non_negative_integer(
        document["extraction_max_retries"], "execution extraction_max_retries"
    )
    model_attempts = _require_positive_integer(
        document["model_max_attempts"], "execution model_max_attempts"
    )
    model_transport_retries = _require_non_negative_integer(
        document["model_transport_max_retries"], "execution model_transport_max_retries"
    )
    if (extraction_retries, model_attempts, model_transport_retries) != (10, 6, 2):
        raise BenchmarkConfigurationError("execution layered retry values do not match the profile")
    return EvaluationControls(
        max_retries_per_operation=retries,
        operation_timeout_seconds=timeout,
        extraction_max_retries=extraction_retries,
        model_max_attempts=model_attempts,
        model_transport_max_retries=model_transport_retries,
    )


def _parse_decision(value: object, *, selection: str) -> DecisionConfiguration | None:
    if selection == "lme6":
        if value is not None:
            raise BenchmarkConfigurationError("decision must be absent for descriptive LME-6")
        return None
    if value is None:
        raise BenchmarkConfigurationError("decision is required for LME-60")
    document = _require_exact_mapping(value, _DECISION_KEYS, "decision")
    values = {
        key: _require_probability_string(document[key], f"decision {key}") for key in _DECISION_KEYS
    }
    expected = {
        "minimum_accuracy_delta": "0.05",
        "maximum_exact_mcnemar_p_value": "0.05",
    }
    if values != expected:
        raise BenchmarkConfigurationError("decision does not match the frozen LME-60 policy")
    return DecisionConfiguration(**values)


def _require_role_id(value: object, label: str) -> ModelRoleId:
    if type(value) is not str or value not in MODEL_ROLE_IDS:
        raise BenchmarkConfigurationError(f"{label} is not a configured model role")
    return value


def _require_environment_reference(value: object, label: str) -> str:
    if type(value) is not str or _ENVIRONMENT_REFERENCE.fullmatch(value) is None:
        raise BenchmarkConfigurationError(f"{label} must name a supported environment variable")
    return value


def _require_credential_reference(value: object, *, role_id: ModelRoleId) -> str:
    return _require_environment_reference(value, f"model role {role_id} credential")


def _require_text(value: object, label: str) -> str:
    if type(value) is not str or not value or value.strip() != value:
        raise BenchmarkConfigurationError(f"{label} must be non-empty text")
    return value


def _require_positive_integer(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise BenchmarkConfigurationError(f"{label} must be a positive finite integer")
    return value


def _require_non_negative_integer(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise BenchmarkConfigurationError(f"{label} must be a non-negative finite integer")
    return value


def _require_probability_string(value: object, label: str) -> str:
    if type(value) is not str:
        raise BenchmarkConfigurationError(f"{label} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise BenchmarkConfigurationError(f"{label} must be a decimal string") from exc
    if not parsed.is_finite() or parsed <= 0 or parsed > 1:
        raise BenchmarkConfigurationError(f"{label} must be within (0, 1]")
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


def _require_mapping_with_optional_keys(
    value: object,
    *,
    required_keys: frozenset[str],
    optional_keys: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if not isinstance(value, dict) or any(type(key) is not str for key in value):
        raise BenchmarkConfigurationError(f"{label} must be a mapping with text keys")
    document = cast(dict[str, object], value)
    actual_keys = frozenset(document)
    missing = required_keys - actual_keys
    extra = actual_keys - required_keys - optional_keys
    if missing or extra:
        raise BenchmarkConfigurationError(
            f"{label} keys do not match (missing: {', '.join(sorted(missing)) or 'none'}; "
            f"extra: {', '.join(sorted(extra)) or 'none'})"
        )
    return document


__all__ = [
    "BenchmarkConfiguration",
    "BenchmarkConfigurationError",
    "CellConfiguration",
    "DEEPSEEK_THINKING_EFFORT_SCALE",
    "DatasetConfiguration",
    "DecisionConfiguration",
    "EvaluationControls",
    "GenerativeThinkingEffort",
    "MODEL_ROLE_IDS",
    "MODEL_EXECUTION_OWNER_BY_ROLE",
    "ModelRoleConfiguration",
    "ModelRoleConfigurations",
    "ModelRoleId",
    "RetrievalBindingConfiguration",
    "RetrievalConfiguration",
    "T10_CELL_IDS",
    "T10_RETRIEVAL_BINDING_IDS",
    "ThinkingEffort",
    "expected_cell_ids_for_selection",
    "load_benchmark_configuration",
]
