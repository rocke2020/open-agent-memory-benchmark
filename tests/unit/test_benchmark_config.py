from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from oamb.config.benchmark import (
    BenchmarkConfiguration,
    BenchmarkConfigurationError,
    load_benchmark_configuration,
    load_resume_concurrency,
)
from tests.benchmark_configuration import MODEL_ENVIRONMENT

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "benchmark.yml"

EXPECTED_ROLE_IDS = (
    "hindsight_extraction",
    "mem0_extraction",
    "openviking_semantic_understanding",
    "answer",
    "judge",
    "embedding",
)
EXPECTED_CELL_IDS = ("hindsight-lme60", "mem0-lme60", "openviking-lme60")
EXPECTED_DEEPSEEK_EFFORT_SCALE = ("low", "high", "max")


def _valid_configuration_yaml() -> str:
    return BENCHMARK_CONFIG_PATH.read_text(encoding="utf-8")


def _write_configuration(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "benchmark.yml"
    path.write_text(content, encoding="utf-8")
    return path


def _load(path: Path) -> BenchmarkConfiguration:
    return load_benchmark_configuration(path, model_environment=MODEL_ENVIRONMENT)


@pytest.mark.parametrize("value", ("true", "0", "-1", "1.5", "'3'", "null"))
def test_resume_concurrency_rejects_non_positive_integer_caps(tmp_path: Path, value: str) -> None:
    path = _write_configuration(
        tmp_path,
        "execution:\n"
        f"  max_parallel_history_ingestions_per_provider: {value}\n"
        "  max_parallel_questions_per_provider: 3\n",
    )
    with pytest.raises(BenchmarkConfigurationError, match="positive finite integer"):
        load_resume_concurrency(path)


@pytest.mark.parametrize("content", ("{}", "execution: []", "execution: {}"))
def test_resume_concurrency_requires_explicit_execution_caps(tmp_path: Path, content: str) -> None:
    with pytest.raises(BenchmarkConfigurationError):
        load_resume_concurrency(_write_configuration(tmp_path, content))


def test_checked_in_configuration_selects_one_lme60_three_provider_comparison() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    assert configuration.comparison_id == "v0.1-lme60"
    assert (
        configuration.dataset.dataset_id,
        configuration.dataset.workload_id,
        configuration.dataset.selection,
        configuration.dataset.path,
        configuration.dataset.revision,
        configuration.dataset.source_sha256,
        configuration.dataset.case_manifest_hash,
    ) == (
        "longmemeval-s-cleaned",
        "lme60-balanced-v1",
        "lme60",
        "datasets/longmemeval-cleaned/longmemeval_s_cleaned.json",
        "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "20afb9ae0d18a5821871e76a1b90df573fd5f155511c264249fdb49154088962",
    )
    assert tuple(cell.cell_id for cell in configuration.cells) == EXPECTED_CELL_IDS
    assert tuple(
        (cell.provider_id, cell.adapter_profile_id, cell.producer_role)
        for cell in configuration.cells
    ) == (
        ("hindsight", "hindsight-rest-v1", "hindsight_extraction"),
        ("mem0", "mem0-rest-v1", "mem0_extraction"),
        (
            "openviking",
            "openviking-session-rest-v1",
            "openviking_semantic_understanding",
        ),
    )
    assert all(
        (cell.embedding_role, cell.answer_role, cell.judge_role) == ("embedding", "answer", "judge")
        for cell in configuration.cells
    )
    assert configuration.decision is not None
    assert configuration.decision.minimum_accuracy_delta == "0.05"
    assert configuration.decision.maximum_exact_mcnemar_p_value == "0.05"


def test_checked_in_generative_models_resolve_from_model_environment() -> None:
    configuration = load_benchmark_configuration(
        BENCHMARK_CONFIG_PATH,
        model_environment={
            "LLM_LIGHT_MODEL": "fixture-light-model",
            "LLM_DEEP_MODEL": "fixture-deep-model",
        },
    )

    assert configuration.llm_profiles.light_model == "fixture-light-model"
    assert configuration.llm_profiles.deep_model == "fixture-deep-model"
    assert {role_id: role.model for role_id, role in configuration.models.ordered_items()} == {
        "hindsight_extraction": "fixture-light-model",
        "mem0_extraction": "fixture-light-model",
        "openviking_semantic_understanding": "fixture-light-model",
        "answer": "fixture-deep-model",
        "judge": "fixture-light-model",
        "embedding": "qwen3-embedding:0.6b",
    }


def test_checked_in_configuration_closes_six_model_roles_and_recipients() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)
    light_model = configuration.llm_profiles.light_model
    deep_model = configuration.llm_profiles.deep_model

    assert configuration.models.ordered_role_ids == EXPECTED_ROLE_IDS
    assert tuple(
        (
            role_id,
            binding.model,
            binding.thinking_effort,
            binding.thinking_effort_scale,
            binding.thinking_effort_rank_1_indexed,
            binding.endpoint_variable,
            binding.credential_variable,
            binding.recipient,
            binding.execution_owner,
            binding.proof_kind,
            binding.proof_reference,
            binding.usage_coverage,
        )
        for role_id, binding in configuration.models.ordered_items()
    ) == (
        (
            "hindsight_extraction",
            light_model,
            "low",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            1,
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "llm-api",
            "provider_internal",
            "effective_config_and_outbound_request",
            "reasoning_effort=low",
            "classified",
        ),
        (
            "mem0_extraction",
            light_model,
            "low",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            1,
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "llm-api",
            "provider_internal",
            "effective_config_and_outbound_request",
            "reasoning_effort=low",
            "classified",
        ),
        (
            "openviking_semantic_understanding",
            light_model,
            "low",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            1,
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "llm-api",
            "provider_internal",
            "effective_config_and_outbound_request",
            "reasoning_effort=low",
            "classified",
        ),
        (
            "answer",
            deep_model,
            "low",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            1,
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "llm-api",
            "harness",
            "sealed_outbound_request_and_response",
            "reasoning_effort=low",
            "classified",
        ),
        (
            "judge",
            light_model,
            "high",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            2,
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "llm-api",
            "harness",
            "sealed_outbound_request_and_response",
            "reasoning_effort=high",
            "classified",
        ),
        (
            "embedding",
            configuration.models.embedding.model,
            "not_applicable",
            (),
            None,
            "OAMB_EMBEDDING_BASE_URL",
            "OAMB_EMBEDDING_API_KEY",
            "embedding-api",
            "provider_internal",
            "model_dimension_probe",
            "not_applicable",
            "classified",
        ),
    )


def test_loader_rejects_a_role_model_outside_llm_profiles(tmp_path: Path) -> None:
    content = _valid_configuration_yaml().replace(
        "  mem0_extraction:\n    model: *light_model\n",
        "  mem0_extraction:\n    model: configured-by-benchmark\n",
        1,
    )

    with pytest.raises(BenchmarkConfigurationError, match="model role mem0_extraction model"):
        _load(_write_configuration(tmp_path, content))


def test_loader_uses_benchmark_yaml_as_the_thinking_effort_source(tmp_path: Path) -> None:
    content = _valid_configuration_yaml().replace(
        "  mem0_extraction:\n    model: *light_model\n    thinking_effort: low\n"
        "    endpoint_variable: LLM_BASE_URL\n"
        "    credential_variable: LLM_API_KEY\n"
        "    recipient: llm-api\n    execution_owner: provider_internal\n"
        "    proof_kind: effective_config_and_outbound_request\n"
        "    proof_reference: reasoning_effort=low\n",
        "  mem0_extraction:\n    model: *light_model\n    thinking_effort: high\n"
        "    endpoint_variable: LLM_BASE_URL\n"
        "    credential_variable: LLM_API_KEY\n"
        "    recipient: llm-api\n    execution_owner: provider_internal\n"
        "    proof_kind: effective_config_and_outbound_request\n"
        "    proof_reference: reasoning_effort=high\n",
        1,
    )

    configuration = _load(_write_configuration(tmp_path, content))

    assert configuration.models.mem0_extraction.thinking_effort == "high"


def test_checked_in_configuration_freezes_generation_free_retrieval_bindings() -> None:
    retrieval = _load(BENCHMARK_CONFIG_PATH).retrieval

    assert retrieval.generation == "disabled"
    assert retrieval.top_k == 150
    assert tuple(
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
        for item in retrieval.bindings
    ) == (
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
    assert "rerank=false" not in _valid_configuration_yaml()


def test_checked_in_configuration_freezes_all_layered_retry_controls() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    assert configuration.evaluation_controls.as_tuple() == (2, 10, 6, 2, 900)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda text: text.replace("comparison: v0.1-lme60\n", "", 1),
        lambda text: text.replace(
            "comparison: v0.1-lme60\n", "comparison: v0.1-lme60\nextra: x\n", 1
        ),
        lambda text: text.replace(
            "  judge:\n",
            "  unknown_role:\n    model: x\n  judge:\n",
            1,
        ),
        lambda text: text.replace("  embedding:\n", "", 1),
        lambda text: text.replace(
            "    recipient: local-hindsight-service\n",
            "    recipient: local-hindsight-service\n    api_key: forbidden\n",
            1,
        ),
    ),
)
def test_loader_rejects_missing_or_unknown_keys(
    tmp_path: Path,
    mutator: Callable[[str], str],
) -> None:
    with pytest.raises(BenchmarkConfigurationError):
        _load(_write_configuration(tmp_path, mutator(_valid_configuration_yaml())))


@pytest.mark.parametrize(
    "mutator",
    (
        lambda text: text + "comparison: duplicate\n",
        lambda text: text.replace(
            "    model: *deep_model\n",
            "    model: *deep_model\n    model: duplicate\n",
            1,
        ),
        lambda text: text.replace(
            "  operation_timeout_seconds: 900\n",
            "  operation_timeout_seconds: 900\n  operation_timeout_seconds: 901\n",
            1,
        ),
    ),
)
def test_loader_rejects_duplicate_yaml_keys(
    tmp_path: Path,
    mutator: Callable[[str], str],
) -> None:
    with pytest.raises(BenchmarkConfigurationError, match="duplicate"):
        _load(_write_configuration(tmp_path, mutator(_valid_configuration_yaml())))


@pytest.mark.parametrize(
    "original,replacement",
    (
        (
            "  workload_id: lme60-balanced-v1",
            "  workload_id: memoryagentbench",
        ),
        ("  dataset_id: longmemeval-s-cleaned", "  dataset_id: memoryagentbench-mab5"),
        ("    provider_id: hindsight", "    provider_id: unknown"),
        ("    adapter_profile_id: hindsight-rest-v1", "    adapter_profile_id: unknown-v1"),
        ("    producer_role: hindsight_extraction", "    producer_role: mem0_extraction"),
    ),
)
def test_loader_rejects_mab_unknown_profiles_and_wrong_role_routes(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    content = _valid_configuration_yaml().replace(original, replacement, 1)
    with pytest.raises(BenchmarkConfigurationError):
        _load(_write_configuration(tmp_path, content))


@pytest.mark.parametrize(
    "original,replacement",
    (
        ("generation: disabled", "generation: enabled"),
        ("route: recall", "route: reflect"),
        ("route: /search", "route: /search-with-rerank"),
        ("request_constraint: rerank_field_absent_by_schema", "request_constraint: rerank=false"),
        ("route: /api/v1/search/find", "route: /api/v1/search"),
        ("request_constraint: no_session_id_on_find", "request_constraint: allow_session_id"),
    ),
)
def test_loader_rejects_non_exact_retrieval_configuration(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    content = _valid_configuration_yaml().replace(original, replacement, 1)
    with pytest.raises(BenchmarkConfigurationError):
        _load(_write_configuration(tmp_path, content))


@pytest.mark.parametrize(
    "original,replacement",
    (
        ("thinking_effort: low", "thinking_effort: none"),
        ("thinking_effort: low", "thinking_effort: not_applicable"),
        ("thinking_effort: not_applicable", "thinking_effort: low"),
        ("credential_variable: LLM_API_KEY", "credential_variable: literal-secret"),
        ("endpoint_variable: LLM_BASE_URL", "endpoint_variable: https://secret"),
        ("recipient: llm-api", "recipient:"),
    ),
)
def test_loader_rejects_invalid_model_effort_or_recipient_closure(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    content = _valid_configuration_yaml().replace(original, replacement, 1)
    with pytest.raises(BenchmarkConfigurationError):
        _load(_write_configuration(tmp_path, content))


@pytest.mark.parametrize(
    "original,replacement",
    (
        ("max_retries_per_operation: 2", "max_retries_per_operation: -1"),
        ("max_retries_per_operation: 2", "max_retries_per_operation: null"),
        ("operation_timeout_seconds: 900", "operation_timeout_seconds: 0"),
        ("operation_timeout_seconds: 900", "operation_timeout_seconds: .inf"),
    ),
)
def test_loader_rejects_invalid_evaluation_controls(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    content = _valid_configuration_yaml().replace(original, replacement, 1)
    with pytest.raises(BenchmarkConfigurationError):
        _load(_write_configuration(tmp_path, content))


@pytest.mark.parametrize("retry_count", (0, 1))
def test_loader_rejects_non_profile_batch_retry_count(tmp_path: Path, retry_count: int) -> None:
    content = _valid_configuration_yaml().replace(
        "max_retries_per_operation: 2",
        f"max_retries_per_operation: {retry_count}",
        1,
    )

    with pytest.raises(BenchmarkConfigurationError, match="batch retry"):
        _load(_write_configuration(tmp_path, content))


@pytest.mark.parametrize(
    ("original", "replacement"),
    (
        ('minimum_accuracy_delta: "0.05"', 'minimum_accuracy_delta: "0.00"'),
        (
            'maximum_exact_mcnemar_p_value: "0.05"',
            'maximum_exact_mcnemar_p_value: "1.01"',
        ),
        ("  maximum_exact_mcnemar_p_value:", "  unknown_decision_key:"),
    ),
)
def test_loader_rejects_invalid_or_unknown_lme60_decision_policy(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    content = _valid_configuration_yaml().replace(original, replacement, 1)

    with pytest.raises(BenchmarkConfigurationError, match="decision"):
        _load(_write_configuration(tmp_path, content))
