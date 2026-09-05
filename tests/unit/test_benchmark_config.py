from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from oamb.config.benchmark import (
    BenchmarkConfiguration,
    BenchmarkConfigurationError,
    load_benchmark_configuration,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "benchmark.yml"
LME6_CONFIG_PATH = REPOSITORY_ROOT / "tests/fixtures/configs/t10-lme6.yml"

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
    return load_benchmark_configuration(path)


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
        "90b2669f7b893e59d404549f5803882bcd6640ce82520a9bf09672cc79464c80",
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


def test_preserved_lme6_configuration_keeps_original_selection_and_concurrency() -> None:
    configuration = _load(LME6_CONFIG_PATH)

    assert configuration.comparison_id == "t10-lme6"
    assert configuration.dataset.selection == "lme6"
    assert configuration.dataset.workload_id == "lme30-native-smoke-plus-v1"
    assert tuple(cell.cell_id for cell in configuration.cells) == (
        "hindsight-lme6",
        "mem0-lme6",
        "openviking-lme6",
    )
    assert configuration.evaluation_controls.as_tuple() == (2, 900)
    assert configuration.decision is None


def test_checked_in_configuration_closes_six_model_roles_and_recipients() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    assert configuration.models.ordered_role_ids == EXPECTED_ROLE_IDS
    assert tuple(
        (
            role_id,
            binding.model,
            binding.runtime_model,
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
            "deepseek-v4-flash",
            "deepseek-v4-flash",
            "low",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            1,
            "OAMB_HINDSIGHT_LLM_BASE_URL",
            "OAMB_HINDSIGHT_LLM_API_KEY",
            "deepseek-api",
            "provider_internal",
            "effective_config_and_outbound_request",
            "reasoning_effort=low",
            "classified",
        ),
        (
            "mem0_extraction",
            "deepseek-v4-flash",
            "deepseek-v4-flash",
            "low",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            1,
            "OAMB_MEM0_LLM_BASE_URL",
            "OAMB_MEM0_LLM_API_KEY",
            "deepseek-api",
            "provider_internal",
            "effective_config_and_outbound_request",
            "reasoning_effort=low",
            "classified",
        ),
        (
            "openviking_semantic_understanding",
            "deepseek-v4-flash",
            "deepseek-v4-flash",
            "low",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            1,
            "OAMB_OPENVIKING_VLM_BASE_URL",
            "OAMB_OPENVIKING_VLM_API_KEY",
            "deepseek-api",
            "provider_internal",
            "effective_config_and_outbound_request",
            "reasoning_effort=low",
            "classified",
        ),
        (
            "answer",
            "deepseek-v4-pro",
            "deepseek-v4-pro",
            "low",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            1,
            "OAMB_DEEPSEEK_BASE_URL",
            "OAMB_DEEPSEEK_API_KEY",
            "deepseek-api",
            "harness",
            "sealed_outbound_request_and_response",
            "reasoning_effort=low",
            "classified",
        ),
        (
            "judge",
            "deepseek-v4-flash",
            "deepseek-v4-flash",
            "high",
            EXPECTED_DEEPSEEK_EFFORT_SCALE,
            2,
            "OAMB_DEEPSEEK_BASE_URL",
            "OAMB_DEEPSEEK_API_KEY",
            "deepseek-api",
            "harness",
            "sealed_outbound_request_and_response",
            "reasoning_effort=high",
            "classified",
        ),
        (
            "embedding",
            "qwen3-embedding:0.6b",
            "qwen3-embedding:0.6b",
            "not_applicable",
            (),
            None,
            "OAMB_EMBEDDING_BASE_URL",
            "not_applicable",
            "local-vllm-metal",
            "provider_internal",
            "model_dimension_probe",
            "not_applicable",
            "classified",
        ),
    )


def test_loader_uses_benchmark_yaml_as_the_model_source(tmp_path: Path) -> None:
    content = _valid_configuration_yaml().replace(
        "  mem0_extraction:\n    model: deepseek-v4-flash\n    runtime_model: deepseek-v4-flash\n",
        "  mem0_extraction:\n"
        "    model: configured-by-benchmark\n"
        "    runtime_model: configured-by-benchmark\n",
        1,
    )

    configuration = _load(_write_configuration(tmp_path, content))

    assert configuration.models.mem0_extraction.model == "configured-by-benchmark"
    assert configuration.models.mem0_extraction.runtime_model == "configured-by-benchmark"


def test_loader_uses_benchmark_yaml_as_the_thinking_effort_source(tmp_path: Path) -> None:
    content = _valid_configuration_yaml().replace(
        "  mem0_extraction:\n    model: deepseek-v4-flash\n"
        "    runtime_model: deepseek-v4-flash\n    thinking_effort: low\n"
        "    endpoint_variable: OAMB_MEM0_LLM_BASE_URL\n"
        "    credential_variable: OAMB_MEM0_LLM_API_KEY\n"
        "    recipient: deepseek-api\n    execution_owner: provider_internal\n"
        "    proof_kind: effective_config_and_outbound_request\n"
        "    proof_reference: reasoning_effort=low\n",
        "  mem0_extraction:\n    model: deepseek-v4-flash\n"
        "    runtime_model: deepseek-v4-flash\n    thinking_effort: high\n"
        "    endpoint_variable: OAMB_MEM0_LLM_BASE_URL\n"
        "    credential_variable: OAMB_MEM0_LLM_API_KEY\n"
        "    recipient: deepseek-api\n    execution_owner: provider_internal\n"
        "    proof_kind: effective_config_and_outbound_request\n"
        "    proof_reference: reasoning_effort=high\n",
        1,
    )

    configuration = _load(_write_configuration(tmp_path, content))

    assert configuration.models.mem0_extraction.thinking_effort == "high"


def test_checked_in_configuration_freezes_generation_free_retrieval_bindings() -> None:
    retrieval = _load(BENCHMARK_CONFIG_PATH).retrieval

    assert retrieval.generation == "disabled"
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


def test_checked_in_configuration_has_only_two_finite_evaluation_controls() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    assert configuration.evaluation_controls.as_tuple() == (2, 900)


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
            "    model: deepseek-v4-pro\n",
            "    model: deepseek-v4-pro\n    model: duplicate\n",
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
        ("credential_variable: OAMB_DEEPSEEK_API_KEY", "credential_variable: literal-secret"),
        ("endpoint_variable: OAMB_DEEPSEEK_BASE_URL", "endpoint_variable: https://secret"),
        ("recipient: deepseek-api", "recipient:"),
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


def test_loader_rejects_unproved_provider_runtime_model_drift(tmp_path: Path) -> None:
    content = _valid_configuration_yaml().replace(
        "runtime_model: deepseek-v4-flash",
        "runtime_model: deepseek-v4-pro",
        1,
    )

    with pytest.raises(BenchmarkConfigurationError, match="configured and runtime model"):
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


@pytest.mark.parametrize(
    "legacy_line",
    (
        "  max_attempts_per_case: 2\n",
        "  max_budgeted_attempts: 10000\n",
        "  total_wall_time_seconds: 2940300\n",
        "  max_input_tokens: 70000000\n",
        "  max_output_tokens: 47000000\n",
        "  max_recall_context_tokens_per_case: 32768\n",
        "  max_storage_bytes: 10737418240\n",
        "  max_peak_memory_bytes: 8589934592\n",
        '  max_cost: "100.00"\n',
        "  currency: CNY\n",
        "  max_parallel_datasets: 1\n",
        "  max_parallel_providers_per_dataset: 3\n",
        "  max_parallel_history_ingestions_per_provider: 2\n",
        "  max_parallel_questions_per_provider: 2\n",
    ),
)
def test_loader_rejects_each_removed_ceiling_or_concurrency_key(
    tmp_path: Path,
    legacy_line: str,
) -> None:
    content = _valid_configuration_yaml().replace(
        "  operation_timeout_seconds: 900\n",
        f"  operation_timeout_seconds: 900\n{legacy_line}",
        1,
    )

    with pytest.raises(BenchmarkConfigurationError, match="execution keys"):
        _load(_write_configuration(tmp_path, content))


def test_loader_accepts_zero_retries_as_one_initial_attempt(tmp_path: Path) -> None:
    content = _valid_configuration_yaml().replace(
        "max_retries_per_operation: 2",
        "max_retries_per_operation: 0",
        1,
    )

    configuration = _load(_write_configuration(tmp_path, content))

    assert configuration.evaluation_controls.as_tuple() == (0, 900)


def test_configuration_is_frozen() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    with pytest.raises(AttributeError):
        configuration.cells = ()  # type: ignore[misc]
    assert replace(configuration.evaluation_controls).as_tuple() == (2, 900)


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
