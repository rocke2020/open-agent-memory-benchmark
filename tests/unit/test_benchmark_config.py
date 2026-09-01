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

EXPECTED_ROLE_IDS = (
    "hindsight_extraction",
    "mem0_extraction",
    "openviking_semantic_understanding",
    "answer",
    "judge",
    "embedding",
)
EXPECTED_CELL_IDS = ("hindsight-lme6", "mem0-lme6", "openviking-lme6")
EXPECTED_DEEPSEEK_EFFORT_SCALE = ("low", "high", "max")


def _valid_configuration_yaml() -> str:
    return BENCHMARK_CONFIG_PATH.read_text(encoding="utf-8")


def _write_configuration(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "benchmark.yml"
    path.write_text(content, encoding="utf-8")
    return path


def _load(path: Path) -> BenchmarkConfiguration:
    return load_benchmark_configuration(path)


def test_checked_in_configuration_selects_one_lme6_three_provider_comparison() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    assert configuration.comparison_id == "t10-lme6"
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
        "lme30-native-smoke-plus-v1",
        "lme6",
        "datasets/longmemeval-cleaned/longmemeval_s_cleaned.json",
        "98d7416c24c778c2fee6e6f3006e7a073259d48f",
        "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
        "3c0bc0e2e539b3f7ceca81569531c6a0fb5823ccc55426cb2b2cf9d295fa33e4",
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


@pytest.mark.parametrize(
    "role_id,target_model,drifted_model",
    (
        ("hindsight_extraction", "deepseek-v4-flash", "deepseek-v4-pro"),
        ("mem0_extraction", "deepseek-v4-flash", "deepseek-v4-pro"),
        ("openviking_semantic_understanding", "deepseek-v4-flash", "deepseek-v4-pro"),
        ("answer", "deepseek-v4-pro", "deepseek-v4-flash"),
        ("judge", "deepseek-v4-flash", "deepseek-v4-pro"),
    ),
)
def test_loader_rejects_drift_from_target_generative_model_profile(
    tmp_path: Path,
    role_id: str,
    target_model: str,
    drifted_model: str,
) -> None:
    target_binding = f"  {role_id}:\n    model: {target_model}\n    runtime_model: {target_model}\n"
    drifted_binding = (
        f"  {role_id}:\n    model: {drifted_model}\n    runtime_model: {drifted_model}\n"
    )
    content = _valid_configuration_yaml().replace(target_binding, drifted_binding, 1)

    with pytest.raises(BenchmarkConfigurationError, match="model profile"):
        _load(_write_configuration(tmp_path, content))


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


def test_checked_in_configuration_has_only_finite_positive_ceilings() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    assert configuration.limits.as_tuple() == (
        2,
        1000,
        900,
        600,
        310_500,
        5_000_000,
        1_200_000,
        32_768,
        10_737_418_240,
        8_589_934_592,
        "100.00",
        "CNY",
    )
    assert configuration.execution.as_tuple() == (1, 3, 2, 2)


@pytest.mark.parametrize(
    "mutator",
    (
        lambda text: text.replace("comparison: t10-lme6\n", "", 1),
        lambda text: text.replace("comparison: t10-lme6\n", "comparison: t10-lme6\nextra: x\n", 1),
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
            "  max_parallel_datasets: 1\n",
            "  max_parallel_datasets: 1\n  max_parallel_datasets: 2\n",
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
            "  workload_id: lme30-native-smoke-plus-v1",
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


@pytest.mark.parametrize(
    "original,replacement",
    (
        ("max_attempts_per_case: 2", "max_attempts_per_case: 0"),
        ("max_budgeted_attempts: 1000", "max_budgeted_attempts: null"),
        (
            "memory_operation_timeout_seconds: 900",
            "memory_operation_timeout_seconds: .inf",
        ),
        ("model_call_timeout_seconds: 600", "model_call_timeout_seconds: 0"),
        ("max_input_tokens: 5000000", "max_input_tokens: unbounded"),
        ('max_cost: "100.00"', "max_cost: null"),
        (
            "max_parallel_history_ingestions_per_provider: 2",
            "max_parallel_history_ingestions_per_provider: -1",
        ),
        (
            "max_parallel_questions_per_provider: 2",
            "max_parallel_questions_per_provider: 0",
        ),
    ),
)
def test_loader_rejects_missing_nonfinite_or_unbounded_ceilings(
    tmp_path: Path,
    original: str,
    replacement: str,
) -> None:
    content = _valid_configuration_yaml().replace(original, replacement, 1)
    with pytest.raises(BenchmarkConfigurationError):
        _load(_write_configuration(tmp_path, content))


@pytest.mark.parametrize(
    "legacy_key",
    (
        "max_parallel_ingestion_plans_per_provider",
        "max_parallel_cases_per_provider",
        "max_parallel_model_calls_per_role",
    ),
)
def test_loader_rejects_each_legacy_parallelism_key(
    tmp_path: Path,
    legacy_key: str,
) -> None:
    content = _valid_configuration_yaml().replace(
        "  max_parallel_questions_per_provider: 2\n",
        f"  max_parallel_questions_per_provider: 2\n  {legacy_key}: 2\n",
        1,
    )

    with pytest.raises(BenchmarkConfigurationError, match="execution keys"):
        _load(_write_configuration(tmp_path, content))


def test_configuration_is_frozen() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    with pytest.raises(AttributeError):
        configuration.cells = ()  # type: ignore[misc]
    assert replace(configuration.execution).as_tuple() == (1, 3, 2, 2)
