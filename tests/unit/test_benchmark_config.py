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
from tests.benchmark_configuration import load_lme6_configuration

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
    configuration = load_lme6_configuration()

    assert configuration.comparison_id == "t10-lme6"
    assert configuration.dataset.selection == "lme6"
    assert configuration.dataset.workload_id == "lme30-native-smoke-plus-v1"
    assert tuple(cell.cell_id for cell in configuration.cells) == (
        "hindsight-lme6",
        "mem0-lme6",
        "openviking-lme6",
    )
    assert configuration.evaluation_controls.as_tuple() == (2, 10, 6, 2, 900)
    assert configuration.decision is None


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
            "not_applicable",
            "local-vllm-metal",
            "provider_internal",
            "model_dimension_probe",
            "not_applicable",
            "classified",
        ),
    )


def test_light_model_alias_updates_every_linked_role(tmp_path: Path) -> None:
    source = _valid_configuration_yaml()
    light_model = _load(BENCHMARK_CONFIG_PATH).llm_profiles.light_model
    anchor = f"light_model: &light_model {light_model}"
    assert anchor in source
    configuration = _load(
        _write_configuration(
            tmp_path,
            source.replace(
                anchor,
                "light_model: &light_model alternate-light-model",
                1,
            ),
        )
    )

    assert (
        tuple(
            binding.model
            for role_id, binding in configuration.models.ordered_items()
            if role_id
            in {
                "hindsight_extraction",
                "mem0_extraction",
                "openviking_semantic_understanding",
                "judge",
            }
        )
        == ("alternate-light-model",) * 4
    )


def test_deep_model_alias_updates_every_linked_role(tmp_path: Path) -> None:
    source = _valid_configuration_yaml()
    deep_model = _load(BENCHMARK_CONFIG_PATH).llm_profiles.deep_model
    anchor = f"deep_model: &deep_model {deep_model}"
    assert anchor in source
    configuration = _load(
        _write_configuration(
            tmp_path,
            source.replace(
                anchor,
                "deep_model: &deep_model alternate-deep-model",
                1,
            ),
        )
    )

    answer = configuration.models.answer
    assert answer.model == "alternate-deep-model"


def test_loader_rejects_a_role_model_outside_llm_profiles(tmp_path: Path) -> None:
    content = _valid_configuration_yaml().replace(
        "  mem0_extraction:\n    model: *light_model\n",
        "  mem0_extraction:\n    model: configured-by-benchmark\n",
        1,
    )

    with pytest.raises(BenchmarkConfigurationError, match="light_model profile"):
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


@pytest.mark.parametrize("retry_count", (0, 1))
def test_loader_rejects_non_profile_batch_retry_count(tmp_path: Path, retry_count: int) -> None:
    content = _valid_configuration_yaml().replace(
        "max_retries_per_operation: 2",
        f"max_retries_per_operation: {retry_count}",
        1,
    )

    with pytest.raises(BenchmarkConfigurationError, match="batch retry"):
        _load(_write_configuration(tmp_path, content))


def test_configuration_is_frozen() -> None:
    configuration = _load(BENCHMARK_CONFIG_PATH)

    with pytest.raises(AttributeError):
        configuration.cells = ()  # type: ignore[misc]
    assert replace(configuration.evaluation_controls).as_tuple() == (2, 10, 6, 2, 900)


def test_loader_rejects_plan_without_layered_retry_fields(tmp_path: Path) -> None:
    content = _valid_configuration_yaml()
    for line in (
        "  extraction_max_retries: 10\n",
        "  model_max_attempts: 6\n",
        "  model_transport_max_retries: 2\n",
    ):
        content = content.replace(line, "", 1)
        with pytest.raises(BenchmarkConfigurationError, match="execution keys"):
            _load(_write_configuration(tmp_path, content))
        content = _valid_configuration_yaml()


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
