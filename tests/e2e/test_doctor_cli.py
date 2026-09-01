from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from oamb.cli import app
from oamb.config.doctor import ResolvedPlanError, load_resolved_plan_for_run
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "benchmark.yml"
EXPECTED_CELL_IDS = ("hindsight-lme6", "mem0-lme6", "openviking-lme6")
EXPECTED_ROLE_IDS = (
    "hindsight_extraction",
    "mem0_extraction",
    "openviking_semantic_understanding",
    "answer",
    "judge",
    "embedding",
)


def _invoke_doctor(*, config: Path, output: Path, extra: tuple[str, ...] = ()) -> Result:
    return CliRunner().invoke(
        app,
        ["doctor", str(config), "--output", str(output), *extra],
    )


def _write_mutated_config(tmp_path: Path, original: str, replacement: str) -> Path:
    path = tmp_path / f"benchmark-{hashlib.sha256(replacement.encode()).hexdigest()[:8]}.yml"
    path.write_text(
        BENCHMARK_CONFIG_PATH.read_text(encoding="utf-8").replace(original, replacement, 1),
        encoding="utf-8",
    )
    return path


def _rehash_plan(document: dict[str, object]) -> bytes:
    payload = dict(document)
    payload.pop("resolved_plan_hash", None)
    document["resolved_plan_hash"] = canonical_sha256(["oamb-resolved-plan-initial-v1", payload])
    return canonical_json_bytes(document)


def test_doctor_public_interface_has_no_free_provider_dataset_or_workload_selection() -> None:
    result = CliRunner().invoke(app, ["doctor", "--help"])

    assert result.exit_code == 0, result.output
    assert "CONFIG" in result.output
    assert "--output" in result.output
    assert "--provider" not in result.output
    assert "--dataset" not in result.output
    assert "--workload" not in result.output
    lowered = result.output.lower()
    assert "approval" not in lowered
    assert "signature" not in lowered
    assert "phase" not in lowered
    assert "t10" not in lowered


def test_doctor_rejects_legacy_free_selection_flags_without_output(tmp_path: Path) -> None:
    output = tmp_path / "comparison"

    result = _invoke_doctor(
        config=BENCHMARK_CONFIG_PATH,
        output=output,
        extra=(
            "--provider",
            "arbitrary-provider",
            "--dataset",
            "ai-hyz/MemoryAgentBench",
            "--workload",
            "arbitrary-workload",
        ),
    )

    assert result.exit_code != 0
    assert not (output / "resolved-plan.json").exists()


def test_doctor_writes_one_canonical_plan_with_three_ordered_cell_specs(tmp_path: Path) -> None:
    output = tmp_path / "comparison"

    result = _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=output)

    assert result.exit_code == 0, result.output
    plan_path = output / "resolved-plan.json"
    content = plan_path.read_bytes()
    document = json.loads(content)
    assert content == json.dumps(
        document,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert document["schema_name"] == "resolved_plan"
    assert document["schema_version"] == 1
    assert document["comparison_id"] == "t10-lme6"
    assert document["dataset"]["workload_id"] == "lme30-native-smoke-plus-v1"
    assert document["dataset"]["selection"] == "lme6"
    assert tuple(cell["cell_id"] for cell in document["cells"]) == EXPECTED_CELL_IDS
    assert tuple(cell["ordinal_1_indexed"] for cell in document["cells"]) == (1, 2, 3)
    assert all(len(cell["cell_spec_hash"]) == 64 for cell in document["cells"])
    assert len(document["resolved_plan_hash"]) == 64
    assert f"resolved plan: {plan_path}" in result.output
    assert f"resolved plan sha256: {hashlib.sha256(content).hexdigest()}" in result.output


def test_doctor_plan_closes_models_retrieval_recipients_and_limits(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    result = _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=output)
    assert result.exit_code == 0, result.output

    document = json.loads((output / "resolved-plan.json").read_bytes())
    assert tuple(item["role_id"] for item in document["model_roles"]) == EXPECTED_ROLE_IDS
    assert tuple(
        (
            item["configured_model"],
            item["thinking_effort"],
            item["thinking_effort_rank_1_indexed"],
        )
        for item in document["model_roles"]
    ) == (
        ("deepseek-v4-flash", "low", 1),
        ("deepseek-v4-flash", "low", 1),
        ("deepseek-v4-flash", "low", 1),
        ("deepseek-v4-pro", "low", 1),
        ("deepseek-v4-flash", "high", 2),
        ("qwen3-embedding:0.6b", "not_applicable", None),
    )
    assert document["retrieval"]["generation"] == "disabled"
    assert tuple(
        (item["binding_id"], item["route"], item["request_constraint"])
        for item in document["retrieval"]["bindings"]
    ) == (
        ("hindsight", "recall", "omit_reflect"),
        ("mem0", "/search", "rerank_field_absent_by_schema"),
        ("openviking", "/api/v1/search/find", "no_session_id_on_find"),
    )
    assert "rerank=false" not in (output / "resolved-plan.json").read_text(encoding="utf-8")
    assert document["limits"] == {
        "currency": "CNY",
        "max_attempts_per_case": 2,
        "max_cost": "100.00",
        "max_input_tokens": 5_000_000,
        "max_output_tokens": 1_200_000,
        "max_peak_memory_bytes": 8_589_934_592,
        "max_budgeted_attempts": 1000,
        "max_recall_context_tokens_per_case": 32_768,
        "max_storage_bytes": 10_737_418_240,
        "memory_operation_timeout_seconds": 900,
        "model_call_timeout_seconds": 600,
        "total_wall_time_seconds": 310_500,
    }
    assert all(cell["recipient"] for cell in document["cells"])
    assert all(role["recipient"] for role in document["model_roles"])


def test_doctor_prints_redacted_human_summary_only(tmp_path: Path) -> None:
    output = tmp_path / "comparison"

    result = _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=output)

    assert result.exit_code == 0, result.output
    assert "comparison: t10-lme6" in result.output
    assert "cells: 3" in result.output
    assert "retrieval generation: disabled" in result.output
    assert "deepseek-v4-pro / low (rank 1/3)" in result.output
    assert "deepseek-v4-flash / high (rank 2/3)" in result.output
    assert "recipient=deepseek-api" in result.output
    assert "credential values: [REDACTED]" in result.output
    assert "api_key" not in result.output.lower()
    assert "authorization:" not in result.output.lower()


def test_loaded_plan_is_frozen_and_does_not_reopen_mutated_yaml(tmp_path: Path) -> None:
    config = tmp_path / "benchmark.yml"
    config.write_bytes(BENCHMARK_CONFIG_PATH.read_bytes())
    output = tmp_path / "comparison"
    result = _invoke_doctor(config=config, output=output)
    assert result.exit_code == 0, result.output
    plan_path = output / "resolved-plan.json"

    config.write_text(
        config.read_text(encoding="utf-8").replace("deepseek-v4-pro", "changed-model", 1),
        encoding="utf-8",
    )
    plan = load_resolved_plan_for_run(plan_path)

    assert plan.comparison_id == "t10-lme6"
    assert plan.model_roles[0].configured_model == "deepseek-v4-flash"
    with pytest.raises(FrozenInstanceError):
        plan.comparison_id = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        plan.cells[0].provider_id = "changed"  # type: ignore[misc]


def test_plan_loader_rejects_noncanonical_plan_hash_and_cell_mutation(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    result = _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=output)
    assert result.exit_code == 0, result.output
    original_path = output / "resolved-plan.json"

    noncanonical = tmp_path / "noncanonical.json"
    noncanonical.write_bytes(original_path.read_bytes() + b"\n")
    with pytest.raises(ResolvedPlanError, match="canonical"):
        load_resolved_plan_for_run(noncanonical)

    hash_drifted_document = json.loads(original_path.read_bytes())
    hash_drifted_document["comparison_id"] = "changed"
    hash_drifted = tmp_path / "hash-drifted.json"
    hash_drifted.write_bytes(canonical_json_bytes(hash_drifted_document))
    with pytest.raises(ResolvedPlanError, match="hash"):
        load_resolved_plan_for_run(hash_drifted)

    cell_drifted_document = json.loads(original_path.read_bytes())
    cell_drifted_document["cells"][0]["recipient"] = "changed-recipient"
    cell_drifted = tmp_path / "cell-drifted.json"
    cell_drifted.write_bytes(_rehash_plan(cell_drifted_document))
    with pytest.raises(ResolvedPlanError, match="cell spec hash"):
        load_resolved_plan_for_run(cell_drifted)


def test_plan_loader_rejects_cross_cell_substitution_with_rehashed_plan(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    result = _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=output)
    assert result.exit_code == 0, result.output
    document = json.loads((output / "resolved-plan.json").read_bytes())
    document["cells"][0], document["cells"][1] = document["cells"][1], document["cells"][0]
    substituted = tmp_path / "substituted.json"
    substituted.write_bytes(_rehash_plan(document))

    with pytest.raises(ResolvedPlanError, match="cell order"):
        load_resolved_plan_for_run(substituted)


@pytest.mark.parametrize(
    ("role_mutation", "error"),
    (
        ({"runtime_model": "deepseek-v4-pro"}, "provider-internal model identity"),
        ({"execution_owner": "harness"}, "execution owner"),
    ),
)
def test_plan_loader_rejects_rehashed_provider_internal_identity_drift(
    tmp_path: Path,
    role_mutation: dict[str, str],
    error: str,
) -> None:
    output = tmp_path / "comparison"
    result = _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=output)
    assert result.exit_code == 0, result.output
    document = json.loads((output / "resolved-plan.json").read_bytes())
    role = document["model_roles"][0]
    role.update(role_mutation)
    role_payload = dict(role)
    role_payload.pop("binding_hash")
    role["binding_hash"] = canonical_sha256(
        ["oamb-model-execution-binding-initial-v1", role_payload]
    )
    for cell in document["cells"]:
        for binding in cell["model_role_binding_hashes"]:
            if binding[0] == role["role_id"]:
                binding[1] = role["binding_hash"]
        cell_payload = dict(cell)
        cell_payload.pop("cell_spec_hash")
        cell["cell_spec_hash"] = canonical_sha256(["oamb-cell-spec-initial-v1", cell_payload])
    drifted = tmp_path / "provider-runtime-drifted.json"
    drifted.write_bytes(_rehash_plan(document))

    with pytest.raises(ResolvedPlanError, match=error):
        load_resolved_plan_for_run(drifted)


def test_model_endpoint_and_concurrency_changes_change_cell_identity(tmp_path: Path) -> None:
    first_output = tmp_path / "first"
    assert _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=first_output).exit_code == 0
    original = json.loads((first_output / "resolved-plan.json").read_bytes())

    model_config = _write_mutated_config(
        tmp_path,
        "endpoint_variable: OAMB_DEEPSEEK_BASE_URL",
        "endpoint_variable: OAMB_ALTERNATE_BASE_URL",
    )
    model_output = tmp_path / "model"
    assert _invoke_doctor(config=model_config, output=model_output).exit_code == 0
    model_changed = json.loads((model_output / "resolved-plan.json").read_bytes())

    concurrency_config = _write_mutated_config(
        tmp_path,
        "max_parallel_questions_per_provider: 2",
        "max_parallel_questions_per_provider: 1",
    )
    concurrency_output = tmp_path / "concurrency"
    assert _invoke_doctor(config=concurrency_config, output=concurrency_output).exit_code == 0
    concurrency_changed = json.loads((concurrency_output / "resolved-plan.json").read_bytes())

    assert tuple(cell["cell_spec_hash"] for cell in model_changed["cells"]) != tuple(
        cell["cell_spec_hash"] for cell in original["cells"]
    )
    assert tuple(cell["cell_spec_hash"] for cell in concurrency_changed["cells"]) != tuple(
        cell["cell_spec_hash"] for cell in original["cells"]
    )


def test_doctor_rejects_mab_or_invalid_retrieval_without_output(tmp_path: Path) -> None:
    for original, replacement in (
        (
            "workload_id: lme30-native-smoke-plus-v1",
            "workload_id: memoryagentbench",
        ),
        ("generation: disabled", "generation: enabled"),
    ):
        config = _write_mutated_config(tmp_path, original, replacement)
        output = tmp_path / hashlib.sha256(replacement.encode()).hexdigest()[:8]

        result = _invoke_doctor(config=config, output=output)

        assert result.exit_code != 0
        assert not (output / "resolved-plan.json").exists()


def test_doctor_output_is_create_only(tmp_path: Path) -> None:
    output = tmp_path / "comparison"
    first = _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=output)
    assert first.exit_code == 0, first.output
    plan_path = output / "resolved-plan.json"
    original = plan_path.read_bytes()

    second = _invoke_doctor(config=BENCHMARK_CONFIG_PATH, output=output)

    assert second.exit_code != 0
    assert plan_path.read_bytes() == original
