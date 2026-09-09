from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

import oamb.config.doctor as doctor_module
from oamb.config.benchmark import (
    BenchmarkConfiguration,
    EvaluationControls,
    load_benchmark_configuration,
)
from oamb.config.doctor import build_resolved_plan
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import ModelRole, SourceEvidenceBinding, SourceEvidenceKind
from oamb.live import build_live_cell

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "benchmark.yml"
NOW = datetime(2026, 9, 6, tzinfo=UTC)


def _configuration(*, retries: int) -> BenchmarkConfiguration:
    configuration = load_benchmark_configuration(BENCHMARK_CONFIG_PATH)
    return replace(
        configuration,
        evaluation_controls=replace(
            configuration.evaluation_controls,
            max_retries_per_operation=retries,
        ),
    )


def _provider_evidence() -> SourceEvidenceBinding:
    source_root_hash = canonical_sha256(["history-rebuild-provider-proof"])
    validation_result_hash = canonical_sha256(["history-rebuild-service-receipt"])
    return SourceEvidenceBinding(
        binding_id=canonical_sha256(
            ["history-rebuild-provider-evidence", source_root_hash, validation_result_hash]
        ),
        source_kind=SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity="oamb-providers-history-rebuild-test",
        source_root_hash=source_root_hash,
        validation_result_hash=validation_result_hash,
        source_schema_versions=("provider_service_evidence_manifest@1",),
    )


def _environment() -> dict[str, str]:
    models = load_benchmark_configuration(BENCHMARK_CONFIG_PATH).models
    return {
        "OAMB_HINDSIGHT_BASE_URL": "http://127.0.0.1:64888",
        "OAMB_HINDSIGHT_LLM_MODEL": models.hindsight_extraction.model,
        "OAMB_MEM0_BASE_URL": "http://127.0.0.1:64889",
        "OAMB_MEM0_ADMIN_API_KEY": "mem0-key",
        "OAMB_MEM0_LLM_MODEL": models.mem0_extraction.model,
        "OAMB_MEM0_INSPECTOR_BASE_URL": "http://127.0.0.1:64333",
        "OAMB_MEM0_INSPECTOR_API_KEY": "inspector-key",
        "OAMB_OPENVIKING_BASE_URL": "http://127.0.0.1:64930",
        "OAMB_OPENVIKING_USER_API_KEY": "openviking-key",
        "OAMB_OPENVIKING_VLM_MODEL": models.openviking_semantic_understanding.model,
        "OAMB_OPENVIKING_ACCOUNT_ID": "benchmark-account",
        "OAMB_OPENVIKING_ADMIN_USER_ID": "benchmark-user",
        "LLM_URL_TYPE": "openai_chat",
        "LLM_BASE_URL": "https://model.example/v1",
        "LLM_API_KEY": "model-key",
        "OAMB_EMBEDDING_BASE_URL": "http://127.0.0.1:18000/v1",
        "OAMB_EMBEDDING_API_KEY": "",
        "OAMB_EMBEDDING_MODEL": models.embedding.model,
    }


def test_lme60_resolved_execution_freezes_whole_history_budget() -> None:
    plan = build_resolved_plan(_configuration(retries=2))

    assert plan.execution.ingestion_retry_unit == "batch_dispatch"
    assert plan.execution.ingestion_recovery_strategy == "same_scope_batch_retry_skip_v1"
    assert plan.execution.extraction_max_retries == 10
    assert plan.execution.model_max_attempts == 6
    assert plan.execution.model_transport_max_retries == 2
    assert plan.execution.per_cell_max_operation_attempt_count == 10_999
    assert plan.execution.per_cell_max_owner_authorization_count == 28_015
    assert plan.execution.comparison_max_operation_attempt_count == 32_997
    assert plan.execution.comparison_max_owner_authorization_count == 84_045


@pytest.mark.parametrize(
    "controls",
    (
        EvaluationControls(1, 900, 10, 6, 2),
        EvaluationControls(2, 900, 9, 6, 2),
        EvaluationControls(2, 900, 10, 5, 2),
        EvaluationControls(2, 900, 10, 6, 1),
    ),
)
def test_resolved_execution_parser_rejects_self_consistent_non_profile_controls(
    controls: EvaluationControls,
) -> None:
    document = doctor_module._execution_document(
        doctor_module._resolve_execution("lme60", controls)
    )

    with pytest.raises(ValueError, match="layered retry"):
        doctor_module._parse_execution(document, selection="lme60")


def test_resolved_execution_uses_uneven_manifest_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Three groups contain 1, 4, and 2 sources: the production resolver consumes
    # their manifest-derived group and source totals.
    monkeypatch.setattr(doctor_module, "LME60_EXPECTED_QUESTION_IDS", ("q1", "q2", "q3"))
    monkeypatch.setattr(doctor_module, "LME60_EXPECTED_SESSION_COUNT", 7)

    execution = doctor_module._resolve_execution(
        "lme60",
        _configuration(retries=2).evaluation_controls,
    )

    assert execution.per_cell_base_operation_count == 32
    assert execution.per_cell_base_owner_authorization_count == 49
    assert execution.per_cell_retry_eligible_operation_count == 13
    assert execution.per_cell_max_operation_attempt_count == 148
    assert execution.per_cell_max_owner_authorization_count == 193


def test_live_budget_expands_ingestion_stages_and_internal_owners(tmp_path: Path) -> None:
    plan = build_resolved_plan(_configuration(retries=2))
    built = build_live_cell(
        plan=plan,
        cell_id="hindsight-lme60",
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-history-rebuild-test",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="history-rebuild-budget",
        observed_at=NOW,
        code_revision="source-tree-test",
    )
    budget = built.control.budget
    operations = {
        item.operation_kind: item.max_attempts for item in budget.provider_operation_ceilings
    }
    roles = {item.role_binding_id: item.max_attempts for item in budget.role_ceilings}
    bindings = {binding.role: binding.binding_id for binding in built.control.role_bindings}

    assert operations == {
        "runtime_resolve": 1,
        "scope_allocate": 60,
        "memory_ingest": 8_478,
        "memory_readiness": 60,
        "memory_projection": 60,
        "pre_query_projection": 60,
        "memory_query": 60,
        "post_query_projection": 60,
    }
    assert roles[bindings[ModelRole.MEMORY_EXTRACTION]] == 8_478
    assert roles[bindings[ModelRole.EMBEDDING]] == 8_538
    assert roles[bindings[ModelRole.ANSWER]] == 1_080
    assert roles[bindings[ModelRole.JUDGE]] == 1_080
    assert budget.max_attempts == 28_015


def test_old_dispatch_retry_execution_document_is_not_reinterpreted(tmp_path: Path) -> None:
    plan = build_resolved_plan(_configuration(retries=2))
    document = json.loads(doctor_module.resolved_plan_bytes(plan))
    execution = document["execution"]
    assert isinstance(execution, dict)
    execution.pop("ingestion_retry_unit", None)
    execution.pop("ingestion_recovery_strategy", None)
    execution["per_cell_retry_eligible_operation_count"] = 120
    execution["per_cell_max_operation_attempt_count"] = 9_199
    execution["per_cell_max_owner_authorization_count"] = 26_215
    payload = dict(document)
    payload.pop("resolved_plan_hash")
    document["resolved_plan_hash"] = canonical_sha256(["oamb-resolved-plan-initial-v1", payload])
    path = tmp_path / "old-dispatch-retry-plan.json"
    path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")))

    with pytest.raises(ValueError):
        doctor_module.load_resolved_plan_for_run(path)


def test_old_connection_only_rebuild_plan_is_rejected_before_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as previous_policy:
        previous_policy.setattr(
            doctor_module, "INGESTION_RECOVERY_STRATEGY", "fresh_scope_full_history_rebuild"
        )
        old_plan = build_resolved_plan(_configuration(retries=2))
        old_bytes = doctor_module.resolved_plan_bytes(old_plan)
    path = tmp_path / "old-connection-only-rebuild-plan.json"
    path.write_bytes(old_bytes)

    with pytest.raises(ValueError):
        doctor_module.load_resolved_plan_for_run(path)
