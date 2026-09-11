from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import build_resolved_plan, resolved_plan_bytes
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.workloads.longmemeval import LME60_EXPECTED_QUESTION_IDS
from tests.benchmark_configuration import MODEL_ENVIRONMENT
from tests.unit.test_question_results import _judged_result

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MIGRATION_SCRIPT = REPOSITORY_ROOT / "scripts" / "migrate_managed_local_resume.py"
PROVIDERS = ("hindsight", "mem0", "openviking")
PROGRESS_SCHEMA_NAMES = {
    "question_result": "full_progress_entry",
    "result_answer": "progress_answer",
    "result_attempt_counts": "progress_attempt_counts",
    "result_evaluation": "progress_evaluation",
    "result_ingestion": "progress_ingestion",
    "result_measurement": "progress_measurement",
    "result_retrieval": "progress_retrieval",
    "result_text": "progress_text",
}


def _migration_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "managed_local_resume_migration", MIGRATION_SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _legacy_plan_bytes() -> bytes:
    plan = build_resolved_plan(
        load_benchmark_configuration(
            REPOSITORY_ROOT / "configs" / "benchmark.yml",
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    document = json.loads(resolved_plan_bytes(plan))
    document.pop("embedding_endpoint")
    payload = dict(document)
    payload.pop("resolved_plan_hash")
    document["resolved_plan_hash"] = canonical_sha256(["oamb-resolved-plan-initial-v1", payload])
    return canonical_json_bytes(document)


def _progress_entry(result: object) -> object:
    if isinstance(result, dict):
        converted = {key: _progress_entry(value) for key, value in result.items()}
        schema_name = converted.get("schema_name")
        if isinstance(schema_name, str) and schema_name in PROGRESS_SCHEMA_NAMES:
            converted["schema_name"] = PROGRESS_SCHEMA_NAMES[schema_name]
        return converted
    if isinstance(result, list):
        return [_progress_entry(value) for value in result]
    return result


def _write_legacy_resume_source(root: Path, *, mismatch: bool = False) -> tuple[Path, Path, bytes]:
    run_label = "legacy-full-run"
    precheck_label = "legacy-precheck"
    precheck_plan = (
        root / "outputs" / "tmp" / "precheck" / precheck_label / "plan" / "resolved-plan.json"
    )
    full_root = root / "outputs" / "full-test" / run_label
    results_root = full_root / "results"
    precheck_plan.parent.mkdir(parents=True)
    results_root.mkdir(parents=True)
    plan_bytes = _legacy_plan_bytes()
    precheck_plan.write_bytes(plan_bytes)
    (full_root / "resolved-plan.json").write_bytes(plan_bytes)
    plan_hash = json.loads(plan_bytes)["resolved_plan_hash"]
    question_ids = LME60_EXPECTED_QUESTION_IDS[:2]
    results = {question_id: _judged_result(question_id) for question_id in question_ids}
    result_bytes = canonical_json_bytes(results)
    for index, provider in enumerate(PROVIDERS):
        cell_id = f"{provider}-lme60"
        (results_root / f"{provider}.json").write_bytes(result_bytes)
        progress_question_id = LME60_EXPECTED_QUESTION_IDS[2] if mismatch and index == 0 else None
        progress = {
            "schema_name": "full_progress",
            "schema_version": 1,
            "resolved_plan_hash": plan_hash,
            "cell_id": cell_id,
            "provider_id": provider,
            "workload_id": "lme60-balanced-v1",
            "case_manifest_hash": (
                "90b2669f7b893e59d404549f5803882bcd6640ce82520a9bf09672cc79464c80"
            ),
            "ordered_question_ids": list(LME60_EXPECTED_QUESTION_IDS),
            "results": [
                _progress_entry(
                    _judged_result(progress_question_id)
                    if progress_question_id is not None and result_index == 1
                    else results[question_id]
                )
                for result_index, question_id in enumerate(question_ids)
            ],
        }
        (results_root / f"progress-{provider}.json").write_bytes(canonical_json_bytes(progress))
    quick_state = {
        "run_label": precheck_label,
        "work_dir": str(precheck_plan.parents[1]),
        "resolved_plan": str(precheck_plan),
        "dataset_source": str(
            REPOSITORY_ROOT / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
        ),
        "question_id": "72e3ee87",
    }
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(canonical_json_bytes(quick_state))
    (root / "outputs" / "tmp" / "full-test-current").write_text(f"{run_label}\n", encoding="utf-8")
    dataset = root / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(
        (
            REPOSITORY_ROOT / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
        ).read_bytes()
    )
    quick_state["dataset_source"] = str(dataset)
    state_path.write_text(json.dumps(quick_state, indent=2) + "\n", encoding="utf-8")
    return full_root, dataset, result_bytes


def _write_target_precheck(root: Path) -> tuple[bytes, Path]:
    plan = build_resolved_plan(
        load_benchmark_configuration(
            REPOSITORY_ROOT / "configs" / "benchmark.yml",
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    label = "current-new-plan-precheck"
    work_dir = root / "outputs" / "tmp" / "precheck" / label
    plan_path = work_dir / "plan" / "resolved-plan.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_bytes(resolved_plan_bytes(plan))
    dataset = root / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_bytes(
        (
            REPOSITORY_ROOT / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
        ).read_bytes()
    )
    state = {
        "run_label": label,
        "work_dir": str(work_dir),
        "resolved_plan": str(plan_path),
        "dataset_source": str(dataset),
        "question_id": "72e3ee87",
        "embedding_ownership": "embedding_local_fallback",
    }
    state_bytes = canonical_json_bytes(state)
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    state_path.write_bytes(state_bytes)
    env_lines = []
    for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line == "LLM_BASE_URL=change-me":
            line = "LLM_BASE_URL=https://models.example/v1"
        elif line == "LLM_API_KEY=change-me":
            line = "LLM_API_KEY=test-model-key"
        elif line.endswith("=change-me"):
            line = f"{line.removesuffix('=change-me')}=test-value"
        env_lines.append(line)
    (root / ".env").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    return state_bytes, plan_path


def test_migration_preserves_completed_results_and_selects_only_unfinished_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    source_full_root, dataset_source, result_bytes = _write_legacy_resume_source(source)
    target_state_bytes, target_plan = _write_target_precheck(target)
    readiness_validations: list[object] = []
    migration = _migration_module()
    models = load_benchmark_configuration(
        REPOSITORY_ROOT / "configs" / "benchmark.yml",
        model_environment=MODEL_ENVIRONMENT,
    ).models

    def validate_readiness(**kwargs: object) -> Path:
        environment = kwargs["environment"]
        assert isinstance(environment, dict)
        assert environment["OAMB_HINDSIGHT_LLM_MODEL"] == models.hindsight_extraction.model
        assert environment["OAMB_MEM0_LLM_MODEL"] == models.mem0_extraction.model
        assert (
            environment["OAMB_OPENVIKING_VLM_MODEL"]
            == models.openviking_semantic_understanding.model
        )
        assert environment["OAMB_EMBEDDING_MODEL"] == models.embedding.model
        readiness_validations.append(kwargs)
        return Path("validated-receipt.json")

    monkeypatch.setattr(
        migration,
        "validate_live_readiness_receipt",
        validate_readiness,
        raising=False,
    )
    original_source_hashes = {
        item.relative_to(source): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in source.rglob("*")
        if item.is_file()
    }

    receipt = migration.migrate_resume_data(
        source_repository=source,
        target_repository=target,
        benchmark_config=REPOSITORY_ROOT / "configs" / "benchmark.yml",
        dataset_source=dataset_source,
        target_label="managed-local-migration",
        model_environment=MODEL_ENVIRONMENT,
    )

    target_full_root = target / "outputs" / "full-test" / "managed-local-migration"
    for provider in PROVIDERS:
        assert (target_full_root / "results" / f"{provider}.json").read_bytes() == result_bytes
    migrated_plan = json.loads((target_full_root / "resolved-plan.json").read_bytes())
    assert migrated_plan["embedding_endpoint"] == {
        "effective_endpoint": "http://127.0.0.1:18000/v1",
        "ownership": "embedding_local_fallback",
    }
    assert receipt["completed_result_counts"] == {
        "hindsight": 2,
        "mem0": 2,
        "openviking": 2,
    }
    assert receipt["remaining_result_counts"] == {
        "hindsight": 58,
        "mem0": 58,
        "openviking": 58,
    }
    assert receipt["external_calls_dispatched"] == 0
    assert (target / "outputs" / "tmp" / "full-test-current").read_text(
        encoding="utf-8"
    ) == "managed-local-migration\n"
    assert (
        target / "outputs" / "tmp" / "quick-start-current.json"
    ).read_bytes() == target_state_bytes
    state = json.loads(target_state_bytes)
    assert state["embedding_ownership"] == "embedding_local_fallback"
    assert (
        Path(state["resolved_plan"]).read_bytes()
        == (target_full_root / "resolved-plan.json").read_bytes()
    )
    assert Path(state["resolved_plan"]) == target_plan
    assert len(readiness_validations) == 1
    assert {
        item.relative_to(source): hashlib.sha256(item.read_bytes()).hexdigest()
        for item in source.rglob("*")
        if item.is_file()
    } == original_source_hashes
    assert source_full_root.is_dir()


def test_migration_requires_new_plan_readiness_before_publishing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _source_full_root, dataset_source, _result_bytes = _write_legacy_resume_source(source)
    target_state_bytes, _target_plan = _write_target_precheck(target)

    with pytest.raises(ValueError, match="target new-plan readiness"):
        _migration_module().migrate_resume_data(
            source_repository=source,
            target_repository=target,
            benchmark_config=REPOSITORY_ROOT / "configs" / "benchmark.yml",
            dataset_source=dataset_source,
            target_label="managed-local-migration",
            model_environment=MODEL_ENVIRONMENT,
        )

    assert (
        target / "outputs" / "tmp" / "quick-start-current.json"
    ).read_bytes() == target_state_bytes
    assert not (target / "outputs" / "tmp" / "full-test-current").exists()
    assert not (target / "outputs" / "full-test" / "managed-local-migration").exists()


def test_migration_mismatch_leaves_active_state_and_target_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    target.mkdir()
    _source_full_root, dataset_source, _result_bytes = _write_legacy_resume_source(
        source, mismatch=True
    )
    previous_state = b'{"previous":true}\n'
    previous_selector = b"previous-run\n"
    target_tmp = target / "outputs" / "tmp"
    target_tmp.mkdir(parents=True)
    (target_tmp / "quick-start-current.json").write_bytes(previous_state)
    (target_tmp / "full-test-current").write_bytes(previous_selector)

    with pytest.raises(ValueError, match="progress and ordinary result IDs differ"):
        _migration_module().migrate_resume_data(
            source_repository=source,
            target_repository=target,
            benchmark_config=REPOSITORY_ROOT / "configs" / "benchmark.yml",
            dataset_source=dataset_source,
            target_label="managed-local-migration",
            model_environment=MODEL_ENVIRONMENT,
        )

    assert (target_tmp / "quick-start-current.json").read_bytes() == previous_state
    assert (target_tmp / "full-test-current").read_bytes() == previous_selector
    assert not (target / "outputs" / "full-test" / "managed-local-migration").exists()
