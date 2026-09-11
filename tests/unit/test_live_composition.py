from __future__ import annotations

import hashlib
import importlib.util
import json
import multiprocessing
import os
import signal
import time
from dataclasses import replace
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import ResolvedPlan, build_resolved_plan
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    RunPreflightRecord,
    SourceEvidenceBinding,
    SourceEvidenceKind,
)
from tests.benchmark_configuration import MODEL_ENVIRONMENT, load_lme6_configuration

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
LME60_BENCHMARK_CONFIG = REPOSITORY_ROOT / "configs" / "benchmark.yml"
NOW = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)


def _plan() -> ResolvedPlan:
    return build_resolved_plan(load_lme6_configuration())


def _lme60_plan() -> ResolvedPlan:
    return build_resolved_plan(
        load_benchmark_configuration(
            LME60_BENCHMARK_CONFIG,
            model_environment=MODEL_ENVIRONMENT,
        )
    )


def _provider_evidence() -> SourceEvidenceBinding:
    source_root_hash = canonical_sha256(["provider-proof-root"])
    validation_result_hash = canonical_sha256(["provider-verification"])
    return SourceEvidenceBinding(
        binding_id=canonical_sha256(
            ["provider-evidence", source_root_hash, validation_result_hash]
        ),
        source_kind=SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity="oamb-providers-test-live",
        source_root_hash=source_root_hash,
        validation_result_hash=validation_result_hash,
        source_schema_versions=("provider_service_evidence_manifest@1",),
    )


def _environment() -> dict[str, str]:
    models = load_benchmark_configuration(
        LME60_BENCHMARK_CONFIG, model_environment=MODEL_ENVIRONMENT
    ).models
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
        "LLM_LIGHT_MODEL": models.hindsight_extraction.model,
        "LLM_DEEP_MODEL": models.answer.model,
        "OAMB_EMBEDDING_BASE_URL": "http://127.0.0.1:18000/v1",
        "OAMB_EMBEDDING_API_KEY": "",
        "OAMB_EMBEDDING_MODEL": models.embedding.model,
    }


@pytest.mark.parametrize(
    ("cell_id", "memory_system_id", "profile_id", "role_ids"),
    (
        (
            "hindsight-lme6",
            "hindsight",
            "hindsight-rest-v1",
            ("hindsight_extraction", "embedding", "answer", "judge"),
        ),
        (
            "mem0-lme6",
            "mem0",
            "mem0-rest-v1",
            ("mem0_extraction", "embedding", "answer", "judge"),
        ),
        (
            "openviking-lme6",
            "openviking",
            "openviking-session-rest-v1",
            ("openviking_semantic_understanding", "embedding", "answer", "judge"),
        ),
    ),
)
def test_builds_one_closed_live_cell_without_approval_or_client_construction(
    tmp_path: Path,
    cell_id: str,
    memory_system_id: str,
    profile_id: str,
    role_ids: tuple[str, ...],
) -> None:
    from oamb.live import build_live_cell

    built = build_live_cell(
        plan=_plan(),
        cell_id=cell_id,
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment={**_environment(), "OAMB_RUN_SH_PROCESS_ID": "4242"},
        run_label="fresh-rerun-1",
        observed_at=NOW,
        code_revision="source-tree-test",
    )

    assert built.cell.cell_id == cell_id
    assert built.control.run_spec.memory_system_id == memory_system_id
    assert built.control.preflight_record.adapter_profile_id == profile_id
    assert built.control.run_spec.case_manifest_hash == _plan().dataset.case_manifest_hash
    assert built.capsule_root == tmp_path / "capsules" / built.run_id
    assert built.control.shutdown_owner_process_ids == (4242, os.getpid())
    assert (
        built.control.provider_runtime_directory
        == (tmp_path / "provider-runtime" / "lifecycle-domains" / memory_system_id).resolve()
    )
    assert (
        built.control.provider_lifecycle_coordination_directory
        == (tmp_path / "provider-runtime").resolve()
    )
    assert tuple(binding.role.value for binding in built.control.role_bindings) == (
        "memory_extraction",
        "embedding",
        "answer",
        "judge",
    )
    assert (
        next(
            binding.provider
            for binding in built.control.role_bindings
            if binding.role.value == "embedding"
        )
        == "openai_embeddings"
    )
    assert built.role_ids == role_ids
    assert isinstance(built.control.preflight_record, RunPreflightRecord)
    assert built.control.preflight_record.provider_profile_evidence == _provider_evidence()
    serialized = built.control.preflight_record.model_dump_json().lower()
    assert "approval" not in serialized
    assert "signature" not in serialized
    assert not built.capsule_root.exists()


def test_missing_runtime_value_or_existing_capsule_fails_before_factory(
    tmp_path: Path,
) -> None:
    from oamb.live import LiveCell, LiveConfigurationError, build_live_cell

    plan = _plan()

    def build(environment: dict[str, str]) -> LiveCell:
        return build_live_cell(
            plan=plan,
            cell_id="mem0-lme6",
            output_root=tmp_path / "capsules",
            provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
            provider_project_id="oamb-providers-test-live",
            provider_evidence=_provider_evidence(),
            environment=environment,
            run_label="fresh-rerun-2",
            observed_at=NOW,
            code_revision="source-tree-test",
        )

    missing = _environment()
    missing.pop("OAMB_MEM0_INSPECTOR_API_KEY")
    with pytest.raises(LiveConfigurationError, match="OAMB_MEM0_INSPECTOR_API_KEY"):
        build(missing)

    first = build(_environment())
    first.capsule_root.mkdir(parents=True)
    with pytest.raises(LiveConfigurationError, match="already exists"):
        build(_environment())


def test_missing_embedding_api_key_is_resolved_as_empty(tmp_path: Path) -> None:
    from oamb.live import build_live_cell

    environment = _environment()
    environment.pop("OAMB_EMBEDDING_API_KEY")

    built = build_live_cell(
        plan=_plan(),
        cell_id="hindsight-lme6",
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=environment,
        run_label="missing-embedding-key",
        observed_at=NOW,
        code_revision="source-tree-test",
    )

    assert built.environment["OAMB_EMBEDDING_API_KEY"] == ""


@pytest.mark.parametrize(
    ("cell_id", "adapter_module", "adapter_name"),
    (
        (
            "hindsight-lme6",
            "oamb.memory_systems.hindsight.adapter",
            "HindsightAdapter",
        ),
        ("mem0-lme6", "oamb.memory_systems.mem0.adapter", "Mem0RestAdapter"),
        (
            "openviking-lme6",
            "oamb.memory_systems.openviking.session_adapter",
            "OpenVikingSessionAdapter",
        ),
    ),
)
def test_live_memory_factory_receives_the_resolved_memory_operation_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cell_id: str,
    adapter_module: str,
    adapter_name: str,
) -> None:
    from oamb import live
    from oamb.runtime import native_run
    from oamb.workloads.fake import GeneratedFakeWorkload

    captured: dict[str, object] = {}

    class RecordingAdapter:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    def record_memory_factory(**kwargs: object) -> object:
        memory_factory = kwargs["memory_factory"]
        assert callable(memory_factory)
        memory_factory(object(), object())
        return SimpleNamespace(capsule_root=tmp_path / "capsule")

    monkeypatch.setattr(import_module(adapter_module), adapter_name, RecordingAdapter)
    monkeypatch.setattr(live, "_validated_cell_internal_retry_count", lambda _cell: 0)
    monkeypatch.setattr(native_run, "run_native_vertical_slice", record_memory_factory)
    monkeypatch.setattr(live, "build_longmemeval_bundle", lambda _path, _selection: object())
    monkeypatch.setattr(
        live,
        "LongMemEvalWorkload",
        lambda _bundle: GeneratedFakeWorkload(),
    )
    built = live.build_live_cell(
        plan=_plan(),
        cell_id=cell_id,
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="timeout-wiring",
        observed_at=NOW,
        code_revision="source-tree-test",
    )

    live.execute_live_cell(built)

    assert captured["read_timeout_seconds"] == 900.0
    assert captured["total_timeout_seconds"] == 900.0
    assert captured["internal_retry_count"] == 0
    if cell_id == "openviking-lme6":
        assert captured["maximum_task_polls"] == 9_000


def test_live_model_factories_receive_the_model_call_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live
    from oamb.model_clients import openai_compatible
    from oamb.runtime import native_run
    from oamb.workloads.fake import GeneratedFakeWorkload

    captured: list[dict[str, object]] = []

    class RecordingModelClient:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    def record_model_factories(**kwargs: object) -> object:
        model_factory = kwargs["model_factory"]
        judge_model_factory = kwargs["judge_model_factory"]
        assert callable(model_factory)
        assert callable(judge_model_factory)
        model_factory(object())
        judge_model_factory(object())
        return SimpleNamespace(capsule_root=tmp_path / "capsule")

    monkeypatch.setattr(
        openai_compatible,
        "OpenAICompatibleModelClient",
        RecordingModelClient,
    )
    monkeypatch.setattr(native_run, "run_native_vertical_slice", record_model_factories)
    monkeypatch.setattr(live, "build_longmemeval_bundle", lambda _path, _selection: object())
    monkeypatch.setattr(
        live,
        "LongMemEvalWorkload",
        lambda _bundle: GeneratedFakeWorkload(),
    )
    built = live.build_live_cell(
        plan=_plan(),
        cell_id="hindsight-lme6",
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="model-timeout-wiring",
        observed_at=NOW,
        code_revision="source-tree-test",
    )

    live.execute_live_cell(built)

    assert [item["read_timeout_seconds"] for item in captured] == [900.0, 900.0]
    assert [item["total_timeout_seconds"] for item in captured] == [900.0, 900.0]


def test_dangling_capsule_root_symlink_is_not_a_fresh_cell(tmp_path: Path) -> None:
    from oamb.live import LiveConfigurationError, build_live_cell

    arguments = {
        "plan": _plan(),
        "cell_id": "hindsight-lme6",
        "output_root": tmp_path / "capsules",
        "provider_runtime_directory": (tmp_path / "provider-runtime").resolve(),
        "provider_project_id": "oamb-providers-test-live",
        "provider_evidence": _provider_evidence(),
        "environment": _environment(),
        "run_label": "dangling-root",
        "observed_at": NOW,
        "code_revision": "source-tree-test",
    }
    built = build_live_cell(**arguments)  # type: ignore[arg-type]
    built.capsule_root.parent.mkdir(parents=True)
    built.capsule_root.symlink_to(tmp_path / "missing-capsule", target_is_directory=True)

    with pytest.raises(LiveConfigurationError, match="already exists"):
        build_live_cell(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "cell_id",
    ("hindsight-lme6", "mem0-lme6", "openviking-lme6"),
)
def test_lme6_budget_accepts_the_complete_owner_allocation_matrix(
    tmp_path: Path,
    cell_id: str,
) -> None:
    from oamb.live import build_live_cell
    from oamb.runtime.budget import BudgetOwnerAllocation, ReservationRequest
    from oamb.runtime.native_run import (
        _live_budget_ledger,
        _reservation_allocations_for_route,
    )

    built = build_live_cell(
        plan=_plan(),
        cell_id=cell_id,
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="budget-closure",
        observed_at=NOW,
        code_revision="source-tree-test",
    )
    ledger = _live_budget_ledger(built.control)
    dispatch_counts = {
        "runtime_resolve": 1,
        "scope_allocate": 6,
        "memory_ingest": 300,
        "memory_readiness": 6,
        "memory_projection": 6,
        "pre_query_projection": 6,
        "memory_query": 6,
        "post_query_projection": 6,
        "answer": 6,
        "judge": 6,
    }
    expected_route_wall_seconds = {
        "runtime_resolve": 900,
        "scope_allocate": 900,
        "memory_ingest": 900,
        "memory_readiness": 900,
        "memory_projection": 900,
        "pre_query_projection": 900,
        "memory_query": 900,
        "post_query_projection": 900,
        "answer": 900,
        "judge": 900,
    }

    reservation_ordinal = 0
    for stage, count in dispatch_counts.items():
        route = built.control.require_budget_route(stage=stage)
        maximum_output_tokens = 8192 if stage == "answer" else 1024 if stage == "judge" else 0
        for _ in range(count):
            reservation_ordinal += 1
            _evidence, allocations, maximum = _reservation_allocations_for_route(
                built.control.budget,
                route,
                maximum_output_tokens=maximum_output_tokens,
            )
            assert maximum.wall_seconds == expected_route_wall_seconds[stage]
            owner_allocations = tuple(
                BudgetOwnerAllocation(item.owner_id, item.maximum) for item in allocations
            )
            reservation_id = f"reservation-{reservation_ordinal}"
            ledger.reserve(
                ReservationRequest(
                    reservation_id=reservation_id,
                    maximum=maximum,
                    owner_allocations=owner_allocations,
                )
            )
            ledger.commit(
                reservation_id,
                observed=maximum,
                owner_observed=owner_allocations,
            )

    snapshot = ledger.snapshot()
    assert sum(dispatch_counts.values()) == 349
    assert snapshot.reserved.attempts == 0
    assert snapshot.committed.attempts == 955
    assert snapshot.committed.output_tokens >= 6 * 8192


@pytest.mark.parametrize(
    ("cell_id", "producer_role_id"),
    (
        ("hindsight-lme60", "hindsight_extraction"),
        ("mem0-lme60", "mem0_extraction"),
        ("openviking-lme60", "openviking_semantic_understanding"),
    ),
)
def test_lme60_cells_use_their_frozen_workload_and_producer_role(
    tmp_path: Path,
    cell_id: str,
    producer_role_id: str,
) -> None:
    from oamb.live import build_live_cell

    built = build_live_cell(
        plan=_lme60_plan(),
        cell_id=cell_id,
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="lme60-cell-closure",
        observed_at=NOW,
        code_revision="source-tree-test",
    )

    assert built.role_ids == (producer_role_id, "embedding", "answer", "judge")
    assert built.control.run_spec.workload_id == "lme60-balanced-v1"
    assert built.control.run_spec.case_manifest_hash == (
        "90b2669f7b893e59d404549f5803882bcd6640ce82520a9bf09672cc79464c80"
    )
    assert built.control.budget.max_attempts == 28_015
    assert built.control.max_retries_per_operation == 2
    attempts_by_binding = {
        ceiling.role_binding_id: ceiling.max_attempts
        for ceiling in built.control.budget.role_ceilings
    }
    binding_by_role = dict(zip(built.role_ids, built.control.role_bindings, strict=True))
    assert attempts_by_binding == {
        binding_by_role[producer_role_id].binding_id: 8_478,
        binding_by_role["embedding"].binding_id: 8_538,
        binding_by_role["answer"].binding_id: 1_080,
        binding_by_role["judge"].binding_id: 1_080,
    }


def test_lme60_budget_accepts_all_9019_owner_allocations(tmp_path: Path) -> None:
    from oamb.live import build_live_cell
    from oamb.runtime.budget import BudgetOwnerAllocation, ReservationRequest
    from oamb.runtime.native_run import (
        _live_budget_ledger,
        _reservation_allocations_for_route,
    )

    built = build_live_cell(
        plan=_lme60_plan(),
        cell_id="hindsight-lme60",
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="lme60-budget-closure",
        observed_at=NOW,
        code_revision="source-tree-test",
    )
    ledger = _live_budget_ledger(built.control)
    dispatch_counts = {
        "runtime_resolve": 1,
        "scope_allocate": 60,
        "memory_ingest": 2_826,
        "memory_readiness": 60,
        "memory_projection": 60,
        "pre_query_projection": 60,
        "memory_query": 60,
        "post_query_projection": 60,
        "answer": 60,
        "judge": 60,
    }

    reservation_ordinal = 0
    for stage, count in dispatch_counts.items():
        route = built.control.require_budget_route(stage=stage)
        maximum_output_tokens = 8192 if stage == "answer" else 1024 if stage == "judge" else 0
        for _ in range(count):
            reservation_ordinal += 1
            _evidence, allocations, maximum = _reservation_allocations_for_route(
                built.control.budget,
                route,
                maximum_output_tokens=maximum_output_tokens,
            )
            owner_allocations = tuple(
                BudgetOwnerAllocation(item.owner_id, item.maximum) for item in allocations
            )
            reservation_id = f"lme60-reservation-{reservation_ordinal}"
            ledger.reserve(
                ReservationRequest(
                    reservation_id=reservation_id,
                    maximum=maximum,
                    owner_allocations=owner_allocations,
                )
            )
            ledger.commit(
                reservation_id,
                observed=maximum,
                owner_observed=owner_allocations,
            )

    snapshot = ledger.snapshot()
    assert sum(dispatch_counts.values()) == 3_307
    assert snapshot.reserved.attempts == 0
    assert snapshot.committed.attempts == 9_019
    assert snapshot.committed.attempts <= built.control.budget.max_attempts


def test_unknown_or_out_of_order_cell_selection_is_rejected() -> None:
    from oamb.live import LiveConfigurationError, select_live_cells

    plan = _plan()
    assert tuple(cell.cell_id for cell in select_live_cells(plan, ())) == (
        "hindsight-lme6",
        "mem0-lme6",
        "openviking-lme6",
    )
    with pytest.raises(LiveConfigurationError, match="canonical order"):
        select_live_cells(plan, ("mem0-lme6", "hindsight-lme6"))
    with pytest.raises(LiveConfigurationError, match="unknown"):
        select_live_cells(plan, ("not-a-cell",))


def test_public_question_selector_resolves_frozen_raw_id_before_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live

    first_case_id = canonical_sha256(["first-case"])
    second_case_id = canonical_sha256(["second-case"])
    bundle = SimpleNamespace(
        case_manifest=SimpleNamespace(
            cases=(
                SimpleNamespace(
                    raw_question_id="72e3ee87",
                    case_manifest_entry_id=first_case_id,
                ),
                SimpleNamespace(
                    raw_question_id="22d2cb42",
                    case_manifest_entry_id=second_case_id,
                ),
            )
        )
    )
    observed: list[tuple[Path, str]] = []

    def build(path: Path, selection: str) -> object:
        observed.append((path, selection))
        return bundle

    monkeypatch.setattr(live, "build_longmemeval_bundle", build)

    assert live.resolve_live_question_case_ids(
        _lme60_plan(),
        Path("dataset.json"),
        ("72e3ee87",),
    ) == (first_case_id,)
    assert observed == [(Path("dataset.json"), "lme60")]
    with pytest.raises(live.LiveConfigurationError, match="unknown frozen question"):
        live.resolve_live_question_case_ids(
            _lme60_plan(),
            Path("dataset.json"),
            ("not-frozen",),
        )
    with pytest.raises(live.LiveConfigurationError, match="duplicate"):
        live.resolve_live_question_case_ids(
            _lme60_plan(),
            Path("dataset.json"),
            ("72e3ee87", "72e3ee87"),
        )


def test_live_model_env_file_overrides_process_generic_connection(
    tmp_path: Path,
) -> None:
    from oamb import live

    provider_env = tmp_path / "provider.env"
    provider_env.write_text(
        "\n".join(f"{name}={value}" for name, value in _environment().items()) + "\n",
        encoding="utf-8",
    )
    model_env = tmp_path / "model.env"
    model_env.write_text(
        "LLM_URL_TYPE=openai_chat\n"
        "LLM_BASE_URL=https://file-model.example/v1\n"
        "LLM_API_KEY=file-answer-key\n"
        "LLM_LIGHT_MODEL=file-light-model\n"
        "LLM_DEEP_MODEL=file-deep-model\n",
        encoding="utf-8",
    )

    environment = live.load_live_environment(
        plan=_lme60_plan(),
        provider_env_path=provider_env,
        model_env_path=model_env,
        provider_runtime_directory=tmp_path / "runtime",
        base_environment={
            "LLM_URL_TYPE": "openai_chat",
            "LLM_BASE_URL": "https://process-model.example/v1",
            "LLM_API_KEY": "process-answer-key",
        },
    )

    assert environment["LLM_URL_TYPE"] == "openai_chat"
    assert environment["LLM_BASE_URL"] == "https://file-model.example/v1"
    assert environment["LLM_API_KEY"] == "file-answer-key"
    assert environment["LLM_LIGHT_MODEL"] == "file-light-model"
    assert environment["LLM_DEEP_MODEL"] == "file-deep-model"


def test_runtime_role_bindings_use_current_generative_model_names() -> None:
    from oamb import live

    bindings = live._role_bindings(
        _lme60_plan().model_roles,
        {
            "LLM_LIGHT_MODEL": "runtime-light-model",
            "LLM_DEEP_MODEL": "runtime-deep-model",
            "LLM_BASE_URL": "https://model.example/v1",
            "LLM_API_KEY": "model-key",
            "OAMB_EMBEDDING_BASE_URL": "http://127.0.0.1:18000/v1",
            "OAMB_EMBEDDING_API_KEY": "",
        },
    )

    assert tuple(binding.model for binding in bindings) == (
        "runtime-light-model",
        "runtime-light-model",
        "runtime-light-model",
        "runtime-deep-model",
        "runtime-light-model",
        "qwen3-embedding:0.6b",
    )


def test_live_readiness_hash_binds_llm_url_type() -> None:
    from oamb import live

    plan = _plan()
    environment = _environment()

    assert live.live_readiness_environment_hash(plan, environment) != (
        live.live_readiness_environment_hash(
            plan,
            {**environment, "LLM_URL_TYPE": "future_protocol"},
        )
    )


def _write_model_readiness_evidence(
    runtime: Path,
    *,
    plan: ResolvedPlan,
    environment: dict[str, str],
    project: str,
    service_bytes: bytes,
) -> tuple[Path, dict[str, object], Path]:
    from oamb import live
    from oamb.config.benchmark import MODEL_ROLE_IDS

    receipts = runtime / "service-verification-receipts"
    receipts.mkdir(parents=True)
    service_hash = hashlib.sha256(service_bytes).hexdigest()
    service_path = receipts / f"{service_hash}.json"
    service_path.write_bytes(service_bytes)
    attempt = {
        "schema_version": "oamb-provider-model-readiness-attempt-v1",
        "provider_project": project,
        "resolved_plan_hash": plan.resolved_plan_hash,
        "started_at_utc": "2026-09-02T00:00:00Z",
        "calls_reserved": 7,
        "calls_dispatched": 7,
        "operation_timeout_seconds": 900,
        "attempts": [
            {
                "role": role,
                "status": "succeeded",
                "dispatched_at_utc": "2026-09-02T00:00:00Z",
                "completed_at_utc": "2026-09-02T00:00:01Z",
            }
            for role in (
                "embedding",
                "hindsight_extraction",
                "mem0_extraction",
                "openviking_semantic_understanding",
                "answer",
                "judge",
                "openviking-ready",
            )
        ],
        "billing_complete": False,
        "cost_usd": None,
        "memory_state_created": False,
    }
    attempt_bytes = json.dumps(attempt).encode()
    (runtime / "model-readiness-attempt.json").write_bytes(attempt_bytes)
    receipt = {
        "schema_version": "oamb-provider-model-readiness-receipt-v1",
        "provider_project": project,
        "resolved_plan_hash": plan.resolved_plan_hash,
        "service_verification_receipt_sha256": service_hash,
        "attempt_sha256": hashlib.sha256(attempt_bytes).hexdigest(),
        "completed_at_utc": "2026-09-02T00:00:02Z",
        "operation_timeout_seconds": 900,
        "model_calls_dispatched": 7,
        "role_ids": list(MODEL_ROLE_IDS),
        "provider_internal_retries": 0,
        "environment_hash": live.live_readiness_environment_hash(plan, environment),
        "billing_complete": False,
        "cost_usd": None,
        "memory_state_created": False,
    }
    receipt_path = runtime / "model-readiness-receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return service_path, receipt, receipt_path


def test_live_readiness_receipt_accepts_rotated_model_connection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live

    monkeypatch.setattr(
        live,
        "validate_live_service_verification_receipt",
        lambda **_kwargs: None,
    )
    plan = _lme60_plan()
    environment = _environment()
    runtime = tmp_path / "runtime"
    service_path, _, _ = _write_model_readiness_evidence(
        runtime,
        plan=plan,
        environment=environment,
        project="oamb-providers-readiness-test",
        service_bytes=json.dumps({"provider_project": "oamb-providers-readiness-test"}).encode(),
    )

    rotated_connection = {
        **environment,
        "LLM_BASE_URL": "https://rotated-model.example/v1",
        "LLM_API_KEY": "rotated-model-key",
    }

    assert (
        live.validate_live_readiness_receipt(
            plan=plan,
            provider_runtime_directory=runtime,
            environment=rotated_connection,
        )
        == service_path
    )


def test_live_readiness_receipt_binds_all_roles_plan_and_zero_internal_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live

    monkeypatch.setattr(
        live,
        "validate_live_service_verification_receipt",
        lambda **_kwargs: None,
    )

    plan = _lme60_plan()
    environment = _environment()
    runtime = tmp_path / "runtime"
    service_path, receipt, receipt_path = _write_model_readiness_evidence(
        runtime,
        plan=plan,
        environment=environment,
        project="oamb-providers-readiness-test",
        service_bytes=json.dumps({"provider_project": "oamb-providers-readiness-test"}).encode(),
    )
    receipts = service_path.parent
    unrelated_bytes = json.dumps({"provider_project": "unrelated-history"}).encode()
    unrelated_hash = hashlib.sha256(unrelated_bytes).hexdigest()
    (receipts / f"{unrelated_hash}.json").write_bytes(unrelated_bytes)
    (receipts / "malformed-history.json").write_text("not json", encoding="utf-8")
    (receipts / "service-verification-current.sha256").write_text(
        "malformed current selection\n", encoding="utf-8"
    )

    assert (
        live.validate_live_readiness_receipt(
            plan=plan,
            provider_runtime_directory=runtime,
            environment=environment,
        )
        == service_path
    )
    (receipts / "service-verification-current.sha256").unlink()
    assert (
        live.validate_live_readiness_receipt(
            plan=plan,
            provider_runtime_directory=runtime,
            environment=environment,
        )
        == service_path
    )

    drifted_embedding = {**environment, "OAMB_EMBEDDING_MODEL": "different-embedding"}
    with pytest.raises(live.LiveConfigurationError, match="embedding.*model"):
        live.validate_live_readiness_receipt(
            plan=plan,
            provider_runtime_directory=runtime,
            environment=drifted_embedding,
        )

    for invalid in (None, True, False, 1):
        receipt["provider_internal_retries"] = invalid
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        with pytest.raises(live.LiveConfigurationError, match="resolved plan"):
            live.validate_live_readiness_receipt(
                plan=plan,
                provider_runtime_directory=runtime,
                environment=environment,
            )


def test_live_readiness_rejects_bound_service_receipt_with_stale_attestation(
    tmp_path: Path,
) -> None:
    from oamb import live
    from oamb.config.provider_services import ProviderServiceBindingError

    plan = _lme60_plan()
    environment = _environment()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    project = "oamb-providers-readiness-test"
    attestation = f"project={project}\n".encode()
    (runtime / "provider-project.attestation").write_bytes(attestation)
    actual_attestation_hash = hashlib.sha256(attestation).hexdigest()
    stale_attestation_hash = "a" * 64
    assert stale_attestation_hash != actual_attestation_hash
    service_bytes = json.dumps(
        {
            "profiles": [],
            "project_attestation_sha256": stale_attestation_hash,
            "provider_project": project,
            "schema_name": "oamb_provider_service_verification",
            "schema_version": 1,
            "verified_at_utc": "2026-09-02T00:00:00Z",
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    _write_model_readiness_evidence(
        runtime,
        plan=plan,
        environment=environment,
        project=project,
        service_bytes=service_bytes,
    )

    with pytest.raises(
        live.LiveConfigurationError,
        match="selected service verification receipt or proof is malformed",
    ) as failure:
        live.validate_live_readiness_receipt(
            plan=plan,
            provider_runtime_directory=runtime,
            environment=environment,
        )

    assert isinstance(failure.value.__cause__, ProviderServiceBindingError)
    assert str(failure.value.__cause__) == "provider project attestation hash does not match"


def test_service_receipt_resolution_uses_current_selection_or_explicit_binding(
    tmp_path: Path,
) -> None:
    from oamb import live

    runtime = tmp_path / "runtime"
    receipts = runtime / "service-verification-receipts"
    receipts.mkdir(parents=True)
    first_bytes = b'{"receipt":"first"}'
    second_bytes = b'{"receipt":"second"}'
    first_hash = hashlib.sha256(first_bytes).hexdigest()
    second_hash = hashlib.sha256(second_bytes).hexdigest()
    first_path = receipts / f"{first_hash}.json"
    second_path = receipts / f"{second_hash}.json"
    first_path.write_bytes(first_bytes)
    second_path.write_bytes(second_bytes)
    (receipts / "service-verification-current.sha256").write_text(
        f"{second_hash}\n", encoding="utf-8"
    )

    assert live.resolve_service_verification_receipt(runtime) == second_path
    assert (
        live.resolve_service_verification_receipt(runtime, expected_sha256=first_hash) == first_path
    )


def test_service_receipt_resolution_rejects_missing_bound_receipt(tmp_path: Path) -> None:
    from oamb import live

    runtime = tmp_path / "runtime"
    (runtime / "service-verification-receipts").mkdir(parents=True)

    with pytest.raises(live.LiveConfigurationError, match="missing or malformed"):
        live.resolve_service_verification_receipt(runtime, expected_sha256="a" * 64)


@pytest.mark.parametrize("provider", ("hindsight", "mem0", "openviking"))
def test_cell_retry_guard_reopens_bound_proof_with_reduced_environment(
    tmp_path: Path, provider: str
) -> None:
    from oamb import live
    from tests.unit.test_provider_service_binding import (
        PROJECT,
        _write_receipt,
        _write_service_artifacts,
    )

    plan = _lme60_plan()
    environment = _environment()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    attestation = f"project={PROJECT}\n".encode()
    (runtime / "provider-project.attestation").write_bytes(attestation)
    (runtime / "embedding-ready-response.json").write_bytes(b'{"dimension":1024}')
    original_receipt = _write_service_artifacts(tmp_path / "proof-fixture")
    receipt_document = json.loads(original_receipt.read_bytes())
    receipt_document["project_attestation_sha256"] = hashlib.sha256(attestation).hexdigest()
    receipts = runtime / "service-verification-receipts"
    original_receipt.parent.rename(receipts)
    receipt_path = _write_receipt(receipts, receipt_document)
    project, evidence = live.load_live_provider_evidence(
        plan=plan,
        provider_runtime_directory=runtime,
        environment=environment,
        service_receipt_path=receipt_path,
    )
    cell = live.build_live_cell(
        plan=plan,
        cell_id=f"{provider}-lme60",
        output_root=tmp_path / "capsules",
        provider_runtime_directory=runtime,
        provider_project_id=project,
        provider_evidence=evidence[provider],
        environment=environment,
        run_label="reduced-environment-proof",
        observed_at=NOW,
        code_revision="source-tree-test",
    )
    assert "OAMB_HINDSIGHT_LLM_MODEL" not in cell.environment
    assert set(cell.environment) < set(environment)
    # A later current-pointer selection cannot replace this cell's frozen receipt.
    (receipts / "service-verification-current.sha256").write_text("a" * 64 + "\n")
    assert live._validated_cell_internal_retry_count(cell) == 10

    manifest_hash = receipt_document["profiles"][0]["proof_manifest_sha256"]
    manifest = json.loads((receipts / f"proofs/manifests/{manifest_hash}.json").read_bytes())
    retry_blob = next(item for item in manifest["files"] if "retry-config" in item["relative_path"])
    (receipts / f"proofs/blobs/{retry_blob['sha256']}").write_bytes(b"{}")
    with pytest.raises(live.LiveConfigurationError, match="receipt or proof is malformed"):
        live._validated_cell_internal_retry_count(cell)


def test_readiness_plan_binds_current_receipt_before_attempt_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = REPOSITORY_ROOT / "provider-services/lib/readiness_plan.py"
    spec = importlib.util.spec_from_file_location("provider_readiness_plan", module_path)
    assert spec is not None and spec.loader is not None
    readiness_plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readiness_plan)
    plan = _lme60_plan()
    environment = _environment()
    runtime = tmp_path / "runtime"
    receipts = runtime / "service-verification-receipts"
    receipts.mkdir(parents=True)
    receipt_bytes = b'{"receipt":"current"}'
    receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
    (receipts / f"{receipt_hash}.json").write_bytes(receipt_bytes)
    pointer = receipts / "service-verification-current.sha256"
    pointer.write_text(f"{receipt_hash}\n", encoding="utf-8")
    validated_paths: list[Path] = []
    monkeypatch.setattr(readiness_plan, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(readiness_plan, "load_live_environment", lambda **_kwargs: environment)
    monkeypatch.setattr(
        readiness_plan,
        "validate_live_service_verification_receipt",
        lambda **kwargs: validated_paths.append(kwargs["service_receipt_path"]),
        raising=False,
    )

    document = readiness_plan.readiness_plan_document(
        tmp_path / "resolved-plan.json",
        provider_env_path=tmp_path / "provider.env",
        model_env_path=tmp_path / "model.env",
        provider_runtime_directory=runtime,
    )

    assert document["service_verification_receipt_sha256"] == receipt_hash
    assert validated_paths == [receipts / f"{receipt_hash}.json"]
    pointer.write_text("malformed\n", encoding="utf-8")
    with pytest.raises(ValueError):
        readiness_plan.readiness_plan_document(
            tmp_path / "resolved-plan.json",
            provider_env_path=tmp_path / "provider.env",
            model_env_path=tmp_path / "model.env",
            provider_runtime_directory=runtime,
        )
    assert not (runtime / "model-readiness-plan.json").exists()
    assert not (runtime / "model-readiness-attempt.json").exists()


def test_readiness_plan_rejects_content_addressed_malformed_service_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = REPOSITORY_ROOT / "provider-services/lib/readiness_plan.py"
    spec = importlib.util.spec_from_file_location("invalid_provider_readiness_plan", module_path)
    assert spec is not None and spec.loader is not None
    readiness_plan = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(readiness_plan)
    runtime = tmp_path / "runtime"
    receipts = runtime / "service-verification-receipts"
    receipts.mkdir(parents=True)
    (runtime / "provider-project.attestation").write_text(
        "project=oamb-providers-readiness-test\n", encoding="utf-8"
    )
    receipt_bytes = b"not-json-but-content-addressed"
    receipt_hash = hashlib.sha256(receipt_bytes).hexdigest()
    (receipts / f"{receipt_hash}.json").write_bytes(receipt_bytes)
    (receipts / "service-verification-current.sha256").write_text(
        f"{receipt_hash}\n", encoding="utf-8"
    )
    monkeypatch.setattr(readiness_plan, "load_resolved_plan_for_run", lambda _path: _lme60_plan())
    monkeypatch.setattr(readiness_plan, "load_live_environment", lambda **_kwargs: _environment())

    with pytest.raises(ValueError):
        readiness_plan.readiness_plan_document(
            tmp_path / "resolved-plan.json",
            provider_env_path=tmp_path / "provider.env",
            model_env_path=tmp_path / "model.env",
            provider_runtime_directory=runtime,
        )
    assert not (runtime / "model-readiness-plan.json").exists()
    assert not (runtime / "model-readiness-attempt.json").exists()


@pytest.mark.parametrize(
    "failure",
    ("missing", "malformed", "symlink", "non-hex", "absent-target", "mismatch"),
)
def test_service_receipt_resolution_rejects_invalid_current_selection(
    tmp_path: Path, failure: str
) -> None:
    from oamb import live

    runtime = tmp_path / "runtime"
    receipts = runtime / "service-verification-receipts"
    receipts.mkdir(parents=True)
    pointer = receipts / "service-verification-current.sha256"
    content = b'{"receipt":"current"}'
    content_hash = hashlib.sha256(content).hexdigest()
    if failure == "malformed":
        pointer.write_text(content_hash, encoding="utf-8")
    elif failure == "symlink":
        target = tmp_path / "pointer-target"
        target.write_text(f"{content_hash}\n", encoding="utf-8")
        pointer.symlink_to(target)
    elif failure == "non-hex":
        pointer.write_text(f"{'g' * 64}\n", encoding="utf-8")
    elif failure == "absent-target":
        pointer.write_text(f"{content_hash}\n", encoding="utf-8")
    elif failure == "mismatch":
        pointer.write_text(f"{content_hash}\n", encoding="utf-8")
        (receipts / f"{content_hash}.json").write_bytes(b"different")

    with pytest.raises(live.LiveConfigurationError):
        live.resolve_service_verification_receipt(runtime)


@pytest.mark.parametrize("failure", ("malformed-hash", "symlink", "mismatch"))
def test_service_receipt_resolution_rejects_invalid_bound_receipt(
    tmp_path: Path, failure: str
) -> None:
    from oamb import live

    runtime = tmp_path / "runtime"
    receipts = runtime / "service-verification-receipts"
    receipts.mkdir(parents=True)
    content = b'{"receipt":"bound"}'
    content_hash = hashlib.sha256(content).hexdigest()
    bound_path = receipts / f"{content_hash}.json"
    expected_hash = content_hash
    if failure == "malformed-hash":
        expected_hash = "not-a-hash"
    elif failure == "symlink":
        target = tmp_path / "bound-target"
        target.write_bytes(content)
        bound_path.symlink_to(target)
    else:
        bound_path.write_bytes(b"different")

    with pytest.raises(live.LiveConfigurationError):
        live.resolve_service_verification_receipt(runtime, expected_sha256=expected_hash)


def test_three_isolated_live_cells_are_dispatched_in_parallel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live

    cells = tuple(
        live.build_live_cell(
            plan=_plan(),
            cell_id=cell_id,
            output_root=tmp_path / "capsules",
            provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
            provider_project_id="oamb-providers-test-live",
            provider_evidence=_provider_evidence(),
            environment=_environment(),
            run_label="parallel-cells",
            observed_at=NOW,
            code_revision="source-tree-test",
        )
        for cell_id in ("hindsight-lme6", "mem0-lme6", "openviking-lme6")
    )
    context = multiprocessing.get_context("fork")
    started = context.Value("i", 0)
    all_started = context.Event()

    def execute(cell: live.LiveCell, **_kwargs: Any) -> object:
        with started.get_lock():
            started.value += 1
            if started.value == 3:
                all_started.set()
        if not all_started.wait(timeout=2):
            raise RuntimeError("cells did not overlap")
        return SimpleNamespace(capsule_root=cell.capsule_root)

    monkeypatch.setattr(live, "execute_live_cell", execute)

    completed = live.execute_live_cells(cells)

    assert started.value == 3
    assert tuple(item.cell_id for item in completed) == tuple(cell.cell.cell_id for cell in cells)
    assert tuple(item.capsule_root for item in completed) == tuple(
        cell.capsule_root for cell in cells
    )


@pytest.mark.parametrize("limits", (None, (3, 4), (8, 7)))
def test_resume_concurrency_reaches_scheduler_without_changing_saved_plan(
    tmp_path: Path, limits: tuple[int, int] | None
) -> None:
    from oamb.config.benchmark import ResumeConcurrency
    from oamb.live import build_live_cell

    configuration = load_lme6_configuration()
    plan = build_resolved_plan(
        replace(
            configuration,
            evaluation_controls=replace(
                configuration.evaluation_controls,
                max_parallel_history_ingestions_per_provider=6,
                max_parallel_questions_per_provider=6,
            ),
        )
    )
    built = build_live_cell(
        plan=plan,
        cell_id=plan.cells[0].cell_id,
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="resume-concurrency",
        observed_at=NOW,
        code_revision="source-tree-test",
        resume_concurrency=None if limits is None else ResumeConcurrency(*limits),
    )

    assert (
        built.control.max_parallel_history_ingestions,
        built.control.max_parallel_questions,
    ) == ((6, 6) if limits is None else limits)
    assert built.plan is plan
    assert plan.execution.max_parallel_history_ingestions_per_provider == 6
    assert plan.execution.max_parallel_questions_per_provider == 6


@pytest.mark.parametrize("limit", (1, 2, 3))
@pytest.mark.parametrize("stop", (None, "failure", "signal"))
def test_configured_provider_cap_releases_slots_and_drains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit: int, stop: str | None
) -> None:
    from oamb import live

    configuration = load_lme6_configuration()
    plan = build_resolved_plan(
        replace(
            configuration,
            evaluation_controls=replace(
                configuration.evaluation_controls,
                max_parallel_providers_per_dataset=limit,
                max_parallel_history_ingestions_per_provider=4,
                max_parallel_questions_per_provider=5,
            ),
        )
    )
    cells = tuple(
        live.build_live_cell(
            plan=plan,
            cell_id=spec.cell_id,
            output_root=tmp_path / "capsules",
            provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
            provider_project_id="oamb-providers-test-live",
            provider_evidence=_provider_evidence(),
            environment=_environment(),
            run_label="limited-cells",
            observed_at=NOW,
            code_revision="source-tree-test",
        )
        for spec in plan.cells
    )
    context = multiprocessing.get_context("fork")
    active = context.Value("i", 0)
    peak = context.Value("i", 0)
    started = context.Value("i", 0)
    first_wave = context.Event()
    third_started = context.Event()

    def execute(cell: live.LiveCell, *, stop_event: Any) -> object:
        assert cell.control.max_parallel_history_ingestions == 4
        assert cell.control.max_parallel_questions == 5
        with active.get_lock():
            active.value += 1
            peak.value = max(peak.value, active.value)
            started.value += 1
            if started.value == limit:
                first_wave.set()
        if cell.cell.provider_id == "openviking":
            third_started.set()
        try:
            assert first_wave.wait(timeout=3), "configured provider slots did not overlap"
            if stop is not None:
                if cell.cell.provider_id == "hindsight":
                    if stop == "failure":
                        raise RuntimeError("planted limited-provider failure")
                    os.kill(os.getppid(), signal.SIGTERM)
                assert stop_event.wait(timeout=3), "stop did not reach admitted peers"
            elif limit == 2 and cell.cell.provider_id == "hindsight":
                # Mem0 finishes first: its slot must start OpenViking without
                # waiting for this earlier cell to finish.
                assert third_started.wait(timeout=3), "completed slot was not replenished"
            return SimpleNamespace(capsule_root=cell.capsule_root)
        finally:
            with active.get_lock():
                active.value -= 1

    monkeypatch.setattr(live, "execute_live_cell", execute)

    if stop == "failure" or (stop == "signal" and limit < 3):
        with pytest.raises(live.LiveCellExecutionError) as failure:
            live.execute_live_cells(cells)
        assert started.value == limit
        assert tuple(item.status for item in failure.value.outcomes)[limit:] == ("not_started",) * (
            3 - limit
        )
    else:
        completed = live.execute_live_cells(cells)
        assert tuple(item.cell_id for item in completed) == tuple(
            spec.cell_id for spec in plan.cells
        )
        assert started.value == 3
    assert peak.value == limit
    assert active.value == 0


@pytest.mark.parametrize("pointer_name", ("active-operation", "active-provider-attempt"))
def test_selected_cell_build_rejects_active_lifecycle_ownership(
    tmp_path: Path,
    pointer_name: str,
) -> None:
    from oamb import live

    runtime = (tmp_path / "provider-runtime").resolve()
    lifecycle_domain = runtime / "lifecycle-domains" / "hindsight"
    lifecycle_domain.mkdir(parents=True)
    (lifecycle_domain / pointer_name).write_text("active\n", encoding="utf-8")

    with pytest.raises(live.LiveConfigurationError, match="lifecycle domain is active"):
        live.build_live_cell(
            plan=_lme60_plan(),
            cell_id="hindsight-lme60",
            output_root=tmp_path / "selected-capsules",
            provider_runtime_directory=runtime,
            provider_project_id="oamb-providers-test-live",
            provider_evidence=_provider_evidence(),
            environment=_environment(),
            run_label="selected-active-owner",
            observed_at=NOW,
            code_revision="source-tree-test",
            requested_case_manifest_entry_ids=(canonical_sha256(["remaining-case"]),),
        )


@pytest.mark.parametrize("stop_signal", (signal.SIGTERM, signal.SIGHUP))
def test_parallel_live_cells_drain_after_root_stop_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stop_signal: signal.Signals,
) -> None:
    from oamb import live

    cells = tuple(
        live.build_live_cell(
            plan=_plan(),
            cell_id=cell_id,
            output_root=tmp_path / "capsules",
            provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
            provider_project_id="oamb-providers-test-live",
            provider_evidence=_provider_evidence(),
            environment=_environment(),
            run_label="root-sigterm",
            observed_at=NOW,
            code_revision="source-tree-test",
        )
        for cell_id in ("hindsight-lme6", "mem0-lme6", "openviking-lme6")
    )
    context = multiprocessing.get_context("fork")
    started = context.Value("i", 0)
    all_started = context.Event()
    drained = context.Value("i", 0)

    def execute(cell: live.LiveCell, *, stop_event: Any | None = None) -> object:
        with started.get_lock():
            started.value += 1
            if started.value == len(cells):
                all_started.set()
        deadline = time.monotonic() + 3
        while stop_event is None or not stop_event.is_set():
            if time.monotonic() >= deadline or os.getppid() == 1:
                raise RuntimeError("root stop signal was not forwarded")
            time.sleep(0.01)
        with drained.get_lock():
            drained.value += 1
        return SimpleNamespace(capsule_root=cell.capsule_root)

    monkeypatch.setattr(live, "execute_live_cell", execute)

    def supervise() -> None:
        os.setsid()
        completed = live.execute_live_cells(cells)
        assert len(completed) == len(cells)

    supervisor = context.Process(target=supervise)
    supervisor.start()
    assert all_started.wait(timeout=2), "parallel cells did not reach the planted barrier"
    assert supervisor.pid is not None
    os.killpg(supervisor.pid, stop_signal)
    supervisor.join(timeout=5)
    if supervisor.is_alive():
        supervisor.terminate()
        supervisor.join(timeout=1)

    assert supervisor.exitcode == 0
    assert drained.value == len(cells)


def test_parallel_live_cell_failure_stops_and_drains_peer_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live

    cells = tuple(
        live.build_live_cell(
            plan=_plan(),
            cell_id=cell_id,
            output_root=tmp_path / "capsules",
            provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
            provider_project_id="oamb-providers-test-live",
            provider_evidence=_provider_evidence(),
            environment=_environment(),
            run_label="peer-terminal-failure",
            observed_at=NOW,
            code_revision="source-tree-test",
        )
        for cell_id in ("hindsight-lme6", "mem0-lme6", "openviking-lme6")
    )
    context = multiprocessing.get_context("fork")
    started = context.Value("i", 0)
    all_started = context.Event()
    drained = context.Value("i", 0)

    def execute(cell: live.LiveCell, *, stop_event: Any | None = None) -> object:
        with started.get_lock():
            started.value += 1
            if started.value == len(cells):
                all_started.set()
        if not all_started.wait(timeout=2):
            raise RuntimeError("parallel cells did not reach the planted barrier")
        if cell.cell.provider_id == "hindsight":
            raise RuntimeError("planted terminal cell failure")
        if stop_event is None or not stop_event.wait(timeout=2):
            raise RuntimeError("terminal peer failure did not request a shared stop")
        with drained.get_lock():
            drained.value += 1
        return SimpleNamespace(capsule_root=cell.capsule_root)

    monkeypatch.setattr(live, "execute_live_cell", execute)

    with pytest.raises(
        live.LiveCellExecutionError, match="planted terminal cell failure"
    ) as failure:
        live.execute_live_cells(cells)

    assert drained.value == 2
    assert tuple(outcome.status for outcome in failure.value.outcomes) == (
        "failed",
        "completed",
        "completed",
    )


def test_single_live_cell_rejects_mismatched_completion_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live

    cell = live.build_live_cell(
        plan=_plan(),
        cell_id="hindsight-lme6",
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="mismatched-single-cell",
        observed_at=NOW,
        code_revision="source-tree-test",
    )
    monkeypatch.setattr(
        live,
        "execute_live_cell",
        lambda _cell, **_kwargs: SimpleNamespace(capsule_root=tmp_path / "wrong"),
    )

    with pytest.raises(live.LiveCellExecutionError, match="mismatched completion identity"):
        live.execute_live_cells((cell,))


def test_partial_cell_admission_failure_settles_the_started_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live

    cells = tuple(
        live.build_live_cell(
            plan=_plan(),
            cell_id=cell_id,
            output_root=tmp_path / "capsules",
            provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
            provider_project_id="oamb-providers-test-live",
            provider_evidence=_provider_evidence(),
            environment=_environment(),
            run_label="partial-admission",
            observed_at=NOW,
            code_revision="source-tree-test",
        )
        for cell_id in ("hindsight-lme6", "mem0-lme6", "openviking-lme6")
    )

    class FakeConnection:
        def __init__(self) -> None:
            self.message: tuple[str, str, str] | None = None
            self.closed = False

        def recv(self) -> object:
            assert self.message is not None
            return self.message

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self, index: int, cell: live.LiveCell, receiver: FakeConnection) -> None:
            self.index = index
            self.cell = cell
            self.receiver = receiver
            self.pid: int | None = None
            self.exitcode: int | None = None
            self.started = False
            self.joined = False
            self.closed = False

        def start(self) -> None:
            if self.index == 1:
                raise RuntimeError("planted start failure")
            self.started = True
            self.pid = 100 + self.index
            self.exitcode = 0
            self.receiver.message = (
                "completed",
                self.cell.cell.cell_id,
                str(self.cell.capsule_root),
            )

        def join(self) -> None:
            self.joined = True

        def close(self) -> None:
            self.closed = True

    class FakeContext:
        def __init__(self) -> None:
            self.pending_receiver: FakeConnection | None = None
            self.processes: list[FakeProcess] = []

        def Event(self) -> Any:
            return SimpleNamespace(set=lambda: None, is_set=lambda: False)

        def Pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is False
            receiver = FakeConnection()
            self.pending_receiver = receiver
            return receiver, FakeConnection()

        def Process(self, *, args: tuple[Any, ...], **_kwargs: Any) -> FakeProcess:
            assert self.pending_receiver is not None
            process = FakeProcess(len(self.processes), args[0], self.pending_receiver)
            self.processes.append(process)
            self.pending_receiver = None
            return process

    context = FakeContext()
    monkeypatch.setattr(multiprocessing, "get_context", lambda _method: context)

    with pytest.raises(
        live.LiveCellExecutionError,
        match="planted start failure",
    ) as caught:
        live.execute_live_cells(cells)

    assert tuple(outcome.status for outcome in caught.value.outcomes) == (
        "completed",
        "failed",
        "not_started",
    )
    assert caught.value.outcomes[0].capsule_root == cells[0].capsule_root

    assert len(context.processes) == 2
    assert context.processes[0].started is True
    assert context.processes[0].joined is True
    assert context.processes[0].closed is True
    assert context.processes[1].started is False
    assert context.processes[1].closed is True
