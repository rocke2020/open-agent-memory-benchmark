from __future__ import annotations

import multiprocessing
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import ResolvedPlan, build_resolved_plan
from oamb.contracts.evidence import RunRecord
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    RunPreflightRecord,
    SourceEvidenceBinding,
    SourceEvidenceKind,
)
from oamb.contracts.states import ResumeDisposition, RunState
from oamb.runtime.source_records import seal_source_contract

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = REPOSITORY_ROOT / "configs" / "benchmark.yml"
NOW = datetime(2026, 8, 30, 10, 0, tzinfo=UTC)


def _plan() -> ResolvedPlan:
    return build_resolved_plan(load_benchmark_configuration(BENCHMARK_CONFIG))


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
    return {
        "OAMB_HINDSIGHT_BASE_URL": "http://127.0.0.1:64888",
        "OAMB_HINDSIGHT_LLM_BASE_URL": "https://model.example/v1",
        "OAMB_HINDSIGHT_LLM_API_KEY": "provider-model-key",
        "OAMB_MEM0_BASE_URL": "http://127.0.0.1:64889",
        "OAMB_MEM0_ADMIN_API_KEY": "mem0-key",
        "OAMB_MEM0_LLM_BASE_URL": "https://model.example/v1",
        "OAMB_MEM0_LLM_API_KEY": "provider-model-key",
        "OAMB_MEM0_INSPECTOR_BASE_URL": "http://127.0.0.1:64333",
        "OAMB_MEM0_INSPECTOR_API_KEY": "inspector-key",
        "OAMB_OPENVIKING_BASE_URL": "http://127.0.0.1:64930",
        "OAMB_OPENVIKING_USER_API_KEY": "openviking-key",
        "OAMB_OPENVIKING_VLM_BASE_URL": "https://model.example/v1",
        "OAMB_OPENVIKING_VLM_API_KEY": "provider-model-key",
        "OAMB_OPENVIKING_ACCOUNT_ID": "benchmark-account",
        "OAMB_OPENVIKING_ADMIN_USER_ID": "benchmark-user",
        "OAMB_DEEPSEEK_BASE_URL": "https://model.example/v1",
        "OAMB_DEEPSEEK_API_KEY": "answer-key",
        "OAMB_EMBEDDING_BASE_URL": "http://127.0.0.1:18000/v1",
    }


def _aborted_openviking_capsule(
    tmp_path: Path,
    *,
    plan: ResolvedPlan,
    environment: dict[str, str],
    provider_evidence: SourceEvidenceBinding,
) -> Path:
    from oamb.live import build_live_cell

    built = build_live_cell(
        plan=plan,
        cell_id="openviking-lme6",
        output_root=tmp_path / "base-capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=provider_evidence,
        environment=environment,
        run_label="continuation-base",
        observed_at=NOW,
        code_revision="source-tree-test",
    )
    store = ArtifactStore(built.capsule_root)
    run_spec = built.control.run_spec
    preflight = built.control.preflight_record
    budget = built.control.budget
    seal_source_contract(
        store,
        relative_path="source/specs/run-spec.json",
        record_id=run_spec.run_id,
        record=run_spec,
    )
    seal_source_contract(
        store,
        relative_path="source/specs/run-preflight.json",
        record_id=preflight.preflight_record_hash,
        record=preflight,
    )
    seal_source_contract(
        store,
        relative_path="source/specs/budget.json",
        record_id=budget.budget_id,
        record=budget,
    )
    for binding in built.control.role_bindings:
        seal_source_contract(
            store,
            relative_path=f"source/model-role-bindings/{binding.binding_id}.json",
            record_id=binding.binding_id,
            record=binding,
        )
    run = RunRecord(
        run_id=built.run_id,
        run_spec_hash=canonical_sha256(run_spec),
        state=RunState.ABORTED,
        resume_disposition=ResumeDisposition.NOT_APPLICABLE,
        started_at=NOW,
        ended_at=NOW,
        ingestion_occurrence_ids=(),
        case_occurrence_ids=(),
    )
    seal_source_contract(
        store,
        relative_path=f"source/run/{built.run_id}.json",
        record_id=built.run_id,
        record=run,
    )
    store.finalize_capsule(run_id=built.run_id, run_spec_hash=canonical_sha256(run_spec))
    return built.capsule_root


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
        environment=_environment(),
        run_label="fresh-rerun-1",
        observed_at=NOW,
        code_revision="source-tree-test",
    )

    assert built.cell.cell_id == cell_id
    assert built.control.run_spec.memory_system_id == memory_system_id
    assert built.control.preflight_record.adapter_profile_id == profile_id
    assert built.control.run_spec.case_manifest_hash == _plan().dataset.case_manifest_hash
    assert built.capsule_root == tmp_path / "capsules" / built.run_id
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
        == "vllm-metal"
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


def test_continuation_rejects_resolved_plan_drift_before_creating_target(
    tmp_path: Path,
) -> None:
    from oamb.live import LiveConfigurationError, build_live_continuation_cell

    base_plan = _plan()
    evidence = _provider_evidence()
    base = _aborted_openviking_capsule(
        tmp_path,
        plan=base_plan,
        environment=_environment(),
        provider_evidence=evidence,
    )
    mutated_config = tmp_path / "benchmark.yml"
    mutated_config.write_text(
        BENCHMARK_CONFIG.read_text(encoding="utf-8").replace(
            "max_parallel_questions_per_provider: 2",
            "max_parallel_questions_per_provider: 1",
            1,
        ),
        encoding="utf-8",
    )
    changed_plan = build_resolved_plan(load_benchmark_configuration(mutated_config))

    with pytest.raises(LiveConfigurationError, match="resolved plan"):
        build_live_continuation_cell(
            plan=changed_plan,
            cell_id="openviking-lme6",
            base_capsule_root=base,
            output_root=tmp_path / "continuation",
            provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
            provider_project_id="oamb-providers-test-live",
            provider_evidence=evidence,
            environment=_environment(),
        )
    assert not (tmp_path / "continuation").exists()


def test_continuation_accepts_the_exact_frozen_plan_and_runtime_binding(
    tmp_path: Path,
) -> None:
    from oamb.live import build_live_continuation_cell

    plan = _plan()
    evidence = _provider_evidence()
    environment = _environment()
    base = _aborted_openviking_capsule(
        tmp_path,
        plan=plan,
        environment=environment,
        provider_evidence=evidence,
    )

    continued = build_live_continuation_cell(
        plan=plan,
        cell_id="openviking-lme6",
        base_capsule_root=base,
        output_root=tmp_path / "continuation",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=evidence,
        environment=environment,
    )

    assert continued.run_id == base.name
    assert continued.capsule_root == tmp_path / "continuation" / base.name
    assert continued.continuation is not None
    assert continued.control.max_parallel_history_ingestions == 2
    assert continued.control.max_parallel_questions == 2


@pytest.mark.parametrize("drift", ["environment", "provider_evidence"])
def test_continuation_rejects_runtime_binding_drift_before_creating_target(
    tmp_path: Path,
    drift: str,
) -> None:
    from oamb.live import LiveConfigurationError, build_live_continuation_cell

    plan = _plan()
    evidence = _provider_evidence()
    environment = _environment()
    base = _aborted_openviking_capsule(
        tmp_path,
        plan=plan,
        environment=environment,
        provider_evidence=evidence,
    )
    current_environment = dict(environment)
    current_evidence = evidence
    if drift == "environment":
        current_environment["OAMB_OPENVIKING_BASE_URL"] = "http://127.0.0.1:64931"
    else:
        current_evidence = evidence.model_copy(
            update={
                "binding_id": canonical_sha256(["changed-provider-evidence"]),
                "source_root_hash": canonical_sha256(["changed-provider-root"]),
            }
        )

    with pytest.raises(LiveConfigurationError, match="continuation .* differs"):
        build_live_continuation_cell(
            plan=plan,
            cell_id="openviking-lme6",
            base_capsule_root=base,
            output_root=tmp_path / "continuation",
            provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
            provider_project_id="oamb-providers-test-live",
            provider_evidence=current_evidence,
            environment=current_environment,
        )
    assert not (tmp_path / "continuation").exists()


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
    monkeypatch.setattr(native_run, "run_native_vertical_slice", record_memory_factory)
    monkeypatch.setattr(live, "build_lme30_bundle", lambda _path: object())
    monkeypatch.setattr(live, "build_lme6_bundle", lambda _bundle: object())
    monkeypatch.setattr(live, "LongMemEvalWorkload", lambda _bundle: object())
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
    if cell_id == "openviking-lme6":
        assert captured["maximum_task_polls"] == 9_000


def test_live_model_factories_receive_the_resolved_model_call_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live
    from oamb.model_clients import openai_compatible
    from oamb.runtime import native_run

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
    monkeypatch.setattr(live, "build_lme30_bundle", lambda _path: object())
    monkeypatch.setattr(live, "build_lme6_bundle", lambda _bundle: object())
    monkeypatch.setattr(live, "LongMemEvalWorkload", lambda _bundle: object())
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

    assert [item["read_timeout_seconds"] for item in captured] == [600.0, 600.0]
    assert [item["total_timeout_seconds"] for item in captured] == [600.0, 600.0]


def test_unknown_live_case_partition_fails_before_native_runner_or_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import live
    from oamb.runtime import native_run
    from oamb.runtime.case_partition import CasePartitionSelectionError
    from oamb.workloads.fake import GeneratedFakeWorkload

    def forbidden_runner(**_kwargs: object) -> object:
        raise AssertionError("native runner must not receive an invalid case partition")

    monkeypatch.setattr(native_run, "run_native_vertical_slice", forbidden_runner)
    monkeypatch.setattr(live, "build_lme30_bundle", lambda _path: object())
    monkeypatch.setattr(live, "build_lme6_bundle", lambda _bundle: object())
    monkeypatch.setattr(live, "LongMemEvalWorkload", lambda _bundle: GeneratedFakeWorkload())
    built = live.build_live_cell(
        plan=_plan(),
        cell_id="hindsight-lme6",
        output_root=tmp_path / "capsules",
        provider_runtime_directory=(tmp_path / "provider-runtime").resolve(),
        provider_project_id="oamb-providers-test-live",
        provider_evidence=_provider_evidence(),
        environment=_environment(),
        run_label="invalid-case-selection",
        observed_at=NOW,
        code_revision="source-tree-test",
        requested_case_manifest_entry_ids=(canonical_sha256(["unknown-live-case"]),),
    )

    with pytest.raises(CasePartitionSelectionError, match="unknown"):
        live.execute_live_cell(built)


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
        "answer": 600,
        "judge": 600,
    }

    reservation_ordinal = 0
    for stage, count in dispatch_counts.items():
        route = built.control.require_budget_route(stage=stage)
        maximum_output_tokens = 8192 if stage == "answer" else 10 if stage == "judge" else 0
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

    def execute(cell: live.LiveCell) -> object:
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

    with pytest.raises(live.LiveCellExecutionError, match="planted start failure"):
        live.execute_live_cells(cells)

    assert len(context.processes) == 2
    assert context.processes[0].started is True
    assert context.processes[0].joined is True
    assert context.processes[0].closed is True
    assert context.processes[1].started is False
    assert context.processes[1].closed is True
