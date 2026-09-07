from __future__ import annotations

import asyncio
import hashlib
import json
import multiprocessing
import os
import signal
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from typer.testing import CliRunner

import oamb.artifacts.composition as composition_module
from oamb.artifacts.atomic import read_regular_file
from oamb.artifacts.composition import CapsuleCompositionError, compose_capsules
from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.source_root import validate_source_root
from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import ResolvedPlan, build_resolved_plan
from oamb.contracts.evidence import (
    CapsuleCompositionPartBinding,
    CapsuleManifest,
    capsule_composition_part_binding_hash,
)
from oamb.contracts.ids import (
    canonical_json_bytes,
    canonical_sha256,
    ingestion_occurrence_id,
)
from oamb.contracts.ports import (
    ArtifactStorePort,
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionRequest,
    ModelClientPort,
    NativeEvidenceBatch,
    RetrievalRequest,
    ScopeAllocationRequest,
    ScopeReceipt,
)
from oamb.contracts.specifications import (
    INFRASTRUCTURE_RETRY_POLICY_HASH,
    CasePartitionSpec,
    run_preflight_record_hash,
)
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.comparison_project import (
    ValidatedCellRoot,
    build_comparison_project,
)
from oamb.runtime.case_partition import build_case_partition_spec
from oamb.runtime.native_run import run_native_vertical_slice
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY
from tests.e2e.test_native_fixture_vertical_slice import (
    _memory_factory,
    _NativeFixtureWorkload,
    _RecordedNativeMemory,
    _RecordedNativeModel,
)
from tests.unit.test_openai_compatible_model_client import _client
from tests.unit.test_t10_native_run_control import _control


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _set_partition_run_state(root: Path, state: str) -> None:
    manifest_path = root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(manifest_path.read_bytes())
    run_entry = next(
        entry for entry in manifest.source_entries if entry.record_kind == "run_record"
    )
    run_path = root / run_entry.relative_path
    run = json.loads(run_path.read_bytes())
    run["state"] = state
    run_path.write_bytes(canonical_json_bytes(run))
    source_entries = tuple(
        entry.model_copy(
            update={"sha256": hashlib.sha256((root / entry.relative_path).read_bytes()).hexdigest()}
        )
        for entry in manifest.source_entries
    )
    source_manifest_hash = canonical_sha256(
        [
            "oamb-source-manifest-v1",
            tuple(entry.model_dump(mode="python") for entry in source_entries),
        ]
    )
    manifest_path.write_bytes(
        canonical_json_bytes(
            manifest.model_copy(
                update={
                    "capsule_id": canonical_sha256(
                        [
                            "oamb-capsule-v1",
                            manifest.run_id,
                            manifest.run_spec_hash,
                            source_manifest_hash,
                        ]
                    ),
                    "source_entries": source_entries,
                    "source_manifest_hash": source_manifest_hash,
                }
            )
        )
    )


def _run_part(tmp_path: Path, run_id: str, plan_index: int) -> Path:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    selected_plan = manifest.ingestion_plans[plan_index]
    partition = build_case_partition_spec(
        run_id=run_id,
        resolved_plan_hash=canonical_sha256(["composition-plan"]),
        cell_spec_hash=canonical_sha256(["composition-cell"]),
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=(selected_plan.ordered_case_manifest_entry_ids[0],),
        budget_policy_hash=canonical_sha256(["composition-budget-policy"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    completed = run_native_vertical_slice(
        output_root=tmp_path / "parts",
        run_id=run_id,
        adapter_profile_id="recorded-native-fixture-v1",
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
        partition=partition,
    )
    return completed.capsule_root


def _comparison_plan(workload: _NativeFixtureWorkload) -> ResolvedPlan:
    base = build_resolved_plan(load_benchmark_configuration(Path("configs/benchmark.yml")))
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    source_sha256 = dataset.source_files[0].sha256
    resolved_dataset = replace(
        base.dataset,
        dataset_id=dataset.dataset_id,
        workload_id=manifest.workload_id,
        selection="fixture-4",
        revision=dataset.revision,
        source_sha256=source_sha256,
        case_manifest_hash=manifest.manifest_hash,
    )
    cells = tuple(
        replace(
            base.cells[index],
            cell_spec_hash=canonical_sha256(["composition-comparison-cell", index]),
            cell_id=f"composition-cell-{index + 1}",
            provider_id="fake-memory",
            adapter_profile_id="recorded-native-fixture-v1",
            dataset_id=dataset.dataset_id,
            workload_id=manifest.workload_id,
            selection="fixture-4",
            source_sha256=source_sha256,
            case_manifest_hash=manifest.manifest_hash,
        )
        for index in range(2)
    )
    return replace(
        base,
        resolved_plan_hash=canonical_sha256(["resolved-plan"]),
        comparison_id=canonical_sha256(["composition-comparison"]),
        dataset=resolved_dataset,
        cells=cells,
    )


def _run_controlled(
    tmp_path: Path,
    *,
    plan: ResolvedPlan,
    cell_index: int,
    run_id: str,
    plan_index: int | None,
    model_factory: Callable[[ArtifactStorePort], ModelClientPort] = _RecordedNativeModel,
    code_revision: str = "fixture-revision",
    model_connection_identity: str | None = None,
    model_name: str | None = None,
    model_endpoint_reference: str | None = None,
) -> Path:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    cell = plan.cells[cell_index]
    control = _control(
        run_id=run_id,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=manifest.manifest_hash,
        workload_id=manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id=cell.adapter_profile_id,
        adapter_profile_hash=cell.cell_spec_hash,
        answer_role_binding_id="recorded-answer-v1",
        provider_runtime_directory=(tmp_path / f"provider-{run_id}").resolve(),
        code_revision=code_revision,
    )
    if any(
        value is not None
        for value in (model_connection_identity, model_name, model_endpoint_reference)
    ):
        role_bindings = []
        for binding in control.role_bindings:
            updates = {}
            if model_connection_identity is not None:
                updates["redacted_endpoint_fingerprint"] = canonical_sha256(
                    ["model-connection", model_connection_identity, binding.binding_id]
                )
            if model_name is not None:
                updates["model"] = model_name
                updates["configuration_fingerprint"] = canonical_sha256(
                    ["model-configuration", model_name, binding.binding_id]
                )
            if model_endpoint_reference is not None:
                updates["endpoint_reference"] = model_endpoint_reference
            role_bindings.append(binding.model_copy(update=updates))
        frozen_role_bindings = tuple(role_bindings)
        run_spec = control.run_spec
        if model_connection_identity is not None:
            run_spec = run_spec.model_copy(
                update={
                    "environment_hash": canonical_sha256(["environment", model_connection_identity])
                }
            )
        preflight_fields = control.preflight_record.model_dump(
            mode="python",
            exclude={"preflight_record_hash"},
        )
        preflight_fields.update(
            {
                "run_spec_hash": canonical_sha256(run_spec),
                "redacted_endpoint_fingerprints": tuple(
                    binding.redacted_endpoint_fingerprint for binding in frozen_role_bindings
                ),
            }
        )
        preflight = type(control.preflight_record).model_validate(
            {
                "preflight_record_hash": run_preflight_record_hash(preflight_fields),
                **preflight_fields,
            }
        )
        control = replace(
            control,
            run_spec=run_spec,
            preflight_record=preflight,
            role_bindings=frozen_role_bindings,
        )
    partition = None
    if plan_index is not None:
        selected_plan = manifest.ingestion_plans[plan_index]
        partition = build_case_partition_spec(
            run_id=run_id,
            resolved_plan_hash=plan.resolved_plan_hash,
            cell_spec_hash=cell.cell_spec_hash,
            dataset_manifest_hash=dataset.manifest_hash,
            case_manifest=manifest,
            case_plans=workload.iter_case_plans(manifest),
            requested_case_manifest_entry_ids=(selected_plan.ordered_case_manifest_entry_ids[0],),
            budget_policy_hash=cell.authorization_hash,
            retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
        )
    completed = run_native_vertical_slice(
        output_root=tmp_path / "controlled",
        run_id=run_id,
        adapter_profile_id=cell.adapter_profile_id,
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=model_factory,
        answer_role_binding_id="recorded-answer-v1",
        control=control,
        partition=partition,
    )
    return completed.capsule_root


def test_composes_complementary_parts_without_changing_sources(tmp_path: Path) -> None:
    first = _run_part(tmp_path, "composition-part-a", 0)
    second = _run_part(tmp_path, "composition-part-b", 1)
    source_hashes = (_tree_digest(first), _tree_digest(second))

    forward = compose_capsules((first, second), tmp_path / "forward")
    reverse = compose_capsules((second, first), tmp_path / "reverse")
    forward_recovery = composition_module.analyze_capsule_recovery((first, second))
    reverse_recovery = composition_module.analyze_capsule_recovery((second, first))

    assert validate_source_root(forward.capsule_root).disposition == (
        ValidationDisposition.VALIDATED
    )
    assert forward.manifest.capsule_id == reverse.manifest.capsule_id
    assert forward_recovery == reverse_recovery
    assert forward_recovery.remaining_ingestion_plan_ids == ()
    assert _tree_digest(forward.capsule_root) == _tree_digest(reverse.capsule_root)
    assert source_hashes == (_tree_digest(first), _tree_digest(second))
    assert len(forward.composition.ordered_contributions) == 2
    assert (
        sum(len(item.case_manifest_entry_ids) for item in forward.composition.ordered_contributions)
        == 4
    )


def test_recovery_target_is_the_selected_partition_not_the_full_manifest(
    tmp_path: Path,
) -> None:
    selected_part = _run_part(tmp_path, "selected-single-plan", 0)

    recovery = composition_module.analyze_capsule_recovery((selected_part,))

    assert len(recovery.reusable_ingestion_plan_ids) == 1
    assert recovery.quarantined_ingestion_plan_ids == ()
    assert recovery.remaining_ingestion_plan_ids == ()
    assert recovery.remaining_case_manifest_entry_ids == ()


def test_composition_rejects_part_drift_before_publishing_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _run_part(tmp_path, "composition-drift-part-a", 0)
    second = _run_part(tmp_path, "composition-drift-part-b", 1)
    manifest = CapsuleManifest.model_validate_json((first / "capsule-manifest.json").read_bytes())
    target_entry = manifest.source_entries[0]
    target_path = first / target_entry.relative_path
    original_read = read_regular_file
    reads = 0

    def mutate_before_publish(path: Path) -> bytes:
        nonlocal reads
        if path == target_path:
            reads += 1
            if reads == 2:
                path.write_bytes(path.read_bytes() + b"\n")
        return original_read(path)

    monkeypatch.setattr(composition_module, "read_regular_file", mutate_before_publish)
    output = tmp_path / "drifted-composition"

    with pytest.raises(CapsuleCompositionError, match="modified composition part"):
        compose_capsules((first, second), output)

    assert not output.exists()


def test_composition_rejects_output_nested_in_a_source_part(tmp_path: Path) -> None:
    first = _run_part(tmp_path, "composition-contained-part-a", 0)
    second = _run_part(tmp_path, "composition-contained-part-b", 1)
    source_digests = (_tree_digest(first), _tree_digest(second))
    output = first / "nested-composition"

    with pytest.raises(CapsuleCompositionError, match="overlaps a source part"):
        compose_capsules((first, second), output)

    assert not output.exists()
    assert source_digests == (_tree_digest(first), _tree_digest(second))
    assert validate_source_root(first).disposition == ValidationDisposition.VALIDATED
    assert validate_source_root(second).disposition == ValidationDisposition.VALIDATED


@pytest.mark.parametrize(
    "embedded_root",
    ("../../outside", "/tmp/outside", "source/parts/not-the-capsule"),
)
def test_composition_part_binding_rejects_noncanonical_embedded_root(
    tmp_path: Path,
    embedded_root: str,
) -> None:
    first = _run_part(tmp_path, "composition-embedded-root", 0)
    composed = compose_capsules(
        (first, _run_part(tmp_path, "composition-peer", 1)), tmp_path / "ok"
    )
    binding = composed.composition.ordered_parts[0]
    fields = {
        **binding.model_dump(mode="python", exclude={"part_binding_hash"}),
        "embedded_root": embedded_root,
    }

    with pytest.raises(ValueError, match="embedded root"):
        CapsuleCompositionPartBinding.model_validate(
            {
                "part_binding_hash": capsule_composition_part_binding_hash(fields),
                **fields,
            }
        )


def test_recovery_rejects_ordinary_failed_history_without_retry_allowance(tmp_path: Path) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    case_plans = workload.iter_case_plans(manifest)

    def partition(run_id: str, requested: tuple[str, ...]) -> CasePartitionSpec:
        return build_case_partition_spec(
            run_id=run_id,
            resolved_plan_hash=canonical_sha256(["aborted-composition-plan"]),
            cell_spec_hash=canonical_sha256(["aborted-composition-cell"]),
            dataset_manifest_hash=dataset.manifest_hash,
            case_manifest=manifest,
            case_plans=case_plans,
            requested_case_manifest_entry_ids=requested,
            budget_policy_hash=canonical_sha256(["aborted-composition-budget"]),
            retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
        )

    aborted_partition = partition(
        "aborted-part",
        tuple(case.case_manifest_entry_id for case in manifest.cases),
    )
    failed_plan_id = manifest.ingestion_plans[1].ingestion_plan_id

    class FailingSecondPlanMemory(_RecordedNativeMemory):
        def __init__(self, store: ArtifactStorePort) -> None:
            super().__init__(store)
            self.first_group_complete = asyncio.Event()
            self.first_group_retrievals = 0

        async def allocate_ingestion_scope(
            self,
            request: ScopeAllocationRequest,
        ) -> ScopeReceipt:
            if request.ingestion_plan_id == failed_plan_id:
                await self.first_group_complete.wait()
                raise RuntimeError("planted second-plan failure")
            return await super().allocate_ingestion_scope(request)

        async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
            result = await super().retrieve(request)
            self.first_group_retrievals += 1
            if self.first_group_retrievals == 3:
                self.first_group_complete.set()
            return result

    with pytest.raises(RuntimeError, match="planted second-plan failure"):
        run_native_vertical_slice(
            output_root=tmp_path / "aborted-parts",
            run_id=aborted_partition.run_id,
            adapter_profile_id="recorded-native-fixture-v1",
            workload=workload,
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=lambda store, _plans: FailingSecondPlanMemory(store),
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
            partition=aborted_partition,
        )
    with pytest.raises(CapsuleCompositionError, match="no eligible pending retry allowance"):
        composition_module.analyze_capsule_recovery(
            (tmp_path / "aborted-parts" / aborted_partition.run_id,)
        )


@pytest.mark.parametrize(
    "stop_signal",
    (signal.SIGINT, signal.SIGTERM),
    ids=("sigint", "sigterm"),
)
def test_process_signal_drains_and_seals_completed_groups_for_fresh_scope_recovery(
    tmp_path: Path,
    stop_signal: signal.Signals,
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    case_plans = workload.iter_case_plans(manifest)
    run_id = "sigint-recovery-part"
    partition = build_case_partition_spec(
        run_id=run_id,
        resolved_plan_hash=canonical_sha256(["sigint-recovery-plan"]),
        cell_spec_hash=canonical_sha256(["sigint-recovery-cell"]),
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=case_plans,
        requested_case_manifest_entry_ids=tuple(
            case.case_manifest_entry_id for case in manifest.cases
        ),
        budget_policy_hash=canonical_sha256(["sigint-recovery-budget"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    second_plan_id = manifest.ingestion_plans[1].ingestion_plan_id
    started = tmp_path / "second-plan-started"
    release = tmp_path / "release-second-plan"
    outcome = tmp_path / "supervisor-outcome"

    def run_until_interrupted() -> None:
        class DrainableMemory(_RecordedNativeMemory):
            def __init__(self, store: ArtifactStorePort) -> None:
                super().__init__(store)
                self.first_group_complete = asyncio.Event()
                self.first_group_retrievals = 0

            async def allocate_ingestion_scope(
                self,
                request: ScopeAllocationRequest,
            ) -> ScopeReceipt:
                if request.ingestion_plan_id == second_plan_id:
                    await self.first_group_complete.wait()
                return await super().allocate_ingestion_scope(request)

            def plan_ingestion(
                self,
                request: IngestionRequest,
            ) -> tuple[IngestionDispatch, ...]:
                if self._plan_by_scope[request.scope.scope_id] != second_plan_id:
                    return super().plan_ingestion(request)
                return tuple(
                    IngestionDispatch(
                        dispatch_ordinal_1_indexed=index,
                        operation_kind="signal-test-ingest",
                        request_fingerprint=canonical_sha256(
                            ["signal-test-ingest", request.scope.scope_id, index]
                        ),
                        ordered_source_units=(source,),
                    )
                    for index, source in enumerate(request.ordered_source_units, start=1)
                )

            async def ingest(
                self,
                request: IngestionDispatchRequest,
            ) -> IngestionDispatchReceipt:
                if (
                    self._plan_by_scope[request.scope.scope_id] == second_plan_id
                    and request.dispatch.dispatch_ordinal_1_indexed == 1
                ):
                    started.write_text("started", encoding="utf-8")
                    while not release.exists():
                        await asyncio.sleep(0.01)
                return await super().ingest(request)

            async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
                result = await super().retrieve(request)
                self.first_group_retrievals += 1
                if self.first_group_retrievals == 3:
                    self.first_group_complete.set()
                return result

        try:
            run_native_vertical_slice(
                output_root=tmp_path / "sigint-parts",
                run_id=run_id,
                adapter_profile_id="recorded-native-fixture-v1",
                workload=workload,
                visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
                artifact_store_factory=ArtifactStore,
                memory_factory=lambda store, _plans: DrainableMemory(store),
                model_factory=_RecordedNativeModel,
                answer_role_binding_id="recorded-answer-v1",
                partition=partition,
            )
        except BaseException as exc:
            outcome.write_text(type(exc).__name__, encoding="utf-8")

    process = multiprocessing.get_context("fork").Process(target=run_until_interrupted)
    process.start()
    deadline = time.monotonic() + 5
    while not started.exists() and process.is_alive() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert started.exists(), "second plan never reached the planted in-flight barrier"
    assert process.pid is not None
    os.kill(process.pid, stop_signal)
    release.write_text("release", encoding="utf-8")
    process.join(timeout=5)
    if process.is_alive():
        process.terminate()
        process.join(timeout=1)
    assert not process.is_alive()
    assert outcome.read_text(encoding="utf-8") == "NativeRunInterrupted"

    capsule_root = tmp_path / "sigint-parts" / run_id
    assert validate_source_root(capsule_root).disposition == ValidationDisposition.VALIDATED
    with pytest.raises(CapsuleCompositionError, match="no eligible pending retry allowance"):
        composition_module.analyze_capsule_recovery((capsule_root,))
    interrupted_second_occurrence_id = ingestion_occurrence_id(
        run_id,
        "fake-memory",
        second_plan_id,
    )
    interrupted_second_ingests = tuple(
        document
        for path in (capsule_root / "source" / "attempts").glob("*.json")
        for document in (json.loads(path.read_bytes()),)
        if document["parent_id"] == interrupted_second_occurrence_id
        and document["stage"] == "memory_ingest"
    )
    assert tuple(document["outcome"] for document in interrupted_second_ingests) == ("succeeded",)


def test_recovery_run_derives_remaining_groups_before_loading_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb import cli, live
    from oamb.artifacts.composition import CapsuleRecoveryPlan
    from oamb.config import doctor

    plan = build_resolved_plan(load_benchmark_configuration(Path("configs/benchmark.yml")))
    selected_cell = plan.cells[0]
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    source_part = tmp_path / "aborted-part"
    remaining_case_id = canonical_sha256(["remaining-case"])
    call_order: list[str] = []
    captured: dict[str, object] = {}
    recovery_targets: list[object] = []

    def analyze(*_args: object, **_kwargs: object) -> CapsuleRecoveryPlan:
        call_order.append("analyze")
        recovery_targets.append(_kwargs.get("target"))
        return CapsuleRecoveryPlan(
            source_capsule_ids=(canonical_sha256(["source-capsule"]),),
            source_manifest_sha256s=(canonical_sha256(["source-manifest"]),),
            reusable_ingestion_plan_ids=(canonical_sha256(["reusable-plan"]),),
            quarantined_ingestion_plan_ids=(canonical_sha256(["quarantined-plan"]),),
            remaining_ingestion_plan_ids=(canonical_sha256(["remaining-plan"]),),
            remaining_case_manifest_entry_ids=(remaining_case_id,),
        )

    def load_environment(**_kwargs: object) -> dict[str, str]:
        call_order.append("environment")
        return {}

    def build_cell(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(composition_module, "analyze_capsule_recovery", analyze)
    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "validate_live_readiness_receipt", lambda **_kwargs: None)
    monkeypatch.setattr(live, "load_live_environment", load_environment)
    monkeypatch.setattr(
        live,
        "load_live_provider_evidence",
        lambda **_kwargs: ("provider-project", {selected_cell.provider_id: object()}),
    )
    monkeypatch.setattr(live, "build_live_cell", build_cell)
    full_target = object()

    def composition_target(_cell: object) -> object:
        call_order.append("target")
        return full_target

    def execute(_cells: object) -> tuple[live.LiveCellCompletion, ...]:
        call_order.append("execute")
        return (live.LiveCellCompletion(selected_cell.cell_id, tmp_path / "recovery-capsule"),)

    monkeypatch.setattr(
        live,
        "live_composition_target",
        composition_target,
    )
    monkeypatch.setattr(live, "execute_live_cells", execute)

    arguments = [
        "run",
        str(resolved_plan),
        "--output-root",
        str(tmp_path / "output"),
        "--cell",
        selected_cell.cell_id,
        "--recover-from",
        str(source_part),
        "--run-label",
        "recovery-1",
    ]
    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0, result.output
    assert call_order == ["analyze", "environment", "target", "analyze", "execute"]
    assert recovery_targets[1] is full_target
    assert captured["requested_case_manifest_entry_ids"] == (remaining_case_id,)
    assert "recovery reusable groups:" in result.output
    assert "recovery remaining groups:" in result.output


def test_infrastructure_blocked_partition_requires_terminal_retry_evidence(
    tmp_path: Path,
) -> None:
    root = _run_part(tmp_path, "blocked-without-retry-event", 0)
    _set_partition_run_state(root, "infrastructure_blocked")

    validation = validate_source_root(root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "infrastructure-retry-evidence-mismatch" in {issue.code for issue in validation.issues}


def test_composed_cell_uses_normal_comparison_and_report_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = _NativeFixtureWorkload()
    plan = _comparison_plan(workload)
    calls = multiprocessing.get_context("fork").Value("i", 0)
    rejection = canonical_json_bytes(
        {
            "error": {
                "origin": "model_supplier",
                "failure_kind": "rate_limited",
                "status": 429,
                "acceptance": "not_accepted",
                "provider_mutation": "none",
                "retryable": True,
                "internal_retry_count": 0,
            }
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        with calls.get_lock():
            calls.value += 1
            call_number = calls.value
        if call_number == 1:
            return httpx.Response(429, content=rejection)
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "alpha"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def no_wait(_seconds: int) -> None:
        return None

    monkeypatch.setattr(
        "oamb.runtime.native_run._infrastructure_retry_sleep",
        no_wait,
    )
    first = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="controlled-part-a",
        plan_index=0,
        model_factory=lambda store: _client(
            cast(Any, store),
            handler,
            binding_id="recorded-answer-v1",
        ),
    )
    second = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="controlled-part-b",
        plan_index=1,
    )
    peer = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=1,
        run_id="controlled-peer",
        plan_index=None,
    )
    _set_partition_run_state(first, "aborted")
    composed = compose_capsules((first, second), tmp_path / "composed-cell")
    composed_validation = validate_source_root(composed.capsule_root)
    peer_validation = validate_source_root(peer)

    built = build_comparison_project(
        plan,
        {
            plan.cells[0].cell_id: ValidatedCellRoot(
                root=composed.capsule_root,
                validation_result=composed_validation,
            ),
            plan.cells[1].cell_id: ValidatedCellRoot(
                root=peer,
                validation_result=peer_validation,
            ),
        },
        output_root=tmp_path / "comparison-report",
    )

    assert built.export_path.is_file()
    assert built.html_path.is_file()
    export = json.loads(built.export_path.read_bytes())
    composed_cell = next(
        cell for cell in export["cells"] if cell["cell_id"] == plan.cells[0].cell_id
    )
    assert export["coverage"]["provider_specific_result_count"] == 8
    assert composed_cell["accounting"]["infrastructure_retries"] == {
        "rejection_count": 0,
        "internal_retry_count": 0,
        "scheduled_retry_count": 0,
        "total_retry_count": 0,
        "backoff_seconds": 0,
        "measurement_coverage": "complete",
    }
    assert composed_cell["observed_time"]["indexing_ready"]["status"] == "measured"
    assert composed_cell["observed_time"]["indexing_ready"]["count"] == 2
    assert composed_cell["accounting"]["attempts"]["retry_count"] == 1
    assert composed_cell["accounting"]["attempts"]["failed_count"] == 1


@pytest.mark.parametrize("invalid_kind", ["missing", "duplicate", "overlap"])
def test_composition_rejects_non_exact_union_before_output(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    first = _run_part(tmp_path, "composition-invalid-a", 0)
    second = _run_part(tmp_path, "composition-invalid-b", 1)
    overlapping = _run_part(tmp_path, "composition-invalid-c", 0)
    parts = {
        "missing": (first,),
        "duplicate": (first, second, first),
        "overlap": (first, second, overlapping),
    }[invalid_kind]
    output = tmp_path / f"invalid-{invalid_kind}"

    with pytest.raises(CapsuleCompositionError, match=invalid_kind):
        compose_capsules(parts, output)

    assert not output.exists()


def test_composition_rejects_different_execution_configuration(tmp_path: Path) -> None:
    workload = _NativeFixtureWorkload()
    plan = _comparison_plan(workload)
    first = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="execution-config-part-a",
        plan_index=0,
        code_revision="revision-a",
    )
    second = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="execution-config-part-b",
        plan_index=1,
        code_revision="revision-b",
    )
    output = tmp_path / "incompatible-execution-configuration"

    with pytest.raises(CapsuleCompositionError, match="incompatible composition parts"):
        compose_capsules((first, second), output)

    assert not output.exists()


def test_composition_accepts_rotated_model_connection(tmp_path: Path) -> None:
    workload = _NativeFixtureWorkload()
    plan = _comparison_plan(workload)
    first = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="model-connection-part-a",
        plan_index=0,
        model_connection_identity="before-rotation",
    )
    second = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="model-connection-part-b",
        plan_index=1,
        model_connection_identity="after-rotation",
    )

    composed = compose_capsules((first, second), tmp_path / "rotated-model-connection")

    assert validate_source_root(composed.capsule_root).disposition == (
        ValidationDisposition.VALIDATED
    )


def test_composition_rejects_changed_model_configuration(tmp_path: Path) -> None:
    workload = _NativeFixtureWorkload()
    plan = _comparison_plan(workload)
    first = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="model-configuration-part-a",
        plan_index=0,
    )
    second = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="model-configuration-part-b",
        plan_index=1,
        model_name="different-model",
    )
    output = tmp_path / "changed-model-configuration"

    with pytest.raises(CapsuleCompositionError, match="incompatible composition parts"):
        compose_capsules((first, second), output)

    assert not output.exists()


def test_composition_rejects_changed_non_llm_endpoint(tmp_path: Path) -> None:
    workload = _NativeFixtureWorkload()
    plan = _comparison_plan(workload)
    first = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="embedding-connection-part-a",
        plan_index=0,
        model_connection_identity="embedding-before",
        model_endpoint_reference="OAMB_EMBEDDING_BASE_URL",
    )
    second = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="embedding-connection-part-b",
        plan_index=1,
        model_connection_identity="embedding-after",
        model_endpoint_reference="OAMB_EMBEDDING_BASE_URL",
    )
    output = tmp_path / "changed-embedding-connection"

    with pytest.raises(CapsuleCompositionError, match="incompatible composition parts"):
        compose_capsules((first, second), output)

    assert not output.exists()


def test_recovery_rejects_target_execution_configuration_before_remaining_work(
    tmp_path: Path,
) -> None:
    workload = _NativeFixtureWorkload()
    plan = _comparison_plan(workload)
    first = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="recovery-execution-config-a",
        plan_index=0,
        code_revision="revision-a",
    )
    incompatible = _run_controlled(
        tmp_path,
        plan=plan,
        cell_index=0,
        run_id="recovery-execution-config-b",
        plan_index=1,
        code_revision="revision-b",
    )
    incompatible_hash = composition_module.capsule_execution_configuration_hash(incompatible)
    cell = plan.cells[0]

    with pytest.raises(
        CapsuleCompositionError,
        match="requested execution configuration",
    ):
        composition_module.analyze_capsule_recovery(
            (first,),
            target=composition_module.CapsuleCompositionTarget(
                resolved_plan_hash=plan.resolved_plan_hash,
                cell_spec_hash=cell.cell_spec_hash,
                target_case_manifest_hash=cell.case_manifest_hash,
                budget_policy_hash=cell.authorization_hash,
                retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
                execution_configuration_hash=incompatible_hash,
            ),
        )
