from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from oamb.artifacts.validation.source_root import validate_source_root
from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import ResolvedPlan, build_resolved_plan
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    INFRASTRUCTURE_RETRY_POLICY_HASH,
    BudgetSpecV4,
    RunPreflightRecord,
    budget_spec_v4_hash,
    run_preflight_record_hash,
)
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.comparison_project import ValidatedCellRoot, build_comparison_project
from oamb.runtime.case_partition import build_case_partition_spec
from oamb.workloads.longmemeval import LME_JUDGE_PROMPT_PACK_ID
from tests.e2e.test_history_rebuild_vertical_slice import _run, _two_source_workload
from tests.e2e.test_native_recorded_exact_profile import (
    ANSWER_ROLE_ID,
    RUNTIME_BINDING_HASH,
)
from tests.unit.test_t10_native_run_control import _control


def _comparison_plan() -> ResolvedPlan:
    workload = _two_source_workload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    base = build_resolved_plan(load_benchmark_configuration(Path("configs/benchmark.yml")))
    dataset_spec = replace(
        base.dataset,
        dataset_id=dataset.dataset_id,
        workload_id=manifest.workload_id,
        selection="history-rebuild-fixture-1",
        revision=dataset.revision,
        source_sha256=dataset.source_files[0].sha256,
        case_manifest_hash=manifest.manifest_hash,
    )
    cells = tuple(
        replace(
            base.cells[0],
            ordinal_1_indexed=index + 1,
            cell_id=f"recorded-hindsight-{label}",
            cell_spec_hash=canonical_sha256(["history-rebuild-comparison-cell", label]),
            dataset_id=dataset.dataset_id,
            workload_id=manifest.workload_id,
            selection=dataset_spec.selection,
            source_sha256=dataset.source_files[0].sha256,
            case_manifest_hash=manifest.manifest_hash,
            execution_hash=canonical_sha256(["history-rebuild-execution", label]),
        )
        for index, label in enumerate(("rebuilt", "clean"))
    )
    return replace(
        base,
        resolved_plan_hash=canonical_sha256(["resolved-plan"]),
        comparison_id=canonical_sha256(["history-rebuild-comparison"]),
        dataset=dataset_spec,
        cells=cells,
    )


def _controlled_cell(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    plan: ResolvedPlan,
    cell_index: int,
    fail_first_history: bool,
    recovery_parts: tuple[Path, ...] = (),
    stop_during_backoff: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    workload = _two_source_workload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    cases = workload.iter_case_plans(manifest)
    cell = plan.cells[cell_index]
    initial_run_id = f"history-comparison-{cell_index + 1}"
    run_id = initial_run_id + ("-resumed" if recovery_parts else "")
    control = _control(
        run_id=run_id,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=manifest.manifest_hash,
        workload_id=manifest.workload_id,
        memory_system_id="hindsight",
        runtime_binding_hash=RUNTIME_BINDING_HASH,
        adapter_profile_id=cell.adapter_profile_id,
        adapter_profile_hash=cell.cell_spec_hash,
        answer_role_binding_id=ANSWER_ROLE_ID,
        judge_role_binding_id=LME_JUDGE_PROMPT_PACK_ID,
        provider_runtime_directory=(tmp_path / f"provider-{run_id}").resolve(),
    )
    role_ceilings = tuple(
        item.model_copy(update={"max_output_tokens": 100_000})
        for item in control.budget.role_ceilings
    )
    budget_fields = control.budget.model_dump(
        mode="python", exclude={"schema_name", "schema_version", "budget_hash"}
    )
    budget_fields["max_output_tokens"] = 1_000_000
    budget_fields["role_ceilings"] = role_ceilings
    budget = BudgetSpecV4.model_validate(
        {"budget_hash": budget_spec_v4_hash(budget_fields), **budget_fields}
    )
    preflight_fields = control.preflight_record.model_dump(
        mode="python", exclude={"schema_name", "schema_version", "preflight_record_hash"}
    )
    preflight_fields["budget_hash"] = canonical_sha256(budget)
    preflight = RunPreflightRecord.model_validate(
        {
            "preflight_record_hash": run_preflight_record_hash(preflight_fields),
            **preflight_fields,
        }
    )
    control = replace(control, budget=budget, preflight_record=preflight)
    partition = build_case_partition_spec(
        run_id=run_id,
        resolved_plan_hash=plan.resolved_plan_hash,
        cell_spec_hash=cell.cell_spec_hash,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=cases,
        requested_case_manifest_entry_ids=tuple(item.case_manifest_entry_id for item in cases),
        budget_policy_hash=cell.authorization_hash,
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    completed, service, _waits = _run(
        tmp_path / f"cell-{cell_index}",
        monkeypatch,
        run_id=run_id,
        fail_first_history=fail_first_history,
        control=control,
        partition=partition,
        occurrence_run_id=initial_run_id,
        recovery_parts=recovery_parts,
        stop_during_backoff=stop_during_backoff,
    )
    return completed.capsule_root, service.occurrences


def test_partial_and_clean_recorded_hindsight_cells_publish_normal_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorded HTTP/model fixtures prove the public consumer path without provider calls."""

    plan = _comparison_plan()
    rebuilt_root, rebuilt_occurrences = _controlled_cell(
        tmp_path,
        monkeypatch,
        plan=plan,
        cell_index=0,
        fail_first_history=True,
    )
    clean_root, clean_occurrences = _controlled_cell(
        tmp_path, monkeypatch, plan=plan, cell_index=1, fail_first_history=False
    )
    rebuilt_validation = validate_source_root(rebuilt_root)
    clean_validation = validate_source_root(clean_root)
    assert (
        rebuilt_validation.disposition
        == clean_validation.disposition
        == (ValidationDisposition.VALIDATED)
    )

    built = build_comparison_project(
        plan,
        {
            plan.cells[0].cell_id: ValidatedCellRoot(rebuilt_root, rebuilt_validation),
            plan.cells[1].cell_id: ValidatedCellRoot(clean_root, clean_validation),
        },
        output_root=tmp_path / "comparison",
    )

    assert built.html_path.is_file()
    export = json.loads(built.export_path.read_bytes())
    cells = {item["cell_id"]: item for item in export["cells"]}
    rebuilt = cells[plan.cells[0].cell_id]
    clean = cells[plan.cells[1].cell_id]
    rebuilt_plans = tuple(
        json.loads(path.read_bytes())
        for path in (rebuilt_root / "source/ingestion-plans").glob("*.json")
    )
    clean_plans = tuple(
        json.loads(path.read_bytes())
        for path in (clean_root / "source/ingestion-plans").glob("*.json")
    )
    assert tuple(item["ingestion_occurrence_id"] for item in rebuilt_plans) == (
        rebuilt_occurrences[0],
    )
    assert tuple(item["ingestion_occurrence_id"] for item in clean_plans) == (clean_occurrences[0],)
    assert rebuilt["case_count"] == clean["case_count"] == 1
    assert len(rebuilt["results"]) == len(clean["results"]) == 1
    assert (
        rebuilt["accounting"]["attempts"]["attempt_count"]
        > clean["accounting"]["attempts"]["attempt_count"]
    )
    assert rebuilt["accounting"]["attempts"]["failed_count"] == 3
    assert clean["accounting"]["attempts"]["failed_count"] == 0
    rebuilt_indexing = rebuilt["accounting"]["tokens"]["indexing"]
    clean_indexing = clean["accounting"]["tokens"]["indexing"]
    assert rebuilt_indexing["supplier_usage_coverage"]["record_count"] == 4
    assert clean_indexing["supplier_usage_coverage"]["record_count"] == 2
    assert rebuilt_indexing["totals"]["input_tokens"]["value"] == 11
    assert clean_indexing["totals"]["input_tokens"]["value"] == 22
    assert export["coverage"]["provider_specific_result_count"] == 2
