from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from oamb import live
from oamb.cli import app
from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import ResolvedPlan, build_resolved_plan
from oamb.contracts.ids import canonical_sha256
from tests.benchmark_configuration import load_lme6_configuration

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = REPOSITORY_ROOT / "configs/benchmark.yml"


def _plan() -> ResolvedPlan:
    return build_resolved_plan(load_benchmark_configuration(BENCHMARK_CONFIG))


def _lme6_plan() -> ResolvedPlan:
    return build_resolved_plan(load_lme6_configuration())


def test_continuation_stops_before_environment_or_provider_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config.doctor import resolved_plan_bytes

    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_bytes(resolved_plan_bytes(_lme6_plan()))
    entered_environment = False

    def forbidden_environment(**_kwargs: object) -> dict[str, str]:
        nonlocal entered_environment
        entered_environment = True
        raise AssertionError("unsupported continuation reached runtime preparation")

    monkeypatch.setattr(live, "load_live_environment", forbidden_environment)
    output = tmp_path / "successor"
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--cell",
            "openviking-lme6",
            "--continue-from",
            str(tmp_path / "aborted-base"),
            "--output-root",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "no-mutation" in result.output
    assert not entered_environment
    assert not output.exists()


def test_root_environment_template_contains_model_placeholders_without_credentials() -> None:
    entries = {}
    for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            entries[key] = value

    assert entries["LLM_URL_TYPE"] == "openai_chat"
    assert entries["LLM_BASE_URL"] == "change-me"
    assert entries["LLM_API_KEY"] == "change-me"
    assert "DEEPSEEK_BASE_URL" not in entries
    assert "DEEPSEEK_API_KEY" not in entries
    assert not any(value.startswith("sk-") for value in entries.values())


def test_live_question_run_writes_machine_readable_result_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    capsule_root = tmp_path / "capsules" / "hindsight-question"
    capsule_root.mkdir(parents=True)
    result_map = tmp_path / "hindsight-result.json"
    bound_service_receipt = tmp_path / "bound-service-receipt.json"

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(
        live,
        "validate_live_readiness_receipt",
        lambda **_kwargs: bound_service_receipt,
    )
    monkeypatch.setattr(live, "load_live_environment", lambda **_kwargs: {})

    def load_provider_evidence(**kwargs: object) -> tuple[str, dict[str, object]]:
        assert kwargs["service_receipt_path"] == bound_service_receipt
        return "provider-project", {"hindsight": object()}

    monkeypatch.setattr(
        live,
        "load_live_provider_evidence",
        load_provider_evidence,
    )
    monkeypatch.setattr(
        live,
        "resolve_live_question_case_ids",
        lambda *_args, **_kwargs: (canonical_sha256(["question-case"]),),
    )
    monkeypatch.setattr(live, "build_live_cell", lambda **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        live,
        "execute_live_cells",
        lambda _cells: (
            live.LiveCellCompletion(
                cell_id="hindsight-lme60",
                capsule_root=capsule_root,
            ),
        ),
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--cell",
            "hindsight-lme60",
            "--question",
            "72e3ee87",
            "--run-label",
            "quick-start-smoke",
            "--result-map",
            str(result_map),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result_map.read_text(encoding="utf-8")) == {
        "capsule_roots": {
            "hindsight-lme60": str(capsule_root.resolve()),
        },
        "cells": [
            {
                "capsule_root": str(capsule_root.resolve()),
                "cell_id": "hindsight-lme60",
                "detail": None,
                "status": "completed",
            }
        ],
        "resolved_plan_hash": plan.resolved_plan_hash,
        "schema_name": "live_run_result_map",
        "schema_version": 1,
        "status": "completed",
    }


def test_full_lme60_dispatches_all_cells_without_bounded_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    capsule_roots = {
        cell.cell_id: tmp_path / "capsules" / f"{cell.cell_id}-capsule" for cell in plan.cells
    }
    for capsule_root in capsule_roots.values():
        capsule_root.mkdir(parents=True)
    selected_cell_ids: tuple[str, ...] = ()

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "validate_live_readiness_receipt", lambda **_kwargs: None)
    monkeypatch.setattr(live, "load_live_environment", lambda **_kwargs: {})
    monkeypatch.setattr(
        live,
        "load_live_provider_evidence",
        lambda **_kwargs: (
            "provider-project",
            {cell.provider_id: object() for cell in plan.cells},
        ),
    )
    monkeypatch.setattr(
        live,
        "build_live_cell",
        lambda **kwargs: SimpleNamespace(cell_id=kwargs["cell_id"]),
    )

    def execute(cells: tuple[SimpleNamespace, ...]) -> tuple[live.LiveCellCompletion, ...]:
        nonlocal selected_cell_ids
        selected_cell_ids = tuple(cell.cell_id for cell in cells)
        return tuple(
            live.LiveCellCompletion(cell_id, capsule_roots[cell_id])
            for cell_id in selected_cell_ids
        )

    monkeypatch.setattr(live, "execute_live_cells", execute)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--run-label",
            "quick-start-full",
        ],
    )

    assert result.exit_code == 0, result.output
    assert selected_cell_ids == tuple(cell.cell_id for cell in plan.cells)
    assert "bounded" not in result.output


@pytest.mark.parametrize("selector", ("--question", "--case"))
@pytest.mark.parametrize("selected_count", (2, 59))
def test_multi_case_lme60_selection_uses_readiness_without_bounded_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    selected_count: int,
) -> None:
    from oamb.config import doctor

    plan = _plan()
    selected_cell = plan.cells[0]
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    capsule_root = tmp_path / "capsules" / "selected-cases"
    capsule_root.mkdir(parents=True)
    runtime_loaded = False
    requested_case_ids: tuple[str, ...] = ()

    def load_environment(**_kwargs: object) -> dict[str, str]:
        nonlocal runtime_loaded
        runtime_loaded = True
        return {}

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(
        live,
        "validate_live_readiness_receipt",
        lambda **_kwargs: tmp_path / "readiness.json",
    )
    monkeypatch.setattr(live, "load_live_environment", load_environment)
    monkeypatch.setattr(
        live,
        "load_live_provider_evidence",
        lambda **_kwargs: ("provider-project", {selected_cell.provider_id: object()}),
    )
    if selector == "--question":
        monkeypatch.setattr(
            live,
            "resolve_live_question_case_ids",
            lambda *_args, **_kwargs: tuple(
                canonical_sha256(["selected-question", ordinal])
                for ordinal in range(selected_count)
            ),
        )
        selected_values = [f"q{ordinal}" for ordinal in range(selected_count)]
    else:
        selected_values = [
            canonical_sha256(["selected-case", ordinal]) for ordinal in range(selected_count)
        ]

    def build_cell(**kwargs: object) -> SimpleNamespace:
        nonlocal requested_case_ids
        selected = kwargs["requested_case_manifest_entry_ids"]
        assert isinstance(selected, tuple)
        requested_case_ids = selected
        return SimpleNamespace(cell_id=kwargs["cell_id"])

    monkeypatch.setattr(live, "build_live_cell", build_cell)
    monkeypatch.setattr(
        live,
        "execute_live_cells",
        lambda _cells: (live.LiveCellCompletion(selected_cell.cell_id, capsule_root),),
    )
    arguments = [
        "run",
        str(resolved_plan),
        "--output-root",
        str(tmp_path / "capsules"),
        "--cell",
        "hindsight-lme60",
        "--run-label",
        "multi-case-without-proof",
    ]
    for selected_value in selected_values:
        arguments.extend((selector, selected_value))

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    assert runtime_loaded is True
    assert len(requested_case_ids) == selected_count
    assert "bounded" not in result.output


@pytest.mark.parametrize("has_remaining", (True, False))
def test_recovery_analysis_output_closes_current_execution_without_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_remaining: bool,
) -> None:
    from oamb.artifacts import composition
    from oamb.config import doctor

    plan = _lme6_plan()
    selected_cell = plan.cells[0]
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    recovery_root = tmp_path / "part"
    recovery_root.mkdir()
    analysis_output = tmp_path / "recovery-analysis.json"
    executed = False
    recovery = SimpleNamespace(
        source_capsule_ids=(canonical_sha256(["source-capsule"]),),
        source_manifest_sha256s=(canonical_sha256(["source-manifest"]),),
        reusable_ingestion_plan_ids=(canonical_sha256(["reusable-plan"]),),
        quarantined_ingestion_plan_ids=(canonical_sha256(["quarantined-plan"]),),
        remaining_ingestion_plan_ids=(
            (canonical_sha256(["remaining-plan"]),) if has_remaining else ()
        ),
        remaining_case_manifest_entry_ids=(
            (canonical_sha256(["remaining-case"]),) if has_remaining else ()
        ),
    )

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(composition, "analyze_capsule_recovery", lambda *_a, **_k: recovery)
    monkeypatch.setattr(live, "load_live_environment", lambda **_kwargs: {})
    monkeypatch.setattr(
        live,
        "load_live_provider_evidence",
        lambda **_kwargs: ("provider-project", {selected_cell.provider_id: object()}),
    )
    monkeypatch.setattr(live, "build_live_cell", lambda **_kwargs: object())
    monkeypatch.setattr(live, "live_composition_target", lambda _cell: object())

    def execute(_cells: object) -> tuple[live.LiveCellCompletion, ...]:
        nonlocal executed
        executed = True
        return ()

    monkeypatch.setattr(live, "execute_live_cells", execute)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--cell",
            selected_cell.cell_id,
            "--run-label",
            "recovery-analysis",
            "--recover-from",
            str(recovery_root),
            "--recovery-analysis-output",
            str(analysis_output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert executed is False
    assert json.loads(analysis_output.read_bytes()) == {
        "cell_id": selected_cell.cell_id,
        "quarantined_ingestion_plan_ids": list(recovery.quarantined_ingestion_plan_ids),
        "remaining_case_manifest_entry_ids": list(recovery.remaining_case_manifest_entry_ids),
        "remaining_ingestion_plan_ids": list(recovery.remaining_ingestion_plan_ids),
        "resolved_plan_hash": plan.resolved_plan_hash,
        "reusable_ingestion_plan_ids": list(recovery.reusable_ingestion_plan_ids),
        "schema_name": "capsule_recovery_analysis",
        "schema_version": 1,
        "source_manifest_sha256s": list(recovery.source_manifest_sha256s),
    }


def test_failed_parallel_run_preserves_completed_capsule_in_result_map(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor

    plan = _lme6_plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    result_map = tmp_path / "results" / "full.json"
    completed_root = tmp_path / "capsules" / "hindsight"
    completed_root.mkdir(parents=True)
    outcomes = (
        live.LiveCellOutcome("hindsight-lme6", "completed", completed_root, None),
        live.LiveCellOutcome("mem0-lme6", "failed", None, "planted provider failure"),
        live.LiveCellOutcome("openviking-lme6", "not_started", None, None),
    )

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "load_live_environment", lambda **_kwargs: {})
    monkeypatch.setattr(
        live,
        "load_live_provider_evidence",
        lambda **_kwargs: (
            "provider-project",
            {cell.provider_id: object() for cell in plan.cells},
        ),
    )
    monkeypatch.setattr(
        live, "build_live_cell", lambda **kwargs: SimpleNamespace(cell_id=kwargs["cell_id"])
    )

    def fail(_cells: object) -> tuple[live.LiveCellCompletion, ...]:
        raise live.LiveCellExecutionError("planted aggregate failure", outcomes=outcomes)

    monkeypatch.setattr(live, "execute_live_cells", fail)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--run-label",
            "partial-result-map",
            "--result-map",
            str(result_map),
        ],
    )

    assert result.exit_code != 0
    assert "planted aggregate failure" in result.output
    assert str(completed_root.resolve()) in result.output
    assert json.loads(result_map.read_text(encoding="utf-8")) == {
        "capsule_roots": {"hindsight-lme6": str(completed_root.resolve())},
        "cells": [
            {
                "capsule_root": str(completed_root.resolve()),
                "cell_id": "hindsight-lme6",
                "detail": None,
                "status": "completed",
            },
            {
                "capsule_root": None,
                "cell_id": "mem0-lme6",
                "detail": "planted provider failure",
                "status": "failed",
            },
            {
                "capsule_root": None,
                "cell_id": "openviking-lme6",
                "detail": None,
                "status": "not_started",
            },
        ],
        "resolved_plan_hash": plan.resolved_plan_hash,
        "schema_name": "live_run_result_map",
        "schema_version": 1,
        "status": "failed",
    }


@pytest.mark.parametrize("publication_failure", ("collision", "write-error"))
def test_result_map_publication_failure_preserves_worker_error_and_console_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publication_failure: str,
) -> None:
    from oamb import cli
    from oamb.config import doctor

    plan = _lme6_plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    result_map = tmp_path / "results" / "full.json"
    completed_root = tmp_path / "capsules" / "hindsight"
    completed_root.mkdir(parents=True)
    outcomes = (
        live.LiveCellOutcome("hindsight-lme6", "completed", completed_root, None),
        live.LiveCellOutcome("mem0-lme6", "failed", None, "planted worker failure"),
        live.LiveCellOutcome("openviking-lme6", "not_started", None, None),
    )

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "load_live_environment", lambda **_kwargs: {})
    monkeypatch.setattr(
        live,
        "load_live_provider_evidence",
        lambda **_kwargs: (
            "provider-project",
            {cell.provider_id: object() for cell in plan.cells},
        ),
    )
    monkeypatch.setattr(
        live, "build_live_cell", lambda **kwargs: SimpleNamespace(cell_id=kwargs["cell_id"])
    )

    def fail(_cells: object) -> tuple[live.LiveCellCompletion, ...]:
        if publication_failure == "collision":
            result_map.parent.mkdir(parents=True)
            result_map.write_text("preserve concurrent bytes", encoding="utf-8")
        raise live.LiveCellExecutionError("planted aggregate failure", outcomes=outcomes)

    monkeypatch.setattr(live, "execute_live_cells", fail)
    if publication_failure == "write-error":
        monkeypatch.setattr(
            cli,
            "atomic_write_bytes",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("planted map write error")),
        )

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--run-label",
            "result-map-publication-failure",
            "--result-map",
            str(result_map),
        ],
    )

    assert result.exit_code != 0
    assert "planted aggregate failure" in result.output
    assert "planted worker failure" in result.output
    assert str(completed_root.resolve()) in result.output
    if publication_failure == "collision":
        assert "different bytes" in result.output
        assert result_map.read_text(encoding="utf-8") == "preserve concurrent bytes"
    else:
        assert "planted map write error" in result.output


def test_live_run_rejects_existing_result_map_before_runtime_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    result_map = tmp_path / "existing-result.json"
    result_map.write_text("preserve me", encoding="utf-8")
    runtime_loaded = False

    def load_environment(**_kwargs: object) -> dict[str, str]:
        nonlocal runtime_loaded
        runtime_loaded = True
        return {}

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "load_live_environment", load_environment)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--cell",
            "hindsight-lme60",
            "--question",
            "72e3ee87",
            "--run-label",
            "quick-start-smoke",
            "--result-map",
            str(result_map),
        ],
    )

    assert result.exit_code != 0
    assert "result map already exists" in result.output
    assert result_map.read_text(encoding="utf-8") == "preserve me"
    assert runtime_loaded is False
