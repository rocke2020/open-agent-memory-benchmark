from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

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


def test_run_help_excludes_deleted_recovery_options() -> None:
    result = CliRunner().invoke(app, ["run", "--help"])

    assert result.exit_code == 0, result.output
    for option in ("--continue-from", "--recover-from", "--recovery-analysis-output"):
        assert option not in result.output


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


def test_full_lme60_requires_provider_result_files_before_dispatch(
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

    assert result.exit_code == 2
    assert "full LME-60 requires one result file per provider" in result.output
    assert selected_cell_ids == ()


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


def test_question_result_failure_stops_before_environment_or_provider_preparation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor
    from oamb.runtime.question_results import QuestionResultsError

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    runtime_loaded = False

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(
        live,
        "load_live_question_results",
        lambda **_kwargs: (_ for _ in ()).throw(
            QuestionResultsError("question results cannot parse or validate")
        ),
    )

    def forbidden_environment(**_kwargs: object) -> dict[str, str]:
        nonlocal runtime_loaded
        runtime_loaded = True
        raise AssertionError("invalid question results reached environment loading")

    monkeypatch.setattr(live, "load_live_environment", forbidden_environment)
    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--run-label",
            "simple-resume",
            "--results-root",
            str(tmp_path / "results"),
        ],
    )

    assert result.exit_code != 0
    assert "question results cannot parse" in result.output
    assert runtime_loaded is False


def test_question_results_build_only_cells_with_remaining_questions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    remaining = {
        plan.cells[0].cell_id: (canonical_sha256(["remaining-hindsight"]),),
        plan.cells[1].cell_id: (),
        plan.cells[2].cell_id: (canonical_sha256(["remaining-openviking"]),),
    }
    results_by_cell = {
        cell.cell_id: {str(index): object() for index in range(60 - len(remaining[cell.cell_id]))}
        for cell in plan.cells
    }
    selection = SimpleNamespace(
        results_by_cell=results_by_cell,
        remaining_case_manifest_entry_ids=remaining,
        ordered_question_ids=tuple(canonical_sha256(["question", index]) for index in range(60)),
    )
    built: list[dict[str, object]] = []
    capsule_roots = {cell.cell_id: tmp_path / "capsules" / cell.cell_id for cell in plan.cells}
    for root in capsule_roots.values():
        root.mkdir(parents=True)

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "load_live_question_results", lambda **_kwargs: selection)
    monkeypatch.setattr(live, "validate_live_readiness_receipt", lambda **_kwargs: None)

    def load_environment(**_kwargs: object) -> dict[str, str]:
        return {}

    monkeypatch.setattr(live, "load_live_environment", load_environment)
    monkeypatch.setattr(
        live,
        "load_live_provider_evidence",
        lambda **_kwargs: (
            "provider-project",
            {cell.provider_id: object() for cell in plan.cells},
        ),
    )

    def build_cell(**kwargs: object) -> SimpleNamespace:
        built.append(kwargs)
        return SimpleNamespace(cell_id=kwargs["cell_id"])

    monkeypatch.setattr(live, "build_live_cell", build_cell)
    monkeypatch.setattr(
        live,
        "execute_live_cells",
        lambda cells: tuple(
            live.LiveCellCompletion(cell.cell_id, capsule_roots[cell.cell_id]) for cell in cells
        ),
    )
    results_root = tmp_path / "fresh-run" / "results"
    results_root.mkdir(parents=True)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--run-label",
            "simple-resume",
            "--results-root",
            str(results_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert tuple(item["cell_id"] for item in built) == (
        plan.cells[0].cell_id,
        plan.cells[2].cell_id,
    )
    for item in built:
        cell_id = str(item["cell_id"])
        provider_id = next(cell.provider_id for cell in plan.cells if cell.cell_id == cell_id)
        assert item["requested_case_manifest_entry_ids"] == remaining[cell_id]
        assert item["ordered_question_ids"] is selection.ordered_question_ids
        assert item["results_path"] == results_root / f"{provider_id}.json"


def test_complete_question_results_return_before_runtime_or_provider_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    selection = SimpleNamespace(
        results_by_cell={
            cell.cell_id: {str(index): object() for index in range(60)} for cell in plan.cells
        },
        remaining_case_manifest_entry_ids={cell.cell_id: () for cell in plan.cells},
        ordered_question_ids=tuple(canonical_sha256(["question", index]) for index in range(60)),
    )

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "load_live_question_results", lambda **_kwargs: selection)

    def forbidden_environment(**_kwargs: object) -> dict[str, str]:
        raise AssertionError("complete question results reached runtime loading")

    monkeypatch.setattr(live, "load_live_environment", forbidden_environment)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--run-label",
            "complete-results",
            "--results-root",
            str(tmp_path / "results"),
        ],
    )

    assert result.exit_code == 0, result.output
    assert "question results: PASS complete=180 zero_dispatch=true" in result.output


def test_compare_accepts_only_the_three_provider_result_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor
    from oamb.reporting import comparison_project

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    results_by_cell = {cell.cell_id: object() for cell in plan.cells}
    selection = SimpleNamespace(results_by_cell=results_by_cell, case_manifest=object())
    captured: dict[str, object] = {}
    report_root = tmp_path / "report"

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "load_live_question_results", lambda **_kwargs: selection)

    def build_results_report(*args: object, **kwargs: object) -> SimpleNamespace:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            comparison_paths=(),
            export_path=report_root / "report.json",
            html_path=report_root / "report.html",
        )

    monkeypatch.setattr(
        comparison_project,
        "build_question_results_comparison_project",
        build_results_report,
    )

    result = CliRunner().invoke(
        app,
        [
            "compare",
            str(resolved_plan),
            "--results-root",
            str(tmp_path / "results"),
            "--output-root",
            str(report_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["args"] == (plan, results_by_cell)
    assert cast(dict[str, object], captured["kwargs"])["case_manifest"] is selection.case_manifest


def test_compare_builds_the_report_analysis_generator_from_explicit_model_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches the normal compare CLI silently omitting requested LLM analysis."""

    from oamb.config import doctor
    from oamb.reporting import comparison_project, report_analysis

    factory = getattr(report_analysis, "build_report_analysis_generator", None)
    assert factory is not None, "report analysis generator factory is not implemented"
    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    model_env = tmp_path / ".env"
    model_env.write_text(
        "LLM_URL_TYPE=openai_chat\nLLM_BASE_URL=https://models.example/v1\nLLM_API_KEY=test-key\n",
        encoding="utf-8",
    )
    results_by_cell = {cell.cell_id: object() for cell in plan.cells}
    selection = SimpleNamespace(results_by_cell=results_by_cell, case_manifest=object())
    report_root = tmp_path / "report"
    cache_root = tmp_path / "analysis-cache"
    captured: dict[str, object] = {}

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "load_live_question_results", lambda **_kwargs: selection)

    def sentinel_generator(_export: object) -> None:
        return None

    def build_generator(**kwargs: object) -> object:
        captured["factory"] = kwargs
        return sentinel_generator

    monkeypatch.setattr(report_analysis, "build_report_analysis_generator", build_generator)

    def build_results_report(*_args: object, **kwargs: object) -> SimpleNamespace:
        captured["builder"] = kwargs
        return SimpleNamespace(
            comparison_paths=(),
            export_path=report_root / "report.json",
            html_path=report_root / "report.html",
            analysis_path=None,
        )

    monkeypatch.setattr(
        comparison_project,
        "build_question_results_comparison_project",
        build_results_report,
    )

    result = CliRunner().invoke(
        app,
        [
            "compare",
            str(resolved_plan),
            "--results-root",
            str(tmp_path / "results"),
            "--output-root",
            str(report_root),
            "--analysis-model-env",
            str(model_env),
            "--analysis-cache-root",
            str(cache_root),
        ],
    )

    assert result.exit_code == 0, result.output
    assert cast(dict[str, object], captured["factory"])["plan"] is plan
    assert cast(dict[str, object], captured["factory"])["model_env_path"] == model_env
    assert cast(dict[str, object], captured["factory"])["cache_root"] == cache_root
    assert cast(dict[str, object], captured["builder"])["analysis_generator"] is sentinel_generator
    assert "report analysis: unavailable" in result.output
