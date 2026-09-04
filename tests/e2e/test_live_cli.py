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

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = REPOSITORY_ROOT / "configs/benchmark.yml"


def _plan() -> ResolvedPlan:
    return build_resolved_plan(load_benchmark_configuration(BENCHMARK_CONFIG))


def _lme6_plan() -> ResolvedPlan:
    return build_resolved_plan(
        load_benchmark_configuration(REPOSITORY_ROOT / "tests/fixtures/configs/t10-lme6.yml")
    )


def test_root_model_environment_template_contains_only_placeholders() -> None:
    entries = {}
    for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            entries[key] = value

    assert entries == {
        "DEEPSEEK_BASE_URL": "change-me",
        "DEEPSEEK_API_KEY": "change-me",
    }


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
            "quick-start-bounded",
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


def test_full_lme60_rejects_missing_bounded_proof_before_runtime_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.config import doctor

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "validate_live_readiness_receipt", lambda **_kwargs: None)

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

    assert result.exit_code != 0
    assert "full LME-60 requires three bounded capsules and validations" in result.output


@pytest.mark.parametrize("selector", ("--question", "--case"))
@pytest.mark.parametrize("selected_count", (2, 59, 60))
def test_multi_case_lme60_selection_requires_bounded_proof_before_runtime_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    selected_count: int,
) -> None:
    from oamb.config import doctor

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    runtime_loaded = False

    def load_environment(**_kwargs: object) -> dict[str, str]:
        nonlocal runtime_loaded
        runtime_loaded = True
        return {}

    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "validate_live_readiness_receipt", lambda **_kwargs: None)
    monkeypatch.setattr(live, "load_live_environment", load_environment)
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

    assert result.exit_code != 0
    assert "LME-60 multi-case or recovery run requires bounded proofs" in result.output
    assert runtime_loaded is False


def test_lme60_recovery_requires_bounded_proof_before_runtime_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.artifacts import composition
    from oamb.config import doctor

    plan = _plan()
    resolved_plan = tmp_path / "resolved-plan.json"
    resolved_plan.write_text('{"schema_name":"resolved_plan"}', encoding="utf-8")
    recovery_root = tmp_path / "part"
    recovery_root.mkdir()
    runtime_loaded = False

    def load_environment(**_kwargs: object) -> dict[str, str]:
        nonlocal runtime_loaded
        runtime_loaded = True
        return {}

    recovery = SimpleNamespace(
        source_capsule_ids=("source",),
        source_manifest_sha256s=(canonical_sha256(["source-manifest"]),),
        reusable_ingestion_plan_ids=("reusable",),
        quarantined_ingestion_plan_ids=(),
        remaining_ingestion_plan_ids=("remaining",),
        remaining_case_manifest_entry_ids=(canonical_sha256(["remaining-case"]),),
    )
    monkeypatch.setattr(doctor, "load_resolved_plan_for_run", lambda _path: plan)
    monkeypatch.setattr(live, "validate_live_readiness_receipt", lambda **_kwargs: None)
    monkeypatch.setattr(live, "load_live_environment", load_environment)
    monkeypatch.setattr(composition, "analyze_capsule_recovery", lambda *_a, **_k: recovery)

    result = CliRunner().invoke(
        app,
        [
            "run",
            str(resolved_plan),
            "--output-root",
            str(tmp_path / "capsules"),
            "--cell",
            "hindsight-lme60",
            "--run-label",
            "recovery-without-proof",
            "--recover-from",
            str(recovery_root),
        ],
    )

    assert result.exit_code != 0
    assert "LME-60 multi-case or recovery run requires bounded proofs" in result.output
    assert runtime_loaded is False


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
            "quick-start-bounded",
            "--result-map",
            str(result_map),
        ],
    )

    assert result.exit_code != 0
    assert "result map already exists" in result.output
    assert result_map.read_text(encoding="utf-8") == "preserve me"
    assert runtime_loaded is False
