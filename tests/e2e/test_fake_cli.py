from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from oamb.cli import app


def test_fake_cli_runs_manifest_through_committed_offline_report(tmp_path: Path) -> None:
    runner = CliRunner()
    inputs = tmp_path / "inputs"
    capsules = tmp_path / "capsules"
    resolved_plan = tmp_path / "resolved-plan.json"
    evidence = tmp_path / "evidence-validation.json"
    summary = tmp_path / "run-summary.json"
    reports = tmp_path / "reports"
    run_id = "fake-cli-run"

    manifest_result = runner.invoke(
        app,
        [
            "manifest",
            "build",
            "--workload",
            "fake",
            "--run-id",
            run_id,
            "--output",
            str(inputs),
        ],
    )
    assert manifest_result.exit_code == 0, manifest_result.output
    case_manifest = json.loads((inputs / "case-manifest.json").read_bytes())
    assert len(case_manifest["ingestion_plans"]) == 2
    assert len(case_manifest["cases"]) == 4

    preflight_result = runner.invoke(
        app,
        [
            "preflight",
            "--run-spec",
            str(inputs / "run-spec.json"),
            "--artifact-root",
            str(capsules),
            "--output",
            str(resolved_plan),
        ],
    )
    assert preflight_result.exit_code == 0, preflight_result.output
    resolved = json.loads(resolved_plan.read_bytes())
    durability = resolved["artifact_durability"]
    assert durability["available_bytes"] > 0
    assert durability["lease_supported"] is True
    assert durability["file_fsync_supported"] is True
    assert durability["directory_fsync_supported"] is True
    assert durability["no_replace_supported"] is True

    run_result = runner.invoke(app, ["run", "--resolved-plan", str(resolved_plan)])
    assert run_result.exit_code == 0, run_result.output
    capsule_root = capsules / run_id
    assert (capsule_root / "capsule-manifest.json").is_file()

    validate_result = runner.invoke(
        app,
        [
            "capsule",
            "validate",
            str(capsule_root),
            "--output",
            str(evidence),
        ],
    )
    assert validate_result.exit_code == 0, validate_result.output
    assert json.loads(evidence.read_bytes())["disposition"] == "validated"

    summarize_result = runner.invoke(
        app,
        [
            "summarize",
            str(capsule_root),
            "--validation",
            str(evidence),
            "--output",
            str(summary),
        ],
    )
    assert summarize_result.exit_code == 0, summarize_result.output
    assert json.loads(summary.read_bytes())["ready_ingestion_plans"] == 1

    report_result = runner.invoke(
        app,
        [
            "report",
            "build",
            str(capsule_root),
            "--validation",
            str(evidence),
            "--output-root",
            str(reports),
            "--audience",
            "public",
        ],
    )
    assert report_result.exit_code == 0, report_result.output
    report_path = Path(report_result.output.strip().split("report: ", 1)[1])
    assert report_path.is_file()
    assert report_path.as_uri().startswith("file://")


def test_fake_preflight_rejects_any_run_spec_field_drift(tmp_path: Path) -> None:
    runner = CliRunner()
    inputs = tmp_path / "inputs"
    manifest_result = runner.invoke(
        app,
        [
            "manifest",
            "build",
            "--workload",
            "fake",
            "--run-id",
            "fake-preflight-drift",
            "--output",
            str(inputs),
        ],
    )
    assert manifest_result.exit_code == 0, manifest_result.output
    run_spec_path = inputs / "run-spec.json"
    run_spec = json.loads(run_spec_path.read_bytes())
    run_spec["protocol_id"] = "drifted-protocol"
    run_spec_path.write_text(json.dumps(run_spec), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "preflight",
            "--run-spec",
            str(run_spec_path),
            "--artifact-root",
            str(tmp_path / "capsules"),
            "--output",
            str(tmp_path / "resolved-plan.json"),
        ],
    )

    assert result.exit_code != 0
    assert "does not exactly match" in result.output
