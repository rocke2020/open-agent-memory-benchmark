from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from click import unstyle

from oamb.artifacts.store import ArtifactStore
from oamb.cli import run_generated_fake_vertical_slice
from oamb.contracts.evidence import ValidationResult
from oamb.contracts.ports import ModelReceipt, ModelRequest, ThinkingEffort
from oamb.contracts.states import ValidationDisposition
from oamb.model_clients.fake import ScriptedFakeModelClient
from oamb.runtime.native_run import NativeRunArtifacts, run_native_vertical_slice
from oamb.workloads.longmemeval import (
    LME30_WORKLOAD_ID,
    LongMemEvalWorkload,
    _build_bundle,
    load_longmemeval_rows,
)
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY
from tests.e2e.test_native_fixture_vertical_slice import (
    _memory_factory,
    _run_fixture_capsule,
)
from tests.unit.test_longmemeval import (
    _dataset_manifest_for_test,
    _row,
    _write_rows,
)

FORBIDDEN_PUBLIC_TERMS = ("t10", "phase", "approval", "signature")


def _installed_oamb(*arguments: str) -> subprocess.CompletedProcess[str]:
    executable = Path(sys.executable).with_name("oamb")
    assert executable.is_file()
    return subprocess.run(
        [str(executable), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def _report_path(result: subprocess.CompletedProcess[str]) -> Path:
    assert result.returncode == 0, result.stdout + result.stderr
    return Path(result.stdout.strip().split("report: ", 1)[1])


def _published_bytes(report_path: Path) -> dict[str, bytes]:
    final_directory = report_path.parents[1]
    return {
        path.relative_to(final_directory).as_posix(): path.read_bytes()
        for path in sorted(final_directory.rglob("*"))
        if path.is_file()
    }


def _run_lme_identity_fixture_capsule(tmp_path: Path) -> NativeRunArtifacts:
    class SuccessfulJudgeModel(ScriptedFakeModelClient):
        def thinking_effort_for(
            self,
            *,
            stage: str,
            role_binding_id: str,
        ) -> ThinkingEffort:
            if stage == "judge" and role_binding_id == "oamb-lme-judge-v1":
                return "high"
            return super().thinking_effort_for(
                stage=stage,
                role_binding_id=role_binding_id,
            )

        async def complete(self, request: ModelRequest) -> ModelReceipt:
            if request.stage != "judge":
                return await super().complete(request)
            raw_reference, usage_ids = self._seal_call(
                request,
                outcome="success",
                output_text="yes",
            )
            return ModelReceipt(
                raw_reference=raw_reference,
                output_text="yes",
                usage_reference_ids=usage_ids,
                model="fixture-judge-model",
            )

    source = tmp_path / "longmemeval.json"
    source_hash = _write_rows(source, [_row()])
    row = load_longmemeval_rows(source, expected_sha256=source_hash)[0]
    bundle = _build_bundle(
        _dataset_manifest_for_test(source),
        (row,),
        workload_id=LME30_WORKLOAD_ID,
    )
    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="installed-lme-identity-source-root",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=LongMemEvalWorkload(bundle),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=SuccessfulJudgeModel,
        answer_role_binding_id="fake-answer-v1",
        judge_model_factory=SuccessfulJudgeModel,
        judge_role_binding_id="oamb-lme-judge-v1",
    )


def test_installed_source_root_commands_have_no_fake_only_or_review_surface() -> None:
    for command in (("capsule", "validate", "--help"), ("report", "build", "--help")):
        result = _installed_oamb(*command)

        assert result.returncode == 0, result.stdout + result.stderr
        output = (result.stdout + result.stderr).lower()
        assert "source root" in output
        assert "fake" not in output
        assert all(term not in output for term in FORBIDDEN_PUBLIC_TERMS)


def test_installed_run_exposes_plain_results_without_resume_protocol_options() -> None:
    run_help = _installed_oamb("run", "--help")
    compose_help = _installed_oamb("capsule", "compose", "--help")

    assert run_help.returncode == 0, run_help.stdout + run_help.stderr
    run_output = unstyle(run_help.stdout + run_help.stderr)
    assert "--case" in run_output
    assert "--results-root" in run_output
    assert "--full-progress-root" not in run_output
    assert "--full-resume-lock" not in run_output
    assert "--full-resume-pointer" not in run_output
    assert "--full-resume-rehearsal" not in run_output
    assert "--recover-from" not in run_output
    assert "--continue-from" not in run_output
    assert compose_help.returncode != 0


def test_installed_native_source_root_validates_and_builds_deterministic_offline_report(
    tmp_path: Path,
) -> None:
    completed = _run_fixture_capsule(tmp_path, "installed-native-source-root")
    validation_path = tmp_path / "native-validation.json"

    validation = _installed_oamb(
        "capsule",
        "validate",
        str(completed.capsule_root),
        "--output",
        str(validation_path),
    )

    assert validation.returncode == 0, validation.stdout + validation.stderr
    result = ValidationResult.model_validate_json(validation_path.read_bytes())
    assert result.disposition == ValidationDisposition.VALIDATED
    assert result.validation_profile_id == "oamb-t8-native-evidence-v1"

    first = _installed_oamb(
        "report",
        "build",
        str(completed.capsule_root),
        "--validation",
        str(validation_path),
        "--output-root",
        str(tmp_path / "reports-first"),
    )
    second = _installed_oamb(
        "report",
        "build",
        str(completed.capsule_root),
        "--validation",
        str(validation_path),
        "--output-root",
        str(tmp_path / "reports-second"),
    )
    first_path = _report_path(first)
    second_path = _report_path(second)

    assert _published_bytes(first_path) == _published_bytes(second_path)
    html = first_path.read_text(encoding="utf-8")
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "prefers-color-scheme: dark" in html
    assert "http://" not in html and "https://" not in html
    public_bytes = validation_path.read_bytes() + first_path.read_bytes()
    lowered = public_bytes.lower()
    assert all(term.encode() not in lowered for term in FORBIDDEN_PUBLIC_TERMS)


def test_installed_native_report_uses_the_actual_validation_profile_not_the_workload_name(
    tmp_path: Path,
) -> None:
    completed = _run_lme_identity_fixture_capsule(tmp_path)
    validation_path = tmp_path / "lme-validation.json"
    validation = _installed_oamb(
        "capsule",
        "validate",
        str(completed.capsule_root),
        "--output",
        str(validation_path),
    )

    assert validation.returncode == 0, validation.stdout + validation.stderr
    report = _installed_oamb(
        "report",
        "build",
        str(completed.capsule_root),
        "--validation",
        str(validation_path),
        "--output-root",
        str(tmp_path / "lme-report"),
    )

    assert report.returncode == 0, report.stdout + report.stderr
    assert _report_path(report).is_file()


def test_installed_report_reopens_native_source_root_and_rejects_stale_validation(
    tmp_path: Path,
) -> None:
    completed = _run_fixture_capsule(tmp_path, "installed-native-stale-validation")
    validation_path = tmp_path / "native-validation.json"
    validated = _installed_oamb(
        "capsule",
        "validate",
        str(completed.capsule_root),
        "--output",
        str(validation_path),
    )
    assert validated.returncode == 0, validated.stdout + validated.stderr
    manifest = json.loads((completed.capsule_root / "capsule-manifest.json").read_bytes())
    run_entry = next(
        entry for entry in manifest["source_entries"] if entry["record_kind"] == "run_record"
    )
    source_path = completed.capsule_root / run_entry["relative_path"]
    source_path.write_bytes(source_path.read_bytes() + b"\n")

    report_root = tmp_path / "stale-report"
    report = _installed_oamb(
        "report",
        "build",
        str(completed.capsule_root),
        "--validation",
        str(validation_path),
        "--output-root",
        str(report_root),
    )

    assert report.returncode != 0
    assert not (report_root / "derivations").exists()


def test_installed_report_rejects_invalid_result_even_with_diagnostic_opt_in(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="installed-invalid-fake-source-root",
    )
    run_path = completed.capsule_root / "source" / "run" / "installed-invalid-fake-source-root.json"
    run_path.write_bytes(run_path.read_bytes() + b"\n")
    validation_path = tmp_path / "invalid-validation.json"
    validation = _installed_oamb(
        "capsule",
        "validate",
        str(completed.capsule_root),
        "--output",
        str(validation_path),
        "--diagnostic",
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr
    assert (
        ValidationResult.model_validate_json(validation_path.read_bytes()).disposition
        == ValidationDisposition.INVALID
    )

    report_root = tmp_path / "invalid-report"
    report = _installed_oamb(
        "report",
        "build",
        str(completed.capsule_root),
        "--validation",
        str(validation_path),
        "--output-root",
        str(report_root),
        "--diagnostic",
    )

    assert report.returncode != 0
    assert "validated evidence result" in (report.stdout + report.stderr).lower()
    assert not (report_root / "derivations").exists()
