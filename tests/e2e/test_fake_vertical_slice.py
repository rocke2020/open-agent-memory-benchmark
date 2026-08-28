from __future__ import annotations

import importlib
import json
import re
from pathlib import Path
from types import ModuleType

import pytest

from oamb.artifacts.capsule import verify_published_directory
from oamb.artifacts.store import ArtifactStore
from oamb.contracts.ports import (
    ArtifactStorePort,
    IngestionPlan,
    MemorySystemPort,
    ModelReceipt,
    ModelRequest,
    RuntimeResolution,
)
from oamb.contracts.states import (
    AttemptOutcome,
    ResumeDisposition,
    RunState,
    ValidationDisposition,
)
from oamb.memory_systems.fake import ScriptedFakeMemorySystem
from oamb.model_clients.fake import ScriptedFakeModelClient
from oamb.runtime.fake_run import FakeRunScenario, run_fake_vertical_slice
from oamb.workloads.fake import GeneratedFakeWorkload


def require(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} is not implemented", pytrace=False)


def test_fake_run_seals_valid_retry_unjudged_and_shared_plan_evidence(tmp_path: Path) -> None:
    composition = require("oamb.cli")
    fake_validation = require("oamb.artifacts.validation.fake")
    reduce = require("oamb.reporting.reduce")

    completed = composition.run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-valid",
    )
    validation = fake_validation.validate_fake_capsule(completed.capsule_root)
    summary = reduce.reduce_run_summary(completed.capsule_root, validation)

    assert validation.disposition == ValidationDisposition.VALIDATED
    assert summary.intended_logical_contexts == 2
    assert summary.intended_ingestion_plans == 2
    assert summary.ready_ingestion_plans == 1
    assert summary.intended_cases == 4
    assert summary.terminal_cases == 4
    assert summary.completed_cases == 2
    assert summary.errored_cases == 2
    assert summary.parsed_cases == 3
    assert summary.evaluated_cases == 2
    assert summary.judged_cases == 0
    assert summary.unjudged_cases == 1
    assert summary.billing_complete is False
    assert summary.cost_complete is False

    source_documents = fake_validation.load_fake_capsule(completed.capsule_root).contracts
    answer_attempts = tuple(
        sorted(
            (
                document
                for document in source_documents
                if getattr(document, "schema_name", None) == "attempt_record"
                and getattr(document, "stage", None) == "answer"
            ),
            key=lambda attempt: attempt.started_at,
        )
    )
    assert tuple(attempt.outcome for attempt in answer_attempts) == (
        AttemptOutcome.SUCCEEDED,
        AttemptOutcome.FAILED,
        AttemptOutcome.SUCCEEDED,
        AttemptOutcome.SUCCEEDED,
    )
    failed = answer_attempts[1]
    retried = answer_attempts[2]
    assert retried.retry_of_attempt_id == failed.attempt_id

    cases = tuple(
        document
        for document in source_documents
        if getattr(document, "schema_name", None) == "case_record"
    )
    assert sum(case.evaluation_disposition == "unjudged" for case in cases) == 1
    plan_usage_ids = tuple(
        usage_id
        for document in source_documents
        if getattr(document, "schema_name", None) == "ingestion_plan_record"
        for usage_id in document.usage_record_ids
    )
    assert len(plan_usage_ids) == 2
    assert len(set(plan_usage_ids)) == 2


def test_fake_runtime_sends_the_rendered_visible_evidence_prompt_to_the_model(
    tmp_path: Path,
) -> None:
    captured_requests: list[ModelRequest] = []
    workload = GeneratedFakeWorkload()

    class CapturingModelClient:
        def __init__(self, store: ArtifactStorePort) -> None:
            self._delegate = ScriptedFakeModelClient(store)

        async def complete(self, request: ModelRequest) -> ModelReceipt:
            captured_requests.append(request)
            return await self._delegate.complete(request)

        async def close(self) -> None:
            await self._delegate.close()

    def memory_factory(
        store: ArtifactStorePort,
        plans: tuple[IngestionPlan, ...],
    ) -> MemorySystemPort:
        return ScriptedFakeMemorySystem(
            store=store,
            delayed_readiness_plan_ids=(plans[0].ingestion_plan_id,),
            partial_ingestion_plan_ids=(plans[1].ingestion_plan_id,),
        )

    run_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-rendered-prompt",
        workload=workload,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=CapturingModelClient,
    )

    answer_requests = [request for request in captured_requests if request.stage == "answer"]
    assert answer_requests
    assert all('"visible_evidence"' in request.messages[0][1] for request in answer_requests)


def test_fake_runtime_closes_both_ports_when_case_execution_raises(tmp_path: Path) -> None:
    closed = {"memory": False, "model": False}
    workload = GeneratedFakeWorkload()

    class TrackingMemorySystem(ScriptedFakeMemorySystem):
        async def close(self) -> None:
            closed["memory"] = True
            await super().close()

    class FailingModelClient:
        async def complete(self, request: ModelRequest) -> ModelReceipt:
            raise RuntimeError(f"planted model failure at {request.stage}")

        async def close(self) -> None:
            closed["model"] = True

    def memory_factory(
        store: ArtifactStorePort,
        plans: tuple[IngestionPlan, ...],
    ) -> MemorySystemPort:
        return TrackingMemorySystem(
            store=store,
            delayed_readiness_plan_ids=(plans[0].ingestion_plan_id,),
            partial_ingestion_plan_ids=(plans[1].ingestion_plan_id,),
        )

    with pytest.raises(RuntimeError, match="planted model failure"):
        run_fake_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="fake-close-on-error",
            workload=workload,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=lambda _store: FailingModelClient(),
        )

    assert closed == {"memory": True, "model": True}
    assert (tmp_path / "capsules" / "fake-close-on-error" / "capsule-manifest.json").is_file()


def test_fake_runtime_closes_both_ports_when_runtime_resolution_raises(tmp_path: Path) -> None:
    closed = {"memory": False, "model": False}
    workload = GeneratedFakeWorkload()

    class ResolveFailingMemorySystem(ScriptedFakeMemorySystem):
        async def resolve(self) -> RuntimeResolution:
            raise RuntimeError("planted runtime resolution failure")

        async def close(self) -> None:
            closed["memory"] = True
            await super().close()

    class TrackingModelClient(ScriptedFakeModelClient):
        async def close(self) -> None:
            closed["model"] = True
            await super().close()

    def memory_factory(
        store: ArtifactStorePort,
        plans: tuple[IngestionPlan, ...],
    ) -> MemorySystemPort:
        return ResolveFailingMemorySystem(
            store=store,
            delayed_readiness_plan_ids=(plans[0].ingestion_plan_id,),
            partial_ingestion_plan_ids=(plans[1].ingestion_plan_id,),
        )

    with pytest.raises(RuntimeError, match="planted runtime resolution failure"):
        run_fake_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="fake-close-on-resolution-error",
            workload=workload,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=TrackingModelClient,
        )

    assert closed == {"memory": True, "model": True}


def test_fake_runtime_records_close_error_and_still_seals_terminal_manifest(
    tmp_path: Path,
) -> None:
    workload = GeneratedFakeWorkload()

    class CloseFailingModelClient(ScriptedFakeModelClient):
        async def close(self) -> None:
            await super().close()
            raise RuntimeError("planted close failure")

    def memory_factory(
        store: ArtifactStorePort,
        plans: tuple[IngestionPlan, ...],
    ) -> MemorySystemPort:
        return ScriptedFakeMemorySystem(
            store=store,
            delayed_readiness_plan_ids=(plans[0].ingestion_plan_id,),
            partial_ingestion_plan_ids=(plans[1].ingestion_plan_id,),
        )

    completed = run_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-close-error",
        workload=workload,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=CloseFailingModelClient,
    )

    assert (completed.capsule_root / "capsule-manifest.json").is_file()
    source_documents = (
        require("oamb.artifacts.validation.fake")
        .load_fake_capsule(completed.capsule_root)
        .contracts
    )
    assert (
        sum(
            getattr(document, "schema_name", None) == "close_error_record"
            for document in source_documents
        )
        == 1
    )
    assert (
        require("oamb.artifacts.validation.fake")
        .validate_fake_capsule(completed.capsule_root)
        .disposition
        == ValidationDisposition.VALIDATED
    )


def test_fake_run_rejects_an_escaping_run_id_before_creating_artifacts(tmp_path: Path) -> None:
    escaped_root = tmp_path / "escaped"

    with pytest.raises(ValueError, match="one safe path component"):
        require("oamb.cli").run_generated_fake_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="../escaped",
        )

    assert not escaped_root.exists()


def test_invalid_capsule_cannot_reduce_but_remains_available_for_diagnostics(
    tmp_path: Path,
) -> None:
    composition = require("oamb.cli")
    fake_validation = require("oamb.artifacts.validation.fake")
    reduce = require("oamb.reporting.reduce")

    completed = composition.run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-invalid",
    )
    run_path = completed.capsule_root / "source" / "run" / "fake-invalid.json"
    run_document = json.loads(run_path.read_text(encoding="utf-8"))
    run_document["state"] = "running"
    run_path.write_text(json.dumps(run_document), encoding="utf-8")

    validation = fake_validation.validate_fake_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    with pytest.raises(fake_validation.EvidenceNotValidatedError):
        reduce.reduce_run_summary(completed.capsule_root, validation)


def test_validated_capsule_must_be_revalidated_after_source_bytes_change(
    tmp_path: Path,
) -> None:
    composition = require("oamb.cli")
    fake_validation = require("oamb.artifacts.validation.fake")
    reduce = require("oamb.reporting.reduce")
    completed = composition.run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-stale-validation",
    )
    validation = fake_validation.validate_fake_capsule(completed.capsule_root)
    run_path = completed.capsule_root / "source" / "run" / "fake-stale-validation.json"
    run_path.write_bytes(run_path.read_bytes() + b"\n")

    with pytest.raises(fake_validation.EvidenceNotValidatedError, match="current capsule"):
        reduce.reduce_run_summary(completed.capsule_root, validation)


@pytest.mark.parametrize(
    ("scenario_name", "expected_state", "expected_resume", "unknown_attempts"),
    (
        ("cancelled", RunState.ABORTED, ResumeDisposition.NOT_APPLICABLE, 0),
        (
            "unknown_outcome",
            RunState.INTERRUPTED,
            ResumeDisposition.REPLACEMENT_RUN_REQUIRED,
            1,
        ),
    ),
)
def test_terminal_diagnostic_runs_preserve_cancel_and_unknown_outcomes(
    tmp_path: Path,
    scenario_name: str,
    expected_state: RunState,
    expected_resume: ResumeDisposition,
    unknown_attempts: int,
) -> None:
    composition = require("oamb.cli")
    fake_validation = require("oamb.artifacts.validation.fake")
    completed = composition.run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=f"fake-{scenario_name}",
        scenario=FakeRunScenario(scenario_name),
    )

    validation = fake_validation.validate_fake_capsule(completed.capsule_root)
    source_documents = fake_validation.load_fake_capsule(completed.capsule_root).contracts
    attempts = tuple(
        document
        for document in source_documents
        if getattr(document, "schema_name", None) == "attempt_record"
    )

    assert validation.disposition == ValidationDisposition.VALIDATED
    assert completed.run_record.state == expected_state
    assert completed.run_record.resume_disposition == expected_resume
    assert sum(attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME for attempt in attempts) == (
        unknown_attempts
    )
    if scenario_name == "unknown_outcome":
        assert len(attempts) == 1


def test_report_build_is_deterministic_self_contained_and_committed_last(
    tmp_path: Path,
) -> None:
    composition = require("oamb.cli")
    fake_validation = require("oamb.artifacts.validation.fake")
    report = require("oamb.reporting.html")
    completed = composition.run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-report",
    )
    validation = fake_validation.validate_fake_capsule(completed.capsule_root)

    first = report.build_fake_report(
        completed.capsule_root,
        validation,
        output_root=tmp_path / "reports",
        audience="public",
    )
    second = report.build_fake_report(
        completed.capsule_root,
        validation,
        output_root=tmp_path / "reports",
        audience="public",
    )

    assert first.derivation_id == second.derivation_id
    assert first.report_path.read_bytes() == second.report_path.read_bytes()
    verify_published_directory(first.final_directory, "derived-manifest.json")
    html = first.report_path.read_text(encoding="utf-8")
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "Content-Security-Policy" in html
    assert "prefers-color-scheme: dark" in html
    assert "Execution completeness" in html
    assert "http://" not in html and "https://" not in html
    assert not re.search(r"<(?:img|link|iframe)\b", html, re.IGNORECASE)


def test_invalid_capsule_requires_explicit_restricted_diagnostic_report(
    tmp_path: Path,
) -> None:
    composition = require("oamb.cli")
    fake_validation = require("oamb.artifacts.validation.fake")
    report = require("oamb.reporting.html")
    completed = composition.run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-diagnostic",
    )
    run_path = completed.capsule_root / "source" / "run" / "fake-diagnostic.json"
    run_path.write_bytes(run_path.read_bytes() + b"\n")
    validation = fake_validation.validate_fake_capsule(completed.capsule_root)

    with pytest.raises(fake_validation.EvidenceNotValidatedError):
        report.build_fake_report(
            completed.capsule_root,
            validation,
            output_root=tmp_path / "reports",
            audience="public",
        )
    diagnostic = report.build_fake_report(
        completed.capsule_root,
        validation,
        output_root=tmp_path / "reports",
        audience="public",
        diagnostic=True,
    )

    html = diagnostic.report_path.read_text(encoding="utf-8")
    assert "DIAGNOSTIC — INVALID EVIDENCE" in html
    assert "Final score" not in html
    assert "Winner" not in html
