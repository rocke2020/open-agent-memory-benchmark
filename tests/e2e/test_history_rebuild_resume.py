from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

import oamb.runtime.native_run as native
from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.native import validate_native_capsule, validate_partition_capsule
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import ArtifactStorePort, IngestionPlan
from oamb.contracts.specifications import INFRASTRUCTURE_RETRY_POLICY_HASH, ModelRole
from oamb.contracts.states import ValidationDisposition
from oamb.memory_systems.hindsight import HindsightAdapter
from oamb.reporting.native_reduce import reduce_native_run_report
from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash
from oamb.reporting.roots import build_report_spec
from oamb.runtime.case_partition import build_case_partition_spec
from oamb.workloads.longmemeval import LME_JUDGE_PROMPT_PACK_ID
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY
from tests.e2e.test_history_rebuild_vertical_slice import (
    _RebuildingHindsightService,
    _two_source_workload,
)
from tests.e2e.test_native_recorded_exact_profile import (
    ANSWER_ROLE_ID,
    RUNTIME_BINDING_HASH,
    _model_factory,
)


def _run(
    tmp_path: Path,
    *,
    run_id: str,
    occurrence_run_id: str,
    fail_first_history: bool,
    failure_history_ordinal: int = 1,
    recovery_parts: tuple[Path, ...] = (),
) -> tuple[native.NativeRunArtifacts, _RebuildingHindsightService]:
    workload = _two_source_workload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    plans = workload.iter_ingestion_plans(manifest)
    cases = workload.iter_case_plans(manifest)
    partition = build_case_partition_spec(
        run_id=run_id,
        resolved_plan_hash=canonical_sha256(["history-resume-plan"]),
        cell_spec_hash=canonical_sha256(["history-resume-cell"]),
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=cases,
        requested_case_manifest_entry_ids=tuple(item.case_manifest_entry_id for item in cases),
        budget_policy_hash=canonical_sha256(["history-resume-budget"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    service = _RebuildingHindsightService(
        occurrence_run_id,
        plans,
        fail_first_history=fail_first_history,
        old_partial_count=1,
        journal=tmp_path / f"{run_id}-requests.jsonl",
    )
    if failure_history_ordinal != 1:
        banks = service.banks
        service.banks = (
            banks[failure_history_ordinal - 1],
            *(bank for index, bank in enumerate(banks, 1) if index != failure_history_ordinal),
        )

    def memory_factory(
        store: ArtifactStorePort, frozen_plans: tuple[IngestionPlan, ...]
    ) -> HindsightAdapter:
        assert frozen_plans == (service.plan,)
        return HindsightAdapter(
            store=store,
            base_url="https://hindsight.example",
            extraction_model="fixture-extractor",
            runtime_binding_hash=RUNTIME_BINDING_HASH,
            transport=httpx.MockTransport(service),
            internal_retry_count=0,
        )

    completed = native.run_native_vertical_slice(
        output_root=tmp_path,
        run_id=run_id,
        adapter_profile_id="hindsight-rest-v1",
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.ANSWER,
            binding_id=ANSWER_ROLE_ID,
            model="fixture-answer-model",
            output="An ordinary incorrect answer.",
            thinking_effort="low",
        ),
        answer_role_binding_id=ANSWER_ROLE_ID,
        judge_model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.JUDGE,
            binding_id=LME_JUDGE_PROMPT_PACK_ID,
            model="fixture-judge-model",
            output="no",
            thinking_effort="high",
        ),
        judge_role_binding_id=LME_JUDGE_PROMPT_PACK_ID,
        partition=partition,
        recovery_parts=recovery_parts,
    )
    for line in service.journal.read_text().splitlines():
        event = json.loads(line)
        if event["kind"] == "create":
            service.created_bank_ids.add(event["bank"])
        elif event["kind"] == "retain":
            service.retains.append((event["bank"], event["item"]))
        elif event["kind"] == "recall":
            service.recalls.append(event["bank"])
    return completed, service


def test_interrupted_pending_history_retry_resumes_exact_fresh_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorded HTTP/model fixtures exercise public recovery without provider calls."""

    predecessor_run_id = "history-resume-predecessor"

    async def interrupt_backoff(seconds: int) -> None:
        assert seconds == 1
        raise native.NativeRunInterrupted("fixture interruption during durable backoff")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", interrupt_backoff)
    with pytest.raises(native.NativeRunInterrupted, match="durable backoff"):
        _run(
            tmp_path / "runs",
            run_id=predecessor_run_id,
            occurrence_run_id=predecessor_run_id,
            fail_first_history=True,
        )
    predecessor_root = tmp_path / "runs" / predecessor_run_id
    events = tuple(
        json.loads(path.read_bytes())
        for path in (predecessor_root / "source/history-retries").glob("*.json")
    )
    assert len(events) == 1 and events[0]["retry_scheduled"] is True
    predecessor_validation = validate_partition_capsule(predecessor_root)
    assert predecessor_validation.disposition == ValidationDisposition.VALIDATED, (
        predecessor_validation.issues
    )

    resumed_wait = tmp_path / "resumed-backoff"

    async def resume_backoff(seconds: int) -> None:
        resumed_wait.write_text(str(seconds), encoding="utf-8")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", resume_backoff)
    resumed, service = _run(
        tmp_path / "runs",
        run_id="history-resume-successor",
        occurrence_run_id=predecessor_run_id,
        fail_first_history=False,
        recovery_parts=(predecessor_root,),
    )

    assert resumed_wait.read_text(encoding="utf-8") == str(events[0]["backoff_seconds"])
    assert service.created_bank_ids == {service.banks[1]}
    retained = tuple(item for bank, item in service.retains if bank == service.banks[1])
    assert tuple(item["document_id"] for item in retained) == tuple(
        item.source_unit_id for item in service.plan.ordered_source_units
    )
    assert service.recalls == [service.banks[1]]
    assert resumed.ingestion_plan_records[0].ingestion_occurrence_id == service.occurrences[1]
    assert resumed.case_records[0].ingestion_occurrence_id == service.occurrences[1]
    validation = validate_native_capsule(resumed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    run = json.loads(next((resumed.capsule_root / "source/run").glob("*.json")).read_bytes())
    assert run["ingestion_occurrence_ids"] == [service.occurrences[1]]
    histories = tuple(
        json.loads(path.read_bytes())
        for path in (resumed.capsule_root / "source/history-attempts").glob("*.json")
    )
    assert tuple(item["history_attempt_ordinal"] for item in histories) == (2,)
    assert histories[0]["previous_retry_event_id"] == events[0]["history_retry_event_id"]
    carries = tuple(
        json.loads(path.read_bytes())
        for path in (resumed.capsule_root / "source/history-carries").glob("*.json")
    )
    assert len(carries) == 1
    binding = carries[0]["source_part_bindings"][0]
    embedded = resumed.capsule_root / binding["embedded_root"]
    embedded_histories = tuple(
        json.loads(path.read_bytes())
        for path in (embedded / "source/history-attempts").glob("*.json")
    )
    assert tuple(item["ingestion_occurrence_id"] for item in embedded_histories) == (
        service.occurrences[0],
    )


def test_nested_pending_history_retry_resumes_ordinal_three_without_reset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predecessor_run_id = "history-nested-a"

    async def stop_after_first_failure(seconds: int) -> None:
        assert seconds == 1
        raise native.NativeRunInterrupted("stop A")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", stop_after_first_failure)
    with pytest.raises(native.NativeRunInterrupted, match="stop A"):
        _run(
            tmp_path / "runs",
            run_id=predecessor_run_id,
            occurrence_run_id=predecessor_run_id,
            fail_first_history=True,
        )
    part_a = tmp_path / "runs" / predecessor_run_id

    waits_b = tmp_path / "waits-b"

    async def stop_after_second_failure(seconds: int) -> None:
        with waits_b.open("a", encoding="utf-8") as stream:
            stream.write(f"{seconds}\n")
        if seconds == 2:
            raise native.NativeRunInterrupted("stop B")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", stop_after_second_failure)
    with pytest.raises(native.NativeRunInterrupted, match="stop B"):
        _run(
            tmp_path / "runs",
            run_id="history-nested-b",
            occurrence_run_id=predecessor_run_id,
            fail_first_history=True,
            failure_history_ordinal=2,
            recovery_parts=(part_a,),
        )
    assert waits_b.read_text(encoding="utf-8").splitlines() == ["1", "2"]
    part_b = tmp_path / "runs/history-nested-b"
    assert validate_partition_capsule(part_b).disposition == ValidationDisposition.VALIDATED

    wait_c = tmp_path / "wait-c"

    async def complete_pending_wait(seconds: int) -> None:
        wait_c.write_text(str(seconds), encoding="utf-8")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", complete_pending_wait)
    completed, service = _run(
        tmp_path / "runs",
        run_id="history-nested-c",
        occurrence_run_id="history-nested-b",
        fail_first_history=False,
        recovery_parts=(part_b,),
    )
    assert wait_c.read_text(encoding="utf-8") == "2"
    assert completed.ingestion_plan_records[0].ingestion_occurrence_id == service.occurrences[2]
    assert completed.case_records[0].ingestion_occurrence_id == service.occurrences[2]
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    report = reduce_native_run_report(
        completed.capsule_root,
        validation,
        report_spec=build_report_spec(
            report_kind="run",
            audience="public",
            preview_max_field_bytes=4096,
            preview_total_bytes=65536,
            display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
            renderer_hash=offline_renderer_hash(),
            asset_hashes=offline_asset_hashes(),
            export_profile_selector_id="public-run-v1",
            export_profile_selector_version=1,
        ),
    )
    physical_attempt_ids = {
        json.loads(path.read_bytes())["attempt_id"]
        for path in completed.capsule_root.rglob("source/attempts/*.json")
    }
    assert set(report.attempt_ids) == physical_attempt_ids
    assert report.ingestion_occurrence_ids == (service.occurrences[2],)
    physical_accounting_ids = {
        json.loads(path.read_bytes())[identity]
        for directory, identity in (
            ("usage", "usage_record_id"),
            ("resources", "resource_record_id"),
            ("costs", "cost_record_id"),
        )
        for path in completed.capsule_root.rglob(f"source/{directory}/*.json")
    }
    assert {
        record_id for line in report.measurement_lines for record_id in line.source_record_ids
    } == physical_accounting_ids
    monkeypatch.chdir(tmp_path)
    local_report = reduce_native_run_report(
        completed.capsule_root.relative_to(tmp_path),
        validation,
        report_spec=build_report_spec(
            report_kind="run",
            audience="local",
            preview_max_field_bytes=4096,
            preview_total_bytes=65536,
            display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
            renderer_hash=offline_renderer_hash(),
            asset_hashes=offline_asset_hashes(),
            export_profile_selector_id="local-run-v1",
            export_profile_selector_version=1,
        ),
    )
    failed_attempts = tuple(
        item
        for item in local_report.record_projections
        if item.axis == "attempt" and item.status == "failed"
    )
    assert len(failed_attempts) == 2
    assert all(item.raw_evidence_present for item in failed_attempts)
    assert sum(len(item.display_previews) for item in failed_attempts) == 1
    local_history = json.loads(
        next((completed.capsule_root / "source/history-attempts").glob("*.json")).read_bytes()
    )
    assert local_history["history_attempt_ordinal"] == 3
    assert local_history["previous_retry_event_id"] is not None
    assert local_history["admission_claim_raw_ref"] is not None
    assert (
        completed.capsule_root
        / "source/raw"
        / f"{local_history['admission_claim_raw_ref']}.json.gz"
    ).is_file()
    carry = json.loads(
        next((completed.capsule_root / "source/history-carries").glob("*.json")).read_bytes()
    )
    assert carry["allowances"][0]["consumed_retries"] == 2
    assert carry["allowances"][0]["next_history_attempt_ordinal"] == 3
    nested_histories = tuple(
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/parts").rglob(
            "source/history-attempts/*.json"
        )
    )
    assert {item["history_attempt_ordinal"] for item in nested_histories} == {1, 2}
    second = next(item for item in nested_histories if item["history_attempt_ordinal"] == 2)
    assert second["admission_claim_raw_ref"] is not None

    dedup_wait = tmp_path / "wait-c-dedup"

    async def complete_dedup_wait(seconds: int) -> None:
        dedup_wait.write_text(str(seconds), encoding="utf-8")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", complete_dedup_wait)
    deduped, _service = _run(
        tmp_path / "runs",
        run_id="history-nested-c-dedup",
        occurrence_run_id="history-nested-b",
        fail_first_history=False,
        recovery_parts=(part_a, part_b),
    )
    assert dedup_wait.read_text(encoding="utf-8") == "2"
    dedup_validation = validate_native_capsule(deduped.capsule_root)
    assert dedup_validation.disposition == ValidationDisposition.VALIDATED, dedup_validation.issues
    embedded_manifest_files = tuple(
        path for path in (deduped.capsule_root / "source/parts").rglob("capsule-manifest.json")
    )
    a_capsule_id = json.loads((part_a / "capsule-manifest.json").read_bytes())["capsule_id"]
    repeated_a = tuple(
        path
        for path in embedded_manifest_files
        if json.loads(path.read_bytes())["capsule_id"] == a_capsule_id
    )
    assert len(repeated_a) == 2
    assert len({hashlib.sha256(path.read_bytes()).hexdigest() for path in repeated_a}) == 1


def test_exhausted_nested_history_cannot_reset_retry_allowance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def stop_a(_seconds: int) -> None:
        raise native.NativeRunInterrupted("stop exhausted A")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", stop_a)
    with pytest.raises(native.NativeRunInterrupted):
        _run(
            tmp_path / "runs",
            run_id="history-exhausted-a",
            occurrence_run_id="history-exhausted-a",
            fail_first_history=True,
        )
    part_a = tmp_path / "runs/history-exhausted-a"

    async def stop_b(seconds: int) -> None:
        if seconds == 2:
            raise native.NativeRunInterrupted("stop exhausted B")

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", stop_b)
    with pytest.raises(native.NativeRunInterrupted):
        _run(
            tmp_path / "runs",
            run_id="history-exhausted-b",
            occurrence_run_id="history-exhausted-a",
            fail_first_history=True,
            failure_history_ordinal=2,
            recovery_parts=(part_a,),
        )
    part_b = tmp_path / "runs/history-exhausted-b"

    async def pending_c(seconds: int) -> None:
        assert seconds == 2

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", pending_c)
    with pytest.raises(RuntimeError, match="retry limit exhausted"):
        _run(
            tmp_path / "runs",
            run_id="history-exhausted-c",
            occurrence_run_id="history-exhausted-b",
            fail_first_history=True,
            failure_history_ordinal=3,
            recovery_parts=(part_b,),
        )
    exhausted = tmp_path / "runs/history-exhausted-c"
    exhausted_validation = validate_partition_capsule(exhausted)
    assert exhausted_validation.disposition == ValidationDisposition.VALIDATED, (
        exhausted_validation.issues
    )
    exhausted_events = tuple(
        json.loads(path.read_bytes())
        for path in (exhausted / "source/history-retries").glob("*.json")
    )
    assert len(exhausted_events) == 1
    assert exhausted_events[0]["retry_scheduled"] is False
    assert exhausted_events[0]["failed_history_attempt_ordinal"] == 3

    with pytest.raises(ValueError, match="no single pending scheduled retry"):
        _run(
            tmp_path / "runs",
            run_id="history-exhausted-reset",
            occurrence_run_id="history-exhausted-c",
            fail_first_history=False,
            recovery_parts=(exhausted,),
        )
    assert not (tmp_path / "runs/history-exhausted-reset-requests.jsonl").exists()
