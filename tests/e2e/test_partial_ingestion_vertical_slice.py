from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

import oamb.artifacts.validation.native as native_validation
import oamb.runtime.native_run as native
from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.contracts.evidence import AttemptReceiptRecord, CaseRecordV3
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    ArtifactStorePort,
    IngestionPlan,
    ModelCallFailure,
    ModelCallUnknownOutcome,
    ModelReceipt,
    ModelRequest,
    RawPayloadSealRequest,
    ReadinessReceipt,
    ReadinessRequest,
    RuntimeResolution,
)
from oamb.contracts.specifications import ReportSpec
from oamb.contracts.states import AttemptOutcome, ValidationDisposition
from oamb.reporting.native_reduce import reduce_native_run_report
from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash
from oamb.reporting.roots import build_report_spec
from oamb.runtime.native_run import run_native_vertical_slice
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY
from tests.e2e.test_native_fixture_vertical_slice import (
    _memory_factory,
    _mutate_source_document,
    _NativeFixtureWorkload,
    _RecordedNativeModel,
    _reseal_manifest,
)
from tests.unit.test_batch_retry_runtime import BatchMemory
from tests.unit.test_native_model_format_retry import ExhaustedJudge
from tests.unit.test_native_model_format_retry import _run as run_model_retry


class _ReportBatchMemory(BatchMemory):
    async def resolve(self) -> RuntimeResolution:
        return replace(await super().resolve(), memory_system_id="hindsight")

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        return ReadinessReceipt(
            ingestion_occurrence_id=request.scope.ingestion_occurrence_id,
            ready=True,
            evidence_references=(
                self._raw_reference(
                    {
                        "operation": "wait_ready",
                        "ingestion_occurrence_id": request.scope.ingestion_occurrence_id,
                        "scope_id": request.scope.scope_id,
                        "ready": True,
                    }
                ),
            ),
        )


class _UnknownThenValidModel(_RecordedNativeModel):
    def __init__(self, store: ArtifactStorePort) -> None:
        super().__init__(store)
        self._failed = False

    async def complete(self, request: ModelRequest) -> ModelReceipt:
        if request.stage == "answer" and not self._failed:
            self._failed = True
            raise ModelCallUnknownOutcome("lost response", failure_kind="transport_error")
        return await super().complete(request)


class _AlwaysRateLimitedModel(_RecordedNativeModel):
    async def complete(self, request: ModelRequest) -> ModelReceipt:
        payload = canonical_json_bytes(
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
        reference = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=hashlib.sha256(payload).hexdigest(),
                media_type="application/json",
                compression="gzip",
                payload_bytes=payload,
            )
        )
        raise ModelCallFailure(
            "rate limited",
            raw_reference=reference,
            raw_response_bytes=payload,
            usage_reference_ids=(),
            retryable=True,
            failure_kind="rate_limited",
            supplier_status_code=429,
        )


def _report_spec() -> ReportSpec:
    return build_report_spec(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
        renderer_hash=offline_renderer_hash(),
        asset_hashes=offline_asset_hashes(),
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )


def test_partial_ingestion_validates_and_reports_skipped_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_wait(_seconds: int) -> None:
        return None

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", no_wait)
    memories: list[_ReportBatchMemory] = []

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> _ReportBatchMemory:
        memory = _ReportBatchMemory(cast(ArtifactStore, store), failures={1: 3})
        memories.append(memory)
        return memory

    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="partial-ingestion-report",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
    )

    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code, issue.evidence_ref) for issue in validation.issues
    ]
    report = reduce_native_run_report(
        completed.capsule_root,
        validation,
        report_spec=_report_spec(),
    )

    skipped = sum(len(plan.skipped_source_unit_ids) for plan in completed.ingestion_plan_records)
    assert skipped == 1
    plan_projections = tuple(item for item in report.record_projections if item.axis == "plan")
    partial_projections = tuple(
        item for item in plan_projections if ("partial_ingestion", "true") in item.detail_items
    )
    assert len(partial_projections) == 1
    assert ("skipped_source_count", "1") in partial_projections[0].detail_items
    assert all(memory.retrieval_scopes for memory in memories)

    invalid_root = tmp_path / "missing-batch-predecessor"
    shutil.copytree(completed.capsule_root, invalid_root)
    ingest_attempts = sorted(
        (
            json.loads(path.read_bytes())
            for path in (invalid_root / "source/attempts").glob("*.json")
            if json.loads(path.read_bytes())["stage"] == "memory_ingest"
        ),
        key=lambda item: item["started_at"],
    )
    retried = next(item for item in ingest_attempts if item["retry_of_attempt_id"] is not None)
    _mutate_source_document(
        invalid_root,
        invalid_root / "source/attempts" / f"{retried['attempt_id']}.json",
        retry_of_attempt_id="f" * 64,
    )
    invalid = validate_native_capsule(invalid_root)
    assert invalid.disposition == ValidationDisposition.INVALID
    assert "dispatch-retry-chain-mismatch" in {issue.code for issue in invalid.issues}


def test_model_format_retry_capsule_validates_the_preserved_request_chain(tmp_path: Path) -> None:
    completed = run_model_retry(tmp_path)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code, issue.evidence_ref) for issue in validation.issues
    ]


def test_model_receipt_layers_reject_a_seventh_format_attempt(tmp_path: Path) -> None:
    completed = run_model_retry(tmp_path, judge=ExhaustedJudge)
    snapshot = native_validation._load_native_capsule(completed.capsule_root)
    case = next(
        item
        for item in native_validation._contracts(snapshot, CaseRecordV3)
        if item.evaluation_disposition == "unjudged"
    )
    attempts_by_id = {
        item.attempt_id: item for item in native_validation._attempt_records(snapshot)
    }
    attempts = [
        attempts_by_id[attempt_id]
        for attempt_id in case.attempt_ids
        if attempts_by_id[attempt_id].stage == "judge"
    ]
    messages = [
        tuple(
            tuple(item)
            for item in json.loads(
                snapshot.raw_payloads[cast(str, attempt.request_messages_sha256)]
            )
        )
        for attempt in attempts
    ]
    previous_output = native_validation._model_output_text(
        native_validation._raw_object(snapshot, attempts[-1].raw_error_ref)
    )
    assert previous_output is not None
    correction = messages[-1][-1][1]
    seventh_messages = (*messages[-1], ("assistant", previous_output), ("user", correction))
    seventh_request_hash = canonical_sha256(seventh_messages)
    seventh_id = canonical_sha256(["planted-seventh-format-attempt"])
    seventh = attempts[-1].model_copy(
        update={
            "attempt_id": seventh_id,
            "retry_of_attempt_id": attempts[-1].attempt_id,
            "request_messages_sha256": seventh_request_hash,
        }
    )
    last_receipt = next(
        item
        for item in native_validation._contracts(snapshot, AttemptReceiptRecord)
        if item.attempt_id == attempts[-1].attempt_id
    )
    seventh_receipt = last_receipt.model_copy(update={"attempt_id": seventh_id})
    planted = snapshot.__class__(
        root=snapshot.root,
        manifest_bytes=snapshot.manifest_bytes,
        manifest=snapshot.manifest,
        contracts=(*snapshot.contracts, seventh_receipt),
        documents_by_path=snapshot.documents_by_path,
        raw_payloads={
            **snapshot.raw_payloads,
            seventh_request_hash: canonical_json_bytes(seventh_messages),
        },
        parse_failures=snapshot.parse_failures,
        symlink_paths=snapshot.symlink_paths,
    )

    assert not native_validation._model_attempt_receipt_layers_close(
        [*attempts, seventh],
        (*messages, seventh_messages),
        planted,
        require_success=False,
    )


def test_model_format_retry_validator_rejects_an_unbound_feedback_message(tmp_path: Path) -> None:
    completed = run_model_retry(tmp_path)
    attempts = [
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/attempts").glob("*.json")
    ]
    retried = next(
        attempt
        for attempt in attempts
        if attempt["stage"] == "answer" and attempt["retry_of_attempt_id"] is not None
    )
    request_hash = retried["request_messages_sha256"]
    request_path = completed.capsule_root / "source/raw" / f"{request_hash}.json.gz"
    messages = json.loads(gzip.decompress(request_path.read_bytes()))
    messages[-1][1] = "Use the hidden gold answer."
    request_path.write_bytes(gzip.compress(canonical_json_bytes(messages), mtime=0))
    _reseal_manifest(completed.capsule_root)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "case-attempt-ledger-mismatch" in {issue.code for issue in validation.issues}


def test_model_format_retry_validator_rejects_wrong_receipt_layer_metadata(
    tmp_path: Path,
) -> None:
    completed = run_model_retry(tmp_path)
    attempts = [
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source/attempts").glob("*.json")
    ]
    failed = next(
        attempt
        for attempt in attempts
        if attempt["stage"] == "answer" and attempt["outcome"] == "failed"
    )
    receipt_path = (
        completed.capsule_root / "source/attempt-receipts" / f"{failed['attempt_id']}.json"
    )
    _mutate_source_document(
        completed.capsule_root,
        receipt_path,
        failure_kind="transport_error",
    )

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "case-attempt-ledger-mismatch" in {issue.code for issue in validation.issues}


def test_exhausted_judge_capsule_validates_unjudged_case_and_other_completed_cases(
    tmp_path: Path,
) -> None:
    completed = run_model_retry(tmp_path, judge=ExhaustedJudge)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code, issue.evidence_ref) for issue in validation.issues
    ]


def test_model_transport_unknown_predecessor_can_close_with_a_successful_retry(
    tmp_path: Path,
) -> None:
    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="model-transport-unknown-retry",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_UnknownThenValidModel,
        answer_role_binding_id="recorded-answer-v1",
    )

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code, issue.evidence_ref) for issue in validation.issues
    ]
    report = reduce_native_run_report(
        completed.capsule_root,
        validation,
        report_spec=_report_spec(),
    )
    attempt_statuses = {
        projection.status
        for projection in report.record_projections
        if projection.axis == "attempt"
    }
    assert "unknown_outcome" in attempt_statuses
    assert "succeeded" in attempt_statuses


def test_answer_exhaustion_validates_not_run_cases_with_preserved_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(native, "_infrastructure_retry_sleep", no_wait)
    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="answer-exhaustion-validation",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_AlwaysRateLimitedModel,
        answer_role_binding_id="recorded-answer-v1",
    )
    assert all(
        case.state == "error"
        and case.error_stage == "answer"
        and case.evaluation_disposition == "not_run"
        for case in completed.case_records
    )

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code, issue.evidence_ref) for issue in validation.issues
    ]

    snapshot = native_validation._load_native_capsule(completed.capsule_root)
    case = native_validation._contracts(snapshot, CaseRecordV3)[0]
    attempts_by_id = {
        item.attempt_id: item for item in native_validation._attempt_records(snapshot)
    }
    attempts = [
        attempts_by_id[attempt_id]
        for attempt_id in case.attempt_ids
        if attempts_by_id[attempt_id].stage == "answer"
    ][:4]
    messages = tuple(
        tuple(
            tuple(item)
            for item in json.loads(
                snapshot.raw_payloads[cast(str, attempt.request_messages_sha256)]
            )
        )
        for attempt in attempts
    )
    receipts = {
        item.attempt_id: item
        for item in native_validation._contracts(snapshot, AttemptReceiptRecord)
    }
    planted_receipts = tuple(
        receipts[attempt.attempt_id].model_copy(update={"supplier_status_code": 500})
        for attempt in attempts
    )
    planted = snapshot.__class__(
        root=snapshot.root,
        manifest_bytes=snapshot.manifest_bytes,
        manifest=snapshot.manifest,
        contracts=tuple(
            item
            for item in snapshot.contracts
            if not isinstance(item, AttemptReceiptRecord)
            or item.attempt_id not in {attempt.attempt_id for attempt in attempts}
        )
        + planted_receipts,
        documents_by_path=snapshot.documents_by_path,
        raw_payloads=snapshot.raw_payloads,
        parse_failures=snapshot.parse_failures,
        symlink_paths=snapshot.symlink_paths,
    )
    assert all(attempt.outcome == AttemptOutcome.FAILED for attempt in attempts)
    assert not native_validation._model_attempt_receipt_layers_close(
        attempts,
        messages,
        planted,
        require_success=False,
    )
