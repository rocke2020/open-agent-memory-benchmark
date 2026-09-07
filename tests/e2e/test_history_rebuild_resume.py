from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import oamb.artifacts.validation.native as native_validation
import oamb.runtime.native_run as native
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.contracts.evidence import (
    CapsuleCompositionPartBinding,
    CapsuleManifest,
    HistoryAttemptRecord,
    HistoryRetryAllowance,
    HistoryRetryCarryRecord,
    HistoryRetryEvent,
    capsule_composition_part_binding_hash,
    history_attempt_id,
    history_retry_carry_id,
    history_retry_event_id,
)
from oamb.contracts.ids import canonical_sha256, ingestion_occurrence_id
from oamb.contracts.specifications import CasePartitionSpec
from oamb.contracts.states import RunState, ValidationDisposition
from tests.e2e.test_history_rebuild_vertical_slice import _run


def test_interrupted_batch_retry_rebuilds_in_an_explicit_fresh_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    with pytest.raises(native.NativeRunInterrupted):
        _run(
            runs,
            monkeypatch,
            run_id="batch-retry-predecessor",
            fail_first_history=True,
            stop_during_backoff=True,
        )
    predecessor = runs / "batch-retry-predecessor"

    completed, service, waits = _run(
        runs,
        monkeypatch,
        run_id="batch-retry-successor",
        fail_first_history=False,
        recovery_parts=(predecessor,),
    )

    assert validate_native_capsule(completed.capsule_root).disposition == (
        ValidationDisposition.VALIDATED
    )
    assert service.created_bank_ids == {service.banks[0]}
    assert waits == []


def test_native_validation_rejects_legacy_retry_event_allowance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "legacy-runs"
    with pytest.raises(native.NativeRunInterrupted):
        _run(
            runs,
            monkeypatch,
            run_id="legacy-retry-source",
            fail_first_history=True,
            stop_during_backoff=True,
        )
    source = runs / "legacy-retry-source"
    source_snapshot = native_validation._load_native_capsule(source)
    assert source_snapshot.manifest is not None
    partition = native_validation._contracts(source_snapshot, CasePartitionSpec)[0]
    source_history = native_validation._contracts(source_snapshot, HistoryAttemptRecord)[0]
    history_fields = source_history.model_dump(mode="python", exclude={"history_attempt_id"})
    history_fields.update(
        status="retryable_failed_settled",
        max_retries_per_operation=2,
        settlement_basis="hindsight_sync_extraction_drained_v1",
        settlement_evidence_refs=(canonical_sha256(["legacy-settlement"]),),
        settlement_status_code=500,
        internal_retry_count=10,
        failure_kind="supplier_connection",
        ingestion_plan_record_hash=None,
    )
    history = HistoryAttemptRecord.model_validate(
        {"history_attempt_id": history_attempt_id(history_fields), **history_fields}
    )
    event_fields = {
        "schema_name": "history_retry_event",
        "schema_version": 1,
        "run_id": source_snapshot.manifest.run_id,
        "ingestion_plan_id": history.ingestion_plan_id,
        "failed_history_attempt_id": history.history_attempt_id,
        "failed_history_attempt_ordinal": 1,
        "retry_ordinal": 1,
        "retry_scheduled": True,
        "successor_ingestion_occurrence_id": ingestion_occurrence_id(
            "legacy-successor",
            history.memory_system_id,
            history.ingestion_plan_id,
            history_attempt_ordinal=2,
        ),
        "successor_execution_run_id": "legacy-successor",
        "successor_history_attempt_ordinal": 2,
        "retry_policy_hash": history.retry_policy_hash,
        "max_retries_per_operation": 2,
        "backoff_seconds": 1,
        "observed_at": history.ended_at,
    }
    event = HistoryRetryEvent.model_validate(
        {"history_retry_event_id": history_retry_event_id(event_fields), **event_fields}
    )

    wrapper_root = tmp_path / "legacy-wrapper"
    embedded = wrapper_root / "source" / "parts" / source_snapshot.manifest.capsule_id
    shutil.copytree(source, embedded)
    binding_fields = {
        "schema_name": "capsule_composition_part_binding",
        "schema_version": 1,
        "capsule_id": source_snapshot.manifest.capsule_id,
        "run_id": source_snapshot.manifest.run_id,
        "manifest_sha256": hashlib.sha256(source_snapshot.manifest_bytes).hexdigest(),
        "partition_id": partition.partition_id,
        "embedded_root": f"source/parts/{source_snapshot.manifest.capsule_id}",
        "run_state": RunState.ABORTED,
    }
    binding = CapsuleCompositionPartBinding.model_validate(
        {
            "part_binding_hash": capsule_composition_part_binding_hash(binding_fields),
            **binding_fields,
        }
    )
    allowance = HistoryRetryAllowance(
        ingestion_plan_id=history.ingestion_plan_id,
        next_history_attempt_ordinal=event.successor_history_attempt_ordinal or 2,
        execution_run_id=event.successor_execution_run_id or "legacy-successor",
        previous_retry_event_id=event.history_retry_event_id,
        predecessor_history_attempt_id=history.history_attempt_id,
        consumed_retries=event.retry_ordinal,
    )
    carry_fields = {
        "schema_name": "history_retry_carry_record",
        "schema_version": 1,
        "run_id": "legacy-wrapper",
        "source_part_bindings": (binding,),
        "allowances": (allowance,),
        "retry_policy_hash": partition.retry_policy_hash,
        "max_retries_per_operation": history.max_retries_per_operation,
    }
    carry = HistoryRetryCarryRecord.model_validate(
        {"carry_record_id": history_retry_carry_id(carry_fields), **carry_fields}
    )
    wrapper_manifest = CapsuleManifest(
        capsule_id=canonical_sha256(["legacy-wrapper-capsule"]),
        run_id="legacy-wrapper",
        run_spec_hash=canonical_sha256(["legacy-wrapper-run-spec"]),
        source_entries=(),
        source_manifest_hash=canonical_sha256(["legacy-wrapper-source"]),
    )
    wrapper_snapshot = native_validation._NativeCapsuleSnapshot(
        root=wrapper_root,
        manifest_bytes=b"{}",
        manifest=wrapper_manifest,
        contracts=(partition.model_copy(update={"run_id": "legacy-wrapper"}), carry),
        documents_by_path={},
        raw_payloads={},
        parse_failures=(),
        symlink_paths=(),
    )
    synthetic_predecessor = native_validation._NativeCapsuleSnapshot(
        root=source_snapshot.root,
        manifest_bytes=source_snapshot.manifest_bytes,
        manifest=source_snapshot.manifest,
        contracts=(
            *(
                item
                for item in source_snapshot.contracts
                if not isinstance(item, HistoryAttemptRecord)
            ),
            history,
            event,
        ),
        documents_by_path=source_snapshot.documents_by_path,
        raw_payloads=source_snapshot.raw_payloads,
        parse_failures=source_snapshot.parse_failures,
        symlink_paths=source_snapshot.symlink_paths,
    )
    original_carried_history_snapshots = native_validation._carried_history_snapshots

    def carried_history_snapshots(
        snapshot: native_validation._NativeCapsuleSnapshot,
        seen: dict[str, str] | None = None,
    ) -> tuple[native_validation._NativeCapsuleSnapshot, ...]:
        if snapshot.root == wrapper_root:
            return (synthetic_predecessor,)
        return original_carried_history_snapshots(snapshot, seen)

    monkeypatch.setattr(
        native_validation,
        "_carried_history_snapshots",
        carried_history_snapshots,
    )

    issues = native_validation._history_carry_issues(wrapper_snapshot)

    assert "history-carry-allowance-mismatch" in {issue.code for issue in issues}

    monkeypatch.setattr(
        native_validation,
        "_carried_history_snapshots",
        original_carried_history_snapshots,
    )
    fresh_allowance = HistoryRetryAllowance(
        ingestion_plan_id=history.ingestion_plan_id,
        next_history_attempt_ordinal=1,
        execution_run_id="legacy-wrapper",
        previous_retry_event_id=None,
        predecessor_history_attempt_id=None,
        consumed_retries=0,
    )
    drifted_carry_fields = {
        **carry_fields,
        "allowances": (fresh_allowance,),
        "execution_configuration_family_hash": canonical_sha256(["drifted-execution-family"]),
    }
    drifted_carry = HistoryRetryCarryRecord.model_validate(
        {
            "carry_record_id": history_retry_carry_id(drifted_carry_fields),
            **drifted_carry_fields,
        }
    )

    drifted_issues = native_validation._history_carry_issues(
        replace(
            wrapper_snapshot,
            contracts=(
                partition.model_copy(update={"run_id": "legacy-wrapper"}),
                drifted_carry,
            ),
        )
    )

    assert "history-carry-execution-configuration-family-mismatch" in {
        issue.code for issue in drifted_issues
    }
