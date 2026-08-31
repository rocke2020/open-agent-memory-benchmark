"""Create-only import of an aborted native capsule for same-run continuation."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
from oamb.contracts.evidence import (
    AttemptRecordV4,
    CapsuleManifest,
    CaseRecordV3,
    IngestionPlanRecordV2,
    RunLeaseRecord,
    RunRecord,
)
from oamb.contracts.ids import ingestion_occurrence_id
from oamb.contracts.ports import (
    CasePlan,
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionPlan,
    RawReferenceHandle,
    ScopeReceipt,
)
from oamb.contracts.states import RunState


@dataclass(frozen=True, slots=True)
class InitializedContinuation:
    run_id: str
    base_capsule_root: Path
    target_root: Path
    base_manifest: CapsuleManifest
    aborted_run: RunRecord


@dataclass(frozen=True, slots=True)
class ImportedAttemptAccounting:
    attempt_id: str
    usage_record_ids: tuple[str, ...]
    resource_record_id: str
    cost_record_id: str


@dataclass(frozen=True, slots=True)
class ImportedDispatch:
    ordinal: int
    receipt: IngestionDispatchReceipt
    accounting: ImportedAttemptAccounting
    task_id: str
    evidence_references: tuple[RawReferenceHandle, ...]


@dataclass(frozen=True, slots=True)
class PartialPlanContinuation:
    ingestion_plan_id: str
    ingestion_occurrence_id: str
    scope: ScopeReceipt
    scope_accounting: ImportedAttemptAccounting
    completed_dispatches: tuple[ImportedDispatch, ...]
    failed_attempt: AttemptRecordV4
    failed_task_id: str


@dataclass(frozen=True, slots=True)
class NativeContinuation:
    initialized: InitializedContinuation
    previous_lease: RunLeaseRecord
    completed_plan_records: tuple[IngestionPlanRecordV2, ...]
    completed_case_records: tuple[CaseRecordV3, ...]
    partial_plan: PartialPlanContinuation


def initialize_continuation_root(
    base_capsule_root: Path,
    output_root: Path,
) -> InitializedContinuation:
    """Verify an immutable base, then copy its source evidence without terminal records."""

    base = Path(base_capsule_root).resolve(strict=True)
    manifest_path = base / "capsule-manifest.json"
    manifest_bytes = read_regular_file(manifest_path)
    try:
        manifest = CapsuleManifest.model_validate_json(manifest_bytes)
    except ValueError as exc:
        raise ValueError("continuation base manifest is invalid") from exc

    expected_paths = {entry.relative_path for entry in manifest.source_entries}
    actual_paths = {
        path.relative_to(base).as_posix() for path in (base / "source").rglob("*") if path.is_file()
    }
    if actual_paths != expected_paths:
        raise ValueError("continuation base source inventory mismatch")

    verified_bytes: dict[str, bytes] = {}
    for entry in manifest.source_entries:
        relative = Path(entry.relative_path)
        if relative.is_absolute() or not relative.parts or relative.parts[0] != "source":
            raise ValueError("continuation base manifest path is outside source")
        source = base / relative
        if source.is_symlink():
            raise ValueError("continuation base contains a symbolic link")
        payload = read_regular_file(source)
        if hashlib.sha256(payload).hexdigest() != entry.sha256:
            raise ValueError(f"continuation base hash mismatch: {entry.relative_path}")
        verified_bytes[entry.relative_path] = payload

    run_relative_path = f"source/run/{manifest.run_id}.json"
    run_bytes = verified_bytes.get(run_relative_path)
    if run_bytes is None:
        raise ValueError("continuation base has no terminal run record")
    try:
        run = RunRecord.model_validate_json(run_bytes)
    except ValueError as exc:
        raise ValueError("continuation base run record is invalid") from exc
    if (
        run.run_id != manifest.run_id
        or run.run_spec_hash != manifest.run_spec_hash
        or run.state != RunState.ABORTED
    ):
        raise ValueError("continuation base must contain the matching aborted run")

    target = Path(output_root).absolute() / manifest.run_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"continuation target already exists: {target}")

    for relative_path, payload in verified_bytes.items():
        if relative_path == run_relative_path:
            continue
        atomic_write_bytes(
            target / relative_path,
            payload,
            trusted_root=target,
        )
    return InitializedContinuation(
        run_id=manifest.run_id,
        base_capsule_root=base,
        target_root=target,
        base_manifest=manifest,
        aborted_run=run,
    )


def load_openviking_continuation(
    initialized: InitializedContinuation,
    *,
    plans: tuple[IngestionPlan, ...],
    case_plans: tuple[CasePlan, ...],
) -> NativeContinuation:
    """Close one OpenViking continuation seed from imported immutable evidence."""

    source = initialized.target_root / "source"
    plan_records = tuple(
        IngestionPlanRecordV2.model_validate_json(read_regular_file(path))
        for path in sorted((source / "ingestion-plans").glob("*.json"))
    )
    case_records = tuple(
        CaseRecordV3.model_validate_json(read_regular_file(path))
        for path in sorted((source / "cases").glob("*.json"))
    )
    leases = tuple(
        RunLeaseRecord.model_validate_json(read_regular_file(path))
        for path in sorted((source / "run-leases").glob("*.json"))
    )
    if len(leases) != 1 or leases[0].lease_epoch != 1:
        raise ValueError("OpenViking continuation requires one base lease epoch")
    previous_lease = leases[0]
    if previous_lease.run_id != initialized.run_id:
        raise ValueError("OpenViking continuation lease has the wrong run")

    expected_occurrences = {
        plan.ingestion_plan_id: ingestion_occurrence_id(
            initialized.run_id,
            "openviking",
            plan.ingestion_plan_id,
        )
        for plan in plans
    }
    if len(plan_records) >= len(plans):
        raise ValueError("OpenViking continuation base has no unfinished plan")
    for record in plan_records:
        if (
            record.run_id != initialized.run_id
            or record.memory_system_id != "openviking"
            or record.adapter_profile_id != "openviking-session-rest-v1"
            or expected_occurrences.get(record.ingestion_plan_id) != record.ingestion_occurrence_id
        ):
            raise ValueError("OpenViking imported completed plan identity mismatch")
    complete_plan_ids = {record.ingestion_plan_id for record in plan_records}
    expected_complete_cases = {
        case_id
        for plan in plans
        if plan.ingestion_plan_id in complete_plan_ids
        for case_id in plan.ordered_case_manifest_entry_ids
    }
    if {record.case_manifest_entry_id for record in case_records} != expected_complete_cases:
        raise ValueError("OpenViking imported completed case inventory mismatch")
    if not expected_complete_cases <= {case.case_manifest_entry_id for case in case_plans}:
        raise ValueError("OpenViking imported case is outside the frozen workload")

    attempts = tuple(
        AttemptRecordV4.model_validate_json(read_regular_file(path))
        for path in sorted((source / "attempts").glob("*.json"))
    )
    incomplete = tuple(plan for plan in plans if plan.ingestion_plan_id not in complete_plan_ids)
    partial_candidates = []
    for plan in incomplete:
        occurrence_id = expected_occurrences[plan.ingestion_plan_id]
        plan_attempts = tuple(
            attempt
            for attempt in attempts
            if attempt.parent_id == occurrence_id and attempt.stage == "memory_ingest"
        )
        if plan_attempts:
            partial_candidates.append((plan, occurrence_id, plan_attempts))
    if len(partial_candidates) != 1:
        raise ValueError("OpenViking continuation requires exactly one partial plan")
    partial_plan, partial_occurrence, partial_attempts = partial_candidates[0]
    succeeded_by_ordinal = {
        attempt.ordinal: attempt
        for attempt in partial_attempts
        if attempt.outcome.value == "succeeded"
    }
    completed_count = len(succeeded_by_ordinal)
    if set(succeeded_by_ordinal) != set(range(1, completed_count + 1)):
        raise ValueError("OpenViking continuation success prefix is not contiguous")
    failed = tuple(
        attempt
        for attempt in partial_attempts
        if attempt.ordinal == completed_count + 1 and attempt.outcome.value == "failed"
    )
    if len(failed) != 1 or len(partial_attempts) != completed_count + 1:
        raise ValueError("OpenViking continuation requires one next-ordinal failure")

    scope_attempts = tuple(
        attempt
        for attempt in attempts
        if attempt.parent_id == partial_occurrence
        and attempt.stage == "scope_allocate"
        and attempt.outcome.value == "succeeded"
    )
    if len(scope_attempts) != 1 or scope_attempts[0].raw_response_ref is None:
        raise ValueError("OpenViking continuation has no exact scope evidence")
    scope_attempt = scope_attempts[0]
    scope_raw_reference = scope_attempt.raw_response_ref
    if scope_raw_reference is None:
        raise AssertionError("validated scope attempt lost its raw reference")
    actor_peer_id = (
        "oamb-" + hashlib.sha256(b"peer\0" + partial_occurrence.encode("utf-8")).hexdigest()
    )
    scope = ScopeReceipt(
        ingestion_occurrence_id=partial_occurrence,
        scope_id=f"viking://user/oamb-admin/peers/{actor_peer_id}/memories",
        raw_reference=RawReferenceHandle(scope_raw_reference),
    )

    raw_payloads = _raw_payloads(source / "raw")
    dispatches = _openviking_dispatches(scope, partial_plan)
    imported_dispatches = []
    for dispatch in dispatches[:completed_count]:
        attempt = succeeded_by_ordinal[dispatch.dispatch_ordinal_1_indexed]
        raw_ref = attempt.raw_response_ref
        if raw_ref is None or raw_ref not in raw_payloads:
            raise ValueError("OpenViking completed dispatch response is missing")
        source_id = dispatch.ordered_source_units[0].source_unit_id
        session_id = _openviking_session_id(source_id)
        task_ids = _completed_task_ids(raw_payloads, session_id)
        if len(task_ids) != 1:
            raise ValueError("OpenViking completed source has no exact terminal task")
        imported_dispatches.append(
            ImportedDispatch(
                ordinal=dispatch.dispatch_ordinal_1_indexed,
                receipt=IngestionDispatchReceipt(
                    attempt_id=attempt.attempt_id,
                    dispatch=dispatch,
                    accepted_source_unit_ids=(source_id,),
                    rejected_source_unit_ids=(),
                    raw_reference=RawReferenceHandle(raw_ref),
                    raw_response_bytes=raw_payloads[raw_ref],
                    usage_records=(),
                ),
                accounting=_attempt_accounting(source, attempt.attempt_id),
                task_id=task_ids[0],
                evidence_references=tuple(
                    RawReferenceHandle(reference)
                    for reference in _session_evidence_references(
                        raw_payloads,
                        session_id=session_id,
                        source_payload=dispatch.ordered_source_units[0].payload_bytes,
                        dispatch_raw_reference=raw_ref,
                    )
                ),
            )
        )

    failed_source = dispatches[completed_count].ordered_source_units[0]
    failed_session_id = _openviking_session_id(failed_source.source_unit_id)
    failed_task_ids = _failed_task_ids(raw_payloads, failed_session_id)
    if len(failed_task_ids) != 1:
        raise ValueError("OpenViking failed source has no exact terminal task")
    return NativeContinuation(
        initialized=initialized,
        previous_lease=previous_lease,
        completed_plan_records=plan_records,
        completed_case_records=case_records,
        partial_plan=PartialPlanContinuation(
            ingestion_plan_id=partial_plan.ingestion_plan_id,
            ingestion_occurrence_id=partial_occurrence,
            scope=scope,
            scope_accounting=_attempt_accounting(source, scope_attempt.attempt_id),
            completed_dispatches=tuple(imported_dispatches),
            failed_attempt=failed[0],
            failed_task_id=failed_task_ids[0],
        ),
    )


def _raw_payloads(raw_root: Path) -> dict[str, bytes]:
    result = {}
    for path in sorted(raw_root.glob("*.json.gz")):
        payload = gzip.decompress(read_regular_file(path))
        reference = path.name.removesuffix(".json.gz")
        if hashlib.sha256(payload).hexdigest() != reference:
            raise ValueError("OpenViking imported raw payload hash mismatch")
        result[reference] = payload
    return result


def _openviking_dispatches(
    scope: ScopeReceipt,
    plan: IngestionPlan,
) -> tuple[IngestionDispatch, ...]:
    result = []
    for source in plan.ordered_source_units:
        session_id = _openviking_session_id(source.source_unit_id)
        timestamp = source.occurred_at
        result.append(
            IngestionDispatch(
                dispatch_ordinal_1_indexed=source.ordinal_1_indexed,
                operation_kind="openviking_session_commit",
                request_fingerprint=_openviking_request_fingerprint(
                    scope.scope_id,
                    session_id,
                    source.source_unit_id,
                    source.payload_sha256,
                    timestamp,
                ),
                ordered_source_units=(source,),
            )
        )
    return tuple(result)


def _openviking_request_fingerprint(
    memory_root: str,
    session_id: str,
    source_unit_id: str,
    payload_sha256: str,
    timestamp: str | None,
) -> str:
    from oamb.contracts.ids import canonical_sha256

    return canonical_sha256(
        [
            "oamb-openviking-session-commit-v1",
            memory_root,
            session_id,
            source_unit_id,
            payload_sha256,
            timestamp,
        ]
    )


def _openviking_session_id(source_unit_id: str) -> str:
    return "oamb-" + hashlib.sha256(b"session\0" + source_unit_id.encode("utf-8")).hexdigest()


def _completed_task_ids(raw_payloads: dict[str, bytes], session_id: str) -> tuple[str, ...]:
    return _task_ids(raw_payloads, session_id, "completed")


def _failed_task_ids(raw_payloads: dict[str, bytes], session_id: str) -> tuple[str, ...]:
    return _task_ids(raw_payloads, session_id, "failed")


def _task_ids(
    raw_payloads: dict[str, bytes],
    session_id: str,
    status: str,
) -> tuple[str, ...]:
    result = set()
    for payload in raw_payloads.values():
        try:
            document = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError):
            continue
        value = document.get("result") if isinstance(document, dict) else None
        if (
            isinstance(value, dict)
            and value.get("resource_id") == session_id
            and value.get("status") == status
            and isinstance(value.get("task_id"), str)
        ):
            result.add(value["task_id"])
    return tuple(sorted(result))


def _session_evidence_references(
    raw_payloads: dict[str, bytes],
    *,
    session_id: str,
    source_payload: bytes,
    dispatch_raw_reference: str,
) -> tuple[str, ...]:
    session_references: list[str] = []
    completed_references: list[str] = []
    archive_references: list[str] = []
    for reference, payload in raw_payloads.items():
        try:
            document = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError):
            continue
        result = document.get("result") if isinstance(document, dict) else None
        if isinstance(result, dict):
            if (
                result.get("resource_id") == session_id
                and result.get("status") == "completed"
                and isinstance(result.get("result"), dict)
            ):
                completed_references.append(reference)
                continue
            if _archive_matches_source(result, source_payload):
                archive_references.append(reference)
                continue
        if session_id.encode("utf-8") in payload:
            session_references.append(reference)
    if len(completed_references) != 1 or len(archive_references) != 1:
        raise ValueError("OpenViking completed source terminal evidence is ambiguous")
    return tuple(
        dict.fromkeys(
            (
                *session_references,
                *completed_references,
                *archive_references,
                dispatch_raw_reference,
            )
        )
    )


def _archive_matches_source(result: dict[str, object], source_payload: bytes) -> bool:
    archived_messages = result.get("messages")
    if not isinstance(result.get("archive_id"), str) or not isinstance(archived_messages, list):
        return False
    try:
        source_messages = json.loads(source_payload)
    except (UnicodeError, json.JSONDecodeError):
        return False
    if not isinstance(source_messages, list):
        return False
    normalized_archive = []
    for message in archived_messages:
        if not isinstance(message, dict) or not isinstance(message.get("parts"), list):
            return False
        text = "".join(
            part.get("text", "")
            for part in message["parts"]
            if isinstance(part, dict) and part.get("type") == "text"
        )
        normalized_archive.append({"role": message.get("role"), "content": text})
    return normalized_archive == source_messages


def _attempt_accounting(source: Path, attempt_id: str) -> ImportedAttemptAccounting:
    def matching(collection: str, id_field: str) -> tuple[str, ...]:
        values = []
        for path in sorted((source / collection).glob("*.json")):
            document = json.loads(read_regular_file(path))
            if document.get("attempt_id") == attempt_id:
                value = document.get(id_field)
                if not isinstance(value, str):
                    raise ValueError("OpenViking imported accounting ID is invalid")
                values.append(value)
        return tuple(values)

    usage = matching("usage", "usage_record_id")
    resources = matching("resources", "resource_record_id")
    costs = matching("costs", "cost_record_id")
    if len(resources) != 1 or len(costs) != 1:
        raise ValueError("OpenViking imported attempt accounting is incomplete")
    return ImportedAttemptAccounting(attempt_id, usage, resources[0], costs[0])


__all__ = [
    "ImportedAttemptAccounting",
    "ImportedDispatch",
    "InitializedContinuation",
    "NativeContinuation",
    "PartialPlanContinuation",
    "initialize_continuation_root",
    "load_openviking_continuation",
]
