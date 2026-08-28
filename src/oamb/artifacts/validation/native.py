"""Closed root-only validation for fixture-produced native source capsules."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from oamb.artifacts.atomic import ArtifactCollisionError, read_regular_file, sha256_file
from oamb.artifacts.validation.hindsight_evidence import (
    HindsightProjectionEvidence,
    reconstruct_hindsight_projection,
)
from oamb.artifacts.validation.openviking_evidence import (
    OPENVIKING_PROFILE_ID,
    OpenVikingPlanEvidence,
    reconstruct_openviking_candidates,
    reconstruct_openviking_plan,
    reconstruct_openviking_projection,
    reconstruct_openviking_runtime_identity,
)
from oamb.contracts.accounting import (
    CostRecord,
    ResourceUsageRecord,
    TokenUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
)
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptReceiptRecord,
    AttemptRecordV2,
    BudgetReservationRecord,
    CapsuleManifest,
    CaseEvaluationDisposition,
    CaseRecordV3,
    CloseErrorRecord,
    IngestionPlanRecordV2,
    OccurrenceClaimRecord,
    RunLeaseRecord,
    RunRecord,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.ids import (
    canonical_sha256,
    case_occurrence_id,
    ingestion_occurrence_id,
)
from oamb.contracts.ports import NativeEvidenceCandidate
from oamb.contracts.specifications import CaseManifest, DatasetManifest
from oamb.contracts.states import (
    AttemptOutcome,
    CaseState,
    IngestionPlanState,
    RunState,
    ValidationDisposition,
)
from oamb.memory_systems.hindsight.normalize import normalize_recall
from oamb.memory_systems.hindsight.profiles import (
    PROFILE_ID as HINDSIGHT_PROFILE_ID,
)
from oamb.memory_systems.hindsight.profiles import (
    parse_bank_config,
    parse_bank_profile,
    parse_retain_response,
)

NATIVE_EVIDENCE_PROFILE_ID = "oamb-t8-native-evidence-v1"
NATIVE_EVIDENCE_RULE_IDS = (
    "native.manifest-schema.v1",
    "native.raw-closure.v1",
    "native.plan-closure.v1",
    "native.retrieval-closure.v1",
    "native.visible-context.v1",
    "native.query-state.v1",
    "native.metric-fraction.v1",
)


@dataclass(frozen=True, slots=True)
class _NativeCapsuleSnapshot:
    root: Path
    manifest_bytes: bytes
    manifest: CapsuleManifest | None
    contracts: tuple[BaseModel, ...]
    documents_by_path: dict[str, bytes]
    raw_payloads: dict[str, bytes]
    parse_failures: tuple[str, ...]
    symlink_paths: tuple[str, ...]


def validate_native_capsule(capsule_root: Path) -> ValidationResult:
    """Reopen and validate a native fixture capsule using no transient receipts."""

    snapshot = _load_native_capsule(Path(capsule_root))
    target_hash = (
        snapshot.manifest.source_manifest_hash
        if snapshot.manifest is not None
        else hashlib.sha256(snapshot.manifest_bytes).hexdigest()
    )
    rules = (
        (NATIVE_EVIDENCE_RULE_IDS[0], _manifest_schema_rule),
        (NATIVE_EVIDENCE_RULE_IDS[1], _raw_closure_rule),
        (NATIVE_EVIDENCE_RULE_IDS[2], _plan_closure_rule),
        (NATIVE_EVIDENCE_RULE_IDS[3], _retrieval_closure_rule),
        (NATIVE_EVIDENCE_RULE_IDS[4], _visible_context_rule),
        (NATIVE_EVIDENCE_RULE_IDS[5], _query_state_rule),
        (NATIVE_EVIDENCE_RULE_IDS[6], _metric_fraction_rule),
    )
    passed: list[str] = []
    failed: list[str] = []
    issues: list[ValidationIssue] = []
    for rule_id, evaluate in rules:
        rule_issues = evaluate(snapshot)
        if rule_issues:
            failed.append(rule_id)
            issues.extend(rule_issues)
        else:
            passed.append(rule_id)
    return ValidationResult(
        validation_profile_id=NATIVE_EVIDENCE_PROFILE_ID,
        target_hash=target_hash,
        disposition=(
            ValidationDisposition.VALIDATED if not failed else ValidationDisposition.INVALID
        ),
        required_rule_ids=NATIVE_EVIDENCE_RULE_IDS,
        executed_rule_ids=NATIVE_EVIDENCE_RULE_IDS,
        passed_rule_ids=tuple(passed),
        failed_rule_ids=tuple(failed),
        not_applicable_rule_ids=(),
        missing_rule_ids=(),
        implementation_versions=tuple(f"{rule_id}@1" for rule_id in NATIVE_EVIDENCE_RULE_IDS),
        issues=tuple(issues),
    )


def _load_native_capsule(root: Path) -> _NativeCapsuleSnapshot:
    symlink_paths = _find_symlink_paths(root)
    manifest_path = root / "capsule-manifest.json"
    try:
        if _below_symlink("capsule-manifest.json", symlink_paths):
            raise ArtifactCollisionError("capsule manifest is below a symbolic link")
        manifest_bytes = read_regular_file(manifest_path)
    except (OSError, ArtifactCollisionError):
        manifest_bytes = b"{}"
    try:
        manifest = CapsuleManifest.model_validate_json(manifest_bytes)
    except Exception:
        manifest = None
    documents: dict[str, bytes] = {}
    raw_payloads: dict[str, bytes] = {}
    contracts: list[BaseModel] = []
    failures: list[str] = []
    if manifest is not None:
        for entry in manifest.source_entries:
            if not entry.relative_path.startswith("source/") or _below_symlink(
                entry.relative_path,
                symlink_paths,
            ):
                failures.append(entry.relative_path)
                continue
            path = root / entry.relative_path
            try:
                content = read_regular_file(path)
            except (OSError, ArtifactCollisionError):
                failures.append(entry.relative_path)
                continue
            documents[entry.relative_path] = content
            if entry.record_kind == "raw_payload":
                try:
                    payload = gzip.decompress(content) if path.name.endswith(".gz") else content
                except (OSError, EOFError):
                    failures.append(entry.relative_path)
                    continue
                raw_payloads[entry.record_id] = payload
                continue
            try:
                document = json.loads(content)
                if not isinstance(document, dict):
                    raise ValueError("source contract is not an object")
                contract = _parse_native_contract(document, content)
                if contract is not None:
                    contracts.append(contract)
                if (
                    document.get("schema_name") != entry.record_kind
                    or _source_record_id(document) != entry.record_id
                    or path.stem != entry.record_id
                ):
                    raise ValueError("source contract identity does not match its manifest entry")
            except Exception:
                failures.append(entry.relative_path)
    return _NativeCapsuleSnapshot(
        root=root,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        contracts=tuple(contracts),
        documents_by_path=documents,
        raw_payloads=raw_payloads,
        parse_failures=tuple(failures),
        symlink_paths=symlink_paths,
    )


def _parse_native_contract(document: dict[str, Any], content: bytes) -> BaseModel | None:
    identity = (document.get("schema_name"), document.get("schema_version"))
    if identity == ("attempt_intent_record", 1):
        return AttemptIntentRecord.model_validate_json(content)
    if identity == ("attempt_receipt_record", 1):
        return AttemptReceiptRecord.model_validate_json(content)
    if identity == ("attempt_record", 2):
        return AttemptRecordV2.model_validate_json(content)
    if identity == ("budget_reservation_record", 1):
        return BudgetReservationRecord.model_validate_json(content)
    if identity == ("case_record", 3):
        return CaseRecordV3.model_validate_json(content)
    if identity == ("close_error_record", 1):
        return CloseErrorRecord.model_validate_json(content)
    if identity == ("ingestion_plan_record", 2):
        return IngestionPlanRecordV2.model_validate_json(content)
    if identity == ("occurrence_claim_record", 1):
        return OccurrenceClaimRecord.model_validate_json(content)
    if identity == ("run_lease_record", 1):
        return RunLeaseRecord.model_validate_json(content)
    if identity == ("run_record", 1):
        return RunRecord.model_validate_json(content)
    if identity == ("case_manifest", 1):
        return CaseManifest.model_validate_json(content)
    if identity == ("dataset_manifest", 1):
        return DatasetManifest.model_validate_json(content)
    if identity == ("token_usage_record", 1):
        return TokenUsageRecord.model_validate_json(content)
    if identity == ("token_usage_record", 2):
        return TokenUsageRecordV2.model_validate_json(content)
    if identity == ("token_usage_record", 3):
        return TokenUsageRecordV3.model_validate_json(content)
    if identity == ("resource_usage_record", 1):
        return ResourceUsageRecord.model_validate_json(content)
    if identity == ("cost_record", 1):
        return CostRecord.model_validate_json(content)
    if not isinstance(identity[0], str) or not isinstance(identity[1], int):
        raise ValueError("source contract has no schema identity")
    raise ValueError(f"unsupported source contract schema identity: {identity[0]}@{identity[1]}")


def _source_record_id(document: dict[str, Any]) -> str | None:
    identity_fields = {
        "attempt_intent_record": "attempt_id",
        "attempt_receipt_record": "attempt_id",
        "attempt_record": "attempt_id",
        "budget_reservation_record": "reservation_id",
        "case_record": "case_occurrence_id",
        "close_error_record": "close_error_id",
        "ingestion_plan_record": "ingestion_occurrence_id",
        "occurrence_claim_record": "claim_id",
        "run_lease_record": "lease_record_hash",
        "run_record": "run_id",
        "token_usage_record": "usage_record_id",
        "resource_usage_record": "resource_record_id",
        "cost_record": "cost_record_id",
    }
    fixed_ids = {
        "case_manifest": "case-manifest",
        "dataset_manifest": "dataset-manifest",
    }
    schema_name = document.get("schema_name")
    field = identity_fields.get(schema_name) if isinstance(schema_name, str) else None
    if field is not None:
        value = document.get(field)
        return value if isinstance(value, str) else None
    return fixed_ids.get(schema_name) if isinstance(schema_name, str) else None


def _manifest_schema_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.manifest-schema.v1"
    manifest = snapshot.manifest
    if manifest is None:
        return (_issue(rule_id, "capsule-manifest.json", "manifest-invalid"),)
    issues = [_issue(rule_id, path, "schema-invalid") for path in snapshot.parse_failures]
    issues.extend(_issue(rule_id, path, "source-symlink") for path in snapshot.symlink_paths)
    expected_paths = {entry.relative_path for entry in manifest.source_entries}
    source_root = snapshot.root / "source"
    actual_paths = (
        {
            path.relative_to(snapshot.root).as_posix()
            for path in source_root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        if source_root.is_dir()
        else set()
    )
    if expected_paths != actual_paths:
        issues.append(_issue(rule_id, manifest.capsule_id, "source-inventory-mismatch"))
    for entry in manifest.source_entries:
        path = snapshot.root / entry.relative_path
        try:
            matches = (
                not _below_symlink(
                    entry.relative_path,
                    snapshot.symlink_paths,
                )
                and sha256_file(path) == entry.sha256
            )
        except (OSError, ArtifactCollisionError):
            matches = False
        if not matches:
            issues.append(_issue(rule_id, entry.record_id, "source-hash-mismatch"))
    expected_source_hash = canonical_sha256(
        [
            "oamb-source-manifest-v1",
            tuple(entry.model_dump(mode="python") for entry in manifest.source_entries),
        ]
    )
    expected_capsule_id = canonical_sha256(
        ["oamb-capsule-v1", manifest.run_id, manifest.run_spec_hash, expected_source_hash]
    )
    if (
        manifest.source_manifest_hash != expected_source_hash
        or manifest.capsule_id != expected_capsule_id
    ):
        issues.append(_issue(rule_id, manifest.capsule_id, "manifest-identity-mismatch"))

    plans = _contracts(snapshot, IngestionPlanRecordV2)
    cases = _contracts(snapshot, CaseRecordV3)
    runs = _contracts(snapshot, RunRecord)
    datasets = _contracts(snapshot, DatasetManifest)
    case_manifests = _contracts(snapshot, CaseManifest)
    expected_plan_occurrence_ids: tuple[str, ...] = ()
    expected_case_occurrence_ids: tuple[str, ...] = ()
    if not plans or not cases or len(datasets) != 1 or len(case_manifests) != 1:
        issues.append(_issue(rule_id, manifest.capsule_id, "native-record-inventory-incomplete"))
    elif any(record.run_id != manifest.run_id for record in (*plans, *cases)):
        issues.append(_issue(rule_id, manifest.capsule_id, "native-run-parent-mismatch"))
    else:
        first = plans[0]
        if any(
            (
                plan.memory_system_id,
                plan.runtime_binding_hash,
                plan.adapter_profile_id,
            )
            != (
                first.memory_system_id,
                first.runtime_binding_hash,
                first.adapter_profile_id,
            )
            for plan in plans
        ):
            issues.append(_issue(rule_id, manifest.capsule_id, "runtime-binding-drift"))
        expected_run_spec_hash = canonical_sha256(
            [
                "oamb-native-fixture-run-spec-v1",
                manifest.run_id,
                datasets[0].manifest_hash,
                case_manifests[0].manifest_hash,
                first.memory_system_id,
                first.runtime_binding_hash,
                first.adapter_profile_id,
            ]
        )
        if manifest.run_spec_hash != expected_run_spec_hash:
            issues.append(_issue(rule_id, manifest.capsule_id, "run-spec-binding-mismatch"))
        case_manifest = case_manifests[0]
        plan_records_by_id = {plan.ingestion_plan_id: plan for plan in plans}
        case_records_by_manifest_id = {case.case_manifest_entry_id: case for case in cases}
        inventory_closed = bool(
            len(plan_records_by_id) == len(plans)
            and len(case_records_by_manifest_id) == len(cases)
            and set(plan_records_by_id)
            == {plan.ingestion_plan_id for plan in case_manifest.ingestion_plans}
            and set(case_records_by_manifest_id)
            == {case.case_manifest_entry_id for case in case_manifest.cases}
        )
        plan_occurrences: list[str] = []
        case_occurrences: list[str] = []
        if inventory_closed:
            for manifest_plan in case_manifest.ingestion_plans:
                plan_record = plan_records_by_id[manifest_plan.ingestion_plan_id]
                plan_occurrence = ingestion_occurrence_id(
                    manifest.run_id,
                    plan_record.memory_system_id,
                    manifest_plan.ingestion_plan_id,
                )
                plan_occurrences.append(plan_occurrence)
                planned_case_occurrences = tuple(
                    case_occurrence_id(plan_occurrence, case_manifest_entry_id)
                    for case_manifest_entry_id in manifest_plan.ordered_case_manifest_entry_ids
                )
                case_occurrences.extend(planned_case_occurrences)
                if (
                    plan_record.ingestion_occurrence_id != plan_occurrence
                    or plan_record.ordered_member_context_manifest_entry_ids
                    != manifest_plan.ordered_member_context_manifest_entry_ids
                    or plan_record.ordered_case_occurrence_ids != planned_case_occurrences
                ):
                    inventory_closed = False
            for case_record in cases:
                expected_case_occurrence = case_occurrence_id(
                    case_record.ingestion_occurrence_id,
                    case_record.case_manifest_entry_id,
                )
                if (
                    case_record.case_occurrence_id != expected_case_occurrence
                    or case_record.case_occurrence_id not in case_occurrences
                ):
                    inventory_closed = False
        expected_plan_occurrence_ids = tuple(plan_occurrences)
        expected_case_occurrence_ids = tuple(case_occurrences)
        if not inventory_closed:
            issues.append(
                _issue(
                    rule_id,
                    manifest.capsule_id,
                    "native-manifest-record-inventory-mismatch",
                )
            )
    if len(runs) != 1:
        issues.append(_issue(rule_id, manifest.capsule_id, "native-run-terminal-mismatch"))
    else:
        run = runs[0]
        expected_plan_ids = set(expected_plan_occurrence_ids)
        expected_case_ids = set(expected_case_occurrence_ids)
        if (
            run.run_id != manifest.run_id
            or run.run_spec_hash != manifest.run_spec_hash
            or run.state != RunState.FINALIZED
            or len(run.ingestion_occurrence_ids) != len(expected_plan_ids)
            or set(run.ingestion_occurrence_ids) != expected_plan_ids
            or len(run.case_occurrence_ids) != len(expected_case_ids)
            or set(run.case_occurrence_ids) != expected_case_ids
        ):
            issues.append(_issue(rule_id, manifest.capsule_id, "native-run-terminal-mismatch"))
    leases = _contracts(snapshot, RunLeaseRecord)
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    intent_records = _contracts(snapshot, AttemptIntentRecord)
    intents = {item.attempt_id: item for item in intent_records}
    receipts = {item.attempt_id: item for item in _contracts(snapshot, AttemptReceiptRecord)}
    claim_records = _contracts(snapshot, OccurrenceClaimRecord)
    claims = {item.claim_id: item for item in claim_records}
    reservation_records = _contracts(snapshot, BudgetReservationRecord)
    reservations = {item.reservation_id: item for item in reservation_records}
    for lease in leases:
        expected_lease_hash = canonical_sha256(
            lease.model_dump(mode="python", exclude={"lease_record_hash"})
        )
        if lease.lease_record_hash != expected_lease_hash:
            issues.append(_issue(rule_id, lease.lease_record_hash, "run-lease-identity-mismatch"))
    for claim in claim_records:
        claim_fields = claim.model_dump(
            mode="python",
            exclude={"schema_name", "schema_version", "claim_id"},
        )
        expected_claim_id = canonical_sha256(["oamb-native-occurrence-claim-v1", claim_fields])
        if claim.claim_id != expected_claim_id:
            issues.append(_issue(rule_id, claim.claim_id, "attempt-claim-identity-mismatch"))
    for reservation in reservation_records:
        reservation_fields = reservation.model_dump(
            mode="python",
            exclude={"schema_name", "schema_version", "reservation_id"},
        )
        expected_reservation_id = canonical_sha256(
            ["oamb-native-budget-reservation-v1", reservation_fields]
        )
        if reservation.reservation_id != expected_reservation_id:
            issues.append(
                _issue(
                    rule_id,
                    reservation.reservation_id,
                    "attempt-reservation-identity-mismatch",
                )
            )
    if not _has_exact_reference_inventory(
        tuple(item.claim_id for item in claim_records),
        tuple(item.claim_id for item in intent_records),
    ):
        issues.append(_issue(rule_id, manifest.capsule_id, "attempt-claim-inventory-mismatch"))
    if not _has_exact_reference_inventory(
        tuple(item.reservation_id for item in reservation_records),
        tuple(item.reservation_id for item in intent_records),
    ):
        issues.append(
            _issue(rule_id, manifest.capsule_id, "attempt-reservation-inventory-mismatch")
        )
    if len(leases) != 1 or set(attempts) != set(intents) or set(attempts) != set(receipts):
        issues.append(_issue(rule_id, manifest.capsule_id, "attempt-evidence-incomplete"))
    else:
        lease = leases[0]
        for attempt_id, attempt in attempts.items():
            intent = intents[attempt_id]
            receipt = receipts[attempt_id]
            claim = claims.get(intent.claim_id)
            reservation = reservations.get(intent.reservation_id)
            aligned = bool(
                claim is not None
                and reservation is not None
                and claim.lease_record_hash == lease.lease_record_hash
                and claim.lease_epoch == lease.lease_epoch
                and claim.owner_id == lease.owner_id
                and claim.occurrence_id == attempt.parent_id == intent.parent_id
                and claim.stage == attempt.stage == intent.stage
                and claim.request_fingerprint
                == attempt.request_fingerprint
                == intent.request_fingerprint
                and reservation.attempt_id == attempt_id
                and reservation.scope_id == attempt.parent_id
                and reservation.role_binding_id == intent.role_binding_id
                and intent.sealed_at <= attempt.started_at
                and receipt.dispatch_started_at == attempt.started_at
                and receipt.receipt_observed_at == attempt.ended_at
                and receipt.raw_response_ref == attempt.raw_response_ref
                and receipt.raw_error_ref == attempt.raw_error_ref
            )
            if not aligned:
                issues.append(_issue(rule_id, attempt_id, "attempt-evidence-mismatch"))
    return tuple(issues)


def _raw_closure_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.raw-closure.v1"
    issues: list[ValidationIssue] = []
    for raw_reference in _all_raw_references(snapshot):
        payload = snapshot.raw_payloads.get(raw_reference)
        if payload is None:
            issues.append(_issue(rule_id, raw_reference, "raw-reference-missing"))
        elif hashlib.sha256(payload).hexdigest() != raw_reference:
            issues.append(_issue(rule_id, raw_reference, "raw-payload-hash-mismatch"))
    return tuple(issues)


def _plan_closure_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.plan-closure.v1"
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    usage_records = {item.usage_record_id: item for item in _token_usage_records(snapshot)}
    resource_records = {
        item.resource_record_id: item for item in _contracts(snapshot, ResourceUsageRecord)
    }
    cost_records = {item.cost_record_id: item for item in _contracts(snapshot, CostRecord)}
    issues: list[ValidationIssue] = []
    for plan in _contracts(snapshot, IngestionPlanRecordV2):
        if plan.state != IngestionPlanState.SEALED or plan.scope_id is None:
            issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "plan-not-sealed"))
        if not _accounting_ledger_closes(
            parent_kind="ingestion_plan",
            parent_id=plan.ingestion_occurrence_id,
            attempt_ids=plan.attempt_ids,
            usage_record_ids=plan.usage_record_ids,
            resource_record_ids=plan.resource_record_ids,
            cost_record_ids=plan.cost_record_ids,
            attempts=attempts,
            usage_records=usage_records,
            resource_records=resource_records,
            cost_records=cost_records,
        ):
            issues.append(
                _issue(
                    rule_id,
                    plan.ingestion_occurrence_id,
                    "ingestion-accounting-ledger-mismatch",
                )
            )
        if plan.adapter_profile_id == HINDSIGHT_PROFILE_ID:
            issues.extend(_hindsight_plan_closure(snapshot, plan, attempts))
            if not _plan_usage_closes(plan, attempts, usage_records):
                issues.append(
                    _issue(
                        rule_id,
                        plan.ingestion_occurrence_id,
                        "ingestion-usage-ledger-mismatch",
                    )
                )
            continue
        if plan.adapter_profile_id == OPENVIKING_PROFILE_ID:
            try:
                _openviking_native_plan_evidence(snapshot, plan, attempts)
            except ValueError:
                issues.append(
                    _issue(
                        rule_id,
                        plan.ingestion_occurrence_id,
                        "openviking-plan-evidence-mismatch",
                    )
                )
            if not _plan_usage_closes(plan, attempts, usage_records):
                issues.append(
                    _issue(
                        rule_id,
                        plan.ingestion_occurrence_id,
                        "ingestion-usage-ledger-mismatch",
                    )
                )
            continue
        scope_document = _raw_object(
            snapshot, plan.scope_raw_refs[0] if plan.scope_raw_refs else None
        )
        if not _matches(
            scope_document,
            operation="allocate_scope",
            ingestion_occurrence_id=plan.ingestion_occurrence_id,
            ingestion_plan_id=plan.ingestion_plan_id,
            scope_id=plan.scope_id,
        ):
            issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "scope-binding-mismatch"))

        accepted: list[str] = []
        rejected: list[str] = []
        dispatched: list[str] = []
        for attempt_id, expected_dispatch_source_ids in zip(
            plan.ordered_dispatch_attempt_ids,
            plan.ordered_dispatch_source_unit_ids,
            strict=True,
        ):
            attempt = attempts.get(attempt_id)
            if (
                attempt is None
                or attempt.parent_kind != "ingestion_plan"
                or attempt.parent_id != plan.ingestion_occurrence_id
                or attempt.stage != "memory_ingest"
                or attempt.outcome != AttemptOutcome.SUCCEEDED
                or attempt.raw_response_ref is None
            ):
                issues.append(_issue(rule_id, attempt_id, "dispatch-attempt-mismatch"))
                continue
            document = _raw_object(snapshot, attempt.raw_response_ref)
            if not _matches(
                document,
                operation="ingest",
                attempt_id=attempt_id,
                ingestion_occurrence_id=plan.ingestion_occurrence_id,
                scope_id=plan.scope_id,
            ):
                issues.append(_issue(rule_id, attempt_id, "dispatch-receipt-mismatch"))
                continue
            if document is None:
                issues.append(_issue(rule_id, attempt_id, "dispatch-receipt-mismatch"))
                continue
            source_units = document.get("source_units")
            if not isinstance(source_units, list):
                issues.append(_issue(rule_id, attempt_id, "dispatch-source-ledger-invalid"))
                continue
            source_ids: list[str] = []
            for item in source_units:
                if not isinstance(item, dict) or not isinstance(item.get("source_unit_id"), str):
                    issues.append(_issue(rule_id, attempt_id, "dispatch-source-ledger-invalid"))
                    source_ids = []
                    break
                source_ids.append(item["source_unit_id"])
            if len(source_ids) != len(source_units):
                continue
            if tuple(source_ids) != expected_dispatch_source_ids:
                issues.append(_issue(rule_id, attempt_id, "dispatch-source-ledger-mismatch"))
            dispatched.extend(source_ids)
            accepted.extend(_string_list(document.get("accepted_source_unit_ids")))
            rejected.extend(_string_list(document.get("rejected_source_unit_ids")))
        if tuple(dispatched) != plan.ordered_source_unit_ids:
            issues.append(
                _issue(rule_id, plan.ingestion_occurrence_id, "dispatch-source-order-mismatch")
            )
        if (
            tuple(accepted) != plan.accepted_source_unit_ids
            or tuple(rejected) != plan.rejected_source_unit_ids
        ):
            issues.append(
                _issue(rule_id, plan.ingestion_occurrence_id, "dispatch-partition-mismatch")
            )

        readiness_documents = tuple(
            _raw_object(snapshot, reference) for reference in plan.readiness_evidence_refs
        )
        if not readiness_documents or not any(
            _matches(
                document,
                operation="wait_ready",
                ingestion_occurrence_id=plan.ingestion_occurrence_id,
                scope_id=plan.scope_id,
                ready=True,
            )
            for document in readiness_documents
        ):
            issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "readiness-proof-mismatch"))
        inventory = _raw_object(snapshot, plan.inventory_raw_ref)
        projected = _string_list(
            inventory.get("ordered_source_unit_ids") if inventory is not None else None
        )
        if (
            not _matches(
                inventory,
                operation="inventory",
                ingestion_occurrence_id=plan.ingestion_occurrence_id,
                scope_id=plan.scope_id,
            )
            or tuple(projected) != plan.projected_source_unit_ids
            or plan.projected_source_unit_ids != plan.accepted_source_unit_ids
        ):
            issues.append(
                _issue(
                    rule_id,
                    plan.ingestion_occurrence_id,
                    "projection-source-order-mismatch",
                )
            )
        state_document = _operation_document(snapshot, plan.projection_raw_refs, "state_digest")
        if not _matches(
            state_document,
            operation="state_digest",
            ingestion_occurrence_id=plan.ingestion_occurrence_id,
            scope_id=plan.scope_id,
            state_sha256=plan.protected_state_sha256,
        ):
            issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "protected-state-mismatch"))
        if not _plan_usage_closes(plan, attempts, usage_records):
            issues.append(
                _issue(
                    rule_id,
                    plan.ingestion_occurrence_id,
                    "ingestion-usage-ledger-mismatch",
                )
            )
    return tuple(issues)


def _plan_usage_closes(
    plan: IngestionPlanRecordV2,
    attempts: dict[str, AttemptRecordV2],
    usage_records: dict[str, TokenUsageRecord | TokenUsageRecordV2 | TokenUsageRecordV3],
) -> bool:
    if len(plan.usage_record_ids) != len(plan.ordered_dispatch_attempt_ids):
        return False
    for usage_record_id, attempt_id in zip(
        plan.usage_record_ids,
        plan.ordered_dispatch_attempt_ids,
        strict=True,
    ):
        usage = usage_records.get(usage_record_id)
        attempt = attempts.get(attempt_id)
        if (
            usage is None
            or attempt is None
            or usage.parent_kind != "ingestion_plan"
            or usage.parent_id != plan.ingestion_occurrence_id
            or usage.attempt_id != attempt_id
            or usage.raw_response_ref != attempt.raw_response_ref
        ):
            return False
    return True


def _accounting_ledger_closes(
    *,
    parent_kind: str,
    parent_id: str,
    attempt_ids: tuple[str, ...],
    usage_record_ids: tuple[str, ...],
    resource_record_ids: tuple[str, ...],
    cost_record_ids: tuple[str, ...],
    attempts: dict[str, AttemptRecordV2],
    usage_records: dict[str, TokenUsageRecord | TokenUsageRecordV2 | TokenUsageRecordV3],
    resource_records: dict[str, ResourceUsageRecord],
    cost_records: dict[str, CostRecord],
) -> bool:
    if not attempt_ids or not (
        len(attempt_ids)
        == len(usage_record_ids)
        == len(resource_record_ids)
        == len(cost_record_ids)
    ):
        return False
    for attempt_id, usage_id, resource_id, cost_id in zip(
        attempt_ids,
        usage_record_ids,
        resource_record_ids,
        cost_record_ids,
        strict=True,
    ):
        attempt = attempts.get(attempt_id)
        usage = usage_records.get(usage_id)
        resource = resource_records.get(resource_id)
        cost = cost_records.get(cost_id)
        expected_indexing_view = (
            "final_contribution"
            if attempt is not None and attempt.stage == "memory_ingest"
            else "not_applicable"
        )
        if (
            attempt is None
            or usage is None
            or resource is None
            or cost is None
            or attempt.parent_kind != parent_kind
            or attempt.parent_id != parent_id
            or usage.attempt_id != attempt_id
            or usage.parent_kind != parent_kind
            or usage.parent_id != parent_id
            or usage.stage.value != attempt.stage
            or usage.raw_response_ref != attempt.raw_response_ref
            or resource.parent_kind != parent_kind
            or resource.parent_id != parent_id
            or resource.stage != attempt.stage
            or resource.raw_telemetry_ref != attempt.raw_response_ref
            or cost.parent_kind != parent_kind
            or cost.parent_id != parent_id
            or cost.indexing_view.value != expected_indexing_view
            or cost.source_usage_record_ids != (usage_id,)
            or cost.source_resource_record_ids != (resource_id,)
        ):
            return False
    return True


def _openviking_native_plan_evidence(
    snapshot: _NativeCapsuleSnapshot,
    plan: IngestionPlanRecordV2,
    attempts: dict[str, AttemptRecordV2],
) -> OpenVikingPlanEvidence:
    manifests = _contracts(snapshot, CaseManifest)
    if len(manifests) != 1:
        raise ValueError("OpenViking capsule has no exact case manifest")
    manifest_plan = next(
        (
            item
            for item in manifests[0].ingestion_plans
            if item.ingestion_plan_id == plan.ingestion_plan_id
        ),
        None,
    )
    if manifest_plan is None:
        raise ValueError("OpenViking plan is absent from the case manifest")
    return reconstruct_openviking_plan(
        raw_payloads=snapshot.raw_payloads,
        plan=plan,
        manifest_plan=manifest_plan,
        attempts=attempts,
        runtime_identity=reconstruct_openviking_runtime_identity(snapshot.raw_payloads),
    )


def _hindsight_plan_closure(
    snapshot: _NativeCapsuleSnapshot,
    plan: IngestionPlanRecordV2,
    attempts: dict[str, AttemptRecordV2],
) -> tuple[ValidationIssue, ...]:
    rule_id = "native.plan-closure.v1"
    issues: list[ValidationIssue] = []
    expected_scope_id = hashlib.sha256(
        f"oamb-hindsight-bank-v1\0{plan.ingestion_occurrence_id}".encode()
    ).hexdigest()
    scope_payloads = tuple(
        snapshot.raw_payloads.get(reference) for reference in plan.scope_raw_refs
    )
    if (
        plan.scope_id != expected_scope_id
        or not scope_payloads
        or scope_payloads[0] is None
        or not _accepts_hindsight_bank_profile(scope_payloads[0], expected_scope_id)
        or not any(
            _accepts_hindsight_bank_config(payload, expected_scope_id)
            for payload in scope_payloads[1:]
            if payload is not None
        )
    ):
        issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "scope-binding-mismatch"))

    dispatched: list[str] = []
    dispatch_raw_refs: list[str] = []
    for attempt_id, expected_source_ids in zip(
        plan.ordered_dispatch_attempt_ids,
        plan.ordered_dispatch_source_unit_ids,
        strict=True,
    ):
        attempt = attempts.get(attempt_id)
        payload = (
            snapshot.raw_payloads.get(attempt.raw_response_ref)
            if attempt is not None and attempt.raw_response_ref is not None
            else None
        )
        if (
            attempt is None
            or attempt.parent_kind != "ingestion_plan"
            or attempt.parent_id != plan.ingestion_occurrence_id
            or attempt.stage != "memory_ingest"
            or attempt.outcome != AttemptOutcome.SUCCEEDED
            or attempt.raw_response_ref is None
            or payload is None
            or not _accepts_hindsight_retain(
                payload,
                expected_scope_id,
                len(expected_source_ids),
            )
        ):
            issues.append(_issue(rule_id, attempt_id, "dispatch-receipt-mismatch"))
            continue
        dispatched.extend(expected_source_ids)
        dispatch_raw_refs.append(attempt.raw_response_ref)
    if (
        tuple(dispatched) != plan.ordered_source_unit_ids
        or plan.accepted_source_unit_ids != plan.ordered_source_unit_ids
        or plan.rejected_source_unit_ids
    ):
        issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "dispatch-partition-mismatch"))
    if (
        not plan.readiness_evidence_refs
        or not set(dispatch_raw_refs) <= set(plan.readiness_evidence_refs)
        or any(reference not in snapshot.raw_payloads for reference in plan.readiness_evidence_refs)
    ):
        issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "readiness-proof-mismatch"))

    try:
        projection = _hindsight_native_projection_evidence(
            snapshot,
            plan,
            plan.projection_raw_refs,
        )
    except ValueError:
        projection = None
    if (
        projection is None
        or projection.ordered_source_unit_ids != plan.projected_source_unit_ids
        or plan.projected_source_unit_ids != plan.accepted_source_unit_ids
    ):
        issues.append(
            _issue(rule_id, plan.ingestion_occurrence_id, "projection-source-order-mismatch")
        )
    if (
        plan.inventory_raw_ref is None
        or projection is None
        or projection.state_sha256 != plan.protected_state_sha256
        or plan.inventory_raw_ref != projection.state_sha256
        or plan.inventory_raw_ref not in plan.projection_raw_refs
        or any(reference not in snapshot.raw_payloads for reference in plan.projection_raw_refs)
    ):
        issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "protected-state-mismatch"))
    return tuple(issues)


def _retrieval_closure_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.retrieval-closure.v1"
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    usage_records = {item.usage_record_id: item for item in _token_usage_records(snapshot)}
    resource_records = {
        item.resource_record_id: item for item in _contracts(snapshot, ResourceUsageRecord)
    }
    cost_records = {item.cost_record_id: item for item in _contracts(snapshot, CostRecord)}
    plans = {
        item.ingestion_occurrence_id: item for item in _contracts(snapshot, IngestionPlanRecordV2)
    }
    issues: list[ValidationIssue] = []
    for case in _contracts(snapshot, CaseRecordV3):
        plan = plans.get(case.ingestion_occurrence_id)
        if (
            case.state != CaseState.COMPLETED
            or plan is None
            or case.adapter_profile_id != plan.adapter_profile_id
            or case.case_occurrence_id not in plan.ordered_case_occurrence_ids
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "case-parentage-mismatch"))
        candidates = _normalized_native_candidates(snapshot, case, plan)
        if candidates is None:
            issues.append(_issue(rule_id, case.case_occurrence_id, "retrieval-primary-mismatch"))
            continue
        if (
            tuple(item.native_id for item in candidates) != case.ordered_native_candidate_ids
            or tuple(
                hashlib.sha256(item.content.encode("utf-8")).hexdigest() for item in candidates
            )
            != case.ordered_native_content_sha256
            or tuple(item.source_unit_id for item in candidates)
            != case.native_candidate_source_unit_ids
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "retrieval-order-mismatch"))
        if case.adapter_profile_id not in {
            HINDSIGHT_PROFILE_ID,
            OPENVIKING_PROFILE_ID,
        }:
            retrieval = _raw_object(snapshot, case.retrieval_raw_ref)
            primary_supporting_inventory = _string_list(
                retrieval.get("supporting_raw_references") if retrieval is not None else None
            )
            if tuple(primary_supporting_inventory) != case.retrieval_supporting_raw_refs:
                issues.append(
                    _issue(
                        rule_id,
                        case.case_occurrence_id,
                        "retrieval-support-inventory-mismatch",
                    )
                )
            for reference in case.retrieval_supporting_raw_refs:
                supporting = _raw_object(snapshot, reference)
                if not _matches(
                    supporting,
                    operation="retrieve_support",
                    case_occurrence_id=case.case_occurrence_id,
                    ordered_native_ids=list(case.ordered_native_candidate_ids),
                ):
                    issues.append(_issue(rule_id, reference, "retrieval-support-mismatch"))
        elif case.adapter_profile_id == HINDSIGHT_PROFILE_ID and case.retrieval_supporting_raw_refs:
            issues.append(
                _issue(
                    rule_id,
                    case.case_occurrence_id,
                    "retrieval-support-inventory-mismatch",
                )
            )

        case_attempts = [attempts.get(attempt_id) for attempt_id in case.attempt_ids]
        query_attempts = [
            item for item in case_attempts if item is not None and item.stage == "memory_query"
        ]
        answer_attempts = [
            item for item in case_attempts if item is not None and item.stage == "answer"
        ]
        judge_attempts = [
            item for item in case_attempts if item is not None and item.stage == "judge"
        ]
        expected_judge_count = (
            1 if case.evaluation_disposition == CaseEvaluationDisposition.JUDGED else 0
        )
        if (
            len(query_attempts) != 1
            or query_attempts[0].raw_response_ref != case.retrieval_raw_ref
            or len(answer_attempts) != 1
            or answer_attempts[0].raw_response_ref != case.answer_raw_ref
            or len(judge_attempts) != expected_judge_count
            or any(item.raw_response_ref is None for item in judge_attempts)
            or len(case_attempts) != 2 + expected_judge_count
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "case-attempt-ledger-mismatch"))
        if not _accounting_ledger_closes(
            parent_kind="case",
            parent_id=case.case_occurrence_id,
            attempt_ids=case.attempt_ids,
            usage_record_ids=case.usage_record_ids,
            resource_record_ids=case.resource_record_ids,
            cost_record_ids=case.cost_record_ids,
            attempts=attempts,
            usage_records=usage_records,
            resource_records=resource_records,
            cost_records=cost_records,
        ):
            issues.append(
                _issue(
                    rule_id,
                    case.case_occurrence_id,
                    "case-accounting-ledger-mismatch",
                )
            )
        answer = _raw_object(snapshot, case.answer_raw_ref)
        prompt = snapshot.raw_payloads.get(case.prompt_raw_ref or "")
        output = _model_output_text(answer)
        answer_fingerprint = _single_user_message_fingerprint(prompt)
        judge_prompt = snapshot.raw_payloads.get(case.judge_prompt_raw_ref or "")
        judge_fingerprint = _single_user_message_fingerprint(judge_prompt)
        if (
            prompt is None
            or not isinstance(output, str)
            or hashlib.sha256(prompt).hexdigest() != case.prompt_sha256
            or answer_fingerprint is None
            or len(answer_attempts) != 1
            or answer_attempts[0].request_fingerprint != answer_fingerprint
            or hashlib.sha256(output.encode("utf-8")).hexdigest() != case.parsed_answer_sha256
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "prompt-answer-mismatch"))
        if case.evaluation_disposition == CaseEvaluationDisposition.JUDGED:
            if (
                case.judge_prompt_raw_ref is None
                or judge_fingerprint is None
                or len(judge_attempts) != 1
                or judge_attempts[0].request_fingerprint != judge_fingerprint
            ):
                issues.append(_issue(rule_id, case.case_occurrence_id, "judge-prompt-mismatch"))
        elif case.judge_prompt_raw_ref is not None:
            issues.append(_issue(rule_id, case.case_occurrence_id, "judge-prompt-mismatch"))
    return tuple(issues)


def _visible_context_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.visible-context.v1"
    issues: list[ValidationIssue] = []
    for case in _contracts(snapshot, CaseRecordV3):
        visible = snapshot.raw_payloads.get(case.visible_evidence_raw_ref or "")
        ledger = _raw_object(snapshot, case.visible_decision_ledger_raw_ref)
        decisions = ledger.get("decisions") if ledger is not None else None
        if visible is None or not isinstance(decisions, list):
            issues.append(_issue(rule_id, case.case_occurrence_id, "visible-evidence-missing"))
            continue
        kept_ids = tuple(
            item.get("native_id")
            for item in decisions
            if isinstance(item, dict) and item.get("disposition") == "kept"
        )
        truncated_count = sum(
            isinstance(item, dict) and item.get("disposition") == "truncated" for item in decisions
        )
        ledger_values_match = _matches(
            ledger,
            operation="visible_evidence_decision",
            case_occurrence_id=case.case_occurrence_id,
            included_native_ids=list(kept_ids),
            visible_evidence_sha256=case.visible_evidence_sha256,
            visible_evidence_byte_count=case.visible_evidence_byte_count,
            visible_evidence_token_count=case.visible_evidence_token_count,
            visible_evidence_tokenizer_fingerprint=(case.visible_evidence_tokenizer_fingerprint),
            candidate_count=case.native_candidate_count,
            kept_count=case.visible_kept_count,
            dropped_count=case.visible_dropped_count,
            truncated_count=case.visible_truncated_count,
        )
        if (
            not ledger_values_match
            or len(kept_ids) != case.visible_kept_count
            or len(decisions) - len(kept_ids) != case.visible_dropped_count
            or truncated_count != case.visible_truncated_count
            or hashlib.sha256(visible).hexdigest() != case.visible_evidence_sha256
            or len(visible) != case.visible_evidence_byte_count
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "visible-decision-mismatch"))
            continue
        expected_visible = _rebuild_visible(snapshot, case, decisions)
        if expected_visible is None or expected_visible != visible:
            issues.append(_issue(rule_id, case.case_occurrence_id, "visible-bytes-mismatch"))
    return tuple(issues)


def _query_state_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.query-state.v1"
    issues: list[ValidationIssue] = []
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    plans = {
        item.ingestion_occurrence_id: item for item in _contracts(snapshot, IngestionPlanRecordV2)
    }
    for case in _contracts(snapshot, CaseRecordV3):
        if case.adapter_profile_id == HINDSIGHT_PROFILE_ID:
            plan = plans.get(case.ingestion_occurrence_id)
            try:
                if plan is None:
                    raise ValueError("Hindsight case has no plan evidence")
                before = _hindsight_native_projection_evidence(
                    snapshot,
                    plan,
                    case.pre_query_projection_raw_refs,
                )
                after = _hindsight_native_projection_evidence(
                    snapshot,
                    plan,
                    case.post_query_projection_raw_refs,
                )
            except ValueError:
                issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
                continue
            if (
                before.state_sha256 != case.pre_query_state_sha256
                or after.state_sha256 != case.post_query_state_sha256
                or before.state_sha256 != after.state_sha256
                or before.state_sha256 != plan.protected_state_sha256
                or case.query_mutation_status != "unchanged"
            ):
                issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
            continue
        if case.adapter_profile_id == OPENVIKING_PROFILE_ID:
            plan = plans.get(case.ingestion_occurrence_id)
            try:
                if plan is None:
                    raise ValueError("OpenViking case has no plan evidence")
                evidence = _openviking_native_plan_evidence(
                    snapshot,
                    plan,
                    attempts,
                )
                before_projection = reconstruct_openviking_projection(
                    raw_payloads=snapshot.raw_payloads,
                    references=case.pre_query_projection_raw_refs,
                    root_uri=evidence.root_uri,
                    chunk_uris=evidence.chunk_uris,
                    source_payload_sha256=evidence.source_payload_sha256,
                )
                after_projection = reconstruct_openviking_projection(
                    raw_payloads=snapshot.raw_payloads,
                    references=case.post_query_projection_raw_refs,
                    root_uri=evidence.root_uri,
                    chunk_uris=evidence.chunk_uris,
                    source_payload_sha256=evidence.source_payload_sha256,
                )
            except ValueError:
                issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
                continue
            if (
                before_projection.state_sha256 != case.pre_query_state_sha256
                or after_projection.state_sha256 != case.post_query_state_sha256
                or before_projection.state_sha256 != after_projection.state_sha256
                or before_projection.state_sha256 != evidence.projection.state_sha256
                or case.query_mutation_status != "unchanged"
            ):
                issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
            continue
        before_document = _operation_document(
            snapshot, case.pre_query_projection_raw_refs, "state_digest"
        )
        after_document = _operation_document(
            snapshot, case.post_query_projection_raw_refs, "state_digest"
        )
        before_state = before_document.get("state_sha256") if before_document is not None else None
        after_state = after_document.get("state_sha256") if after_document is not None else None
        if (
            before_state != case.pre_query_state_sha256
            or after_state != case.post_query_state_sha256
            or before_state != after_state
            or case.query_mutation_status != "unchanged"
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
    return tuple(issues)


def _metric_fraction_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.metric-fraction.v1"
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    issues: list[ValidationIssue] = []
    for case in _contracts(snapshot, CaseRecordV3):
        evaluation = _raw_object(snapshot, case.evaluation_raw_ref)
        answer = _raw_object(snapshot, case.answer_raw_ref)
        output = _model_output_text(answer)
        fraction_matches = _matches(
            evaluation,
            operation="deterministic_evaluation",
            metric_id=case.metric_id,
            numerator=case.metric_numerator,
            denominator=case.metric_denominator,
            parsed_answer_sha256=case.parsed_answer_sha256,
        )
        trace_hex = evaluation.get("trace_hex") if evaluation is not None else None
        try:
            trace = bytes.fromhex(trace_hex) if isinstance(trace_hex, str) else b""
            trace_document = json.loads(trace)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            trace = b""
            trace_document = None
        judge_attempts = tuple(
            attempts[attempt_id]
            for attempt_id in case.attempt_ids
            if attempt_id in attempts and attempts[attempt_id].stage == "judge"
        )
        judged_trace_matches = True
        if case.evaluation_disposition == CaseEvaluationDisposition.JUDGED:
            if len(judge_attempts) != 1 or judge_attempts[0].raw_response_ref is None:
                judged_trace_matches = False
            else:
                judge_document = _raw_object(snapshot, judge_attempts[0].raw_response_ref)
                judge_output = _model_output_text(judge_document)
                judged_trace_matches = bool(
                    isinstance(judge_output, str)
                    and isinstance(trace_document, dict)
                    and trace_document.get("judge_raw_reference")
                    == judge_attempts[0].raw_response_ref
                    and trace_document.get("judge_answer_sha256")
                    == hashlib.sha256(judge_output.encode("utf-8")).hexdigest()
                )
        elif judge_attempts:
            judged_trace_matches = False
        if (
            not fraction_matches
            or not isinstance(output, str)
            or hashlib.sha256(output.encode("utf-8")).hexdigest() != case.parsed_answer_sha256
            or not isinstance(trace_document, dict)
            or trace_document.get("metric_id") != case.metric_id
            or trace_document.get("numerator") != case.metric_numerator
            or trace_document.get("denominator") != case.metric_denominator
            or trace_document.get("parsed_answer_sha256") != case.parsed_answer_sha256
            or evaluation is None
            or evaluation.get("trace_sha256") != hashlib.sha256(trace).hexdigest()
            or evaluation.get("result_sha256") != hashlib.sha256(trace).hexdigest()
            or not judged_trace_matches
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "metric-fraction-mismatch"))
    return tuple(issues)


def _rebuild_visible(
    snapshot: _NativeCapsuleSnapshot,
    case: CaseRecordV3,
    decisions: list[Any],
) -> bytes | None:
    plans = {
        item.ingestion_occurrence_id: item for item in _contracts(snapshot, IngestionPlanRecordV2)
    }
    candidates = _normalized_native_candidates(
        snapshot,
        case,
        plans.get(case.ingestion_occurrence_id),
    )
    if candidates is None:
        return None
    candidates_by_id = {item.native_id: item for item in candidates}
    lines: list[bytes] = []
    for decision in decisions:
        if not isinstance(decision, dict) or decision.get("disposition") != "kept":
            continue
        decision_native_id = decision.get("native_id")
        if not isinstance(decision_native_id, str):
            return None
        candidate = candidates_by_id.get(decision_native_id)
        if candidate is None:
            return None
        content = candidate.content
        identity = candidate.provider_evidence_identity
        if not isinstance(content, str) or not isinstance(identity, str):
            return None
        if (
            decision.get("provider_evidence_identity") != identity
            or decision.get("normalized_text_sha256")
            != hashlib.sha256(content.encode("utf-8")).hexdigest()
        ):
            return None
        lines.append(
            json.dumps(
                {
                    "provider_evidence_identity": identity,
                    "source_unit_id": candidate.source_unit_id,
                    "evidence_kind": candidate.evidence_kind,
                    "text": content,
                    "occurred_start": _canonical_visible_timestamp(candidate.occurred_start),
                    "occurred_end": _canonical_visible_timestamp(candidate.occurred_end),
                    "mentioned_at": _canonical_visible_timestamp(candidate.mentioned_at),
                },
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
    return b"\n".join(lines)


def _normalized_native_candidates(
    snapshot: _NativeCapsuleSnapshot,
    case: CaseRecordV3,
    plan: IngestionPlanRecordV2 | None,
) -> tuple[NativeEvidenceCandidate, ...] | None:
    payload = snapshot.raw_payloads.get(case.retrieval_raw_ref or "")
    if payload is None:
        return None
    if case.adapter_profile_id == HINDSIGHT_PROFILE_ID:
        if plan is None or plan.adapter_profile_id != HINDSIGHT_PROFILE_ID:
            return None
        try:
            projection = _hindsight_native_projection_evidence(
                snapshot,
                plan,
                plan.projection_raw_refs,
            )
            return normalize_recall(
                payload,
                document_to_source_unit=projection.document_to_source_unit,
            )
        except ValueError:
            return None
    if case.adapter_profile_id == OPENVIKING_PROFILE_ID:
        if plan is None:
            return None
        attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
        try:
            evidence = _openviking_native_plan_evidence(snapshot, plan, attempts)
            return reconstruct_openviking_candidates(
                raw_payloads=snapshot.raw_payloads,
                case=case,
                plan=evidence,
            )
        except ValueError:
            return None
    retrieval = _raw_object(snapshot, case.retrieval_raw_ref)
    raw_candidates = retrieval.get("candidates") if retrieval is not None else None
    if (
        plan is None
        or not _matches(
            retrieval,
            operation="retrieve",
            case_occurrence_id=case.case_occurrence_id,
            scope_id=plan.scope_id,
        )
        or not isinstance(raw_candidates, list)
    ):
        return None
    candidates: list[NativeEvidenceCandidate] = []
    for ordinal, item in enumerate(raw_candidates, start=1):
        if (
            not isinstance(item, dict)
            or item.get("native_rank_1_indexed") != ordinal
            or not isinstance(item.get("native_id"), str)
            or not isinstance(item.get("content"), str)
        ):
            return None
        source_id = item.get("source_unit_id")
        evidence_kind = item.get("evidence_kind")
        candidates.append(
            NativeEvidenceCandidate(
                native_id=item["native_id"],
                native_rank_1_indexed=ordinal,
                content=item["content"],
                native_score=(
                    item.get("native_score") if isinstance(item.get("native_score"), str) else None
                ),
                provider_evidence_identity=(
                    item.get("provider_evidence_identity")
                    if isinstance(item.get("provider_evidence_identity"), str)
                    else item["native_id"]
                ),
                source_unit_id=source_id if isinstance(source_id, str) else None,
                evidence_kind=evidence_kind if isinstance(evidence_kind, str) else "native",
                occurred_start=_optional_string(item.get("occurred_start")),
                occurred_end=_optional_string(item.get("occurred_end")),
                mentioned_at=_optional_string(item.get("mentioned_at")),
            )
        )
    return tuple(candidates)


def _hindsight_native_projection_evidence(
    snapshot: _NativeCapsuleSnapshot,
    plan: IngestionPlanRecordV2,
    references: tuple[str, ...],
) -> HindsightProjectionEvidence:
    manifests = _contracts(snapshot, CaseManifest)
    if len(manifests) != 1 or plan.scope_id is None:
        raise ValueError("Hindsight capsule has no exact manifest or scope")
    manifest_plan = next(
        (
            item
            for item in manifests[0].ingestion_plans
            if item.ingestion_plan_id == plan.ingestion_plan_id
        ),
        None,
    )
    if manifest_plan is None:
        raise ValueError("Hindsight plan is absent from the case manifest")
    return reconstruct_hindsight_projection(
        raw_payloads=snapshot.raw_payloads,
        references=references,
        bank_id=plan.scope_id,
        ordered_source_unit_ids=plan.ordered_source_unit_ids,
        ordered_source_payload_sha256=manifest_plan.ordered_source_unit_bytes_sha256,
    )


def _accepts_hindsight_bank_profile(payload: bytes, scope_id: str) -> bool:
    try:
        parse_bank_profile(payload, expected_bank_id=scope_id)
    except ValueError:
        return False
    return True


def _accepts_hindsight_bank_config(payload: bytes, scope_id: str) -> bool:
    try:
        parse_bank_config(payload, expected_bank_id=scope_id)
    except ValueError:
        return False
    return True


def _accepts_hindsight_retain(payload: bytes, scope_id: str, item_count: int) -> bool:
    try:
        parse_retain_response(
            payload,
            expected_bank_id=scope_id,
            expected_items_count=item_count,
        )
    except ValueError:
        return False
    return True


def _single_user_message_fingerprint(payload: bytes | None) -> str | None:
    if payload is None:
        return None
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return None
    return canonical_sha256((("user", text),))


def _canonical_visible_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return "invalid"
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return "invalid"
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _all_raw_references(snapshot: _NativeCapsuleSnapshot) -> tuple[str, ...]:
    references: list[str] = []
    for plan in _contracts(snapshot, IngestionPlanRecordV2):
        references.extend(plan.scope_raw_refs)
        references.extend(plan.readiness_evidence_refs)
        if plan.inventory_raw_ref is not None:
            references.append(plan.inventory_raw_ref)
        references.extend(plan.projection_raw_refs)
    for case in _contracts(snapshot, CaseRecordV3):
        for reference in (
            case.retrieval_raw_ref,
            case.visible_evidence_raw_ref,
            case.visible_decision_ledger_raw_ref,
            case.prompt_raw_ref,
            case.judge_prompt_raw_ref,
            case.answer_raw_ref,
            case.evaluation_raw_ref,
        ):
            if reference is not None:
                references.append(reference)
        references.extend(case.retrieval_supporting_raw_refs)
        references.extend(case.pre_query_projection_raw_refs)
        references.extend(case.post_query_projection_raw_refs)
    for attempt in _contracts(snapshot, AttemptRecordV2):
        if attempt.raw_response_ref is not None:
            references.append(attempt.raw_response_ref)
        if attempt.raw_error_ref is not None:
            references.append(attempt.raw_error_ref)
    for usage in _token_usage_records(snapshot):
        if usage.raw_response_ref is not None:
            references.append(usage.raw_response_ref)
    return tuple(dict.fromkeys(references))


def _token_usage_records(
    snapshot: _NativeCapsuleSnapshot,
) -> tuple[TokenUsageRecord | TokenUsageRecordV2 | TokenUsageRecordV3, ...]:
    return (
        *_contracts(snapshot, TokenUsageRecord),
        *_contracts(snapshot, TokenUsageRecordV2),
        *_contracts(snapshot, TokenUsageRecordV3),
    )


def _operation_document(
    snapshot: _NativeCapsuleSnapshot,
    references: tuple[str, ...],
    operation: str,
) -> dict[str, Any] | None:
    for reference in references:
        document = _raw_object(snapshot, reference)
        if document is not None and document.get("operation") == operation:
            return document
    return None


def _raw_object(
    snapshot: _NativeCapsuleSnapshot,
    reference: str | None,
) -> dict[str, Any] | None:
    if reference is None:
        return None
    payload = snapshot.raw_payloads.get(reference)
    if payload is None:
        return None
    try:
        document = json.loads(payload)
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _matches(document: dict[str, Any] | None, **expected: object) -> bool:
    return document is not None and all(
        document.get(key) == value for key, value in expected.items()
    )


def _model_output_text(document: dict[str, Any] | None) -> str | None:
    if document is None:
        return None
    direct = document.get("output_text")
    if isinstance(direct, str):
        return direct
    choices = document.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    return content if isinstance(content, str) else None


def _has_exact_reference_inventory(
    record_ids: tuple[str, ...],
    referenced_ids: tuple[str, ...],
) -> bool:
    return (
        len(record_ids) == len(set(record_ids))
        and len(referenced_ids) == len(set(referenced_ids))
        and set(record_ids) == set(referenced_ids)
    )


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        return []
    return value


def _contracts(
    snapshot: _NativeCapsuleSnapshot,
    model: type[BaseModel],
) -> tuple[Any, ...]:
    return tuple(item for item in snapshot.contracts if isinstance(item, model))


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


def _find_symlink_paths(root: Path) -> tuple[str, ...]:
    if root.is_symlink():
        return (".",)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink())
    )


def _below_symlink(relative_path: str, symlink_paths: tuple[str, ...]) -> bool:
    return "." in symlink_paths or any(
        relative_path == path or relative_path.startswith(f"{path}/") for path in symlink_paths
    )
