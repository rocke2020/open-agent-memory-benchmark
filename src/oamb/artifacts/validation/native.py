"""Closed root-only validation for fixture-produced native source capsules."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from oamb.artifacts.atomic import ArtifactCollisionError, read_regular_file, sha256_file
from oamb.artifacts.validation.hindsight_evidence import (
    HindsightProjectionEvidence,
    reconstruct_hindsight_candidate_limit,
    reconstruct_hindsight_projection,
)
from oamb.artifacts.validation.mem0_evidence import (
    MEM0_PROFILE_ID,
    reconstruct_mem0_candidates,
    reconstruct_mem0_plan,
    reconstruct_mem0_projection,
)
from oamb.artifacts.validation.openviking_evidence import (
    OPENVIKING_PROFILE_ID,
    OpenVikingPlanEvidence,
    reconstruct_openviking_candidates,
    reconstruct_openviking_plan,
    reconstruct_openviking_projection,
    reconstruct_openviking_runtime_identity,
)
from oamb.artifacts.validation.openviking_session_evidence import (
    OpenVikingSessionPlanEvidence,
    reconstruct_openviking_session_candidates,
    reconstruct_openviking_session_plan,
    reconstruct_openviking_session_projection,
)
from oamb.artifacts.validation.retrieval_request import (
    GENERATION_FREE_PROOF_PROFILES,
    retrieval_request_proves_generation_free,
)
from oamb.contracts.accounting import (
    CostRecord,
    CostRecordV2,
    ResourceUsageRecord,
    ResourceUsageRecordV2,
    TokenUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
    TokenUsageRecordV5,
)
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptIntentRecordV3,
    AttemptReceiptKind,
    AttemptReceiptRecord,
    AttemptRecordV2,
    AttemptRecordV4,
    BudgetReservationRecord,
    BudgetReservationRecordV3,
    CapsuleManifest,
    CaseEvaluationDisposition,
    CaseRecordV3,
    CloseErrorRecord,
    HistoryAttemptRecord,
    InfrastructureRetryEvent,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
    OccurrenceClaimRecord,
    RunLeaseRecord,
    RunRecord,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    infrastructure_supplier_call_id,
)
from oamb.contracts.ids import (
    canonical_sha256,
    case_occurrence_id,
    ingestion_occurrence_id,
    openviking_session_id,
)
from oamb.contracts.ingestion_failures import (
    HINDSIGHT_SETTLEMENT_BASIS,
    MEM0_SETTLEMENT_BASIS,
    OPENVIKING_SETTLEMENT_BASIS,
    classify_settled_ingestion_failure,
)
from oamb.contracts.ports import NativeEvidenceCandidate
from oamb.contracts.specifications import (
    INFRASTRUCTURE_MAX_TOTAL_RETRIES,
    INFRASTRUCTURE_RETRY_BACKOFF_SECONDS,
    INFRASTRUCTURE_RETRY_POLICY_HASH,
    BudgetSpecV2,
    BudgetSpecV4,
    CaseManifest,
    DatasetManifest,
    ModelRoleBindingV2,
    RunPreflightRecord,
    RunSpec,
)
from oamb.contracts.states import (
    AttemptOutcome,
    CaseState,
    IngestionPlanState,
    RunState,
    ValidationDisposition,
)
from oamb.contracts.supplier_rejection import parse_structured_supplier_rejection
from oamb.memory_systems.hindsight.normalize import normalize_recall
from oamb.memory_systems.hindsight.profiles import (
    PROFILE_ID as HINDSIGHT_PROFILE_ID,
)
from oamb.memory_systems.hindsight.profiles import (
    parse_bank_config,
    parse_bank_profile,
    parse_retain_response,
)
from oamb.memory_systems.openviking.session_adapter import OPENVIKING_SESSION_PROFILE_ID
from oamb.workloads.visible_evidence import (
    count_o200k_tokens,
    render_compact_evidence,
    tokenizer_fingerprint,
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
    "native.retrieval-request.v1",
)

NativePlanRecord = IngestionPlanRecordV2 | IngestionPlanRecordV3
NativeAttemptRecord = AttemptRecordV2 | AttemptRecordV4
NativeIntentRecord = AttemptIntentRecord | AttemptIntentRecordV3
NativeReservationRecord = BudgetReservationRecord | BudgetReservationRecordV3
NativeTokenUsageRecord = (
    TokenUsageRecord | TokenUsageRecordV2 | TokenUsageRecordV3 | TokenUsageRecordV5
)
NativeResourceUsageRecord = ResourceUsageRecord | ResourceUsageRecordV2
NativeCostRecord = CostRecord | CostRecordV2


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
        (NATIVE_EVIDENCE_RULE_IDS[7], _retrieval_request_rule),
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
            if entry.record_kind == "embedded_part_file":
                continue
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
                    or not _source_record_identity_matches(document, entry.record_id)
                    or not _source_path_identity_matches(
                        entry.relative_path,
                        path.stem,
                        entry.record_id,
                        document,
                    )
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
    if identity == ("attempt_intent_record", 3):
        return AttemptIntentRecordV3.model_validate_json(content)
    if identity == ("attempt_receipt_record", 1):
        return AttemptReceiptRecord.model_validate_json(content)
    if identity == ("attempt_record", 2):
        return AttemptRecordV2.model_validate_json(content)
    if identity == ("attempt_record", 4):
        return AttemptRecordV4.model_validate_json(content)
    if identity == ("budget_reservation_record", 1):
        return BudgetReservationRecord.model_validate_json(content)
    if identity == ("budget_reservation_record", 3):
        return BudgetReservationRecordV3.model_validate_json(content)
    if identity == ("case_record", 3):
        return CaseRecordV3.model_validate_json(content)
    if identity == ("close_error_record", 1):
        return CloseErrorRecord.model_validate_json(content)
    if identity == ("ingestion_plan_record", 2):
        return IngestionPlanRecordV2.model_validate_json(content)
    if identity == ("ingestion_plan_record", 3):
        return IngestionPlanRecordV3.model_validate_json(content)
    if identity == ("infrastructure_retry_event", 1):
        return InfrastructureRetryEvent.model_validate_json(content)
    if identity == ("history_attempt_record", 1):
        return HistoryAttemptRecord.model_validate_json(content)
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
    if identity == ("token_usage_record", 5):
        return TokenUsageRecordV5.model_validate_json(content)
    if identity == ("resource_usage_record", 1):
        return ResourceUsageRecord.model_validate_json(content)
    if identity == ("resource_usage_record", 2):
        return ResourceUsageRecordV2.model_validate_json(content)
    if identity == ("cost_record", 1):
        return CostRecord.model_validate_json(content)
    if identity == ("cost_record", 2):
        return CostRecordV2.model_validate_json(content)
    if identity == ("run_spec", 1):
        return RunSpec.model_validate_json(content)
    if identity == ("run_preflight_record", 1):
        return RunPreflightRecord.model_validate_json(content)
    if identity == ("budget_spec", 2):
        return BudgetSpecV2.model_validate_json(content)
    if identity == ("budget_spec", 4):
        return BudgetSpecV4.model_validate_json(content)
    if identity == ("model_role_binding", 2):
        return ModelRoleBindingV2.model_validate_json(content)
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
        "infrastructure_retry_event": "retry_event_id",
        "history_attempt_record": "history_attempt_id",
        "occurrence_claim_record": "claim_id",
        "run_lease_record": "lease_record_hash",
        "run_record": "run_id",
        "token_usage_record": "usage_record_id",
        "resource_usage_record": "resource_record_id",
        "cost_record": "cost_record_id",
        "model_role_binding": "binding_id",
    }
    fixed_ids = {
        "case_manifest": "case-manifest",
        "dataset_manifest": "dataset-manifest",
        "run_spec": "run-spec",
        "run_preflight_record": "run-preflight",
        "budget_spec": "budget",
    }
    schema_name = document.get("schema_name")
    field = identity_fields.get(schema_name) if isinstance(schema_name, str) else None
    if field is not None:
        value = document.get(field)
        return value if isinstance(value, str) else None
    return fixed_ids.get(schema_name) if isinstance(schema_name, str) else None


def _source_record_identity_matches(document: dict[str, Any], record_id: str) -> bool:
    if _source_record_id(document) == record_id:
        return True
    lease_epoch = document.get("lease_epoch")
    return (
        document.get("schema_name") == "run_lease_record"
        and isinstance(lease_epoch, int)
        and str(lease_epoch) == record_id
    )


def _source_path_identity_matches(
    relative_path: str,
    path_stem: str,
    record_id: str,
    document: dict[str, Any],
) -> bool:
    if path_stem == record_id:
        return True
    schema_name = document.get("schema_name")
    fixed_paths = {
        "run_spec": "source/specs/run-spec.json",
        "run_preflight_record": "source/specs/run-preflight.json",
        "budget_spec": "source/specs/budget.json",
    }
    return isinstance(schema_name, str) and fixed_paths.get(schema_name) == relative_path


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

    plans = _ingestion_plans(snapshot)
    cases = _contracts(snapshot, CaseRecordV3)
    runs = _contracts(snapshot, RunRecord)
    datasets = _contracts(snapshot, DatasetManifest)
    case_manifests = _contracts(snapshot, CaseManifest)
    run_specs = _contracts(snapshot, RunSpec)
    run_preflights = _contracts(snapshot, RunPreflightRecord)
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
        if run_specs or run_preflights:
            budgets = _contracts(snapshot, BudgetSpecV4)
            role_bindings = _contracts(snapshot, ModelRoleBindingV2)
            if len(run_specs) != 1 or len(run_preflights) != 1 or len(budgets) != 1:
                issues.append(
                    _issue(rule_id, manifest.capsule_id, "live-control-inventory-mismatch")
                )
                expected_run_spec_hash = ""
            else:
                run_spec = run_specs[0]
                preflight = run_preflights[0]
                budget = budgets[0]
                expected_run_spec_hash = canonical_sha256(run_spec)
                persisted_role_ids = tuple(item.binding_id for item in role_bindings)
                role_inventory_closes = bool(
                    len(persisted_role_ids) == len(set(persisted_role_ids))
                    and run_spec.model_role_binding_ids == preflight.role_binding_ids
                    and set(run_spec.model_role_binding_ids) == set(persisted_role_ids)
                )
                live_control_closes = bool(
                    run_spec.run_id == manifest.run_id
                    and run_spec.dataset_manifest_hash == datasets[0].manifest_hash
                    and run_spec.case_manifest_hash == case_manifests[0].manifest_hash
                    and run_spec.workload_id == case_manifests[0].workload_id
                    and run_spec.memory_system_id == first.memory_system_id
                    and run_spec.runtime_binding_hash == first.runtime_binding_hash
                    and run_spec.budget_id == budget.budget_id
                    and role_inventory_closes
                    and preflight.run_id == manifest.run_id
                    and preflight.run_spec_hash == expected_run_spec_hash
                    and preflight.dataset_manifest_hash == datasets[0].manifest_hash
                    and preflight.subset_manifest_hash == case_manifests[0].manifest_hash
                    and preflight.adapter_profile_id == first.adapter_profile_id
                    and preflight.runtime_binding_hash == first.runtime_binding_hash
                    and preflight.budget_hash == canonical_sha256(budget)
                    and preflight.dispatch_routes == budget.dispatch_routes
                    and budget.scope_id == manifest.run_id
                )
                if not live_control_closes:
                    issues.append(
                        _issue(rule_id, manifest.capsule_id, "live-control-binding-mismatch")
                    )
        else:
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
        expected_manifest_plan_ids = {plan.ingestion_plan_id for plan in plans}
        expected_manifest_case_ids = {case.case_manifest_entry_id for case in cases}
        all_manifest_plan_ids = {plan.ingestion_plan_id for plan in case_manifest.ingestion_plans}
        all_manifest_case_ids = {case.case_manifest_entry_id for case in case_manifest.cases}
        selected_manifest_plans = tuple(
            plan
            for plan in case_manifest.ingestion_plans
            if plan.ingestion_plan_id in expected_manifest_plan_ids
        )
        plan_records_by_id = {plan.ingestion_plan_id: plan for plan in plans}
        case_records_by_manifest_id = {case.case_manifest_entry_id: case for case in cases}
        inventory_closed = bool(
            len(plan_records_by_id) == len(plans)
            and len(case_records_by_manifest_id) == len(cases)
            and set(plan_records_by_id) == expected_manifest_plan_ids
            and set(case_records_by_manifest_id) == expected_manifest_case_ids
            and len(selected_manifest_plans) == len(expected_manifest_plan_ids)
            and bool(expected_manifest_plan_ids)
            and bool(expected_manifest_case_ids)
            and expected_manifest_plan_ids <= all_manifest_plan_ids
            and expected_manifest_case_ids <= all_manifest_case_ids
        )
        plan_occurrences: list[str] = []
        case_occurrences: list[str] = []
        manifest_history_attempts = _contracts(snapshot, HistoryAttemptRecord)
        if inventory_closed:
            for manifest_plan in selected_manifest_plans:
                plan_record = plan_records_by_id[manifest_plan.ingestion_plan_id]
                selected_ready_attempts = tuple(
                    item
                    for item in manifest_history_attempts
                    if item.ingestion_plan_id == manifest_plan.ingestion_plan_id
                    and item.status == "ready"
                )
                plan_occurrence = (
                    selected_ready_attempts[0].ingestion_occurrence_id
                    if len(selected_ready_attempts) == 1
                    else ingestion_occurrence_id(
                        manifest.run_id,
                        plan_record.memory_system_id,
                        manifest_plan.ingestion_plan_id,
                    )
                )
                plan_occurrences.append(plan_occurrence)
                selected_manifest_case_ids = tuple(
                    case_manifest_entry_id
                    for case_manifest_entry_id in manifest_plan.ordered_case_manifest_entry_ids
                    if case_manifest_entry_id in expected_manifest_case_ids
                )
                if not selected_manifest_case_ids:
                    inventory_closed = False
                planned_case_occurrences = tuple(
                    case_occurrence_id(plan_occurrence, case_manifest_entry_id)
                    for case_manifest_entry_id in selected_manifest_case_ids
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
        local_history_attempts = _contracts(snapshot, HistoryAttemptRecord)
        issues.extend(_history_closure_issues(snapshot, plans, local_history_attempts))
        expected_plan_ids = (
            {item.ingestion_occurrence_id for item in local_history_attempts}
            if local_history_attempts
            else set(expected_plan_occurrence_ids)
        )
        expected_history_sequence = tuple(
            item.ingestion_occurrence_id
            for manifest_plan in case_manifests[0].ingestion_plans
            for item in sorted(
                (
                    candidate
                    for candidate in local_history_attempts
                    if candidate.ingestion_plan_id == manifest_plan.ingestion_plan_id
                ),
                key=lambda candidate: candidate.history_attempt_ordinal,
            )
        )
        expected_case_ids = set(expected_case_occurrence_ids)
        if (
            run.run_id != manifest.run_id
            or run.run_spec_hash != manifest.run_spec_hash
            or run.state != RunState.FINALIZED
            or len(run.ingestion_occurrence_ids) != len(expected_plan_ids)
            or set(run.ingestion_occurrence_ids) != expected_plan_ids
            or bool(local_history_attempts)
            and run.ingestion_occurrence_ids != expected_history_sequence
            or len(run.case_occurrence_ids) != len(expected_case_ids)
            or set(run.case_occurrence_ids) != expected_case_ids
        ):
            issues.append(_issue(rule_id, manifest.capsule_id, "native-run-terminal-mismatch"))
    leases = _contracts(snapshot, RunLeaseRecord)
    attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
    intent_records = _attempt_intent_records(snapshot)
    intents = {item.attempt_id: item for item in intent_records}
    receipts = {item.attempt_id: item for item in _contracts(snapshot, AttemptReceiptRecord)}
    claim_records = _contracts(snapshot, OccurrenceClaimRecord)
    claims = {item.claim_id: item for item in claim_records}
    reservation_records = _budget_reservation_records(snapshot)
    reservations = {item.reservation_id: item for item in reservation_records}
    for lease in leases:
        expected_lease_hash = canonical_sha256(
            lease.model_dump(mode="python", exclude={"lease_record_hash"})
        )
        if lease.lease_record_hash != expected_lease_hash:
            issues.append(_issue(rule_id, lease.lease_record_hash, "run-lease-identity-mismatch"))
    leases_by_hash = _strict_run_lease_chain(leases)
    if leases and leases_by_hash is None:
        issues.append(_issue(rule_id, manifest.capsule_id, "run-lease-chain-mismatch"))
    for claim in claim_records:
        claim_fields = claim.model_dump(
            mode="python",
            exclude={"schema_name", "schema_version", "claim_id"},
        )
        expected_claim_id = canonical_sha256(["oamb-native-occurrence-claim-v1", claim_fields])
        if claim.claim_id != expected_claim_id:
            issues.append(_issue(rule_id, claim.claim_id, "attempt-claim-identity-mismatch"))
    for reservation_record in reservation_records:
        if isinstance(reservation_record, BudgetReservationRecordV3):
            continue
        reservation_fields = reservation_record.model_dump(
            mode="python",
            exclude={"schema_name", "schema_version", "reservation_id"},
        )
        expected_reservation_id = canonical_sha256(
            ["oamb-native-budget-reservation-v1", reservation_fields]
        )
        if reservation_record.reservation_id != expected_reservation_id:
            issues.append(
                _issue(
                    rule_id,
                    reservation_record.reservation_id,
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
    if (
        not leases
        or leases_by_hash is None
        or set(attempts) != set(intents)
        or set(attempts) != set(receipts)
    ):
        issues.append(_issue(rule_id, manifest.capsule_id, "attempt-evidence-incomplete"))
    else:
        for attempt_id, attempt in attempts.items():
            intent = intents[attempt_id]
            receipt = receipts[attempt_id]
            claim = claims.get(intent.claim_id)
            lease = leases_by_hash.get(claim.lease_record_hash) if claim is not None else None
            reservation = reservations.get(intent.reservation_id)
            common_aligned = bool(
                claim is not None
                and lease is not None
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
                and intent.sealed_at <= attempt.started_at
                and receipt.dispatch_started_at == attempt.started_at
                and receipt.receipt_observed_at == attempt.ended_at
                and receipt.raw_response_ref == attempt.raw_response_ref
                and receipt.raw_error_ref == attempt.raw_error_ref
            )
            if (
                isinstance(attempt, AttemptRecordV4)
                and isinstance(intent, AttemptIntentRecordV3)
                and isinstance(reservation, BudgetReservationRecordV3)
            ):
                attempt_preflight = run_preflights[0] if len(run_preflights) == 1 else None
                budgets = _contracts(snapshot, BudgetSpecV4)
                budget = budgets[0] if len(budgets) == 1 else None
                route = (
                    next(
                        (
                            item
                            for item in budget.dispatch_routes
                            if item.route_id == attempt.dispatch_route_id
                        ),
                        None,
                    )
                    if budget is not None
                    else None
                )
                aligned = bool(
                    common_aligned
                    and attempt_preflight is not None
                    and budget is not None
                    and route is not None
                    and attempt.run_id == manifest.run_id
                    and attempt.intent_hash == intent.intent_hash
                    and attempt.receipt_record_hash == canonical_sha256(receipt)
                    and intent.reservation_hash == reservation.reservation_hash
                    and intent.scope_id == reservation.scope_id == manifest.run_id
                    and intent.preflight_record_hash == attempt_preflight.preflight_record_hash
                    and intent.budget_id == reservation.budget_id == budget.budget_id
                    and intent.budget_hash == reservation.budget_hash == budget.budget_hash
                    and attempt.dispatch_route_id
                    == intent.dispatch_route_id
                    == reservation.dispatch_route_id
                    == route.route_id
                    and attempt.dispatch_route_hash
                    == intent.dispatch_route_hash
                    == reservation.dispatch_route_hash
                    == route.route_hash
                )
            elif (
                isinstance(attempt, AttemptRecordV2)
                and isinstance(intent, AttemptIntentRecord)
                and isinstance(reservation, BudgetReservationRecord)
            ):
                aligned = bool(
                    common_aligned
                    and reservation.scope_id == attempt.parent_id
                    and reservation.role_binding_id == intent.role_binding_id
                )
            else:
                aligned = False
            if not aligned:
                issues.append(_issue(rule_id, attempt_id, "attempt-evidence-mismatch"))
            if (
                attempt.stage in {"answer", "judge"}
                and attempt.outcome == AttemptOutcome.FAILED
                and attempt.raw_error_ref is not None
                and (
                    rejection := parse_structured_supplier_rejection(
                        snapshot.raw_payloads.get(attempt.raw_error_ref, b""),
                        status_code=429,
                    )
                )
                is not None
                and rejection.internal_retry_count != 0
            ):
                issues.append(_issue(rule_id, attempt_id, "model-supplier-retry-proof-drift"))
    issues.extend(_infrastructure_retry_issues(snapshot, attempts, runs))
    return tuple(issues)


def _history_closure_issues(
    snapshot: _NativeCapsuleSnapshot,
    plans: tuple[NativePlanRecord, ...],
    attempts: tuple[HistoryAttemptRecord, ...],
) -> tuple[ValidationIssue, ...]:
    if not attempts:
        return ()
    rule_id = "native.manifest-schema.v1"
    issues: list[ValidationIssue] = []
    operation_attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
    plans_by_id = {item.ingestion_plan_id: item for item in plans}
    if len({item.history_attempt_id for item in attempts}) != len(attempts):
        return (_issue(rule_id, "history", "history-record-identity-duplicate"),)
    attempts_by_plan: dict[str, list[HistoryAttemptRecord]] = {}
    for item in attempts:
        attempts_by_plan.setdefault(item.ingestion_plan_id, []).append(item)
    if set(attempts_by_plan) != set(plans_by_id):
        issues.append(_issue(rule_id, "history", "history-ready-selection-mismatch"))
    for plan_id, chain_items in attempts_by_plan.items():
        chain = tuple(sorted(chain_items, key=lambda item: item.history_attempt_ordinal))
        first = chain[0]
        if (
            len(chain) != 1
            or first.history_attempt_ordinal != 1
            or first.previous_retry_event_id is not None
        ):
            issues.append(_issue(rule_id, plan_id, "history-attempt-chain-mismatch"))
        plan = plans_by_id.get(plan_id)
        ready = tuple(item for item in chain if item.status == "ready")
        if (
            len(ready) != 1
            or plan is None
            or ready[0].ingestion_occurrence_id != plan.ingestion_occurrence_id
            or ready[0].ingestion_plan_record_hash != canonical_sha256(plan)
        ):
            issues.append(_issue(rule_id, plan_id, "history-ready-selection-mismatch"))
        for history_attempt in chain:
            if any(
                reference not in snapshot.raw_payloads
                for reference in history_attempt.scope_raw_refs
            ):
                issues.append(
                    _issue(
                        rule_id,
                        history_attempt.history_attempt_id,
                        "history-scope-evidence-missing",
                    )
                )
            if any(
                reference not in snapshot.raw_payloads
                for reference in history_attempt.settlement_evidence_refs
            ):
                issues.append(
                    _issue(
                        rule_id,
                        history_attempt.history_attempt_id,
                        "history-settlement-evidence-missing",
                    )
                )
            if any(
                operation_attempts.get(attempt_id) is None
                or operation_attempts[attempt_id].parent_id
                != history_attempt.ingestion_occurrence_id
                for attempt_id in history_attempt.operation_attempt_ids
            ):
                issues.append(
                    _issue(
                        rule_id,
                        history_attempt.history_attempt_id,
                        "history-operation-inventory-mismatch",
                    )
                )
            if history_attempt.status == "retryable_failed_settled":
                terminal = operation_attempts.get(history_attempt.terminal_failure_attempt_id or "")
                raw_ref = terminal.raw_error_ref if terminal is not None else None
                raw = snapshot.raw_payloads.get(raw_ref or "")
                classified = (
                    classify_settled_ingestion_failure(
                        settlement_basis=history_attempt.settlement_basis or "",
                        status_code=history_attempt.settlement_status_code or 0,
                        raw_response_bytes=raw or b"",
                        internal_retry_count=history_attempt.internal_retry_count,
                        expected_task_id=history_attempt.settlement_task_id,
                        expected_session_id=history_attempt.settlement_session_id,
                    )
                    if raw is not None
                    else None
                )
                if (
                    raw_ref not in history_attempt.settlement_evidence_refs
                    or classified != history_attempt.failure_kind
                ):
                    issues.append(
                        _issue(
                            rule_id,
                            history_attempt.history_attempt_id,
                            "history-failure-classification-mismatch",
                        )
                    )
    return tuple(issues)


def _infrastructure_retry_issues(
    snapshot: _NativeCapsuleSnapshot,
    attempts: dict[str, NativeAttemptRecord],
    runs: tuple[RunRecord, ...],
) -> tuple[ValidationIssue, ...]:
    rule_id = "native.manifest-schema.v1"
    events = _contracts(snapshot, InfrastructureRetryEvent)
    expected_policy_hash = INFRASTRUCTURE_RETRY_POLICY_HASH
    if not events:
        if len(runs) == 1 and runs[0].state == RunState.INFRASTRUCTURE_BLOCKED:
            return (
                _issue(
                    rule_id,
                    runs[0].run_id,
                    "infrastructure-retry-evidence-mismatch",
                ),
            )
        return ()
    usage_by_id = {usage.usage_record_id: usage for usage in _token_usage_records(snapshot)}
    issues: list[ValidationIssue] = []
    by_attempt: dict[str, list[InfrastructureRetryEvent]] = {}
    supplier_call_ids: list[str] = []
    infrastructure_usage_ids: list[str] = []
    observed_retry_units = 0
    for event in events:
        by_attempt.setdefault(event.logical_attempt_id, []).append(event)
        supplier_call_ids.append(event.supplier_call_id)
        infrastructure_usage_ids.extend(event.usage_record_ids)
        observed_retry_units += event.internal_retry_count + int(event.retry_scheduled)
    for logical_attempt_id, attempt_events in by_attempt.items():
        ordered = tuple(sorted(attempt_events, key=lambda item: item.supplier_call_ordinal))
        ordinals = tuple(item.supplier_call_ordinal for item in ordered)
        expected_ordinals = tuple(range(1, len(ordered) + 1))
        terminal_unscheduled = tuple(item for item in ordered if not item.retry_scheduled)
        attempt = attempts.get(logical_attempt_id)
        event_usage_closes = all(
            bool(event.usage_record_ids)
            and all(
                (usage := usage_by_id.get(usage_id)) is not None
                and usage.attempt_id == event.supplier_call_id
                and usage.raw_response_ref == event.raw_error_ref
                for usage_id in event.usage_record_ids
            )
            for event in ordered
        )
        event_proof_closes = all(
            (
                classification := parse_structured_supplier_rejection(
                    snapshot.raw_payloads.get(event.raw_error_ref, b""),
                    status_code=event.status,
                )
            )
            is not None
            and (
                classification.origin,
                classification.failure_kind,
                classification.status,
                classification.acceptance,
                classification.provider_mutation,
                classification.retryable,
                classification.internal_retry_count,
            )
            == (
                event.origin,
                event.failure_kind,
                event.status,
                event.acceptance,
                event.provider_mutation,
                event.retryable,
                event.internal_retry_count,
            )
            and event.supplier_call_id
            == infrastructure_supplier_call_id(
                logical_attempt_id,
                event.supplier_call_ordinal,
                event.raw_error_ref,
            )
            for event in ordered
        )
        attempt_binding_closes = (
            bool(
                attempt is not None
                and all(
                    event.run_id == runs[0].run_id
                    and event.parent_kind == attempt.parent_kind
                    and event.parent_id == attempt.parent_id
                    and event.stage == attempt.stage
                    for event in ordered
                )
            )
            if len(runs) == 1
            else False
        )
        sequence_closes = bool(
            attempt_binding_closes
            and ordinals == expected_ordinals
            and all(
                event.retry_policy_hash == expected_policy_hash == INFRASTRUCTURE_RETRY_POLICY_HASH
                and (
                    event.backoff_seconds
                    == INFRASTRUCTURE_RETRY_BACKOFF_SECONDS[event.supplier_call_ordinal - 1]
                    if event.retry_scheduled
                    and event.supplier_call_ordinal <= len(INFRASTRUCTURE_RETRY_BACKOFF_SECONDS)
                    else not event.retry_scheduled
                )
                for event in ordered
            )
            and (
                not terminal_unscheduled
                or terminal_unscheduled == (ordered[-1],)
                and len(runs) == 1
                and runs[0].state == RunState.INFRASTRUCTURE_BLOCKED
            )
            and event_usage_closes
            and event_proof_closes
        )
        if not sequence_closes:
            issues.append(
                _issue(rule_id, logical_attempt_id, "infrastructure-retry-evidence-mismatch")
            )
    run_retry_closes = bool(
        len(set(supplier_call_ids)) == len(supplier_call_ids)
        and len(set(infrastructure_usage_ids)) == len(infrastructure_usage_ids)
        and observed_retry_units <= INFRASTRUCTURE_MAX_TOTAL_RETRIES
        and (
            len(runs) != 1
            or runs[0].state != RunState.INFRASTRUCTURE_BLOCKED
            or any(not event.retry_scheduled for event in events)
        )
    )
    if not run_retry_closes:
        target = runs[0].run_id if len(runs) == 1 else "infrastructure-retry-run"
        issues.append(_issue(rule_id, target, "infrastructure-retry-evidence-mismatch"))
    return tuple(issues)


def _strict_run_lease_chain(
    leases: tuple[RunLeaseRecord, ...],
) -> dict[str, RunLeaseRecord] | None:
    if not leases:
        return {}
    hashes = tuple(lease.lease_record_hash for lease in leases)
    epochs = tuple(lease.lease_epoch for lease in leases)
    if len(set(hashes)) != len(hashes) or len(set(epochs)) != len(epochs):
        return None
    ordered = tuple(sorted(leases, key=lambda lease: lease.lease_epoch))
    if tuple(lease.lease_epoch for lease in ordered) != tuple(range(1, len(ordered) + 1)):
        return None
    first = ordered[0]
    stable_binding = (
        first.run_id,
        first.provider_project_id,
        first.provider_profile_id,
    )
    for predecessor, successor in zip(ordered, ordered[1:], strict=False):
        if (
            successor.predecessor_lease_record_hash != predecessor.lease_record_hash
            or (
                successor.run_id,
                successor.provider_project_id,
                successor.provider_profile_id,
            )
            != stable_binding
        ):
            return None
    return {lease.lease_record_hash: lease for lease in ordered}


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
    attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
    usage_records = {item.usage_record_id: item for item in _token_usage_records(snapshot)}
    resource_records = {item.resource_record_id: item for item in _resource_usage_records(snapshot)}
    cost_records = {item.cost_record_id: item for item in _cost_records(snapshot)}
    receipts = {item.attempt_id: item for item in _contracts(snapshot, AttemptReceiptRecord)}
    issues: list[ValidationIssue] = []
    for plan in _ingestion_plans(snapshot):
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
        for final_attempt_id, dispatch_source_ids in zip(
            plan.ordered_dispatch_attempt_ids,
            plan.ordered_dispatch_source_unit_ids,
            strict=True,
        ):
            final_attempt = attempts.get(final_attempt_id)
            if final_attempt is not None and not _dispatch_retry_chain_closes(
                plan=plan,
                source_ids=dispatch_source_ids,
                final_attempt=final_attempt,
                attempts=attempts,
                receipts=receipts,
                raw_payloads=snapshot.raw_payloads,
            ):
                issues.append(_issue(rule_id, final_attempt_id, "dispatch-retry-chain-mismatch"))
        if plan.adapter_profile_id == HINDSIGHT_PROFILE_ID:
            if not isinstance(plan, IngestionPlanRecordV2):
                issues.append(
                    _issue(
                        rule_id,
                        plan.ingestion_occurrence_id,
                        "hindsight-plan-version-mismatch",
                    )
                )
                continue
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
            if not isinstance(plan, IngestionPlanRecordV2):
                issues.append(
                    _issue(
                        rule_id,
                        plan.ingestion_occurrence_id,
                        "openviking-plan-version-mismatch",
                    )
                )
                continue
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
        if plan.adapter_profile_id == OPENVIKING_SESSION_PROFILE_ID:
            if not isinstance(plan, IngestionPlanRecordV2):
                issues.append(
                    _issue(
                        rule_id,
                        plan.ingestion_occurrence_id,
                        "openviking-session-plan-version-mismatch",
                    )
                )
                continue
            try:
                _openviking_session_native_plan_evidence(snapshot, plan, attempts)
            except ValueError:
                issues.append(
                    _issue(
                        rule_id,
                        plan.ingestion_occurrence_id,
                        "openviking-session-plan-evidence-mismatch",
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
        if plan.adapter_profile_id == MEM0_PROFILE_ID:
            try:
                if not isinstance(plan, IngestionPlanRecordV3):
                    raise ValueError("Mem0 REST requires ingestion-plan evidence v3")
                reconstruct_mem0_plan(
                    raw_payloads=snapshot.raw_payloads,
                    plan=plan,
                    attempts=cast(dict[str, AttemptRecordV2], attempts),
                )
            except ValueError:
                issues.append(
                    _issue(
                        rule_id,
                        plan.ingestion_occurrence_id,
                        "mem0-plan-evidence-mismatch",
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
        skipped: list[str] = []
        dispatched: list[str] = []
        for attempt_id, expected_dispatch_source_ids in zip(
            plan.ordered_dispatch_attempt_ids,
            plan.ordered_dispatch_source_unit_ids,
            strict=True,
        ):
            attempt = attempts.get(attempt_id)
            skipped_dispatch = set(expected_dispatch_source_ids) <= set(
                plan.skipped_source_unit_ids
            )
            if (
                attempt is None
                or attempt.parent_kind != "ingestion_plan"
                or attempt.parent_id != plan.ingestion_occurrence_id
                or attempt.stage != "memory_ingest"
            ):
                issues.append(_issue(rule_id, attempt_id, "dispatch-attempt-mismatch"))
                continue
            if skipped_dispatch:
                if attempt.outcome != AttemptOutcome.FAILED or attempt.raw_error_ref is None:
                    issues.append(_issue(rule_id, attempt_id, "dispatch-attempt-mismatch"))
                    continue
                dispatched.extend(expected_dispatch_source_ids)
                skipped.extend(expected_dispatch_source_ids)
                continue
            if attempt.outcome != AttemptOutcome.SUCCEEDED or attempt.raw_response_ref is None:
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
            or tuple(skipped) != plan.skipped_source_unit_ids
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


def _dispatch_retry_chain_closes(
    *,
    plan: NativePlanRecord,
    source_ids: tuple[str, ...],
    final_attempt: NativeAttemptRecord,
    attempts: dict[str, NativeAttemptRecord],
    receipts: dict[str, AttemptReceiptRecord],
    raw_payloads: Mapping[str, bytes],
) -> bool:
    chain = [final_attempt]
    seen = {final_attempt.attempt_id}
    predecessor_id = final_attempt.retry_of_attempt_id
    while predecessor_id is not None:
        predecessor = attempts.get(predecessor_id)
        if predecessor is None or predecessor_id in seen:
            return False
        chain.append(predecessor)
        seen.add(predecessor_id)
        predecessor_id = predecessor.retry_of_attempt_id
    chain.reverse()
    skipped = set(source_ids) <= set(plan.skipped_source_unit_ids)
    if not 1 <= len(chain) <= 3 or (skipped and len(chain) != 3):
        return False
    if chain[0].retry_of_attempt_id is not None or any(
        current.retry_of_attempt_id != previous.attempt_id
        for previous, current in zip(chain, chain[1:], strict=False)
    ):
        return False
    if any(
        (
            attempt.parent_kind,
            attempt.parent_id,
            attempt.stage,
            attempt.ordinal,
            attempt.request_fingerprint,
        )
        != (
            "ingestion_plan",
            final_attempt.parent_id,
            "memory_ingest",
            final_attempt.ordinal,
            final_attempt.request_fingerprint,
        )
        for attempt in chain
    ):
        return False
    matching_attempt_ids = {
        attempt.attempt_id
        for attempt in attempts.values()
        if (
            attempt.parent_kind,
            attempt.parent_id,
            attempt.stage,
            attempt.ordinal,
            attempt.request_fingerprint,
        )
        == (
            "ingestion_plan",
            final_attempt.parent_id,
            "memory_ingest",
            final_attempt.ordinal,
            final_attempt.request_fingerprint,
        )
    }
    if matching_attempt_ids != seen or not seen <= set(plan.attempt_ids):
        return False
    expected_failed_count = len(chain) if skipped else len(chain) - 1
    if any(
        attempt.outcome != AttemptOutcome.FAILED
        or not _settled_ingestion_receipt_closes(
            attempt,
            receipts.get(attempt.attempt_id),
            raw_payloads,
            plan=plan,
            source_ids=source_ids,
            batch_attempt_ordinal=index + 1,
        )
        for index, attempt in enumerate(chain[:expected_failed_count])
    ):
        return False
    if skipped:
        return final_attempt.outcome == AttemptOutcome.FAILED
    final_receipt = receipts.get(final_attempt.attempt_id)
    return bool(
        final_attempt.outcome == AttemptOutcome.SUCCEEDED
        and final_receipt is not None
        and final_receipt.receipt_kind == AttemptReceiptKind.RESPONSE
        and final_receipt.raw_response_ref == final_attempt.raw_response_ref
    )


def _settled_ingestion_receipt_closes(
    attempt: NativeAttemptRecord,
    receipt: AttemptReceiptRecord | None,
    raw_payloads: Mapping[str, bytes],
    *,
    plan: NativePlanRecord,
    source_ids: tuple[str, ...],
    batch_attempt_ordinal: int,
) -> bool:
    if (
        receipt is None
        or receipt.receipt_kind != AttemptReceiptKind.ERROR
        or receipt.raw_error_ref != attempt.raw_error_ref
        or receipt.raw_error_ref is None
        or receipt.failure_kind is None
        or receipt.supplier_status_code is None
        or receipt.settlement_basis is None
        or receipt.internal_retry_count is None
    ):
        return False
    expected_basis = {
        "hindsight": HINDSIGHT_SETTLEMENT_BASIS,
        "mem0": MEM0_SETTLEMENT_BASIS,
        "openviking": OPENVIKING_SETTLEMENT_BASIS,
    }.get(plan.memory_system_id)
    if receipt.settlement_basis != expected_basis:
        return False
    expected_session_id = receipt.settlement_session_id
    if plan.memory_system_id == "openviking":
        if len(source_ids) != 1:
            return False
        session_identity = source_ids[0]
        if batch_attempt_ordinal > 1:
            session_identity = f"{session_identity}:batch-attempt:{batch_attempt_ordinal}"
        expected_session_id = openviking_session_id(
            plan.ingestion_occurrence_id,
            session_identity,
        )
        if receipt.settlement_session_id != expected_session_id:
            return False
    elif receipt.settlement_task_id is not None or receipt.settlement_session_id is not None:
        return False
    raw = raw_payloads.get(receipt.raw_error_ref)
    return bool(
        raw is not None
        and classify_settled_ingestion_failure(
            settlement_basis=receipt.settlement_basis,
            status_code=receipt.supplier_status_code,
            raw_response_bytes=raw,
            internal_retry_count=receipt.internal_retry_count,
            expected_task_id=receipt.settlement_task_id,
            expected_session_id=receipt.settlement_session_id,
        )
        == receipt.failure_kind
    )


def _plan_usage_closes(
    plan: NativePlanRecord,
    attempts: dict[str, NativeAttemptRecord],
    usage_records: dict[str, NativeTokenUsageRecord],
) -> bool:
    if not plan.ordered_dispatch_attempt_ids or len(plan.usage_record_ids) != len(
        set(plan.usage_record_ids)
    ):
        return False
    dispatch_attempt_ids = {
        attempt_id
        for attempt_id in plan.attempt_ids
        if (attempt := attempts.get(attempt_id)) is not None and attempt.stage == "memory_ingest"
    }
    expected_usage_ids = {
        usage.usage_record_id
        for usage in usage_records.values()
        if usage.attempt_id in dispatch_attempt_ids
    }
    if set(plan.usage_record_ids) != expected_usage_ids:
        return False
    covered_attempt_ids: set[str] = set()
    for usage_record_id in plan.usage_record_ids:
        usage = usage_records.get(usage_record_id)
        attempt = attempts.get(usage.attempt_id) if usage is not None else None
        if (
            usage is None
            or attempt is None
            or usage.attempt_id not in dispatch_attempt_ids
            or usage.parent_kind != "ingestion_plan"
            or usage.parent_id != plan.ingestion_occurrence_id
            or usage.stage.value != "memory_ingest"
            or usage.raw_response_ref != (attempt.raw_response_ref or attempt.raw_error_ref)
        ):
            return False
        covered_attempt_ids.add(usage.attempt_id)
    return covered_attempt_ids == dispatch_attempt_ids


def _accounting_ledger_closes(
    *,
    parent_kind: str,
    parent_id: str,
    attempt_ids: tuple[str, ...],
    usage_record_ids: tuple[str, ...],
    resource_record_ids: tuple[str, ...],
    cost_record_ids: tuple[str, ...],
    attempts: dict[str, NativeAttemptRecord],
    usage_records: dict[str, NativeTokenUsageRecord],
    resource_records: dict[str, NativeResourceUsageRecord],
    cost_records: dict[str, NativeCostRecord],
) -> bool:
    if not attempt_ids:
        return False
    if any(
        (attempt := attempts.get(attempt_id)) is None
        or attempt.parent_kind != parent_kind
        or attempt.parent_id != parent_id
        for attempt_id in attempt_ids
    ):
        return False
    accountable_attempt_ids = tuple(
        attempt_id
        for attempt_id in attempt_ids
        if not (
            attempts[attempt_id].outcome == AttemptOutcome.UNKNOWN_OUTCOME
            and attempts[attempt_id].raw_response_ref is None
            and attempts[attempt_id].raw_error_ref is None
        )
    )
    live_accounting = any(
        isinstance(attempts.get(attempt_id), AttemptRecordV4) for attempt_id in attempt_ids
    )
    if live_accounting:
        if not (
            len(accountable_attempt_ids) == len(resource_record_ids) == len(cost_record_ids)
            and len(usage_record_ids) == len(set(usage_record_ids))
        ):
            return False
        expected_usage_ids = tuple(
            usage.usage_record_id
            for attempt_id in accountable_attempt_ids
            for usage in usage_records.values()
            if usage.attempt_id == attempt_id
        )
        if len(usage_record_ids) != len(expected_usage_ids) or set(usage_record_ids) != set(
            expected_usage_ids
        ):
            return False
    elif not (
        len(accountable_attempt_ids)
        == len(usage_record_ids)
        == len(resource_record_ids)
        == len(cost_record_ids)
    ):
        return False
    for ordinal, (attempt_id, resource_id, cost_id) in enumerate(
        zip(
            accountable_attempt_ids,
            resource_record_ids,
            cost_record_ids,
            strict=True,
        )
    ):
        attempt = attempts.get(attempt_id)
        attempt_usage_ids = tuple(
            usage.usage_record_id
            for usage in usage_records.values()
            if usage.attempt_id == attempt_id
        )
        usage_id = (
            attempt_usage_ids[0]
            if live_accounting and len(attempt_usage_ids) == 1
            else usage_record_ids[ordinal]
            if not live_accounting
            else None
        )
        usage = usage_records.get(usage_id) if usage_id is not None else None
        resource = resource_records.get(resource_id)
        cost = cost_records.get(cost_id)
        expected_indexing_view = (
            "final_contribution"
            if attempt is not None
            and attempt.stage == "memory_ingest"
            and attempt.outcome == AttemptOutcome.SUCCEEDED
            else "attempted"
            if attempt is not None and attempt.stage == "memory_ingest"
            else "not_applicable"
        )
        raw_attempt_ref = (
            attempt.raw_response_ref or attempt.raw_error_ref if attempt is not None else None
        )
        if (
            attempt is None
            or resource is None
            or cost is None
            or attempt.parent_kind != parent_kind
            or attempt.parent_id != parent_id
            or resource.parent_kind != parent_kind
            or resource.parent_id != parent_id
            or resource.stage != attempt.stage
            or resource.raw_telemetry_ref != raw_attempt_ref
            or cost.parent_kind != parent_kind
            or cost.parent_id != parent_id
            or cost.indexing_view.value != expected_indexing_view
            or cost.source_resource_record_ids != (resource_id,)
        ):
            return False
        if live_accounting:
            if (
                len(cost.source_usage_record_ids) != len(attempt_usage_ids)
                or set(cost.source_usage_record_ids) != set(attempt_usage_ids)
                or not isinstance(resource, ResourceUsageRecordV2)
                or not isinstance(cost, CostRecordV2)
                or resource.attempt_id != attempt_id
                or cost.attempt_id != attempt_id
                or not isinstance(attempt, AttemptRecordV4)
                or not (
                    resource.dispatch_route_id
                    == cost.dispatch_route_id
                    == attempt.dispatch_route_id
                )
                or not (
                    resource.dispatch_route_hash
                    == cost.dispatch_route_hash
                    == attempt.dispatch_route_hash
                )
                or resource.budget_owner_kind != cost.budget_owner_kind
                or resource.budget_owner_id != cost.budget_owner_id
            ):
                return False
            for attempt_usage_id in attempt_usage_ids:
                attempt_usage = usage_records.get(attempt_usage_id)
                if (
                    not isinstance(attempt_usage, TokenUsageRecordV5)
                    or attempt_usage.parent_kind != parent_kind
                    or attempt_usage.parent_id != parent_id
                    or attempt_usage.stage.value != attempt.stage
                    or attempt_usage.raw_response_ref != raw_attempt_ref
                    or attempt_usage.dispatch_route_id != attempt.dispatch_route_id
                    or attempt_usage.dispatch_route_hash != attempt.dispatch_route_hash
                ):
                    return False
        elif (
            usage is None
            or usage.attempt_id != attempt_id
            or usage.parent_kind != parent_kind
            or usage.parent_id != parent_id
            or usage.stage.value != attempt.stage
            or usage.raw_response_ref != raw_attempt_ref
            or cost.source_usage_record_ids != (usage_id,)
        ):
            return False
    return True


def _openviking_native_plan_evidence(
    snapshot: _NativeCapsuleSnapshot,
    plan: IngestionPlanRecordV2,
    attempts: dict[str, NativeAttemptRecord],
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
        attempts=cast(dict[str, AttemptRecordV2], attempts),
        runtime_identity=reconstruct_openviking_runtime_identity(snapshot.raw_payloads),
    )


def _openviking_session_native_plan_evidence(
    snapshot: _NativeCapsuleSnapshot,
    plan: IngestionPlanRecordV2,
    attempts: dict[str, NativeAttemptRecord],
) -> OpenVikingSessionPlanEvidence:
    runtime_identity = reconstruct_openviking_runtime_identity(snapshot.raw_payloads)
    return reconstruct_openviking_session_plan(
        raw_payloads=snapshot.raw_payloads,
        plan=plan,
        attempts=attempts,
        runtime_account_id=runtime_identity.account_id,
        runtime_admin_user_id=runtime_identity.user_id,
    )


def _hindsight_plan_closure(
    snapshot: _NativeCapsuleSnapshot,
    plan: IngestionPlanRecordV2,
    attempts: dict[str, NativeAttemptRecord],
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
        skipped_dispatch = set(expected_source_ids) <= set(plan.skipped_source_unit_ids)
        raw_reference = (
            attempt.raw_error_ref
            if skipped_dispatch and attempt is not None
            else attempt.raw_response_ref
            if attempt is not None
            else None
        )
        payload = snapshot.raw_payloads.get(raw_reference) if raw_reference is not None else None
        if (
            attempt is None
            or attempt.parent_kind != "ingestion_plan"
            or attempt.parent_id != plan.ingestion_occurrence_id
            or attempt.stage != "memory_ingest"
            or payload is None
            or (
                skipped_dispatch
                and (attempt.outcome != AttemptOutcome.FAILED or attempt.raw_error_ref is None)
            )
            or (
                not skipped_dispatch
                and (
                    attempt.outcome != AttemptOutcome.SUCCEEDED
                    or attempt.raw_response_ref is None
                    or not _accepts_hindsight_retain(
                        payload,
                        expected_scope_id,
                        len(expected_source_ids),
                    )
                )
            )
        ):
            issues.append(_issue(rule_id, attempt_id, "dispatch-receipt-mismatch"))
            continue
        dispatched.extend(expected_source_ids)
        if raw_reference is not None:
            dispatch_raw_refs.append(raw_reference)
    if tuple(dispatched) != plan.ordered_source_unit_ids or set(
        plan.accepted_source_unit_ids
    ) | set(plan.rejected_source_unit_ids) | set(plan.skipped_source_unit_ids) != set(
        plan.ordered_source_unit_ids
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
    if projection is None or projection.ordered_source_unit_ids != plan.projected_source_unit_ids:
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
    attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
    usage_records = {item.usage_record_id: item for item in _token_usage_records(snapshot)}
    resource_records = {item.resource_record_id: item for item in _resource_usage_records(snapshot)}
    cost_records = {item.cost_record_id: item for item in _cost_records(snapshot)}
    plans = {item.ingestion_occurrence_id: item for item in _ingestion_plans(snapshot)}
    issues: list[ValidationIssue] = []
    for case in _contracts(snapshot, CaseRecordV3):
        plan = plans.get(case.ingestion_occurrence_id)
        if (
            case.state not in {CaseState.COMPLETED, CaseState.ERROR}
            or (
                case.state == CaseState.ERROR
                and not (
                    (
                        case.evaluation_disposition == CaseEvaluationDisposition.UNJUDGED
                        and case.error_stage == "judge"
                    )
                    or (
                        case.evaluation_disposition == CaseEvaluationDisposition.NOT_RUN
                        and case.error_stage == "answer"
                    )
                )
            )
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
        pre_query_attempts = [
            item
            for item in case_attempts
            if item is not None and item.stage == "pre_query_projection"
        ]
        post_query_attempts = [
            item
            for item in case_attempts
            if item is not None and item.stage == "post_query_projection"
        ]
        expected_judge = case.evaluation_disposition == CaseEvaluationDisposition.JUDGED
        expected_judge_attempts = case.evaluation_disposition in {
            CaseEvaluationDisposition.JUDGED,
            CaseEvaluationDisposition.UNJUDGED,
        }
        live_case = any(isinstance(item, AttemptRecordV4) for item in case_attempts)
        projection_attempts_close = (
            len(pre_query_attempts) == len(post_query_attempts) == 1
            and pre_query_attempts[0].raw_response_ref in case.pre_query_projection_raw_refs
            and post_query_attempts[0].raw_response_ref in case.post_query_projection_raw_refs
        )
        if (
            len(query_attempts) != 1
            or query_attempts[0].raw_response_ref != case.retrieval_raw_ref
            or not _model_attempt_chain_closes(
                snapshot,
                answer_attempts,
                prompt_ref=case.prompt_raw_ref,
                final_response_ref=case.answer_raw_ref,
                require_success=(case.evaluation_disposition != CaseEvaluationDisposition.NOT_RUN),
            )
            or bool(judge_attempts) != expected_judge_attempts
            or (
                expected_judge
                and not _model_attempt_chain_closes(
                    snapshot,
                    judge_attempts,
                    prompt_ref=case.judge_prompt_raw_ref,
                    final_response_ref=None,
                    require_success=True,
                )
            )
            or (
                case.evaluation_disposition == CaseEvaluationDisposition.UNJUDGED
                and not _model_attempt_chain_closes(
                    snapshot,
                    judge_attempts,
                    prompt_ref=case.judge_prompt_raw_ref,
                    final_response_ref=None,
                    require_success=False,
                )
            )
            or (
                live_case
                and (
                    not projection_attempts_close
                    or len(case_attempts) != 3 + len(answer_attempts) + len(judge_attempts)
                )
            )
            or (
                not live_case
                and len(case_attempts) != 1 + len(answer_attempts) + len(judge_attempts)
            )
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
        answer_messages_sha256 = _single_user_message_fingerprint(prompt)
        judge_prompt = snapshot.raw_payloads.get(case.judge_prompt_raw_ref or "")
        judge_messages_sha256 = _single_user_message_fingerprint(judge_prompt)
        answer_completed = case.evaluation_disposition != CaseEvaluationDisposition.NOT_RUN
        if (
            prompt is None
            or hashlib.sha256(prompt).hexdigest() != case.prompt_sha256
            or answer_messages_sha256 is None
            or not answer_attempts
            or (
                len(answer_attempts) == 1
                and answer_attempts[0].request_messages_sha256 != answer_messages_sha256
            )
            or (
                answer_completed
                and (
                    not isinstance(output, str)
                    or hashlib.sha256(output.encode("utf-8")).hexdigest()
                    != case.parsed_answer_sha256
                )
            )
            or (
                not answer_completed
                and (case.answer_raw_ref is not None or case.parsed_answer_sha256 is not None)
            )
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "prompt-answer-mismatch"))
        if case.evaluation_disposition == CaseEvaluationDisposition.JUDGED:
            if (
                case.judge_prompt_raw_ref is None
                or judge_messages_sha256 is None
                or not judge_attempts
                or (
                    len(judge_attempts) == 1
                    and judge_attempts[0].request_messages_sha256 != judge_messages_sha256
                )
            ):
                issues.append(_issue(rule_id, case.case_occurrence_id, "judge-prompt-mismatch"))
        elif (
            case.evaluation_disposition
            not in {
                CaseEvaluationDisposition.UNJUDGED,
                CaseEvaluationDisposition.NOT_RUN,
            }
            and case.judge_prompt_raw_ref is not None
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "judge-prompt-mismatch"))
    return tuple(issues)


def _retrieval_request_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.retrieval-request.v1"
    live_capsule = any(isinstance(item, AttemptRecordV4) for item in _attempt_records(snapshot))
    issues: list[ValidationIssue] = []
    for case in _contracts(snapshot, CaseRecordV3):
        reference = case.retrieval_request_raw_ref
        required = live_capsule and case.adapter_profile_id in GENERATION_FREE_PROOF_PROFILES
        if reference is None:
            if required:
                issues.append(
                    _issue(rule_id, case.case_occurrence_id, "retrieval-request-proof-missing")
                )
            continue
        payload = snapshot.raw_payloads.get(reference)
        if payload is None or not retrieval_request_proves_generation_free(
            case.adapter_profile_id,
            payload,
        ):
            issues.append(_issue(rule_id, reference, "retrieval-request-proof-invalid"))
    return tuple(issues)


def _visible_context_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.visible-context.v1"
    issues: list[ValidationIssue] = []
    for case in _contracts(snapshot, CaseRecordV3):
        visible = snapshot.raw_payloads.get(case.visible_evidence_raw_ref or "")
        ledger = _raw_object(snapshot, case.visible_decision_ledger_raw_ref)
        decisions = ledger.get("decisions") if ledger is not None else None
        if visible is None or ledger is None or not isinstance(decisions, list):
            issues.append(_issue(rule_id, case.case_occurrence_id, "visible-evidence-missing"))
            continue
        kept_ids = tuple(
            item.get("native_id")
            for item in decisions
            if isinstance(item, dict) and item.get("disposition") == "kept"
        )
        plans = {item.ingestion_occurrence_id: item for item in _ingestion_plans(snapshot)}
        candidates = _normalized_native_candidates(
            snapshot, case, plans.get(case.ingestion_occurrence_id)
        )
        if candidates is None:
            issues.append(_issue(rule_id, case.case_occurrence_id, "native-evidence-missing"))
            continue
        truncated_count = sum(
            item.native_truncated for item in candidates if item.native_id in kept_ids
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
            or count_o200k_tokens(visible) != case.visible_evidence_token_count
            or tokenizer_fingerprint() != case.visible_evidence_tokenizer_fingerprint
            or ledger.get("policy")
            != {"max_items": None, "max_characters": None, "max_tokens": None}
            or ledger.get("first_exceeded_limit") is not None
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
    attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
    plans = {item.ingestion_occurrence_id: item for item in _ingestion_plans(snapshot)}
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
                if not isinstance(plan, IngestionPlanRecordV2):
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
        if case.adapter_profile_id == OPENVIKING_SESSION_PROFILE_ID:
            plan = plans.get(case.ingestion_occurrence_id)
            try:
                if not isinstance(plan, IngestionPlanRecordV2):
                    raise ValueError("OpenViking session case has no plan evidence")
                session_evidence = _openviking_session_native_plan_evidence(
                    snapshot, plan, attempts
                )
                before_uris = reconstruct_openviking_session_projection(
                    raw_payloads=snapshot.raw_payloads,
                    references=case.pre_query_projection_raw_refs,
                    memory_root=session_evidence.memory_root,
                )
                after_uris = reconstruct_openviking_session_projection(
                    raw_payloads=snapshot.raw_payloads,
                    references=case.post_query_projection_raw_refs,
                    memory_root=session_evidence.memory_root,
                )
                session_before_state = canonical_sha256(
                    [
                        "oamb-openviking-session-projection-v1",
                        plan.ingestion_occurrence_id,
                        plan.projected_source_unit_ids,
                        before_uris,
                    ]
                )
                session_after_state = canonical_sha256(
                    [
                        "oamb-openviking-session-projection-v1",
                        plan.ingestion_occurrence_id,
                        plan.projected_source_unit_ids,
                        after_uris,
                    ]
                )
            except ValueError:
                issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
                continue
            if (
                session_before_state != case.pre_query_state_sha256
                or session_after_state != case.post_query_state_sha256
                or session_before_state != session_after_state
                or session_before_state != session_evidence.state_sha256
                or case.query_mutation_status != "unchanged"
            ):
                issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
            continue
        if case.adapter_profile_id == MEM0_PROFILE_ID:
            plan = plans.get(case.ingestion_occurrence_id)
            try:
                if not isinstance(plan, IngestionPlanRecordV3):
                    raise ValueError("Mem0 case has no v3 plan evidence")
                mem0_plan = reconstruct_mem0_plan(
                    raw_payloads=snapshot.raw_payloads,
                    plan=plan,
                    attempts=cast(dict[str, AttemptRecordV2], attempts),
                )
                mem0_before = reconstruct_mem0_projection(
                    raw_payloads=snapshot.raw_payloads,
                    references=case.pre_query_projection_raw_refs,
                    expected_run_id=plan.ingestion_occurrence_id,
                )
                mem0_after = reconstruct_mem0_projection(
                    raw_payloads=snapshot.raw_payloads,
                    references=case.post_query_projection_raw_refs,
                    expected_run_id=plan.ingestion_occurrence_id,
                )
            except ValueError:
                issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
                continue
            if (
                mem0_before.state_sha256 != case.pre_query_state_sha256
                or mem0_after.state_sha256 != case.post_query_state_sha256
                or mem0_before.state_sha256 != mem0_after.state_sha256
                or mem0_before.state_sha256 != plan.protected_state_sha256
                or mem0_before.capture_sequence <= mem0_plan.projection.capture_sequence
                or mem0_after.capture_sequence != mem0_before.capture_sequence + 1
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
        generic_before_state = (
            before_document.get("state_sha256") if before_document is not None else None
        )
        generic_after_state = (
            after_document.get("state_sha256") if after_document is not None else None
        )
        if (
            generic_before_state != case.pre_query_state_sha256
            or generic_after_state != case.post_query_state_sha256
            or generic_before_state != generic_after_state
            or case.query_mutation_status != "unchanged"
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "query-state-mismatch"))
    return tuple(issues)


def _metric_fraction_rule(snapshot: _NativeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "native.metric-fraction.v1"
    attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
    issues: list[ValidationIssue] = []
    for case in _contracts(snapshot, CaseRecordV3):
        if case.evaluation_disposition == CaseEvaluationDisposition.NOT_RUN:
            if (
                case.state != CaseState.ERROR
                or case.error_stage != "answer"
                or case.answer_raw_ref is not None
                or case.parsed_answer_sha256 is not None
                or case.evaluation_raw_ref is not None
                or case.metric_numerator is not None
                or case.metric_denominator is not None
            ):
                issues.append(_issue(rule_id, case.case_occurrence_id, "metric-fraction-mismatch"))
            continue
        if case.evaluation_disposition == CaseEvaluationDisposition.UNJUDGED:
            if (
                case.state != CaseState.ERROR
                or case.error_stage != "judge"
                or case.evaluation_raw_ref is not None
                or case.metric_numerator is not None
                or case.metric_denominator is not None
            ):
                issues.append(_issue(rule_id, case.case_occurrence_id, "metric-fraction-mismatch"))
            continue
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
            successful_judges = tuple(
                attempt for attempt in judge_attempts if attempt.outcome == AttemptOutcome.SUCCEEDED
            )
            if len(successful_judges) != 1 or successful_judges[0] is not judge_attempts[-1]:
                judged_trace_matches = False
            else:
                judge_attempt = successful_judges[0]
                judge_document = _raw_object(snapshot, judge_attempt.raw_response_ref)
                judge_output = _model_output_text(judge_document)
                judged_trace_matches = bool(
                    isinstance(judge_output, str)
                    and isinstance(trace_document, dict)
                    and trace_document.get("judge_raw_reference") == judge_attempt.raw_response_ref
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
    plans = {item.ingestion_occurrence_id: item for item in _ingestion_plans(snapshot)}
    candidates = _normalized_native_candidates(
        snapshot,
        case,
        plans.get(case.ingestion_occurrence_id),
    )
    if candidates is None:
        return None
    if len(decisions) != len(candidates):
        return None
    seen: dict[str, str] = {}
    kept: list[NativeEvidenceCandidate] = []
    expected_decisions: list[dict[str, Any]] = []
    for candidate in candidates:
        identity = candidate.provider_evidence_identity
        if not identity:
            return None
        content_hash = hashlib.sha256(candidate.content.encode("utf-8")).hexdigest()
        if identity in seen and seen[identity] != content_hash:
            return None
        duplicate = identity in seen
        if not duplicate:
            seen[identity] = content_hash
            kept.append(candidate)
        expected_decisions.append(
            {
                "native_id": candidate.native_id,
                "provider_evidence_identity": identity,
                "normalized_text_sha256": content_hash,
                "disposition": "duplicate" if duplicate else "kept",
                "reason": "duplicate_identity_and_text" if duplicate else None,
            }
        )
    try:
        payload, labels = render_compact_evidence(kept)
    except (ValueError, UnicodeError):
        return None
    labels_by_identity = dict(
        zip((item.provider_evidence_identity for item in kept), labels, strict=True)
    )
    for expected in expected_decisions:
        expected["label"] = labels_by_identity[expected["provider_evidence_identity"]]
    if decisions != expected_decisions:
        return None
    return payload


def _normalized_native_candidates(
    snapshot: _NativeCapsuleSnapshot,
    case: CaseRecordV3,
    plan: NativePlanRecord | None,
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
            hindsight_candidates = normalize_recall(
                payload,
                document_to_source_unit=projection.document_to_source_unit,
            )
            top_k = reconstruct_hindsight_candidate_limit(
                raw_payloads=snapshot.raw_payloads,
                references=case.retrieval_supporting_raw_refs,
                expected_request_sha256=case.retrieval_request_raw_ref,
                expected_response_sha256=case.retrieval_raw_ref,
            )
            return (
                hindsight_candidates
                if top_k is None
                else hindsight_candidates[:top_k]
            )
        except ValueError:
            return None
    if case.adapter_profile_id == OPENVIKING_PROFILE_ID:
        if not isinstance(plan, IngestionPlanRecordV2):
            return None
        attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
        try:
            openviking_evidence = _openviking_native_plan_evidence(snapshot, plan, attempts)
            return reconstruct_openviking_candidates(
                raw_payloads=snapshot.raw_payloads,
                case=case,
                plan=openviking_evidence,
            )
        except ValueError:
            return None
    if case.adapter_profile_id == OPENVIKING_SESSION_PROFILE_ID:
        if not isinstance(plan, IngestionPlanRecordV2):
            return None
        attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
        try:
            session_evidence = _openviking_session_native_plan_evidence(snapshot, plan, attempts)
            return reconstruct_openviking_session_candidates(
                raw_payloads=snapshot.raw_payloads,
                case=case,
                memory_root=session_evidence.memory_root,
            )
        except ValueError:
            return None
    if case.adapter_profile_id == MEM0_PROFILE_ID:
        if not isinstance(plan, IngestionPlanRecordV3):
            return None
        attempts = {item.attempt_id: item for item in _attempt_records(snapshot)}
        try:
            mem0_evidence = reconstruct_mem0_plan(
                raw_payloads=snapshot.raw_payloads,
                plan=plan,
                attempts=cast(dict[str, AttemptRecordV2], attempts),
            )
            return reconstruct_mem0_candidates(
                raw_payloads=snapshot.raw_payloads,
                case=case,
                plan=mem0_evidence,
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
    plan: NativePlanRecord,
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
        ordered_source_unit_ids=plan.projected_source_unit_ids,
        ordered_source_payload_sha256=tuple(
            dict(
                zip(
                    plan.ordered_source_unit_ids,
                    manifest_plan.ordered_source_unit_bytes_sha256,
                    strict=True,
                )
            )[source_id]
            for source_id in plan.projected_source_unit_ids
        ),
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


def _model_attempt_chain_closes(
    snapshot: _NativeCapsuleSnapshot,
    attempts: list[NativeAttemptRecord],
    *,
    prompt_ref: str | None,
    final_response_ref: str | None,
    require_success: bool = True,
) -> bool:
    if not attempts or len(attempts) > 18 or prompt_ref is None:
        return False
    if attempts[0].retry_of_attempt_id is not None:
        return False
    has_receipts = bool(
        set(attempt.attempt_id for attempt in attempts)
        & {receipt.attempt_id for receipt in _contracts(snapshot, AttemptReceiptRecord)}
    )
    prompt = snapshot.raw_payloads.get(prompt_ref)
    if prompt is None:
        return False
    try:
        prompt_text = prompt.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    if require_success and (
        attempts[-1].outcome != AttemptOutcome.SUCCEEDED
        or attempts[-1].raw_response_ref is None
        or (final_response_ref is not None and attempts[-1].raw_response_ref != final_response_ref)
    ):
        return False
    predecessor_attempts = attempts[:-1] if require_success else attempts
    if any(
        attempt.outcome not in {AttemptOutcome.FAILED, AttemptOutcome.UNKNOWN_OUTCOME}
        for attempt in predecessor_attempts
    ):
        return False
    if len(attempts) == 1:
        if attempts[0].request_messages_sha256 != canonical_sha256((("user", prompt_text),)):
            return False
        return not has_receipts or _model_attempt_receipt_layers_close(
            attempts,
            ((("user", prompt_text),),),
            snapshot,
            require_success=require_success,
        )

    request_messages: list[tuple[tuple[str, str], ...]] = []
    for attempt in attempts:
        reference = attempt.request_messages_sha256
        payload = snapshot.raw_payloads.get(reference or "")
        if reference is None or payload is None or hashlib.sha256(payload).hexdigest() != reference:
            return False
        try:
            document = json.loads(payload)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(document, list):
            return False
        messages: list[tuple[str, str]] = []
        for item in document:
            if (
                not isinstance(item, list)
                or len(item) != 2
                or item[0] not in {"user", "assistant"}
                or not isinstance(item[1], str)
            ):
                return False
            messages.append((item[0], item[1]))
        request_messages.append(tuple(messages))
    if request_messages[0] != (("user", prompt_text),):
        return False
    for index, (previous, current) in enumerate(zip(attempts, attempts[1:], strict=False)):
        previous_messages = request_messages[index]
        if current.retry_of_attempt_id != previous.attempt_id:
            return False
        current_messages = request_messages[index + 1]
        previous_output = _model_output_text(
            _raw_object(snapshot, previous.raw_error_ref or previous.raw_response_ref)
        )
        if previous_output is None:
            if current_messages != previous_messages:
                return False
            continue
        if (
            len(current_messages) != len(previous_messages) + 2
            or current_messages[:-2] != previous_messages
            or current_messages[-2] != ("assistant", previous_output)
            or current_messages[-1][0] != "user"
            or not current_messages[-1][1].startswith(
                "The previous response failed output validation: "
            )
            or not current_messages[-1][1].endswith(
                "Return a corrected response that satisfies the required output format."
            )
        ):
            return False
    return not has_receipts or _model_attempt_receipt_layers_close(
        attempts,
        tuple(request_messages),
        snapshot,
        require_success=require_success,
    )


def _model_attempt_receipt_layers_close(
    attempts: list[NativeAttemptRecord],
    request_messages: tuple[tuple[tuple[str, str], ...], ...],
    snapshot: _NativeCapsuleSnapshot,
    *,
    require_success: bool,
) -> bool:
    receipts = {item.attempt_id: item for item in _contracts(snapshot, AttemptReceiptRecord)}
    final_receipt = receipts.get(attempts[-1].attempt_id)
    if require_success and (
        final_receipt is None
        or final_receipt.receipt_kind != AttemptReceiptKind.RESPONSE
        or final_receipt.raw_response_ref != attempts[-1].raw_response_ref
    ):
        return False
    predecessor_count = len(attempts) - 1 if require_success else len(attempts)
    for index in range(predecessor_count):
        attempt = attempts[index]
        receipt = receipts.get(attempt.attempt_id)
        if receipt is None:
            return False
        same_messages = (
            index + 1 < len(attempts) and request_messages[index + 1] == request_messages[index]
        )
        terminal_failure = index + 1 == len(attempts)
        if attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME:
            if (
                receipt.receipt_kind != AttemptReceiptKind.UNKNOWN_OUTCOME
                or receipt.failure_kind not in {"timeout", "transport_error"}
                or (not same_messages and not terminal_failure)
            ):
                return False
            continue
        if (
            receipt.receipt_kind != AttemptReceiptKind.ERROR
            or receipt.raw_error_ref != attempt.raw_error_ref
            or receipt.raw_error_ref is None
        ):
            return False
        retryable_status = receipt.supplier_status_code in {408, 409, 429} or (
            isinstance(receipt.supplier_status_code, int)
            and 500 <= receipt.supplier_status_code <= 599
        )
        if same_messages:
            if not retryable_status:
                return False
        elif terminal_failure and (
            retryable_status
            or (
                receipt.failure_kind == "supplier_error"
                and isinstance(receipt.supplier_status_code, int)
                and 400 <= receipt.supplier_status_code <= 499
            )
        ):
            pass
        elif (
            receipt.failure_kind != "output_contract_error"
            or receipt.supplier_status_code not in {None, 200}
        ):
            return False

    outer_attempts = 0
    segment_start = 0
    while segment_start < len(attempts):
        segment_end = segment_start + 1
        while (
            segment_end < len(attempts)
            and request_messages[segment_end] == request_messages[segment_start]
        ):
            segment_end += 1
        segment_length = segment_end - segment_start
        outer_attempts += (segment_length + 2) // 3
        for boundary in range(segment_start + 2, segment_end - 1, 3):
            receipt = receipts.get(attempts[boundary].attempt_id)
            if receipt is None or receipt.supplier_status_code != 429:
                return False
        segment_start = segment_end
    return outer_attempts <= 6


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _all_raw_references(snapshot: _NativeCapsuleSnapshot) -> tuple[str, ...]:
    references: list[str] = []
    for plan in _ingestion_plans(snapshot):
        references.extend(plan.scope_raw_refs)
        references.extend(plan.readiness_evidence_refs)
        if plan.inventory_raw_ref is not None:
            references.append(plan.inventory_raw_ref)
        references.extend(plan.projection_raw_refs)
    for case in _contracts(snapshot, CaseRecordV3):
        for reference in (
            case.retrieval_raw_ref,
            case.retrieval_request_raw_ref,
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
    for attempt in _attempt_records(snapshot):
        if attempt.raw_response_ref is not None:
            references.append(attempt.raw_response_ref)
        if attempt.raw_error_ref is not None:
            references.append(attempt.raw_error_ref)
    for usage in _token_usage_records(snapshot):
        if usage.raw_response_ref is not None:
            references.append(usage.raw_response_ref)
    for event in _contracts(snapshot, InfrastructureRetryEvent):
        references.append(event.raw_error_ref)
    return tuple(dict.fromkeys(references))


def _token_usage_records(
    snapshot: _NativeCapsuleSnapshot,
) -> tuple[NativeTokenUsageRecord, ...]:
    return tuple(
        cast(NativeTokenUsageRecord, item)
        for item in snapshot.contracts
        if type(item)
        in {
            TokenUsageRecord,
            TokenUsageRecordV2,
            TokenUsageRecordV3,
            TokenUsageRecordV5,
        }
    )


def _attempt_records(snapshot: _NativeCapsuleSnapshot) -> tuple[NativeAttemptRecord, ...]:
    return (
        *_contracts(snapshot, AttemptRecordV2),
        *_contracts(snapshot, AttemptRecordV4),
    )


def _attempt_intent_records(
    snapshot: _NativeCapsuleSnapshot,
) -> tuple[NativeIntentRecord, ...]:
    return (
        *_contracts(snapshot, AttemptIntentRecord),
        *_contracts(snapshot, AttemptIntentRecordV3),
    )


def _budget_reservation_records(
    snapshot: _NativeCapsuleSnapshot,
) -> tuple[NativeReservationRecord, ...]:
    return (
        *_contracts(snapshot, BudgetReservationRecord),
        *_contracts(snapshot, BudgetReservationRecordV3),
    )


def _resource_usage_records(
    snapshot: _NativeCapsuleSnapshot,
) -> tuple[NativeResourceUsageRecord, ...]:
    return tuple(
        cast(NativeResourceUsageRecord, item)
        for item in snapshot.contracts
        if type(item) in {ResourceUsageRecord, ResourceUsageRecordV2}
    )


def _cost_records(snapshot: _NativeCapsuleSnapshot) -> tuple[NativeCostRecord, ...]:
    return tuple(
        cast(NativeCostRecord, item)
        for item in snapshot.contracts
        if type(item) in {CostRecord, CostRecordV2}
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


def _ingestion_plans(
    snapshot: _NativeCapsuleSnapshot,
) -> tuple[NativePlanRecord, ...]:
    return tuple(
        item
        for item in snapshot.contracts
        if isinstance(item, (IngestionPlanRecordV2, IngestionPlanRecordV3))
    )


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
