"""Closed T5 validation profile for generated fake source capsules."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from oamb.artifacts.atomic import ArtifactCollisionError, read_regular_file, sha256_file
from oamb.contracts.accounting import (
    ProofStatus,
    TokenDomain,
    TokenMeasurementSource,
    TokenStage,
    TokenUsageRecord,
    TokenUsageRecordV2,
    count_message_whitespace_tokens,
    count_whitespace_tokens,
)
from oamb.contracts.evidence import (
    AttemptRecord,
    CapsuleManifest,
    CaseEvaluationDisposition,
    CaseRecord,
    CaseRecordV2,
    CloseErrorRecord,
    IngestionPlanRecord,
    LogicalContextRecord,
    OriginRecord,
    RunRecord,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.ids import (
    attempt_id,
    canonical_json_bytes,
    canonical_sha256,
    case_occurrence_id,
    ingestion_occurrence_id,
)
from oamb.contracts.schema import parse_contract
from oamb.contracts.specifications import (
    BudgetScopeKind,
    BudgetSpec,
    CaseManifest,
    CaseManifestEntry,
    DatasetManifest,
    IngestionPlanManifest,
    RunSpec,
    ValidationProfile,
    ValidationStage,
)
from oamb.contracts.states import (
    AttemptOutcome,
    CaseState,
    IndexContribution,
    IngestionPlanState,
    ResumeDisposition,
    RunState,
    ValidationDisposition,
)

from .core import EvidenceNotValidatedError
from .profiles import fake_evidence_profile
from .registry import RuleRegistry, ValidationRule

FAKE_EVIDENCE_RULE_IDS = (
    "fake.schema.v1",
    "fake.manifest-closure.v1",
    "fake.parentage.v1",
    "fake.execution.v1",
    "fake.raw-reference.v1",
    "fake.index-accounting.v1",
)
_FAKE_ATTEMPT_PARENT_KINDS = {
    "memory_ingest": "ingestion_plan",
    "memory_query": "case",
    "answer": "case",
    "judge": "case",
}


@dataclass(frozen=True, slots=True)
class FakeCapsuleSnapshot:
    root: Path
    manifest_bytes: bytes
    manifest: CapsuleManifest | None
    contracts: tuple[BaseModel, ...]
    documents_by_path: dict[str, bytes]
    parse_failures: tuple[str, ...]
    symlink_paths: tuple[str, ...]


def validate_fake_capsule(
    capsule_root: Path,
    *,
    profile: ValidationProfile | None = None,
    registry: RuleRegistry | None = None,
) -> ValidationResult:
    return validate_fake_snapshot(
        load_fake_capsule(capsule_root),
        profile=profile,
        registry=registry,
    )


def validate_fake_snapshot(
    snapshot: FakeCapsuleSnapshot,
    *,
    profile: ValidationProfile | None = None,
    registry: RuleRegistry | None = None,
) -> ValidationResult:
    selected_profile = profile or fake_evidence_profile()
    selected_registry = registry or fake_evidence_registry()
    target_hash = (
        snapshot.manifest.source_manifest_hash
        if snapshot.manifest is not None
        else hashlib.sha256(snapshot.manifest_bytes).hexdigest()
    )
    required_ids = tuple(requirement.rule_id for requirement in selected_profile.required_rules)
    expected_inventory_hash = canonical_sha256(
        [
            "oamb-required-rule-inventory-v1",
            tuple(
                (requirement.rule_id, requirement.minimum_version)
                for requirement in selected_profile.required_rules
            ),
        ]
    )
    if selected_profile.stage != ValidationStage.EVIDENCE:
        return _profile_failure(
            selected_profile,
            target_hash,
            required_ids,
            "validation.stage.v1",
            "wrong-validation-stage",
        )
    if selected_profile.required_rule_inventory_hash != expected_inventory_hash:
        return _profile_failure(
            selected_profile,
            target_hash,
            required_ids,
            "validation.profile-inventory.v1",
            "profile-inventory-mismatch",
        )
    if required_ids != FAKE_EVIDENCE_RULE_IDS:
        return _profile_failure(
            selected_profile,
            target_hash,
            required_ids,
            "validation.profile-coverage.v1",
            "profile-rule-coverage-mismatch",
        )
    executed: list[str] = []
    passed: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    issues: list[ValidationIssue] = []
    implementation_versions: list[str] = []
    for requirement in selected_profile.required_rules:
        rule = selected_registry.compatible(
            requirement.rule_id,
            requirement.minimum_version,
        )
        if rule is None:
            missing.append(requirement.rule_id)
            continue
        executed.append(rule.rule_id)
        implementation_versions.append(f"{rule.rule_id}@{rule.version}")
        rule_issues = rule.evaluate(snapshot)
        if rule_issues:
            failed.append(rule.rule_id)
            issues.extend(rule_issues)
        else:
            passed.append(rule.rule_id)
    return ValidationResult(
        validation_profile_id=selected_profile.profile_id,
        target_hash=target_hash,
        disposition=(
            ValidationDisposition.VALIDATED
            if not failed and not missing
            else ValidationDisposition.INVALID
        ),
        required_rule_ids=required_ids,
        executed_rule_ids=tuple(executed),
        passed_rule_ids=tuple(passed),
        failed_rule_ids=tuple(failed),
        not_applicable_rule_ids=(),
        missing_rule_ids=tuple(missing),
        implementation_versions=tuple(implementation_versions),
        issues=tuple(issues),
    )


def fake_evidence_registry(*, exclude: set[str] | None = None) -> RuleRegistry:
    omitted = exclude or set()
    registry = RuleRegistry()
    for rule in FAKE_EVIDENCE_RULES:
        if rule.rule_id not in omitted:
            registry.register(rule)
    return registry


def load_fake_capsule(capsule_root: Path) -> FakeCapsuleSnapshot:
    root = Path(capsule_root)
    manifest_path = root / "capsule-manifest.json"
    symlink_paths = _find_symlink_paths(root)
    try:
        if _path_is_below_symlink("capsule-manifest.json", symlink_paths):
            raise ArtifactCollisionError("capsule manifest is below a symbolic link")
        manifest_bytes = read_regular_file(manifest_path)
    except (OSError, ArtifactCollisionError):
        manifest_bytes = b"{}"
    manifest: CapsuleManifest | None = None
    try:
        manifest = CapsuleManifest.model_validate_json(manifest_bytes)
    except Exception:
        pass
    documents_by_path: dict[str, bytes] = {}
    contracts: list[BaseModel] = []
    failures: list[str] = []
    if manifest is not None:
        for entry in manifest.source_entries:
            if not entry.relative_path.startswith("source/") or _path_is_below_symlink(
                entry.relative_path, symlink_paths
            ):
                failures.append(entry.relative_path)
                continue
            path = root / entry.relative_path
            try:
                content = read_regular_file(path)
            except (OSError, ArtifactCollisionError):
                failures.append(entry.relative_path)
                continue
            documents_by_path[entry.relative_path] = content
            if entry.relative_path.startswith("source/raw/"):
                if entry.record_kind != "raw_payload" or entry.record_id != path.name.removesuffix(
                    ".json.gz"
                ):
                    failures.append(entry.relative_path)
                continue
            try:
                document = json.loads(content)
                if not isinstance(document, dict):
                    raise ValueError("contract is not an object")
                contract = parse_contract(document)
                if (
                    document.get("schema_name") != entry.record_kind
                    or path.stem != entry.record_id
                    or _expected_source_record_id(document) != entry.record_id
                ):
                    raise ValueError("manifest identity does not match source contract")
                contracts.append(contract)
            except Exception:
                failures.append(entry.relative_path)
    return FakeCapsuleSnapshot(
        root=root,
        manifest_bytes=manifest_bytes,
        manifest=manifest,
        contracts=tuple(contracts),
        documents_by_path=documents_by_path,
        parse_failures=tuple(failures),
        symlink_paths=symlink_paths,
    )


def _find_symlink_paths(root: Path) -> tuple[str, ...]:
    if root.is_symlink():
        return (".",)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_symlink())
    )


def _path_is_below_symlink(relative_path: str, symlink_paths: tuple[str, ...]) -> bool:
    return "." in symlink_paths or any(
        relative_path == symlink_path or relative_path.startswith(f"{symlink_path}/")
        for symlink_path in symlink_paths
    )


def _expected_source_record_id(document: dict[str, object]) -> str | None:
    schema_name = document.get("schema_name")
    identity_fields = {
        "attempt_record": "attempt_id",
        "case_record": "case_occurrence_id",
        "close_error_record": "close_error_id",
        "ingestion_plan_record": "ingestion_occurrence_id",
        "logical_context_record": "context_manifest_entry_id",
        "origin_record": "origin_id",
        "run_record": "run_id",
        "token_usage_record": "usage_record_id",
    }
    fixed_ids = {
        "budget_spec": "budget",
        "case_manifest": "case-manifest",
        "dataset_manifest": "dataset-manifest",
        "run_spec": "run-spec",
    }
    if schema_name in identity_fields:
        value = document.get(identity_fields[schema_name])
        return value if isinstance(value, str) else None
    return fixed_ids.get(schema_name) if isinstance(schema_name, str) else None


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


def _profile_failure(
    profile: ValidationProfile,
    target_hash: str,
    required_ids: tuple[str, ...],
    rule_id: str,
    code: str,
) -> ValidationResult:
    return ValidationResult(
        validation_profile_id=profile.profile_id,
        target_hash=target_hash,
        disposition=ValidationDisposition.INVALID,
        required_rule_ids=required_ids,
        executed_rule_ids=(),
        passed_rule_ids=(),
        failed_rule_ids=(rule_id,),
        not_applicable_rule_ids=(),
        missing_rule_ids=required_ids,
        implementation_versions=(),
        issues=(_issue(rule_id, profile.profile_id, code),),
    )


def _schema_rule(snapshot: FakeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    if snapshot.manifest is None:
        return (_issue("fake.schema.v1", "capsule-manifest.json", "schema-invalid"),)
    issues = [_issue("fake.schema.v1", path, "schema-invalid") for path in snapshot.parse_failures]
    allowed_types = {
        AttemptRecord,
        BudgetSpec,
        CaseManifest,
        CaseRecordV2,
        CloseErrorRecord,
        DatasetManifest,
        IngestionPlanRecord,
        LogicalContextRecord,
        OriginRecord,
        RunRecord,
        RunSpec,
        TokenUsageRecord,
    }
    if any(type(item) not in allowed_types for item in snapshot.contracts):
        issues.append(_issue("fake.schema.v1", snapshot.manifest.run_id, "contract-not-allowed"))
    return tuple(issues)


def _manifest_rule(snapshot: FakeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.manifest-closure.v1"
    manifest = snapshot.manifest
    if manifest is None:
        return (_issue(rule_id, "capsule-manifest.json", "manifest-invalid"),)
    issues: list[ValidationIssue] = [
        _issue(rule_id, path, "source-symlink") for path in snapshot.symlink_paths
    ]
    expected_paths = {entry.relative_path for entry in manifest.source_entries}
    actual_paths = {
        path.relative_to(snapshot.root).as_posix()
        for path in (snapshot.root / "source").rglob("*")
        if not path.is_symlink() and path.is_file()
    }
    if expected_paths != actual_paths:
        issues.append(_issue(rule_id, manifest.capsule_id, "source-inventory-mismatch"))
    for entry in manifest.source_entries:
        path = snapshot.root / entry.relative_path
        try:
            matches = (
                not _path_is_below_symlink(entry.relative_path, snapshot.symlink_paths)
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
        [
            "oamb-capsule-v1",
            manifest.run_id,
            manifest.run_spec_hash,
            expected_source_hash,
        ]
    )
    if (
        manifest.source_manifest_hash != expected_source_hash
        or manifest.capsule_id != expected_capsule_id
    ):
        issues.append(_issue(rule_id, manifest.capsule_id, "manifest-identity-mismatch"))
    return tuple(issues)


def _parentage_rule(snapshot: FakeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.parentage.v1"
    manifest = snapshot.manifest
    if manifest is None:
        return (_issue(rule_id, "capsule-manifest.json", "parentage-unavailable"),)
    runs = tuple(item for item in snapshot.contracts if isinstance(item, RunRecord))
    plans = tuple(item for item in snapshot.contracts if isinstance(item, IngestionPlanRecord))
    cases = tuple(
        item for item in snapshot.contracts if isinstance(item, (CaseRecord, CaseRecordV2))
    )
    contexts = tuple(item for item in snapshot.contracts if isinstance(item, LogicalContextRecord))
    origins = tuple(item for item in snapshot.contracts if isinstance(item, OriginRecord))
    close_errors = tuple(item for item in snapshot.contracts if isinstance(item, CloseErrorRecord))
    attempts = tuple(item for item in snapshot.contracts if isinstance(item, AttemptRecord))
    usage = tuple(
        item
        for item in snapshot.contracts
        if isinstance(item, (TokenUsageRecord, TokenUsageRecordV2))
    )
    run_specs = tuple(item for item in snapshot.contracts if isinstance(item, RunSpec))
    datasets = tuple(item for item in snapshot.contracts if isinstance(item, DatasetManifest))
    case_manifests = tuple(item for item in snapshot.contracts if isinstance(item, CaseManifest))
    budgets = tuple(item for item in snapshot.contracts if isinstance(item, BudgetSpec))
    if len(runs) != 1:
        return (_issue(rule_id, manifest.run_id, "run-inventory-mismatch"),)
    run = runs[0]
    issues: list[ValidationIssue] = []
    if not (len(run_specs) == len(datasets) == len(case_manifests) == len(budgets) == 1):
        issues.append(_issue(rule_id, run.run_id, "control-plane-inventory-mismatch"))
        return tuple(issues)
    run_spec = run_specs[0]
    dataset = datasets[0]
    case_manifest = case_manifests[0]
    budget = budgets[0]
    if run.run_id != manifest.run_id or run.run_spec_hash != manifest.run_spec_hash:
        issues.append(_issue(rule_id, run.run_id, "run-manifest-mismatch"))
    expected_case_manifest_hash = canonical_sha256(
        [
            "oamb-case-manifest-v1",
            {
                "manifest_id": case_manifest.manifest_id,
                "workload_id": case_manifest.workload_id,
                "logical_contexts": case_manifest.logical_contexts,
                "ingestion_plans": case_manifest.ingestion_plans,
                "cases": case_manifest.cases,
            },
        ]
    )
    if (
        run_spec.run_id != run.run_id
        or canonical_sha256(run_spec) != run.run_spec_hash
        or run_spec.dataset_manifest_hash != dataset.manifest_hash
        or run_spec.case_manifest_hash != case_manifest.manifest_hash
        or run_spec.workload_id != case_manifest.workload_id
        or case_manifest.manifest_hash != expected_case_manifest_hash
        or budget.budget_id != run_spec.budget_id
        or budget.scope_id != run.run_id
        or budget.scope_kind != BudgetScopeKind.RUN
        or budget.approval_id is not None
        or budget.max_attempts != 0
        or budget.max_input_tokens != 0
        or budget.max_output_tokens != 0
        or budget.max_wall_seconds != 0
        or budget.max_cost is not None
        or budget.currency is not None
        or run_spec.protocol_id != "oamb-fake-v1"
        or run_spec.memory_system_id != "fake-memory"
        or run_spec.model_role_binding_ids != ("fake-answer-v1", "fake-judge-v1")
        or run_spec.code_revision != "generated-fake-v1"
    ):
        issues.append(_issue(rule_id, run.run_id, "control-plane-binding-mismatch"))
    plan_occurrence_ids = tuple(plan.ingestion_occurrence_id for plan in plans)
    if (
        len(run.ingestion_occurrence_ids) != len(plan_occurrence_ids)
        or len(set(run.ingestion_occurrence_ids)) != len(run.ingestion_occurrence_ids)
        or len(set(plan_occurrence_ids)) != len(plan_occurrence_ids)
        or set(run.ingestion_occurrence_ids) != set(plan_occurrence_ids)
    ):
        issues.append(_issue(rule_id, run.run_id, "plan-parentage-mismatch"))
    case_occurrence_ids = tuple(case.case_occurrence_id for case in cases)
    if (
        len(run.case_occurrence_ids) != len(case_occurrence_ids)
        or len(set(run.case_occurrence_ids)) != len(run.case_occurrence_ids)
        or len(set(case_occurrence_ids)) != len(case_occurrence_ids)
        or set(run.case_occurrence_ids) != set(case_occurrence_ids)
    ):
        issues.append(_issue(rule_id, run.run_id, "case-parentage-mismatch"))
    plan_ids = {plan.ingestion_occurrence_id for plan in plans}
    if any(case.ingestion_occurrence_id not in plan_ids for case in cases):
        issues.append(_issue(rule_id, run.run_id, "case-plan-mismatch"))
    if any(plan.run_id != run.run_id for plan in plans) or any(
        case.run_id != run.run_id for case in cases
    ):
        issues.append(_issue(rule_id, run.run_id, "cross-run-reference"))
    context_by_id = {item.context_manifest_entry_id: item for item in contexts}
    manifest_context_by_id = {
        item.context_manifest_entry_id: item for item in case_manifest.logical_contexts
    }
    if (
        len(context_by_id) != len(contexts)
        or len(manifest_context_by_id) != len(case_manifest.logical_contexts)
        or set(context_by_id) != set(manifest_context_by_id)
    ):
        issues.append(_issue(rule_id, run.run_id, "logical-context-inventory-mismatch"))
    elif any(
        context.context_content_id != manifest_context_by_id[context_id].context_content_id
        or context.source_file_sha256 != manifest_context_by_id[context_id].source_file_sha256
        or context.source_row_number_1_indexed
        != manifest_context_by_id[context_id].source_row_number_1_indexed
        or context.context_bytes_sha256 != manifest_context_by_id[context_id].context_bytes_sha256
        or context.ordered_case_manifest_entry_ids
        != tuple(
            case.case_manifest_entry_id
            for case in case_manifest.cases
            if case.context_manifest_entry_id == context_id
        )
        for context_id, context in context_by_id.items()
    ):
        issues.append(_issue(rule_id, run.run_id, "logical-context-binding-mismatch"))
    origin_by_id = {item.origin_id: item for item in origins}
    referenced_origin_ids = {context.origin_id for context in contexts}
    dataset_source_hashes = {item.sha256 for item in dataset.source_files}
    if (
        len(origin_by_id) != len(origins)
        or set(origin_by_id) != referenced_origin_ids
        or any(
            context.origin_id not in origin_by_id
            or origin_by_id[context.origin_id].source_sha256 not in dataset_source_hashes
            for context in contexts
        )
    ):
        issues.append(_issue(rule_id, run.run_id, "origin-binding-mismatch"))
    plan_manifest_by_id = {item.ingestion_plan_id: item for item in case_manifest.ingestion_plans}
    plan_record_by_plan_id = {item.ingestion_plan_id: item for item in plans}
    if (
        len(plan_manifest_by_id) != len(case_manifest.ingestion_plans)
        or len(plan_record_by_plan_id) != len(plans)
        or set(plan_manifest_by_id) != set(plan_record_by_plan_id)
    ):
        issues.append(_issue(rule_id, run.run_id, "plan-manifest-inventory-mismatch"))
    else:
        for plan_id, plan_record in plan_record_by_plan_id.items():
            plan_manifest = plan_manifest_by_id[plan_id]
            expected_occurrence_id = ingestion_occurrence_id(
                run.run_id,
                plan_record.memory_system_id,
                plan_id,
            )
            expected_case_ids = tuple(
                case_occurrence_id(expected_occurrence_id, case_id)
                for case_id in plan_manifest.ordered_case_manifest_entry_ids
            )
            if (
                plan_record.memory_system_id != run_spec.memory_system_id
                or plan_record.ingestion_occurrence_id != expected_occurrence_id
                or plan_record.ordered_member_context_manifest_entry_ids
                != plan_manifest.ordered_member_context_manifest_entry_ids
                or plan_record.ordered_case_occurrence_ids != expected_case_ids
            ):
                issues.append(_issue(rule_id, plan_id, "plan-manifest-binding-mismatch"))
    manifest_case_ids = tuple(item.case_manifest_entry_id for item in case_manifest.cases)
    record_case_ids = tuple(item.case_manifest_entry_id for item in cases)
    if (
        len(set(manifest_case_ids)) != len(manifest_case_ids)
        or len(set(record_case_ids)) != len(record_case_ids)
        or set(record_case_ids) != set(manifest_case_ids)
    ):
        issues.append(_issue(rule_id, run.run_id, "case-manifest-inventory-mismatch"))
    if any(
        case.case_occurrence_id
        != case_occurrence_id(case.ingestion_occurrence_id, case.case_manifest_entry_id)
        for case in cases
    ):
        issues.append(_issue(rule_id, run.run_id, "case-manifest-binding-mismatch"))
    parents = {
        "ingestion_plan": {plan.ingestion_occurrence_id for plan in plans},
        "case": {case.case_occurrence_id for case in cases},
    }
    attempt_by_id = {attempt.attempt_id: attempt for attempt in attempts}
    if len(attempt_by_id) != len(attempts):
        issues.append(_issue(rule_id, run.run_id, "attempt-identity-duplicated"))
    if any(
        item.attempt_id
        != attempt_id(
            item.parent_id,
            item.stage,
            item.ordinal,
            item.request_fingerprint,
        )
        for item in attempts
    ):
        issues.append(_issue(rule_id, run.run_id, "attempt-identity-mismatch"))
    referenced_attempt_ids = {
        attempt_id_value for parent in (*plans, *cases) for attempt_id_value in parent.attempt_ids
    }
    if set(attempt_by_id) != referenced_attempt_ids:
        issues.append(_issue(rule_id, run.run_id, "attempt-inventory-mismatch"))
    for plan in plans:
        if len(plan.attempt_ids) != len(set(plan.attempt_ids)) or any(
            attempt_id_value not in attempt_by_id
            or attempt_by_id[attempt_id_value].parent_kind != "ingestion_plan"
            or attempt_by_id[attempt_id_value].parent_id != plan.ingestion_occurrence_id
            for attempt_id_value in plan.attempt_ids
        ):
            issues.append(_issue(rule_id, plan.ingestion_occurrence_id, "plan-attempt-mismatch"))
    for case in cases:
        if len(case.attempt_ids) != len(set(case.attempt_ids)) or any(
            attempt_id_value not in attempt_by_id
            or attempt_by_id[attempt_id_value].parent_kind != "case"
            or attempt_by_id[attempt_id_value].parent_id != case.case_occurrence_id
            for attempt_id_value in case.attempt_ids
        ):
            issues.append(_issue(rule_id, case.case_occurrence_id, "case-attempt-mismatch"))
    usage_by_id = {item.usage_record_id: item for item in usage}
    if len(usage_by_id) != len(usage):
        issues.append(_issue(rule_id, run.run_id, "usage-identity-duplicated"))
    usage_by_attempt: dict[str, list[TokenUsageRecord | TokenUsageRecordV2]] = {}
    for item in usage:
        usage_by_attempt.setdefault(item.attempt_id, []).append(item)
    if any(
        plan.resource_record_ids
        or plan.cost_record_ids
        or plan.usage_record_ids
        != tuple(
            item.usage_record_id
            for attempt_id_value in plan.attempt_ids
            for item in usage_by_attempt.get(attempt_id_value, ())
            if item.parent_kind == "ingestion_plan"
            and item.parent_id == plan.ingestion_occurrence_id
        )
        or any(
            usage_id not in usage_by_id
            or usage_by_id[usage_id].parent_kind != "ingestion_plan"
            or usage_by_id[usage_id].parent_id != plan.ingestion_occurrence_id
            for usage_id in plan.usage_record_ids
        )
        for plan in plans
    ):
        issues.append(_issue(rule_id, run.run_id, "plan-usage-mismatch"))
    if any(
        item.parent_kind not in parents
        or item.parent_id not in parents[item.parent_kind]
        or item.attempt_id not in attempt_by_id
        or attempt_by_id[item.attempt_id].parent_kind != item.parent_kind
        or attempt_by_id[item.attempt_id].parent_id != item.parent_id
        for item in usage
    ):
        issues.append(_issue(rule_id, run.run_id, "usage-attempt-parentage-mismatch"))
    close_error_by_id = {item.close_error_id: item for item in close_errors}
    if (
        len(close_error_by_id) != len(close_errors)
        or len({item.client_profile_id for item in close_errors}) != len(close_errors)
        or any(
            item.owner_kind != "run"
            or item.owner_id != run.run_id
            or item.client_profile_id not in {"fake-model", "fake-memory"}
            or item.close_error_id
            != canonical_sha256(
                [
                    "oamb-fake-close-error-v1",
                    run.run_id,
                    item.client_profile_id,
                    item.error_ref,
                ]
            )
            for item in close_errors
        )
    ):
        issues.append(_issue(rule_id, run.run_id, "close-error-parentage-mismatch"))
    return tuple(issues)


def _execution_rule(snapshot: FakeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.execution.v1"
    runs = tuple(item for item in snapshot.contracts if isinstance(item, RunRecord))
    plans = tuple(item for item in snapshot.contracts if isinstance(item, IngestionPlanRecord))
    cases = tuple(
        item for item in snapshot.contracts if isinstance(item, (CaseRecord, CaseRecordV2))
    )
    attempts = tuple(item for item in snapshot.contracts if isinstance(item, AttemptRecord))
    if len(runs) != 1:
        return (_issue(rule_id, "run", "terminal-run-missing"),)
    terminal_runs = {RunState.FINALIZED, RunState.ABORTED, RunState.INTERRUPTED}
    issues: list[ValidationIssue] = []
    if runs[0].state not in terminal_runs:
        issues.append(_issue(rule_id, runs[0].run_id, "run-not-terminal"))
    if len(plans) != 2 or len(cases) != 4:
        issues.append(_issue(rule_id, runs[0].run_id, "fake-counts-mismatch"))
    terminal_plan_states = {
        IngestionPlanState.SEALED,
        IngestionPlanState.ERROR,
        IngestionPlanState.CANCELLED,
        IngestionPlanState.INTERRUPTED_UNKNOWN_OUTCOME,
    }
    terminal_case_states = {CaseState.COMPLETED, CaseState.ERROR, CaseState.CANCELLED}
    if any(plan.state not in terminal_plan_states for plan in plans):
        issues.append(_issue(rule_id, runs[0].run_id, "plan-not-terminal"))
    if any(case.state not in terminal_case_states for case in cases):
        issues.append(_issue(rule_id, runs[0].run_id, "case-not-terminal"))
    if any(
        _FAKE_ATTEMPT_PARENT_KINDS.get(attempt.stage) != attempt.parent_kind for attempt in attempts
    ):
        issues.append(_issue(rule_id, runs[0].run_id, "attempt-stage-parent-mismatch"))
    if any(not _attempt_raw_outcome_matches(attempt) for attempt in attempts):
        issues.append(_issue(rule_id, runs[0].run_id, "attempt-outcome-raw-mismatch"))
    attempt_ids = {attempt.attempt_id for attempt in attempts}
    referenced_attempt_ids = tuple(
        attempt_id_value for plan in plans for attempt_id_value in plan.attempt_ids
    ) + tuple(attempt_id_value for case in cases for attempt_id_value in case.attempt_ids)
    if any(attempt_id_value not in attempt_ids for attempt_id_value in referenced_attempt_ids):
        issues.append(_issue(rule_id, runs[0].run_id, "attempt-reference-missing"))
    attempt_by_id = {attempt.attempt_id: attempt for attempt in attempts}
    if any(not _plan_attempt_chain_matches(plan, attempt_by_id) for plan in plans):
        issues.append(_issue(rule_id, runs[0].run_id, "plan-attempt-chain-mismatch"))
    if any(
        not isinstance(case, CaseRecordV2) or not _case_attempt_chain_matches(case, attempt_by_id)
        for case in cases
    ):
        issues.append(_issue(rule_id, runs[0].run_id, "case-attempt-chain-mismatch"))
    retry_edges_are_valid = True
    seen_attempt_slots: set[tuple[str, str, str, int]] = set()
    for attempt in attempts:
        slot = (
            attempt.parent_kind,
            attempt.parent_id,
            attempt.stage,
            attempt.ordinal,
        )
        if slot in seen_attempt_slots:
            retry_edges_are_valid = False
        seen_attempt_slots.add(slot)
        if attempt.ordinal == 1:
            if attempt.retry_of_attempt_id is not None:
                retry_edges_are_valid = False
            continue
        predecessor = attempt_by_id.get(attempt.retry_of_attempt_id or "")
        if (
            predecessor is None
            or predecessor.parent_kind != attempt.parent_kind
            or predecessor.parent_id != attempt.parent_id
            or predecessor.stage != attempt.stage
            or predecessor.request_fingerprint != attempt.request_fingerprint
            or predecessor.ordinal + 1 != attempt.ordinal
            or predecessor.outcome != AttemptOutcome.FAILED
        ):
            retry_edges_are_valid = False
    if not retry_edges_are_valid:
        issues.append(_issue(rule_id, runs[0].run_id, "retry-edge-mismatch"))
    if runs[0].state == RunState.FINALIZED:
        answer_attempts = tuple(item for item in attempts if item.stage == "answer")
        failed_answers = tuple(
            item for item in answer_attempts if item.outcome == AttemptOutcome.FAILED
        )
        retries = tuple(item for item in answer_attempts if item.retry_of_attempt_id is not None)
        if (
            len(failed_answers) != 1
            or len(retries) != 1
            or retries[0].outcome != AttemptOutcome.SUCCEEDED
        ):
            issues.append(_issue(rule_id, runs[0].run_id, "bounded-retry-mismatch"))
        if sum(case.error_stage == "judge" for case in cases) != 1:
            issues.append(_issue(rule_id, runs[0].run_id, "unjudged-count-mismatch"))
    unknown_attempts = tuple(
        attempt for attempt in attempts if attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME
    )
    interrupted_plans = tuple(
        plan for plan in plans if plan.state == IngestionPlanState.INTERRUPTED_UNKNOWN_OUTCOME
    )
    if unknown_attempts:
        unknown = unknown_attempts[0]
        unknown_propagates = (
            len(unknown_attempts) == 1
            and len(attempts) == 1
            and len(interrupted_plans) == 1
            and unknown.parent_kind == "ingestion_plan"
            and unknown.stage == "memory_ingest"
            and unknown.raw_response_ref is None
            and unknown.raw_error_ref is None
            and interrupted_plans[0].ingestion_occurrence_id == unknown.parent_id
            and interrupted_plans[0].attempt_ids == (unknown.attempt_id,)
            and runs[0].state == RunState.INTERRUPTED
            and runs[0].resume_disposition == ResumeDisposition.REPLACEMENT_RUN_REQUIRED
            and all(
                case.state == CaseState.ERROR
                and case.error_stage == "unknown_outcome"
                and not case.attempt_ids
                for case in cases
            )
        )
        if not unknown_propagates:
            issues.append(_issue(rule_id, runs[0].run_id, "unknown-outcome-propagation-mismatch"))
    elif (
        interrupted_plans
        or runs[0].state == RunState.INTERRUPTED
        or runs[0].resume_disposition == ResumeDisposition.REPLACEMENT_RUN_REQUIRED
    ):
        issues.append(_issue(rule_id, runs[0].run_id, "unknown-outcome-propagation-mismatch"))
    return tuple(issues)


def _attempt_raw_outcome_matches(attempt: AttemptRecord) -> bool:
    if attempt.outcome == AttemptOutcome.SUCCEEDED:
        return attempt.raw_response_ref is not None and attempt.raw_error_ref is None
    if attempt.outcome == AttemptOutcome.FAILED:
        return attempt.raw_response_ref is None and attempt.raw_error_ref is not None
    if attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME:
        return attempt.raw_response_ref is None and attempt.raw_error_ref is None
    return False


def _plan_attempt_chain_matches(
    plan: IngestionPlanRecord,
    attempt_by_id: dict[str, AttemptRecord],
) -> bool:
    chain = tuple(
        attempt_by_id[attempt_id_value]
        for attempt_id_value in plan.attempt_ids
        if attempt_id_value in attempt_by_id
    )
    if len(chain) != len(plan.attempt_ids):
        return False
    if not chain:
        return (
            plan.state in {IngestionPlanState.ERROR, IngestionPlanState.CANCELLED}
            and plan.accepted_source_count == 0
            and plan.failed_source_count == 0
            and not plan.readiness_evidence_refs
            and not plan.usage_record_ids
        )
    if len(chain) != 1 or chain[0].stage != "memory_ingest":
        return False
    attempt = chain[0]
    if attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME:
        return (
            plan.state == IngestionPlanState.INTERRUPTED_UNKNOWN_OUTCOME
            and plan.accepted_source_count == 0
            and plan.failed_source_count == 0
            and not plan.readiness_evidence_refs
            and not plan.usage_record_ids
        )
    if attempt.outcome == AttemptOutcome.SUCCEEDED:
        return (
            plan.state in {IngestionPlanState.SEALED, IngestionPlanState.ERROR}
            and plan.accepted_source_count + plan.failed_source_count == plan.intended_source_count
            and len(plan.readiness_evidence_refs) == 2
            and len(plan.usage_record_ids) == 1
        )
    return False


def _case_attempt_chain_matches(
    case: CaseRecordV2,
    attempt_by_id: dict[str, AttemptRecord],
) -> bool:
    chain = tuple(
        attempt_by_id[attempt_id_value]
        for attempt_id_value in case.attempt_ids
        if attempt_id_value in attempt_by_id
    )
    if len(chain) != len(case.attempt_ids):
        return False
    if case.evaluation_disposition == CaseEvaluationDisposition.NOT_RUN:
        return (
            not chain
            and case.retrieval_raw_ref is None
            and case.prompt_sha256 is None
            and case.answer_raw_ref is None
            and case.parsed_answer_sha256 is None
            and case.evaluation_raw_ref is None
        )
    shape = tuple((attempt.stage, attempt.outcome) for attempt in chain)
    if case.evaluation_disposition == CaseEvaluationDisposition.DETERMINISTIC_EVALUATED:
        return shape in {
            (
                ("memory_query", AttemptOutcome.SUCCEEDED),
                ("answer", AttemptOutcome.SUCCEEDED),
            ),
            (
                ("memory_query", AttemptOutcome.SUCCEEDED),
                ("answer", AttemptOutcome.FAILED),
                ("answer", AttemptOutcome.SUCCEEDED),
            ),
        }
    if case.evaluation_disposition == CaseEvaluationDisposition.UNJUDGED:
        return shape == (
            ("memory_query", AttemptOutcome.SUCCEEDED),
            ("answer", AttemptOutcome.SUCCEEDED),
            ("judge", AttemptOutcome.FAILED),
        )
    return False


def _raw_reference_rule(snapshot: FakeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.raw-reference.v1"
    raw_ids = (
        {
            entry.record_id
            for entry in snapshot.manifest.source_entries
            if entry.relative_path.startswith("source/raw/")
        }
        if snapshot.manifest is not None
        else set()
    )
    referenced: set[str] = set()
    for item in snapshot.contracts:
        document = item.model_dump(mode="python")
        _collect_raw_references(document, referenced)
    missing = sorted(referenced - raw_ids)
    issues = [_issue(rule_id, reference, "raw-reference-missing") for reference in missing]
    raw_documents, raw_issues = _decode_raw_documents(snapshot, rule_id)
    issues.extend(raw_issues)

    lifecycle_ids: set[str] = set()
    run_specs = tuple(item for item in snapshot.contracts if isinstance(item, RunSpec))
    resolve_items = tuple(
        (raw_id, document)
        for raw_id, document in raw_documents.items()
        if document.get("operation") == "resolve"
    )
    if len(run_specs) != 1 or len(resolve_items) != 1:
        issues.append(_issue(rule_id, "fake-memory", "runtime-resolution-raw-mismatch"))
    else:
        resolve_id, resolve_document = resolve_items[0]
        run_spec = run_specs[0]
        if (
            resolve_document.get("memory_system_id") != run_spec.memory_system_id
            or resolve_document.get("runtime_binding_hash") != run_spec.runtime_binding_hash
        ):
            issues.append(_issue(rule_id, resolve_id, "runtime-resolution-raw-mismatch"))
        else:
            lifecycle_ids.add(resolve_id)

    plans = tuple(item for item in snapshot.contracts if isinstance(item, IngestionPlanRecord))
    attempts = {
        item.attempt_id: item for item in snapshot.contracts if isinstance(item, AttemptRecord)
    }
    expected_scopes = {
        (plan.ingestion_occurrence_id, plan.ingestion_plan_id)
        for plan in plans
        if any(
            attempt_id_value in attempts
            and attempts[attempt_id_value].stage == "memory_ingest"
            and attempts[attempt_id_value].outcome != AttemptOutcome.UNKNOWN_OUTCOME
            for attempt_id_value in plan.attempt_ids
        )
    }
    scope_items: dict[tuple[str, str], list[str]] = {}
    for raw_id, document in raw_documents.items():
        if document.get("operation") != "allocate_scope":
            continue
        occurrence_id = document.get("ingestion_occurrence_id")
        plan_id = document.get("ingestion_plan_id")
        scope_id = document.get("scope_id")
        if not isinstance(occurrence_id, str) or not isinstance(plan_id, str):
            issues.append(_issue(rule_id, raw_id, "scope-allocation-raw-mismatch"))
            continue
        key = (occurrence_id, plan_id)
        if scope_id != canonical_sha256(["oamb-fake-scope-v1", occurrence_id, plan_id]):
            issues.append(_issue(rule_id, raw_id, "scope-allocation-raw-mismatch"))
            continue
        scope_items.setdefault(key, []).append(raw_id)
    if set(scope_items) != expected_scopes or any(
        len(raw_reference_ids) != 1 for raw_reference_ids in scope_items.values()
    ):
        issues.append(_issue(rule_id, "ingestion-plans", "scope-allocation-raw-mismatch"))
    else:
        lifecycle_ids.update(raw_reference_ids[0] for raw_reference_ids in scope_items.values())

    for raw_id in sorted((raw_ids - referenced) - lifecycle_ids):
        issues.append(_issue(rule_id, raw_id, "raw-reference-orphan"))
    issues.extend(_fake_raw_mapping_issues(snapshot, raw_documents))
    return tuple(issues)


def _decode_raw_documents(
    snapshot: FakeCapsuleSnapshot,
    rule_id: str,
) -> tuple[dict[str, dict[str, object]], list[ValidationIssue]]:
    raw_documents: dict[str, dict[str, object]] = {}
    issues: list[ValidationIssue] = []
    if snapshot.manifest is not None:
        for entry in snapshot.manifest.source_entries:
            if not entry.relative_path.startswith("source/raw/"):
                continue
            content = snapshot.documents_by_path.get(entry.relative_path)
            try:
                payload = gzip.decompress(content) if content is not None else None
            except (gzip.BadGzipFile, EOFError, OSError):
                payload = None
            if payload is None or hashlib.sha256(payload).hexdigest() != entry.record_id:
                issues.append(_issue(rule_id, entry.record_id, "raw-payload-identity-mismatch"))
                continue
            try:
                raw_document: object = json.loads(payload)
            except (json.JSONDecodeError, UnicodeDecodeError):
                raw_document = None
            if not isinstance(raw_document, dict):
                issues.append(_issue(rule_id, entry.record_id, "raw-payload-shape-invalid"))
                continue
            raw_documents[entry.record_id] = raw_document
    return raw_documents, issues


_FAKE_RETRIEVAL_CANDIDATES = [
    {
        "native_id": "fake-native-1",
        "native_rank_1_indexed": 1,
        "content": "shared answer alpha",
        "native_score": "0.900",
    },
    {
        "native_id": "fake-native-2",
        "native_rank_1_indexed": 2,
        "content": "shared answer beta",
        "native_score": "0.800",
    },
]


def _fake_raw_mapping_issues(
    snapshot: FakeCapsuleSnapshot,
    raw_documents: dict[str, dict[str, object]],
) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.raw-reference.v1"
    issues: list[ValidationIssue] = []
    plans = tuple(item for item in snapshot.contracts if isinstance(item, IngestionPlanRecord))
    cases = tuple(item for item in snapshot.contracts if isinstance(item, CaseRecordV2))
    attempts = tuple(item for item in snapshot.contracts if isinstance(item, AttemptRecord))
    usage = tuple(
        item
        for item in snapshot.contracts
        if isinstance(item, (TokenUsageRecord, TokenUsageRecordV2))
    )
    manifests = tuple(item for item in snapshot.contracts if isinstance(item, CaseManifest))
    if len(manifests) != 1:
        return (_issue(rule_id, "case-manifest", "raw-normalized-mapping-mismatch"),)
    case_manifest = manifests[0]
    plan_manifest_by_id = {plan.ingestion_plan_id: plan for plan in case_manifest.ingestion_plans}
    plan_by_occurrence = {plan.ingestion_occurrence_id: plan for plan in plans}
    case_by_occurrence = {case.case_occurrence_id: case for case in cases}
    manifest_case_by_id = {case.case_manifest_entry_id: case for case in case_manifest.cases}
    usage_by_attempt: dict[str, list[TokenUsageRecord | TokenUsageRecordV2]] = {}
    for item in usage:
        usage_by_attempt.setdefault(item.attempt_id, []).append(item)

    for attempt in attempts:
        if attempt.stage == "memory_ingest":
            if attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME:
                continue
            plan = plan_by_occurrence.get(attempt.parent_id)
            document = raw_documents.get(attempt.raw_response_ref or "")
            plan_manifest = (
                plan_manifest_by_id.get(plan.ingestion_plan_id) if plan is not None else None
            )
            if (
                plan is None
                or plan_manifest is None
                or not _ingest_raw_matches(document, plan, plan_manifest)
            ):
                issues.append(
                    _issue(rule_id, attempt.attempt_id, "raw-normalized-mapping-mismatch")
                )
        elif attempt.stage == "memory_query":
            case = case_by_occurrence.get(attempt.parent_id)
            document = raw_documents.get(attempt.raw_response_ref or "")
            manifest_case = (
                manifest_case_by_id.get(case.case_manifest_entry_id) if case is not None else None
            )
            plan = (
                plan_by_occurrence.get(case.ingestion_occurrence_id) if case is not None else None
            )
            if (
                case is None
                or manifest_case is None
                or plan is None
                or not _retrieval_raw_matches(
                    document,
                    attempt.raw_response_ref or "",
                    case,
                    manifest_case,
                    plan,
                )
            ):
                issues.append(
                    _issue(rule_id, attempt.attempt_id, "raw-normalized-mapping-mismatch")
                )
        elif attempt.stage in {"answer", "judge"}:
            case = case_by_occurrence.get(attempt.parent_id)
            raw_id = attempt.raw_response_ref or attempt.raw_error_ref or ""
            document = raw_documents.get(raw_id)
            if not _model_raw_matches(
                document,
                raw_id,
                attempt,
                case,
                manifest_case_by_id.get(case.case_manifest_entry_id) if case is not None else None,
                raw_documents.get(case.retrieval_raw_ref or "") if case is not None else None,
                usage_by_attempt.get(attempt.attempt_id, []),
            ):
                issues.append(
                    _issue(rule_id, attempt.attempt_id, "raw-normalized-mapping-mismatch")
                )

    for plan in plans:
        readiness_documents = [
            raw_documents.get(reference) for reference in plan.readiness_evidence_refs
        ]
        if not _readiness_raw_matches(readiness_documents, plan):
            issues.append(
                _issue(rule_id, plan.ingestion_occurrence_id, "raw-normalized-mapping-mismatch")
            )

    for case in cases:
        if case.evaluation_disposition != CaseEvaluationDisposition.DETERMINISTIC_EVALUATED:
            continue
        document = raw_documents.get(case.evaluation_raw_ref or "")
        manifest_case = manifest_case_by_id.get(case.case_manifest_entry_id)
        if not _evaluation_raw_matches(document, case, manifest_case):
            issues.append(
                _issue(rule_id, case.case_occurrence_id, "raw-normalized-mapping-mismatch")
            )
    return tuple(issues)


def _ingest_raw_matches(
    document: dict[str, object] | None,
    plan: IngestionPlanRecord,
    plan_manifest: IngestionPlanManifest,
) -> bool:
    if document is None:
        return False
    accepted = document.get("accepted_source_unit_ids")
    rejected = document.get("rejected_source_unit_ids")
    source_units = document.get("source_units")
    expected_source_ids = _expected_source_unit_ids(plan)
    expected_scope_id = canonical_sha256(
        ["oamb-fake-scope-v1", plan.ingestion_occurrence_id, plan.ingestion_plan_id]
    )
    return (
        document.get("operation") == "ingest"
        and document.get("ingestion_occurrence_id") == plan.ingestion_occurrence_id
        and document.get("scope_id") == expected_scope_id
        and isinstance(accepted, list)
        and isinstance(rejected, list)
        and _source_units_match(source_units, expected_source_ids, plan_manifest)
        and accepted == list(expected_source_ids[: plan.accepted_source_count])
        and rejected == list(expected_source_ids[plan.accepted_source_count :])
        and len(rejected) == plan.failed_source_count
    )


def _source_units_match(
    value: object,
    expected_source_ids: tuple[str, ...],
    plan_manifest: IngestionPlanManifest,
) -> bool:
    if not isinstance(value, list) or len(value) != len(expected_source_ids):
        return False
    for source_id, payload_sha256, item in zip(
        expected_source_ids,
        plan_manifest.ordered_source_unit_bytes_sha256,
        value,
        strict=True,
    ):
        if not isinstance(item, dict):
            return False
        payload = item.get("payload")
        if (
            set(item) != {"source_unit_id", "payload_sha256", "payload"}
            or item.get("source_unit_id") != source_id
            or item.get("payload_sha256") != payload_sha256
            or not isinstance(payload, str)
            or hashlib.sha256(payload.encode("utf-8")).hexdigest() != payload_sha256
        ):
            return False
    return True


def _readiness_raw_matches(
    documents: list[dict[str, object] | None],
    plan: IngestionPlanRecord,
) -> bool:
    if not plan.attempt_ids or plan.state in {
        IngestionPlanState.CANCELLED,
        IngestionPlanState.INTERRUPTED_UNKNOWN_OUTCOME,
    }:
        return not documents
    expected_scope_id = canonical_sha256(
        ["oamb-fake-scope-v1", plan.ingestion_occurrence_id, plan.ingestion_plan_id]
    )
    expected_source_ids = list(_expected_source_unit_ids(plan)[: plan.accepted_source_count])
    expected_ready = [False, True] if plan.state == IngestionPlanState.SEALED else [False, False]
    if len(documents) != 2:
        return False
    for ordinal, document in enumerate(documents, start=1):
        if document is None:
            return False
        raw_expected_source_ids = document.get("expected_source_unit_ids")
        if not (
            document.get("operation") == "wait_ready"
            and document.get("ingestion_occurrence_id") == plan.ingestion_occurrence_id
            and document.get("scope_id") == expected_scope_id
            and document.get("check_ordinal") == ordinal
            and document.get("ready") is expected_ready[ordinal - 1]
            and raw_expected_source_ids == expected_source_ids
        ):
            return False
    return True


def _expected_source_unit_ids(plan: IngestionPlanRecord) -> tuple[str, ...]:
    return tuple(
        canonical_sha256(["oamb-fake-source-unit-v1", plan.ingestion_plan_id, ordinal])
        for ordinal in range(1, plan.intended_source_count + 1)
    )


def _retrieval_raw_matches(
    document: dict[str, object] | None,
    raw_id: str,
    case: CaseRecordV2,
    manifest_case: CaseManifestEntry,
    plan: IngestionPlanRecord,
) -> bool:
    if document is None:
        return False
    expected_scope_id = canonical_sha256(
        ["oamb-fake-scope-v1", plan.ingestion_occurrence_id, plan.ingestion_plan_id]
    )
    query = document.get("query")
    return (
        document.get("operation") == "retrieve"
        and document.get("case_occurrence_id") == case.case_occurrence_id
        and document.get("scope_id") == expected_scope_id
        and isinstance(query, str)
        and hashlib.sha256(query.encode("utf-8")).hexdigest() == manifest_case.question_bytes_sha256
        and document.get("query_sha256") == manifest_case.question_bytes_sha256
        and document.get("top_k") == 3
        and document.get("candidates") == _FAKE_RETRIEVAL_CANDIDATES
        and case.retrieval_raw_ref == raw_id
    )


def _model_raw_matches(
    document: dict[str, object] | None,
    raw_id: str,
    attempt: AttemptRecord,
    case: CaseRecordV2 | None,
    manifest_case: CaseManifestEntry | None,
    retrieval_document: dict[str, object] | None,
    usage: list[TokenUsageRecord | TokenUsageRecordV2],
) -> bool:
    if (
        document is None
        or case is None
        or manifest_case is None
        or retrieval_document is None
        or len(usage) != 1
    ):
        return False
    expected_outcome = (
        "success"
        if attempt.outcome == AttemptOutcome.SUCCEEDED
        else "judge-unavailable"
        if attempt.stage == "judge"
        else "billed-failure"
    )
    output_text = document.get("output_text")
    usage_record = usage[0]
    request_matches = _model_request_matches(
        document,
        attempt,
        case,
        manifest_case,
        retrieval_document,
    )
    expected_usage = _expected_model_usage(attempt, raw_id, document)
    usage_matches = (
        isinstance(usage_record, TokenUsageRecord)
        and expected_usage is not None
        and usage_record == expected_usage
    )
    if not isinstance(output_text, str):
        return False
    if attempt.outcome == AttemptOutcome.SUCCEEDED and attempt.stage == "answer":
        output_matches = (
            hashlib.sha256(output_text.encode("utf-8")).hexdigest() == case.parsed_answer_sha256
            and case.answer_raw_ref == raw_id
        )
    elif attempt.outcome == AttemptOutcome.FAILED:
        output_matches = output_text == ""
    else:
        output_matches = True
    return (
        document.get("attempt_id") == attempt.attempt_id
        and document.get("outcome") == expected_outcome
        and request_matches
        and usage_matches
        and output_matches
    )


def _model_request_matches(
    document: dict[str, object],
    attempt: AttemptRecord,
    case: CaseRecordV2,
    manifest_case: CaseManifestEntry,
    retrieval_document: dict[str, object],
) -> bool:
    messages = _model_messages(document)
    if messages is None:
        return False
    message_content = messages[0][1]
    if (
        canonical_sha256(messages) != attempt.request_fingerprint
        or document.get("messages_sha256") != attempt.request_fingerprint
    ):
        return False
    query = retrieval_document.get("query")
    visible = _visible_evidence_bytes(retrieval_document)
    if not isinstance(query, str) or visible is None:
        return False
    if attempt.stage == "answer":
        expected_prompt = canonical_json_bytes(
            {
                "question": query,
                "visible_evidence_sha256": hashlib.sha256(visible).hexdigest(),
                "visible_evidence": visible.decode("utf-8"),
            }
        )
        return (
            document.get("role_binding_id") == "fake-answer-v1"
            and message_content.encode("utf-8") == expected_prompt
            and case.prompt_sha256 == hashlib.sha256(expected_prompt).hexdigest()
        )
    if attempt.stage == "judge":
        if not manifest_case.answer_value_sha256:
            return False
        expected_prompt = canonical_json_bytes(
            {
                "question": query,
                "answer_sha256": case.parsed_answer_sha256,
                "reference_sha256": manifest_case.answer_value_sha256[0],
            }
        )
        return (
            document.get("role_binding_id") == "fake-judge-v1"
            and message_content.encode("utf-8") == expected_prompt
        )
    return False


def _visible_evidence_bytes(retrieval_document: dict[str, object]) -> bytes | None:
    candidates = retrieval_document.get("candidates")
    if not isinstance(candidates, list):
        return None
    visible: list[dict[str, str]] = []
    for item in candidates:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("native_id"), str)
            or not isinstance(item.get("content"), str)
        ):
            return None
        visible.append({"native_id": item["native_id"], "content": item["content"]})
    return canonical_json_bytes(visible)


def _expected_model_usage(
    attempt: AttemptRecord,
    raw_id: str,
    document: dict[str, object],
) -> TokenUsageRecord | None:
    raw_usage = document.get("usage")
    messages = _model_messages(document)
    output_text = document.get("output_text")
    if not isinstance(raw_usage, dict) or messages is None or not isinstance(output_text, str):
        return None
    input_tokens = count_message_whitespace_tokens(messages)
    output_tokens = count_whitespace_tokens(output_text)
    if (
        raw_usage.get("input_tokens") != input_tokens
        or raw_usage.get("output_tokens") != output_tokens
    ):
        return None
    return TokenUsageRecord(
        usage_record_id=canonical_sha256(
            [
                "oamb-fake-model-usage-v1",
                attempt.attempt_id,
                input_tokens,
                output_tokens,
                raw_id,
            ]
        ),
        attempt_id=attempt.attempt_id,
        parent_kind=attempt.parent_kind,
        parent_id=attempt.parent_id,
        stage=TokenStage(attempt.stage),
        operation_kind=f"fake_{attempt.stage}_completion",
        token_domain=TokenDomain.EXTERNAL_LLM,
        measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=input_tokens,
        visible_output_tokens=output_tokens,
        supplier_reported_total_tokens=input_tokens + output_tokens,
        context_view_tokens=None,
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
        raw_response_ref=raw_id,
    )


def _model_messages(document: dict[str, object]) -> tuple[tuple[str, str], ...] | None:
    raw_messages = document.get("messages")
    if not isinstance(raw_messages, list) or len(raw_messages) != 1:
        return None
    raw_message = raw_messages[0]
    if (
        not isinstance(raw_message, list)
        or len(raw_message) != 2
        or raw_message[0] != "user"
        or not isinstance(raw_message[1], str)
    ):
        return None
    return (("user", raw_message[1]),)


def _evaluation_raw_matches(
    document: dict[str, object] | None,
    case: CaseRecordV2,
    manifest_case: CaseManifestEntry | None,
) -> bool:
    if document is None or manifest_case is None or case.parsed_answer_sha256 is None:
        return False
    expected_trace = canonical_json_bytes(
        {
            "metric_id": "fake-exact-v1",
            "numerator": int(case.parsed_answer_sha256 in manifest_case.answer_value_sha256),
            "denominator": 1,
            "parsed_answer_sha256": case.parsed_answer_sha256,
        }
    )
    expected_result = hashlib.sha256(expected_trace).hexdigest()
    return (
        set(document) == {"metric_id", "result_sha256"}
        and document.get("metric_id") == "fake-exact-v1"
        and document.get("result_sha256") == expected_result
    )


def _collect_raw_references(value: object, target: set[str], key: str | None = None) -> None:
    if isinstance(value, dict):
        for child_key, child in value.items():
            _collect_raw_references(child, target, child_key)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _collect_raw_references(child, target, key)
    elif (
        isinstance(value, str)
        and key is not None
        and (
            key.endswith("_raw_ref")
            or key.endswith("_response_ref")
            or key.endswith("_error_ref")
            or key == "error_ref"
            or key == "readiness_evidence_refs"
        )
    ):
        target.add(value)


def _index_accounting_rule(snapshot: FakeCapsuleSnapshot) -> tuple[ValidationIssue, ...]:
    rule_id = "fake.index-accounting.v1"
    runs = tuple(item for item in snapshot.contracts if isinstance(item, RunRecord))
    plans = tuple(item for item in snapshot.contracts if isinstance(item, IngestionPlanRecord))
    attempts = tuple(item for item in snapshot.contracts if isinstance(item, AttemptRecord))
    usage = tuple(
        item
        for item in snapshot.contracts
        if isinstance(item, (TokenUsageRecord, TokenUsageRecordV2))
    )
    raw_documents, _raw_issues = _decode_raw_documents(snapshot, rule_id)
    issues: list[ValidationIssue] = []
    usage_ids = tuple(usage_id for plan in plans for usage_id in plan.usage_record_ids)
    if len(usage_ids) != len(set(usage_ids)):
        issues.append(_issue(rule_id, "ingestion-plans", "index-usage-duplicated"))
    if runs and runs[0].state == RunState.FINALIZED and len(usage_ids) != 2:
        issues.append(_issue(rule_id, runs[0].run_id, "index-usage-count-mismatch"))
    plan_by_occurrence = {plan.ingestion_occurrence_id: plan for plan in plans}
    usage_by_attempt: dict[str, list[TokenUsageRecord | TokenUsageRecordV2]] = {}
    for item in usage:
        usage_by_attempt.setdefault(item.attempt_id, []).append(item)
    for attempt in attempts:
        plan = plan_by_occurrence.get(attempt.parent_id)
        if attempt.stage == "memory_ingest":
            expected_contribution = (
                IndexContribution.FINAL
                if attempt.outcome == AttemptOutcome.SUCCEEDED
                and plan is not None
                and plan.state == IngestionPlanState.SEALED
                else IndexContribution.NONE
            )
        else:
            expected_contribution = IndexContribution.NOT_APPLICABLE
        if attempt.index_contribution != expected_contribution:
            issues.append(_issue(rule_id, attempt.attempt_id, "index-contribution-invalid"))
        expected_usage = _expected_attempt_usage(attempt, raw_documents)
        actual_usage = usage_by_attempt.get(attempt.attempt_id, [])
        if expected_usage is None:
            issues.append(_issue(rule_id, attempt.attempt_id, "attempt-usage-mismatch"))
            continue
        actual_by_id = {item.usage_record_id: item for item in actual_usage}
        expected_by_id = {item.usage_record_id: item for item in expected_usage}
        if len(actual_usage) != len(expected_usage) or actual_by_id != expected_by_id:
            issues.append(_issue(rule_id, attempt.attempt_id, "attempt-usage-mismatch"))
    return tuple(issues)


def _expected_attempt_usage(
    attempt: AttemptRecord,
    raw_documents: dict[str, dict[str, object]],
) -> tuple[TokenUsageRecord, ...] | None:
    if attempt.outcome == AttemptOutcome.UNKNOWN_OUTCOME:
        return ()
    raw_id = attempt.raw_response_ref or attempt.raw_error_ref
    if raw_id is None:
        return None
    raw_document = raw_documents.get(raw_id)
    if raw_document is None:
        return None
    if attempt.stage == "memory_ingest":
        input_tokens = _ingest_raw_input_tokens(raw_document)
        if input_tokens is None:
            return None
        return (
            TokenUsageRecord(
                usage_record_id=canonical_sha256(
                    [
                        "oamb-fake-runtime-usage-v1",
                        attempt.attempt_id,
                        TokenStage.MEMORY_INGEST.value,
                        input_tokens,
                        0,
                    ]
                ),
                attempt_id=attempt.attempt_id,
                parent_kind="ingestion_plan",
                parent_id=attempt.parent_id,
                stage=TokenStage.MEMORY_INGEST,
                operation_kind="fake_memory_ingest",
                token_domain=TokenDomain.EXTERNAL_LLM,
                measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
                input_tokens=input_tokens,
                visible_output_tokens=0,
                supplier_reported_total_tokens=input_tokens,
                context_view_tokens=None,
                proof_status=ProofStatus.MEASURED_COMPLETE,
                reason=None,
                raw_response_ref=raw_id,
            ),
        )
    if attempt.stage == "memory_query":
        visible = _visible_evidence_bytes(raw_document)
        if visible is None:
            return None
        context_tokens = len(visible.decode("utf-8").split())
        return (
            TokenUsageRecord(
                usage_record_id=canonical_sha256(
                    ["oamb-fake-query-usage-v1", attempt.attempt_id, "not-applicable"]
                ),
                attempt_id=attempt.attempt_id,
                parent_kind="case",
                parent_id=attempt.parent_id,
                stage=TokenStage.MEMORY_QUERY,
                operation_kind="fake_memory_query",
                token_domain=TokenDomain.EXTERNAL_LLM,
                measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
                input_tokens=None,
                visible_output_tokens=None,
                supplier_reported_total_tokens=None,
                context_view_tokens=None,
                proof_status=ProofStatus.NOT_APPLICABLE,
                reason="fake-memory-query-has-no-model-meter",
                raw_response_ref=raw_id,
            ),
            TokenUsageRecord(
                usage_record_id=canonical_sha256(
                    ["oamb-fake-context-usage-v1", attempt.attempt_id, context_tokens]
                ),
                attempt_id=attempt.attempt_id,
                parent_kind="case",
                parent_id=attempt.parent_id,
                stage=TokenStage.CONTEXT_VIEW,
                operation_kind="fake_visible_context",
                token_domain=TokenDomain.LOCAL_CONTEXT_VIEW,
                measurement_source=TokenMeasurementSource.LOCAL_TOKENIZER,
                input_tokens=None,
                visible_output_tokens=None,
                supplier_reported_total_tokens=None,
                context_view_tokens=context_tokens,
                proof_status=ProofStatus.MEASURED_COMPLETE,
                reason=None,
                raw_response_ref=None,
            ),
        )
    if attempt.stage in {"answer", "judge"}:
        expected = _expected_model_usage(attempt, raw_id, raw_document)
        return (expected,) if expected is not None else None
    return None


def _ingest_raw_input_tokens(document: dict[str, object]) -> int | None:
    source_units = document.get("source_units")
    if not isinstance(source_units, list):
        return None
    token_count = 0
    for item in source_units:
        if not isinstance(item, dict) or not isinstance(item.get("payload"), str):
            return None
        token_count += len(item["payload"].split())
    return token_count


FAKE_EVIDENCE_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("fake.schema.v1", 1, _schema_rule),
    ValidationRule("fake.manifest-closure.v1", 1, _manifest_rule),
    ValidationRule("fake.parentage.v1", 1, _parentage_rule),
    ValidationRule("fake.execution.v1", 1, _execution_rule),
    ValidationRule("fake.raw-reference.v1", 1, _raw_reference_rule),
    ValidationRule("fake.index-accounting.v1", 1, _index_accounting_rule),
)


__all__ = [
    "EvidenceNotValidatedError",
    "FAKE_EVIDENCE_RULE_IDS",
    "FakeCapsuleSnapshot",
    "fake_evidence_registry",
    "load_fake_capsule",
    "validate_fake_capsule",
    "validate_fake_snapshot",
]
