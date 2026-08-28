"""Composite native-run evidence validation for report reduction and publication."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oamb.artifacts.atomic import read_regular_file
from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.artifacts.validation.engine import validate_closed_profile
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.artifacts.validation.registry import RuleRegistry, ValidationRule
from oamb.contracts.evidence import (
    CapsuleManifest,
    IngestionPlanRecordV2,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.specifications import (
    CaseManifest,
    DatasetManifest,
    ValidationProfile,
    ValidationRuleRequirement,
    ValidationStage,
)
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.native_reduce import build_native_accounting_validation_input
from oamb.workloads.longmemeval import LME6_MANIFEST_ID, LME30_WORKLOAD_ID, LongMemEvalBundle
from oamb.workloads.memoryagentbench import MAB5_MANIFEST_ID, MAB65_WORKLOAD_ID, MabManifestBundle

NATIVE_RUN_EVIDENCE_PROFILE_ID = "oamb-t8-native-run-evidence-v1"
NATIVE_RUN_EVIDENCE_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("run-evidence.source-coherence.v1", 1),
    ("run-evidence.native-capsule.v1", 1),
    ("run-evidence.workload-profile.v1", 1),
    ("run-evidence.adapter-profile.v1", 1),
    ("run-evidence.accounting-profile.v1", 1),
)

_WORKLOAD_MANIFEST_IDS = {
    "oamb-t8-workload-lme6-v1": LME6_MANIFEST_ID,
    "oamb-t8-workload-lme30-v1": f"{LME30_WORKLOAD_ID}-manifest-v1",
    "oamb-t8-workload-mab5-v1": MAB5_MANIFEST_ID,
    "oamb-t8-workload-mab65-v1": MAB65_WORKLOAD_ID,
}
_ADAPTER_PROFILE_IDS = {
    "oamb-t8-adapter-hindsight-rest-v1": "hindsight-rest-v1",
    "oamb-t8-adapter-openviking-rest-v1": "openviking-rest-v1",
}


@dataclass(frozen=True, slots=True)
class NativeRunEvidenceValidationInput:
    capsule_root: Path
    workload_profile_id: str
    workload_target: LongMemEvalBundle | MabManifestBundle
    adapter_profile_id: str
    accounting_target: Any


def native_run_evidence_validation_input(
    *,
    capsule_root: Path,
    workload_profile_id: str,
    workload_target: LongMemEvalBundle | MabManifestBundle,
    adapter_profile_id: str,
) -> NativeRunEvidenceValidationInput:
    if workload_profile_id not in _WORKLOAD_MANIFEST_IDS:
        raise ValueError("unsupported exact workload profile for native-run evidence")
    if adapter_profile_id not in _ADAPTER_PROFILE_IDS:
        raise ValueError("unsupported exact adapter profile for native-run evidence")
    root = Path(capsule_root)
    return NativeRunEvidenceValidationInput(
        capsule_root=root,
        workload_profile_id=workload_profile_id,
        workload_target=workload_target,
        adapter_profile_id=adapter_profile_id,
        accounting_target=build_native_accounting_validation_input(root),
    )


def validate_native_run_evidence(
    target: NativeRunEvidenceValidationInput,
) -> ValidationResult:
    return validate_closed_profile(
        target,
        target_hash=_capsule_source_root(target.capsule_root),
        profile=_native_run_evidence_profile(),
        expected_stage=ValidationStage.EVIDENCE,
        expected_inventory=NATIVE_RUN_EVIDENCE_RULE_INVENTORY,
        registry=_native_run_evidence_registry(),
    )


def _native_run_evidence_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id=NATIVE_RUN_EVIDENCE_PROFILE_ID,
        stage=ValidationStage.EVIDENCE,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=version)
            for rule_id, version in NATIVE_RUN_EVIDENCE_RULE_INVENTORY
        ),
        applicability=("origin=native", "publication=run"),
    )


def _native_run_evidence_registry() -> RuleRegistry:
    registry = RuleRegistry()
    for rule in NATIVE_RUN_EVIDENCE_RULES:
        registry.register(rule)
    return registry


def _source_coherence_rule(
    target: NativeRunEvidenceValidationInput,
) -> tuple[ValidationIssue, ...]:
    rule_id = "run-evidence.source-coherence.v1"
    try:
        manifest = _capsule_manifest(target.capsule_root)
        dataset = _single_contract(
            target.capsule_root,
            manifest,
            "dataset_manifest",
            DatasetManifest,
        )
        case_manifest = _single_contract(
            target.capsule_root,
            manifest,
            "case_manifest",
            CaseManifest,
        )
        plans = _contracts(
            target.capsule_root,
            manifest,
            "ingestion_plan_record",
            IngestionPlanRecordV2,
        )
        expected_manifest_id = _WORKLOAD_MANIFEST_IDS[target.workload_profile_id]
        expected_adapter_profile_id = _ADAPTER_PROFILE_IDS[target.adapter_profile_id]
        rebuilt_accounting = build_native_accounting_validation_input(target.capsule_root)
        exact = bool(
            dataset == target.workload_target.dataset_manifest
            and case_manifest == target.workload_target.case_manifest
            and case_manifest.manifest_id == expected_manifest_id
            and plans
            and all(plan.adapter_profile_id == expected_adapter_profile_id for plan in plans)
            and rebuilt_accounting == target.accounting_target
        )
    except (KeyError, OSError, TypeError, ValueError):
        exact = False
    return () if exact else (_issue(rule_id, "native-run", "run-source-coherence-drift"),)


def _native_capsule_rule(
    target: NativeRunEvidenceValidationInput,
) -> tuple[ValidationIssue, ...]:
    return _component_rule(
        "run-evidence.native-capsule.v1",
        validate_native_capsule(target.capsule_root),
        "native-capsule-invalid",
    )


def _workload_profile_rule(
    target: NativeRunEvidenceValidationInput,
) -> tuple[ValidationIssue, ...]:
    rule_id = "run-evidence.workload-profile.v1"
    try:
        validation = validate_catalog_profile(
            target.workload_profile_id,
            target.workload_target,
        )
    except (KeyError, OSError, TypeError, ValueError):
        validation = None
    return _component_rule(rule_id, validation, "workload-profile-invalid")


def _adapter_profile_rule(
    target: NativeRunEvidenceValidationInput,
) -> tuple[ValidationIssue, ...]:
    rule_id = "run-evidence.adapter-profile.v1"
    try:
        validation = validate_catalog_profile(target.adapter_profile_id, target.capsule_root)
    except (KeyError, OSError, TypeError, ValueError):
        validation = None
    return _component_rule(rule_id, validation, "adapter-profile-invalid")


def _accounting_profile_rule(
    target: NativeRunEvidenceValidationInput,
) -> tuple[ValidationIssue, ...]:
    rule_id = "run-evidence.accounting-profile.v1"
    try:
        validation = validate_catalog_profile(
            "oamb-t8-accounting-native-v1",
            target.accounting_target,
        )
    except (KeyError, OSError, TypeError, ValueError):
        validation = None
    return _component_rule(rule_id, validation, "accounting-profile-invalid")


def _component_rule(
    rule_id: str,
    validation: ValidationResult | None,
    code: str,
) -> tuple[ValidationIssue, ...]:
    exact = bool(
        validation is not None
        and validation.disposition == ValidationDisposition.VALIDATED
        and validation.required_rule_ids == validation.executed_rule_ids
        and validation.required_rule_ids == validation.passed_rule_ids
        and not validation.failed_rule_ids
        and not validation.not_applicable_rule_ids
        and not validation.missing_rule_ids
        and not validation.issues
    )
    return () if exact else (_issue(rule_id, "native-run", code),)


def _capsule_source_root(root: Path) -> str:
    try:
        return _capsule_manifest(root).source_manifest_hash
    except (OSError, TypeError, ValueError):
        try:
            content = read_regular_file(Path(root) / "capsule-manifest.json")
        except OSError:
            content = b""
        import hashlib

        return hashlib.sha256(content).hexdigest()


def _capsule_manifest(root: Path) -> CapsuleManifest:
    return CapsuleManifest.model_validate_json(
        read_regular_file(Path(root) / "capsule-manifest.json")
    )


def _single_contract(
    root: Path,
    manifest: CapsuleManifest,
    record_kind: str,
    contract_type: type[Any],
) -> Any:
    values = _contracts(root, manifest, record_kind, contract_type)
    if len(values) != 1:
        raise ValueError(f"native run requires one {record_kind}")
    return values[0]


def _contracts(
    root: Path,
    manifest: CapsuleManifest,
    record_kind: str,
    contract_type: type[Any],
) -> tuple[Any, ...]:
    return tuple(
        contract_type.model_validate_json(read_regular_file(Path(root) / entry.relative_path))
        for entry in manifest.source_entries
        if entry.record_kind == record_kind
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


NATIVE_RUN_EVIDENCE_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("run-evidence.source-coherence.v1", 1, _source_coherence_rule),
    ValidationRule("run-evidence.native-capsule.v1", 1, _native_capsule_rule),
    ValidationRule("run-evidence.workload-profile.v1", 1, _workload_profile_rule),
    ValidationRule("run-evidence.adapter-profile.v1", 1, _adapter_profile_rule),
    ValidationRule("run-evidence.accounting-profile.v1", 1, _accounting_profile_rule),
)


__all__ = [
    "NATIVE_RUN_EVIDENCE_PROFILE_ID",
    "NATIVE_RUN_EVIDENCE_RULE_INVENTORY",
    "NativeRunEvidenceValidationInput",
    "native_run_evidence_validation_input",
    "validate_native_run_evidence",
]
