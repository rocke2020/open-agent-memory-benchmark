"""Shared fail-closed execution for validation profiles with frozen inventories."""

from __future__ import annotations

from typing import Any

from oamb.artifacts.validation.registry import RuleRegistry
from oamb.contracts.evidence import (
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import ValidationProfile, ValidationStage
from oamb.contracts.states import ValidationDisposition


def validate_closed_profile(
    target: Any,
    *,
    target_hash: str,
    profile: ValidationProfile,
    expected_stage: ValidationStage,
    expected_inventory: tuple[tuple[str, int], ...],
    registry: RuleRegistry,
) -> ValidationResult:
    required_ids = tuple(item.rule_id for item in profile.required_rules)
    expected_ids = tuple(rule_id for rule_id, _version in expected_inventory)
    expected_inventory_hash = canonical_sha256(
        ["oamb-required-rule-inventory-v1", expected_inventory]
    )
    if profile.stage != expected_stage:
        return _preflight_failure(
            profile,
            target_hash,
            required_ids,
            "validation.stage.v1",
            "wrong-validation-stage",
        )
    if (
        required_ids != expected_ids
        or tuple((item.rule_id, item.minimum_version) for item in profile.required_rules)
        != expected_inventory
        or profile.required_rule_inventory_hash != expected_inventory_hash
    ):
        return _preflight_failure(
            profile,
            target_hash,
            required_ids,
            "validation.profile-inventory.v1",
            "profile-inventory-mismatch",
        )
    duplicated_required_ids = tuple(
        rule_id for rule_id in expected_ids if rule_id in registry.duplicate_rule_ids
    )
    if duplicated_required_ids:
        return _preflight_failure(
            profile,
            target_hash,
            required_ids,
            "validation.rule-duplication.v1",
            "duplicate-required-rule",
        )

    executed: list[str] = []
    passed: list[str] = []
    failed: list[str] = []
    missing: list[str] = []
    implementation_versions: list[str] = []
    issues: list[ValidationIssue] = []
    for requirement in profile.required_rules:
        rule = registry.compatible(requirement.rule_id, requirement.minimum_version)
        if rule is None:
            missing.append(requirement.rule_id)
            continue
        executed.append(rule.rule_id)
        implementation_versions.append(f"{rule.rule_id}@{rule.version}")
        try:
            rule_issues = rule.evaluate(target)
        except Exception:
            rule_issues = (_issue(rule.rule_id, target_hash, "validation-rule-execution-error"),)
        if rule_issues:
            failed.append(rule.rule_id)
            issues.extend(rule_issues)
        else:
            passed.append(rule.rule_id)
    if missing:
        availability_rule_id = "validation.rule-availability.v1"
        failed.append(availability_rule_id)
        issues.append(
            _issue(
                availability_rule_id,
                ",".join(missing),
                "validation-rule-missing-or-incompatible",
            )
        )
    return ValidationResult(
        validation_profile_id=profile.profile_id,
        target_hash=target_hash,
        disposition=(
            ValidationDisposition.VALIDATED
            if not failed and not missing and tuple(executed) == expected_ids
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


def _preflight_failure(
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


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


__all__ = ["validate_closed_profile"]
