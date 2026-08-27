"""Closed evidence profiles for structural and provider-service roots."""

from __future__ import annotations

from oamb.contracts.specifications import (
    ValidationProfile,
    ValidationRuleRequirement,
    ValidationStage,
)

from .core import STRUCTURAL_RULES

T4_PROVIDER_SERVICE_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("t4-provider-schema", 1),
    ("t4-provider-manifest-closure", 1),
    ("t4-provider-path-safety", 1),
    ("t4-provider-occurrence-terminal", 1),
    ("t4-provider-parent-isolation", 1),
    ("t4-provider-budget-closure", 1),
    ("t4-provider-attempt-ordering", 1),
    ("t4-provider-unknown-outcome", 1),
    ("t4-provider-runtime-binding", 1),
    ("t4-provider-gate-separation", 1),
    ("t4-provider-control-redaction", 1),
)


def structural_evidence_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id="oamb-t3-structural-evidence-v1",
        stage=ValidationStage.EVIDENCE,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule.rule_id, minimum_version=rule.version)
            for rule in STRUCTURAL_RULES
        ),
        applicability=("t3-contract-kernel",),
    )


def provider_service_evidence_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id="oamb-t4-provider-service-evidence-v1",
        stage=ValidationStage.EVIDENCE,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=minimum_version)
            for rule_id, minimum_version in T4_PROVIDER_SERVICE_RULE_INVENTORY
        ),
        applicability=(
            "source_kind=provider_service",
            "operation=model_readiness",
        ),
    )
