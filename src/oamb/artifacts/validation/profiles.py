"""Closed T3 evidence profile for schema and structural rules."""

from __future__ import annotations

from oamb.contracts.specifications import (
    ValidationProfile,
    ValidationRuleRequirement,
    ValidationStage,
)

from .core import STRUCTURAL_RULES


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
