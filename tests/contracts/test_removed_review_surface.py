from __future__ import annotations

from importlib.util import find_spec

from oamb.artifacts.validation.profiles import (
    REPORT_KINDS,
    validation_profile_catalog,
)
from oamb.contracts.schema import CONTRACT_REGISTRY

REMOVED_MODULES = (
    "oamb.phase_cli",
    "oamb.phase_cli_io",
    "oamb.phase_review_profiles",
    "oamb.phase_review_repository",
    "oamb.phase_review_run",
    "oamb.artifacts.validation.phase",
    "oamb.reporting.human_review",
    "oamb.reporting.review",
    "oamb.reporting.t10_phase_assemble",
)

REMOVED_CONTRACT_NAMES = {
    "acceptance_report_spec",
    "ai_quality_review_record",
    "ai_review_batch",
    "ai_review_batch_result",
    "ai_review_case_projection",
    "ai_review_case_result",
    "ai_review_finding",
    "ai_review_integrity_projection",
    "ai_review_integrity_result",
    "ai_review_plan",
    "evaluation_phase_gate",
    "evaluation_review_bundle",
    "human_quality_review_record",
    "human_review_decision",
    "human_review_key_binding",
    "phase_acceptance_report",
    "phase_review_occurrence_record",
    "signature_verification_record",
}


def test_removed_review_modules_are_not_packaged() -> None:
    assert tuple(module for module in REMOVED_MODULES if find_spec(module) is not None) == ()


def test_removed_review_contracts_and_profiles_are_not_registered() -> None:
    registered_contract_names = {name for name, _version in CONTRACT_REGISTRY}

    assert registered_contract_names.isdisjoint(REMOVED_CONTRACT_NAMES)
    assert REPORT_KINDS == ("run", "comparison", "evaluation", "release")
    assert all("phase" not in profile_id for profile_id in validation_profile_catalog())
