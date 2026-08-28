from __future__ import annotations

import importlib
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest

from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.contracts.states import ValidationDisposition

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SOURCE = REPOSITORY_ROOT / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"

EXPECTED_IDS = (
    "72e3ee87",
    "22d2cb42",
    "e61a7584",
    "eace081b",
    "8fb83627",
    "21d02d0d",
    "gpt4_a56e767c",
    "gpt4_15e38248",
    "4bc144e2",
    "157a136e",
    "e3fc4d6e",
    "41275add",
    "fca762bc",
    "488d3006",
    "8cf51dda",
    "32260d93",
    "1d4e3b97",
    "0a34ad58",
    "07b6f563",
    "d6233ab6",
    "6b168ec8",
    "bc8a6e93_abs",
    "ccb36322",
    "8e9d538c",
    "c14c00dd",
    "a3045048",
    "gpt4_ec93e27f",
    "gpt4_8279ba03",
    "gpt4_4edbafa2",
    "b46e15ed",
)
EXPECTED_LME6_IDS = (
    "72e3ee87",
    "21d02d0d",
    "e3fc4d6e",
    "32260d93",
    "6b168ec8",
    "a3045048",
)


def require_longmemeval() -> ModuleType:
    try:
        return importlib.import_module("oamb.workloads.longmemeval")
    except ModuleNotFoundError:
        pytest.fail("oamb.workloads.longmemeval is not implemented", pytrace=False)


@pytest.mark.skipif(not SOURCE.exists(), reason="download the pinned LongMemEval S dataset")
def test_real_pinned_source_builds_exact_lme30_and_lme6_manifests() -> None:
    lme = require_longmemeval()

    full = lme.build_lme30_bundle(SOURCE)
    smoke = lme.build_lme6_bundle(full)
    reversed_selection = lme.select_lme30(tuple(reversed(lme.load_longmemeval_rows(SOURCE))))

    assert tuple(row.question_id for row in full.selected_rows) == EXPECTED_IDS
    assert tuple(row.question_id for row in reversed_selection) == EXPECTED_IDS
    assert full.dataset_manifest.manifest_hash == lme.LME_DATASET_MANIFEST_HASH
    assert full.case_manifest.manifest_hash == lme.LME30_CASE_MANIFEST_HASH
    assert full.dataset_manifest.source_files[0].license_id == "NOASSERTION"
    assert len(full.case_manifest.logical_contexts) == 30
    assert len(full.case_manifest.ingestion_plans) == 30
    assert len(full.case_manifest.cases) == 30
    assert {case.metric_id for case in full.case_plans} == {"lme-judged-accuracy-v1"}
    assert {case.answer_max_output_tokens for case in full.case_plans} == {8_192}
    assert sum(plan.intended_source_count for plan in full.ingestion_plans) == 1_418
    assert all(
        source.source_reference is not None
        and source.occurred_at is not None
        and source.context_text
        == f"LongMemEval session {source.source_reference} at {source.occurred_at}"
        for plan in full.ingestion_plans
        for source in plan.ordered_source_units
    )
    assert all(case.query_timestamp is not None for case in full.case_plans)
    assert full.label_mismatch_question_ids == (
        "22d2cb42",
        "eace081b",
        "bc8a6e93_abs",
    )

    assert tuple(row.question_id for row in smoke.selected_rows) == EXPECTED_LME6_IDS
    assert len(smoke.case_manifest.ingestion_plans) == 6
    assert len(smoke.case_manifest.cases) == 6
    assert sum(plan.intended_source_count for plan in smoke.ingestion_plans) == 300
    assert tuple(case.case_manifest_entry_id for case in smoke.case_manifest.cases) == tuple(
        full.case_manifest.cases[full.selected_rows.index(row)].case_manifest_entry_id
        for row in smoke.selected_rows
    )
    assert smoke.case_manifest.manifest_id == "lme6-live-smoke-v1"
    assert smoke.case_manifest.manifest_hash == lme.LME6_CASE_MANIFEST_HASH
    assert smoke.case_manifest.workload_id == "lme30-native-smoke-plus-v1"
    full_plans_by_case = {
        plan.ordered_case_manifest_entry_ids[0]: plan for plan in full.case_manifest.ingestion_plans
    }
    for smoke_plan in smoke.case_manifest.ingestion_plans:
        full_plan = full_plans_by_case[smoke_plan.ordered_case_manifest_entry_ids[0]]
        assert smoke_plan == full_plan


@pytest.mark.skipif(not SOURCE.exists(), reason="download the pinned LongMemEval S dataset")
def test_real_lme_bundles_pass_closed_production_profiles_and_manifest_drift_fails() -> None:
    lme = require_longmemeval()
    full = lme.build_lme30_bundle(SOURCE)
    smoke = lme.build_lme6_bundle(full)

    for profile_id, bundle in (
        ("oamb-t8-workload-lme30-v1", full),
        ("oamb-t8-workload-lme6-v1", smoke),
    ):
        result = validate_catalog_profile(profile_id, bundle)
        assert result.disposition == ValidationDisposition.VALIDATED, result.issues
        assert result.executed_rule_ids == result.required_rule_ids

    drifted = replace(
        full,
        case_manifest=full.case_manifest.model_copy(update={"manifest_hash": "f" * 64}),
    )
    drifted_result = validate_catalog_profile("oamb-t8-workload-lme30-v1", drifted)
    assert drifted_result.disposition == ValidationDisposition.INVALID
    assert "workload.lme30.selector-coverage.v1" in drifted_result.failed_rule_ids


@pytest.mark.skipif(not SOURCE.exists(), reason="download the pinned LongMemEval S dataset")
def test_lme_profile_binds_complete_dataset_source_identity() -> None:
    lme = require_longmemeval()
    full = lme.build_lme30_bundle(SOURCE)
    original_dataset = full.dataset_manifest
    original_file = original_dataset.source_files[0]
    baseline = validate_catalog_profile("oamb-t8-workload-lme30-v1", full)
    drifted_datasets = (
        original_dataset.model_copy(update={"split": "changed"}),
        original_dataset.model_copy(update={"payload_policy": "changed"}),
        original_dataset.model_copy(
            update={
                "source_files": (
                    original_file.model_copy(update={"relative_path": "changed.json"}),
                )
            }
        ),
        original_dataset.model_copy(
            update={"source_files": (original_file.model_copy(update={"byte_count": 1}),)}
        ),
        original_dataset.model_copy(
            update={"source_files": (original_file.model_copy(update={"license_id": "MIT"}),)}
        ),
    )

    for dataset in drifted_datasets:
        result = validate_catalog_profile(
            "oamb-t8-workload-lme30-v1",
            replace(full, dataset_manifest=dataset),
        )
        assert result.disposition == ValidationDisposition.INVALID
        assert "workload.lme.source-manifest.v1" in result.failed_rule_ids
        assert result.target_hash != baseline.target_hash


@pytest.mark.skipif(not SOURCE.exists(), reason="download the pinned LongMemEval S dataset")
def test_lme_profile_rejects_case_reference_payload_tamper() -> None:
    lme = require_longmemeval()
    smoke = lme.build_lme6_bundle(lme.build_lme30_bundle(SOURCE))
    baseline = validate_catalog_profile("oamb-t8-workload-lme6-v1", smoke)
    tampered = replace(
        smoke,
        case_plans=(
            replace(smoke.case_plans[0], reference_payload=b'"attacker-gold"'),
            *smoke.case_plans[1:],
        ),
    )

    result = validate_catalog_profile("oamb-t8-workload-lme6-v1", tampered)

    assert result.target_hash != baseline.target_hash
    assert result.disposition == ValidationDisposition.INVALID
    assert "workload.lme.prompt-answer-parser.v1" in result.failed_rule_ids


@pytest.mark.skipif(not SOURCE.exists(), reason="download the pinned LongMemEval S dataset")
def test_lme_profile_rejects_runtime_source_payload_tamper() -> None:
    lme = require_longmemeval()
    smoke = lme.build_lme6_bundle(lme.build_lme30_bundle(SOURCE))
    baseline = validate_catalog_profile("oamb-t8-workload-lme6-v1", smoke)
    original_plan = smoke.ingestion_plans[0]
    original_unit = original_plan.ordered_source_units[0]
    tampered_plan = replace(
        original_plan,
        ordered_source_units=(
            replace(original_unit, payload_bytes=b"attacker source payload"),
            *original_plan.ordered_source_units[1:],
        ),
    )
    tampered = replace(smoke, ingestion_plans=(tampered_plan, *smoke.ingestion_plans[1:]))

    result = validate_catalog_profile("oamb-t8-workload-lme6-v1", tampered)

    assert result.target_hash != baseline.target_hash
    assert result.disposition == ValidationDisposition.INVALID
    assert "workload.lme.source-manifest.v1" in result.failed_rule_ids
