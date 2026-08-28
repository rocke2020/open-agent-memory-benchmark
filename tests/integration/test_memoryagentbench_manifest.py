from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.contracts.states import ValidationDisposition

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPOSITORY_ROOT / "datasets" / "MemoryAgentBench"
SNAPSHOT_PATH = (
    REPOSITORY_ROOT
    / "tests"
    / "fixtures"
    / "workloads"
    / "memoryagentbench"
    / "selection-snapshot.json"
)


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_pinned_mab65_manifest_matches_frozen_selection_snapshot() -> None:
    from oamb.workloads.memoryagentbench import MAB65_CASE_MANIFEST_HASH, build_mab65_manifest

    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    bundle = build_mab65_manifest(DATASET_ROOT)

    assert bundle.dataset_manifest.manifest_hash == expected["dataset_manifest_hash"]
    assert {item.license_id for item in bundle.dataset_manifest.source_files} == {"NOASSERTION"}
    assert len(bundle.case_manifest.logical_contexts) == 29
    assert len(bundle.case_manifest.ingestion_plans) == 25
    assert len(bundle.case_manifest.cases) == 65
    assert bundle.selected_case_digest == expected["selected_case_digest"]
    assert bundle.selected_plan_digest == expected["selected_plan_digest"]
    assert bundle.case_manifest.manifest_hash == MAB65_CASE_MANIFEST_HASH
    assert bundle.entity_catalog_sha256 == (
        "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5"
    )
    assert len(bundle.movie_catalog.movies) == 31_161
    assert bundle.movie_catalog.unicode_version == "14.0.0"
    assert bundle.unicode_fingerprint == (
        "1a846dffe314101b43cb68132979e8a1f052b7d2115532981d9866db1eeb352a"
    )
    assert [plan.plan_manifest_entry_id for plan in bundle.case_manifest.ingestion_plans] == [
        plan[0] for plan in expected["plans"]
    ]
    actual_snapshot = []
    for plan in bundle.plans:
        positions_by_member = [
            [
                case.entry.source_question_number_1_indexed
                for case in bundle.cases
                if case.entry.context_manifest_entry_id == member_id
                and case.entry.case_manifest_entry_id
                in plan.manifest.ordered_case_manifest_entry_ids
            ]
            for member_id in plan.manifest.ordered_member_context_manifest_entry_ids
        ]
        actual_snapshot.append(
            [
                plan.manifest.plan_manifest_entry_id,
                list(plan.member_labels),
                positions_by_member,
            ]
        )
    assert actual_snapshot == expected["plans"]
    assert Counter(case.component for case in bundle.cases) == {
        "ar": 15,
        "icl": 10,
        "recsys": 10,
        "lru": 15,
        "cr_sf": 15,
    }
    assert len({case.entry.raw_question_id for case in bundle.cases}) == 64
    assert len({case.entry.case_manifest_entry_id for case in bundle.cases}) == 65
    assert {
        case.case_plan.prompt_binding_id: case.case_plan.answer_max_output_tokens
        for case in bundle.cases
    } == {
        "oamb-mab-eventqa-rag-v1": 40,
        "oamb-mab-icl-rag-v1": 20,
        "oamb-mab-redial-rag-v1": 512,
        "oamb-mab-detectiveqa-rag-v1": 2_000,
        "oamb-mab-factconsolidation-rag-v1": 10,
    }
    assert all(case.source_relative_path.startswith("data/") for case in bundle.cases)


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_mab5_filters_five_cases_without_splitting_262k_grouped_plan() -> None:
    from oamb.workloads.memoryagentbench import (
        MAB5_CASE_MANIFEST_HASH,
        MAB5_SELECTED_CASE_DIGEST,
        MAB5_SELECTED_PLAN_DIGEST,
        build_mab5_manifest,
        build_mab65_manifest,
    )

    full = build_mab65_manifest(DATASET_ROOT)
    smoke = build_mab5_manifest(full)

    assert len(smoke.case_manifest.ingestion_plans) == 5
    assert len(smoke.case_manifest.cases) == 5
    assert smoke.case_manifest.manifest_hash == MAB5_CASE_MANIFEST_HASH
    assert smoke.selected_case_digest == MAB5_SELECTED_CASE_DIGEST
    assert smoke.selected_plan_digest == MAB5_SELECTED_PLAN_DIGEST
    assert [case.logical_case_label for case in smoke.cases] == [
        "eventqa_65536@q6",
        "recsys_redial_full@q33",
        "icl_banking77_5900shot_balance@q56",
        "detective_qa@q8",
        "factconsolidation_mh_262k@q21",
    ]
    grouped = smoke.case_manifest.ingestion_plans[-1]
    assert len(grouped.ordered_member_context_manifest_entry_ids) == 2
    assert grouped.plan_manifest_entry_id == (
        "31e2161abe21a8f936d389e706270dd7ebfe1ed9302d4aa10cf3a0783340d9c1"
    )
    assert len(grouped.ordered_case_manifest_entry_ids) == 1


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_mab65_selection_is_independent_of_loaded_row_iteration_order() -> None:
    from oamb.workloads import memoryagentbench as mab
    from oamb.workloads.memoryagentbench import MabDatasetRow

    rows = tuple(
        row
        for relative_path, sha256, split, _byte_count in mab.MAB_PINNED_SOURCE_FILES
        for row in mab.read_aligned_parquet_rows(
            DATASET_ROOT / relative_path,
            split=split,
            expected_sha256=sha256,
            source_relative_path=relative_path,
        )
    )

    def snapshot(loaded_rows: tuple[MabDatasetRow, ...]) -> tuple[object, ...]:
        return tuple(
            (
                tuple(row.context_entry.context_manifest_entry_id for row in members),
                position_groups,
            )
            for members, position_groups in mab._selected_groups(loaded_rows)
        )

    assert snapshot(rows) == snapshot(tuple(reversed(rows)))


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_complete_real_mab65_manifest_can_emit_the_secondary_index() -> None:
    from oamb.workloads.mab65_reduction import reduce_mab65
    from oamb.workloads.memoryagentbench import build_mab65_manifest

    bundle = build_mab65_manifest(DATASET_ROOT)
    result = reduce_mab65(
        bundle,
        case_metrics={case.entry.case_manifest_entry_id: Fraction(1) for case in bundle.cases},
        ready_plan_manifest_ids=frozenset(
            plan.manifest.plan_manifest_entry_id for plan in bundle.plans
        ),
        capsule_valid=True,
        query_state_unchanged=True,
        expected_catalog_sha256=bundle.entity_catalog_sha256,
        expected_unicode_fingerprint=bundle.unicode_fingerprint,
        expected_interaction_fingerprint="f" * 64,
        observed_interaction_fingerprint="f" * 64,
        comparison_controls_closed=True,
    )

    assert result.available is True
    assert result.index_value == 100


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_real_mab_bundles_pass_closed_production_profiles_and_manifest_drift_fails() -> None:
    from oamb.workloads.memoryagentbench import build_mab5_manifest, build_mab65_manifest

    full = build_mab65_manifest(DATASET_ROOT)
    smoke = build_mab5_manifest(full)
    for profile_id, bundle in (
        ("oamb-t8-workload-mab65-v1", full),
        ("oamb-t8-workload-mab5-v1", smoke),
    ):
        result = validate_catalog_profile(profile_id, bundle)
        assert result.disposition == ValidationDisposition.VALIDATED, result.issues
        assert result.executed_rule_ids == result.required_rule_ids

    drifted = replace(
        full,
        case_manifest=full.case_manifest.model_copy(update={"manifest_hash": "f" * 64}),
    )
    drifted_result = validate_catalog_profile("oamb-t8-workload-mab65-v1", drifted)
    assert drifted_result.disposition == ValidationDisposition.INVALID
    assert "workload.mab65.selector-grouping.v1" in drifted_result.failed_rule_ids


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_mab_profile_binds_complete_dataset_source_identity() -> None:
    from oamb.workloads.memoryagentbench import build_mab65_manifest

    full = build_mab65_manifest(DATASET_ROOT)
    original_dataset = full.dataset_manifest
    original_file = original_dataset.source_files[0]
    baseline = validate_catalog_profile("oamb-t8-workload-mab65-v1", full)
    drifted_datasets = (
        original_dataset.model_copy(update={"split": "changed"}),
        original_dataset.model_copy(update={"payload_policy": "changed"}),
        original_dataset.model_copy(update={"source_files": ()}),
        original_dataset.model_copy(
            update={
                "source_files": (
                    original_file.model_copy(update={"relative_path": "changed.parquet"}),
                    *original_dataset.source_files[1:],
                )
            }
        ),
        original_dataset.model_copy(
            update={
                "source_files": (
                    original_file.model_copy(update={"byte_count": 1}),
                    *original_dataset.source_files[1:],
                )
            }
        ),
        original_dataset.model_copy(
            update={
                "source_files": (
                    original_file.model_copy(update={"license_id": "MIT"}),
                    *original_dataset.source_files[1:],
                )
            }
        ),
    )

    for dataset in drifted_datasets:
        result = validate_catalog_profile(
            "oamb-t8-workload-mab65-v1",
            replace(full, dataset_manifest=dataset),
        )
        assert result.disposition == ValidationDisposition.INVALID
        assert "workload.mab.source-manifest.v1" in result.failed_rule_ids
        assert result.target_hash != baseline.target_hash


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_mab_profile_rejects_case_source_provenance_tamper() -> None:
    from oamb.workloads.memoryagentbench import build_mab65_manifest

    full = build_mab65_manifest(DATASET_ROOT)
    baseline = validate_catalog_profile("oamb-t8-workload-mab65-v1", full)
    tampered_cases = (
        replace(
            full.cases[0],
            source_relative_path="data/Test_Time_Learning-00000-of-00001.parquet",
        ),
        replace(full.cases[0], source_row_number_1_indexed=2),
    )

    for tampered_case in tampered_cases:
        tampered = replace(full, cases=(tampered_case, *full.cases[1:]))
        result = validate_catalog_profile("oamb-t8-workload-mab65-v1", tampered)

        assert result.target_hash != baseline.target_hash
        assert result.disposition == ValidationDisposition.INVALID
        assert "workload.mab.source-manifest.v1" in result.failed_rule_ids


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_mab_profile_rejects_reference_payload_tamper_with_synchronized_hash() -> None:
    from oamb.workloads.memoryagentbench import build_mab65_manifest

    full = build_mab65_manifest(DATASET_ROOT)
    baseline = validate_catalog_profile("oamb-t8-workload-mab65-v1", full)
    attacker_payload = b'["attacker-gold"]'
    original_case = full.cases[0]
    tampered_case_plan = replace(
        original_case.case_plan,
        reference_payload=attacker_payload,
        reference_payload_sha256=hashlib.sha256(attacker_payload).hexdigest(),
    )
    tampered = replace(
        full,
        cases=(replace(original_case, case_plan=tampered_case_plan), *full.cases[1:]),
    )

    result = validate_catalog_profile("oamb-t8-workload-mab65-v1", tampered)

    assert result.target_hash != baseline.target_hash
    assert result.disposition == ValidationDisposition.INVALID
    assert "workload.mab.prompt-answer-parser.v1" in result.failed_rule_ids


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_mab_profile_rejects_runtime_source_payload_tamper() -> None:
    from oamb.workloads.memoryagentbench import build_mab65_manifest

    full = build_mab65_manifest(DATASET_ROOT)
    baseline = validate_catalog_profile("oamb-t8-workload-mab65-v1", full)
    original_plan = full.plans[0]
    original_unit = original_plan.runtime_plan.ordered_source_units[0]
    tampered_runtime_plan = replace(
        original_plan.runtime_plan,
        ordered_source_units=(
            replace(original_unit, payload_bytes=b"attacker source payload"),
            *original_plan.runtime_plan.ordered_source_units[1:],
        ),
    )
    tampered = replace(
        full,
        plans=(replace(original_plan, runtime_plan=tampered_runtime_plan), *full.plans[1:]),
    )

    result = validate_catalog_profile("oamb-t8-workload-mab65-v1", tampered)

    assert result.target_hash != baseline.target_hash
    assert result.disposition == ValidationDisposition.INVALID
    assert "workload.mab.source-manifest.v1" in result.failed_rule_ids


@pytest.mark.skipif(not DATASET_ROOT.is_dir(), reason="pinned MemoryAgentBench data not downloaded")
def test_mab_profile_rejects_catalog_and_unicode_identity_tamper() -> None:
    from oamb.workloads.memoryagentbench import build_mab65_manifest

    full = build_mab65_manifest(DATASET_ROOT)
    baseline = validate_catalog_profile("oamb-t8-workload-mab65-v1", full)
    first_movie = full.movie_catalog.movies[0]
    drifted_catalog = replace(
        full.movie_catalog,
        movies=(
            replace(first_movie, normalized_title="attacker-normalization"),
            *full.movie_catalog.movies[1:],
        ),
    )
    tampered_bundles = (
        replace(full, entity_catalog_sha256="f" * 64),
        replace(full, unicode_fingerprint="f" * 64),
        replace(full, movie_catalog=drifted_catalog),
    )

    for tampered in tampered_bundles:
        result = validate_catalog_profile("oamb-t8-workload-mab65-v1", tampered)

        assert result.target_hash != baseline.target_hash
        assert result.disposition == ValidationDisposition.INVALID
        assert "workload.mab.source-manifest.v1" in result.failed_rule_ids
