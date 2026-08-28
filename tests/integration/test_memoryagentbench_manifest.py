from __future__ import annotations

import json
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

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
        for relative_path, sha256, split in mab._PINNED_FILES
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
