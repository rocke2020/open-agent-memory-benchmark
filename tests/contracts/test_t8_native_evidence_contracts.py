from __future__ import annotations

import importlib
from datetime import UTC, datetime
from types import ModuleType
from typing import cast

import pytest
from pydantic import ValidationError

from oamb.contracts.evidence import (
    CaseEvaluationDisposition,
    CaseRecordV3,
    IngestionPlanRecordV2,
)
from oamb.contracts.states import CaseState, IngestionPlanState

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
NOW = datetime(2026, 8, 28, tzinfo=UTC)


def _evidence() -> ModuleType:
    module = importlib.import_module("oamb.contracts.evidence")
    missing = tuple(
        name
        for name in ("IngestionPlanRecordV2", "CaseRecordV3", "PhaseReviewOccurrenceRecord")
        if not hasattr(module, name)
    )
    if missing:
        pytest.fail(f"missing native evidence contracts: {', '.join(missing)}", pytrace=False)
    return module


def _plan(evidence: ModuleType) -> IngestionPlanRecordV2:
    return cast(
        IngestionPlanRecordV2,
        evidence.IngestionPlanRecordV2(
            ingestion_occurrence_id=SHA_A,
            run_id="native-run",
            memory_system_id="fixture-native-rest",
            adapter_profile_id="fixture-native-rest-v1",
            runtime_binding_hash=SHA_B,
            ingestion_plan_id=SHA_C,
            ordered_member_context_manifest_entry_ids=(SHA_A,),
            ordered_case_occurrence_ids=(SHA_D,),
            state=IngestionPlanState.SEALED,
            scope_id="fixture-scope",
            scope_raw_refs=(SHA_A,),
            ordered_source_unit_ids=(SHA_B, SHA_C),
            ordered_dispatch_attempt_ids=(SHA_A,),
            ordered_dispatch_source_unit_ids=((SHA_B, SHA_C),),
            accepted_source_unit_ids=(SHA_B, SHA_C),
            rejected_source_unit_ids=(),
            readiness_evidence_refs=(SHA_B,),
            inventory_raw_ref=SHA_C,
            projected_source_unit_ids=(SHA_B, SHA_C),
            projection_raw_refs=(SHA_C, SHA_D),
            protected_state_sha256=SHA_D,
            attempt_ids=(SHA_A,),
            usage_record_ids=(SHA_B,),
            resource_record_ids=(SHA_C,),
            cost_record_ids=(SHA_D,),
        ),
    )


def _case(evidence: ModuleType) -> CaseRecordV3:
    return cast(
        CaseRecordV3,
        evidence.CaseRecordV3(
            case_occurrence_id=SHA_D,
            run_id="native-run",
            ingestion_occurrence_id=SHA_A,
            case_manifest_entry_id=SHA_B,
            adapter_profile_id="fixture-native-rest-v1",
            state=CaseState.COMPLETED,
            retrieval_raw_ref=SHA_A,
            retrieval_supporting_raw_refs=(SHA_B,),
            ordered_native_candidate_ids=("native-1", "native-2"),
            ordered_native_content_sha256=(SHA_C, SHA_D),
            native_candidate_source_unit_ids=(SHA_B, None),
            visible_evidence_raw_ref=SHA_B,
            visible_evidence_sha256=SHA_C,
            visible_evidence_byte_count=128,
            visible_evidence_token_count=32,
            visible_evidence_tokenizer_fingerprint=SHA_D,
            native_candidate_count=2,
            visible_kept_count=1,
            visible_dropped_count=1,
            visible_truncated_count=0,
            visible_decision_ledger_raw_ref=SHA_C,
            pre_query_projection_raw_refs=(SHA_A,),
            pre_query_state_sha256=SHA_D,
            post_query_projection_raw_refs=(SHA_B,),
            post_query_state_sha256=SHA_D,
            query_mutation_status="unchanged",
            prompt_raw_ref=SHA_A,
            prompt_sha256=SHA_A,
            judge_prompt_raw_ref=None,
            answer_raw_ref=SHA_B,
            parsed_answer_sha256=SHA_C,
            metric_id="fixture-metric-v1",
            metric_numerator=1,
            metric_denominator=1,
            evaluation_raw_ref=SHA_D,
            evaluation_disposition=CaseEvaluationDisposition.DETERMINISTIC_EVALUATED,
            attempt_ids=(SHA_A, SHA_B),
            usage_record_ids=(SHA_C,),
            resource_record_ids=(SHA_D,),
            cost_record_ids=(),
            error_stage=None,
        ),
    )


def test_native_ingestion_record_closes_scope_dispatch_and_projection() -> None:
    evidence = _evidence()
    plan = _plan(evidence)

    assert plan.accepted_source_unit_ids == plan.ordered_source_unit_ids
    with pytest.raises(ValidationError, match="projection"):
        evidence.IngestionPlanRecordV2(
            **{
                **plan.model_dump(mode="python"),
                "projection_raw_refs": (),
            }
        )
    with pytest.raises(ValidationError, match="partition"):
        evidence.IngestionPlanRecordV2(
            **{
                **plan.model_dump(mode="python"),
                "accepted_source_unit_ids": (SHA_B,),
            }
        )
    with pytest.raises(ValidationError, match="dispatch source ledger"):
        evidence.IngestionPlanRecordV2(
            **{
                **plan.model_dump(mode="python"),
                "ordered_dispatch_source_unit_ids": ((SHA_C, SHA_B),),
            }
        )


def test_completed_native_case_closes_ordered_retrieval_context_and_metric() -> None:
    evidence = _evidence()
    case = _case(evidence)

    assert case.query_mutation_status == "unchanged"
    assert case.metric_numerator == case.metric_denominator == 1
    with pytest.raises(ValidationError, match="aligned"):
        evidence.CaseRecordV3(
            **{
                **case.model_dump(mode="python"),
                "ordered_native_content_sha256": (SHA_C,),
            }
        )
    with pytest.raises(ValidationError, match="read-only"):
        evidence.CaseRecordV3(
            **{
                **case.model_dump(mode="python"),
                "post_query_state_sha256": SHA_A,
                "query_mutation_status": "changed",
            }
        )
    with pytest.raises(ValidationError, match="metric fraction"):
        evidence.CaseRecordV3(
            **{
                **case.model_dump(mode="python"),
                "metric_numerator": 2,
            }
        )
    with pytest.raises(ValidationError, match="prompt evidence"):
        evidence.CaseRecordV3(
            **{
                **case.model_dump(mode="python"),
                "prompt_raw_ref": None,
            }
        )


def test_phase_review_occurrence_is_not_parented_by_a_benchmark_run() -> None:
    evidence = _evidence()
    occurrence_id = evidence.phase_review_occurrence_id(
        phase_id="fixture_phase",
        review_bundle_hash=SHA_B,
        reviewer_role_binding_hash=SHA_C,
        ordinal=1,
    )
    occurrence = evidence.PhaseReviewOccurrenceRecord(
        phase_review_occurrence_id=occurrence_id,
        phase_id="fixture_phase",
        review_bundle_hash=SHA_B,
        reviewer_role_binding_hash=SHA_C,
        ordinal=1,
        approval_record_id=SHA_D,
        budget_id="phase-review-budget",
        state="sealed",
        started_at=NOW,
        ended_at=NOW,
    )

    assert not hasattr(occurrence, "run_id")
