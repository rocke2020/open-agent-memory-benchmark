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
    IngestionPlanRecordV3,
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
        for name in (
            "IngestionPlanRecordV2",
            "IngestionPlanRecordV3",
            "CaseRecordV3",
        )
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
            retrieval_request_raw_ref=SHA_C,
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


def _plan_v3(evidence: ModuleType, projected: tuple[str, ...]) -> IngestionPlanRecordV3:
    v2 = _plan(evidence)
    values = v2.model_dump(mode="python")
    values.update(
        schema_version=3,
        memory_system_id="mem0",
        adapter_profile_id="mem0-rest-v1",
        projected_source_unit_ids=projected,
        projection_semantics="retrieval_visible_subset",
    )
    return cast(IngestionPlanRecordV3, evidence.IngestionPlanRecordV3(**values))


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


@pytest.mark.parametrize("projected", ((), (SHA_B,), (SHA_B, SHA_C)))
def test_mem0_v3_separates_completed_sources_from_visible_projection(
    projected: tuple[str, ...],
) -> None:
    evidence = _evidence()
    plan = _plan_v3(evidence, projected)

    assert plan.accepted_source_unit_ids == (SHA_B, SHA_C)
    assert plan.projected_source_unit_ids == projected
    assert plan.projection_semantics == "retrieval_visible_subset"


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"projected_source_unit_ids": (SHA_C, SHA_B)}, "source order"),
        ({"projected_source_unit_ids": (SHA_D,)}, "accepted"),
        ({"adapter_profile_id": "hindsight-rest-v0.9-v1"}, "adapter_profile_id"),
        ({"projection_semantics": "accepted_exact"}, "projection_semantics"),
    ),
)
def test_mem0_v3_rejects_invalid_subset_or_writer(
    overrides: dict[str, object],
    message: str,
) -> None:
    evidence = _evidence()
    plan = _plan_v3(evidence, (SHA_B,))

    with pytest.raises(ValidationError, match=message):
        evidence.IngestionPlanRecordV3(
            **{
                **plan.model_dump(mode="python"),
                **overrides,
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
