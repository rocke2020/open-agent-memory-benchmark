from __future__ import annotations

from datetime import UTC, datetime

import pytest

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    HumanQualityReviewRecord,
    QualityReviewStatus,
    ai_quality_review_record_identity,
)
from oamb.reporting.human_review import (
    HumanReviewConfirmationError,
    ReviewHistoryError,
    create_human_quality_review_record,
    derive_evaluation_phase_gate,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
AI_REVIEW_RECORD_ID = "38a29206e3d7c9b66e6bb15be1f245b6fe39401e03076c072f3dae75d7c78c80"
CREATED_AT = datetime(2026, 8, 28, tzinfo=UTC)


def _ai_record(
    status: QualityReviewStatus = QualityReviewStatus.PASS,
) -> AIQualityReviewRecord:
    outcome_kind = "complete" if status == QualityReviewStatus.PASS else "semantic_failure"
    finding_codes = () if status == QualityReviewStatus.PASS else ("REPORT_CLAIM_DEFECT",)
    fields = {
        "review_bundle_hash": SHA_B,
        "review_plan_hash": SHA_C,
        "projection_spec_hash": SHA_A,
        "finding_registry_hash": SHA_B,
        "prompt_pack_hash": SHA_C,
        "output_contract_hash": SHA_D,
        "parser_hash": SHA_A,
        "aggregate_version": "ai-quality-aggregate-v1",
        "reviewer_role_binding_hash": SHA_B,
        "reviewer_model_hash": SHA_C,
        "reviewer_runtime_hash": SHA_D,
        "reviewer_configuration_hash": SHA_A,
        "occurrence_id": SHA_D,
        "ordinal": 1,
        "previous_ai_review_record_hash": None,
        "previous_history_root_hash": None,
        "case_coverage_hash": SHA_C,
        "batch_result_hashes": (SHA_A,),
        "integrity_result_hash": SHA_B,
        "status": status,
        "review_outcome_kind": outcome_kind,
        "finding_codes": finding_codes,
        "attempt_ids": (SHA_D,),
        "usage_record_ids": (SHA_A,),
        "resource_record_ids": (SHA_B,),
        "cost_record_ids": (SHA_C,),
        "accounting_closed": True,
        "created_at": CREATED_AT,
    }
    record_id = ai_quality_review_record_identity(fields)
    record = AIQualityReviewRecord.model_validate(
        {
            "ai_review_record_id": record_id,
            "history_root_hash": canonical_sha256(["oamb-ai-review-history-v1", None, record_id]),
            **fields,
        }
    )
    if status == QualityReviewStatus.PASS:
        assert record.ai_review_record_id == AI_REVIEW_RECORD_ID
    return record


def test_local_human_confirmation_creates_v2_record_and_v2_gate_without_a_key() -> None:
    ai_record = _ai_record()
    record = create_human_quality_review_record(
        canonical_ai_record=ai_record,
        status="pass",
        nonce="nonce-001",
        operator_id="operator",
        finding_codes=(),
        evidence_references=(),
        used_nonces=(),
        created_at=CREATED_AT,
    )

    assert isinstance(record, HumanQualityReviewRecord)
    assert record.schema_version == 2
    assert record.review_bundle_hash == SHA_B
    assert record.ai_review_record_hash == AI_REVIEW_RECORD_ID
    assert record.status == "pass"
    assert record.operator_id == "operator"
    assert record.confirmation_nonce == "nonce-001"
    assert record.confirmation_method == "interactive_exact_hash_phrase_v1"
    assert "signature" not in record.model_dump(mode="json")
    assert "key" not in record.model_dump(mode="json")

    gate = derive_evaluation_phase_gate(
        phase_id="fixture_phase",
        review_bundle_hash=SHA_B,
        ordered_ai_history=(ai_record,),
        human_records=(record,),
    )

    assert gate.schema_version == 2
    assert gate.passed_by_ai is True
    assert gate.passed_by_human is True
    assert gate.human_record == record


def test_local_human_confirmation_rejects_replayed_nonce_and_nonpassing_ai() -> None:
    ai_record = _ai_record()
    with pytest.raises(HumanReviewConfirmationError, match="nonce"):
        create_human_quality_review_record(
            canonical_ai_record=ai_record,
            status="pass",
            nonce="nonce-001",
            operator_id="operator",
            finding_codes=(),
            evidence_references=(),
            used_nonces=("nonce-001",),
            created_at=CREATED_AT,
        )
    with pytest.raises(HumanReviewConfirmationError, match="AI PASS"):
        create_human_quality_review_record(
            canonical_ai_record=_ai_record(QualityReviewStatus.FAIL),
            status="fail",
            nonce="nonce-002",
            operator_id="operator",
            finding_codes=("REPORT_CLAIM_DEFECT",),
            evidence_references=(),
            used_nonces=(),
            created_at=CREATED_AT,
        )


def test_phase_gate_hashes_complete_ai_then_human_history_and_rejects_duplicates() -> None:
    ai_record = _ai_record()
    human_record = create_human_quality_review_record(
        canonical_ai_record=ai_record,
        status="pass",
        nonce="nonce-001",
        operator_id="operator",
        finding_codes=(),
        evidence_references=(),
        used_nonces=(),
        created_at=CREATED_AT,
    )

    gate = derive_evaluation_phase_gate(
        phase_id="fixture_phase",
        review_bundle_hash=SHA_B,
        ordered_ai_history=(ai_record,),
        human_records=(human_record,),
    )

    assert gate.passed_by_ai is True
    assert gate.passed_by_human is True
    assert gate.review_history_root_hash != ai_record.history_root_hash
    with pytest.raises(ReviewHistoryError, match="human review"):
        derive_evaluation_phase_gate(
            phase_id="fixture_phase",
            review_bundle_hash=SHA_B,
            ordered_ai_history=(ai_record,),
            human_records=(human_record, human_record),
        )
