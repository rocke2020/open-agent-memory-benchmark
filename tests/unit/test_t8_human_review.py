from __future__ import annotations

import importlib
from datetime import UTC, datetime
from types import ModuleType

import pytest

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    QualityReviewStatus,
    ai_quality_review_record_identity,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
AI_REVIEW_RECORD_ID = "38a29206e3d7c9b66e6bb15be1f245b6fe39401e03076c072f3dae75d7c78c80"
PUBLIC_KEY_BASE64 = "C8aRhmD01ZtA8t71pqhqWswBAaFCabPNqUETOcDkY/g="
SIGNATURE_BASE64 = (
    "mMNlMSX1DwgRjPpxCiCITURhVSOeJO5DuiZnkn+EPx8bJPjSEWJ1OL5xSTm6g/AT5IKNSLv12VkdHR+uvM2CCA=="
)
EXPECTED_DECISION_JSON = (
    '{"ai_review_record_hash":"38a29206e3d7c9b66e6bb15be1f245b6fe39401e03076c072f3dae75d7c78c80",'
    '"decided_at":"2026-08-28T01:00:00+00:00",'
    '"decision_id":"4ff51f94e4462fc02bff3ccd5fc8fa8991b3c51cefd2164292534db235ef1316",'
    '"evidence_references":[],"finding_codes":[],"nonce":"nonce-001",'
    '"review_bundle_hash":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",'
    '"reviewer_label":"operator","schema_name":"human_review_decision","schema_version":1,'
    '"status":"pass"}'
)
CREATED_AT = datetime(2026, 8, 28, tzinfo=UTC)
DECIDED_AT = datetime(2026, 8, 28, 1, tzinfo=UTC)


def _human_review() -> ModuleType:
    try:
        return importlib.import_module("oamb.reporting.human_review")
    except ModuleNotFoundError:
        pytest.fail("oamb.reporting.human_review is not implemented", pytrace=False)


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


def test_signed_human_decision_import_uses_only_the_pinned_public_key() -> None:
    human_review = _human_review()
    ai_record = _ai_record()
    key_binding = human_review.build_human_review_key_binding(
        source_kind="raw",
        source_reference="test-vector",
        public_key_base64=PUBLIC_KEY_BASE64,
    )
    prepared = human_review.prepare_human_review_decision(
        review_bundle_hash=SHA_B,
        ai_review_record_hash=ai_record.ai_review_record_id,
        status="pass",
        decided_at=DECIDED_AT,
        nonce="nonce-001",
        reviewer_label="operator",
        finding_codes=(),
        evidence_references=(),
    )

    assert prepared.canonical_bytes.decode("utf-8") == EXPECTED_DECISION_JSON
    record = human_review.import_human_review_decision(
        decision_bytes=prepared.canonical_bytes,
        signature_base64=SIGNATURE_BASE64,
        key_binding=key_binding,
        canonical_ai_record=ai_record,
        used_nonces=(),
        imported_at=datetime(2026, 8, 28, 2, tzinfo=UTC),
    )

    assert record.status == "pass"
    assert record.decision_nonce == "nonce-001"
    assert record.trusted_key_fingerprint == key_binding.fingerprint
    assert record.signature_verification.algorithm == "ed25519"


def test_human_import_rejects_bad_signature_replay_and_nonpassing_ai() -> None:
    human_review = _human_review()
    ai_record = _ai_record()
    key_binding = human_review.build_human_review_key_binding(
        source_kind="raw",
        source_reference="test-vector",
        public_key_base64=PUBLIC_KEY_BASE64,
    )
    prepared = human_review.prepare_human_review_decision(
        review_bundle_hash=SHA_B,
        ai_review_record_hash=ai_record.ai_review_record_id,
        status="pass",
        decided_at=DECIDED_AT,
        nonce="nonce-001",
        reviewer_label="operator",
        finding_codes=(),
        evidence_references=(),
    )

    with pytest.raises(human_review.HumanReviewImportError, match="signature"):
        human_review.import_human_review_decision(
            decision_bytes=prepared.canonical_bytes,
            signature_base64=SIGNATURE_BASE64[:-2] + "AA",
            key_binding=key_binding,
            canonical_ai_record=ai_record,
            used_nonces=(),
            imported_at=datetime(2026, 8, 28, 2, tzinfo=UTC),
        )
    with pytest.raises(human_review.HumanReviewImportError, match="nonce"):
        human_review.import_human_review_decision(
            decision_bytes=prepared.canonical_bytes,
            signature_base64=SIGNATURE_BASE64,
            key_binding=key_binding,
            canonical_ai_record=ai_record,
            used_nonces=("nonce-001",),
            imported_at=datetime(2026, 8, 28, 2, tzinfo=UTC),
        )
    with pytest.raises(human_review.HumanReviewImportError, match="AI PASS"):
        human_review.import_human_review_decision(
            decision_bytes=prepared.canonical_bytes,
            signature_base64=SIGNATURE_BASE64,
            key_binding=key_binding,
            canonical_ai_record=_ai_record(QualityReviewStatus.FAIL),
            used_nonces=(),
            imported_at=datetime(2026, 8, 28, 2, tzinfo=UTC),
        )


def test_phase_gate_hashes_complete_ai_then_human_history_and_rejects_duplicates() -> None:
    human_review = _human_review()
    ai_record = _ai_record()
    key_binding = human_review.build_human_review_key_binding(
        source_kind="raw",
        source_reference="test-vector",
        public_key_base64=PUBLIC_KEY_BASE64,
    )
    prepared = human_review.prepare_human_review_decision(
        review_bundle_hash=SHA_B,
        ai_review_record_hash=ai_record.ai_review_record_id,
        status="pass",
        decided_at=DECIDED_AT,
        nonce="nonce-001",
        reviewer_label="operator",
        finding_codes=(),
        evidence_references=(),
    )
    human_record = human_review.import_human_review_decision(
        decision_bytes=prepared.canonical_bytes,
        signature_base64=SIGNATURE_BASE64,
        key_binding=key_binding,
        canonical_ai_record=ai_record,
        used_nonces=(),
        imported_at=datetime(2026, 8, 28, 2, tzinfo=UTC),
    )

    gate = human_review.derive_evaluation_phase_gate(
        phase_id="fixture_phase",
        review_bundle_hash=SHA_B,
        ordered_ai_history=(ai_record,),
        human_records=(human_record,),
    )

    assert gate.passed_by_ai is True
    assert gate.passed_by_human is True
    assert gate.review_history_root_hash != ai_record.history_root_hash
    with pytest.raises(human_review.ReviewHistoryError, match="human review"):
        human_review.derive_evaluation_phase_gate(
            phase_id="fixture_phase",
            review_bundle_hash=SHA_B,
            ordered_ai_history=(ai_record,),
            human_records=(human_record, human_record),
        )
