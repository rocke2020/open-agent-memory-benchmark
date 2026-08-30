"""Local human confirmation and phase-gate reduction."""

from __future__ import annotations

from datetime import datetime

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    EvaluationPhaseGate,
    HumanQualityReviewRecord,
    QualityReviewStatus,
    evaluation_phase_gate_id,
    human_quality_review_record_id,
)


class HumanReviewConfirmationError(ValueError):
    """The local human confirmation cannot enter canonical review history."""


class ReviewHistoryError(ValueError):
    """The append-only AI/human review chain is incomplete or forked."""


def create_human_quality_review_record(
    *,
    canonical_ai_record: AIQualityReviewRecord,
    status: str,
    nonce: str,
    operator_id: str,
    finding_codes: tuple[str, ...],
    evidence_references: tuple[str, ...],
    used_nonces: tuple[str, ...],
    created_at: datetime,
) -> HumanQualityReviewRecord:
    if canonical_ai_record.status != QualityReviewStatus.PASS:
        raise HumanReviewConfirmationError("human review requires the canonical AI PASS record")
    if nonce in set(used_nonces):
        raise HumanReviewConfirmationError("human review confirmation nonce was already used")
    record_fields = {
        "review_bundle_hash": canonical_ai_record.review_bundle_hash,
        "ai_review_record_hash": canonical_ai_record.ai_review_record_id,
        "status": status,
        "operator_id": operator_id,
        "confirmation_method": "interactive_exact_hash_phrase_v1",
        "confirmation_nonce": nonce,
        "finding_codes": finding_codes,
        "evidence_references": evidence_references,
        "created_at": created_at,
    }
    return HumanQualityReviewRecord.model_validate(
        {
            "human_review_record_id": human_quality_review_record_id(**record_fields),
            **record_fields,
        }
    )


def derive_evaluation_phase_gate(
    *,
    phase_id: str,
    review_bundle_hash: str,
    ordered_ai_history: tuple[AIQualityReviewRecord, ...],
    human_records: tuple[HumanQualityReviewRecord, ...],
) -> EvaluationPhaseGate:
    validate_ai_review_history(review_bundle_hash, ordered_ai_history)
    if len(human_records) > 1:
        raise ReviewHistoryError("review history contains more than one human review")
    ai_head = ordered_ai_history[-1]
    human_record = human_records[0] if human_records else None
    if human_record is not None:
        if ai_head.status != QualityReviewStatus.PASS:
            raise ReviewHistoryError("human review requires the canonical AI PASS head")
        if (
            human_record.review_bundle_hash != review_bundle_hash
            or human_record.ai_review_record_hash != ai_head.ai_review_record_id
        ):
            raise ReviewHistoryError("human review does not bind the canonical AI head")
    ordered_ai_hashes = tuple(item.ai_review_record_id for item in ordered_ai_history)
    human_hash = human_record.human_review_record_id if human_record is not None else None
    history_root = canonical_sha256(["oamb-review-history-v2", ordered_ai_hashes, human_hash])
    passed_by_ai = ai_head.status == QualityReviewStatus.PASS
    passed_by_human = passed_by_ai and human_record is not None and human_record.status == "pass"
    gate_fields = {
        "phase_id": phase_id,
        "review_bundle_hash": review_bundle_hash,
        "review_history_root_hash": history_root,
        "ordered_ai_review_record_hashes": ordered_ai_hashes,
        "canonical_ai_review_record_hash": ai_head.ai_review_record_id,
        "human_review_record_hash": human_hash,
        "passed_by_ai": passed_by_ai,
        "passed_by_human": passed_by_human,
    }
    return EvaluationPhaseGate.model_validate(
        {
            "gate_id": evaluation_phase_gate_id(**gate_fields),
            "ai_record": ai_head,
            "human_record": human_record,
            **gate_fields,
        }
    )


def validate_ai_review_history(
    review_bundle_hash: str,
    records: tuple[AIQualityReviewRecord, ...],
) -> None:
    if not records:
        raise ReviewHistoryError("review history requires an AI record")
    if len({item.ai_review_record_id for item in records}) != len(records):
        raise ReviewHistoryError("AI review history contains a duplicate or fork")
    for index, record in enumerate(records):
        try:
            validated_record = AIQualityReviewRecord.model_validate(
                record.model_dump(mode="python")
            )
        except Exception as exc:
            raise ReviewHistoryError("AI review history contains an invalid record") from exc
        if validated_record != record:
            raise ReviewHistoryError("AI review history contains a non-canonical record")
        expected_ordinal = index + 1
        if record.review_bundle_hash != review_bundle_hash or record.ordinal != expected_ordinal:
            raise ReviewHistoryError("AI review history has a bundle mismatch or ordinal gap")
        if index == 0:
            if (
                record.previous_ai_review_record_hash is not None
                or record.previous_history_root_hash is not None
            ):
                raise ReviewHistoryError("first AI review record names a predecessor")
            continue
        previous = records[index - 1]
        if (
            record.previous_ai_review_record_hash != previous.ai_review_record_id
            or record.previous_history_root_hash != previous.history_root_hash
        ):
            raise ReviewHistoryError("AI review history contains a fork or omitted predecessor")
        if (
            previous.status != QualityReviewStatus.INCONCLUSIVE
            or previous.review_outcome_kind != "operational_inconclusive"
        ):
            raise ReviewHistoryError("only operational AI INCONCLUSIVE may be superseded")


__all__ = [
    "HumanReviewConfirmationError",
    "ReviewHistoryError",
    "create_human_quality_review_record",
    "derive_evaluation_phase_gate",
    "validate_ai_review_history",
]
