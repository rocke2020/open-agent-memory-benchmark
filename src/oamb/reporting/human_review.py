"""Public-key-only human decision preparation, import, and phase-gate reduction."""

from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from datetime import datetime

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    EvaluationPhaseGate,
    HumanQualityReviewRecord,
    HumanReviewDecision,
    QualityReviewStatus,
    SignatureVerificationRecord,
    evaluation_phase_gate_id,
    human_quality_review_record_id,
    human_review_decision_id,
    signature_verification_record_id,
)
from oamb.contracts.specifications import (
    HumanReviewKeyBinding,
    human_review_key_binding_id,
)


class HumanReviewImportError(ValueError):
    """The externally signed decision cannot enter canonical review history."""


class ReviewHistoryError(ValueError):
    """The append-only AI/human review chain is incomplete or forked."""


@dataclass(frozen=True, slots=True)
class PreparedHumanReviewDecision:
    decision: HumanReviewDecision
    canonical_bytes: bytes


def build_human_review_key_binding(
    *,
    source_kind: str,
    source_reference: str,
    public_key_base64: str,
) -> HumanReviewKeyBinding:
    try:
        public_key = base64.b64decode(public_key_base64, validate=True)
    except Exception as exc:
        raise ValueError("human review public key is not canonical base64") from exc
    public_key_sha256 = hashlib.sha256(public_key).hexdigest()
    fingerprint = f"SHA256:{public_key_sha256}"
    fields = {
        "source_kind": source_kind,
        "source_reference": source_reference,
        "public_key_base64": public_key_base64,
        "public_key_sha256": public_key_sha256,
        "fingerprint": fingerprint,
    }
    return HumanReviewKeyBinding.model_validate(
        {
            "key_binding_id": human_review_key_binding_id(**fields),
            "algorithm": "ed25519",
            **fields,
        }
    )


def prepare_human_review_decision(
    *,
    review_bundle_hash: str,
    ai_review_record_hash: str,
    status: str,
    decided_at: datetime,
    nonce: str,
    reviewer_label: str,
    finding_codes: tuple[str, ...],
    evidence_references: tuple[str, ...],
) -> PreparedHumanReviewDecision:
    fields = {
        "review_bundle_hash": review_bundle_hash,
        "ai_review_record_hash": ai_review_record_hash,
        "status": status,
        "decided_at": decided_at,
        "nonce": nonce,
        "reviewer_label": reviewer_label,
        "finding_codes": finding_codes,
        "evidence_references": evidence_references,
    }
    decision = HumanReviewDecision.model_validate(
        {
            "decision_id": human_review_decision_id(**fields),
            **fields,
        }
    )
    return PreparedHumanReviewDecision(
        decision=decision,
        canonical_bytes=canonical_json_bytes(decision),
    )


def import_human_review_decision(
    *,
    decision_bytes: bytes,
    signature_base64: str,
    key_binding: HumanReviewKeyBinding,
    canonical_ai_record: AIQualityReviewRecord,
    used_nonces: tuple[str, ...],
    imported_at: datetime,
) -> HumanQualityReviewRecord:
    if canonical_ai_record.status != QualityReviewStatus.PASS:
        raise HumanReviewImportError("human review requires the canonical AI PASS record")
    try:
        decision = HumanReviewDecision.model_validate_json(decision_bytes)
    except Exception as exc:
        raise HumanReviewImportError("human review decision contract is invalid") from exc
    if canonical_json_bytes(decision) != decision_bytes:
        raise HumanReviewImportError("human review decision bytes are not canonical")
    if (
        decision.review_bundle_hash != canonical_ai_record.review_bundle_hash
        or decision.ai_review_record_hash != canonical_ai_record.ai_review_record_id
    ):
        raise HumanReviewImportError("human decision does not bind the canonical review inputs")
    if decision.nonce in set(used_nonces):
        raise HumanReviewImportError("human review decision nonce was already used")
    public_key_bytes = _canonical_base64(key_binding.public_key_base64, label="public key")
    signature = _canonical_base64(signature_base64, label="signature")
    if len(signature) != 64:
        raise HumanReviewImportError("human review signature must contain exactly 64 bytes")
    try:
        Ed25519PublicKey.from_public_bytes(public_key_bytes).verify(signature, decision_bytes)
    except (InvalidSignature, ValueError) as exc:
        raise HumanReviewImportError("human review signature verification failed") from exc

    verification_fields = {
        "key_binding_id": key_binding.key_binding_id,
        "trusted_key_fingerprint": key_binding.fingerprint,
        "public_key_sha256": key_binding.public_key_sha256,
        "signed_payload_sha256": hashlib.sha256(decision_bytes).hexdigest(),
        "signature_sha256": hashlib.sha256(signature).hexdigest(),
        "verified_at": imported_at,
    }
    verification = SignatureVerificationRecord.model_validate(
        {
            "verification_id": signature_verification_record_id(**verification_fields),
            "algorithm": "ed25519",
            **verification_fields,
        }
    )
    record_fields = {
        "review_bundle_hash": decision.review_bundle_hash,
        "ai_review_record_hash": decision.ai_review_record_hash,
        "decision_hash": hashlib.sha256(decision_bytes).hexdigest(),
        "signature_verification": verification,
        "status": decision.status,
        "finding_codes": decision.finding_codes,
        "evidence_references": decision.evidence_references,
        "reviewer_label": decision.reviewer_label,
        "decision_nonce": decision.nonce,
        "trusted_key_fingerprint": key_binding.fingerprint,
        "created_at": imported_at,
    }
    return HumanQualityReviewRecord.model_validate(
        {
            "human_review_record_id": human_quality_review_record_id(**record_fields),
            **record_fields,
        }
    )


def verify_human_review_record_evidence(
    *,
    record: HumanQualityReviewRecord,
    decision_bytes: bytes,
    signature_base64: str,
    key_binding: HumanReviewKeyBinding,
    canonical_ai_record: AIQualityReviewRecord,
) -> bool:
    try:
        rebuilt = import_human_review_decision(
            decision_bytes=decision_bytes,
            signature_base64=signature_base64,
            key_binding=key_binding,
            canonical_ai_record=canonical_ai_record,
            used_nonces=(),
            imported_at=record.created_at,
        )
    except HumanReviewImportError:
        return False
    return rebuilt == record


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
    history_root = canonical_sha256(["oamb-review-history-v1", ordered_ai_hashes, human_hash])
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


def _canonical_base64(value: str, *, label: str) -> bytes:
    try:
        decoded = base64.b64decode(value, validate=True)
    except Exception as exc:
        raise HumanReviewImportError(f"human review {label} is not canonical base64") from exc
    if base64.b64encode(decoded).decode("ascii") != value:
        raise HumanReviewImportError(f"human review {label} is not canonical base64")
    return decoded


__all__ = [
    "HumanReviewImportError",
    "PreparedHumanReviewDecision",
    "ReviewHistoryError",
    "build_human_review_key_binding",
    "derive_evaluation_phase_gate",
    "import_human_review_decision",
    "prepare_human_review_decision",
    "validate_ai_review_history",
    "verify_human_review_record_evidence",
]
