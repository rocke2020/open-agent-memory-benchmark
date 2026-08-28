"""Provider-order visible-evidence normalization and exact context counting."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from functools import cache
from importlib.metadata import version

import tiktoken

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import (
    EvidenceDecision,
    NativeEvidenceBatch,
    NativeEvidenceCandidate,
    VisibleEvidence,
    VisibleEvidencePolicy,
)

TOKENIZER_PACKAGE = "tiktoken"
TOKENIZER_VERSION = "0.14.0"
TOKENIZER_ENCODING = "o200k_base"
LME_VISIBLE_EVIDENCE_POLICY = VisibleEvidencePolicy(
    max_items=256,
    max_characters=262_144,
    max_tokens=32_768,
)


class EvidenceConflictError(ValueError):
    pass


@cache
def o200k_encoding() -> tiktoken.Encoding:
    installed_version = version(TOKENIZER_PACKAGE)
    if installed_version != TOKENIZER_VERSION:
        raise RuntimeError(
            f"tiktoken version drift: expected {TOKENIZER_VERSION}, got {installed_version}"
        )
    return tiktoken.get_encoding(TOKENIZER_ENCODING)


@cache
def tokenizer_fingerprint() -> str:
    encoding = o200k_encoding()
    mergeable_ranks = encoding._mergeable_ranks
    special_tokens = encoding._special_tokens
    asset_hasher = hashlib.sha256()
    for token_bytes, rank in sorted(mergeable_ranks.items(), key=lambda item: item[0]):
        asset_hasher.update(len(token_bytes).to_bytes(4, "big"))
        asset_hasher.update(token_bytes)
        asset_hasher.update(rank.to_bytes(4, "big"))
    for token_text, rank in sorted(special_tokens.items()):
        encoded = token_text.encode("utf-8")
        asset_hasher.update(len(encoded).to_bytes(4, "big"))
        asset_hasher.update(encoded)
        asset_hasher.update(rank.to_bytes(4, "big"))
    return canonical_sha256(
        [
            "oamb-tokenizer-fingerprint-v1",
            TOKENIZER_PACKAGE,
            TOKENIZER_VERSION,
            TOKENIZER_ENCODING,
            "encode_ordinary",
            asset_hasher.hexdigest(),
        ]
    )


def count_o200k_tokens(payload: bytes) -> int:
    text = payload.decode("utf-8", errors="strict")
    return len(o200k_encoding().encode_ordinary(text))


def _canonical_timestamp(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid evidence timestamp: {value}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("evidence timestamp requires an explicit offset")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def _identity(candidate: NativeEvidenceCandidate) -> str:
    identity = candidate.provider_evidence_identity
    if not identity:
        raise ValueError("provider evidence identity must not be empty")
    return identity


def verify_visible_evidence_hash(evidence: VisibleEvidence) -> None:
    if hashlib.sha256(evidence.canonical_bytes).hexdigest() != evidence.sha256:
        raise ValueError("visible evidence hash does not match its exact bytes")


def _item_bytes(candidate: NativeEvidenceCandidate, identity: str) -> bytes:
    if not candidate.evidence_kind or not candidate.content:
        raise ValueError("visible evidence requires non-empty kind and text")
    item = {
        "provider_evidence_identity": identity,
        "source_unit_id": candidate.source_unit_id,
        "evidence_kind": candidate.evidence_kind,
        "text": candidate.content,
        "occurred_start": _canonical_timestamp(candidate.occurred_start),
        "occurred_end": _canonical_timestamp(candidate.occurred_end),
        "mentioned_at": _canonical_timestamp(candidate.mentioned_at),
    }
    return json.dumps(
        item,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def build_lme_visible_evidence(
    native_batch: NativeEvidenceBatch,
    policy: VisibleEvidencePolicy = LME_VISIBLE_EVIDENCE_POLICY,
) -> VisibleEvidence:
    """Keep complete JSONL items until the first frozen ceiling is exceeded."""

    seen_content: dict[str, str] = {}
    lines: list[bytes] = []
    included_ids: list[str] = []
    decisions: list[EvidenceDecision] = []
    stopped = False
    first_exceeded: str | None = None

    for expected_rank, candidate in enumerate(native_batch.candidates, start=1):
        if candidate.native_rank_1_indexed != expected_rank:
            raise ValueError("native evidence ranks must preserve provider order")
        if candidate.native_truncated:
            raise ValueError("visible evidence rejects native truncation")
        identity = _identity(candidate)
        text_hash = hashlib.sha256(candidate.content.encode()).hexdigest()
        previous_hash = seen_content.get(identity)
        if previous_hash is not None and previous_hash != text_hash:
            raise EvidenceConflictError(
                f"provider evidence identity has conflicting content: {identity}"
            )
        if previous_hash == text_hash:
            decisions.append(
                EvidenceDecision(
                    native_id=candidate.native_id,
                    provider_evidence_identity=identity,
                    normalized_text_sha256=text_hash,
                    disposition="duplicate",
                    reason="duplicate_identity_and_text",
                )
            )
            continue
        seen_content[identity] = text_hash
        if stopped:
            decisions.append(
                EvidenceDecision(
                    native_id=candidate.native_id,
                    provider_evidence_identity=identity,
                    normalized_text_sha256=text_hash,
                    disposition="budget_dropped",
                    reason=first_exceeded,
                )
            )
            continue
        item = _item_bytes(candidate, identity)
        proposed = b"\n".join((*lines, item))
        exceeded: str | None = None
        if len(lines) + 1 > policy.max_items:
            exceeded = "max_items"
        elif len(proposed.decode("utf-8")) > policy.max_characters:
            exceeded = "max_characters"
        elif count_o200k_tokens(proposed) > policy.max_tokens:
            exceeded = "max_tokens"
        if exceeded is not None:
            stopped = True
            first_exceeded = exceeded
            decisions.append(
                EvidenceDecision(
                    native_id=candidate.native_id,
                    provider_evidence_identity=identity,
                    normalized_text_sha256=text_hash,
                    disposition="budget_dropped",
                    reason=exceeded,
                )
            )
            continue
        lines.append(item)
        included_ids.append(candidate.native_id)
        decisions.append(
            EvidenceDecision(
                native_id=candidate.native_id,
                provider_evidence_identity=identity,
                normalized_text_sha256=text_hash,
                disposition="kept",
                reason=None,
            )
        )

    payload = b"\n".join(lines)
    if native_batch.candidates and not payload:
        raise ValueError("non-empty native evidence produced an empty visible context")
    dropped = sum(item.disposition != "kept" for item in decisions)
    return VisibleEvidence(
        canonical_bytes=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        included_native_ids=tuple(included_ids),
        token_count=count_o200k_tokens(payload),
        candidate_count=len(native_batch.candidates),
        kept_count=len(lines),
        dropped_count=dropped,
        truncated_count=0,
        first_exceeded_limit=first_exceeded,
        decisions=tuple(decisions),
        payload_byte_count=len(payload),
        character_count=len(payload.decode("utf-8")),
        tokenizer_fingerprint=tokenizer_fingerprint(),
    )
