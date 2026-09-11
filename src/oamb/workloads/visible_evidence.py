"""Provider-order visible-evidence normalization and exact context counting."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace
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
    max_items=None,
    max_characters=None,
    max_tokens=None,
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


def render_compact_evidence(
    candidates: Sequence[NativeEvidenceCandidate],
) -> tuple[bytes, tuple[str, ...]]:
    """Render unchanged evidence text with deterministic short source references."""

    labels: list[str] = []
    fact_count = source_count = 0
    source_labels: dict[str, str] = {}
    for candidate in candidates:
        if candidate.evidence_kind == "source_chunk":
            source_count += 1
            label = f"S{source_count}"
            if candidate.native_reference is not None:
                source_labels[candidate.native_reference] = label
        else:
            fact_count += 1
            label = f"F{fact_count}"
        labels.append(label)
    blocks: list[str] = []
    for candidate, label in zip(candidates, labels, strict=True):
        if not candidate.evidence_kind:
            raise ValueError("visible evidence requires a non-empty kind")
        header = [f"[{label}]", candidate.evidence_kind]
        if candidate.evidence_kind != "source_chunk":
            source_label = source_labels.get(candidate.native_reference or "")
            if source_label is not None:
                header.append(f"source={source_label}")
        for name in ("occurred_start", "occurred_end", "mentioned_at"):
            timestamp = _canonical_timestamp(getattr(candidate, name))
            if timestamp is not None:
                header.append(f"{name}={timestamp}")
        if candidate.native_truncated:
            header.append("native_truncated=true")
        blocks.append(" ".join(header) + "\n" + candidate.content)
    return "\n\n".join(blocks).encode("utf-8"), tuple(labels)


def build_lme_visible_evidence(
    native_batch: NativeEvidenceBatch,
    policy: VisibleEvidencePolicy = LME_VISIBLE_EVIDENCE_POLICY,
) -> VisibleEvidence:
    """Preserve provider-returned evidence without a harness admission ceiling."""

    if policy != LME_VISIBLE_EVIDENCE_POLICY:
        raise ValueError("LongMemEval renders without a harness evidence ceiling")
    seen_content: dict[str, str] = {}
    kept: list[NativeEvidenceCandidate] = []
    decisions: list[EvidenceDecision] = []
    for expected_rank, candidate in enumerate(native_batch.candidates, start=1):
        if candidate.native_rank_1_indexed != expected_rank:
            raise ValueError("native evidence ranks must preserve provider order")
        identity = _identity(candidate)
        text_hash = hashlib.sha256(candidate.content.encode()).hexdigest()
        previous_hash = seen_content.get(identity)
        if previous_hash is not None and previous_hash != text_hash:
            raise EvidenceConflictError(
                f"provider evidence identity has conflicting content: {identity}"
            )
        duplicate = previous_hash == text_hash
        if not duplicate:
            kept.append(candidate)
            seen_content[identity] = text_hash
        decisions.append(
            EvidenceDecision(
                native_id=candidate.native_id,
                provider_evidence_identity=identity,
                normalized_text_sha256=text_hash,
                disposition="duplicate" if duplicate else "kept",
                reason="duplicate_identity_and_text" if duplicate else None,
            )
        )
    payload, labels = render_compact_evidence(kept)
    labels_by_identity = dict(zip((_identity(item) for item in kept), labels, strict=True))
    decisions = [
        replace(item, label=labels_by_identity[item.provider_evidence_identity])
        for item in decisions
    ]
    return VisibleEvidence(
        canonical_bytes=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        included_native_ids=tuple(item.native_id for item in kept),
        token_count=count_o200k_tokens(payload),
        candidate_count=len(native_batch.candidates),
        kept_count=len(kept),
        dropped_count=len(decisions) - len(kept),
        truncated_count=sum(item.native_truncated for item in kept),
        first_exceeded_limit=None,
        decisions=tuple(decisions),
        payload_byte_count=len(payload),
        character_count=len(payload.decode("utf-8")),
        tokenizer_fingerprint=tokenizer_fingerprint(),
    )
