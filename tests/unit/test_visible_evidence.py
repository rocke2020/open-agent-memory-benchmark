from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from oamb.contracts.ports import NativeEvidenceBatch, NativeEvidenceCandidate, RawReferenceHandle
from oamb.workloads.visible_evidence import (
    LME_VISIBLE_EVIDENCE_POLICY,
    EvidenceConflictError,
    build_lme_visible_evidence,
    count_o200k_tokens,
)


def _candidate(identity: str, text: str, rank: int) -> NativeEvidenceCandidate:
    return NativeEvidenceCandidate(
        native_id=f"native-{rank}",
        native_rank_1_indexed=rank,
        content=text,
        native_score=None,
        provider_evidence_identity=identity,
        source_unit_id=None,
        evidence_kind="memory",
        occurred_start=None,
        occurred_end=None,
        mentioned_at=None,
        native_reference=f"raw/{rank}",
        native_truncated=False,
    )


def test_lme_evidence_preserves_provider_order_deduplicates_and_counts_exact_bytes() -> None:
    batch = NativeEvidenceBatch(
        raw_reference=RawReferenceHandle("a" * 64),
        candidates=(
            _candidate("id-1", "你好 <|endoftext|>", 1),
            _candidate("id-1", "你好 <|endoftext|>", 2),
            _candidate("id-2", "second", 3),
        ),
    )

    visible = build_lme_visible_evidence(batch, LME_VISIBLE_EVIDENCE_POLICY)
    assert visible.canonical_bytes == (
        "[F1] memory\n你好 <|endoftext|>\n\n[F2] memory\nsecond".encode()
    )
    assert visible.included_native_ids == ("native-1", "native-3")
    assert visible.candidate_count == 3
    assert visible.kept_count == 2
    assert visible.dropped_count == 1
    assert visible.truncated_count == 0
    assert visible.token_count == count_o200k_tokens(visible.canonical_bytes)
    assert visible.payload_byte_count == len(visible.canonical_bytes)
    assert visible.character_count == len(visible.canonical_bytes.decode())
    assert visible.tokenizer_fingerprint is not None
    assert not visible.canonical_bytes.endswith(b"\n")


def test_lme_evidence_rejects_a_harness_ceiling_and_conflicting_identity() -> None:
    batch = NativeEvidenceBatch(
        raw_reference=RawReferenceHandle("a" * 64),
        candidates=(_candidate("id-1", "first", 1), _candidate("id-2", "second", 2)),
    )
    policy = LME_VISIBLE_EVIDENCE_POLICY.__class__(
        max_items=1,
        max_characters=LME_VISIBLE_EVIDENCE_POLICY.max_characters,
        max_tokens=LME_VISIBLE_EVIDENCE_POLICY.max_tokens,
    )

    with pytest.raises(ValueError, match="without.*ceiling"):
        build_lme_visible_evidence(batch, policy)
    conflict = NativeEvidenceBatch(
        raw_reference=batch.raw_reference,
        candidates=(_candidate("same", "one", 1), _candidate("same", "two", 2)),
    )
    with pytest.raises(EvidenceConflictError, match="conflicting content"):
        build_lme_visible_evidence(conflict, LME_VISIBLE_EVIDENCE_POLICY)


def test_lme_evidence_keeps_partial_chunk_and_complete_fact_with_exact_labels() -> None:
    fact = replace(
        _candidate("fact-id", "Leave it for ten minutes.", 1),
        evidence_kind="world",
        native_reference="chunk-id",
        occurred_start="2023-04-18T10:00:00+02:00",
        mentioned_at="2023-04-19T00:00:00+00:00",
    )
    chunk = replace(
        _candidate("chunk:chunk-id", '[{"role":"assistant","content":"ten minutes', 2),
        evidence_kind="source_chunk",
        native_reference="chunk-id",
        native_truncated=True,
    )
    visible = build_lme_visible_evidence(
        NativeEvidenceBatch(
            raw_reference=RawReferenceHandle("a" * 64),
            candidates=(fact, chunk),
        )
    )
    assert visible.canonical_bytes == (
        b"[F1] world source=S1 occurred_start=2023-04-18T08:00:00.000000+00:00 "
        b"mentioned_at=2023-04-19T00:00:00.000000+00:00\nLeave it for ten minutes.\n\n"
        b'[S1] source_chunk native_truncated=true\n[{"role":"assistant","content":"ten minutes'
    )
    assert (
        visible.candidate_count,
        visible.kept_count,
        visible.dropped_count,
        visible.truncated_count,
    ) == (2, 2, 0, 1)
    assert [d.disposition for d in visible.decisions] == ["kept", "kept"]
    assert [d.label for d in visible.decisions] == ["F1", "S1"]


def test_lme_evidence_preserves_content_beyond_all_former_limits() -> None:
    text = "marker: alpha beta gamma delta " * 12000
    candidates = tuple(
        _candidate(f"id-{i}", text if i == 257 else str(i), i) for i in range(1, 258)
    )
    visible = build_lme_visible_evidence(
        NativeEvidenceBatch(
            raw_reference=RawReferenceHandle("a" * 64),
            candidates=candidates,
        )
    )
    assert visible.kept_count == 257
    assert visible.dropped_count == 0
    assert visible.first_exceeded_limit is None
    assert visible.character_count > 262144
    assert visible.token_count > 65536
    assert visible.canonical_bytes.endswith(text.encode())


def test_lme_evidence_preserves_empty_provider_text_in_order() -> None:
    batch = NativeEvidenceBatch(
        raw_reference=RawReferenceHandle("a" * 64),
        candidates=(
            _candidate("empty-abstract", "", 1),
            _candidate("with-abstract", "provider content", 2),
        ),
    )

    visible = build_lme_visible_evidence(batch)
    assert visible.canonical_bytes == b"[F1] memory\n\n\n[F2] memory\nprovider content"
    assert visible.included_native_ids == ("native-1", "native-2")
    assert visible.kept_count == visible.candidate_count == 2
    assert visible.dropped_count == 0
    assert visible.sha256 == hashlib.sha256(visible.canonical_bytes).hexdigest()
    assert visible.token_count == count_o200k_tokens(visible.canonical_bytes)


def test_lme_evidence_requires_an_explicit_provider_evidence_identity() -> None:
    missing_identity = replace(
        _candidate("id-1", "content", 1),
        provider_evidence_identity=None,
    )

    with pytest.raises(ValueError, match="provider evidence identity"):
        build_lme_visible_evidence(
            NativeEvidenceBatch(
                raw_reference=RawReferenceHandle("a" * 64),
                candidates=(missing_identity,),
            )
        )
