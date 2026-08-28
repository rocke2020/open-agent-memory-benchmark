from __future__ import annotations

import json
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
    lines = visible.canonical_bytes.decode().splitlines()

    assert len(lines) == 2
    assert list(json.loads(lines[0])) == [
        "provider_evidence_identity",
        "source_unit_id",
        "evidence_kind",
        "text",
        "occurred_start",
        "occurred_end",
        "mentioned_at",
    ]
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


def test_lme_evidence_stops_at_first_whole_item_and_conflicting_identity_fails() -> None:
    batch = NativeEvidenceBatch(
        raw_reference=RawReferenceHandle("a" * 64),
        candidates=(_candidate("id-1", "first", 1), _candidate("id-2", "second", 2)),
    )
    policy = LME_VISIBLE_EVIDENCE_POLICY.__class__(
        max_items=1,
        max_characters=LME_VISIBLE_EVIDENCE_POLICY.max_characters,
        max_tokens=LME_VISIBLE_EVIDENCE_POLICY.max_tokens,
    )

    visible = build_lme_visible_evidence(batch, policy)

    assert visible.kept_count == 1
    assert visible.dropped_count == 1
    assert visible.first_exceeded_limit == "max_items"
    conflict = NativeEvidenceBatch(
        raw_reference=batch.raw_reference,
        candidates=(_candidate("same", "one", 1), _candidate("same", "two", 2)),
    )
    with pytest.raises(EvidenceConflictError, match="conflicting content"):
        build_lme_visible_evidence(conflict, LME_VISIBLE_EVIDENCE_POLICY)


def test_lme_evidence_rejects_a_native_read_truncation() -> None:
    truncated = replace(_candidate("id-1", "partial", 1), native_truncated=True)

    with pytest.raises(ValueError, match="native truncation"):
        build_lme_visible_evidence(
            NativeEvidenceBatch(
                raw_reference=RawReferenceHandle("a" * 64),
                candidates=(truncated,),
            )
        )


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
