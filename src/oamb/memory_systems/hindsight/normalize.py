"""Fail-closed Hindsight v0.9.2 recall normalization."""

from __future__ import annotations

import json
from typing import Any, Final

from oamb.contracts.ports import NativeEvidenceCandidate
from oamb.memory_systems.rest import parse_exact_json_object

RECALL_FIELDS: Final = frozenset(
    {"results", "trace", "entities", "chunks", "source_facts", "source_facts_truncated"}
)
RESULT_FIELDS: Final = frozenset(
    {
        "id",
        "text",
        "type",
        "entities",
        "context",
        "occurred_start",
        "occurred_end",
        "mentioned_at",
        "document_id",
        "metadata",
        "chunk_id",
        "tags",
        "source_fact_ids",
        "scores",
    }
)
CHUNK_FIELDS: Final = frozenset({"id", "text", "chunk_index", "truncated"})
SCORE_FIELDS: Final = frozenset({"final", "reranker", "semantic", "keyword"})


def _exact_object(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != fields:
        raise ValueError(f"Hindsight recall {label} fields do not match the exact profile")
    return value


def _nullable_string(value: object, label: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Hindsight recall {label} must be a string or null")
    return value


def _string_list_or_none(value: object, label: str) -> list[str] | None:
    if value is None:
        return None
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Hindsight recall {label} must be an array of strings or null")
    return value


def _native_score(value: object) -> str | None:
    if value is None:
        return None
    scores = _exact_object(value, SCORE_FIELDS, "scores")
    final_score = scores["final"]
    if type(final_score) not in {int, float}:
        raise ValueError("Hindsight recall final score must be numeric")
    for name in ("reranker", "semantic", "keyword"):
        score = scores[name]
        if score is not None and type(score) not in {int, float}:
            raise ValueError(f"Hindsight recall {name} score must be numeric or null")
    if scores["reranker"] is not None:
        raise ValueError("Hindsight recall returned a reranker score after reranking was disabled")
    return json.dumps(final_score, allow_nan=False, separators=(",", ":"))


def _chunks(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict):
        raise ValueError("Hindsight recall chunks must be an object")
    chunks: dict[str, dict[str, Any]] = {}
    for key, raw_chunk in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Hindsight recall chunk key must be non-empty")
        chunk = _exact_object(raw_chunk, CHUNK_FIELDS, "chunk")
        if chunk["id"] != key:
            raise ValueError("Hindsight recall chunk ID disagrees with its map key")
        if not isinstance(chunk["text"], str):
            raise ValueError("Hindsight recall chunk text must be a string")
        if type(chunk["chunk_index"]) is not int or chunk["chunk_index"] < 0:
            raise ValueError("Hindsight recall chunk index must be non-negative")
        if type(chunk["truncated"]) is not bool:
            raise ValueError("Hindsight recall chunk truncated must be boolean")
        chunks[key] = chunk
    return chunks


def normalize_recall(
    raw_bytes: bytes,
    *,
    document_to_source_unit: dict[str, str],
) -> tuple[NativeEvidenceCandidate, ...]:
    document = parse_exact_json_object(raw_bytes, expected_fields=RECALL_FIELDS)
    if document["entities"] is not None:
        raise ValueError("Hindsight recall returned entities after they were disabled")
    if document["source_facts"] is not None or document["source_facts_truncated"] is not None:
        raise ValueError("Hindsight recall returned unrequested source facts")
    if document["trace"] is not None and not isinstance(document["trace"], dict):
        raise ValueError("Hindsight recall trace must be an object or null")
    chunks = _chunks(document["chunks"])
    results = document["results"]
    if not isinstance(results, list):
        raise ValueError("Hindsight recall results must be an array")

    candidates: list[NativeEvidenceCandidate] = []
    seen_ids: set[str] = set()
    referenced_chunks: set[str] = set()
    for rank, raw_result in enumerate(results, start=1):
        result = _exact_object(raw_result, RESULT_FIELDS, "result")
        memory_id = result["id"]
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("Hindsight recall memory ID must be non-empty")
        if memory_id in seen_ids:
            raise ValueError("Hindsight recall returned a duplicate memory ID")
        seen_ids.add(memory_id)
        text = result["text"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Hindsight recall result text must be non-empty")
        fact_type = result["type"]
        if fact_type not in {"world", "experience"}:
            raise ValueError("Hindsight recall returned an unrequested fact type")
        _string_list_or_none(result["entities"], "entities")
        _string_list_or_none(result["tags"], "tags")
        if result["metadata"] is not None and not isinstance(result["metadata"], dict):
            raise ValueError("Hindsight recall metadata must be an object or null")
        if result["source_fact_ids"] is not None:
            raise ValueError("Hindsight raw fact unexpectedly returned source_fact_ids")
        document_id = _nullable_string(result["document_id"], "document_id")
        source_unit_id = None
        if document_id is not None:
            try:
                source_unit_id = document_to_source_unit[document_id]
            except KeyError as exc:
                raise ValueError(
                    "Hindsight recall result names an unknown source document"
                ) from exc
        chunk_id = _nullable_string(result["chunk_id"], "chunk_id")
        if chunk_id is not None:
            if chunk_id not in chunks:
                raise ValueError("Hindsight recall result names an unknown chunk")
            referenced_chunks.add(chunk_id)
        occurred_start = _nullable_string(result["occurred_start"], "occurred_start")
        occurred_end = _nullable_string(result["occurred_end"], "occurred_end")
        mentioned_at = _nullable_string(result["mentioned_at"], "mentioned_at")
        _nullable_string(result["context"], "context")
        candidates.append(
            NativeEvidenceCandidate(
                native_id=memory_id,
                native_rank_1_indexed=rank,
                content=text,
                native_score=_native_score(result["scores"]),
                provider_evidence_identity=memory_id,
                source_unit_id=source_unit_id,
                evidence_kind=fact_type,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
                native_reference=chunk_id,
                native_truncated=chunks[chunk_id]["truncated"] if chunk_id is not None else False,
            )
        )
    if set(chunks) != referenced_chunks:
        raise ValueError("Hindsight recall returned an unreferenced raw chunk")
    return tuple(candidates)
