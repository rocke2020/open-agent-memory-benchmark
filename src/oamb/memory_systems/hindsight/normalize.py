"""Fail-closed Hindsight v0.9.2 recall normalization."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Final

from oamb.contracts.ports import NativeEvidenceCandidate
from oamb.memory_systems.rest import parse_exact_json_object

RECALL_FIELDS: Final = frozenset(
    {"results", "trace", "entities", "chunks", "source_facts", "source_facts_truncated"}
)
REQUIRED_RECALL_FIELDS: Final = frozenset({"results", "trace"})
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
REQUIRED_RESULT_FIELDS: Final = frozenset({"id", "text", "type"})
CHUNK_FIELDS: Final = frozenset({"id", "text", "chunk_index", "truncated"})
SCORE_FIELDS: Final = frozenset({"final", "reranker", "semantic", "keyword"})
REQUIRED_SCORE_FIELDS: Final = frozenset({"final"})


def _exact_object(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != fields:
        raise ValueError(f"Hindsight recall {label} fields do not match the exact profile")
    return value


def _closed_object(
    value: object,
    *,
    allowed_fields: frozenset[str],
    required_fields: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Hindsight recall {label} must be an object")
    actual_fields = frozenset(value)
    if not required_fields <= actual_fields or not actual_fields <= allowed_fields:
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
    scores = _closed_object(
        value,
        allowed_fields=SCORE_FIELDS,
        required_fields=REQUIRED_SCORE_FIELDS,
        label="scores",
    )
    final_score = scores["final"]
    if type(final_score) not in {int, float}:
        raise ValueError("Hindsight recall final score must be numeric")
    for name in ("reranker", "semantic", "keyword"):
        score = scores.get(name)
        if score is not None and type(score) not in {int, float}:
            raise ValueError(f"Hindsight recall {name} score must be numeric or null")
    if scores.get("reranker") is not None:
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
    decoded = json.loads(raw_bytes)
    if not isinstance(decoded, dict):
        raise ValueError("Hindsight recall response root must be an object")
    actual_fields = frozenset(decoded)
    if not REQUIRED_RECALL_FIELDS <= actual_fields or not actual_fields <= RECALL_FIELDS:
        raise ValueError("Hindsight recall response fields do not match the exact profile")
    document = parse_exact_json_object(raw_bytes, expected_fields=actual_fields)
    if document.get("entities") is not None:
        raise ValueError("Hindsight recall returned entities after they were disabled")
    if (
        document.get("source_facts") is not None
        or document.get("source_facts_truncated") is not None
    ):
        raise ValueError("Hindsight recall returned unrequested source facts")
    if document["trace"] is not None and not isinstance(document["trace"], dict):
        raise ValueError("Hindsight recall trace must be an object or null")
    chunks = _chunks(document.get("chunks", {}))
    results = document["results"]
    if not isinstance(results, list):
        raise ValueError("Hindsight recall results must be an array")

    facts: list[NativeEvidenceCandidate] = []
    chunk_sources: dict[str, str] = {}
    seen_ids: set[str] = set()
    for rank, raw_result in enumerate(results, start=1):
        result = _closed_object(
            raw_result,
            allowed_fields=RESULT_FIELDS,
            required_fields=REQUIRED_RESULT_FIELDS,
            label="result",
        )
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
        _string_list_or_none(result.get("entities"), "entities")
        _string_list_or_none(result.get("tags"), "tags")
        if result.get("metadata") is not None and not isinstance(result["metadata"], dict):
            raise ValueError("Hindsight recall metadata must be an object or null")
        if result.get("source_fact_ids") is not None:
            raise ValueError("Hindsight raw fact unexpectedly returned source_fact_ids")
        document_id = _nullable_string(result.get("document_id"), "document_id")
        source_unit_id = None
        if document_id is not None:
            try:
                source_unit_id = document_to_source_unit[document_id]
            except KeyError as exc:
                raise ValueError(
                    "Hindsight recall result names an unknown source document"
                ) from exc
        chunk_id = _nullable_string(result.get("chunk_id"), "chunk_id")
        if chunk_id is not None and chunk_id in chunks and source_unit_id is not None:
            previous_source = chunk_sources.setdefault(chunk_id, source_unit_id)
            if previous_source != source_unit_id:
                raise ValueError("Hindsight recall chunk has conflicting source identity")
        occurred_start = _nullable_string(result.get("occurred_start"), "occurred_start")
        occurred_end = _nullable_string(result.get("occurred_end"), "occurred_end")
        mentioned_at = _nullable_string(result.get("mentioned_at"), "mentioned_at")
        _nullable_string(result.get("context"), "context")
        facts.append(
            NativeEvidenceCandidate(
                native_id=memory_id,
                native_rank_1_indexed=rank,
                content=text,
                native_score=_native_score(result.get("scores")),
                provider_evidence_identity=memory_id,
                source_unit_id=source_unit_id,
                evidence_kind=fact_type,
                occurred_start=occurred_start,
                occurred_end=occurred_end,
                mentioned_at=mentioned_at,
                native_reference=chunk_id,
                native_truncated=False,
            )
        )
    if any(f"chunk:{chunk_id}" in seen_ids for chunk_id in chunks):
        raise ValueError("Hindsight recall fact and chunk identity collision")

    candidates: list[NativeEvidenceCandidate] = []
    emitted_chunks: set[str] = set()

    def append_chunk(chunk_id: str) -> None:
        if chunk_id in emitted_chunks or chunk_id not in chunks:
            return
        chunk = chunks[chunk_id]
        identity = f"chunk:{chunk_id}"
        candidates.append(
            NativeEvidenceCandidate(
                native_id=identity,
                native_rank_1_indexed=len(candidates) + 1,
                content=chunk["text"],
                native_score=None,
                provider_evidence_identity=identity,
                source_unit_id=chunk_sources.get(chunk_id),
                evidence_kind="source_chunk",
                native_reference=chunk_id,
                native_truncated=chunk["truncated"],
            )
        )
        emitted_chunks.add(chunk_id)

    for fact in facts:
        candidates.append(replace(fact, native_rank_1_indexed=len(candidates) + 1))
        if fact.native_reference is not None:
            append_chunk(fact.native_reference)
    for chunk_id in chunks:
        append_chunk(chunk_id)
    return tuple(candidates)
