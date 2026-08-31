"""Complete protected-state projection for pinned Hindsight v0.9.2."""

from __future__ import annotations

import hashlib
import uuid
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Final

from oamb.contracts.ids import canonical_json_bytes
from oamb.contracts.ports import (
    MemorySystemCallFailure,
    MemorySystemReadCancelled,
    RawReferenceHandle,
    SourceUnit,
)
from oamb.memory_systems.rest import (
    SealedRestResponse,
    link_preceding_raw_references,
    link_preceding_read_cancellation,
    parse_exact_json_object,
    sealed_response_validation_failure,
)

from .client import HindsightClient
from .profiles import PROFILE_ID

PAGE_LIMIT: Final = 1000
ENTITY_JOIN_MATCH_STATE_LIMIT: Final = 4096
PAGE_FIELDS: Final = frozenset({"items", "total", "limit", "offset"})
DOCUMENT_ITEM_FIELDS: Final = frozenset(
    {
        "id",
        "bank_id",
        "content_hash",
        "created_at",
        "updated_at",
        "text_length",
        "memory_unit_count",
        "retain_params",
        "document_metadata",
        "tags",
    }
)
DOCUMENT_DETAIL_FIELDS: Final = frozenset(
    {
        "id",
        "bank_id",
        "original_text",
        "content_hash",
        "memory_unit_count",
        "nodes_by_fact_type",
        "created_at",
        "updated_at",
        "tags",
        "document_metadata",
        "retain_params",
        "observation_scopes",
    }
)
MEMORY_ITEM_FIELDS: Final = frozenset(
    {
        "id",
        "text",
        "context",
        "date",
        "fact_type",
        "document_id",
        "mentioned_at",
        "occurred_start",
        "occurred_end",
        "entities",
        "chunk_id",
        "proof_count",
        "tags",
        "metadata",
        "consolidated_at",
        "consolidation_failed_at",
        "state",
        "invalidation_reason",
        "invalidated_at",
        "edited_at",
        "updated_at",
        "source_memory_ids",
    }
)
MEMORY_DETAIL_FIELDS: Final = frozenset(
    {
        "id",
        "text",
        "context",
        "date",
        "type",
        "mentioned_at",
        "occurred_start",
        "occurred_end",
        "entities",
        "document_id",
        "chunk_id",
        "tags",
        "metadata",
        "observation_scopes",
        "state",
        "invalidation_reason",
        "invalidated_at",
        "edited_at",
    }
)


@dataclass(frozen=True, slots=True)
class ProjectionSnapshot:
    ordered_source_unit_ids: tuple[str, ...]
    state_sha256: str
    canonical_bytes: bytes
    raw_references: tuple[RawReferenceHandle, ...]
    document_to_source_unit: dict[str, str]


@dataclass(frozen=True, slots=True)
class _Page:
    items: tuple[dict[str, Any], ...]
    total: int


def _strict_non_negative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Hindsight {label} must be a non-negative integer")
    return value


def _exact_object(value: object, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or frozenset(value) != fields:
        raise ValueError(f"Hindsight {label} fields do not match the exact profile")
    return value


def _page(
    raw_bytes: bytes,
    *,
    item_fields: frozenset[str],
    expected_limit: int,
    expected_offset: int,
    label: str,
) -> _Page:
    document = parse_exact_json_object(raw_bytes, expected_fields=PAGE_FIELDS)
    items_value = document["items"]
    if not isinstance(items_value, list):
        raise ValueError(f"Hindsight {label} items must be an array")
    items = tuple(_exact_object(item, item_fields, f"{label} item") for item in items_value)
    total = _strict_non_negative_int(document["total"], f"{label} total")
    limit = _strict_non_negative_int(document["limit"], f"{label} limit")
    offset = _strict_non_negative_int(document["offset"], f"{label} offset")
    if limit != expected_limit or offset != expected_offset:
        raise ValueError(f"Hindsight {label} page did not echo the requested boundary")
    return _Page(items=items, total=total)


async def _documents(
    client: HindsightClient,
    bank_id: str,
    responses: list[SealedRestResponse],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_total: int | None = None
    offset = 0
    while True:
        response = await client.list_documents(bank_id, limit=PAGE_LIMIT, offset=offset)
        responses.append(response)
        page = _page(
            response.raw_bytes,
            item_fields=DOCUMENT_ITEM_FIELDS,
            expected_limit=PAGE_LIMIT,
            expected_offset=offset,
            label="document",
        )
        if expected_total is None:
            expected_total = page.total
        elif page.total != expected_total:
            raise ValueError("Hindsight document total changed during pagination")
        for item in page.items:
            document_id = item["id"]
            if not isinstance(document_id, str) or not document_id:
                raise ValueError("Hindsight document ID must be non-empty")
            if document_id in seen:
                raise ValueError("Hindsight document pagination returned a duplicate ID")
            seen.add(document_id)
            items.append(item)
        if len(items) > expected_total:
            raise ValueError("Hindsight document pagination exceeds its total")
        if not page.items:
            if len(items) != expected_total:
                raise ValueError("Hindsight document pagination ended before its total")
            break
        offset += len(page.items)
    return items


async def _memories(
    client: HindsightClient,
    bank_id: str,
    responses: list[SealedRestResponse],
) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_total: int | None = None
    offset = 0
    while True:
        response = await client.list_memories(bank_id, limit=PAGE_LIMIT, offset=offset)
        responses.append(response)
        page = _page(
            response.raw_bytes,
            item_fields=MEMORY_ITEM_FIELDS,
            expected_limit=PAGE_LIMIT,
            expected_offset=offset,
            label="memory",
        )
        if expected_total is None:
            expected_total = page.total
        elif page.total != expected_total:
            raise ValueError("Hindsight memory total changed during pagination")
        for item in page.items:
            memory_id = item["id"]
            if not isinstance(memory_id, str) or not memory_id:
                raise ValueError("Hindsight memory ID must be non-empty")
            try:
                uuid.UUID(memory_id)
            except ValueError as exc:
                raise ValueError("Hindsight memory ID must be a UUID") from exc
            if memory_id in seen:
                raise ValueError("Hindsight memory pagination returned a duplicate ID")
            seen.add(memory_id)
            items.append(item)
        if len(items) > expected_total:
            raise ValueError("Hindsight memory pagination exceeds its total")
        if not page.items:
            if len(items) != expected_total:
                raise ValueError("Hindsight memory pagination ended before its total")
            break
        offset += len(page.items)
    return items


def _validate_document_item(item: dict[str, Any], *, bank_id: str) -> None:
    if item["bank_id"] != bank_id:
        raise ValueError("Hindsight document names a different bank")
    for field in ("content_hash", "created_at", "updated_at"):
        if not isinstance(item[field], str) or not item[field]:
            raise ValueError(f"Hindsight document {field} must be non-empty")
    _strict_non_negative_int(item["text_length"], "document text length")
    _strict_non_negative_int(item["memory_unit_count"], "document memory count")
    if item["retain_params"] is not None and not isinstance(item["retain_params"], dict):
        raise ValueError("Hindsight document retain_params must be an object or null")
    if item["document_metadata"] is not None and not isinstance(item["document_metadata"], dict):
        raise ValueError("Hindsight document metadata must be an object or null")
    if not isinstance(item["tags"], list) or any(not isinstance(tag, str) for tag in item["tags"]):
        raise ValueError("Hindsight document tags must be strings")


def _document_detail(
    raw_bytes: bytes,
    *,
    list_item: dict[str, Any],
    source: SourceUnit,
    bank_id: str,
) -> dict[str, Any]:
    detail = parse_exact_json_object(raw_bytes, expected_fields=DOCUMENT_DETAIL_FIELDS)
    for field in (
        "id",
        "bank_id",
        "content_hash",
        "memory_unit_count",
        "created_at",
        "updated_at",
        "tags",
        "document_metadata",
        "retain_params",
    ):
        if detail[field] != list_item[field]:
            raise ValueError(f"Hindsight document detail {field} disagrees with its page")
    if detail["bank_id"] != bank_id or detail["id"] != source.source_unit_id:
        raise ValueError("Hindsight document detail identity does not match its source")
    original_text = detail["original_text"]
    if not isinstance(original_text, str):
        raise ValueError("Hindsight document detail did not preserve original_text")
    if original_text != source.payload_bytes.decode("utf-8", errors="strict"):
        raise ValueError("Hindsight document detail text does not match the frozen source")
    if len(original_text) != list_item["text_length"]:
        raise ValueError("Hindsight document detail text length disagrees with its page")
    if detail["content_hash"] != source.payload_sha256:
        raise ValueError("Hindsight document content_hash does not match the frozen source")
    nodes = detail["nodes_by_fact_type"]
    if nodes is not None and (
        not isinstance(nodes, dict)
        or any(
            not isinstance(name, str) or type(count) is not int or count < 0
            for name, count in nodes.items()
        )
    ):
        raise ValueError("Hindsight document nodes_by_fact_type is invalid")
    if detail["observation_scopes"] is not None:
        raise ValueError("Hindsight observations-disabled document has observation scopes")
    return detail


def _validate_memory_item(item: dict[str, Any], *, document_ids: set[str]) -> None:
    for field in ("text", "context", "date", "fact_type"):
        if not isinstance(item[field], str):
            raise ValueError(f"Hindsight memory {field} must be a string")
    if not item["text"].strip():
        raise ValueError("Hindsight memory text must be non-empty")
    if item["fact_type"] not in {"world", "experience"}:
        raise ValueError("Hindsight observations-disabled projection returned another fact type")
    if item["document_id"] not in document_ids:
        raise ValueError("Hindsight memory names an unknown source document")
    if not isinstance(item["entities"], str):
        raise ValueError("Hindsight memory page entities must be a string")
    _strict_non_negative_int(item["proof_count"], "memory proof count")
    if not isinstance(item["tags"], list) or not isinstance(item["metadata"], dict):
        raise ValueError("Hindsight memory tags or metadata have an invalid type")
    if item["state"] != "valid":
        raise ValueError("Hindsight protected projection contains an invalidated memory")
    if any(item[name] is not None for name in ("invalidation_reason", "invalidated_at")):
        raise ValueError("Hindsight valid memory carries invalidation state")
    if item["source_memory_ids"] != []:
        raise ValueError("Hindsight source fact unexpectedly carries observation lineage")


def _memory_detail(
    raw_bytes: bytes,
    *,
    list_item: dict[str, Any],
) -> dict[str, Any]:
    detail = parse_exact_json_object(raw_bytes, expected_fields=MEMORY_DETAIL_FIELDS)
    field_pairs = (
        ("id", "id"),
        ("text", "text"),
        ("context", "context"),
        ("date", "date"),
        ("type", "fact_type"),
        ("mentioned_at", "mentioned_at"),
        ("occurred_start", "occurred_start"),
        ("occurred_end", "occurred_end"),
        ("document_id", "document_id"),
        ("chunk_id", "chunk_id"),
        ("tags", "tags"),
        ("metadata", "metadata"),
        ("state", "state"),
        ("invalidation_reason", "invalidation_reason"),
        ("invalidated_at", "invalidated_at"),
        ("edited_at", "edited_at"),
    )
    for detail_field, list_field in field_pairs:
        if detail[detail_field] != list_item[list_field]:
            raise ValueError(f"Hindsight memory detail {detail_field} disagrees with its page")
    entities = detail["entities"]
    if not isinstance(entities, list) or any(
        not isinstance(name, str) or not name for name in entities
    ):
        raise ValueError("Hindsight memory detail entities must be strings")
    if not _unordered_entity_join_matches(tuple(entities), list_item["entities"]):
        raise ValueError("Hindsight memory detail entities disagree with its page")
    if detail["observation_scopes"] is not None:
        raise ValueError("Hindsight observations-disabled memory has observation scopes")
    return detail


def _unordered_entity_join_matches(entities: tuple[str, ...], serialized: str) -> bool:
    """Match the page's ambiguous comma join against the detail's unordered entities."""

    if not entities:
        return serialized == ""
    counts = Counter(entities)
    names = tuple(sorted(counts, key=lambda name: (-len(name), name)))
    initial_counts = tuple(counts[name] for name in names)
    visited_states = 0

    @lru_cache(maxsize=ENTITY_JOIN_MATCH_STATE_LIMIT)
    def match(remaining: str, remaining_counts: tuple[int, ...]) -> bool:
        nonlocal visited_states
        visited_states += 1
        if visited_states > ENTITY_JOIN_MATCH_STATE_LIMIT:
            return False
        remaining_total = sum(remaining_counts)
        if remaining_total == 0:
            return remaining == ""
        for index, (name, count) in enumerate(zip(names, remaining_counts, strict=True)):
            if count == 0:
                continue
            if remaining_total == 1:
                suffix = "" if remaining == name else None
            else:
                prefix = f"{name}, "
                suffix = remaining[len(prefix) :] if remaining.startswith(prefix) else None
            if suffix is None:
                continue
            next_counts = list(remaining_counts)
            next_counts[index] -= 1
            if match(suffix, tuple(next_counts)):
                return True
        return False

    return match(serialized, initial_counts)


async def build_projection(
    client: HindsightClient,
    *,
    bank_id: str,
    ordered_sources: tuple[SourceUnit, ...],
) -> ProjectionSnapshot:
    responses: list[SealedRestResponse] = []
    try:
        return await _build_projection(
            client,
            bank_id=bank_id,
            ordered_sources=ordered_sources,
            responses=responses,
        )
    except MemorySystemReadCancelled as exc:
        if not responses:
            raise
        raise link_preceding_read_cancellation(
            exc,
            preceding_raw_references=tuple(response.raw_reference for response in responses),
        ) from exc
    except MemorySystemCallFailure as exc:
        if not responses:
            raise
        raise link_preceding_raw_references(
            exc,
            preceding_raw_references=tuple(response.raw_reference for response in responses),
        ) from exc
    except ValueError as exc:
        if not responses:
            raise
        current_response = responses[-1]
        raise sealed_response_validation_failure(
            current_response,
            message="Hindsight projection response failed exact-profile validation",
            supporting_raw_references=tuple(response.raw_reference for response in responses[:-1]),
        ) from exc


async def _build_projection(
    client: HindsightClient,
    *,
    bank_id: str,
    ordered_sources: tuple[SourceUnit, ...],
    responses: list[SealedRestResponse],
) -> ProjectionSnapshot:
    if not ordered_sources:
        raise ValueError("Hindsight projection requires at least one frozen source")
    source_by_id = {source.source_unit_id: source for source in ordered_sources}
    if len(source_by_id) != len(ordered_sources):
        raise ValueError("Hindsight frozen source IDs must be unique")
    document_items = await _documents(client, bank_id, responses)
    if {item["id"] for item in document_items} != set(source_by_id):
        raise ValueError(
            "Hindsight document inventory does not equal the expected source inventory"
        )
    document_records: list[dict[str, Any]] = []
    for item in sorted(document_items, key=lambda value: value["id"]):
        _validate_document_item(item, bank_id=bank_id)
        response = await client.get_document(bank_id, item["id"])
        responses.append(response)
        detail = _document_detail(
            response.raw_bytes,
            list_item=item,
            source=source_by_id[item["id"]],
            bank_id=bank_id,
        )
        document_records.append({"page": item, "detail": detail})

    memory_items = await _memories(client, bank_id, responses)
    memory_records: list[dict[str, Any]] = []
    for item in sorted(memory_items, key=lambda value: value["id"]):
        _validate_memory_item(item, document_ids=set(source_by_id))
        response = await client.get_memory(bank_id, item["id"])
        responses.append(response)
        detail = _memory_detail(response.raw_bytes, list_item=item)
        memory_records.append({"page": item, "detail": detail})

    mental_response = await client.list_mental_models(bank_id, limit=PAGE_LIMIT, offset=0)
    responses.append(mental_response)
    mental_page = _page(
        mental_response.raw_bytes,
        item_fields=frozenset(),
        expected_limit=PAGE_LIMIT,
        expected_offset=0,
        label="mental model",
    )
    if mental_page.items or mental_page.total != 0:
        raise ValueError("Hindsight observations-disabled profile has mental models")

    observation_response = await client.list_memories(
        bank_id,
        fact_type="observation",
        limit=PAGE_LIMIT,
        offset=0,
    )
    responses.append(observation_response)
    observation_page = _page(
        observation_response.raw_bytes,
        item_fields=MEMORY_ITEM_FIELDS,
        expected_limit=PAGE_LIMIT,
        expected_offset=0,
        label="observation",
    )
    if observation_page.items or observation_page.total != 0:
        raise ValueError("Hindsight observations-disabled profile returned observations")

    canonical_bytes = canonical_json_bytes(
        {
            "profile": PROFILE_ID,
            "bank_id": bank_id,
            "documents": document_records,
            "memories": memory_records,
            "mental_models": [],
            "observations": [],
        }
    )
    return ProjectionSnapshot(
        ordered_source_unit_ids=tuple(source.source_unit_id for source in ordered_sources),
        state_sha256=hashlib.sha256(canonical_bytes).hexdigest(),
        canonical_bytes=canonical_bytes,
        raw_references=tuple(response.raw_reference for response in responses),
        document_to_source_unit={source_id: source_id for source_id in source_by_id},
    )
