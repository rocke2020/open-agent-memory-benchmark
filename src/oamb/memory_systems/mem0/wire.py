"""Strict, side-effect-free Mem0 v2.0.19 REST wire codecs."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from oamb.memory_systems.rest import parse_exact_json_object

MEM0_SEARCH_THRESHOLD = 0.1
MEM0_REST_ROUTE_ALLOWLIST = (
    ("POST", "/memories"),
    ("POST", "/search"),
)

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_ADD_RESPONSE_FIELDS = frozenset({"results"})
_ADD_EVENT_FIELDS = frozenset({"id", "memory", "event"})
_ADD_EVENT_VALUES = frozenset({"ADD", "UPDATE", "DELETE"})
_SEARCH_RESPONSE_FIELDS = frozenset({"results"})
_SEARCH_REQUIRED_ITEM_FIELDS = frozenset(
    {
        "id",
        "memory",
        "hash",
        "metadata",
        "score",
        "created_at",
        "updated_at",
        "run_id",
    }
)
_SEARCH_OPTIONAL_ITEM_FIELDS = frozenset({"attributed_to"})
_SEARCH_ALLOWED_ITEM_FIELDS = _SEARCH_REQUIRED_ITEM_FIELDS | _SEARCH_OPTIONAL_ITEM_FIELDS
_SOURCE_METADATA_FIELDS = frozenset(
    {
        "oamb_ingestion_occurrence_id",
        "oamb_ingestion_plan_id",
        "oamb_source_unit_id",
        "oamb_source_ordinal",
    }
)


@dataclass(frozen=True, slots=True)
class Mem0SourceMetadata:
    ingestion_occurrence_id: str
    ingestion_plan_id: str
    source_unit_id: str
    source_ordinal: int

    def __post_init__(self) -> None:
        _require_non_empty_string(
            self.ingestion_occurrence_id,
            field_name="ingestion_occurrence_id",
        )
        _require_non_empty_string(self.ingestion_plan_id, field_name="ingestion_plan_id")
        _require_non_empty_string(self.source_unit_id, field_name="source_unit_id")
        if isinstance(self.source_ordinal, bool) or self.source_ordinal < 1:
            raise ValueError("source_ordinal must be a positive integer")

    def to_wire(self) -> dict[str, str | int]:
        return {
            "oamb_ingestion_occurrence_id": self.ingestion_occurrence_id,
            "oamb_ingestion_plan_id": self.ingestion_plan_id,
            "oamb_source_unit_id": self.source_unit_id,
            "oamb_source_ordinal": self.source_ordinal,
        }


@dataclass(frozen=True, slots=True)
class Mem0RestRequest:
    method: str
    path: str
    body: bytes

    def __post_init__(self) -> None:
        if (self.method, self.path) not in MEM0_REST_ROUTE_ALLOWLIST:
            raise ValueError("Mem0 route is not in the exact nondestructive allowlist")
        if not self.body:
            raise ValueError("Mem0 request body must not be empty")


class Mem0AddDisposition(StrEnum):
    ACCEPTED = "accepted"
    EMPTY_PROVIDER_OUTCOME = "empty_provider_outcome"


@dataclass(frozen=True, slots=True)
class Mem0AddEvent:
    native_id: str
    memory: str
    event: str


@dataclass(frozen=True, slots=True)
class Mem0AddResult:
    disposition: Mem0AddDisposition
    events: tuple[Mem0AddEvent, ...]


@dataclass(frozen=True, slots=True)
class Mem0SearchItem:
    native_id: str
    native_rank_1_indexed: int
    memory: str
    memory_hash: str | None
    metadata: Mem0SourceMetadata | None
    native_score: float
    created_at: str | None
    updated_at: str | None
    run_id: str
    attributed_to: str | None


def encode_add_request(
    *,
    messages: tuple[tuple[str, str], ...],
    run_id: str,
    metadata: Mem0SourceMetadata,
) -> bytes:
    _require_run_id(run_id)
    if not messages:
        raise ValueError("messages must not be empty")
    encoded_messages: list[dict[str, str]] = []
    for role, content in messages:
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        encoded_messages.append(
            {
                "role": _require_non_empty_string(role, field_name="message role"),
                "content": content,
            }
        )
    return _encode_exact_json(
        {
            "messages": encoded_messages,
            "run_id": run_id,
            "metadata": metadata.to_wire(),
            "infer": True,
        }
    )


def encode_search_request(*, query: str, run_id: str, top_k: int) -> bytes:
    _require_non_empty_string(query, field_name="query")
    _require_run_id(run_id)
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top_k must be a positive integer")
    return _encode_exact_json(
        {
            "query": query,
            "filters": {"run_id": run_id},
            "top_k": top_k,
            "threshold": MEM0_SEARCH_THRESHOLD,
        }
    )


def build_add_http_request(
    *,
    messages: tuple[tuple[str, str], ...],
    run_id: str,
    metadata: Mem0SourceMetadata,
) -> Mem0RestRequest:
    return Mem0RestRequest(
        method="POST",
        path="/memories",
        body=encode_add_request(messages=messages, run_id=run_id, metadata=metadata),
    )


def build_search_http_request(*, query: str, run_id: str, top_k: int) -> Mem0RestRequest:
    return Mem0RestRequest(
        method="POST",
        path="/search",
        body=encode_search_request(query=query, run_id=run_id, top_k=top_k),
    )


def parse_add_response(raw_bytes: bytes) -> Mem0AddResult:
    document = parse_exact_json_object(raw_bytes, expected_fields=_ADD_RESPONSE_FIELDS)
    raw_results = document["results"]
    if not isinstance(raw_results, list):
        raise ValueError("add response results must be a list")

    events: list[Mem0AddEvent] = []
    seen_ids: set[str] = set()
    for raw_event in raw_results:
        event = _require_exact_object(
            raw_event,
            expected_fields=_ADD_EVENT_FIELDS,
            description="add event",
        )
        native_id = _require_non_empty_string(event["id"], field_name="add event id")
        if native_id in seen_ids:
            raise ValueError(f"duplicate add event id: {native_id}")
        seen_ids.add(native_id)
        memory = _require_non_empty_string(event["memory"], field_name="add event memory")
        event_name = _require_non_empty_string(event["event"], field_name="add event event")
        if event_name not in _ADD_EVENT_VALUES:
            raise ValueError(f"unsupported exact-profile add event: {event_name}")
        events.append(Mem0AddEvent(native_id=native_id, memory=memory, event=event_name))

    if not events:
        return Mem0AddResult(
            disposition=Mem0AddDisposition.EMPTY_PROVIDER_OUTCOME,
            events=(),
        )
    return Mem0AddResult(
        disposition=Mem0AddDisposition.ACCEPTED,
        events=tuple(events),
    )


def parse_search_response(
    raw_bytes: bytes,
    *,
    expected_run_id: str,
) -> tuple[Mem0SearchItem, ...]:
    _require_run_id(expected_run_id)
    document = parse_exact_json_object(raw_bytes, expected_fields=_SEARCH_RESPONSE_FIELDS)
    raw_results = document["results"]
    if not isinstance(raw_results, list):
        raise ValueError("search response results must be a list")

    items: list[Mem0SearchItem] = []
    seen_ids: set[str] = set()
    for native_rank_1_indexed, raw_item in enumerate(raw_results, start=1):
        item = _require_allowed_object(
            raw_item,
            required_fields=_SEARCH_REQUIRED_ITEM_FIELDS,
            allowed_fields=_SEARCH_ALLOWED_ITEM_FIELDS,
            description="search item",
        )
        native_id = _require_non_empty_string(item["id"], field_name="search item id")
        if native_id in seen_ids:
            raise ValueError(f"duplicate search item id: {native_id}")
        seen_ids.add(native_id)

        run_id = _require_non_empty_string(item["run_id"], field_name="search item run_id")
        if run_id != expected_run_id:
            raise ValueError("search item run_id does not match the requested scope")

        score = item["score"]
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("search item score must be a finite number")
        native_score = float(score)
        if not math.isfinite(native_score):
            raise ValueError("search item score must be a finite number")

        items.append(
            Mem0SearchItem(
                native_id=native_id,
                native_rank_1_indexed=native_rank_1_indexed,
                memory=_require_non_empty_string(
                    item["memory"],
                    field_name="search item memory",
                ),
                memory_hash=_require_optional_string(
                    item["hash"],
                    field_name="search item hash",
                ),
                metadata=_parse_optional_source_metadata(item["metadata"]),
                native_score=native_score,
                created_at=_require_optional_string(
                    item["created_at"],
                    field_name="search item created_at",
                ),
                updated_at=_require_optional_string(
                    item["updated_at"],
                    field_name="search item updated_at",
                ),
                run_id=run_id,
                attributed_to=_require_optional_string(
                    item.get("attributed_to"),
                    field_name="search item attributed_to",
                ),
            )
        )
    return tuple(items)


def _parse_source_metadata(value: object) -> Mem0SourceMetadata:
    metadata = _require_exact_object(
        value,
        expected_fields=_SOURCE_METADATA_FIELDS,
        description="search item metadata",
    )
    source_ordinal = metadata["oamb_source_ordinal"]
    if isinstance(source_ordinal, bool) or not isinstance(source_ordinal, int):
        raise ValueError("search item metadata source ordinal must be an integer")
    return Mem0SourceMetadata(
        ingestion_occurrence_id=_require_non_empty_string(
            metadata["oamb_ingestion_occurrence_id"],
            field_name="search item ingestion occurrence id",
        ),
        ingestion_plan_id=_require_non_empty_string(
            metadata["oamb_ingestion_plan_id"],
            field_name="search item ingestion plan id",
        ),
        source_unit_id=_require_non_empty_string(
            metadata["oamb_source_unit_id"],
            field_name="search item source unit id",
        ),
        source_ordinal=source_ordinal,
    )


def _parse_optional_source_metadata(value: object) -> Mem0SourceMetadata | None:
    if value == {}:
        return None
    return _parse_source_metadata(value)


def _encode_exact_json(document: object) -> bytes:
    return json.dumps(
        document,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_exact_object(
    value: object,
    *,
    expected_fields: frozenset[str],
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    actual_fields = frozenset(value)
    if actual_fields != expected_fields:
        raise ValueError(
            f"{description} fields do not match the exact profile: "
            f"expected {sorted(expected_fields)}, got {sorted(actual_fields)}"
        )
    return value


def _require_allowed_object(
    value: object,
    *,
    required_fields: frozenset[str],
    allowed_fields: frozenset[str],
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    fields = frozenset(value)
    if not required_fields.issubset(fields) or not fields.issubset(allowed_fields):
        raise ValueError(
            f"{description} fields do not match the exact profile: "
            f"required {sorted(required_fields)}, allowed {sorted(allowed_fields)}, "
            f"got {sorted(fields)}"
        )
    return value


def _require_run_id(value: object) -> str:
    run_id = _require_non_empty_string(value, field_name="run_id")
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must be a full lowercase SHA-256 identifier")
    return run_id


def _require_non_empty_string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, field_name=field_name)


__all__ = [
    "MEM0_REST_ROUTE_ALLOWLIST",
    "MEM0_SEARCH_THRESHOLD",
    "Mem0AddDisposition",
    "Mem0AddEvent",
    "Mem0AddResult",
    "Mem0RestRequest",
    "Mem0SearchItem",
    "Mem0SourceMetadata",
    "build_add_http_request",
    "build_search_http_request",
    "encode_add_request",
    "encode_search_request",
    "parse_add_response",
    "parse_search_response",
]
