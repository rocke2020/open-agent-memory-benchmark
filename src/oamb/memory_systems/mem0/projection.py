"""Strict parser for the read-only Mem0 projection inspector."""

from __future__ import annotations

import base64
import binascii
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from oamb.contracts.ids import canonical_sha256
from oamb.memory_systems.rest import parse_exact_json_object

from .wire import Mem0SourceMetadata

MEM0_PROJECTION_MAX_PAGES = 4096
MEM0_PROJECTION_MAX_POINTS = 1_000_000

_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{64}")
_CURSOR_PATTERN = re.compile(r"[A-Za-z0-9_-]+")
_PAGE_FIELDS = frozenset({"collection", "run_id", "count", "points", "next_cursor"})
_POINT_FIELDS = frozenset({"id", "payload"})
_REQUIRED_PAYLOAD_FIELDS = frozenset(
    {
        "data",
        "text_lemmatized",
        "hash",
        "created_at",
        "updated_at",
        "run_id",
    }
)
_SOURCE_METADATA_FIELDS = frozenset(
    {
        "oamb_ingestion_occurrence_id",
        "oamb_ingestion_plan_id",
        "oamb_source_unit_id",
        "oamb_source_ordinal",
    }
)
_OPTIONAL_PAYLOAD_FIELDS = frozenset({"attributed_to"}) | _SOURCE_METADATA_FIELDS
_ALLOWED_PAYLOAD_FIELDS = _REQUIRED_PAYLOAD_FIELDS | _OPTIONAL_PAYLOAD_FIELDS


@dataclass(frozen=True, slots=True)
class Mem0ProjectionPoint:
    native_id: str
    memory: str
    text_lemmatized: str
    memory_hash: str
    created_at: str
    updated_at: str
    run_id: str
    metadata: Mem0SourceMetadata | None
    attributed_to: str | None


@dataclass(frozen=True, slots=True)
class Mem0Projection:
    collection: str
    run_id: str
    declared_count: int
    points: tuple[Mem0ProjectionPoint, ...]
    page_count: int


def ordered_projection_points(
    projection: Mem0Projection,
) -> tuple[Mem0ProjectionPoint, ...]:
    return tuple(
        sorted(
            projection.points,
            key=lambda point: (
                math.inf if point.metadata is None else point.metadata.source_ordinal,
                point.native_id,
            ),
        )
    )


def projection_source_unit_ids(projection: Mem0Projection) -> tuple[str, ...]:
    source_ids: list[str] = []
    for point in ordered_projection_points(projection):
        if point.metadata is None:
            continue
        if point.metadata.source_unit_id not in source_ids:
            source_ids.append(point.metadata.source_unit_id)
    return tuple(source_ids)


def projection_state_sha256(projection: Mem0Projection) -> str:
    return canonical_sha256(
        [
            "oamb-mem0-main-projection-v1",
            projection.collection,
            projection.run_id,
            tuple(
                _projection_point_tuple(point) for point in ordered_projection_points(projection)
            ),
        ]
    )


def parse_projection_pages(
    raw_pages: tuple[bytes, ...],
    *,
    expected_collection: str,
    expected_run_id: str,
    max_pages: int = MEM0_PROJECTION_MAX_PAGES,
    max_points: int = MEM0_PROJECTION_MAX_POINTS,
) -> Mem0Projection:
    _require_non_empty_string(expected_collection, field_name="expected collection")
    _require_run_id(expected_run_id)
    _require_positive_limit(max_pages, field_name="projection page limit")
    _require_positive_limit(max_points, field_name="projection point limit")
    if not raw_pages:
        raise ValueError("projection requires at least one terminal projection page")
    if len(raw_pages) > max_pages:
        raise ValueError("projection page limit exceeded")

    declared_count: int | None = None
    points: list[Mem0ProjectionPoint] = []
    seen_point_ids: set[str] = set()
    seen_cursors: set[str] = set()
    final_cursor: str | None = None

    for page_number, raw_page in enumerate(raw_pages, start=1):
        page = parse_exact_json_object(raw_page, expected_fields=_PAGE_FIELDS)
        collection = _require_non_empty_string(
            page["collection"],
            field_name="projection collection",
        )
        if collection != expected_collection:
            raise ValueError("projection collection does not match the exact profile")
        run_id = _require_non_empty_string(page["run_id"], field_name="projection run_id")
        if run_id != expected_run_id:
            raise ValueError("projection run_id does not match the requested scope")

        page_count = page["count"]
        if isinstance(page_count, bool) or not isinstance(page_count, int) or page_count < 0:
            raise ValueError("projection count must be a non-negative integer")
        if page_count > max_points:
            raise ValueError("projection point limit exceeded")
        if declared_count is None:
            declared_count = page_count
        elif page_count != declared_count:
            raise ValueError("projection count changed between pages")

        raw_points = page["points"]
        if not isinstance(raw_points, list):
            raise ValueError("projection points must be a list")
        final_cursor = _parse_cursor(page["next_cursor"])
        if final_cursor is not None:
            if final_cursor in seen_cursors:
                raise ValueError("projection cursor repeated before terminal page")
            seen_cursors.add(final_cursor)
            if not raw_points:
                raise ValueError("non-terminal projection page is empty")
        elif page_number != len(raw_pages):
            raise ValueError("terminal projection page must be last")

        for raw_point in raw_points:
            point = _parse_projection_point(raw_point, expected_run_id=expected_run_id)
            if point.native_id in seen_point_ids:
                raise ValueError(f"duplicate projection point id: {point.native_id}")
            seen_point_ids.add(point.native_id)
            points.append(point)
            if len(points) > max_points:
                raise ValueError("projection point limit exceeded")

    if final_cursor is not None:
        raise ValueError("projection requires a terminal projection page")
    if declared_count is None:
        raise ValueError("projection requires a declared count")
    if len(points) != declared_count:
        raise ValueError(
            "projection point cardinality does not match the inspector count: "
            f"expected {declared_count}, got {len(points)}"
        )
    return Mem0Projection(
        collection=expected_collection,
        run_id=expected_run_id,
        declared_count=declared_count,
        points=tuple(points),
        page_count=len(raw_pages),
    )


def _parse_projection_point(
    value: object,
    *,
    expected_run_id: str,
) -> Mem0ProjectionPoint:
    point = _require_exact_object(
        value,
        expected_fields=_POINT_FIELDS,
        description="projection point",
    )
    native_id = _require_non_empty_string(point["id"], field_name="projection point id")
    payload = _require_payload(point["payload"])
    run_id = _require_non_empty_string(payload["run_id"], field_name="projection point run_id")
    if run_id != expected_run_id:
        raise ValueError("projection point run_id does not match the requested scope")
    metadata = _parse_optional_source_metadata(payload)
    if metadata is not None and metadata.ingestion_occurrence_id != expected_run_id:
        raise ValueError("projection point ingestion occurrence does not match the requested scope")
    attributed_to = payload.get("attributed_to")
    return Mem0ProjectionPoint(
        native_id=native_id,
        memory=_require_non_empty_string(payload["data"], field_name="projection point data"),
        text_lemmatized=_require_non_empty_string(
            payload["text_lemmatized"],
            field_name="projection point text_lemmatized",
        ),
        memory_hash=_require_non_empty_string(
            payload["hash"],
            field_name="projection point hash",
        ),
        created_at=_require_non_empty_string(
            payload["created_at"],
            field_name="projection point created_at",
        ),
        updated_at=_require_non_empty_string(
            payload["updated_at"],
            field_name="projection point updated_at",
        ),
        run_id=run_id,
        metadata=metadata,
        attributed_to=(
            None
            if attributed_to is None
            else _require_non_empty_string(
                attributed_to,
                field_name="projection point attributed_to",
            )
        ),
    )


def _projection_point_tuple(point: Mem0ProjectionPoint) -> tuple[object, ...]:
    metadata = point.metadata
    return (
        point.native_id,
        point.memory,
        point.text_lemmatized,
        point.memory_hash,
        point.created_at,
        point.updated_at,
        point.run_id,
        None
        if metadata is None
        else (
            metadata.ingestion_occurrence_id,
            metadata.ingestion_plan_id,
            metadata.source_unit_id,
            metadata.source_ordinal,
        ),
        point.attributed_to,
    )


def _parse_source_metadata(payload: dict[str, Any]) -> Mem0SourceMetadata:
    source_ordinal = payload["oamb_source_ordinal"]
    if isinstance(source_ordinal, bool) or not isinstance(source_ordinal, int):
        raise ValueError("projection point source ordinal must be an integer")
    return Mem0SourceMetadata(
        ingestion_occurrence_id=_require_non_empty_string(
            payload["oamb_ingestion_occurrence_id"],
            field_name="projection point ingestion occurrence id",
        ),
        ingestion_plan_id=_require_non_empty_string(
            payload["oamb_ingestion_plan_id"],
            field_name="projection point ingestion plan id",
        ),
        source_unit_id=_require_non_empty_string(
            payload["oamb_source_unit_id"],
            field_name="projection point source unit id",
        ),
        source_ordinal=source_ordinal,
    )


def _parse_optional_source_metadata(
    payload: dict[str, Any],
) -> Mem0SourceMetadata | None:
    present = frozenset(payload) & _SOURCE_METADATA_FIELDS
    if not present:
        return None
    if present != _SOURCE_METADATA_FIELDS:
        raise ValueError("projection point source metadata must be complete when present")
    return _parse_source_metadata(payload)


def _parse_cursor(value: object) -> str | None:
    if value is None:
        return None
    cursor = _require_non_empty_string(value, field_name="projection next_cursor")
    if _CURSOR_PATTERN.fullmatch(cursor) is None:
        raise ValueError("projection next_cursor is not an exact inspector cursor")
    padding = "=" * (-len(cursor) % 4)
    try:
        decoded = base64.b64decode(cursor + padding, altchars=b"-_", validate=True)
        payload = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("projection next_cursor is not an exact inspector cursor") from exc
    if isinstance(payload, (dict, list, bool)) or payload is None:
        raise ValueError("projection next_cursor payload must be a string or number")
    if isinstance(payload, float) and not math.isfinite(payload):
        raise ValueError("projection next_cursor payload must be finite")
    if not isinstance(payload, (str, int, float)):
        raise ValueError("projection next_cursor payload must be a string or number")
    return cursor


def _require_payload(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("projection point payload must be an object")
    fields = frozenset(value)
    if not _REQUIRED_PAYLOAD_FIELDS.issubset(fields) or not fields.issubset(
        _ALLOWED_PAYLOAD_FIELDS
    ):
        raise ValueError(
            "projection payload fields do not match the exact profile: "
            f"required {sorted(_REQUIRED_PAYLOAD_FIELDS)}, "
            f"allowed {sorted(_ALLOWED_PAYLOAD_FIELDS)}, got {sorted(fields)}"
        )
    return value


def _require_exact_object(
    value: object,
    *,
    expected_fields: frozenset[str],
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    fields = frozenset(value)
    if fields != expected_fields:
        raise ValueError(
            f"{description} fields do not match the exact profile: "
            f"expected {sorted(expected_fields)}, got {sorted(fields)}"
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


def _require_positive_limit(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    return value


__all__ = [
    "MEM0_PROJECTION_MAX_PAGES",
    "MEM0_PROJECTION_MAX_POINTS",
    "Mem0Projection",
    "Mem0ProjectionPoint",
    "ordered_projection_points",
    "parse_projection_pages",
    "projection_source_unit_ids",
    "projection_state_sha256",
]
