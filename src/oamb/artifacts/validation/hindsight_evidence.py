"""Strict reconstruction of Hindsight projections from sealed provider receipts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from oamb.contracts.ids import canonical_json_bytes
from oamb.contracts.ports import SourceUnit
from oamb.memory_systems.hindsight.profiles import PROFILE_ID
from oamb.memory_systems.hindsight.projection import (
    DOCUMENT_DETAIL_FIELDS,
    DOCUMENT_ITEM_FIELDS,
    MEMORY_ITEM_FIELDS,
    PAGE_LIMIT,
    _canonical_memory_record,
    _document_detail,
    _memory_detail,
    _page,
    _validate_document_item,
    _validate_memory_item,
)
from oamb.memory_systems.rest import parse_exact_json_object


@dataclass(frozen=True, slots=True)
class HindsightProjectionEvidence:
    state_sha256: str
    ordered_source_unit_ids: tuple[str, ...]
    document_to_source_unit: dict[str, str]


def reconstruct_hindsight_projection(
    *,
    raw_payloads: Mapping[str, bytes],
    references: tuple[str, ...],
    bank_id: str,
    ordered_source_unit_ids: tuple[str, ...],
    ordered_source_payload_sha256: tuple[str, ...],
) -> HindsightProjectionEvidence:
    if (
        len(references) < 4
        or references[0] != references[1]
        or len(ordered_source_unit_ids) != len(ordered_source_payload_sha256)
        or len(set(ordered_source_unit_ids)) != len(ordered_source_unit_ids)
    ):
        raise ValueError("Hindsight projection reference or source inventory is invalid")
    synthetic_reference = references[0]
    supporting_references = references[2:]
    supporting_payloads = tuple(
        _required_raw(raw_payloads, reference) for reference in supporting_references
    )
    source_hash_by_id = dict(
        zip(ordered_source_unit_ids, ordered_source_payload_sha256, strict=True)
    )

    cursor = 0
    document_items, cursor = _consume_pages(
        supporting_payloads,
        cursor,
        item_fields=DOCUMENT_ITEM_FIELDS,
        label="document",
    )
    if {item.get("id") for item in document_items} != set(ordered_source_unit_ids):
        raise ValueError("Hindsight document page inventory differs from the frozen sources")
    document_records: list[dict[str, Any]] = []
    for item in sorted(document_items, key=lambda value: value["id"]):
        _validate_document_item(item, bank_id=bank_id)
        document_id = item["id"]
        if not isinstance(document_id, str):
            raise ValueError("Hindsight document ID is not a string")
        detail_payload, cursor = _next_payload(supporting_payloads, cursor)
        raw_detail = parse_exact_json_object(
            detail_payload,
            expected_fields=DOCUMENT_DETAIL_FIELDS,
        )
        original_text = raw_detail.get("original_text")
        if not isinstance(original_text, str):
            raise ValueError("Hindsight document original text is not a string")
        expected_payload_hash = source_hash_by_id[document_id]
        if hashlib.sha256(original_text.encode("utf-8")).hexdigest() != expected_payload_hash:
            raise ValueError("Hindsight document text differs from the manifest payload hash")
        source = SourceUnit(
            source_unit_id=document_id,
            context_manifest_entry_id=document_id,
            ordinal_1_indexed=ordered_source_unit_ids.index(document_id) + 1,
            payload_sha256=expected_payload_hash,
            payload_bytes=original_text.encode("utf-8"),
        )
        detail = _document_detail(
            detail_payload,
            list_item=item,
            source=source,
            bank_id=bank_id,
        )
        document_records.append({"page": item, "detail": detail})

    memory_items, cursor = _consume_pages(
        supporting_payloads,
        cursor,
        item_fields=MEMORY_ITEM_FIELDS,
        label="memory",
    )
    memory_records: list[dict[str, Any]] = []
    for item in sorted(memory_items, key=lambda value: value["id"]):
        _validate_memory_item(item, document_ids=set(ordered_source_unit_ids))
        detail_payload, cursor = _next_payload(supporting_payloads, cursor)
        detail = _memory_detail(detail_payload, list_item=item)
        memory_records.append(_canonical_memory_record(item, detail))

    mental_payload, cursor = _next_payload(supporting_payloads, cursor)
    mental_page = _page(
        mental_payload,
        item_fields=frozenset(),
        expected_limit=PAGE_LIMIT,
        expected_offset=0,
        label="mental model",
    )
    observation_payload, cursor = _next_payload(supporting_payloads, cursor)
    observation_page = _page(
        observation_payload,
        item_fields=MEMORY_ITEM_FIELDS,
        expected_limit=PAGE_LIMIT,
        expected_offset=0,
        label="observation",
    )
    if (
        mental_page.items
        or mental_page.total != 0
        or observation_page.items
        or observation_page.total != 0
        or cursor != len(supporting_payloads)
    ):
        raise ValueError("Hindsight projection terminal inventories are not exact")

    expected_bytes = canonical_json_bytes(
        {
            "profile": PROFILE_ID,
            "bank_id": bank_id,
            "documents": document_records,
            "memories": memory_records,
            "mental_models": [],
            "observations": [],
        }
    )
    if _required_raw(raw_payloads, synthetic_reference) != expected_bytes:
        raise ValueError("Hindsight synthetic projection differs from provider receipts")
    state_sha256 = hashlib.sha256(expected_bytes).hexdigest()
    if state_sha256 != synthetic_reference:
        raise ValueError("Hindsight projection reference differs from reconstructed bytes")
    return HindsightProjectionEvidence(
        state_sha256=state_sha256,
        ordered_source_unit_ids=ordered_source_unit_ids,
        document_to_source_unit={source_id: source_id for source_id in ordered_source_unit_ids},
    )


def _consume_pages(
    payloads: tuple[bytes, ...],
    cursor: int,
    *,
    item_fields: frozenset[str],
    label: str,
) -> tuple[tuple[dict[str, Any], ...], int]:
    items: list[dict[str, Any]] = []
    expected_total: int | None = None
    offset = 0
    while True:
        payload, cursor = _next_payload(payloads, cursor)
        page = _page(
            payload,
            item_fields=item_fields,
            expected_limit=PAGE_LIMIT,
            expected_offset=offset,
            label=label,
        )
        if expected_total is None:
            expected_total = page.total
        elif page.total != expected_total:
            raise ValueError(f"Hindsight {label} total changed during sealed pagination")
        items.extend(page.items)
        if len(items) > expected_total:
            raise ValueError(f"Hindsight {label} pages exceed their declared total")
        if len(items) == expected_total:
            if page.items:
                offset = expected_total
                continue
            return tuple(items), cursor
        if not page.items:
            raise ValueError(f"Hindsight {label} pages ended before their declared total")
        offset += len(page.items)


def _next_payload(payloads: tuple[bytes, ...], cursor: int) -> tuple[bytes, int]:
    if cursor >= len(payloads):
        raise ValueError("Hindsight projection supporting evidence is incomplete")
    return payloads[cursor], cursor + 1


def _required_raw(raw_payloads: Mapping[str, bytes], reference: str) -> bytes:
    try:
        return raw_payloads[reference]
    except KeyError as exc:
        raise ValueError("Hindsight projection raw reference is missing") from exc


__all__ = ["HindsightProjectionEvidence", "reconstruct_hindsight_projection"]
