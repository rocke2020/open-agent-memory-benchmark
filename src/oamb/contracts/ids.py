"""Canonical serialization and length-safe OAMB identity formulas."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

from pydantic import BaseModel

from .base import canonical_decimal_text


def _canonical_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonical_value(value.model_dump(mode="python"))
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, Decimal):
        return canonical_decimal_text(value)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must include an explicit timezone offset")
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON map keys must be strings")
        return {key: _canonical_value(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise TypeError("floating-point values are not canonical OAMB values")
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return compact UTF-8 JSON with sorted maps and preserved sequences."""

    normalized = _canonical_value(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_identity(prefix: str, parts: Sequence[Any]) -> str:
    if not prefix:
        raise ValueError("identity prefix must not be empty")
    return canonical_sha256([prefix, *parts])


def context_content_id(
    dataset_revision: str,
    split: str,
    source: str,
    context_bytes_sha256: str,
) -> str:
    return sha256_identity(
        "oamb-context-content-v1",
        (dataset_revision, split, source, context_bytes_sha256),
    )


def context_manifest_entry_id(
    dataset_manifest_hash: str,
    source_file_sha256: str,
    source_row_number_1_indexed: int,
    content_id: str,
) -> str:
    _require_positive_ordinal(source_row_number_1_indexed)
    return sha256_identity(
        "oamb-context-manifest-entry-v1",
        (
            dataset_manifest_hash,
            source_file_sha256,
            source_row_number_1_indexed,
            content_id,
        ),
    )


def case_manifest_entry_id(
    context_manifest_id: str,
    source_question_number_1_indexed: int,
    question_bytes_sha256: str,
    raw_question_id: str,
) -> str:
    _require_positive_ordinal(source_question_number_1_indexed)
    return sha256_identity(
        "oamb-case-manifest-entry-v1",
        (
            context_manifest_id,
            source_question_number_1_indexed,
            question_bytes_sha256,
            raw_question_id,
        ),
    )


def plan_manifest_entry_id(
    workload_id: str,
    ordered_member_context_manifest_entry_ids: Sequence[str],
) -> str:
    if not ordered_member_context_manifest_entry_ids:
        raise ValueError("an ingestion plan requires at least one logical member")
    return sha256_identity(
        "oamb-ingestion-plan-manifest-entry-v1",
        (workload_id, tuple(ordered_member_context_manifest_entry_ids)),
    )


def ingestion_payload_hash(ordered_source_unit_bytes_sha256: Sequence[str]) -> str:
    if not ordered_source_unit_bytes_sha256:
        raise ValueError("an ingestion payload requires at least one source unit")
    return sha256_identity(
        "oamb-ingestion-payload-v1",
        (tuple(ordered_source_unit_bytes_sha256),),
    )


def ingestion_plan_id(plan_manifest_id: str, payload_hash: str) -> str:
    return sha256_identity("oamb-ingestion-plan-v1", (plan_manifest_id, payload_hash))


def ingestion_occurrence_id(run_id: str, memory_system_id: str, plan_id: str) -> str:
    return sha256_identity("oamb-ingestion-occurrence-v1", (run_id, memory_system_id, plan_id))


def case_occurrence_id(ingestion_occurrence: str, case_manifest_id: str) -> str:
    return sha256_identity("oamb-case-occurrence-v1", (ingestion_occurrence, case_manifest_id))


def attempt_id(
    parent_occurrence_id: str,
    stage: str,
    ordinal_1_indexed: int,
    request_fingerprint: str,
) -> str:
    _require_positive_ordinal(ordinal_1_indexed)
    return sha256_identity(
        "oamb-attempt-v1",
        (parent_occurrence_id, stage, ordinal_1_indexed, request_fingerprint),
    )


def _require_positive_ordinal(value: int) -> None:
    if isinstance(value, bool) or value < 1:
        raise ValueError("ordinal must be a positive 1-indexed integer")
