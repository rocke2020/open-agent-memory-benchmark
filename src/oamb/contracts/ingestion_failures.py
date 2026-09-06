"""Narrow terminal-ingestion failure recognition for the pinned REST profiles.

Runtime producers and offline validators use the same raw-receipt policy. These
shapes prove that the failed operation settled; they do not prove no mutation.
Fresh-scope ownership and the verified source/runtime binding are separate gates.
"""

from __future__ import annotations

import json
import re
from typing import Literal

HINDSIGHT_SETTLEMENT_BASIS = "hindsight_sync_extraction_drained_v1"
MEM0_SETTLEMENT_BASIS = "mem0_sync_add_returned_v1"
OPENVIKING_SETTLEMENT_BASIS = "openviking_task_work_drained_v1"

_HINDSIGHT_FAILURE_SUMMARY_LIMIT = 5
_HINDSIGHT_AGGREGATE = re.compile(
    r"Fact extraction failed: ([1-9][0-9]*)/([1-9][0-9]*) chunks failed\. "
    r"First failures: (.+)"
)
_HINDSIGHT_CONNECTION_FAILURE = re.compile(
    r"chunk (0|[1-9][0-9]*): APIConnectionError: Connection error\."
)
_HINDSIGHT_INVALID_JSON_FAILURE = re.compile(
    r"chunk (0|[1-9][0-9]*): JSONDecodeError: Invalid control character at: "
    r"line [1-9][0-9]* column [1-9][0-9]* \(char (?:0|[1-9][0-9]*)\)"
)
_MEM0_REQUEST_ID = re.compile(r"[0-9a-f]{8}")
_OPENVIKING_WRAPPER_FIELDS = frozenset({"status", "result", "error", "profile", "telemetry"})

SettledFailureKind = Literal[
    "supplier_connection", "supplier_rate_limit", "supplier_invalid_json_output"
]


def classify_settled_ingestion_failure(
    *,
    settlement_basis: str,
    status_code: int,
    raw_response_bytes: bytes,
    internal_retry_count: int | None,
    expected_task_id: str | None = None,
    expected_session_id: str | None = None,
) -> SettledFailureKind | None:
    """Recognize only a complete pinned receipt with proven internal retries off."""

    if (
        type(internal_retry_count) is not int
        or internal_retry_count != 0
        or type(status_code) is not int
        or not isinstance(raw_response_bytes, bytes)
        or not raw_response_bytes
    ):
        return None
    try:
        value = json.loads(
            raw_response_bytes,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (ValueError, UnicodeError):
        return None
    if not isinstance(value, dict):
        return None

    if settlement_basis == HINDSIGHT_SETTLEMENT_BASIS:
        if status_code == 500:
            return _hindsight_extraction_failure(value)
    elif settlement_basis == MEM0_SETTLEMENT_BASIS:
        if (
            status_code != 502
            or set(value) != {"detail", "code", "request_id"}
            or not isinstance(value["detail"], str)
            or not value["detail"].strip()
            or not isinstance(value["request_id"], str)
            or _MEM0_REQUEST_ID.fullmatch(value["request_id"]) is None
        ):
            return None
        if value["code"] == "provider_unavailable":
            return "supplier_connection"
        if value["code"] == "provider_rate_limited":
            return "supplier_rate_limit"
    elif settlement_basis == OPENVIKING_SETTLEMENT_BASIS:
        if (
            status_code == 200
            and isinstance(expected_task_id, str)
            and bool(expected_task_id)
            and isinstance(expected_session_id, str)
            and bool(expected_session_id)
            and set(value) <= _OPENVIKING_WRAPPER_FIELDS
            and value.get("status") == "ok"
            and value.get("error") is None
            and isinstance(value.get("result"), dict)
        ):
            task = value["result"]
            if (
                task.get("task_id") == expected_task_id
                and task.get("resource_id") == expected_session_id
                and task.get("task_type") == "session_commit"
                and task.get("status") == "failed"
                and task.get("stage") == "failed"
                and task.get("error") == "Connection error."
            ):
                return "supplier_connection"
    return None


def _hindsight_extraction_failure(value: dict[str, object]) -> SettledFailureKind | None:
    if set(value) != {"detail"} or not isinstance(value["detail"], str):
        return None
    aggregate = _HINDSIGHT_AGGREGATE.fullmatch(value["detail"])
    if aggregate is None:
        return None
    try:
        failed, total = int(aggregate[1]), int(aggregate[2])
        if failed > min(total, _HINDSIGHT_FAILURE_SUMMARY_LIMIT):
            return None
        failures = aggregate[3].split(", ")
        if len(failures) != failed:
            return None
        indices: set[int] = set()
        failure_kind: SettledFailureKind | None = None
        for failure in failures:
            match = _HINDSIGHT_CONNECTION_FAILURE.fullmatch(failure)
            member_kind: SettledFailureKind = "supplier_connection"
            if match is None:
                match = _HINDSIGHT_INVALID_JSON_FAILURE.fullmatch(failure)
                member_kind = "supplier_invalid_json_output"
            if match is None or (failure_kind is not None and member_kind != failure_kind):
                return None
            index = int(match[1])
            if index >= total or index in indices:
                return None
            indices.add(index)
            failure_kind = member_kind
    except ValueError:
        return None
    return failure_kind


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("non-finite JSON constant")
