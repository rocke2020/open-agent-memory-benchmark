"""Pinned parser for retry-safe structured model-supplier rejection evidence."""

from __future__ import annotations

import json

from .ports import ModelSupplierRejectionClassification

_REJECTION_ROOT_FIELDS = frozenset({"error"})
_REJECTION_FIELDS = frozenset(
    {
        "origin",
        "failure_kind",
        "status",
        "acceptance",
        "provider_mutation",
        "retryable",
        "internal_retry_count",
    }
)


class _DuplicateJsonKey(ValueError):
    pass


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise _DuplicateJsonKey(key)
        document[key] = value
    return document


def parse_structured_supplier_rejection(
    raw_bytes: bytes,
    *,
    status_code: int,
) -> ModelSupplierRejectionClassification | None:
    """Parse only the pinned exact schema; status or message text alone proves nothing."""

    if status_code != 429:
        return None
    try:
        document = json.loads(raw_bytes, object_pairs_hook=_reject_duplicate_keys)
    except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(document, dict) or set(document) != _REJECTION_ROOT_FIELDS:
        return None
    error = document.get("error")
    if not isinstance(error, dict) or set(error) != _REJECTION_FIELDS:
        return None
    if (
        error.get("origin") != "model_supplier"
        or error.get("failure_kind") != "rate_limited"
        or error.get("status") != 429
        or error.get("acceptance") != "not_accepted"
        or error.get("provider_mutation") != "none"
        or error.get("retryable") is not True
        or type(error.get("internal_retry_count")) is not int
        or error["internal_retry_count"] < 0
    ):
        return None
    return ModelSupplierRejectionClassification(
        origin="model_supplier",
        failure_kind="rate_limited",
        status=429,
        acceptance="not_accepted",
        provider_mutation="none",
        retryable=True,
        internal_retry_count=error["internal_retry_count"],
    )


__all__ = ["parse_structured_supplier_rejection"]
