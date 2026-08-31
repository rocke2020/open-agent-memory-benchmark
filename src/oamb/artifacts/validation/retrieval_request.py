"""Exact generation-free retrieval request proof shared by validation and reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from oamb.artifacts.validation.mem0_evidence import MEM0_PROFILE_ID
from oamb.memory_systems.hindsight.profiles import PROFILE_ID as HINDSIGHT_PROFILE_ID
from oamb.memory_systems.openviking.session_adapter import OPENVIKING_SESSION_PROFILE_ID

GENERATION_FREE_PROOF_PROFILES = frozenset(
    {HINDSIGHT_PROFILE_ID, MEM0_PROFILE_ID, OPENVIKING_SESSION_PROFILE_ID}
)


def retrieval_request_proves_generation_free(profile_id: str, payload: bytes) -> bool:
    try:
        proof = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(proof, dict):
        return False
    if proof.get("schema_name") != "oamb_rest_request_proof" or proof.get("schema_version") != 1:
        return False
    if proof.get("method") != "POST" or proof.get("params") != {}:
        return False
    body = proof.get("json_payload")
    if not isinstance(body, dict) or proof.get("write_intent") is not False:
        return False
    path = proof.get("path")
    if profile_id == HINDSIGHT_PROFILE_ID:
        return (
            isinstance(path, str)
            and path.endswith("/memories/recall")
            and set(body)
            == {"query", "types", "budget", "max_tokens", "query_timestamp", "trace", "include"}
            and "reflect" not in body
        )
    if profile_id == MEM0_PROFILE_ID:
        return path == "/search" and set(body) == {
            "query",
            "filters",
            "top_k",
            "threshold",
        }
    if profile_id == OPENVIKING_SESSION_PROFILE_ID:
        return (
            path == "/api/v1/search/find"
            and set(body) == {"query", "target_uri", "context_type", "limit"}
            and body.get("context_type") == "memory"
            and "session_id" not in body
        )
    return False


def _unique_object(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


__all__ = [
    "GENERATION_FREE_PROOF_PROFILES",
    "retrieval_request_proves_generation_free",
]
