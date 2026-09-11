from __future__ import annotations

import json
from typing import Any

import pytest

from oamb.artifacts.validation.retrieval_request import retrieval_request_proves_generation_free
from oamb.memory_systems.hindsight.profiles import PROFILE_ID


def _hindsight_proof() -> dict[str, Any]:
    return {
        "schema_name": "oamb_rest_request_proof",
        "schema_version": 1,
        "method": "POST",
        "path": "/v1/default/banks/isolated-bank/memories/recall",
        "params": {},
        "write_intent": False,
        "json_payload": {
            "query": "How long?",
            "types": ["world", "experience"],
            "query_timestamp": "2023-04-18T00:00:00+00:00",
            "trace": True,
            "include": {"entities": None, "chunks": {}},
        },
    }


def test_hindsight_request_proof_accepts_provider_owned_retrieval_volume() -> None:
    assert retrieval_request_proves_generation_free(
        PROFILE_ID, json.dumps(_hindsight_proof()).encode()
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"budget": "high"},
        {"max_tokens": 32768},
        {"include": {"entities": None, "chunks": {"max_tokens": 1000}}},
        {"include": {"entities": None}},
        {"types": ["world", "experience", "observation"]},
        {"reflect": True},
    ],
)
def test_hindsight_request_proof_rejects_size_or_evidence_route_drift(
    changes: dict[str, Any],
) -> None:
    proof = _hindsight_proof()
    proof["json_payload"].update(changes)
    assert not retrieval_request_proves_generation_free(PROFILE_ID, json.dumps(proof).encode())
