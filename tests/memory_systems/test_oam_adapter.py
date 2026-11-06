import json

import httpx
import pytest

from oamb.memory_systems.oam import OAMAdapter, pack_messages


def test_packing_preserves_utf8_and_splits_oversized_messages():
    original = [
        {
            "role": "user",
            "content": "猫猫猫",
            "event_time": "2024-01-01T00:00:00Z",
            "learning_eligible": True,
        },
        {
            "role": "assistant",
            "content": "OK",
            "event_time": "2024-01-01T00:00:00Z",
            "learning_eligible": True,
        },
    ]
    batches = pack_messages(original, max_messages=1, max_bytes=6)
    assert [[m["content"] for m in b] for b in batches] == [["猫猫"], ["猫"], ["OK"]]
    assert [m["role"] for b in batches for m in b] == ["user", "user", "assistant"]
    assert all(m["event_time"] == "2024-01-01T00:00:00Z" for b in batches for m in b)


@pytest.mark.parametrize(
    "status,ready", [("failed", False), ("partial", True), ("succeeded", False)]
)
def test_operation_rejects_failed_partial_and_incomplete_success(status, ready):
    def handle(request):
        return httpx.Response(
            200,
            json={
                "operation_id": "op",
                "status": status,
                "readiness": {
                    "source_searchable": ready,
                    "memory_ready": ready,
                    "links_ready": ready,
                    "temporal_links_ready": ready,
                },
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        adapter = OAMAdapter(http, "http://oam", lambda *args: None, timeout=0.1)
        with pytest.raises(ValueError, match="operation"):
            adapter.wait_ready("u", "key", "op")


@pytest.mark.parametrize("mutation", ["generation", "mode", "degraded", "partial"])
def test_recall_rejects_wrong_mode_generation_and_degradation(mutation):
    diagnostics = {
        "generation_id": "g",
        "recall_level": "fast",
        "generative_calls": 0,
        "degraded": False,
        "partial": False,
        "readiness": "ready",
    }
    if mutation == "generation":
        diagnostics["generative_calls"] = 1
    elif mutation == "mode":
        diagnostics["recall_level"] = "normal"
    else:
        diagnostics[mutation] = True
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, json={"context": "evidence", "diagnostics": diagnostics}
            )
        )
    ) as http:
        adapter = OAMAdapter(http, "http://oam", lambda *args: None)
        with pytest.raises(ValueError, match="recall"):
            adapter.recall("u", "key", "whole original question", "fast", "g", None)


def test_retain_is_idempotent_and_never_records_user_secret():
    records = []

    def handle(request):
        assert request.headers["Idempotency-Key"].startswith("oamb-")
        assert json.loads(request.content)["messages"][0]["role"] == "assistant"
        return httpx.Response(202, json={"operation_id": "op", "source_id": "s"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        adapter = OAMAdapter(http, "http://oam", lambda *args: records.append(args))
        payload = [
            {
                "role": "assistant",
                "content": "hello",
                "event_time": "2024-01-01T00:00:00Z",
                "learning_eligible": True,
            }
        ]
        assert adapter.retain("u", "private-key", payload)["operation_id"] == "op"
    assert "private-key" not in json.dumps(records)


def test_recall_accepts_the_real_empty_degradation_list():
    body = {
        "context": "Juniper",
        "diagnostics": {
            "generation_id": "g",
            "recall_level": "fast",
            "generative_calls": 0,
            "degraded": [],
            "partial": False,
            "readiness": "ready",
        },
    }
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    ) as http:
        adapter = OAMAdapter(http, "http://oam", lambda *args: None)
        assert adapter.recall("u", "key", "Name?", "fast", "g", None)["context"] == "Juniper"


def test_identical_payloads_have_distinct_occurrences_but_same_resume_key():
    keys = []

    def handle(request):
        keys.append(request.headers["Idempotency-Key"])
        return httpx.Response(202, json={"operation_id": "op"})

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        adapter = OAMAdapter(http, "http://oam", lambda *args: None)
        for occurrence in ("batch-0", "batch-1", "batch-0"):
            adapter.retain("u", "key", [{"content": "same"}], occurrence_id=occurrence)
    assert keys[0] != keys[1]
    assert keys[0] == keys[2]


def test_successful_operation_with_partial_extraction_is_rejected():
    body = {
        "operation_id": "op",
        "status": "succeeded",
        "readiness": {
            "source_searchable": True,
            "memory_ready": True,
            "links_ready": True,
            "temporal_links_ready": True,
        },
        "stage_state": {"extraction_coverage": "partial"},
    }
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body))
    ) as http:
        adapter = OAMAdapter(http, "http://oam", lambda *args: None)
        with pytest.raises(ValueError, match="extraction"):
            adapter.wait_ready("u", "key", "op")


def test_owner_creation_recovers_after_committed_response_is_lost():
    created = False
    records = []

    def handle(request):
        nonlocal created
        assert json.loads(request.content) == {"user_id": "u"}
        if request.url.path == "/v1/users":
            if not created:
                created = True
                raise httpx.ReadError("response lost after commit", request=request)
            return httpx.Response(
                422,
                json={"detail": {"code": "InvalidRequest", "message": "user_id already exists"}},
            )
        assert request.url.path == "/v1/credentials"
        return httpx.Response(
            201, json={"principal_id": "user:u", "state": "active", "secret": "recovered-key"}
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        adapter = OAMAdapter(http, "http://oam", lambda *args: records.append(args))
        with pytest.raises(httpx.ReadError):
            adapter.create_owner("u", "root-key")
        assert adapter.create_owner("u", "root-key") == "recovered-key"
    assert "recovered-key" not in json.dumps(records)
