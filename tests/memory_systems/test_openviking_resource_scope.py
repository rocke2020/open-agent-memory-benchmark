from __future__ import annotations

import hashlib
import json
from pathlib import Path

import httpx
import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.ports import ScopeAllocationRequest
from oamb.memory_systems.openviking.adapter import OpenVikingRestAdapter


@pytest.mark.asyncio
async def test_resource_scope_allocates_an_isolated_plan_root(tmp_path: Path) -> None:
    peer_key = hashlib.sha256(b"peer\0occurrence-1").hexdigest()
    plan_key = hashlib.sha256(b"plan\0plan-1").hexdigest()
    peer_root = f"viking://user/oamb-admin/peers/oamb-{peer_key}"
    expected_root = f"{peer_root}/resources/oamb/plans/{plan_key}"
    requested_uris: list[str] = []

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "healthy": True,
                    "version": "v0.4.19",
                    "auth_mode": "api_key",
                    "account_id": "oamb-benchmark",
                    "user_id": "oamb-admin",
                    "role": "admin",
                },
            )
        assert request.headers["X-OpenViking-Actor-Peer"] == f"oamb-{peer_key}"
        if request.method == "GET" and request.url.path == "/api/v1/fs/stat":
            uri = request.url.params["uri"]
            requested_uris.append(uri)
            return httpx.Response(
                404,
                json={
                    "status": "error",
                    "result": None,
                    "telemetry": None,
                    "profile": None,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "File not found",
                        "details": {"type": "file", "resource": uri},
                    },
                },
            )
        assert request.method == "POST" and request.url.path == "/api/v1/fs/mkdir"
        assert json.loads(request.content) == {"uri": expected_root}
        return httpx.Response(
            200,
            json={
                "status": "ok",
                "result": {"uri": expected_root},
                "error": None,
                "telemetry": None,
                "profile": None,
            },
        )

    adapter = OpenVikingRestAdapter(
        store=ArtifactStore(tmp_path),
        base_url="http://provider.invalid",
        api_key="test-key",
        benchmark_account="oamb-benchmark",
        benchmark_user="oamb-admin",
        runtime_binding_hash="a" * 64,
        transport=httpx.MockTransport(respond),
    )
    try:
        await adapter.resolve()
        receipt = await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id="occurrence-1", ingestion_plan_id="plan-1"
            )
        )
        assert receipt.scope_id == expected_root
        assert requested_uris == [peer_root, expected_root]
        assert len(receipt.supporting_raw_references) == 2
        with pytest.raises(ValueError, match="cannot replay"):
            await adapter.allocate_ingestion_scope(
                ScopeAllocationRequest(
                    ingestion_occurrence_id="occurrence-1", ingestion_plan_id="plan-1"
                )
            )
    finally:
        await adapter.close()
