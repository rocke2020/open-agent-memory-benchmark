from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.ports import MemorySystemCallFailure
from oamb.memory_systems.hindsight.adapter import HindsightAdapter
from oamb.memory_systems.mem0.adapter import Mem0RestAdapter
from oamb.memory_systems.openviking.session_adapter import OpenVikingSessionAdapter

REQUEST_TIMEOUT_SECONDS = 0.01
OUTER_TEST_TIMEOUT_SECONDS = 0.1


async def _slow_response(_request: httpx.Request) -> httpx.Response:
    await asyncio.sleep(1)
    return httpx.Response(200, json={})


def _hindsight(tmp_path: Path) -> HindsightAdapter:
    return HindsightAdapter(
        store=ArtifactStore(tmp_path / "hindsight"),
        base_url="https://hindsight.example",
        extraction_model="configured-model",
        runtime_binding_hash="a" * 64,
        transport=httpx.MockTransport(_slow_response),
        read_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        total_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


def _mem0(tmp_path: Path) -> Mem0RestAdapter:
    transport = httpx.MockTransport(_slow_response)
    return Mem0RestAdapter(
        store=ArtifactStore(tmp_path / "mem0"),
        base_url="https://mem0.example",
        api_key="mem0-key",
        inspector_base_url="https://mem0-inspector.example",
        inspector_api_key="inspector-key",
        runtime_binding_hash="b" * 64,
        transport=transport,
        inspector_transport=transport,
        read_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        total_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


def _openviking(tmp_path: Path) -> OpenVikingSessionAdapter:
    return OpenVikingSessionAdapter(
        store=ArtifactStore(tmp_path / "openviking"),
        base_url="https://openviking.example",
        api_key="openviking-key",
        benchmark_account="benchmark-account",
        benchmark_user="benchmark-user",
        runtime_binding_hash="c" * 64,
        transport=httpx.MockTransport(_slow_response),
        read_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
        total_timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("build_adapter", (_hindsight, _mem0, _openviking))
async def test_memory_adapter_enforces_the_supplied_request_timeout(
    tmp_path: Path,
    build_adapter: Callable[[Path], Any],
) -> None:
    adapter = build_adapter(tmp_path)
    try:
        with pytest.raises(MemorySystemCallFailure, match="read timed out"):
            async with asyncio.timeout(OUTER_TEST_TIMEOUT_SECONDS):
                await adapter.resolve()
    finally:
        await adapter.close()
