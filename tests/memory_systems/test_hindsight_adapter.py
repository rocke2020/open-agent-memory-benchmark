from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import replace
from decimal import Decimal
from importlib import import_module
from itertools import count
from pathlib import Path
from typing import Any

import httpx
import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.accounting import ProofStatus
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import (
    IngestionDispatchRequest,
    IngestionReceipt,
    IngestionRequest,
    MemorySystemCallFailure,
    MemorySystemCallUnknownOutcome,
    MemorySystemReadCancelled,
    RawReferenceHandle,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
    ScopeReceipt,
    SourceUnit,
)
from oamb.runtime.memory_query import (
    MemorySystemProjectionMutation,
    ReadOnlyRetrievalFailure,
    execute_read_only_retrieval,
)

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "adapters" / "hindsight"


def _hindsight_module() -> Any:
    try:
        return import_module("oamb.memory_systems.hindsight.adapter")
    except ModuleNotFoundError:
        pytest.fail("the Hindsight v0.9.2 adapter is not implemented")


def _version_bytes() -> bytes:
    return (FIXTURE_ROOT / "version-v0.9.2.json").read_bytes()


def _fixture_bytes(name: str) -> bytes:
    return (FIXTURE_ROOT / name).read_bytes()


def _bank_config_bytes(bank_id: str) -> bytes:
    return _fixture_bytes("bank-config-v0.9.2.json").replace(b"BANK_ID", bank_id.encode())


def _adapter(tmp_path: Path, handler: httpx.MockTransport) -> Any:
    return _hindsight_module().HindsightAdapter(
        store=ArtifactStore(tmp_path / "capsule"),
        base_url="https://hindsight.example",
        extraction_model="fixture-extractor",
        runtime_binding_hash="f" * 64,
        transport=handler,
    )


def _source(source_id: str, ordinal: int) -> SourceUnit:
    payload = f'[{{"role":"user","content":"source {ordinal}"}}]'.encode()
    return SourceUnit(
        source_unit_id=source_id,
        context_manifest_entry_id="context-1",
        ordinal_1_indexed=ordinal,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=payload,
        source_reference=f"session-{ordinal}",
        occurred_at=f"2026-01-{ordinal:02d}T00:00:00+00:00",
        context_text=f"LongMemEval session session-{ordinal} at 2026-01-{ordinal:02d}T00:00:00+00:00",
        source_metadata=(("question_id", "question-1"), ("session_id", f"session-{ordinal}")),
    )


def _bank_id(ingestion_occurrence_id: str) -> str:
    return hashlib.sha256(f"oamb-hindsight-bank-v1\0{ingestion_occurrence_id}".encode()).hexdigest()


def _dispatch_fingerprint(
    scope_id: str,
    ordinal: int,
    sources: tuple[SourceUnit, ...],
) -> str:
    return canonical_sha256(
        [
            "oamb-hindsight-retain-dispatch-v1",
            scope_id,
            ordinal,
            tuple(
                {
                    "content": source.payload_bytes.decode(),
                    "timestamp": source.occurred_at,
                    "context": source.context_text,
                    "metadata": dict(source.source_metadata),
                    "document_id": source.source_unit_id,
                }
                for source in sources
            ),
        ]
    )


def _bank_response(bank_id: str) -> dict[str, object]:
    return {
        "bank_id": bank_id,
        "name": bank_id,
        "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
        "mission": "",
        "background": "",
    }


async def _resolve_and_allocate(adapter: Any, occurrence_id: str) -> Any:
    await adapter.resolve()
    return await adapter.allocate_ingestion_scope(
        ScopeAllocationRequest(
            ingestion_occurrence_id=occurrence_id,
            ingestion_plan_id="plan-1",
        )
    )


class _HindsightFixtureService:
    def __init__(
        self,
        occurrence_id: str,
        *,
        mutate_after_recall: bool = False,
        unknown_document_field: bool = False,
        mismatched_document_detail: bool = False,
        truncated_document_page: bool = False,
        malformed_recall: bool = False,
        malformed_projection_after_recall: bool = False,
        truncated_projection_after_recall: bool = False,
        reordered_memory_detail_entities: bool = False,
        retain_fixture_name: str = "retain-zero-usage.json",
        lose_retain_receipt: bool = False,
    ) -> None:
        self.bank_id = _bank_id(occurrence_id)
        self.mutate_after_recall = mutate_after_recall
        self.unknown_document_field = unknown_document_field
        self.mismatched_document_detail = mismatched_document_detail
        self.truncated_document_page = truncated_document_page
        self.malformed_recall = malformed_recall
        self.malformed_projection_after_recall = malformed_projection_after_recall
        self.truncated_projection_after_recall = truncated_projection_after_recall
        self.reordered_memory_detail_entities = reordered_memory_detail_entities
        self.retain_fixture_name = retain_fixture_name
        self.lose_retain_receipt = lose_retain_receipt
        self.recall_seen = False
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/version":
            return httpx.Response(200, content=_version_bytes())
        if path == "/v1/default/banks":
            return httpx.Response(200, content=_fixture_bytes("bank-list-empty.json"))
        if request.method == "PUT" and path.endswith(self.bank_id):
            return httpx.Response(200, json=_bank_response(self.bank_id))
        if path.endswith("/config"):
            return httpx.Response(200, content=_bank_config_bytes(self.bank_id))
        if path.endswith("/profile"):
            return httpx.Response(200, json=_bank_response(self.bank_id))
        if request.method == "POST" and path.endswith("/memories/recall"):
            self.recall_seen = True
            if not self.malformed_recall:
                return httpx.Response(200, content=_fixture_bytes("recall-success.json"))
            payload = json.loads(_fixture_bytes("recall-success.json"))
            payload["results"][0]["future_field"] = True
            return httpx.Response(200, json=payload)
        if request.method == "POST" and path.endswith("/memories"):
            if self.lose_retain_receipt:
                raise httpx.ReadTimeout("lost receipt", request=request)
            return httpx.Response(
                200,
                content=_fixture_bytes(self.retain_fixture_name).replace(
                    b"BANK_ID", self.bank_id.encode()
                ),
            )
        if request.method == "GET" and path.endswith("/documents"):
            return self._documents_page(request)
        if request.method == "GET" and path.endswith("/documents/source-1"):
            document = json.loads(
                _fixture_bytes("document-detail.json").replace(b"BANK_ID", self.bank_id.encode())
            )
            if self.mismatched_document_detail:
                document["content_hash"] = "f" * 64
            if self.mutate_after_recall and self.recall_seen:
                document["updated_at"] = "2026-01-01T00:00:09+00:00"
            return httpx.Response(200, json=document)
        if request.method == "GET" and path.endswith("/memories/list"):
            return self._memories_page(request)
        if request.method == "GET" and path.endswith(
            "/memories/11111111-1111-4111-8111-111111111111"
        ):
            detail = json.loads(_fixture_bytes("memory-detail-world.json"))
            if self.reordered_memory_detail_entities:
                detail["entities"] = [
                    "mixed media",
                    "consumption",
                    "pollution",
                    "toy packaging",
                    "waste",
                    "Waste Not, Want Not",
                ]
            return httpx.Response(200, json=detail)
        if request.method == "GET" and path.endswith(
            "/memories/22222222-2222-4222-8222-222222222222"
        ):
            return httpx.Response(200, content=_fixture_bytes("memory-detail-experience.json"))
        if request.method == "GET" and path.endswith("/mental-models"):
            assert dict(request.url.params) == {
                "detail": "full",
                "limit": "1000",
                "offset": "0",
            }
            return httpx.Response(200, content=_fixture_bytes("empty-page.json"))
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    def _documents_page(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        assert params.get("limit") == "1000"
        offset = int(params["offset"])
        projection_malformed = self.unknown_document_field or (
            self.malformed_projection_after_recall and self.recall_seen
        )
        projection_truncated = self.truncated_document_page or (
            self.truncated_projection_after_recall and self.recall_seen
        )
        if offset == 0:
            document = json.loads(
                _fixture_bytes("documents-page.json").replace(b"BANK_ID", self.bank_id.encode())
            )
            if projection_malformed:
                document["items"][0]["future_field"] = True
            if projection_truncated:
                document["total"] = 2
            if self.mutate_after_recall and self.recall_seen:
                document["items"][0]["updated_at"] = "2026-01-01T00:00:09+00:00"
            return httpx.Response(200, json=document)
        expected_total = 2 if projection_truncated else 1
        assert offset == 1
        return httpx.Response(
            200,
            json={"items": [], "total": expected_total, "limit": 1000, "offset": offset},
        )

    def _memories_page(self, request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        assert params.get("limit") == "1000"
        offset = int(params["offset"])
        if params.get("type") == "observation":
            assert offset == 0
            return httpx.Response(200, content=_fixture_bytes("empty-page.json"))
        assert set(params) == {"limit", "offset"}
        if offset == 0:
            page = json.loads(_fixture_bytes("memories-page.json"))
            if self.reordered_memory_detail_entities:
                page["items"][0]["entities"] = (
                    "consumption, Waste Not, Want Not, pollution, waste, toy packaging, mixed media"
                )
            return httpx.Response(200, json=page)
        assert offset == 2
        return httpx.Response(
            200,
            json={"items": [], "total": 2, "limit": 1000, "offset": 2},
        )


async def _ingested_fixture_lifecycle(
    tmp_path: Path,
    service: _HindsightFixtureService,
) -> tuple[Any, ScopeReceipt, tuple[SourceUnit, ...], IngestionReceipt]:
    adapter = _adapter(tmp_path, httpx.MockTransport(service))
    occurrence_id = "a" * 64
    scope = await _resolve_and_allocate(adapter, occurrence_id)
    sources = (_source("source-1", 1),)
    dispatch = adapter.plan_ingestion(IngestionRequest(scope=scope, ordered_source_units=sources))[
        0
    ]
    dispatch_receipt = await adapter.ingest(
        IngestionDispatchRequest(
            scope=scope,
            attempt_id="c" * 64,
            dispatch=dispatch,
        )
    )
    receipt = IngestionReceipt(
        ingestion_occurrence_id=occurrence_id,
        accepted_source_unit_ids=("source-1",),
        rejected_source_unit_ids=(),
        raw_references=(dispatch_receipt.raw_reference,),
        dispatch_receipts=(dispatch_receipt,),
    )
    return adapter, scope, sources, receipt


@pytest.mark.asyncio
async def test_resolve_requires_exact_v092_version_and_feature_closure(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=_version_bytes())

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))

    resolution = await adapter.resolve()
    capabilities = await adapter.capabilities()

    assert [request.method for request in calls] == ["GET"]
    assert [request.url.path for request in calls] == ["/version"]
    assert resolution.memory_system_id == "hindsight"
    assert resolution.runtime_binding_hash == "f" * 64
    assert capabilities.provider_order_preserved is True
    assert capabilities.native_reranking_disabled is True
    assert "complete-projection" in capabilities.capability_ids
    await adapter.close()


@pytest.mark.asyncio
async def test_resolve_rejects_an_unknown_v092_feature_before_scope_allocation(
    tmp_path: Path,
) -> None:
    payload = json.loads(_version_bytes())
    payload["features"]["future_feature"] = True
    raw_response = json.dumps(payload, separators=(",", ":")).encode()
    adapter = _adapter(
        tmp_path,
        httpx.MockTransport(lambda request: httpx.Response(200, content=raw_response)),
    )

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.resolve()
    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_response_bytes == raw_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(raw_response).hexdigest()
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_allocate_scope_proves_absence_then_creates_config_and_profile_in_order(
    tmp_path: Path,
) -> None:
    occurrence_id = "a" * 64
    expected_bank_id = _bank_id(occurrence_id)
    calls: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/version":
            return httpx.Response(200, content=_version_bytes())
        if request.url.path == "/v1/default/banks":
            assert dict(request.url.params) == {"limit": "1000", "offset": "0"}
            return httpx.Response(200, content=_fixture_bytes("bank-list-empty.json"))
        if request.method == "PUT":
            assert json.loads(request.content) == {"enable_observations": False}
            return httpx.Response(200, json=_bank_response(expected_bank_id))
        if request.url.path.endswith("/config"):
            return httpx.Response(200, content=_bank_config_bytes(expected_bank_id))
        if request.url.path.endswith("/profile"):
            return httpx.Response(200, json=_bank_response(expected_bank_id))
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    scope = await _resolve_and_allocate(adapter, occurrence_id)

    assert scope.scope_id == expected_bank_id
    assert len(scope.supporting_raw_references) == 3
    assert (
        scope.supporting_raw_references[0].sha256
        == hashlib.sha256(_fixture_bytes("bank-list-empty.json")).hexdigest()
    )
    assert calls == [
        ("GET", "/version"),
        ("GET", "/v1/default/banks"),
        ("PUT", f"/v1/default/banks/{expected_bank_id}"),
        ("GET", f"/v1/default/banks/{expected_bank_id}/config"),
        ("GET", f"/v1/default/banks/{expected_bank_id}/profile"),
    ]
    await adapter.close()


@pytest.mark.asyncio
async def test_allocate_scope_accepts_unrelated_preexisting_bank_as_isolated_baseline(
    tmp_path: Path,
) -> None:
    occurrence_id = "a" * 64
    expected_bank_id = _bank_id(occurrence_id)
    baseline_bank_id = "b" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/version":
            return httpx.Response(200, content=_version_bytes())
        if request.url.path == "/v1/default/banks":
            offset = int(request.url.params["offset"])
            return httpx.Response(
                200,
                json={
                    "banks": [
                        {
                            "bank_id": baseline_bank_id,
                            "name": baseline_bank_id,
                            "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                            "mission": "",
                            "created_at": "2026-01-01T00:00:00+00:00",
                            "updated_at": "2026-01-01T00:00:00+00:00",
                            "fact_count": 1,
                        }
                    ]
                    if offset == 0
                    else [],
                    "total": 1,
                    "limit": 1000,
                    "offset": offset,
                },
            )
        if request.method == "PUT":
            return httpx.Response(200, json=_bank_response(expected_bank_id))
        if request.url.path.endswith("/config"):
            return httpx.Response(200, content=_bank_config_bytes(expected_bank_id))
        if request.url.path.endswith("/profile"):
            return httpx.Response(200, json=_bank_response(expected_bank_id))
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    scope = await _resolve_and_allocate(adapter, occurrence_id)

    assert scope.scope_id == expected_bank_id
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed_stage", ["inventory", "create", "config", "profile"])
async def test_scope_allocation_validation_failure_preserves_sealed_response(
    tmp_path: Path,
    malformed_stage: str,
) -> None:
    occurrence_id = "a" * 64
    bank_id = _bank_id(occurrence_id)
    malformed_response = b'{"unexpected":true}'

    def response(stage: str, valid: bytes) -> httpx.Response:
        return httpx.Response(
            200,
            content=malformed_response if stage == malformed_stage else valid,
        )

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/version":
            return httpx.Response(200, content=_version_bytes())
        if request.url.path == "/v1/default/banks":
            return response("inventory", _fixture_bytes("bank-list-empty.json"))
        if request.method == "PUT":
            return response(
                "create",
                json.dumps(_bank_response(bank_id), separators=(",", ":")).encode(),
            )
        if request.url.path.endswith("/config"):
            return response("config", _bank_config_bytes(bank_id))
        if request.url.path.endswith("/profile"):
            return response(
                "profile",
                json.dumps(_bank_response(bank_id), separators=(",", ":")).encode(),
            )
        raise AssertionError(f"unexpected request: {request.method} {request.url}")

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    await adapter.resolve()

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.allocate_ingestion_scope(ScopeAllocationRequest(occurrence_id, "plan-1"))

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_response_bytes == malformed_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(malformed_response).hexdigest()
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_scope_allocation_requires_version_resolution_before_any_bank_call(
    tmp_path: Path,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(500)

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="resolve"):
        await adapter.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id="a" * 64,
                ingestion_plan_id="plan-1",
            )
        )
    assert calls == 0
    await adapter.close()


@pytest.mark.asyncio
async def test_lost_bank_create_receipt_retires_allocation_without_replay(
    tmp_path: Path,
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/version":
            return httpx.Response(200, content=_version_bytes())
        if request.url.path == "/v1/default/banks":
            return httpx.Response(200, content=_fixture_bytes("bank-list-empty.json"))
        raise httpx.ReadTimeout("lost bank create receipt", request=request)

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    await adapter.resolve()
    allocation = ScopeAllocationRequest(
        ingestion_occurrence_id="a" * 64,
        ingestion_plan_id="plan-1",
    )

    with pytest.raises(MemorySystemCallUnknownOutcome):
        await adapter.allocate_ingestion_scope(allocation)
    call_count = len(calls)
    with pytest.raises(ValueError, match="attempted|replay"):
        await adapter.allocate_ingestion_scope(allocation)

    assert len(calls) == call_count == 3
    await adapter.close()


@pytest.mark.asyncio
async def test_concurrent_scope_allocation_dispatches_create_once(tmp_path: Path) -> None:
    occurrence_id = "a" * 64
    service = _HindsightFixtureService(occurrence_id)

    async def handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0)
        return service(request)

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    await adapter.resolve()
    allocation = ScopeAllocationRequest(occurrence_id, "plan-1")

    results = await asyncio.gather(
        adapter.allocate_ingestion_scope(allocation),
        adapter.allocate_ingestion_scope(allocation),
        return_exceptions=True,
    )

    assert sum(isinstance(result, ScopeReceipt) for result in results) == 1
    failures = [result for result in results if isinstance(result, BaseException)]
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert re.search("attempted|replay", str(failures[0])) is not None
    assert sum(request.method == "PUT" for request in service.requests) == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_cancelled_allocation_preflight_releases_owner_for_waiter(tmp_path: Path) -> None:
    occurrence_id = "a" * 64
    service = _HindsightFixtureService(occurrence_id)
    first_inventory_started = asyncio.Event()
    block_first_inventory = True

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal block_first_inventory
        if request.url.path == "/v1/default/banks" and block_first_inventory:
            block_first_inventory = False
            service.requests.append(request)
            first_inventory_started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled inventory request unexpectedly resumed")
        return service(request)

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    await adapter.resolve()
    allocation = ScopeAllocationRequest(occurrence_id, "plan-1")
    owner = asyncio.create_task(adapter.allocate_ingestion_scope(allocation))
    await first_inventory_started.wait()
    waiter = asyncio.create_task(adapter.allocate_ingestion_scope(allocation))
    await asyncio.sleep(0)

    owner.cancel()
    with pytest.raises(asyncio.CancelledError):
        await owner
    scope = await waiter

    assert scope.scope_id == _bank_id(occurrence_id)
    assert sum(request.method == "PUT" for request in service.requests) == 1
    await adapter.close()


@pytest.mark.asyncio
async def test_bank_collision_preserves_terminal_inventory_response(tmp_path: Path) -> None:
    occurrence_id = "a" * 64
    bank_id = _bank_id(occurrence_id)
    first_page = json.dumps(
        {
            "banks": [
                {
                    "bank_id": bank_id,
                    "name": bank_id,
                    "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                    "mission": "",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "updated_at": "2026-01-01T00:00:00+00:00",
                    "fact_count": 0,
                    "last_document_at": None,
                    "last_write_at": None,
                }
            ],
            "total": 1,
            "limit": 1000,
            "offset": 0,
        },
        separators=(",", ":"),
    ).encode()
    terminal_page = b'{"banks":[],"total":1,"limit":1000,"offset":1}'
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/version":
            return httpx.Response(200, content=_version_bytes())
        assert request.url.path == "/v1/default/banks"
        response_bytes = first_page if request.url.params["offset"] == "0" else terminal_page
        return httpx.Response(200, content=response_bytes)

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    await adapter.resolve()

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.allocate_ingestion_scope(ScopeAllocationRequest(occurrence_id, "plan-1"))

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_response_bytes == terminal_page
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(terminal_page).hexdigest()
    )
    assert failure.value.supporting_raw_references == (
        RawReferenceHandle(hashlib.sha256(first_page).hexdigest()),
    )
    assert all(request.method != "PUT" for request in calls)
    await adapter.close()


def test_bank_config_rejects_unknown_nested_profile_field() -> None:
    profiles = import_module("oamb.memory_systems.hindsight.profiles")
    bank_id = _bank_id("a" * 64)
    raw_bytes = json.dumps(
        {
            "bank_id": bank_id,
            "config": {
                "enable_observations": False,
                "enable_reranking": False,
                "store_document_text": True,
                "future_score_path": False,
            },
            "overrides": {"enable_observations": False},
        },
        separators=(",", ":"),
    ).encode()

    with pytest.raises(ValueError, match="config|profile|field"):
        profiles.parse_bank_config(raw_bytes, expected_bank_id=bank_id)


def test_bank_config_accepts_exact_v092_complete_profile() -> None:
    profiles = import_module("oamb.memory_systems.hindsight.profiles")
    bank_id = _bank_id("a" * 64)
    raw_bytes = _bank_config_bytes(bank_id)

    profiles.parse_bank_config(raw_bytes, expected_bank_id=bank_id)


def test_bank_page_accepts_v092_fresh_bank_without_activity_timestamps() -> None:
    profiles = import_module("oamb.memory_systems.hindsight.profiles")
    bank_id = _bank_id("a" * 64)
    raw_bytes = json.dumps(
        {
            "banks": [
                {
                    "bank_id": bank_id,
                    "name": bank_id,
                    "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
                    "mission": "",
                    "created_at": "2026-08-30T04:33:21.816160+00:00",
                    "updated_at": "2026-08-30T04:33:21.818630+00:00",
                    "fact_count": 0,
                }
            ],
            "total": 1,
            "limit": 1000,
            "offset": 0,
        },
        separators=(",", ":"),
    ).encode()

    page = profiles.parse_bank_page(raw_bytes, expected_limit=1000, expected_offset=0)

    assert page.bank_ids == (bank_id,)


@pytest.mark.asyncio
async def test_plan_ingestion_rejects_an_unallocated_scope_without_dispatch(
    tmp_path: Path,
) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        raise AssertionError("unallocated Hindsight planning must not dispatch")

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    occurrence_id = "a" * 64
    scope = ScopeReceipt(
        ingestion_occurrence_id=occurrence_id,
        scope_id=_bank_id(occurrence_id),
        raw_reference=RawReferenceHandle("b" * 64),
    )

    with pytest.raises(ValueError, match="allocated|scope"):
        adapter.plan_ingestion(
            IngestionRequest(
                scope=scope,
                ordered_source_units=(_source("source-1", 1),),
            )
        )

    assert calls == []
    await adapter.close()


@pytest.mark.asyncio
async def test_plan_ingestion_freezes_one_source_per_retain(
    tmp_path: Path,
) -> None:
    occurrence_id = "a" * 64
    service = _HindsightFixtureService(occurrence_id)
    adapter = _adapter(tmp_path, httpx.MockTransport(service))
    scope = await _resolve_and_allocate(adapter, occurrence_id)
    sources = tuple(_source(f"source-{ordinal}", ordinal) for ordinal in range(1, 4))

    dispatches = adapter.plan_ingestion(IngestionRequest(scope=scope, ordered_source_units=sources))

    assert [len(dispatch.ordered_source_units) for dispatch in dispatches] == [1, 1, 1]
    assert [dispatch.dispatch_ordinal_1_indexed for dispatch in dispatches] == [1, 2, 3]
    assert all(dispatch.operation_kind == "retain_extraction" for dispatch in dispatches)
    assert len({dispatch.request_fingerprint for dispatch in dispatches}) == 3
    with pytest.raises(ValueError, match="planned|frozen"):
        adapter.plan_ingestion(IngestionRequest(scope=scope, ordered_source_units=sources[:1]))
    await adapter.close()


@pytest.mark.asyncio
async def test_ingest_rejects_self_consistent_dispatch_outside_frozen_plan(
    tmp_path: Path,
) -> None:
    occurrence_id = "a" * 64
    service = _HindsightFixtureService(occurrence_id)
    adapter = _adapter(tmp_path, httpx.MockTransport(service))
    scope = await _resolve_and_allocate(adapter, occurrence_id)
    sources = (_source("source-1", 1), _source("source-2", 2))
    dispatch = adapter.plan_ingestion(IngestionRequest(scope=scope, ordered_source_units=sources))[
        0
    ]
    forged_sources = sources[1:]
    forged_dispatch = replace(
        dispatch,
        ordered_source_units=forged_sources,
        request_fingerprint=_dispatch_fingerprint(
            scope.scope_id,
            dispatch.dispatch_ordinal_1_indexed,
            forged_sources,
        ),
    )
    call_count = len(service.requests)

    with pytest.raises(ValueError, match="frozen plan|planned dispatch"):
        await adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="c" * 64,
                dispatch=forged_dispatch,
            )
        )

    assert len(service.requests) == call_count
    await adapter.close()


@pytest.mark.asyncio
async def test_readiness_rejects_a_truncated_frozen_dispatch_set_before_projection(
    tmp_path: Path,
) -> None:
    occurrence_id = "a" * 64
    service = _HindsightFixtureService(occurrence_id)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/memories"):
            service.requests.append(request)
            payload = json.loads(
                _fixture_bytes("retain-zero-usage.json").replace(
                    b"BANK_ID", service.bank_id.encode()
                )
            )
            payload["items_count"] = len(json.loads(request.content)["items"])
            return httpx.Response(200, json=payload)
        return service(request)

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    scope = await _resolve_and_allocate(adapter, occurrence_id)
    sources = tuple(_source(f"source-{ordinal}", ordinal) for ordinal in range(1, 22))
    dispatches = adapter.plan_ingestion(IngestionRequest(scope=scope, ordered_source_units=sources))
    first_receipt = await adapter.ingest(
        IngestionDispatchRequest(
            scope=scope,
            attempt_id="c" * 64,
            dispatch=dispatches[0],
        )
    )
    truncated_source_ids = first_receipt.accepted_source_unit_ids
    truncated_receipt = IngestionReceipt(
        ingestion_occurrence_id=occurrence_id,
        accepted_source_unit_ids=truncated_source_ids,
        rejected_source_unit_ids=(),
        raw_references=(first_receipt.raw_reference,),
        dispatch_receipts=(first_receipt,),
    )
    call_count = len(service.requests)

    with pytest.raises(ValueError, match="planned|dispatch|source"):
        await adapter.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=truncated_source_ids,
                ingestion_receipt=truncated_receipt,
            )
        )

    assert len(service.requests) == call_count
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("fixture_name", "expected_status"),
    [
        ("retain-success-one-source.json", ProofStatus.MEASURED_PARTIAL),
        ("retain-zero-usage.json", ProofStatus.UNAVAILABLE),
    ],
)
async def test_ingest_dispatches_exact_retain_and_maps_v3_usage(
    tmp_path: Path,
    fixture_name: str,
    expected_status: ProofStatus,
) -> None:
    occurrence_id = "a" * 64
    bank_id = _bank_id(occurrence_id)
    sources = (_source("source-1", 1),)
    service = _HindsightFixtureService(
        occurrence_id,
        retain_fixture_name=fixture_name,
    )
    adapter = _adapter(tmp_path, httpx.MockTransport(service))
    scope = await _resolve_and_allocate(adapter, occurrence_id)
    dispatch = adapter.plan_ingestion(IngestionRequest(scope=scope, ordered_source_units=sources))[
        0
    ]

    receipt = await adapter.ingest(
        IngestionDispatchRequest(
            scope=scope,
            attempt_id="c" * 64,
            dispatch=dispatch,
        )
    )

    retain_calls = [
        request
        for request in service.requests
        if request.method == "POST" and request.url.path.endswith("/memories")
    ]
    assert len(retain_calls) == 1
    assert retain_calls[0].url.path == f"/v1/default/banks/{bank_id}/memories"
    body = json.loads(retain_calls[0].content)
    assert body == {
        "items": [
            {
                "content": source.payload_bytes.decode(),
                "timestamp": source.occurred_at,
                "context": source.context_text,
                "metadata": dict(source.source_metadata),
                "document_id": source.source_unit_id,
            }
            for source in sources
        ],
        "async": False,
    }
    assert receipt.accepted_source_unit_ids == tuple(source.source_unit_id for source in sources)
    assert receipt.rejected_source_unit_ids == ()
    assert receipt.usage_records[0].proof_status == expected_status
    assert receipt.usage_records[0].billing_complete is False
    if expected_status == ProofStatus.MEASURED_PARTIAL:
        assert receipt.usage_records[0].covered_dimensions == (
            "input_tokens",
            "visible_output_tokens",
            "supplier_reported_total_tokens",
        )
    else:
        assert receipt.usage_records[0].input_tokens is None
        assert receipt.usage_records[0].reason == "provider_zero_usage_sentinel"
    await adapter.close()


@pytest.mark.asyncio
async def test_lost_retain_response_is_unknown_outcome_and_never_retried(tmp_path: Path) -> None:
    occurrence_id = "a" * 64
    service = _HindsightFixtureService(occurrence_id, lose_retain_receipt=True)
    adapter = _adapter(tmp_path, httpx.MockTransport(service))
    scope = await _resolve_and_allocate(adapter, occurrence_id)
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(_source("source-1", 1),))
    )[0]

    with pytest.raises(MemorySystemCallUnknownOutcome):
        await adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="c" * 64,
                dispatch=dispatch,
            )
        )
    with pytest.raises(ValueError, match="replay|attempted"):
        await adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="d" * 64,
                dispatch=dispatch,
            )
        )
    with pytest.raises(ValueError, match="fingerprint|dispatch"):
        await adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="e" * 64,
                dispatch=replace(dispatch, request_fingerprint="f" * 64),
            )
        )
    assert (
        sum(
            request.method == "POST" and request.url.path.endswith("/memories")
            for request in service.requests
        )
        == 1
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_retain_validation_failure_preserves_sealed_response(tmp_path: Path) -> None:
    occurrence_id = "a" * 64
    service = _HindsightFixtureService(occurrence_id)
    malformed_response = b'{"unexpected":true}'

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST" and request.url.path.endswith("/memories"):
            service.requests.append(request)
            return httpx.Response(200, content=malformed_response)
        return service(request)

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    scope = await _resolve_and_allocate(adapter, occurrence_id)
    dispatch = adapter.plan_ingestion(
        IngestionRequest(scope=scope, ordered_source_units=(_source("source-1", 1),))
    )[0]

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.ingest(
            IngestionDispatchRequest(
                scope=scope,
                attempt_id="c" * 64,
                dispatch=dispatch,
            )
        )

    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_response_bytes == malformed_response
    assert failure.value.raw_reference == RawReferenceHandle(
        hashlib.sha256(malformed_response).hexdigest()
    )
    await adapter.close()


@pytest.mark.asyncio
async def test_readiness_projection_and_recall_close_the_exact_v092_lifecycle(
    tmp_path: Path,
) -> None:
    service = _HindsightFixtureService("a" * 64)
    adapter, scope, sources, ingestion_receipt = await _ingested_fixture_lifecycle(
        tmp_path, service
    )

    readiness = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=("source-1",),
            ingestion_receipt=ingestion_receipt,
        )
    )
    ticks = count()
    query_receipt = await execute_read_only_retrieval(
        memory=adapter,
        request=RetrievalRequest(
            scope=scope,
            case_occurrence_id="d" * 64,
            query_bytes=b"What does Alice care about?",
            top_k=1,
            query_timestamp="2026-01-02T00:00:00+00:00",
        ),
        clock=lambda: Decimal(next(ticks)),
    )
    batch = query_receipt.native_batch
    await adapter.close()

    assert readiness.ready is True
    assert readiness.ingestion_occurrence_id == scope.ingestion_occurrence_id
    assert readiness.evidence_references[0] == ingestion_receipt.raw_references[0]
    assert len(readiness.evidence_references) == 10
    assert [candidate.native_id for candidate in batch.candidates] == [
        "22222222-2222-4222-8222-222222222222",
        "11111111-1111-4111-8111-111111111111",
    ]
    assert [candidate.native_rank_1_indexed for candidate in batch.candidates] == [1, 2]
    assert [candidate.source_unit_id for candidate in batch.candidates] == [
        sources[0].source_unit_id,
        sources[0].source_unit_id,
    ]
    assert [candidate.evidence_kind for candidate in batch.candidates] == [
        "experience",
        "world",
    ]
    assert [candidate.native_reference for candidate in batch.candidates] == [
        "44444444-4444-4444-8444-444444444444",
        "33333333-3333-4333-8333-333333333333",
    ]
    assert len(batch.candidates) == 2
    assert batch.supporting_raw_references == ()
    assert batch.request_raw_reference is not None
    assert batch.request_raw_reference != batch.raw_reference
    assert len(query_receipt.before_projection.supporting_raw_references) == 9
    assert len(query_receipt.after_projection.supporting_raw_references) == 9
    assert query_receipt.timing.provider_request.seconds == Decimal("1")
    assert query_receipt.timing.projection_verification.seconds == Decimal("2")
    recall_request = next(
        request for request in service.requests if request.url.path.endswith("/memories/recall")
    )
    assert json.loads(recall_request.content) == {
        "query": "What does Alice care about?",
        "types": ["world", "experience"],
        "budget": "high",
        "max_tokens": 32768,
        "query_timestamp": "2026-01-02T00:00:00+00:00",
        "trace": True,
        "include": {"entities": None, "chunks": {}},
    }
    assert (
        batch.raw_reference.sha256
        == hashlib.sha256(_fixture_bytes("recall-success.json")).hexdigest()
    )
    assert batch.raw_reference not in batch.supporting_raw_references
    assert {request.method for request in service.requests}.isdisjoint({"DELETE", "PATCH"})


@pytest.mark.asyncio
async def test_readiness_accepts_live_memory_entity_order_drift_between_list_and_detail(
    tmp_path: Path,
) -> None:
    """Catches treating Hindsight's unordered entity set as an ordered wire field."""

    service = _HindsightFixtureService(
        "a" * 64,
        reordered_memory_detail_entities=True,
    )
    adapter, scope, _sources, ingestion_receipt = await _ingested_fixture_lifecycle(
        tmp_path,
        service,
    )

    readiness = await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=("source-1",),
            ingestion_receipt=ingestion_receipt,
        )
    )

    assert readiness.ready is True
    await adapter.close()


@pytest.mark.asyncio
async def test_readiness_rejects_receipt_binding_before_projection(tmp_path: Path) -> None:
    service = _HindsightFixtureService("a" * 64)
    adapter, scope, _, ingestion_receipt = await _ingested_fixture_lifecycle(tmp_path, service)
    call_count = len(service.requests)

    with pytest.raises(ValueError, match="receipt|source"):
        await adapter.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=("different-source",),
                ingestion_receipt=ingestion_receipt,
            )
        )

    assert len(service.requests) == call_count

    forged_dispatch_receipt = replace(
        ingestion_receipt.dispatch_receipts[0],
        raw_response_bytes=b"{}",
    )
    forged_ingestion_receipt = replace(
        ingestion_receipt,
        dispatch_receipts=(forged_dispatch_receipt,),
    )
    with pytest.raises(ValueError, match="raw|hash|receipt"):
        await adapter.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=("source-1",),
                ingestion_receipt=forged_ingestion_receipt,
            )
        )
    assert len(service.requests) == call_count
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("service_options", "message"),
    [
        ({"unknown_document_field": True}, "field"),
        ({"mismatched_document_detail": True}, "detail|content_hash"),
        ({"truncated_document_page": True}, "pagination|total"),
    ],
)
async def test_readiness_projection_fails_closed_on_planted_wire_drift(
    tmp_path: Path,
    service_options: dict[str, Any],
    message: str,
) -> None:
    service = _HindsightFixtureService("a" * 64, **service_options)
    adapter, scope, _, ingestion_receipt = await _ingested_fixture_lifecycle(tmp_path, service)

    with pytest.raises(MemorySystemCallFailure) as failure:
        await adapter.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=("source-1",),
                ingestion_receipt=ingestion_receipt,
            )
        )
    assert failure.value.failure_kind == "response_validation"
    assert failure.value.raw_reference is not None
    assert isinstance(failure.value.__cause__, ValueError)
    assert re.search(message, str(failure.value.__cause__)) is not None
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure_mode", "expected_failure_kind"),
    [
        ("http_status", "http_status"),
        ("transport_error", "transport_error"),
        ("cancelled", "cancelled"),
    ],
)
async def test_projection_call_failure_preserves_prior_response_references(
    tmp_path: Path,
    failure_mode: str,
    expected_failure_kind: str,
) -> None:
    occurrence_id = "a" * 64
    service = _HindsightFixtureService(occurrence_id)
    failure_response = b'{"error":"projection failed"}'
    fail_document_detail = False
    post_projection_page_references: list[RawReferenceHandle] = []
    failed_detail_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if (
            fail_document_detail
            and request.method == "GET"
            and request.url.path.endswith("/documents/source-1")
        ):
            service.requests.append(request)
            if failure_mode == "http_status":
                return httpx.Response(500, content=failure_response)
            if failure_mode == "transport_error":
                raise httpx.ConnectError("projection transport failed", request=request)
            failed_detail_started.set()
            await asyncio.Event().wait()
            raise AssertionError("cancelled projection detail unexpectedly resumed")
        response = service(request)
        if fail_document_detail and request.url.path.endswith("/documents"):
            post_projection_page_references.append(
                RawReferenceHandle(hashlib.sha256(response.content).hexdigest())
            )
        return response

    adapter = _adapter(tmp_path, httpx.MockTransport(handler))
    scope = await _resolve_and_allocate(adapter, occurrence_id)
    sources = (_source("source-1", 1),)
    dispatch = adapter.plan_ingestion(IngestionRequest(scope=scope, ordered_source_units=sources))[
        0
    ]
    dispatch_receipt = await adapter.ingest(
        IngestionDispatchRequest(
            scope=scope,
            attempt_id="c" * 64,
            dispatch=dispatch,
        )
    )
    ingestion_receipt = IngestionReceipt(
        ingestion_occurrence_id=occurrence_id,
        accepted_source_unit_ids=("source-1",),
        rejected_source_unit_ids=(),
        raw_references=(dispatch_receipt.raw_reference,),
        dispatch_receipts=(dispatch_receipt,),
    )
    await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=("source-1",),
            ingestion_receipt=ingestion_receipt,
        )
    )
    fail_document_detail = True

    call_failure: MemorySystemCallFailure | MemorySystemReadCancelled
    if failure_mode == "cancelled":
        projection_task = asyncio.create_task(adapter.project(scope))
        await failed_detail_started.wait()
        projection_task.cancel()
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await projection_task
        assert isinstance(cancelled.value, MemorySystemReadCancelled)
        call_failure = cancelled.value
    else:
        with pytest.raises(MemorySystemCallFailure) as failure:
            await adapter.project(scope)
        call_failure = failure.value

    assert call_failure.failure_kind == expected_failure_kind
    assert call_failure.raw_response_bytes == (
        failure_response if failure_mode == "http_status" else None
    )
    assert len(post_projection_page_references) == 2
    assert call_failure.supporting_raw_references[:2] == tuple(post_projection_page_references)
    await adapter.close()


@pytest.mark.asyncio
async def test_retrieve_rejects_query_side_projection_mutation(tmp_path: Path) -> None:
    service = _HindsightFixtureService("a" * 64, mutate_after_recall=True)
    adapter, scope, _, ingestion_receipt = await _ingested_fixture_lifecycle(tmp_path, service)
    await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=("source-1",),
            ingestion_receipt=ingestion_receipt,
        )
    )

    ticks = count()
    with pytest.raises(MemorySystemProjectionMutation):
        await execute_read_only_retrieval(
            memory=adapter,
            request=RetrievalRequest(
                scope=scope,
                case_occurrence_id="d" * 64,
                query_bytes=b"What does Alice care about?",
                top_k=1,
                query_timestamp="2026-01-02T00:00:00+00:00",
            ),
            clock=lambda: Decimal(next(ticks)),
        )
    await adapter.close()


@pytest.mark.asyncio
async def test_retrieve_preserves_sealed_raw_reference_on_response_validation_failure(
    tmp_path: Path,
) -> None:
    service = _HindsightFixtureService("a" * 64, malformed_recall=True)
    adapter, scope, _, ingestion_receipt = await _ingested_fixture_lifecycle(tmp_path, service)
    await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=("source-1",),
            ingestion_receipt=ingestion_receipt,
        )
    )

    ticks = count()
    with pytest.raises(ReadOnlyRetrievalFailure) as failure:
        await execute_read_only_retrieval(
            memory=adapter,
            request=RetrievalRequest(
                scope=scope,
                case_occurrence_id="d" * 64,
                query_bytes=b"What does Alice care about?",
                top_k=1,
                query_timestamp="2026-01-02T00:00:00+00:00",
            ),
            clock=lambda: Decimal(next(ticks)),
        )

    receipt = failure.value.receipt
    assert receipt.after_projection is not None
    assert isinstance(receipt.provider_error, MemorySystemCallFailure)
    provider_error = receipt.provider_error
    assert provider_error.failure_kind == "response_validation"
    assert provider_error.raw_reference is not None
    assert provider_error.raw_response_bytes is not None
    assert (
        provider_error.raw_reference.sha256
        == hashlib.sha256(provider_error.raw_response_bytes).hexdigest()
    )
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "service_options",
    [
        {"malformed_projection_after_recall": True},
        {"truncated_projection_after_recall": True},
    ],
)
async def test_post_projection_validation_failure_preserves_ordered_raw_evidence(
    tmp_path: Path,
    service_options: dict[str, Any],
) -> None:
    service = _HindsightFixtureService("a" * 64, **service_options)
    adapter, scope, _, ingestion_receipt = await _ingested_fixture_lifecycle(tmp_path, service)
    await adapter.wait_ready(
        ReadinessRequest(
            scope=scope,
            expected_source_unit_ids=("source-1",),
            ingestion_receipt=ingestion_receipt,
        )
    )
    ticks = count()

    with pytest.raises(ReadOnlyRetrievalFailure) as failure:
        await execute_read_only_retrieval(
            memory=adapter,
            request=RetrievalRequest(
                scope=scope,
                case_occurrence_id="d" * 64,
                query_bytes=b"What does Alice care about?",
                top_k=1,
                query_timestamp="2026-01-02T00:00:00+00:00",
            ),
            clock=lambda: Decimal(next(ticks)),
        )

    receipt = failure.value.receipt
    assert receipt.native_batch is not None
    assert (
        receipt.native_batch.raw_reference.sha256
        == hashlib.sha256(_fixture_bytes("recall-success.json")).hexdigest()
    )
    assert isinstance(receipt.post_projection_error, MemorySystemCallFailure)
    projection_error = receipt.post_projection_error
    assert projection_error.failure_kind == "response_validation"
    assert projection_error.raw_reference is not None
    assert projection_error.raw_response_bytes is not None
    assert (
        projection_error.raw_reference.sha256
        == hashlib.sha256(projection_error.raw_response_bytes).hexdigest()
    )
    if service_options.get("truncated_projection_after_recall"):
        assert projection_error.supporting_raw_references
    assert receipt.timing.provider_request.seconds == Decimal("1")
    assert receipt.timing.projection_verification.seconds == Decimal("2")
    await adapter.close()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload["results"].append(payload["results"][0]), "duplicate"),
        (lambda payload: payload["results"][0].update({"future_field": True}), "field"),
        (lambda payload: payload["results"][0].update({"document_id": "unknown"}), "document"),
        (
            lambda payload: payload["results"][0].update(
                {
                    "scores": {
                        "final": 0.9,
                        "reranker": 0.9,
                        "semantic": 0.8,
                        "keyword": None,
                    }
                }
            ),
            "rerank",
        ),
    ],
)
def test_recall_normalization_fails_closed_on_planted_result_drift(
    mutate: Any,
    message: str,
) -> None:
    payload = json.loads(_fixture_bytes("recall-success.json"))
    mutate(payload)
    try:
        normalize = import_module("oamb.memory_systems.hindsight.normalize")
    except ModuleNotFoundError:
        pytest.fail("the Hindsight recall normalizer is not implemented")

    with pytest.raises(ValueError, match=message):
        normalize.normalize_recall(
            json.dumps(payload, separators=(",", ":")).encode(),
            document_to_source_unit={"source-1": "source-1"},
        )


def test_recall_normalization_accepts_live_empty_optional_field_shape() -> None:
    normalize = import_module("oamb.memory_systems.hindsight.normalize")

    assert (
        normalize.normalize_recall(
            _fixture_bytes("recall-empty-live-v0.9.2.json"),
            document_to_source_unit={},
        )
        == ()
    )


def test_recall_normalization_accepts_live_omitted_nullable_result_fields() -> None:
    normalize = import_module("oamb.memory_systems.hindsight.normalize")

    candidates = normalize.normalize_recall(
        _fixture_bytes("recall-result-live-v0.9.2.json"),
        document_to_source_unit={"source-1": "source-unit-1"},
    )

    assert len(candidates) == 1
    assert candidates[0].source_unit_id == "source-unit-1"
    assert candidates[0].native_score == "0.9"


def test_recall_normalization_accepts_budget_limited_chunk_hydration() -> None:
    normalize = import_module("oamb.memory_systems.hindsight.normalize")
    payload = json.loads(_fixture_bytes("recall-result-live-v0.9.2.json"))
    payload["chunks"] = {
        "prefilter-chunk": {
            "id": "prefilter-chunk",
            "text": "A hydrated chunk whose fact did not fit the result budget.",
            "chunk_index": 0,
            "truncated": False,
        }
    }

    candidates = normalize.normalize_recall(
        json.dumps(payload, separators=(",", ":")).encode(),
        document_to_source_unit={"source-1": "source-unit-1"},
    )

    assert len(candidates) == 1
    assert candidates[0].native_reference == "chunk-1"
    assert candidates[0].native_truncated is False


def test_recall_normalization_does_not_relabel_a_complete_fact_as_truncated() -> None:
    normalize = import_module("oamb.memory_systems.hindsight.normalize")
    payload = json.loads(_fixture_bytes("recall-result-live-v0.9.2.json"))
    payload["chunks"] = {
        "chunk-1": {
            "id": "chunk-1",
            "text": "A source chunk cut only by the independent hydration budget.",
            "chunk_index": 0,
            "truncated": True,
        }
    }

    candidates = normalize.normalize_recall(
        json.dumps(payload, separators=(",", ":")).encode(),
        document_to_source_unit={"source-1": "source-unit-1"},
    )

    assert len(candidates) == 1
    assert candidates[0].content == "The memory conformance code is ALPHA-417."
    assert candidates[0].native_reference == "chunk-1"
    assert candidates[0].native_truncated is False
