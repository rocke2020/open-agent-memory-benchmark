from __future__ import annotations

import asyncio
import gzip
import importlib
import json
import os
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.ports import (
    IngestionRequest,
    MemorySystemPort,
    ModelCallFailure,
    ModelClientPort,
    ModelRequest,
    NativeEvidenceBatch,
    ReadinessRequest,
    RetrievalRequest,
    ScopeAllocationRequest,
    WorkloadPort,
)
from oamb.runtime.preflight import PreflightRejected


def require(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} is not implemented", pytrace=False)


def test_generated_fake_workload_has_one_reused_plan_and_one_partial_plan() -> None:
    fake = require("oamb.workloads.fake")
    workload = fake.GeneratedFakeWorkload()

    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    plans = workload.iter_ingestion_plans(manifest)
    case_plans = workload.iter_case_plans(manifest)

    assert isinstance(workload, WorkloadPort)
    assert len(manifest.logical_contexts) == 2
    assert len(manifest.ingestion_plans) == 2
    assert len(manifest.cases) == 4
    assert tuple(plan.case_manifest_entry_id for plan in case_plans) == tuple(
        case.case_manifest_entry_id for case in manifest.cases
    )
    assert tuple(len(plan.ordered_case_manifest_entry_ids) for plan in plans) == (3, 1)
    assert tuple(len(plan.ordered_source_units) for plan in plans) == (1, 2)
    assert plans[0].ordered_case_manifest_entry_ids == tuple(
        case.case_manifest_entry_id for case in manifest.cases[:3]
    )
    assert plans[1].ordered_case_manifest_entry_ids == (manifest.cases[3].case_manifest_entry_id,)


def test_fake_memory_requires_an_artifact_store_for_raw_reference_closure() -> None:
    fake_memory = require("oamb.memory_systems.fake")

    with pytest.raises(TypeError):
        fake_memory.ScriptedFakeMemorySystem()


def test_fake_preflight_fails_when_the_artifact_fsync_probe_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_run = require("oamb.runtime.fake_run")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("planted fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(PreflightRejected, match="artifact root"):
        fake_run.resolve_fake_preflight(tmp_path / "capsule", "fake-preflight-fsync")


def test_fake_memory_delays_readiness_preserves_order_and_reports_partial_ingest(
    tmp_path: Path,
) -> None:
    fake_workload = require("oamb.workloads.fake")
    fake_memory = require("oamb.memory_systems.fake")
    workload = fake_workload.GeneratedFakeWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    ready_plan, partial_plan = workload.iter_ingestion_plans(manifest)
    memory = fake_memory.ScriptedFakeMemorySystem(
        store=ArtifactStore(tmp_path / "capsule"),
        delayed_readiness_plan_ids=(ready_plan.ingestion_plan_id,),
        partial_ingestion_plan_ids=(partial_plan.ingestion_plan_id,),
    )

    async def exercise() -> None:
        assert isinstance(memory, MemorySystemPort)
        scope = await memory.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id="a" * 64,
                ingestion_plan_id=ready_plan.ingestion_plan_id,
            )
        )
        receipt = await memory.ingest(
            IngestionRequest(scope=scope, ordered_source_units=ready_plan.ordered_source_units)
        )
        first = await memory.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=receipt.accepted_source_unit_ids,
            )
        )
        second = await memory.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=receipt.accepted_source_unit_ids,
            )
        )
        native = await memory.retrieve(
            RetrievalRequest(
                scope=scope,
                case_occurrence_id="b" * 64,
                query_bytes=b"shared answer",
                top_k=3,
            )
        )

        assert first.ready is False
        assert second.ready is True
        assert tuple(candidate.native_rank_1_indexed for candidate in native.candidates) == (1, 2)
        assert tuple(candidate.content for candidate in native.candidates) == (
            "shared answer alpha",
            "shared answer beta",
        )

        partial_scope = await memory.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id="c" * 64,
                ingestion_plan_id=partial_plan.ingestion_plan_id,
            )
        )
        partial = await memory.ingest(
            IngestionRequest(
                scope=partial_scope,
                ordered_source_units=partial_plan.ordered_source_units,
            )
        )
        assert partial.accepted_source_unit_ids == (
            partial_plan.ordered_source_units[0].source_unit_id,
        )
        assert partial.rejected_source_unit_ids == (
            partial_plan.ordered_source_units[1].source_unit_id,
        )
        await memory.close()

    asyncio.run(exercise())


def test_fake_memory_seals_every_returned_raw_reference(tmp_path: Path) -> None:
    fake_workload = require("oamb.workloads.fake")
    fake_memory = require("oamb.memory_systems.fake")
    workload = fake_workload.GeneratedFakeWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    plan = workload.iter_ingestion_plans(manifest)[0]
    store = ArtifactStore(tmp_path / "capsule")
    memory = fake_memory.ScriptedFakeMemorySystem(store=store)

    async def exercise() -> tuple[str, ...]:
        resolved = await memory.resolve()
        scope = await memory.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id="a" * 64,
                ingestion_plan_id=plan.ingestion_plan_id,
            )
        )
        receipt = await memory.ingest(
            IngestionRequest(scope=scope, ordered_source_units=plan.ordered_source_units)
        )
        readiness = await memory.wait_ready(
            ReadinessRequest(
                scope=scope,
                expected_source_unit_ids=receipt.accepted_source_unit_ids,
            )
        )
        projection = await memory.project(scope)
        native = await memory.retrieve(
            RetrievalRequest(
                scope=scope,
                case_occurrence_id="b" * 64,
                query_bytes=b"shared",
                top_k=3,
            )
        )
        await memory.close()
        return (
            resolved.raw_reference.sha256,
            scope.raw_reference.sha256,
            *(item.sha256 for item in receipt.raw_references),
            *(item.sha256 for item in readiness.evidence_references),
            projection.inventory.raw_reference.sha256,
            projection.state_digest.raw_reference.sha256,
            native.raw_reference.sha256,
        )

    returned_hashes = asyncio.run(exercise())
    sealed_hashes = {
        path.name.removesuffix(".json.gz")
        for path in (store.root / "source" / "raw").glob("*.json.gz")
    }
    assert set(returned_hashes) <= sealed_hashes


def test_fake_retrieval_raw_payload_reconstructs_returned_candidates(tmp_path: Path) -> None:
    fake_workload = require("oamb.workloads.fake")
    fake_memory = require("oamb.memory_systems.fake")
    workload = fake_workload.GeneratedFakeWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    plan = workload.iter_ingestion_plans(manifest)[0]
    store = ArtifactStore(tmp_path / "capsule")
    memory = fake_memory.ScriptedFakeMemorySystem(store=store)

    async def exercise() -> NativeEvidenceBatch:
        scope = await memory.allocate_ingestion_scope(
            ScopeAllocationRequest(
                ingestion_occurrence_id="a" * 64,
                ingestion_plan_id=plan.ingestion_plan_id,
            )
        )
        await memory.ingest(
            IngestionRequest(scope=scope, ordered_source_units=plan.ordered_source_units)
        )
        batch = cast(
            NativeEvidenceBatch,
            await memory.retrieve(
                RetrievalRequest(
                    scope=scope,
                    case_occurrence_id="b" * 64,
                    query_bytes=b"shared",
                    top_k=3,
                )
            ),
        )
        await memory.close()
        return batch

    batch = asyncio.run(exercise())
    raw_path = store.root / "source" / "raw" / f"{batch.raw_reference.sha256}.json.gz"
    payload = json.loads(gzip.decompress(raw_path.read_bytes()))

    assert payload["operation"] == "retrieve"
    assert payload["candidates"] == [
        {
            "content": candidate.content,
            "native_id": candidate.native_id,
            "native_rank_1_indexed": candidate.native_rank_1_indexed,
            "native_score": candidate.native_score,
        }
        for candidate in batch.candidates
    ]


def test_scripted_fake_model_seals_failed_usage_then_succeeds_on_explicit_retry(
    tmp_path: Path,
) -> None:
    fake_model = require("oamb.model_clients.fake")
    store = ArtifactStore(tmp_path / "capsule")
    client = fake_model.ScriptedFakeModelClient(store)
    parent_id = "d" * 64
    messages = (("user", "Which token is second?"),)

    async def exercise() -> None:
        assert isinstance(client, ModelClientPort)
        first = ModelRequest(
            attempt_id="1" * 64,
            parent_kind="case",
            parent_id=parent_id,
            stage="answer",
            role_binding_id="fake-answer-v1",
            messages_sha256="2" * 64,
            messages=messages,
        )
        with pytest.raises(ModelCallFailure) as failed:
            await client.complete(first)
        assert failed.value.retryable is True
        assert len(failed.value.usage_reference_ids) == 1

        retry = ModelRequest(
            attempt_id="3" * 64,
            parent_kind="case",
            parent_id=parent_id,
            stage="answer",
            role_binding_id="fake-answer-v1",
            messages_sha256="2" * 64,
            messages=messages,
        )
        receipt = await client.complete(retry)
        assert receipt.output_text == "beta"
        assert len(receipt.usage_reference_ids) == 1
        await client.close()

    asyncio.run(exercise())
    usage_files = tuple((store.root / "source" / "usage").glob("*.json"))
    raw_files = tuple((store.root / "source" / "raw").glob("*.json.gz"))
    assert len(usage_files) == 2
    assert len(raw_files) == 2
