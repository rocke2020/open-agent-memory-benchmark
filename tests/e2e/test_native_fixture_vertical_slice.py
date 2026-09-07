from __future__ import annotations

import asyncio
import gzip
import hashlib
import importlib
import json
import multiprocessing
import os
import queue
import shutil
import threading
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

import oamb.runtime.native_run as native_run_module
from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.artifacts.validation.source_root import validate_source_root
from oamb.contracts.accounting import (
    ProofStatus,
    TokenDomain,
    TokenMeasurementSource,
    TokenStageV2,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
    TokenUsageRecordV5,
)
from oamb.contracts.evidence import (
    CapsuleManifest,
    IngestionPlanRecordV3,
    OccurrenceClaimRecord,
    RunLeaseRecord,
    history_attempt_id,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    AnswerValue,
    ArtifactSealReceipt,
    ArtifactStorePort,
    ArtifactWriteRequest,
    CasePlan,
    DeterministicEvaluation,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionPlan,
    InventoryReceipt,
    JudgeRequest,
    MemorySystemCallCancelledBeforeDispatch,
    MemorySystemCallCancelledUnknownOutcome,
    MemorySystemCallUnknownOutcome,
    ModelReceipt,
    ModelRequest,
    NativeEvidenceBatch,
    NativeEvidenceCandidate,
    ProjectionReceipt,
    RawPayloadSealRequest,
    RetrievalRequest,
    RuntimeResolution,
    ScopeAllocationRequest,
    ScopeReceipt,
    StateDigestReceipt,
    ThinkingEffort,
    VisibleEvidence,
    VisibleEvidencePolicy,
)
from oamb.contracts.specifications import INFRASTRUCTURE_RETRY_POLICY_HASH
from oamb.contracts.states import ValidationDisposition
from oamb.memory_systems.fake import ScriptedFakeMemorySystem
from oamb.memory_systems.mem0 import Mem0RestAdapter
from oamb.reporting import native_reduce
from oamb.runtime.case_partition import build_case_partition_spec
from oamb.runtime.native_run import (
    NativeCancellationEvidenceError,
    NativeRunArtifacts,
    NativeRunInterrupted,
    NativeRunProcessError,
    run_native_vertical_slice,
)
from oamb.workloads.fake import GeneratedFakeWorkload
from oamb.workloads.visible_evidence import (
    LME_VISIBLE_EVIDENCE_POLICY,
    build_lme_visible_evidence,
)
from tests.unit.test_openai_compatible_model_client import _client
from tests.unit.test_t10_native_run_control import _control

HARD_CLOSE_STAGE_REACH_BOUND_SECONDS = 2.0
HARD_CLOSE_RETURN_BOUND_SECONDS = 0.30


class _NativeFixtureWorkload(GeneratedFakeWorkload):
    def build_visible_evidence(
        self,
        native_batch: NativeEvidenceBatch,
        policy: VisibleEvidencePolicy,
    ) -> VisibleEvidence:
        return build_lme_visible_evidence(native_batch, policy)

    def evaluate(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
    ) -> DeterministicEvaluation | JudgeRequest:
        reference_payload = case_plan.reference_payload
        metric_id = case_plan.metric_id
        numerator = int(answer.parsed_value == reference_payload)
        trace = canonical_json_bytes(
            {
                "metric_id": metric_id,
                "numerator": numerator,
                "denominator": 1,
                "parsed_answer_sha256": answer.parsed_value_sha256,
            }
        )
        return DeterministicEvaluation(
            metric_id=metric_id,
            result_sha256=hashlib.sha256(trace).hexdigest(),
            numerator=numerator,
            denominator=1,
            trace_bytes=trace,
        )


class _NativeFixtureJudgeWorkload(_NativeFixtureWorkload):
    def evaluate(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
    ) -> DeterministicEvaluation | JudgeRequest:
        if case_plan.judge_binding_id is not None:
            return GeneratedFakeWorkload.evaluate(self, case_plan, answer)
        return super().evaluate(case_plan, answer)


class _RecordedNativeMemory(ScriptedFakeMemorySystem):
    async def ingest(
        self,
        request: IngestionDispatchRequest,
    ) -> IngestionDispatchReceipt:
        receipt = await super().ingest(request)
        usage_record_id = canonical_sha256(
            [
                "oamb-native-fixture-ingest-usage-v1",
                request.attempt_id,
                receipt.raw_reference.sha256,
            ]
        )
        usage = TokenUsageRecordV3(
            usage_record_id=usage_record_id,
            attempt_id=request.attempt_id,
            parent_kind="ingestion_plan",
            parent_id=request.scope.ingestion_occurrence_id,
            stage=TokenStageV2.MEMORY_INGEST,
            operation_kind="recorded_fixture_ingest",
            token_domain=TokenDomain.EXTERNAL_LLM,
            measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
            input_tokens=None,
            visible_output_tokens=None,
            supplier_reported_total_tokens=None,
            context_view_tokens=None,
            cached_input_tokens=None,
            reasoning_tokens=None,
            model="recorded-extractor",
            meter_schema_id="recorded-fixture-usage-v1",
            raw_field_paths=(),
            covered_dimensions=(),
            unavailable_dimensions=(
                "input_tokens",
                "visible_output_tokens",
                "supplier_reported_total_tokens",
                "cached_input_tokens",
                "reasoning_tokens",
            ),
            not_applicable_dimensions=(),
            inclusion_relationships=(),
            token_measurement_complete=False,
            billing_complete=False,
            proof_status=ProofStatus.UNAVAILABLE,
            reason="recorded_fixture_has_no_supplier_meter",
            raw_response_ref=receipt.raw_reference.sha256,
        )
        return replace(receipt, usage_records=(usage,))

    async def allocate_ingestion_scope(
        self,
        request: ScopeAllocationRequest,
    ) -> ScopeReceipt:
        receipt = await super().allocate_ingestion_scope(request)
        supporting = self._raw_reference(
            {
                "operation": "scope_support",
                "ingestion_occurrence_id": receipt.ingestion_occurrence_id,
                "scope_id": receipt.scope_id,
            }
        )
        return replace(receipt, supporting_raw_references=(supporting,))

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        projection = await super().project(scope)
        supporting = self._raw_reference(
            {
                "operation": "projection_support",
                "ingestion_occurrence_id": scope.ingestion_occurrence_id,
                "scope_id": scope.scope_id,
                "ordered_source_unit_ids": projection.inventory.ordered_source_unit_ids,
                "state_sha256": projection.state_digest.state_sha256,
            }
        )
        return replace(projection, supporting_raw_references=(supporting,))

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        source_ids = self._accepted_by_scope[request.scope.scope_id]
        source_unit_id = source_ids[0]
        candidates = (
            NativeEvidenceCandidate(
                native_id="fixture-native-1",
                native_rank_1_indexed=1,
                content="shared answer alpha beta partial",
                native_score="0.900",
                provider_evidence_identity="fixture-evidence-1",
                source_unit_id=source_unit_id,
            ),
            NativeEvidenceCandidate(
                native_id="fixture-native-2",
                native_rank_1_indexed=2,
                content="secondary recorded evidence",
                native_score="0.800",
                provider_evidence_identity="fixture-evidence-2",
                source_unit_id=source_unit_id,
            ),
        )
        supporting = self._raw_reference(
            {
                "operation": "retrieve_support",
                "case_occurrence_id": request.case_occurrence_id,
                "ordered_native_ids": tuple(candidate.native_id for candidate in candidates),
            }
        )
        primary = self._raw_reference(
            {
                "operation": "retrieve",
                "case_occurrence_id": request.case_occurrence_id,
                "scope_id": request.scope.scope_id,
                "query_sha256": hashlib.sha256(request.query_bytes).hexdigest(),
                "top_k": request.top_k,
                "supporting_raw_references": (supporting.sha256,),
                "candidates": tuple(
                    {
                        "native_id": candidate.native_id,
                        "native_rank_1_indexed": candidate.native_rank_1_indexed,
                        "content": candidate.content,
                        "source_unit_id": candidate.source_unit_id,
                        "provider_evidence_identity": candidate.provider_evidence_identity,
                    }
                    for candidate in candidates
                ),
            }
        )
        return NativeEvidenceBatch(
            raw_reference=primary,
            candidates=candidates,
            supporting_raw_references=(supporting,),
        )


class _RecordedNativeModel:
    def __init__(self, store: ArtifactStorePort) -> None:
        self._store = store
        self._closed = False

    def thinking_effort_for(self, *, stage: str, role_binding_id: str) -> ThinkingEffort:
        if stage == "answer" and role_binding_id == "recorded-answer-v1":
            return "low"
        if stage == "judge" and role_binding_id == "fake-judge-v1":
            return "high"
        raise ValueError("recorded model request differs from its fixture role binding")

    async def complete(self, request: ModelRequest) -> ModelReceipt:
        if self._closed:
            raise RuntimeError("recorded model is closed")
        prompt_text = request.messages[0][1]
        output_text = _answer_for_prompt(prompt_text)
        payload = canonical_json_bytes(
            {
                "operation": "model_complete",
                "attempt_id": request.attempt_id,
                "stage": request.stage,
                "prompt": prompt_text,
                "output_text": output_text,
            }
        )
        raw_sha256 = hashlib.sha256(payload).hexdigest()
        raw_reference = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=raw_sha256,
                media_type="application/json",
                compression="gzip",
                payload_bytes=payload,
            )
        )
        return ModelReceipt(
            raw_reference=raw_reference,
            output_text=output_text,
            usage_reference_ids=(),
            model="fixture-model",
            raw_response_bytes=payload,
        )

    async def close(self) -> None:
        self._closed = True


class _MeasuredLiveModel(_RecordedNativeModel):
    async def complete(self, request: ModelRequest) -> ModelReceipt:
        receipt = await super().complete(request)
        usage = self._usage_record(request, receipt.raw_reference.sha256)
        payload = canonical_json_bytes(usage)
        self._store.seal_source_record(
            ArtifactWriteRequest(
                record_id=usage.usage_record_id,
                relative_path=f"source/usage/{usage.usage_record_id}.json",
                canonical_sha256=hashlib.sha256(payload).hexdigest(),
                canonical_bytes=payload,
            )
        )
        return replace(receipt, usage_reference_ids=(usage.usage_record_id,))

    def _usage_record(self, request: ModelRequest, raw_response_ref: str) -> TokenUsageRecordV2:
        usage_id = canonical_sha256(
            ["measured-live-model-usage", request.attempt_id, raw_response_ref]
        )
        return TokenUsageRecordV2(
            usage_record_id=usage_id,
            attempt_id=request.attempt_id,
            parent_kind=request.parent_kind,
            parent_id=request.parent_id,
            stage=TokenStageV2(request.stage),
            operation_kind="openai_chat_completion",
            token_domain=TokenDomain.EXTERNAL_LLM,
            measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
            input_tokens=13,
            visible_output_tokens=5,
            supplier_reported_total_tokens=18,
            context_view_tokens=None,
            proof_status=ProofStatus.MEASURED_COMPLETE,
            reason=None,
            raw_response_ref=raw_response_ref,
        )


class _MismatchedLiveModel(_MeasuredLiveModel):
    def _usage_record(self, request: ModelRequest, raw_response_ref: str) -> TokenUsageRecordV2:
        return (
            super()
            ._usage_record(request, raw_response_ref)
            .model_copy(update={"parent_id": "wrong-parent"})
        )


class _TamperedLiveModel(_MeasuredLiveModel):
    async def complete(self, request: ModelRequest) -> ModelReceipt:
        receipt = await _RecordedNativeModel.complete(self, request)
        usage = self._usage_record(request, receipt.raw_reference.sha256)
        payload = canonical_json_bytes(usage)
        wrong_id = canonical_sha256(["tampered-live-usage-id", usage.usage_record_id])
        self._store.seal_source_record(
            ArtifactWriteRequest(
                record_id=wrong_id,
                relative_path=f"source/usage/{wrong_id}.json",
                canonical_sha256=hashlib.sha256(payload).hexdigest(),
                canonical_bytes=payload,
            )
        )
        return replace(receipt, usage_reference_ids=(wrong_id,))


class _ExtraUnreferencedLiveModel(_MeasuredLiveModel):
    async def complete(self, request: ModelRequest) -> ModelReceipt:
        receipt = await super().complete(request)
        extra = self._usage_record(request, receipt.raw_reference.sha256).model_copy(
            update={
                "usage_record_id": canonical_sha256(["unreferenced-live-usage", request.attempt_id])
            }
        )
        payload = canonical_json_bytes(extra)
        self._store.seal_source_record(
            ArtifactWriteRequest(
                record_id=extra.usage_record_id,
                relative_path=f"source/usage/{extra.usage_record_id}.json",
                canonical_sha256=hashlib.sha256(payload).hexdigest(),
                canonical_bytes=payload,
            )
        )
        return receipt


def _answer_for_prompt(prompt: str) -> str:
    if '"answer_sha256"' in prompt:
        return "yes"
    if "Which token is first?" in prompt:
        return "alpha"
    if "Which token is second?" in prompt:
        return "beta"
    if "Is the shared phrase grounded?" in prompt:
        return "shared answer"
    if "What survived the partial ingest?" in prompt:
        return "partial"
    raise AssertionError(f"unexpected recorded prompt: {prompt}")


def _memory_factory(
    store: ArtifactStorePort,
    _plans: tuple[IngestionPlan, ...],
) -> _RecordedNativeMemory:
    return _RecordedNativeMemory(store)


class _RecordedMem0SubsetMemory(_RecordedNativeMemory):
    async def resolve(self) -> RuntimeResolution:
        resolution = await super().resolve()
        return replace(resolution, memory_system_id="mem0")

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        state_sha256 = canonical_sha256(["oamb-mem0-main-projection-v1", ()])
        inventory = InventoryReceipt(
            ingestion_occurrence_id=scope.ingestion_occurrence_id,
            ordered_source_unit_ids=(),
            raw_reference=self._raw_reference(
                {
                    "operation": "inventory",
                    "ingestion_occurrence_id": scope.ingestion_occurrence_id,
                    "scope_id": scope.scope_id,
                    "ordered_source_unit_ids": (),
                }
            ),
        )
        state = StateDigestReceipt(
            ingestion_occurrence_id=scope.ingestion_occurrence_id,
            state_sha256=state_sha256,
            raw_reference=self._raw_reference(
                {
                    "operation": "state_digest",
                    "ingestion_occurrence_id": scope.ingestion_occurrence_id,
                    "scope_id": scope.scope_id,
                    "state_sha256": state_sha256,
                }
            ),
        )
        supporting = self._raw_reference(
            {
                "operation": "projection_support",
                "ingestion_occurrence_id": scope.ingestion_occurrence_id,
                "scope_id": scope.scope_id,
                "ordered_source_unit_ids": (),
                "state_sha256": state_sha256,
            }
        )
        return ProjectionReceipt(
            inventory=inventory,
            state_digest=state,
            supporting_raw_references=(supporting,),
        )

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        return NativeEvidenceBatch(
            raw_reference=self._raw_reference(
                {
                    "operation": "retrieve",
                    "case_occurrence_id": request.case_occurrence_id,
                    "scope_id": request.scope.scope_id,
                    "query_sha256": hashlib.sha256(request.query_bytes).hexdigest(),
                    "top_k": request.top_k,
                    "candidates": (),
                }
            ),
            candidates=(),
        )


def _mem0_subset_memory_factory(
    store: ArtifactStorePort,
    _plans: tuple[IngestionPlan, ...],
) -> _RecordedMem0SubsetMemory:
    return _RecordedMem0SubsetMemory(store)


def _run_fixture_capsule(tmp_path: Path, run_id: str) -> NativeRunArtifacts:
    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=run_id,
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
    )


def _run_judged_fixture_capsule(tmp_path: Path, run_id: str) -> NativeRunArtifacts:
    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=run_id,
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureJudgeWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
        judge_model_factory=_RecordedNativeModel,
        judge_role_binding_id="fake-judge-v1",
    )


def test_native_partition_executes_only_selected_whole_plan(tmp_path: Path) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    first_plan = manifest.ingestion_plans[0]
    partition = build_case_partition_spec(
        run_id="native-partition-first-plan",
        resolved_plan_hash=canonical_sha256(["partition-plan"]),
        cell_spec_hash=canonical_sha256(["partition-cell"]),
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=(first_plan.ordered_case_manifest_entry_ids[1],),
        budget_policy_hash=canonical_sha256(["partition-budget-policy"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )

    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=partition.run_id,
        adapter_profile_id="recorded-native-fixture-v1",
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
        partition=partition,
    )

    assert (
        tuple(record.ingestion_plan_id for record in completed.ingestion_plan_records)
        == partition.selected_ingestion_plan_ids
    )
    assert (
        tuple(record.case_manifest_entry_id for record in completed.case_records)
        == partition.selected_case_manifest_entry_ids
    )
    assert (
        completed.capsule_root / "source" / "specs" / f"{partition.partition_id}.json"
    ).is_file()
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues


@pytest.mark.parametrize("unknown_transport", (False, True))
def test_native_model_429_retries_retain_each_physical_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unknown_transport: bool,
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    first_plan = manifest.ingestion_plans[0]
    provider_runtime = (tmp_path / "model-retry-provider-runtime").resolve()
    control = _control(
        run_id="native-structured-429",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=manifest.manifest_hash,
        workload_id=manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="answer-binding",
        provider_runtime_directory=provider_runtime,
    )
    partition = build_case_partition_spec(
        run_id="native-structured-429",
        resolved_plan_hash=control.preflight_record.resolved_plan_hash,
        cell_spec_hash=control.preflight_record.adapter_profile_hash,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=(first_plan.ordered_case_manifest_entry_ids[0],),
        budget_policy_hash=canonical_sha256(["structured-429-budget"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    calls = multiprocessing.get_context("fork").Value("i", 0)
    rejection = canonical_json_bytes(
        {
            "error": {
                "origin": "model_supplier",
                "failure_kind": "rate_limited",
                "status": 429,
                "acceptance": "not_accepted",
                "provider_mutation": "none",
                "retryable": True,
                "internal_retry_count": 0,
            }
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        with calls.get_lock():
            calls.value += 1
            call_number = calls.value
        if call_number <= 2:
            if unknown_transport:
                raise httpx.ReadError("response lost", request=_request)
            return httpx.Response(429, content=rejection)
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "alpha"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def no_wait(_seconds: int) -> None:
        return None

    monkeypatch.setattr(native_run_module, "_infrastructure_retry_sleep", no_wait)
    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=partition.run_id,
        adapter_profile_id="recorded-native-fixture-v1",
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=lambda store: _client(cast(Any, store), handler),
        answer_role_binding_id="answer-binding",
        partition=partition,
        control=control,
    )

    events = tuple(
        json.loads(path.read_bytes())
        for path in sorted(
            (completed.capsule_root / "source/infrastructure-retries").glob("*.json")
        )
    )
    attempts = tuple(
        json.loads(path.read_bytes())
        for path in sorted((completed.capsule_root / "source/attempts").glob("*.json"))
    )
    usage_records = {
        record["usage_record_id"]: record
        for path in (completed.capsule_root / "source/usage").glob("*.json")
        for record in (json.loads(path.read_bytes()),)
    }
    assert calls.value == 5
    assert not events
    answer_attempts = [attempt for attempt in attempts if attempt["stage"] == "answer"]
    assert len(answer_attempts) == 5
    failed_outcome = "unknown_outcome" if unknown_transport else "failed"
    assert sum(attempt["outcome"] == failed_outcome for attempt in answer_attempts) == 2
    assert sum(attempt["outcome"] == "succeeded" for attempt in answer_attempts) == 3
    assert len({attempt["attempt_id"] for attempt in answer_attempts}) == 5
    answer_ids = {
        attempt["attempt_id"]
        for attempt in answer_attempts
        if attempt["outcome"] != "unknown_outcome"
    }
    assert {
        record["attempt_id"] for record in usage_records.values() if record["stage"] == "answer"
    } == answer_ids
    referenced_usage = {
        identity for case in completed.case_records for identity in case.usage_record_ids
    }
    assert all(
        identity in referenced_usage
        for identity, record in usage_records.items()
        if record["attempt_id"] in answer_ids
    )
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    assert not (provider_runtime / "active-operation").exists()
    active_attempts = provider_runtime / "active-provider-attempts"
    assert not active_attempts.exists() or not tuple(active_attempts.iterdir())
    invalid_root = tmp_path / "invalid-model-retry-chain"
    shutil.copytree(completed.capsule_root, invalid_root)
    retry_attempt = next(attempt for attempt in answer_attempts if attempt["retry_of_attempt_id"])
    path = invalid_root / "source/attempts" / f"{retry_attempt['attempt_id']}.json"
    _mutate_source_document(invalid_root, path, retry_of_attempt_id="f" * 64)
    invalid = validate_native_capsule(invalid_root)
    assert invalid.disposition == ValidationDisposition.INVALID


def test_native_model_429_exhaustion_retains_error_cases_and_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    first_plan = manifest.ingestion_plans[0]
    provider_runtime = (tmp_path / "structured-429-provider-runtime").resolve()
    control = _control(
        run_id="native-structured-429-exhausted",
        model_attempts=54,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=manifest.manifest_hash,
        workload_id=manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="answer-binding",
        provider_runtime_directory=provider_runtime,
    )
    partition = build_case_partition_spec(
        run_id=control.run_spec.run_id,
        resolved_plan_hash=control.preflight_record.resolved_plan_hash,
        cell_spec_hash=control.preflight_record.adapter_profile_hash,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=(first_plan.ordered_case_manifest_entry_ids[0],),
        budget_policy_hash=canonical_sha256(["structured-429-exhausted-budget"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    calls = multiprocessing.get_context("fork").Value("i", 0)
    rejection = canonical_json_bytes(
        {
            "error": {
                "origin": "model_supplier",
                "failure_kind": "rate_limited",
                "status": 429,
                "acceptance": "not_accepted",
                "provider_mutation": "none",
                "retryable": True,
                "internal_retry_count": 0,
            }
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        with calls.get_lock():
            calls.value += 1
        return httpx.Response(429, content=rejection)

    async def no_wait(_seconds: int) -> None:
        return None

    monkeypatch.setattr(native_run_module, "_infrastructure_retry_sleep", no_wait)
    run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=partition.run_id,
        adapter_profile_id="recorded-native-fixture-v1",
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=lambda store: _client(cast(Any, store), handler),
        answer_role_binding_id="answer-binding",
        control=control,
        partition=partition,
    )

    root = tmp_path / "capsules" / partition.run_id
    run = json.loads((root / "source/run" / f"{partition.run_id}.json").read_bytes())
    events = tuple(
        json.loads(path.read_bytes())
        for path in sorted((root / "source/infrastructure-retries").glob("*.json"))
    )
    resources = tuple((root / "source/resources").glob("*.json"))
    costs = tuple((root / "source/costs").glob("*.json"))
    assert calls.value == 54  # Three cases independently exhaust 6 x 3 physical calls.
    assert run["state"] == "finalized"
    assert not events
    cases = [json.loads(path.read_bytes()) for path in (root / "source/cases").glob("*.json")]
    assert len(cases) == 3
    assert all(
        case["state"] == "error" and case["evaluation_disposition"] == "not_run" for case in cases
    )
    assert len(resources) == len(costs)
    validation = validate_source_root(root)
    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code, issue.evidence_ref) for issue in validation.issues
    ]
    assert not (provider_runtime / "active-operation").exists()
    attempts_directory = provider_runtime / "active-provider-attempts"
    assert not attempts_directory.exists() or not tuple(attempts_directory.iterdir())


def test_live_cancellation_during_retry_backoff_clears_provider_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    first_plan = manifest.ingestion_plans[0]
    provider_runtime = (tmp_path / "cancelled-backoff-provider-runtime").resolve()
    control = _control(
        run_id="native-structured-429-cancelled-backoff",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=manifest.manifest_hash,
        workload_id=manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="answer-binding",
        provider_runtime_directory=provider_runtime,
    )
    partition = build_case_partition_spec(
        run_id=control.run_spec.run_id,
        resolved_plan_hash=control.preflight_record.resolved_plan_hash,
        cell_spec_hash=control.preflight_record.adapter_profile_hash,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=(first_plan.ordered_case_manifest_entry_ids[0],),
        budget_policy_hash=canonical_sha256(["cancelled-backoff-budget"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    calls = multiprocessing.get_context("fork").Value("i", 0)
    rejection = canonical_json_bytes(
        {
            "error": {
                "origin": "model_supplier",
                "failure_kind": "rate_limited",
                "status": 429,
                "acceptance": "not_accepted",
                "provider_mutation": "none",
                "retryable": True,
                "internal_retry_count": 0,
            }
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        with calls.get_lock():
            calls.value += 1
        return httpx.Response(429, content=rejection)

    async def cancel_backoff(_seconds: int) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(native_run_module, "_infrastructure_retry_sleep", cancel_backoff)

    with pytest.raises(asyncio.CancelledError):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id=partition.run_id,
            adapter_profile_id="recorded-native-fixture-v1",
            workload=workload,
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=lambda store: _client(cast(Any, store), handler),
            answer_role_binding_id="answer-binding",
            control=control,
            partition=partition,
        )

    root = tmp_path / "capsules" / partition.run_id
    run = json.loads((root / "source/run" / f"{partition.run_id}.json").read_bytes())
    events = tuple((root / "source/infrastructure-retries").glob("*.json"))
    assert calls.value == 1
    assert run["state"] == "aborted"
    assert not events
    validation = validate_source_root(root)
    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code, issue.evidence_ref) for issue in validation.issues
    ]
    assert not (provider_runtime / "active-operation").exists()
    attempts_directory = provider_runtime / "active-provider-attempts"
    assert not attempts_directory.exists() or not tuple(attempts_directory.iterdir())


def test_planned_stop_during_retry_backoff_starts_no_later_supplier_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    first_plan = manifest.ingestion_plans[0]
    provider_runtime = (tmp_path / "stopped-backoff-provider-runtime").resolve()
    control = _control(
        run_id="native-structured-429-stopped-backoff",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=manifest.manifest_hash,
        workload_id=manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="answer-binding",
        provider_runtime_directory=provider_runtime,
    )
    partition = build_case_partition_spec(
        run_id=control.run_spec.run_id,
        resolved_plan_hash=control.preflight_record.resolved_plan_hash,
        cell_spec_hash=control.preflight_record.adapter_profile_hash,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=(first_plan.ordered_case_manifest_entry_ids[0],),
        budget_policy_hash=canonical_sha256(["stopped-backoff-budget"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    context = multiprocessing.get_context("fork")
    calls = context.Value("i", 0)
    stop_event = context.Event()
    rejection = canonical_json_bytes(
        {
            "error": {
                "origin": "model_supplier",
                "failure_kind": "rate_limited",
                "status": 429,
                "acceptance": "not_accepted",
                "provider_mutation": "none",
                "retryable": True,
                "internal_retry_count": 0,
            }
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        with calls.get_lock():
            calls.value += 1
        return httpx.Response(429, content=rejection)

    async def stop_backoff(_seconds: int) -> None:
        stop_event.set()

    monkeypatch.setattr(native_run_module, "_infrastructure_retry_sleep", stop_backoff)

    with pytest.raises(NativeRunInterrupted):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id=partition.run_id,
            adapter_profile_id="recorded-native-fixture-v1",
            workload=workload,
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=lambda store: _client(cast(Any, store), handler),
            answer_role_binding_id="answer-binding",
            control=control,
            partition=partition,
            stop_event=stop_event,
        )

    assert calls.value == 1
    root = tmp_path / "capsules" / partition.run_id
    validation = validate_source_root(root)
    assert validation.disposition == ValidationDisposition.VALIDATED, [
        (issue.rule_id, issue.code, issue.evidence_ref) for issue in validation.issues
    ]
    assert not (provider_runtime / "active-operation").exists()


def test_native_validation_rejects_supplier_internal_retry_budget_overflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    first_plan = manifest.ingestion_plans[0]
    partition = build_case_partition_spec(
        run_id="native-structured-429-internal-overflow",
        resolved_plan_hash=canonical_sha256(["internal-overflow-plan"]),
        cell_spec_hash=canonical_sha256(["internal-overflow-cell"]),
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=(first_plan.ordered_case_manifest_entry_ids[0],),
        budget_policy_hash=canonical_sha256(["internal-overflow-budget"]),
        retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
    )
    calls = multiprocessing.get_context("fork").Value("i", 0)
    rejection = canonical_json_bytes(
        {
            "error": {
                "origin": "model_supplier",
                "failure_kind": "rate_limited",
                "status": 429,
                "acceptance": "not_accepted",
                "provider_mutation": "none",
                "retryable": True,
                "internal_retry_count": 4,
            }
        }
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        with calls.get_lock():
            calls.value += 1
        return httpx.Response(429, content=rejection)

    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr(native_run_module, "_infrastructure_retry_sleep", no_wait)
    with pytest.raises(RuntimeError, match="supplier internal retry configuration drift"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id=partition.run_id,
            adapter_profile_id="recorded-native-fixture-v1",
            workload=workload,
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=lambda store: _client(cast(Any, store), handler),
            answer_role_binding_id="answer-binding",
            partition=partition,
        )

    root = tmp_path / "capsules" / partition.run_id
    validation = validate_source_root(root)
    assert calls.value == 1
    assert validation.disposition == ValidationDisposition.INVALID
    assert "model-supplier-retry-proof-drift" in {issue.code for issue in validation.issues}


def test_native_mem0_rest_writes_v3_for_empty_visible_subset(tmp_path: Path) -> None:
    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="native-mem0-subset",
        adapter_profile_id="mem0-rest-v1",
        workload=_NativeFixtureWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_mem0_subset_memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
    )

    assert completed.ingestion_plan_records
    assert all(
        isinstance(plan, IngestionPlanRecordV3)
        and plan.projection_semantics == "retrieval_visible_subset"
        and plan.accepted_source_unit_ids
        and plan.projected_source_unit_ids == ()
        for plan in completed.ingestion_plan_records
    )
    accounting = native_reduce.build_native_accounting_validation_input(completed.capsule_root)
    assert accounting.expected_plan_ids == frozenset(
        plan.ingestion_occurrence_id for plan in completed.ingestion_plan_records
    )


def test_recorded_empty_mem0_rest_capsule_validates_black_box_evidence(
    tmp_path: Path,
) -> None:
    def public_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/openapi.json":
            return httpx.Response(
                200,
                json={
                    "openapi": "3.1.0",
                    "info": {"title": "Mem0 REST APIs", "version": "1.0.0"},
                    "paths": {"/memories": {"post": {}}, "/search": {"post": {}}},
                },
            )
        if request.url.path in {"/memories", "/search"}:
            return httpx.Response(200, content=b'{"results":[]}')
        raise AssertionError(f"unexpected Mem0 request: {request.method} {request.url}")

    def inspector_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(
                200,
                content=b'{"status":"ok","mode":"read_only_projection"}',
            )
        if request.url.path == "/v1/projection":
            run_id = request.url.params["run_id"]
            return httpx.Response(
                200,
                json={
                    "collection": "oamb_memories",
                    "run_id": run_id,
                    "count": 0,
                    "points": [],
                    "next_cursor": None,
                },
            )
        raise AssertionError(f"unexpected inspector request: {request.method} {request.url}")

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> Mem0RestAdapter:
        return Mem0RestAdapter(
            store=store,
            base_url="https://mem0.example",
            api_key="fixture-secret",
            inspector_base_url="https://mem0-inspector.example",
            inspector_api_key="fixture-inspector-secret",
            runtime_binding_hash="f" * 64,
            transport=httpx.MockTransport(public_handler),
            inspector_transport=httpx.MockTransport(inspector_handler),
        )

    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="native-mem0-recorded-empty",
        adapter_profile_id="mem0-rest-v1",
        workload=_NativeFixtureWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
    )

    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, tuple(
        (issue.rule_id, issue.code) for issue in validation.issues
    )
    adapter_validation = validate_catalog_profile(
        "oamb-t10-adapter-mem0-rest-blackbox-v1",
        completed.capsule_root,
    )
    assert adapter_validation.disposition == ValidationDisposition.VALIDATED, tuple(
        (issue.rule_id, issue.code) for issue in adapter_validation.issues
    )

    invalid_root = tmp_path / "mem0-readiness-single-snapshot"
    shutil.copytree(completed.capsule_root, invalid_root)
    plan_path = _first_source_document(invalid_root, "ingestion_plan_record", 3)
    plan_document = json.loads(plan_path.read_bytes())
    readiness_references = plan_document["readiness_evidence_refs"]
    summary_references = tuple(
        reference
        for reference in readiness_references
        if _raw_payload_document(invalid_root, reference).get("schema")
        == "oamb-mem0-projection-receipt-v1"
    )
    assert len(summary_references) == 2
    _mutate_source_document(
        invalid_root,
        plan_path,
        readiness_evidence_refs=tuple(
            reference for reference in readiness_references if reference != summary_references[-1]
        ),
    )

    invalid = validate_catalog_profile(
        "oamb-t10-adapter-mem0-rest-blackbox-v1",
        invalid_root,
    )
    assert invalid.disposition == ValidationDisposition.INVALID
    assert "adapter.mem0.rest.scope-dispatch-projection.v1" in invalid.failed_rule_ids

    replayed_post_root = tmp_path / "mem0-replayed-post-query-projection"
    shutil.copytree(completed.capsule_root, replayed_post_root)
    case_path = _first_source_document(replayed_post_root, "case_record", 3)
    case_document = json.loads(case_path.read_bytes())
    _mutate_source_document(
        replayed_post_root,
        case_path,
        post_query_projection_raw_refs=case_document["pre_query_projection_raw_refs"],
        post_query_state_sha256=case_document["pre_query_state_sha256"],
    )

    replayed_post = validate_catalog_profile(
        "oamb-t10-adapter-mem0-rest-blackbox-v1",
        replayed_post_root,
    )
    assert replayed_post.disposition == ValidationDisposition.INVALID
    assert "adapter.mem0.rest.query-mutation.v1" in replayed_post.failed_rule_ids


def _first_source_document(root: Path, schema_name: str, schema_version: int) -> Path:
    manifest = CapsuleManifest.model_validate_json((root / "capsule-manifest.json").read_bytes())
    for entry in manifest.source_entries:
        if entry.record_kind != schema_name:
            continue
        path = root / entry.relative_path
        document = json.loads(path.read_bytes())
        if document.get("schema_version") == schema_version:
            return path
    raise AssertionError(f"missing {schema_name} v{schema_version}")


def _mutate_source_document(root: Path, path: Path, **updates: object) -> None:
    document = json.loads(path.read_bytes())
    document.update(updates)
    path.write_bytes(canonical_json_bytes(document))
    _reseal_manifest(root)


def _remove_raw_reference(root: Path, raw_reference: str) -> None:
    manifest = CapsuleManifest.model_validate_json((root / "capsule-manifest.json").read_bytes())
    entry = next(
        item
        for item in manifest.source_entries
        if item.record_kind == "raw_payload" and item.record_id == raw_reference
    )
    (root / entry.relative_path).unlink()
    _reseal_manifest(root)


def _raw_payload_document(root: Path, raw_reference: str) -> dict[str, object]:
    manifest = CapsuleManifest.model_validate_json((root / "capsule-manifest.json").read_bytes())
    entry = next(
        item
        for item in manifest.source_entries
        if item.record_kind == "raw_payload" and item.record_id == raw_reference
    )
    compressed = (root / entry.relative_path).read_bytes()
    document = json.loads(gzip.decompress(compressed))
    if not isinstance(document, dict):
        raise AssertionError("raw payload fixture is not an object")
    return document


def _reseal_manifest(root: Path) -> None:
    path = root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(path.read_bytes())
    source_entries = tuple(
        entry.model_copy(
            update={"sha256": hashlib.sha256((root / entry.relative_path).read_bytes()).hexdigest()}
        )
        for entry in manifest.source_entries
        if (root / entry.relative_path).is_file()
    )
    source_manifest_hash = canonical_sha256(
        [
            "oamb-source-manifest-v1",
            tuple(entry.model_dump(mode="python") for entry in source_entries),
        ]
    )
    capsule_id = canonical_sha256(
        ["oamb-capsule-v1", manifest.run_id, manifest.run_spec_hash, source_manifest_hash]
    )
    path.write_bytes(
        canonical_json_bytes(
            manifest.model_copy(
                update={
                    "capsule_id": capsule_id,
                    "source_entries": source_entries,
                    "source_manifest_hash": source_manifest_hash,
                }
            )
        )
    )


def _append_source_contract_copy(
    root: Path,
    *,
    schema_name: str,
    identity_field: str,
    record_id: str,
    relative_parent: Path | None = None,
) -> None:
    manifest_path = root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(manifest_path.read_bytes())
    template_entry = next(
        entry for entry in manifest.source_entries if entry.record_kind == schema_name
    )
    template_path = root / template_entry.relative_path
    document = json.loads(template_path.read_bytes())
    document[identity_field] = record_id
    content = canonical_json_bytes(document)
    destination_parent = relative_parent or Path(template_entry.relative_path).parent
    relative_path = (destination_parent / f"{record_id}.json").as_posix()
    destination = root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    copied_entry = template_entry.model_copy(
        update={
            "record_id": record_id,
            "relative_path": relative_path,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    manifest_path.write_bytes(
        canonical_json_bytes(
            manifest.model_copy(update={"source_entries": (*manifest.source_entries, copied_entry)})
        )
    )
    _reseal_manifest(root)


def _successor_run_lease(
    predecessor: RunLeaseRecord,
    **updates: object,
) -> RunLeaseRecord:
    candidate = predecessor.model_copy(
        update={
            "lease_record_hash": "0" * 64,
            "lease_epoch": predecessor.lease_epoch + 1,
            "owner_id": f"{predecessor.owner_id}-successor",
            "predecessor_lease_record_hash": predecessor.lease_record_hash,
            **updates,
        }
    )
    return candidate.model_copy(
        update={
            "lease_record_hash": canonical_sha256(
                candidate.model_dump(mode="python", exclude={"lease_record_hash"})
            )
        }
    )


def _append_run_lease(
    root: Path,
    lease: RunLeaseRecord,
    *,
    relative_path: str | None = None,
) -> None:
    manifest_path = root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(manifest_path.read_bytes())
    template_entry = next(
        entry for entry in manifest.source_entries if entry.record_kind == "run_lease_record"
    )
    path = relative_path or f"source/run-leases/{lease.lease_epoch}.json"
    content = canonical_json_bytes(lease)
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    entry = template_entry.model_copy(
        update={
            "record_id": Path(path).stem,
            "relative_path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    manifest_path.write_bytes(
        canonical_json_bytes(
            manifest.model_copy(update={"source_entries": (*manifest.source_entries, entry)})
        )
    )
    _reseal_manifest(root)


def _rebind_first_occurrence_claim(
    root: Path,
    lease: RunLeaseRecord,
    **updates: object,
) -> None:
    manifest_path = root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(manifest_path.read_bytes())
    claim_entry = next(
        entry for entry in manifest.source_entries if entry.record_kind == "occurrence_claim_record"
    )
    claim_path = root / claim_entry.relative_path
    previous = OccurrenceClaimRecord.model_validate_json(claim_path.read_bytes())
    candidate = previous.model_copy(
        update={
            "claim_id": "0" * 64,
            "lease_record_hash": lease.lease_record_hash,
            "lease_epoch": lease.lease_epoch,
            "owner_id": lease.owner_id,
            **updates,
        }
    )
    claim_fields = candidate.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "claim_id"},
    )
    successor_claim = candidate.model_copy(
        update={"claim_id": canonical_sha256(["oamb-native-occurrence-claim-v1", claim_fields])}
    )
    intent_entry = next(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "attempt_intent_record"
        and json.loads((root / entry.relative_path).read_bytes())["claim_id"] == previous.claim_id
    )
    intent_path = root / intent_entry.relative_path
    intent = json.loads(intent_path.read_bytes())
    intent["claim_id"] = successor_claim.claim_id
    intent_path.write_bytes(canonical_json_bytes(intent))

    successor_path = Path("source/occurrence-claims") / f"{successor_claim.claim_id}.json"
    successor_content = canonical_json_bytes(successor_claim)
    (root / successor_path).write_bytes(successor_content)
    claim_path.unlink()
    source_entries = tuple(
        entry.model_copy(
            update={
                "record_id": successor_claim.claim_id,
                "relative_path": successor_path.as_posix(),
                "sha256": hashlib.sha256(successor_content).hexdigest(),
            }
        )
        if entry.relative_path == claim_entry.relative_path
        else entry
        for entry in manifest.source_entries
    )
    manifest_path.write_bytes(
        canonical_json_bytes(manifest.model_copy(update={"source_entries": source_entries}))
    )
    _reseal_manifest(root)


def _issue_codes(root: Path) -> set[str]:
    return {issue.code for issue in validate_native_capsule(root).issues}


def test_recorded_native_ports_seal_a_root_only_validatable_capsule(tmp_path: Path) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-valid")

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    assert validation.failed_rule_ids == ()
    assert len(completed.ingestion_plan_records) == 2
    assert len(completed.case_records) == 4
    assert all(record.schema_version == 2 for record in completed.ingestion_plan_records)
    assert all(record.schema_version == 3 for record in completed.case_records)
    assert all(record.retrieval_supporting_raw_refs for record in completed.case_records)
    assert all(record.usage_record_ids for record in completed.ingestion_plan_records)
    assert all(record.metric_denominator == 1 for record in completed.case_records)


def test_live_entrypoint_owns_provider_lifecycle_through_terminal_capsule(
    tmp_path: Path,
) -> None:
    from oamb.artifacts.validation.source_root import validate_source_root

    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    control = _control(
        run_id="controlled-native-valid",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        workload_id=case_manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="recorded-answer-v1",
        provider_runtime_directory=tmp_path / "provider-runtime",
    )

    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=control.run_spec.run_id,
        adapter_profile_id=control.preflight_record.adapter_profile_id,
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
        control=control,
    )

    assert completed.manifest.run_id == control.run_spec.run_id
    assert not (control.provider_runtime_directory / "active-operation").exists()
    attempts_directory = control.provider_runtime_directory / "active-provider-attempts"
    assert not attempts_directory.exists() or not any(attempts_directory.iterdir())
    reservations = tuple(
        json.loads(path.read_bytes())
        for path in sorted((completed.capsule_root / "source/budget-reservations").glob("*.json"))
    )
    intents = tuple(
        json.loads(path.read_bytes())
        for path in sorted((completed.capsule_root / "source/attempt-intents").glob("*.json"))
    )
    attempts = tuple(
        json.loads(path.read_bytes())
        for path in sorted((completed.capsule_root / "source/attempts").glob("*.json"))
    )
    assert reservations and {item["schema_version"] for item in reservations} == {3}
    assert intents and {item["schema_version"] for item in intents} == {3}
    assert attempts and {item["schema_version"] for item in attempts} == {4}
    assert {
        "runtime_resolve",
        "scope_allocate",
        "memory_ingest",
        "memory_readiness",
        "memory_projection",
        "pre_query_projection",
        "memory_query",
        "post_query_projection",
        "answer",
    } <= {item["stage"] for item in attempts}
    validation = validate_source_root(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues

    hash_drift_root = tmp_path / "live-attempt-hash-drift"
    shutil.copytree(completed.capsule_root, hash_drift_root)
    attempt_path = _first_source_document(hash_drift_root, "attempt_record", 4)
    _mutate_source_document(
        hash_drift_root,
        attempt_path,
        request_fingerprint="f" * 64,
    )
    assert validate_source_root(hash_drift_root).disposition == ValidationDisposition.INVALID

    path_drift_root = tmp_path / "live-attempt-path-drift"
    shutil.copytree(completed.capsule_root, path_drift_root)
    manifest_path = path_drift_root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(manifest_path.read_bytes())
    attempt_entry = next(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "attempt_record"
        and json.loads((path_drift_root / entry.relative_path).read_bytes())["schema_version"] == 4
    )
    wrong_relative_path = f"source/attempts/wrong-{attempt_entry.record_id}.json"
    (path_drift_root / attempt_entry.relative_path).rename(path_drift_root / wrong_relative_path)
    manifest_path.write_bytes(
        canonical_json_bytes(
            manifest.model_copy(
                update={
                    "source_entries": tuple(
                        entry.model_copy(update={"relative_path": wrong_relative_path})
                        if entry == attempt_entry
                        else entry
                        for entry in manifest.source_entries
                    )
                }
            )
        )
    )
    _reseal_manifest(path_drift_root)
    assert validate_source_root(path_drift_root).disposition == ValidationDisposition.INVALID


def test_live_validation_matches_role_inventory_independent_of_manifest_path_order(
    tmp_path: Path,
) -> None:
    workload = _NativeFixtureJudgeWorkload()
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    control = _control(
        run_id="controlled-native-role-path-order",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        workload_id=case_manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="recorded-answer-v1",
        judge_role_binding_id="fake-judge-v1",
        provider_runtime_directory=tmp_path / "provider-runtime-role-path-order",
    )

    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=control.run_spec.run_id,
        adapter_profile_id=control.preflight_record.adapter_profile_id,
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
        judge_model_factory=_RecordedNativeModel,
        judge_role_binding_id="fake-judge-v1",
        control=control,
    )

    persisted_role_ids = tuple(
        json.loads((completed.capsule_root / entry.relative_path).read_bytes())["binding_id"]
        for entry in completed.manifest.source_entries
        if entry.record_kind == "model_role_binding"
    )
    assert persisted_role_ids == ("fake-judge-v1", "recorded-answer-v1")
    assert control.run_spec.model_role_binding_ids == (
        "recorded-answer-v1",
        "fake-judge-v1",
    )
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues


def test_live_validation_accepts_multiple_provider_usage_records_with_model_role_owners(
    tmp_path: Path,
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    control = _control(
        run_id="controlled-native-multiple-provider-usage",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        workload_id=case_manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="recorded-answer-v1",
        provider_runtime_directory=tmp_path / "provider-runtime-multiple-usage",
    )
    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=control.run_spec.run_id,
        adapter_profile_id=control.preflight_record.adapter_profile_id,
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_RecordedNativeModel,
        answer_role_binding_id="recorded-answer-v1",
        control=control,
    )
    root = completed.capsule_root
    plan = completed.ingestion_plan_records[0]
    attempt_id = plan.ordered_dispatch_attempt_ids[0]
    usage_path = next(
        path
        for path in (root / "source/usage").glob("*.json")
        if json.loads(path.read_bytes())["attempt_id"] == attempt_id
    )
    original_usage = json.loads(usage_path.read_bytes())
    original_usage_id = original_usage["usage_record_id"]
    additional_usage_id = canonical_sha256(
        ["oamb-fixture-secondary-provider-model-usage-v1", attempt_id]
    )
    _append_source_contract_copy(
        root,
        schema_name="token_usage_record",
        identity_field="usage_record_id",
        record_id=additional_usage_id,
    )
    _mutate_source_document(
        root,
        root / f"source/usage/{additional_usage_id}.json",
        attempt_id=attempt_id,
        parent_kind="ingestion_plan",
        parent_id=plan.ingestion_occurrence_id,
        stage="memory_ingest",
        operation_kind="recorded_fixture_secondary_ingest",
        model="deepseek-chat",
        budget_owner_kind="model_role",
        budget_owner_id="recorded-answer-v1",
        raw_response_ref=original_usage["raw_response_ref"],
        dispatch_route_id=original_usage["dispatch_route_id"],
        dispatch_route_hash=original_usage["dispatch_route_hash"],
    )
    plan_path = root / f"source/ingestion-plans/{plan.ingestion_occurrence_id}.json"
    plan_document = json.loads(plan_path.read_bytes())
    usage_index = plan_document["usage_record_ids"].index(original_usage_id)
    plan_usage_ids = list(plan_document["usage_record_ids"])
    plan_usage_ids[usage_index : usage_index + 1] = [additional_usage_id, original_usage_id]
    _mutate_source_document(root, plan_path, usage_record_ids=plan_usage_ids)
    capsule_manifest_path = root / "capsule-manifest.json"
    capsule_manifest = CapsuleManifest.model_validate_json(capsule_manifest_path.read_bytes())
    history_entry = next(
        entry
        for entry in capsule_manifest.source_entries
        if entry.record_kind == "history_attempt_record"
        and json.loads((root / entry.relative_path).read_bytes())["ingestion_occurrence_id"]
        == plan.ingestion_occurrence_id
    )
    history_path = root / history_entry.relative_path
    history_document = json.loads(history_path.read_bytes())
    history_document["ingestion_plan_record_hash"] = canonical_sha256(
        json.loads(plan_path.read_bytes())
    )
    history_document.pop("history_attempt_id")
    history_id = history_attempt_id(history_document)
    history_document["history_attempt_id"] = history_id
    new_history_path = history_path.with_name(f"{history_id}.json")
    history_path.rename(new_history_path)
    new_history_path.write_bytes(canonical_json_bytes(history_document))
    capsule_manifest_path.write_bytes(
        canonical_json_bytes(
            capsule_manifest.model_copy(
                update={
                    "source_entries": tuple(
                        entry.model_copy(
                            update={
                                "record_id": history_id,
                                "relative_path": new_history_path.relative_to(root).as_posix(),
                            }
                        )
                        if entry == history_entry
                        else entry
                        for entry in capsule_manifest.source_entries
                    )
                }
            )
        )
    )
    _reseal_manifest(root)
    cost_path = next(
        path
        for path in (root / "source/costs").glob("*.json")
        if json.loads(path.read_bytes())["attempt_id"] == attempt_id
    )
    _mutate_source_document(
        root,
        cost_path,
        source_usage_record_ids=[additional_usage_id, original_usage_id],
    )

    validation = validate_native_capsule(root)

    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues


def test_live_answer_and_judge_usage_is_converted_to_route_bound_v5(
    tmp_path: Path,
) -> None:
    workload = _NativeFixtureJudgeWorkload()
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    control = _control(
        run_id="controlled-native-measured-usage",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        workload_id=case_manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="recorded-answer-v1",
        judge_role_binding_id="fake-judge-v1",
        provider_runtime_directory=tmp_path / "provider-runtime-usage",
    )

    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=control.run_spec.run_id,
        adapter_profile_id=control.preflight_record.adapter_profile_id,
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=_MeasuredLiveModel,
        answer_role_binding_id="recorded-answer-v1",
        judge_model_factory=_MeasuredLiveModel,
        judge_role_binding_id="fake-judge-v1",
        control=control,
    )

    usage_documents = tuple(
        json.loads(path.read_bytes())
        for path in sorted((completed.capsule_root / "source/usage").glob("*.json"))
    )
    measured = tuple(
        TokenUsageRecordV5.model_validate_json(canonical_json_bytes(document))
        for document in usage_documents
        if document["stage"] in {"answer", "judge"}
    )
    assert {document["schema_version"] for document in usage_documents} == {5}
    assert {record.stage.value for record in measured} == {"answer", "judge"}
    assert all(
        (
            record.input_tokens,
            record.visible_output_tokens,
            record.supplier_reported_total_tokens,
        )
        == (13, 5, 18)
        for record in measured
    )
    assert all(
        record.unavailable_dimensions == ("cached_input_tokens", "reasoning_tokens")
        and record.proof_status == ProofStatus.MEASURED_PARTIAL
        and not record.billing_complete
        for record in measured
    )
    resource_documents = tuple(
        json.loads(path.read_bytes())
        for path in sorted((completed.capsule_root / "source/resources").glob("*.json"))
    )
    assert resource_documents
    assert {item["environment_hash"] for item in resource_documents} == {
        control.run_spec.environment_hash
    }
    cost_documents = tuple(
        json.loads(path.read_bytes())
        for path in sorted((completed.capsule_root / "source/costs").glob("*.json"))
    )
    measured_ids = {record.usage_record_id for record in measured}
    assert measured_ids <= {
        usage_id for cost in cost_documents for usage_id in cost["source_usage_record_ids"]
    }


@pytest.mark.parametrize("model_type", (_MismatchedLiveModel, _TamperedLiveModel))
def test_live_usage_mismatch_or_tamper_fails_without_orphaning_v2_usage(
    tmp_path: Path,
    model_type: type[_MeasuredLiveModel],
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    control = _control(
        run_id=f"controlled-native-rejected-usage-{model_type.__name__}",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        workload_id=case_manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="recorded-answer-v1",
        provider_runtime_directory=tmp_path / f"provider-runtime-{model_type.__name__}",
    )

    with pytest.raises(ValueError, match="usage"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id=control.run_spec.run_id,
            adapter_profile_id=control.preflight_record.adapter_profile_id,
            workload=workload,
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=model_type,
            answer_role_binding_id="recorded-answer-v1",
            control=control,
        )

    capsule_root = tmp_path / "capsules" / control.run_spec.run_id
    manifest = CapsuleManifest.model_validate_json(
        (capsule_root / "capsule-manifest.json").read_bytes()
    )
    assert manifest.run_id == control.run_spec.run_id
    assert all(
        json.loads(path.read_bytes())["schema_version"] == 5
        for path in (capsule_root / "source/usage").glob("*.json")
    )


def test_live_terminal_seal_rejects_unreferenced_captured_usage(tmp_path: Path) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    control = _control(
        run_id="controlled-native-unreferenced-usage",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        workload_id=case_manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="recorded-answer-v1",
        provider_runtime_directory=tmp_path / "provider-runtime-unreferenced",
    )

    with pytest.raises(ValueError, match="unreferenced records"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id=control.run_spec.run_id,
            adapter_profile_id=control.preflight_record.adapter_profile_id,
            workload=workload,
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=_ExtraUnreferencedLiveModel,
            answer_role_binding_id="recorded-answer-v1",
            control=control,
        )


def test_private_live_path_rejects_missing_lifecycle_composition(
    tmp_path: Path,
) -> None:
    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    control = _control(
        run_id="controlled-native-private-path",
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        workload_id=case_manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="recorded-answer-v1",
        provider_runtime_directory=tmp_path / "provider-runtime-private",
    )
    output_root = tmp_path / "capsules"
    native_run_module._acquire_native_run_owner(
        output_root / control.run_spec.run_id, control.run_spec.run_id, control=control
    )
    with pytest.raises(ValueError, match="budget and provider lifecycle ownership"):
        native_run_module._run_native_supervised(
            native_run_module._NativeRunRequest(
                output_root=output_root,
                run_id=control.run_spec.run_id,
                adapter_profile_id=control.preflight_record.adapter_profile_id,
                workload=workload,
                visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
                artifact_store_factory=ArtifactStore,
                memory_factory=_memory_factory,
                model_factory=_RecordedNativeModel,
                answer_role_binding_id="recorded-answer-v1",
                judge_model_factory=None,
                judge_role_binding_id=None,
                close_timeout_seconds=1.0,
                control=control,
            )
        )


def test_native_validator_rejects_case_inventory_removed_from_run_and_root(
    tmp_path: Path,
) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-missing-case-inventory")
    invalid_root = tmp_path / "missing-case-inventory"
    shutil.copytree(completed.capsule_root, invalid_root)
    removed_case = completed.case_records[0]
    removed_path = invalid_root / "source" / "cases" / f"{removed_case.case_occurrence_id}.json"
    manifest_path = invalid_root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(manifest_path.read_bytes())
    removed_relative_path = removed_path.relative_to(invalid_root).as_posix()
    removed_path.unlink()
    manifest_path.write_bytes(
        canonical_json_bytes(
            manifest.model_copy(
                update={
                    "source_entries": tuple(
                        entry
                        for entry in manifest.source_entries
                        if entry.relative_path != removed_relative_path
                    )
                }
            )
        )
    )
    run_path = _first_source_document(invalid_root, "run_record", 1)
    run_document = json.loads(run_path.read_bytes())
    _mutate_source_document(
        invalid_root,
        run_path,
        case_occurrence_ids=tuple(
            case_id
            for case_id in run_document["case_occurrence_ids"]
            if case_id != removed_case.case_occurrence_id
        ),
    )

    validation = validate_native_capsule(invalid_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "native-manifest-record-inventory-mismatch" in {
        issue.code for issue in validation.issues
    }


def test_recorded_native_judge_path_seals_judge_attempt_and_exact_fraction(
    tmp_path: Path,
) -> None:
    completed = _run_judged_fixture_capsule(tmp_path, "native-fixture-judged")

    validation = validate_native_capsule(completed.capsule_root)
    judged = next(
        record for record in completed.case_records if record.evaluation_disposition == "judged"
    )

    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    assert len(judged.attempt_ids) == 3
    assert (judged.metric_numerator, judged.metric_denominator) == (1, 1)


def test_native_judge_path_uses_a_separately_bound_model_client(tmp_path: Path) -> None:
    class BoundedJudgeWorkload(_NativeFixtureJudgeWorkload):
        def evaluate(
            self,
            case_plan: CasePlan,
            answer: AnswerValue,
        ) -> DeterministicEvaluation | JudgeRequest:
            evaluation = super().evaluate(case_plan, answer)
            if isinstance(evaluation, JudgeRequest):
                return replace(evaluation, max_output_tokens=1024)
            return evaluation

    class AnswerOnlyModel(_RecordedNativeModel):
        async def complete(self, request: ModelRequest) -> ModelReceipt:
            if request.stage != "answer":
                raise AssertionError("answer client received a non-answer request")
            if request.max_output_tokens is not None:
                raise AssertionError("answer client received an output ceiling")
            return await super().complete(request)

    class JudgeOnlyModel(_RecordedNativeModel):
        async def complete(self, request: ModelRequest) -> ModelReceipt:
            if request.stage != "judge":
                raise AssertionError("judge client received a non-judge request")
            if request.max_output_tokens is not None:
                raise AssertionError("judge client received an output ceiling")
            return await super().complete(request)

    completed = run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="native-fixture-separate-judge",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=BoundedJudgeWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=AnswerOnlyModel,
        answer_role_binding_id="recorded-answer-v1",
        judge_model_factory=JudgeOnlyModel,
        judge_role_binding_id="fake-judge-v1",
    )

    assert validate_native_capsule(completed.capsule_root).disposition == (
        ValidationDisposition.VALIDATED
    )


def test_native_validator_rejects_a_missing_supporting_raw(tmp_path: Path) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-missing-support")
    invalid_root = tmp_path / "missing-support"
    shutil.copytree(completed.capsule_root, invalid_root)
    case = completed.case_records[0]
    case_path = invalid_root / "source" / "cases" / f"{case.case_occurrence_id}.json"

    _remove_raw_reference(invalid_root, case.retrieval_supporting_raw_refs[0])
    _mutate_source_document(invalid_root, case_path, retrieval_supporting_raw_refs=())

    assert "retrieval-support-inventory-mismatch" in _issue_codes(invalid_root)


def test_native_validator_binds_answer_prompt_bytes_to_attempt_fingerprint(
    tmp_path: Path,
) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-answer-prompt-drift")
    case_path = _first_source_document(completed.capsule_root, "case_record", 3)
    case = json.loads(case_path.read_bytes())

    _mutate_source_document(
        completed.capsule_root,
        case_path,
        prompt_raw_ref=case["answer_raw_ref"],
        prompt_sha256=case["answer_raw_ref"],
    )

    assert "prompt-answer-mismatch" in _issue_codes(completed.capsule_root)


def test_native_validator_binds_judge_prompt_bytes_to_attempt_fingerprint(
    tmp_path: Path,
) -> None:
    completed = _run_judged_fixture_capsule(tmp_path, "native-fixture-judge-prompt-drift")
    judged_case = next(
        item for item in completed.case_records if item.evaluation_disposition == "judged"
    )
    case_path = (
        completed.capsule_root / "source" / "cases" / f"{judged_case.case_occurrence_id}.json"
    )
    case = json.loads(case_path.read_bytes())

    _mutate_source_document(
        completed.capsule_root,
        case_path,
        judge_prompt_raw_ref=case["answer_raw_ref"],
    )

    assert "judge-prompt-mismatch" in _issue_codes(completed.capsule_root)


@pytest.mark.parametrize(
    ("schema_name", "identity_field", "issue_code"),
    (
        (
            "occurrence_claim_record",
            "claim_id",
            "attempt-claim-inventory-mismatch",
        ),
        (
            "budget_reservation_record",
            "reservation_id",
            "attempt-reservation-inventory-mismatch",
        ),
    ),
)
def test_native_validator_rejects_orphan_attempt_prerequisites(
    tmp_path: Path,
    schema_name: str,
    identity_field: str,
    issue_code: str,
) -> None:
    completed = _run_fixture_capsule(tmp_path, f"native-fixture-orphan-{schema_name}")
    invalid_root = tmp_path / f"orphan-{schema_name}"
    shutil.copytree(completed.capsule_root, invalid_root)

    _append_source_contract_copy(
        invalid_root,
        schema_name=schema_name,
        identity_field=identity_field,
        record_id="f" * 64,
    )

    validation = validate_native_capsule(invalid_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert issue_code in {issue.code for issue in validation.issues}


def test_native_validator_rejects_a_known_contract_with_an_unknown_version(
    tmp_path: Path,
) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-unknown-contract-version")
    unknown_record_id = "f" * 64
    relative_parent = Path("source") / "unknown-contract-version"
    _append_source_contract_copy(
        completed.capsule_root,
        schema_name="case_record",
        identity_field="case_occurrence_id",
        record_id=unknown_record_id,
        relative_parent=relative_parent,
    )
    copied_path = completed.capsule_root / relative_parent / f"{unknown_record_id}.json"
    _mutate_source_document(completed.capsule_root, copied_path, schema_version=999)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "schema-invalid" in {issue.code for issue in validation.issues}


@pytest.mark.parametrize(
    ("schema_name", "identity_field", "issue_code"),
    (
        (
            "occurrence_claim_record",
            "claim_id",
            "attempt-claim-inventory-mismatch",
        ),
        (
            "budget_reservation_record",
            "reservation_id",
            "attempt-reservation-inventory-mismatch",
        ),
    ),
)
def test_native_validator_rejects_duplicate_attempt_prerequisite_relations(
    tmp_path: Path,
    schema_name: str,
    identity_field: str,
    issue_code: str,
) -> None:
    completed = _run_fixture_capsule(tmp_path, f"native-fixture-duplicate-{schema_name}")
    invalid_root = tmp_path / f"duplicate-{schema_name}"
    shutil.copytree(completed.capsule_root, invalid_root)
    template_path = _first_source_document(invalid_root, schema_name, 1)
    template = json.loads(template_path.read_bytes())

    _append_source_contract_copy(
        invalid_root,
        schema_name=schema_name,
        identity_field=identity_field,
        record_id=template[identity_field],
        relative_parent=Path("source") / "duplicate-attempt-prerequisites" / schema_name,
    )

    validation = validate_native_capsule(invalid_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert issue_code in {issue.code for issue in validation.issues}


@pytest.mark.parametrize(
    ("schema_name", "field_name", "field_value", "issue_code"),
    (
        (
            "run_lease_record",
            "acquired_at",
            "2026-01-01T00:00:01Z",
            "run-lease-identity-mismatch",
        ),
        (
            "occurrence_claim_record",
            "claimed_at",
            "2026-01-01T00:00:01Z",
            "attempt-claim-identity-mismatch",
        ),
        (
            "budget_reservation_record",
            "reserved_at",
            "2026-01-01T00:00:01Z",
            "attempt-reservation-identity-mismatch",
        ),
    ),
)
def test_native_validator_recomputes_durable_prerequisite_identities(
    tmp_path: Path,
    schema_name: str,
    field_name: str,
    field_value: str,
    issue_code: str,
) -> None:
    completed = _run_fixture_capsule(tmp_path, f"native-fixture-identity-{schema_name}")
    path = _first_source_document(completed.capsule_root, schema_name, 1)

    _mutate_source_document(completed.capsule_root, path, **{field_name: field_value})

    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.INVALID
    assert issue_code in {issue.code for issue in validation.issues}


def test_native_validator_accepts_a_strict_successor_lease_chain_and_claim_binding(
    tmp_path: Path,
) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-successor-lease")
    lease_path = _first_source_document(completed.capsule_root, "run_lease_record", 1)
    predecessor = RunLeaseRecord.model_validate_json(lease_path.read_bytes())
    successor = _successor_run_lease(predecessor)
    _append_run_lease(completed.capsule_root, successor)
    _rebind_first_occurrence_claim(completed.capsule_root, successor)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues


@pytest.mark.parametrize(
    "invalid_chain",
    (
        "gap",
        "wrong_predecessor",
        "mixed_run",
        "mixed_provider_project",
        "mixed_provider_profile",
    ),
)
def test_native_validator_rejects_a_non_contiguous_or_mixed_successor_lease(
    tmp_path: Path,
    invalid_chain: str,
) -> None:
    completed = _run_fixture_capsule(tmp_path, f"native-fixture-lease-chain-{invalid_chain}")
    lease_path = _first_source_document(completed.capsule_root, "run_lease_record", 1)
    predecessor = RunLeaseRecord.model_validate_json(lease_path.read_bytes())
    updates: dict[str, object] = {}
    if invalid_chain == "gap":
        updates["lease_epoch"] = 3
    elif invalid_chain == "wrong_predecessor":
        updates["predecessor_lease_record_hash"] = "f" * 64
    elif invalid_chain == "mixed_run":
        updates["run_id"] = "another-run"
    elif invalid_chain == "mixed_provider_project":
        updates["provider_project_id"] = "another-provider-project"
    elif invalid_chain == "mixed_provider_profile":
        updates["provider_profile_id"] = "another-provider-profile"
    successor = _successor_run_lease(predecessor, **updates)
    _append_run_lease(completed.capsule_root, successor)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "run-lease-chain-mismatch" in {issue.code for issue in validation.issues}


@pytest.mark.parametrize("duplicate_kind", ("epoch", "hash"))
def test_native_validator_rejects_duplicate_lease_epochs_or_hashes(
    tmp_path: Path,
    duplicate_kind: str,
) -> None:
    completed = _run_fixture_capsule(tmp_path, f"native-fixture-duplicate-lease-{duplicate_kind}")
    lease_path = _first_source_document(completed.capsule_root, "run_lease_record", 1)
    predecessor = RunLeaseRecord.model_validate_json(lease_path.read_bytes())
    successor = _successor_run_lease(predecessor)
    _append_run_lease(completed.capsule_root, successor)
    duplicate = (
        _successor_run_lease(predecessor, owner_id="another-successor-owner")
        if duplicate_kind == "epoch"
        else successor
    )
    _append_run_lease(
        completed.capsule_root,
        duplicate,
        relative_path=f"source/duplicate-run-leases/{duplicate.lease_record_hash}.json",
    )

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "run-lease-chain-mismatch" in {issue.code for issue in validation.issues}


@pytest.mark.parametrize("claim_mismatch", ("hash", "epoch", "owner"))
def test_native_validator_binds_each_claim_to_its_referenced_lease(
    tmp_path: Path,
    claim_mismatch: str,
) -> None:
    completed = _run_fixture_capsule(tmp_path, f"native-fixture-claim-lease-{claim_mismatch}")
    lease_path = _first_source_document(completed.capsule_root, "run_lease_record", 1)
    predecessor = RunLeaseRecord.model_validate_json(lease_path.read_bytes())
    successor = _successor_run_lease(predecessor)
    _append_run_lease(completed.capsule_root, successor)
    updates: dict[str, object] = {}
    if claim_mismatch == "hash":
        updates["lease_record_hash"] = "f" * 64
    elif claim_mismatch == "epoch":
        updates["lease_epoch"] = predecessor.lease_epoch
    elif claim_mismatch == "owner":
        updates["owner_id"] = predecessor.owner_id
    _rebind_first_occurrence_claim(completed.capsule_root, successor, **updates)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "attempt-evidence-mismatch" in {issue.code for issue in validation.issues}


def test_native_validator_rejects_projection_drift(tmp_path: Path) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-projection-drift")
    invalid_root = tmp_path / "projection-drift"
    shutil.copytree(completed.capsule_root, invalid_root)
    plan_path = _first_source_document(invalid_root, "ingestion_plan_record", 2)
    _mutate_source_document(
        invalid_root,
        plan_path,
        state="error",
        projected_source_unit_ids=(),
    )

    assert "projection-source-order-mismatch" in _issue_codes(invalid_root)


def test_native_validator_rejects_visible_decision_drift(tmp_path: Path) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-visible-drift")
    invalid_root = tmp_path / "visible-drift"
    shutil.copytree(completed.capsule_root, invalid_root)
    case_path = _first_source_document(invalid_root, "case_record", 3)
    case = json.loads(case_path.read_bytes())

    _mutate_source_document(
        invalid_root,
        case_path,
        visible_kept_count=case["visible_kept_count"] - 1,
        visible_dropped_count=case["visible_dropped_count"] + 1,
    )

    assert "visible-decision-mismatch" in _issue_codes(invalid_root)


def test_native_validator_rejects_query_mutation(tmp_path: Path) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-query-mutation")
    invalid_root = tmp_path / "query-mutation"
    shutil.copytree(completed.capsule_root, invalid_root)
    case_path = _first_source_document(invalid_root, "case_record", 3)

    _mutate_source_document(
        invalid_root,
        case_path,
        state="error",
        post_query_state_sha256="f" * 64,
        query_mutation_status="changed",
        error_stage="memory_query",
    )

    assert "query-state-mismatch" in _issue_codes(invalid_root)


def test_native_validator_rejects_metric_fraction_drift(tmp_path: Path) -> None:
    completed = _run_fixture_capsule(tmp_path, "native-fixture-metric-drift")
    invalid_root = tmp_path / "metric-drift"
    shutil.copytree(completed.capsule_root, invalid_root)
    case_path = _first_source_document(invalid_root, "case_record", 3)

    _mutate_source_document(invalid_root, case_path, metric_numerator=0)

    assert "metric-fraction-mismatch" in _issue_codes(invalid_root)


def test_native_runtime_attempts_both_port_closes_before_final_seal(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    memory_closed = context.Event()
    model_closed = context.Event()

    class TrackingMemory(_RecordedNativeMemory):
        async def close(self) -> None:
            memory_closed.set()
            await super().close()

    class FailingCloseModel(_RecordedNativeModel):
        async def close(self) -> None:
            model_closed.set()
            await super().close()
            raise RuntimeError("planted recorded-model close failure")

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> TrackingMemory:
        return TrackingMemory(store)

    with pytest.raises(RuntimeError, match="planted recorded-model close failure"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-close-failure",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=FailingCloseModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    assert memory_closed.is_set()
    assert model_closed.is_set()
    root = tmp_path / "capsules" / "native-fixture-close-failure"
    assert (root / "capsule-manifest.json").is_file()
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


def test_native_runtime_closes_memory_when_model_construction_fails(tmp_path: Path) -> None:
    memory_closed = multiprocessing.get_context("fork").Event()

    class TrackingMemory(_RecordedNativeMemory):
        async def close(self) -> None:
            memory_closed.set()
            await super().close()

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> TrackingMemory:
        return TrackingMemory(store)

    def failing_model_factory(_store: ArtifactStorePort) -> _RecordedNativeModel:
        raise RuntimeError("planted recorded-model construction failure")

    with pytest.raises(RuntimeError, match="planted recorded-model construction failure"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-model-construction-failure",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=failing_model_factory,
            answer_role_binding_id="recorded-answer-v1",
        )

    assert memory_closed.is_set()
    root = tmp_path / "capsules" / "native-fixture-model-construction-failure"
    assert (root / "capsule-manifest.json").is_file()
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


@pytest.mark.parametrize("invalid_timeout", (float("nan"), float("inf"), float("-inf")))
def test_native_runtime_rejects_non_finite_close_timeouts(
    tmp_path: Path,
    invalid_timeout: float,
) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id=f"native-fixture-invalid-timeout-{str(invalid_timeout).replace('-', 'negative-')}",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
            close_timeout_seconds=invalid_timeout,
        )


def test_native_run_ownership_blocks_repeat_dispatch_before_port_construction(
    tmp_path: Path,
) -> None:
    factory_events = multiprocessing.get_context("fork").Queue()

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> _RecordedNativeMemory:
        factory_events.put("memory_factory")
        return _RecordedNativeMemory(store)

    def model_factory(store: ArtifactStorePort) -> _RecordedNativeModel:
        factory_events.put("model_factory")
        return _RecordedNativeModel(store)

    run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="native-fixture-single-owner",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=model_factory,
        answer_role_binding_id="recorded-answer-v1",
    )

    with pytest.raises(Exception, match="already owned"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-single-owner",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=model_factory,
            answer_role_binding_id="recorded-answer-v1",
        )

    assert {factory_events.get(timeout=1), factory_events.get(timeout=1)} == {
        "memory_factory",
        "model_factory",
    }
    with pytest.raises(queue.Empty):
        factory_events.get(timeout=0.05)


def test_dispatched_answer_failure_seals_intent_receipt_and_terminal_attempt(
    tmp_path: Path,
) -> None:
    class FailingAnswerModel(_RecordedNativeModel):
        async def complete(self, request: ModelRequest) -> ModelReceipt:
            if request.stage == "answer":
                raise RuntimeError("planted answer dispatch failure")
            return await super().complete(request)

    root = tmp_path / "capsules" / "native-fixture-answer-failure"
    with pytest.raises(RuntimeError, match="planted answer dispatch failure"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-answer-failure",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=FailingAnswerModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    intents = tuple((root / "source" / "attempt-intents").glob("*.json"))
    receipts = tuple((root / "source" / "attempt-receipts").glob("*.json"))
    terminals = tuple((root / "source" / "attempts").glob("*.json"))
    answer_intents = [json.loads(path.read_bytes()) for path in intents]
    answer_intents = [item for item in answer_intents if item["stage"] == "answer"]
    answer_terminals = [json.loads(path.read_bytes()) for path in terminals]
    answer_terminals = [item for item in answer_terminals if item["stage"] == "answer"]
    assert len(answer_intents) == 1
    assert len(receipts) >= 1
    assert len(answer_terminals) == 1
    assert answer_terminals[0]["outcome"] in {"failed", "unknown_outcome"}
    assert (root / "capsule-manifest.json").is_file()
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


def test_before_dispatch_cancellation_seals_known_cancelled_attempt(
    tmp_path: Path,
) -> None:
    class CancelledBeforeDispatchMemory(_RecordedNativeMemory):
        async def ingest(
            self,
            request: IngestionDispatchRequest,
        ) -> IngestionDispatchReceipt:
            raise MemorySystemCallCancelledBeforeDispatch(
                f"planted cancellation before dispatch for {request.attempt_id}"
            )

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> CancelledBeforeDispatchMemory:
        return CancelledBeforeDispatchMemory(store)

    root = tmp_path / "capsules" / "native-fixture-cancel-before-dispatch"
    with pytest.raises(
        MemorySystemCallCancelledBeforeDispatch,
        match="planted cancellation before dispatch",
    ):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-cancel-before-dispatch",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    attempt = json.loads(next((root / "source" / "attempts").glob("*.json")).read_bytes())
    receipt = json.loads(next((root / "source" / "attempt-receipts").glob("*.json")).read_bytes())
    assert attempt["outcome"] == "cancelled"
    assert receipt["receipt_kind"] == "error"
    assert isinstance(attempt["raw_error_ref"], str)
    assert receipt["raw_error_ref"] == attempt["raw_error_ref"]
    assert (root / "capsule-manifest.json").is_file()
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


@pytest.mark.parametrize(
    ("error_type", "message"),
    (
        (MemorySystemCallCancelledBeforeDispatch, "planted query cancel before dispatch"),
        (MemorySystemCallCancelledUnknownOutcome, "planted query cancel with unknown outcome"),
    ),
)
def test_supervised_query_cancellation_preserves_the_provider_error_type(
    tmp_path: Path,
    error_type: type[asyncio.CancelledError],
    message: str,
) -> None:
    class CancelledQueryMemory(_RecordedNativeMemory):
        async def retrieve(self, _request: RetrievalRequest) -> NativeEvidenceBatch:
            raise error_type(message)

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> CancelledQueryMemory:
        return CancelledQueryMemory(store)

    with pytest.raises(error_type, match=message):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id=f"native-fixture-query-{error_type.__name__.lower()}",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
        )


def test_supervised_boundary_preserves_explicit_unknown_outcome_type(tmp_path: Path) -> None:
    class UnknownOutcomeMemory(_RecordedNativeMemory):
        async def ingest(
            self,
            _request: IngestionDispatchRequest,
        ) -> IngestionDispatchReceipt:
            raise MemorySystemCallUnknownOutcome(
                "planted dispatched write without a receipt",
                failure_kind="planted_unknown",
            )

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> UnknownOutcomeMemory:
        return UnknownOutcomeMemory(store)

    root = tmp_path / "capsules" / "native-fixture-explicit-unknown"
    with pytest.raises(
        MemorySystemCallUnknownOutcome,
        match="planted dispatched write without a receipt",
    ) as captured:
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-explicit-unknown",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    assert captured.value.failure_kind == "planted_unknown"
    attempt = json.loads(next((root / "source" / "attempts").glob("*.json")).read_bytes())
    assert attempt["outcome"] == "unknown_outcome"


def test_child_crash_reconstructs_an_aborted_terminal_root(tmp_path: Path) -> None:
    def crashing_model_factory(_store: ArtifactStorePort) -> _RecordedNativeModel:
        os._exit(17)

    root = tmp_path / "capsules" / "native-fixture-child-crash"
    with pytest.raises(NativeRunProcessError, match="status 17|closed its result channel"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-child-crash",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=crashing_model_factory,
            answer_role_binding_id="recorded-answer-v1",
        )

    manifest = CapsuleManifest.model_validate_json((root / "capsule-manifest.json").read_bytes())
    run_records = [
        json.loads((root / entry.relative_path).read_bytes())
        for entry in manifest.source_entries
        if entry.record_kind == "run_record"
    ]
    assert [record["state"] for record in run_records] == ["aborted"]
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


def test_supervised_child_crash_releases_lifecycle_for_a_fresh_run(tmp_path: Path) -> None:
    class CrashingIngestMemory(_RecordedNativeMemory):
        async def ingest(
            self,
            _request: IngestionDispatchRequest,
        ) -> IngestionDispatchReceipt:
            os._exit(17)

    workload = _NativeFixtureWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    provider_runtime = (tmp_path / "provider-runtime").resolve()
    run_id = "native-fixture-supervised-crash"
    control = _control(
        run_id=run_id,
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest_hash=manifest.manifest_hash,
        workload_id=manifest.workload_id,
        memory_system_id="fake-memory",
        runtime_binding_hash=canonical_sha256(["oamb-fake-runtime-v1"]),
        adapter_profile_id="recorded-native-fixture-v1",
        answer_role_binding_id="answer-binding",
        provider_runtime_directory=provider_runtime,
    )

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> CrashingIngestMemory:
        return CrashingIngestMemory(store)

    root = tmp_path / "capsules" / run_id
    with pytest.raises(NativeRunProcessError, match="status 17|closed its result channel"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id=run_id,
            adapter_profile_id="recorded-native-fixture-v1",
            workload=workload,
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="answer-binding",
            control=control,
        )

    run_record = json.loads((root / "source" / "run" / f"{run_id}.json").read_bytes())
    assert run_record["state"] == "aborted"
    assert not (provider_runtime / "active-operation").exists()
    attempts = provider_runtime / "active-provider-attempts"
    assert not attempts.exists() or not tuple(attempts.iterdir())


def test_child_crash_after_successful_closes_cannot_leave_a_finalized_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_run = native_run_module._run_native_vertical_slice

    async def crash_after_successful_closes(**kwargs: Any) -> None:
        await original_run(**kwargs)
        os._exit(73)

    monkeypatch.setattr(
        native_run_module,
        "_run_native_vertical_slice",
        crash_after_successful_closes,
    )
    root = tmp_path / "capsules" / "native-fixture-post-close-crash"

    with pytest.raises(NativeRunProcessError, match="status 73|closed its result channel"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-post-close-crash",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    manifest = CapsuleManifest.model_validate_json((root / "capsule-manifest.json").read_bytes())
    run_records = [
        json.loads((root / entry.relative_path).read_bytes())
        for entry in manifest.source_entries
        if entry.record_kind == "run_record"
    ]
    assert [record["state"] for record in run_records] == ["aborted"]
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


def test_child_crash_during_close_records_active_and_skipped_clients(tmp_path: Path) -> None:
    class CrashingCloseModel(_RecordedNativeModel):
        async def close(self) -> None:
            os._exit(72)

    root = tmp_path / "capsules" / "native-fixture-close-crash"
    with pytest.raises(BaseExceptionGroup):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-close-crash",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=CrashingCloseModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    close_errors = [
        json.loads(path.read_bytes()) for path in (root / "source" / "close-errors").glob("*.json")
    ]
    assert {record["shutdown_stage"] for record in close_errors} == {
        "model_close",
        "memory_close",
    }
    assert len(close_errors) == 2
    assert (root / "capsule-manifest.json").is_file()
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


def test_dispatch_and_failure_evidence_errors_are_both_preserved(tmp_path: Path) -> None:
    class FailingAnswerModel(_RecordedNativeModel):
        async def complete(self, request: ModelRequest) -> ModelReceipt:
            if request.stage == "answer":
                raise LookupError("planted provider answer failure")
            return await super().complete(request)

    class FailingFailureReceiptStore(ArtifactStore):
        def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
            document = json.loads(request.canonical_bytes)
            if (
                document.get("schema_name") == "attempt_receipt_record"
                and document.get("receipt_kind") == "error"
            ):
                raise RuntimeError("planted failure-receipt seal failure")
            return super().seal_source_record(request)

    with pytest.raises(BaseExceptionGroup) as captured:
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-dispatch-and-seal-failure",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=FailingFailureReceiptStore,
            memory_factory=_memory_factory,
            model_factory=FailingAnswerModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    messages = tuple(str(error) for error in captured.value.exceptions)
    assert any("planted provider answer failure" in message for message in messages)
    assert any("planted failure-receipt seal failure" in message for message in messages)


def test_cancellation_semantics_survive_a_failure_evidence_seal_error(tmp_path: Path) -> None:
    class CancelledBeforeDispatchMemory(_RecordedNativeMemory):
        async def ingest(
            self,
            _request: IngestionDispatchRequest,
        ) -> IngestionDispatchReceipt:
            raise MemorySystemCallCancelledBeforeDispatch("planted zero-dispatch cancel")

    class FailingFailureReceiptStore(ArtifactStore):
        def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
            document = json.loads(request.canonical_bytes)
            if document.get("schema_name") == "attempt_receipt_record":
                raise RuntimeError("planted cancellation-receipt seal failure")
            return super().seal_source_record(request)

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> CancelledBeforeDispatchMemory:
        return CancelledBeforeDispatchMemory(store)

    with pytest.raises(NativeCancellationEvidenceError) as captured:
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-cancel-and-seal-failure",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=FailingFailureReceiptStore,
            memory_factory=memory_factory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    primary_error, evidence_error = captured.value.errors
    assert isinstance(primary_error, MemorySystemCallCancelledBeforeDispatch)
    assert "planted cancellation-receipt seal failure" in str(evidence_error)


def test_close_timeout_retires_process_and_persists_later_close_as_skipped(
    tmp_path: Path,
) -> None:
    class HangingModel(_RecordedNativeModel):
        async def close(self) -> None:
            await asyncio.Event().wait()

    class TrackingMemory(_RecordedNativeMemory):
        async def close(self) -> None:
            raise RuntimeError("planted memory close failure")

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> TrackingMemory:
        return TrackingMemory(store)

    root = tmp_path / "capsules" / "native-fixture-close-timeout"
    with pytest.raises(BaseExceptionGroup) as captured:
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-close-timeout",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=HangingModel,
            answer_role_binding_id="recorded-answer-v1",
            close_timeout_seconds=0.01,
        )

    assert len(captured.value.exceptions) == 2
    close_records = tuple((root / "source" / "close-errors").glob("*.json"))
    assert len(close_records) == 2
    assert {json.loads(path.read_bytes())["shutdown_stage"] for path in close_records} == {
        "model_close",
        "memory_close",
    }
    assert (root / "capsule-manifest.json").is_file()
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


def test_close_timeout_is_hard_when_client_swallows_cancellation(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    close_started_receiver, close_started_sender = context.Pipe(duplex=False)
    release_close = context.Event()
    runner_errors: list[BaseException] = []

    class CancellationSwallowingModel(_RecordedNativeModel):
        async def close(self) -> None:
            close_started_sender.send(True)
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                while not release_close.is_set():
                    await asyncio.sleep(0.01)

    root = tmp_path / "capsules" / "native-fixture-hard-close-timeout"

    def run_fixture() -> None:
        try:
            run_native_vertical_slice(
                output_root=tmp_path / "capsules",
                run_id="native-fixture-hard-close-timeout",
                adapter_profile_id="recorded-native-fixture-v1",
                workload=_NativeFixtureWorkload(),
                visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
                artifact_store_factory=ArtifactStore,
                memory_factory=_memory_factory,
                model_factory=CancellationSwallowingModel,
                answer_role_binding_id="recorded-answer-v1",
                close_timeout_seconds=0.01,
            )
        except BaseException as exc:
            runner_errors.append(exc)

    runner = threading.Thread(target=run_fixture, daemon=True)
    runner.start()
    reached_close_stage = close_started_receiver.poll(HARD_CLOSE_STAGE_REACH_BOUND_SECONDS)
    if reached_close_stage:
        close_started_receiver.recv()
    runner.join(timeout=HARD_CLOSE_RETURN_BOUND_SECONDS)
    returned_within_bound = not runner.is_alive()
    release_close.set()
    runner.join(timeout=HARD_CLOSE_STAGE_REACH_BOUND_SECONDS)
    close_started_receiver.close()
    close_started_sender.close()

    assert reached_close_stage
    assert returned_within_bound
    assert not runner.is_alive()
    assert runner_errors
    assert tuple((root / "source" / "close-errors").glob("*.json"))
    assert (root / "capsule-manifest.json").is_file()
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


def test_successful_shutdown_orders_model_then_memory_then_final_manifest(tmp_path: Path) -> None:
    events = multiprocessing.get_context("fork").Queue()

    class TrackingStore(ArtifactStore):
        def finalize_capsule(self, *, run_id: str, run_spec_hash: str) -> CapsuleManifest:
            events.put("finalize")
            return super().finalize_capsule(run_id=run_id, run_spec_hash=run_spec_hash)

    class TrackingModel(_RecordedNativeModel):
        async def close(self) -> None:
            events.put("model.close")
            await super().close()

    class TrackingMemory(_RecordedNativeMemory):
        async def close(self) -> None:
            events.put("memory.close")
            await super().close()

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> TrackingMemory:
        return TrackingMemory(store)

    run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="native-fixture-shutdown-order",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=TrackingStore,
        memory_factory=memory_factory,
        model_factory=TrackingModel,
        answer_role_binding_id="recorded-answer-v1",
    )

    assert [events.get(timeout=1) for _ in range(3)] == [
        "model.close",
        "memory.close",
        "finalize",
    ]


def test_successful_dual_model_shutdown_orders_judge_answer_memory_then_manifest(
    tmp_path: Path,
) -> None:
    events = multiprocessing.get_context("fork").Queue()

    class TrackingStore(ArtifactStore):
        def finalize_capsule(self, *, run_id: str, run_spec_hash: str) -> CapsuleManifest:
            events.put("finalize")
            return super().finalize_capsule(run_id=run_id, run_spec_hash=run_spec_hash)

    class TrackingAnswerModel(_RecordedNativeModel):
        async def close(self) -> None:
            events.put("answer.close")
            await super().close()

    class TrackingJudgeModel(_RecordedNativeModel):
        async def close(self) -> None:
            events.put("judge.close")
            await super().close()

    class TrackingMemory(_RecordedNativeMemory):
        async def close(self) -> None:
            events.put("memory.close")
            await super().close()

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> TrackingMemory:
        return TrackingMemory(store)

    run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="native-fixture-dual-model-shutdown-order",
        adapter_profile_id="recorded-native-fixture-v1",
        workload=_NativeFixtureJudgeWorkload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=TrackingStore,
        memory_factory=memory_factory,
        model_factory=TrackingAnswerModel,
        answer_role_binding_id="recorded-answer-v1",
        judge_model_factory=TrackingJudgeModel,
        judge_role_binding_id="fake-judge-v1",
    )

    assert [events.get(timeout=1) for _ in range(4)] == [
        "judge.close",
        "answer.close",
        "memory.close",
        "finalize",
    ]


def test_judge_close_crash_records_answer_and_memory_as_skipped(tmp_path: Path) -> None:
    class CrashingJudgeModel(_RecordedNativeModel):
        async def close(self) -> None:
            os._exit(74)

    root = tmp_path / "capsules" / "native-fixture-judge-close-crash"
    with pytest.raises(BaseExceptionGroup):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-judge-close-crash",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureJudgeWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=_memory_factory,
            model_factory=_RecordedNativeModel,
            answer_role_binding_id="recorded-answer-v1",
            judge_model_factory=CrashingJudgeModel,
            judge_role_binding_id="fake-judge-v1",
        )

    close_errors = [
        json.loads(path.read_bytes()) for path in (root / "source" / "close-errors").glob("*.json")
    ]
    assert {record["shutdown_stage"] for record in close_errors} == {
        "judge_model_close",
        "model_close",
        "memory_close",
    }
    assert validate_native_capsule(root).disposition == ValidationDisposition.INVALID


def test_concurrent_same_run_has_exactly_one_dispatch_owner(tmp_path: Path) -> None:
    context = multiprocessing.get_context("fork")
    entered_factory = context.Event()
    release_factory = context.Event()
    first_errors: list[BaseException] = []

    def blocking_memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> _RecordedNativeMemory:
        entered_factory.set()
        assert release_factory.wait(timeout=5)
        return _RecordedNativeMemory(store)

    def first_run() -> None:
        try:
            run_native_vertical_slice(
                output_root=tmp_path / "capsules",
                run_id="native-fixture-concurrent-owner",
                adapter_profile_id="recorded-native-fixture-v1",
                workload=_NativeFixtureWorkload(),
                visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
                artifact_store_factory=ArtifactStore,
                memory_factory=blocking_memory_factory,
                model_factory=_RecordedNativeModel,
                answer_role_binding_id="recorded-answer-v1",
            )
        except BaseException as exc:
            first_errors.append(exc)

    thread = threading.Thread(target=first_run)
    thread.start()
    assert entered_factory.wait(timeout=5)
    try:
        with pytest.raises(Exception, match="already owned"):
            _run_fixture_capsule(tmp_path, "native-fixture-concurrent-owner")
    finally:
        release_factory.set()
        thread.join(timeout=10)

    assert not thread.is_alive()
    assert first_errors == []


def test_execution_and_close_failures_share_one_terminal_diagnostic_root(
    tmp_path: Path,
) -> None:
    class FailingAnswerModel(_RecordedNativeModel):
        async def complete(self, request: ModelRequest) -> ModelReceipt:
            if request.stage == "answer":
                raise RuntimeError("planted execution failure")
            return await super().complete(request)

    class FailingCloseMemory(_RecordedNativeMemory):
        async def close(self) -> None:
            raise RuntimeError("planted close failure")

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> FailingCloseMemory:
        return FailingCloseMemory(store)

    root = tmp_path / "capsules" / "native-fixture-execution-close-failure"
    with pytest.raises(BaseExceptionGroup) as captured:
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-execution-close-failure",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=ArtifactStore,
            memory_factory=memory_factory,
            model_factory=FailingAnswerModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    messages = tuple(str(item) for item in captured.value.exceptions)
    assert any("execution failure" in message for message in messages)
    assert any("close failure" in message for message in messages)
    assert tuple((root / "source" / "close-errors").glob("*.json"))
    assert tuple((root / "source" / "attempt-intents").glob("*.json"))
    assert (root / "capsule-manifest.json").is_file()
    validation = validate_native_capsule(root)
    assert validation.disposition == ValidationDisposition.INVALID
    assert (
        validation.target_hash
        == CapsuleManifest.model_validate_json(
            (root / "capsule-manifest.json").read_bytes()
        ).source_manifest_hash
    )


def test_finalize_fault_occurs_after_closes_without_a_false_manifest(tmp_path: Path) -> None:
    events = multiprocessing.get_context("fork").Queue()

    class FailingFinalizeStore(ArtifactStore):
        def finalize_capsule(self, *, run_id: str, run_spec_hash: str) -> CapsuleManifest:
            events.put("finalize")
            raise RuntimeError("planted finalize failure")

    class TrackingModel(_RecordedNativeModel):
        async def close(self) -> None:
            events.put("model.close")
            await super().close()

    class TrackingMemory(_RecordedNativeMemory):
        async def close(self) -> None:
            events.put("memory.close")
            await super().close()

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> TrackingMemory:
        return TrackingMemory(store)

    root = tmp_path / "capsules" / "native-fixture-finalize-failure"
    with pytest.raises(RuntimeError, match="planted finalize failure"):
        run_native_vertical_slice(
            output_root=tmp_path / "capsules",
            run_id="native-fixture-finalize-failure",
            adapter_profile_id="recorded-native-fixture-v1",
            workload=_NativeFixtureWorkload(),
            visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
            artifact_store_factory=FailingFinalizeStore,
            memory_factory=memory_factory,
            model_factory=TrackingModel,
            answer_role_binding_id="recorded-answer-v1",
        )

    assert [events.get(timeout=1) for _ in range(3)] == [
        "model.close",
        "memory.close",
        "finalize",
    ]
    assert not (root / "capsule-manifest.json").exists()


def test_validated_native_capsule_reduces_and_publishes_without_transient_receipts(
    tmp_path: Path,
) -> None:
    try:
        native_reduce = importlib.import_module("oamb.reporting.native_reduce")
    except ModuleNotFoundError:
        pytest.fail("native report reduction is not implemented", pytrace=False)
    from oamb.contracts.reporting import RunReportModelV3, run_report_model_v3_id
    from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash
    from oamb.reporting.publication import ReportExportError, build_report_derivation
    from oamb.reporting.roots import build_report_spec

    completed = _run_fixture_capsule(tmp_path, "native-fixture-report")
    validation = validate_native_capsule(completed.capsule_root)
    css_hash, script_hash = offline_asset_hashes()
    report_spec = build_report_spec(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
        renderer_hash=offline_renderer_hash(),
        asset_hashes=(css_hash, script_hash),
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )

    model = native_reduce.reduce_native_run_report(
        completed.capsule_root,
        validation,
        report_spec=report_spec,
    )
    built = build_report_derivation(
        model=model,
        report_spec=report_spec,
        ordered_source_bindings=model.ordered_source_bindings,
        evidence_validations=(validation,),
        evidence_validation_targets=(completed.capsule_root,),
        transform_spec_hash="c" * 64,
        schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
        output_root=tmp_path / "reports",
        committed_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    rebuilt = build_report_derivation(
        model=model,
        report_spec=report_spec,
        ordered_source_bindings=model.ordered_source_bindings,
        evidence_validations=(validation,),
        evidence_validation_targets=(completed.capsule_root,),
        transform_spec_hash="c" * 64,
        schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
        output_root=tmp_path / "reports",
        committed_at=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert model.summary.intended_logical_contexts == 2
    assert model.summary.intended_ingestion_plans == 2
    assert model.summary.intended_cases == 4
    assert sum(item.input_count for item in model.metric_summaries) == 4
    assert model.origin_kind == "native"
    assert built.report_path.is_file()
    assert rebuilt.derivation_id == built.derivation_id
    assert rebuilt.report_path.read_bytes() == built.report_path.read_bytes()

    local_report_spec = build_report_spec(
        report_kind="run",
        audience="local",
        preview_max_field_bytes=32,
        preview_total_bytes=4096,
        display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
        renderer_hash=offline_renderer_hash(),
        asset_hashes=(css_hash, script_hash),
        export_profile_selector_id="local-run-v1",
        export_profile_selector_version=1,
    )
    local_model = native_reduce.reduce_native_run_report(
        completed.capsule_root,
        validation,
        report_spec=local_report_spec,
    )
    previews = tuple(
        preview
        for projection in local_model.record_projections
        for preview in projection.display_previews
    )
    assert previews
    assert all(preview.shown_bytes <= 32 for preview in previews)
    assert sum(preview.shown_bytes for preview in previews) <= 4096
    assert all(preview.sha256 in preview.source_reference for preview in previews)
    assert "text/plain; charset=utf-8" in {preview.media_type for preview in previews}
    assert "application/vnd.oamb.visible-evidence" in {preview.media_type for preview in previews}

    tampered_fields = model.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "report_id"},
    )
    tampered_fields["measurement_lines"] = ()
    tampered = RunReportModelV3.model_validate(
        {"report_id": run_report_model_v3_id(**tampered_fields), **tampered_fields}
    )
    with pytest.raises(ReportExportError) as captured:
        build_report_derivation(
            model=tampered,
            report_spec=report_spec,
            ordered_source_bindings=tampered.ordered_source_bindings,
            evidence_validations=(validation,),
            evidence_validation_targets=(completed.capsule_root,),
            transform_spec_hash="c" * 64,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path / "tampered-reports",
            committed_at=datetime(2026, 8, 28, tzinfo=UTC),
        )
    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}


def test_invalid_native_capsule_requires_explicit_restricted_diagnostic_publication(
    tmp_path: Path,
) -> None:
    from oamb.contracts.reporting import DiagnosticRunReportModel
    from oamb.contracts.schema import parse_contract
    from oamb.contracts.specifications import SourceEvidenceBinding, SourceEvidenceKind
    from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash
    from oamb.reporting.public import build_diagnostic_run_report_model
    from oamb.reporting.publication import build_report_derivation
    from oamb.reporting.roots import build_report_spec

    run_id = "native-invalid-diagnostic"
    completed = _run_fixture_capsule(tmp_path, run_id)
    run_path = completed.capsule_root / "source" / "run" / f"{run_id}.json"
    run_path.write_bytes(run_path.read_bytes() + b"\n")
    validation = validate_native_capsule(completed.capsule_root)
    assert validation.disposition == ValidationDisposition.INVALID
    manifest = CapsuleManifest.model_validate_json(
        (completed.capsule_root / "capsule-manifest.json").read_bytes()
    )
    validation_hash = canonical_sha256(validation)
    source = SourceEvidenceBinding(
        binding_id=canonical_sha256(["diagnostic-source", manifest.source_manifest_hash]),
        source_kind=SourceEvidenceKind.RUN,
        source_identity=run_id,
        source_root_hash=manifest.source_manifest_hash,
        validation_result_hash=validation_hash,
        source_schema_versions=("capsule_manifest@1",),
    )
    css_hash, script_hash = offline_asset_hashes()
    spec = build_report_spec(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "validation", "limitations"),
        renderer_hash=offline_renderer_hash(),
        asset_hashes=(css_hash, script_hash),
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )
    profile_hash = canonical_sha256(
        [
            "oamb-validation-profile-binding-v1",
            validation.validation_profile_id,
            validation.required_rule_ids,
            validation.implementation_versions,
        ]
    )
    model = build_diagnostic_run_report_model(
        report_spec_hash=canonical_sha256(spec),
        source_binding=source,
        evidence_validation_profile_hash=profile_hash,
        evidence_validation_result_hash=validation_hash,
        origin_kind="native",
        run_id=run_id,
        validation_issue_codes=tuple(dict.fromkeys(issue.code for issue in validation.issues)),
        limitations=("diagnostic-only; no benchmark quality or cost claims",),
    )
    assert isinstance(model, DiagnosticRunReportModel)

    def publish_diagnostic(*, diagnostic: bool = False) -> Any:
        return build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(validation,),
            evidence_validation_targets=(completed.capsule_root,),
            transform_spec_hash="c" * 64,
            schema_versions=(
                "diagnostic_run_report_model@1",
                "report_artifact_manifest@2",
            ),
            output_root=tmp_path / "diagnostic-reports",
            committed_at=datetime(2026, 8, 28, tzinfo=UTC),
            diagnostic=diagnostic,
        )

    with pytest.raises(ValueError, match="explicit opt-in"):
        publish_diagnostic()
    built = publish_diagnostic(diagnostic=True)

    html = built.report_path.read_text(encoding="utf-8")
    assert "DIAGNOSTIC — INVALID EVIDENCE" in html
    assert "No final quality or cost claims are present" in html
    assert '"metric_summaries"' not in html
    assert '"measurement_lines"' not in html
    parsed_model = parse_contract(
        json.loads((built.final_directory / "outputs" / "report-model.json").read_bytes())
    )
    assert parsed_model == model
    assert built.export_validation.disposition == ValidationDisposition.VALIDATED
