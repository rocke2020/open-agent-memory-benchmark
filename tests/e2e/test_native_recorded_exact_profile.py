from __future__ import annotations

import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.artifacts.validation.run_evidence import (
    native_run_evidence_validation_input,
    validate_native_run_evidence,
)
from oamb.contracts.evidence import CapsuleManifest
from oamb.contracts.ids import canonical_sha256, ingestion_occurrence_id
from oamb.contracts.ports import ArtifactStorePort, IngestionPlan, SourceUnit, ThinkingEffort
from oamb.contracts.specifications import (
    BindingKind,
    DatasetFile,
    DatasetManifest,
    ExecutionOwner,
    ModelRole,
    ModelRoleBindingV2,
    RoleBindingStatus,
)
from oamb.contracts.states import ValidationDisposition
from oamb.memory_systems.hindsight import HindsightAdapter
from oamb.memory_systems.openviking import OpenVikingRestAdapter, OpenVikingSessionAdapter
from oamb.model_clients.openai_compatible import OpenAICompatibleModelClient
from oamb.reporting.native_reduce import reduce_native_run_report
from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash
from oamb.reporting.publication import build_report_derivation
from oamb.reporting.roots import build_report_spec
from oamb.runtime.native_run import NativeRunArtifacts, run_native_vertical_slice
from oamb.workloads.longmemeval import (
    LME30_WORKLOAD_ID,
    LME_JUDGE_PROMPT_PACK_ID,
    LongMemEvalMessage,
    LongMemEvalRow,
    LongMemEvalSession,
    LongMemEvalWorkload,
    _build_bundle,
    build_lme6_bundle,
    build_lme30_bundle,
)
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY

RUN_ID = "recorded-hindsight-lme-fixture"
REAL_LME6_RUN_ID = "recorded-hindsight-lme6"
ANSWER_ROLE_ID = "recorded-answer-v1"
RUNTIME_BINDING_HASH = "f" * 64
OPENVIKING_RUN_ID = "recorded-openviking-lme-fixture"
OPENVIKING_SESSION_RUN_ID = "recorded-openviking-session-lme-fixture"
OPENVIKING_USER = "oamb-admin"
REAL_LME_SOURCE = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "longmemeval-cleaned"
    / "longmemeval_s_cleaned.json"
)
HINDSIGHT_FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "adapters" / "hindsight"


def _hindsight_bank_config_bytes(bank_id: str) -> bytes:
    return (
        (HINDSIGHT_FIXTURE_ROOT / "bank-config-v0.9.2.json")
        .read_bytes()
        .replace(b"BANK_ID", bank_id.encode())
    )


def _workload() -> LongMemEvalWorkload:
    source_file = DatasetFile(
        relative_path="fixtures/recorded-lme.json",
        sha256="a" * 64,
        byte_count=1,
        license_id="NOASSERTION",
    )
    dataset = DatasetManifest(
        dataset_id="recorded-lme-fixture",
        revision="recorded-v1",
        split="fixture",
        manifest_hash=canonical_sha256(["recorded-lme-fixture", source_file]),
        source_files=(source_file,),
        payload_policy="generated-fixture",
    )
    session = LongMemEvalSession(
        session_id="session-1",
        raw_timestamp="2026/01/01 (Thu) 00:00",
        canonical_timestamp="2026-01-01T00:00:00+00:00",
        messages=(LongMemEvalMessage(role="user", content="source 1", has_answer=True),),
    )
    row = LongMemEvalRow(
        source_row_number_1_indexed=1,
        question_id="question-1",
        question_type="multi-session",
        question="What does Alice care about?",
        answer="Alice prefers exact evidence.",
        raw_question_timestamp="2026/01/02 (Fri) 00:00",
        canonical_question_timestamp="2026-01-02T00:00:00+00:00",
        answer_session_ids=(session.session_id,),
        message_has_answer_session_ids=(session.session_id,),
        sessions=(session,),
    )
    return LongMemEvalWorkload(
        _build_bundle(dataset, (row,), workload_id="recorded-lme-fixture-v1")
    )


def _bank_id(occurrence_id: str) -> str:
    return hashlib.sha256(f"oamb-hindsight-bank-v1\0{occurrence_id}".encode()).hexdigest()


def _bank_response(bank_id: str) -> dict[str, object]:
    return {
        "bank_id": bank_id,
        "name": bank_id,
        "disposition": {"skepticism": 3, "literalism": 3, "empathy": 3},
        "mission": "",
        "background": "",
    }


def _bank_list_item(bank_id: str) -> dict[str, object]:
    return {
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


class _RecordedHindsightService:
    def __init__(self, sources_by_bank: dict[str, tuple[SourceUnit, ...]]) -> None:
        self.sources_by_bank = sources_by_bank
        self.created_bank_ids: set[str] = set()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/version":
            return httpx.Response(
                200,
                json={
                    "api_version": "0.9.2",
                    "features": {
                        "observations": True,
                        "mcp": True,
                        "worker": True,
                        "bank_config_api": True,
                        "bank_llm_health": False,
                        "file_upload_api": True,
                        "document_export_api": True,
                        "document_import_api": True,
                        "audit_log": False,
                        "llm_trace": True,
                        "store_document_text": True,
                    },
                },
            )
        if path == "/v1/default/banks":
            ordered_bank_ids = tuple(sorted(self.created_bank_ids))
            offset = int(request.url.params["offset"])
            items = ordered_bank_ids if offset == 0 else ()
            return httpx.Response(
                200,
                json={
                    "banks": [_bank_list_item(bank_id) for bank_id in items],
                    "total": len(ordered_bank_ids),
                    "limit": 1000,
                    "offset": offset,
                },
            )
        bank_id = self._bank_id_from_path(path)
        if request.method == "PUT" and bank_id is not None and path.endswith(bank_id):
            self.created_bank_ids.add(bank_id)
            return httpx.Response(200, json=_bank_response(bank_id))
        if path.endswith("/config"):
            assert bank_id is not None
            return httpx.Response(200, content=_hindsight_bank_config_bytes(bank_id))
        if path.endswith("/profile"):
            assert bank_id is not None
            return httpx.Response(200, json=_bank_response(bank_id))
        if request.method == "POST" and path.endswith("/memories"):
            assert bank_id is not None
            body = json.loads(request.content)
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "bank_id": bank_id,
                    "items_count": len(body["items"]),
                    "async": False,
                    "operation_id": None,
                    "operation_ids": None,
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 2,
                        "total_tokens": 13,
                        "cached_tokens": 0,
                        "thoughts_tokens": 0,
                    },
                },
            )
        if request.method == "GET" and path.endswith("/documents"):
            assert bank_id is not None
            return self._documents_page(request, bank_id)
        if request.method == "GET" and "/documents/" in path:
            assert bank_id is not None
            source_id = path.rsplit("/", 1)[-1]
            source = next(
                source
                for source in self.sources_by_bank[bank_id]
                if source.source_unit_id == source_id
            )
            return httpx.Response(200, json=self._document_detail(bank_id, source))
        if request.method == "GET" and path.endswith("/memories/list"):
            return httpx.Response(200, json={"items": [], "total": 0, "limit": 1000, "offset": 0})
        if request.method == "GET" and path.endswith("/mental-models"):
            return httpx.Response(200, json={"items": [], "total": 0, "limit": 1000, "offset": 0})
        if request.method == "POST" and path.endswith("/memories/recall"):
            return httpx.Response(
                200,
                json={
                    "results": [],
                    "trace": {
                        "query": "What does Alice care about?",
                        "num_results": 0,
                        "time_seconds": 0.01,
                    },
                    "entities": None,
                    "chunks": {},
                    "source_facts": None,
                    "source_facts_truncated": None,
                },
            )
        raise AssertionError(f"unexpected Hindsight request: {request.method} {request.url}")

    def _documents_page(self, request: httpx.Request, bank_id: str) -> httpx.Response:
        sources = self.sources_by_bank[bank_id]
        offset = int(request.url.params["offset"])
        if offset == len(sources):
            return httpx.Response(
                200,
                json={"items": [], "total": len(sources), "limit": 1000, "offset": offset},
            )
        if offset != 0:
            raise AssertionError(f"unexpected document offset: {offset}")
        return httpx.Response(
            200,
            json={
                "items": [self._document_summary(bank_id, source) for source in sources],
                "total": len(sources),
                "limit": 1000,
                "offset": 0,
            },
        )

    @staticmethod
    def _retain_metadata(source: SourceUnit) -> dict[str, object]:
        return {
            "context": source.context_text,
            "event_date": source.occurred_at,
            "metadata": dict(source.source_metadata),
        }

    def _document_summary(self, bank_id: str, source: SourceUnit) -> dict[str, object]:
        return {
            "id": source.source_unit_id,
            "bank_id": bank_id,
            "content_hash": source.payload_sha256,
            "created_at": "2026-01-01T00:00:01+00:00",
            "updated_at": "2026-01-01T00:00:02+00:00",
            "text_length": len(source.payload_bytes.decode("utf-8")),
            "memory_unit_count": 0,
            "retain_params": self._retain_metadata(source),
            "document_metadata": dict(source.source_metadata),
            "tags": [],
        }

    def _document_detail(self, bank_id: str, source: SourceUnit) -> dict[str, object]:
        return {
            "id": source.source_unit_id,
            "bank_id": bank_id,
            "original_text": source.payload_bytes.decode("utf-8"),
            "content_hash": source.payload_sha256,
            "memory_unit_count": 0,
            "nodes_by_fact_type": {},
            "created_at": "2026-01-01T00:00:01+00:00",
            "updated_at": "2026-01-01T00:00:02+00:00",
            "tags": [],
            "document_metadata": dict(source.source_metadata),
            "retain_params": self._retain_metadata(source),
            "observation_scopes": None,
        }

    def _bank_id_from_path(self, path: str) -> str | None:
        for bank_id in self.sources_by_bank:
            if f"/banks/{bank_id}" in path:
                return bank_id
        return None


def _memory_factory(
    store: ArtifactStorePort,
    plans: tuple[IngestionPlan, ...],
) -> HindsightAdapter:
    service = _RecordedHindsightService(
        {
            _bank_id(ingestion_occurrence_id(RUN_ID, "hindsight", plan.ingestion_plan_id)): (
                plan.ordered_source_units
            )
            for plan in plans
        }
    )
    return HindsightAdapter(
        store=store,
        base_url="https://hindsight.example",
        configured_extraction_model="fixture-extractor",
        runtime_extraction_model="fixture-extractor@runtime",
        runtime_binding_hash=RUNTIME_BINDING_HASH,
        transport=httpx.MockTransport(service),
    )


def _real_lme6_memory_factory(
    store: ArtifactStorePort,
    plans: tuple[IngestionPlan, ...],
) -> HindsightAdapter:
    service = _RecordedHindsightService(
        {
            _bank_id(
                ingestion_occurrence_id(
                    REAL_LME6_RUN_ID,
                    "hindsight",
                    plan.ingestion_plan_id,
                )
            ): plan.ordered_source_units
            for plan in plans
        }
    )
    return HindsightAdapter(
        store=store,
        base_url="https://hindsight.example",
        configured_extraction_model="fixture-extractor",
        runtime_extraction_model="fixture-extractor@runtime",
        runtime_binding_hash=RUNTIME_BINDING_HASH,
        transport=httpx.MockTransport(service),
    )


def _model_binding(
    role: ModelRole,
    binding_id: str,
    model: str,
    thinking_effort: ThinkingEffort,
) -> ModelRoleBindingV2:
    return ModelRoleBindingV2(
        binding_id=binding_id,
        role=role,
        role_status=RoleBindingStatus.SELECTED,
        execution_owner=ExecutionOwner.HARNESS,
        binding_kind=BindingKind.MODEL_CLIENT,
        provider="fixture-provider",
        endpoint_reference="fixture-model-endpoint",
        credential_variable_name="OAMB_FIXTURE_MODEL_API_KEY",
        configured_model=model,
        resolved_model=model,
        thinking_effort=thinking_effort,
        parameters_fingerprint="1" * 64,
        retry_policy_id="no-retry-v1",
        configuration_fingerprint="2" * 64,
        redacted_endpoint_fingerprint="3" * 64,
    )


def _model_factory(
    store: ArtifactStorePort,
    *,
    role: ModelRole,
    binding_id: str,
    model: str,
    output: str,
    thinking_effort: ThinkingEffort,
) -> OpenAICompatibleModelClient:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["reasoning_effort"] == thinking_effort
        return httpx.Response(
            200,
            json={
                "id": f"{model}-response",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": output},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            },
        )

    return OpenAICompatibleModelClient(
        store=store,
        base_url="https://models.example/v1",
        api_key="fixture-secret",
        role_binding=_model_binding(role, binding_id, model, thinking_effort),
        runtime_model_policy="require_match",
        transport=httpx.MockTransport(handler),
    )


def _run_recorded_hindsight(tmp_path: Path) -> NativeRunArtifacts:
    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=RUN_ID,
        adapter_profile_id="hindsight-rest-v1",
        workload=_workload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_memory_factory,
        model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.ANSWER,
            binding_id=ANSWER_ROLE_ID,
            model="fixture-answer-model",
            output="Alice prefers exact evidence.",
            thinking_effort="low",
        ),
        answer_role_binding_id=ANSWER_ROLE_ID,
        judge_model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.JUDGE,
            binding_id=LME_JUDGE_PROMPT_PACK_ID,
            model="fixture-judge-model",
            output="yes",
            thinking_effort="high",
        ),
        judge_role_binding_id=LME_JUDGE_PROMPT_PACK_ID,
    )


def _run_recorded_hindsight_lme6(
    tmp_path: Path,
    workload: LongMemEvalWorkload,
) -> NativeRunArtifacts:
    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=REAL_LME6_RUN_ID,
        adapter_profile_id="hindsight-rest-v1",
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_real_lme6_memory_factory,
        model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.ANSWER,
            binding_id=ANSWER_ROLE_ID,
            model="fixture-answer-model",
            output="Recorded exact LME-6 answer.",
            thinking_effort="low",
        ),
        answer_role_binding_id=ANSWER_ROLE_ID,
        judge_model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.JUDGE,
            binding_id=LME_JUDGE_PROMPT_PACK_ID,
            model="fixture-judge-model",
            output="yes",
            thinking_effort="high",
        ),
        judge_role_binding_id=LME_JUDGE_PROMPT_PACK_ID,
    )


class _RecordedOpenVikingSessionService:
    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "healthy": True,
                    "version": "v0.4.16",
                    "auth_mode": "api_key",
                    "account_id": "oamb-benchmark",
                    "user_id": OPENVIKING_USER,
                    "role": "admin",
                },
            )
        if path == "/api/v1/fs/stat":
            resource = request.url.params["uri"]
            return httpx.Response(
                404,
                json={
                    "status": "error",
                    "error": {
                        "code": "NOT_FOUND",
                        "message": f"File not found: {resource}",
                        "details": {"resource": resource, "type": "file"},
                    },
                },
            )
        if path.startswith("/api/v1/sessions/") and path.count("/") == 4:
            session_id = path.rsplit("/", 1)[-1]
            return httpx.Response(
                404,
                json={
                    "status": "error",
                    "error": {
                        "code": "NOT_FOUND",
                        "message": f"Session {session_id} not found",
                    },
                },
            )
        if path == "/api/v1/sessions" and request.method == "POST":
            session_id = json.loads(request.content)["session_id"]
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "session_id": session_id,
                        "auto_commit_policy": None,
                    },
                    "error": None,
                    "telemetry": None,
                    "profile": None,
                },
            )
        if path.endswith("/messages/batch"):
            session_id = path.split("/")[4]
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {"session_id": session_id},
                    "error": None,
                    "telemetry": None,
                    "profile": None,
                },
            )
        if path.endswith("/commit"):
            session_id = path.split("/")[4]
            archive_uri = f"viking://user/{OPENVIKING_USER}/sessions/{session_id}/archive/a1"
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "session_id": session_id,
                        "status": "accepted",
                        "task_id": f"task-{session_id}",
                        "archive_uri": archive_uri,
                        "archived": True,
                    },
                },
            )
        if path.startswith("/api/v1/tasks/"):
            task_id = path.rsplit("/", 1)[-1]
            session_id = task_id.removeprefix("task-")
            archive_uri = f"viking://user/{OPENVIKING_USER}/sessions/{session_id}/archive/a1"
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "task_id": task_id,
                        "task_type": "session_commit",
                        "status": "completed",
                        "resource_id": session_id,
                        "result": {
                            "session_id": session_id,
                            "archive_uri": archive_uri,
                            "token_usage": {"llm": {}, "embedding": {}, "total": {}},
                        },
                    },
                    "error": None,
                    "telemetry": None,
                    "profile": None,
                },
            )
        if "/archives/" in path:
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "archive_id": "a1",
                        "abstract": "# Working Memory",
                        "messages": [],
                        "overview": "fixture archive",
                    },
                    "error": None,
                    "telemetry": None,
                    "profile": None,
                },
            )
        if path == "/api/v1/fs/ls":
            memory_root = request.url.params["uri"]
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": [f"{memory_root}/events/event-1.md"],
                    "error": None,
                    "telemetry": None,
                    "profile": None,
                },
            )
        if path == "/api/v1/search/find":
            memory_root = json.loads(request.content)["target_uri"]
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "memories": [
                            {
                                "context_type": "memory",
                                "uri": f"{memory_root}/events/event-1.md",
                                "level": 2,
                                "score": 0.9,
                                "abstract": "Alice prefers exact evidence.",
                                "tags": [],
                            }
                        ],
                        "resources": [],
                        "skills": [],
                        "total": 1,
                    },
                },
            )
        raise AssertionError(f"unexpected OpenViking session request: {request.method} {path}")


def _run_recorded_openviking_session(tmp_path: Path) -> NativeRunArtifacts:
    service = _RecordedOpenVikingSessionService()

    def memory_factory(
        store: ArtifactStorePort,
        _plans: tuple[IngestionPlan, ...],
    ) -> OpenVikingSessionAdapter:
        return OpenVikingSessionAdapter(
            store=store,
            base_url="https://openviking.example",
            api_key="fixture-secret",
            benchmark_account="oamb-benchmark",
            benchmark_user=OPENVIKING_USER,
            runtime_binding_hash=RUNTIME_BINDING_HASH,
            transport=httpx.MockTransport(service),
            task_poll_interval_seconds=0,
            maximum_task_polls=1,
        )

    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=OPENVIKING_SESSION_RUN_ID,
        adapter_profile_id="openviking-session-rest-v1",
        workload=_workload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.ANSWER,
            binding_id=ANSWER_ROLE_ID,
            model="fixture-answer-model",
            output="Alice prefers exact evidence.",
            thinking_effort="low",
        ),
        answer_role_binding_id=ANSWER_ROLE_ID,
        judge_model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.JUDGE,
            binding_id=LME_JUDGE_PROMPT_PACK_ID,
            model="fixture-judge-model",
            output="yes",
            thinking_effort="high",
        ),
        judge_role_binding_id=LME_JUDGE_PROMPT_PACK_ID,
    )


def test_openviking_session_profile_seals_a_fresh_validatable_native_capsule(
    tmp_path: Path,
) -> None:
    completed = _run_recorded_openviking_session(tmp_path)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    assert completed.ingestion_plan_records[0].adapter_profile_id == ("openviking-session-rest-v1")
    assert completed.case_records[0].retrieval_request_raw_ref is not None


def test_openviking_session_capsule_rejects_a_resealed_generating_retrieval_request(
    tmp_path: Path,
) -> None:
    completed = _run_recorded_openviking_session(tmp_path)
    request_reference = completed.case_records[0].retrieval_request_raw_ref
    assert request_reference is not None
    manifest = CapsuleManifest.model_validate_json(
        (completed.capsule_root / "capsule-manifest.json").read_bytes()
    )
    entry = next(
        item
        for item in manifest.source_entries
        if item.record_kind == "raw_payload" and item.record_id == request_reference
    )
    proof = json.loads(gzip.decompress((completed.capsule_root / entry.relative_path).read_bytes()))
    proof["json_payload"]["session_id"] = "forbidden-generative-session"
    _replace_raw_payload(
        completed.capsule_root,
        request_reference,
        json.dumps(proof, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "retrieval-request-proof-invalid" in {issue.code for issue in validation.issues}


def test_real_lme_hindsight_and_model_clients_seal_one_root_validatable_capsule(
    tmp_path: Path,
) -> None:
    completed = _run_recorded_hindsight(tmp_path)

    profile_id = "oamb-t8-adapter-hindsight-rest-v1"
    validation = validate_catalog_profile(profile_id, completed.capsule_root)
    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    assert validation.executed_rule_ids == validation.required_rule_ids
    native_validation = validate_native_capsule(completed.capsule_root)
    assert native_validation.disposition == ValidationDisposition.VALIDATED, (
        native_validation.issues
    )
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
    report = reduce_native_run_report(
        completed.capsule_root,
        native_validation,
        report_spec=report_spec,
    )
    assert report.summary.completed_cases == 1
    publication = build_report_derivation(
        model=report,
        report_spec=report_spec,
        ordered_source_bindings=report.ordered_source_bindings,
        evidence_validations=(native_validation,),
        evidence_validation_targets=(completed.capsule_root,),
        transform_spec_hash="c" * 64,
        schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
        output_root=tmp_path / "reports",
        committed_at=datetime(2026, 8, 28, tzinfo=UTC),
    )
    assert publication.report_path.is_file()
    assert len(completed.ingestion_plan_records) == 1
    assert len(completed.case_records) == 1
    assert completed.ingestion_plan_records[0].adapter_profile_id == "hindsight-rest-v1"
    assert completed.ingestion_plan_records[0].ordered_dispatch_source_unit_ids == (
        completed.ingestion_plan_records[0].ordered_source_unit_ids,
    )
    assert completed.case_records[0].prompt_raw_ref == completed.case_records[0].prompt_sha256
    assert completed.case_records[0].judge_prompt_raw_ref is not None
    assert completed.case_records[0].evaluation_disposition == "judged"


@pytest.mark.skipif(not REAL_LME_SOURCE.is_file(), reason="pinned LongMemEval data is absent")
def test_exact_lme6_hindsight_capsule_requires_composite_validation_before_publication(
    tmp_path: Path,
) -> None:
    bundle = build_lme6_bundle(build_lme30_bundle(REAL_LME_SOURCE))
    completed = _run_recorded_hindsight_lme6(tmp_path, LongMemEvalWorkload(bundle))
    validation_target = native_run_evidence_validation_input(
        capsule_root=completed.capsule_root,
        workload_profile_id="oamb-t8-workload-lme6-v1",
        workload_target=bundle,
        adapter_profile_id="oamb-t8-adapter-hindsight-rest-v1",
    )
    validation = validate_native_run_evidence(validation_target)

    assert validation.disposition == ValidationDisposition.VALIDATED, validation.issues
    assert validation.required_rule_ids == validation.executed_rule_ids
    assert validation.required_rule_ids == validation.passed_rule_ids

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
    report = reduce_native_run_report(
        completed.capsule_root,
        validation,
        report_spec=report_spec,
        validation_target=validation_target,
    )
    publication = build_report_derivation(
        model=report,
        report_spec=report_spec,
        ordered_source_bindings=report.ordered_source_bindings,
        evidence_validations=(validation,),
        evidence_validation_targets=(validation_target,),
        transform_spec_hash="c" * 64,
        schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
        output_root=tmp_path / "reports",
        committed_at=datetime(2026, 8, 28, tzinfo=UTC),
    )

    assert report.summary.intended_cases == 6
    assert report.workload_id == LME30_WORKLOAD_ID
    assert publication.export_validation.disposition == ValidationDisposition.VALIDATED


def test_hindsight_profile_rejects_resealed_synthetic_projection_self_attestation(
    tmp_path: Path,
) -> None:
    completed = _run_recorded_hindsight(tmp_path)
    plan = completed.ingestion_plan_records[0]
    projection_reference = plan.projection_raw_refs[0]
    manifest = CapsuleManifest.model_validate_json(
        (completed.capsule_root / "capsule-manifest.json").read_bytes()
    )
    projection_entry = next(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "raw_payload" and entry.record_id == projection_reference
    )
    projection = json.loads(
        gzip.decompress((completed.capsule_root / projection_entry.relative_path).read_bytes())
    )
    projection["documents"][0]["detail"]["original_text"] += " tampered"
    projection["documents"][0]["detail"]["content_hash"] = "0" * 64
    _replace_raw_payload(
        completed.capsule_root,
        projection_reference,
        json.dumps(projection, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    profile_validation = validate_catalog_profile(
        "oamb-t8-adapter-hindsight-rest-v1",
        completed.capsule_root,
    )
    native_validation = validate_native_capsule(completed.capsule_root)

    assert profile_validation.disposition == ValidationDisposition.INVALID
    assert "adapter.hindsight.projection-readiness.v1" in profile_validation.failed_rule_ids
    assert native_validation.disposition == ValidationDisposition.INVALID
    assert "native.plan-closure.v1" in native_validation.failed_rule_ids


def test_native_profile_rejects_resealed_missing_attempt_accounting_references(
    tmp_path: Path,
) -> None:
    completed = _run_recorded_hindsight(tmp_path)
    plan = completed.ingestion_plan_records[0]
    case = completed.case_records[0]
    plan_path = (
        completed.capsule_root
        / "source"
        / "ingestion-plans"
        / f"{plan.ingestion_occurrence_id}.json"
    )
    case_path = completed.capsule_root / "source" / "cases" / f"{case.case_occurrence_id}.json"
    plan_document = json.loads(plan_path.read_bytes())
    case_document = json.loads(case_path.read_bytes())
    plan_document["resource_record_ids"] = []
    plan_document["cost_record_ids"] = []
    case_document["usage_record_ids"] = case_document["usage_record_ids"][1:]
    plan_path.write_bytes(
        json.dumps(plan_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    case_path.write_bytes(
        json.dumps(case_document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    _reseal_capsule(completed.capsule_root)

    validation = validate_native_capsule(completed.capsule_root)

    assert validation.disposition == ValidationDisposition.INVALID
    assert "native.plan-closure.v1" in validation.failed_rule_ids
    assert "native.retrieval-closure.v1" in validation.failed_rule_ids
    assert {issue.code for issue in validation.issues} >= {
        "ingestion-accounting-ledger-mismatch",
        "case-accounting-ledger-mismatch",
    }


class _RecordedOpenVikingService:
    def __init__(self, occurrence_id: str, plan: IngestionPlan) -> None:
        self.source = plan.ordered_source_units[0]
        actor_peer_id = (
            "oamb-" + hashlib.sha256(b"peer\0" + occurrence_id.encode("utf-8")).hexdigest()
        )
        plan_key = hashlib.sha256(b"plan\0" + plan.ingestion_plan_id.encode("utf-8")).hexdigest()
        self.peer_root = f"viking://user/{OPENVIKING_USER}/peers/{actor_peer_id}"
        self.root_uri = f"{self.peer_root}/resources/oamb/mab65-v1/{plan_key}"
        self.chunk_uri = f"{self.root_uri}/chunk-0001.txt"
        self.abstract_uri = f"{self.root_uri}/.abstract.md"
        self.overview_uri = f"{self.root_uri}/.overview.md"
        self.contents = {
            self.abstract_uri: "recorded abstract",
            self.overview_uri: "recorded overview",
            self.chunk_uri: self.source.payload_bytes.decode("utf-8"),
        }

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "healthy": True,
                    "version": "v0.4.16",
                    "auth_mode": "api_key",
                    "account_id": "oamb-benchmark",
                    "user_id": OPENVIKING_USER,
                    "role": "admin",
                },
            )
        if path == "/api/v1/fs/stat":
            return httpx.Response(
                404,
                json={
                    "status": "error",
                    "result": None,
                    "error": {
                        "code": "NOT_FOUND",
                        "message": "not found",
                        "details": {
                            "type": "file",
                            "resource": request.url.params["uri"],
                        },
                    },
                    "telemetry": None,
                    "profile": None,
                },
            )
        if path == "/api/v1/fs/mkdir":
            return httpx.Response(200, json=_openviking_success({"uri": self.root_uri}))
        if path == "/api/v1/content/batch-write":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "root_uri": self.root_uri,
                        "created": [self.chunk_uri],
                        "updated": [],
                        "unchanged": [],
                        "queue_status": {
                            "Semantic": {
                                "processed": 1,
                                "requeue_count": 0,
                                "error_count": 0,
                                "errors": [],
                            },
                            "Embedding": {
                                "processed": 1,
                                "requeue_count": 0,
                                "error_count": 0,
                                "errors": [],
                            },
                        },
                    },
                },
            )
        if path == "/api/v1/fs/ls":
            return httpx.Response(
                200,
                json=_openviking_success([self.overview_uri, self.chunk_uri, self.abstract_uri]),
            )
        if path == "/api/v1/fs/attrs":
            uri = request.url.params["uri"]
            return httpx.Response(
                200,
                json=_openviking_success(
                    {"uri": uri, "context_type": "resource", "attrs": {"tags": []}}
                ),
            )
        if path == "/api/v1/content/read":
            return httpx.Response(
                200,
                json=_openviking_success(self.contents[request.url.params["uri"]]),
            )
        if path == "/api/v1/search/find":
            return httpx.Response(
                200,
                json={
                    "status": "ok",
                    "result": {
                        "memories": [],
                        "resources": [
                            {
                                "context_type": "resource",
                                "uri": self.abstract_uri,
                                "level": 0,
                                "score": 0.9,
                                "abstract": self.contents[self.abstract_uri],
                                "tags": [],
                            },
                            {
                                "context_type": "resource",
                                "uri": self.chunk_uri,
                                "level": 2,
                                "score": 0.8,
                                "abstract": "chunk preview",
                                "tags": [],
                            },
                        ],
                        "skills": [],
                        "total": 2,
                    },
                },
            )
        raise AssertionError(f"unexpected OpenViking request: {request.method} {request.url}")


def _openviking_success(result: object) -> dict[str, object]:
    return {
        "status": "ok",
        "result": result,
        "error": None,
        "telemetry": None,
        "profile": None,
    }


def _openviking_memory_factory(
    store: ArtifactStorePort,
    plans: tuple[IngestionPlan, ...],
) -> OpenVikingRestAdapter:
    occurrence_id = ingestion_occurrence_id(
        OPENVIKING_RUN_ID,
        "openviking",
        plans[0].ingestion_plan_id,
    )
    service = _RecordedOpenVikingService(occurrence_id, plans[0])
    return OpenVikingRestAdapter(
        store=store,
        base_url="https://openviking.example",
        api_key="fixture-secret",
        benchmark_account="oamb-benchmark",
        benchmark_user=OPENVIKING_USER,
        runtime_binding_hash=RUNTIME_BINDING_HASH,
        transport=httpx.MockTransport(service),
    )


def _run_recorded_openviking(tmp_path: Path) -> NativeRunArtifacts:
    return run_native_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=OPENVIKING_RUN_ID,
        adapter_profile_id="openviking-rest-v1",
        workload=_workload(),
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=_openviking_memory_factory,
        model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.ANSWER,
            binding_id=ANSWER_ROLE_ID,
            model="fixture-answer-model",
            output="Alice prefers exact evidence.",
            thinking_effort="low",
        ),
        answer_role_binding_id=ANSWER_ROLE_ID,
        judge_model_factory=lambda store: _model_factory(
            store,
            role=ModelRole.JUDGE,
            binding_id=LME_JUDGE_PROMPT_PACK_ID,
            model="fixture-judge-model",
            output="yes",
            thinking_effort="high",
        ),
        judge_role_binding_id=LME_JUDGE_PROMPT_PACK_ID,
    )


def _replace_raw_payload(root: Path, old_reference: str, payload: bytes) -> str:
    new_reference = hashlib.sha256(payload).hexdigest()
    manifest_path = root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(manifest_path.read_bytes())
    raw_entry = next(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "raw_payload" and entry.record_id == old_reference
    )
    raw_path = root / raw_entry.relative_path
    raw_path.write_bytes(gzip.compress(payload, mtime=0))

    def replace_reference(value: object) -> object:
        if value == old_reference:
            return new_reference
        if isinstance(value, list):
            return [replace_reference(item) for item in value]
        if isinstance(value, dict):
            return {key: replace_reference(item) for key, item in value.items()}
        return value

    for path in (root / "source").rglob("*.json"):
        document = json.loads(path.read_bytes())
        path.write_bytes(
            json.dumps(
                replace_reference(document),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
    _reseal_capsule(root, record_id_updates={old_reference: new_reference})
    return new_reference


def _reseal_capsule(
    root: Path,
    *,
    record_id_updates: dict[str, str] | None = None,
) -> None:
    manifest_path = root / "capsule-manifest.json"
    manifest = CapsuleManifest.model_validate_json(manifest_path.read_bytes())
    updates = record_id_updates or {}
    source_entries = tuple(
        entry.model_copy(
            update={
                "record_id": updates.get(entry.record_id, entry.record_id),
                "sha256": hashlib.sha256((root / entry.relative_path).read_bytes()).hexdigest(),
            }
        )
        for entry in manifest.source_entries
    )
    source_manifest_hash = canonical_sha256(
        [
            "oamb-source-manifest-v1",
            tuple(entry.model_dump(mode="python") for entry in source_entries),
        ]
    )
    manifest_path.write_bytes(
        json.dumps(
            manifest.model_copy(
                update={
                    "capsule_id": canonical_sha256(
                        [
                            "oamb-capsule-v1",
                            manifest.run_id,
                            manifest.run_spec_hash,
                            source_manifest_hash,
                        ]
                    ),
                    "source_entries": source_entries,
                    "source_manifest_hash": source_manifest_hash,
                }
            ).model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def test_real_lme_openviking_capsule_closes_exact_profile_native_validation_and_reduction(
    tmp_path: Path,
) -> None:
    completed = _run_recorded_openviking(tmp_path)

    profile_validation = validate_catalog_profile(
        "oamb-t8-adapter-openviking-rest-v1",
        completed.capsule_root,
    )
    native_validation = validate_native_capsule(completed.capsule_root)

    assert profile_validation.disposition == ValidationDisposition.VALIDATED, (
        profile_validation.issues
    )
    assert native_validation.disposition == ValidationDisposition.VALIDATED, (
        native_validation.issues
    )
    css_hash, script_hash = offline_asset_hashes()
    report = reduce_native_run_report(
        completed.capsule_root,
        native_validation,
        report_spec=build_report_spec(
            report_kind="run",
            audience="public",
            preview_max_field_bytes=4096,
            preview_total_bytes=65536,
            display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
            renderer_hash=offline_renderer_hash(),
            asset_hashes=(css_hash, script_hash),
            export_profile_selector_id="public-run-v1",
            export_profile_selector_version=1,
        ),
    )
    assert report.summary.completed_cases == 1


def test_openviking_profile_rejects_resealed_raw_retrieval_order_drift(tmp_path: Path) -> None:
    completed = _run_recorded_openviking(tmp_path)
    case = completed.case_records[0]
    assert case.retrieval_raw_ref is not None
    manifest = CapsuleManifest.model_validate_json(
        (completed.capsule_root / "capsule-manifest.json").read_bytes()
    )
    retrieval_entry = next(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "raw_payload" and entry.record_id == case.retrieval_raw_ref
    )
    retrieval = json.loads(
        gzip.decompress((completed.capsule_root / retrieval_entry.relative_path).read_bytes())
    )
    retrieval["result"]["resources"].reverse()
    _replace_raw_payload(
        completed.capsule_root,
        case.retrieval_raw_ref,
        json.dumps(retrieval, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    validation = validate_catalog_profile(
        "oamb-t8-adapter-openviking-rest-v1",
        completed.capsule_root,
    )

    assert validation.disposition == ValidationDisposition.INVALID
    assert "adapter.openviking.retrieval-order.v1" in validation.failed_rule_ids


def test_openviking_profile_rejects_resealed_raw_scope_drift(tmp_path: Path) -> None:
    completed = _run_recorded_openviking(tmp_path)
    plan = completed.ingestion_plan_records[0]
    manifest = CapsuleManifest.model_validate_json(
        (completed.capsule_root / "capsule-manifest.json").read_bytes()
    )
    scope_entry = next(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "raw_payload" and entry.record_id == plan.scope_raw_refs[0]
    )
    scope_payload = json.loads(
        gzip.decompress((completed.capsule_root / scope_entry.relative_path).read_bytes())
    )
    scope_payload["result"]["uri"] += "/tampered"
    _replace_raw_payload(
        completed.capsule_root,
        plan.scope_raw_refs[0],
        json.dumps(scope_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    validation = validate_catalog_profile(
        "oamb-t8-adapter-openviking-rest-v1",
        completed.capsule_root,
    )

    assert validation.disposition == ValidationDisposition.INVALID
    assert "adapter.openviking.runtime-auth-scope.v1" in validation.failed_rule_ids


def test_openviking_profile_rejects_resealed_raw_dispatch_drift(tmp_path: Path) -> None:
    completed = _run_recorded_openviking(tmp_path)
    plan = completed.ingestion_plan_records[0]
    attempt = json.loads(
        (
            completed.capsule_root
            / "source"
            / "attempts"
            / f"{plan.ordered_dispatch_attempt_ids[0]}.json"
        ).read_bytes()
    )
    dispatch_reference = attempt["raw_response_ref"]
    manifest = CapsuleManifest.model_validate_json(
        (completed.capsule_root / "capsule-manifest.json").read_bytes()
    )
    dispatch_entry = next(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "raw_payload" and entry.record_id == dispatch_reference
    )
    dispatch = json.loads(
        gzip.decompress((completed.capsule_root / dispatch_entry.relative_path).read_bytes())
    )
    dispatch["result"]["queue_status"]["Semantic"]["error_count"] = 1
    dispatch["result"]["queue_status"]["Semantic"]["errors"] = [{"message": "planted failure"}]
    _replace_raw_payload(
        completed.capsule_root,
        dispatch_reference,
        json.dumps(dispatch, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    validation = validate_catalog_profile(
        "oamb-t8-adapter-openviking-rest-v1",
        completed.capsule_root,
    )

    assert validation.disposition == ValidationDisposition.INVALID
    assert "adapter.openviking.dispatch-projection.v1" in validation.failed_rule_ids


def test_openviking_profile_rejects_resealed_raw_projection_drift(tmp_path: Path) -> None:
    completed = _run_recorded_openviking(tmp_path)
    plan = completed.ingestion_plan_records[0]
    state_reference = plan.projection_raw_refs[1]
    manifest = CapsuleManifest.model_validate_json(
        (completed.capsule_root / "capsule-manifest.json").read_bytes()
    )
    state_entry = next(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "raw_payload" and entry.record_id == state_reference
    )
    state = json.loads(
        gzip.decompress((completed.capsule_root / state_entry.relative_path).read_bytes())
    )
    state["entries"][1]["content"] += " tampered"
    _replace_raw_payload(
        completed.capsule_root,
        state_reference,
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8"),
    )

    validation = validate_catalog_profile(
        "oamb-t8-adapter-openviking-rest-v1",
        completed.capsule_root,
    )

    assert validation.disposition == ValidationDisposition.INVALID
    assert "adapter.openviking.dispatch-projection.v1" in validation.failed_rule_ids


def test_openviking_profile_rejects_post_query_raw_projection_substitution(
    tmp_path: Path,
) -> None:
    completed = _run_recorded_openviking(tmp_path)
    case = completed.case_records[0]
    case_path = completed.capsule_root / "source" / "cases" / f"{case.case_occurrence_id}.json"
    case_document = json.loads(case_path.read_bytes())
    case_document["post_query_projection_raw_refs"] = [case.answer_raw_ref]
    case_path.write_bytes(
        json.dumps(
            case_document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    _reseal_capsule(completed.capsule_root)

    validation = validate_catalog_profile(
        "oamb-t8-adapter-openviking-rest-v1",
        completed.capsule_root,
    )

    assert validation.disposition == ValidationDisposition.INVALID
    assert "adapter.openviking.query-mutation.v1" in validation.failed_rule_ids
