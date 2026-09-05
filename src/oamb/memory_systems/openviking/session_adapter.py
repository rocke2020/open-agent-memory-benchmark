"""OpenViking v0.4.16 LongMemEval session/message/commit profile."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any

import httpx

from oamb.contracts.ids import canonical_sha256, openviking_session_id
from oamb.contracts.ports import (
    ArtifactStorePort,
    CapabilitySet,
    IngestionDispatch,
    IngestionDispatchReceipt,
    IngestionDispatchRequest,
    IngestionRequest,
    InventoryReceipt,
    MemorySystemCallFailure,
    MemorySystemCallUnknownOutcome,
    NativeEvidenceBatch,
    NativeEvidenceCandidate,
    ProjectionReceipt,
    RawReferenceHandle,
    ReadinessReceipt,
    ReadinessRequest,
    RetrievalRequest,
    RuntimeResolution,
    ScopeAllocationRequest,
    ScopeReceipt,
    SourceUnit,
    StateDigestReceipt,
)
from oamb.memory_systems.rest import (
    DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS,
    DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS,
    SealedRestClient,
    SealedRestResponse,
    bind_sealed_response_validation,
    parse_exact_json_object,
)

from .adapter import (
    ACTOR_PEER_HEADER,
    OPENVIKING_AUTH_MODE,
    OPENVIKING_MEMORY_SYSTEM_ID,
    OPENVIKING_USER_ROLE,
    OPENVIKING_VERSION,
)

OPENVIKING_SESSION_COMMIT_OPERATION = "openviking_session_commit"
OPENVIKING_SESSION_PROFILE_ID = "openviking-session-rest-v1"
MAX_MESSAGES_PER_BATCH = 100
DEFAULT_MAXIMUM_TASK_POLLS = 1_800
DEFAULT_TASK_POLL_INTERVAL_SECONDS = 0.1
DEFAULT_TASK_POLL_TIMEOUT_SECONDS = 180.0
NATIVE_RETRIEVAL_TOP_K = 100
_ACTIVE_TASK_STATUSES = frozenset({"pending", "running", "cancelling"})
_TERMINAL_FAILURE_TASK_STATUSES = frozenset({"failed", "cancelled"})
_MESSAGE_FIELDS = frozenset({"role", "content"})


def maximum_task_polls_for_timeout(
    timeout_seconds: int,
    *,
    poll_interval_seconds: float = DEFAULT_TASK_POLL_INTERVAL_SECONDS,
) -> int:
    """Cover the outer call timeout without a shorter hidden poll ceiling."""

    return math.ceil(timeout_seconds / poll_interval_seconds)


class OpenVikingSessionProfileError(ValueError):
    """The exact OpenViking LME session profile did not close."""


@dataclass(frozen=True, slots=True)
class _ScopeBinding:
    ingestion_occurrence_id: str
    ingestion_plan_id: str
    actor_peer_id: str
    peer_root: str
    memory_root: str


@dataclass(frozen=True, slots=True)
class _PlannedSession:
    dispatch: IngestionDispatch
    source: SourceUnit
    session_id: str
    messages: tuple[dict[str, str], ...]


@dataclass(frozen=True, slots=True)
class _CompletedSession:
    session_id: str
    archive_uri: str
    task_id: str
    evidence_references: tuple[RawReferenceHandle, ...]


@dataclass(frozen=True, slots=True)
class _ScopeContinuation:
    completed_session_ids: tuple[str, ...]
    failed_session_id: str
    failed_evidence_references: tuple[RawReferenceHandle, ...]


class OpenVikingSessionAdapter:
    """One LME source session per create-only OpenViking native session."""

    def __init__(
        self,
        *,
        store: ArtifactStorePort,
        base_url: str,
        api_key: str,
        benchmark_account: str,
        benchmark_user: str,
        runtime_binding_hash: str,
        transport: httpx.AsyncBaseTransport | None = None,
        task_poll_interval_seconds: float = DEFAULT_TASK_POLL_INTERVAL_SECONDS,
        maximum_task_polls: int = DEFAULT_MAXIMUM_TASK_POLLS,
        task_poll_timeout_seconds: float = DEFAULT_TASK_POLL_TIMEOUT_SECONDS,
        read_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS,
        total_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS,
    ) -> None:
        for label, value in (
            ("api_key", api_key),
            ("benchmark_account", benchmark_account),
            ("benchmark_user", benchmark_user),
            ("runtime_binding_hash", runtime_binding_hash),
        ):
            if not value:
                raise ValueError(f"OpenViking session {label} is required")
        if re.fullmatch(r"[0-9a-f]{64}", runtime_binding_hash) is None:
            raise ValueError("OpenViking session runtime binding hash must be lowercase SHA-256")
        if task_poll_interval_seconds < 0:
            raise ValueError("OpenViking task poll interval must not be negative")
        if maximum_task_polls < 1:
            raise ValueError("OpenViking maximum task polls must be positive")
        if not math.isfinite(task_poll_timeout_seconds) or task_poll_timeout_seconds <= 0:
            raise ValueError("OpenViking task poll timeout must be finite positive")
        self._benchmark_account = benchmark_account
        self._benchmark_user = benchmark_user
        self._runtime_binding_hash = runtime_binding_hash
        self._task_poll_interval_seconds = task_poll_interval_seconds
        self._maximum_task_polls = maximum_task_polls
        self._task_poll_timeout_seconds = task_poll_timeout_seconds
        self._client = SealedRestClient(
            store=store,
            base_url=base_url,
            headers={"X-API-Key": api_key},
            transport=transport,
            read_timeout_seconds=read_timeout_seconds,
            total_timeout_seconds=total_timeout_seconds,
        )
        self._resolved = False
        self._attempted_allocations: set[str] = set()
        self._scopes: dict[str, _ScopeBinding] = {}
        self._plans: dict[str, tuple[_PlannedSession, ...]] = {}
        self._attempted_sessions: set[str] = set()
        self._completed: dict[str, _CompletedSession] = {}
        self._continuations: dict[str, _ScopeContinuation] = {}
        self._ready_scopes: set[str] = set()

    async def resolve(self) -> RuntimeResolution:
        response = await self._client.request("GET", "/health")
        with bind_sealed_response_validation(
            response,
            message="OpenViking session health response failed exact-profile validation",
        ):
            document = _exact_object(
                response.raw_bytes,
                frozenset(
                    {
                        "status",
                        "healthy",
                        "version",
                        "auth_mode",
                        "account_id",
                        "user_id",
                        "role",
                    }
                ),
                "health",
            )
            if document != {
                "status": "ok",
                "healthy": True,
                "version": OPENVIKING_VERSION,
                "auth_mode": OPENVIKING_AUTH_MODE,
                "account_id": self._benchmark_account,
                "user_id": self._benchmark_user,
                "role": OPENVIKING_USER_ROLE,
            }:
                raise OpenVikingSessionProfileError(
                    "OpenViking session health identity does not match the profile"
                )
        self._resolved = True
        return RuntimeResolution(
            memory_system_id=OPENVIKING_MEMORY_SYSTEM_ID,
            runtime_binding_hash=self._runtime_binding_hash,
            raw_reference=response.raw_reference,
        )

    async def capabilities(self) -> CapabilitySet:
        return CapabilitySet(
            capability_ids=(
                "create-only-peer-session",
                "native-session-message-commit",
                "intent-free-peer-memory-find",
            ),
            provider_order_preserved=True,
            native_reranking_disabled=True,
        )

    async def allocate_ingestion_scope(
        self,
        request: ScopeAllocationRequest,
    ) -> ScopeReceipt:
        if not self._resolved:
            raise RuntimeError("OpenViking session resolve must pass before scope allocation")
        binding = self._scope_binding(request)
        if binding.memory_root in self._attempted_allocations:
            raise OpenVikingSessionProfileError(
                "OpenViking create-only peer scope was already attempted and cannot be reused"
            )
        absence = await self._require_absent(
            path="/api/v1/fs/stat",
            expected_resource=binding.memory_root,
            expected_type="file",
            params={"uri": binding.memory_root},
            headers={ACTOR_PEER_HEADER: binding.actor_peer_id},
        )
        self._attempted_allocations.add(binding.memory_root)
        self._scopes[binding.memory_root] = binding
        return ScopeReceipt(
            ingestion_occurrence_id=request.ingestion_occurrence_id,
            scope_id=binding.memory_root,
            raw_reference=absence,
        )

    async def adopt_ingestion_scope(
        self,
        request: ScopeAllocationRequest,
        *,
        completed_session_task_ids: tuple[str, ...],
        failed_session_task_id: str,
    ) -> ScopeReceipt:
        """Adopt one proven continuation without replaying unknown provider work."""

        if not self._resolved:
            raise RuntimeError("OpenViking session resolve must pass before scope adoption")
        binding = self._scope_binding(request)
        task_ids = (*completed_session_task_ids, failed_session_task_id)
        if (
            not failed_session_task_id
            or any(not task_id for task_id in task_ids)
            or len(set(task_ids)) != len(task_ids)
        ):
            raise OpenVikingSessionProfileError(
                "OpenViking continuation task identities must be non-empty and unique"
            )
        if binding.memory_root in self._attempted_allocations:
            raise OpenVikingSessionProfileError(
                "OpenViking peer scope was already attempted and cannot be reused"
            )

        headers = {ACTOR_PEER_HEADER: binding.actor_peer_id}
        completed: list[_CompletedSession] = []
        for task_id in completed_session_task_ids:
            task, task_reference = await self._read_task(task_id=task_id, headers=headers)
            session_id = _task_session_id(task, task_id)
            result = task.get("result")
            if task.get("status") != "completed" or task.get("error") is not None:
                raise OpenVikingSessionProfileError(
                    "OpenViking continuation prefix task is not completed"
                )
            if not isinstance(result, dict) or result.get("session_id") != session_id:
                raise OpenVikingSessionProfileError(
                    "OpenViking continuation prefix task result has the wrong session"
                )
            archive_uri = _task_archive_uri(result, session_id)
            _require_usage_snapshot(result)
            archive_id = archive_uri.rstrip("/").rsplit("/", 1)[-1]
            archive = await self._client.request(
                "GET",
                f"/api/v1/sessions/{session_id}/archives/{archive_id}",
                request_headers=headers,
            )
            archive_result = _clean_result(archive, "continued session archive")
            if not isinstance(archive_result, dict) or archive_result.get("session_id") not in {
                None,
                session_id,
            }:
                raise OpenVikingSessionProfileError(
                    "OpenViking continued session archive has the wrong session"
                )
            completed.append(
                _CompletedSession(
                    session_id=session_id,
                    archive_uri=archive_uri,
                    task_id=task_id,
                    evidence_references=(task_reference, archive.raw_reference),
                )
            )

        failed_task, failed_task_reference = await self._read_task(
            task_id=failed_session_task_id,
            headers=headers,
        )
        failed_session_id = _task_session_id(failed_task, failed_session_task_id)
        if (
            failed_task.get("status") != "failed"
            or failed_task.get("result") is not None
            or not isinstance(failed_task.get("error"), str)
            or not failed_task["error"]
        ):
            raise OpenVikingSessionProfileError(
                "OpenViking continuation task is not explicitly terminal failed"
            )
        if failed_session_id in {item.session_id for item in completed}:
            raise OpenVikingSessionProfileError(
                "OpenViking failed continuation session overlaps its completed prefix"
            )
        failed_session_reference = await self._require_zero_failed_session(
            session_id=failed_session_id,
            headers=headers,
        )

        projection_response, _memory_uris = await self._capture_memory_projection(binding)
        self._attempted_allocations.add(binding.memory_root)
        self._scopes[binding.memory_root] = binding
        for item in completed:
            self._completed[item.session_id] = item
        self._continuations[binding.memory_root] = _ScopeContinuation(
            completed_session_ids=tuple(item.session_id for item in completed),
            failed_session_id=failed_session_id,
            failed_evidence_references=(failed_task_reference, failed_session_reference),
        )
        return ScopeReceipt(
            ingestion_occurrence_id=request.ingestion_occurrence_id,
            scope_id=binding.memory_root,
            raw_reference=projection_response.raw_reference,
        )

    def plan_ingestion(self, request: IngestionRequest) -> tuple[IngestionDispatch, ...]:
        binding = self._require_scope(request.scope)
        if binding.memory_root in self._plans:
            raise OpenVikingSessionProfileError(
                "OpenViking session ingestion is already planned for this scope"
            )
        sources = request.ordered_source_units
        if not sources:
            raise OpenVikingSessionProfileError("OpenViking session plan requires source sessions")
        if tuple(source.ordinal_1_indexed for source in sources) != tuple(
            range(1, len(sources) + 1)
        ):
            raise OpenVikingSessionProfileError(
                "OpenViking source-session ordinals must be contiguous and ordered"
            )
        if len({source.source_unit_id for source in sources}) != len(sources):
            raise OpenVikingSessionProfileError("OpenViking source-session IDs must be unique")

        planned: list[_PlannedSession] = []
        for source in sources:
            if hashlib.sha256(source.payload_bytes).hexdigest() != source.payload_sha256:
                raise OpenVikingSessionProfileError(
                    "OpenViking source-session payload hash does not match bytes"
                )
            timestamp = _canonical_timestamp(source.occurred_at)
            messages = _parse_messages(source.payload_bytes)
            session_id = openviking_session_id(
                binding.ingestion_occurrence_id,
                source.source_unit_id,
            )
            dispatch = IngestionDispatch(
                dispatch_ordinal_1_indexed=source.ordinal_1_indexed,
                operation_kind=OPENVIKING_SESSION_COMMIT_OPERATION,
                request_fingerprint=canonical_sha256(
                    [
                        "oamb-openviking-session-commit-v1",
                        binding.memory_root,
                        session_id,
                        source.source_unit_id,
                        source.payload_sha256,
                        timestamp,
                    ]
                ),
                ordered_source_units=(source,),
            )
            planned.append(
                _PlannedSession(
                    dispatch=dispatch,
                    source=source,
                    session_id=session_id,
                    messages=messages,
                )
            )
        result = tuple(planned)
        continuation = self._continuations.get(binding.memory_root)
        if continuation is not None:
            completed_count = len(continuation.completed_session_ids)
            session_ids = tuple(item.session_id for item in result)
            if (
                session_ids[:completed_count] != continuation.completed_session_ids
                or completed_count >= len(session_ids)
                or session_ids[completed_count] != continuation.failed_session_id
            ):
                raise OpenVikingSessionProfileError(
                    "OpenViking continuation must be a contiguous completed prefix followed "
                    "by its failed session"
                )
        self._plans[binding.memory_root] = result
        return tuple(item.dispatch for item in result)

    async def ingest(self, request: IngestionDispatchRequest) -> IngestionDispatchReceipt:
        binding = self._require_scope(request.scope)
        planned = self._planned_for_dispatch(binding, request.dispatch)
        if planned.session_id in self._attempted_sessions:
            raise OpenVikingSessionProfileError(
                "OpenViking create-only native session was already attempted and cannot be reused"
            )
        completed_count = sum(
            item.session_id in self._completed for item in self._plans[binding.memory_root]
        )
        if request.dispatch.dispatch_ordinal_1_indexed != completed_count + 1:
            raise OpenVikingSessionProfileError(
                "OpenViking source sessions must be ingested in frozen chronological order"
            )

        headers = {ACTOR_PEER_HEADER: binding.actor_peer_id}
        self._attempted_sessions.add(planned.session_id)
        continuation = self._continuations.get(binding.memory_root)
        if continuation is not None and continuation.failed_session_id == planned.session_id:
            replay_failed_session = True
            evidence_references = list(continuation.failed_evidence_references)
            evidence_references.append(
                await self._require_zero_failed_session(
                    session_id=planned.session_id,
                    headers=headers,
                )
            )
        else:
            replay_failed_session = False
            session_absence = await self._require_absent(
                path=f"/api/v1/sessions/{planned.session_id}",
                expected_resource=planned.session_id,
                expected_type="session",
                params={"auto_create": False},
                headers=headers,
            )
            evidence_references = [session_absence]

            create = await self._client.request(
                "POST",
                "/api/v1/sessions",
                json_payload={
                    "session_id": planned.session_id,
                    "auto_commit_policy": None,
                },
                request_headers=headers,
                write_intent=True,
            )
            evidence_references.append(create.raw_reference)
            create_result = _clean_result(create, "session create")
            if (
                not isinstance(create_result, dict)
                or create_result.get("session_id") != planned.session_id
                or create_result.get("auto_commit_policy", object()) is not None
            ):
                raise OpenVikingSessionProfileError(
                    "OpenViking session create did not prove auto-commit disabled"
                )

        timestamp = _canonical_timestamp(planned.source.occurred_at)
        message_payloads = tuple(
            {
                "role": message["role"],
                "content": message["content"],
                "peer_id": binding.actor_peer_id,
                "created_at": timestamp,
            }
            for message in planned.messages
        )
        for start in range(0, len(message_payloads), MAX_MESSAGES_PER_BATCH):
            batch = message_payloads[start : start + MAX_MESSAGES_PER_BATCH]
            message_response = await self._client.request(
                "POST",
                f"/api/v1/sessions/{planned.session_id}/messages/batch",
                json_payload={"messages": batch},
                request_headers=headers,
                write_intent=True,
            )
            evidence_references.append(message_response.raw_reference)
            message_result = _clean_result(message_response, "session messages")
            if (
                not isinstance(message_result, dict)
                or message_result.get("session_id") != planned.session_id
            ):
                raise OpenVikingSessionProfileError(
                    "OpenViking session message response has the wrong session"
                )

        commit = await self._client.request(
            "POST",
            f"/api/v1/sessions/{planned.session_id}/commit",
            json_payload={"keep_recent_count": 0},
            request_headers=headers,
            write_intent=True,
        )
        evidence_references.append(commit.raw_reference)
        commit_result = _clean_result(commit, "session commit")
        task_id, archive_uri = _commit_identity(commit_result, planned.session_id)

        task_result, task_references = await self._poll_terminal_task(
            task_id=task_id,
            session_id=planned.session_id,
            archive_uri=archive_uri,
            headers=headers,
        )
        evidence_references.extend(task_references)
        _require_usage_snapshot(task_result)

        archive_id = archive_uri.rstrip("/").rsplit("/", 1)[-1]
        archive = await self._client.request(
            "GET",
            f"/api/v1/sessions/{planned.session_id}/archives/{archive_id}",
            request_headers=headers,
        )
        evidence_references.append(archive.raw_reference)
        archive_result = _clean_result(archive, "session archive")
        if not isinstance(archive_result, dict) or archive_result.get("session_id") not in {
            None,
            planned.session_id,
        }:
            raise OpenVikingSessionProfileError("OpenViking session archive has the wrong session")

        projection_response, _uris = await self._capture_memory_projection(binding)
        evidence_references.append(projection_response.raw_reference)
        completed = _CompletedSession(
            session_id=planned.session_id,
            archive_uri=archive_uri,
            task_id=task_id,
            evidence_references=tuple(evidence_references),
        )
        self._completed[planned.session_id] = completed
        if replay_failed_session:
            del self._continuations[binding.memory_root]
        return IngestionDispatchReceipt(
            attempt_id=request.attempt_id,
            dispatch=request.dispatch,
            accepted_source_unit_ids=(planned.source.source_unit_id,),
            rejected_source_unit_ids=(),
            raw_reference=projection_response.raw_reference,
            raw_response_bytes=projection_response.raw_bytes,
            usage_records=(),
        )

    async def wait_ready(self, request: ReadinessRequest) -> ReadinessReceipt:
        binding = self._require_scope(request.scope)
        planned = self._plans.get(binding.memory_root)
        if planned is None:
            raise OpenVikingSessionProfileError("OpenViking session readiness has no frozen plan")
        expected_source_ids = tuple(item.source.source_unit_id for item in planned)
        requested_source_ids = tuple(request.expected_source_unit_ids)
        receipt = request.ingestion_receipt
        if (
            requested_source_ids != expected_source_ids
            or receipt.ingestion_occurrence_id != binding.ingestion_occurrence_id
        ):
            return ReadinessReceipt(
                ingestion_occurrence_id=binding.ingestion_occurrence_id,
                ready=False,
                evidence_references=(),
            )
        receipt_source_ids = tuple(
            source_id
            for dispatch_receipt in receipt.dispatch_receipts
            for source_id in dispatch_receipt.accepted_source_unit_ids
        )
        completed_source_ids = tuple(
            item.source.source_unit_id for item in planned if item.session_id in self._completed
        )
        if (
            receipt_source_ids != expected_source_ids
            or receipt.accepted_source_unit_ids != expected_source_ids
            or receipt.rejected_source_unit_ids
            or completed_source_ids != expected_source_ids
            or len(receipt.dispatch_receipts) != len(planned)
        ):
            return ReadinessReceipt(
                ingestion_occurrence_id=binding.ingestion_occurrence_id,
                ready=False,
                evidence_references=tuple(
                    reference
                    for item in planned
                    if (completed := self._completed.get(item.session_id)) is not None
                    for reference in completed.evidence_references
                ),
            )
        for expected, actual in zip(planned, receipt.dispatch_receipts, strict=True):
            if (
                actual.dispatch != expected.dispatch
                or hashlib.sha256(actual.raw_response_bytes).hexdigest()
                != actual.raw_reference.sha256
            ):
                raise OpenVikingSessionProfileError(
                    "OpenViking session readiness receipt binding is invalid"
                )
        projection = await self.project(request.scope)
        if projection.inventory.ordered_source_unit_ids != expected_source_ids:
            raise OpenVikingSessionProfileError(
                "OpenViking session readiness projection lost source-session order"
            )
        self._ready_scopes.add(binding.memory_root)
        return ReadinessReceipt(
            ingestion_occurrence_id=binding.ingestion_occurrence_id,
            ready=True,
            evidence_references=(
                *(
                    reference
                    for item in planned
                    for reference in self._completed[item.session_id].evidence_references
                ),
                projection.inventory.raw_reference,
                *projection.supporting_raw_references,
            ),
        )

    async def inventory(self, scope: ScopeReceipt) -> InventoryReceipt:
        return (await self.project(scope)).inventory

    async def state_digest(self, scope: ScopeReceipt) -> StateDigestReceipt:
        return (await self.project(scope)).state_digest

    async def project(self, scope: ScopeReceipt) -> ProjectionReceipt:
        binding = self._require_scope(scope)
        response, memory_uris = await self._capture_memory_projection(binding)
        planned = self._plans.get(binding.memory_root, ())
        source_ids = tuple(
            item.source.source_unit_id for item in planned if item.session_id in self._completed
        )
        state_sha256 = canonical_sha256(
            [
                "oamb-openviking-session-projection-v1",
                binding.ingestion_occurrence_id,
                source_ids,
                memory_uris,
            ]
        )
        return ProjectionReceipt(
            inventory=InventoryReceipt(
                ingestion_occurrence_id=binding.ingestion_occurrence_id,
                ordered_source_unit_ids=source_ids,
                raw_reference=response.raw_reference,
            ),
            state_digest=StateDigestReceipt(
                ingestion_occurrence_id=binding.ingestion_occurrence_id,
                state_sha256=state_sha256,
                raw_reference=response.raw_reference,
            ),
        )

    async def retrieve(self, request: RetrievalRequest) -> NativeEvidenceBatch:
        binding = self._require_scope(request.scope)
        if binding.memory_root not in self._ready_scopes:
            raise OpenVikingSessionProfileError(
                "OpenViking session scope is not ready for retrieval"
            )
        if request.top_k != NATIVE_RETRIEVAL_TOP_K:
            raise OpenVikingSessionProfileError("OpenViking session find top_k must equal 100")
        try:
            query = request.query_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise OpenVikingSessionProfileError(
                "OpenViking session find query must be UTF-8"
            ) from exc
        response = await self._client.request(
            "POST",
            "/api/v1/search/find",
            json_payload={
                "query": query,
                "target_uri": binding.memory_root,
                "context_type": "memory",
                "limit": NATIVE_RETRIEVAL_TOP_K,
            },
            request_headers={ACTOR_PEER_HEADER: binding.actor_peer_id},
            request_evidence=True,
        )
        result = _clean_result(response, "peer-memory find")
        hits = _memory_hits(result, binding.memory_root)
        return NativeEvidenceBatch(
            raw_reference=response.raw_reference,
            request_raw_reference=response.request_reference,
            candidates=tuple(
                NativeEvidenceCandidate(
                    native_id=f"{hit['uri']}#level={hit['level']}",
                    native_rank_1_indexed=rank,
                    content=hit["abstract"],
                    native_score=_canonical_score(hit["score"]),
                    provider_evidence_identity=f"{hit['uri']}#level={hit['level']}",
                    source_unit_id=None,
                    evidence_kind="native_memory",
                    native_reference=hit["uri"],
                    native_truncated=False,
                )
                for rank, hit in enumerate(hits, start=1)
            ),
        )

    async def close(self) -> None:
        await self._client.close()

    async def _poll_terminal_task(
        self,
        *,
        task_id: str,
        session_id: str,
        archive_uri: str,
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], tuple[RawReferenceHandle, ...]]:
        references: list[RawReferenceHandle] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._task_poll_timeout_seconds
        for _poll in range(self._maximum_task_polls):
            remaining_seconds = deadline - loop.time()
            if remaining_seconds <= 0:
                break
            try:
                response = await self._client.request(
                    "GET",
                    f"/api/v1/tasks/{task_id}",
                    request_headers=headers,
                    total_timeout_seconds=remaining_seconds,
                )
            except MemorySystemCallFailure as exc:
                if exc.failure_kind != "timeout":
                    raise
                raise MemorySystemCallUnknownOutcome(
                    "OpenViking session commit task did not become terminal before timeout",
                    failure_kind="timeout",
                ) from exc
            references.append(response.raw_reference)
            result = _clean_result(response, "session commit task")
            if not isinstance(result, dict):
                raise OpenVikingSessionProfileError(
                    "OpenViking session commit task result is not an object"
                )
            if (
                result.get("task_id") != task_id
                or result.get("task_type") != "session_commit"
                or result.get("resource_id") != session_id
            ):
                raise OpenVikingSessionProfileError(
                    "OpenViking session commit task identity is invalid"
                )
            status = result.get("status")
            if status == "completed":
                terminal_result = result.get("result")
                if (
                    not isinstance(terminal_result, dict)
                    or terminal_result.get("session_id") != session_id
                    or terminal_result.get("archive_uri") != archive_uri
                ):
                    raise OpenVikingSessionProfileError(
                        "OpenViking completed task result is not bound to its archive"
                    )
                return terminal_result, tuple(references)
            if status in _TERMINAL_FAILURE_TASK_STATUSES:
                raise OpenVikingSessionProfileError(
                    f"OpenViking session commit task ended as {status}"
                )
            if status not in _ACTIVE_TASK_STATUSES:
                raise OpenVikingSessionProfileError(
                    "OpenViking session commit task status is invalid"
                )
            remaining_seconds = deadline - loop.time()
            if remaining_seconds <= 0:
                break
            await asyncio.sleep(min(self._task_poll_interval_seconds, remaining_seconds))
        raise MemorySystemCallUnknownOutcome(
            "OpenViking session commit task did not become terminal before timeout",
            failure_kind="timeout",
        )

    async def _read_task(
        self,
        *,
        task_id: str,
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], RawReferenceHandle]:
        response = await self._client.request(
            "GET",
            f"/api/v1/tasks/{task_id}",
            request_headers=headers,
        )
        result = _clean_result(response, "continuation task")
        if not isinstance(result, dict):
            raise OpenVikingSessionProfileError(
                "OpenViking continuation task result is not an object"
            )
        if result.get("task_id") != task_id or result.get("task_type") != "session_commit":
            raise OpenVikingSessionProfileError("OpenViking continuation task identity is invalid")
        return result, response.raw_reference

    async def _require_zero_failed_session(
        self,
        *,
        session_id: str,
        headers: dict[str, str],
    ) -> RawReferenceHandle:
        session = await self._client.request(
            "GET",
            f"/api/v1/sessions/{session_id}",
            params={"auto_create": False},
            request_headers=headers,
        )
        result = _clean_result(session, "failed continuation session")
        _require_zero_failed_session(result, session_id)
        return session.raw_reference

    async def _capture_memory_projection(
        self,
        binding: _ScopeBinding,
    ) -> tuple[SealedRestResponse, tuple[str, ...]]:
        try:
            response = await self._client.request(
                "GET",
                "/api/v1/fs/ls",
                params={
                    "uri": binding.memory_root,
                    "simple": True,
                    "recursive": True,
                    "output": "original",
                    "show_all_hidden": True,
                },
                request_headers={ACTOR_PEER_HEADER: binding.actor_peer_id},
            )
        except MemorySystemCallFailure as exc:
            if (
                exc.failure_kind != "http_status"
                or exc.status_code != 404
                or exc.raw_reference is None
                or exc.raw_response_bytes is None
            ):
                raise
            _parse_not_found(
                exc.raw_response_bytes,
                expected_resource=binding.memory_root,
                expected_type="directory",
            )
            return (
                SealedRestResponse(
                    status_code=exc.status_code,
                    raw_bytes=exc.raw_response_bytes,
                    raw_reference=exc.raw_reference,
                ),
                (),
            )
        result = _clean_result(response, "peer-memory projection")
        if (
            not isinstance(result, list)
            or any(not isinstance(item, str) for item in result)
            or len(set(result)) != len(result)
            or any(not item.startswith(f"{binding.memory_root}/") for item in result)
        ):
            raise OpenVikingSessionProfileError(
                "OpenViking peer-memory projection is incomplete or outside the root"
            )
        return response, tuple(result)

    async def _require_absent(
        self,
        *,
        path: str,
        expected_resource: str,
        expected_type: str,
        params: dict[str, str | bool],
        headers: dict[str, str],
    ) -> RawReferenceHandle:
        try:
            await self._client.request(
                "GET",
                path,
                params=params,
                request_headers=headers,
            )
        except MemorySystemCallFailure as exc:
            if (
                exc.failure_kind != "http_status"
                or exc.status_code != 404
                or exc.raw_reference is None
                or exc.raw_response_bytes is None
            ):
                raise
            _parse_not_found(
                exc.raw_response_bytes,
                expected_resource=expected_resource,
                expected_type=expected_type,
            )
            return exc.raw_reference
        raise OpenVikingSessionProfileError(
            f"OpenViking create-only {expected_type} already exists: {expected_resource}"
        )

    def _planned_for_dispatch(
        self,
        binding: _ScopeBinding,
        dispatch: IngestionDispatch,
    ) -> _PlannedSession:
        for item in self._plans.get(binding.memory_root, ()):
            if item.dispatch == dispatch:
                return item
        raise OpenVikingSessionProfileError(
            "OpenViking session dispatch does not match the frozen plan"
        )

    def _scope_binding(self, request: ScopeAllocationRequest) -> _ScopeBinding:
        actor_peer_id = (
            "oamb-"
            + hashlib.sha256(
                b"peer\0" + request.ingestion_occurrence_id.encode("utf-8")
            ).hexdigest()
        )
        peer_root = f"viking://user/{self._benchmark_user}/peers/{actor_peer_id}"
        return _ScopeBinding(
            ingestion_occurrence_id=request.ingestion_occurrence_id,
            ingestion_plan_id=request.ingestion_plan_id,
            actor_peer_id=actor_peer_id,
            peer_root=peer_root,
            memory_root=f"{peer_root}/memories",
        )

    def _require_scope(self, scope: ScopeReceipt) -> _ScopeBinding:
        binding = self._scopes.get(scope.scope_id)
        if binding is None or binding.ingestion_occurrence_id != scope.ingestion_occurrence_id:
            raise OpenVikingSessionProfileError(
                "OpenViking session scope is not allocated by this adapter"
            )
        return binding


def _canonical_timestamp(value: str | None) -> str:
    if not isinstance(value, str) or not value:
        raise OpenVikingSessionProfileError(
            "OpenViking source session requires a canonical created_at"
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise OpenVikingSessionProfileError(
            "OpenViking source session created_at is invalid"
        ) from exc
    if parsed.tzinfo is None or parsed.isoformat(timespec="seconds") != value:
        raise OpenVikingSessionProfileError("OpenViking source session created_at is not canonical")
    return value


def _parse_messages(payload: bytes) -> tuple[dict[str, str], ...]:
    try:
        document = json.loads(payload, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OpenVikingSessionProfileError(
            "OpenViking source session payload is not valid JSON"
        ) from exc
    if not isinstance(document, list) or not document:
        raise OpenVikingSessionProfileError(
            "OpenViking source session requires a non-empty message array"
        )
    messages: list[dict[str, str]] = []
    for value in document:
        if not isinstance(value, dict) or frozenset(value) != _MESSAGE_FIELDS:
            raise OpenVikingSessionProfileError(
                "OpenViking source session message has an invalid shape"
            )
        role = value["role"]
        content = value["content"]
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise OpenVikingSessionProfileError(
                "OpenViking source session message requires role and text"
            )
        messages.append({"role": role, "content": content})
    return tuple(messages)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise OpenVikingSessionProfileError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _exact_object(raw_bytes: bytes, fields: frozenset[str], label: str) -> dict[str, Any]:
    try:
        return parse_exact_json_object(raw_bytes, expected_fields=fields)
    except ValueError as exc:
        raise OpenVikingSessionProfileError(f"invalid OpenViking session {label} response") from exc


def _clean_result(response: SealedRestResponse, label: str) -> object:
    with bind_sealed_response_validation(
        response,
        message=f"OpenViking {label} response failed exact-profile validation",
    ):
        document = _exact_one_of(
            response.raw_bytes,
            (
                frozenset({"status", "result"}),
                frozenset({"status", "result", "error", "telemetry", "profile"}),
            ),
            label,
        )
        if document["status"] != "ok" or any(
            document.get(field) is not None for field in ("error", "telemetry", "profile")
        ):
            raise OpenVikingSessionProfileError(
                f"OpenViking {label} did not return a clean success"
            )
        return document["result"]


def _parse_not_found(
    raw_bytes: bytes,
    *,
    expected_resource: str,
    expected_type: str,
) -> None:
    document = _exact_one_of(
        raw_bytes,
        (
            frozenset({"status", "error"}),
            frozenset({"status", "result", "error", "telemetry", "profile"}),
        ),
        "not-found",
    )
    error = document["error"]
    if expected_type == "session":
        expected_details: object = None
        expected_message = f"Session {expected_resource} not found"
    else:
        expected_details = {"resource": expected_resource, "type": expected_type}
        expected_message = f"{expected_type.capitalize()} not found: {expected_resource}"
    if (
        document["status"] != "error"
        or document.get("result") is not None
        or document.get("telemetry") is not None
        or document.get("profile") is not None
        or not isinstance(error, dict)
        or error.get("code") != "NOT_FOUND"
        or error.get("message") != expected_message
        or error.get("details") != expected_details
    ):
        raise OpenVikingSessionProfileError("OpenViking create-only absence response is invalid")


def _exact_one_of(
    raw_bytes: bytes,
    field_sets: tuple[frozenset[str], ...],
    label: str,
) -> dict[str, Any]:
    for fields in field_sets:
        try:
            return parse_exact_json_object(raw_bytes, expected_fields=fields)
        except ValueError:
            continue
    expected = " or ".join(str(sorted(fields)) for fields in field_sets)
    raise OpenVikingSessionProfileError(
        f"invalid OpenViking session {label} response; expected fields {expected}"
    )


def _commit_identity(result: object, session_id: str) -> tuple[str, str]:
    if not isinstance(result, dict):
        raise OpenVikingSessionProfileError("OpenViking session commit result is not an object")
    task_id = result.get("task_id")
    archive_uri = result.get("archive_uri")
    if (
        result.get("session_id") != session_id
        or result.get("status") != "accepted"
        or result.get("archived") is not True
        or not isinstance(task_id, str)
        or not task_id
        or not isinstance(archive_uri, str)
        or not archive_uri
    ):
        raise OpenVikingSessionProfileError(
            "OpenViking session commit did not close archive and task identity"
        )
    return task_id, archive_uri


def _task_session_id(task: dict[str, Any], task_id: str) -> str:
    session_id = task.get("resource_id")
    if not isinstance(session_id, str) or not session_id:
        raise OpenVikingSessionProfileError(
            f"OpenViking continuation task {task_id} has no session identity"
        )
    return session_id


def _task_archive_uri(task_result: dict[str, Any], session_id: str) -> str:
    archive_uri = task_result.get("archive_uri")
    if not isinstance(archive_uri, str) or not archive_uri:
        raise OpenVikingSessionProfileError(
            "OpenViking completed continuation task has no archive identity"
        )
    session_marker = f"/sessions/{session_id}/history/"
    if session_marker not in archive_uri or archive_uri.endswith("/"):
        raise OpenVikingSessionProfileError(
            "OpenViking completed continuation task archive is outside its session"
        )
    return archive_uri


def _require_zero_failed_session(result: object, session_id: str) -> None:
    if not isinstance(result, dict) or result.get("session_id") != session_id:
        raise OpenVikingSessionProfileError(
            "OpenViking failed continuation proof has the wrong session"
        )
    if (
        result.get("message_count") != 0
        or not _positive_integer(result.get("total_message_count"))
        or not _positive_integer(result.get("commit_count"))
        or result.get("pending_tokens") != 0
        or not _zero_integer_mapping(result.get("memories_extracted"))
        or not _zero_integer_mapping(result.get("llm_token_usage"))
        or not _zero_integer_mapping(result.get("embedding_token_usage"))
    ):
        raise OpenVikingSessionProfileError(
            "OpenViking failed continuation session does not prove zero memory and usage"
        )


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _zero_integer_mapping(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            not isinstance(item, bool) and isinstance(item, int) and item == 0
            for item in value.values()
        )
    )


def _require_usage_snapshot(task_result: dict[str, Any]) -> None:
    usage = task_result.get("token_usage")
    if (
        not isinstance(usage, dict)
        or not isinstance(usage.get("llm"), dict)
        or not isinstance(usage.get("embedding"), dict)
        or not isinstance(usage.get("total"), dict)
    ):
        raise OpenVikingSessionProfileError("OpenViking completed task has no usage snapshot")


def _memory_hits(result: object, memory_root: str) -> tuple[dict[str, Any], ...]:
    if not isinstance(result, dict) or frozenset(result) != frozenset(
        {"memories", "resources", "skills", "total"}
    ):
        raise OpenVikingSessionProfileError("OpenViking peer-memory find result shape is invalid")
    memories = result["memories"]
    if not isinstance(memories, list) or result["resources"] != [] or result["skills"] != []:
        raise OpenVikingSessionProfileError(
            "OpenViking peer-memory find returned a non-memory bucket"
        )
    if result["total"] != len(memories):
        raise OpenVikingSessionProfileError("OpenViking peer-memory find total does not match hits")
    hits: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    fields = frozenset({"context_type", "uri", "level", "score", "abstract", "tags"})
    for value in memories:
        if not isinstance(value, dict) or frozenset(value) != fields:
            raise OpenVikingSessionProfileError("OpenViking peer-memory hit shape is invalid")
        uri = value["uri"]
        level = value["level"]
        score = value["score"]
        if (
            value["context_type"] != "memory"
            or not isinstance(uri, str)
            or not uri.startswith(f"{memory_root}/")
            or isinstance(level, bool)
            or not isinstance(level, int)
            or level not in {0, 1, 2}
            or isinstance(score, bool)
            or not isinstance(score, (int, float))
            or not math.isfinite(float(score))
            or not isinstance(value["abstract"], str)
            or not isinstance(value["tags"], list)
            or any(not isinstance(tag, str) for tag in value["tags"])
        ):
            raise OpenVikingSessionProfileError(
                "OpenViking peer-memory hit is outside the exact profile"
            )
        identity = (uri, level)
        if identity in seen:
            raise OpenVikingSessionProfileError(
                "OpenViking peer-memory find returned a duplicate hit"
            )
        seen.add(identity)
        hits.append(value)
    return tuple(hits)


def _canonical_score(value: int | float) -> str:
    decimal = Decimal(str(value))
    text = format(decimal, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in {"", "-0"} else text


__all__ = [
    "OpenVikingSessionAdapter",
    "OpenVikingSessionProfileError",
    "OPENVIKING_SESSION_PROFILE_ID",
]
