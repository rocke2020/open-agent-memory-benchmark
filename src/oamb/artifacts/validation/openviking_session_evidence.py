"""Strict reconstruction for the OpenViking LME session profile."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from oamb.contracts.evidence import CaseRecordV3, IngestionPlanRecordV2
from oamb.contracts.ids import canonical_sha256, openviking_session_id
from oamb.contracts.ports import NativeEvidenceCandidate
from oamb.contracts.states import AttemptOutcome
from oamb.memory_systems.openviking.session_adapter import (
    OPENVIKING_SESSION_PROFILE_ID,
    _canonical_score,
    _memory_hits,
    _parse_not_found,
)


@dataclass(frozen=True, slots=True)
class OpenVikingSessionPlanEvidence:
    memory_root: str
    memory_uris: tuple[str, ...]
    state_sha256: str


@dataclass(frozen=True, slots=True)
class OpenVikingSessionIndexingUsage:
    attempt_id: str
    raw_response_ref: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cached_tokens: int
    reasoning_tokens: int


def reconstruct_openviking_session_indexing_usage(
    *,
    raw_payloads: Mapping[str, bytes],
    plan: Mapping[str, Any],
) -> tuple[OpenVikingSessionIndexingUsage, ...]:
    if plan.get("adapter_profile_id") != OPENVIKING_SESSION_PROFILE_ID:
        raise ValueError("OpenViking indexing usage has the wrong adapter profile")
    attempt_ids = _ordered_text(plan, "ordered_dispatch_attempt_ids")
    source_ids = _ordered_text(plan, "ordered_source_unit_ids")
    readiness_refs = _ordered_text(plan, "readiness_evidence_refs")
    ingestion_occurrence_id = plan.get("ingestion_occurrence_id")
    if (
        not attempt_ids
        or not isinstance(ingestion_occurrence_id, str)
        or not ingestion_occurrence_id
        or len(attempt_ids) != len(source_ids)
        or len(set(attempt_ids)) != len(attempt_ids)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError("OpenViking indexing usage dispatch ledger is invalid")

    expected = {
        openviking_session_id(ingestion_occurrence_id, source_id): attempt_id
        for source_id, attempt_id in zip(source_ids, attempt_ids, strict=True)
    }
    accepted: dict[str, set[tuple[str, str]]] = {session_id: set() for session_id in expected}
    completed: dict[str, OpenVikingSessionIndexingUsage] = {}
    completed_identities: dict[str, tuple[str, str]] = {}
    task_ids: set[str] = set()
    for reference in dict.fromkeys(readiness_refs):
        payload = _payload(raw_payloads, reference)
        if hashlib.sha256(payload).hexdigest() != reference:
            raise ValueError("OpenViking indexing usage raw identity does not close")
        try:
            document = json.loads(payload)
        except (UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(document, dict):
            continue
        result = document.get("result")
        if isinstance(result, dict) and result.get("status") == "accepted":
            session_id = result.get("session_id")
            if session_id not in expected:
                continue
            task_id = result.get("task_id")
            archive_uri = result.get("archive_uri")
            if (
                document.get("status") != "ok"
                or any(
                    document.get(field) is not None for field in ("error", "telemetry", "profile")
                )
                or result.get("archived") is not True
                or not isinstance(task_id, str)
                or not task_id
                or not isinstance(archive_uri, str)
                or not archive_uri
            ):
                raise ValueError("OpenViking indexing usage commit identity does not close")
            accepted[session_id].add((task_id, archive_uri))
            continue
        if (
            not isinstance(result, dict)
            or result.get("task_type") != "session_commit"
            or result.get("status") != "completed"
        ):
            continue
        task_id = result.get("task_id")
        task_result = result.get("result")
        session_id = result.get("resource_id")
        if isinstance(session_id, str) and session_id not in expected:
            continue
        archive_uri = task_result.get("archive_uri") if isinstance(task_result, dict) else None
        if (
            document.get("status") != "ok"
            or any(document.get(field) is not None for field in ("error", "telemetry", "profile"))
            or not isinstance(task_id, str)
            or not task_id
            or not isinstance(task_result, dict)
            or not isinstance(session_id, str)
            or task_result.get("session_id") != session_id
            or not isinstance(archive_uri, str)
            or not archive_uri
        ):
            raise ValueError("OpenViking indexing usage task identity does not close")
        if task_id in task_ids or session_id in completed:
            raise ValueError("OpenViking indexing usage has conflicting completed snapshots")
        task_ids.add(task_id)
        usage = _task_indexing_usage(
            task_result.get("token_usage"),
            attempt_id=expected[session_id],
            raw_response_ref=reference,
        )
        completed[session_id] = usage
        completed_identities[session_id] = (task_id, archive_uri)

    expected_sessions = tuple(
        openviking_session_id(ingestion_occurrence_id, source_id) for source_id in source_ids
    )
    if any(
        completed_identities[session_id] not in accepted[session_id] for session_id in completed
    ):
        raise ValueError("OpenViking indexing usage task is not bound to its accepted commit")
    if set(completed) != set(expected_sessions):
        return ()
    return tuple(completed[session_id] for session_id in expected_sessions)


def reconstruct_openviking_session_plan(
    *,
    raw_payloads: Mapping[str, bytes],
    plan: IngestionPlanRecordV2,
    attempts: Mapping[str, Any],
    runtime_user_id: str,
) -> OpenVikingSessionPlanEvidence:
    if plan.adapter_profile_id != OPENVIKING_SESSION_PROFILE_ID or plan.scope_id is None:
        raise ValueError("OpenViking session plan profile or scope is invalid")
    peer_id = (
        "oamb-"
        + hashlib.sha256(b"peer\0" + plan.ingestion_occurrence_id.encode("utf-8")).hexdigest()
    )
    memory_root = f"viking://user/{runtime_user_id}/peers/{peer_id}/memories"
    if plan.scope_id != memory_root or not plan.scope_raw_refs:
        raise ValueError("OpenViking session scope identity is invalid")
    scope_payload = _payload(raw_payloads, plan.scope_raw_refs[0])
    _parse_not_found(scope_payload, expected_resource=memory_root, expected_type="file")
    if (
        set(plan.accepted_source_unit_ids)
        | set(plan.rejected_source_unit_ids)
        | set(plan.skipped_source_unit_ids)
        != set(plan.ordered_source_unit_ids)
        or not set(plan.projected_source_unit_ids)
        <= (set(plan.accepted_source_unit_ids) | set(plan.skipped_source_unit_ids))
        or len(plan.ordered_dispatch_attempt_ids) != len(plan.ordered_source_unit_ids)
    ):
        raise ValueError("OpenViking session source ledger is not exact and ordered")
    for source_id, attempt_id in zip(
        plan.ordered_source_unit_ids, plan.ordered_dispatch_attempt_ids, strict=True
    ):
        attempt = attempts.get(attempt_id)
        skipped = source_id in plan.skipped_source_unit_ids
        if attempt is None or attempt.stage != "memory_ingest":
            raise ValueError("OpenViking session dispatch attempt is incomplete")
        if skipped:
            if (
                attempt.outcome != AttemptOutcome.FAILED
                or attempt.raw_error_ref not in plan.readiness_evidence_refs
            ):
                raise ValueError("OpenViking skipped session lacks terminal failure evidence")
        elif (
            attempt.outcome != AttemptOutcome.SUCCEEDED
            or attempt.raw_response_ref not in plan.readiness_evidence_refs
        ):
            raise ValueError("OpenViking session dispatch attempt is incomplete")
    final_attempts_by_source = dict(
        zip(plan.ordered_source_unit_ids, plan.ordered_dispatch_attempt_ids, strict=True)
    )
    expected_sessions = tuple(
        _session_id_for_attempt(
            plan.ingestion_occurrence_id,
            source_id,
            _batch_attempt_ordinal(attempts[final_attempts_by_source[source_id]], attempts),
        )
        for source_id in plan.accepted_source_unit_ids
    )
    created, committed, completed, archived = _session_terminal_evidence(
        raw_payloads,
        plan.readiness_evidence_refs,
    )
    if any(
        session_id not in evidence
        for session_id in expected_sessions
        for evidence in (created, committed, completed, archived)
    ):
        raise ValueError("OpenViking session terminal evidence is incomplete")
    memory_uris = reconstruct_openviking_session_projection(
        raw_payloads=raw_payloads,
        references=plan.projection_raw_refs,
        memory_root=memory_root,
    )
    state_sha256 = canonical_sha256(
        [
            "oamb-openviking-session-projection-v1",
            plan.ingestion_occurrence_id,
            plan.projected_source_unit_ids,
            memory_uris,
        ]
    )
    if (
        plan.inventory_raw_ref not in plan.projection_raw_refs
        or state_sha256 != plan.protected_state_sha256
    ):
        raise ValueError("OpenViking session projection state is invalid")
    return OpenVikingSessionPlanEvidence(memory_root, memory_uris, state_sha256)


def _batch_attempt_ordinal(attempt: Any, attempts: Mapping[str, Any]) -> int:
    ordinal = 1
    seen = {attempt.attempt_id}
    predecessor_id = attempt.retry_of_attempt_id
    while predecessor_id is not None:
        if predecessor_id in seen or predecessor_id not in attempts:
            raise ValueError("OpenViking session retry chain is incomplete")
        seen.add(predecessor_id)
        ordinal += 1
        predecessor_id = attempts[predecessor_id].retry_of_attempt_id
    if ordinal > 3:
        raise ValueError("OpenViking session retry chain exceeds the batch allowance")
    return ordinal


def _session_id_for_attempt(
    ingestion_occurrence_id: str,
    source_id: str,
    batch_attempt_ordinal: int,
) -> str:
    identity = (
        source_id
        if batch_attempt_ordinal == 1
        else f"{source_id}:batch-attempt:{batch_attempt_ordinal}"
    )
    return openviking_session_id(ingestion_occurrence_id, identity)


def reconstruct_openviking_session_projection(
    *,
    raw_payloads: Mapping[str, bytes],
    references: tuple[str, ...],
    memory_root: str,
) -> tuple[str, ...]:
    if not references:
        raise ValueError("OpenViking session projection evidence is absent")
    payload = _payload(raw_payloads, references[0])
    document = _object(payload)
    if document.get("status") == "error":
        _parse_not_found(payload, expected_resource=memory_root, expected_type="file")
        return ()
    result = document.get("result")
    if (
        document.get("status") != "ok"
        or any(document.get(field) is not None for field in ("error", "telemetry", "profile"))
        or not isinstance(result, list)
        or any(not isinstance(uri, str) or not uri.startswith(f"{memory_root}/") for uri in result)
        or len(set(result)) != len(result)
    ):
        raise ValueError("OpenViking session projection payload is invalid")
    return tuple(result)


def reconstruct_openviking_session_candidates(
    *,
    raw_payloads: Mapping[str, bytes],
    case: CaseRecordV3,
    memory_root: str,
) -> tuple[NativeEvidenceCandidate, ...]:
    payload = _payload(raw_payloads, case.retrieval_raw_ref)
    document = _object(payload)
    if document.get("status") != "ok":
        raise ValueError("OpenViking session find response is not successful")
    hits = _memory_hits(document.get("result"), memory_root)
    return tuple(
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
    )


def _session_terminal_evidence(
    raw_payloads: Mapping[str, bytes],
    references: tuple[str, ...],
) -> tuple[set[str], set[str], set[str], set[str]]:
    created: set[str] = set()
    committed: set[str] = set()
    completed: set[str] = set()
    archived: set[str] = set()
    pending_archive: tuple[str, str] | None = None
    for reference in references:
        try:
            document = _object(_payload(raw_payloads, reference))
        except ValueError:
            continue
        if document.get("status") != "ok":
            continue
        result = document.get("result")
        if not isinstance(result, dict):
            continue
        task_result = result.get("result")
        resource_id = result.get("resource_id")
        if (
            result.get("status") == "completed"
            and isinstance(resource_id, str)
            and isinstance(task_result, dict)
            and task_result.get("session_id") == resource_id
            and isinstance(task_result.get("token_usage"), dict)
        ):
            completed.add(resource_id)
            archive_uri = task_result.get("archive_uri")
            if isinstance(archive_uri, str) and archive_uri.rstrip("/"):
                pending_archive = (resource_id, archive_uri.rstrip("/").rsplit("/", 1)[-1])
        session_id = result.get("session_id")
        archive_id = result.get("archive_id")
        if not isinstance(session_id, str) and isinstance(archive_id, str):
            if (
                pending_archive is not None
                and pending_archive[1] == archive_id
                and isinstance(result.get("messages"), list)
                and isinstance(result.get("abstract"), str)
                and isinstance(result.get("overview"), str)
            ):
                archived.add(pending_archive[0])
                pending_archive = None
            continue
        if not isinstance(session_id, str):
            continue
        if result.get("auto_commit_policy", object()) is None:
            created.add(session_id)
        if result.get("status") == "accepted" and isinstance(result.get("task_id"), str):
            committed.add(session_id)
        if isinstance(archive_id, str):
            archived.add(session_id)
        token_usage = result.get("token_usage")
        if isinstance(token_usage, dict) and isinstance(token_usage.get("total"), dict):
            completed.add(session_id)
    return created, committed, completed, archived


def _ordered_text(plan: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = plan.get(field)
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"OpenViking indexing usage {field} is invalid")
    return tuple(value)


def _task_indexing_usage(
    value: object,
    *,
    attempt_id: str,
    raw_response_ref: str,
) -> OpenVikingSessionIndexingUsage:
    if not isinstance(value, dict):
        raise ValueError("OpenViking indexing usage snapshot is absent")
    llm = value.get("llm")
    embedding = value.get("embedding")
    total = value.get("total")
    if not isinstance(llm, dict) or not isinstance(embedding, dict) or not isinstance(total, dict):
        raise ValueError("OpenViking indexing usage snapshot is malformed")
    prompt_tokens = _non_negative_integer(llm, "prompt_tokens")
    completion_tokens = _non_negative_integer(llm, "completion_tokens")
    llm_total_tokens = _non_negative_integer(llm, "total_tokens")
    cached_tokens = _non_negative_integer(llm, "cached_tokens")
    reasoning_tokens = _non_negative_integer(llm, "reasoning_tokens")
    embedding_tokens = _non_negative_integer(embedding, "total_tokens")
    combined_tokens = _non_negative_integer(total, "total_tokens")
    combined_cached_tokens = _non_negative_integer(total, "cached_tokens")
    combined_reasoning_tokens = _non_negative_integer(total, "reasoning_tokens")
    if (
        prompt_tokens + completion_tokens != llm_total_tokens
        or llm_total_tokens + embedding_tokens != combined_tokens
        or cached_tokens != combined_cached_tokens
        or reasoning_tokens != combined_reasoning_tokens
        or cached_tokens > prompt_tokens
        or reasoning_tokens > completion_tokens
    ):
        raise ValueError("OpenViking indexing usage token equations do not close")
    return OpenVikingSessionIndexingUsage(
        attempt_id=attempt_id,
        raw_response_ref=raw_response_ref,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=llm_total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _non_negative_integer(value: Mapping[str, Any], field: str) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int) or item < 0:
        raise ValueError(f"OpenViking indexing usage {field} is invalid")
    return item


def _payload(raw_payloads: Mapping[str, bytes], reference: str | None) -> bytes:
    if reference is None or reference not in raw_payloads:
        raise ValueError("OpenViking session raw evidence is absent")
    return raw_payloads[reference]


def _object(payload: bytes) -> dict[str, Any]:
    try:
        document = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("OpenViking session evidence is not JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("OpenViking session evidence is not an object")
    return document


__all__ = [
    "OpenVikingSessionIndexingUsage",
    "OpenVikingSessionPlanEvidence",
    "reconstruct_openviking_session_indexing_usage",
    "reconstruct_openviking_session_candidates",
    "reconstruct_openviking_session_plan",
    "reconstruct_openviking_session_projection",
]
