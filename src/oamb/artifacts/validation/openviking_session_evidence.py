"""Strict reconstruction for the OpenViking LME session profile."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from oamb.contracts.evidence import CaseRecordV3, IngestionPlanRecordV2
from oamb.contracts.ids import canonical_sha256
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
        plan.ordered_source_unit_ids != plan.accepted_source_unit_ids
        or plan.rejected_source_unit_ids
        or plan.projected_source_unit_ids != plan.ordered_source_unit_ids
        or len(plan.ordered_dispatch_attempt_ids) != len(plan.ordered_source_unit_ids)
    ):
        raise ValueError("OpenViking session source ledger is not exact and ordered")
    for attempt_id in plan.ordered_dispatch_attempt_ids:
        attempt = attempts.get(attempt_id)
        if (
            attempt is None
            or attempt.stage != "memory_ingest"
            or attempt.outcome != AttemptOutcome.SUCCEEDED
            or attempt.raw_response_ref not in plan.readiness_evidence_refs
        ):
            raise ValueError("OpenViking session dispatch attempt is incomplete")
    expected_sessions = tuple(_session_id(source_id) for source_id in plan.ordered_source_unit_ids)
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


def _session_id(source_unit_id: str) -> str:
    return "oamb-" + hashlib.sha256(b"session\0" + source_unit_id.encode("utf-8")).hexdigest()


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
    "OpenVikingSessionPlanEvidence",
    "reconstruct_openviking_session_candidates",
    "reconstruct_openviking_session_plan",
    "reconstruct_openviking_session_projection",
]
