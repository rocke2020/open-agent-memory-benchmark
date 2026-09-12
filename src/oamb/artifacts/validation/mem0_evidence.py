"""Source-reconstructable evidence helpers for Mem0 REST black-box capsules."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

from oamb.contracts.evidence import AttemptRecordV2, CaseRecordV3, IngestionPlanRecordV3
from oamb.contracts.ports import NativeEvidenceCandidate
from oamb.contracts.states import AttemptOutcome
from oamb.memory_systems.mem0.projection import (
    Mem0Projection,
    parse_projection_pages,
    projection_source_unit_ids,
    projection_state_sha256,
)
from oamb.memory_systems.mem0.wire import parse_add_response, parse_search_response
from oamb.memory_systems.rest import parse_exact_json_object

MEM0_PROFILE_ID = "mem0-rest-v1"
MEM0_COLLECTION = "oamb_memories"

_PROJECTION_PAGE_FIELDS = frozenset({"collection", "run_id", "count", "points", "next_cursor"})
_PROJECTION_RECEIPT_FIELDS = frozenset(
    {
        "schema",
        "collection",
        "run_id",
        "ordered_source_unit_ids",
        "state_sha256",
        "capture_sequence",
        "page_raw_refs",
    }
)
_PAIR_ADD_RECEIPT_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "source_unit_id",
        "ordered_response_sha256",
    }
)


@dataclass(frozen=True, slots=True)
class Mem0ProjectionEvidence:
    projection: Mem0Projection
    ordered_source_unit_ids: tuple[str, ...]
    state_sha256: str
    capture_sequence: int
    summary_raw_ref: str
    page_raw_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Mem0PlanEvidence:
    plan: IngestionPlanRecordV3
    projection: Mem0ProjectionEvidence


def reconstruct_mem0_projection(
    *,
    raw_payloads: dict[str, bytes],
    references: tuple[str, ...],
    expected_run_id: str,
) -> Mem0ProjectionEvidence:
    unique_references = tuple(dict.fromkeys(references))
    summary_ref: str | None = None
    summary: dict[str, object] | None = None
    page_refs: list[str] = []
    page_payloads: list[bytes] = []
    for reference in unique_references:
        payload = raw_payloads.get(reference)
        if payload is None:
            raise ValueError("Mem0 projection raw reference is missing")
        document = _parse_unique_json(payload)
        if not isinstance(document, dict):
            continue
        fields = frozenset(document)
        if fields == _PROJECTION_RECEIPT_FIELDS:
            if summary is not None:
                raise ValueError("Mem0 projection has multiple summary receipts")
            summary_ref = reference
            summary = document
        elif fields == _PROJECTION_PAGE_FIELDS:
            page_refs.append(reference)
            page_payloads.append(payload)
    if summary is None or summary_ref is None or not page_payloads:
        raise ValueError("Mem0 projection evidence is incomplete")
    projection = parse_projection_pages(
        tuple(page_payloads),
        expected_collection=MEM0_COLLECTION,
        expected_run_id=expected_run_id,
    )
    source_ids = projection_source_unit_ids(projection)
    state_sha256 = projection_state_sha256(projection)
    capture_sequence = summary.get("capture_sequence")
    if (
        summary.get("schema") != "oamb-mem0-projection-receipt-v1"
        or summary.get("collection") != MEM0_COLLECTION
        or summary.get("run_id") != expected_run_id
        or summary.get("ordered_source_unit_ids") != list(source_ids)
        or summary.get("state_sha256") != state_sha256
        or not isinstance(capture_sequence, int)
        or isinstance(capture_sequence, bool)
        or capture_sequence < 1
        or summary.get("page_raw_refs") != page_refs
    ):
        raise ValueError("Mem0 projection summary does not bind its raw pages")
    return Mem0ProjectionEvidence(
        projection=projection,
        ordered_source_unit_ids=source_ids,
        state_sha256=state_sha256,
        capture_sequence=capture_sequence,
        summary_raw_ref=summary_ref,
        page_raw_refs=tuple(page_refs),
    )


def reconstruct_mem0_projection_snapshots(
    *,
    raw_payloads: dict[str, bytes],
    references: tuple[str, ...],
    expected_run_id: str,
) -> tuple[Mem0ProjectionEvidence, ...]:
    reference_set = set(references)
    summary_refs: list[str] = []
    for reference in references:
        if reference in summary_refs:
            continue
        payload = raw_payloads.get(reference)
        if payload is None:
            raise ValueError("Mem0 readiness raw reference is missing")
        document = _parse_unique_json(payload)
        if isinstance(document, dict) and frozenset(document) == _PROJECTION_RECEIPT_FIELDS:
            page_refs = document.get("page_raw_refs")
            if (
                not isinstance(page_refs, list)
                or any(not isinstance(item, str) for item in page_refs)
                or not set(page_refs) <= reference_set
            ):
                raise ValueError("Mem0 readiness summary pages are not evidence-bound")
            summary_refs.append(reference)
    return tuple(
        reconstruct_mem0_projection(
            raw_payloads=raw_payloads,
            references=(summary_ref, *_summary_page_refs(raw_payloads, summary_ref)),
            expected_run_id=expected_run_id,
        )
        for summary_ref in summary_refs
    )


def reconstruct_mem0_plan(
    *,
    raw_payloads: dict[str, bytes],
    plan: IngestionPlanRecordV3,
    attempts: dict[str, AttemptRecordV2],
) -> Mem0PlanEvidence:
    if (
        plan.adapter_profile_id != MEM0_PROFILE_ID
        or plan.memory_system_id != "mem0"
        or plan.scope_id != plan.ingestion_occurrence_id
    ):
        raise ValueError("Mem0 plan identity does not match the black-box profile")
    absence = reconstruct_mem0_projection(
        raw_payloads=raw_payloads,
        references=plan.scope_raw_refs,
        expected_run_id=plan.ingestion_occurrence_id,
    )
    if absence.projection.points or absence.ordered_source_unit_ids:
        raise ValueError("Mem0 scope was not empty before add dispatch")

    dispatched: list[str] = []
    add_raw_refs: list[str] = []
    for attempt_id, source_ids in zip(
        plan.ordered_dispatch_attempt_ids,
        plan.ordered_dispatch_source_unit_ids,
        strict=True,
    ):
        attempt = attempts.get(attempt_id)
        skipped_dispatch = set(source_ids) <= set(plan.skipped_source_unit_ids)
        if (
            attempt is None
            or attempt.parent_kind != "ingestion_plan"
            or attempt.parent_id != plan.ingestion_occurrence_id
            or attempt.stage != "memory_ingest"
            or len(source_ids) != 1
        ):
            raise ValueError("Mem0 add attempt does not close its source dispatch")
        if skipped_dispatch:
            if (
                attempt.outcome != AttemptOutcome.FAILED
                or attempt.raw_error_ref is None
                or attempt.raw_error_ref not in raw_payloads
            ):
                raise ValueError("Mem0 skipped add lacks its terminal failure evidence")
            dispatched.extend(source_ids)
            continue
        if attempt.outcome != AttemptOutcome.SUCCEEDED or attempt.raw_response_ref is None:
            raise ValueError("Mem0 add attempt does not close its source dispatch")
        payload = raw_payloads.get(attempt.raw_response_ref)
        if payload is None:
            raise ValueError("Mem0 add raw receipt is missing")
        _validate_add_receipt(
            payload,
            raw_payloads=raw_payloads,
            expected_source_unit_id=source_ids[0],
        )
        dispatched.extend(source_ids)
        add_raw_refs.append(attempt.raw_response_ref)
    if (
        tuple(dispatched) != plan.ordered_source_unit_ids
        or set(plan.accepted_source_unit_ids)
        | set(plan.rejected_source_unit_ids)
        | set(plan.skipped_source_unit_ids)
        != set(plan.ordered_source_unit_ids)
        or not set(add_raw_refs) <= set(plan.readiness_evidence_refs)
    ):
        raise ValueError("Mem0 completed source ledger does not close")

    readiness_snapshots = reconstruct_mem0_projection_snapshots(
        raw_payloads=raw_payloads,
        references=plan.readiness_evidence_refs,
        expected_run_id=plan.ingestion_occurrence_id,
    )
    if len(readiness_snapshots) < 2:
        raise ValueError("Mem0 readiness lacks two projection observations")
    readiness_before, readiness_after = readiness_snapshots[-2:]
    if (
        readiness_after.capture_sequence != readiness_before.capture_sequence + 1
        or readiness_before.ordered_source_unit_ids != readiness_after.ordered_source_unit_ids
        or readiness_before.state_sha256 != readiness_after.state_sha256
    ):
        raise ValueError("Mem0 readiness projections are not consecutive and stable")

    projection = reconstruct_mem0_projection(
        raw_payloads=raw_payloads,
        references=plan.projection_raw_refs,
        expected_run_id=plan.ingestion_occurrence_id,
    )
    observable = set(plan.accepted_source_unit_ids) | set(plan.skipped_source_unit_ids)
    source_positions = {
        source_id: index for index, source_id in enumerate(plan.ordered_source_unit_ids)
    }
    for point in projection.projection.points:
        metadata = point.metadata
        if metadata is None:
            continue
        if (
            metadata.ingestion_occurrence_id != plan.ingestion_occurrence_id
            or metadata.ingestion_plan_id != plan.ingestion_plan_id
            or metadata.source_unit_id not in observable
            or metadata.source_ordinal != source_positions[metadata.source_unit_id] + 1
        ):
            raise ValueError("Mem0 visible projection contains foreign source metadata")
    if (
        projection.ordered_source_unit_ids != plan.projected_source_unit_ids
        or projection.state_sha256 != plan.protected_state_sha256
        or projection.capture_sequence != readiness_after.capture_sequence + 1
        or plan.inventory_raw_ref != projection.summary_raw_ref
        or readiness_after.ordered_source_unit_ids != projection.ordered_source_unit_ids
        or readiness_after.state_sha256 != projection.state_sha256
    ):
        raise ValueError("Mem0 visible projection does not bind the plan")
    return Mem0PlanEvidence(plan=plan, projection=projection)


def reconstruct_mem0_candidates(
    *,
    raw_payloads: dict[str, bytes],
    case: CaseRecordV3,
    plan: Mem0PlanEvidence,
) -> tuple[NativeEvidenceCandidate, ...]:
    if case.retrieval_raw_ref is None:
        raise ValueError("Mem0 case has no raw search response")
    payload = raw_payloads.get(case.retrieval_raw_ref)
    if payload is None:
        raise ValueError("Mem0 search raw response is missing")
    items = parse_search_response(
        payload,
        expected_run_id=plan.plan.ingestion_occurrence_id,
    )
    points = {point.native_id: point for point in plan.projection.projection.points}
    candidates: list[NativeEvidenceCandidate] = []
    for item in items:
        point = points.get(item.native_id)
        if (
            point is None
            or item.memory != point.memory
            or item.run_id != point.run_id
            or item.metadata != point.metadata
            or (item.memory_hash is not None and item.memory_hash != point.memory_hash)
            or item.attributed_to != point.attributed_to
            or item.created_at != point.created_at
            or item.updated_at != point.updated_at
        ):
            raise ValueError("Mem0 search candidate is outside the sealed projection")
        candidates.append(
            NativeEvidenceCandidate(
                native_id=item.native_id,
                native_rank_1_indexed=item.native_rank_1_indexed,
                content=item.memory,
                native_score=_canonical_score(item.native_score),
                provider_evidence_identity=item.native_id,
                source_unit_id=(None if item.metadata is None else item.metadata.source_unit_id),
                evidence_kind="native_memory",
                native_reference=item.native_id,
                native_truncated=False,
            )
        )
    return tuple(candidates)


def _canonical_score(value: float) -> str:
    if not math.isfinite(value):
        raise ValueError("Mem0 native score must be finite")
    return str(value)


def _validate_add_receipt(
    payload: bytes,
    *,
    raw_payloads: dict[str, bytes],
    expected_source_unit_id: str,
) -> None:
    document = _parse_unique_json(payload)
    if not isinstance(document, dict) or document.get("schema_name") != (
        "oamb_mem0_pair_add_receipt"
    ):
        parse_add_response(payload)
        return
    if frozenset(document) != _PAIR_ADD_RECEIPT_FIELDS:
        raise ValueError("Mem0 pair-add receipt fields are invalid")
    response_refs = document["ordered_response_sha256"]
    if (
        document["schema_version"] != 1
        or document["source_unit_id"] != expected_source_unit_id
        or not isinstance(response_refs, list)
        or len(response_refs) < 2
        or any(not isinstance(reference, str) for reference in response_refs)
    ):
        raise ValueError("Mem0 pair-add receipt identity is invalid")
    for reference in response_refs:
        response = raw_payloads.get(reference)
        if response is None or hashlib.sha256(response).hexdigest() != reference:
            raise ValueError("Mem0 pair-add response is missing or corrupt")
        parse_add_response(response)


def _summary_page_refs(raw_payloads: dict[str, bytes], summary_ref: str) -> tuple[str, ...]:
    payload = raw_payloads.get(summary_ref)
    if payload is None:
        raise ValueError("Mem0 projection summary is missing")
    document = parse_exact_json_object(
        payload,
        expected_fields=_PROJECTION_RECEIPT_FIELDS,
    )
    page_refs = document["page_raw_refs"]
    if not isinstance(page_refs, list) or any(not isinstance(item, str) for item in page_refs):
        raise ValueError("Mem0 projection summary page references are invalid")
    return tuple(page_refs)


def _parse_unique_json(payload: bytes) -> object:
    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        return json.loads(payload, object_pairs_hook=unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Mem0 projection evidence is not JSON") from exc


__all__ = [
    "MEM0_PROFILE_ID",
    "Mem0PlanEvidence",
    "Mem0ProjectionEvidence",
    "reconstruct_mem0_candidates",
    "reconstruct_mem0_plan",
    "reconstruct_mem0_projection",
    "reconstruct_mem0_projection_snapshots",
]
