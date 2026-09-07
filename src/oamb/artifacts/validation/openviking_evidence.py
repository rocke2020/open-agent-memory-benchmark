"""Strict reconstruction of OpenViking evidence from sealed raw payloads."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from oamb.contracts.evidence import AttemptRecordV2, CaseRecordV3, IngestionPlanRecordV2
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import NativeEvidenceCandidate
from oamb.contracts.specifications import IngestionPlanManifest
from oamb.contracts.states import AttemptOutcome
from oamb.memory_systems.openviking.adapter import (
    OPENVIKING_AUTH_MODE,
    OPENVIKING_USER_ROLE,
    OPENVIKING_VERSION,
    _canonical_score,
    _parse_attrs_response,
    _parse_batch_response,
    _parse_exact_object,
    _parse_find_response,
    _parse_not_found_response,
    _parse_standard_success,
    _require_exact_dict,
    _require_string_list,
)

OPENVIKING_PROFILE_ID = "openviking-rest-v1"


@dataclass(frozen=True, slots=True)
class OpenVikingRuntimeIdentity:
    account_id: str
    user_id: str


@dataclass(frozen=True, slots=True)
class OpenVikingScopeEvidence:
    actor_peer_id: str
    peer_root: str
    root_uri: str


@dataclass(frozen=True, slots=True)
class OpenVikingProjectionEvidence:
    state_sha256: str
    content_by_uri: tuple[tuple[str, str], ...]
    state_evidence_raw_ref: str

    def content(self, uri: str) -> str:
        for candidate_uri, content in self.content_by_uri:
            if candidate_uri == uri:
                return content
        raise ValueError("OpenViking projection does not contain the requested URI")


@dataclass(frozen=True, slots=True)
class OpenVikingPlanEvidence:
    actor_peer_id: str
    peer_root: str
    root_uri: str
    chunk_uris: tuple[str, ...]
    ordered_source_unit_ids: tuple[str, ...]
    source_payload_sha256: tuple[str, ...]
    projection: OpenVikingProjectionEvidence
    dispatch_raw_ref: str


def reconstruct_openviking_runtime_identity(
    raw_payloads: Mapping[str, bytes],
) -> OpenVikingRuntimeIdentity:
    identities: list[OpenVikingRuntimeIdentity] = []
    expected_fields = frozenset(
        {
            "status",
            "healthy",
            "version",
            "auth_mode",
            "account_id",
            "user_id",
            "role",
        }
    )
    for payload in raw_payloads.values():
        try:
            document = _parse_exact_object(payload, expected_fields, "health evidence")
        except ValueError:
            continue
        account_id = document.get("account_id")
        user_id = document.get("user_id")
        if (
            document.get("status") == "ok"
            and document.get("healthy") is True
            and document.get("version") == OPENVIKING_VERSION
            and document.get("auth_mode") == OPENVIKING_AUTH_MODE
            and document.get("role") == OPENVIKING_USER_ROLE
            and isinstance(account_id, str)
            and account_id
            and isinstance(user_id, str)
            and user_id
        ):
            identities.append(OpenVikingRuntimeIdentity(account_id, user_id))
    if len(identities) != 1:
        raise ValueError("OpenViking capsule requires one exact runtime identity payload")
    return identities[0]


def reconstruct_openviking_plan(
    *,
    raw_payloads: Mapping[str, bytes],
    plan: IngestionPlanRecordV2,
    manifest_plan: IngestionPlanManifest,
    attempts: Mapping[str, AttemptRecordV2],
    runtime_identity: OpenVikingRuntimeIdentity,
) -> OpenVikingPlanEvidence:
    scope = reconstruct_openviking_scope(
        raw_payloads=raw_payloads,
        plan=plan,
        runtime_identity=runtime_identity,
    )
    root_uri = scope.root_uri
    if plan.ingestion_plan_id != manifest_plan.ingestion_plan_id:
        raise ValueError("OpenViking plan identity differs from its manifest")
    source_hashes = manifest_plan.ordered_source_unit_bytes_sha256
    if len(source_hashes) != len(plan.ordered_source_unit_ids):
        raise ValueError("OpenViking source count does not match the sealed plan manifest")
    chunk_uris = tuple(
        f"{root_uri}/chunk-{ordinal:04d}.txt" for ordinal in range(1, len(source_hashes) + 1)
    )
    if (
        len(plan.ordered_dispatch_attempt_ids) != 1
        or len(plan.ordered_dispatch_source_unit_ids) != 1
        or plan.ordered_dispatch_source_unit_ids[0] != plan.ordered_source_unit_ids
    ):
        raise ValueError("OpenViking exact profile requires one ordered batch dispatch")
    dispatch_attempt_id = plan.ordered_dispatch_attempt_ids[0]
    dispatch_attempt = attempts.get(dispatch_attempt_id)
    skipped_dispatch = bool(plan.skipped_source_unit_ids)
    if (
        dispatch_attempt is None
        or dispatch_attempt.parent_kind != "ingestion_plan"
        or dispatch_attempt.parent_id != plan.ingestion_occurrence_id
        or dispatch_attempt.stage != "memory_ingest"
    ):
        raise ValueError("OpenViking dispatch attempt evidence is incomplete")
    if skipped_dispatch:
        if (
            dispatch_attempt.outcome != AttemptOutcome.FAILED
            or dispatch_attempt.raw_error_ref is None
        ):
            raise ValueError("OpenViking skipped batch lacks terminal failure evidence")
        dispatch_raw_ref = dispatch_attempt.raw_error_ref
        _required_raw(raw_payloads, dispatch_raw_ref)
    else:
        if (
            dispatch_attempt.outcome != AttemptOutcome.SUCCEEDED
            or dispatch_attempt.raw_response_ref is None
        ):
            raise ValueError("OpenViking dispatch attempt evidence is incomplete")
        dispatch_raw_ref = dispatch_attempt.raw_response_ref
        created, queue_ready = _parse_batch_response(
            _required_raw(raw_payloads, dispatch_raw_ref),
            expected_root=root_uri,
            expected_chunk_uris=chunk_uris,
        )
        if created != chunk_uris or not queue_ready:
            raise ValueError("OpenViking batch response does not close the source partition")
    if set(plan.accepted_source_unit_ids) | set(plan.rejected_source_unit_ids) | set(
        plan.skipped_source_unit_ids
    ) != set(plan.ordered_source_unit_ids):
        raise ValueError("OpenViking batch response does not close the source partition")

    projection = reconstruct_openviking_projection(
        raw_payloads=raw_payloads,
        references=plan.projection_raw_refs,
        root_uri=root_uri,
        chunk_uris=chunk_uris,
        source_payload_sha256=source_hashes,
    )
    if (
        plan.inventory_raw_ref != plan.projection_raw_refs[0]
        or not set(plan.projected_source_unit_ids)
        <= (set(plan.accepted_source_unit_ids) | set(plan.skipped_source_unit_ids))
        or plan.protected_state_sha256 != projection.state_sha256
        or plan.readiness_evidence_refs
        != (dispatch_raw_ref, plan.inventory_raw_ref, projection.state_evidence_raw_ref)
    ):
        raise ValueError("OpenViking readiness or projection binding is inconsistent")
    return OpenVikingPlanEvidence(
        actor_peer_id=scope.actor_peer_id,
        peer_root=scope.peer_root,
        root_uri=root_uri,
        chunk_uris=chunk_uris,
        ordered_source_unit_ids=plan.ordered_source_unit_ids,
        source_payload_sha256=source_hashes,
        projection=projection,
        dispatch_raw_ref=dispatch_raw_ref,
    )


def reconstruct_openviking_scope(
    *,
    raw_payloads: Mapping[str, bytes],
    plan: IngestionPlanRecordV2,
    runtime_identity: OpenVikingRuntimeIdentity,
) -> OpenVikingScopeEvidence:
    actor_peer_id = (
        "oamb-"
        + hashlib.sha256(b"peer\0" + plan.ingestion_occurrence_id.encode("utf-8")).hexdigest()
    )
    plan_key = hashlib.sha256(b"plan\0" + plan.ingestion_plan_id.encode("utf-8")).hexdigest()
    peer_root = f"viking://user/{runtime_identity.user_id}/peers/{actor_peer_id}"
    root_uri = f"{peer_root}/resources/oamb/mab65-v1/{plan_key}"
    if plan.scope_id != root_uri:
        raise ValueError("OpenViking scope does not match the sealed plan identity")
    if len(plan.scope_raw_refs) != 3:
        raise ValueError("OpenViking scope evidence inventory is not exact")
    mkdir_payload, peer_absence_payload, root_absence_payload = (
        _required_raw(raw_payloads, reference) for reference in plan.scope_raw_refs
    )
    mkdir_document = _parse_standard_success(mkdir_payload, "mkdir evidence")
    mkdir_result = _require_exact_dict(
        mkdir_document["result"], frozenset({"uri"}), "mkdir evidence.result"
    )
    if mkdir_result["uri"] != root_uri:
        raise ValueError("OpenViking mkdir evidence has the wrong root")
    _parse_not_found_response(peer_absence_payload, expected_uri=peer_root)
    _parse_not_found_response(root_absence_payload, expected_uri=root_uri)
    return OpenVikingScopeEvidence(
        actor_peer_id=actor_peer_id,
        peer_root=peer_root,
        root_uri=root_uri,
    )


def reconstruct_openviking_projection(
    *,
    raw_payloads: Mapping[str, bytes],
    references: tuple[str, ...],
    root_uri: str,
    chunk_uris: tuple[str, ...],
    source_payload_sha256: tuple[str, ...],
) -> OpenVikingProjectionEvidence:
    if len(references) < 2:
        raise ValueError("OpenViking projection evidence is incomplete")
    inventory_raw_ref, state_evidence_raw_ref, *supporting_refs = references
    inventory_document = _parse_standard_success(
        _required_raw(raw_payloads, inventory_raw_ref),
        "projection inventory evidence",
    )
    inventory_uris = tuple(
        sorted(
            (
                f"{root_uri}/.abstract.md",
                f"{root_uri}/.overview.md",
                *chunk_uris,
            )
        )
    )
    returned_uris = _require_string_list(
        inventory_document["result"], "projection inventory evidence.result"
    )
    if len(set(returned_uris)) != len(returned_uris) or frozenset(returned_uris) != frozenset(
        inventory_uris
    ):
        raise ValueError("OpenViking hidden inventory is incomplete")

    state_document = _parse_exact_object(
        _required_raw(raw_payloads, state_evidence_raw_ref),
        frozenset({"schema", "root_uri", "inventory_raw_reference", "state_sha256", "entries"}),
        "projection state evidence",
    )
    entries = state_document["entries"]
    expected_entry_uris = (root_uri, *inventory_uris)
    if (
        state_document["schema"] != "oamb-openviking-projection-evidence-v1"
        or state_document["root_uri"] != root_uri
        or state_document["inventory_raw_reference"] != inventory_raw_ref
        or not isinstance(entries, list)
        or len(entries) != len(expected_entry_uris)
    ):
        raise ValueError("OpenViking projection state envelope is invalid")

    state_entries: list[tuple[str, str | None, tuple[str, ...]]] = []
    content_by_uri: list[tuple[str, str]] = []
    for entry, expected_uri in zip(entries, expected_entry_uris, strict=True):
        entry_object = _require_exact_dict(
            entry,
            frozenset({"uri", "content", "tags", "attrs_raw_reference", "content_raw_reference"}),
            "projection state evidence.entries[]",
        )
        tags = _require_string_list(entry_object["tags"], "projection state evidence.tags")
        attrs_ref = entry_object["attrs_raw_reference"]
        if (
            entry_object["uri"] != expected_uri
            or not isinstance(attrs_ref, str)
            or _parse_attrs_response(
                _required_raw(raw_payloads, attrs_ref), expected_uri=expected_uri
            )
            != tags
        ):
            raise ValueError("OpenViking projected attrs do not match sealed raw evidence")
        content = entry_object["content"]
        content_ref = entry_object["content_raw_reference"]
        if expected_uri == root_uri:
            if content is not None or content_ref is not None:
                raise ValueError("OpenViking projection root must not contain file content")
        else:
            if not isinstance(content, str) or not isinstance(content_ref, str):
                raise ValueError("OpenViking projection entry lacks content evidence")
            content_document = _parse_standard_success(
                _required_raw(raw_payloads, content_ref),
                "projection content evidence",
            )
            if content_document["result"] != content:
                raise ValueError("OpenViking projection content does not match its raw response")
            content_by_uri.append((expected_uri, content))
        state_entries.append((expected_uri, content, tags))

    entry_attrs_refs = tuple(entry["attrs_raw_reference"] for entry in entries)
    entry_content_refs = tuple(
        entry["content_raw_reference"] for entry in entries if entry["content_raw_reference"]
    )
    ordered_supporting_refs = (*entry_attrs_refs, *entry_content_refs)
    if tuple(supporting_refs) != ordered_supporting_refs:
        raise ValueError("OpenViking projection supporting evidence order is invalid")

    content_lookup = dict(content_by_uri)
    for uri, payload_hash in zip(chunk_uris, source_payload_sha256, strict=True):
        if hashlib.sha256(content_lookup[uri].encode("utf-8")).hexdigest() != payload_hash:
            raise ValueError("OpenViking projected source bytes differ from the plan manifest")
    state_sha256 = canonical_sha256(
        ["oamb-openviking-projection-v1", root_uri, tuple(state_entries)]
    )
    if state_document["state_sha256"] != state_sha256:
        raise ValueError("OpenViking protected state digest does not match raw evidence")
    return OpenVikingProjectionEvidence(
        state_sha256=state_sha256,
        content_by_uri=tuple(content_by_uri),
        state_evidence_raw_ref=state_evidence_raw_ref,
    )


def reconstruct_openviking_candidates(
    *,
    raw_payloads: Mapping[str, bytes],
    case: CaseRecordV3,
    plan: OpenVikingPlanEvidence,
) -> tuple[NativeEvidenceCandidate, ...]:
    if case.retrieval_raw_ref is None:
        raise ValueError("OpenViking case has no retrieval response")
    hits = _parse_find_response(
        _required_raw(raw_payloads, case.retrieval_raw_ref),
        root_uri=plan.root_uri,
        chunk_uris=plan.chunk_uris,
    )
    l2_hits = tuple(hit for hit in hits if hit["level"] == 2)
    if len(l2_hits) != len(case.retrieval_supporting_raw_refs):
        raise ValueError("OpenViking L2 hydration evidence inventory is incomplete")
    hydration_by_uri: dict[str, str] = {}
    for hit, reference in zip(l2_hits, case.retrieval_supporting_raw_refs, strict=True):
        hydration = _parse_standard_success(
            _required_raw(raw_payloads, reference),
            "retrieval L2 hydration evidence",
        )["result"]
        if not isinstance(hydration, str) or hydration != plan.projection.content(hit["uri"]):
            raise ValueError("OpenViking L2 hydration differs from the protected projection")
        hydration_by_uri[hit["uri"]] = hydration

    candidates: list[NativeEvidenceCandidate] = []
    source_by_uri = dict(zip(plan.chunk_uris, plan.ordered_source_unit_ids, strict=True))
    for rank, hit in enumerate(hits, start=1):
        uri = hit["uri"]
        level = hit["level"]
        if level == 0:
            content = hit["abstract"]
            source_unit_id = None
            evidence_kind = "native_abstract"
        elif level == 1:
            content = hit["abstract"]
            source_unit_id = None
            evidence_kind = "native_overview"
        else:
            content = hydration_by_uri[uri]
            source_unit_id = source_by_uri[uri]
            evidence_kind = "source_content"
        native_id = f"{uri}#level={level}"
        candidates.append(
            NativeEvidenceCandidate(
                native_id=native_id,
                native_rank_1_indexed=rank,
                content=content,
                native_score=_canonical_score(hit["score"]),
                provider_evidence_identity=native_id,
                source_unit_id=source_unit_id,
                evidence_kind=evidence_kind,
                native_reference=uri,
                native_truncated=False,
            )
        )
    return tuple(candidates)


def _required_raw(raw_payloads: Mapping[str, bytes], reference: str) -> bytes:
    payload = raw_payloads.get(reference)
    if payload is None:
        raise ValueError("OpenViking raw evidence reference is missing")
    return payload
