"""Exact Mem0 reference-profile identities and fail-closed support verdicts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import MemorySystemProfileUnsupported
from oamb.memory_systems.rest import parse_exact_json_object

MEM0_RELEASE_VERSION = "2.0.19"
MEM0_SOURCE_REVISION = "dc82354e143c2581d505d581a00286d6ef8c3605"
MEM0_SOURCE_ARCHIVE_SHA256 = "5443d9dd99196e33fdde31ef663518c022a8f4a86ef032e7b4bd7b00285705c2"


class Mem0TransportKind(StrEnum):
    REST_API = "rest_api"
    PYTHON_SDK = "python_sdk"


@dataclass(frozen=True, slots=True)
class Mem0ExactProfile:
    profile_id: str
    transport_kind: Mem0TransportKind
    release_version: str
    source_revision: str
    source_archive_sha256: str
    is_default_comparison_transport: bool


@dataclass(frozen=True, slots=True)
class Mem0OperationAuditEvent:
    event_id: str
    sequence: int
    operation_kind: str
    unsupported_profile_id: str
    unsupported_reason_codes: tuple[str, ...]
    dispatcher_count_before: int
    dispatcher_count_after: int


@dataclass(frozen=True, slots=True)
class Mem0ZeroDispatchAudit:
    audit_id: str
    profile: Mem0ExactProfile
    unsupported_profile_id: str
    unsupported_reason_codes: tuple[str, ...]
    ordered_events: tuple[Mem0OperationAuditEvent, ...]
    producer_attestation: object = field(repr=False, compare=False)

    @property
    def rejected_operation_count(self) -> int:
        return len(self.ordered_events)

    @property
    def dispatched_operation_count(self) -> int:
        return sum(
            event.dispatcher_count_after - event.dispatcher_count_before
            for event in self.ordered_events
        )


_MEM0_ADAPTER_AUDIT_PRODUCER = object()


MEM0_REST_PROFILE = Mem0ExactProfile(
    profile_id="mem0-rest-v1",
    transport_kind=Mem0TransportKind.REST_API,
    release_version=MEM0_RELEASE_VERSION,
    source_revision=MEM0_SOURCE_REVISION,
    source_archive_sha256=MEM0_SOURCE_ARCHIVE_SHA256,
    is_default_comparison_transport=True,
)

MEM0_SDK_PROFILE = Mem0ExactProfile(
    profile_id="mem0-sdk-v1",
    transport_kind=Mem0TransportKind.PYTHON_SDK,
    release_version=MEM0_RELEASE_VERSION,
    source_revision=MEM0_SOURCE_REVISION,
    source_archive_sha256=MEM0_SOURCE_ARCHIVE_SHA256,
    is_default_comparison_transport=False,
)

_V2_0_19_UNSUPPORTED_REASONS = (
    "implicit_entity_store",
    "implicit_bm25_scoring",
    "implicit_entity_boost",
    "incomplete_main_entity_history_projection",
    "swallowed_internal_errors",
    "audited_empty_unproven",
)
MEM0_ZERO_DISPATCH_OPERATIONS = (
    "resolve",
    "allocate_ingestion_scope",
    "plan_ingestion",
    "ingest",
    "wait_ready",
    "inventory",
    "state_digest",
    "project",
    "retrieve",
)

_PROFILE_VERDICT_FIELDS = frozenset(
    {
        "profile_id",
        "transport_kind",
        "release_version",
        "source_revision",
        "source_archive_sha256",
        "configuration",
        "audit",
        "unsupported_reason_codes",
    }
)
_CONFIGURATION_FIELDS = frozenset(
    {
        "api_version",
        "vector_store_provider",
        "collection_name",
        "embedding_model",
        "embedding_dimension",
        "reranker",
    }
)
_AUDIT_FIELDS = frozenset(
    {
        "entity_store_control",
        "bm25_scoring_control",
        "entity_boost_control",
        "protected_projection",
        "internal_error_propagation",
        "audited_empty_response",
    }
)
_REFERENCE_CONFIGURATION = {
    "api_version": "v1.1",
    "vector_store_provider": "qdrant",
    "collection_name": "oamb_memories",
    "embedding_model": "qwen3-embedding:0.6b",
    "embedding_dimension": 1024,
    "reranker": None,
}
_REFERENCE_AUDIT = {
    "entity_store_control": "implicit",
    "bm25_scoring_control": "implicit",
    "entity_boost_control": "implicit",
    "protected_projection": "main_only_incomplete_entity_history",
    "internal_error_propagation": "swallowed",
    "audited_empty_response": False,
}


@dataclass(frozen=True, slots=True)
class Mem0ReferenceProfileVerdict:
    profile: Mem0ExactProfile
    api_version: str
    vector_store_provider: str
    collection_name: str
    embedding_model: str
    embedding_dimension: int
    reranker: None
    entity_store_control: str
    bm25_scoring_control: str
    entity_boost_control: str
    protected_projection: str
    internal_error_propagation: str
    audited_empty_response: bool
    unsupported_reason_codes: tuple[str, ...]


def reference_profile_unsupported(profile: Mem0ExactProfile) -> MemorySystemProfileUnsupported:
    if profile not in {MEM0_REST_PROFILE, MEM0_SDK_PROFILE}:
        raise ValueError("unknown Mem0 exact reference profile")
    return MemorySystemProfileUnsupported(
        profile.profile_id,
        reason_codes=_V2_0_19_UNSUPPORTED_REASONS,
    )


def build_mem0_operation_audit_event(
    *,
    profile: Mem0ExactProfile,
    sequence: int,
    operation_kind: str,
    dispatcher_count_before: int,
    dispatcher_count_after: int,
) -> Mem0OperationAuditEvent:
    if operation_kind not in MEM0_ZERO_DISPATCH_OPERATIONS:
        raise ValueError("unknown Mem0 zero-dispatch operation")
    unsupported = reference_profile_unsupported(profile)
    fields = {
        "sequence": sequence,
        "operation_kind": operation_kind,
        "unsupported_profile_id": unsupported.profile_id,
        "unsupported_reason_codes": unsupported.reason_codes,
        "dispatcher_count_before": dispatcher_count_before,
        "dispatcher_count_after": dispatcher_count_after,
    }
    return Mem0OperationAuditEvent(
        event_id=canonical_sha256(["oamb-mem0-operation-audit-event-v1", fields]),
        sequence=sequence,
        operation_kind=operation_kind,
        unsupported_profile_id=unsupported.profile_id,
        unsupported_reason_codes=unsupported.reason_codes,
        dispatcher_count_before=dispatcher_count_before,
        dispatcher_count_after=dispatcher_count_after,
    )


def _seal_mem0_zero_dispatch_audit(
    *,
    profile: Mem0ExactProfile,
    ordered_events: tuple[Mem0OperationAuditEvent, ...],
) -> Mem0ZeroDispatchAudit:
    unsupported = reference_profile_unsupported(profile)
    audit_id = _mem0_zero_dispatch_audit_id(
        profile=profile,
        unsupported_profile_id=unsupported.profile_id,
        unsupported_reason_codes=unsupported.reason_codes,
        ordered_events=ordered_events,
    )
    return Mem0ZeroDispatchAudit(
        audit_id=audit_id,
        profile=profile,
        unsupported_profile_id=unsupported.profile_id,
        unsupported_reason_codes=unsupported.reason_codes,
        ordered_events=ordered_events,
        producer_attestation=_MEM0_ADAPTER_AUDIT_PRODUCER,
    )


def mem0_zero_dispatch_audit_is_valid(audit: Mem0ZeroDispatchAudit) -> bool:
    if audit.producer_attestation is not _MEM0_ADAPTER_AUDIT_PRODUCER:
        return False
    try:
        unsupported = reference_profile_unsupported(audit.profile)
    except ValueError:
        return False
    if (
        audit.unsupported_profile_id != unsupported.profile_id
        or audit.unsupported_reason_codes != unsupported.reason_codes
        or not audit.ordered_events
    ):
        return False
    for sequence, event in enumerate(audit.ordered_events, 1):
        try:
            expected = build_mem0_operation_audit_event(
                profile=audit.profile,
                sequence=sequence,
                operation_kind=event.operation_kind,
                dispatcher_count_before=event.dispatcher_count_before,
                dispatcher_count_after=event.dispatcher_count_after,
            )
        except ValueError:
            return False
        if (
            event != expected
            or event.dispatcher_count_before != 0
            or event.dispatcher_count_after != 0
        ):
            return False
    return audit.audit_id == _mem0_zero_dispatch_audit_id(
        profile=audit.profile,
        unsupported_profile_id=audit.unsupported_profile_id,
        unsupported_reason_codes=audit.unsupported_reason_codes,
        ordered_events=audit.ordered_events,
    )


def mem0_zero_dispatch_audit_binding(audit: Mem0ZeroDispatchAudit) -> tuple[object, ...]:
    return (
        audit.audit_id,
        audit.profile.profile_id,
        audit.profile.transport_kind,
        audit.profile.release_version,
        audit.profile.source_revision,
        audit.profile.source_archive_sha256,
        audit.unsupported_profile_id,
        audit.unsupported_reason_codes,
        tuple(
            (
                event.event_id,
                event.sequence,
                event.operation_kind,
                event.unsupported_profile_id,
                event.unsupported_reason_codes,
                event.dispatcher_count_before,
                event.dispatcher_count_after,
            )
            for event in audit.ordered_events
        ),
    )


def _mem0_zero_dispatch_audit_id(
    *,
    profile: Mem0ExactProfile,
    unsupported_profile_id: str,
    unsupported_reason_codes: tuple[str, ...],
    ordered_events: tuple[Mem0OperationAuditEvent, ...],
) -> str:
    return canonical_sha256(
        [
            "oamb-mem0-zero-dispatch-audit-v1",
            profile.profile_id,
            profile.transport_kind,
            profile.release_version,
            profile.source_revision,
            profile.source_archive_sha256,
            unsupported_profile_id,
            unsupported_reason_codes,
            tuple(
                (
                    event.event_id,
                    event.sequence,
                    event.operation_kind,
                    event.unsupported_profile_id,
                    event.unsupported_reason_codes,
                    event.dispatcher_count_before,
                    event.dispatcher_count_after,
                )
                for event in ordered_events
            ),
        ]
    )


def parse_reference_profile_verdict(
    raw_bytes: bytes,
    *,
    expected_profile: Mem0ExactProfile,
) -> Mem0ReferenceProfileVerdict:
    if expected_profile not in {MEM0_REST_PROFILE, MEM0_SDK_PROFILE}:
        raise ValueError("unknown Mem0 exact reference profile")
    document = parse_exact_json_object(raw_bytes, expected_fields=_PROFILE_VERDICT_FIELDS)
    configuration = _require_exact_object(
        document["configuration"],
        expected_fields=_CONFIGURATION_FIELDS,
        description="configuration",
    )
    audit = _require_exact_object(
        document["audit"],
        expected_fields=_AUDIT_FIELDS,
        description="audit",
    )
    expected_identity = {
        "profile_id": expected_profile.profile_id,
        "transport_kind": expected_profile.transport_kind.value,
        "release_version": expected_profile.release_version,
        "source_revision": expected_profile.source_revision,
        "source_archive_sha256": expected_profile.source_archive_sha256,
    }
    for field_name, expected_value in expected_identity.items():
        if document[field_name] != expected_value:
            raise ValueError(f"profile verdict {field_name} does not match the exact profile")
    if configuration != _REFERENCE_CONFIGURATION:
        raise ValueError("configuration values do not match the exact profile")
    if audit != _REFERENCE_AUDIT:
        raise ValueError("audit values do not match the exact profile")
    raw_reasons = document["unsupported_reason_codes"]
    if not isinstance(raw_reasons, list) or tuple(raw_reasons) != _V2_0_19_UNSUPPORTED_REASONS:
        raise ValueError("unsupported reason codes do not match the exact profile")
    return Mem0ReferenceProfileVerdict(
        profile=expected_profile,
        api_version="v1.1",
        vector_store_provider="qdrant",
        collection_name="oamb_memories",
        embedding_model="qwen3-embedding:0.6b",
        embedding_dimension=1024,
        reranker=None,
        entity_store_control="implicit",
        bm25_scoring_control="implicit",
        entity_boost_control="implicit",
        protected_projection="main_only_incomplete_entity_history",
        internal_error_propagation="swallowed",
        audited_empty_response=False,
        unsupported_reason_codes=_V2_0_19_UNSUPPORTED_REASONS,
    )


def _require_exact_object(
    value: object,
    *,
    expected_fields: frozenset[str],
    description: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object")
    fields = frozenset(value)
    if fields != expected_fields:
        raise ValueError(
            f"{description} fields do not match the exact profile: "
            f"expected {sorted(expected_fields)}, got {sorted(fields)}"
        )
    return value


__all__ = [
    "MEM0_RELEASE_VERSION",
    "MEM0_REST_PROFILE",
    "MEM0_SDK_PROFILE",
    "MEM0_SOURCE_ARCHIVE_SHA256",
    "MEM0_SOURCE_REVISION",
    "MEM0_ZERO_DISPATCH_OPERATIONS",
    "Mem0ExactProfile",
    "Mem0OperationAuditEvent",
    "Mem0ReferenceProfileVerdict",
    "Mem0TransportKind",
    "Mem0ZeroDispatchAudit",
    "build_mem0_operation_audit_event",
    "mem0_zero_dispatch_audit_binding",
    "mem0_zero_dispatch_audit_is_valid",
    "parse_reference_profile_verdict",
    "reference_profile_unsupported",
]
