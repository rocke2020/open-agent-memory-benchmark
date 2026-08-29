"""Production semantic rules for exact native adapter profiles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oamb.artifacts.validation.hindsight_evidence import (
    HindsightProjectionEvidence,
    reconstruct_hindsight_projection,
)
from oamb.artifacts.validation.mem0_evidence import (
    MEM0_PROFILE_ID,
    reconstruct_mem0_candidates,
    reconstruct_mem0_plan,
    reconstruct_mem0_projection,
)
from oamb.artifacts.validation.native import (
    _contracts,
    _ingestion_plans,
    _load_native_capsule,
    _manifest_schema_rule,
    _NativeCapsuleSnapshot,
    _raw_closure_rule,
)
from oamb.artifacts.validation.openviking_evidence import (
    OpenVikingPlanEvidence,
    reconstruct_openviking_candidates,
    reconstruct_openviking_plan,
    reconstruct_openviking_projection,
    reconstruct_openviking_runtime_identity,
    reconstruct_openviking_scope,
)
from oamb.artifacts.validation.registry import ValidationRule
from oamb.contracts.evidence import (
    AttemptRecordV2,
    CaseRecordV3,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
    RunRecord,
    ValidationIssue,
    ValidationSeverity,
)
from oamb.contracts.specifications import CaseManifest
from oamb.contracts.states import AttemptOutcome, CaseState, IngestionPlanState, RunState
from oamb.memory_systems.hindsight.normalize import normalize_recall
from oamb.memory_systems.hindsight.profiles import (
    MEMORY_SYSTEM_ID as HINDSIGHT_MEMORY_SYSTEM_ID,
)
from oamb.memory_systems.hindsight.profiles import (
    PROFILE_ID as HINDSIGHT_PROFILE_ID,
)
from oamb.memory_systems.hindsight.profiles import (
    parse_bank_config,
    parse_bank_profile,
    parse_retain_response,
    parse_version_response,
)
from oamb.memory_systems.mem0.profiles import (
    MEM0_REST_PROFILE,
    MEM0_SDK_PROFILE,
    Mem0ExactProfile,
    Mem0ZeroDispatchAudit,
    mem0_zero_dispatch_audit_is_valid,
    reference_profile_unsupported,
)
from oamb.memory_systems.openviking.adapter import OPENVIKING_MEMORY_SYSTEM_ID

OPENVIKING_PROFILE_ID = "openviking-rest-v1"


@dataclass(frozen=True, slots=True)
class Mem0AdapterValidationInput:
    adapter: object
    audit: Mem0ZeroDispatchAudit

    @property
    def profile(self) -> Mem0ExactProfile:
        return self.audit.profile

    @property
    def unsupported_profile_id(self) -> str:
        return self.audit.unsupported_profile_id

    @property
    def unsupported_reason_codes(self) -> tuple[str, ...]:
        return self.audit.unsupported_reason_codes

    @property
    def rejected_operation_count(self) -> int:
        return self.audit.rejected_operation_count

    @property
    def dispatched_operation_count(self) -> int:
        return self.audit.dispatched_operation_count

    @property
    def comparison_eligible(self) -> bool:
        return self.audit.profile.is_default_comparison_transport


def mem0_adapter_validation_input(adapter: object) -> Mem0AdapterValidationInput:
    validation_audit = getattr(adapter, "validation_audit", None)
    if not callable(validation_audit):
        raise TypeError("Mem0 validation input requires an adapter audit producer")
    audit = validation_audit()
    if not isinstance(audit, Mem0ZeroDispatchAudit):
        raise TypeError("Mem0 adapter returned an invalid audit type")
    return Mem0AdapterValidationInput(adapter=adapter, audit=audit)


def _native_snapshot(target: Any) -> _NativeCapsuleSnapshot | None:
    if not isinstance(target, Path):
        return None
    return _load_native_capsule(target)


def _native_structural_closes(snapshot: Any) -> bool:
    return not _manifest_schema_rule(snapshot) and not _raw_closure_rule(snapshot)


def _hindsight_runtime_profile_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.hindsight.runtime-profile.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    plans = _contracts(snapshot, IngestionPlanRecordV2)
    runs = _contracts(snapshot, RunRecord)
    version_closed = any(
        _accepts(parse_version_response, payload) for payload in snapshot.raw_payloads.values()
    )
    exact = bool(
        _native_structural_closes(snapshot)
        and plans
        and all(
            plan.adapter_profile_id == HINDSIGHT_PROFILE_ID
            and plan.memory_system_id == HINDSIGHT_MEMORY_SYSTEM_ID
            for plan in plans
        )
        and len(runs) == 1
        and runs[0].state == RunState.FINALIZED
        and version_closed
    )
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "hindsight-runtime-profile-drift"),)


def _hindsight_scope_dispatch_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.hindsight.scope-dispatch.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    exact = bool(_contracts(snapshot, IngestionPlanRecordV2))
    for plan in _contracts(snapshot, IngestionPlanRecordV2):
        expected_scope_id = hashlib.sha256(
            f"oamb-hindsight-bank-v1\0{plan.ingestion_occurrence_id}".encode()
        ).hexdigest()
        scope_payloads = tuple(
            snapshot.raw_payloads.get(reference) for reference in plan.scope_raw_refs
        )
        scope_closed = bool(
            plan.state == IngestionPlanState.SEALED
            and plan.scope_id == expected_scope_id
            and scope_payloads
            and scope_payloads[0] is not None
            and _accepts_bank_profile(scope_payloads[0], expected_scope_id)
            and any(
                _accepts_bank_config(payload, expected_scope_id)
                for payload in scope_payloads[1:]
                if payload is not None
            )
        )
        dispatch_closed = bool(
            len(plan.ordered_dispatch_attempt_ids) == len(plan.ordered_dispatch_source_unit_ids)
            and tuple(
                source_id
                for source_ids in plan.ordered_dispatch_source_unit_ids
                for source_id in source_ids
            )
            == plan.ordered_source_unit_ids
            and plan.accepted_source_unit_ids == plan.ordered_source_unit_ids
            and not plan.rejected_source_unit_ids
        )
        for attempt_id, source_ids in zip(
            plan.ordered_dispatch_attempt_ids,
            plan.ordered_dispatch_source_unit_ids,
            strict=True,
        ):
            item_count = len(source_ids)
            attempt = attempts.get(attempt_id)
            payload = (
                snapshot.raw_payloads.get(attempt.raw_response_ref)
                if attempt is not None and attempt.raw_response_ref is not None
                else None
            )
            dispatch_closed = dispatch_closed and bool(
                attempt is not None
                and attempt.parent_kind == "ingestion_plan"
                and attempt.parent_id == plan.ingestion_occurrence_id
                and attempt.stage == "memory_ingest"
                and attempt.outcome == AttemptOutcome.SUCCEEDED
                and payload is not None
                and _accepts_retain(payload, expected_scope_id, item_count)
            )
        exact = exact and scope_closed and dispatch_closed
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "hindsight-scope-dispatch-drift"),)


def _hindsight_projection_readiness_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.hindsight.projection-readiness.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    plans = _contracts(snapshot, IngestionPlanRecordV2)
    exact = bool(plans)
    for plan in plans:
        try:
            projection = _hindsight_projection_evidence(
                snapshot,
                plan,
                plan.projection_raw_refs,
            )
        except ValueError:
            exact = False
            continue
        exact = exact and bool(
            projection.ordered_source_unit_ids == plan.projected_source_unit_ids
            and plan.projected_source_unit_ids == plan.accepted_source_unit_ids
            and projection.state_sha256 == plan.protected_state_sha256
            and plan.inventory_raw_ref == projection.state_sha256
            and plan.inventory_raw_ref in plan.projection_raw_refs
            and set(plan.ordered_dispatch_attempt_ids)
            <= {
                attempt.attempt_id
                for attempt in _contracts(snapshot, AttemptRecordV2)
                if attempt.raw_response_ref in plan.readiness_evidence_refs
            }
            and all(
                reference in snapshot.raw_payloads for reference in plan.readiness_evidence_refs
            )
            and all(reference in snapshot.raw_payloads for reference in plan.projection_raw_refs)
        )
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "hindsight-projection-readiness-drift"),)


def _hindsight_retrieval_mutation_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.hindsight.retrieval-mutation.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    plans = {
        item.ingestion_occurrence_id: item for item in _contracts(snapshot, IngestionPlanRecordV2)
    }
    cases = _contracts(snapshot, CaseRecordV3)
    exact = bool(cases)
    for case in cases:
        plan = plans.get(case.ingestion_occurrence_id)
        raw_retrieval = snapshot.raw_payloads.get(case.retrieval_raw_ref or "")
        try:
            if plan is None:
                raise ValueError("Hindsight case has no ingestion plan")
            projection = _hindsight_projection_evidence(
                snapshot,
                plan,
                plan.projection_raw_refs,
            )
            before = _hindsight_projection_evidence(
                snapshot,
                plan,
                case.pre_query_projection_raw_refs,
            )
            after = _hindsight_projection_evidence(
                snapshot,
                plan,
                case.post_query_projection_raw_refs,
            )
            candidates = (
                normalize_recall(
                    raw_retrieval,
                    document_to_source_unit=projection.document_to_source_unit,
                )
                if raw_retrieval is not None
                else ()
            )
        except ValueError:
            candidates = ()
            retrieval_valid = False
        else:
            retrieval_valid = raw_retrieval is not None
        exact = exact and bool(
            plan is not None
            and case.adapter_profile_id == HINDSIGHT_PROFILE_ID
            and case.state == CaseState.COMPLETED
            and retrieval_valid
            and tuple(item.native_id for item in candidates) == case.ordered_native_candidate_ids
            and tuple(
                hashlib.sha256(item.content.encode("utf-8")).hexdigest() for item in candidates
            )
            == case.ordered_native_content_sha256
            and tuple(item.source_unit_id for item in candidates)
            == case.native_candidate_source_unit_ids
            and case.query_mutation_status == "unchanged"
            and case.pre_query_state_sha256 == case.post_query_state_sha256
            and case.pre_query_state_sha256 == plan.protected_state_sha256
            and before.state_sha256 == case.pre_query_state_sha256
            and after.state_sha256 == case.post_query_state_sha256
            and all(
                reference in snapshot.raw_payloads
                for reference in (
                    *case.pre_query_projection_raw_refs,
                    *case.post_query_projection_raw_refs,
                    *case.retrieval_supporting_raw_refs,
                )
            )
        )
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "hindsight-retrieval-mutation-drift"),)


def _hindsight_projection_evidence(
    snapshot: _NativeCapsuleSnapshot,
    plan: IngestionPlanRecordV2,
    references: tuple[str, ...],
) -> HindsightProjectionEvidence:
    manifests = _contracts(snapshot, CaseManifest)
    if len(manifests) != 1 or plan.scope_id is None:
        raise ValueError("Hindsight capsule has no exact manifest or scope")
    manifest_plan = next(
        (
            item
            for item in manifests[0].ingestion_plans
            if item.ingestion_plan_id == plan.ingestion_plan_id
        ),
        None,
    )
    if manifest_plan is None:
        raise ValueError("Hindsight plan is absent from the case manifest")
    return reconstruct_hindsight_projection(
        raw_payloads=snapshot.raw_payloads,
        references=references,
        bank_id=plan.scope_id,
        ordered_source_unit_ids=plan.ordered_source_unit_ids,
        ordered_source_payload_sha256=manifest_plan.ordered_source_unit_bytes_sha256,
    )


def _mem0_profile(target: Any, expected: Mem0ExactProfile, rule_id: str) -> bool:
    from oamb.memory_systems.mem0.adapter import Mem0ReferenceNegativeAdapter
    from oamb.memory_systems.mem0.sdk import Mem0SdkAdapter

    expected_type = (
        Mem0ReferenceNegativeAdapter if expected == MEM0_REST_PROFILE else Mem0SdkAdapter
    )
    return bool(
        isinstance(target, Mem0AdapterValidationInput)
        and isinstance(target.adapter, expected_type)
        and target.audit == target.adapter.validation_audit()
        and mem0_zero_dispatch_audit_is_valid(target.audit)
        and target.profile == expected
        and target.unsupported_profile_id == expected.profile_id
        and target.unsupported_reason_codes == reference_profile_unsupported(expected).reason_codes
        and target.rejected_operation_count > 0
    )


def _mem0_rest_runtime_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.rest.runtime-profile.v1"
    return () if _mem0_profile(target, MEM0_REST_PROFILE, rule_id) else _wrong_target(rule_id)


def _mem0_rest_unsupported_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.rest.unsupported-verdict.v1"
    exact = _mem0_profile(target, MEM0_REST_PROFILE, rule_id)
    return () if exact else (_issue(rule_id, "mem0-rest-v1", "mem0-unsupported-verdict-drift"),)


def _mem0_rest_zero_dispatch_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.rest.zero-dispatch.v1"
    exact = bool(
        _mem0_profile(target, MEM0_REST_PROFILE, rule_id) and target.dispatched_operation_count == 0
    )
    return () if exact else (_issue(rule_id, "mem0-rest-v1", "mem0-zero-dispatch-drift"),)


def _mem0_rest_blackbox_runtime_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.rest.runtime-profile.v2"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    plans = tuple(
        plan for plan in _ingestion_plans(snapshot) if isinstance(plan, IngestionPlanRecordV3)
    )
    runs = _contracts(snapshot, RunRecord)
    runtime_hashes = {plan.runtime_binding_hash for plan in plans}
    has_resolution = any(
        _accepts_mem0_runtime_resolution(
            payload,
            raw_payloads=snapshot.raw_payloads,
            expected_runtime_binding_hash=next(iter(runtime_hashes), ""),
        )
        for payload in snapshot.raw_payloads.values()
    )
    exact = bool(
        _native_structural_closes(snapshot)
        and plans
        and len(plans) == len(_ingestion_plans(snapshot))
        and all(
            plan.adapter_profile_id == MEM0_PROFILE_ID
            and plan.memory_system_id == "mem0"
            and plan.projection_semantics == "retrieval_visible_subset"
            for plan in plans
        )
        and len(runs) == 1
        and runs[0].state == RunState.FINALIZED
        and len(runtime_hashes) == 1
        and has_resolution
    )
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "mem0-runtime-profile-drift"),)


def _mem0_rest_scope_dispatch_projection_rule(
    target: Any,
) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.rest.scope-dispatch-projection.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    plans = _ingestion_plans(snapshot)
    exact = bool(plans)
    for plan in plans:
        try:
            if not isinstance(plan, IngestionPlanRecordV3):
                raise ValueError("Mem0 REST plan is not evidence v3")
            reconstruct_mem0_plan(
                raw_payloads=snapshot.raw_payloads,
                plan=plan,
                attempts=attempts,
            )
        except ValueError:
            exact = False
            break
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "mem0-scope-dispatch-projection-drift"),)


def _mem0_rest_retrieval_order_scope_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.rest.retrieval-order-scope.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    plans = {
        item.ingestion_occurrence_id: item
        for item in _ingestion_plans(snapshot)
        if isinstance(item, IngestionPlanRecordV3)
    }
    cases = _contracts(snapshot, CaseRecordV3)
    exact = bool(cases)
    for case in cases:
        plan = plans.get(case.ingestion_occurrence_id)
        try:
            if (
                plan is None
                or case.adapter_profile_id != MEM0_PROFILE_ID
                or case.state != CaseState.COMPLETED
            ):
                raise ValueError("Mem0 case parentage is invalid")
            evidence = reconstruct_mem0_plan(
                raw_payloads=snapshot.raw_payloads,
                plan=plan,
                attempts=attempts,
            )
            candidates = reconstruct_mem0_candidates(
                raw_payloads=snapshot.raw_payloads,
                case=case,
                plan=evidence,
            )
            if (
                tuple(item.native_id for item in candidates) != case.ordered_native_candidate_ids
                or tuple(
                    hashlib.sha256(item.content.encode("utf-8")).hexdigest() for item in candidates
                )
                != case.ordered_native_content_sha256
                or tuple(item.source_unit_id for item in candidates)
                != case.native_candidate_source_unit_ids
            ):
                raise ValueError("Mem0 candidate order or scope drifted")
        except ValueError:
            exact = False
            break
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "mem0-retrieval-order-drift"),)


def _mem0_rest_query_mutation_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.rest.query-mutation.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    plans = {
        item.ingestion_occurrence_id: item
        for item in _ingestion_plans(snapshot)
        if isinstance(item, IngestionPlanRecordV3)
    }
    cases = _contracts(snapshot, CaseRecordV3)
    exact = bool(cases)
    for case in cases:
        plan = plans.get(case.ingestion_occurrence_id)
        try:
            if plan is None:
                raise ValueError("Mem0 case has no v3 plan")
            plan_evidence = reconstruct_mem0_plan(
                raw_payloads=snapshot.raw_payloads,
                plan=plan,
                attempts=attempts,
            )
            before = reconstruct_mem0_projection(
                raw_payloads=snapshot.raw_payloads,
                references=case.pre_query_projection_raw_refs,
                expected_run_id=plan.ingestion_occurrence_id,
            )
            after = reconstruct_mem0_projection(
                raw_payloads=snapshot.raw_payloads,
                references=case.post_query_projection_raw_refs,
                expected_run_id=plan.ingestion_occurrence_id,
            )
            if (
                before.state_sha256 != plan.protected_state_sha256
                or after.state_sha256 != plan.protected_state_sha256
                or before.capture_sequence <= plan_evidence.projection.capture_sequence
                or after.capture_sequence != before.capture_sequence + 1
                or case.pre_query_state_sha256 != plan.protected_state_sha256
                or case.post_query_state_sha256 != plan.protected_state_sha256
                or case.query_mutation_status != "unchanged"
            ):
                raise ValueError("Mem0 query state differs from readiness")
        except ValueError:
            exact = False
            break
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "mem0-query-state-drift"),)


def _accepts_mem0_openapi(payload: bytes) -> bool:
    try:
        document = json.loads(payload)
        paths = document["paths"]
        return bool(
            document["openapi"] == "3.1.0"
            and document["info"]["title"] == "Mem0 REST APIs"
            and document["info"]["version"] == "1.0.0"
            and isinstance(paths["/memories"].get("post"), dict)
            and isinstance(paths["/search"].get("post"), dict)
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _accepts_mem0_inspector_health(payload: bytes) -> bool:
    try:
        return bool(json.loads(payload) == {"status": "ok", "mode": "read_only_projection"})
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _accepts_mem0_runtime_resolution(
    payload: bytes,
    *,
    raw_payloads: dict[str, bytes],
    expected_runtime_binding_hash: str,
) -> bool:
    try:
        document = json.loads(payload)
        if not isinstance(document, dict) or frozenset(document) != frozenset(
            {
                "schema",
                "release_version",
                "runtime_binding_hash",
                "openapi_raw_ref",
                "inspector_health_raw_ref",
            }
        ):
            return False
        openapi_ref = document["openapi_raw_ref"]
        health_ref = document["inspector_health_raw_ref"]
        return bool(
            document["schema"] == "oamb-mem0-runtime-resolution-v1"
            and document["release_version"] == "2.0.19"
            and document["runtime_binding_hash"] == expected_runtime_binding_hash
            and isinstance(openapi_ref, str)
            and isinstance(health_ref, str)
            and openapi_ref in raw_payloads
            and health_ref in raw_payloads
            and _accepts_mem0_openapi(raw_payloads[openapi_ref])
            and _accepts_mem0_inspector_health(raw_payloads[health_ref])
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return False


def _mem0_sdk_runtime_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.sdk.runtime-profile.v1"
    return () if _mem0_profile(target, MEM0_SDK_PROFILE, rule_id) else _wrong_target(rule_id)


def _mem0_sdk_unsupported_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.sdk.unsupported-verdict.v1"
    exact = _mem0_profile(target, MEM0_SDK_PROFILE, rule_id)
    return () if exact else (_issue(rule_id, "mem0-sdk-v1", "mem0-unsupported-verdict-drift"),)


def _mem0_sdk_zero_dispatch_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.sdk.zero-dispatch.v1"
    exact = bool(
        _mem0_profile(target, MEM0_SDK_PROFILE, rule_id) and target.dispatched_operation_count == 0
    )
    return () if exact else (_issue(rule_id, "mem0-sdk-v1", "mem0-zero-dispatch-drift"),)


def _mem0_sdk_comparison_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.mem0.sdk.comparison-ineligible.v1"
    exact = bool(
        _mem0_profile(target, MEM0_SDK_PROFILE, rule_id) and not target.comparison_eligible
    )
    return () if exact else (_issue(rule_id, "mem0-sdk-v1", "mem0-comparison-eligibility-drift"),)


def _openviking_profile_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.openviking.runtime-auth-scope.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    plans = _contracts(snapshot, IngestionPlanRecordV2)
    runs = _contracts(snapshot, RunRecord)
    try:
        runtime_identity = reconstruct_openviking_runtime_identity(snapshot.raw_payloads)
        for plan in plans:
            reconstruct_openviking_scope(
                raw_payloads=snapshot.raw_payloads,
                plan=plan,
                runtime_identity=runtime_identity,
            )
    except ValueError:
        runtime_closed = False
    else:
        runtime_closed = True
    exact = bool(
        _native_structural_closes(snapshot)
        and plans
        and len(runs) == 1
        and runs[0].state == RunState.FINALIZED
        and runtime_closed
        and all(
            plan.adapter_profile_id == OPENVIKING_PROFILE_ID
            and plan.memory_system_id == OPENVIKING_MEMORY_SYSTEM_ID
            for plan in plans
        )
    )
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "openviking-runtime-scope-drift"),)


def _openviking_dispatch_projection_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.openviking.dispatch-projection.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    try:
        exact = bool(_openviking_plan_evidence(snapshot))
    except ValueError:
        exact = False
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "openviking-dispatch-projection-drift"),)


def _openviking_retrieval_order_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.openviking.retrieval-order.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    cases = _contracts(snapshot, CaseRecordV3)
    try:
        plans = _openviking_plan_evidence(snapshot)
        exact = bool(cases)
        for case in cases:
            candidates = reconstruct_openviking_candidates(
                raw_payloads=snapshot.raw_payloads,
                case=case,
                plan=plans[case.ingestion_occurrence_id],
            )
            exact = exact and bool(
                case.adapter_profile_id == OPENVIKING_PROFILE_ID
                and case.state == CaseState.COMPLETED
                and tuple(item.native_id for item in candidates)
                == case.ordered_native_candidate_ids
                and tuple(
                    hashlib.sha256(item.content.encode("utf-8")).hexdigest() for item in candidates
                )
                == case.ordered_native_content_sha256
                and tuple(item.source_unit_id for item in candidates)
                == case.native_candidate_source_unit_ids
            )
    except (KeyError, ValueError):
        exact = False
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "openviking-retrieval-order-drift"),)


def _openviking_query_mutation_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "adapter.openviking.query-mutation.v1"
    snapshot = _native_snapshot(target)
    if snapshot is None:
        return _wrong_target(rule_id)
    cases = _contracts(snapshot, CaseRecordV3)
    try:
        plans = _openviking_plan_evidence(snapshot)
        exact = bool(cases)
        for case in cases:
            plan = plans[case.ingestion_occurrence_id]
            before = reconstruct_openviking_projection(
                raw_payloads=snapshot.raw_payloads,
                references=case.pre_query_projection_raw_refs,
                root_uri=plan.root_uri,
                chunk_uris=plan.chunk_uris,
                source_payload_sha256=plan.source_payload_sha256,
            )
            after = reconstruct_openviking_projection(
                raw_payloads=snapshot.raw_payloads,
                references=case.post_query_projection_raw_refs,
                root_uri=plan.root_uri,
                chunk_uris=plan.chunk_uris,
                source_payload_sha256=plan.source_payload_sha256,
            )
            exact = exact and bool(
                case.query_mutation_status == "unchanged"
                and before.state_sha256 == case.pre_query_state_sha256
                and after.state_sha256 == case.post_query_state_sha256
                and before.state_sha256 == after.state_sha256
                and before.state_sha256 == plan.projection.state_sha256
            )
    except (KeyError, ValueError):
        exact = False
    evidence_ref = snapshot.manifest.source_manifest_hash if snapshot.manifest else "manifest"
    return () if exact else (_issue(rule_id, evidence_ref, "openviking-query-mutation-drift"),)


def _openviking_plan_evidence(
    snapshot: _NativeCapsuleSnapshot,
) -> dict[str, OpenVikingPlanEvidence]:
    plans = _contracts(snapshot, IngestionPlanRecordV2)
    manifests = _contracts(snapshot, CaseManifest)
    if not plans or len(manifests) != 1:
        raise ValueError("OpenViking native capsule has no exact plan inventory")
    manifest_plans = {item.ingestion_plan_id: item for item in manifests[0].ingestion_plans}
    attempts = {item.attempt_id: item for item in _contracts(snapshot, AttemptRecordV2)}
    runtime_identity = reconstruct_openviking_runtime_identity(snapshot.raw_payloads)
    reconstructed: dict[str, OpenVikingPlanEvidence] = {}
    for plan in plans:
        if (
            plan.adapter_profile_id != OPENVIKING_PROFILE_ID
            or plan.memory_system_id != OPENVIKING_MEMORY_SYSTEM_ID
            or plan.state != IngestionPlanState.SEALED
        ):
            raise ValueError("OpenViking plan does not match the exact adapter profile")
        manifest_plan = manifest_plans.get(plan.ingestion_plan_id)
        if manifest_plan is None:
            raise ValueError("OpenViking plan is absent from the case manifest")
        reconstructed[plan.ingestion_occurrence_id] = reconstruct_openviking_plan(
            raw_payloads=snapshot.raw_payloads,
            plan=plan,
            manifest_plan=manifest_plan,
            attempts=attempts,
            runtime_identity=runtime_identity,
        )
    return reconstructed


def _hindsight_projected_source_ids(documents: object) -> tuple[str, ...]:
    if not isinstance(documents, list):
        return ()
    source_ids: list[str] = []
    for item in documents:
        if not isinstance(item, dict):
            return ()
        detail = item.get("detail")
        if not isinstance(detail, dict) or not isinstance(detail.get("id"), str):
            return ()
        source_ids.append(detail["id"])
    return tuple(source_ids)


def _accepts(parser: Callable[[bytes], object], payload: bytes | None) -> bool:
    if payload is None:
        return False
    try:
        parser(payload)
    except (TypeError, ValueError):
        return False
    return True


def _accepts_bank_profile(payload: bytes, scope_id: str) -> bool:
    try:
        parse_bank_profile(payload, expected_bank_id=scope_id)
    except ValueError:
        return False
    return True


def _accepts_bank_config(payload: bytes, scope_id: str) -> bool:
    try:
        parse_bank_config(payload, expected_bank_id=scope_id)
    except ValueError:
        return False
    return True


def _accepts_retain(payload: bytes, scope_id: str, item_count: int) -> bool:
    try:
        parse_retain_response(
            payload,
            expected_bank_id=scope_id,
            expected_items_count=item_count,
        )
    except ValueError:
        return False
    return True


def _wrong_target(rule_id: str) -> tuple[ValidationIssue, ...]:
    return (_issue(rule_id, "target", "validation-target-type-mismatch"),)


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


ADAPTER_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("adapter.hindsight.runtime-profile.v1", 1, _hindsight_runtime_profile_rule),
    ValidationRule("adapter.hindsight.scope-dispatch.v1", 1, _hindsight_scope_dispatch_rule),
    ValidationRule(
        "adapter.hindsight.projection-readiness.v1",
        1,
        _hindsight_projection_readiness_rule,
    ),
    ValidationRule(
        "adapter.hindsight.retrieval-mutation.v1",
        1,
        _hindsight_retrieval_mutation_rule,
    ),
    ValidationRule("adapter.mem0.rest.runtime-profile.v1", 1, _mem0_rest_runtime_rule),
    ValidationRule("adapter.mem0.rest.unsupported-verdict.v1", 1, _mem0_rest_unsupported_rule),
    ValidationRule("adapter.mem0.rest.zero-dispatch.v1", 1, _mem0_rest_zero_dispatch_rule),
    ValidationRule(
        "adapter.mem0.rest.runtime-profile.v2",
        2,
        _mem0_rest_blackbox_runtime_rule,
    ),
    ValidationRule(
        "adapter.mem0.rest.scope-dispatch-projection.v1",
        1,
        _mem0_rest_scope_dispatch_projection_rule,
    ),
    ValidationRule(
        "adapter.mem0.rest.retrieval-order-scope.v1",
        1,
        _mem0_rest_retrieval_order_scope_rule,
    ),
    ValidationRule(
        "adapter.mem0.rest.query-mutation.v1",
        1,
        _mem0_rest_query_mutation_rule,
    ),
    ValidationRule("adapter.mem0.sdk.runtime-profile.v1", 1, _mem0_sdk_runtime_rule),
    ValidationRule("adapter.mem0.sdk.unsupported-verdict.v1", 1, _mem0_sdk_unsupported_rule),
    ValidationRule("adapter.mem0.sdk.zero-dispatch.v1", 1, _mem0_sdk_zero_dispatch_rule),
    ValidationRule(
        "adapter.mem0.sdk.comparison-ineligible.v1",
        1,
        _mem0_sdk_comparison_rule,
    ),
    ValidationRule(
        "adapter.openviking.runtime-auth-scope.v1",
        1,
        _openviking_profile_rule,
    ),
    ValidationRule(
        "adapter.openviking.dispatch-projection.v1",
        1,
        _openviking_dispatch_projection_rule,
    ),
    ValidationRule(
        "adapter.openviking.retrieval-order.v1",
        1,
        _openviking_retrieval_order_rule,
    ),
    ValidationRule(
        "adapter.openviking.query-mutation.v1",
        1,
        _openviking_query_mutation_rule,
    ),
)


__all__ = [
    "ADAPTER_RULES",
    "Mem0AdapterValidationInput",
    "mem0_adapter_validation_input",
]
