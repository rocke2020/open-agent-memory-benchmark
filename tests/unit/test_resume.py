from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from oamb.artifacts.atomic import atomic_write_bytes
from oamb.artifacts.store import ArtifactStore
from oamb.contracts.evidence import (
    CapsuleManifestEntry,
    MemoryConformanceEvidenceManifest,
    MemoryConformanceOccurrenceRecord,
    MemoryConformanceOccurrenceState,
    memory_conformance_evidence_manifest_hash,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import ArtifactWriteRequest
from oamb.contracts.specifications import (
    MEMORY_CONFORMANCE_ROUTE_STAGES,
    DispatchBudgetOwnerKind,
    DispatchBudgetRoute,
    MemoryConformanceSpec,
    dispatch_budget_route_hash,
    memory_conformance_spec_hash,
)
from oamb.contracts.states import AttemptOutcome
from oamb.runtime.resume import (
    AttemptEvidenceState,
    ProviderLifecycleBridge,
    RecoveryAction,
    ResumeFingerprint,
    ResumeRejectedError,
    StaleLeaseOwner,
    authorize_stale_lease_recovery,
    classify_attempt_recovery,
    host_identity_fingerprint,
    require_matching_fingerprint,
)


def fingerprint(**changes: str) -> ResumeFingerprint:
    values = {
        "protocol": "protocol-a",
        "dataset": "dataset-a",
        "case_manifest": "cases-a",
        "prompt_pack": "prompt-a",
        "metric": "metric-a",
        "code_revision": "code-a",
        "memory_runtime": "runtime-a",
        "model_roles": "roles-a",
        "budget": "budget-a",
        "normalizer": "normalizer-a",
    }
    values.update(changes)
    return ResumeFingerprint(**values)


def test_resume_rejects_every_fingerprint_mismatch() -> None:
    expected = fingerprint()

    for field_name in expected.__dataclass_fields__:
        with pytest.raises(ResumeRejectedError, match=field_name):
            require_matching_fingerprint(
                expected,
                fingerprint(**{field_name: f"different-{field_name}"}),
            )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            AttemptEvidenceState(
                claim=True,
                intent=False,
                receipt=False,
                terminal_outcome=None,
            ),
            RecoveryAction.REACQUIRE,
        ),
        (
            AttemptEvidenceState(
                claim=True,
                intent=True,
                receipt=True,
                terminal_outcome=None,
            ),
            RecoveryAction.FINISH_LOCAL,
        ),
        (
            AttemptEvidenceState(
                claim=True,
                intent=True,
                receipt=False,
                terminal_outcome=None,
            ),
            RecoveryAction.UNKNOWN_OUTCOME,
        ),
        (
            AttemptEvidenceState(
                claim=True,
                intent=True,
                receipt=True,
                terminal_outcome=AttemptOutcome.SUCCEEDED,
            ),
            RecoveryAction.NO_ACTION,
        ),
        (
            AttemptEvidenceState(
                claim=True,
                intent=True,
                receipt=False,
                terminal_outcome=AttemptOutcome.UNKNOWN_OUTCOME,
            ),
            RecoveryAction.NO_ACTION,
        ),
    ],
)
def test_attempt_recovery_never_replays_an_intent_without_receipt(
    state: AttemptEvidenceState,
    expected: RecoveryAction,
) -> None:
    assert classify_attempt_recovery(state) is expected


def test_terminal_unknown_rejects_a_provider_receipt() -> None:
    with pytest.raises(ResumeRejectedError, match="unknown-outcome terminal cannot have receipt"):
        classify_attempt_recovery(
            AttemptEvidenceState(
                claim=True,
                intent=True,
                receipt=True,
                terminal_outcome=AttemptOutcome.UNKNOWN_OUTCOME,
            )
        )


def test_provider_lifecycle_bridge_is_create_only_and_hash_guarded(tmp_path: Path) -> None:
    bridge = ProviderLifecycleBridge(tmp_path)

    authority = bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )

    pointer = json.loads((tmp_path / "active-operation").read_text(encoding="utf-8"))
    assert pointer == {
        "kind": "benchmark_run",
        "lease_epoch": 1,
        "lease_record_sha256": "a" * 64,
        "owner": "run-a",
        "profile_id": "mem0-rest-v1",
        "provider_project": "oamb-providers-test-a",
        "run_id": "run-a",
        "schema_name": "oamb_provider_active_operation_pointer",
        "schema_version": 1,
    }
    with pytest.raises(ResumeRejectedError, match="active provider operation"):
        bridge.acquire_run(
            run_id="run-b",
            provider_project="oamb-providers-test-a",
            profile_id="mem0-rest-v1",
            lease_epoch=1,
            lease_record_hash="b" * 64,
        )
    with pytest.raises(ResumeRejectedError, match="authority"):
        bridge.release_run(replace(authority))
    assert (tmp_path / "active-operation").exists()

    bridge.release_run(authority)

    assert not (tmp_path / "active-operation").exists()


def test_malformed_operational_pointer_fails_closed_instead_of_releasing(tmp_path: Path) -> None:
    bridge = ProviderLifecycleBridge(tmp_path)
    authority = bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )
    pointer_path = tmp_path / "active-operation"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    pointer["unexpected"] = "field"
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")

    with pytest.raises(ResumeRejectedError, match="malformed"):
        bridge.release_run(authority)

    assert pointer_path.exists()


def test_provider_attempt_pointer_blocks_run_release_until_durable_receipt(tmp_path: Path) -> None:
    bridge = ProviderLifecycleBridge(tmp_path)
    authority = bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )

    bridge.mark_attempt_dispatched(attempt_id="b" * 64, intent_record_hash="c" * 64)

    attempt_pointer_path = tmp_path / "active-provider-attempts" / f"{'b' * 64}.json"
    attempt_pointer = json.loads(attempt_pointer_path.read_text(encoding="utf-8"))
    assert attempt_pointer["attempt_id"] == "b" * 64
    assert attempt_pointer["intent_record_sha256"] == "c" * 64
    with pytest.raises(ResumeRejectedError, match="provider attempt"):
        bridge.release_run(authority)
    with pytest.raises(ResumeRejectedError, match="hash"):
        bridge.clear_attempt_after_receipt(
            attempt_id="b" * 64,
            expected_intent_record_hash="d" * 64,
        )

    bridge.clear_attempt_after_receipt(
        attempt_id="b" * 64,
        expected_intent_record_hash="c" * 64,
    )
    bridge.release_run(authority)

    assert not attempt_pointer_path.exists()
    assert not (tmp_path / "active-operation").exists()


def test_new_process_bridge_cannot_dispatch_under_a_stale_run_pointer(
    tmp_path: Path,
) -> None:
    owner = ProviderLifecycleBridge(tmp_path)
    owner.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )

    with pytest.raises(ResumeRejectedError, match="operation authority"):
        ProviderLifecycleBridge(tmp_path).mark_attempt_dispatched(
            attempt_id="b" * 64,
            intent_record_hash="c" * 64,
        )

    assert not (tmp_path / "active-provider-attempts").exists()


def test_provider_lifecycle_bridge_allows_only_one_racing_owner(tmp_path: Path) -> None:
    def acquire(ordinal: int) -> str:
        bridge = ProviderLifecycleBridge(tmp_path)
        try:
            bridge.acquire_run(
                run_id=f"run-{ordinal}",
                provider_project="oamb-providers-test-a",
                profile_id="mem0-rest-v1",
                lease_epoch=1,
                lease_record_hash=str(ordinal) * 64,
            )
        except ResumeRejectedError:
            return "rejected"
        return "acquired"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = sorted(pool.map(acquire, (1, 2)))

    assert results == ["acquired", "rejected"]


def test_legacy_active_run_lease_blocks_run_acquisition(tmp_path: Path) -> None:
    (tmp_path / "active-run-lease").write_text("legacy\n", encoding="utf-8")
    bridge = ProviderLifecycleBridge(tmp_path)

    with pytest.raises(ResumeRejectedError, match="legacy active run lease"):
        bridge.acquire_run(
            run_id="run-a",
            provider_project="oamb-providers-test-a",
            profile_id="mem0-rest-v1",
            lease_epoch=1,
            lease_record_hash="a" * 64,
        )
    assert not (tmp_path / "active-operation").exists()


def _conformance_spec() -> MemoryConformanceSpec:
    routes = []
    for stage in MEMORY_CONFORMANCE_ROUTE_STAGES:
        route_fields: dict[str, object] = {
            "route_id": f"conformance-{stage}",
            "stage": stage,
            "dispatch_owner_kind": DispatchBudgetOwnerKind.PROVIDER_OPERATION,
            "dispatch_model_role_binding_id": None,
            "provider_operation_ceiling_id": f"operation-{stage}",
            "adapter_profile_id": "mem0-rest-v1",
            "operation_kind": stage,
            "billing_unit": "request",
            "internal_usage_role_binding_ids": (),
        }
        routes.append(
            DispatchBudgetRoute.model_validate(
                {
                    "route_hash": dispatch_budget_route_hash(route_fields),
                    **route_fields,
                }
            )
        )
    spec_fields: dict[str, object] = {
        "conformance_spec_id": "mem0-conformance-spec-1",
        "occurrence_id": "mem0-conformance-1",
        "provider": "mem0",
        "provider_project_id": "oamb-mem0-conformance-1",
        "provider_profile_id": "mem0-rest-v1",
        "runtime_binding_hash": "a" * 64,
        "budget_id": "budget-1",
        "budget_hash": "c" * 64,
        "dispatch_routes": tuple(routes),
        "minimal_source_sha256": "d" * 64,
        "minimal_query_sha256": "e" * 64,
        "expected_marker_sha256": "f" * 64,
        "stop_condition_ids": ("unknown-outcome",),
    }
    return MemoryConformanceSpec.model_validate(
        {
            "conformance_spec_hash": memory_conformance_spec_hash(spec_fields),
            **spec_fields,
        }
    )


def _seal_conformance_spec(root: Path, spec: MemoryConformanceSpec) -> None:
    content = canonical_json_bytes(spec)
    ArtifactStore(root).seal_source_record(
        ArtifactWriteRequest(
            record_id=spec.conformance_spec_id,
            relative_path="source/specs/memory-conformance-spec.json",
            canonical_sha256=canonical_sha256(spec),
            canonical_bytes=content,
        )
    )


def test_memory_conformance_acquisition_reads_registered_canonical_spec(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    spec = _conformance_spec()
    _seal_conformance_spec(evidence_root, spec)
    bridge = ProviderLifecycleBridge(runtime)

    bridge.acquire_memory_conformance(
        evidence_root=evidence_root,
        owner="operator-1",
    )

    assert json.loads((runtime / "active-operation").read_text(encoding="utf-8")) == {
        "schema_name": "oamb_provider_active_operation_pointer",
        "schema_version": 1,
        "kind": "memory_conformance",
        "owner": "operator-1",
        "occurrence_id": spec.occurrence_id,
        "conformance_spec_sha256": spec.conformance_spec_hash,
        "provider_project": spec.provider_project_id,
        "profile_id": spec.provider_profile_id,
    }


def test_memory_conformance_acquisition_rejects_absent_or_noncanonical_spec(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    bridge = ProviderLifecycleBridge(runtime)

    with pytest.raises(ResumeRejectedError, match="registered specification"):
        bridge.acquire_memory_conformance(evidence_root=evidence_root, owner="operator-1")
    assert not (runtime / "active-operation").exists()

    spec = _conformance_spec()
    _seal_conformance_spec(evidence_root, spec)
    spec_path = evidence_root / "source/specs/memory-conformance-spec.json"
    spec_path.write_bytes(spec_path.read_bytes() + b"\n")
    with pytest.raises(ResumeRejectedError, match="canonical"):
        bridge.acquire_memory_conformance(evidence_root=evidence_root, owner="operator-1")
    assert not (runtime / "active-operation").exists()


_CONFORMANCE_REQUIRED_KINDS = (
    "budget_spec",
    "occurrence_claim_record",
    "budget_reservation_record",
    "attempt_intent_record",
    "attempt_receipt_record",
    "attempt_record",
    "token_usage_record",
    "resource_usage_record",
    "cost_record",
    "projection_before",
    "projection_after",
)


def _write_conformance_terminal(
    root: Path,
    spec: MemoryConformanceSpec,
    *,
    state: MemoryConformanceOccurrenceState = MemoryConformanceOccurrenceState.SEALED,
) -> MemoryConformanceEvidenceManifest:
    store = ArtifactStore(root)
    source_entries: list[CapsuleManifestEntry] = []
    spec_bytes = canonical_json_bytes(spec)
    source_entries.append(
        CapsuleManifestEntry(
            record_kind="memory_conformance_spec",
            record_id=spec.conformance_spec_id,
            relative_path="source/specs/memory-conformance-spec.json",
            sha256=hashlib.sha256(spec_bytes).hexdigest(),
        )
    )
    now = datetime(2026, 8, 30, 12, 0, tzinfo=UTC)
    occurrence = MemoryConformanceOccurrenceRecord(
        occurrence_id=spec.occurrence_id,
        provider=spec.provider,
        provider_project_id=spec.provider_project_id,
        provider_profile_id=spec.provider_profile_id,
        provider_scope_id="fresh-conformance-scope-1",
        runtime_binding_hash=spec.runtime_binding_hash,
        budget_id=spec.budget_id,
        state=state,
        operation_claim_ids=("1" * 64,),
        dispatch_route_ids=tuple(route.route_id for route in spec.dispatch_routes),
        attempt_ids=("2" * 64,),
        usage_record_ids=("3" * 64,),
        resource_record_ids=("4" * 64,),
        cost_record_ids=("5" * 64,),
        protected_state_before_hash="6" * 64,
        protected_state_after_hash="6" * 64,
        started_at=now,
        ended_at=now + timedelta(seconds=1),
    )
    occurrence_bytes = canonical_json_bytes(occurrence)
    occurrence_hash = hashlib.sha256(occurrence_bytes).hexdigest()
    occurrence_path = "source/occurrence.json"
    store.seal_source_record(
        ArtifactWriteRequest(
            record_id=occurrence.occurrence_id,
            relative_path=occurrence_path,
            canonical_sha256=occurrence_hash,
            canonical_bytes=occurrence_bytes,
        )
    )
    source_entries.append(
        CapsuleManifestEntry(
            record_kind="memory_conformance_occurrence_record",
            record_id=occurrence.occurrence_id,
            relative_path=occurrence_path,
            sha256=occurrence_hash,
        )
    )
    for ordinal, kind in enumerate(_CONFORMANCE_REQUIRED_KINDS, start=1):
        payload = canonical_json_bytes({"kind": kind, "ordinal": ordinal})
        digest = hashlib.sha256(payload).hexdigest()
        relative_path = f"source/fixtures/{ordinal:02d}-{kind}.json"
        store.seal_source_record(
            ArtifactWriteRequest(
                record_id=digest,
                relative_path=relative_path,
                canonical_sha256=digest,
                canonical_bytes=payload,
            )
        )
        source_entries.append(
            CapsuleManifestEntry(
                record_kind=kind,
                record_id=digest,
                relative_path=relative_path,
                sha256=digest,
            )
        )
    raw_bytes = b'{"provider":"mem0","status":"ok"}'
    raw_hash = hashlib.sha256(raw_bytes).hexdigest()
    raw_path = root / "raw/response.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(raw_path, raw_bytes, trusted_root=root)
    raw_entries = (
        CapsuleManifestEntry(
            record_kind="raw_reference",
            record_id=raw_hash,
            relative_path="raw/response.json",
            sha256=raw_hash,
        ),
    )
    fields = {
        "conformance_spec_id": spec.conformance_spec_id,
        "conformance_spec_hash": spec.conformance_spec_hash,
        "occurrence_id": spec.occurrence_id,
        "terminal_occurrence_hash": occurrence_hash,
        "terminal_state": state,
        "source_entries": tuple(source_entries),
        "raw_entries": raw_entries,
        "attempt_root_hash": canonical_sha256(["attempt-root"]),
        "token_usage_root_hash": canonical_sha256(["token-root"]),
        "resource_usage_root_hash": canonical_sha256(["resource-root"]),
        "cost_root_hash": canonical_sha256(["cost-root"]),
        "created_at": now + timedelta(seconds=2),
    }
    manifest = MemoryConformanceEvidenceManifest.model_validate(
        {
            "manifest_hash": memory_conformance_evidence_manifest_hash(fields),
            **fields,
        }
    )
    atomic_write_bytes(
        root / "memory-conformance-manifest.json",
        canonical_json_bytes(manifest),
        trusted_root=root,
    )
    return manifest


def test_memory_conformance_release_requires_verified_terminal_manifest(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    spec = _conformance_spec()
    _seal_conformance_spec(evidence_root, spec)
    bridge = ProviderLifecycleBridge(runtime)
    authority = bridge.acquire_memory_conformance(
        evidence_root=evidence_root,
        owner="operator-1",
    )

    with pytest.raises(ResumeRejectedError, match="terminal manifest"):
        bridge.release_memory_conformance(authority)
    assert (runtime / "active-operation").exists()

    _write_conformance_terminal(evidence_root, spec)
    bridge.release_memory_conformance(authority)
    assert not (runtime / "active-operation").exists()


def test_memory_conformance_manifest_mismatch_or_unknown_retains_pointers(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    spec = _conformance_spec()
    _seal_conformance_spec(evidence_root, spec)
    bridge = ProviderLifecycleBridge(runtime)
    authority = bridge.acquire_memory_conformance(
        evidence_root=evidence_root,
        owner="operator-1",
    )
    _write_conformance_terminal(
        evidence_root,
        spec,
        state=MemoryConformanceOccurrenceState.INTERRUPTED_UNKNOWN_OUTCOME,
    )

    with pytest.raises(ResumeRejectedError, match="unknown outcome"):
        bridge.release_memory_conformance(authority)
    assert (runtime / "active-operation").exists()


def test_memory_conformance_active_attempt_blocks_release(tmp_path: Path) -> None:
    evidence_root = tmp_path / "evidence"
    runtime = tmp_path / "runtime"
    spec = _conformance_spec()
    _seal_conformance_spec(evidence_root, spec)
    bridge = ProviderLifecycleBridge(runtime)
    authority = bridge.acquire_memory_conformance(
        evidence_root=evidence_root,
        owner="operator-1",
    )
    _write_conformance_terminal(evidence_root, spec)
    bridge.mark_attempt_dispatched(attempt_id="7" * 64, intent_record_hash="8" * 64)

    with pytest.raises(ResumeRejectedError, match="active provider attempt"):
        bridge.release_memory_conformance(authority)
    assert (runtime / "active-operation").exists()
    assert (runtime / "active-provider-attempts" / f"{'7' * 64}.json").exists()


def test_same_host_live_process_is_never_treated_as_stale() -> None:
    owner = StaleLeaseOwner(
        host="host-a",
        host_fingerprint=host_identity_fingerprint("host-a"),
        process_id=123,
    )

    with pytest.raises(ResumeRejectedError, match="still alive"):
        authorize_stale_lease_recovery(
            owner,
            current_host="host-a",
            process_is_alive=lambda process_id: process_id == 123,
            explicitly_authorized=True,
        )


def test_same_host_dead_process_and_authorized_foreign_owner_are_distinct() -> None:
    same_host = authorize_stale_lease_recovery(
        StaleLeaseOwner(
            host="host-a",
            host_fingerprint=host_identity_fingerprint("host-a"),
            process_id=123,
        ),
        current_host="host-a",
        process_is_alive=lambda _process_id: False,
        explicitly_authorized=False,
    )
    assert same_host.reason == "same_host_process_absent"

    with pytest.raises(ResumeRejectedError, match="explicit authorization"):
        authorize_stale_lease_recovery(
            StaleLeaseOwner(
                host="host-b",
                host_fingerprint=host_identity_fingerprint("host-b"),
                process_id=456,
            ),
            current_host="host-a",
            process_is_alive=lambda _process_id: False,
            explicitly_authorized=False,
        )

    foreign = authorize_stale_lease_recovery(
        StaleLeaseOwner(
            host="host-b",
            host_fingerprint=host_identity_fingerprint("host-b"),
            process_id=456,
        ),
        current_host="host-a",
        process_is_alive=lambda _process_id: False,
        explicitly_authorized=True,
    )
    assert foreign.reason == "foreign_owner_explicitly_authorized"
