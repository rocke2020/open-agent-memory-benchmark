from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

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

    pointer = json.loads((tmp_path / "active-run-lease").read_text(encoding="utf-8"))
    assert pointer == {
        "lease_epoch": 1,
        "lease_record_sha256": "a" * 64,
        "profile_id": "mem0-rest-v1",
        "provider_project": "oamb-providers-test-a",
        "run_id": "run-a",
        "schema_name": "oamb_provider_active_run_pointer",
        "schema_version": 1,
    }
    with pytest.raises(ResumeRejectedError, match="active run"):
        bridge.acquire_run(
            run_id="run-b",
            provider_project="oamb-providers-test-a",
            profile_id="mem0-rest-v1",
            lease_epoch=1,
            lease_record_hash="b" * 64,
        )
    with pytest.raises(ResumeRejectedError, match="authority"):
        bridge.release_run(replace(authority))
    assert (tmp_path / "active-run-lease").exists()

    bridge.release_run(authority)

    assert not (tmp_path / "active-run-lease").exists()


def test_malformed_operational_pointer_fails_closed_instead_of_releasing(tmp_path: Path) -> None:
    bridge = ProviderLifecycleBridge(tmp_path)
    authority = bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )
    pointer_path = tmp_path / "active-run-lease"
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

    attempt_pointer = json.loads((tmp_path / "active-provider-attempt").read_text(encoding="utf-8"))
    assert attempt_pointer["attempt_id"] == "b" * 64
    assert attempt_pointer["intent_record_sha256"] == "c" * 64
    with pytest.raises(ResumeRejectedError, match="provider attempt"):
        bridge.release_run(authority)
    with pytest.raises(ResumeRejectedError, match="hash"):
        bridge.clear_attempt_after_receipt("d" * 64)

    bridge.clear_attempt_after_receipt("c" * 64)
    bridge.release_run(authority)

    assert not (tmp_path / "active-provider-attempt").exists()
    assert not (tmp_path / "active-run-lease").exists()


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

    with pytest.raises(ResumeRejectedError, match="lease authority"):
        ProviderLifecycleBridge(tmp_path).mark_attempt_dispatched(
            attempt_id="b" * 64,
            intent_record_hash="c" * 64,
        )

    assert not (tmp_path / "active-provider-attempt").exists()


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
