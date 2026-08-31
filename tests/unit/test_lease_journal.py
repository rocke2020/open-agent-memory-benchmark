from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.evidence import (
    RecoveryDecisionRecord,
    RecoveryDisposition,
    RunLeaseHeartbeatRecord,
    RunLeaseRecord,
)
from oamb.contracts.ids import canonical_sha256
from oamb.runtime.resume import (
    LeaseJournal,
    ProviderLifecycleBridge,
    ResumeRejectedError,
    StaleLeaseOwner,
    host_identity_fingerprint,
)
from oamb.runtime.source_records import seal_source_contract

NOW = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)


def lease(*, epoch: int = 1, predecessor: str | None = None) -> RunLeaseRecord:
    candidate = RunLeaseRecord(
        lease_record_hash="0" * 64,
        run_id="run-1",
        provider_project_id="oamb-providers-test-a",
        provider_profile_id="mem0-rest-v1",
        lease_epoch=epoch,
        owner_id="runner-1",
        host_fingerprint=host_identity_fingerprint("host-a"),
        process_id=123 if epoch == 1 else os.getpid(),
        predecessor_lease_record_hash=predecessor,
        acquired_at=NOW + timedelta(minutes=2 * (epoch - 1)),
    )
    return candidate.model_copy(
        update={
            "lease_record_hash": canonical_sha256(
                candidate.model_dump(mode="python", exclude={"lease_record_hash"})
            )
        }
    )


def heartbeat(
    lease_record: RunLeaseRecord,
    *,
    sequence: int,
    predecessor: str | None,
) -> RunLeaseHeartbeatRecord:
    candidate = RunLeaseHeartbeatRecord(
        heartbeat_record_hash="0" * 64,
        lease_record_hash=lease_record.lease_record_hash,
        heartbeat_sequence=sequence,
        predecessor_heartbeat_hash=predecessor,
        observed_at=NOW + timedelta(seconds=sequence),
    )
    return candidate.model_copy(
        update={
            "heartbeat_record_hash": canonical_sha256(
                candidate.model_dump(mode="python", exclude={"heartbeat_record_hash"})
            )
        }
    )


def recovery_decision(
    previous: RunLeaseRecord,
    new: RunLeaseRecord,
) -> RecoveryDecisionRecord:
    candidate = RecoveryDecisionRecord(
        recovery_decision_id="0" * 64,
        run_id=previous.run_id,
        previous_lease_record_hash=previous.lease_record_hash,
        new_lease_record_hash=new.lease_record_hash,
        previous_lease_epoch=previous.lease_epoch,
        new_lease_epoch=new.lease_epoch,
        authorization_id="same_host_process_absent",
        archived_checkpoint_hashes=(),
        disposition=RecoveryDisposition.RESUME_SAFE,
        decided_at=NOW + timedelta(minutes=1),
    )
    return candidate.model_copy(
        update={
            "recovery_decision_id": canonical_sha256(
                candidate.model_dump(
                    mode="python",
                    exclude={"recovery_decision_id"},
                )
            )
        }
    )


def test_lease_journal_seals_canonical_lease_before_provider_pointer(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    provider_runtime = tmp_path / "provider-runtime"
    journal = LeaseJournal(store, ProviderLifecycleBridge(provider_runtime))
    record = lease()

    journal.acquire(record)

    assert (store.root / "source/run-leases/1.json").is_file()
    assert (provider_runtime / "active-operation").is_file()
    assert journal.current_lease_hash == record.lease_record_hash

    journal.release()

    assert not (provider_runtime / "active-operation").exists()
    assert (store.root / "source/run-leases/1.json").is_file()
    assert journal.current_lease_hash is None


def test_released_lease_record_cannot_be_reused_as_a_new_acquisition(tmp_path: Path) -> None:
    journal = LeaseJournal(
        ArtifactStore(tmp_path / "capsule"),
        ProviderLifecycleBridge(tmp_path / "provider-runtime"),
    )
    record = lease()
    journal.acquire(record)
    journal.release()

    with pytest.raises(ResumeRejectedError, match="already sealed"):
        journal.acquire(record)

    assert not (tmp_path / "provider-runtime" / "active-operation").exists()


def test_stale_lease_recovery_seals_audited_chain_before_pointer_switch(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    runtime = tmp_path / "provider-runtime"
    previous = lease()
    LeaseJournal(store, ProviderLifecycleBridge(runtime)).acquire(previous)
    successor = lease(epoch=2, predecessor=previous.lease_record_hash)
    decision = recovery_decision(previous, successor)
    recovering = LeaseJournal(store, ProviderLifecycleBridge(runtime))

    recovering.recover_stale(
        previous=previous,
        successor=successor,
        decision=decision,
        owner=StaleLeaseOwner(
            host="host-a",
            host_fingerprint=previous.host_fingerprint,
            process_id=previous.process_id,
        ),
        current_host="host-a",
        process_is_alive=lambda _process_id: False,
        explicitly_authorized=False,
    )

    assert (store.root / "source/run-leases/2.json").is_file()
    assert (
        store.root / f"source/recovery-decisions/{decision.recovery_decision_id}.json"
    ).is_file()
    pointer = (runtime / "active-operation").read_text(encoding="utf-8")
    assert successor.lease_record_hash in pointer
    assert previous.lease_record_hash not in pointer
    assert recovering.current_lease_hash == successor.lease_record_hash


def test_stale_lease_recovery_rejects_a_live_owner_without_writes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    runtime = tmp_path / "provider-runtime"
    previous = lease()
    LeaseJournal(store, ProviderLifecycleBridge(runtime)).acquire(previous)
    successor = lease(epoch=2, predecessor=previous.lease_record_hash)
    decision = recovery_decision(previous, successor)

    with pytest.raises(ResumeRejectedError, match="still alive"):
        LeaseJournal(store, ProviderLifecycleBridge(runtime)).recover_stale(
            previous=previous,
            successor=successor,
            decision=decision,
            owner=StaleLeaseOwner(
                host="host-a",
                host_fingerprint=previous.host_fingerprint,
                process_id=previous.process_id,
            ),
            current_host="host-a",
            process_is_alive=lambda process_id: process_id == previous.process_id,
            explicitly_authorized=False,
        )

    assert not (store.root / "source/run-leases/2.json").exists()
    assert not (store.root / "source/recovery-decisions").exists()
    assert previous.lease_record_hash in (runtime / "active-operation").read_text(encoding="utf-8")


def test_stale_recovery_rejects_a_caller_tampered_previous_lease(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    runtime = tmp_path / "provider-runtime"
    previous = lease()
    LeaseJournal(store, ProviderLifecycleBridge(runtime)).acquire(previous)
    tampered = previous.model_copy(update={"process_id": 999})
    successor = lease(epoch=2, predecessor=previous.lease_record_hash)

    with pytest.raises(ResumeRejectedError, match="canonical stored evidence"):
        LeaseJournal(store, ProviderLifecycleBridge(runtime)).recover_stale(
            previous=tampered,
            successor=successor,
            decision=recovery_decision(previous, successor),
            owner=StaleLeaseOwner(
                host="host-a",
                host_fingerprint=previous.host_fingerprint,
                process_id=previous.process_id,
            ),
            current_host="host-a",
            process_is_alive=lambda _process_id: False,
            explicitly_authorized=False,
        )

    assert not (store.root / "source/run-leases/2.json").exists()


def test_stale_recovery_binds_authorization_and_current_successor_owner(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    runtime = tmp_path / "provider-runtime"
    previous = lease()
    LeaseJournal(store, ProviderLifecycleBridge(runtime)).acquire(previous)
    successor = lease(epoch=2, predecessor=previous.lease_record_hash)
    owner = StaleLeaseOwner(
        host="host-a",
        host_fingerprint=previous.host_fingerprint,
        process_id=previous.process_id,
    )

    wrong_authorization = recovery_decision(previous, successor).model_copy(
        update={"authorization_id": "different"}
    )
    with pytest.raises(ResumeRejectedError, match="authorization"):
        LeaseJournal(store, ProviderLifecycleBridge(runtime)).recover_stale(
            previous=previous,
            successor=successor,
            decision=wrong_authorization,
            owner=owner,
            current_host="host-a",
            process_is_alive=lambda _process_id: False,
            explicitly_authorized=False,
        )

    wrong_owner = successor.model_copy(update={"process_id": os.getpid() + 1})
    with pytest.raises(ResumeRejectedError, match="current owner"):
        LeaseJournal(store, ProviderLifecycleBridge(runtime)).recover_stale(
            previous=previous,
            successor=wrong_owner,
            decision=recovery_decision(previous, wrong_owner),
            owner=owner,
            current_host="host-a",
            process_is_alive=lambda _process_id: False,
            explicitly_authorized=False,
        )


def test_stale_recovery_resumes_identical_partial_and_completed_wal_states(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    runtime = tmp_path / "provider-runtime"
    previous = lease()
    LeaseJournal(store, ProviderLifecycleBridge(runtime)).acquire(previous)
    successor = lease(epoch=2, predecessor=previous.lease_record_hash)
    decision = recovery_decision(previous, successor)
    seal_source_contract(
        store,
        relative_path=f"source/recovery-decisions/{decision.recovery_decision_id}.json",
        record_id=decision.recovery_decision_id,
        record=decision,
    )
    owner = StaleLeaseOwner(
        host="host-a",
        host_fingerprint=previous.host_fingerprint,
        process_id=previous.process_id,
    )

    first = LeaseJournal(store, ProviderLifecycleBridge(runtime))
    first.recover_stale(
        previous=previous,
        successor=successor,
        decision=decision,
        owner=owner,
        current_host="host-a",
        process_is_alive=lambda _process_id: False,
        explicitly_authorized=False,
    )
    second = LeaseJournal(store, ProviderLifecycleBridge(runtime))
    second.recover_stale(
        previous=previous,
        successor=successor,
        decision=decision,
        owner=owner,
        current_host="host-a",
        process_is_alive=lambda _process_id: False,
        explicitly_authorized=False,
    )

    assert second.current_lease_hash == successor.lease_record_hash
    assert not hasattr(ProviderLifecycleBridge(runtime), "replace_stale_run")


def test_attempt_pointer_clear_requires_the_current_in_process_lease_authority(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "provider-runtime"
    owner = ProviderLifecycleBridge(runtime)
    owner.acquire_run(
        run_id="run-1",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )
    owner.mark_attempt_dispatched(attempt_id="b" * 64, intent_record_hash="c" * 64)

    with pytest.raises(ResumeRejectedError, match="current operation authority"):
        ProviderLifecycleBridge(runtime).clear_attempt_after_receipt(
            attempt_id="b" * 64,
            expected_intent_record_hash="c" * 64,
        )

    assert (runtime / "active-provider-attempts" / f"{'b' * 64}.json").exists()


def test_lease_heartbeat_is_append_only_and_predecessor_bound(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    journal = LeaseJournal(
        store,
        ProviderLifecycleBridge(tmp_path / "provider-runtime"),
    )
    lease_record = lease()
    journal.acquire(lease_record)
    first = heartbeat(lease_record, sequence=1, predecessor=None)
    second = heartbeat(
        lease_record,
        sequence=2,
        predecessor=first.heartbeat_record_hash,
    )

    journal.append_heartbeat(first)
    journal.append_heartbeat(second)

    assert (store.root / "source/run-lease-heartbeats/1-1.json").is_file()
    assert (store.root / "source/run-lease-heartbeats/1-2.json").is_file()
    fork = heartbeat(lease_record, sequence=3, predecessor="f" * 64)
    with pytest.raises(ResumeRejectedError, match="predecessor"):
        journal.append_heartbeat(fork)
    assert not (store.root / "source/run-lease-heartbeats/1-3.json").exists()
