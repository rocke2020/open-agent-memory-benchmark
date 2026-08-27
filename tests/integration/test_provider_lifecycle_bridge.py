from __future__ import annotations

import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.evidence import RunLeaseRecord
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import ArtifactSealReceipt, ArtifactWriteRequest
from oamb.runtime.resume import LeaseJournal, ProviderLifecycleBridge

ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE = ROOT / "provider-services" / "lib" / "lifecycle.sh"


def shell_lifecycle_acquire(runtime: Path) -> subprocess.CompletedProcess[str]:
    program = """
die() { printf '%s\n' "$*" >&2; exit 1; }
RUNTIME_DIR=$1
LIFECYCLE_LOCK="$RUNTIME_DIR/provider-lifecycle.lock"
LIFECYCLE_LOCK_HELD=false
. "$2"
acquire_lifecycle_lock
printf 'acquired\n'
"""
    return subprocess.run(
        ["sh", "-c", program, "sh", str(runtime), str(LIFECYCLE)],
        check=False,
        capture_output=True,
        text=True,
    )


def test_python_run_pointer_blocks_shell_lifecycle_mutation(tmp_path: Path) -> None:
    runtime = tmp_path / "provider-runtime"
    bridge = ProviderLifecycleBridge(runtime)
    authority = bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )

    blocked = shell_lifecycle_acquire(runtime)

    assert blocked.returncode != 0
    assert "active OAMB run lease" in blocked.stderr
    assert not (runtime / "provider-lifecycle.lock").exists()

    bridge.release_run(authority)
    acquired = shell_lifecycle_acquire(runtime)
    assert acquired.returncode == 0, acquired.stderr
    assert acquired.stdout == "acquired\n"


def test_python_dispatched_attempt_pointer_blocks_shell_lifecycle_mutation(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "provider-runtime"
    bridge = ProviderLifecycleBridge(runtime)
    bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )
    bridge.mark_attempt_dispatched(attempt_id="b" * 64, intent_record_hash="c" * 64)
    (runtime / "active-run-lease").unlink()

    blocked = shell_lifecycle_acquire(runtime)

    assert blocked.returncode != 0
    assert "active provider attempt" in blocked.stderr
    assert not (runtime / "provider-lifecycle.lock").exists()


def test_lifecycle_lock_covers_durable_lease_seal_and_pointer_creation(
    tmp_path: Path,
) -> None:
    seal_started = threading.Event()
    allow_seal = threading.Event()

    class BlockingStore(ArtifactStore):
        def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
            seal_started.set()
            assert allow_seal.wait(timeout=2)
            return super().seal_source_record(request)

    candidate = RunLeaseRecord(
        lease_record_hash="0" * 64,
        run_id="run-a",
        provider_project_id="oamb-providers-test-a",
        provider_profile_id="mem0-rest-v1",
        lease_epoch=1,
        owner_id="runner-1",
        host_fingerprint="d" * 64,
        process_id=123,
        predecessor_lease_record_hash=None,
        acquired_at=datetime(2026, 8, 27, 12, 0, tzinfo=UTC),
    )
    lease = candidate.model_copy(
        update={
            "lease_record_hash": canonical_sha256(
                candidate.model_dump(mode="python", exclude={"lease_record_hash"})
            )
        }
    )
    runtime = tmp_path / "provider-runtime"
    journal = LeaseJournal(
        BlockingStore(tmp_path / "capsule"),
        ProviderLifecycleBridge(runtime),
    )
    thread = threading.Thread(target=journal.acquire, args=(lease,))
    thread.start()
    assert seal_started.wait(timeout=2)

    racing_lifecycle = shell_lifecycle_acquire(runtime)

    allow_seal.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert racing_lifecycle.returncode != 0
    assert "another provider lifecycle operation" in racing_lifecycle.stderr
    assert (runtime / "active-run-lease").is_file()
