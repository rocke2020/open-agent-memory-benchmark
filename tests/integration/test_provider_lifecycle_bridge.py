from __future__ import annotations

import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.evidence import RunLeaseRecord
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import ArtifactSealReceipt, ArtifactWriteRequest
from oamb.runtime.resume import LeaseJournal, ProviderLifecycleBridge, ResumeRejectedError

ROOT = Path(__file__).resolve().parents[2]
LIFECYCLE = ROOT / "provider-services" / "lib" / "lifecycle.sh"


def shell_lifecycle_acquire(runtime: Path) -> subprocess.CompletedProcess[str]:
    program = """
set -eu
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
    assert "active OAMB provider operation" in blocked.stderr
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
    (runtime / "active-operation").unlink()

    blocked = shell_lifecycle_acquire(runtime)

    assert blocked.returncode != 0
    assert "active provider attempt" in blocked.stderr
    assert not (runtime / "provider-lifecycle.lock").exists()


def test_provider_domains_overlap_but_same_domain_cannot_reenter(
    tmp_path: Path,
) -> None:
    runtime = tmp_path / "provider-runtime"
    profiles = (
        ("hindsight-rest-v1", "run-hindsight", "a" * 64),
        ("mem0-rest-v1", "run-mem0", "b" * 64),
        ("openviking-session-rest-v1", "run-openviking", "c" * 64),
    )
    bridges: list[ProviderLifecycleBridge] = []
    authorities = []
    for profile_id, run_id, lease_record_hash in profiles:
        domain = runtime / "lifecycle-domains" / profile_id
        bridge = ProviderLifecycleBridge(domain, coordination_directory=runtime)
        authority = bridge.acquire_run(
            run_id=run_id,
            provider_project="oamb-providers-test-a",
            profile_id=profile_id,
            lease_epoch=1,
            lease_record_hash=lease_record_hash,
        )
        bridges.append(bridge)
        authorities.append(authority)
        assert (domain / "active-operation").is_file()

    mem0_runtime = runtime / "lifecycle-domains" / "mem0-rest-v1"
    competing_mem0 = ProviderLifecycleBridge(mem0_runtime, coordination_directory=runtime)
    with pytest.raises(ResumeRejectedError, match="active provider operation"):
        competing_mem0.acquire_run(
            run_id="run-mem0-duplicate",
            provider_project="oamb-providers-test-a",
            profile_id="mem0-rest-v1",
            lease_epoch=1,
            lease_record_hash="d" * 64,
        )

    blocked = shell_lifecycle_acquire(runtime)
    assert blocked.returncode != 0
    assert "active OAMB provider operation" in blocked.stderr
    assert not (runtime / "provider-lifecycle.lock").exists()

    for bridge, authority in zip(bridges, authorities, strict=True):
        bridge.release_run(authority)
    acquired = shell_lifecycle_acquire(runtime)
    assert acquired.returncode == 0, acquired.stderr


def test_root_stack_mutation_lock_blocks_domain_acquisition(tmp_path: Path) -> None:
    runtime = tmp_path / "provider-runtime"
    runtime.mkdir()
    root_lock = runtime / "provider-lifecycle.lock"
    root_lock.mkdir()
    domain = runtime / "lifecycle-domains" / "mem0-rest-v1"
    bridge = ProviderLifecycleBridge(domain, coordination_directory=runtime)

    with pytest.raises(ResumeRejectedError, match="provider lifecycle operation"):
        bridge.acquire_run(
            run_id="run-mem0",
            provider_project="oamb-providers-test-a",
            profile_id="mem0-rest-v1",
            lease_epoch=1,
            lease_record_hash="a" * 64,
        )

    assert not (domain / "active-operation").exists()


def test_domain_admission_blocks_racing_stack_mutation(tmp_path: Path) -> None:
    runtime = tmp_path / "provider-runtime"
    domain = runtime / "lifecycle-domains" / "mem0-rest-v1"
    bridge = ProviderLifecycleBridge(domain, coordination_directory=runtime)
    admission_started = threading.Event()
    allow_pointer = threading.Event()
    authorities = []

    def acquire_domain() -> None:
        def block_before_pointer() -> None:
            admission_started.set()
            assert allow_pointer.wait(timeout=2)

        authorities.append(
            bridge.acquire_run(
                run_id="run-mem0",
                provider_project="oamb-providers-test-a",
                profile_id="mem0-rest-v1",
                lease_epoch=1,
                lease_record_hash="a" * 64,
                durable_lease=block_before_pointer,
            )
        )

    thread = threading.Thread(target=acquire_domain)
    thread.start()
    assert admission_started.wait(timeout=2)

    blocked = shell_lifecycle_acquire(runtime)

    assert blocked.returncode != 0
    assert "provider lifecycle domain admission" in blocked.stderr
    assert not (runtime / "provider-lifecycle.lock").exists()
    allow_pointer.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert (domain / "active-operation").is_file()
    bridge.release_run(authorities[0])


@pytest.mark.parametrize(
    "symbolic_component",
    ("coordination-root", "domains-parent", "provider-domain"),
)
def test_symbolic_lifecycle_path_rejects_before_pointer_creation(
    tmp_path: Path,
    symbolic_component: str,
) -> None:
    runtime = tmp_path / "provider-runtime"
    runtime.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    coordination = runtime
    domains_parent = runtime / "lifecycle-domains"
    domain = domains_parent / "mem0-rest-v1"
    if symbolic_component == "coordination-root":
        coordination = tmp_path / "coordination"
        coordination.symlink_to(external, target_is_directory=True)
    elif symbolic_component == "domains-parent":
        domains_parent.symlink_to(external, target_is_directory=True)
    else:
        domains_parent.mkdir()
        domain.symlink_to(external, target_is_directory=True)
    bridge = ProviderLifecycleBridge(domain, coordination_directory=coordination)

    with pytest.raises(ResumeRejectedError, match="unsafe|symbolic"):
        bridge.acquire_run(
            run_id="run-mem0",
            provider_project="oamb-providers-test-a",
            profile_id="mem0-rest-v1",
            lease_epoch=1,
            lease_record_hash="a" * 64,
        )

    assert not (external / "active-operation").exists()
    assert not (external / "mem0-rest-v1" / "active-operation").exists()


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
    assert (runtime / "active-operation").is_file()
