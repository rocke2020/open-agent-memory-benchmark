from __future__ import annotations

import concurrent.futures
import json
import threading
from pathlib import Path

import pytest

from oamb.runtime.resume import ProviderLifecycleBridge, ResumeRejectedError


def test_one_run_tracks_parallel_provider_attempts_independently(tmp_path: Path) -> None:
    bridge = ProviderLifecycleBridge(tmp_path)
    authority = bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )

    bridge.mark_attempt_dispatched(attempt_id="b" * 64, intent_record_hash="c" * 64)
    bridge.mark_attempt_dispatched(attempt_id="d" * 64, intent_record_hash="e" * 64)

    pointers = tmp_path / "active-provider-attempts"
    assert sorted(path.name for path in pointers.iterdir()) == [
        f"{'b' * 64}.json",
        f"{'d' * 64}.json",
    ]
    assert (
        json.loads((pointers / f"{'b' * 64}.json").read_text(encoding="utf-8"))[
            "intent_record_sha256"
        ]
        == "c" * 64
    )
    with pytest.raises(ResumeRejectedError, match="provider attempt"):
        bridge.release_run(authority)

    bridge.clear_attempt_after_receipt(
        attempt_id="b" * 64,
        expected_intent_record_hash="c" * 64,
    )
    assert not (pointers / f"{'b' * 64}.json").exists()
    assert (pointers / f"{'d' * 64}.json").exists()

    bridge.clear_attempt_after_receipt(
        attempt_id="d" * 64,
        expected_intent_record_hash="e" * 64,
    )
    bridge.release_run(authority)

    assert not any(pointers.iterdir())
    assert not (tmp_path / "active-operation").exists()


def test_distinct_provider_domains_can_hold_runs_under_one_project_gate(
    tmp_path: Path,
) -> None:
    domains = tmp_path / "lifecycle-domains"
    hindsight = ProviderLifecycleBridge(
        domains / "hindsight",
        coordination_directory=tmp_path,
    )
    mem0 = ProviderLifecycleBridge(
        domains / "mem0",
        coordination_directory=tmp_path,
    )

    hindsight_authority = hindsight.acquire_run(
        run_id="run-hindsight",
        provider_project="oamb-providers-test-a",
        profile_id="hindsight-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )
    mem0_authority = mem0.acquire_run(
        run_id="run-mem0",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="b" * 64,
    )

    assert (domains / "hindsight" / "active-operation").is_file()
    assert (domains / "mem0" / "active-operation").is_file()
    hindsight.release_run(hindsight_authority)
    mem0.release_run(mem0_authority)


def test_project_mutation_gate_blocks_new_provider_domain_acquisition(tmp_path: Path) -> None:
    bridge = ProviderLifecycleBridge(
        tmp_path / "lifecycle-domains" / "hindsight",
        coordination_directory=tmp_path,
    )
    (tmp_path / "provider-lifecycle.lock").mkdir()

    with pytest.raises(ResumeRejectedError, match="lifecycle operation"):
        bridge.acquire_run(
            run_id="run-hindsight",
            provider_project="oamb-providers-test-a",
            profile_id="hindsight-rest-v1",
            lease_epoch=1,
            lease_record_hash="a" * 64,
        )

    assert not (tmp_path / "lifecycle-domains" / "hindsight" / "active-operation").exists()


def test_distinct_provider_domains_cross_the_admission_gate_concurrently(
    tmp_path: Path,
) -> None:
    domains = tmp_path / "lifecycle-domains"
    bridges = (
        ProviderLifecycleBridge(domains / "hindsight", coordination_directory=tmp_path),
        ProviderLifecycleBridge(domains / "mem0", coordination_directory=tmp_path),
    )
    barrier = threading.Barrier(2, timeout=1)

    def acquire(index: int) -> object:
        def durable_lease() -> None:
            barrier.wait()

        return bridges[index].acquire_run(
            run_id=f"run-{index}",
            provider_project="oamb-providers-test-a",
            profile_id=("hindsight-rest-v1", "mem0-rest-v1")[index],
            lease_epoch=1,
            lease_record_hash=("a", "b")[index] * 64,
            durable_lease=durable_lease,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        authorities = tuple(executor.map(acquire, range(2)))

    for bridge, authority in zip(bridges, authorities, strict=True):
        bridge.release_run(authority)  # type: ignore[arg-type]
