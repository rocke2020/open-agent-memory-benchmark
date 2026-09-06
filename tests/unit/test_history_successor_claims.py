from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
from typing import Any, cast

import pytest

from oamb.runtime.resume import ProviderLifecycleBridge, ResumeRejectedError

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def _bridge(root: Path, *, profile_id: str = "mem0-rest-v1") -> ProviderLifecycleBridge:
    runtime = root / "lifecycle-domains" / profile_id
    bridge = ProviderLifecycleBridge(runtime, coordination_directory=root)
    bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id=profile_id,
        lease_epoch=1,
        lease_record_hash=SHA_A,
    )
    return bridge


def _claim(bridge: ProviderLifecycleBridge, **overrides: object) -> bytes:
    values: dict[str, object] = {
        "retry_event_id": SHA_B,
        "retry_event_sha256": SHA_C,
        "ingestion_plan_id": SHA_D,
        "successor_ingestion_occurrence_id": SHA_E,
        "claimant_run_id": "run-a",
        "claimant_capsule_id": "run-a",
        "source_part_manifest_bindings": ((SHA_A, SHA_B),),
    }
    values.update(overrides)
    return bridge.claim_history_successor(**values)  # type: ignore[arg-type]


def test_claim_is_canonical_idempotent_in_process_and_survives_release(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    first = _claim(bridge)
    assert _claim(bridge) == first
    document = json.loads(first)
    assert document["provider_project"] == "oamb-providers-test-a"
    assert document["profile_id"] == "mem0-rest-v1"
    assert document["operation_record_sha256"] == SHA_A
    claim_path = tmp_path / "history-successor-claims" / f"{SHA_B}.json"
    assert claim_path.read_bytes() == first

    authority = bridge._active_authority
    assert authority is not None
    bridge.release_run(cast(Any, authority))
    assert claim_path.read_bytes() == first


def test_conflicting_or_omitted_source_branch_cannot_reclaim_event(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    _claim(bridge)
    with pytest.raises(ResumeRejectedError, match="already claimed"):
        _claim(bridge, source_part_manifest_bindings=())
    with pytest.raises(ResumeRejectedError, match="already claimed"):
        _claim(bridge, successor_ingestion_occurrence_id="1" * 64)


def test_claim_rejects_malformed_values_and_unsafe_claim_directory(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path)
    with pytest.raises(ResumeRejectedError, match="SHA-256"):
        _claim(bridge, retry_event_id="short")
    with pytest.raises(ResumeRejectedError, match="pre-final run identity"):
        _claim(bridge, claimant_capsule_id="other-run")
    with pytest.raises(ResumeRejectedError, match="canonical"):
        _claim(
            bridge,
            source_part_manifest_bindings=((SHA_B, SHA_C), (SHA_A, SHA_B)),
        )

    unsafe_root = tmp_path / "unsafe"
    unsafe_root.mkdir()
    (unsafe_root / "history-successor-claims").symlink_to(tmp_path, target_is_directory=True)
    unsafe_bridge = _bridge(unsafe_root)
    with pytest.raises(ResumeRejectedError, match="unsafe|symbolic"):
        _claim(unsafe_bridge)


def test_claim_is_bound_to_current_domain_owner(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, profile_id="mem0-rest-v1")
    document = json.loads(_claim(bridge))
    assert document["runtime_domain"] == "lifecycle-domains/mem0-rest-v1"

    other_root = tmp_path / "other"
    other = _bridge(other_root, profile_id="openviking-session-v1")
    other_document = json.loads(_claim(other))
    assert other_document["runtime_domain"] == "lifecycle-domains/openviking-session-v1"
    assert other_document["profile_id"] == "openviking-session-v1"


def _claim_in_child(
    bridge: ProviderLifecycleBridge, queue: object, bindings: tuple[tuple[str, str], ...]
) -> None:
    try:
        _claim(bridge, source_part_manifest_bindings=bindings)
    except BaseException as exc:
        queue.put(("error", type(exc).__name__))  # type: ignore[attr-defined]
    else:
        queue.put(("ok", ""))  # type: ignore[attr-defined]


def test_separate_process_branches_have_exactly_one_claim_winner(tmp_path: Path) -> None:
    try:
        context = multiprocessing.get_context("fork")
    except ValueError:
        pytest.skip("history successor claim concurrency requires fork")
    bridge = _bridge(tmp_path)
    queue = context.Queue()
    processes = (
        context.Process(target=_claim_in_child, args=(bridge, queue, ((SHA_A, SHA_B),))),
        context.Process(target=_claim_in_child, args=(bridge, queue, ())),
    )
    for process in processes:
        process.start()
    results = tuple(queue.get(timeout=5) for _ in processes)
    for process in processes:
        process.join(5)
        assert process.exitcode == 0
    assert sorted(status for status, _detail in results) == ["error", "ok"]
