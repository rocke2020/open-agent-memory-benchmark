from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from oamb.artifacts import atomic as atomic_module
from oamb.artifacts.atomic import (
    ATOMIC_WRITE_BOUNDARIES,
    ArtifactCollisionError,
    AtomicWriteBoundary,
    atomic_write_bytes,
)


class InjectedCrash(RuntimeError):
    pass


def test_atomic_write_seals_exact_bytes_and_reports_every_boundary(tmp_path: Path) -> None:
    target = tmp_path / "records" / "attempt.json"
    observed: list[AtomicWriteBoundary] = []

    result = atomic_write_bytes(
        target,
        b'{"attempt":1}',
        fault_hook=lambda boundary, _path: observed.append(boundary),
    )

    assert target.read_bytes() == b'{"attempt":1}'
    assert result.sha256 == hashlib.sha256(b'{"attempt":1}').hexdigest()
    assert result.byte_count == 13
    assert result.created is True
    assert tuple(observed) == ATOMIC_WRITE_BOUNDARIES
    assert not tuple(target.parent.glob(".*.tmp-*"))


def test_each_new_parent_directory_has_a_durable_creation_boundary(tmp_path: Path) -> None:
    target = tmp_path / "first" / "second" / "record.json"
    observed: list[AtomicWriteBoundary] = []

    atomic_write_bytes(
        target,
        b"durable parents",
        fault_hook=lambda boundary, _path: observed.append(boundary),
    )

    assert observed[:2] == [
        AtomicWriteBoundary.AFTER_PARENT_DIRECTORY_FSYNC,
        AtomicWriteBoundary.AFTER_PARENT_DIRECTORY_FSYNC,
    ]


def test_atomic_write_allows_a_system_symlink_above_the_trusted_root() -> None:
    with tempfile.TemporaryDirectory(dir="/var/tmp") as temporary:
        trusted_root = Path(temporary)
        target = trusted_root / "records" / "record.json"

        result = atomic_write_bytes(target, b"portable", trusted_root=trusted_root)

        assert result.created is True
        assert target.read_bytes() == b"portable"


def test_atomic_write_is_idempotent_for_identical_bytes_and_rejects_collision(
    tmp_path: Path,
) -> None:
    target = tmp_path / "sealed.json"

    first = atomic_write_bytes(target, b"first")
    repeated = atomic_write_bytes(target, b"first")

    assert first.created is True
    assert repeated.created is False
    with pytest.raises(ArtifactCollisionError, match="different bytes"):
        atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"first"
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_atomic_write_rejects_an_existing_symbolic_link_target(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"sealed")
    target = tmp_path / "sealed.json"
    target.symlink_to(outside)

    with pytest.raises(ArtifactCollisionError, match="symbolic link"):
        atomic_write_bytes(target, b"sealed")

    assert target.is_symlink()
    assert outside.read_bytes() == b"sealed"
    assert not tuple(tmp_path.glob(".*.tmp-*"))


@pytest.mark.parametrize("boundary", ATOMIC_WRITE_BOUNDARIES)
def test_atomic_fault_hook_can_interrupt_each_durability_boundary(
    tmp_path: Path,
    boundary: AtomicWriteBoundary,
) -> None:
    target = tmp_path / boundary.value / "record.json"

    def crash_at(current: AtomicWriteBoundary, _path: Path) -> None:
        if current == boundary:
            raise InjectedCrash(current.value)

    with pytest.raises(InjectedCrash, match=boundary.value):
        atomic_write_bytes(target, b"durable evidence", fault_hook=crash_at)

    if boundary in {
        AtomicWriteBoundary.AFTER_TARGET_PUBLISH,
        AtomicWriteBoundary.AFTER_TARGET_DIRECTORY_FSYNC,
        AtomicWriteBoundary.AFTER_TEMPORARY_CLEANUP,
        AtomicWriteBoundary.AFTER_CLEANUP_DIRECTORY_FSYNC,
    }:
        assert target.read_bytes() == b"durable evidence"
    else:
        assert not target.exists()


def test_atomic_write_does_not_use_replacing_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "sealed.json"

    def reject_replace(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("replace-style publication is forbidden")

    monkeypatch.setattr("os.replace", reject_replace)
    monkeypatch.setattr("os.rename", reject_replace)

    atomic_write_bytes(target, b"create only")

    assert target.read_bytes() == b"create only"


def test_atomic_replace_publishes_new_complete_bytes_over_an_existing_file(
    tmp_path: Path,
) -> None:
    target = tmp_path / "progress.json"
    target.write_bytes(b"old complete progress")

    result = atomic_module.atomic_replace_bytes(target, b"new complete progress")

    assert target.read_bytes() == b"new complete progress"
    assert result.sha256 == hashlib.sha256(b"new complete progress").hexdigest()
    assert result.byte_count == len(b"new complete progress")
    assert result.created is False
    assert not tuple(tmp_path.glob(".progress.json.tmp-*"))


@pytest.mark.parametrize(
    ("boundary", "expected"),
    [
        ("after_temporary_write", b"old complete progress"),
        ("after_file_fsync", b"old complete progress"),
        ("after_target_replace", b"new complete progress"),
        ("after_target_directory_fsync", b"new complete progress"),
    ],
)
def test_atomic_replace_exposes_only_old_or_new_complete_bytes_after_a_fault(
    tmp_path: Path,
    boundary: str,
    expected: bytes,
) -> None:
    target = tmp_path / "progress.json"
    target.write_bytes(b"old complete progress")

    def crash_at(current: object, _path: Path) -> None:
        if str(current) == boundary:
            raise InjectedCrash(boundary)

    with pytest.raises(InjectedCrash, match=boundary):
        atomic_module.atomic_replace_bytes(
            target,
            b"new complete progress",
            fault_hook=crash_at,
        )

    assert target.read_bytes() == expected
