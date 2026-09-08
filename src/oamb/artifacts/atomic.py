"""Create-only durable file publication."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

HASH_READ_CHUNK_BYTES = 1024 * 1024


class ArtifactCollisionError(FileExistsError):
    """A sealed target exists with bytes different from the candidate."""


class AtomicWriteBoundary(StrEnum):
    AFTER_PARENT_DIRECTORY_FSYNC = "after_parent_directory_fsync"
    AFTER_TEMPORARY_WRITE = "after_temporary_write"
    AFTER_FILE_FSYNC = "after_file_fsync"
    AFTER_TARGET_PUBLISH = "after_target_publish"
    AFTER_TARGET_DIRECTORY_FSYNC = "after_target_directory_fsync"
    AFTER_TEMPORARY_CLEANUP = "after_temporary_cleanup"
    AFTER_CLEANUP_DIRECTORY_FSYNC = "after_cleanup_directory_fsync"


ATOMIC_WRITE_BOUNDARIES = tuple(AtomicWriteBoundary)


class AtomicReplaceBoundary(StrEnum):
    AFTER_TEMPORARY_WRITE = "after_temporary_write"
    AFTER_FILE_FSYNC = "after_file_fsync"
    AFTER_TARGET_REPLACE = "after_target_replace"
    AFTER_TARGET_DIRECTORY_FSYNC = "after_target_directory_fsync"


ATOMIC_REPLACE_BOUNDARIES = tuple(AtomicReplaceBoundary)

FaultHook = Callable[[AtomicWriteBoundary, Path], None]
ReplaceFaultHook = Callable[[AtomicReplaceBoundary, Path], None]


@dataclass(frozen=True)
class AtomicWriteResult:
    path: Path
    sha256: str
    byte_count: int
    created: bool


def atomic_replace_bytes(
    target: Path,
    content: bytes,
    *,
    fault_hook: ReplaceFaultHook | None = None,
    trusted_root: Path | None = None,
) -> AtomicWriteResult:
    """Durably publish mutable ``content`` with one same-directory replacement."""

    target = Path(target).absolute()
    lexical_root = Path(trusted_root).absolute() if trusted_root is not None else target.parent
    _require_lexical_containment(lexical_root, target.parent)
    resolved_root = lexical_root.resolve(strict=False)
    resolved_parent = target.parent.resolve(strict=False)
    if not resolved_parent.is_relative_to(resolved_root):
        raise ArtifactCollisionError("artifact path escapes its trusted root")
    target = resolved_parent / target.name
    _ensure_durable_directory(
        target.parent,
        trusted_root=resolved_root,
        target=target,
        fault_hook=None,
    )
    try:
        metadata = target.lstat()
    except FileNotFoundError:
        created = True
    else:
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactCollisionError(f"{target} is a symbolic link or non-regular file")
        created = False

    expected_sha256 = hashlib.sha256(content).hexdigest()
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    temporary_path = Path(temporary_name)
    try:
        _write_all(file_descriptor, content)
        _notify_replace(fault_hook, AtomicReplaceBoundary.AFTER_TEMPORARY_WRITE, target)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    _notify_replace(fault_hook, AtomicReplaceBoundary.AFTER_FILE_FSYNC, target)
    os.replace(temporary_path, target)
    _notify_replace(fault_hook, AtomicReplaceBoundary.AFTER_TARGET_REPLACE, target)
    _fsync_directory(target.parent)
    _notify_replace(fault_hook, AtomicReplaceBoundary.AFTER_TARGET_DIRECTORY_FSYNC, target)
    return AtomicWriteResult(
        path=target,
        sha256=expected_sha256,
        byte_count=len(content),
        created=created,
    )


def atomic_write_bytes(
    target: Path,
    content: bytes,
    *,
    fault_hook: FaultHook | None = None,
    trusted_root: Path | None = None,
) -> AtomicWriteResult:
    """Durably publish ``content`` without replacing an existing target.

    The hard link is the create-only commit point.  A fault raised by the hook
    deliberately leaves the same-filesystem temporary file in place so crash
    recovery can compare it with any published target.
    """

    target = Path(target).absolute()
    lexical_root = Path(trusted_root).absolute() if trusted_root is not None else target.parent
    _require_lexical_containment(lexical_root, target.parent)
    resolved_root = lexical_root.resolve(strict=False)
    resolved_parent = target.parent.resolve(strict=False)
    if not resolved_parent.is_relative_to(resolved_root):
        raise ArtifactCollisionError("artifact path escapes its trusted root")
    target = resolved_parent / target.name
    _ensure_durable_directory(
        target.parent,
        trusted_root=resolved_root,
        target=target,
        fault_hook=fault_hook,
    )
    expected_sha256 = hashlib.sha256(content).hexdigest()
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.tmp-", dir=target.parent
    )
    temporary_path = Path(temporary_name)
    try:
        _write_all(file_descriptor, content)
        _notify(fault_hook, AtomicWriteBoundary.AFTER_TEMPORARY_WRITE, target)
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)
    _notify(fault_hook, AtomicWriteBoundary.AFTER_FILE_FSYNC, target)

    created = True
    try:
        os.link(temporary_path, target)
    except FileExistsError:
        created = False
        try:
            actual_sha256, actual_size = _sha256_regular_file(target)
            if actual_sha256 != expected_sha256 or actual_size != len(content):
                raise ArtifactCollisionError(
                    f"{target} already exists with different bytes"
                ) from None
        except BaseException:
            temporary_path.unlink()
            _fsync_directory(target.parent)
            raise
    _notify(fault_hook, AtomicWriteBoundary.AFTER_TARGET_PUBLISH, target)

    _fsync_directory(target.parent)
    _notify(fault_hook, AtomicWriteBoundary.AFTER_TARGET_DIRECTORY_FSYNC, target)
    temporary_path.unlink()
    _notify(fault_hook, AtomicWriteBoundary.AFTER_TEMPORARY_CLEANUP, target)
    _fsync_directory(target.parent)
    _notify(fault_hook, AtomicWriteBoundary.AFTER_CLEANUP_DIRECTORY_FSYNC, target)
    return AtomicWriteResult(
        path=target,
        sha256=expected_sha256,
        byte_count=len(content),
        created=created,
    )


def _write_all(file_descriptor: int, content: bytes) -> None:
    view = memoryview(content)
    written = 0
    while written < len(view):
        count = os.write(file_descriptor, view[written:])
        if count <= 0:
            raise OSError("temporary artifact write made no progress")
        written += count


def _ensure_durable_directory(
    directory: Path,
    *,
    trusted_root: Path,
    target: Path,
    fault_hook: FaultHook | None,
) -> None:
    if not directory.is_relative_to(trusted_root):
        raise ArtifactCollisionError("artifact directory escapes its trusted root")
    missing: list[Path] = []
    cursor = directory
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                raise ArtifactCollisionError(
                    "artifact directory has no existing ancestor"
                ) from None
            cursor = parent
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise ArtifactCollisionError(
                f"{cursor} is a symbolic link or non-directory artifact parent"
            )
        break

    for path in reversed(missing):
        try:
            path.mkdir()
        except FileExistsError:
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactCollisionError(
                    f"{path} is a symbolic link or non-directory artifact parent"
                ) from None
        _fsync_directory(path.parent)
        _notify(fault_hook, AtomicWriteBoundary.AFTER_PARENT_DIRECTORY_FSYNC, target)


def _require_lexical_containment(root: Path, directory: Path) -> None:
    if not directory.is_relative_to(root):
        raise ArtifactCollisionError("artifact path is outside its trusted root")
    current = root
    relative = directory.relative_to(root)
    for component in (None, *relative.parts):
        if component is not None:
            current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ArtifactCollisionError(f"{current} is a symbolic link in the artifact path")


def sha256_file(path: Path) -> str:
    return _sha256_regular_file(path)[0]


def read_regular_file(path: Path) -> bytes:
    file_descriptor, _metadata = _open_regular_file(path)
    chunks: list[bytes] = []
    try:
        while chunk := os.read(file_descriptor, HASH_READ_CHUNK_BYTES):
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(file_descriptor)


def _sha256_regular_file(path: Path) -> tuple[str, int]:
    file_descriptor, opened = _open_regular_file(path)
    digest = hashlib.sha256()
    try:
        while chunk := os.read(file_descriptor, HASH_READ_CHUNK_BYTES):
            digest.update(chunk)
        return digest.hexdigest(), opened.st_size
    finally:
        os.close(file_descriptor)


def _open_regular_file(path: Path) -> tuple[int, os.stat_result]:
    try:
        before_open = path.lstat()
    except OSError:
        raise
    if not stat.S_ISREG(before_open.st_mode):
        raise ArtifactCollisionError(f"{path} is a symbolic link or non-regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        file_descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactCollisionError(
            f"{path} is a symbolic link or changed during verification"
        ) from exc
    opened = os.fstat(file_descriptor)
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_dev != before_open.st_dev
        or opened.st_ino != before_open.st_ino
    ):
        os.close(file_descriptor)
        raise ArtifactCollisionError(f"{path} is non-regular or changed during verification")
    return file_descriptor, opened


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    file_descriptor = os.open(directory, flags)
    try:
        os.fsync(file_descriptor)
    finally:
        os.close(file_descriptor)


def _notify(hook: FaultHook | None, boundary: AtomicWriteBoundary, target: Path) -> None:
    if hook is not None:
        hook(boundary, target)


def _notify_replace(
    hook: ReplaceFaultHook | None,
    boundary: AtomicReplaceBoundary,
    target: Path,
) -> None:
    if hook is not None:
        hook(boundary, target)
