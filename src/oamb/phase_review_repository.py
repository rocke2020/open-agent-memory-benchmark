"""No-follow artifact-repository binding for one phase-review occurrence."""

from __future__ import annotations

import os
import socket
import stat
from pathlib import Path

from oamb.contracts.ids import canonical_sha256


def prepare_artifact_repository(path: Path) -> tuple[Path, str]:
    repository = Path(os.path.abspath(os.fspath(path)))
    metadata = _open_or_create_directory_no_follow(repository)
    fingerprint = canonical_sha256(
        [
            "oamb-phase-review-artifact-repository-v1",
            repository.as_posix(),
            socket.gethostname(),
            metadata.st_dev,
            metadata.st_ino,
        ]
    )
    return repository, fingerprint


def phase_review_occurrence_root(repository: Path, occurrence_id: str) -> Path:
    root = repository / "phase-reviews" / occurrence_id
    _open_or_create_directory_no_follow(root)
    return root


def phase_review_runtime_directory(repository: Path) -> Path:
    runtime = repository / ".runtime" / "phase-review"
    _open_or_create_directory_no_follow(runtime)
    return runtime


def _open_or_create_directory_no_follow(path: Path) -> os.stat_result:
    if not path.is_absolute():
        raise ValueError("artifact repository path must be absolute after normalization")
    descriptor = os.open(
        path.anchor,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        for part in path.parts[1:]:
            try:
                child = os.open(
                    part,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except FileNotFoundError:
                os.mkdir(part, mode=0o700, dir_fd=descriptor)
                os.fsync(descriptor)
                child = os.open(
                    part,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                    dir_fd=descriptor,
                )
            except OSError as exc:
                raise ValueError(
                    f"artifact repository path contains a symlink or non-directory: {path}"
                ) from exc
            os.close(descriptor)
            descriptor = child
        metadata = os.fstat(descriptor)
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("artifact repository root is not a directory")
        return metadata
    finally:
        os.close(descriptor)


__all__ = [
    "phase_review_occurrence_root",
    "phase_review_runtime_directory",
    "prepare_artifact_repository",
]
