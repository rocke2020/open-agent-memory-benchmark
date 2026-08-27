"""Capsule sealing and checksum-before-marker publication."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import DerivationSpec

from .atomic import atomic_write_bytes, sha256_file
from .store import UnsafeArtifactPathError, safe_path_component, safe_relative_path

CHECKSUMS_NAME = "CHECKSUMS.sha256"
CHECKSUM_LINE = re.compile(r"^(?P<sha256>[0-9a-f]{64})  (?P<path>[^\r\n]+)$")


class PublicationBoundary(StrEnum):
    AFTER_PAYLOADS_DURABLE = "after_payloads_durable"
    AFTER_CHECKSUM_DURABLE = "after_checksum_durable"
    BEFORE_MARKER_WRITE = "before_marker_write"
    AFTER_MARKER_DURABLE = "after_marker_durable"


PUBLICATION_BOUNDARIES = tuple(PublicationBoundary)
PublicationFaultHook = Callable[[PublicationBoundary, Path], None]


@dataclass(frozen=True)
class PublicationResult:
    directory: Path
    marker_path: Path
    checksums_path: Path


def derive_publication_id(
    spec: DerivationSpec,
    evidence_validation_result_hash: str,
    export_validation_result_hash: str,
) -> str:
    return canonical_sha256(
        [
            spec.derivation_kind,
            tuple(binding.source_root_hash for binding in spec.ordered_source_bindings),
            evidence_validation_result_hash,
            spec.transform_spec_hash,
            spec.report_spec_hash,
            spec.reducer_and_renderer_input_hashes,
            export_validation_result_hash,
        ]
    )


def publish_with_last_marker(
    directory: Path,
    *,
    payloads: Mapping[str, bytes],
    marker_name: str,
    marker_bytes: bytes,
    fault_hook: PublicationFaultHook | None = None,
) -> PublicationResult:
    directory = Path(directory)
    marker_component = safe_path_component(marker_name)
    if marker_component == CHECKSUMS_NAME:
        raise ValueError("the commit marker and checksum file must be distinct")
    if marker_name in payloads or CHECKSUMS_NAME in payloads:
        raise ValueError("payloads cannot contain the publication envelope")

    normalized_payloads: list[tuple[str, bytes]] = []
    for relative_name, content in payloads.items():
        normalized = safe_relative_path(relative_name).as_posix()
        normalized_payloads.append((normalized, content))
    normalized_payloads.sort(key=lambda item: item[0])

    _require_publication_containment(
        directory,
        tuple(relative_name for relative_name, _content in normalized_payloads)
        + (CHECKSUMS_NAME, marker_name),
    )

    for relative_name, content in normalized_payloads:
        atomic_write_bytes(directory / relative_name, content, trusted_root=directory)
    _notify(fault_hook, PublicationBoundary.AFTER_PAYLOADS_DURABLE, directory)

    checksum_entries = [
        (relative_name, hashlib.sha256(content).hexdigest())
        for relative_name, content in normalized_payloads
    ]
    checksum_entries.append((marker_name, hashlib.sha256(marker_bytes).hexdigest()))
    checksum_entries.sort(key=lambda item: item[0])
    checksum_bytes = "".join(
        f"{sha256}  {relative_name}\n" for relative_name, sha256 in checksum_entries
    ).encode("utf-8")
    checksums_path = atomic_write_bytes(
        directory / CHECKSUMS_NAME,
        checksum_bytes,
        trusted_root=directory,
    ).path
    _notify(fault_hook, PublicationBoundary.AFTER_CHECKSUM_DURABLE, directory)
    _notify(fault_hook, PublicationBoundary.BEFORE_MARKER_WRITE, directory)
    marker_path = atomic_write_bytes(
        directory / marker_name,
        marker_bytes,
        trusted_root=directory,
    ).path
    _notify(fault_hook, PublicationBoundary.AFTER_MARKER_DURABLE, directory)
    return PublicationResult(
        directory=directory,
        marker_path=marker_path,
        checksums_path=checksums_path,
    )


def verify_published_directory(directory: Path, marker_name: str) -> None:
    directory = Path(directory)
    safe_path_component(marker_name)
    marker_path = directory / marker_name
    if not marker_path.is_file():
        raise ValueError("publication commit marker is absent")
    checksums_path = directory / CHECKSUMS_NAME
    if not checksums_path.is_file():
        raise ValueError("publication checksum file is absent")

    indexed_paths: set[str] = set()
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        match = CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise ValueError("invalid checksum line")
        relative_name = safe_relative_path(match.group("path")).as_posix()
        if relative_name == CHECKSUMS_NAME or relative_name in indexed_paths:
            raise ValueError("invalid or duplicate checksum path")
        indexed_paths.add(relative_name)
        path = directory / relative_name
        if not path.is_file() or sha256_file(path) != match.group("sha256"):
            raise ValueError(f"checksum mismatch for {relative_name}")
    if marker_name not in indexed_paths:
        raise ValueError("commit marker is not covered by checksums")
    committed_files: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("published directory cannot contain symbolic links")
        if path.is_file() and path != checksums_path:
            committed_files.add(path.relative_to(directory).as_posix())
    if committed_files != indexed_paths:
        raise ValueError("published directory contains an unindexed or missing file")


def _notify(
    hook: PublicationFaultHook | None,
    boundary: PublicationBoundary,
    directory: Path,
) -> None:
    if hook is not None:
        hook(boundary, directory)


def _require_publication_containment(
    directory: Path,
    relative_names: tuple[str, ...],
) -> None:
    if directory.is_symlink():
        raise UnsafeArtifactPathError("publication directory cannot be a symbolic link")
    if directory.exists() and not directory.is_dir():
        raise UnsafeArtifactPathError("publication root must be a directory")
    for relative_name in relative_names:
        current = directory
        for component in safe_relative_path(relative_name).parts[:-1]:
            current /= component
            if current.is_symlink():
                raise UnsafeArtifactPathError("publication path cannot traverse a symbolic link")
            if current.exists() and not current.is_dir():
                raise UnsafeArtifactPathError("publication path parent must be a directory")
