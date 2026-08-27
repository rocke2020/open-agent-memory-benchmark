"""Typed create-only storage for source artifacts."""

from __future__ import annotations

import gzip
import hashlib
import io
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from oamb.contracts.base import StrictContract
from oamb.contracts.evidence import (
    CapsuleManifest,
    CapsuleManifestEntry,
    CheckpointManifest,
    RawReference,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    ArtifactReadRequest,
    ArtifactSealReceipt,
    ArtifactWriteRequest,
    RawPayloadSealRequest,
    RawReferenceHandle,
)

from .atomic import (
    ArtifactCollisionError,
    AtomicWriteResult,
    atomic_write_bytes,
    read_regular_file,
    sha256_file,
)


class UnsafeArtifactPathError(ValueError):
    """An artifact path is absolute, escaping, or otherwise ambiguous."""


@dataclass(frozen=True)
class StoredRawArtifact:
    path: Path
    reference: RawReference
    created: bool


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve(strict=False)

    def write_source_record(
        self, collection: str, record_id: str, record: StrictContract
    ) -> AtomicWriteResult:
        collection_path = safe_relative_path(collection)
        record_component = safe_path_component(record_id)
        target = self._contained_target(
            Path("source") / collection_path / f"{record_component}.json"
        )
        return atomic_write_bytes(target, canonical_json_bytes(record), trusted_root=self.root)

    def write_raw(self, payload: bytes, *, media_type: str) -> StoredRawArtifact:
        raw_sha256 = hashlib.sha256(payload).hexdigest()
        compressed = _deterministic_gzip(payload)
        result = atomic_write_bytes(
            self._contained_target(Path("source/raw") / f"{raw_sha256}.json.gz"),
            compressed,
            trusted_root=self.root,
        )
        return StoredRawArtifact(
            path=result.path,
            reference=RawReference(
                sha256=raw_sha256,
                media_type=media_type,
                byte_count=len(payload),
                compression="gzip",
            ),
            created=result.created,
        )

    def seal_raw(self, request: RawPayloadSealRequest) -> RawReferenceHandle:
        actual_sha256 = hashlib.sha256(request.payload_bytes).hexdigest()
        if actual_sha256 != request.sha256:
            raise ArtifactCollisionError("raw request hash does not match payload bytes")
        if request.compression == "gzip":
            self.write_raw(request.payload_bytes, media_type=request.media_type)
        else:
            atomic_write_bytes(
                self._contained_target(Path("source/raw") / request.sha256),
                request.payload_bytes,
                trusted_root=self.root,
            )
        return RawReferenceHandle(sha256=request.sha256)

    def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        relative_path = safe_relative_path(request.relative_path)
        if relative_path.parts[0] != "source":
            raise UnsafeArtifactPathError("source record path must be below source/")
        result = self._seal_request(request)
        return ArtifactSealReceipt(
            request.record_id,
            request.canonical_sha256,
            created=result.created,
        )

    def seal_checkpoint(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        self._verify_request_hash(request)
        checkpoint = CheckpointManifest.model_validate_json(request.canonical_bytes)
        expected_path = f"checkpoints/{checkpoint.lease_epoch}-{checkpoint.sequence}.json"
        if request.relative_path != expected_path:
            raise UnsafeArtifactPathError("checkpoint path does not match its epoch and sequence")
        if request.record_id != checkpoint.checkpoint_manifest_hash:
            raise ArtifactCollisionError("checkpoint record ID does not match its content hash")
        result = self.append_checkpoint(checkpoint)
        return ArtifactSealReceipt(
            request.record_id,
            request.canonical_sha256,
            created=result.created,
        )

    def seal_source_manifest(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        self._verify_request_hash(request)
        if request.relative_path != "capsule-manifest.json":
            raise UnsafeArtifactPathError("source manifest must use capsule-manifest.json")
        manifest = CapsuleManifest.model_validate_json(request.canonical_bytes)
        if request.record_id != manifest.capsule_id:
            raise ArtifactCollisionError("capsule record ID does not match the manifest")
        result = self.seal_capsule(manifest)
        return ArtifactSealReceipt(
            request.record_id,
            request.canonical_sha256,
            created=result.created,
        )

    def read_verified(self, request: ArtifactReadRequest) -> bytes:
        relative_path = safe_relative_path(request.relative_path)
        target = self._contained_target(Path(relative_path.as_posix()))
        if target.is_symlink():
            raise UnsafeArtifactPathError("artifact read target cannot be a symbolic link")
        if not target.is_file():
            raise FileNotFoundError(target)
        try:
            content = read_regular_file(target)
        except ArtifactCollisionError as exc:
            raise UnsafeArtifactPathError("artifact read target must be a regular file") from exc
        if hashlib.sha256(content).hexdigest() != request.expected_sha256:
            raise ArtifactCollisionError("artifact read hash does not match expected hash")
        return content

    def seal_capsule(self, manifest: CapsuleManifest) -> AtomicWriteResult:
        self._verify_source_entries(manifest.source_entries)
        return atomic_write_bytes(
            self._contained_target(Path("capsule-manifest.json")),
            canonical_json_bytes(manifest),
            trusted_root=self.root,
        )

    def append_checkpoint(self, manifest: CheckpointManifest) -> AtomicWriteResult:
        expected_hash = canonical_sha256(
            manifest.model_dump(mode="python", exclude={"checkpoint_manifest_hash"})
        )
        if manifest.checkpoint_manifest_hash != expected_hash:
            raise ArtifactCollisionError("checkpoint content hash does not match its fields")
        self._verify_source_entries(manifest.source_entries)

        checkpoint_directory = self._contained_target(Path("checkpoints/placeholder")).parent
        target = checkpoint_directory / f"{manifest.lease_epoch}-{manifest.sequence}.json"
        if not target.exists():
            existing_sequences = _checkpoint_sequences(checkpoint_directory, manifest.lease_epoch)
            expected_sequence = (max(existing_sequences) + 1) if existing_sequences else 1
            if manifest.sequence != expected_sequence:
                raise ArtifactCollisionError("checkpoint sequence is not append-only")
        if manifest.sequence > 1:
            predecessor_path = checkpoint_directory / (
                f"{manifest.lease_epoch}-{manifest.sequence - 1}.json"
            )
            try:
                predecessor_bytes = read_regular_file(predecessor_path)
            except FileNotFoundError:
                raise ArtifactCollisionError("checkpoint predecessor is missing") from None
            predecessor = CheckpointManifest.model_validate_json(predecessor_bytes)
            if predecessor.checkpoint_manifest_hash != manifest.predecessor_checkpoint_hash:
                raise ArtifactCollisionError("checkpoint predecessor hash does not match")
        return atomic_write_bytes(
            target,
            canonical_json_bytes(manifest),
            trusted_root=self.root,
        )

    def _contained_target(self, relative_path: Path) -> Path:
        target = self.root / relative_path
        resolved_parent = target.parent.resolve(strict=False)
        if not resolved_parent.is_relative_to(self.root):
            raise UnsafeArtifactPathError("artifact path escapes the store through a symlink")
        return target

    def _seal_request(self, request: ArtifactWriteRequest) -> AtomicWriteResult:
        self._verify_request_hash(request)
        relative_path = safe_relative_path(request.relative_path)
        return atomic_write_bytes(
            self._contained_target(Path(relative_path.as_posix())),
            request.canonical_bytes,
            trusted_root=self.root,
        )

    @staticmethod
    def _verify_request_hash(request: ArtifactWriteRequest) -> None:
        if hashlib.sha256(request.canonical_bytes).hexdigest() != request.canonical_sha256:
            raise ArtifactCollisionError("artifact request hash does not match canonical bytes")

    def _verify_source_entries(self, entries: tuple[CapsuleManifestEntry, ...]) -> None:
        for entry in entries:
            relative_name = entry.relative_path
            expected_sha256 = entry.sha256
            relative_path = safe_relative_path(relative_name)
            if not relative_path.parts or relative_path.parts[0] != "source":
                raise UnsafeArtifactPathError("manifest entries must be below source/")
            source_path = self._contained_target(Path(relative_path.as_posix()))
            if not source_path.is_file() or sha256_file(source_path) != expected_sha256:
                raise ArtifactCollisionError(
                    f"source entry hash does not match sealed bytes: {relative_name}"
                )


def safe_relative_path(value: str) -> PurePosixPath:
    """Return an unambiguous, non-escaping POSIX artifact path."""

    if not value or "\x00" in value or "\\" in value or value.startswith("/"):
        raise UnsafeArtifactPathError("artifact path must be a safe relative POSIX path")
    components = value.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise UnsafeArtifactPathError("artifact path must be a safe relative POSIX path")
    return PurePosixPath(*components)


def safe_path_component(value: str) -> str:
    path = safe_relative_path(value)
    if len(path.parts) != 1:
        raise UnsafeArtifactPathError("artifact identifier must be one path component")
    return value


def _deterministic_gzip(payload: bytes) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(payload)
    return buffer.getvalue()


def _checkpoint_sequences(directory: Path, lease_epoch: int) -> tuple[int, ...]:
    if not directory.exists():
        return ()
    prefix = f"{lease_epoch}-"
    sequences: list[int] = []
    for path in directory.glob(f"{prefix}*.json"):
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(metadata.st_mode):
            continue
        suffix = path.name.removeprefix(prefix).removesuffix(".json")
        if suffix.isdigit():
            sequences.append(int(suffix))
    return tuple(sequences)
