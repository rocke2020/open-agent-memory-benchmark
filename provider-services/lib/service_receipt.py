"""Secret-free, content-addressed provider service verification receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

PROFILE_IDS = (
    "hindsight-rest-v1",
    "mem0-rest-v1",
    "openviking-rest-v1",
)
PROFILE_PROOF_FILES = {
    "hindsight-rest-v1": (
        "hindsight-health.json",
        "hindsight-version.json",
        "hindsight-model-config.json",
        "hindsight-retry-config.json",
    ),
    "mem0-rest-v1": (
        "mem0-openapi.json",
        "mem0-empty-projection.json",
        "mem0-config-redacted.json",
        "mem0-retry-config.json",
    ),
    "openviking-rest-v1": (
        "openviking-health.json",
        "openviking-auth-identity.json",
        "openviking-storage.json",
        "openviking-model-config.json",
        "openviking-retry-config.json",
    ),
}
PRIVATE_PATH_PREFIXES = ("/Users/", "/home/", "/private/var/", "C:\\Users\\")
REDACTED_VALUE = "[redacted]"


def build_service_verification_receipt(
    *,
    provider_project: str,
    project_attestation_sha256: str,
    verified_at_utc: str,
    profile_proof_manifest_sha256: Mapping[str, str],
) -> dict[str, object]:
    _require_sha256(project_attestation_sha256)
    if tuple(sorted(profile_proof_manifest_sha256)) != tuple(sorted(PROFILE_IDS)):
        raise ValueError("service receipt requires exactly the three REST profiles")
    profiles: list[dict[str, object]] = []
    for profile_id in PROFILE_IDS:
        proof_manifest_hash = profile_proof_manifest_sha256[profile_id]
        _require_sha256(proof_manifest_hash)
        profiles.append(
            {
                "profile_id": profile_id,
                "proof_manifest_sha256": proof_manifest_hash,
                "liveness": "passed",
                "storage_configuration": "passed",
                "runtime_identity": "passed",
                "model_readiness": "not_run",
                "memory_conformance": "not_run",
            }
        )
    return {
        "schema_name": "oamb_provider_service_verification",
        "schema_version": 1,
        "provider_project": provider_project,
        "project_attestation_sha256": project_attestation_sha256,
        "verified_at_utc": verified_at_utc,
        "profiles": profiles,
    }


def seal_profile_proof_manifest(
    proof_store: Path,
    *,
    profile_id: str,
    proof_directory: Path,
    trusted_root: Path,
) -> str:
    try:
        filenames = PROFILE_PROOF_FILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"unknown provider profile: {profile_id}") from exc
    _require_real_directory_path(proof_directory, trusted_root=trusted_root)
    entries: list[dict[str, object]] = []
    for filename in filenames:
        content = _public_proof_bytes(_read_regular_leaf(proof_directory / filename))
        content_hash = hashlib.sha256(content).hexdigest()
        _seal_content_addressed_bytes(
            proof_store / "blobs",
            content,
            suffix="",
            trusted_root=trusted_root,
        )
        entries.append(
            {
                "relative_path": filename,
                "sha256": content_hash,
                "byte_count": len(content),
            }
        )
    manifest = {
        "schema_name": "oamb_provider_profile_proof_manifest",
        "schema_version": 1,
        "profile_id": profile_id,
        "files": entries,
    }
    manifest_path = _seal_content_addressed_bytes(
        proof_store / "manifests",
        _canonical_json_bytes(manifest),
        suffix=".json",
        trusted_root=trusted_root,
    )
    return manifest_path.stem


def _public_proof_bytes(content: bytes) -> bytes:
    document = json.loads(content)
    redacted = _redact_control_authorities(document)
    if redacted == document:
        return content
    return _canonical_json_bytes(redacted)


def _redact_control_authorities(value: object) -> object:
    if isinstance(value, dict):
        return {key: _redact_control_authorities(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_control_authorities(item) for item in value]
    if isinstance(value, str) and (
        "://" in value
        or value.startswith(("localhost:", "127.0.0.1:", *PRIVATE_PATH_PREFIXES))
    ):
        return REDACTED_VALUE
    return value


def seal_service_verification_receipt(
    output_directory: Path,
    receipt: Mapping[str, object],
) -> Path:
    return _seal_content_addressed_bytes(
        output_directory,
        _canonical_json_bytes(receipt),
        suffix=".json",
        trusted_root=output_directory,
    )


def _seal_content_addressed_bytes(
    output_directory: Path,
    content: bytes,
    *,
    suffix: str,
    trusted_root: Path,
) -> Path:
    output_directory = _ensure_durable_directory(
        output_directory,
        trusted_root=trusted_root,
    )
    content_hash = hashlib.sha256(content).hexdigest()
    target = output_directory / f"{content_hash}{suffix}"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".content-addressed-",
        suffix=".partial",
        dir=output_directory,
    )
    temporary = Path(temporary_name)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short service receipt write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    try:
        try:
            os.link(temporary, target)
        except FileExistsError:
            if _read_regular_leaf(target) != content:
                raise RuntimeError("content-addressed hash collision or corruption") from None
        _fsync_directory(output_directory)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(output_directory)
    return target


def _ensure_durable_directory(directory: Path, *, trusted_root: Path) -> Path:
    directory = directory.absolute()
    lexical_root = trusted_root.absolute()
    if not directory.is_relative_to(lexical_root):
        raise ValueError("content-addressed path is outside its trusted root")
    current = lexical_root
    for component in (None, *directory.relative_to(lexical_root).parts):
        if component is not None:
            current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{current} is a symbolic link in the proof path")
    resolved_root = lexical_root.resolve(strict=False)
    directory = directory.resolve(strict=False)
    if not directory.is_relative_to(resolved_root):
        raise ValueError("content-addressed path escapes its trusted root")
    missing: list[Path] = []
    cursor = directory
    while True:
        try:
            metadata = cursor.lstat()
        except FileNotFoundError:
            missing.append(cursor)
            parent = cursor.parent
            if parent == cursor:
                raise ValueError("content-addressed directory has no existing ancestor") from None
            cursor = parent
            continue
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"{cursor} is a symbolic link or non-directory proof parent")
        break
    for path in reversed(missing):
        try:
            path.mkdir()
        except FileExistsError:
            metadata = path.lstat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(
                    f"{path} is a symbolic link or non-directory proof parent"
                ) from None
        _fsync_directory(path.parent)
    return directory


def _read_regular_leaf(path: Path) -> bytes:
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError as exc:
        raise ValueError(f"missing proof file: {path.name}") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise ValueError(f"proof leaf must be a regular proof file, not a symlink: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened_stat = os.fstat(descriptor)
        if (path_stat.st_dev, path_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
            raise ValueError(f"proof leaf changed while opening: {path.name}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _require_real_directory_path(directory: Path, *, trusted_root: Path) -> None:
    directory = directory.absolute()
    trusted_root = trusted_root.absolute()
    if not directory.is_relative_to(trusted_root):
        raise ValueError("proof directory is outside its trusted root")
    cursor = trusted_root
    for component in (None, *directory.relative_to(trusted_root).parts):
        if component is not None:
            cursor /= component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError as exc:
            raise ValueError("proof directory is missing") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("proof directory contains a symbolic link or non-directory")


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_sha256(value: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError("expected lowercase SHA-256")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--proof-dir", type=Path, required=True)
    parser.add_argument("--trusted-root", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--attestation-sha256", required=True)
    parser.add_argument("--verified-at-utc", required=True)
    arguments = parser.parse_args()
    proof_store = arguments.output_dir / "proofs"
    proof_manifest_hashes = {
        profile_id: seal_profile_proof_manifest(
            proof_store,
            profile_id=profile_id,
            proof_directory=arguments.proof_dir,
            trusted_root=arguments.trusted_root,
        )
        for profile_id in PROFILE_IDS
    }
    receipt = build_service_verification_receipt(
        provider_project=arguments.project,
        project_attestation_sha256=arguments.attestation_sha256,
        verified_at_utc=arguments.verified_at_utc,
        profile_proof_manifest_sha256=proof_manifest_hashes,
    )
    path = seal_service_verification_receipt(arguments.output_dir, receipt)
    print(path)


if __name__ == "__main__":
    main()
