#!/usr/bin/env python3
"""Read-only OpenViking v0.4.16 local-storage acceptance probe."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any


EXPECTED_OPENVIKING_VERSION = "0.4.16"
EXPECTED_COLLECTION = "context"
EXPECTED_DIMENSION = 1024
MAX_METADATA_BYTES = 1024 * 1024


class ProbeError(RuntimeError):
    pass


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ProbeError(f"required regular metadata file is absent: {path.name}")
    if path.stat().st_size > MAX_METADATA_BYTES:
        raise ProbeError(f"metadata file exceeds limit: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError(f"metadata is unreadable: {path.name}") from exc
    if not isinstance(value, dict):
        raise ProbeError(f"metadata is not an object: {path.name}")
    return value


def _storage_entries(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ProbeError(f"required storage directory is absent: {root.name}")
    entries: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in sorted(directory_names + file_names):
            path = current / name
            if path.is_symlink():
                raise ProbeError(f"symlink is forbidden in storage tree: {path.name}")
            entries.append(path)
    return entries


def _required_paths(workspace: Path) -> tuple[Path, Path]:
    collection = workspace / "vectordb" / EXPECTED_COLLECTION
    return (
        collection / "collection_meta.json",
        workspace / "viking",
    )


def snapshot_required_files(workspace: Path) -> dict[str, str]:
    workspace = workspace.resolve(strict=True)
    collection_meta, agfs_root = _required_paths(workspace)
    entries = _storage_entries(agfs_root)
    files = [collection_meta] + [path for path in entries if path.is_file()]
    snapshot: dict[str, str] = {}
    for path in files:
        relative = str(path.relative_to(workspace))
        snapshot[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def _validate_metadata_shape(collection_meta: dict[str, Any]) -> None:
    if collection_meta.get("CollectionName") != EXPECTED_COLLECTION:
        raise ProbeError("unexpected OpenViking collection name")
    if collection_meta.get("Dimension") != EXPECTED_DIMENSION:
        raise ProbeError("unexpected OpenViking collection dimension")
    if collection_meta.get("PrimaryKey") != "id":
        raise ProbeError("unexpected OpenViking primary key")
    vector_fields = [
        field
        for field in collection_meta.get("Fields", [])
        if isinstance(field, dict) and field.get("FieldType") == "vector"
    ]
    if len(vector_fields) != 1 or vector_fields[0].get("Dim") != EXPECTED_DIMENSION:
        raise ProbeError("unexpected OpenViking vector-field dimension")


def require_read_only_mount(workspace: Path, mountinfo: str) -> None:
    expected = str(workspace)
    for line in mountinfo.splitlines():
        left = line.split(" - ", 1)[0].split()
        if len(left) >= 6 and left[4] == expected:
            if "ro" not in left[5].split(","):
                raise ProbeError("OpenViking workspace mount is writable")
            return
    raise ProbeError("OpenViking workspace is not a distinct container mount")


def _validate_with_exact_release(
    collection_meta: dict[str, Any], workspace: Path, agfs_root: Path
) -> None:
    installed_version = importlib.metadata.version("openviking")
    normalized_version = installed_version.removeprefix("v").split("+", 1)[0]
    if normalized_version != EXPECTED_OPENVIKING_VERSION:
        raise ProbeError("storage probe is not running in OpenViking v0.4.16")

    from openviking.pyagfs import get_binding_client
    from openviking.storage.vectordb.store.file_store import FileStore

    binding_client, _ = get_binding_client()
    agfs = binding_client(
        None,
        config={
            "pathlock": {
                "provider": "memory",
                "lock_timeout_secs": 0.0,
                "lock_expire_secs": 30.0,
            }
        },
    )
    agfs.mount("localfs", "/local", {"local_dir": str(agfs_root)})
    mounts = {mount.get("path"): mount.get("fstype") for mount in agfs.mounts()}
    if mounts.get("/local") != "localfs":
        raise ProbeError("exact-release AGFS interface did not mount local storage")
    root_stat = agfs.stat("/local")
    root_entries = agfs.ls("/local")
    if root_stat.get("isDir") is not True or not isinstance(root_entries, list):
        raise ProbeError("exact-release AGFS interface rejected storage root")
    if root_entries:
        names = sorted(str(entry["name"]) for entry in root_entries)
        representative = names[0]
        if representative in {".", ".."} or "/" in representative:
            raise ProbeError("exact-release AGFS interface returned an unsafe entry")
        agfs.stat(f"/local/{representative}")

    vector_store = FileStore(base_path=str(workspace / "vectordb"))
    raw_metadata = vector_store.get(f"{EXPECTED_COLLECTION}/collection_meta.json")
    if not raw_metadata:
        raise ProbeError("exact-release vector store could not read collection metadata")
    try:
        exact_metadata = json.loads(raw_metadata)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProbeError("exact-release vector store returned invalid metadata") from exc
    if exact_metadata != collection_meta:
        raise ProbeError("exact-release vector store returned inconsistent metadata")


def probe_storage(workspace: Path, *, exact_release: bool = True) -> dict[str, Any]:
    workspace = workspace.resolve(strict=True)
    collection_path, agfs_root = _required_paths(workspace)
    collection_meta = _read_json_object(collection_path)
    before = snapshot_required_files(workspace)
    _validate_metadata_shape(collection_meta)
    agfs_entries = _storage_entries(agfs_root)
    if exact_release:
        mountinfo = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
        require_read_only_mount(workspace, mountinfo)
        _validate_with_exact_release(collection_meta, workspace, agfs_root)
    after = snapshot_required_files(workspace)
    if before != after:
        raise ProbeError("storage changed during read-only probe")
    return {
        "status": "ok",
        "mode": "read_only_storage",
        "openviking_version": EXPECTED_OPENVIKING_VERSION,
        "collection": EXPECTED_COLLECTION,
        "dimension": EXPECTED_DIMENSION,
        "agfs_entries": len(agfs_entries),
        "files_verified": len(after),
    }


def _install_no_write_no_network_audit() -> None:
    forbidden_events = {
        "os.mkdir",
        "os.remove",
        "os.rename",
        "os.rmdir",
        "os.truncate",
        "shutil.copyfile",
        "shutil.move",
        "socket.bind",
        "socket.connect",
        "socket.connect_ex",
    }

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event in forbidden_events:
            raise ProbeError(f"forbidden probe operation: {event}")
        if event == "open" and len(args) >= 2:
            mode = args[1]
            flags = args[2] if len(args) >= 3 else None
            if isinstance(mode, str) and any(character in mode for character in "wax+"):
                raise ProbeError("forbidden write-mode file open")
            if isinstance(flags, int) and flags & os.O_ACCMODE != os.O_RDONLY:
                raise ProbeError("forbidden writable file descriptor")

    sys.addaudithook(audit)
    socket.setdefaulttimeout(0.1)


if __name__ == "__main__":
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    _install_no_write_no_network_audit()
    try:
        result = probe_storage(
            Path(os.environ.get("OAMB_OPENVIKING_WORKSPACE", "/var/lib/openviking"))
        )
    except (OSError, ProbeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, sort_keys=True))
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
