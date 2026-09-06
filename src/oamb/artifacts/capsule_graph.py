"""Read an immutable embedded-capsule graph with global identity deduplication."""

from __future__ import annotations

import hashlib
from pathlib import Path

from oamb.artifacts.atomic import read_regular_file
from oamb.contracts.evidence import CapsuleManifest


def read_capsule_graph(
    root: Path,
    seen: dict[str, str] | None = None,
) -> tuple[tuple[Path, CapsuleManifest], ...]:
    """Return root-first unique capsules after verifying every manifest-owned byte."""

    identities = seen if seen is not None else {}
    resolved_root = Path(root).resolve(strict=True)
    manifest_bytes = read_regular_file(resolved_root / "capsule-manifest.json")
    manifest = CapsuleManifest.model_validate_json(manifest_bytes)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    previous = identities.get(manifest.capsule_id)
    if previous is not None:
        if previous != manifest_sha256:
            raise ValueError("embedded capsule identity has conflicting manifest bytes")
        return ()
    identities[manifest.capsule_id] = manifest_sha256

    embedded: list[Path] = []
    for entry in manifest.source_entries:
        content = read_regular_file(resolved_root / entry.relative_path)
        if hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ValueError("embedded capsule source entry hash drifted")
        if entry.record_kind == "embedded_part_file" and entry.relative_path.endswith(
            "/capsule-manifest.json"
        ):
            embedded.append((resolved_root / entry.relative_path).parent)
    graph: list[tuple[Path, CapsuleManifest]] = [(resolved_root, manifest)]
    for embedded_root in embedded:
        graph.extend(read_capsule_graph(embedded_root, identities))
    return tuple(graph)


__all__ = ["read_capsule_graph"]
