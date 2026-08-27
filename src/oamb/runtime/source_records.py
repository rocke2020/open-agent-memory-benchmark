"""Canonical runtime writes through the dependency-free artifact port."""

from __future__ import annotations

import hashlib

from oamb.contracts.base import StrictContract
from oamb.contracts.ids import canonical_json_bytes
from oamb.contracts.ports import ArtifactSealReceipt, ArtifactStorePort, ArtifactWriteRequest


def seal_source_contract(
    store: ArtifactStorePort,
    *,
    relative_path: str,
    record_id: str,
    record: StrictContract,
) -> ArtifactSealReceipt:
    content = canonical_json_bytes(record)
    return store.seal_source_record(
        ArtifactWriteRequest(
            record_id=record_id,
            relative_path=relative_path,
            canonical_sha256=hashlib.sha256(content).hexdigest(),
            canonical_bytes=content,
        )
    )
