from __future__ import annotations

import gzip
import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from oamb.artifacts.atomic import ArtifactCollisionError
from oamb.artifacts.capsule import (
    PUBLICATION_BOUNDARIES,
    PublicationBoundary,
    derive_publication_id,
    publish_with_last_marker,
    verify_published_directory,
)
from oamb.artifacts.store import ArtifactStore, UnsafeArtifactPathError
from oamb.contracts.evidence import (
    CapsuleManifest,
    CapsuleManifestEntry,
    CheckpointManifest,
    OriginClass,
    OriginRecord,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    ArtifactReadRequest,
    ArtifactStorePort,
    ArtifactWriteRequest,
    RawPayloadSealRequest,
)
from oamb.contracts.specifications import DerivationSpec, SourceEvidenceBinding, SourceEvidenceKind


def _origin() -> OriginRecord:
    return OriginRecord(
        origin_id="origin-1",
        origin_class=OriginClass.OAMB_NATIVE,
        producer="fixture",
        source_sha256="a" * 64,
        license_id="Apache-2.0",
    )


def _checkpoint(
    *, sequence: int, predecessor: str | None, entries: tuple[CapsuleManifestEntry, ...]
) -> CheckpointManifest:
    candidate = CheckpointManifest(
        checkpoint_manifest_hash="0" * 64,
        run_id="run-1",
        lease_record_hash="f" * 64,
        lease_epoch=1,
        sequence=sequence,
        predecessor_checkpoint_hash=predecessor,
        source_entries=entries,
        created_at=datetime(2026, 8, 27, 12, sequence, tzinfo=UTC),
    )
    checkpoint_hash = canonical_sha256(
        candidate.model_dump(mode="python", exclude={"checkpoint_manifest_hash"})
    )
    return candidate.model_copy(update={"checkpoint_manifest_hash": checkpoint_hash})


def test_store_seals_canonical_occurrence_records_and_rejects_unsafe_paths(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "capsule")

    result = store.write_source_record("specs/origins", "origin-1", _origin())

    assert result.path.relative_to(store.root).as_posix() == "source/specs/origins/origin-1.json"
    assert result.path.read_bytes() == canonical_json_bytes(_origin())
    with pytest.raises(UnsafeArtifactPathError):
        store.write_source_record("../outside", "origin-2", _origin())
    with pytest.raises(UnsafeArtifactPathError):
        store.write_source_record("attempts", "nested/id", _origin())
    assert not (tmp_path / "outside").exists()


def test_store_rejects_an_existing_symlink_that_escapes_the_artifact_root(tmp_path: Path) -> None:
    store_root = tmp_path / "capsule"
    outside = tmp_path / "outside"
    store_root.mkdir()
    outside.mkdir()
    (store_root / "source").symlink_to(outside, target_is_directory=True)
    store = ArtifactStore(store_root)

    with pytest.raises(UnsafeArtifactPathError, match="escapes"):
        store.write_source_record("attempts", "attempt-1", _origin())

    assert not tuple(outside.iterdir())


def test_verified_read_rejects_a_leaf_symlink_to_matching_external_bytes(
    tmp_path: Path,
) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    source = store.root / "source" / "attempts"
    source.mkdir(parents=True)
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"matching bytes")
    target = source / "attempt.json"
    target.symlink_to(outside)

    with pytest.raises(UnsafeArtifactPathError, match="symbolic link"):
        store.read_verified(
            ArtifactReadRequest(
                relative_path="source/attempts/attempt.json",
                expected_sha256=hashlib.sha256(b"matching bytes").hexdigest(),
            )
        )


def test_raw_payload_hashes_original_bytes_and_uses_deterministic_gzip(tmp_path: Path) -> None:
    payload = b'{"provider":"fixture","value":"\xe9\x9b\xaa"}'
    first_store = ArtifactStore(tmp_path / "one")
    second_store = ArtifactStore(tmp_path / "two")

    first = first_store.write_raw(payload, media_type="application/json")
    repeated = first_store.write_raw(payload, media_type="application/json")
    second = second_store.write_raw(payload, media_type="application/json")

    raw_sha256 = hashlib.sha256(payload).hexdigest()
    assert first.reference.sha256 == raw_sha256
    assert first.reference.byte_count == len(payload)
    assert first.reference.compression == "gzip"
    assert first.path.name == f"{raw_sha256}.json.gz"
    assert gzip.decompress(first.path.read_bytes()) == payload
    assert first.path.read_bytes() == second.path.read_bytes()
    assert repeated.created is False


def test_store_implements_typed_port_and_verifies_request_hashes(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    raw_bytes = b"provider receipt"
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    raw_handle = store.seal_raw(
        RawPayloadSealRequest(
            sha256=raw_sha256,
            media_type="application/octet-stream",
            compression="gzip",
            payload_bytes=raw_bytes,
        )
    )
    record_bytes = canonical_json_bytes(_origin())
    write_request = ArtifactWriteRequest(
        record_id="origin-1",
        relative_path="source/specs/origins/origin-1.json",
        canonical_sha256=hashlib.sha256(record_bytes).hexdigest(),
        canonical_bytes=record_bytes,
    )
    receipt = store.seal_source_record(write_request)

    assert isinstance(store, ArtifactStorePort)
    assert raw_handle.sha256 == raw_sha256
    assert receipt.record_id == "origin-1"
    assert (
        store.read_verified(
            ArtifactReadRequest(
                relative_path=write_request.relative_path,
                expected_sha256=write_request.canonical_sha256,
            )
        )
        == record_bytes
    )
    with pytest.raises(ArtifactCollisionError, match="request hash"):
        store.seal_source_record(
            ArtifactWriteRequest(
                record_id="origin-2",
                relative_path="source/specs/origins/origin-2.json",
                canonical_sha256="0" * 64,
                canonical_bytes=record_bytes,
            )
        )


def test_capsule_manifest_is_final_and_indexes_only_verified_source_files(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    record = store.write_source_record("specs/origins", "origin-1", _origin())
    entry = CapsuleManifestEntry(
        record_kind="origin_record",
        record_id="origin-1",
        relative_path="source/specs/origins/origin-1.json",
        sha256=record.sha256,
    )
    manifest = CapsuleManifest(
        capsule_id="b" * 64,
        run_id="run-1",
        run_spec_hash="c" * 64,
        source_entries=(entry,),
        source_manifest_hash="d" * 64,
    )

    sealed = store.seal_capsule(manifest)

    assert sealed.path.name == "capsule-manifest.json"
    assert sealed.path.read_bytes() == canonical_json_bytes(manifest)
    assert all(
        not item.relative_path.startswith("checkpoints/") for item in manifest.source_entries
    )
    record.path.write_bytes(b"corrupted after seal")
    with pytest.raises(ArtifactCollisionError, match="source entry hash"):
        store.seal_capsule(manifest)


def test_checkpoint_store_requires_an_append_only_predecessor_chain(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    record = store.write_source_record("specs/origins", "origin-1", _origin())
    entry = CapsuleManifestEntry(
        record_kind="origin_record",
        record_id="origin-1",
        relative_path="source/specs/origins/origin-1.json",
        sha256=record.sha256,
    )
    first = _checkpoint(sequence=1, predecessor=None, entries=(entry,))
    second = _checkpoint(
        sequence=2,
        predecessor=first.checkpoint_manifest_hash,
        entries=(entry,),
    )

    first_result = store.append_checkpoint(first)
    second_result = store.append_checkpoint(second)

    assert first_result.path.name == "1-1.json"
    assert second_result.path.name == "1-2.json"
    assert store.append_checkpoint(second).created is False
    fork = _checkpoint(sequence=3, predecessor="9" * 64, entries=(entry,))
    with pytest.raises(ArtifactCollisionError, match="predecessor"):
        store.append_checkpoint(fork)
    assert not (store.root / "checkpoints" / "1-3.json").exists()


def test_checkpoint_store_rejects_a_false_content_hash(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    invalid = CheckpointManifest(
        checkpoint_manifest_hash="1" * 64,
        run_id="run-1",
        lease_record_hash="f" * 64,
        lease_epoch=1,
        sequence=1,
        predecessor_checkpoint_hash=None,
        source_entries=(),
        created_at=datetime(2026, 8, 27, 12, 0, tzinfo=UTC),
    )

    with pytest.raises(ArtifactCollisionError, match="content hash"):
        store.append_checkpoint(invalid)


def test_checkpoint_sequence_scan_ignores_symlink_leaves(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    checkpoint_directory = store.root / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    outside = tmp_path / "outside-checkpoint.json"
    outside.write_bytes(b"not a checkpoint")
    (checkpoint_directory / "1-99.json").symlink_to(outside)
    first = _checkpoint(sequence=1, predecessor=None, entries=())

    result = store.append_checkpoint(first)

    assert result.path.name == "1-1.json"
    assert result.created is True


def test_checkpoint_predecessor_read_rejects_a_symlink_leaf(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "capsule")
    checkpoint_directory = store.root / "checkpoints"
    checkpoint_directory.mkdir(parents=True)
    first = _checkpoint(sequence=1, predecessor=None, entries=())
    second = _checkpoint(
        sequence=2,
        predecessor=first.checkpoint_manifest_hash,
        entries=(),
    )
    outside = tmp_path / "outside-checkpoint.json"
    outside.write_bytes(canonical_json_bytes(first))
    (checkpoint_directory / "1-1.json").symlink_to(outside)
    (checkpoint_directory / "1-2.json").write_bytes(canonical_json_bytes(second))

    with pytest.raises(ArtifactCollisionError, match="symbolic link|non-regular"):
        store.append_checkpoint(second)


def test_generic_publisher_writes_checksums_before_last_marker(tmp_path: Path) -> None:
    final_directory = tmp_path / "derivations" / ("e" * 64)
    marker_bytes = b'{"schema_name":"derived_manifest","derivation_id":"' + b"e" * 64 + b'"}'
    observed: list[PublicationBoundary] = []

    result = publish_with_last_marker(
        final_directory,
        payloads={"outputs/report.html": b"<p>offline</p>", "source-roots.json": b"[]"},
        marker_name="derived-manifest.json",
        marker_bytes=marker_bytes,
        fault_hook=lambda boundary, _path: observed.append(boundary),
    )

    assert tuple(observed) == PUBLICATION_BOUNDARIES
    assert result.marker_path.read_bytes() == marker_bytes
    checksum_text = (final_directory / "CHECKSUMS.sha256").read_text()
    assert f"{hashlib.sha256(marker_bytes).hexdigest()}  derived-manifest.json\n" in checksum_text
    assert "CHECKSUMS.sha256" not in checksum_text
    verify_published_directory(final_directory, "derived-manifest.json")


def test_published_directory_rejects_an_unindexed_extra_file(tmp_path: Path) -> None:
    final_directory = tmp_path / "derivation"
    publish_with_last_marker(
        final_directory,
        payloads={"output.json": b"{}"},
        marker_name="derived-manifest.json",
        marker_bytes=b'{"committed":true}',
    )
    (final_directory / "unindexed.txt").write_text("not committed", encoding="utf-8")

    with pytest.raises(ValueError, match="unindexed"):
        verify_published_directory(final_directory, "derived-manifest.json")


def test_publication_id_binds_ordered_roots_and_both_validation_results() -> None:
    first = SourceEvidenceBinding(
        binding_id="1" * 64,
        source_kind=SourceEvidenceKind.RUN,
        source_identity="run-1",
        source_root_hash="2" * 64,
        validation_result_hash="3" * 64,
        source_schema_versions=("capsule_manifest@1",),
    )
    second = SourceEvidenceBinding(
        binding_id="4" * 64,
        source_kind=SourceEvidenceKind.RUN,
        source_identity="run-2",
        source_root_hash="5" * 64,
        validation_result_hash="6" * 64,
        source_schema_versions=("capsule_manifest@1",),
    )
    spec = DerivationSpec(
        derivation_kind="comparison",
        ordered_source_bindings=(first, second),
        ordered_source_root_hash="7" * 64,
        transform_spec_hash="8" * 64,
        report_spec_hash="9" * 64,
        reducer_and_renderer_input_hashes=("a" * 64,),
        derivation_input_hash="b" * 64,
    )
    evidence_hash = "c" * 64
    export_hash = "d" * 64
    baseline = derive_publication_id(spec, evidence_hash, export_hash)

    assert (
        derive_publication_id(
            spec.model_copy(update={"ordered_source_bindings": (second, first)}),
            evidence_hash,
            export_hash,
        )
        != baseline
    )
    assert derive_publication_id(spec, "e" * 64, export_hash) != baseline
    assert derive_publication_id(spec, evidence_hash, "f" * 64) != baseline


@pytest.mark.parametrize("boundary", PUBLICATION_BOUNDARIES)
def test_publisher_fault_hook_preserves_partial_state_without_false_commit(
    tmp_path: Path,
    boundary: PublicationBoundary,
) -> None:
    final_directory = tmp_path / boundary.value

    def crash_at(current: PublicationBoundary, _path: Path) -> None:
        if current == boundary:
            raise RuntimeError(current.value)

    with pytest.raises(RuntimeError, match=boundary.value):
        publish_with_last_marker(
            final_directory,
            payloads={"output.json": b"{}"},
            marker_name="derived-manifest.json",
            marker_bytes=b'{"committed":true}',
            fault_hook=crash_at,
        )

    marker_exists = (final_directory / "derived-manifest.json").exists()
    assert marker_exists is (boundary == PublicationBoundary.AFTER_MARKER_DURABLE)
    if marker_exists:
        verify_published_directory(final_directory, "derived-manifest.json")


def test_publisher_retries_identical_partial_bytes_but_never_overwrites_collision(
    tmp_path: Path,
) -> None:
    final_directory = tmp_path / "derivation"
    (final_directory / "outputs").mkdir(parents=True)
    (final_directory / "outputs" / "report.html").write_bytes(b"wrong")

    with pytest.raises(ArtifactCollisionError, match="different bytes"):
        publish_with_last_marker(
            final_directory,
            payloads={"outputs/report.html": b"expected"},
            marker_name="derived-manifest.json",
            marker_bytes=b"marker",
        )
    assert (final_directory / "outputs" / "report.html").read_bytes() == b"wrong"
    assert not (final_directory / "derived-manifest.json").exists()


def test_publisher_rejects_nested_symlink_before_any_external_write(tmp_path: Path) -> None:
    final_directory = tmp_path / "derivation"
    outside = tmp_path / "outside"
    final_directory.mkdir()
    outside.mkdir()
    (final_directory / "outputs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(UnsafeArtifactPathError, match="symbolic link"):
        publish_with_last_marker(
            final_directory,
            payloads={"outputs/report.html": b"must stay contained"},
            marker_name="derived-manifest.json",
            marker_bytes=b"marker",
        )

    assert not tuple(outside.iterdir())
    assert not (final_directory / "CHECKSUMS.sha256").exists()
    assert not (final_directory / "derived-manifest.json").exists()
