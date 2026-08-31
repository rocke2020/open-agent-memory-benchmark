from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.contracts.evidence import RunRecord
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.states import ResumeDisposition, RunState
from oamb.runtime.native_continuation import initialize_continuation_root


def _sealed_capsule(
    root: Path,
    *,
    run_id: str = "run-a",
    state: RunState = RunState.ABORTED,
) -> Path:
    store = ArtifactStore(root)
    run = RunRecord(
        run_id=run_id,
        run_spec_hash=canonical_sha256(["run-spec"]),
        state=state,
        resume_disposition=ResumeDisposition.NOT_APPLICABLE,
        started_at=datetime(2026, 8, 31, tzinfo=UTC),
        ended_at=datetime(2026, 8, 31, 0, 1, tzinfo=UTC),
        ingestion_occurrence_ids=(),
        case_occurrence_ids=(),
    )
    store.write_source_record("run", run_id, run)
    store.write_raw(b'{"kept":true}', media_type="application/json")
    store.finalize_capsule(run_id=run_id, run_spec_hash=run.run_spec_hash)
    (root / "native-run-owner.json").write_bytes(b"old owner")
    return root


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_initialize_continuation_root_preserves_base_and_omits_terminal_files(
    tmp_path: Path,
) -> None:
    base = _sealed_capsule(tmp_path / "base")
    before = _tree_hashes(base)

    initialized = initialize_continuation_root(base, tmp_path / "successor")

    assert initialized.run_id == "run-a"
    assert initialized.target_root == tmp_path / "successor" / "run-a"
    assert not (initialized.target_root / "capsule-manifest.json").exists()
    assert not (initialized.target_root / "native-run-owner.json").exists()
    assert not (initialized.target_root / "source/run/run-a.json").exists()
    assert list((initialized.target_root / "source/raw").iterdir())
    assert _tree_hashes(base) == before


def test_initialize_continuation_root_is_create_only(tmp_path: Path) -> None:
    base = _sealed_capsule(tmp_path / "base")
    target = tmp_path / "successor" / "run-a"
    target.mkdir(parents=True)
    (target / "sentinel").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        initialize_continuation_root(base, tmp_path / "successor")

    assert (target / "sentinel").read_text(encoding="utf-8") == "keep"


def test_initialize_continuation_root_rejects_tampered_base_before_writing(
    tmp_path: Path,
) -> None:
    base = _sealed_capsule(tmp_path / "base")
    raw = next((base / "source/raw").iterdir())
    raw.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="hash mismatch"):
        initialize_continuation_root(base, tmp_path / "successor")

    assert not (tmp_path / "successor").exists()


def test_initialize_continuation_root_requires_aborted_run(tmp_path: Path) -> None:
    base = _sealed_capsule(tmp_path / "base", state=RunState.FINALIZED)

    with pytest.raises(ValueError, match="must contain the matching aborted run"):
        initialize_continuation_root(base, tmp_path / "successor")
