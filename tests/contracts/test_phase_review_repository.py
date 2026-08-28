from __future__ import annotations

from pathlib import Path

import pytest

from oamb.phase_review_repository import (
    phase_review_occurrence_root,
    prepare_artifact_repository,
)


def test_phase_review_repository_fingerprint_is_storage_bound(tmp_path: Path) -> None:
    first, first_fingerprint = prepare_artifact_repository(tmp_path / "first")
    second, second_fingerprint = prepare_artifact_repository(tmp_path / "second")

    assert first.is_dir()
    assert second.is_dir()
    assert first_fingerprint != second_fingerprint


def test_phase_review_repository_rejects_symlinked_root(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        prepare_artifact_repository(linked)


def test_phase_review_repository_rejects_symlinked_ancestor_before_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        prepare_artifact_repository(linked / "must-not-exist")

    assert not (outside / "must-not-exist").exists()


def test_phase_review_occurrence_root_rejects_symlink_before_write(tmp_path: Path) -> None:
    repository, _fingerprint = prepare_artifact_repository(tmp_path / "repository")
    outside = tmp_path / "outside"
    outside.mkdir()
    (repository / "phase-reviews").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        phase_review_occurrence_root(repository, "a" * 64)

    assert not (outside / ("a" * 64)).exists()
