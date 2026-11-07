"""Fail-closed scan for private references in tracked public artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_FORBIDDEN_PUBLIC_FRAGMENTS = (
    "oamb" + "-" + "plan",
    "agent" + "-" + "memory" + "-" + "eval",
    "/Users/" + "rocke_dong",
)


@dataclass(frozen=True, slots=True)
class PublicBoundaryIssue:
    kind: Literal["path", "content", "symlink"]
    relative_path: str
    rule_id: str


def scan_public_paths(
    repository_root: Path,
    paths: tuple[Path, ...],
) -> tuple[PublicBoundaryIssue, ...]:
    root = Path(repository_root).resolve(strict=True)
    forbidden_bytes = tuple(fragment.encode("utf-8") for fragment in _FORBIDDEN_PUBLIC_FRAGMENTS)
    issues: list[PublicBoundaryIssue] = []
    for candidate in paths:
        path = Path(candidate)
        try:
            relative_path = path.relative_to(root).as_posix()
        except ValueError:
            relative_path = path.name
        if relative_path == "AGENTS.override.md" or any(
            fragment in relative_path for fragment in _FORBIDDEN_PUBLIC_FRAGMENTS
        ):
            issues.append(
                PublicBoundaryIssue(
                    kind="path",
                    relative_path=relative_path,
                    rule_id="public-boundary.private-path.v1",
                )
            )
            continue
        if path.is_symlink():
            target = path.readlink()
            resolved_target = (path.parent / target).resolve()
            if target.is_absolute() or not resolved_target.is_relative_to(root):
                issues.append(
                    PublicBoundaryIssue(
                        kind="symlink",
                        relative_path=relative_path,
                        rule_id="public-boundary.local-symlink.v1",
                    )
                )
                continue
            content = target.as_posix().encode("utf-8")
        elif path.is_file():
            content = path.read_bytes()
        else:
            continue
        if any(fragment in content for fragment in forbidden_bytes):
            issues.append(
                PublicBoundaryIssue(
                    kind="content",
                    relative_path=relative_path,
                    rule_id="public-boundary.private-content.v1",
                )
            )
    return tuple(issues)


__all__ = ["PublicBoundaryIssue", "scan_public_paths"]
