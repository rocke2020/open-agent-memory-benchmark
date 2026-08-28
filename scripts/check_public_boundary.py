#!/usr/bin/env python3
"""Check every Git-tracked path for forbidden local-only references."""

from __future__ import annotations

import subprocess
from pathlib import Path

from oamb.public_boundary import scan_public_paths


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    paths = tuple(repository_root / value.decode("utf-8") for value in tracked if value)
    issues = scan_public_paths(repository_root, paths)
    for issue in issues:
        print(f"{issue.rule_id}:{issue.kind}:{issue.relative_path}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
