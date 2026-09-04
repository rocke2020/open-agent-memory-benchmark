from __future__ import annotations

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "verify_mem0_source.sh"
EXPECTED_COMMIT = "dc82354e143c2581d505d581a00286d6ef8c3605"


def _write_fake_git(bin_directory: Path) -> None:
    fake_git = bin_directory / "git"
    fake_git.write_text(
        """#!/bin/sh
if [ "$#" -ne 4 ] || [ "$1" != "-C" ] || \
   [ "$2" != "$OAMB_TEST_MEM0_CHECKOUT" ] || \
   [ "$3" != "rev-parse" ] || [ "$4" != "v2.0.19^{commit}" ]; then
  exit 97
fi
if [ "${OAMB_TEST_GIT_FAIL:-0}" = "1" ]; then
  exit 2
fi
printf '%s\\n' "$OAMB_TEST_GIT_COMMIT"
""",
        encoding="utf-8",
    )
    fake_git.chmod(0o755)


def _run_script(
    tmp_path: Path,
    *,
    resolved_commit: str = EXPECTED_COMMIT,
    git_fails: bool = False,
) -> subprocess.CompletedProcess[str]:
    assert SCRIPT_PATH.is_file(), "Mem0 source verification script is missing"
    assert os.access(SCRIPT_PATH, os.X_OK), "Mem0 source verification script is not executable"
    checkout = tmp_path / "mem0 checkout"
    checkout.mkdir()
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    _write_fake_git(bin_directory)
    return subprocess.run(
        [str(SCRIPT_PATH), str(checkout)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_directory}:/usr/bin:/bin",
            "OAMB_TEST_MEM0_CHECKOUT": str(checkout),
            "OAMB_TEST_GIT_COMMIT": resolved_commit,
            "OAMB_TEST_GIT_FAIL": "1" if git_fails else "0",
        },
    )


def test_verify_mem0_source_prints_pass_for_the_pinned_release(tmp_path: Path) -> None:
    result = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    assert result.stdout == f"PASS: Mem0 v2.0.19 resolves to {EXPECTED_COMMIT}\n"
    assert result.stderr == ""


def test_verify_mem0_source_reports_a_commit_mismatch(tmp_path: Path) -> None:
    mismatched_commit = "0" * 40

    result = _run_script(tmp_path, resolved_commit=mismatched_commit)

    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr == (
        f"FAIL: Mem0 v2.0.19 expected {EXPECTED_COMMIT}, got {mismatched_commit}\n"
    )


def test_verify_mem0_source_reports_an_unresolvable_tag(tmp_path: Path) -> None:
    result = _run_script(tmp_path, git_fails=True)

    assert result.returncode != 0
    assert result.stdout == ""
    assert "FAIL: cannot resolve Mem0 tag v2.0.19 in " in result.stderr
