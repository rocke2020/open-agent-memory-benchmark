from __future__ import annotations

import subprocess
from pathlib import Path

from oamb.public_boundary import scan_public_paths


def _private_fragments() -> tuple[str, ...]:
    return (
        "oamb" + "-" + "plan",
        "agent" + "-" + "memory" + "-" + "eval",
        "/Users/" + "rocke_dong",
    )


def test_tracked_public_tree_has_canonical_instruction_pair_and_no_private_inputs() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    tracked_paths = tuple(repository_root / path.decode("utf-8") for path in tracked if path)

    assert scan_public_paths(repository_root, tracked_paths) == ()
    assert (
        (repository_root / "AGENTS.md")
        .read_text(encoding="utf-8")
        .startswith(
            "# AGENTS.md\nThis file provides context for AI coding assistants "
            "(Claude Code, Codex, etc.)\n"
        )
    )
    assert (repository_root / "CLAUDE.md").is_symlink()
    assert (repository_root / "CLAUDE.md").readlink() == Path("AGENTS.md")
    tracked_names = {path.relative_to(repository_root).as_posix() for path in tracked_paths}
    assert "AGENTS.override.md" not in tracked_names
    assert not any(name.startswith(_private_fragments()[0] + "/") for name in tracked_names)


def test_public_boundary_scan_rejects_private_path_names_and_content(tmp_path: Path) -> None:
    for index, fragment in enumerate(_private_fragments(), 1):
        content_path = tmp_path / f"public-{index}.txt"
        content_path.write_text(f"unsafe reference: {fragment}\n", encoding="utf-8")
        issues = scan_public_paths(tmp_path, (content_path,))
        assert len(issues) == 1
        assert issues[0].kind == "content"
        assert issues[0].relative_path == content_path.name

    named_path = tmp_path / _private_fragments()[0]
    named_path.mkdir()
    nested = named_path / "public.txt"
    nested.write_text("safe content\n", encoding="utf-8")
    issues = scan_public_paths(tmp_path, (nested,))
    assert len(issues) == 1
    assert issues[0].kind == "path"

    override = tmp_path / "AGENTS.override.md"
    override.write_text("local instructions\n", encoding="utf-8")
    issues = scan_public_paths(tmp_path, (override,))
    assert len(issues) == 1
    assert issues[0].kind == "path"

    local_target = tmp_path.parent / "local-instructions.md"
    link = tmp_path / "LOCAL.md"
    link.symlink_to(local_target)
    issues = scan_public_paths(tmp_path, (link,))
    assert len(issues) == 1
    assert issues[0].kind == "symlink"
