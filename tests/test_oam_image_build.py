import os
import subprocess
from pathlib import Path


def test_image_build_uses_committed_source_without_untracked_files(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    subprocess.run(
        ["git", "-C", str(source), "config", "user.email", "test@example.com"], check=True
    )
    subprocess.run(["git", "-C", str(source), "config", "user.name", "Test"], check=True)
    (source / "tracked.txt").write_text("committed")
    subprocess.run(["git", "-C", str(source), "add", "tracked.txt"], check=True)
    subprocess.run(["git", "-C", str(source), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    (source / "tracked.txt").write_text("modified")
    (source / ".env").write_text("private")
    binary = tmp_path / "bin"
    binary.mkdir()
    docker = binary / "docker"
    docker.write_text(
        '#!/bin/sh\nfor last do :; done\ntest "$(cat "$last/tracked.txt")" = committed && test ! -e "$last/.env"\n'
    )
    docker.chmod(0o755)
    script = Path(__file__).resolve().parents[1] / "provider-services/oam/build-image.sh"
    result = subprocess.run(
        [str(script), str(source), revision],
        env={**os.environ, "PATH": f"{binary}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
