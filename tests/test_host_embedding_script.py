from __future__ import annotations

import os
import subprocess
from pathlib import Path

RESOLVER = Path(__file__).parents[1] / "provider-services" / "lib" / "host_embedding.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _resolve(tmp_path: Path, *, system_name: str, url: str) -> subprocess.CompletedProcess[str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "uname", f"printf '%s\\n' {system_name}")
    _write_executable(fake_bin / "docker", "printf '%s\\n' 172.17.0.1")
    return subprocess.run(
        ["/bin/sh", "-c", '. "$1"; resolve_host_embedding_base "$2"', "sh", RESOLVER, url],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )


def test_linux_host_embedding_uses_the_docker_bridge_gateway(tmp_path: Path) -> None:
    result = _resolve(
        tmp_path,
        system_name="Linux",
        url="http://host.docker.internal:18000/v1",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "http://172.17.0.1:18000/v1\n"


def test_macos_host_embedding_uses_loopback(tmp_path: Path) -> None:
    result = _resolve(
        tmp_path,
        system_name="Darwin",
        url="http://host.docker.internal:18000/v1",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "http://127.0.0.1:18000/v1\n"


def test_online_embedding_url_is_not_rewritten(tmp_path: Path) -> None:
    result = _resolve(
        tmp_path,
        system_name="Linux",
        url="https://embedding.example/v1",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "https://embedding.example/v1\n"
