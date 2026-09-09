from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

RESOLVER = Path(__file__).parents[1] / "provider-services" / "lib" / "host_embedding.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _write_recording_embedding_curl(path: Path) -> None:
    _write_executable(
        path,
        """
python3 -c '
import json
import sys
from pathlib import Path
Path(sys.argv[1]).write_text(
    json.dumps({"arguments": sys.argv[2:], "stdin": sys.stdin.read()}),
    encoding="utf-8",
)
print(json.dumps({"data": [{"embedding": [0.0] * 1024}]}))
' "$OAMB_TEST_TRACE" "$@"
""".strip(),
    )


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


def test_embedding_probe_uses_configured_service_api_key(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    trace = tmp_path / "curl-arguments.json"
    curl = fake_bin / "curl"
    _write_recording_embedding_curl(curl)

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; probe_embedding "$2" "$3"',
            "sh",
            str(RESOLVER),
            "https://embedding.example/v1",
            "paid-embedding-key",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "OAMB_EMBEDDING_MODEL": "qwen3-embedding:0.6b",
            "OAMB_TEST_TRACE": str(trace),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    invocation = json.loads(trace.read_text(encoding="utf-8"))
    assert "paid-embedding-key" not in " ".join(invocation["arguments"])
    assert invocation["stdin"] == 'header = "Authorization: Bearer paid-embedding-key"\n'


def test_embedding_probe_omits_authorization_for_keyless_service(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    trace = tmp_path / "curl-arguments.json"
    curl = fake_bin / "curl"
    _write_recording_embedding_curl(curl)

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; probe_embedding "$2" ""',
            "sh",
            str(RESOLVER),
            "http://127.0.0.1:18000/v1",
        ],
        capture_output=True,
        text=True,
        check=False,
        env={
            **os.environ,
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
            "OAMB_EMBEDDING_MODEL": "qwen3-embedding:0.6b",
            "OAMB_TEST_TRACE": str(trace),
        },
    )

    assert result.returncode == 0, result.stdout + result.stderr
    invocation = json.loads(trace.read_text(encoding="utf-8"))
    assert invocation["stdin"] == ""
    assert not any(argument.startswith("Authorization:") for argument in invocation["arguments"])


def test_online_embedding_url_is_not_rewritten(tmp_path: Path) -> None:
    result = _resolve(
        tmp_path,
        system_name="Linux",
        url="https://embedding.example/v1",
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "https://embedding.example/v1\n"
