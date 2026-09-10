from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from oamb.runtime.provider_env import load_t10_provider_environment

REPOSITORY_ROOT = Path(__file__).parents[1]
RESOLVER = REPOSITORY_ROOT / "provider-services" / "lib" / "host_embedding.sh"
ENV_FILE = REPOSITORY_ROOT / ".env"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
EXPECTED_DIMENSION = 1024
NORMALIZATION_TOLERANCE = 1e-6
PROBE_TEXT = "OAMB embedding normalization probe"
EMBEDDING_ENVIRONMENT_KEYS = frozenset({"OAMB_EMBEDDING_BASE_URL", "OAMB_EMBEDDING_API_KEY"})


def _load_embedding_configuration() -> tuple[str, str]:
    environment = load_t10_provider_environment(
        ENV_FILE,
        expected_keys=EMBEDDING_ENVIRONMENT_KEYS,
    )
    base_url = environment.get("OAMB_EMBEDDING_BASE_URL", "")
    if not base_url or base_url == "change-me":
        raise ValueError("OAMB_EMBEDDING_BASE_URL must be configured in .env")
    return base_url, environment.get("OAMB_EMBEDDING_API_KEY", "")


def _request_embedding(base_url: str, api_key: str) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(timeout=10.0, trust_env=False) as client:
        response = client.post(
            f"{base_url.rstrip('/')}/embeddings",
            headers=headers,
            json={
                "model": EMBEDDING_MODEL,
                "input": PROBE_TEXT,
                "dimensions": EXPECTED_DIMENSION,
            },
        )
        response.raise_for_status()
        document = response.json()
    if not isinstance(document, dict):
        raise ValueError("embedding response must be a JSON object")
    return document


def _read_embedding(document: dict[str, Any]) -> list[float]:
    if document.get("model") != EMBEDDING_MODEL:
        raise ValueError("embedding response model does not match the requested model")
    data = document.get("data")
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("embedding response must contain exactly one vector")
    vector = data[0].get("embedding")
    if not isinstance(vector, list) or len(vector) != EXPECTED_DIMENSION:
        raise ValueError(f"embedding vector must have {EXPECTED_DIMENSION} dimensions")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ):
        raise ValueError("embedding vector contains a non-finite number")
    return [float(value) for value in vector]


def main() -> None:
    try:
        base_url, api_key = _load_embedding_configuration()
        vector = _read_embedding(_request_embedding(base_url, api_key))
    except (OSError, ValueError, httpx.HTTPError) as exc:
        raise SystemExit(f"embedding service test failed: {exc}") from exc

    norm = math.sqrt(sum(value * value for value in vector))
    normalized = math.isclose(
        norm,
        1.0,
        rel_tol=NORMALIZATION_TOLERANCE,
        abs_tol=NORMALIZATION_TOLERANCE,
    )
    print(f"model: {EMBEDDING_MODEL}")
    print(f"api_key: {'configured' if api_key else 'not configured'}")
    print(f"dimension: {len(vector)}")
    print(f"l2_norm: {norm:.9f}")
    print(f"normalized: {str(normalized).lower()}")
    if not normalized:
        raise SystemExit("embedding vector is not normalized")


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


def test_linux_managed_local_start_uses_loopback_listener_and_gateway_relay(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repository"
    helper = root / "scripts" / "start_local_embedding" / "start_ollama_embedding.sh"
    helper.parent.mkdir(parents=True)
    trace = tmp_path / "helper-environment"
    host_ready = tmp_path / "host-ready"
    gateway_ready = tmp_path / "gateway-ready"
    _write_executable(
        helper,
        'printf "%s|%s\\n" "$OAMB_OLLAMA_BIND_HOST" "$OAMB_OLLAMA_RELAY_HOST" '
        '> "$OAMB_TEST_TRACE"; touch "$OAMB_TEST_HOST_READY" "$OAMB_TEST_GATEWAY_READY"; '
        "while :; do sleep 1; done",
    )
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(fake_bin / "uname", "printf '%s\\n' Linux")
    _write_executable(fake_bin / "docker", "printf '%s\\n' 172.17.0.1")
    program = f"""
ROOT={root}
WORK_DIR={tmp_path}
EMBEDDING_STARTUP_ATTEMPTS=3
. "{RESOLVER}"
die() {{ printf '%s\n' "$*" >&2; exit 1; }}
probe_embedding() {{
  case "$1" in
    http://127.0.0.1:18000/v1) [ -f "$OAMB_TEST_HOST_READY" ] ;;
    http://172.17.0.1:18000/v1) [ -f "$OAMB_TEST_GATEWAY_READY" ] ;;
    *) return 1 ;;
  esac
}}
start_local_embedding 'http://127.0.0.1:18000/v1' ''
"""

    pid_path = tmp_path / "embedding.pid"
    try:
        result = subprocess.run(
            ["/bin/sh", "-c", program],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
            env={
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "OAMB_TEST_TRACE": str(trace),
                "OAMB_TEST_HOST_READY": str(host_ready),
                "OAMB_TEST_GATEWAY_READY": str(gateway_ready),
            },
        )
    finally:
        if pid_path.is_file():
            helper_pid = int(pid_path.read_text(encoding="utf-8"))
            os.kill(helper_pid, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                try:
                    os.kill(helper_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)

    assert result.returncode == 0, result.stdout + result.stderr
    assert trace.read_text(encoding="utf-8") == "127.0.0.1|172.17.0.1\n"
    assert "embedding: PASS (start_ollama_embedding.sh" in result.stdout


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


if __name__ == "__main__":
    main()
