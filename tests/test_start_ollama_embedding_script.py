from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from tests.benchmark_configuration import load_canonical_configuration

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "start_local_embedding" / "start_ollama_embedding.sh"
)
EMBEDDING_MODEL = load_canonical_configuration().models.embedding.model


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/bash\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _fake_environment(
    tmp_path: Path,
    *,
    ready: bool,
    embedding_dimension: int = 1024,
) -> tuple[dict[str, str], Path, Path, Path]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    ready_file = tmp_path / "ready"
    stopped_file = tmp_path / "stopped"
    ollama_calls = tmp_path / "ollama-calls"
    curl_calls = tmp_path / "curl-calls"
    if ready:
        ready_file.touch()

    _write_executable(
        fake_bin / "ollama",
        """
printf 'OLLAMA_HOST=%s %s\n' "${OLLAMA_HOST:-}" "$*" >> "$FAKE_OLLAMA_CALLS"
case "${1:-}" in
  serve)
    touch "$FAKE_READY_FILE"
    if [ "${FAKE_SERVER_EXIT_AFTER_READY:-}" = "1" ]; then
      sleep 1
      exit 91
    fi
    trap 'touch "$FAKE_STOPPED_FILE"; exit 0' TERM INT
    while :; do sleep 1; done
    ;;
  pull)
    [ "${2:-}" = "__EMBEDDING_MODEL__" ]
    ;;
  *)
    exit 2
    ;;
esac
""".replace("__EMBEDDING_MODEL__", EMBEDDING_MODEL).strip(),
    )
    _write_executable(fake_bin / "docker", "printf '%s\\n' 172.17.0.1")
    _write_executable(
        fake_bin / "curl",
        """
printf '%s\n' "$*" >> "$FAKE_CURL_CALLS"
if [ "${FAKE_RELAY_UNREADY:-}" = "1" ] && \
   [ "$(wc -l < "$FAKE_CURL_CALLS")" -ge 2 ]; then
  exit 22
fi
case "$*" in
  */api/version*)
    [ -f "$FAKE_READY_FILE" ] || exit 22
    printf '{"version":"test"}\n'
    ;;
  */v1/embeddings*)
    python3 - "$FAKE_EMBEDDING_DIMENSION" <<'PY'
import json
import sys

dimension = int(sys.argv[1])
print(json.dumps({
    "object": "list",
    "data": [{"object": "embedding", "embedding": [0.25] * dimension, "index": 0}],
    "model": "__EMBEDDING_MODEL__",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}))
PY
    ;;
  */api/ps*)
    printf '%s\n' '{"models":[{"name":"__EMBEDDING_MODEL__","model":"__EMBEDDING_MODEL__","context_length":8192}]}'
    ;;
  *)
    exit 2
    ;;
esac
""".replace("__EMBEDDING_MODEL__", EMBEDDING_MODEL).strip(),
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "FAKE_READY_FILE": str(ready_file),
        "FAKE_STOPPED_FILE": str(stopped_file),
        "FAKE_OLLAMA_CALLS": str(ollama_calls),
        "FAKE_CURL_CALLS": str(curl_calls),
        "FAKE_EMBEDDING_DIMENSION": str(embedding_dimension),
        "OAMB_OLLAMA_PORT": "18001",
        "OAMB_OLLAMA_BIND_HOST": "127.0.0.1",
        "OAMB_OLLAMA_RELAY_HOST": "127.0.0.1",
        "OAMB_EMBEDDING_STARTUP_ATTEMPTS": "1",
        "OAMB_EMBEDDING_MODEL": EMBEDDING_MODEL,
    }
    return env, ollama_calls, curl_calls, stopped_file


def test_start_ollama_embedding_is_executable_from_readme_command() -> None:
    assert os.access(SCRIPT_PATH, os.X_OK)


def test_start_ollama_embedding_reuses_server_and_verifies_exact_vector(
    tmp_path: Path,
) -> None:
    env, ollama_calls, curl_calls, _ = _fake_environment(tmp_path, ready=True)
    output_path = tmp_path / "output"
    with output_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [str(SCRIPT_PATH)],
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                output.flush()
                if "PASS:" in output_path.read_text(encoding="utf-8"):
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("startup verification timed out")
            assert process.poll() is None, output_path.read_text(encoding="utf-8")
        finally:
            process.terminate()
            process.wait(timeout=5)

    assert ollama_calls.read_text(encoding="utf-8").splitlines() == [
        f"OLLAMA_HOST=http://127.0.0.1:18001 pull {EMBEDDING_MODEL}"
    ]
    curl_call_lines = curl_calls.read_text(encoding="utf-8").splitlines()
    embedding_call = next(line for line in curl_call_lines if "/v1/embeddings" in line)
    assert "http://127.0.0.1:18001/v1/embeddings" in embedding_call
    assert '"dimensions":1024' in embedding_call
    assert any("http://127.0.0.1:18001/api/ps" in line for line in curl_call_lines)
    assert (
        f"PASS: {EMBEDDING_MODEL} returned one finite 1024-dimensional vector "
        "with an 8192-token context" in output_path.read_text(encoding="utf-8")
    )


def test_start_ollama_embedding_starts_and_owns_missing_server(tmp_path: Path) -> None:
    env, ollama_calls, _, stopped_file = _fake_environment(tmp_path, ready=False)
    output_path = tmp_path / "output"
    with output_path.open("w", encoding="utf-8") as output:
        process = subprocess.Popen(
            [str(SCRIPT_PATH)],
            stdout=output,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                output.flush()
                if output_path.exists() and "PASS:" in output_path.read_text(encoding="utf-8"):
                    break
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("startup verification timed out")

            assert process.poll() is None, output_path.read_text(encoding="utf-8")
            assert ollama_calls.read_text(encoding="utf-8").splitlines() == [
                "OLLAMA_HOST=127.0.0.1:18001 serve",
                f"OLLAMA_HOST=http://127.0.0.1:18001 pull {EMBEDDING_MODEL}",
            ]
        finally:
            process.terminate()
            process.wait(timeout=5)

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not stopped_file.exists():
        time.sleep(0.1)
    assert stopped_file.exists()


def test_start_ollama_embedding_rejects_wrong_vector_dimension(tmp_path: Path) -> None:
    env, _, _, _ = _fake_environment(
        tmp_path,
        ready=True,
        embedding_dimension=3,
    )

    result = subprocess.run(
        [str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode != 0
    assert "expected exactly 1024 embedding values" in result.stderr


def test_start_ollama_embedding_rejects_relay_that_stays_unready(tmp_path: Path) -> None:
    env, _, _, _ = _fake_environment(tmp_path, ready=True)
    env["FAKE_RELAY_UNREADY"] = "1"

    result = subprocess.run(
        [str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode != 0
    assert "Docker-gateway relay did not become ready" in result.stderr


def test_start_ollama_embedding_exits_when_owned_server_dies(tmp_path: Path) -> None:
    env, _, _, _ = _fake_environment(tmp_path, ready=False)
    env["FAKE_SERVER_EXIT_AFTER_READY"] = "1"

    result = subprocess.run(
        [str(SCRIPT_PATH)],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )

    assert result.returncode != 0
    assert "owned Ollama server exited after readiness" in result.stderr
