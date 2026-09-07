from __future__ import annotations

import json
import os
import shutil
import signal
import stat
import subprocess
import time
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
PRECHECK_SCRIPT = REPOSITORY_ROOT / "precheck.sh"
RUN_SCRIPT = REPOSITORY_ROOT / "run.sh"
HOST_EMBEDDING_SCRIPT = REPOSITORY_ROOT / "provider-services" / "lib" / "host_embedding.sh"
PLAN_ENVIRONMENT_SCRIPT = REPOSITORY_ROOT / "provider-services" / "lib" / "plan_environment.sh"
PROVIDER_ENVIRONMENT_SCRIPT = REPOSITORY_ROOT / "provider-services" / "lib" / "env.sh"
RESOLVED_PLAN_HASH = "a" * 64
LONGMEMEVAL_SOURCE = (
    REPOSITORY_ROOT / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
)

PROVIDER_MODEL_CONNECTION_ALIASES = (
    "OAMB_HINDSIGHT_LLM_BASE_URL",
    "OAMB_HINDSIGHT_LLM_API_KEY",
    "OAMB_MEM0_LLM_BASE_URL",
    "OAMB_MEM0_LLM_API_KEY",
    "OAMB_OPENVIKING_VLM_BASE_URL",
    "OAMB_OPENVIKING_VLM_API_KEY",
)


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/bash\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _write_configured_environment(root: Path) -> None:
    configured = (
        (REPOSITORY_ROOT / ".env.example")
        .read_text(encoding="utf-8")
        .replace(
            "LLM_BASE_URL=change-me\nLLM_API_KEY=change-me",
            "LLM_BASE_URL=https://models.example/v1\nLLM_API_KEY=test-model-key",
        )
    )
    (root / ".env").write_text(
        configured,
        encoding="utf-8",
    )
    (root / ".env").chmod(0o600)


def _copy_quick_start_script(source: Path, root: Path) -> Path:
    assert source.is_file(), f"quick-start script is missing: {source.name}"
    destination = root / source.name
    shutil.copy2(source, destination)
    if "plan_environment.sh" in source.read_text(encoding="utf-8"):
        helper = root / "provider-services" / "lib" / "plan_environment.sh"
        helper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PLAN_ENVIRONMENT_SCRIPT, helper)
    return destination


def test_root_env_template_owns_one_generic_llm_connection() -> None:
    assignment_names = [
        line.split("=", 1)[0]
        for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert assignment_names.count("LLM_URL_TYPE") == 1
    assert assignment_names.count("LLM_BASE_URL") == 1
    assert assignment_names.count("LLM_API_KEY") == 1
    assert "DEEPSEEK_BASE_URL" not in assignment_names
    assert "DEEPSEEK_API_KEY" not in assignment_names
    assert set(assignment_names).isdisjoint(PROVIDER_MODEL_CONNECTION_ALIASES)
    assert "LLM_URL_TYPE=openai_chat" in (REPOSITORY_ROOT / ".env.example").read_text(
        encoding="utf-8"
    )


def test_provider_shell_derives_model_connections_from_generic_llm_pair(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_URL_TYPE=openai_chat\n"
        "LLM_BASE_URL=https://models.example/v1\n"
        "LLM_API_KEY=test-model-key\n",
        encoding="utf-8",
    )
    requested = " ".join(PROVIDER_MODEL_CONNECTION_ALIASES)

    result = subprocess.run(
        [
            "sh",
            "-c",
            f'. "{PROVIDER_ENVIRONMENT_SCRIPT}"; '
            f'for name in {requested}; do printf "%s=" "$name"; '
            'read_env_value "$1" "$name"; done',
            "sh",
            str(env_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        "OAMB_HINDSIGHT_LLM_BASE_URL=https://models.example/v1",
        "OAMB_HINDSIGHT_LLM_API_KEY=test-model-key",
        "OAMB_MEM0_LLM_BASE_URL=https://models.example/v1",
        "OAMB_MEM0_LLM_API_KEY=test-model-key",
        "OAMB_OPENVIKING_VLM_BASE_URL=https://models.example/v1",
        "OAMB_OPENVIKING_VLM_API_KEY=test-model-key",
    ]


def test_live_environment_derives_provider_connections_from_generic_llm_pair(
    tmp_path: Path,
) -> None:
    from oamb.live import load_live_environment

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_URL_TYPE=openai_chat\n"
        "LLM_BASE_URL=https://models.example/v1\n"
        "LLM_API_KEY=test-model-key\n",
        encoding="utf-8",
    )

    environment = load_live_environment(
        provider_env_path=env_file,
        model_env_path=env_file,
        provider_runtime_directory=tmp_path / "runtime",
        base_environment={
            "OAMB_MEM0_LLM_BASE_URL": "https://stale.example/v1",
            "OAMB_MEM0_LLM_API_KEY": "stale-model-key",
        },
    )

    for name in PROVIDER_MODEL_CONNECTION_ALIASES:
        expected = "https://models.example/v1" if name.endswith("BASE_URL") else "test-model-key"
        assert environment[name] == expected


def test_live_environment_rejects_unsupported_llm_url_type(tmp_path: Path) -> None:
    from oamb.live import LiveConfigurationError, load_live_environment

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_URL_TYPE=anthropic\n"
        "LLM_BASE_URL=https://models.example/v1\n"
        "LLM_API_KEY=test-model-key\n",
        encoding="utf-8",
    )

    with pytest.raises(LiveConfigurationError, match="LLM_URL_TYPE.*openai_chat"):
        load_live_environment(
            provider_env_path=env_file,
            model_env_path=env_file,
            provider_runtime_directory=tmp_path / "runtime",
            base_environment={},
        )


def _quick_start_fixture(tmp_path: Path, *, system_name: str) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / "open-agent-memory-benchmark"
    root.mkdir()
    trace = tmp_path / "trace.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_configured_environment(root)
    (root / "configs").mkdir()
    (root / "configs" / "benchmark.yml").write_text("comparison: test\n", encoding="utf-8")
    (root / "datasets" / "longmemeval-cleaned").mkdir(parents=True)
    (root / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json").write_text(
        "[]\n", encoding="utf-8"
    )
    (root / "outputs" / "tmp" / "provider-source" / "mem0" / ".git").mkdir(parents=True)

    _write_executable(fake_bin / "uname", f"printf '%s\\n' {system_name}")
    _write_executable(fake_bin / "docker", "printf '%s\\n' 172.17.0.1")
    _write_executable(
        fake_bin / "curl",
        """
printf 'curl %s\n' "$*" >> "$OAMB_TEST_TRACE"
case "$*" in
  *host.docker.internal*|*127.0.0.1*|*172.17.0.1*) [ -f "$OAMB_TEST_EMBED_READY" ] || exit 22 ;;
esac
python3 - <<'PY'
import json
print(json.dumps({"object": "list", "data": [{"embedding": [0.0] * 1024, "index": 0}]}))
PY
""".strip(),
    )
    _write_executable(
        fake_bin / "uv",
        """
printf 'uv %s\n' "$*" >> "$OAMB_TEST_TRACE"
case " $* " in
  *" python - "*"longmemeval_s_cleaned.json"*)
    if [ "${OAMB_TEST_INPUT_ENCODING_FAIL:-}" = "1" ]; then
      printf 'planted full-input encoding failure\n' >&2
      exit 46
    fi
    ;;
  *" oamb doctor "*)
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--output" ]; then
        mkdir -p "$2"
        printf '%s\n' '{"schema_name":"resolved_comparison_plan","execution":{"extraction_max_retries":10},"model_roles":[{"role_id":"hindsight_extraction","model":"plan-hindsight","thinking_effort":"low"},{"role_id":"mem0_extraction","model":"plan-mem0","thinking_effort":"high"},{"role_id":"openviking_semantic_understanding","model":"plan-openviking","thinking_effort":"max"},{"role_id":"embedding","model":"plan-embedding","thinking_effort":"not_applicable"}]}' > "$2/resolved-plan.json"
        break
      fi
      shift
    done
    ;;
esac
""".strip(),
    )
    _write_executable(
        root / "scripts" / "download" / "longmemeval.sh",
        'printf \'download %s\\n\' "$*" >> "$OAMB_TEST_TRACE"',
    )
    _write_executable(
        root / "scripts" / "verify_mem0_source.sh",
        'printf \'verify-mem0 %s\\n\' "$*" >> "$OAMB_TEST_TRACE"',
    )
    for name in ("start_vllm_metal.sh", "start_ollama_embedding.sh"):
        _write_executable(
            root / "scripts" / "start_local_embedding" / name,
            f"printf 'embedding {name}\\n' >> \"$OAMB_TEST_TRACE\"; "
            'touch "$OAMB_TEST_EMBED_READY"; sleep 2',
        )
    _write_executable(
        root / "provider-services" / "bin" / "provider-services",
        """
printf 'provider-config %s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
  "${OAMB_HINDSIGHT_LLM_MODEL:-missing}" \
  "${OAMB_MEM0_LLM_MODEL:-missing}" \
  "${OAMB_OPENVIKING_VLM_MODEL:-missing}" \
  "${OAMB_EMBEDDING_MODEL:-missing}" \
  "${OAMB_HINDSIGHT_LLM_REASONING_EFFORT:-missing}" \
  "${OAMB_MEM0_LLM_REASONING_EFFORT:-missing}" \
  "${OAMB_OPENVIKING_VLM_REASONING_EFFORT:-missing}" \
  "${OAMB_HINDSIGHT_LLM_PROVIDER:-missing}" \
  "${OAMB_OPENVIKING_VLM_PROVIDER:-missing}" >> "$OAMB_TEST_TRACE"
printf 'provider-services %s\n' "$*" >> "$OAMB_TEST_TRACE"
""".strip(),
    )
    resolver = root / "provider-services" / "lib" / "host_embedding.sh"
    resolver.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HOST_EMBEDDING_SCRIPT, resolver)
    (root / ".env.example").write_text("", encoding="utf-8")

    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "OAMB_TEST_TRACE": str(trace),
        "OAMB_TEST_EMBED_READY": str(tmp_path / "embedding-ready"),
        "OAMB_EMBEDDING_STARTUP_ATTEMPTS": "3",
    }
    return root, env, trace


@pytest.mark.skipif(
    not LONGMEMEVAL_SOURCE.exists(), reason="download the pinned LongMemEval S dataset"
)
def test_full_input_precheck_encodes_every_frozen_lme60_mem0_source() -> None:
    from oamb.live import validate_lme60_mem0_input_encoding

    assert validate_lme60_mem0_input_encoding(LONGMEMEVAL_SOURCE) == 2_826


def test_precheck_stops_before_provider_start_when_full_input_encoding_fails(
    tmp_path: Path,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env["OAMB_TEST_INPUT_ENCODING_FAIL"] = "1"

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 46
    assert "planted full-input encoding failure" in result.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert any(
        "uv run --locked python -" in line and "longmemeval_s_cleaned.json" in line
        for line in calls
    )
    assert not any(line.startswith("provider-services ") for line in calls)
    assert not any(line.startswith("embedding ") for line in calls)
    assert not any(line.startswith("curl ") for line in calls)


@pytest.mark.parametrize(
    ("system_name", "expected_helper"),
    (("Darwin", "start_vllm_metal.sh"), ("Linux", "start_ollama_embedding.sh")),
)
def test_precheck_routes_default_local_embedding_by_operating_system(
    tmp_path: Path,
    system_name: str,
    expected_helper: str,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name=system_name)
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8")
    assert f"embedding {expected_helper}" in calls
    assert "uv sync --locked --all-groups" in calls
    assert "provider-services verify --services" in calls
    assert "provider-services verify --model-readiness" in calls
    if system_name == "Linux":
        assert "http://172.17.0.1:18000/v1/embeddings" in calls
    state = json.loads((root / "outputs" / "tmp" / "quick-start-current.json").read_bytes())
    assert Path(state["resolved_plan"]).is_file()
    assert state["question_id"] == "72e3ee87"


def test_precheck_publishes_state_and_plan_below_outputs_root(tmp_path: Path) -> None:
    root, env, _trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    state = json.loads(state_path.read_bytes())
    outputs_root = (root / "outputs").resolve()
    work_dir = Path(state["work_dir"])
    resolved_plan = Path(state["resolved_plan"])
    assert work_dir.parent == outputs_root / "tmp" / "precheck"
    assert resolved_plan == work_dir / "plan" / "resolved-plan.json"
    assert not (root / ".local-demo").exists()


def test_precheck_online_embedding_url_skips_local_server(tmp_path: Path) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)

    result = subprocess.run(
        [str(script), "--embedding-api-url", "https://embedding.example/v1"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8")
    assert "embedding start_" not in calls
    root_env = (root / ".env").read_text(encoding="utf-8")
    assert "OAMB_EMBEDDING_BASE_URL=https://embedding.example/v1" in root_env
    assert not (root / "provider-services" / ".env").exists()


def test_precheck_completes_single_root_env_without_provider_copy(tmp_path: Path) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    root_template = REPOSITORY_ROOT / ".env.example"
    configured = root_template.read_text(encoding="utf-8").replace(
        "LLM_BASE_URL=change-me\nLLM_API_KEY=change-me",
        "LLM_BASE_URL=https://models.example/v1\nLLM_API_KEY=test-model-key",
    )
    (root / ".env.example").write_text(root_template.read_text(encoding="utf-8"), encoding="utf-8")
    configured_lines = [
        line
        for line in configured.splitlines()
        if line.split("=", 1)[0] not in PROVIDER_MODEL_CONNECTION_ALIASES
    ]
    legacy_provider_connections = "\n".join(
        f"{name}=" + ("https://legacy.example/v1" if name.endswith("BASE_URL") else "legacy-key")
        for name in PROVIDER_MODEL_CONNECTION_ALIASES
    )
    (root / ".env").write_text(
        "\n".join(configured_lines)
        + "\n"
        + legacy_provider_connections
        + "\n"
        + "OMBA_ANSWER_LLM=openai\n"
        + "OMBA_ANSWER_MODEL=legacy-answer\n"
        + "OPENAI_BASE_URL=https://legacy.example/v1\n"
        + "OPENAI_API_KEY=legacy-key\n"
        + "# legacy alternate credentials\n"
        + "# DEEPSEEK_BASE_URL=https://legacy.example/v1\n"
        + "# DEEPSEEK_API_KEY=legacy-key\n",
        encoding="utf-8",
    )
    (root / ".env").chmod(0o600)
    (root / "provider-services" / ".env").unlink(missing_ok=True)
    (root / "provider-services" / ".env.example").unlink(missing_ok=True)

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    root_env_path = root / ".env"
    root_env = root_env_path.read_text(encoding="utf-8")
    assert "OAMB_HINDSIGHT_PORT=18888" in root_env
    for alias in PROVIDER_MODEL_CONNECTION_ALIASES:
        assert f"{alias}=" not in root_env
    assert "OAMB_PROVIDER_PROJECT=oamb-providers-lme60-" in root_env
    assert "OAMB_MEM0_SOURCE_CHECKOUT=" in root_env
    for plan_owned in (
        "OAMB_EMBEDDING_MODEL",
        "OAMB_HINDSIGHT_LLM_PROVIDER",
        "OAMB_HINDSIGHT_LLM_MODEL",
        "OAMB_HINDSIGHT_LLM_REASONING_EFFORT",
        "OAMB_MEM0_LLM_MODEL",
        "OAMB_MEM0_LLM_REASONING_EFFORT",
        "OAMB_OPENVIKING_VLM_PROVIDER",
        "OAMB_OPENVIKING_VLM_MODEL",
        "OAMB_OPENVIKING_VLM_REASONING_EFFORT",
    ):
        assert f"{plan_owned}=" not in root_env
    for stale in (
        "OMBA_ANSWER_LLM",
        "OMBA_ANSWER_MODEL",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "legacy alternate credentials",
        "# DEEPSEEK_BASE_URL=",
        "# DEEPSEEK_API_KEY=",
    ):
        assert stale not in root_env
    assert (
        "provider-config plan-hindsight|plan-mem0|plan-openviking|plan-embedding|low|high|max|openai|openai"
        in (trace.read_text(encoding="utf-8"))
    )
    assert any(
        line.startswith("curl ") and "plan-embedding" in line
        for line in trace.read_text(encoding="utf-8").splitlines()
    )
    assert not any(
        line.split("=", 1)[1].startswith("change-me")
        for line in root_env.splitlines()
        if line.startswith("OAMB_") and "=" in line
    )
    assert stat.S_IMODE(root_env_path.stat().st_mode) == 0o600
    assert not (root / "provider-services" / ".env").exists()


def test_precheck_upgrades_legacy_two_key_root_env(tmp_path: Path) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    root_env_path = root / ".env"
    root_env_path.write_text(
        "DEEPSEEK_BASE_URL=https://models.example/v1\nDEEPSEEK_API_KEY=test-model-key\n",
        encoding="utf-8",
    )
    root_env_path.chmod(0o600)

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    root_env = root_env_path.read_text(encoding="utf-8")
    assert "LLM_URL_TYPE=openai_chat" in root_env
    assert "LLM_BASE_URL=https://models.example/v1" in root_env
    assert "LLM_API_KEY=test-model-key" in root_env
    assert "DEEPSEEK_BASE_URL=" not in root_env
    assert "DEEPSEEK_API_KEY=" not in root_env
    for expected in (
        "OAMB_HINDSIGHT_PORT=18888",
        "OAMB_MEM0_PORT=18889",
        "OAMB_MEM0_INSPECTOR_PORT=16333",
        "OAMB_OPENVIKING_PORT=19330",
        "OAMB_OPENVIKING_ACCOUNT_ID=oamb-benchmark",
        "OAMB_OPENVIKING_ADMIN_USER_ID=oamb-admin",
    ):
        assert expected in root_env
    assert (
        "provider-config plan-hindsight|plan-mem0|plan-openviking|plan-embedding|low|high|max|openai|openai"
        in (trace.read_text(encoding="utf-8"))
    )
    assert not (root / "provider-services" / ".env").exists()


@pytest.mark.parametrize(
    ("key", "duplicate_value"),
    (
        ("DEEPSEEK_BASE_URL", "https://duplicate.example/v1"),
        ("DEEPSEEK_API_KEY", "duplicate-model-key"),
    ),
)
def test_precheck_rejects_duplicate_legacy_llm_assignment_without_mutation(
    tmp_path: Path,
    key: str,
    duplicate_value: str,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    env_path.write_text(
        "DEEPSEEK_BASE_URL=https://models.example/v1\n"
        "DEEPSEEK_API_KEY=test-model-key\n"
        f"{key}={duplicate_value}\n",
        encoding="utf-8",
    )
    original = env_path.read_bytes()

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert f"precheck: FAIL: {key} must occur exactly once in .env" in result.stderr
    assert env_path.read_bytes() == original
    assert "provider-services " not in trace.read_text(encoding="utf-8")


def test_precheck_rejects_symlinked_root_env_without_mutating_target(tmp_path: Path) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    target = tmp_path / "shared.env"
    env_path.replace(target)
    env_path.symlink_to(target)
    original = target.read_bytes()

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert "precheck: FAIL: .env must be a regular file, not a symbolic link" in result.stderr
    assert env_path.is_symlink()
    assert target.read_bytes() == original
    assert "provider-services " not in trace.read_text(encoding="utf-8")


def test_precheck_rejects_unsupported_llm_url_type(tmp_path: Path) -> None:
    root, env, _trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "LLM_URL_TYPE=openai_chat",
            "LLM_URL_TYPE=anthropic",
            1,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert "LLM_URL_TYPE" in result.stderr
    assert "openai_chat" in result.stderr


@pytest.mark.parametrize(
    ("key", "duplicate_value"),
    (
        ("LLM_URL_TYPE", "openai_chat"),
        ("LLM_BASE_URL", "https://duplicate.example/v1"),
        ("LLM_API_KEY", "duplicate-model-key"),
    ),
)
def test_precheck_rejects_duplicate_generic_llm_assignment_before_provider_start(
    tmp_path: Path,
    key: str,
    duplicate_value: str,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    with env_path.open("a", encoding="utf-8") as stream:
        stream.write(f"{key}={duplicate_value}\n")

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert f"precheck: FAIL: {key} must occur exactly once in .env" in result.stderr
    assert "provider-services " not in trace.read_text(encoding="utf-8")


def test_precheck_requires_operator_prepared_root_env(tmp_path: Path) -> None:
    root, env, _trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    (root / ".env").unlink()

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert "prepare .env from .env.example" in result.stderr
    assert not (root / ".env").exists()


def _write_fake_oamb(fake_bin: Path) -> None:
    _write_executable(
        fake_bin / "uv",
        r"""
	trace_lock="$OAMB_TEST_TRACE.lock"
	trace_lock_owned=false
	cleanup_trace_lock() {
	  if [ "$trace_lock_owned" = true ]; then
	    rmdir "$trace_lock" 2>/dev/null || true
	  fi
	}
	trap 'cleanup_trace_lock; exit 143' TERM INT HUP
	trap cleanup_trace_lock EXIT
	while ! mkdir "$trace_lock" 2>/dev/null; do /bin/sleep 0.01; done
	trace_lock_owned=true
	printf 'uv %s\n' "$*" >> "$OAMB_TEST_TRACE"
	rmdir "$trace_lock"
	trace_lock_owned=false
	trap - EXIT TERM INT HUP
	if [[ " $* " == *" python - "*"longmemeval_s_cleaned.json"* ]] && \
	   [ "${OAMB_TEST_INPUT_ENCODING_FAIL:-}" = "1" ]; then
	  printf 'planted full-input encoding failure\n' >&2
	  exit 46
	fi
	case " $* " in
	  *" oamb run "*)
	    cells=()
	    result_map=""
	    output_root=""
	    recovery_analysis_output=""
	    recover_from=()
	    while [ "$#" -gt 0 ]; do
	      case "$1" in
	        --cell) cells+=("$2"); shift 2 ;;
	        --result-map) result_map=$2; shift 2 ;;
	        --output-root) output_root=$2; shift 2 ;;
	        --recover-from) recover_from+=("$2"); shift 2 ;;
	        --recovery-analysis-output) recovery_analysis_output=$2; shift 2 ;;
	        *) shift ;;
	      esac
	    done
	    if [ -n "$recovery_analysis_output" ]; then
	      cell=${cells[0]}
	      if [ "${OAMB_TEST_RECOVERY_ANALYSIS_FAIL_CELL:-}" = "$cell" ]; then
	        printf 'planted recovery analysis failure for %s\n' "$cell" >&2
	        exit 43
	      fi
	      mkdir -p "$(dirname "$recovery_analysis_output")"
	      recovery_reusable=${OAMB_TEST_RECOVERY_REUSABLE:-30}
	      recovery_remaining=${OAMB_TEST_RECOVERY_REMAINING:-30}
	      recovery_quarantined=${OAMB_TEST_RECOVERY_QUARANTINED:-1}
	      for recovery_part in "${recover_from[@]}"; do
	        if [ -f "$recovery_part/fake-completed" ]; then
	          recovery_reusable=60
	          recovery_remaining=0
	          recovery_quarantined=0
	        elif [ -f "$recovery_part/fake-failed" ] && [ "$recovery_reusable" -lt 45 ]; then
	          recovery_reusable=45
	          recovery_remaining=15
	          recovery_quarantined=1
	        fi
	      done
	      python3 - "$recovery_analysis_output" "$cell" "$OAMB_TEST_PLAN_HASH" \
	        "$recovery_reusable" "$recovery_remaining" "$recovery_quarantined" \
	        "${#recover_from[@]}" <<'PY'
import json
import sys
from pathlib import Path

output, cell, plan_hash = sys.argv[1:4]
reusable, remaining, quarantined, source_count = map(int, sys.argv[4:])
document = {
    "schema_name": "capsule_recovery_analysis",
    "schema_version": 1,
    "resolved_plan_hash": plan_hash,
    "cell_id": cell,
    "source_manifest_sha256s": [f"{index + 1:064x}" for index in range(source_count)],
    "reusable_ingestion_plan_ids": [f"reusable-{index}" for index in range(reusable)],
    "quarantined_ingestion_plan_ids": [
        f"quarantined-{index}" for index in range(quarantined)
    ],
    "remaining_ingestion_plan_ids": [f"remaining-{index}" for index in range(remaining)],
    "remaining_case_manifest_entry_ids": [f"case-{index}" for index in range(remaining)],
}
Path(output).write_text(json.dumps(document) + "\n", encoding="utf-8")
PY
	      exit 0
	    fi
	    mkdir -p "$output_root" "$(dirname "$result_map")"
	    if [ "${#recover_from[@]}" -gt 0 ]; then
	      cell=${cells[0]}
	      capsule="$output_root/$cell-recovery-capsule"
	      mkdir -p "$capsule"
	      active_marker=""
	      cleanup_recovery_marker() {
	        [ -z "$active_marker" ] || rm -f "$active_marker"
	      }
	      write_interrupted_recovery() {
	        printf '{"schema_name":"live_run_result_map","schema_version":1,"resolved_plan_hash":"%s","status":"failed","cells":[{"cell_id":"%s","status":"failed","capsule_root":"%s","detail":"planted interruption"}],"capsule_roots":{}}\n' \
	          "$OAMB_TEST_PLAN_HASH" "$cell" "$capsule" > "$result_map"
	      }
	      handle_recovery_signal() {
	        : > "$capsule/fake-failed"
	        write_interrupted_recovery
	        cleanup_recovery_marker
	        exit 44
	      }
	      trap handle_recovery_signal TERM INT HUP
	      if [ -n "${OAMB_TEST_RECOVERY_ACTIVE_DIR:-}" ]; then
	        mkdir -p "$OAMB_TEST_RECOVERY_ACTIVE_DIR"
	        active_marker="$OAMB_TEST_RECOVERY_ACTIVE_DIR/$cell"
	        : > "$active_marker"
	        active_count="$(find "$OAMB_TEST_RECOVERY_ACTIVE_DIR" -type f | wc -l | tr -d ' ')"
	        if [ "$active_count" -gt "${OAMB_TEST_RECOVERY_MAX_ACTIVE:-3}" ]; then
	          cleanup_recovery_marker
	          exit 46
	        fi
	        if [ "${OAMB_TEST_RECOVERY_PAIR_BARRIER:-0}" = "1" ] && \
	          [ "$cell" != "openviking-lme60" ]; then
	          deadline=$((SECONDS + 5))
	          if [ "$(find "$OAMB_TEST_RECOVERY_ACTIVE_DIR" -type f | wc -l | tr -d ' ')" -ge 2 ]; then
	            : > "$OAMB_TEST_RECOVERY_ACTIVE_DIR.pair-ready"
	          fi
	          while [ ! -f "$OAMB_TEST_RECOVERY_ACTIVE_DIR.pair-ready" ]; do
	            if [ "$(find "$OAMB_TEST_RECOVERY_ACTIVE_DIR" -type f | wc -l | tr -d ' ')" -ge 2 ]; then
	              : > "$OAMB_TEST_RECOVERY_ACTIVE_DIR.pair-ready"
	            fi
	            [ "$SECONDS" -lt "$deadline" ] || exit 47
	            /bin/sleep 0.01
	          done
	        fi
	      fi
	      if [ -n "${OAMB_TEST_RECOVERY_BARRIER:-}" ]; then
	        mkdir -p "$OAMB_TEST_RECOVERY_BARRIER"
	        : > "$OAMB_TEST_RECOVERY_BARRIER/$cell"
	        deadline=$((SECONDS + 5))
	        while [ "$(find "$OAMB_TEST_RECOVERY_BARRIER" -type f | wc -l | tr -d ' ')" -lt \
	          "${OAMB_TEST_RECOVERY_BARRIER_COUNT:-3}" ]; do
	          [ "$SECONDS" -lt "$deadline" ] || exit 45
	          /bin/sleep 0.01
	        done
	      fi
	      if [ -n "${OAMB_TEST_RECOVERY_SIGNAL_READY:-}" ]; then
	        mkdir -p "$OAMB_TEST_RECOVERY_SIGNAL_READY"
	        : > "$OAMB_TEST_RECOVERY_SIGNAL_READY/$cell"
	        while :; do /bin/sleep 0.05; done
	      fi
	      if [ "${OAMB_TEST_RECOVERY_FAIL:-0}" = "1" ] || \
	        [ "${OAMB_TEST_RECOVERY_FAIL_CELL:-}" = "$cell" ]; then
	        : > "$capsule/fake-failed"
	        write_interrupted_recovery
	        cleanup_recovery_marker
	        exit 44
	      fi
	      printf '{"schema_name":"live_run_result_map","schema_version":1,"resolved_plan_hash":"%s","status":"completed","cells":[{"cell_id":"%s","status":"completed","capsule_root":"%s","detail":null}],"capsule_roots":{"%s":"%s"}}\n' \
	        "$OAMB_TEST_PLAN_HASH" "$cell" "$capsule" "$cell" "$capsule" > "$result_map"
	      : > "$capsule/fake-completed"
	      cleanup_recovery_marker
	      exit 0
	    fi
	    progress_cases=${OAMB_TEST_PROGRESS_CASES:-0}
    if [ "$progress_cases" -gt 0 ]; then
      progress_cells=()
      if [ "${#cells[@]}" -gt 0 ]; then
        progress_cells=("${cells[@]}")
      else
        progress_cells=("hindsight-lme60" "mem0-lme60" "openviking-lme60")
      fi
      for progress_cell in "${progress_cells[@]}"; do
        progress_provider=${progress_cell%-lme60}
        case "$progress_provider" in
          hindsight) provider_progress_cases=${OAMB_TEST_PROGRESS_HINDSIGHT:-$progress_cases} ;;
          mem0) provider_progress_cases=${OAMB_TEST_PROGRESS_MEM0:-$progress_cases} ;;
          openviking) provider_progress_cases=${OAMB_TEST_PROGRESS_OPENVIKING:-$progress_cases} ;;
        esac
        progress_root="$output_root/progress-$progress_provider"
        mkdir -p "$progress_root/source/specs" "$progress_root/source/cases"
        printf '{"memory_system_id":"%s","run_id":"%s-run"}\n' \
          "$progress_provider" "$progress_provider" \
          > "$progress_root/source/specs/run-spec.json"
        progress_index=0
        while [ "$progress_index" -lt "$provider_progress_cases" ]; do
          printf '{}\n' > "$progress_root/source/cases/$progress_index.json"
          progress_index=$((progress_index + 1))
        done
      done
    fi
    if [ -n "${OAMB_TEST_PROGRESS_RUN_RECORDS:-}" ]; then
      python3 - "$output_root" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
for provider, records in json.loads(os.environ["OAMB_TEST_PROGRESS_RUN_RECORDS"]).items():
    run_root = output_root / f"progress-{provider}" / "source" / "run"
    run_root.mkdir(parents=True)
    for index, record in enumerate(records):
        name = f"{provider}-run" if index == 0 else f"duplicate-{index}"
        payload = record if isinstance(record, str) else json.dumps(record)
        (run_root / f"{name}.json").write_text(payload, encoding="utf-8")
duplicate = os.environ.get("OAMB_TEST_PROGRESS_DUPLICATE_PROVIDER")
if duplicate:
    shutil.copytree(output_root / f"progress-{duplicate}", output_root / "duplicate-provider")
PY
    fi
    if [ "${OAMB_TEST_OAMB_RUN_DELAY:-0}" = "1" ]; then
      /bin/sleep 1
    fi
    if [ "${OAMB_TEST_OAMB_RUN_FAIL:-0}" = "1" ]; then
      exit 44
    fi
    [ ! -e "$result_map" ] || exit 91
    if [ "${#cells[@]}" -gt 0 ]; then
      python3 - "$result_map" "$output_root" "$OAMB_TEST_PLAN_HASH" "${cells[@]}" <<'PY'
import json
import sys
from pathlib import Path

result_map = Path(sys.argv[1])
output_root = Path(sys.argv[2])
plan_hash = sys.argv[3]
cells = sys.argv[4:]
roots = {cell: output_root / f"{cell}-capsule" for cell in cells}
for root in roots.values():
    root.mkdir(parents=True, exist_ok=True)
document = {
    "schema_name": "live_run_result_map",
    "schema_version": 1,
    "resolved_plan_hash": plan_hash,
    "status": "completed",
    "cells": [
        {
            "cell_id": cell,
            "status": "completed",
            "capsule_root": str(roots[cell]),
            "detail": None,
        }
        for cell in cells
    ],
    "capsule_roots": {cell: str(roots[cell]) for cell in cells},
}
result_map.write_text(json.dumps(document) + "\n", encoding="utf-8")
PY
    else
      hindsight="$output_root/hindsight-lme60-capsule"
      mem0="$output_root/mem0-lme60-capsule"
      openviking="$output_root/openviking-lme60-capsule"
      mkdir -p "$hindsight" "$mem0" "$openviking"
      printf '{"schema_name":"live_run_result_map","schema_version":1,"resolved_plan_hash":"%s","status":"completed","cells":[{"cell_id":"hindsight-lme60","status":"completed","capsule_root":"%s","detail":null},{"cell_id":"mem0-lme60","status":"completed","capsule_root":"%s","detail":null},{"cell_id":"openviking-lme60","status":"completed","capsule_root":"%s","detail":null}],"capsule_roots":{"hindsight-lme60":"%s","mem0-lme60":"%s","openviking-lme60":"%s"}}\n' \
        "$OAMB_TEST_PLAN_HASH" "$hindsight" "$mem0" "$openviking" \
        "$hindsight" "$mem0" "$openviking" > "$result_map"
	    fi
	    ;;
	  *" oamb capsule compose "*)
	    output=""
	    cell=""
	    while [ "$#" -gt 0 ]; do
	      case "$1" in
	        --cell) cell=$2; shift 2 ;;
	        --output) output=$2; shift 2 ;;
	        *) shift ;;
	      esac
	    done
	    [ ! -e "$output" ] || exit 93
	    mkdir -p "$output"
	    if [ "${OAMB_TEST_COMPOSE_SIGNAL_CELL:-}" = "$cell" ]; then
	      mkdir -p "$OAMB_TEST_COMPOSE_SIGNAL_READY"
	      : > "$OAMB_TEST_COMPOSE_SIGNAL_READY/$cell"
	      while [ ! -f "$OAMB_TEST_COMPOSE_SIGNAL_RELEASE" ]; do /bin/sleep 0.02; done
	    fi
	    ;;
	  *" oamb capsule validate "*)
	    if [ -n "${OAMB_TEST_INVALID_RECOVERY_VALIDATION_CELL:-}" ] && \
	      [[ "$*" == *"${OAMB_TEST_INVALID_RECOVERY_VALIDATION_CELL}-recovery-capsule"* ]]; then
	      exit 48
	    fi
    output=""
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--output" ]; then output=$2; break; fi
      shift
    done
    mkdir -p "$(dirname "$output")"
    printf '%s\n' '{"disposition":"validated"}' > "$output"
    ;;
  *" oamb compare "*)
    output_root=""
    diagnostic=false
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --output-root) output_root=$2; shift 2 ;;
        --diagnostic) diagnostic=true; shift ;;
        *) shift ;;
      esac
    done
    [ ! -e "$output_root" ] || exit 92
    mkdir -p "$output_root"
    if [ "$diagnostic" = true ]; then cases=1; results=3; else cases=60; results=180; fi
    printf '{"diagnostic":%s,"coverage":{"cell_count":3,"unique_case_count":%s,"provider_specific_result_count":%s}}\n' \
      "$diagnostic" "$cases" "$results" > "$output_root/report.json"
    printf '%s\n' '<!doctype html><title>OAMB comparison report</title>' > "$output_root/report.html"
    ;;
esac
""".strip(),
    )


def _run_fixture(
    tmp_path: Path, *, system_name: str = "Darwin"
) -> tuple[Path, dict[str, str], Path]:
    root = tmp_path / "open-agent-memory-benchmark"
    root.mkdir()
    trace = tmp_path / "trace.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_fake_oamb(fake_bin)
    _write_executable(fake_bin / "uname", f"printf '%s\\n' {system_name}")
    _write_executable(
        fake_bin / ("open" if system_name == "Darwin" else "xdg-open"),
        'printf \'open %s\\n\' "$*" >> "$OAMB_TEST_TRACE"',
    )
    _write_executable(
        root / "provider-services" / "bin" / "provider-services",
        """
printf 'provider-config %s|%s|%s|%s|%s|%s|%s|%s|%s\n' \
  "${OAMB_HINDSIGHT_LLM_MODEL:-missing}" \
  "${OAMB_MEM0_LLM_MODEL:-missing}" \
  "${OAMB_OPENVIKING_VLM_MODEL:-missing}" \
  "${OAMB_EMBEDDING_MODEL:-missing}" \
  "${OAMB_HINDSIGHT_LLM_REASONING_EFFORT:-missing}" \
  "${OAMB_MEM0_LLM_REASONING_EFFORT:-missing}" \
  "${OAMB_OPENVIKING_VLM_REASONING_EFFORT:-missing}" \
  "${OAMB_HINDSIGHT_LLM_PROVIDER:-missing}" \
  "${OAMB_OPENVIKING_VLM_PROVIDER:-missing}" >> "$OAMB_TEST_TRACE"
printf 'provider-services %s\n' "$*" >> "$OAMB_TEST_TRACE"
""".strip(),
    )
    (root / "provider-services" / ".runtime").mkdir()
    (root / ".env").write_text(
        "LLM_URL_TYPE=openai_chat\nLLM_BASE_URL=test\nLLM_API_KEY=test\n",
        encoding="utf-8",
    )
    (root / ".env").chmod(0o600)
    work_dir = root / "outputs" / "tmp" / "precheck" / "lme60-test"
    plan = work_dir / "plan" / "resolved-plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "resolved_plan_hash": RESOLVED_PLAN_HASH,
                "execution": {
                    "max_parallel_providers_per_dataset": 3,
                    "extraction_max_retries": 10,
                },
                "model_roles": [
                    {
                        "role_id": "hindsight_extraction",
                        "model": "plan-hindsight",
                        "thinking_effort": "low",
                    },
                    {
                        "role_id": "mem0_extraction",
                        "model": "plan-mem0",
                        "thinking_effort": "high",
                    },
                    {
                        "role_id": "openviking_semantic_understanding",
                        "model": "plan-openviking",
                        "thinking_effort": "max",
                    },
                    {
                        "role_id": "embedding",
                        "model": "plan-embedding",
                        "thinking_effort": "not_applicable",
                    },
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = root / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("[]\n", encoding="utf-8")
    state = {
        "run_label": "lme60-test",
        "work_dir": str(work_dir),
        "resolved_plan": str(plan),
        "dataset_source": str(dataset),
        "question_id": "72e3ee87",
    }
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "OAMB_TEST_TRACE": str(trace),
        "OAMB_TEST_PLAN_HASH": RESOLVED_PLAN_HASH,
    }
    return root, env, trace


@pytest.mark.parametrize("doctor_exit_code", (0, 47))
def test_run_saves_stdout_and_stderr_without_changing_exit_status(
    tmp_path: Path, doctor_exit_code: int
) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    release = tmp_path / "release-doctor"
    env["OAMB_TEST_LOG_RELEASE"] = str(release)
    _write_executable(
        root / "provider-services" / "bin" / "provider-services",
        "printf 'doctor output sentinel\\n'\n"
        "printf 'doctor error sentinel\\n' >&2\n"
        'while [ ! -f "$OAMB_TEST_LOG_RELEASE" ]; do /bin/sleep 0.02; done\n'
        f"exit {doctor_exit_code}",
    )

    process = subprocess.Popen(
        [str(script), "--full_test", "--dry-run"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            logs = tuple((root / "outputs" / "tmp").glob("run-full-*.log"))
            if len(logs) == 1:
                saved = logs[0].read_text(encoding="utf-8")
                if "doctor output sentinel" in saved and "doctor error sentinel" in saved:
                    break
            time.sleep(0.02)
        else:
            pytest.fail("stdout and stderr were not logged while the command was running")
        assert process.poll() is None
    finally:
        release.touch()
        stdout, stderr = process.communicate(timeout=20)

    assert process.returncode == doctor_exit_code, stdout + stderr
    assert "doctor output sentinel" in stdout
    assert "doctor error sentinel" in stderr
    logs = tuple((root / "outputs" / "tmp").glob("run-full-*.log"))
    assert len(logs) == 1
    log = logs[0]
    assert f"run: log={log}" in stdout
    saved = log.read_text(encoding="utf-8")
    assert all(line in saved for line in (stdout + stderr).splitlines())
    assert stat.S_IMODE(log.stat().st_mode) == 0o600
    assert not tuple((root / "outputs" / "tmp").glob(".run-log.*"))


@pytest.mark.parametrize(
    ("arguments", "expected_mode", "expected_questions"),
    ((["--dry-run"], "smoke", 1), (["--full_test", "--dry-run"], "full", 60)),
)
def test_run_dry_run_validates_without_dispatch(
    tmp_path: Path,
    arguments: list[str],
    expected_mode: str,
    expected_questions: int,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    result = subprocess.run(
        [str(script), *arguments],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8")
    assert "provider-services doctor" in calls
    assert (
        "provider-config plan-hindsight|plan-mem0|plan-openviking|plan-embedding|low|high|max|openai|openai"
        in calls
    )
    assert "uv run --locked python -" in calls
    assert "longmemeval_s_cleaned.json" in calls
    assert "oamb run" not in calls
    assert "oamb capsule validate" not in calls
    assert "oamb compare" not in calls
    assert "open " not in calls
    output_branch = "smoke-test" if expected_mode == "smoke" else "full-test"
    assert not (root / "outputs" / output_branch / "lme60-test").exists()
    assert (
        f"run: PASS (dry-run, {expected_mode}, {expected_questions} questions, "
        "3 providers, zero model/provider calls)"
    ) in result.stdout


def test_run_full_input_encoding_failure_prevents_benchmark_dispatch(tmp_path: Path) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    env["OAMB_TEST_INPUT_ENCODING_FAIL"] = "1"

    result = subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 46
    assert "planted full-input encoding failure" in result.stderr
    calls = trace.read_text(encoding="utf-8")
    assert "provider-services doctor" in calls
    assert "longmemeval_s_cleaned.json" in calls
    assert "oamb run" not in calls


def test_run_smoke_runs_the_dry_run_gate_before_dispatch(tmp_path: Path) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    result = subprocess.run(
        [str(script), "--smoke_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    doctor_index = calls.index("provider-services doctor")
    readiness_index = next(
        index for index, line in enumerate(calls) if "uv run --locked python -" in line
    )
    dispatch_index = next(index for index, line in enumerate(calls) if "oamb run" in line)
    assert doctor_index < readiness_index < dispatch_index


@pytest.mark.parametrize(
    ("arguments", "output_branch"),
    (([], "smoke-test"), (["--full_test"], "full-test")),
)
def test_run_without_resume_starts_a_fresh_execution_every_time(
    tmp_path: Path,
    arguments: list[str],
    output_branch: str,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    first = subprocess.run(
        [str(script), *arguments],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    original_logs = {
        path: path.read_bytes() for path in (root / "outputs" / "tmp").glob("run-*.log")
    }
    assert len(original_logs) == 1
    first_trace_lines = trace.read_text(encoding="utf-8").splitlines()
    second = subprocess.run(
        [str(script), *arguments],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    logs = tuple((root / "outputs" / "tmp").glob("run-*.log"))
    assert len(logs) == 2
    assert all(path.read_bytes() == content for path, content in original_logs.items())
    execution_roots = sorted((root / "outputs" / output_branch).iterdir())
    assert len(execution_roots) == 2
    assert execution_roots[0] != execution_roots[1]
    assert "reusing-completed-capsule" not in second.stdout
    second_trace_lines = trace.read_text(encoding="utf-8").splitlines()[len(first_trace_lines) :]
    assert any("oamb run" in line for line in second_trace_lines)
    assert not any(f"{execution_roots[0]}/" in line for line in second_trace_lines)


@pytest.mark.parametrize(
    ("arguments", "output_branch"),
    ((["--smoke_test"], "smoke-test"), (["--full_test"], "full-test")),
)
def test_run_routes_results_to_mode_specific_outputs_directory(
    tmp_path: Path,
    arguments: list[str],
    output_branch: str,
) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    result = subprocess.run(
        [str(script), *arguments],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    report = root / "outputs" / output_branch / "lme60-test" / "comparison" / "report.json"
    assert report.is_file()
    logs = tuple((root / "outputs" / "tmp").glob("run-*.log"))
    assert len(logs) == 1
    saved = logs[0].read_text(encoding="utf-8")
    assert "run: PASS" in saved
    assert all(line in saved for line in (result.stdout + result.stderr).splitlines())
    assert not (root / ".local-demo").exists()


def test_run_defaults_to_smoke_and_builds_one_question_comparison(tmp_path: Path) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    result = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8")
    run_calls = [line for line in calls.splitlines() if "oamb run" in line]
    assert len(run_calls) == 1
    for cell in ("hindsight-lme60", "mem0-lme60", "openviking-lme60"):
        assert f"--cell {cell}" in run_calls[0]
    assert "--question 72e3ee87" in run_calls[0]
    assert calls.count("oamb capsule validate") == 3
    assert "oamb compare" in calls and "--diagnostic" in calls
    assert "open " in calls and "/outputs/smoke-test/lme60-test/comparison/report.html" in calls
    report = json.loads(
        (root / "outputs" / "smoke-test" / "lme60-test" / "comparison" / "report.json").read_bytes()
    )
    assert report["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 1,
        "provider_specific_result_count": 3,
    }


def test_run_reports_live_cell_progress_while_provider_is_running(tmp_path: Path) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    fake_bin = Path(env["PATH"].split(os.pathsep, 1)[0])
    _write_executable(fake_bin / "sleep", "/bin/sleep 0.05")
    env["OAMB_TEST_OAMB_RUN_DELAY"] = "1"
    env["OAMB_TEST_PROGRESS_CASES"] = "1"

    result = subprocess.run(
        [str(script), "--smoke_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "run: providers=hindsight,mem0,openviking status=starting (smoke, 1 question each)"
        in result.stdout
    )
    for provider in ("hindsight", "mem0", "openviking"):
        progress_line = next(
            line
            for line in result.stdout.splitlines()
            if f"provider={provider} status=running" in line
        )
        assert progress_line.startswith(f"provider={provider} status=running, elapsed=")
        assert "history_rebuild_attempts=0" in progress_line
        assert progress_line.endswith("completed_questions=1 (1/1, 100%)")
        assert f"run: provider={provider} status=validating" in result.stdout
        assert f"run: provider={provider} status=completed" in result.stdout
    assert "run: status=building-comparison" in result.stdout
    assert "run: status=opening-report" in result.stdout


def test_run_full_reports_progress_per_provider_out_of_sixty(tmp_path: Path) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    fake_bin = Path(env["PATH"].split(os.pathsep, 1)[0])
    _write_executable(fake_bin / "sleep", "/bin/sleep 0.05")
    env["OAMB_TEST_OAMB_RUN_DELAY"] = "1"
    env["OAMB_TEST_PROGRESS_CASES"] = "1"
    env["OAMB_TEST_PROGRESS_HINDSIGHT"] = "1"
    env["OAMB_TEST_PROGRESS_MEM0"] = "2"
    env["OAMB_TEST_PROGRESS_OPENVIKING"] = "3"

    result = subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    expected_progress = {
        "hindsight": (1, 1),
        "mem0": (2, 3),
        "openviking": (3, 5),
    }
    for provider, (completed, percentage) in expected_progress.items():
        assert (
            f"run: provider={provider} status=starting, completed_operations=0, "
            "completed_questions=0, question_progress=0% (0/60)"
        ) in result.stdout
        progress_line = next(
            line
            for line in result.stdout.splitlines()
            if f"provider={provider} status=running" in line
        )
        assert progress_line.startswith(f"provider={provider} status=running, elapsed=")
        assert "history_rebuild_attempts=0" in progress_line
        assert progress_line.endswith(
            f"completed_questions={completed} ({completed}/60, {percentage}%)"
        )
        assert "completed_operations=" not in progress_line
        assert "question_progress=" not in progress_line
    assert "/180" not in result.stdout
    assert "/1)" not in result.stdout


def _progress_run_record(provider: str, state: str) -> dict[str, object]:
    return {
        "schema_name": "run_record",
        "schema_version": 1,
        "run_id": f"{provider}-run",
        "run_spec_hash": RESOLVED_PLAN_HASH,
        "state": state,
        "resume_disposition": "not_applicable",
        "started_at": "2026-09-06T01:10:34.090719+00:00",
        "ended_at": "2026-09-06T01:33:02.823301+00:00",
        "ingestion_occurrence_ids": [],
        "case_occurrence_ids": [],
    }


def _run_progress_fixture(
    tmp_path: Path,
    records: dict[str, list[object]],
    *,
    duplicate_provider: str = "",
) -> subprocess.CompletedProcess[str]:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    fake_bin = Path(env["PATH"].split(os.pathsep, 1)[0])
    _write_executable(fake_bin / "sleep", "/bin/sleep 0.05")
    env["OAMB_TEST_OAMB_RUN_DELAY"] = "1"
    env["OAMB_TEST_PROGRESS_CASES"] = "1"
    env["OAMB_TEST_PROGRESS_RUN_RECORDS"] = json.dumps(records)
    env["OAMB_TEST_PROGRESS_DUPLICATE_PROVIDER"] = duplicate_provider
    env["OAMB_TEST_OAMB_RUN_FAIL"] = "1"
    return subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )


@pytest.mark.parametrize("failed_state", ("aborted", "infrastructure_blocked"))
def test_run_progress_preserves_terminal_provider_state_while_peer_runs(
    tmp_path: Path, failed_state: str
) -> None:
    result = _run_progress_fixture(
        tmp_path,
        {
            "hindsight": [_progress_run_record("hindsight", failed_state)],
            "mem0": [_progress_run_record("mem0", "finalized")],
        },
    )

    assert result.returncode == 44, result.stdout + result.stderr
    assert f"provider=hindsight status={failed_state}, elapsed=1348s" in result.stdout
    assert "provider=mem0 status=execution-completed, elapsed=1348s" in result.stdout
    assert "provider=openviking status=running, elapsed=" in result.stdout
    assert "provider=hindsight status=running" not in result.stdout
    assert "provider=mem0 status=running" not in result.stdout
    assert "provider=openviking status=unavailable, elapsed=unavailable" in result.stdout
    assert "providers=hindsight,mem0,openviking status=failed" not in result.stderr
    assert "provider=mem0 status=failed" not in result.stdout + result.stderr
    assert "status=validating" not in result.stdout


@pytest.mark.parametrize("invalid_record", ("malformed", "duplicate", "identity", "timestamp"))
def test_run_progress_reports_invalid_terminal_record_as_unavailable(
    tmp_path: Path, invalid_record: str
) -> None:
    record = _progress_run_record("hindsight", "aborted")
    records: list[object] = [record]
    if invalid_record == "malformed":
        records = ["{"]
    elif invalid_record == "duplicate":
        records = [record, record]
    elif invalid_record == "identity":
        record["run_id"] = "another-run"
    else:
        record["ended_at"] = "not-a-timestamp"

    result = _run_progress_fixture(tmp_path, {"hindsight": records})

    assert result.returncode == 44, result.stdout + result.stderr
    assert "provider=hindsight status=unavailable, elapsed=unavailable" in result.stdout
    assert "provider=hindsight status=running" not in result.stdout
    assert "provider=hindsight status=aborted" not in result.stdout


def test_run_progress_reports_ambiguous_provider_root_as_unavailable(tmp_path: Path) -> None:
    result = _run_progress_fixture(tmp_path, {}, duplicate_provider="hindsight")

    assert result.returncode == 44, result.stdout + result.stderr
    assert "provider=hindsight status=unavailable, elapsed=unavailable" in result.stdout
    assert "provider=hindsight status=running" not in result.stdout


@pytest.mark.parametrize(
    ("arguments", "expected_question"),
    ((["--smoke_test"], True), (["--full_test"], False)),
)
def test_run_dispatches_all_provider_cells_together(
    tmp_path: Path,
    arguments: list[str],
    expected_question: bool,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    result = subprocess.run(
        [str(script), *arguments],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    run_calls = [
        line for line in trace.read_text(encoding="utf-8").splitlines() if "oamb run" in line
    ]
    assert len(run_calls) == 1
    if expected_question:
        assert "--question 72e3ee87" in run_calls[0]
        for cell in ("hindsight-lme60", "mem0-lme60", "openviking-lme60"):
            assert f"--cell {cell}" in run_calls[0]
    else:
        assert "--question" not in run_calls[0]
        assert "--bounded-capsule" not in run_calls[0]
        assert "--bounded-validation" not in run_calls[0]


def test_run_smoke_ignores_old_capsule_and_preserves_invalid_validation(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    smoke = root / "outputs" / "smoke-test" / "lme60-test"
    capsule = smoke / "capsules" / "smoke" / "hindsight-completed"
    capsule.mkdir(parents=True)
    result_map = smoke / "results" / "smoke.json"
    result_map.parent.mkdir(parents=True)
    result_map.write_text(
        json.dumps(
            {
                "schema_name": "live_run_result_map",
                "schema_version": 1,
                "resolved_plan_hash": RESOLVED_PLAN_HASH,
                "status": "completed",
                "cells": [
                    {
                        "cell_id": "hindsight-lme60",
                        "status": "completed",
                        "capsule_root": str(capsule),
                        "detail": None,
                    }
                ],
                "capsule_roots": {"hindsight-lme60": str(capsule)},
            }
        ),
        encoding="utf-8",
    )
    invalid_validation = smoke / "validations" / "smoke-hindsight.json"
    invalid_validation.parent.mkdir(parents=True)
    invalid_validation.write_text('{"disposition":"invalid"}\n', encoding="utf-8")

    result = subprocess.run(
        [str(script), "--smoke_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8")
    assert "oamb run" in calls
    assert "--cell hindsight-lme60" in calls
    assert "--cell mem0-lme60" in calls
    assert "--cell openviking-lme60" in calls
    assert invalid_validation.read_text(encoding="utf-8") == '{"disposition":"invalid"}\n'
    assert not tuple(invalid_validation.parent.glob("smoke-hindsight-retry-*.json"))


def test_run_smoke_starts_fresh_on_every_invocation(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    first = subprocess.run(
        [str(script), "--smoke_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    result_map = root / "outputs" / "smoke-test" / "lme60-test" / "results" / "smoke.json"
    failed = json.loads(result_map.read_bytes())
    failed["status"] = "failed"
    result_map.write_text(json.dumps(failed) + "\n", encoding="utf-8")

    for _ in range(2):
        retried = subprocess.run(
            [str(script), "--smoke_test"],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=20,
        )
        assert retried.returncode == 0, retried.stdout + retried.stderr

    smoke_run_calls = [
        line
        for line in trace.read_text(encoding="utf-8").splitlines()
        if "oamb run" in line and "--question 72e3ee87" in line
    ]
    assert len(smoke_run_calls) == 3
    assert json.loads(result_map.read_bytes())["status"] == "failed"
    assert not tuple(result_map.parent.glob("smoke-retry-*.json"))


def test_run_full_test_runs_directly_and_validates_sixty_case_report(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    result = subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8")
    assert "--bounded-capsule" not in calls
    assert "--bounded-validation" not in calls
    assert calls.count("oamb capsule validate") == 3
    assert (
        "run: providers=hindsight,mem0,openviking status=starting (full, 60 questions each)"
        in result.stdout
    )
    comparison_call = next(line for line in calls.splitlines() if "oamb compare" in line)
    assert "--diagnostic" not in comparison_call
    report = json.loads(
        (root / "outputs" / "full-test" / "lme60-test" / "comparison" / "report.json").read_bytes()
    )
    assert report["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 60,
        "provider_specific_result_count": 180,
    }


def test_run_full_test_ignores_completed_old_capsules_and_preserves_validation(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    first = subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    full_validation = (
        root / "outputs" / "full-test" / "lme60-test" / "validations" / "full-hindsight.json"
    )
    full_validation.write_text('{"disposition":"invalid"}\n', encoding="utf-8")

    second = subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert second.returncode == 0, second.stdout + second.stderr
    calls = trace.read_text(encoding="utf-8")
    full_run_calls = [
        line for line in calls.splitlines() if "oamb run" in line and "--cell" not in line
    ]
    assert len(full_run_calls) == 2
    assert full_validation.read_text(encoding="utf-8") == '{"disposition":"invalid"}\n'
    assert not tuple(full_validation.parent.glob("full-hindsight-retry-*.json"))


def test_run_full_test_starts_fresh_instead_of_reusing_failed_result_map(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    first = subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    full_result = root / "outputs" / "full-test" / "lme60-test" / "results" / "full.json"
    failed = json.loads(full_result.read_bytes())
    failed["status"] = "failed"
    full_result.write_text(json.dumps(failed) + "\n", encoding="utf-8")

    second = subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert second.returncode == 0, second.stdout + second.stderr
    third = subprocess.run(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert third.returncode == 0, third.stdout + third.stderr
    full_run_calls = [
        line
        for line in trace.read_text(encoding="utf-8").splitlines()
        if "oamb run" in line and "--cell" not in line
    ]
    assert len(full_run_calls) == 3
    assert json.loads(full_result.read_bytes())["status"] == "failed"
    assert not tuple(full_result.parent.glob("full-retry-*.json"))


def _write_interrupted_full_result_map(root: Path, *, suffix: str = "") -> tuple[Path, ...]:
    full = root / "outputs" / "full-test" / "lme60-test"
    resume_pointer = root / "outputs" / "tmp" / "precheck" / "lme60-test" / "full-test-current"
    resume_pointer.write_text("lme60-test\n", encoding="utf-8")
    cells = ("hindsight-lme60", "mem0-lme60", "openviking-lme60")
    capsule_roots = tuple(full / "capsules" / "full" / f"{cell}-part{suffix}" for cell in cells)
    for capsule_root in capsule_roots:
        capsule_root.mkdir(parents=True)
        (capsule_root / "preserved.txt").write_text("immutable source\n", encoding="utf-8")
    result_path = full / "results" / f"full{suffix}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(
            {
                "schema_name": "live_run_result_map",
                "schema_version": 1,
                "resolved_plan_hash": RESOLVED_PLAN_HASH,
                "status": "failed",
                "cells": [
                    {
                        "cell_id": cell,
                        "status": "failed",
                        "capsule_root": str(capsule_root),
                        "detail": "interrupted",
                    }
                    for cell, capsule_root in zip(cells, capsule_roots, strict=True)
                ],
                "capsule_roots": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return capsule_roots


@pytest.mark.parametrize("arguments", (["--resume"], ["--smoke_test", "--resume"]))
def test_run_rejects_resume_without_full_before_dispatch(
    tmp_path: Path,
    arguments: list[str],
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)

    result = subprocess.run(
        [str(script), *arguments],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert "--resume requires --full_test" in result.stderr
    assert not trace.exists() or "oamb run" not in trace.read_text(encoding="utf-8")


def test_run_full_resume_recovers_providers_concurrently_and_composes_report(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    source_roots = _write_interrupted_full_result_map(root)
    barrier = tmp_path / "recovery-barrier"
    env["OAMB_TEST_RECOVERY_BARRIER"] = str(barrier)

    result = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    analysis_calls = [line for line in calls if "--recovery-analysis-output" in line]
    recovery_calls = [
        line
        for line in calls
        if "oamb run" in line
        and "--recover-from" in line
        and "--recovery-analysis-output" not in line
    ]
    assert len(analysis_calls) == 3
    assert len(recovery_calls) == 3
    assert all("--bounded-capsule" not in line for line in calls)
    assert all("--bounded-validation" not in line for line in calls)
    assert all(
        sum(f"--recover-from {root}" in line for root in source_roots) == 1
        for line in recovery_calls
    )
    assert len([line for line in calls if "oamb capsule compose" in line]) == 3
    assert len(tuple(barrier.iterdir())) == 3
    assert (
        "reused=30, remaining=30, running=unavailable, completed=0, "
        "failed=unavailable, quarantined=1" in result.stdout
    )
    assert "running=30" not in result.stdout
    assert "question_progress=100% (60/60)" in result.stdout
    report = root / "outputs" / "full-test" / "lme60-test" / "comparison" / "report.json"
    assert json.loads(report.read_bytes())["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 60,
        "provider_specific_result_count": 180,
    }
    state = json.loads(
        (
            root / "outputs" / "full-test" / "lme60-test" / "results" / "full-resume-state.json"
        ).read_bytes()
    )
    assert state["attempt"] == 1
    assert [len(cell["parts"]) for cell in state["cells"]] == [2, 2, 2]
    assert all(cell["final_capsule_root"] for cell in state["cells"])


def test_run_full_resume_obeys_two_provider_cap_while_preserving_overlap(tmp_path: Path) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    plan_path = root / "outputs" / "tmp" / "precheck" / "lme60-test" / "plan" / "resolved-plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["execution"]["max_parallel_providers_per_dataset"] = 2
    plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    active = tmp_path / "active-recoveries"
    env["OAMB_TEST_RECOVERY_ACTIVE_DIR"] = str(active)
    env["OAMB_TEST_RECOVERY_MAX_ACTIVE"] = "2"
    env["OAMB_TEST_RECOVERY_PAIR_BARRIER"] = "1"

    result = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "active-recoveries.pair-ready").is_file()
    assert not any(active.iterdir())


def test_run_full_resume_rejects_a_concurrent_state_owner_before_analysis(tmp_path: Path) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    lock = root / "outputs" / "full-test" / "lme60-test" / "results" / "full-resume.lock"
    lock.mkdir()

    result = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert "another full-test resume owns the local state" in result.stderr
    assert not trace.exists() or "--recovery-analysis-output" not in trace.read_text(
        encoding="utf-8"
    )


def test_run_full_resume_stops_admission_when_signal_arrives_between_launches(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    signal_sentinel = tmp_path / "admission-signal-sent"
    bash_env = tmp_path / "signal-during-admission.bash"
    bash_env.write_text(
        """
OAMB_TEST_TOP_PID=$$
set -T
oamb_test_signal_during_admission() {
  if [[ "${BASH_SUBSHELL:-0}" == "0" &&
        "$$" == "$OAMB_TEST_TOP_PID" &&
        "${index:-}" == "1" &&
        "${pending_statuses[0]:-}" == "running" &&
        ! -e "$OAMB_TEST_ADMISSION_SIGNAL_SENTINEL" ]]; then
    trap - DEBUG
    : > "$OAMB_TEST_ADMISSION_SIGNAL_SENTINEL"
    kill -TERM "$$"
  fi
}
trap oamb_test_signal_during_admission DEBUG
""",
        encoding="utf-8",
    )
    env["BASH_ENV"] = str(bash_env)
    env["OAMB_TEST_ADMISSION_SIGNAL_SENTINEL"] = str(signal_sentinel)

    result = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert signal_sentinel.is_file(), result.stdout + result.stderr
    recovery_calls = [
        line
        for line in trace.read_text(encoding="utf-8").splitlines()
        if "oamb run" in line
        and "--recover-from" in line
        and "--recovery-analysis-output" not in line
    ]
    assert not any("--cell mem0-lme60" in line for line in recovery_calls)
    assert not any("--cell openviking-lme60" in line for line in recovery_calls)


def test_run_full_resume_stops_new_batch_after_failure_and_reports_each_provider(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    plan_path = root / "outputs" / "tmp" / "precheck" / "lme60-test" / "plan" / "resolved-plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["execution"]["max_parallel_providers_per_dataset"] = 2
    plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
    active = tmp_path / "active-recoveries"
    env["OAMB_TEST_RECOVERY_ACTIVE_DIR"] = str(active)
    env["OAMB_TEST_RECOVERY_MAX_ACTIVE"] = "2"
    env["OAMB_TEST_RECOVERY_PAIR_BARRIER"] = "1"
    env["OAMB_TEST_RECOVERY_FAIL_CELL"] = "hindsight-lme60"

    result = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    recovery_calls = [
        line
        for line in trace.read_text(encoding="utf-8").splitlines()
        if "oamb run" in line
        and "--recover-from" in line
        and "--recovery-analysis-output" not in line
    ]
    assert len(recovery_calls) == 2
    assert not any("--cell openviking-lme60" in line for line in recovery_calls)
    state = json.loads(
        (
            root / "outputs" / "full-test" / "lme60-test" / "results" / "full-resume-state.json"
        ).read_bytes()
    )
    assert [len(cell["parts"]) for cell in state["cells"]] == [2, 2, 1]
    assert "provider=hindsight status=failed" in result.stderr
    assert "provider=mem0 status=completed" in result.stderr
    assert "provider=openviking status=not_started" in result.stderr


def test_run_full_resume_records_valid_siblings_when_one_new_part_is_invalid(
    tmp_path: Path,
) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    env["OAMB_TEST_INVALID_RECOVERY_VALIDATION_CELL"] = "hindsight-lme60"

    result = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    state = json.loads(
        (
            root / "outputs" / "full-test" / "lme60-test" / "results" / "full-resume-state.json"
        ).read_bytes()
    )
    assert [len(cell["parts"]) for cell in state["cells"]] == [1, 2, 2]
    assert "provider=hindsight status=failed" in result.stderr
    assert "provider=mem0 status=completed" in result.stderr
    assert "provider=openviking status=completed" in result.stderr


def test_run_full_resume_stops_after_composition_when_operator_signals(tmp_path: Path) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    ready = tmp_path / "compose-ready"
    release = tmp_path / "compose-release"
    env["OAMB_TEST_COMPOSE_SIGNAL_CELL"] = "hindsight-lme60"
    env["OAMB_TEST_COMPOSE_SIGNAL_READY"] = str(ready)
    env["OAMB_TEST_COMPOSE_SIGNAL_RELEASE"] = str(release)
    process = subprocess.Popen(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if (ready / "hindsight-lme60").is_file():
            break
        time.sleep(0.02)
    else:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"composition did not reach signal barrier\n{stdout}\n{stderr}")

    process.send_signal(signal.SIGTERM)
    release.touch()
    stdout, stderr = process.communicate(timeout=20)

    assert process.returncode != 0, stdout + stderr
    calls = trace.read_text(encoding="utf-8")
    assert "oamb compare" not in calls
    assert "run: PASS" not in stdout
    assert "operator stop preserved resume state" in stderr


def test_run_full_resume_drains_signal_and_records_parts_for_another_resume(tmp_path: Path) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    signal_ready = tmp_path / "signal-ready"
    env["OAMB_TEST_RECOVERY_SIGNAL_READY"] = str(signal_ready)
    process = subprocess.Popen(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if signal_ready.is_dir() and len(tuple(signal_ready.iterdir())) == 3:
            break
        time.sleep(0.02)
    else:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        pytest.fail(f"recovery workers did not reach signal barrier\n{stdout}\n{stderr}")

    process.send_signal(signal.SIGTERM)
    stdout, stderr = process.communicate(timeout=20)

    assert process.returncode != 0, stdout + stderr
    state_path = (
        root / "outputs" / "full-test" / "lme60-test" / "results" / "full-resume-state.json"
    )
    interrupted_state = json.loads(state_path.read_bytes())
    assert [len(cell["parts"]) for cell in interrupted_state["cells"]] == [2, 2, 2]
    assert "preserved parts will be reused by the next --resume" in stderr

    env.pop("OAMB_TEST_RECOVERY_SIGNAL_READY")
    resumed = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    final_state = json.loads(state_path.read_bytes())
    assert final_state["attempt"] == 2
    assert [len(cell["parts"]) for cell in final_state["cells"]] == [3, 3, 3]
    assert all(cell["final_capsule_root"] for cell in final_state["cells"])


def test_run_full_resume_preserves_second_interruption_for_next_resume(tmp_path: Path) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    source_roots = _write_interrupted_full_result_map(root)
    source_bytes = tuple((path / "preserved.txt").read_bytes() for path in source_roots)
    env["OAMB_TEST_RECOVERY_FAIL"] = "1"

    interrupted = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert interrupted.returncode != 0
    state_path = (
        root / "outputs" / "full-test" / "lme60-test" / "results" / "full-resume-state.json"
    )
    interrupted_state = json.loads(state_path.read_bytes())
    assert [len(cell["parts"]) for cell in interrupted_state["cells"]] == [2, 2, 2]
    assert all(cell["final_capsule_root"] is None for cell in interrupted_state["cells"])
    assert tuple((path / "preserved.txt").read_bytes() for path in source_roots) == source_bytes

    env.pop("OAMB_TEST_RECOVERY_FAIL")
    resumed = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    second_analysis_calls = [
        line
        for line in calls
        if "--recovery-analysis-output" in line and "resume-analysis-2" in line
    ]
    assert len(second_analysis_calls) == 3
    assert all(line.count("--recover-from") == 2 for line in second_analysis_calls)
    final_state = json.loads(state_path.read_bytes())
    assert final_state["attempt"] == 2
    assert [len(cell["parts"]) for cell in final_state["cells"]] == [3, 3, 3]
    assert tuple((path / "preserved.txt").read_bytes() for path in source_roots) == source_bytes


def test_run_full_resume_fails_global_analysis_barrier_before_recovery_dispatch(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    env["OAMB_TEST_RECOVERY_ANALYSIS_FAIL_CELL"] = "mem0-lme60"

    result = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert len([line for line in calls if "--recovery-analysis-output" in line]) == 3
    assert not any(
        "oamb run" in line and "--recover-from" in line and "--recovery-analysis-output" not in line
        for line in calls
    )


def test_run_full_resume_rejects_ambiguous_initial_source_selection(tmp_path: Path) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_interrupted_full_result_map(root)
    _write_interrupted_full_result_map(root, suffix="-retry-ambiguous")

    result = subprocess.run(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode != 0
    assert "ambiguous full-run source selection" in result.stderr
    assert not trace.exists() or "--recover-from" not in trace.read_text(encoding="utf-8")
