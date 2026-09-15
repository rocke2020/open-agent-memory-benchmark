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

from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import (
    build_resolved_plan,
    load_resolved_plan_for_run,
    resolved_plan_bytes,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from tests.benchmark_configuration import MODEL_ENVIRONMENT
from tests.unit.test_question_results import _judged_result

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
    source_text = source.read_text(encoding="utf-8")
    for helper_source in (
        HOST_EMBEDDING_SCRIPT,
        PLAN_ENVIRONMENT_SCRIPT,
        PROVIDER_ENVIRONMENT_SCRIPT,
    ):
        if helper_source.name not in source_text:
            continue
        helper = root / "provider-services" / "lib" / helper_source.name
        helper.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(helper_source, helper)
    return destination


def test_root_env_template_owns_one_generic_llm_connection() -> None:
    assignment_names = [
        line.split("=", 1)[0]
        for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert assignment_names.count("LLM_URL_TYPE") == 1
    assert assignment_names.count("LLM_LIGHT_MODEL") == 1
    assert assignment_names.count("LLM_DEEP_MODEL") == 1
    assert assignment_names.count("LLM_BASE_URL") == 1
    assert assignment_names.count("LLM_API_KEY") == 1
    assert "DEEPSEEK_BASE_URL" not in assignment_names
    assert "DEEPSEEK_API_KEY" not in assignment_names
    assert set(assignment_names).isdisjoint(PROVIDER_MODEL_CONNECTION_ALIASES)
    assert "LLM_URL_TYPE=openai_chat" in (REPOSITORY_ROOT / ".env.example").read_text(
        encoding="utf-8"
    )
    example = (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8")
    example_values = dict(
        line.split("=", 1) for line in example.splitlines() if line and not line.startswith("#")
    )
    assert example_values["LLM_LIGHT_MODEL"] == "deepseek-flash"
    assert example_values["LLM_DEEP_MODEL"] == "deepseek-flash"
    assert example_values["OAMB_EMBEDDING_BASE_URL"] == "change-me"
    assert example_values["OAMB_EMBEDDING_API_KEY"] == ""


@pytest.mark.parametrize("model_variable", ("LLM_LIGHT_MODEL", "LLM_DEEP_MODEL"))
def test_precheck_rejects_an_empty_model_name(
    tmp_path: Path,
    model_variable: str,
) -> None:
    root, env, _trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    configured = env_path.read_text(encoding="utf-8")
    configured = configured.replace(
        f"{model_variable}=deepseek-flash",
        f"{model_variable}=",
        1,
    )
    env_path.write_text(configured, encoding="utf-8")

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
    assert f"configure {model_variable} in .env" in result.stderr


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


def test_provider_shell_defaults_missing_embedding_api_key_for_containers(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:18000/v1\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; '
            'resolve_container_embedding_base "$(read_env_value "$2" OAMB_EMBEDDING_BASE_URL)"; '
            'effective_embedding_api_key "$2"',
            "sh",
            str(PROVIDER_ENVIRONMENT_SCRIPT),
            str(env_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.splitlines() == [
        "http://host.docker.internal:18000/v1",
        "oamb-no-auth",
    ]


def test_compose_uses_plan_endpoint_and_translates_only_at_container_boundary(
    tmp_path: Path,
) -> None:
    env_file = tmp_path / ".env"
    configured = (
        (REPOSITORY_ROOT / ".env.example")
        .read_text(encoding="utf-8")
        .replace("LLM_BASE_URL=change-me", "LLM_BASE_URL=https://models.example/v1")
        .replace("LLM_API_KEY=change-me", "LLM_API_KEY=test-key")
    )
    env_file.write_text(configured, encoding="utf-8")
    plan = build_resolved_plan(
        load_benchmark_configuration(
            REPOSITORY_ROOT / "configs" / "benchmark.yml",
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    plan_path = tmp_path / "resolved-plan.json"
    plan_path.write_bytes(resolved_plan_bytes(plan))
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _write_executable(
        fake_bin / "docker",
        'printf "%s|%s\\n" "$OAMB_EMBEDDING_BASE_URL" "$OAMB_EMBEDDING_API_KEY"',
    )

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; . "$2"; . "$3"; load_plan_model_environment "$4" "$6"; '
            'oamb_compose "$5" "$6" config',
            "sh",
            str(PROVIDER_ENVIRONMENT_SCRIPT),
            str(REPOSITORY_ROOT / "provider-services" / "lib" / "compose.sh"),
            str(PLAN_ENVIRONMENT_SCRIPT),
            str(plan_path),
            str(REPOSITORY_ROOT / "provider-services"),
            str(env_file),
        ],
        env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == "http://host.docker.internal:18000/v1|oamb-no-auth\n"


def test_plan_environment_uses_current_dotenv_generative_models(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_LIGHT_MODEL=runtime-light-model\nLLM_DEEP_MODEL=runtime-deep-model\n",
        encoding="utf-8",
    )
    plan = build_resolved_plan(
        load_benchmark_configuration(
            REPOSITORY_ROOT / "configs" / "benchmark.yml",
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    plan_path = tmp_path / "resolved-plan.json"
    plan_path.write_bytes(resolved_plan_bytes(plan))

    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; . "$2"; load_plan_model_environment "$3" "$4"; '
            'printf "%s|%s|%s|%s\\n" "$OAMB_HINDSIGHT_LLM_MODEL" '
            '"$OAMB_MEM0_LLM_MODEL" "$OAMB_OPENVIKING_VLM_MODEL" '
            '"$OAMB_EMBEDDING_MODEL"',
            "sh",
            str(REPOSITORY_ROOT / "provider-services" / "lib" / "env.sh"),
            str(PLAN_ENVIRONMENT_SCRIPT),
            str(plan_path),
            str(env_file),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == (
        "runtime-light-model|runtime-light-model|runtime-light-model|qwen3-embedding:0.6b\n"
    )


@pytest.mark.parametrize(
    "url",
    (
        "http://127.0.0.1.embedding.example/v1",
        "http://localhost.embedding.example/v1",
    ),
)
def test_provider_shell_does_not_rewrite_host_name_lookalikes(
    tmp_path: Path,
    url: str,
) -> None:
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; resolve_container_embedding_base "$2"',
            "sh",
            str(PROVIDER_ENVIRONMENT_SCRIPT),
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout == f"{url}\n"


@pytest.mark.parametrize(
    "url",
    (
        "https://127.0.0.1:18000/v1",
        "https://localhost:18000/v1",
        "http://[::1]:18000/v1",
    ),
)
def test_provider_shell_rejects_unsupported_loopback_embedding_urls(
    tmp_path: Path,
    url: str,
) -> None:
    result = subprocess.run(
        [
            "/bin/sh",
            "-c",
            '. "$1"; resolve_container_embedding_base "$2"',
            "sh",
            str(PROVIDER_ENVIRONMENT_SCRIPT),
            url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert result.stdout == ""


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
        plan=build_resolved_plan(
            load_benchmark_configuration(
                REPOSITORY_ROOT / "configs" / "benchmark.yml",
                model_environment=MODEL_ENVIRONMENT,
            )
        ),
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


@pytest.mark.parametrize(
    "embedding_key_assignment",
    ("", "OAMB_EMBEDDING_API_KEY=\n"),
    ids=("missing", "empty"),
)
def test_live_environment_treats_missing_and_empty_embedding_keys_as_keyless(
    tmp_path: Path,
    embedding_key_assignment: str,
) -> None:
    from oamb.live import load_live_environment

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_URL_TYPE=openai_chat\n"
        "LLM_BASE_URL=https://models.example/v1\n"
        "LLM_API_KEY=test-model-key\n"
        "OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:18000/v1\n"
        f"{embedding_key_assignment}",
        encoding="utf-8",
    )

    environment = load_live_environment(
        plan=build_resolved_plan(
            load_benchmark_configuration(
                REPOSITORY_ROOT / "configs" / "benchmark.yml",
                model_environment=MODEL_ENVIRONMENT,
            )
        ),
        provider_env_path=env_file,
        model_env_path=env_file,
        provider_runtime_directory=tmp_path / "runtime",
        base_environment={"OAMB_EMBEDDING_API_KEY": "inherited-process-key"},
    )

    assert environment["OAMB_EMBEDDING_API_KEY"] == ""


def test_live_environment_uses_frozen_managed_local_endpoint_without_dotenv_writeback(
    tmp_path: Path,
) -> None:
    from oamb.config.benchmark import load_benchmark_configuration
    from oamb.config.doctor import build_resolved_plan
    from oamb.live import load_live_environment

    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_URL_TYPE=openai_chat\n"
        "LLM_BASE_URL=https://models.example/v1\n"
        "LLM_API_KEY=test-model-key\n"
        "OAMB_EMBEDDING_BASE_URL=change-me\n",
        encoding="utf-8",
    )
    plan = build_resolved_plan(
        load_benchmark_configuration(
            REPOSITORY_ROOT / "configs" / "benchmark.yml",
            model_environment=MODEL_ENVIRONMENT,
        )
    )

    environment = load_live_environment(
        plan=plan,
        provider_env_path=env_file,
        model_env_path=env_file,
        provider_runtime_directory=tmp_path / "runtime",
        base_environment={},
    )

    assert environment["OAMB_EMBEDDING_BASE_URL"] == "http://127.0.0.1:18000/v1"


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
            plan=build_resolved_plan(
                load_benchmark_configuration(
                    REPOSITORY_ROOT / "configs" / "benchmark.yml",
                    model_environment=MODEL_ENVIRONMENT,
                )
            ),
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
    _write_executable(
        fake_bin / "docker",
        r"""
printf 'docker %s\n' "$*" >> "$OAMB_TEST_TRACE"
case "$*" in
  network\ rm\ *) exit 0 ;;
esac
printf '%s\n' 172.17.0.1
""".strip(),
    )
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
    embedding_url="http://127.0.0.1:18000/v1"
    embedding_ownership="embedding_local_fallback"
    previous=""
    for argument in "$@"; do
      if [ "$previous" = "--embedding-api-url" ]; then
        embedding_url=$argument
        embedding_ownership="external"
      fi
      previous=$argument
    done
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--output" ]; then
        mkdir -p "$2"
        python3 - "$2/resolved-plan.json" "$embedding_url" "$embedding_ownership" <<'PY'
import json
import sys

target, endpoint, ownership = sys.argv[1:]
document = {"schema_name":"resolved_comparison_plan","embedding_endpoint":{"effective_endpoint":endpoint,"ownership":ownership},"execution":{"extraction_max_retries":10},"model_roles":[{"role_id":"hindsight_extraction","model":"LLM_LIGHT_MODEL","thinking_effort":"low"},{"role_id":"mem0_extraction","model":"LLM_LIGHT_MODEL","thinking_effort":"high"},{"role_id":"openviking_semantic_understanding","model":"LLM_LIGHT_MODEL","thinking_effort":"max"},{"role_id":"embedding","model":"plan-embedding","thinking_effort":"not_applicable"}]}
with open(target, "w", encoding="utf-8") as output:
    json.dump(document, output)
PY
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
if [ "${1:-}" = "prepare-new-run" ]; then
  new_project=$2
  preserved_runtime=$3
  provider_root=$(cd "$(dirname "$0")/.." && pwd)
  runtime="$provider_root/.runtime"
  mkdir -p "$preserved_runtime"
  shopt -s dotglob nullglob
  for runtime_entry in "$runtime"/*; do
    mv "$runtime_entry" "$preserved_runtime/"
  done
  OAMB_ENV_FILE="$provider_root/../.env" OAMB_NEW_PROJECT="$new_project" python3 - <<'PY'
import os
from pathlib import Path

path = Path(os.environ["OAMB_ENV_FILE"])
lines = path.read_text(encoding="utf-8").splitlines()
lines = [
    f"OAMB_PROVIDER_PROJECT={os.environ['OAMB_NEW_PROJECT']}"
    if line.startswith("OAMB_PROVIDER_PROJECT=")
    else line
    for line in lines
]
path.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
PY
fi
""".strip(),
    )
    resolver = root / "provider-services" / "lib" / "host_embedding.sh"
    resolver.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HOST_EMBEDDING_SCRIPT, resolver)
    shutil.copy2(REPOSITORY_ROOT / ".env.example", root / ".env.example")

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
    assert "OAMB_EMBEDDING_BASE_URL=change-me" in (root / ".env").read_text(encoding="utf-8")


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
    root_env = (root / ".env").read_text(encoding="utf-8")
    assert "OAMB_EMBEDDING_BASE_URL=change-me" in root_env
    assert "OAMB_EMBEDDING_BASE_URL=http://host.docker.internal:18000/v1" not in root_env
    assert "OAMB_EMBEDDING_API_KEY=" in root_env
    assert "uv sync --locked --all-groups" in calls
    assert "provider-services verify --services" in calls
    assert "provider-services verify --model-readiness" in calls
    if system_name == "Linux":
        assert "http://127.0.0.1:18000/v1/embeddings" in calls
        assert "http://172.17.0.1:18000/v1/embeddings" in calls
    state = json.loads((root / "outputs" / "tmp" / "quick-start-current.json").read_bytes())
    assert Path(state["resolved_plan"]).is_file()
    assert state["question_id"] == "72e3ee87"
    assert state["embedding_ownership"] == "embedding_local_fallback"


def test_precheck_managed_local_embedding_leaves_prepared_dotenv_byte_identical(
    tmp_path: Path,
) -> None:
    root, env, _trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    replacements = {
        "OAMB_PROVIDER_PROJECT": "oamb-providers-byte-identical",
        "OAMB_MEM0_SOURCE_CHECKOUT": str(root / "outputs" / "tmp" / "provider-source" / "mem0"),
        "OAMB_MEM0_ADMIN_API_KEY": "test-admin-key",
        "OAMB_MEM0_JWT_SECRET": "test-jwt-secret",
        "OAMB_MEM0_POSTGRES_PASSWORD": "test-postgres-password",
        "OAMB_MEM0_INSPECTOR_API_KEY": "test-inspector-key",
        "OAMB_OPENVIKING_ROOT_API_KEY": "test-openviking-key",
    }
    prepared_lines = []
    for line in env_path.read_text(encoding="utf-8").splitlines():
        name = line.split("=", 1)[0]
        prepared_lines.append(f"{name}={replacements[name]}" if name in replacements else line)
    env_path.write_text("\n".join(prepared_lines) + "\n", encoding="utf-8")
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

    assert result.returncode == 0, result.stdout + result.stderr
    assert env_path.read_bytes() == original


def test_precheck_replaces_existing_provider_project_for_new_run(
    tmp_path: Path,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    old_project = "oamb-providers-old-project"
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "OAMB_PROVIDER_PROJECT=change-me",
            f"OAMB_PROVIDER_PROJECT={old_project}",
        ),
        encoding="utf-8",
    )
    provider_root = root / "provider-services"
    runtime = provider_root / ".runtime"
    runtime.mkdir()
    (runtime / "provider-project.attestation").write_text(
        f"project={old_project}\ncompose_sha256={'a' * 64}\nversions_sha256={'b' * 64}\n",
        encoding="utf-8",
    )
    (runtime / "preserved-evidence.json").write_text("{}\n", encoding="utf-8")

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
    state = json.loads((root / "outputs" / "tmp" / "quick-start-current.json").read_bytes())
    new_project = next(
        line.split("=", 1)[1]
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("OAMB_PROVIDER_PROJECT=")
    )
    assert new_project == f"oamb-providers-{state['run_label']}"
    assert new_project != old_project
    preserved_runtime = provider_root / f".runtime-preserved-{state['run_label']}"
    assert (preserved_runtime / "preserved-evidence.json").read_text(encoding="utf-8") == "{}\n"
    calls = trace.read_text(encoding="utf-8").splitlines()
    prepare_call = next(
        line for line in calls if line.startswith("provider-services prepare-new-run ")
    )
    assert calls.index(prepare_call) < calls.index("provider-services doctor")
    assert f"docker network rm {old_project}_default" in calls


def test_precheck_does_not_replace_an_active_provider_project(tmp_path: Path) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    first = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    old_project = next(
        line.split("=", 1)[1]
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("OAMB_PROVIDER_PROJECT=")
    )
    original_env = env_path.read_bytes()
    provider_root = root / "provider-services"
    runtime = provider_root / ".runtime"
    runtime.mkdir()
    attestation = runtime / "provider-project.attestation"
    attestation.write_text(
        f"project={old_project}\ncompose_sha256={'a' * 64}\nversions_sha256={'b' * 64}\n",
        encoding="utf-8",
    )
    active_operation = runtime / "active-operation"
    active_operation.write_text('{"kind":"benchmark_run"}\n', encoding="utf-8")
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    original_state = state_path.read_bytes()
    _write_executable(
        provider_root / "bin" / "provider-services",
        """
printf 'provider-services %s\n' "$*" >> "$OAMB_TEST_TRACE"
if [ "${1:-}" = "prepare-new-run" ] && [ -e "$OAMB_TEST_ACTIVE_OPERATION" ]; then
  printf 'provider-services: active OAMB provider operation exists; refusing provider lifecycle mutation\n' >&2
  exit 1
fi
""".strip(),
    )
    env["OAMB_TEST_ACTIVE_OPERATION"] = str(active_operation)
    trace.write_text("", encoding="utf-8")

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
    assert "active OAMB provider operation" in result.stderr
    assert env_path.read_bytes() == original_env
    assert state_path.read_bytes() == original_state
    assert attestation.is_file()
    assert active_operation.is_file()
    assert not tuple(provider_root.glob(".runtime-preserved-*"))
    calls = trace.read_text(encoding="utf-8").splitlines()
    assert any(line.startswith("provider-services prepare-new-run ") for line in calls)
    assert not any(line.startswith("docker network rm ") for line in calls)


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


def test_precheck_configured_embedding_service_pair_skips_local_server(tmp_path: Path) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        .replace(
            "OAMB_EMBEDDING_BASE_URL=change-me",
            "OAMB_EMBEDDING_BASE_URL=https://embedding.example/v1",
        )
        .replace(
            "OAMB_EMBEDDING_API_KEY=change-me",
            "OAMB_EMBEDDING_API_KEY=paid-embedding-key",
        )
        .replace(
            "OAMB_EMBEDDING_API_KEY=\n",
            "OAMB_EMBEDDING_API_KEY=paid-embedding-key\n",
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

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "embedding: local startup skipped; configured API will be verified directly"
        in result.stdout
    )
    calls = trace.read_text(encoding="utf-8")
    assert "embedding start_" not in calls
    root_env = (root / ".env").read_text(encoding="utf-8")
    assert "OAMB_EMBEDDING_BASE_URL=https://embedding.example/v1" in root_env
    assert "OAMB_EMBEDDING_API_KEY=paid-embedding-key" in root_env
    assert not (root / "provider-services" / ".env").exists()
    state = json.loads((root / "outputs" / "tmp" / "quick-start-current.json").read_bytes())
    plan = json.loads(Path(state["resolved_plan"]).read_bytes())
    assert state["embedding_ownership"] == "external"
    assert plan["embedding_endpoint"] == {
        "effective_endpoint": "https://embedding.example/v1",
        "ownership": "external",
    }


def test_precheck_embedding_url_override_is_external_without_dotenv_writeback(
    tmp_path: Path,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)

    result = subprocess.run(
        [str(script), "--embedding-api-url", "https://override.example/v1"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "embedding start_" not in trace.read_text(encoding="utf-8")
    assert "OAMB_EMBEDDING_BASE_URL=change-me" in (root / ".env").read_text(encoding="utf-8")
    state = json.loads((root / "outputs" / "tmp" / "quick-start-current.json").read_bytes())
    plan = json.loads(Path(state["resolved_plan"]).read_bytes())
    assert state["embedding_ownership"] == "external"
    assert plan["embedding_endpoint"] == {
        "effective_endpoint": "https://override.example/v1",
        "ownership": "external",
    }


def test_precheck_keyless_embedding_service_skips_local_server(tmp_path: Path) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        .replace(
            "OAMB_EMBEDDING_BASE_URL=change-me",
            "OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:18000/v1",
        )
        .replace("OAMB_EMBEDDING_API_KEY=\n", ""),
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

    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        "embedding: local startup skipped; configured API will be verified directly"
        in result.stdout
    )
    assert "embedding start_" not in trace.read_text(encoding="utf-8")
    assert "OAMB_EMBEDDING_API_KEY" not in env_path.read_text(encoding="utf-8")
    state = json.loads((root / "outputs" / "tmp" / "quick-start-current.json").read_bytes())
    assert state["embedding_ownership"] == "external"


def test_precheck_explicit_fallback_address_does_not_take_startup_ownership(
    tmp_path: Path,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "OAMB_EMBEDDING_BASE_URL=change-me",
            "OAMB_EMBEDDING_BASE_URL=http://host.docker.internal:18000/v1",
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

    assert result.returncode == 0, result.stdout + result.stderr
    assert "embedding start_" not in trace.read_text(encoding="utf-8")
    state = json.loads((root / "outputs" / "tmp" / "quick-start-current.json").read_bytes())
    assert state["embedding_ownership"] == "external"


def test_precheck_explicit_fallback_address_ignores_stale_local_ownership(
    tmp_path: Path,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "OAMB_EMBEDDING_BASE_URL=change-me",
            "OAMB_EMBEDDING_BASE_URL=http://host.docker.internal:18000/v1",
        ),
        encoding="utf-8",
    )
    ownership_path = root / "outputs" / "tmp" / "embedding-ownership"
    ownership_path.write_text("local-fallback\n", encoding="utf-8")

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
    assert (
        "embedding: local startup skipped; configured API will be verified directly"
        in result.stdout
    )
    assert "embedding start_" not in trace.read_text(encoding="utf-8")
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    assert json.loads(state_path.read_bytes())["embedding_ownership"] == "external"


def test_precheck_does_not_publish_state_when_provider_preparation_fails(
    tmp_path: Path,
) -> None:
    root, env, _trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    _write_executable(
        root / "provider-services" / "bin" / "provider-services",
        "exit 73",
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

    assert result.returncode == 73
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    assert not state_path.exists()
    assert "OAMB_EMBEDDING_BASE_URL=change-me" in (root / ".env").read_text(encoding="utf-8")


def test_failed_reprecheck_preserves_previous_quick_start_selection(tmp_path: Path) -> None:
    root, env, _trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    first = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    assert first.returncode == 0, first.stdout + first.stderr
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    original_state = state_path.read_bytes()
    _write_executable(
        root / "provider-services" / "bin" / "provider-services",
        "exit 73",
    )

    second = subprocess.run(
        [str(script)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert second.returncode == 73
    assert state_path.read_bytes() == original_state


def test_precheck_rejects_embedding_api_key_without_service_url(
    tmp_path: Path,
) -> None:
    root, env, trace = _quick_start_fixture(tmp_path, system_name="Darwin")
    script = _copy_quick_start_script(PRECHECK_SCRIPT, root)
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8")
        .replace("OAMB_EMBEDDING_API_KEY=change-me", "OAMB_EMBEDDING_API_KEY=orphaned-key")
        .replace("OAMB_EMBEDDING_API_KEY=\n", "OAMB_EMBEDDING_API_KEY=orphaned-key\n"),
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
    assert "OAMB_EMBEDDING_API_KEY requires OAMB_EMBEDDING_BASE_URL" in result.stderr
    assert "embedding start_" not in trace.read_text(encoding="utf-8")


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
        "provider-config deepseek-flash|deepseek-flash|deepseek-flash|plan-embedding|low|high|max|openai|openai"
        in (trace.read_text(encoding="utf-8"))
    )
    assert any(
        line.startswith("curl ") and "plan-embedding" in line
        for line in trace.read_text(encoding="utf-8").splitlines()
    )
    assert [
        line.split("=", 1)[0]
        for line in root_env.splitlines()
        if line.startswith("OAMB_")
        and "=" in line
        and line.split("=", 1)[1].startswith("change-me")
    ] == ["OAMB_EMBEDDING_BASE_URL"]
    assert stat.S_IMODE(root_env_path.stat().st_mode) == 0o600
    assert not (root / "provider-services" / ".env").exists()


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
	if [[ " $* " == *" oamb run "* ]]; then
	  printf 'run-owner-pid %s\n' "${OAMB_RUN_SH_PROCESS_ID:-missing}" >> "$OAMB_TEST_TRACE"
	fi
	rmdir "$trace_lock"
	trace_lock_owned=false
	trap - EXIT TERM INT HUP
	if [[ " $* " == *" python - "*"longmemeval_s_cleaned.json"* ]] && \
	   [ "${OAMB_TEST_INPUT_ENCODING_FAIL:-}" = "1" ]; then
	  printf 'planted full-input encoding failure\n' >&2
	  exit 46
	fi
	if [ "$#" -eq 6 ] && [ "$3" = "python" ] && [ "$4" = "-" ]; then
	  case "$6" in
	    */results)
	      mkdir -p "$6"
	      for provider in hindsight mem0 openviking; do
	        printf '{}' > "$6/$provider.json"
	      done
	      exit 0
	      ;;
	  esac
	fi
	case " $* " in
	  *" oamb run "*)
	    cells=()
	    result_map=""
	    output_root=""
	    results_root=""
	    while [ "$#" -gt 0 ]; do
	      case "$1" in
	        --cell) cells+=("$2"); shift 2 ;;
	        --result-map) result_map=$2; shift 2 ;;
	        --output-root) output_root=$2; shift 2 ;;
	        --results-root) results_root=$2; shift 2 ;;
	        *) shift ;;
	      esac
	    done
	    if [ "${OAMB_TEST_RUN_SIGNAL_BLOCK:-0}" = "1" ]; then
	      printf '%s\n' "$$" > "$OAMB_TEST_RUN_SIGNAL_PID"
	      touch "$OAMB_TEST_RUN_SIGNAL_STARTED"
	      trap 'printf "INT\n" >> "$OAMB_TEST_RUN_SIGNALS"' INT
	      trap 'printf "TERM\n" >> "$OAMB_TEST_RUN_SIGNALS"; exit 143' TERM
	      trap 'printf "HUP\n" >> "$OAMB_TEST_RUN_SIGNALS"; exit 129' HUP
	      while :; do /bin/sleep 0.1 || true; done
	    fi
	    if [ -n "$results_root" ]; then
	      if [ -n "${OAMB_TEST_RESULT_UPDATES:-}" ]; then
	        python3 - "$output_root" "$results_root" <<'PY'
import json
import os
import sys
import time
from pathlib import Path

output_root, results_root = map(Path, sys.argv[1:])
source = output_root / "progress-openviking" / "source"
(source / "specs").mkdir(parents=True)
(source / "cases").mkdir()
(source / "specs" / "run-spec.json").write_text(
    json.dumps({"memory_system_id": "openviking", "run_id": "openviking-run"})
)
path = results_root / "openviking.json"
results = json.loads(path.read_text())
print("fixture-run-active", flush=True)
for index, result in enumerate(json.loads(os.environ["OAMB_TEST_RESULT_UPDATES"])):
    results[result["question_id"]] = result
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results))
    temporary.replace(path)
    (source / "cases" / f"{index}.json").write_text("{}")
    time.sleep(0.5)
print("fixture-run-finished", flush=True)
PY
	        exit 44
	      fi
	      if [ -z "$result_map" ] && \
	        [ "${OAMB_TEST_RESUME_FAIL:-0}" = "1" ]; then
	        printf 'planted simple resume failure\n' >&2
	        exit 44
	      fi
	      if [ -z "$result_map" ]; then
	        exit 0
	      fi
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
        if [ -n "$results_root" ]; then
          python3 - "$results_root/$progress_provider.json" "$provider_progress_cases" <<'PY'
import json
import os
import sys
from pathlib import Path

template = json.loads(os.environ["OAMB_TEST_QUESTION_RESULT"])
results = {f"q-{i}": dict(template, question_id=f"q-{i}") for i in range(int(sys.argv[2]))}
Path(sys.argv[1]).write_text(json.dumps(results))
PY
        fi
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
	  *" oamb capsule validate "*)
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
    if [ "${OAMB_TEST_COMPARE_BLOCK:-0}" = "1" ]; then
      printf '%s\n' "$$" > "$OAMB_TEST_COMPARE_PID"
      touch "$OAMB_TEST_COMPARE_STARTED"
      trap 'touch "$OAMB_TEST_COMPARE_STOPPED"; exit 143' TERM INT HUP
      while :; do
        if [ "${OAMB_TEST_COMPARE_EXIT_WITH_OWNER:-0}" = "1" ] && \
          ! kill -0 "$OAMB_RUN_SH_PROCESS_ID" 2>/dev/null; then
          exit 0
        fi
        /bin/sleep 0.01
      done
    fi
    [ ! -e "$output_root" ] || exit 92
    mkdir -p "$output_root"
    if [ "$diagnostic" = true ]; then cases=1; results=3; else cases=60; results=180; fi
    printf '{"diagnostic":%s,"coverage":{"cell_count":3,"unique_case_count":%s,"provider_specific_result_count":%s}}\n' \
      "$diagnostic" "$cases" "$results" > "$output_root/report.json"
    printf '%s\n' '<!doctype html><title>OAMB comparison report</title>' > "$output_root/report.html"
    ;;
  *" oamb report saved-results "*)
    output_root=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --output-root) output_root=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    [ ! -e "$output_root" ] || exit 92
    mkdir -p "$output_root"
    printf '%s\n' '{"coverage":{"cell_count":3,"unique_case_count":60,"provider_specific_result_count":180}}' > "$output_root/report.json"
    printf '%s\n' '<!doctype html><title>OAMB saved results report</title>' > "$output_root/report.html"
    printf '%s\n' '{"schema_name":"report_analysis"}' > "$output_root/report-analysis.json"
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
        fake_bin / "curl",
        """
python3 - <<'PY'
import json
print(json.dumps({"data": [{"embedding": [0.0] * 1024}]}))
PY
""".strip(),
    )
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
    (root / "provider-services" / ".runtime" / "embedding-ready-request.json").write_text(
        '{"dimensions":1024}\n',
        encoding="utf-8",
    )
    (root / ".env").write_text(
        "LLM_URL_TYPE=openai_chat\n"
        "LLM_BASE_URL=test\n"
        "LLM_API_KEY=test\n"
        "LLM_LIGHT_MODEL=runtime-light-model\n"
        "LLM_DEEP_MODEL=runtime-deep-model\n"
        "OAMB_EMBEDDING_BASE_URL=change-me\n"
        "OAMB_EMBEDDING_API_KEY=\n",
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
                "embedding_endpoint": {
                    "effective_endpoint": "http://127.0.0.1:18000/v1",
                    "ownership": "embedding_local_fallback",
                },
                "dataset": {"path": "datasets/longmemeval-cleaned/longmemeval_s_cleaned.json"},
                "execution": {
                    "max_parallel_providers_per_dataset": 3,
                    "extraction_max_retries": 10,
                },
                "model_roles": [
                    {
                        "role_id": "hindsight_extraction",
                        "model": "LLM_LIGHT_MODEL",
                        "thinking_effort": "low",
                    },
                    {
                        "role_id": "mem0_extraction",
                        "model": "LLM_LIGHT_MODEL",
                        "thinking_effort": "high",
                    },
                    {
                        "role_id": "openviking_semantic_understanding",
                        "model": "LLM_LIGHT_MODEL",
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
        "embedding_ownership": "embedding_local_fallback",
    }
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "OAMB_TEST_TRACE": str(trace),
        "OAMB_TEST_PLAN_HASH": RESOLVED_PLAN_HASH,
        "OAMB_TEST_QUESTION_RESULT": json.dumps(_judged_result()),
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
        "provider-config runtime-light-model|runtime-light-model|runtime-light-model|plan-embedding|low|high|max|openai|openai"
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
        timeout=60,
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
            f"provider={provider} status=starting, elapsed=0s, completed_questions=0 (0/60, 0%)"
        ) in result.stdout
        progress_lines = tuple(
            line
            for line in result.stdout.splitlines()
            if line.startswith(f"provider={provider} status=running, elapsed=")
        )
        assert any(
            line.endswith(f"completed_questions={completed} ({completed}/60, {percentage}%)")
            for line in progress_lines
        )
        assert all("completed_operations=" not in line for line in progress_lines)
        assert all("question_progress=" not in line for line in progress_lines)
    assert "/180" not in result.stdout
    assert "/1)" not in result.stdout


@pytest.mark.parametrize("resume", (False, True), ids=("fresh", "resume"))
def test_full_run_logs_saved_question_updates_before_command_finishes(
    tmp_path: Path, resume: bool
) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    fake_bin = Path(env["PATH"].split(os.pathsep, 1)[0])
    _write_executable(fake_bin / "sleep", "/bin/sleep 0.05")
    arguments = [str(script), "--full_test"]
    initial = 46 if resume else 0
    if resume:
        arguments.append("--resume")
        _full_root, result_paths = _write_provider_result_run(root)
        for path, count in zip(result_paths, (60, 60, 46), strict=True):
            path.write_text(
                json.dumps({f"q-{i}": _judged_result(f"q-{i}") for i in range(count)}),
                encoding="utf-8",
            )
    env["OAMB_TEST_RESULT_UPDATES"] = json.dumps(
        [_judged_result(f"q-{initial}"), _judged_result(f"q-{initial + 1}")]
    )

    result = subprocess.run(
        arguments, cwd=root, env=env, capture_output=True, text=True, check=False, timeout=20
    )

    assert result.returncode == 44, result.stdout + result.stderr
    live_output = result.stdout.split("fixture-run-active\n", 1)[1].split(
        "fixture-run-finished\n", 1
    )[0]
    # A polling monitor can coalesce consecutive writes; the latest update must
    # still be observable while the command is running, before its final output.
    count, percent = (48, 80) if resume else (2, 3)
    assert any(
        line.startswith("provider=openviking status=running, elapsed=")
        and line.endswith(f"completed_questions={count} ({count}/60, {percent}%)")
        for line in live_output.splitlines()
    ), result.stdout
    if resume:
        for provider in ("hindsight", "mem0"):
            assert (
                f"provider={provider} status=execution-completed, elapsed=unavailable, "
                "completed_questions=60 (60/60, 100%)"
            ) in live_output
            assert f"provider={provider} status=starting" not in result.stdout
        assert "completed_questions=46 (46/60, 76%)" in result.stdout
    log = next((root / "outputs" / "tmp").glob("run-full-*.log"))
    assert live_output in log.read_text(encoding="utf-8")


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


def _last_provider_progress_line(output: str, provider: str) -> str:
    matches = tuple(
        line for line in output.splitlines() if line.startswith(f"provider={provider} status=")
    )
    assert matches, f"missing progress for provider={provider}"
    return matches[-1]


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
    assert _last_provider_progress_line(result.stdout, "hindsight").startswith(
        f"provider=hindsight status={failed_state},"
    )
    assert _last_provider_progress_line(result.stdout, "mem0").startswith(
        "provider=mem0 status=execution-completed,"
    )
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
    assert _last_provider_progress_line(result.stdout, "hindsight").startswith(
        "provider=hindsight status=unavailable,"
    )
    assert "provider=hindsight status=aborted" not in result.stdout


def test_run_progress_reports_ambiguous_provider_root_as_unavailable(tmp_path: Path) -> None:
    result = _run_progress_fixture(tmp_path, {}, duplicate_provider="hindsight")

    assert result.returncode == 44, result.stdout + result.stderr
    assert "provider=hindsight status=unavailable, elapsed=unavailable" in result.stdout
    assert _last_provider_progress_line(result.stdout, "hindsight").startswith(
        "provider=hindsight status=unavailable,"
    )


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
    full_run_call = next(
        line for line in calls.splitlines() if "oamb run" in line and "--results-root" in line
    )
    assert "--full-progress-root" not in full_run_call
    assert "--full-resume-lock" not in full_run_call
    assert "--full-resume-pointer" not in full_run_call
    assert "python -" in calls and "/results" in calls
    assert calls.count("oamb capsule validate") == 3
    assert (
        "run: providers=hindsight,mem0,openviking status=starting (full, 60 questions each)"
        in result.stdout
    )
    comparison_call = next(line for line in calls.splitlines() if "oamb compare" in line)
    assert "--diagnostic" not in comparison_call
    assert "--results-root" in comparison_call
    assert "--analysis-model-env" in comparison_call
    assert "--analysis-cache-root" in comparison_call
    assert "--cell-root" not in comparison_call
    assert "--validation" not in comparison_call
    report = json.loads(
        (root / "outputs" / "full-test" / "lme60-test" / "comparison" / "report.json").read_bytes()
    )
    assert report["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 60,
        "provider_specific_result_count": 180,
    }
    full_root = root / "outputs" / "full-test" / "lme60-test"
    source_plan = (
        root / "outputs" / "tmp" / "precheck" / "lme60-test" / "plan" / "resolved-plan.json"
    )
    assert (full_root / "resolved-plan.json").read_bytes() == source_plan.read_bytes()
    assert tuple(
        (full_root / "results" / f"{provider}.json").read_bytes()
        for provider in ("hindsight", "mem0", "openviking")
    ) == (b"{}", b"{}", b"{}")
    assert (root / "outputs" / "tmp" / "full-test-current").read_text(
        encoding="utf-8"
    ) == "lme60-test\n"


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
    full_run_directories = tuple(sorted((root / "outputs" / "full-test").glob("lme60-test*")))
    assert len(full_run_directories) == 3
    assert all((directory / "resolved-plan.json").is_file() for directory in full_run_directories)
    selected_label = (
        (root / "outputs" / "tmp" / "full-test-current").read_text(encoding="utf-8").strip()
    )
    assert root / "outputs" / "full-test" / selected_label in full_run_directories
    assert json.loads(full_result.read_bytes())["status"] == "failed"
    assert not tuple(full_result.parent.glob("full-retry-*.json"))


def _write_provider_result_run(root: Path) -> tuple[Path, tuple[Path, ...]]:
    full_root = root / "outputs" / "full-test" / "lme60-test"
    results_root = full_root / "results"
    results_root.mkdir(parents=True)
    result_paths = tuple(
        results_root / f"{provider_id}.json" for provider_id in ("hindsight", "mem0", "openviking")
    )
    for path in result_paths:
        path.write_text("{}", encoding="utf-8")
    source_plan = (
        root / "outputs" / "tmp" / "precheck" / "lme60-test" / "plan" / "resolved-plan.json"
    )
    shutil.copyfile(source_plan, full_root / "resolved-plan.json")
    selector = root / "outputs" / "tmp" / "full-test-current"
    selector.write_text("lme60-test\n", encoding="utf-8")
    return full_root, result_paths


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


def test_run_full_resume_runs_once_and_compares_provider_results(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    full_root, result_paths = _write_provider_result_run(root)
    original_results = tuple(path.read_bytes() for path in result_paths)
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "LLM_LIGHT_MODEL=runtime-light-model",
            "LLM_LIGHT_MODEL=changed-runtime-model",
        ),
        encoding="utf-8",
    )

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
    run_calls = [line for line in calls if "oamb run" in line]
    compare_calls = [line for line in calls if "oamb compare" in line]
    doctor_calls = [line for line in calls if line == "provider-services doctor"]
    provider_up_calls = [line for line in calls if line == "provider-services up"]
    provider_verify_calls = [
        line for line in calls if line == "provider-services verify --services"
    ]

    assert len(run_calls) == 1
    assert f"oamb run {full_root}/resolved-plan.json" in run_calls[0]
    assert f"--resume-concurrency-config {root}/configs/benchmark.yml" in run_calls[0]
    owner_lines = [line for line in calls if line.startswith("run-owner-pid ")]
    assert len(owner_lines) == 1
    assert owner_lines[0].removeprefix("run-owner-pid ").isdigit()
    assert len(compare_calls) == 1
    assert len(doctor_calls) == 1
    assert len(provider_up_calls) == 1
    assert len(provider_verify_calls) == 1
    results_root = full_root / "results"
    assert all(f"--results-root {results_root}" in line for line in run_calls)
    assert all("--full-progress-root" not in line for line in run_calls)
    assert all("--full-resume-lock" not in line for line in run_calls)
    assert f"--output-root {full_root}/capsules/resume/simple-resume-" in run_calls[0]
    assert f"--results-root {results_root}" in compare_calls[0]
    assert "--cell-root" not in compare_calls[0]
    assert "--validation" not in compare_calls[0]
    assert calls.index(doctor_calls[0]) < calls.index(provider_up_calls[0])
    assert calls.index(provider_up_calls[0]) < calls.index(provider_verify_calls[0])
    assert calls.index(provider_verify_calls[0]) < calls.index(run_calls[0])
    assert calls.index(run_calls[0]) < calls.index(compare_calls[0])
    trace_text = "\n".join(calls)
    assert all(
        obsolete not in trace_text
        for obsolete in (
            "--recover-from",
            "--continue-from",
            "--recovery-analysis-output",
            "oamb capsule compose",
        )
    )
    assert tuple(path.read_bytes() for path in result_paths) == original_results
    provider_config = next(line for line in calls if line.startswith("provider-config "))
    assert provider_config.startswith(
        "provider-config changed-runtime-model|changed-runtime-model|changed-runtime-model|plan-embedding|"
    )
    assert "run: PASS (full, 60 questions, 180 provider results)" in result.stdout


def test_run_generate_report_skips_precheck_services_and_provider_calls(tmp_path: Path) -> None:
    """Catches the offline saved-result report accidentally entering the live run path."""

    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    result_root = root / "outputs" / "saved-results" / "provider-native"
    result_root.mkdir(parents=True)
    (root / "outputs" / "tmp" / "quick-start-current.json").unlink()
    env["OAMB_NO_OPEN"] = "1"

    result = subprocess.run(
        [
            str(script),
            "--generate-report",
            f"--result-dir={result_root}",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    calls = trace.read_text(encoding="utf-8").splitlines()
    report_calls = [line for line in calls if "oamb report saved-results" in line]
    assert len(report_calls) == 1
    assert str(result_root) in report_calls[0]
    assert f"--output-root {result_root}/comparison" in report_calls[0]
    assert f"--analysis-model-env {root}/.env" in report_calls[0]
    assert f"--analysis-cache-root {result_root}/report-analysis-cache" in report_calls[0]
    assert not any("provider-services" in line for line in calls)
    assert not any("oamb run" in line for line in calls)
    assert not any("oamb compare" in line for line in calls)
    assert "run: PASS (saved report, 60 questions, 180 provider results)" in result.stdout
    assert f"report: {result_root}/comparison/report.html" in result.stdout


@pytest.mark.parametrize(
    "embedding_url",
    ("http://127.0.0.1:19000/v1", "https://embedding.example/v1"),
    ids=("explicit_loopback", "remote"),
)
def test_run_full_resume_probes_configured_embedding_without_starting_local(
    tmp_path: Path,
    embedding_url: str,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    full_root, _result_paths = _write_provider_result_run(root)
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["embedding_ownership"] = "external"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    for plan_path in (
        Path(state["resolved_plan"]),
        full_root / "resolved-plan.json",
    ):
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        document["embedding_endpoint"] = {
            "effective_endpoint": embedding_url,
            "ownership": "external",
        }
        plan_path.write_text(json.dumps(document), encoding="utf-8")
    _write_executable(
        Path(env["PATH"].split(":", 1)[0]) / "curl",
        'printf "embedding-probe %s\\n" "$*" >> "$OAMB_TEST_TRACE"; '
        'python3 -c \'import json; print(json.dumps({"data": [{"embedding": [0.0] * 1024}]}))\'',
    )

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
    assert "embedding: PASS (finite 1024-dimensional vector)" in result.stdout
    calls = trace.read_text(encoding="utf-8")
    probe_line = next(line for line in calls.splitlines() if line.startswith("embedding-probe "))
    assert probe_line.endswith(f"{embedding_url}/embeddings")
    assert "provider-services up" in calls


def test_run_full_resume_ignores_embedding_ownership_state(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_provider_result_run(root)
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["embedding_ownership"] = "false"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    (root / "provider-services" / ".runtime" / "embedding-ready-request.json").write_text(
        '{"dimensions":768}\n', encoding="utf-8"
    )
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "OAMB_EMBEDDING_BASE_URL=change-me",
            "OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:19001/v1",
        ),
        encoding="utf-8",
    )
    _write_executable(
        Path(env["PATH"].split(":", 1)[0]) / "curl",
        'printf "embedding-probe %s\\n" "$*" >> "$OAMB_TEST_TRACE"; '
        'python3 -c \'import json; print(json.dumps({"data": [{"embedding": [0.0] * 768}]}))\'',
    )

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
    calls = trace.read_text(encoding="utf-8")
    assert "http://127.0.0.1:19001/v1/embeddings" in calls
    assert "embedding: PASS (finite 768-dimensional vector)" in result.stdout
    assert "provider-services up" in calls
    assert "oamb run" in calls


def test_run_full_resume_accepts_legacy_plan_without_embedding_identity(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    full_root, _result_paths = _write_provider_result_run(root)
    state_path = root / "outputs" / "tmp" / "quick-start-current.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.pop("embedding_ownership")
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env_path = root / ".env"
    env_path.write_text(
        env_path.read_text(encoding="utf-8").replace(
            "OAMB_EMBEDDING_BASE_URL=change-me",
            "OAMB_EMBEDDING_BASE_URL=http://127.0.0.1:19000/v1",
        ),
        encoding="utf-8",
    )
    for plan_path in (Path(state["resolved_plan"]), full_root / "resolved-plan.json"):
        document = json.loads(plan_path.read_text(encoding="utf-8"))
        document.pop("embedding_endpoint")
        plan_path.write_text(json.dumps(document) + "\n", encoding="utf-8")

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
    calls = trace.read_text(encoding="utf-8")
    assert "embedding-start" not in calls
    assert "provider-services up" in calls
    assert "oamb run" in calls


def test_plan_loader_accepts_legacy_resume_with_runtime_embedding_endpoint(
    tmp_path: Path,
) -> None:
    plan = build_resolved_plan(
        load_benchmark_configuration(
            REPOSITORY_ROOT / "configs/benchmark.yml",
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    document = json.loads(resolved_plan_bytes(plan))
    document.pop("embedding_endpoint")
    payload = dict(document)
    payload.pop("resolved_plan_hash")
    legacy_hash = canonical_sha256(["oamb-resolved-plan-initial-v1", payload])
    document["resolved_plan_hash"] = legacy_hash
    plan_path = tmp_path / "legacy-plan.json"
    plan_path.write_bytes(canonical_json_bytes(document))

    loaded = load_resolved_plan_for_run(
        plan_path,
        resume_embedding_endpoint="http://127.0.0.1:19000/v1",
    )

    assert loaded.resolved_plan_hash == legacy_hash
    assert loaded.embedding_endpoint.effective_endpoint == "http://127.0.0.1:19000/v1"
    assert loaded.embedding_endpoint.ownership == "external"


def test_plan_loader_ignores_saved_embedding_identity_during_resume(
    tmp_path: Path,
) -> None:
    plan = build_resolved_plan(
        load_benchmark_configuration(
            REPOSITORY_ROOT / "configs/benchmark.yml",
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    document = json.loads(resolved_plan_bytes(plan))
    document["embedding_endpoint"] = {}
    payload = dict(document)
    payload.pop("resolved_plan_hash")
    saved_hash = canonical_sha256(["oamb-resolved-plan-initial-v1", payload])
    document["resolved_plan_hash"] = saved_hash
    plan_path = tmp_path / "saved-plan.json"
    plan_path.write_bytes(canonical_json_bytes(document))

    loaded = load_resolved_plan_for_run(
        plan_path,
        resume_embedding_endpoint="http://127.0.0.1:19000/v1",
    )

    assert loaded.resolved_plan_hash == saved_hash
    assert loaded.embedding_endpoint.effective_endpoint == "http://127.0.0.1:19000/v1"
    assert loaded.embedding_endpoint.ownership == "external"


def test_run_full_resume_rejects_wrong_embedding_dimension_without_starting_it(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_provider_result_run(root)
    _write_executable(
        Path(env["PATH"].split(":", 1)[0]) / "curl",
        """
printf 'embedding-probe\n' >> "$OAMB_TEST_TRACE"
python3 - <<'PY'
import json
print(json.dumps({"data": [{"embedding": [0.0] * 768}]}))
PY
""".strip(),
    )
    _write_executable(
        root / "scripts" / "start_local_embedding" / "start_vllm_metal.sh",
        """
printf 'embedding-start\n' >> "$OAMB_TEST_TRACE"
""".strip(),
    )
    provider_ready = tmp_path / "provider-ready"
    _write_executable(
        root / "provider-services" / "bin" / "provider-services",
        """
printf 'provider-services %s\n' "$*" >> "$OAMB_TEST_TRACE"
if [ "$1" = "up" ]; then
  [ -f "$OAMB_TEST_EMBED_READY" ] || exit 71
  touch "$OAMB_TEST_PROVIDER_READY"
fi
if [ "$1" = "verify" ] && [ "$2" = "--services" ]; then
  [ -f "$OAMB_TEST_EMBED_READY" ] || exit 72
  [ -f "$OAMB_TEST_PROVIDER_READY" ] || exit 73
fi
""".strip(),
    )
    env.update(
        {
            "OAMB_TEST_PROVIDER_READY": str(provider_ready),
            "OAMB_EMBEDDING_STARTUP_ATTEMPTS": "3",
        }
    )

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
    assert "embedding-start" not in calls
    assert "provider-services up" not in calls
    assert not any("oamb run" in line for line in calls)
    assert not provider_ready.exists()


def test_run_full_resume_does_not_wait_for_recorded_embedding_startup(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_provider_result_run(root)
    state = json.loads(
        (root / "outputs" / "tmp" / "quick-start-current.json").read_text(encoding="utf-8")
    )
    (Path(state["work_dir"]) / "embedding.pid").write_text(
        f"{os.getpid()}\n",
        encoding="utf-8",
    )
    _write_executable(
        Path(env["PATH"].split(":", 1)[0]) / "curl",
        """
count=0
if [ -f "$OAMB_TEST_EMBED_PROBE_COUNT" ]; then
  count=$(cat "$OAMB_TEST_EMBED_PROBE_COUNT")
fi
count=$((count + 1))
printf '%s\n' "$count" > "$OAMB_TEST_EMBED_PROBE_COUNT"
printf 'embedding-probe %s\n' "$count" >> "$OAMB_TEST_TRACE"
[ "$count" -ge 2 ] || exit 22
python3 -c 'import json; print(json.dumps({"data": [{"embedding": [0.0] * 1024}]}))'
""".strip(),
    )
    _write_executable(
        root / "scripts" / "start_local_embedding" / "start_vllm_metal.sh",
        "printf 'embedding-start\n' >> \"$OAMB_TEST_TRACE\"; exit 81",
    )
    env.update(
        {
            "OAMB_TEST_EMBED_PROBE_COUNT": str(tmp_path / "embedding-probe-count"),
            "OAMB_EMBEDDING_STARTUP_ATTEMPTS": "3",
        }
    )

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
    assert "embedding-start" not in calls
    assert calls.count("embedding-probe 1") == 1
    assert "embedding-probe 2" not in calls
    assert "embedding: waiting for existing startup" not in result.stdout
    assert not any("oamb run" in line for line in calls)


def test_recorded_embedding_timeout_names_pid_without_claiming_a_new_log(
    tmp_path: Path,
) -> None:
    pid_file = tmp_path / "embedding.pid"
    pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
    log_file = tmp_path / "new-attempt.log"
    program = f"""
ROOT={REPOSITORY_ROOT}
WORK_DIR={tmp_path}
EMBEDDING_STARTUP_ATTEMPTS=1
. "{HOST_EMBEDDING_SCRIPT}"
die() {{ printf '%s\n' "$*" >&2; exit 1; }}
uname() {{ printf 'Darwin\n'; }}
resolve_host_embedding_base() {{ printf 'http://127.0.0.1:18000/v1\n'; }}
probe_embedding() {{ return 1; }}
start_local_embedding 'http://host.docker.internal:18000/v1' 'local-key' '{log_file}' '{pid_file}'
"""

    result = subprocess.run(
        ["sh", "-c", program],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )

    assert result.returncode != 0
    assert f"recorded embedding helper did not become ready: {os.getpid()}" in result.stderr
    assert str(log_file) not in result.stderr


def test_run_full_resume_failure_stops_before_comparison(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_provider_result_run(root)
    env["OAMB_TEST_RESUME_FAIL"] = "1"

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
    run_calls = [line for line in calls if "oamb run" in line]
    assert len(run_calls) == 1
    assert "--full-resume-rehearsal" not in run_calls[0]
    doctor_index = calls.index("provider-services doctor")
    provider_up_index = calls.index("provider-services up")
    provider_verify_index = calls.index("provider-services verify --services")
    resume_index = calls.index(run_calls[0])
    assert doctor_index < provider_up_index < provider_verify_index < resume_index
    assert not any("oamb compare" in line for line in calls)


def test_run_sh_owner_death_stops_in_flight_comparison(tmp_path: Path) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    _write_provider_result_run(root)
    compare_pid_path = tmp_path / "compare.pid"
    compare_started = tmp_path / "compare-started"
    compare_stopped = tmp_path / "compare-stopped"
    env.update(
        {
            "OAMB_TEST_COMPARE_BLOCK": "1",
            "OAMB_TEST_COMPARE_PID": str(compare_pid_path),
            "OAMB_TEST_COMPARE_STARTED": str(compare_started),
            "OAMB_TEST_COMPARE_STOPPED": str(compare_stopped),
            "OAMB_TEST_COMPARE_EXIT_WITH_OWNER": "1",
        }
    )

    process = subprocess.Popen(
        [str(script), "--full_test", "--resume"],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    compare_pid = 0
    try:
        deadline = time.monotonic() + 10
        while not compare_started.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert compare_started.is_file(), "comparison did not reach the planted barrier"
        compare_pid = int(compare_pid_path.read_text(encoding="utf-8"))

        os.kill(process.pid, signal.SIGKILL)
        process.wait(timeout=1)
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                os.kill(compare_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("comparison process did not exit after its stop handler")
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if "provider-services stop" in trace.read_text(encoding="utf-8"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("owner-death watchdog did not stop provider services")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=1)
        if compare_pid:
            try:
                os.kill(compare_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.parametrize(
    ("stop_signal", "send_to_group"),
    ((signal.SIGINT, True), (signal.SIGTERM, False)),
)
def test_one_signal_stops_owned_full_run_tree_once(
    tmp_path: Path,
    stop_signal: signal.Signals,
    send_to_group: bool,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    owned_pid_path = tmp_path / "owned-run.pid"
    run_started = tmp_path / "owned-run-started"
    received_signals = tmp_path / "owned-run-signals"
    env.update(
        {
            "OAMB_TEST_RUN_SIGNAL_BLOCK": "1",
            "OAMB_TEST_RUN_SIGNAL_PID": str(owned_pid_path),
            "OAMB_TEST_RUN_SIGNAL_STARTED": str(run_started),
            "OAMB_TEST_RUN_SIGNALS": str(received_signals),
        }
    )

    process = subprocess.Popen(
        [str(script), "--full_test"],
        cwd=root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    owned_pid = 0
    try:
        deadline = time.monotonic() + 10
        while not run_started.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert run_started.is_file(), "full run did not reach the planted barrier"
        owned_pid = int(owned_pid_path.read_text(encoding="utf-8"))
        if send_to_group:
            os.killpg(process.pid, stop_signal)
        else:
            os.kill(process.pid, stop_signal)
        process.wait(timeout=10)

        assert not received_signals.exists()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if "provider-services stop" in trace.read_text(encoding="utf-8"):
                break
            time.sleep(0.01)
        else:
            pytest.fail("stop handler did not stop provider services")
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            try:
                os.kill(owned_pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.01)
        else:
            pytest.fail("owned full-run command remained alive")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=1)
        if owned_pid:
            try:
                os.kill(owned_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
