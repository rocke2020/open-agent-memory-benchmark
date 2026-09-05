from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
PRECHECK_SCRIPT = REPOSITORY_ROOT / "precheck.sh"
RUN_SCRIPT = REPOSITORY_ROOT / "run.sh"
HOST_EMBEDDING_SCRIPT = REPOSITORY_ROOT / "provider-services" / "lib" / "host_embedding.sh"
PLAN_ENVIRONMENT_SCRIPT = REPOSITORY_ROOT / "provider-services" / "lib" / "plan_environment.sh"
RESOLVED_PLAN_HASH = "a" * 64


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/bash\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _write_configured_environment(root: Path) -> None:
    configured = (
        (REPOSITORY_ROOT / ".env.example")
        .read_text(encoding="utf-8")
        .replace(
            "DEEPSEEK_BASE_URL=change-me\nDEEPSEEK_API_KEY=change-me",
            "DEEPSEEK_BASE_URL=https://models.example/v1\nDEEPSEEK_API_KEY=test-model-key",
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
    (root / ".local-demo" / "provider-source" / "mem0" / ".git").mkdir(parents=True)

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
  *" oamb doctor "*)
    while [ "$#" -gt 0 ]; do
      if [ "$1" = "--output" ]; then
        mkdir -p "$2"
        printf '%s\n' '{"schema_name":"resolved_comparison_plan","model_roles":[{"role_id":"hindsight_extraction","configured_model":"plan-hindsight","thinking_effort":"low"},{"role_id":"mem0_extraction","configured_model":"plan-mem0","thinking_effort":"high"},{"role_id":"openviking_semantic_understanding","configured_model":"plan-openviking","thinking_effort":"max"},{"role_id":"embedding","configured_model":"plan-embedding","thinking_effort":"not_applicable"}]}' > "$2/resolved-plan.json"
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
    state = json.loads((root / ".local-demo" / "quick-start-current.json").read_bytes())
    assert Path(state["resolved_plan"]).is_file()
    assert state["question_id"] == "72e3ee87"


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
        "DEEPSEEK_BASE_URL=change-me\nDEEPSEEK_API_KEY=change-me",
        "DEEPSEEK_BASE_URL=https://models.example/v1\nDEEPSEEK_API_KEY=test-model-key",
    )
    (root / ".env.example").write_text(root_template.read_text(encoding="utf-8"), encoding="utf-8")
    (root / ".env").write_text(
        configured
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
    assert "OAMB_HINDSIGHT_LLM_BASE_URL=https://models.example/v1" in root_env
    assert "OAMB_HINDSIGHT_LLM_API_KEY=test-model-key" in root_env
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
printf 'uv %s\n' "$*" >> "$OAMB_TEST_TRACE"
case " $* " in
  *" oamb run "*)
    cells=()
    result_map=""
    output_root=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --cell) cells+=("$2"); shift 2 ;;
        --result-map) result_map=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$output_root" "$(dirname "$result_map")"
    progress_cases=${OAMB_TEST_PROGRESS_CASES:-0}
    if [ "$progress_cases" -gt 0 ]; then
      mkdir -p "$output_root/progress/source/cases"
      progress_index=0
      while [ "$progress_index" -lt "$progress_cases" ]; do
        printf '{}\n' > "$output_root/progress/source/cases/$progress_index.json"
        progress_index=$((progress_index + 1))
      done
    fi
    if [ "${OAMB_TEST_OAMB_RUN_DELAY:-0}" = "1" ]; then
      /bin/sleep 1
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
    (root / ".env").write_text("DEEPSEEK_BASE_URL=test\nDEEPSEEK_API_KEY=test\n", encoding="utf-8")
    (root / ".env").chmod(0o600)
    work_dir = root / ".local-demo" / "lme60-test"
    plan = work_dir / "plan" / "resolved-plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(
        json.dumps(
            {
                "resolved_plan_hash": RESOLVED_PLAN_HASH,
                "model_roles": [
                    {
                        "role_id": "hindsight_extraction",
                        "configured_model": "plan-hindsight",
                        "thinking_effort": "low",
                    },
                    {
                        "role_id": "mem0_extraction",
                        "configured_model": "plan-mem0",
                        "thinking_effort": "high",
                    },
                    {
                        "role_id": "openviking_semantic_understanding",
                        "configured_model": "plan-openviking",
                        "thinking_effort": "max",
                    },
                    {
                        "role_id": "embedding",
                        "configured_model": "plan-embedding",
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
    state_path = root / ".local-demo" / "quick-start-current.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "OAMB_TEST_TRACE": str(trace),
        "OAMB_TEST_PLAN_HASH": RESOLVED_PLAN_HASH,
    }
    return root, env, trace


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
    assert "oamb run" not in calls
    assert "oamb capsule validate" not in calls
    assert "oamb compare" not in calls
    assert "open " not in calls
    assert not (root / ".local-demo" / "lme60-test" / expected_mode).exists()
    assert (
        f"run: PASS (dry-run, {expected_mode}, {expected_questions} questions, "
        "3 providers, zero model/provider calls)"
    ) in result.stdout


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
    for cell in ("hindsight-lme60", "mem0-lme60", "openviking-lme60"):
        assert f"--cell {cell}" in calls
    assert "--question 72e3ee87" in calls
    assert calls.count("oamb capsule validate") == 3
    assert "oamb compare" in calls and "--diagnostic" in calls
    assert "open " in calls and "/smoke/comparison/report.html" in calls
    report = json.loads(
        (root / ".local-demo" / "lme60-test" / "smoke" / "comparison" / "report.json").read_bytes()
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
        "run: step 1/5 provider=hindsight status=starting (smoke, question 72e3ee87)"
    ) in result.stdout
    assert "run: providers=hindsight,mem0,openviking status=running" in result.stdout
    assert "completed_questions=1, question_progress=33% (1/3)" in result.stdout
    assert "question_progress=100% (3/3)" in result.stdout
    assert "run: step 1/5 provider=hindsight status=validating" in result.stdout
    assert "run: step 1/5 provider=hindsight status=completed" in result.stdout
    assert "run: step 2/5 provider=mem0 status=starting" in result.stdout
    assert "run: step 3/5 provider=openviking status=starting" in result.stdout
    assert "run: step 4/5 status=building-comparison" in result.stdout
    assert "run: step 4/5 status=completed" in result.stdout
    assert "run: step 5/5 status=opening-report" in result.stdout
    assert "run: step 5/5 status=completed" in result.stdout


def test_run_full_reports_bounded_and_full_question_percentages(tmp_path: Path) -> None:
    root, env, _trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    fake_bin = Path(env["PATH"].split(os.pathsep, 1)[0])
    _write_executable(fake_bin / "sleep", "/bin/sleep 0.05")
    env["OAMB_TEST_OAMB_RUN_DELAY"] = "1"

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
    assert "question_progress=0% (0/3)" in result.stdout
    assert "question_progress=100% (3/3)" in result.stdout
    assert "question_progress=0% (0/180)" in result.stdout
    assert "question_progress=100% (180/180)" in result.stdout


@pytest.mark.parametrize("arguments", (["--smoke_test"], ["--full_test"]))
def test_run_dispatches_all_provider_cells_together(
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

    assert result.returncode == 0, result.stdout + result.stderr
    bounded_run_calls = [
        line
        for line in trace.read_text(encoding="utf-8").splitlines()
        if "oamb run" in line and "--question 72e3ee87" in line
    ]
    assert len(bounded_run_calls) == 1
    for cell in ("hindsight-lme60", "mem0-lme60", "openviking-lme60"):
        assert f"--cell {cell}" in bounded_run_calls[0]


def test_run_smoke_reuses_completed_capsule_and_preserves_invalid_validation(
    tmp_path: Path,
) -> None:
    root, env, trace = _run_fixture(tmp_path)
    script = _copy_quick_start_script(RUN_SCRIPT, root)
    smoke = root / ".local-demo" / "lme60-test" / "smoke"
    capsule = smoke / "capsules" / "bounded" / "hindsight-completed"
    capsule.mkdir(parents=True)
    result_map = smoke / "results" / "bounded-hindsight.json"
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
    invalid_validation = smoke / "validations" / "bounded-hindsight.json"
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
    assert "oamb run" in calls and "--cell hindsight-lme60" not in calls
    assert "--cell mem0-lme60" in calls
    assert "--cell openviking-lme60" in calls
    assert invalid_validation.read_text(encoding="utf-8") == '{"disposition":"invalid"}\n'
    retry_validations = tuple(invalid_validation.parent.glob("bounded-hindsight-retry-*.json"))
    assert len(retry_validations) == 1
    assert json.loads(retry_validations[0].read_bytes())["disposition"] == "validated"


def test_run_smoke_reuses_a_successful_combined_retry_result_map_on_third_invocation(
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
    result_map = root / ".local-demo" / "lme60-test" / "smoke" / "results" / "bounded.json"
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

    bounded_run_calls = [
        line
        for line in trace.read_text(encoding="utf-8").splitlines()
        if "oamb run" in line and "--question 72e3ee87" in line
    ]
    assert len(bounded_run_calls) == 2
    retry_maps = tuple(result_map.parent.glob("bounded-retry-*.json"))
    assert len(retry_maps) == 1
    assert json.loads(retry_maps[0].read_bytes())["status"] == "completed"


def test_run_full_test_uses_bounded_proofs_and_validates_sixty_case_report(
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
    assert calls.count("--bounded-capsule") == 3
    assert calls.count("--bounded-validation") == 3
    assert calls.count("oamb capsule validate") == 6
    assert "run: step 4/6 providers=hindsight,mem0,openviking status=starting" in result.stdout
    assert "run: step 4/6 providers=hindsight,mem0,openviking status=validating" in result.stdout
    assert "run: step 4/6 providers=hindsight,mem0,openviking status=completed" in result.stdout
    comparison_call = next(line for line in calls.splitlines() if "oamb compare" in line)
    assert "--diagnostic" not in comparison_call
    report = json.loads(
        (root / ".local-demo" / "lme60-test" / "full" / "comparison" / "report.json").read_bytes()
    )
    assert report["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 60,
        "provider_specific_result_count": 180,
    }


def test_run_full_test_reuses_completed_full_capsules_and_preserves_validation(
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
        root / ".local-demo" / "lme60-test" / "full" / "validations" / "full-hindsight.json"
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
    assert len(full_run_calls) == 1
    assert full_validation.read_text(encoding="utf-8") == '{"disposition":"invalid"}\n'
    retries = tuple(full_validation.parent.glob("full-hindsight-retry-*.json"))
    assert len(retries) == 1
    assert json.loads(retries[0].read_bytes())["disposition"] == "validated"


def test_run_full_test_retries_instead_of_reusing_failed_full_result_map(
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
    full_result = root / ".local-demo" / "lme60-test" / "full" / "results" / "full.json"
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
    assert len(full_run_calls) == 2
    assert json.loads(full_result.read_bytes())["status"] == "failed"
    retry_maps = tuple(full_result.parent.glob("full-retry-*.json"))
    assert len(retry_maps) == 1
    assert json.loads(retry_maps[0].read_bytes())["status"] == "completed"
