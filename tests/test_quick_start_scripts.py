from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).parents[1]
PRECHECK_SCRIPT = REPOSITORY_ROOT / "precheck.sh"
RUN_SCRIPT = REPOSITORY_ROOT / "run.sh"
HOST_EMBEDDING_SCRIPT = REPOSITORY_ROOT / "provider-services" / "lib" / "host_embedding.sh"
RESOLVED_PLAN_HASH = "a" * 64


def _write_executable(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"#!/bin/bash\nset -eu\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _write_configured_environment(root: Path) -> None:
    (root / ".env").write_text(
        "DEEPSEEK_BASE_URL=https://models.example/v1\nDEEPSEEK_API_KEY=test-model-key\n",
        encoding="utf-8",
    )
    (root / ".env").chmod(0o600)
    provider_env = root / "provider-services" / ".env"
    provider_env.parent.mkdir(parents=True, exist_ok=True)
    provider_env.write_text(
        "\n".join(
            (
                "OAMB_PROVIDER_PROJECT=oamb-providers-test",
                f"OAMB_MEM0_SOURCE_CHECKOUT={root / '.local-demo/provider-source/mem0'}",
                "OAMB_EMBEDDING_BASE_URL=http://host.docker.internal:18000/v1",
                "OAMB_EMBEDDING_MODEL=qwen3-embedding:0.6b",
                "OAMB_HINDSIGHT_LLM_BASE_URL=https://models.example/v1",
                "OAMB_HINDSIGHT_LLM_API_KEY=test-model-key",
                "OAMB_MEM0_LLM_BASE_URL=https://models.example/v1",
                "OAMB_MEM0_LLM_API_KEY=test-model-key",
                "OAMB_MEM0_ADMIN_API_KEY=test-admin-key-long-enough-0001",
                "OAMB_MEM0_JWT_SECRET=test-jwt-secret-long-enough-0000000000000001",
                "OAMB_MEM0_POSTGRES_PASSWORD=test-postgres-password",
                "OAMB_MEM0_INSPECTOR_API_KEY=test-inspector-key-long-enough-01",
                "OAMB_OPENVIKING_VLM_BASE_URL=https://models.example/v1",
                "OAMB_OPENVIKING_VLM_API_KEY=test-model-key",
                "OAMB_OPENVIKING_ROOT_API_KEY=test-openviking-key-long-enough-01",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    provider_env.chmod(0o600)


def _copy_quick_start_script(source: Path, root: Path) -> Path:
    assert source.is_file(), f"quick-start script is missing: {source.name}"
    destination = root / source.name
    shutil.copy2(source, destination)
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
        printf '%s\n' '{"schema_name":"resolved_comparison_plan"}' > "$2/resolved-plan.json"
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
        'printf \'provider-services %s\\n\' "$*" >> "$OAMB_TEST_TRACE"',
    )
    (root / "provider-services" / ".env.example").write_text("", encoding="utf-8")
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
    provider_env = (root / "provider-services" / ".env").read_text(encoding="utf-8")
    assert "OAMB_EMBEDDING_BASE_URL=https://embedding.example/v1" in provider_env


def _write_fake_oamb(fake_bin: Path) -> None:
    _write_executable(
        fake_bin / "uv",
        r"""
printf 'uv %s\n' "$*" >> "$OAMB_TEST_TRACE"
case " $* " in
  *" oamb run "*)
    cell=""
    result_map=""
    output_root=""
    while [ "$#" -gt 0 ]; do
      case "$1" in
        --cell) cell=$2; shift 2 ;;
        --result-map) result_map=$2; shift 2 ;;
        --output-root) output_root=$2; shift 2 ;;
        *) shift ;;
      esac
    done
    mkdir -p "$output_root" "$(dirname "$result_map")"
    [ ! -e "$result_map" ] || exit 91
    if [ -n "$cell" ]; then
      root="$output_root/$cell-capsule"
      mkdir -p "$root"
      printf '{"schema_name":"live_run_result_map","schema_version":1,"resolved_plan_hash":"%s","status":"completed","cells":[{"cell_id":"%s","status":"completed","capsule_root":"%s","detail":null}],"capsule_roots":{"%s":"%s"}}\n' \
        "$OAMB_TEST_PLAN_HASH" "$cell" "$root" "$cell" "$root" > "$result_map"
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
    work_dir = root / ".local-demo" / "lme60-test"
    plan = work_dir / "plan" / "resolved-plan.json"
    plan.parent.mkdir(parents=True)
    plan.write_text(json.dumps({"resolved_plan_hash": RESOLVED_PLAN_HASH}) + "\n", encoding="utf-8")
    dataset = root / "datasets" / "longmemeval-cleaned" / "longmemeval_s_cleaned.json"
    dataset.parent.mkdir(parents=True)
    dataset.write_text("[]\n", encoding="utf-8")
    state = {
        "schema_version": 1,
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
        assert f"--cell {cell} --question 72e3ee87" in calls
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


def test_run_smoke_reuses_a_successful_retry_result_map_on_third_invocation(
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
    result_map = root / ".local-demo" / "lme60-test" / "smoke" / "results" / "bounded-mem0.json"
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

    mem0_run_calls = [
        line
        for line in trace.read_text(encoding="utf-8").splitlines()
        if "oamb run" in line and "--cell mem0-lme60" in line
    ]
    assert len(mem0_run_calls) == 2
    retry_maps = tuple(result_map.parent.glob("bounded-mem0-retry-*.json"))
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
