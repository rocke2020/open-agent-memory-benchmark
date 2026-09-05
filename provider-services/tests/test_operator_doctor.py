from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parent
MEM0_COMMIT = "dc82354e143c2581d505d581a00286d6ef8c3605"
PROVIDER_MODEL_VARIABLES = (
    "OAMB_HINDSIGHT_LLM_MODEL",
    "OAMB_MEM0_LLM_MODEL",
    "OAMB_OPENVIKING_VLM_MODEL",
    "OAMB_EMBEDDING_MODEL",
)


def _available_ports(count: int) -> tuple[int, ...]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return tuple(item.getsockname()[1] for item in sockets)
    finally:
        for item in sockets:
            item.close()


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


class OperatorDoctorTests(unittest.TestCase):
    def test_mem0_bootstrap_sanitizes_compose_exec_and_restart(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        bundle = directory / "provider-services"
        for relative_path in ("mem0/bootstrap.sh", "lib/env.sh", "lib/compose.sh"):
            source = ROOT / relative_path
            if source.exists():
                target = bundle / relative_path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

        values: list[str] = []
        for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            name, value = line.split("=", 1)
            if name == "OAMB_PROVIDER_PROJECT":
                value = "file-project"
            elif name == "OAMB_MEM0_POSTGRES_PASSWORD":
                value = "file-postgres-password"
            elif value.startswith("change-me"):
                value = f"file-value-{name.lower()}"
            values.append(f"{name}={value}")
        env_file = directory / ".env"
        env_file.write_text("\n".join(values) + "\n", encoding="utf-8")
        env_file.chmod(0o600)

        fake_bin = directory / "bin"
        fake_bin.mkdir()
        trace = directory / "docker-trace"
        _write_executable(
            fake_bin / "docker",
            "#!/bin/sh\n"
            "printf '%s|%s|%s\\n' \"$OAMB_PROVIDER_PROJECT\" "
            '"$OAMB_MEM0_POSTGRES_PASSWORD" "$*" >> "$OAMB_TEST_TRACE"\n'
            'case "$*" in\n'
            "  *' psql '*) printf 't\\n' ;;\n"
            "  *' alembic current'*) printf '006 \\n' ;;\n"
            "esac\n",
        )
        for command_name in ("jq", "curl"):
            _write_executable(fake_bin / command_name, "#!/bin/sh\nprintf '{}\\n'\n")
        _write_executable(fake_bin / "cmp", "#!/bin/sh\nexit 0\n")

        result = subprocess.run(
            [str(bundle / "mem0/bootstrap.sh"), str(bundle), str(env_file)],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                "OAMB_TEST_TRACE": str(trace),
                "OAMB_PROVIDER_PROJECT": "process-project",
                "OAMB_MEM0_POSTGRES_PASSWORD": "process-postgres-password",
                "OAMB_MEM0_LLM_MODEL": "plan-mem0",
                "OAMB_MEM0_LLM_REASONING_EFFORT": "high",
                "OAMB_EMBEDDING_MODEL": "plan-embedding",
                "OAMB_HINDSIGHT_LLM_MODEL": "plan-hindsight",
                "OAMB_HINDSIGHT_LLM_REASONING_EFFORT": "low",
                "OAMB_OPENVIKING_VLM_MODEL": "plan-openviking",
                "OAMB_OPENVIKING_VLM_REASONING_EFFORT": "max",
            },
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = trace.read_text(encoding="utf-8").splitlines()
        self.assertTrue(any(" exec " in call for call in calls), calls)
        self.assertTrue(any(" restart mem0" in call for call in calls), calls)
        for call in calls:
            self.assertTrue(call.startswith("file-project|file-postgres-password|"), call)
            self.assertIn("compose -p file-project ", call)

    def _run_operator(
        self,
        *command: str,
        model_overrides: dict[str, str] | None = None,
        process_overrides: dict[str, str] | None = None,
        env_extra: str = "",
    ) -> subprocess.CompletedProcess[str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        bundle = directory / "provider-services"
        for relative_path in (
            "bin/provider-services",
            "compose.yaml",
            "lib/compose.sh",
            "lib/env.sh",
            "lib/host_embedding.sh",
            "lib/lifecycle.sh",
            "lib/plan_environment.sh",
            "retry-guard/sitecustomize.py",
            "versions.env",
        ):
            target = bundle / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative_path, target)

        checkout = directory / "mem0"
        (checkout / ".git").mkdir(parents=True)
        ports = _available_ports(4)
        file_overrides = {
            "OAMB_PROVIDER_PROJECT": "oamb-providers-model-profile-test",
            "OAMB_MEM0_SOURCE_CHECKOUT": str(checkout),
            "OAMB_HINDSIGHT_PORT": str(ports[0]),
            "OAMB_MEM0_PORT": str(ports[1]),
            "OAMB_MEM0_INSPECTOR_PORT": str(ports[2]),
            "OAMB_OPENVIKING_PORT": str(ports[3]),
        }
        lines: list[str] = []
        for line in (REPOSITORY_ROOT / ".env.example").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                lines.append(line)
                continue
            name, value = line.split("=", 1)
            if name in PROVIDER_MODEL_VARIABLES:
                continue
            if name in file_overrides:
                value = file_overrides[name]
            elif value.startswith("change-me"):
                value = f"test-value-{name.lower()}"
            lines.append(f"{name}={value}")
        env_file = directory / ".env"
        env_file.write_text("\n".join(lines) + "\n" + env_extra, encoding="utf-8")
        env_file.chmod(0o600)

        plan = directory / "outputs" / "tmp" / "precheck" / "operator-plan" / "resolved-plan.json"
        plan.parent.mkdir(parents=True)
        plan.write_text(
            json.dumps(
                {
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
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        state = directory / "outputs" / "tmp" / "quick-start-current.json"
        state.write_text(json.dumps({"resolved_plan": str(plan)}) + "\n", encoding="utf-8")

        runtime = bundle / ".runtime"
        runtime.mkdir()
        compose_hash = hashlib.sha256((bundle / "compose.yaml").read_bytes()).hexdigest()
        versions_hash = hashlib.sha256((bundle / "versions.env").read_bytes()).hexdigest()
        (runtime / "provider-project.attestation").write_text(
            "project=oamb-providers-model-profile-test\n"
            f"compose_sha256={compose_hash}\n"
            f"versions_sha256={versions_hash}\n",
            encoding="utf-8",
        )

        fake_bin = directory / "bin"
        fake_bin.mkdir()
        _write_executable(
            fake_bin / "docker",
            "#!/bin/sh\n"
            'if [ "${1:-}" = compose ] && [ "${2:-}" != version ]; then\n'
            '  [ -n "${OAMB_MEM0_LLM_MODEL:-}" ] || exit 41\n'
            '  [ -n "${OAMB_EMBEDDING_MODEL:-}" ] || exit 42\n'
            '  [ "${OAMB_MEM0_LLM_REASONING_EFFORT:-}" = high ] || exit 43\n'
            '  printf \'compose-env=%s|%s|%s\\n\' "$OAMB_MEM0_POSTGRES_PASSWORD" "$OAMB_MEM0_LLM_MODEL" "$OAMB_MEM0_LLM_REASONING_EFFORT"\n'
            "fi\n"
            "exit 0\n",
        )
        _write_executable(
            fake_bin / "git",
            "#!/bin/sh\n"
            'case " $* " in\n'
            f"  *' rev-parse '*) printf '%s\\n' '{MEM0_COMMIT}' ;;\n"
            "esac\n"
            "exit 0\n",
        )
        return subprocess.run(
            [str(bundle / "bin" / "provider-services"), *(command or ("doctor",))],
            check=False,
            capture_output=True,
            text=True,
            env={
                **os.environ,
                "PATH": f"{fake_bin}:{os.environ['PATH']}",
                **(model_overrides or {}),
                **(process_overrides or {}),
            },
        )

    def test_doctor_accepts_plan_models_from_the_process_environment(self) -> None:
        result = self._run_operator(
            model_overrides={
                "OAMB_HINDSIGHT_LLM_MODEL": "plan-hindsight",
                "OAMB_HINDSIGHT_LLM_REASONING_EFFORT": "low",
                "OAMB_MEM0_LLM_MODEL": "plan-mem0",
                "OAMB_MEM0_LLM_REASONING_EFFORT": "high",
                "OAMB_OPENVIKING_VLM_MODEL": "plan-openviking",
                "OAMB_OPENVIKING_VLM_REASONING_EFFORT": "max",
                "OAMB_EMBEDDING_MODEL": "plan-embedding",
            },
            process_overrides={"OAMB_MEM0_POSTGRES_PASSWORD": "PROCESS_OVERRIDE_SENTINEL"},
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("doctor: PASS", result.stdout)
        self.assertIn(
            "compose-env=test-value-oamb_mem0_postgres_password|plan-mem0|high",
            result.stdout,
        )
        self.assertNotIn("PROCESS_OVERRIDE_SENTINEL", result.stdout)

    def test_fresh_shell_loads_current_plan_for_doctor_status_and_stop(self) -> None:
        for command in ("doctor", "status", "stop"):
            with self.subTest(command=command):
                result = self._run_operator(command)

                self.assertEqual(result.returncode, 0, result.stderr)

    def test_operator_rejects_plan_owned_model_assignments_in_dotenv(self) -> None:
        result = self._run_operator(
            model_overrides={
                "OAMB_HINDSIGHT_LLM_MODEL": "plan-hindsight",
                "OAMB_HINDSIGHT_LLM_REASONING_EFFORT": "low",
                "OAMB_MEM0_LLM_MODEL": "plan-mem0",
                "OAMB_MEM0_LLM_REASONING_EFFORT": "high",
                "OAMB_OPENVIKING_VLM_MODEL": "plan-openviking",
                "OAMB_OPENVIKING_VLM_REASONING_EFFORT": "max",
                "OAMB_EMBEDDING_MODEL": "plan-embedding",
            },
            env_extra=(
                "OAMB_MEM0_LLM_MODEL=stale-file-model\nOAMB_MEM0_LLM_REASONING_EFFORT=max\n"
            ),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("plan-owned", result.stderr)


if __name__ == "__main__":
    unittest.main()
