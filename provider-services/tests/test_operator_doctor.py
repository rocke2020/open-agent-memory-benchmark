from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEM0_COMMIT = "dc82354e143c2581d505d581a00286d6ef8c3605"
TARGET_PROVIDER_MODEL = "deepseek-v4-flash"
PROVIDER_MODEL_VARIABLES = (
    "OAMB_HINDSIGHT_LLM_MODEL",
    "OAMB_MEM0_LLM_MODEL",
    "OAMB_OPENVIKING_VLM_MODEL",
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
    def _run_doctor(self, model_overrides: dict[str, str]) -> subprocess.CompletedProcess[str]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        bundle = directory / "provider-services"
        for relative_path in (
            ".env.example",
            "bin/provider-services",
            "compose.yaml",
            "lib/env.sh",
            "lib/host_embedding.sh",
            "lib/lifecycle.sh",
            "retry-guard/sitecustomize.py",
            "versions.env",
        ):
            target = bundle / relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative_path, target)

        checkout = directory / "mem0"
        (checkout / ".git").mkdir(parents=True)
        ports = _available_ports(4)
        overrides = {
            "OAMB_PROVIDER_PROJECT": "oamb-providers-model-profile-test",
            "OAMB_MEM0_SOURCE_CHECKOUT": str(checkout),
            "OAMB_HINDSIGHT_PORT": str(ports[0]),
            "OAMB_MEM0_PORT": str(ports[1]),
            "OAMB_MEM0_INSPECTOR_PORT": str(ports[2]),
            "OAMB_OPENVIKING_PORT": str(ports[3]),
            **dict.fromkeys(PROVIDER_MODEL_VARIABLES, TARGET_PROVIDER_MODEL),
            **model_overrides,
        }
        lines: list[str] = []
        for line in (bundle / ".env.example").read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                lines.append(line)
                continue
            name, value = line.split("=", 1)
            if name in overrides:
                value = overrides[name]
            elif value.startswith("change-me"):
                value = f"test-value-{name.lower()}"
            lines.append(f"{name}={value}")
        env_file = bundle / ".env"
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        env_file.chmod(0o600)

        fake_bin = directory / "bin"
        fake_bin.mkdir()
        _write_executable(fake_bin / "docker", "#!/bin/sh\nexit 0\n")
        _write_executable(
            fake_bin / "git",
            "#!/bin/sh\n"
            'case " $* " in\n'
            f"  *' rev-parse '*) printf '%s\\n' '{MEM0_COMMIT}' ;;\n"
            "esac\n"
            "exit 0\n",
        )
        return subprocess.run(
            [str(bundle / "bin" / "provider-services"), "doctor"],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"},
        )

    def test_doctor_accepts_the_target_provider_model_profile(self) -> None:
        result = self._run_doctor({})

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("doctor: PASS", result.stdout)

    def test_doctor_rejects_each_provider_model_outside_the_target_profile(self) -> None:
        for variable in PROVIDER_MODEL_VARIABLES:
            with self.subTest(variable=variable):
                result = self._run_doctor({variable: "deepseek-v4-pro"})

                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    f"{variable} must be {TARGET_PROVIDER_MODEL}",
                    result.stderr,
                )


if __name__ == "__main__":
    unittest.main()
