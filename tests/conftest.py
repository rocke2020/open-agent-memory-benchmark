from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest


@pytest.fixture
def install_wheel_in_isolated_environment() -> Callable[[Path, Path], Path]:
    def install(wheel: Path, environment: Path) -> Path:
        create_environment = subprocess.run(
            ["uv", "venv", "--python", sys.executable, str(environment)],
            check=False,
            capture_output=True,
            text=True,
        )
        if create_environment.returncode != 0:
            raise AssertionError(create_environment.stderr)
        installed_python = environment / "bin" / "python"
        installation = subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(installed_python),
                str(wheel),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if installation.returncode != 0:
            raise AssertionError(installation.stderr)
        return installed_python

    return install
