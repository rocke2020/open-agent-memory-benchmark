import json
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("state,expected", [("ready", 0), ("shared", 1), ("busy", 1)])
def test_beam_provisioning_requires_ready_owner(tmp_path: Path, state: str, expected: int) -> None:
    docker = tmp_path / "docker"
    report = {
        "busy": state == "busy",
        "owners": [{"tenant_id": "t", "user_id": "u", "state": state}],
    }
    docker.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in *'/opt/oam-eval/config.yaml'*) ;; *) exit 3 ;; esac\n"
        "printf '%s\\n' '" + json.dumps(report) + "'\n"
    )
    docker.chmod(0o755)
    result = subprocess.run(
        [
            "python3",
            str(ROOT / "provider-services/oam/provision-beam-index.py"),
            "container",
            "t",
            "u",
        ],
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
    )
    assert (result.returncode == 0) == (expected == 0), result.stderr
