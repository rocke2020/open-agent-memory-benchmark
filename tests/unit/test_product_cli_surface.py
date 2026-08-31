from __future__ import annotations

import subprocess
import sys

from oamb.cli import app


def test_product_cli_has_no_phase_or_t10_specific_surface() -> None:
    command_names = {command.name for command in app.registered_commands}
    group_names = {group.name for group in app.registered_groups}

    assert "t10-run" not in command_names
    assert "phase" not in group_names


def test_core_cli_imports_no_phase_review_or_human_review_module() -> None:
    script = (
        "import json,sys,oamb.cli; "
        "prefixes=('oamb.phase_', 'oamb.reporting.review', "
        "'oamb.reporting.human_review', 'oamb.artifacts.validation.phase'); "
        "loaded=sorted(name for name in sys.modules "
        "if any(name == prefix or name.startswith(prefix) for prefix in prefixes)); "
        "print(json.dumps(loaded)); "
        "raise SystemExit(bool(loaded))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
