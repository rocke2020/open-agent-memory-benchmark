from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path


def test_installed_wheel_rebuilds_external_report_byte_identically(
    tmp_path: Path,
    install_wheel_in_isolated_environment: Callable[[Path, Path], Path],
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    distribution_root = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(distribution_root)],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    wheel = next(distribution_root.glob("*.whl"))
    environment = tmp_path / "installed"
    install_wheel_in_isolated_environment(
        wheel,
        environment,
    )
    installed_cli = environment / "bin" / "oamb"
    help_result = subprocess.run(
        [str(installed_cli), "external", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert {"import", "validate", "report"} <= set(help_result.stdout.split())

    validation_path = tmp_path / "external-validation.json"
    validation = subprocess.run(
        [
            str(installed_cli),
            "external",
            "validate",
            "--output",
            str(validation_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert validation.returncode == 0, validation.stderr
    assert validation_path.is_file()

    report_roots = (tmp_path / "report-one", tmp_path / "report-two")
    for report_root in report_roots:
        report = subprocess.run(
            [
                str(installed_cli),
                "external",
                "report",
                "--validation",
                str(validation_path),
                "--output-root",
                str(report_root),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert report.returncode == 0, report.stderr

    first_model = next(report_roots[0].glob("derivations/*/outputs/report-model.json"))
    second_model = next(report_roots[1].glob("derivations/*/outputs/report-model.json"))
    first_html = next(report_roots[0].glob("derivations/*/outputs/report.html"))
    second_html = next(report_roots[1].glob("derivations/*/outputs/report.html"))
    assert first_model.read_bytes() == second_model.read_bytes()
    assert first_html.read_bytes() == second_html.read_bytes()
    assert b'"correct_cases":448' in first_model.read_bytes()
    assert b'"amb_formatted_view_context_tokens_total":24812616' in first_model.read_bytes()
