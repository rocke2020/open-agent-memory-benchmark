from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from typer.testing import CliRunner

from oamb.public_boundary import scan_public_paths


def require(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} is not implemented", pytrace=False)


def test_core_import_and_cli_help_do_not_load_optional_provider_dependencies() -> None:
    command = (
        "import oamb, sys; "
        "from oamb.cli import app; "
        "forbidden=('mem0','hindsight','openviking','qdrant_client','psycopg'); "
        "loaded=sorted(name for name in sys.modules if name.split('.')[0] in forbidden); "
        "assert not loaded, loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    cli = require("oamb.cli")
    result = CliRunner().invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "schema" not in result.output.lower()


def test_contract_ports_do_not_import_implementation_packages() -> None:
    command = (
        "import oamb.contracts.ports, sys; "
        "forbidden=('oamb.runtime','oamb.memory_systems','oamb.model_clients'); "
        "loaded=sorted(name for name in sys.modules "
        "if any(name == item or name.startswith(item + '.') for item in forbidden)); "
        "assert not loaded, loaded"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_artifact_validation_does_not_import_reporting_implementation() -> None:
    command = (
        "import oamb.artifacts.validation.fake, sys; "
        "forbidden='oamb.reporting'; "
        "loaded=sorted(name for name in sys.modules "
        "if name == forbidden or name.startswith(forbidden + '.')); "
        "assert not loaded, loaded"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_source_checkout_runs_without_installing_a_versioned_distribution(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    for name in ("src", "prompt-packs", "configs"):
        shutil.copytree(
            repository_root / name,
            checkout / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    for name in ("pyproject.toml", "uv.lock", "README.md", "precheck.sh", "run.sh"):
        shutil.copy2(repository_root / name, checkout / name)
    if (repository_root / ".python-version").is_file():
        shutil.copy2(repository_root / ".python-version", checkout / ".python-version")
    environment = os.environ | {"PYTHONPATH": str(checkout / "src")}
    sync = subprocess.run(
        ["uv", "sync", "--locked", "--offline", "--group", "download"],
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert sync.returncode == 0, sync.stdout + sync.stderr
    source_paths = tuple(path for path in (checkout / "src").rglob("*") if path.is_file())
    assert scan_public_paths(checkout, source_paths) == ()
    source_python = checkout / ".venv" / "bin" / "python"
    resource_probe = subprocess.run(
        [
            str(source_python),
            "-c",
            (
                "from importlib.metadata import distributions; "
                "assert not any(d.metadata['Name'] == 'open-agent-memory-benchmark' "
                "for d in distributions()), 'OAMB must run without distribution metadata'; "
                "from importlib import resources; "
                "from oamb.contracts.schema import CONTRACT_REGISTRY, parse_contract; "
                "assert not resources.files('oamb').joinpath('schemas').is_dir(); "
                'expected={("comparison_control_source_reference",1),'
                '("comparison_control_provenance_binding",1),'
                '("workload_execution_control_record",1),'
                '("controlled_embedding_comparison_projection",1),'
                '("runtime_measurement_control_record",1),'
                '("run_comparison_control_basis_record",1),'
                '("run_preflight_record",2),'
                '("comparison_control_snapshot",2)}; '
                "assert expected <= set(CONTRACT_REGISTRY); "
                'document={"schema_name":"comparison_control_source_reference",'
                '"schema_version":1,"record_kind":"run_spec",'
                '"referenced_schema_name":"run_spec",'
                '"referenced_schema_version":1,"record_id":"run",'
                '"record_sha256":"a"*64,"json_pointers":["/run_id"]}; '
                "assert type(parse_contract(document)).__name__ == "
                '"ComparisonControlSourceReference"'
            ),
        ],
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resource_probe.returncode == 0, resource_probe.stderr
    help_result = subprocess.run(
        [str(source_python), "-m", "oamb.cli", "--help"],
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "schema" not in help_result.stdout.lower()
    removed_command = subprocess.run(
        [str(source_python), "-m", "oamb.cli", "schema", "--help"],
        cwd=checkout,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert removed_command.returncode != 0
    for script in ("precheck.sh", "run.sh"):
        help_result = subprocess.run(
            [str(checkout / script), "--oam", "--help"],
            cwd=tmp_path,
            env=environment | {"PYTHONPATH": ""},
            check=False,
            capture_output=True,
            text=True,
        )
        assert help_result.returncode == 0, help_result.stdout + help_result.stderr
        assert "usage:" in help_result.stdout.lower()
