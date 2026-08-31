from __future__ import annotations

import importlib
import subprocess
import sys
import tarfile
import zipfile
from collections.abc import Callable
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


def test_generated_schema_snapshots_are_absent_from_source_and_package_configuration() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    pyproject = (repository_root / "pyproject.toml").read_text(encoding="utf-8")

    assert not (repository_root / "schemas").exists()
    assert '"/schemas"' not in pyproject
    assert '"schemas" = "oamb/schemas"' not in pyproject


def test_built_and_installed_distributions_expose_no_generated_schema_surface(
    tmp_path: Path,
    install_wheel_in_isolated_environment: Callable[[Path, Path], Path],
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    distribution_root = tmp_path / "dist"
    build = subprocess.run(
        ["uv", "build", "--out-dir", str(distribution_root)],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stderr
    wheel = next(distribution_root.glob("*.whl"))
    source_distribution = next(distribution_root.glob("*.tar.gz"))
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = tuple(archive.namelist())
        wheel_root = tmp_path / "wheel"
        archive.extractall(wheel_root)
    with tarfile.open(source_distribution, "r:gz") as archive:
        source_names = tuple(archive.getnames())
        source_root = tmp_path / "source"
        archive.extractall(source_root, filter="data")
    for names in (wheel_names, source_names):
        assert not any(name.endswith(".schema.json") for name in names)
        assert not any("/schemas/" in f"/{name}/" for name in names)
    for distribution_root in (wheel_root, source_root):
        distribution_paths = tuple(
            path for path in distribution_root.rglob("*") if path.is_file() or path.is_symlink()
        )
        assert scan_public_paths(distribution_root, distribution_paths) == ()

    environment = tmp_path / "installed"
    installed_python = install_wheel_in_isolated_environment(
        wheel,
        environment,
    )
    resource_probe = subprocess.run(
        [
            str(installed_python),
            "-c",
            (
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
        check=False,
        capture_output=True,
        text=True,
    )
    assert resource_probe.returncode == 0, resource_probe.stderr
    installed_cli = environment / "bin" / "oamb"
    help_result = subprocess.run(
        [str(installed_cli), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "schema" not in help_result.stdout.lower()
    removed_command = subprocess.run(
        [str(installed_cli), "schema", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert removed_command.returncode != 0
