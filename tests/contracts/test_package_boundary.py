from __future__ import annotations

import importlib
import subprocess
import sys
from types import ModuleType

import pytest
from typer.testing import CliRunner


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
    assert "schema" in result.output


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


def test_checked_in_schemas_are_available_as_package_data() -> None:
    schema = require("oamb.contracts.schema")

    names = schema.packaged_schema_names()

    assert "protocol_spec.v1.schema.json" in names
    assert "validation_result.v1.schema.json" in names
