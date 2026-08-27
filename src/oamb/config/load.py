"""Load strict JSON configuration while preserving environment references."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")


class ConfigurationLoadError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class EnvironmentReference:
    name: str

    def __post_init__(self) -> None:
        if not _ENVIRONMENT_NAME.fullmatch(self.name):
            raise ConfigurationLoadError(f"invalid environment reference: {self.name!r}")


@dataclass(frozen=True, slots=True)
class LoadedConfiguration:
    document: MappingProxyType[str, Any]


def _freeze_configuration(value: object) -> object:
    if isinstance(value, dict):
        if set(value) == {"env"}:
            name = value["env"]
            if not isinstance(name, str):
                raise ConfigurationLoadError("environment reference name must be a string")
            return EnvironmentReference(name)
        return MappingProxyType(
            {
                key: _freeze_configuration(item)
                for key, item in value.items()
                if isinstance(key, str)
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_configuration(item) for item in value)
    if isinstance(value, str) and "${" in value:
        raise ConfigurationLoadError(
            'environment variables require the explicit {"env": "NAME"} form'
        )
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise ConfigurationLoadError(f"unsupported JSON configuration value: {type(value).__name__}")


def load_json_configuration(path: Path) -> LoadedConfiguration:
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationLoadError(f"cannot load JSON configuration: {path}") from exc
    if not isinstance(parsed, dict):
        raise ConfigurationLoadError("configuration root must be a JSON object")
    frozen = _freeze_configuration(parsed)
    if not isinstance(frozen, MappingProxyType):
        raise AssertionError("configuration object did not freeze as a mapping")
    return LoadedConfiguration(document=frozen)
