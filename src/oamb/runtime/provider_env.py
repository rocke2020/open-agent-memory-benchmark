"""Strict inert-data loader for the T10 provider-services dotenv file."""

from __future__ import annotations

import re
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path

from oamb.artifacts.atomic import read_regular_file

_DOTENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UNSUPPORTED_VALUE_CHARACTERS = frozenset({'"', "$", "`", "\\", "#", "'"})
_MAX_PROVIDER_ENV_BYTES = 1024 * 1024

T10_PROVIDER_ENVIRONMENT_KEYS = frozenset(
    {
        "OAMB_HINDSIGHT_AUTHORIZATION",
        "OAMB_HINDSIGHT_BASE_URL",
        "OAMB_HINDSIGHT_LLM_API_KEY",
        "OAMB_HINDSIGHT_LLM_BASE_URL",
        "OAMB_HINDSIGHT_PORT",
        "OAMB_MEM0_ADMIN_API_KEY",
        "OAMB_MEM0_BASE_URL",
        "OAMB_MEM0_INSPECTOR_API_KEY",
        "OAMB_MEM0_INSPECTOR_BASE_URL",
        "OAMB_MEM0_INSPECTOR_PORT",
        "OAMB_MEM0_LLM_API_KEY",
        "OAMB_MEM0_LLM_BASE_URL",
        "OAMB_MEM0_PORT",
        "OAMB_EMBEDDING_API_KEY",
        "OAMB_EMBEDDING_BASE_URL",
        "OAMB_OPENVIKING_ACCOUNT_ID",
        "OAMB_OPENVIKING_ADMIN_USER_ID",
        "OAMB_OPENVIKING_BASE_URL",
        "OAMB_OPENVIKING_PORT",
        "OAMB_OPENVIKING_VLM_API_KEY",
        "OAMB_OPENVIKING_VLM_BASE_URL",
    }
)


class ProviderEnvironmentFileError(ValueError):
    """Provider dotenv bytes are not safe under the closed inert-data grammar."""


def load_t10_provider_environment(
    path: Path,
    *,
    expected_keys: frozenset[str] = T10_PROVIDER_ENVIRONMENT_KEYS,
) -> dict[str, str]:
    """Read unique dotenv assignments without shell evaluation or interpolation."""

    if not expected_keys or any(_DOTENV_KEY.fullmatch(key) is None for key in expected_keys):
        raise ValueError("expected provider environment keys must be valid and non-empty")
    resolved_path = path.resolve(strict=True)
    metadata = resolved_path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ProviderEnvironmentFileError("provider env path must be a regular file")
    payload = read_regular_file(resolved_path)
    after_read = resolved_path.lstat()
    if (metadata.st_dev, metadata.st_ino, metadata.st_mode, metadata.st_size) != (
        after_read.st_dev,
        after_read.st_ino,
        after_read.st_mode,
        after_read.st_size,
    ):
        raise ProviderEnvironmentFileError("provider env file changed while reading")
    if len(payload) > _MAX_PROVIDER_ENV_BYTES:
        raise ProviderEnvironmentFileError("provider env file is unexpectedly large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProviderEnvironmentFileError("provider env file is not UTF-8") from exc
    return _parse_t10_provider_environment(text.split("\n"), expected_keys=expected_keys)


def merge_t10_provider_environment(
    environment: Mapping[str, str],
    provider_env_file: Path | None,
) -> Mapping[str, str]:
    """Overlay only the T10-consumed provider keys on the caller environment."""

    if provider_env_file is None:
        return environment
    merged = dict(environment)
    merged.update(load_t10_provider_environment(provider_env_file))
    return merged


def _parse_t10_provider_environment(
    lines: Iterable[str],
    *,
    expected_keys: frozenset[str],
) -> dict[str, str]:
    seen: set[str] = set()
    selected: dict[str, str] = {}
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if "=" not in line:
            raise ProviderEnvironmentFileError("invalid dotenv line without =")
        key, value = line.split("=", 1)
        if _DOTENV_KEY.fullmatch(key) is None:
            raise ProviderEnvironmentFileError("invalid dotenv key")
        if key not in expected_keys:
            continue
        if key in seen:
            raise ProviderEnvironmentFileError("duplicate dotenv key")
        seen.add(key)
        if value != value.strip() or any(
            character in value for character in _UNSUPPORTED_VALUE_CHARACTERS
        ):
            raise ProviderEnvironmentFileError("unsupported dotenv value syntax")
        selected[key] = value
    return selected


__all__ = [
    "ProviderEnvironmentFileError",
    "T10_PROVIDER_ENVIRONMENT_KEYS",
    "load_t10_provider_environment",
    "merge_t10_provider_environment",
]
