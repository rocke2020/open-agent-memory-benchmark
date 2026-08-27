"""Build deterministic fingerprints from configuration references, never secret values."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from oamb.contracts.ids import canonical_sha256

from .load import EnvironmentReference

_SENSITIVE_WORD_SEQUENCES = (
    ("api", "key"),
    ("access", "token"),
    ("refresh", "token"),
    ("client", "secret"),
)
_SENSITIVE_WORDS = frozenset({"authorization", "credential", "password", "secret"})
_SAFE_TOKEN_FOLLOWING_WORDS = frozenset(
    {"budget", "budgets", "count", "counts", "limit", "limits", "type", "usage", "usages"}
)
_CAMEL_WORD_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_NON_WORD_CHARACTERS = re.compile(r"[^A-Za-z0-9]+")


def normalized_configuration_key(key: str) -> str:
    separated = _CAMEL_WORD_BOUNDARY.sub("_", key)
    return _NON_WORD_CHARACTERS.sub("_", separated).strip("_").lower()


def is_sensitive_key(key: str) -> bool:
    normalized = normalized_configuration_key(key)
    words = tuple(part for part in normalized.split("_") if part)
    if any(word in _SENSITIVE_WORDS for word in words):
        return True
    if any(
        words[index : index + len(sequence)] == sequence
        for sequence in _SENSITIVE_WORD_SEQUENCES
        for index in range(len(words) - len(sequence) + 1)
    ):
        return True
    return any(
        word == "token"
        and (index + 1 == len(words) or words[index + 1] not in _SAFE_TOKEN_FOLLOWING_WORDS)
        for index, word in enumerate(words)
    )


def _redacted_value(value: object, *, key: str | None = None) -> object:
    if isinstance(value, EnvironmentReference):
        return {"env": value.name}
    if key is not None and is_sensitive_key(key):
        raise ValueError("configuration fingerprints accept secret references, not secret values")
    if isinstance(value, Mapping):
        return {
            str(child_key): _redacted_value(item, key=str(child_key))
            for child_key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redacted_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    raise TypeError(f"unsupported redacted configuration value: {type(value).__name__}")


def redacted_configuration_fingerprint(document: Mapping[str, object]) -> str:
    return canonical_sha256(["oamb-redacted-configuration-v1", _redacted_value(document)])
