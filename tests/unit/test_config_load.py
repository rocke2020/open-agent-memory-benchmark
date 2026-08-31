from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path

import pytest


class RecordingEnvironment(Mapping[str, str]):
    def __init__(self, values: dict[str, str]) -> None:
        self._data = values
        self.read_names: list[str] = []

    def __getitem__(self, key: str) -> str:
        self.read_names.append(key)
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)


def test_json_loader_preserves_explicit_environment_references(tmp_path: Path) -> None:
    from oamb.config.load import EnvironmentReference, load_json_configuration

    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "roles": {
                    "answer": {
                        "selected": True,
                        "credential": {"env": "OAMB_ANSWER_API_KEY"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_json_configuration(path)

    assert loaded.document["roles"]["answer"]["credential"] == EnvironmentReference(
        "OAMB_ANSWER_API_KEY"
    )


def test_json_loader_rejects_implicit_environment_syntax(tmp_path: Path) -> None:
    from oamb.config.load import ConfigurationLoadError, load_json_configuration

    path = tmp_path / "run.json"
    path.write_text('{"credential":"${OAMB_ANSWER_API_KEY}"}', encoding="utf-8")

    with pytest.raises(ConfigurationLoadError, match="explicit.*env"):
        load_json_configuration(path)


def test_selected_environment_validation_does_not_read_unselected_roles() -> None:
    from oamb.config.load import EnvironmentReference
    from oamb.config.resolve import RoleSelection, validate_selected_environment

    environment = RecordingEnvironment(
        {
            "OAMB_ANSWER_API_KEY": "answer-secret",
            "OAMB_JUDGE_API_KEY": "must-not-be-read",
        }
    )
    roles = (
        RoleSelection(
            role="answer",
            selected=True,
            credential_reference=EnvironmentReference("OAMB_ANSWER_API_KEY"),
        ),
        RoleSelection(
            role="judge",
            selected=False,
            credential_reference=EnvironmentReference("OAMB_JUDGE_API_KEY"),
        ),
    )

    references = validate_selected_environment(roles, environment)

    assert references == (EnvironmentReference("OAMB_ANSWER_API_KEY"),)
    assert environment.read_names == ["OAMB_ANSWER_API_KEY"]


def test_selected_environment_validation_rejects_missing_secret_without_echoing_name_value() -> (
    None
):
    from oamb.config.load import EnvironmentReference
    from oamb.config.resolve import (
        ConfigurationResolutionError,
        RoleSelection,
        validate_selected_environment,
    )

    secret_value = "top-secret-value"
    environment = RecordingEnvironment({"UNRELATED": secret_value})
    roles = (
        RoleSelection(
            role="answer",
            selected=True,
            credential_reference=EnvironmentReference("OAMB_ANSWER_API_KEY"),
        ),
    )

    with pytest.raises(ConfigurationResolutionError) as failure:
        validate_selected_environment(roles, environment)

    assert "OAMB_ANSWER_API_KEY" in str(failure.value)
    assert secret_value not in str(failure.value)


def test_lazy_registry_does_not_construct_unselected_factories() -> None:
    from oamb.config.resolve import LazyFactoryRegistry

    constructed: list[str] = []

    def create_fake() -> str:
        constructed.append("fake")
        return "fake-instance"

    def create_mem0() -> str:
        constructed.append("mem0-rest")
        return "mem0-instance"

    registry: LazyFactoryRegistry[str] = LazyFactoryRegistry()
    registry.register(
        "fake",
        metadata={"transport_kind": "fake"},
        factory=create_fake,
    )
    registry.register(
        "mem0-rest",
        metadata={"transport_kind": "rest_api"},
        factory=create_mem0,
    )

    assert registry.metadata("mem0-rest") == {"transport_kind": "rest_api"}
    assert constructed == []
    assert registry.create_selected("fake") == "fake-instance"
    assert constructed == ["fake"]


def test_redacted_fingerprint_uses_reference_name_and_never_secret_value() -> None:
    from oamb.config.load import EnvironmentReference
    from oamb.config.redact import redacted_configuration_fingerprint

    secret_value = "sk-do-not-hash-or-serialize"
    document = {
        "endpoint_class": "openai-compatible",
        "credential": EnvironmentReference("OAMB_ANSWER_API_KEY"),
        "secret_value": secret_value,
    }

    with pytest.raises(ValueError, match="secret values"):
        redacted_configuration_fingerprint(document)

    safe_document = {
        "endpoint_class": "openai-compatible",
        "credential": EnvironmentReference("OAMB_ANSWER_API_KEY"),
    }
    fingerprint = redacted_configuration_fingerprint(safe_document)

    assert len(fingerprint) == 64
    assert secret_value not in repr(safe_document)


@pytest.mark.parametrize(
    "sensitive_key",
    (
        "authorization",
        "apiKey",
        "api-key",
        "clientSecret",
        "accessToken",
        "clientSecretValue",
        "oauthAccessTokenValue",
        "myApiKeyValue",
        "nestedCredentialValue",
    ),
)
def test_redacted_fingerprint_rejects_literal_authorization_values(
    sensitive_key: str,
) -> None:
    from oamb.config.load import EnvironmentReference
    from oamb.config.redact import redacted_configuration_fingerprint

    with pytest.raises(ValueError, match="secret values"):
        redacted_configuration_fingerprint({sensitive_key: "Bearer plaintext-secret"})

    assert (
        len(
            redacted_configuration_fingerprint(
                {sensitive_key: EnvironmentReference("OAMB_PROVIDER_AUTHORIZATION")}
            )
        )
        == 64
    )


def test_redacted_fingerprint_keeps_nonsecret_tokenizer_identity() -> None:
    from oamb.config.redact import redacted_configuration_fingerprint

    fingerprint = redacted_configuration_fingerprint(
        {
            "input_token_count": 10,
            "inputTokenCountValue": 10,
            "providerTokenUsageVersion": "1",
            "token_usage": "supplier_response",
            "tokenizer": "o200k_base",
            "tokenizer_version": "0.14.0",
        }
    )

    assert len(fingerprint) == 64
