from __future__ import annotations

from collections.abc import Iterator, Mapping

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
