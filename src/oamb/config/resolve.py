"""Resolve selected configuration edges without constructing unselected components."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .load import EnvironmentReference

FactoryValue = TypeVar("FactoryValue")


class ConfigurationResolutionError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RoleSelection:
    role: str
    selected: bool
    credential_reference: EnvironmentReference | None


@dataclass(frozen=True, slots=True)
class _FactoryRegistration(Generic[FactoryValue]):
    metadata: Mapping[str, object]
    factory: Callable[[], FactoryValue]


class LazyFactoryRegistry(Generic[FactoryValue]):
    def __init__(self) -> None:
        self._registrations: dict[str, _FactoryRegistration[FactoryValue]] = {}

    def register(
        self,
        identifier: str,
        *,
        metadata: Mapping[str, object],
        factory: Callable[[], FactoryValue],
    ) -> None:
        if not identifier or identifier in self._registrations:
            raise ConfigurationResolutionError(f"duplicate or empty registry ID: {identifier!r}")
        self._registrations[identifier] = _FactoryRegistration(dict(metadata), factory)

    def metadata(self, identifier: str) -> Mapping[str, object]:
        try:
            registration = self._registrations[identifier]
        except KeyError as exc:
            raise ConfigurationResolutionError(f"unknown registry ID: {identifier}") from exc
        return dict(registration.metadata)

    def create_selected(self, identifier: str) -> FactoryValue:
        try:
            registration = self._registrations[identifier]
        except KeyError as exc:
            raise ConfigurationResolutionError(f"unknown registry ID: {identifier}") from exc
        return registration.factory()


def validate_selected_environment(
    roles: tuple[RoleSelection, ...],
    environment: Mapping[str, str],
) -> tuple[EnvironmentReference, ...]:
    references: list[EnvironmentReference] = []
    for role in roles:
        if not role.selected or role.credential_reference is None:
            continue
        reference = role.credential_reference
        try:
            value: Any = environment[reference.name]
        except KeyError as exc:
            raise ConfigurationResolutionError(
                f"selected role {role.role!r} requires environment variable {reference.name}"
            ) from exc
        if not isinstance(value, str) or not value:
            raise ConfigurationResolutionError(
                f"selected role {role.role!r} requires non-empty environment variable "
                f"{reference.name}"
            )
        references.append(reference)
    return tuple(references)
