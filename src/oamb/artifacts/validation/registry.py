"""Unique structural-rule registration with closed inventory lookup."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from oamb.contracts.evidence import ValidationIssue

RuleFunction = Callable[[Any], tuple["ValidationIssue", ...]]


@dataclass(frozen=True, slots=True)
class ValidationRule:
    rule_id: str
    version: int
    evaluate: RuleFunction


class DuplicateRuleError(ValueError):
    pass


class RuleRegistry:
    def __init__(self) -> None:
        self._rules: dict[str, ValidationRule] = {}

    def register(self, rule: ValidationRule) -> None:
        if rule.rule_id in self._rules:
            raise DuplicateRuleError(f"duplicate validation rule: {rule.rule_id}")
        self._rules[rule.rule_id] = rule

    def get(self, rule_id: str) -> ValidationRule:
        return self._rules[rule_id]

    def compatible(self, rule_id: str, minimum_version: int) -> ValidationRule | None:
        rule = self._rules.get(rule_id)
        if rule is None or rule.version < minimum_version:
            return None
        return rule

    @property
    def rule_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._rules))


def structural_registry(*, exclude: set[str] | None = None) -> RuleRegistry:
    from .core import STRUCTURAL_RULES

    omitted = exclude or set()
    registry = RuleRegistry()
    for rule in STRUCTURAL_RULES:
        if rule.rule_id not in omitted:
            registry.register(rule)
    return registry


def provider_service_registry(*, exclude: set[str] | None = None) -> RuleRegistry:
    from .core import PROVIDER_SERVICE_RULES

    omitted = exclude or set()
    registry = RuleRegistry()
    for rule in PROVIDER_SERVICE_RULES:
        if rule.rule_id not in omitted:
            registry.register(rule)
    return registry
