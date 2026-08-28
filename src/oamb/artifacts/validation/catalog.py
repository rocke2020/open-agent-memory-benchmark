"""Fail-closed execution of the frozen T8 validation profile catalog."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

from oamb.artifacts.atomic import read_regular_file
from oamb.artifacts.validation.engine import validate_closed_profile
from oamb.artifacts.validation.profiles import validation_profile_catalog
from oamb.artifacts.validation.registry import RuleFunction, RuleRegistry, ValidationRule
from oamb.contracts.evidence import CapsuleManifest, ValidationResult
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import ValidationProfile


def _build_catalog_registry_for_test(
    profile_id: str,
    implementations: Mapping[str, RuleFunction],
    *,
    disabled_rule_ids: frozenset[str] = frozenset(),
    implementation_versions: Mapping[str, int] | None = None,
) -> RuleRegistry:
    definition = validation_profile_catalog()[profile_id]
    version_overrides = implementation_versions or {}
    registry = RuleRegistry()
    for rule_id, minimum_version in definition.rule_inventory:
        evaluate = implementations.get(rule_id)
        if evaluate is None:
            continue
        registry.register(
            ValidationRule(
                rule_id=rule_id,
                version=version_overrides.get(rule_id, minimum_version),
                evaluate=evaluate,
                enabled=rule_id not in disabled_rule_ids,
            )
        )
    return registry


def _validate_catalog_profile_for_test(
    profile_id: str,
    target: Any,
    *,
    target_hash: str,
    registry: RuleRegistry,
    profile: ValidationProfile | None = None,
) -> ValidationResult:
    definition = validation_profile_catalog()[profile_id]
    return validate_closed_profile(
        target,
        target_hash=target_hash,
        profile=profile or definition.profile,
        expected_stage=definition.profile.stage,
        expected_inventory=definition.rule_inventory,
        registry=registry,
    )


def validate_catalog_profile(profile_id: str, target: Any) -> ValidationResult:
    """Run a catalog profile with repository-owned rules and target-derived identity."""

    return _validate_catalog_profile_for_test(
        profile_id,
        target,
        target_hash=_production_target_hash(target),
        registry=production_catalog_registry(profile_id),
    )


def production_catalog_registry(profile_id: str) -> RuleRegistry:
    """Build the closed registry from repository-owned semantic implementations."""

    from oamb.artifacts.validation.adapter import ADAPTER_RULES
    from oamb.artifacts.validation.phase import PHASE_GATE_RULES
    from oamb.artifacts.validation.reduction import REDUCTION_RULES
    from oamb.artifacts.validation.report_export import REPORT_EXPORT_RULES
    from oamb.artifacts.validation.workload import WORKLOAD_RULES
    from oamb.external_evidence.validation import EXTERNAL_HISTORICAL_RULES

    definition = validation_profile_catalog()[profile_id]
    available = {
        rule.rule_id: rule
        for rule in (
            *WORKLOAD_RULES,
            *ADAPTER_RULES,
            *REDUCTION_RULES,
            *PHASE_GATE_RULES,
            *REPORT_EXPORT_RULES,
            *EXTERNAL_HISTORICAL_RULES,
        )
    }
    registry = RuleRegistry()
    for rule_id, _minimum_version in definition.rule_inventory:
        rule = available.get(rule_id)
        if rule is not None:
            registry.register(rule)
    return registry


def _production_target_hash(target: Any) -> str:
    from oamb.artifacts.validation.adapter import Mem0AdapterValidationInput
    from oamb.artifacts.validation.phase import (
        PhaseGateValidationInput,
        _phase_gate_target_hash,
    )
    from oamb.artifacts.validation.reduction import (
        AccountingValidationInput,
        ComparisonValidationInput,
    )
    from oamb.artifacts.validation.report_export import ReportExportInput
    from oamb.contracts.external import ExternalHistoricalEvidence
    from oamb.memory_systems.mem0.profiles import mem0_zero_dispatch_audit_binding
    from oamb.workloads.longmemeval import LongMemEvalBundle
    from oamb.workloads.memoryagentbench import MabManifestBundle

    if isinstance(target, Path):
        manifest_path = target / "capsule-manifest.json"
        try:
            content = read_regular_file(manifest_path)
            return CapsuleManifest.model_validate_json(content).source_manifest_hash
        except Exception:
            try:
                content = read_regular_file(manifest_path)
            except Exception:
                content = b""
            return hashlib.sha256(content).hexdigest()
    if isinstance(target, LongMemEvalBundle):
        return canonical_sha256(
            [
                "oamb-longmemeval-validation-target-v2",
                target.dataset_manifest,
                target.case_manifest,
                _canonical_target_value(target.selected_rows),
                _canonical_target_value(target.ingestion_plans),
                _canonical_target_value(target.case_plans),
            ]
        )
    if isinstance(target, MabManifestBundle):
        return canonical_sha256(
            [
                "oamb-memoryagentbench-validation-target-v3",
                target.dataset_manifest,
                target.case_manifest,
                _canonical_target_value(target.movie_catalog),
                target.entity_catalog_sha256,
                target.unicode_fingerprint,
                target.selected_case_digest,
                target.selected_plan_digest,
                _canonical_target_value(target.cases),
                _canonical_target_value(target.plans),
            ]
        )
    if isinstance(target, Mem0AdapterValidationInput):
        return canonical_sha256(
            [
                "oamb-mem0-zero-dispatch-validation-target-v1",
                mem0_zero_dispatch_audit_binding(target.audit),
            ]
        )
    if isinstance(target, AccountingValidationInput):
        return canonical_sha256(
            [
                "oamb-accounting-validation-target-v1",
                target.token_records,
                target.resource_records,
                target.cost_records,
                tuple(
                    (item.record_id, item.owner_kind, item.indexing_view)
                    for item in target.record_views
                ),
                tuple(sorted(target.expected_plan_ids)),
                tuple(sorted(target.expected_attempt_ids)),
                target.attempts,
                target.require_attempt_accounting_closure,
            ]
        )
    if isinstance(target, ComparisonValidationInput):
        return canonical_sha256(
            [
                "oamb-comparison-validation-target-v1",
                target.spec,
                target.left,
                target.right,
                target.paired_metric_deltas,
                target.aggregate_metric_delta,
                target.cost_delta,
                target.report,
            ]
        )
    if isinstance(target, PhaseGateValidationInput):
        return _phase_gate_target_hash(target)
    if isinstance(target, ReportExportInput):
        return canonical_sha256(
            [
                "oamb-report-publication-payload-v1",
                tuple(
                    (path, hashlib.sha256(content).hexdigest())
                    for path, content in sorted(target.payloads.items())
                ),
                tuple(sorted(target.scan_paths)),
            ]
        )
    if isinstance(target, ExternalHistoricalEvidence):
        return canonical_sha256(target)
    return canonical_sha256(
        [
            "oamb-unsupported-validation-target-v1",
            f"{type(target).__module__}.{type(target).__qualname__}",
        ]
    )


def _canonical_target_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return ("bytes-v1", len(value), hashlib.sha256(value).hexdigest())
    if isinstance(value, float):
        return ("float-hex-v1", value.hex())
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical_target_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _canonical_target_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_canonical_target_value(item) for item in value)
    if isinstance(value, list):
        return [_canonical_target_value(item) for item in value]
    return value


__all__ = [
    "production_catalog_registry",
    "validate_catalog_profile",
]
