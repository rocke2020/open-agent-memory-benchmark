"""Closed evidence profiles for structural and provider-service roots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from oamb.contracts.specifications import (
    ValidationProfile,
    ValidationRuleRequirement,
    ValidationStage,
)

from .core import STRUCTURAL_RULES

T5_FAKE_EVIDENCE_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("fake.schema.v1", 1),
    ("fake.manifest-closure.v1", 1),
    ("fake.parentage.v1", 1),
    ("fake.execution.v1", 1),
    ("fake.raw-reference.v1", 1),
    ("fake.index-accounting.v1", 1),
)
T5_FAKE_EXPORT_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("fake.export.payload-closure.v1", 1),
    ("fake.export.safe-html.v1", 1),
    ("fake.export.report-binding.v1", 1),
)

T4_PROVIDER_SERVICE_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("t4-provider-schema", 1),
    ("t4-provider-manifest-closure", 1),
    ("t4-provider-path-safety", 1),
    ("t4-provider-occurrence-terminal", 1),
    ("t4-provider-parent-isolation", 1),
    ("t4-provider-budget-closure", 1),
    ("t4-provider-attempt-ordering", 1),
    ("t4-provider-unknown-outcome", 1),
    ("t4-provider-runtime-binding", 1),
    ("t4-provider-gate-separation", 1),
    ("t4-provider-control-redaction", 1),
)

T8_T10_PHASE_GATE_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("phase.schema-identity.v1", 1),
    ("phase.root-coverage.v1", 1),
    ("phase.case-coverage.v1", 1),
    ("phase.derivation-export.v1", 1),
    ("phase.review-order.v1", 1),
    ("phase.human-approval.v1", 1),
)

T8_REPORT_EXPORT_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("report.export.payload-closure.v1", 1),
    ("report.export.canonical-bindings.v1", 1),
    ("report.export.safe-html.v1", 1),
    ("report.export.audience-scan.v1", 1),
    ("report.export.size-budget.v1", 1),
)

T8_WORKLOAD_RULE_INVENTORIES: dict[str, tuple[tuple[str, int], ...]] = {
    "lme6-live-smoke-v1": (
        ("workload.lme.source-manifest.v1", 1),
        ("workload.lme.prompt-answer-parser.v1", 1),
        ("workload.lme.judge-metric.v1", 1),
        ("workload.lme6.selector-membership.v1", 1),
        ("workload.lme6.denominator.v1", 1),
    ),
    "mab5-live-smoke-v1": (
        ("workload.mab.source-manifest.v1", 1),
        ("workload.mab.prompt-answer-parser.v1", 1),
        ("workload.mab.metric-strata.v1", 1),
        ("workload.mab5.selector-grouping.v1", 1),
        ("workload.mab5.smoke-isolation.v1", 1),
        ("workload.mab5.no-aggregate.v1", 1),
    ),
    "lme30-native-smoke-plus-v1": (
        ("workload.lme.source-manifest.v1", 1),
        ("workload.lme.prompt-answer-parser.v1", 1),
        ("workload.lme.judge-metric.v1", 1),
        ("workload.lme30.selector-coverage.v1", 1),
        ("workload.lme30.denominator.v1", 1),
    ),
    "mab65-v1": (
        ("workload.mab.source-manifest.v1", 1),
        ("workload.mab.prompt-answer-parser.v1", 1),
        ("workload.mab.metric-strata.v1", 1),
        ("workload.mab65.selector-grouping.v1", 1),
        ("workload.mab65.complete-reducer.v1", 1),
        ("workload.mab65.capability-index.v1", 1),
    ),
}

T8_ADAPTER_RULE_INVENTORIES: dict[str, tuple[tuple[str, int], ...]] = {
    "hindsight-rest-v1": (
        ("adapter.hindsight.runtime-profile.v1", 1),
        ("adapter.hindsight.scope-dispatch.v1", 1),
        ("adapter.hindsight.projection-readiness.v1", 1),
        ("adapter.hindsight.retrieval-mutation.v1", 1),
    ),
    "mem0-rest-v1": (
        ("adapter.mem0.rest.runtime-profile.v1", 1),
        ("adapter.mem0.rest.unsupported-verdict.v1", 1),
        ("adapter.mem0.rest.zero-dispatch.v1", 1),
    ),
    "mem0-sdk-v1": (
        ("adapter.mem0.sdk.runtime-profile.v1", 1),
        ("adapter.mem0.sdk.unsupported-verdict.v1", 1),
        ("adapter.mem0.sdk.zero-dispatch.v1", 1),
        ("adapter.mem0.sdk.comparison-ineligible.v1", 1),
    ),
    "openviking-rest-v1": (
        ("adapter.openviking.runtime-auth-scope.v1", 1),
        ("adapter.openviking.dispatch-projection.v1", 1),
        ("adapter.openviking.retrieval-order.v1", 1),
        ("adapter.openviking.query-mutation.v1", 1),
    ),
}

T10_MEM0_REST_BLACKBOX_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("adapter.mem0.rest.runtime-profile.v2", 2),
    ("adapter.mem0.rest.scope-dispatch-projection.v1", 1),
    ("adapter.mem0.rest.retrieval-order-scope.v1", 1),
    ("adapter.mem0.rest.query-mutation.v1", 1),
)

T8_ACCOUNTING_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("accounting.attempt-coverage.v1", 1),
    ("accounting.token-meter-coverage.v1", 1),
    ("accounting.resource-cost-ownership.v1", 1),
    ("accounting.completeness-claims.v1", 1),
)

T8_COMPARISON_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("comparison.control-equality.v1", 1),
    ("comparison.metric-pairing.v1", 1),
    ("comparison.cost-basis.v1", 1),
    ("comparison.claim-suppression.v1", 1),
)

T9_EXTERNAL_EVIDENCE_RULE_INVENTORY: tuple[tuple[str, int], ...] = (
    ("external.provenance.v1", 1),
    ("external.transformation.v1", 1),
    ("external.case-aggregate.v1", 1),
    ("external.compatibility.v1", 1),
    ("external.limitations.v1", 1),
)

ReportKind = Literal["run", "comparison", "release", "phase_acceptance"]
ReportAudience = Literal["public", "local"]
REPORT_KINDS: tuple[ReportKind, ...] = (
    "run",
    "comparison",
    "release",
    "phase_acceptance",
)
REPORT_AUDIENCES: tuple[ReportAudience, ...] = ("public", "local")


@dataclass(frozen=True, slots=True)
class ClosedProfileDefinition:
    profile: ValidationProfile
    rule_inventory: tuple[tuple[str, int], ...]


def _closed_profile(
    *,
    profile_id: str,
    stage: ValidationStage,
    rule_inventory: tuple[tuple[str, int], ...],
    applicability: tuple[str, ...],
) -> ClosedProfileDefinition:
    profile = ValidationProfile.create(
        profile_id=profile_id,
        stage=stage,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=minimum_version)
            for rule_id, minimum_version in rule_inventory
        ),
        applicability=applicability,
    )
    return ClosedProfileDefinition(profile=profile, rule_inventory=rule_inventory)


def structural_evidence_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id="oamb-t3-structural-evidence-v1",
        stage=ValidationStage.EVIDENCE,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule.rule_id, minimum_version=rule.version)
            for rule in STRUCTURAL_RULES
        ),
        applicability=("t3-contract-kernel",),
    )


def provider_service_evidence_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id="oamb-t4-provider-service-evidence-v1",
        stage=ValidationStage.EVIDENCE,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=minimum_version)
            for rule_id, minimum_version in T4_PROVIDER_SERVICE_RULE_INVENTORY
        ),
        applicability=(
            "source_kind=provider_service",
            "operation=model_readiness",
        ),
    )


def fake_evidence_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id="oamb-t5-fake-evidence-v1",
        stage=ValidationStage.EVIDENCE,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=minimum_version)
            for rule_id, minimum_version in T5_FAKE_EVIDENCE_RULE_INVENTORY
        ),
        applicability=("workload=oamb-fake-vertical-v1",),
    )


def fake_export_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id="oamb-t5-fake-export-v1",
        stage=ValidationStage.EXPORT,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=minimum_version)
            for rule_id, minimum_version in T5_FAKE_EXPORT_RULE_INVENTORY
        ),
        applicability=("report_kind=run", "audience=public|local"),
    )


def t10_phase_gate_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id="oamb-t8-t10-phase-gate-v1",
        stage=ValidationStage.EVIDENCE,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=minimum_version)
            for rule_id, minimum_version in T8_T10_PHASE_GATE_RULE_INVENTORY
        ),
        applicability=(
            "phase=phase_1_smoke_acceptance",
            "capsules=4",
            "unique_cases=11",
            "system_results=22",
        ),
    )


def exact_report_export_profile(
    *,
    report_kind: ReportKind = "run",
    audience: ReportAudience = "public",
) -> ValidationProfile:
    if report_kind not in REPORT_KINDS or audience not in REPORT_AUDIENCES:
        raise ValueError("unknown report kind or audience")
    normalized_kind = report_kind.replace("_", "-")
    return _closed_profile(
        profile_id=f"oamb-t8-export-{normalized_kind}-{audience}-v1",
        stage=ValidationStage.EXPORT,
        rule_inventory=T8_REPORT_EXPORT_RULE_INVENTORY,
        applicability=(f"report_kind={report_kind}", f"audience={audience}"),
    ).profile


def report_export_profile() -> ValidationProfile:
    return ValidationProfile.create(
        profile_id="oamb-t8-report-export-v1",
        stage=ValidationStage.EXPORT,
        required_rules=tuple(
            ValidationRuleRequirement(rule_id=rule_id, minimum_version=minimum_version)
            for rule_id, minimum_version in T8_REPORT_EXPORT_RULE_INVENTORY
        ),
        applicability=(
            "report_kind=run|comparison|release|phase_acceptance",
            "audience=public|local",
        ),
    )


def t8_profile_catalog() -> dict[str, ClosedProfileDefinition]:
    definitions = (
        _closed_profile(
            profile_id="oamb-t8-workload-lme6-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_WORKLOAD_RULE_INVENTORIES["lme6-live-smoke-v1"],
            applicability=("component=workload", "workload=lme6-live-smoke-v1"),
        ),
        _closed_profile(
            profile_id="oamb-t8-workload-mab5-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_WORKLOAD_RULE_INVENTORIES["mab5-live-smoke-v1"],
            applicability=("component=workload", "workload=mab5-live-smoke-v1"),
        ),
        _closed_profile(
            profile_id="oamb-t8-workload-lme30-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_WORKLOAD_RULE_INVENTORIES["lme30-native-smoke-plus-v1"],
            applicability=(
                "component=workload",
                "workload=lme30-native-smoke-plus-v1",
            ),
        ),
        _closed_profile(
            profile_id="oamb-t8-workload-mab65-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_WORKLOAD_RULE_INVENTORIES["mab65-v1"],
            applicability=("component=workload", "workload=mab65-v1"),
        ),
        _closed_profile(
            profile_id="oamb-t8-adapter-hindsight-rest-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_ADAPTER_RULE_INVENTORIES["hindsight-rest-v1"],
            applicability=(
                "component=adapter",
                "adapter_profile=hindsight-rest-v1",
            ),
        ),
        _closed_profile(
            profile_id="oamb-t8-adapter-mem0-rest-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_ADAPTER_RULE_INVENTORIES["mem0-rest-v1"],
            applicability=(
                "component=adapter",
                "adapter_profile=mem0-rest-v1",
                "transport=rest_api",
                "support=unsupported",
            ),
        ),
        _closed_profile(
            profile_id="oamb-t8-adapter-mem0-sdk-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_ADAPTER_RULE_INVENTORIES["mem0-sdk-v1"],
            applicability=(
                "component=adapter",
                "adapter_profile=mem0-sdk-v1",
                "transport=python_sdk",
                "support=unsupported",
                "comparison_eligible=false",
            ),
        ),
        _closed_profile(
            profile_id="oamb-t8-adapter-openviking-rest-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_ADAPTER_RULE_INVENTORIES["openviking-rest-v1"],
            applicability=(
                "component=adapter",
                "adapter_profile=openviking-rest-v1",
            ),
        ),
        _closed_profile(
            profile_id="oamb-t8-accounting-native-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_ACCOUNTING_RULE_INVENTORY,
            applicability=("component=accounting", "accounting=native-v1"),
        ),
        _closed_profile(
            profile_id="oamb-t8-comparison-paired-native-v1",
            stage=ValidationStage.EVIDENCE,
            rule_inventory=T8_COMPARISON_RULE_INVENTORY,
            applicability=(
                "component=comparison",
                "comparison=paired-native-v1",
            ),
        ),
        ClosedProfileDefinition(
            profile=t10_phase_gate_profile(),
            rule_inventory=T8_T10_PHASE_GATE_RULE_INVENTORY,
        ),
        *(
            ClosedProfileDefinition(
                profile=exact_report_export_profile(
                    report_kind=report_kind,
                    audience=audience,
                ),
                rule_inventory=T8_REPORT_EXPORT_RULE_INVENTORY,
            )
            for report_kind in REPORT_KINDS
            for audience in REPORT_AUDIENCES
        ),
    )
    return {definition.profile.profile_id: definition for definition in definitions}


def validation_profile_catalog() -> dict[str, ClosedProfileDefinition]:
    """Return the current catalog without changing the frozen T8 inventory."""

    catalog = t8_profile_catalog()
    external = _closed_profile(
        profile_id="oamb-t9-external-amb-historical-v1",
        stage=ValidationStage.EVIDENCE,
        rule_inventory=T9_EXTERNAL_EVIDENCE_RULE_INVENTORY,
        applicability=(
            "origin=external_amb_generated",
            "producer_protocol=amb-longmemeval-rag",
            "compatibility=unknown",
        ),
    )
    mem0_blackbox = _closed_profile(
        profile_id="oamb-t10-adapter-mem0-rest-blackbox-v1",
        stage=ValidationStage.EVIDENCE,
        rule_inventory=T10_MEM0_REST_BLACKBOX_RULE_INVENTORY,
        applicability=(
            "component=adapter",
            "adapter_profile=mem0-rest-v1",
            "transport=rest_api",
            "evaluation=black_box_quality",
        ),
    )
    return {
        **catalog,
        external.profile.profile_id: external,
        mem0_blackbox.profile.profile_id: mem0_blackbox,
    }
