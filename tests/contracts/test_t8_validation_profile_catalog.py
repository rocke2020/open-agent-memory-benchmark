from __future__ import annotations

from collections import Counter
from typing import Any, cast

import pytest

from oamb.artifacts.validation.catalog import (
    _build_catalog_registry_for_test,
    _validate_catalog_profile_for_test,
    production_catalog_registry,
    validate_catalog_profile,
)
from oamb.artifacts.validation.profiles import (
    ClosedProfileDefinition,
    ReportAudience,
    ReportKind,
    exact_report_export_profile,
    t8_profile_catalog,
    validation_profile_catalog,
)
from oamb.artifacts.validation.registry import RuleRegistry
from oamb.contracts.states import ValidationDisposition

TARGET_HASH = "a" * 64

EXPECTED_PROFILE_IDS = (
    "oamb-t8-workload-lme6-v1",
    "oamb-t8-workload-mab5-v1",
    "oamb-t8-workload-lme30-v1",
    "oamb-t8-workload-mab65-v1",
    "oamb-t8-adapter-hindsight-rest-v1",
    "oamb-t8-adapter-mem0-rest-v1",
    "oamb-t8-adapter-mem0-sdk-v1",
    "oamb-t8-adapter-openviking-rest-v1",
    "oamb-t8-accounting-native-v1",
    "oamb-t8-comparison-paired-native-v1",
    "oamb-t8-t10-phase-gate-v1",
    "oamb-t8-export-run-public-v1",
    "oamb-t8-export-run-local-v1",
    "oamb-t8-export-comparison-public-v1",
    "oamb-t8-export-comparison-local-v1",
    "oamb-t8-export-evaluation-public-v1",
    "oamb-t8-export-evaluation-local-v1",
    "oamb-t8-export-release-public-v1",
    "oamb-t8-export-release-local-v1",
    "oamb-t8-export-phase-acceptance-public-v1",
    "oamb-t8-export-phase-acceptance-local-v1",
)

LME_COMMON_RULE_IDS = (
    "workload.lme.source-manifest.v1",
    "workload.lme.prompt-answer-parser.v1",
    "workload.lme.judge-metric.v1",
)
MAB_COMMON_RULE_IDS = (
    "workload.mab.source-manifest.v1",
    "workload.mab.prompt-answer-parser.v1",
    "workload.mab.metric-strata.v1",
)
EXPORT_RULE_IDS = (
    "report.export.payload-closure.v1",
    "report.export.canonical-bindings.v1",
    "report.export.safe-html.v1",
    "report.export.audience-scan.v1",
    "report.export.size-budget.v1",
)
EXPECTED_RULE_IDS_BY_PROFILE = {
    "oamb-t8-workload-lme6-v1": (
        *LME_COMMON_RULE_IDS,
        "workload.lme6.selector-membership.v1",
        "workload.lme6.denominator.v1",
    ),
    "oamb-t8-workload-mab5-v1": (
        *MAB_COMMON_RULE_IDS,
        "workload.mab5.selector-grouping.v1",
        "workload.mab5.smoke-isolation.v1",
        "workload.mab5.no-aggregate.v1",
    ),
    "oamb-t8-workload-lme30-v1": (
        *LME_COMMON_RULE_IDS,
        "workload.lme30.selector-coverage.v1",
        "workload.lme30.denominator.v1",
    ),
    "oamb-t8-workload-mab65-v1": (
        *MAB_COMMON_RULE_IDS,
        "workload.mab65.selector-grouping.v1",
        "workload.mab65.complete-reducer.v1",
        "workload.mab65.capability-index.v1",
    ),
    "oamb-t8-adapter-hindsight-rest-v1": (
        "adapter.hindsight.runtime-profile.v1",
        "adapter.hindsight.scope-dispatch.v1",
        "adapter.hindsight.projection-readiness.v1",
        "adapter.hindsight.retrieval-mutation.v1",
    ),
    "oamb-t8-adapter-mem0-rest-v1": (
        "adapter.mem0.rest.runtime-profile.v1",
        "adapter.mem0.rest.unsupported-verdict.v1",
        "adapter.mem0.rest.zero-dispatch.v1",
    ),
    "oamb-t8-adapter-mem0-sdk-v1": (
        "adapter.mem0.sdk.runtime-profile.v1",
        "adapter.mem0.sdk.unsupported-verdict.v1",
        "adapter.mem0.sdk.zero-dispatch.v1",
        "adapter.mem0.sdk.comparison-ineligible.v1",
    ),
    "oamb-t8-adapter-openviking-rest-v1": (
        "adapter.openviking.runtime-auth-scope.v1",
        "adapter.openviking.dispatch-projection.v1",
        "adapter.openviking.retrieval-order.v1",
        "adapter.openviking.query-mutation.v1",
    ),
    "oamb-t8-accounting-native-v1": (
        "accounting.attempt-coverage.v1",
        "accounting.token-meter-coverage.v1",
        "accounting.resource-cost-ownership.v1",
        "accounting.completeness-claims.v1",
    ),
    "oamb-t8-comparison-paired-native-v1": (
        "comparison.control-equality.v1",
        "comparison.metric-pairing.v1",
        "comparison.cost-basis.v1",
        "comparison.claim-suppression.v1",
    ),
    "oamb-t8-t10-phase-gate-v1": (
        "phase.schema-identity.v1",
        "phase.root-coverage.v1",
        "phase.case-coverage.v1",
        "phase.derivation-export.v1",
        "phase.review-order.v1",
        "phase.human-approval.v1",
    ),
    **{
        profile_id: EXPORT_RULE_IDS
        for profile_id in EXPECTED_PROFILE_IDS
        if profile_id.startswith("oamb-t8-export-")
    },
}


def _counting_registry(
    definition: ClosedProfileDefinition,
    calls: Counter[str],
    *,
    omitted_rule_ids: frozenset[str] = frozenset(),
    disabled_rule_ids: frozenset[str] = frozenset(),
    implementation_versions: dict[str, int] | None = None,
) -> RuleRegistry:
    implementations = {}
    for rule_id, _version in definition.rule_inventory:
        if rule_id in omitted_rule_ids:
            continue

        def evaluate(_target: object, *, current_rule_id: str = rule_id) -> tuple[()]:
            calls[current_rule_id] += 1
            return ()

        implementations[rule_id] = evaluate
    return _build_catalog_registry_for_test(
        definition.profile.profile_id,
        implementations,
        disabled_rule_ids=disabled_rule_ids,
        implementation_versions=implementation_versions,
    )


def test_t8_catalog_is_exact_and_contains_no_t11_profile() -> None:
    catalog = t8_profile_catalog()

    assert tuple(catalog) == EXPECTED_PROFILE_IDS
    for profile_id, definition in catalog.items():
        assert definition.profile.profile_id == profile_id
        assert (
            tuple(
                (item.rule_id, item.minimum_version) for item in definition.profile.required_rules
            )
            == definition.rule_inventory
        )
        assert (
            tuple(rule_id for rule_id, _version in definition.rule_inventory)
            == (EXPECTED_RULE_IDS_BY_PROFILE[profile_id])
        )
        assert all(version == 1 for _rule_id, version in definition.rule_inventory)
        assert len({rule_id for rule_id, _version in definition.rule_inventory}) == len(
            definition.rule_inventory
        )

    serialized = repr(tuple(catalog.values())).lower()
    assert "phase_2" not in serialized
    assert "repeatability" not in serialized
    assert "unique_cases=95" not in serialized
    assert "system_results=190" not in serialized
    assert "pairs=22" not in serialized
    mem0_rest = catalog["oamb-t8-adapter-mem0-rest-v1"].profile.applicability
    mem0_sdk = catalog["oamb-t8-adapter-mem0-sdk-v1"].profile.applicability
    assert "transport=rest_api" in mem0_rest
    assert "support=unsupported" in mem0_rest
    assert "transport=python_sdk" in mem0_sdk
    assert "support=unsupported" in mem0_sdk
    assert "comparison_eligible=false" in mem0_sdk


def test_current_catalog_extends_frozen_t8_with_t9_and_t10_profiles() -> None:
    t8_catalog = t8_profile_catalog()
    current_catalog = validation_profile_catalog()

    assert tuple(current_catalog) == (
        *tuple(t8_catalog),
        "oamb-t9-external-amb-historical-v1",
        "oamb-t10-adapter-mem0-rest-blackbox-v1",
    )
    external = current_catalog["oamb-t9-external-amb-historical-v1"]
    assert tuple(rule.rule_id for rule in external.profile.required_rules) == (
        "external.provenance.v1",
        "external.transformation.v1",
        "external.case-aggregate.v1",
        "external.compatibility.v1",
        "external.limitations.v1",
    )
    assert "repeatability" not in repr(tuple(current_catalog.values())).lower()
    mem0_blackbox = current_catalog["oamb-t10-adapter-mem0-rest-blackbox-v1"]
    assert tuple(rule.rule_id for rule in mem0_blackbox.profile.required_rules) == (
        "adapter.mem0.rest.runtime-profile.v2",
        "adapter.mem0.rest.scope-dispatch-projection.v1",
        "adapter.mem0.rest.retrieval-order-scope.v1",
        "adapter.mem0.rest.query-mutation.v1",
    )


def test_each_t8_profile_registers_and_executes_every_required_rule_once() -> None:
    for definition in t8_profile_catalog().values():
        calls: Counter[str] = Counter()
        registry = _counting_registry(definition, calls)

        result = _validate_catalog_profile_for_test(
            definition.profile.profile_id,
            object(),
            target_hash=TARGET_HASH,
            registry=registry,
        )

        expected_rule_ids = tuple(rule_id for rule_id, _version in definition.rule_inventory)
        assert result.disposition == ValidationDisposition.VALIDATED, result.issues
        assert result.required_rule_ids == expected_rule_ids
        assert result.executed_rule_ids == expected_rule_ids
        assert result.missing_rule_ids == ()
        assert calls == Counter({rule_id: 1 for rule_id in expected_rule_ids})


def test_production_catalog_registers_every_rule_and_rejects_an_untyped_target() -> None:
    for definition in t8_profile_catalog().values():
        registry = production_catalog_registry(definition.profile.profile_id)
        expected_rule_ids = tuple(rule_id for rule_id, _version in definition.rule_inventory)

        result = validate_catalog_profile(definition.profile.profile_id, object())

        assert registry.rule_ids == tuple(sorted(expected_rule_ids))
        assert result.executed_rule_ids == expected_rule_ids
        assert result.disposition == ValidationDisposition.INVALID
        assert result.failed_rule_ids == expected_rule_ids


def test_post_render_export_profiles_are_exact_per_kind_and_audience() -> None:
    expected: dict[tuple[ReportKind, ReportAudience], str] = {
        ("run", "public"): "oamb-t8-export-run-public-v1",
        ("run", "local"): "oamb-t8-export-run-local-v1",
        ("comparison", "public"): "oamb-t8-export-comparison-public-v1",
        ("comparison", "local"): "oamb-t8-export-comparison-local-v1",
        ("release", "public"): "oamb-t8-export-release-public-v1",
        ("release", "local"): "oamb-t8-export-release-local-v1",
        ("phase_acceptance", "public"): ("oamb-t8-export-phase-acceptance-public-v1"),
        ("phase_acceptance", "local"): "oamb-t8-export-phase-acceptance-local-v1",
    }

    for (report_kind, audience), profile_id in expected.items():
        profile = exact_report_export_profile(
            report_kind=report_kind,
            audience=audience,
        )
        assert profile.profile_id == profile_id
        assert profile.applicability == (
            f"report_kind={report_kind}",
            f"audience={audience}",
        )


def test_post_render_export_profile_rejects_unknown_selector_values() -> None:
    with pytest.raises(ValueError, match="unknown report kind or audience"):
        exact_report_export_profile(
            report_kind=cast(Any, "diagnostic"),
            audience="public",
        )
    with pytest.raises(ValueError, match="unknown report kind or audience"):
        exact_report_export_profile(
            report_kind="run",
            audience=cast(Any, "restricted"),
        )


def test_missing_disabled_and_version_incompatible_rules_are_invalid() -> None:
    definition = t8_profile_catalog()["oamb-t8-workload-lme6-v1"]
    final_rule_id, final_version = definition.rule_inventory[-1]

    missing_calls: Counter[str] = Counter()
    missing_registry = _counting_registry(
        definition,
        missing_calls,
        omitted_rule_ids=frozenset({final_rule_id}),
    )
    missing = _validate_catalog_profile_for_test(
        definition.profile.profile_id,
        object(),
        target_hash=TARGET_HASH,
        registry=missing_registry,
    )

    disabled_calls: Counter[str] = Counter()
    disabled_registry = _counting_registry(
        definition,
        disabled_calls,
        disabled_rule_ids=frozenset({final_rule_id}),
    )
    disabled = _validate_catalog_profile_for_test(
        definition.profile.profile_id,
        object(),
        target_hash=TARGET_HASH,
        registry=disabled_registry,
    )

    old_version_calls: Counter[str] = Counter()
    old_version_registry = _counting_registry(
        definition,
        old_version_calls,
        implementation_versions={final_rule_id: final_version - 1},
    )
    old_version = _validate_catalog_profile_for_test(
        definition.profile.profile_id,
        object(),
        target_hash=TARGET_HASH,
        registry=old_version_registry,
    )

    for result in (missing, disabled, old_version):
        assert result.disposition == ValidationDisposition.INVALID
        assert result.missing_rule_ids == (final_rule_id,)
        assert "validation.rule-availability.v1" in result.failed_rule_ids
    assert missing_calls[final_rule_id] == 0
    assert disabled_calls[final_rule_id] == 0
    assert old_version_calls[final_rule_id] == 0


def test_duplicate_registration_is_retained_as_an_invalid_catalog_state() -> None:
    definition = t8_profile_catalog()["oamb-t8-accounting-native-v1"]
    calls: Counter[str] = Counter()
    registry = _counting_registry(definition, calls)
    duplicate_rule = registry.get(definition.rule_inventory[0][0])

    with pytest.raises(ValueError, match="duplicate validation rule"):
        registry.register(duplicate_rule)

    result = _validate_catalog_profile_for_test(
        definition.profile.profile_id,
        object(),
        target_hash=TARGET_HASH,
        registry=registry,
    )

    assert result.disposition == ValidationDisposition.INVALID
    assert result.executed_rule_ids == ()
    assert result.failed_rule_ids == ("validation.rule-duplication.v1",)
    assert {issue.code for issue in result.issues} == {"duplicate-required-rule"}
    assert calls == Counter()


def test_reordered_profile_inventory_is_invalid_before_any_rule_executes() -> None:
    definition = t8_profile_catalog()["oamb-t8-comparison-paired-native-v1"]
    calls: Counter[str] = Counter()
    registry = _counting_registry(definition, calls)
    reordered = definition.profile.model_copy(
        update={"required_rules": tuple(reversed(definition.profile.required_rules))}
    )

    result = _validate_catalog_profile_for_test(
        definition.profile.profile_id,
        object(),
        target_hash=TARGET_HASH,
        registry=registry,
        profile=reordered,
    )

    assert result.disposition == ValidationDisposition.INVALID
    assert result.executed_rule_ids == ()
    assert result.failed_rule_ids == ("validation.profile-inventory.v1",)
    assert calls == Counter()
