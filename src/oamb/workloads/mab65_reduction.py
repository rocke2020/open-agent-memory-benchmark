"""Complete-only exact-rational reduction for the MAB-65 secondary index."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from oamb.contracts.ids import (
    canonical_sha256,
    ingestion_payload_hash,
    ingestion_plan_id,
)
from oamb.contracts.specifications import CaseManifest
from oamb.workloads.memoryagentbench import (
    MAB65_CASE_MANIFEST_HASH,
    MAB65_CR_32K_PLAN_ID,
    MAB65_SELECTED_CASE_DIGEST,
    MAB65_SELECTED_PLAN_DIGEST,
)

MAB65_INDEX_ID = "mab65-capability-balanced-index-v1"
_COMPONENT_ORDER = ("ar", "icl", "recsys", "lru", "cr_sf")
_EXPECTED_CASE_COUNTS = {"ar": 15, "icl": 10, "recsys": 10, "lru": 15, "cr_sf": 15}
_EXPECTED_PLAN_COUNTS = {"ar": 5, "icl": 5, "recsys": 1, "lru": 10, "cr_sf": 4}
_EXPECTED_PLAN_CASE_COUNTS = {
    "ar": (3, 3, 3, 3, 3),
    "icl": (2, 2, 2, 2, 2),
    "recsys": (10,),
    "lru": (1, 1, 1, 1, 1, 2, 2, 2, 2, 2),
    "cr_sf": (3, 4, 4, 4),
}
_EXPECTED_METRIC_BY_COMPONENT = {
    "ar": "mab-substring-em-v1",
    "icl": "mab-exact-v1",
    "recsys": "mab-redial-recall-at-5-v1",
    "lru": "mab-exact-v1",
    "cr_sf": "mab-substring-em-v1",
}


@dataclass(frozen=True, slots=True)
class MabPlanScore:
    plan_manifest_entry_id: str
    component: str
    case_count: int
    score: Fraction


@dataclass(frozen=True, slots=True)
class MabComponentScore:
    component: str
    plan_count: int
    case_count: int
    score: Fraction


@dataclass(frozen=True, slots=True)
class MabCapabilityContribution:
    capability: str
    score: Fraction
    weight: Fraction
    weighted_contribution: Fraction


@dataclass(frozen=True, slots=True)
class Mab65Reduction:
    reducer_id: str
    available: bool
    unavailable_reason: str | None
    plans: tuple[MabPlanScore, ...]
    components: tuple[MabComponentScore, ...]
    ttl_score: Fraction | None
    capabilities: tuple[MabCapabilityContribution, ...]
    index_value: Fraction | None


def _unavailable(reason: str) -> Mab65Reduction:
    return Mab65Reduction(
        reducer_id=MAB65_INDEX_ID,
        available=False,
        unavailable_reason=reason,
        plans=(),
        components=(),
        ttl_score=None,
        capabilities=(),
        index_value=None,
    )


def _inventory(bundle: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        {case.entry.case_manifest_entry_id: case for case in bundle.cases},
        {plan.manifest.plan_manifest_entry_id: plan for plan in bundle.plans},
    )


def _case_manifest_is_exact(bundle: Any) -> bool:
    try:
        validated = CaseManifest.model_validate(bundle.case_manifest.model_dump(mode="python"))
    except (AttributeError, TypeError, ValueError):
        return False
    return validated.manifest_hash == MAB65_CASE_MANIFEST_HASH


def _source_units_are_exact(plan: Any) -> bool:
    manifest = plan.manifest
    runtime_plan = plan.runtime_plan
    member_ids = manifest.ordered_member_context_manifest_entry_ids
    source_units = runtime_plan.ordered_source_units
    actual_payload_hashes: list[str] = []
    payload_parts: list[bytes] = []
    if not member_ids:
        return False
    for expected_ordinal, source in enumerate(source_units, start=1):
        if not isinstance(source.payload_bytes, bytes):
            return False
        actual_payload_sha256 = hashlib.sha256(source.payload_bytes).hexdigest()
        if (
            source.ordinal_1_indexed != expected_ordinal
            or source.context_manifest_entry_id != member_ids[0]
            or source.payload_sha256 != actual_payload_sha256
            or source.source_unit_id
            != canonical_sha256(
                [runtime_plan.ingestion_plan_id, expected_ordinal, actual_payload_sha256]
            )
        ):
            return False
        actual_payload_hashes.append(actual_payload_sha256)
        payload_parts.append(source.payload_bytes)
    ordered_payload_hashes = tuple(actual_payload_hashes)
    actual_ingestion_payload_hash = ingestion_payload_hash(ordered_payload_hashes)
    expected_ingestion_plan_id = ingestion_plan_id(
        manifest.plan_manifest_entry_id,
        actual_ingestion_payload_hash,
    )
    return bool(
        ordered_payload_hashes == manifest.ordered_source_unit_bytes_sha256
        and actual_ingestion_payload_hash == manifest.ingestion_payload_hash
        and manifest.ingestion_plan_id == expected_ingestion_plan_id
        and runtime_plan.ingestion_plan_id == expected_ingestion_plan_id
        and hashlib.sha256(b"".join(payload_parts)).hexdigest()
        == runtime_plan.shared_context_sha256
    )


def _topology_is_exact(bundle: Any) -> bool:
    manifest = bundle.case_manifest
    if (
        len(manifest.logical_contexts) != 29
        or len(manifest.ingestion_plans) != 25
        or len(manifest.cases) != 65
        or len(bundle.cases) != 65
        or len(bundle.plans) != 25
    ):
        return False
    cases_by_id, _plans_by_id = _inventory(bundle)
    contexts_by_id = {
        context.context_manifest_entry_id: context for context in manifest.logical_contexts
    }
    if len(contexts_by_id) != 29:
        return False
    ordered_case_ids = tuple(case.entry.case_manifest_entry_id for case in bundle.cases)
    ordered_plan_ids = tuple(plan.manifest.plan_manifest_entry_id for plan in bundle.plans)
    if (
        tuple(manifest.cases) != tuple(case.entry for case in bundle.cases)
        or tuple(manifest.ingestion_plans) != tuple(plan.manifest for plan in bundle.plans)
        or tuple(case.case_manifest_entry_id for case in manifest.cases) != ordered_case_ids
        or tuple(plan.plan_manifest_entry_id for plan in manifest.ingestion_plans)
        != ordered_plan_ids
    ):
        return False
    component_case_counts = {component: 0 for component in _COMPONENT_ORDER}
    component_plan_vectors: dict[str, list[int]] = {component: [] for component in _COMPONENT_ORDER}
    flattened_case_ids: list[str] = []
    for case in bundle.cases:
        if case.component not in component_case_counts:
            return False
        component_case_counts[case.component] += 1
    for plan in bundle.plans:
        if plan.component not in component_plan_vectors:
            return False
        case_ids = plan.manifest.ordered_case_manifest_entry_ids
        member_ids = plan.manifest.ordered_member_context_manifest_entry_ids
        runtime_plan = plan.runtime_plan
        if not case_ids or any(
            case_id not in cases_by_id
            or cases_by_id[case_id].component != plan.component
            or cases_by_id[case_id].entry.context_manifest_entry_id not in member_ids
            for case_id in case_ids
        ):
            return False
        if (
            any(member_id not in contexts_by_id for member_id in member_ids)
            or runtime_plan.ingestion_plan_id != plan.manifest.ingestion_plan_id
            or runtime_plan.ordered_member_context_manifest_entry_ids != member_ids
            or runtime_plan.ordered_case_manifest_entry_ids != case_ids
            or runtime_plan.intended_source_count != len(runtime_plan.ordered_source_units)
            or runtime_plan.intended_source_count < 1
            or not _source_units_are_exact(plan)
            or runtime_plan.shared_context_sha256
            != contexts_by_id[member_ids[0]].context_bytes_sha256
        ):
            return False
        flattened_case_ids.extend(case_ids)
        component_plan_vectors[plan.component].append(len(case_ids))
        member_count = len(member_ids)
        if plan.component == "cr_sf":
            if member_count != 2 or len(plan.member_labels) != 2:
                return False
            if (
                len({contexts_by_id[member_id].context_bytes_sha256 for member_id in member_ids})
                != 1
            ):
                return False
            if not (
                "factconsolidation_sh_" in plan.member_labels[0]
                and "factconsolidation_mh_" in plan.member_labels[1]
            ):
                return False
            counts_by_member = tuple(
                sum(
                    cases_by_id[case_id].entry.context_manifest_entry_id == member_id
                    for case_id in case_ids
                )
                for member_id in member_ids
            )
            expected_quota = (
                (2, 1) if plan.manifest.plan_manifest_entry_id == MAB65_CR_32K_PLAN_ID else (2, 2)
            )
            if counts_by_member != expected_quota:
                return False
        elif member_count != 1:
            return False
    return (
        len(flattened_case_ids) == len(ordered_case_ids)
        and len(set(flattened_case_ids)) == len(flattened_case_ids)
        and set(flattened_case_ids) == set(ordered_case_ids)
        and component_case_counts == _EXPECTED_CASE_COUNTS
        and all(
            len(component_plan_vectors[component]) == _EXPECTED_PLAN_COUNTS[component]
            and tuple(sorted(component_plan_vectors[component]))
            == _EXPECTED_PLAN_CASE_COUNTS[component]
            for component in _COMPONENT_ORDER
        )
    )


def _metric_mapping_is_exact(bundle: Any) -> bool:
    return all(
        case.case_plan.metric_id == _EXPECTED_METRIC_BY_COMPONENT.get(case.component)
        for case in bundle.cases
    )


def _manifest_identity_is_exact(bundle: Any) -> bool:
    case_digest = canonical_sha256(
        [
            "oamb-mab65-selected-case-ids-v1",
            tuple(case.entry.case_manifest_entry_id for case in bundle.cases),
        ]
    )
    plan_digest = canonical_sha256(
        [
            "oamb-mab65-selected-plan-manifest-ids-v1",
            tuple(plan.manifest.plan_manifest_entry_id for plan in bundle.plans),
        ]
    )
    return bool(
        case_digest == bundle.selected_case_digest == MAB65_SELECTED_CASE_DIGEST
        and plan_digest == bundle.selected_plan_digest == MAB65_SELECTED_PLAN_DIGEST
    )


def reduce_mab65(
    bundle: Any,
    *,
    case_metrics: Mapping[str, Fraction],
    ready_plan_manifest_ids: frozenset[str],
    capsule_valid: bool,
    query_state_unchanged: bool,
    expected_catalog_sha256: str,
    expected_unicode_fingerprint: str,
    expected_interaction_fingerprint: str,
    observed_interaction_fingerprint: str,
    comparison_controls_closed: bool,
) -> Mab65Reduction:
    """Return the secondary index only from the exact complete MAB-65 shape."""

    if bundle.case_manifest.manifest_id != "mab65-v1":
        return _unavailable("manifest_not_mab65")
    if not _manifest_identity_is_exact(bundle):
        return _unavailable("manifest_identity")
    if not _case_manifest_is_exact(bundle):
        return _unavailable("manifest_topology")
    if bundle.entity_catalog_sha256 != expected_catalog_sha256:
        return _unavailable("catalog_fingerprint")
    if bundle.unicode_fingerprint != expected_unicode_fingerprint:
        return _unavailable("unicode_fingerprint")
    if not comparison_controls_closed:
        return _unavailable("comparison_controls")
    if observed_interaction_fingerprint != expected_interaction_fingerprint:
        return _unavailable("interaction_fingerprint")
    if not capsule_valid:
        return _unavailable("capsule_invalid")
    if not query_state_unchanged:
        return _unavailable("query_state_changed")
    if not _topology_is_exact(bundle):
        return _unavailable("manifest_topology")
    if not _metric_mapping_is_exact(bundle):
        return _unavailable("metric_component_mapping")
    cases_by_id, plans_by_id = _inventory(bundle)
    expected_case_ids = frozenset(cases_by_id)
    if frozenset(case_metrics) != expected_case_ids or any(
        not isinstance(value, Fraction) or value < 0 or value > 1 for value in case_metrics.values()
    ):
        return _unavailable("case_metric_inventory")
    if ready_plan_manifest_ids != frozenset(plans_by_id):
        return _unavailable("ready_plan_inventory")

    plan_scores: list[MabPlanScore] = []
    for plan in bundle.plans:
        case_ids = plan.manifest.ordered_case_manifest_entry_ids
        score = sum((case_metrics[case_id] for case_id in case_ids), Fraction()) / len(case_ids)
        plan_scores.append(
            MabPlanScore(
                plan_manifest_entry_id=plan.manifest.plan_manifest_entry_id,
                component=plan.component,
                case_count=len(case_ids),
                score=score,
            )
        )
    components: list[MabComponentScore] = []
    for component in _COMPONENT_ORDER:
        component_plans = tuple(plan for plan in plan_scores if plan.component == component)
        components.append(
            MabComponentScore(
                component=component,
                plan_count=len(component_plans),
                case_count=sum(plan.case_count for plan in component_plans),
                score=sum((plan.score for plan in component_plans), Fraction())
                / len(component_plans),
            )
        )
    scores = {component.component: component.score for component in components}
    ttl = (scores["icl"] + scores["recsys"]) / 2
    capabilities = (
        MabCapabilityContribution("ar", scores["ar"], Fraction(1, 4), scores["ar"] / 4),
        MabCapabilityContribution("ttl", ttl, Fraction(1, 4), ttl / 4),
        MabCapabilityContribution("lru", scores["lru"], Fraction(1, 4), scores["lru"] / 4),
        MabCapabilityContribution("cr_sf", scores["cr_sf"], Fraction(1, 4), scores["cr_sf"] / 4),
    )
    index_value = 100 * sum(
        (capability.weighted_contribution for capability in capabilities), Fraction()
    )
    return Mab65Reduction(
        reducer_id=MAB65_INDEX_ID,
        available=True,
        unavailable_reason=None,
        plans=tuple(plan_scores),
        components=tuple(components),
        ttl_score=ttl,
        capabilities=capabilities,
        index_value=index_value,
    )
