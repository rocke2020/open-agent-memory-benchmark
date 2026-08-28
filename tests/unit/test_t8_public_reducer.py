from __future__ import annotations

from fractions import Fraction

import pytest

from oamb.contracts.reporting import MabPlanEvidenceBinding
from oamb.reporting.public import build_mab65_report_reduction
from oamb.workloads.mab65_reduction import (
    MAB65_INDEX_ID,
    Mab65Reduction,
    MabCapabilityContribution,
    MabComponentScore,
    MabPlanScore,
)


def _complete_reduction() -> Mab65Reduction:
    plan_counts = {"ar": 5, "icl": 5, "recsys": 1, "lru": 10, "cr_sf": 4}
    case_count_vectors = {
        "ar": (3, 3, 3, 3, 3),
        "icl": (2, 2, 2, 2, 2),
        "recsys": (10,),
        "lru": (1, 1, 1, 1, 1, 2, 2, 2, 2, 2),
        "cr_sf": (3, 4, 4, 4),
    }
    plans = tuple(
        MabPlanScore(
            plan_manifest_entry_id=f"{component}-{ordinal}",
            component=component,
            case_count=case_count,
            score=Fraction(1, 2),
        )
        for component in ("ar", "icl", "recsys", "lru", "cr_sf")
        for ordinal, case_count in enumerate(case_count_vectors[component], start=1)
    )
    components = tuple(
        MabComponentScore(
            component=component,
            plan_count=plan_counts[component],
            case_count=sum(plan.case_count for plan in plans if plan.component == component),
            score=Fraction(1, 2),
        )
        for component in ("ar", "icl", "recsys", "lru", "cr_sf")
    )
    capabilities = tuple(
        MabCapabilityContribution(
            capability=capability,
            score=Fraction(1, 2),
            weight=Fraction(1, 4),
            weighted_contribution=Fraction(1, 8),
        )
        for capability in ("ar", "ttl", "lru", "cr_sf")
    )
    return Mab65Reduction(
        reducer_id=MAB65_INDEX_ID,
        available=True,
        unavailable_reason=None,
        plans=plans,
        components=components,
        ttl_score=Fraction(1, 2),
        capabilities=capabilities,
        index_value=Fraction(50, 1),
    )


def _plan_evidence_bindings(
    reduction: Mab65Reduction,
) -> tuple[MabPlanEvidenceBinding, ...]:
    metric_ids = {
        "ar": "mab-substring-em-v1",
        "icl": "mab-exact-v1",
        "recsys": "mab-redial-recall-at-5-v1",
        "lru": "mab-exact-v1",
        "cr_sf": "mab-substring-em-v1",
    }
    bindings: list[MabPlanEvidenceBinding] = []
    logical_ordinal = 1
    case_ordinal = 1
    for plan_ordinal, plan in enumerate(reduction.plans, start=1):
        member_count = 2 if plan.component == "cr_sf" else 1
        logical_ids = tuple(
            f"{ordinal:064x}" for ordinal in range(logical_ordinal, logical_ordinal + member_count)
        )
        logical_ordinal += member_count
        case_ids = tuple(
            f"{ordinal + 1_000:064x}"
            for ordinal in range(case_ordinal, case_ordinal + plan.case_count)
        )
        case_ordinal += plan.case_count
        member_labels = (
            (
                f"factconsolidation_sh_{plan_ordinal}",
                f"factconsolidation_mh_{plan_ordinal}",
            )
            if plan.component == "cr_sf"
            else (f"{plan.component}_{plan_ordinal}",)
        )
        bindings.append(
            MabPlanEvidenceBinding(
                plan_manifest_entry_id=plan.plan_manifest_entry_id,
                ingestion_plan_id=f"{plan_ordinal + 100:064x}",
                ingestion_occurrence_id=f"{plan_ordinal + 200:064x}",
                ordered_logical_context_ids=logical_ids,
                member_labels=member_labels,
                ordered_case_occurrence_ids=case_ids,
                metric_id=metric_ids[plan.component],
                resolution_evidence_references=(
                    (f"source/redial-resolution-{plan_ordinal}.json.gz",)
                    if plan.component == "recsys"
                    else ()
                ),
            )
        )
    return tuple(bindings)


def test_mab65_report_reduction_preserves_component_first_exact_waterfall() -> None:
    reduction = _complete_reduction()
    reduced = build_mab65_report_reduction(
        reduction,
        plan_evidence_bindings=_plan_evidence_bindings(reduction),
    )

    assert len(reduced.plans) == 25
    assert (
        len(
            {
                context_id
                for item in reduced.plans
                for context_id in item.evidence_binding.ordered_logical_context_ids
            }
        )
        == 29
    )
    assert (
        sum(len(item.evidence_binding.ordered_case_occurrence_ids) for item in reduced.plans) == 65
    )
    assert tuple(item.component for item in reduced.components) == (
        "ar",
        "icl",
        "recsys",
        "lru",
        "cr_sf",
    )
    assert tuple(item.capability for item in reduced.capabilities) == (
        "ar",
        "ttl",
        "lru",
        "cr_sf",
    )
    assert reduced.index_value is not None
    assert (reduced.index_value.numerator, reduced.index_value.denominator) == (50, 1)
    redial = next(item for item in reduced.plans if item.component == "recsys")
    assert redial.evidence_binding.metric_id == "mab-redial-recall-at-5-v1"
    assert redial.evidence_binding.resolution_evidence_references


def test_mab65_report_reduction_rejects_incomplete_plan_evidence_inventory() -> None:
    reduction = _complete_reduction()
    bindings = _plan_evidence_bindings(reduction)

    with pytest.raises(ValueError, match="plan evidence inventory"):
        build_mab65_report_reduction(
            reduction,
            plan_evidence_bindings=bindings[:-1],
        )


def test_mab65_unavailable_report_has_no_index_value() -> None:
    unavailable = Mab65Reduction(
        reducer_id=MAB65_INDEX_ID,
        available=False,
        unavailable_reason="case_metric_inventory",
        plans=(),
        components=(),
        ttl_score=None,
        capabilities=(),
        index_value=None,
    )

    reduced = build_mab65_report_reduction(unavailable)

    assert not reduced.available
    assert reduced.index_value is None
    assert reduced.unavailable_reason == "case_metric_inventory"
