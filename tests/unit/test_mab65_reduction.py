from __future__ import annotations

import copy
import hashlib
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

import pytest

from oamb.contracts.ids import (
    canonical_sha256,
    case_manifest_entry_id,
    ingestion_payload_hash,
    ingestion_plan_id,
    plan_manifest_entry_id,
)
from oamb.contracts.specifications import (
    CaseManifestEntry,
    IngestionPlanManifest,
    LogicalContextManifestEntry,
    case_manifest_hash,
)

CATALOG_SHA256 = "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5"
UNICODE_FINGERPRINT = "1a846dffe314101b43cb68132979e8a1f052b7d2115532981d9866db1eeb352a"
INTERACTION_FINGERPRINT = "f" * 64

_PLAN_SHAPES = (
    *(("ar", 3, 1) for _ in range(5)),
    ("recsys", 10, 1),
    *(("icl", 2, 1) for _ in range(5)),
    *(("lru", count, 1) for count in (2, 2, 1, 2, 1, 1, 1, 1, 2, 2)),
    *(("cr_sf", count, 2) for count in (4, 3, 4, 4)),
)


class _MutableCaseManifest(SimpleNamespace):
    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        if mode != "python":
            raise ValueError("fixture case manifest supports only Python mode")
        return {
            "schema_name": "case_manifest",
            "schema_version": 1,
            "manifest_id": self.manifest_id,
            "manifest_hash": self.manifest_hash,
            "workload_id": self.workload_id,
            "logical_contexts": tuple(vars(item) for item in self.logical_contexts),
            "ingestion_plans": tuple(vars(item) for item in self.ingestion_plans),
            "cases": tuple(vars(item) for item in self.cases),
        }


def _generated_bundle(*, manifest_id: str = "mab65-v1") -> SimpleNamespace:
    cases = []
    plans = []
    context_entries: list[SimpleNamespace] = []
    case_ordinal = 0
    for plan_ordinal, (component, case_count, member_count) in enumerate(_PLAN_SHAPES, start=1):
        member_ids = tuple(
            canonical_sha256(["fixture-context", plan_ordinal, member_ordinal])
            for member_ordinal in range(1, member_count + 1)
        )
        source_payload = f"context-payload-{plan_ordinal}".encode()
        shared_context_sha256 = hashlib.sha256(source_payload).hexdigest()
        context_entries.extend(
            SimpleNamespace(
                **LogicalContextManifestEntry(
                    context_content_id=canonical_sha256(
                        ["fixture-context-content", plan_ordinal, member_ordinal]
                    ),
                    context_manifest_entry_id=member_id,
                    source_file_sha256=canonical_sha256(["fixture-source-file", plan_ordinal]),
                    source_row_number_1_indexed=member_ordinal,
                    context_bytes_sha256=shared_context_sha256,
                ).model_dump(mode="python")
            )
            for member_ordinal, member_id in enumerate(member_ids, start=1)
        )
        plan_case_ids = []
        for case_in_plan in range(1, case_count + 1):
            case_ordinal += 1
            context_id = (
                member_ids[0] if component != "cr_sf" or case_in_plan <= 2 else member_ids[1]
            )
            question_bytes = f"question-{case_ordinal}".encode()
            question_sha256 = hashlib.sha256(question_bytes).hexdigest()
            raw_question_id = f"raw-question-{case_ordinal}"
            case_id = case_manifest_entry_id(
                context_id,
                case_in_plan,
                question_sha256,
                raw_question_id,
            )
            reference_payload = f'["answer-{case_ordinal}"]'.encode()
            reference_sha256 = hashlib.sha256(reference_payload).hexdigest()
            entry = CaseManifestEntry(
                case_manifest_entry_id=case_id,
                context_manifest_entry_id=context_id,
                source_question_number_1_indexed=case_in_plan,
                question_bytes_sha256=question_sha256,
                raw_question_id=raw_question_id,
                answer_value_sha256=(
                    hashlib.sha256(f"answer-{case_ordinal}".encode()).hexdigest(),
                ),
            )
            plan_case_ids.append(case_id)
            cases.append(
                SimpleNamespace(
                    entry=SimpleNamespace(**entry.model_dump(mode="python")),
                    component=component,
                    case_plan=SimpleNamespace(
                        case_manifest_entry_id=case_id,
                        context_manifest_entry_id=context_id,
                        source_question_number_1_indexed=case_in_plan,
                        question_bytes=question_bytes,
                        reference_payload=reference_payload,
                        reference_payload_sha256=reference_sha256,
                        metric_id={
                            "ar": "mab-substring-em-v1",
                            "icl": "mab-exact-v1",
                            "recsys": "mab-redial-recall-at-5-v1",
                            "lru": "mab-exact-v1",
                            "cr_sf": "mab-substring-em-v1",
                        }[component],
                    ),
                )
            )
        plan_id = plan_manifest_entry_id("mab65-v1", member_ids)
        source_hashes = (hashlib.sha256(source_payload).hexdigest(),)
        payload_hash = ingestion_payload_hash(source_hashes)
        physical_plan_id = ingestion_plan_id(plan_id, payload_hash)
        plan_manifest = IngestionPlanManifest(
            plan_manifest_entry_id=plan_id,
            ingestion_payload_hash=payload_hash,
            ingestion_plan_id=physical_plan_id,
            workload_id="mab65-v1",
            ordered_case_manifest_entry_ids=tuple(plan_case_ids),
            ordered_member_context_manifest_entry_ids=member_ids,
            ordered_source_unit_bytes_sha256=source_hashes,
        )
        plans.append(
            SimpleNamespace(
                manifest=SimpleNamespace(**plan_manifest.model_dump(mode="python")),
                runtime_plan=SimpleNamespace(
                    ingestion_plan_id=physical_plan_id,
                    ordered_member_context_manifest_entry_ids=member_ids,
                    shared_context_sha256=shared_context_sha256,
                    intended_source_count=1,
                    ordered_source_units=(
                        SimpleNamespace(
                            source_unit_id=canonical_sha256(
                                [physical_plan_id, 1, source_hashes[0]]
                            ),
                            context_manifest_entry_id=member_ids[0],
                            ordinal_1_indexed=1,
                            payload_sha256=source_hashes[0],
                            payload_bytes=source_payload,
                        ),
                    ),
                    ordered_case_manifest_entry_ids=tuple(plan_case_ids),
                ),
                component=component,
                member_labels=tuple(
                    (
                        f"factconsolidation_{family}_{plan_ordinal}@{member_ordinal}"
                        if component == "cr_sf"
                        else f"member-{plan_ordinal}-{member_ordinal}"
                    )
                    for member_ordinal, family in zip(
                        range(1, member_count + 1), ("sh", "mh"), strict=False
                    )
                ),
            )
        )
    manifest_fields = {
        "manifest_id": manifest_id,
        "workload_id": "mab65-v1",
        "logical_contexts": tuple(vars(context) for context in context_entries),
        "ingestion_plans": tuple(vars(plan.manifest) for plan in plans),
        "cases": tuple(vars(case.entry) for case in cases),
    }
    case_digest = canonical_sha256(
        [
            "oamb-mab65-selected-case-ids-v1",
            tuple(case.entry.case_manifest_entry_id for case in cases),
        ]
    )
    plan_digest = canonical_sha256(
        [
            "oamb-mab65-selected-plan-manifest-ids-v1",
            tuple(plan.manifest.plan_manifest_entry_id for plan in plans),
        ]
    )
    cr_32k_plan_id = next(
        plan.manifest.plan_manifest_entry_id
        for plan in plans
        if plan.component == "cr_sf" and len(plan.manifest.ordered_case_manifest_entry_ids) == 3
    )
    return SimpleNamespace(
        case_manifest=_MutableCaseManifest(
            manifest_id=manifest_id,
            workload_id="mab65-v1",
            logical_contexts=tuple(context_entries),
            ingestion_plans=tuple(plan.manifest for plan in plans),
            cases=tuple(case.entry for case in cases),
            manifest_hash=case_manifest_hash(manifest_fields),
        ),
        cases=tuple(cases),
        plans=tuple(plans),
        selected_case_digest=case_digest,
        selected_plan_digest=plan_digest,
        cr_32k_plan_id=cr_32k_plan_id,
        entity_catalog_sha256=CATALOG_SHA256,
        unicode_fingerprint=UNICODE_FINGERPRINT,
    )


@pytest.fixture
def mab65_bundle(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    from oamb.workloads import mab65_reduction

    bundle = _generated_bundle()
    monkeypatch.setattr(
        mab65_reduction,
        "MAB65_SELECTED_CASE_DIGEST",
        bundle.selected_case_digest,
    )
    monkeypatch.setattr(
        mab65_reduction,
        "MAB65_SELECTED_PLAN_DIGEST",
        bundle.selected_plan_digest,
    )
    monkeypatch.setattr(
        mab65_reduction,
        "MAB65_CASE_MANIFEST_HASH",
        bundle.case_manifest.manifest_hash,
        raising=False,
    )
    monkeypatch.setattr(
        mab65_reduction,
        "MAB65_CR_32K_PLAN_ID",
        bundle.cr_32k_plan_id,
    )
    return bundle


def test_capability_balanced_reducer_uses_exact_plan_macro_fractions(
    mab65_bundle: Any,
) -> None:
    from oamb.workloads.mab65_reduction import reduce_mab65

    component_values = {
        "ar": Fraction(1),
        "icl": Fraction(0),
        "recsys": Fraction(1, 2),
        "lru": Fraction(1, 3),
        "cr_sf": Fraction(1),
    }
    metrics = {
        case.entry.case_manifest_entry_id: component_values[case.component]
        for case in mab65_bundle.cases
    }
    ready = frozenset(plan.manifest.plan_manifest_entry_id for plan in mab65_bundle.plans)

    result = reduce_mab65(
        mab65_bundle,
        case_metrics=metrics,
        ready_plan_manifest_ids=ready,
        capsule_valid=True,
        query_state_unchanged=True,
        expected_catalog_sha256=CATALOG_SHA256,
        expected_unicode_fingerprint=UNICODE_FINGERPRINT,
        expected_interaction_fingerprint=INTERACTION_FINGERPRINT,
        observed_interaction_fingerprint=INTERACTION_FINGERPRINT,
        comparison_controls_closed=True,
    )

    assert result.available is True
    assert result.unavailable_reason is None
    assert {item.component: item.score for item in result.components} == component_values
    assert result.ttl_score == Fraction(1, 4)
    assert result.index_value == Fraction(775, 12)
    assert all(isinstance(plan.score, Fraction) for plan in result.plans)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("missing_case", "case_metric_inventory"),
        ("missing_ready_plan", "ready_plan_inventory"),
        ("invalid_capsule", "capsule_invalid"),
        ("query_mutation", "query_state_changed"),
    ),
)
def test_capability_balanced_reducer_suppresses_index_on_incomplete_evidence(
    mab65_bundle: Any,
    mutation: str,
    reason: str,
) -> None:
    from oamb.workloads.mab65_reduction import reduce_mab65

    metrics = {case.entry.case_manifest_entry_id: Fraction(0) for case in mab65_bundle.cases}
    ready = {plan.manifest.plan_manifest_entry_id for plan in mab65_bundle.plans}
    capsule_valid = True
    query_state_unchanged = True
    if mutation == "missing_case":
        metrics.pop(next(iter(metrics)))
    elif mutation == "missing_ready_plan":
        ready.pop()
    elif mutation == "invalid_capsule":
        capsule_valid = False
    else:
        query_state_unchanged = False

    result = reduce_mab65(
        mab65_bundle,
        case_metrics=metrics,
        ready_plan_manifest_ids=frozenset(ready),
        capsule_valid=capsule_valid,
        query_state_unchanged=query_state_unchanged,
        expected_catalog_sha256=CATALOG_SHA256,
        expected_unicode_fingerprint=UNICODE_FINGERPRINT,
        expected_interaction_fingerprint=INTERACTION_FINGERPRINT,
        observed_interaction_fingerprint=INTERACTION_FINGERPRINT,
        comparison_controls_closed=True,
    )

    assert result.available is False
    assert result.unavailable_reason == reason
    assert result.index_value is None


def test_mab5_cannot_produce_components_or_capability_index(mab65_bundle: Any) -> None:
    from oamb.workloads.mab65_reduction import reduce_mab65

    smoke = _generated_bundle(manifest_id="mab5-live-smoke-v1")
    result = reduce_mab65(
        smoke,
        case_metrics={case.entry.case_manifest_entry_id: Fraction(1) for case in smoke.cases},
        ready_plan_manifest_ids=frozenset(
            plan.manifest.plan_manifest_entry_id for plan in smoke.plans
        ),
        capsule_valid=True,
        query_state_unchanged=True,
        expected_catalog_sha256=CATALOG_SHA256,
        expected_unicode_fingerprint=UNICODE_FINGERPRINT,
        expected_interaction_fingerprint=INTERACTION_FINGERPRINT,
        observed_interaction_fingerprint=INTERACTION_FINGERPRINT,
        comparison_controls_closed=True,
    )

    assert result.available is False
    assert result.unavailable_reason == "manifest_not_mab65"
    assert result.components == ()


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("case_digest", "manifest_identity"),
        ("plan_digest", "manifest_identity"),
        ("case_identity", "manifest_identity"),
        ("plan_identity", "manifest_identity"),
        ("duplicate_membership", "manifest_topology"),
        ("cr_shared_context", "manifest_topology"),
        ("runtime_plan_members", "manifest_topology"),
        ("cr_member_order", "manifest_topology"),
        ("case_context_binding", "manifest_topology"),
        ("cr_32k_count", "manifest_topology"),
        ("cr_member_quota", "manifest_topology"),
        ("metric_mapping", "metric_component_mapping"),
        ("catalog", "catalog_fingerprint"),
        ("unicode", "unicode_fingerprint"),
        ("interaction", "interaction_fingerprint"),
        ("controls", "comparison_controls"),
    ),
)
def test_reducer_rejects_frozen_topology_and_control_drift(
    mab65_bundle: Any,
    mutation: str,
    reason: str,
) -> None:
    from oamb.workloads.mab65_reduction import reduce_mab65

    bundle = copy.deepcopy(mab65_bundle)
    expected_catalog = CATALOG_SHA256
    expected_unicode = UNICODE_FINGERPRINT
    observed_interaction = INTERACTION_FINGERPRINT
    controls_closed = True
    if mutation == "case_digest":
        bundle.selected_case_digest = "0" * 64
    elif mutation == "plan_digest":
        bundle.selected_plan_digest = "0" * 64
    elif mutation == "case_identity":
        bundle.cases[0].entry.case_manifest_entry_id = "changed-case"
    elif mutation == "plan_identity":
        bundle.plans[0].manifest.plan_manifest_entry_id = "changed-plan"
    elif mutation == "duplicate_membership":
        duplicated = bundle.plans[0].manifest.ordered_case_manifest_entry_ids[0]
        second = bundle.plans[1].manifest
        second.ordered_case_manifest_entry_ids = (
            duplicated,
            *second.ordered_case_manifest_entry_ids[1:],
        )
    elif mutation == "cr_shared_context":
        cr_plan = next(plan for plan in bundle.plans if plan.component == "cr_sf")
        second_member_id = cr_plan.manifest.ordered_member_context_manifest_entry_ids[1]
        next(
            context
            for context in bundle.case_manifest.logical_contexts
            if context.context_manifest_entry_id == second_member_id
        ).context_bytes_sha256 = "different-context-hash"
    elif mutation == "runtime_plan_members":
        bundle.plans[0].runtime_plan.ordered_member_context_manifest_entry_ids = (
            "different-context",
        )
    elif mutation == "cr_member_order":
        cr_plan = next(
            plan
            for plan in bundle.plans
            if plan.component == "cr_sf" and len(plan.manifest.ordered_case_manifest_entry_ids) == 4
        )
        reversed_members = tuple(
            reversed(cr_plan.manifest.ordered_member_context_manifest_entry_ids)
        )
        cr_plan.manifest.ordered_member_context_manifest_entry_ids = reversed_members
        cr_plan.runtime_plan.ordered_member_context_manifest_entry_ids = reversed_members
        for source in cr_plan.runtime_plan.ordered_source_units:
            source.context_manifest_entry_id = reversed_members[0]
    elif mutation == "case_context_binding":
        cr_plan = next(
            plan
            for plan in bundle.plans
            if plan.component == "cr_sf" and len(plan.manifest.ordered_case_manifest_entry_ids) == 4
        )
        first_member, second_member = cr_plan.manifest.ordered_member_context_manifest_entry_ids
        first_case = next(
            case
            for case in bundle.cases
            if case.entry.case_manifest_entry_id in cr_plan.manifest.ordered_case_manifest_entry_ids
            and case.entry.context_manifest_entry_id == first_member
        )
        second_case = next(
            case
            for case in bundle.cases
            if case.entry.case_manifest_entry_id in cr_plan.manifest.ordered_case_manifest_entry_ids
            and case.entry.context_manifest_entry_id == second_member
        )
        first_case.entry.context_manifest_entry_id = second_member
        second_case.entry.context_manifest_entry_id = first_member
    elif mutation == "cr_32k_count":
        plan_32k = next(
            plan
            for plan in bundle.plans
            if plan.manifest.plan_manifest_entry_id == bundle.cr_32k_plan_id
        )
        other = bundle.plans[-1]
        moved = other.manifest.ordered_case_manifest_entry_ids[-1]
        plan_32k.manifest.ordered_case_manifest_entry_ids += (moved,)
        other.manifest.ordered_case_manifest_entry_ids = (
            other.manifest.ordered_case_manifest_entry_ids[:-1]
        )
    elif mutation == "cr_member_quota":
        plan_32k = next(
            plan
            for plan in bundle.plans
            if plan.manifest.plan_manifest_entry_id == bundle.cr_32k_plan_id
        )
        case_id = plan_32k.manifest.ordered_case_manifest_entry_ids[1]
        next(
            case for case in bundle.cases if case.entry.case_manifest_entry_id == case_id
        ).entry.context_manifest_entry_id = (
            plan_32k.manifest.ordered_member_context_manifest_entry_ids[1]
        )
    elif mutation == "metric_mapping":
        bundle.cases[0].case_plan.metric_id = "mab-exact-v1"
    elif mutation == "catalog":
        expected_catalog = "0" * 64
    elif mutation == "unicode":
        expected_unicode = "0" * 64
    elif mutation == "interaction":
        observed_interaction = "0" * 64
    else:
        controls_closed = False
    metrics = {case.entry.case_manifest_entry_id: Fraction(1) for case in bundle.cases}
    result = reduce_mab65(
        bundle,
        case_metrics=metrics,
        ready_plan_manifest_ids=frozenset(
            plan.manifest.plan_manifest_entry_id for plan in bundle.plans
        ),
        capsule_valid=True,
        query_state_unchanged=True,
        expected_catalog_sha256=expected_catalog,
        expected_unicode_fingerprint=expected_unicode,
        expected_interaction_fingerprint=INTERACTION_FINGERPRINT,
        observed_interaction_fingerprint=observed_interaction,
        comparison_controls_closed=controls_closed,
    )

    assert result.available is False
    assert result.unavailable_reason == reason


@pytest.mark.parametrize(
    "mutation",
    (
        "source_payload_bytes",
        "source_unit_id",
        "source_ordinal",
        "source_context",
        "derived_ingestion_plan_id",
        "ingestion_payload_hash",
    ),
)
def test_reducer_rejects_source_unit_and_ingestion_identity_drift(
    mab65_bundle: Any,
    mutation: str,
) -> None:
    from oamb.workloads.mab65_reduction import reduce_mab65

    bundle = copy.deepcopy(mab65_bundle)
    plan = bundle.plans[0]
    source = plan.runtime_plan.ordered_source_units[0]
    if mutation == "source_payload_bytes":
        source.payload_bytes = b"tampered payload with the declared hash retained"
    elif mutation == "source_unit_id":
        source.source_unit_id = "0" * 64
    elif mutation == "source_ordinal":
        source.ordinal_1_indexed = 2
    elif mutation == "source_context":
        source.context_manifest_entry_id = "different-context"
    elif mutation == "derived_ingestion_plan_id":
        plan.manifest.ingestion_plan_id = "f" * 64
        plan.runtime_plan.ingestion_plan_id = "f" * 64
    else:
        plan.manifest.ingestion_payload_hash = "e" * 64

    result = reduce_mab65(
        bundle,
        case_metrics={case.entry.case_manifest_entry_id: Fraction(1) for case in bundle.cases},
        ready_plan_manifest_ids=frozenset(
            item.manifest.plan_manifest_entry_id for item in bundle.plans
        ),
        capsule_valid=True,
        query_state_unchanged=True,
        expected_catalog_sha256=CATALOG_SHA256,
        expected_unicode_fingerprint=UNICODE_FINGERPRINT,
        expected_interaction_fingerprint=INTERACTION_FINGERPRINT,
        observed_interaction_fingerprint=INTERACTION_FINGERPRINT,
        comparison_controls_closed=True,
    )

    assert result.available is False
    assert result.unavailable_reason == "manifest_topology"
