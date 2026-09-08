from __future__ import annotations

from types import SimpleNamespace

import pytest

from oamb.contracts.ports import CasePlan, IngestionPlan
from oamb.runtime.case_selection import CaseSelectionError, select_case_execution


def _case(case_id: str) -> CasePlan:
    return CasePlan(
        case_manifest_entry_id=case_id,
        context_manifest_entry_id=f"context-{case_id}",
        source_question_number_1_indexed=1,
        question_bytes=b"question",
        reference_payload=b'"answer"',
        reference_payload_sha256="a" * 64,
        prompt_binding_id="answer-prompt",
        output_contract_id="answer-output",
        metric_id="metric",
        judge_binding_id="judge-prompt",
    )


def _plan(plan_id: str, *case_ids: str) -> IngestionPlan:
    return IngestionPlan(
        ingestion_plan_id=plan_id,
        ordered_member_context_manifest_entry_ids=(f"context-{case_ids[0]}",),
        shared_context_sha256="b" * 64,
        intended_source_count=1,
        ordered_source_units=(),
        ordered_case_manifest_entry_ids=case_ids,
    )


def _manifest() -> SimpleNamespace:
    cases = tuple(
        SimpleNamespace(case_manifest_entry_id=case_id)
        for case_id in ("case-1", "case-2", "case-3")
    )
    plans = (
        SimpleNamespace(
            ingestion_plan_id="plan-1",
            ordered_case_manifest_entry_ids=("case-1", "case-2"),
        ),
        SimpleNamespace(
            ingestion_plan_id="plan-2",
            ordered_case_manifest_entry_ids=("case-3",),
        ),
    )
    return SimpleNamespace(cases=cases, ingestion_plans=plans)


def test_selection_ingests_each_selected_history_but_runs_only_requested_cases() -> None:
    plans = (_plan("plan-1", "case-1", "case-2"), _plan("plan-2", "case-3"))
    cases = tuple(_case(case_id) for case_id in ("case-1", "case-2", "case-3"))

    selected = select_case_execution(
        case_manifest=_manifest(),
        ingestion_plans=plans,
        case_plans=cases,
        requested_case_manifest_entry_ids=("case-2", "case-3"),
    )

    assert tuple(item.ingestion_plan_id for item in selected.ingestion_plans) == (
        "plan-1",
        "plan-2",
    )
    assert tuple(item.ordered_case_manifest_entry_ids for item in selected.ingestion_plans) == (
        ("case-2",),
        ("case-3",),
    )
    assert tuple(item.case_manifest_entry_id for item in selected.case_plans) == (
        "case-2",
        "case-3",
    )


@pytest.mark.parametrize(
    "requested",
    (
        (),
        ("case-1", "case-1"),
        ("case-3", "case-1"),
        ("unknown",),
    ),
)
def test_selection_rejects_empty_duplicate_out_of_order_or_unknown_requests(
    requested: tuple[str, ...],
) -> None:
    plans = (_plan("plan-1", "case-1", "case-2"), _plan("plan-2", "case-3"))
    cases = tuple(_case(case_id) for case_id in ("case-1", "case-2", "case-3"))

    with pytest.raises(CaseSelectionError):
        select_case_execution(
            case_manifest=_manifest(),
            ingestion_plans=plans,
            case_plans=cases,
            requested_case_manifest_entry_ids=requested,
        )
