"""In-memory case selection for fresh isolated executions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from oamb.contracts.ports import CasePlan, IngestionPlan


class CaseSelectionError(ValueError):
    """A requested execution slice differs from the frozen manifest."""


@dataclass(frozen=True, slots=True)
class SelectedCaseExecution:
    ingestion_plans: tuple[IngestionPlan, ...]
    case_plans: tuple[CasePlan, ...]


def select_case_execution(
    *,
    case_manifest: Any,
    ingestion_plans: tuple[IngestionPlan, ...],
    case_plans: tuple[CasePlan, ...],
    requested_case_manifest_entry_ids: tuple[str, ...],
) -> SelectedCaseExecution:
    """Select manifest-ordered cases while retaining each owning history input."""

    manifest_case_ids = tuple(item.case_manifest_entry_id for item in case_manifest.cases)
    if len(set(manifest_case_ids)) != len(manifest_case_ids):
        raise CaseSelectionError("case manifest identities are duplicated")
    case_index = {case_id: index for index, case_id in enumerate(manifest_case_ids)}
    requested = requested_case_manifest_entry_ids
    if not requested or len(set(requested)) != len(requested):
        raise CaseSelectionError("case selection must be nonempty and unique")
    if any(case_id not in case_index for case_id in requested):
        raise CaseSelectionError("case selection contains an unknown identity")
    indexes = tuple(case_index[case_id] for case_id in requested)
    if indexes != tuple(sorted(indexes)):
        raise CaseSelectionError("case selection differs from manifest order")

    runtime_case_ids = tuple(item.case_manifest_entry_id for item in case_plans)
    if runtime_case_ids != manifest_case_ids:
        raise CaseSelectionError("runtime cases differ from the frozen manifest")
    manifest_plan_ids = tuple(item.ingestion_plan_id for item in case_manifest.ingestion_plans)
    runtime_plan_ids = tuple(item.ingestion_plan_id for item in ingestion_plans)
    if runtime_plan_ids != manifest_plan_ids or len(set(runtime_plan_ids)) != len(runtime_plan_ids):
        raise CaseSelectionError("runtime ingestion plans differ from the frozen manifest")
    for manifest_plan, runtime_plan in zip(
        case_manifest.ingestion_plans,
        ingestion_plans,
        strict=True,
    ):
        if tuple(runtime_plan.ordered_case_manifest_entry_ids) != tuple(
            manifest_plan.ordered_case_manifest_entry_ids
        ):
            raise CaseSelectionError("runtime ingestion case binding differs from the manifest")

    requested_set = set(requested)
    selected_plans = tuple(
        replace(
            plan,
            ordered_case_manifest_entry_ids=tuple(
                case_id
                for case_id in plan.ordered_case_manifest_entry_ids
                if case_id in requested_set
            ),
        )
        for plan in ingestion_plans
        if requested_set.intersection(plan.ordered_case_manifest_entry_ids)
    )
    selected_case_plans = tuple(
        case for case in case_plans if case.case_manifest_entry_id in requested_set
    )
    selected_bindings = tuple(
        case_id for plan in selected_plans for case_id in plan.ordered_case_manifest_entry_ids
    )
    if (
        selected_bindings != requested
        or tuple(item.case_manifest_entry_id for item in selected_case_plans) != requested
    ):
        raise CaseSelectionError("selected execution does not close the requested cases")
    return SelectedCaseExecution(
        ingestion_plans=selected_plans,
        case_plans=selected_case_plans,
    )


__all__ = ["CaseSelectionError", "SelectedCaseExecution", "select_case_execution"]
