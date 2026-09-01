"""Deterministic whole-ingestion-plan case partition selection."""

from __future__ import annotations

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import CasePlan, IngestionPlan
from oamb.contracts.specifications import (
    CaseExecutionBinding,
    CaseManifest,
    CasePartitionSpec,
    case_execution_binding_hash,
    case_partition_id,
)


class CasePartitionSelectionError(ValueError):
    """Raised before dispatch when a requested case partition is not canonical."""


def select_partition_execution(
    *,
    partition: CasePartitionSpec,
    dataset_manifest_hash: str,
    case_manifest: CaseManifest,
    ingestion_plans: tuple[IngestionPlan, ...],
    case_plans: tuple[CasePlan, ...],
) -> tuple[tuple[IngestionPlan, ...], tuple[CasePlan, ...]]:
    """Recompute a persisted partition and return its target-ordered execution slice."""

    expected = build_case_partition_spec(
        run_id=partition.run_id,
        resolved_plan_hash=partition.resolved_plan_hash,
        cell_spec_hash=partition.cell_spec_hash,
        dataset_manifest_hash=dataset_manifest_hash,
        case_manifest=case_manifest,
        case_plans=case_plans,
        requested_case_manifest_entry_ids=partition.requested_case_manifest_entry_ids,
        budget_policy_hash=partition.budget_policy_hash,
        retry_policy_hash=partition.retry_policy_hash,
    )
    if partition != expected:
        raise CasePartitionSelectionError(
            "case partition does not match the workload's canonical whole-plan selection"
        )
    plans_by_id = {plan.ingestion_plan_id: plan for plan in ingestion_plans}
    cases_by_id = {case.case_manifest_entry_id: case for case in case_plans}
    try:
        selected_plans = tuple(
            plans_by_id[plan_id] for plan_id in partition.selected_ingestion_plan_ids
        )
        selected_cases = tuple(
            cases_by_id[case_id] for case_id in partition.selected_case_manifest_entry_ids
        )
    except KeyError as exc:
        raise CasePartitionSelectionError(
            "case partition references an unavailable runtime plan or case"
        ) from exc
    return selected_plans, selected_cases


def build_case_partition_spec(
    *,
    run_id: str,
    resolved_plan_hash: str,
    cell_spec_hash: str,
    dataset_manifest_hash: str,
    case_manifest: CaseManifest,
    case_plans: tuple[CasePlan, ...],
    requested_case_manifest_entry_ids: tuple[str, ...],
    budget_policy_hash: str,
    retry_policy_hash: str,
) -> CasePartitionSpec:
    """Expand requested cases to whole plans and bind the full target execution."""

    target_case_ids = tuple(case.case_manifest_entry_id for case in case_manifest.cases)
    target_case_index = {case_id: index for index, case_id in enumerate(target_case_ids)}
    if not requested_case_manifest_entry_ids:
        raise CasePartitionSelectionError("empty case selection")
    if len(set(requested_case_manifest_entry_ids)) != len(requested_case_manifest_entry_ids):
        raise CasePartitionSelectionError("duplicate case selection")
    unknown = tuple(
        case_id for case_id in requested_case_manifest_entry_ids if case_id not in target_case_index
    )
    if unknown:
        raise CasePartitionSelectionError("unknown case selection")
    requested_indexes = tuple(
        target_case_index[case_id] for case_id in requested_case_manifest_entry_ids
    )
    if requested_indexes != tuple(sorted(requested_indexes)):
        raise CasePartitionSelectionError("out of order case selection")

    requested_set = set(requested_case_manifest_entry_ids)
    selected_plans = tuple(
        plan
        for plan in case_manifest.ingestion_plans
        if requested_set.intersection(plan.ordered_case_manifest_entry_ids)
    )
    selected_case_ids = tuple(
        case_id for plan in selected_plans for case_id in plan.ordered_case_manifest_entry_ids
    )

    plans_by_id = {plan.case_manifest_entry_id: plan for plan in case_plans}
    if tuple(plans_by_id) != target_case_ids or len(plans_by_id) != len(case_plans):
        raise CasePartitionSelectionError(
            "case plan inventory does not match the target case manifest"
        )
    target_bindings = tuple(_execution_binding(plans_by_id[case_id]) for case_id in target_case_ids)
    target_bindings_hash = canonical_sha256(
        ["oamb-target-case-execution-bindings-v1", target_bindings]
    )
    fields = {
        "schema_name": "case_partition_spec",
        "schema_version": 1,
        "run_id": run_id,
        "resolved_plan_hash": resolved_plan_hash,
        "cell_spec_hash": cell_spec_hash,
        "dataset_manifest_hash": dataset_manifest_hash,
        "target_case_manifest_hash": case_manifest.manifest_hash,
        "budget_policy_hash": budget_policy_hash,
        "retry_policy_hash": retry_policy_hash,
        "requested_case_manifest_entry_ids": requested_case_manifest_entry_ids,
        "selected_ingestion_plan_ids": tuple(plan.ingestion_plan_id for plan in selected_plans),
        "selected_case_manifest_entry_ids": selected_case_ids,
        "target_case_execution_bindings": target_bindings,
        "target_case_execution_bindings_hash": target_bindings_hash,
    }
    return CasePartitionSpec.model_validate({"partition_id": case_partition_id(fields), **fields})


def _execution_binding(plan: CasePlan) -> CaseExecutionBinding:
    fields = {
        "schema_name": "case_execution_binding",
        "schema_version": 1,
        "case_manifest_entry_id": plan.case_manifest_entry_id,
        "prompt_binding_id": plan.prompt_binding_id,
        "output_contract_id": plan.output_contract_id,
        "metric_id": plan.metric_id,
        "judge_binding_id": plan.judge_binding_id,
        "answer_max_output_tokens": plan.answer_max_output_tokens,
        "query_timestamp": plan.query_timestamp,
    }
    return CaseExecutionBinding.model_validate(
        {"binding_hash": case_execution_binding_hash(fields), **fields}
    )
