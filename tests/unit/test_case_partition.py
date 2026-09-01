from __future__ import annotations

import pytest
from pydantic import ValidationError

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import CasePartitionSpec
from oamb.runtime.case_partition import CasePartitionSelectionError, build_case_partition_spec
from oamb.workloads.fake import GeneratedFakeWorkload


def _sha(label: str) -> str:
    return canonical_sha256([label])


def _build_partition(*requested_case_ids: str) -> CasePartitionSpec:
    workload = GeneratedFakeWorkload()
    dataset = workload.resolve_sources()
    manifest = workload.build_case_manifest(dataset)
    return build_case_partition_spec(
        run_id="partition-run-1",
        resolved_plan_hash=_sha("plan"),
        cell_spec_hash=_sha("cell"),
        dataset_manifest_hash=dataset.manifest_hash,
        case_manifest=manifest,
        case_plans=workload.iter_case_plans(manifest),
        requested_case_manifest_entry_ids=requested_case_ids,
        budget_policy_hash=_sha("budget-policy"),
        retry_policy_hash=_sha("retry-policy"),
    )


def test_selecting_one_case_expands_its_whole_ingestion_plan() -> None:
    workload = GeneratedFakeWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    first_plan = manifest.ingestion_plans[0]

    partition = _build_partition(first_plan.ordered_case_manifest_entry_ids[1])

    assert partition.requested_case_manifest_entry_ids == (
        first_plan.ordered_case_manifest_entry_ids[1],
    )
    assert partition.selected_ingestion_plan_ids == (first_plan.ingestion_plan_id,)
    assert partition.selected_case_manifest_entry_ids == (
        first_plan.ordered_case_manifest_entry_ids
    )
    assert tuple(
        binding.case_manifest_entry_id for binding in partition.target_case_execution_bindings
    ) == tuple(case.case_manifest_entry_id for case in manifest.cases)


def test_selection_preserves_target_plan_and_case_order() -> None:
    workload = GeneratedFakeWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    first_requested = manifest.ingestion_plans[0].ordered_case_manifest_entry_ids[1]
    second_requested = manifest.ingestion_plans[1].ordered_case_manifest_entry_ids[0]

    partition = _build_partition(first_requested, second_requested)

    assert partition.selected_ingestion_plan_ids == tuple(
        plan.ingestion_plan_id for plan in manifest.ingestion_plans
    )
    assert partition.selected_case_manifest_entry_ids == tuple(
        case.case_manifest_entry_id for case in manifest.cases
    )


@pytest.mark.parametrize("invalid_kind", ["empty", "unknown", "duplicate", "out-of-order"])
def test_invalid_case_selection_fails_closed(invalid_kind: str) -> None:
    workload = GeneratedFakeWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    first = manifest.ingestion_plans[0].ordered_case_manifest_entry_ids[0]
    last = manifest.ingestion_plans[-1].ordered_case_manifest_entry_ids[-1]
    requested = {
        "empty": (),
        "unknown": (_sha("unknown-case"),),
        "duplicate": (first, first),
        "out-of-order": (last, first),
    }[invalid_kind]

    with pytest.raises(CasePartitionSelectionError, match=invalid_kind.replace("-", " ")):
        _build_partition(*requested)


def test_partition_identity_and_execution_binding_hash_fail_on_mutation() -> None:
    workload = GeneratedFakeWorkload()
    manifest = workload.build_case_manifest(workload.resolve_sources())
    partition = _build_partition(manifest.cases[0].case_manifest_entry_id)

    with pytest.raises(ValidationError, match="partition identity"):
        partition.model_copy(update={"partition_id": _sha("wrong")}, deep=True).__class__(
            **partition.model_copy(update={"partition_id": _sha("wrong")}).model_dump()
        )
    with pytest.raises(ValidationError, match="execution binding hash"):
        CasePartitionSpec.model_validate(
            partition.model_copy(
                update={"target_case_execution_bindings_hash": _sha("wrong-bindings")}
            ).model_dump()
        )
