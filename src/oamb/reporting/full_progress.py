"""Pure mapping from flat progress into report-ready manifest identities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from oamb.runtime.full_progress import (
    FullProgress,
    FullProgressEntry,
    FullProgressError,
    require_complete_full_progress,
)


class FullProgressAdapterError(ValueError):
    """A progress snapshot cannot close the frozen report inputs."""


@dataclass(frozen=True, slots=True)
class ProgressCaseReductionInput:
    question_id: str
    case_manifest_entry_id: str
    entry: FullProgressEntry


@dataclass(frozen=True, slots=True)
class ProgressCellReductionInput:
    cell: Any
    case_manifest: Any
    progress: FullProgress
    cases: tuple[ProgressCaseReductionInput, ...]


def adapt_full_progress(
    *,
    plan: Any,
    cell: Any,
    case_manifest: Any,
    progress: FullProgress,
) -> ProgressCellReductionInput:
    """Bind one complete progress snapshot to its frozen manifest and cell."""

    try:
        require_complete_full_progress(progress)
    except FullProgressError as exc:
        raise FullProgressAdapterError(str(exc)) from exc

    expected_identity = (
        ("resolved plan", progress.resolved_plan_hash, plan.resolved_plan_hash),
        ("cell", progress.cell_id, cell.cell_id),
        ("provider", progress.provider_id, cell.provider_id),
        ("workload", progress.workload_id, cell.workload_id),
        ("cell manifest", progress.case_manifest_hash, cell.case_manifest_hash),
        ("plan manifest", progress.case_manifest_hash, plan.dataset.case_manifest_hash),
        ("manifest", progress.case_manifest_hash, case_manifest.manifest_hash),
        ("manifest workload", progress.workload_id, case_manifest.workload_id),
    )
    drift = tuple(label for label, actual, expected in expected_identity if actual != expected)
    if drift:
        raise FullProgressAdapterError(f"full progress identity differs at: {', '.join(drift)}")

    manifest_question_ids = tuple(item.raw_question_id for item in case_manifest.cases)
    if (
        len(manifest_question_ids) != 60
        or len(set(manifest_question_ids)) != 60
        or manifest_question_ids != progress.ordered_question_ids
    ):
        raise FullProgressAdapterError(
            "full progress question order differs from the frozen case manifest"
        )
    manifest_case_ids = tuple(item.case_manifest_entry_id for item in case_manifest.cases)
    if len(set(manifest_case_ids)) != 60:
        raise FullProgressAdapterError("frozen case manifest identity is duplicated")

    cases = tuple(
        ProgressCaseReductionInput(
            question_id=entry.question_id,
            case_manifest_entry_id=case_id,
            entry=entry,
        )
        for case_id, entry in zip(manifest_case_ids, progress.results, strict=True)
    )
    return ProgressCellReductionInput(
        cell=cell,
        case_manifest=case_manifest,
        progress=progress,
        cases=cases,
    )


__all__ = [
    "FullProgressAdapterError",
    "ProgressCaseReductionInput",
    "ProgressCellReductionInput",
    "adapt_full_progress",
]
