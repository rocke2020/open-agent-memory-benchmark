"""Pure completion-first reducers over validated source-manifest snapshots."""

from __future__ import annotations

from pathlib import Path

from oamb.artifacts.validation.fake import (
    EvidenceNotValidatedError,
    FakeCapsuleSnapshot,
    load_fake_capsule,
    validate_fake_snapshot,
)
from oamb.contracts.evidence import (
    CaseEvaluationDisposition,
    CaseRecord,
    CaseRecordV2,
    IngestionPlanRecord,
    LogicalContextRecord,
    ValidationResult,
)
from oamb.contracts.reporting import RunSummaryV2
from oamb.contracts.states import CaseState, IngestionPlanState, ValidationDisposition


def reduce_run_summary(capsule_root: Path, validation_result: ValidationResult) -> RunSummaryV2:
    return reduce_run_summary_snapshot(load_fake_capsule(capsule_root), validation_result)


def reduce_run_summary_snapshot(
    snapshot: FakeCapsuleSnapshot,
    validation_result: ValidationResult,
) -> RunSummaryV2:
    if getattr(validation_result, "disposition", None) != ValidationDisposition.VALIDATED:
        raise EvidenceNotValidatedError("normal reduction requires evidence validation PASS")
    if validate_fake_snapshot(snapshot) != validation_result:
        raise EvidenceNotValidatedError(
            "normal reduction requires fresh validation of the current capsule bytes"
        )
    if snapshot.manifest is None:
        raise EvidenceNotValidatedError("validated capsule manifest is unavailable")
    if validation_result.target_hash != snapshot.manifest.source_manifest_hash:
        raise EvidenceNotValidatedError("validation result does not bind this capsule")
    contexts = tuple(item for item in snapshot.contracts if isinstance(item, LogicalContextRecord))
    plans = tuple(item for item in snapshot.contracts if isinstance(item, IngestionPlanRecord))
    cases = tuple(
        item for item in snapshot.contracts if isinstance(item, (CaseRecord, CaseRecordV2))
    )
    terminal_states = {
        CaseState.COMPLETED,
        CaseState.ERROR,
        CaseState.UNSUPPORTED,
        CaseState.CANCELLED,
        CaseState.BUDGET_EXCEEDED,
    }
    v2_cases = tuple(item for item in cases if isinstance(item, CaseRecordV2))
    return RunSummaryV2(
        run_id=snapshot.manifest.run_id,
        intended_logical_contexts=len(contexts),
        intended_ingestion_plans=len(plans),
        ready_ingestion_plans=sum(plan.state == IngestionPlanState.SEALED for plan in plans),
        intended_cases=len(cases),
        terminal_cases=sum(case.state in terminal_states for case in cases),
        completed_cases=sum(case.state == CaseState.COMPLETED for case in cases),
        errored_cases=sum(case.state == CaseState.ERROR for case in cases),
        unsupported_cases=sum(case.state == CaseState.UNSUPPORTED for case in cases),
        cancelled_cases=sum(case.state == CaseState.CANCELLED for case in cases),
        budget_exceeded_cases=sum(case.state == CaseState.BUDGET_EXCEEDED for case in cases),
        parsed_cases=sum(item.parsed_answer_sha256 is not None for item in v2_cases),
        evaluated_cases=sum(
            item.evaluation_disposition
            in {
                CaseEvaluationDisposition.DETERMINISTIC_EVALUATED,
                CaseEvaluationDisposition.JUDGED,
            }
            for item in v2_cases
        ),
        judged_cases=sum(
            item.evaluation_disposition == CaseEvaluationDisposition.JUDGED for item in v2_cases
        ),
        unjudged_cases=sum(
            item.evaluation_disposition == CaseEvaluationDisposition.UNJUDGED for item in v2_cases
        ),
        billing_complete=False,
        cost_complete=False,
    )


def reduce_diagnostic_summary(capsule_root: Path) -> RunSummaryV2:
    """Build only completion counts; diagnostic output never claims billing/cost closure."""

    snapshot = load_fake_capsule(capsule_root)
    contexts = tuple(item for item in snapshot.contracts if isinstance(item, LogicalContextRecord))
    plans = tuple(item for item in snapshot.contracts if isinstance(item, IngestionPlanRecord))
    cases = tuple(
        item for item in snapshot.contracts if isinstance(item, (CaseRecord, CaseRecordV2))
    )
    terminal_states = {
        CaseState.COMPLETED,
        CaseState.ERROR,
        CaseState.UNSUPPORTED,
        CaseState.CANCELLED,
        CaseState.BUDGET_EXCEEDED,
    }
    run_id = snapshot.manifest.run_id if snapshot.manifest is not None else "diagnostic-unbound"
    v2_cases = tuple(item for item in cases if isinstance(item, CaseRecordV2))
    return RunSummaryV2(
        run_id=run_id,
        intended_logical_contexts=len(contexts),
        intended_ingestion_plans=len(plans),
        ready_ingestion_plans=sum(plan.state == IngestionPlanState.SEALED for plan in plans),
        intended_cases=len(cases),
        terminal_cases=sum(case.state in terminal_states for case in cases),
        completed_cases=sum(case.state == CaseState.COMPLETED for case in cases),
        errored_cases=sum(case.state == CaseState.ERROR for case in cases),
        unsupported_cases=sum(case.state == CaseState.UNSUPPORTED for case in cases),
        cancelled_cases=sum(case.state == CaseState.CANCELLED for case in cases),
        budget_exceeded_cases=sum(case.state == CaseState.BUDGET_EXCEEDED for case in cases),
        parsed_cases=sum(item.parsed_answer_sha256 is not None for item in v2_cases),
        evaluated_cases=sum(
            item.evaluation_disposition
            in {
                CaseEvaluationDisposition.DETERMINISTIC_EVALUATED,
                CaseEvaluationDisposition.JUDGED,
            }
            for item in v2_cases
        ),
        judged_cases=sum(
            item.evaluation_disposition == CaseEvaluationDisposition.JUDGED for item in v2_cases
        ),
        unjudged_cases=sum(
            item.evaluation_disposition == CaseEvaluationDisposition.UNJUDGED for item in v2_cases
        ),
        billing_complete=False,
        cost_complete=False,
    )
