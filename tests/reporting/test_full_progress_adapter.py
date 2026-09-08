from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from oamb.reporting.full_progress import FullProgressAdapterError, adapt_full_progress
from oamb.runtime import full_progress as progress_contract
from oamb.workloads.longmemeval import LME60_EXPECTED_QUESTION_IDS


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


QUESTION_IDS = LME60_EXPECTED_QUESTION_IDS
CASE_IDS = tuple(_sha(value) for value in QUESTION_IDS)
PLAN_HASH = _sha("plan")
MANIFEST_HASH = _sha("manifest")


def _progress(
    *,
    resolved_plan_hash: str = PLAN_HASH,
    cell_id: str = "hindsight-lme60",
    provider_id: str = "hindsight",
    workload_id: str = "longmemeval-v1",
    case_manifest_hash: str = MANIFEST_HASH,
) -> progress_contract.FullProgress:
    measured = progress_contract.ProgressMeasurement(status="measured", value=1)
    ingestion = progress_contract.ProgressIngestion(
        status="sealed",
        intended_source_count=1,
        accepted_source_count=1,
        skipped_source_count=0,
        partial=False,
        indexing_tokens=measured,
        indexing_token_coverage="measured_complete",
        indexing_ready_latency_microseconds=measured,
    )
    retrieval = progress_contract.ProgressRetrieval(
        status="succeeded",
        visible_context=progress_contract.ProgressText(status="available", value="x"),
        visible_context_byte_count=measured,
        visible_context_token_count=measured,
        native_candidate_count=measured,
        visible_kept_count=measured,
        visible_dropped_count=progress_contract.ProgressMeasurement(status="measured", value=0),
        visible_truncated_count=progress_contract.ProgressMeasurement(status="measured", value=0),
        request_latency_microseconds=measured,
        generation_proof_disposition="runtime_verified",
    )
    answer = progress_contract.ProgressAnswer(
        status="parsed",
        parsed_answer=progress_contract.ProgressText(status="available", value="answer"),
        protocol_disposition="normal_stop",
    )
    evaluation = progress_contract.JudgedProgressEvaluation(
        disposition="judged",
        metric_id="lme-judged-accuracy-v1",
        numerator=1,
        denominator=1,
        judge_decision="yes",
    )
    attempts = progress_contract.ProgressAttemptCounts(
        ingestion=1,
        retrieval=1,
        answer=1,
        judge=1,
        final_failure_class="none",
    )
    entries = tuple(
        progress_contract.JudgedFullProgressEntry(
            terminal_status="judged",
            question_id=question_id,
            ingestion=ingestion,
            retrieval=retrieval,
            answer=answer,
            evaluation=evaluation,
            attempts=attempts,
        )
        for question_id in QUESTION_IDS
    )
    return progress_contract.FullProgress(
        resolved_plan_hash=resolved_plan_hash,
        cell_id=cell_id,
        provider_id=provider_id,
        workload_id=workload_id,
        case_manifest_hash=case_manifest_hash,
        ordered_question_ids=QUESTION_IDS,
        results=entries,
    )


def _progress_with_model_exhaustion() -> progress_contract.FullProgress:
    progress = _progress()
    judged = cast(progress_contract.JudgedFullProgressEntry, progress.results[0])
    unjudged = progress_contract.UnjudgedFullProgressEntry(
        terminal_status="unjudged",
        question_id=QUESTION_IDS[-2],
        ingestion=judged.ingestion,
        retrieval=judged.retrieval,
        answer=judged.answer,
        evaluation=progress_contract.UnjudgedProgressEvaluation(
            disposition="unjudged",
            reason="judge output remained malformed after correction",
        ),
        attempts=progress_contract.ProgressAttemptCounts(
            ingestion=1,
            retrieval=1,
            answer=1,
            judge=6,
            final_failure_class="output_contract_error",
        ),
        failure_stage="judge",
        failure_kind="output_contract_error",
        failure_reason="judge output remained malformed after correction",
    )
    provider_failed = progress_contract.ProviderFailedFullProgressEntry(
        terminal_status="provider_failed",
        question_id=QUESTION_IDS[-1],
        ingestion=judged.ingestion,
        retrieval=judged.retrieval,
        answer=progress_contract.FailedProgressAnswer(
            status="failed",
            parsed_answer=progress_contract.UnavailableProgressText(
                status="unavailable",
                reason="answer output remained malformed after correction",
            ),
            protocol_disposition="output_contract_error",
        ),
        evaluation=progress_contract.NotRunProgressEvaluation(
            disposition="not_run",
            reason="answer did not satisfy its output contract",
        ),
        attempts=progress_contract.ProgressAttemptCounts(
            ingestion=1,
            retrieval=1,
            answer=6,
            judge=0,
            final_failure_class="output_contract_error",
        ),
        failure_stage="answer",
        failure_kind="output_contract_error",
        failure_reason="answer output remained malformed after correction",
    )
    return progress_contract.FullProgress(
        resolved_plan_hash=progress.resolved_plan_hash,
        cell_id=progress.cell_id,
        provider_id=progress.provider_id,
        workload_id=progress.workload_id,
        case_manifest_hash=progress.case_manifest_hash,
        ordered_question_ids=progress.ordered_question_ids,
        results=(*progress.results[:-2], unjudged, provider_failed),
    )


def _inputs() -> tuple[Any, Any, Any]:
    dataset = SimpleNamespace(
        dataset_id="longmemeval-s-cleaned",
        workload_id="longmemeval-v1",
        case_manifest_hash=MANIFEST_HASH,
        selection="lme60",
    )
    plan = SimpleNamespace(
        comparison_id="fixture-lme60",
        resolved_plan_hash=PLAN_HASH,
        dataset=dataset,
        decision=None,
    )
    cell = SimpleNamespace(
        cell_id="hindsight-lme60",
        provider_id="hindsight",
        workload_id="longmemeval-v1",
        case_manifest_hash=MANIFEST_HASH,
        cell_spec_hash=_sha("cell"),
        adapter_profile_id="hindsight-rest-profile",
        producer_role_id="producer",
        embedding_role_id="embedding",
        answer_role_id="answer",
        judge_role_id="judge",
        retrieval_binding_id="retrieval",
    )
    manifest = SimpleNamespace(
        manifest_hash=MANIFEST_HASH,
        workload_id="longmemeval-v1",
        cases=tuple(
            SimpleNamespace(raw_question_id=question_id, case_manifest_entry_id=case_id)
            for question_id, case_id in zip(QUESTION_IDS, CASE_IDS, strict=True)
        ),
    )
    return plan, cell, manifest


def test_progress_adapter_maps_dataset_question_ids_through_the_frozen_manifest() -> None:
    plan, cell, manifest = _inputs()

    adapted = adapt_full_progress(
        plan=plan,
        cell=cell,
        case_manifest=manifest,
        progress=_progress(),
    )

    assert tuple(item.question_id for item in adapted.cases) == QUESTION_IDS
    assert tuple(item.case_manifest_entry_id for item in adapted.cases) == CASE_IDS
    assert all(item.entry.terminal_status == "judged" for item in adapted.cases)


def test_progress_adapter_rejects_manifest_question_order_drift() -> None:
    plan, cell, manifest = _inputs()
    manifest.cases = tuple(reversed(manifest.cases))

    with pytest.raises(FullProgressAdapterError, match="question order"):
        adapt_full_progress(
            plan=plan,
            cell=cell,
            case_manifest=manifest,
            progress=_progress(),
        )


def test_progress_adapter_requires_all_sixty_terminal_results() -> None:
    plan, cell, manifest = _inputs()
    progress = _progress().model_copy(update={"results": _progress().results[:-1]})

    with pytest.raises(FullProgressAdapterError, match="60 terminal results"):
        adapt_full_progress(
            plan=plan,
            cell=cell,
            case_manifest=manifest,
            progress=progress,
        )


@pytest.mark.parametrize(
    ("owner", "field", "changed"),
    (
        ("plan", "resolved_plan_hash", _sha("other-plan")),
        ("cell", "provider_id", "mem0"),
        ("manifest", "manifest_hash", _sha("other-manifest")),
    ),
)
def test_progress_adapter_rejects_identity_drift(
    owner: str,
    field: str,
    changed: str,
) -> None:
    plan, cell, manifest = _inputs()
    target = {"plan": plan, "cell": cell, "manifest": manifest}[owner]
    setattr(target, field, changed)

    with pytest.raises(FullProgressAdapterError, match="identity"):
        adapt_full_progress(
            plan=plan,
            cell=cell,
            case_manifest=manifest,
            progress=_progress(),
        )


def test_progress_cell_reuses_accuracy_context_indexing_and_latency_reducers() -> None:
    from oamb.reporting.comparison_project import _progress_cell_document

    plan, cell, manifest = _inputs()
    adapted = adapt_full_progress(
        plan=plan,
        cell=cell,
        case_manifest=manifest,
        progress=_progress(),
    )

    document = _progress_cell_document(plan, adapted)

    assert document["case_count"] == 60
    assert document["completed_case_count"] == 60
    assert document["judged_case_count"] == 60
    assert document["judged_numerator"] == 60
    assert document["accuracy"]["all_60"]["denominator"] == 60
    assert document["accounting"]["answer_visible_context_tokens"] == {
        "status": "measured_complete",
        "case_count": 60,
        "measured_case_count": 60,
        "total": 60,
        "mean": "1",
    }
    indexing = document["accounting"]["tokens"]["indexing"]
    assert indexing["supplier_usage_coverage"]["status"] == "measured_complete"
    assert indexing["totals"]["supplier_reported_total_tokens"]["value"] == 60
    assert document["observed_time"]["indexing_ready"]["count"] == 60
    assert document["observed_time"]["provider_request"]["count"] == 60


def test_nonjudged_progress_is_terminal_but_disables_pairwise_accuracy() -> None:
    from oamb.reporting.comparison_project import _pair_document, _progress_cell_document

    plan, cell, manifest = _inputs()
    mixed = _progress_cell_document(
        plan,
        adapt_full_progress(
            plan=plan,
            cell=cell,
            case_manifest=manifest,
            progress=_progress_with_model_exhaustion(),
        ),
    )
    judged = _progress_cell_document(
        plan,
        adapt_full_progress(
            plan=plan,
            cell=cell,
            case_manifest=manifest,
            progress=_progress(),
        ),
    )

    assert mixed["completed_case_count"] == 60
    assert mixed["judged_case_count"] == 58
    assert mixed["judged_denominator"] == 58
    assert mixed["results"][-2]["evaluation_disposition"] == "unjudged"
    assert mixed["results"][-1]["evaluation_disposition"] == "not_run"

    pair = _pair_document(plan, mixed, judged)

    assert pair["comparable"] is False
    assert pair["paired_accuracy"] == "unavailable"
    assert not any("code revision" in item for item in pair["limitations"])


def test_three_closed_progress_files_build_one_deterministic_180_result_report(
    tmp_path: Path,
) -> None:
    from oamb.reporting.comparison_project import build_full_progress_comparison_project
    from tests.reporting.test_comparison_project import _lme60_plan

    plan = _lme60_plan()
    manifest = SimpleNamespace(
        manifest_hash=plan.dataset.case_manifest_hash,
        workload_id=plan.dataset.workload_id,
        cases=tuple(
            SimpleNamespace(raw_question_id=question_id, case_manifest_entry_id=case_id)
            for question_id, case_id in zip(QUESTION_IDS, CASE_IDS, strict=True)
        ),
    )
    progresses = {
        cell.cell_id: _progress(
            resolved_plan_hash=plan.resolved_plan_hash,
            cell_id=cell.cell_id,
            provider_id=cell.provider_id,
            workload_id=cell.workload_id,
            case_manifest_hash=cell.case_manifest_hash,
        )
        for cell in plan.cells
    }

    first = build_full_progress_comparison_project(
        plan,
        progresses,
        case_manifest=manifest,
        output_root=tmp_path / "first",
    )
    second = build_full_progress_comparison_project(
        plan,
        progresses,
        case_manifest=manifest,
        output_root=tmp_path / "second",
    )

    report = json.loads(first.export_path.read_bytes())
    assert report["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 60,
        "provider_specific_result_count": 180,
    }
    assert len(report["questions"]) == 60
    assert len(report["comparisons"]) == 3
    assert first.export_path.read_bytes() == second.export_path.read_bytes()
    assert first.html_path.read_bytes() == second.html_path.read_bytes()
