from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from oamb.runtime import question_results as result_contract
from oamb.workloads.longmemeval import LME60_EXPECTED_QUESTION_IDS


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


QUESTION_IDS = LME60_EXPECTED_QUESTION_IDS
CASE_IDS = tuple(_sha(value) for value in QUESTION_IDS)
PLAN_HASH = _sha("plan")
MANIFEST_HASH = _sha("manifest")


def _results() -> dict[str, result_contract.QuestionResult]:
    measured = result_contract.ResultMeasurement(status="measured", value=1)
    ingestion = result_contract.ResultIngestion(
        status="sealed",
        intended_source_count=1,
        accepted_source_count=1,
        skipped_source_count=0,
        partial=False,
        indexing_tokens=measured,
        indexing_token_coverage="measured_complete",
        indexing_ready_latency_microseconds=measured,
    )
    retrieval = result_contract.ResultRetrieval(
        status="succeeded",
        visible_context=result_contract.ResultText(status="available", value="x"),
        visible_context_byte_count=measured,
        visible_context_token_count=measured,
        native_candidate_count=measured,
        visible_kept_count=measured,
        visible_dropped_count=result_contract.ResultMeasurement(status="measured", value=0),
        visible_truncated_count=result_contract.ResultMeasurement(status="measured", value=0),
        request_latency_microseconds=measured,
        generation_proof_disposition="runtime_verified",
    )
    answer = result_contract.ResultAnswer(
        status="parsed",
        parsed_answer=result_contract.ResultText(status="available", value="answer"),
        protocol_disposition="normal_stop",
    )
    evaluation = result_contract.JudgedResultEvaluation(
        disposition="judged",
        metric_id="lme-judged-accuracy-v1",
        numerator=1,
        denominator=1,
        judge_decision="yes",
    )
    attempts = result_contract.ResultAttemptCounts(
        ingestion=1,
        retrieval=1,
        answer=1,
        judge=1,
        final_failure_class="none",
    )
    entries = tuple(
        result_contract.JudgedQuestionResult(
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
    return dict(zip(QUESTION_IDS, entries, strict=True))


def _results_with_model_exhaustion() -> dict[str, result_contract.QuestionResult]:
    results = _results()
    judged = cast(result_contract.JudgedQuestionResult, results[QUESTION_IDS[0]])
    unjudged = result_contract.UnjudgedQuestionResult(
        terminal_status="unjudged",
        question_id=QUESTION_IDS[-2],
        ingestion=judged.ingestion,
        retrieval=judged.retrieval,
        answer=judged.answer,
        evaluation=result_contract.UnjudgedResultEvaluation(
            disposition="unjudged",
            reason="judge output remained malformed after correction",
        ),
        attempts=result_contract.ResultAttemptCounts(
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
    answer_failed = result_contract.AnswerFailedQuestionResult(
        terminal_status="answer_failed",
        question_id=QUESTION_IDS[-1],
        ingestion=judged.ingestion,
        retrieval=judged.retrieval,
        answer=result_contract.FailedResultAnswer(
            status="failed",
            parsed_answer=result_contract.UnavailableResultText(
                status="unavailable",
                reason="answer output remained malformed after correction",
            ),
            protocol_disposition="output_contract_error",
        ),
        evaluation=result_contract.NotRunResultEvaluation(
            disposition="not_run",
            reason="answer did not satisfy its output contract",
        ),
        attempts=result_contract.ResultAttemptCounts(
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
    return {**results, QUESTION_IDS[-2]: unjudged, QUESTION_IDS[-1]: answer_failed}


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


def test_results_cell_reuses_accuracy_context_indexing_and_latency_reducers() -> None:
    from oamb.reporting.comparison_project import _question_result_cell_document

    plan, cell, manifest = _inputs()

    document = _question_result_cell_document(plan, cell, manifest, _results())

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


def test_nonjudged_results_is_terminal_but_disables_pairwise_accuracy() -> None:
    from oamb.reporting.comparison_project import (
        _pair_document,
        _question_result_cell_document,
    )

    plan, cell, manifest = _inputs()
    mixed = _question_result_cell_document(plan, cell, manifest, _results_with_model_exhaustion())
    judged = _question_result_cell_document(plan, cell, manifest, _results())

    assert mixed["completed_case_count"] == 60
    assert mixed["judged_case_count"] == 58
    assert mixed["judged_denominator"] == 58
    assert mixed["results"][-2]["evaluation_disposition"] == "unjudged"
    assert mixed["results"][-1]["evaluation_disposition"] == "not_run"

    pair = _pair_document(plan, mixed, judged)

    assert pair["comparable"] is False
    assert pair["paired_accuracy"] == "unavailable"
    assert not any("code revision" in item for item in pair["limitations"])


def test_three_closed_results_files_build_one_deterministic_180_result_report(
    tmp_path: Path,
) -> None:
    from oamb.reporting.comparison_project import build_question_results_comparison_project
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
    results_by_cell = {cell.cell_id: _results() for cell in plan.cells}

    first = build_question_results_comparison_project(
        plan,
        results_by_cell,
        case_manifest=manifest,
        output_root=tmp_path / "first",
    )
    second = build_question_results_comparison_project(
        plan,
        results_by_cell,
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


def test_missing_provider_question_prevents_final_report_publication(tmp_path: Path) -> None:
    from oamb.reporting.comparison_project import (
        ComparisonProjectError,
        build_question_results_comparison_project,
    )
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
    results_by_cell = {cell.cell_id: _results() for cell in plan.cells}
    results_by_cell[plan.cells[0].cell_id].pop(QUESTION_IDS[-1])
    output_root = tmp_path / "report"

    with pytest.raises(ComparisonProjectError, match="requires all manifest questions"):
        build_question_results_comparison_project(
            plan,
            results_by_cell,
            case_manifest=manifest,
            output_root=output_root,
        )

    assert not output_root.exists()
