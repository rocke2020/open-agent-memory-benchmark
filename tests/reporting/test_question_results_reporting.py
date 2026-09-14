from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import BaseModel

from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.runtime import question_results as result_contract
from oamb.workloads.longmemeval import LME60_EXPECTED_QUESTION_IDS

LONGMEMEVAL_DATASET = (
    Path(__file__).resolve().parents[2]
    / "datasets"
    / "longmemeval-cleaned"
    / "longmemeval_s_cleaned.json"
)
REQUIRES_LONGMEMEVAL_DATASET = pytest.mark.skipif(
    not LONGMEMEVAL_DATASET.is_file(),
    reason="requires the ignored pinned LongMemEval-S dataset",
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


QUESTION_IDS = LME60_EXPECTED_QUESTION_IDS
CASE_IDS = tuple(_sha(value) for value in QUESTION_IDS)
PLAN_HASH = _sha("plan")
MANIFEST_HASH = _sha("manifest")


class _SerializableManifestCase(BaseModel):
    raw_question_id: str
    case_manifest_entry_id: str


class _SerializableCaseManifest(BaseModel):
    manifest_hash: str
    workload_id: str
    cases: tuple[_SerializableManifestCase, ...]


def _results(*, native_candidate_count: int = 1) -> dict[str, result_contract.QuestionResult]:
    measured = result_contract.ResultMeasurement(status="measured", value=1)
    candidates = result_contract.ResultMeasurement(status="measured", value=native_candidate_count)
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
        native_candidate_count=candidates,
        visible_kept_count=candidates,
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
    manifest = _SerializableCaseManifest(
        manifest_hash=plan.dataset.case_manifest_hash,
        workload_id=plan.dataset.workload_id,
        cases=tuple(
            _SerializableManifestCase(
                raw_question_id=question_id,
                case_manifest_entry_id=case_id,
            )
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
    cell_results = [result for cell in report["cells"] for result in cell["results"]]
    question_results = [
        result for question in report["questions"] for result in question["provider_results"]
    ]
    assert len(cell_results) == len(question_results) == 180
    assert all("model_answer" in result and "injected_context" in result for result in cell_results)
    assert all(
        "model_answer" not in result and "injected_context" not in result
        for result in question_results
    )
    assert first.export_path.read_bytes() == second.export_path.read_bytes()
    assert first.html_path.read_bytes() == second.html_path.read_bytes()
    html = first.html_path.read_text(encoding="utf-8")
    assert "Secondary accounting" not in html
    assert "Concise metric comparison" not in html
    assert "Analysis unavailable" not in html


def test_detailed_question_results_store_answer_and_context_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches detailed report content being duplicated in both result projections."""

    from oamb.reporting import comparison_project
    from tests.reporting.test_comparison_project import _lme60_plan

    plan = _lme60_plan()
    manifest = _SerializableCaseManifest(
        manifest_hash=plan.dataset.case_manifest_hash,
        workload_id=plan.dataset.workload_id,
        cases=tuple(
            _SerializableManifestCase(
                raw_question_id=question_id,
                case_manifest_entry_id=case_id,
            )
            for question_id, case_id in zip(QUESTION_IDS, CASE_IDS, strict=True)
        ),
    )
    results_by_cell = {cell.cell_id: _results() for cell in plan.cells}
    local_question_content = tuple(
        {
            "case_manifest_entry_id": case_id,
            "raw_question_id": question_id,
            "question_type": "single-session-user",
            "question": f"Question {index}",
            "gold_answer": f"Gold answer {index}",
            "answer_sessions": (),
            "has_answer_label_mismatch": False,
        }
        for index, (case_id, question_id) in enumerate(
            zip(CASE_IDS, QUESTION_IDS, strict=True),
            start=1,
        )
    )
    dataset_source = tmp_path / "dataset.json"
    dataset_source.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        comparison_project,
        "_load_local_question_content",
        lambda resolved_plan, source, case_manifest_bytes: local_question_content,
    )

    built = comparison_project.build_question_results_comparison_project(
        plan,
        results_by_cell,
        case_manifest=manifest,
        output_root=tmp_path / "detailed",
        dataset_source=dataset_source,
    )

    report = json.loads(built.export_path.read_bytes())
    cell_results = [result for cell in report["cells"] for result in cell["results"]]
    question_results = [
        result for question in report["questions"] for result in question["provider_results"]
    ]
    assert len(cell_results) == len(question_results) == 180
    assert all(
        "model_answer" not in result
        and "injected_context" not in result
        and "judge_decision" not in result
        and "answer_unavailable_reason" not in result
        for result in cell_results
    )
    assert all(
        result["model_answer"] == "answer"
        and result["injected_context"] == "x"
        and result["judge_decision"] == "yes"
        for result in question_results
    )


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


def _write_saved_result_snapshot(
    root: Path,
    *,
    provider_id: str,
    plan_bytes: bytes,
    native_candidate_count: int = 1,
    report_control_top_k: int | None = None,
) -> None:
    snapshot = root / f"{provider_id}-snapshot"
    results_root = snapshot / "results"
    results_root.mkdir(parents=True)
    (snapshot / "resolved-plan.json").write_bytes(plan_bytes)
    if report_control_top_k is not None:
        (snapshot / "report-control.json").write_bytes(
            canonical_json_bytes(
                {
                    "schema_name": "saved_result_report_control",
                    "schema_version": 1,
                    "provider_id": provider_id,
                    "retrieval_top_k": report_control_top_k,
                    "retrieval_top_k_scope": "report_comparison_normalization_ceiling",
                }
            )
        )
    (results_root / f"{provider_id}.json").write_bytes(
        canonical_json_bytes(
            {
                question_id: result.model_dump(mode="json")
                for question_id, result in _results(
                    native_candidate_count=native_candidate_count
                ).items()
            }
        )
    )


@REQUIRES_LONGMEMEVAL_DATASET
def test_saved_provider_results_use_report_control_when_candidates_fit_its_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches separately run results being relabeled as one uniform frozen plan."""

    from oamb.config.benchmark import load_benchmark_configuration
    from oamb.config.doctor import build_resolved_plan, resolved_plan_bytes
    from oamb.reporting import saved_results
    from tests.benchmark_configuration import MODEL_ENVIRONMENT

    configuration = load_benchmark_configuration(
        Path("configs/benchmark.yml"),
        model_environment=MODEL_ENVIRONMENT,
    )
    plan = build_resolved_plan(configuration)
    result_root = tmp_path / "saved-results"
    current_plan_bytes = resolved_plan_bytes(plan)
    legacy_document = json.loads(current_plan_bytes)
    legacy_document.pop("resolved_plan_hash")
    legacy_document["retrieval"].pop("top_k")
    legacy_document["cells"][0]["cell_spec_hash"] = _sha("legacy-hindsight-cell")
    legacy_plan_hash = canonical_sha256(["oamb-resolved-plan-initial-v1", legacy_document])
    legacy_document["resolved_plan_hash"] = legacy_plan_hash
    _write_saved_result_snapshot(
        result_root,
        provider_id="hindsight",
        plan_bytes=canonical_json_bytes(legacy_document),
        report_control_top_k=plan.retrieval.top_k,
    )
    _write_saved_result_snapshot(
        result_root,
        provider_id="mem0",
        plan_bytes=current_plan_bytes,
    )
    _write_saved_result_snapshot(
        result_root,
        provider_id="openviking",
        plan_bytes=current_plan_bytes,
    )
    source_hashes_before = {
        path.relative_to(result_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in result_root.rglob("*")
        if path.is_file()
    }
    model_env = tmp_path / ".env"
    model_env.write_text("ignored-by-test\n", encoding="utf-8")
    analysis_cache_root = tmp_path / "report-analysis-cache"
    analysis_calls: list[dict[str, object]] = []

    def analysis_generator(export: dict[str, object]) -> None:
        analysis_calls.append(export)
        return None

    captured_factory: dict[str, object] = {}

    def build_analysis_generator(**kwargs: object) -> object:
        captured_factory.update(kwargs)
        return analysis_generator

    monkeypatch.setattr(
        saved_results,
        "build_report_analysis_generator",
        build_analysis_generator,
        raising=False,
    )

    built = saved_results.build_saved_results_report(
        result_root=result_root,
        output_root=tmp_path / "comparison",
        analysis_model_env=model_env,
        analysis_cache_root=analysis_cache_root,
    )

    report = json.loads(built.export_path.read_bytes())
    sources = {item["provider_id"]: item for item in report["saved_result_sources"]}
    pairs = {
        frozenset((item["left_provider_id"], item["right_provider_id"])): item
        for item in report["comparisons"]
    }
    assert report["source_mode"] == "saved_provider_results"
    assert report["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 60,
        "provider_specific_result_count": 180,
    }
    assert sources["hindsight"]["resolved_plan_hash"] == legacy_plan_hash
    assert sources["hindsight"]["source_plan_retrieval_top_k"] == "unavailable"
    assert sources["hindsight"]["retrieval_top_k"] == plan.retrieval.top_k
    assert sources["hindsight"]["retrieval_top_k_source"] == "saved_report_control"
    assert sources["hindsight"]["observed_native_candidate_count_max"] == 1
    assert sources["hindsight"]["cell_spec_hash"] == _sha("legacy-hindsight-cell")
    assert sources["mem0"]["retrieval_top_k"] == plan.retrieval.top_k
    assert pairs[frozenset(("hindsight", "mem0"))]["comparable"] is True
    assert pairs[frozenset(("hindsight", "openviking"))]["comparable"] is True
    assert pairs[frozenset(("mem0", "openviking"))]["comparable"] is True
    assert report["accuracy_decision"]["status"] == "no_clear_accuracy_leader"
    assert captured_factory == {
        "plan": plan,
        "model_env_path": model_env,
        "cache_root": analysis_cache_root,
    }
    assert len(analysis_calls) == 1
    assert analysis_calls[0]["report_id"] == report["report_id"]
    html = built.html_path.read_text(encoding="utf-8")
    assert "separate saved provider results" in html
    assert "Portable result files do not independently reconstruct" in html
    assert "Runtime models were verified against this comparison" not in html
    assert report["controlled_comparison_warning"].startswith(
        "This report combines separately run provider-native results"
    )
    source_hashes_after = {
        path.relative_to(result_root): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in result_root.rglob("*")
        if path.is_file()
    }
    assert source_hashes_after == source_hashes_before


@REQUIRES_LONGMEMEVAL_DATASET
def test_saved_provider_results_reject_report_control_when_candidates_exceed_its_ceiling(
    tmp_path: Path,
) -> None:
    from oamb.config.benchmark import load_benchmark_configuration
    from oamb.config.doctor import build_resolved_plan, resolved_plan_bytes
    from oamb.reporting.saved_results import build_saved_results_report
    from tests.benchmark_configuration import MODEL_ENVIRONMENT

    plan = build_resolved_plan(
        load_benchmark_configuration(
            Path("configs/benchmark.yml"),
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    result_root = tmp_path / "saved-results"
    current_plan_bytes = resolved_plan_bytes(plan)
    legacy_document = json.loads(current_plan_bytes)
    legacy_document.pop("resolved_plan_hash")
    legacy_document["retrieval"].pop("top_k")
    legacy_document["cells"][0]["cell_spec_hash"] = _sha("legacy-cell")
    legacy_document["resolved_plan_hash"] = canonical_sha256(
        ["oamb-resolved-plan-initial-v1", legacy_document]
    )
    configured_ceiling = plan.retrieval.top_k
    _write_saved_result_snapshot(
        result_root,
        provider_id="hindsight",
        plan_bytes=canonical_json_bytes(legacy_document),
        native_candidate_count=configured_ceiling + 1,
        report_control_top_k=configured_ceiling,
    )
    for provider_id in ("mem0", "openviking"):
        _write_saved_result_snapshot(
            result_root,
            provider_id=provider_id,
            plan_bytes=current_plan_bytes,
        )

    built = build_saved_results_report(
        result_root=result_root,
        output_root=tmp_path / "comparison",
    )

    report = json.loads(built.export_path.read_bytes())
    pairs = {
        frozenset((item["left_provider_id"], item["right_provider_id"])): item
        for item in report["comparisons"]
    }
    for peer in ("mem0", "openviking"):
        pair = pairs[frozenset(("hindsight", peer))]
        assert pair["comparable"] is False
        assert any("retrieval" in reason for reason in pair["limitations"])


def test_saved_provider_results_reject_duplicate_provider_before_publication(
    tmp_path: Path,
) -> None:
    from oamb.config.benchmark import load_benchmark_configuration
    from oamb.config.doctor import build_resolved_plan, resolved_plan_bytes
    from oamb.reporting.saved_results import SavedResultsError, build_saved_results_report
    from tests.benchmark_configuration import MODEL_ENVIRONMENT

    plan = build_resolved_plan(
        load_benchmark_configuration(
            Path("configs/benchmark.yml"),
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    result_root = tmp_path / "saved-results"
    plan_bytes = resolved_plan_bytes(plan)
    for provider_id in ("hindsight", "mem0", "openviking"):
        _write_saved_result_snapshot(
            result_root,
            provider_id=provider_id,
            plan_bytes=plan_bytes,
        )
    duplicate = result_root / "duplicate-hindsight" / "results"
    duplicate.mkdir(parents=True)
    (duplicate.parent / "resolved-plan.json").write_bytes(plan_bytes)
    (duplicate / "hindsight.json").write_bytes(
        (result_root / "hindsight-snapshot/results/hindsight.json").read_bytes()
    )
    output_root = tmp_path / "comparison"

    with pytest.raises(SavedResultsError, match="duplicate saved result provider: hindsight"):
        build_saved_results_report(result_root=result_root, output_root=output_root)

    assert not output_root.exists()


def test_saved_provider_results_reject_plan_hash_drift_before_publication(
    tmp_path: Path,
) -> None:
    from oamb.config.benchmark import load_benchmark_configuration
    from oamb.config.doctor import build_resolved_plan, resolved_plan_bytes
    from oamb.reporting.saved_results import SavedResultsError, build_saved_results_report
    from tests.benchmark_configuration import MODEL_ENVIRONMENT

    plan = build_resolved_plan(
        load_benchmark_configuration(
            Path("configs/benchmark.yml"),
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    result_root = tmp_path / "saved-results"
    plan_bytes = resolved_plan_bytes(plan)
    for provider_id in ("hindsight", "mem0", "openviking"):
        _write_saved_result_snapshot(
            result_root,
            provider_id=provider_id,
            plan_bytes=plan_bytes,
        )
    hindsight_plan = result_root / "hindsight-snapshot/resolved-plan.json"
    drifted = json.loads(hindsight_plan.read_bytes())
    drifted["embedding_endpoint"]["ownership"] = "drifted"
    hindsight_plan.write_bytes(canonical_json_bytes(drifted))
    output_root = tmp_path / "comparison"

    with pytest.raises(SavedResultsError, match="resolved plan hash does not match"):
        build_saved_results_report(result_root=result_root, output_root=output_root)

    assert not output_root.exists()
