"""Project one freshly sealed native case into an ordinary question result."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast

from oamb.artifacts.atomic import read_regular_file
from oamb.config.doctor import CellSpec, ResolvedPlan
from oamb.contracts.evidence import (
    AttemptRecordV2,
    AttemptRecordV4,
    CaseRecordV3,
    HistoryAttemptRecord,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
)
from oamb.contracts.specifications import CaseManifest
from oamb.reporting.comparison_project import (
    _duration_interval,
    _indexing_token_stage_document,
    _judge_decision,
    _model_output_text,
    _retrieval_runtime_proof,
    _visible_context_text,
)
from oamb.runtime.question_results import (
    AnswerFailedQuestionResult,
    FailedResultAnswer,
    JudgedQuestionResult,
    JudgedResultEvaluation,
    NotRunResultEvaluation,
    QuestionResult,
    ResultAnswer,
    ResultAttemptCounts,
    ResultIngestion,
    ResultMeasurement,
    ResultRetrieval,
    ResultText,
    UnavailableResultMeasurement,
    UnavailableResultText,
    UnjudgedQuestionResult,
    UnjudgedResultEvaluation,
)

NativePlanRecord = IngestionPlanRecordV2 | IngestionPlanRecordV3
NativeAttemptRecord = AttemptRecordV2 | AttemptRecordV4


class NativeResultProjectionError(ValueError):
    """A terminal native case cannot safely enter ordinary results."""


def project_native_question_result(
    *,
    plan: ResolvedPlan,
    cell: CellSpec,
    capsule_root: Path,
    case_manifest: CaseManifest,
    plan_record: NativePlanRecord,
    case_record: CaseRecordV3,
    attempts: Mapping[str, NativeAttemptRecord],
    history_attempts: tuple[HistoryAttemptRecord, ...],
) -> QuestionResult | None:
    """Return a terminal entry, or ``None`` for a retryable/nonterminal provider error."""

    question_by_case = {
        item.case_manifest_entry_id: item.raw_question_id for item in case_manifest.cases
    }
    question_id = question_by_case.get(case_record.case_manifest_entry_id)
    if question_id is None:
        raise NativeResultProjectionError("terminal case is outside the frozen manifest")
    if case_record.ingestion_occurrence_id != plan_record.ingestion_occurrence_id:
        raise NativeResultProjectionError("terminal case and ingestion occurrence differ")
    relevant_histories = tuple(
        item
        for item in history_attempts
        if item.ingestion_occurrence_id == plan_record.ingestion_occurrence_id
    )
    related_attempt_ids = tuple(
        dict.fromkeys(
            (
                *plan_record.attempt_ids,
                *case_record.attempt_ids,
                *(
                    attempt_id
                    for history in relevant_histories
                    for attempt_id in history.operation_attempt_ids
                ),
            )
        )
    )
    try:
        selected_attempts = tuple(attempts[attempt_id] for attempt_id in related_attempt_ids)
    except KeyError as exc:
        raise NativeResultProjectionError("terminal attempt is unavailable in memory") from exc
    attempt_documents = tuple(item.model_dump(mode="json") for item in selected_attempts)
    plan_document = plan_record.model_dump(mode="json")
    case_document = case_record.model_dump(mode="json")
    history_documents = tuple(item.model_dump(mode="json") for item in relevant_histories)
    usage_ids = tuple(dict.fromkeys((*plan_record.usage_record_ids, *case_record.usage_record_ids)))
    token_usage = tuple(
        _load_source_document(capsule_root, "usage", usage_id) for usage_id in usage_ids
    )
    raw_references = tuple(
        dict.fromkeys(
            (
                *plan_record.readiness_evidence_refs,
                case_record.visible_evidence_raw_ref,
                case_record.retrieval_request_raw_ref,
                case_record.answer_raw_ref,
                case_record.evaluation_raw_ref,
            )
        )
    )
    raw_payloads = {
        reference: _load_raw_payload(capsule_root, reference)
        for reference in raw_references
        if isinstance(reference, str)
    }
    snapshot = SimpleNamespace(
        cell=cell,
        ingestion_plans=(plan_document,),
        attempts=attempt_documents,
        history_attempts=history_documents,
        token_usage=token_usage,
        raw_payloads=raw_payloads,
    )
    indexing = _indexing_token_stage_document(plan, cast(Any, snapshot))
    indexing_coverage = indexing["supplier_usage_coverage"]["status"]
    indexing_value = indexing["totals"]["supplier_reported_total_tokens"]["value"]
    if isinstance(indexing_value, int) and not isinstance(indexing_value, bool):
        indexing_tokens: ResultMeasurement | UnavailableResultMeasurement = ResultMeasurement(
            status="measured", value=indexing_value
        )
    elif indexing_value == "unavailable" and indexing_coverage == "unavailable":
        indexing_tokens = UnavailableResultMeasurement(
            status="unavailable",
            reason="supplier indexing token usage was unavailable in validated source evidence",
        )
    else:
        raise NativeResultProjectionError("indexing token projection is inconsistent")

    visible_context = _visible_context_text(cast(Any, snapshot), case_document)
    ingestion_attempts = tuple(
        item
        for item in attempt_documents
        if item.get("parent_id") == plan_record.ingestion_occurrence_id
        and item.get("stage") == "memory_ingest"
    )
    case_attempts = tuple(
        attempts[attempt_id].model_dump(mode="json") for attempt_id in case_record.attempt_ids
    )
    retrieval_attempts = tuple(
        item for item in case_attempts if item.get("stage") == "memory_query"
    )
    answer_attempts = tuple(item for item in case_attempts if item.get("stage") == "answer")
    judge_attempts = tuple(item for item in case_attempts if item.get("stage") == "judge")
    if not ingestion_attempts or not retrieval_attempts or not answer_attempts:
        raise NativeResultProjectionError("terminal stage attempt inventory is incomplete")
    answer_outer_attempts = _model_outer_attempt_count(
        answer_attempts,
        max_transport_retries=plan.execution.model_transport_max_retries,
    )
    judge_outer_attempts = _model_outer_attempt_count(
        judge_attempts,
        max_transport_retries=plan.execution.model_transport_max_retries,
    )
    retrieval_latency = _one_success_duration(retrieval_attempts, "retrieval")
    indexing_ready_latency = _indexing_ready_duration(
        attempt_documents,
        plan_document,
    )
    proof = _retrieval_runtime_proof(cell, (case_document,), raw_payloads)
    if proof == "unavailable":
        proof = "unattested"
    if proof not in {
        "runtime_verified",
        "build_provenance_verified",
        "unattested",
        "unsupported",
    }:
        raise NativeResultProjectionError("retrieval generation proof is invalid")
    proof_disposition = cast(
        Literal[
            "runtime_verified",
            "build_provenance_verified",
            "unattested",
            "unsupported",
        ],
        proof,
    )

    intended = plan_record.ordered_source_unit_ids
    accepted = plan_record.accepted_source_unit_ids
    skipped = plan_record.skipped_source_unit_ids
    ingestion = ResultIngestion(
        status="partial" if skipped else "sealed",
        intended_source_count=len(intended),
        accepted_source_count=len(accepted),
        skipped_source_count=len(skipped),
        partial=bool(skipped),
        indexing_tokens=indexing_tokens,
        indexing_token_coverage=indexing_coverage,
        indexing_ready_latency_microseconds=ResultMeasurement(
            status="measured", value=indexing_ready_latency
        ),
    )
    retrieval = ResultRetrieval(
        status="succeeded",
        visible_context=ResultText(status="available", value=visible_context),
        visible_context_byte_count=ResultMeasurement(
            status="measured", value=_required_case_int(case_record.visible_evidence_byte_count)
        ),
        visible_context_token_count=ResultMeasurement(
            status="measured", value=_required_case_int(case_record.visible_evidence_token_count)
        ),
        native_candidate_count=ResultMeasurement(
            status="measured", value=case_record.native_candidate_count
        ),
        visible_kept_count=ResultMeasurement(
            status="measured", value=case_record.visible_kept_count
        ),
        visible_dropped_count=ResultMeasurement(
            status="measured", value=case_record.visible_dropped_count
        ),
        visible_truncated_count=ResultMeasurement(
            status="measured", value=case_record.visible_truncated_count
        ),
        request_latency_microseconds=ResultMeasurement(status="measured", value=retrieval_latency),
        generation_proof_disposition=proof_disposition,
    )
    attempt_counts = ResultAttemptCounts(
        ingestion=len(ingestion_attempts),
        retrieval=len(retrieval_attempts),
        answer=answer_outer_attempts,
        judge=judge_outer_attempts,
        final_failure_class=(
            "none" if case_record.state == "completed" else "output_contract_error"
        ),
    )
    if case_record.state == "completed":
        return _judged_entry(
            snapshot=snapshot,
            question_id=question_id,
            case=case_document,
            ingestion=ingestion,
            retrieval=retrieval,
            attempts=attempt_counts,
        )
    if case_record.state != "error" or case_record.error_stage not in {"answer", "judge"}:
        return None
    stage_attempts = answer_attempts if case_record.error_stage == "answer" else judge_attempts
    stage_outer_attempts = (
        answer_outer_attempts if case_record.error_stage == "answer" else judge_outer_attempts
    )
    if stage_outer_attempts != plan.execution.model_max_attempts:
        return None
    final_attempt = max(stage_attempts, key=lambda item: int(item["ordinal"]))
    receipt = _load_source_document(
        capsule_root,
        "attempt-receipts",
        str(final_attempt["attempt_id"]),
    )
    if receipt.get("failure_kind") != "output_contract_error":
        return None
    failure_reason = f"{case_record.error_stage} output remained malformed after correction"
    if case_record.error_stage == "answer":
        return AnswerFailedQuestionResult(
            terminal_status="answer_failed",
            question_id=question_id,
            ingestion=ingestion,
            retrieval=retrieval,
            answer=FailedResultAnswer(
                status="failed",
                parsed_answer=UnavailableResultText(status="unavailable", reason=failure_reason),
                protocol_disposition="output_contract_error",
            ),
            evaluation=NotRunResultEvaluation(
                disposition="not_run",
                reason="answer did not satisfy its output contract",
            ),
            attempts=attempt_counts,
            failure_stage="answer",
            failure_kind="output_contract_error",
            failure_reason=failure_reason,
        )
    answer = _parsed_answer(snapshot, case_document)
    return UnjudgedQuestionResult(
        terminal_status="unjudged",
        question_id=question_id,
        ingestion=ingestion,
        retrieval=retrieval,
        answer=ResultAnswer(
            status="parsed",
            parsed_answer=ResultText(status="available", value=answer),
            protocol_disposition="normal_stop",
        ),
        evaluation=UnjudgedResultEvaluation(disposition="unjudged", reason=failure_reason),
        attempts=attempt_counts,
        failure_stage="judge",
        failure_kind="output_contract_error",
        failure_reason=failure_reason,
    )


def _judged_entry(
    *,
    snapshot: Any,
    question_id: str,
    case: dict[str, Any],
    ingestion: ResultIngestion,
    retrieval: ResultRetrieval,
    attempts: ResultAttemptCounts,
) -> JudgedQuestionResult:
    if (
        case.get("evaluation_disposition") != "judged"
        or case.get("metric_id") != "lme-judged-accuracy-v1"
        or case.get("metric_numerator") not in {0, 1}
        or case.get("metric_denominator") != 1
    ):
        raise NativeResultProjectionError("completed LME case is not judged")
    evaluation_reference = case.get("evaluation_raw_ref")
    evaluation_payload = snapshot.raw_payloads.get(evaluation_reference)
    if evaluation_payload is None:
        raise NativeResultProjectionError("judge evidence is unavailable")
    return JudgedQuestionResult(
        terminal_status="judged",
        question_id=question_id,
        ingestion=ingestion,
        retrieval=retrieval,
        answer=ResultAnswer(
            status="parsed",
            parsed_answer=ResultText(status="available", value=_parsed_answer(snapshot, case)),
            protocol_disposition="normal_stop",
        ),
        evaluation=JudgedResultEvaluation(
            disposition="judged",
            metric_id="lme-judged-accuracy-v1",
            numerator=case["metric_numerator"],
            denominator=1,
            judge_decision=cast(
                Literal["yes", "no"],
                _judge_decision(evaluation_payload, case),
            ),
        ),
        attempts=attempts,
    )


def _parsed_answer(snapshot: Any, case: dict[str, Any]) -> str:
    answer_reference = case.get("answer_raw_ref")
    answer_hash = case.get("parsed_answer_sha256")
    payload = snapshot.raw_payloads.get(answer_reference)
    if payload is None or not isinstance(answer_hash, str):
        raise NativeResultProjectionError("parsed answer evidence is unavailable")
    return _model_output_text(payload, expected_sha256=answer_hash)


def _one_success_duration(attempts: tuple[dict[str, Any], ...], stage: str) -> int:
    successful = tuple(
        interval
        for attempt in attempts
        if attempt.get("outcome") == "succeeded"
        and (interval := _duration_interval(attempt)) is not None
    )
    if len(successful) != 1:
        raise NativeResultProjectionError(f"{stage} latency does not have one successful attempt")
    return successful[0][2]


def _model_outer_attempt_count(
    attempts: tuple[dict[str, Any], ...],
    *,
    max_transport_retries: int,
) -> int:
    """Derive model outer attempts while collapsing physical transport retries."""

    if type(max_transport_retries) is not int or max_transport_retries < 0:
        raise NativeResultProjectionError("model transport retry limit is invalid")
    if not attempts:
        return 0
    ordered = sorted(attempts, key=lambda item: int(item["ordinal"]))
    group_lengths: list[int] = []
    seen_hashes: set[str] = set()
    current_hash: str | None = None
    for attempt in ordered:
        request_hash = attempt.get("request_messages_sha256")
        if not isinstance(request_hash, str) or not request_hash:
            raise NativeResultProjectionError("model attempt lacks its request messages hash")
        if request_hash != current_hash:
            if request_hash in seen_hashes:
                raise NativeResultProjectionError("model correction request hash reappeared")
            seen_hashes.add(request_hash)
            group_lengths.append(0)
            current_hash = request_hash
        group_lengths[-1] += 1
    physical_calls_per_outer = max_transport_retries + 1
    return sum(
        (length + physical_calls_per_outer - 1) // physical_calls_per_outer
        for length in group_lengths
    )


def _indexing_ready_duration(
    attempts: tuple[dict[str, Any], ...],
    plan: dict[str, Any],
) -> int:
    parent_id = plan.get("ingestion_occurrence_id")
    selected = tuple(item for item in attempts if item.get("parent_id") == parent_id)
    ingestions = tuple(
        interval
        for item in selected
        if item.get("stage") == "memory_ingest"
        and (interval := _duration_interval(item)) is not None
    )
    readiness = tuple(
        (item, interval)
        for item in selected
        if item.get("stage") == "memory_readiness"
        and (interval := _duration_interval(item)) is not None
    )
    if not ingestions or not readiness:
        raise NativeResultProjectionError("indexing readiness latency is unavailable")
    final, interval = max(readiness, key=lambda item: item[1][1])
    if final.get("outcome") != "succeeded":
        raise NativeResultProjectionError("final indexing readiness did not succeed")
    started = min(item[0] for item in ingestions)
    ended = interval[1]
    if any(item[1] > ended for item in ingestions):
        raise NativeResultProjectionError("indexing readiness precedes ingestion settlement")
    delta = ended - started
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _load_source_document(root: Path, collection: str, record_id: str) -> dict[str, Any]:
    path = root / "source" / collection / f"{record_id}.json"
    try:
        document = json.loads(read_regular_file(path))
    except (OSError, TypeError, ValueError) as exc:
        raise NativeResultProjectionError(
            f"sealed source record cannot be read: {collection}/{record_id}"
        ) from exc
    if not isinstance(document, dict):
        raise NativeResultProjectionError("sealed source record is not an object")
    return document


def _load_raw_payload(root: Path, reference: str) -> bytes:
    raw_root = root / "source" / "raw"
    plain = raw_root / reference
    compressed = raw_root / f"{reference}.json.gz"
    existing = tuple(
        path for path in (plain, compressed) if path.is_file() and not path.is_symlink()
    )
    if len(existing) != 1:
        raise NativeResultProjectionError(f"raw payload inventory differs: {reference}")
    content = read_regular_file(existing[0])
    try:
        payload = gzip.decompress(content) if existing[0] == compressed else content
    except (OSError, EOFError) as exc:
        raise NativeResultProjectionError(f"raw payload cannot decompress: {reference}") from exc
    if hashlib.sha256(payload).hexdigest() != reference:
        raise NativeResultProjectionError(f"raw payload hash differs: {reference}")
    return payload


def _required_case_int(value: int | None) -> int:
    if value is None:
        raise NativeResultProjectionError("terminal case measurement is unavailable")
    return value


__all__ = ["NativeResultProjectionError", "project_native_question_result"]
