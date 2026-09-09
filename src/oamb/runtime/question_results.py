"""Ordinary per-question results shared by execution, resume, and reporting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from oamb.artifacts.atomic import (
    atomic_replace_bytes,
    atomic_write_bytes,
    read_regular_file,
)
from oamb.contracts.base import NonEmptyStr, NonNegativeInt, PositiveInt, StrictContract
from oamb.contracts.ids import canonical_json_bytes


class QuestionResultsError(ValueError):
    """A provider result file is malformed or does not match its manifest."""


class ResultMeasurement(StrictContract):
    schema_name: Literal["result_measurement"] = "result_measurement"
    schema_version: Literal[1] = 1
    status: Literal["measured"]
    value: NonNegativeInt


class UnavailableResultMeasurement(StrictContract):
    schema_name: Literal["result_measurement"] = "result_measurement"
    schema_version: Literal[1] = 1
    status: Literal["unavailable"]
    reason: NonEmptyStr


class ResultText(StrictContract):
    schema_name: Literal["result_text"] = "result_text"
    schema_version: Literal[1] = 1
    status: Literal["available"]
    value: str


class UnavailableResultText(StrictContract):
    schema_name: Literal["result_text"] = "result_text"
    schema_version: Literal[1] = 1
    status: Literal["unavailable"]
    reason: NonEmptyStr


class ResultIngestion(StrictContract):
    schema_name: Literal["result_ingestion"] = "result_ingestion"
    schema_version: Literal[1] = 1
    status: Literal["sealed", "partial"]
    intended_source_count: NonNegativeInt
    accepted_source_count: NonNegativeInt
    skipped_source_count: NonNegativeInt
    partial: bool
    indexing_tokens: ResultMeasurement | UnavailableResultMeasurement
    indexing_token_coverage: Literal["measured_complete", "measured_partial", "unavailable"]
    indexing_ready_latency_microseconds: ResultMeasurement

    @model_validator(mode="after")
    def counts_close(self) -> Self:
        if self.accepted_source_count + self.skipped_source_count != self.intended_source_count:
            raise ValueError("result ingestion source counts do not close")
        if self.partial != (self.skipped_source_count > 0):
            raise ValueError("result ingestion partial status differs from skipped sources")
        if self.status == "sealed" and self.partial:
            raise ValueError("sealed result ingestion cannot be partial")
        if self.status == "partial" and not self.partial:
            raise ValueError("partial result ingestion requires skipped sources")
        if (self.indexing_tokens.status == "unavailable") != (
            self.indexing_token_coverage == "unavailable"
        ):
            raise ValueError("indexing token value and coverage availability differ")
        return self


class ResultRetrieval(StrictContract):
    schema_name: Literal["result_retrieval"] = "result_retrieval"
    schema_version: Literal[1] = 1
    status: Literal["succeeded"]
    visible_context: ResultText
    visible_context_byte_count: ResultMeasurement
    visible_context_token_count: ResultMeasurement
    native_candidate_count: ResultMeasurement
    visible_kept_count: ResultMeasurement
    visible_dropped_count: ResultMeasurement
    visible_truncated_count: ResultMeasurement
    request_latency_microseconds: ResultMeasurement
    generation_proof_disposition: Literal[
        "runtime_verified", "build_provenance_verified", "unattested", "unsupported"
    ]

    @model_validator(mode="after")
    def visible_context_and_candidates_close(self) -> Self:
        if len(self.visible_context.value.encode("utf-8")) != self.visible_context_byte_count.value:
            raise ValueError("result visible context byte count does not close")
        if self.visible_kept_count.value + self.visible_dropped_count.value != (
            self.native_candidate_count.value
        ):
            raise ValueError("result retrieval candidate counts do not close")
        if self.visible_truncated_count.value > self.visible_kept_count.value:
            raise ValueError("result truncated count exceeds kept candidates")
        return self


class ResultAnswer(StrictContract):
    schema_name: Literal["result_answer"] = "result_answer"
    schema_version: Literal[1] = 1
    status: Literal["parsed"]
    parsed_answer: ResultText
    protocol_disposition: Literal["normal_stop"]


class FailedResultAnswer(StrictContract):
    schema_name: Literal["result_answer"] = "result_answer"
    schema_version: Literal[1] = 1
    status: Literal["failed"]
    parsed_answer: UnavailableResultText
    protocol_disposition: Literal["output_contract_error"]


class JudgedResultEvaluation(StrictContract):
    schema_name: Literal["result_evaluation"] = "result_evaluation"
    schema_version: Literal[1] = 1
    disposition: Literal["judged"]
    metric_id: Literal["lme-judged-accuracy-v1"]
    numerator: NonNegativeInt
    denominator: Literal[1]
    judge_decision: Literal["yes", "no"]

    @model_validator(mode="after")
    def binary_decision_closes(self) -> Self:
        if self.numerator not in {0, 1}:
            raise ValueError("judged result numerator must be binary")
        expected = "yes" if self.numerator == 1 else "no"
        if self.judge_decision != expected:
            raise ValueError("judge decision differs from the result numerator")
        return self


class UnjudgedResultEvaluation(StrictContract):
    schema_name: Literal["result_evaluation"] = "result_evaluation"
    schema_version: Literal[1] = 1
    disposition: Literal["unjudged"]
    reason: NonEmptyStr


class NotRunResultEvaluation(StrictContract):
    schema_name: Literal["result_evaluation"] = "result_evaluation"
    schema_version: Literal[1] = 1
    disposition: Literal["not_run"]
    reason: NonEmptyStr


class ResultAttemptCounts(StrictContract):
    schema_name: Literal["result_attempt_counts"] = "result_attempt_counts"
    schema_version: Literal[1] = 1
    ingestion: PositiveInt
    retrieval: PositiveInt
    answer: PositiveInt
    judge: NonNegativeInt
    final_failure_class: Literal["none", "output_contract_error"]


class JudgedQuestionResult(StrictContract):
    schema_name: Literal["question_result"] = "question_result"
    schema_version: Literal[1] = 1
    terminal_status: Literal["judged"]
    question_id: NonEmptyStr
    ingestion: ResultIngestion
    retrieval: ResultRetrieval
    answer: ResultAnswer
    evaluation: JudgedResultEvaluation
    attempts: ResultAttemptCounts

    @model_validator(mode="after")
    def attempts_close(self) -> Self:
        if self.attempts.judge == 0:
            raise ValueError("judged result requires a judge attempt")
        if self.attempts.final_failure_class != "none":
            raise ValueError("judged result cannot retain a final failure")
        return self


class UnjudgedQuestionResult(StrictContract):
    schema_name: Literal["question_result"] = "question_result"
    schema_version: Literal[1] = 1
    terminal_status: Literal["unjudged"]
    question_id: NonEmptyStr
    ingestion: ResultIngestion
    retrieval: ResultRetrieval
    answer: ResultAnswer
    evaluation: UnjudgedResultEvaluation
    attempts: ResultAttemptCounts
    failure_stage: Literal["judge"]
    failure_kind: Literal["output_contract_error"]
    failure_reason: NonEmptyStr

    @model_validator(mode="after")
    def exhausted_judge_attempts_close(self) -> Self:
        if self.attempts.judge != 6:
            raise ValueError("unjudged result requires six judge attempts")
        if self.attempts.final_failure_class != "output_contract_error":
            raise ValueError("unjudged result requires an output-contract failure")
        if self.evaluation.reason != self.failure_reason:
            raise ValueError("unjudged evaluation and failure reasons differ")
        return self


class AnswerFailedQuestionResult(StrictContract):
    schema_name: Literal["question_result"] = "question_result"
    schema_version: Literal[1] = 1
    terminal_status: Literal["answer_failed"]
    question_id: NonEmptyStr
    ingestion: ResultIngestion
    retrieval: ResultRetrieval
    answer: FailedResultAnswer
    evaluation: NotRunResultEvaluation
    attempts: ResultAttemptCounts
    failure_stage: Literal["answer"]
    failure_kind: Literal["output_contract_error"]
    failure_reason: NonEmptyStr

    @model_validator(mode="after")
    def exhausted_answer_attempts_close(self) -> Self:
        if self.attempts.answer != 6:
            raise ValueError("answer-failed result requires six answer attempts")
        if self.attempts.judge != 0:
            raise ValueError("answer-failed result cannot have a judge attempt")
        if self.attempts.final_failure_class != "output_contract_error":
            raise ValueError("answer-failed result requires an output-contract failure")
        if self.answer.parsed_answer.reason != self.failure_reason:
            raise ValueError("failed answer and result failure reasons differ")
        return self


QuestionResult = Annotated[
    JudgedQuestionResult | UnjudgedQuestionResult | AnswerFailedQuestionResult,
    Field(discriminator="terminal_status"),
]
_QUESTION_RESULTS_ADAPTER = TypeAdapter(dict[str, QuestionResult])
RESULT_PROVIDER_IDS = ("hindsight", "mem0", "openviking")


def provider_result_path(root: Path, provider_id: str) -> Path:
    """Return one provider's only ordinary result-file path."""

    if provider_id not in RESULT_PROVIDER_IDS:
        raise QuestionResultsError(f"unsupported result provider: {provider_id}")
    return Path(root) / f"{provider_id}.json"


def load_question_results(
    path: Path,
    *,
    ordered_question_ids: tuple[str, ...],
) -> dict[str, QuestionResult]:
    """Load one provider's ordinary results against the stored-plan manifest."""

    try:
        content = read_regular_file(path)
        json.loads(
            content,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        parsed = _QUESTION_RESULTS_ADAPTER.validate_json(content)
    except Exception as exc:
        raise QuestionResultsError(f"question results cannot parse or validate: {path}") from exc
    manifest_ids = frozenset(ordered_question_ids)
    if len(manifest_ids) != len(ordered_question_ids):
        raise QuestionResultsError("stored manifest question IDs must be unique")
    for question_id, result in parsed.items():
        if question_id not in manifest_ids:
            raise QuestionResultsError(
                f"question result is outside the stored manifest: {question_id}"
            )
        if result.question_id != question_id:
            raise QuestionResultsError(f"question result key differs from its value: {question_id}")
    return {
        question_id: parsed[question_id]
        for question_id in ordered_question_ids
        if question_id in parsed
    }


def remaining_question_ids(
    results: Mapping[str, QuestionResult],
    ordered_question_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Subtract saved question IDs while preserving manifest order."""

    return tuple(question_id for question_id in ordered_question_ids if question_id not in results)


def initialize_question_results(path: Path) -> None:
    """Create one empty provider result file without replacing existing bytes."""

    result = atomic_write_bytes(path, b"{}", trusted_root=path.parent)
    if not result.created:
        raise FileExistsError(f"question results already exist: {path}")


def add_question_result(
    path: Path,
    result: QuestionResult,
    *,
    ordered_question_ids: tuple[str, ...],
) -> dict[str, QuestionResult]:
    """Atomically add one terminal result without overwriting a saved question."""

    current = load_question_results(path, ordered_question_ids=ordered_question_ids)
    if result.question_id in current:
        raise QuestionResultsError(f"question result already exists: {result.question_id}")
    updated = {**current, result.question_id: result}
    order = {question_id: index for index, question_id in enumerate(ordered_question_ids)}
    try:
        ordered = dict(sorted(updated.items(), key=lambda item: order[item[0]]))
    except KeyError as exc:
        raise QuestionResultsError(
            f"question result is outside the stored manifest: {result.question_id}"
        ) from exc
    content = canonical_json_bytes(
        {question_id: item.model_dump(mode="json") for question_id, item in ordered.items()}
    )
    atomic_replace_bytes(path, content, trusted_root=path.parent)
    return load_question_results(path, ordered_question_ids=ordered_question_ids)


def require_complete_question_results(
    results: Mapping[str, QuestionResult],
    ordered_question_ids: tuple[str, ...],
) -> Mapping[str, QuestionResult]:
    """Require one result for every question in the stored-plan manifest."""

    if tuple(results) != ordered_question_ids:
        raise QuestionResultsError("final report requires all manifest questions")
    return results


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate question-results JSON key: {key}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid question-results JSON constant: {value}")
