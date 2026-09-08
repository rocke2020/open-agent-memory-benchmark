"""Strict flat progress for resumable full benchmark cells."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import stat
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from oamb.artifacts.atomic import atomic_replace_bytes, atomic_write_bytes, read_regular_file
from oamb.contracts.base import NonEmptyStr, NonNegativeInt, PositiveInt, Sha256, StrictContract
from oamb.contracts.ids import canonical_json_bytes


class FullProgressError(ValueError):
    """Raised before dispatch when canonical progress cannot be reused."""


FULL_PROGRESS_PROVIDER_IDS = ("hindsight", "mem0", "openviking")


def canonical_full_progress_path(root: Path, provider_id: str) -> Path:
    """Return the only accepted filename for one frozen full-run provider."""

    if provider_id not in FULL_PROGRESS_PROVIDER_IDS:
        raise FullProgressError(f"unsupported progress provider: {provider_id}")
    return Path(root) / f"progress-{provider_id}.json"


@contextmanager
def acquire_full_resume_lock(path: Path) -> Iterator[Path]:
    """Hold one nonblocking process-owned POSIX lock for a full resume command."""

    target = Path(path).absolute()
    try:
        parent_metadata = target.parent.lstat()
    except OSError as exc:
        raise FullProgressError(f"full resume lock parent is unavailable: {target.parent}") from exc
    if not stat.S_ISDIR(parent_metadata.st_mode):
        raise FullProgressError("full resume lock parent must be a real directory")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as exc:
        raise FullProgressError(f"full resume lock cannot open: {target}") from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise FullProgressError("full resume lock must be a regular file")
        try:
            fcntl.lockf(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno in {errno.EACCES, errno.EAGAIN}:
                raise FullProgressError("another full resume command holds the lock") from exc
            raise FullProgressError(f"full resume lock cannot be acquired: {target}") from exc
        try:
            yield target
        finally:
            fcntl.lockf(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


class ProgressMeasurement(StrictContract):
    schema_name: Literal["progress_measurement"] = "progress_measurement"
    schema_version: Literal[1] = 1
    status: Literal["measured"]
    value: NonNegativeInt


class UnavailableProgressMeasurement(StrictContract):
    schema_name: Literal["progress_measurement"] = "progress_measurement"
    schema_version: Literal[1] = 1
    status: Literal["unavailable"]
    reason: NonEmptyStr


class ProgressText(StrictContract):
    schema_name: Literal["progress_text"] = "progress_text"
    schema_version: Literal[1] = 1
    status: Literal["available"]
    value: str


class UnavailableProgressText(StrictContract):
    schema_name: Literal["progress_text"] = "progress_text"
    schema_version: Literal[1] = 1
    status: Literal["unavailable"]
    reason: NonEmptyStr


class ProgressIngestion(StrictContract):
    schema_name: Literal["progress_ingestion"] = "progress_ingestion"
    schema_version: Literal[1] = 1
    status: Literal["sealed", "partial"]
    intended_source_count: NonNegativeInt
    accepted_source_count: NonNegativeInt
    skipped_source_count: NonNegativeInt
    partial: bool
    indexing_tokens: ProgressMeasurement | UnavailableProgressMeasurement
    indexing_token_coverage: Literal["measured_complete", "measured_partial", "unavailable"]
    indexing_ready_latency_microseconds: ProgressMeasurement

    @model_validator(mode="after")
    def counts_close(self) -> Self:
        if self.accepted_source_count + self.skipped_source_count != self.intended_source_count:
            raise ValueError("progress ingestion source counts do not close")
        if self.partial != (self.skipped_source_count > 0):
            raise ValueError("progress ingestion partial status differs from skipped sources")
        if self.status == "sealed" and self.partial:
            raise ValueError("sealed progress ingestion cannot be partial")
        if self.status == "partial" and not self.partial:
            raise ValueError("partial progress ingestion requires skipped sources")
        if (self.indexing_tokens.status == "unavailable") != (
            self.indexing_token_coverage == "unavailable"
        ):
            raise ValueError("indexing token value and coverage availability differ")
        return self


class ProgressRetrieval(StrictContract):
    schema_name: Literal["progress_retrieval"] = "progress_retrieval"
    schema_version: Literal[1] = 1
    status: Literal["succeeded"]
    visible_context: ProgressText
    visible_context_byte_count: ProgressMeasurement
    visible_context_token_count: ProgressMeasurement
    native_candidate_count: ProgressMeasurement
    visible_kept_count: ProgressMeasurement
    visible_dropped_count: ProgressMeasurement
    visible_truncated_count: ProgressMeasurement
    request_latency_microseconds: ProgressMeasurement
    generation_proof_disposition: Literal[
        "runtime_verified", "build_provenance_verified", "unattested", "unsupported"
    ]

    @model_validator(mode="after")
    def visible_context_and_candidates_close(self) -> Self:
        if len(self.visible_context.value.encode("utf-8")) != self.visible_context_byte_count.value:
            raise ValueError("progress visible context byte count does not close")
        if self.visible_kept_count.value + self.visible_dropped_count.value != (
            self.native_candidate_count.value
        ):
            raise ValueError("progress retrieval candidate counts do not close")
        if self.visible_truncated_count.value > self.visible_kept_count.value:
            raise ValueError("progress truncated count exceeds kept candidates")
        return self


class ProgressAnswer(StrictContract):
    schema_name: Literal["progress_answer"] = "progress_answer"
    schema_version: Literal[1] = 1
    status: Literal["parsed"]
    parsed_answer: ProgressText
    protocol_disposition: Literal["normal_stop"]


class FailedProgressAnswer(StrictContract):
    schema_name: Literal["progress_answer"] = "progress_answer"
    schema_version: Literal[1] = 1
    status: Literal["failed"]
    parsed_answer: UnavailableProgressText
    protocol_disposition: Literal["output_contract_error"]


class JudgedProgressEvaluation(StrictContract):
    schema_name: Literal["progress_evaluation"] = "progress_evaluation"
    schema_version: Literal[1] = 1
    disposition: Literal["judged"]
    metric_id: Literal["lme-judged-accuracy-v1"]
    numerator: NonNegativeInt
    denominator: Literal[1]
    judge_decision: Literal["yes", "no"]

    @model_validator(mode="after")
    def binary_decision_closes(self) -> Self:
        if self.numerator not in {0, 1}:
            raise ValueError("judged progress numerator must be binary")
        expected = "yes" if self.numerator == 1 else "no"
        if self.judge_decision != expected:
            raise ValueError("judge decision differs from the progress numerator")
        return self


class UnjudgedProgressEvaluation(StrictContract):
    schema_name: Literal["progress_evaluation"] = "progress_evaluation"
    schema_version: Literal[1] = 1
    disposition: Literal["unjudged"]
    reason: NonEmptyStr


class NotRunProgressEvaluation(StrictContract):
    schema_name: Literal["progress_evaluation"] = "progress_evaluation"
    schema_version: Literal[1] = 1
    disposition: Literal["not_run"]
    reason: NonEmptyStr


class ProgressAttemptCounts(StrictContract):
    schema_name: Literal["progress_attempt_counts"] = "progress_attempt_counts"
    schema_version: Literal[1] = 1
    ingestion: PositiveInt
    retrieval: PositiveInt
    answer: PositiveInt
    judge: NonNegativeInt
    final_failure_class: Literal["none", "output_contract_error"]


class JudgedFullProgressEntry(StrictContract):
    schema_name: Literal["full_progress_entry"] = "full_progress_entry"
    schema_version: Literal[1] = 1
    terminal_status: Literal["judged"]
    question_id: NonEmptyStr
    ingestion: ProgressIngestion
    retrieval: ProgressRetrieval
    answer: ProgressAnswer
    evaluation: JudgedProgressEvaluation
    attempts: ProgressAttemptCounts

    @model_validator(mode="after")
    def attempts_close(self) -> Self:
        if self.attempts.judge == 0:
            raise ValueError("judged progress requires a judge attempt")
        if self.attempts.final_failure_class != "none":
            raise ValueError("judged progress cannot retain a final failure")
        return self


class UnjudgedFullProgressEntry(StrictContract):
    schema_name: Literal["full_progress_entry"] = "full_progress_entry"
    schema_version: Literal[1] = 1
    terminal_status: Literal["unjudged"]
    question_id: NonEmptyStr
    ingestion: ProgressIngestion
    retrieval: ProgressRetrieval
    answer: ProgressAnswer
    evaluation: UnjudgedProgressEvaluation
    attempts: ProgressAttemptCounts
    failure_stage: Literal["judge"]
    failure_kind: Literal["output_contract_error"]
    failure_reason: NonEmptyStr

    @model_validator(mode="after")
    def exhausted_judge_attempts_close(self) -> Self:
        if self.attempts.judge != 6:
            raise ValueError("unjudged progress requires six judge attempts")
        if self.attempts.final_failure_class != "output_contract_error":
            raise ValueError("unjudged progress requires an output-contract failure")
        if self.evaluation.reason != self.failure_reason:
            raise ValueError("unjudged evaluation and failure reasons differ")
        return self


class ProviderFailedFullProgressEntry(StrictContract):
    schema_name: Literal["full_progress_entry"] = "full_progress_entry"
    schema_version: Literal[1] = 1
    terminal_status: Literal["provider_failed"]
    question_id: NonEmptyStr
    ingestion: ProgressIngestion
    retrieval: ProgressRetrieval
    answer: FailedProgressAnswer
    evaluation: NotRunProgressEvaluation
    attempts: ProgressAttemptCounts
    failure_stage: Literal["answer"]
    failure_kind: Literal["output_contract_error"]
    failure_reason: NonEmptyStr

    @model_validator(mode="after")
    def exhausted_answer_attempts_close(self) -> Self:
        if self.attempts.answer != 6:
            raise ValueError("provider-failed progress requires six answer attempts")
        if self.attempts.judge != 0:
            raise ValueError("provider-failed progress cannot have a judge attempt")
        if self.attempts.final_failure_class != "output_contract_error":
            raise ValueError("provider-failed progress requires an output-contract failure")
        if self.answer.parsed_answer.reason != self.failure_reason:
            raise ValueError("failed answer and progress failure reasons differ")
        return self


FullProgressEntry = Annotated[
    JudgedFullProgressEntry | UnjudgedFullProgressEntry | ProviderFailedFullProgressEntry,
    Field(discriminator="terminal_status"),
]


class FullProgress(StrictContract):
    schema_name: Literal["full_progress"] = "full_progress"
    schema_version: Literal[1] = 1
    resolved_plan_hash: Sha256
    cell_id: NonEmptyStr
    provider_id: NonEmptyStr
    workload_id: NonEmptyStr
    case_manifest_hash: Sha256
    ordered_question_ids: tuple[NonEmptyStr, ...]
    results: tuple[FullProgressEntry, ...]

    @model_validator(mode="after")
    def result_inventory_closes(self) -> Self:
        if len(self.ordered_question_ids) != 60:
            raise ValueError("full progress requires exactly 60 ordered question IDs")
        if len(set(self.ordered_question_ids)) != len(self.ordered_question_ids):
            raise ValueError("full progress question IDs must be unique")
        result_ids = tuple(item.question_id for item in self.results)
        if len(set(result_ids)) != len(result_ids):
            raise ValueError("full progress result IDs must be unique")
        expected_result_ids = tuple(
            question_id for question_id in self.ordered_question_ids if question_id in result_ids
        )
        if result_ids != expected_result_ids:
            raise ValueError("full progress results must be a manifest-ordered subset")
        return self

    @property
    def completed_question_ids(self) -> tuple[str, ...]:
        return tuple(item.question_id for item in self.results)

    @property
    def remaining_question_ids(self) -> tuple[str, ...]:
        completed = frozenset(self.completed_question_ids)
        return tuple(item for item in self.ordered_question_ids if item not in completed)


def empty_full_progress(
    *,
    resolved_plan_hash: str,
    cell_id: str,
    provider_id: str,
    workload_id: str,
    case_manifest_hash: str,
    ordered_question_ids: tuple[str, ...],
) -> FullProgress:
    """Build the explicit empty snapshot for a new non-resume run."""

    return FullProgress(
        resolved_plan_hash=resolved_plan_hash,
        cell_id=cell_id,
        provider_id=provider_id,
        workload_id=workload_id,
        case_manifest_hash=case_manifest_hash,
        ordered_question_ids=ordered_question_ids,
        results=(),
    )


def initialize_full_progress(path: Path, progress: FullProgress) -> FullProgress:
    """Create one canonical progress file and reject every existing target."""

    result = atomic_write_bytes(path, canonical_json_bytes(progress))
    if not result.created:
        raise FileExistsError(f"progress already exists: {path}")
    return progress


def require_complete_full_progress(progress: FullProgress) -> FullProgress:
    """Require the final exact 60-result closure before comparison."""

    if len(progress.results) != 60 or progress.completed_question_ids != (
        progress.ordered_question_ids
    ):
        raise FullProgressError("final progress requires 60 terminal results")
    return progress


class ProviderProgressWriter:
    """Serialize updates to one provider's canonical progress snapshot."""

    def __init__(self, path: Path, *, expected: FullProgress) -> None:
        self._path = Path(path)
        self._expected = expected
        self._lock = threading.Lock()

    def publish(self, entry: FullProgressEntry) -> FullProgress:
        with self._lock:
            current = self._load()
            if entry.question_id not in current.ordered_question_ids:
                raise FullProgressError(f"unknown progress question: {entry.question_id}")
            if entry.question_id in current.completed_question_ids:
                raise FullProgressError(f"question is already complete: {entry.question_id}")
            order = {
                question_id: index for index, question_id in enumerate(current.ordered_question_ids)
            }
            results = tuple(
                sorted((*current.results, entry), key=lambda item: order[item.question_id])
            )
            updated = FullProgress(
                resolved_plan_hash=current.resolved_plan_hash,
                cell_id=current.cell_id,
                provider_id=current.provider_id,
                workload_id=current.workload_id,
                case_manifest_hash=current.case_manifest_hash,
                ordered_question_ids=current.ordered_question_ids,
                results=results,
            )
            atomic_replace_bytes(
                self._path,
                canonical_json_bytes(updated),
                trusted_root=self._path.parent,
            )
            return self._load()

    def _load(self) -> FullProgress:
        return load_full_progress(
            self._path,
            expected_resolved_plan_hash=self._expected.resolved_plan_hash,
            expected_cell_id=self._expected.cell_id,
            expected_provider_id=self._expected.provider_id,
            expected_workload_id=self._expected.workload_id,
            expected_case_manifest_hash=self._expected.case_manifest_hash,
            expected_ordered_question_ids=self._expected.ordered_question_ids,
        )


def load_full_progress(
    path: Path,
    *,
    expected_resolved_plan_hash: str,
    expected_cell_id: str,
    expected_provider_id: str,
    expected_workload_id: str,
    expected_case_manifest_hash: str,
    expected_ordered_question_ids: tuple[str, ...],
) -> FullProgress:
    """Load one canonical snapshot and require its complete runtime identity."""

    try:
        content = read_regular_file(path)
        json.loads(
            content,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        progress = FullProgress.model_validate_json(content)
    except Exception as exc:
        raise FullProgressError(f"progress cannot parse or validate: {path}") from exc
    expected = {
        "resolved_plan_hash": expected_resolved_plan_hash,
        "cell_id": expected_cell_id,
        "provider_id": expected_provider_id,
        "workload_id": expected_workload_id,
        "case_manifest_hash": expected_case_manifest_hash,
        "ordered_question_ids": expected_ordered_question_ids,
    }
    for field_name, expected_value in expected.items():
        if getattr(progress, field_name) != expected_value:
            raise FullProgressError(f"progress {field_name} differs from the current run")
    return progress


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate progress JSON key: {key}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid progress JSON constant: {value}")


__all__ = [
    "FullProgress",
    "FullProgressEntry",
    "FullProgressError",
    "FULL_PROGRESS_PROVIDER_IDS",
    "JudgedFullProgressEntry",
    "ProviderFailedFullProgressEntry",
    "ProviderProgressWriter",
    "UnjudgedFullProgressEntry",
    "acquire_full_resume_lock",
    "canonical_full_progress_path",
    "empty_full_progress",
    "initialize_full_progress",
    "load_full_progress",
    "require_complete_full_progress",
]
