"""Pinned LongMemEval S loading, selection, manifests, and prompt rendering."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oamb.contracts.ids import (
    canonical_json_bytes,
    canonical_sha256,
    case_manifest_entry_id,
    context_content_id,
    context_manifest_entry_id,
    ingestion_payload_hash,
    ingestion_plan_id,
    plan_manifest_entry_id,
)
from oamb.contracts.ports import (
    AnswerValue,
    CasePlan,
    DeterministicEvaluation,
    IngestionPlan,
    JudgeRequest,
    NativeEvidenceBatch,
    RenderedPrompt,
    SourceUnit,
    VisibleEvidence,
    VisibleEvidencePolicy,
    WorkloadRecordSet,
    WorkloadRuleResult,
)
from oamb.contracts.specifications import (
    CaseManifest,
    CaseManifestEntry,
    DatasetFile,
    DatasetManifest,
    IngestionPlanManifest,
    LogicalContextManifestEntry,
    PromptPackManifest,
    PromptTemplateManifest,
    case_manifest_hash,
    prompt_pack_manifest_hash,
)
from oamb.workloads.metrics import LME_ANSWER_MAX_OUTPUT_TOKENS
from oamb.workloads.prompts import PromptPack, render_prompt
from oamb.workloads.visible_evidence import (
    LME_VISIBLE_EVIDENCE_POLICY,
    build_lme_visible_evidence,
    verify_visible_evidence_hash,
)

LME30_WORKLOAD_ID = "lme30-native-smoke-plus-v1"
LME6_MANIFEST_ID = "lme6-live-smoke-v1"
LME_DATASET_ID = "longmemeval-s-cleaned"
LME_DATASET_REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
LME_SOURCE_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"
LME_SOURCE_NAME = "longmemeval_s_cleaned.json"
LME_SOURCE_BYTE_COUNT = 277_383_467
LME_SOURCE_LICENSE_ID = "NOASSERTION"
LME_DATASET_SPLIT = "s-cleaned"
LME_PAYLOAD_POLICY = "download-required-not-redistributed"
LME_DATASET_MANIFEST_HASH = "098fd29291256d5e09dc82db146ee90fadc267e6061c33167bff0b54b98c2a85"
LME30_CASE_MANIFEST_HASH = "b69702c5a643f054b98808ec463ab8babb23d513bdcdd77de79c687eb4d7326d"
LME6_CASE_MANIFEST_HASH = "b5093e3f418eef9cc30eb2323f676be92cf6132f8116aee5a8da6e8a98851563"
LME_ANSWER_PROMPT_PACK_ID = "oamb-lme-answer-v1"
LME_JUDGE_PROMPT_PACK_ID = "oamb-lme-judge-v1"
LME_ANSWER_OUTPUT_CONTRACT_ID = "lme-answer-text-v1"
LME_JUDGE_OUTPUT_CONTRACT_ID = "lme-judge-yes-no-v1"
LME_JUDGE_METRIC_ID = "lme-judged-accuracy-v1"

QUESTION_TYPES = (
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
)
LME30_EXPECTED_QUESTION_IDS = (
    "72e3ee87",
    "22d2cb42",
    "e61a7584",
    "eace081b",
    "8fb83627",
    "21d02d0d",
    "gpt4_a56e767c",
    "gpt4_15e38248",
    "4bc144e2",
    "157a136e",
    "e3fc4d6e",
    "41275add",
    "fca762bc",
    "488d3006",
    "8cf51dda",
    "32260d93",
    "1d4e3b97",
    "0a34ad58",
    "07b6f563",
    "d6233ab6",
    "6b168ec8",
    "bc8a6e93_abs",
    "ccb36322",
    "8e9d538c",
    "c14c00dd",
    "a3045048",
    "gpt4_ec93e27f",
    "gpt4_8279ba03",
    "gpt4_4edbafa2",
    "b46e15ed",
)
LME6_EXPECTED_QUESTION_IDS = (
    "72e3ee87",
    "21d02d0d",
    "e3fc4d6e",
    "32260d93",
    "6b168ec8",
    "a3045048",
)
LME30_EXPECTED_SESSION_COUNT = 1_418
LME6_EXPECTED_SESSION_COUNT = 300

_RAW_TIMESTAMP = re.compile(
    r"(?P<year>[0-9]{4})/(?P<month>[0-9]{2})/(?P<day>[0-9]{2}) "
    r"\((?P<weekday>Mon|Tue|Wed|Thu|Fri|Sat|Sun)\) "
    r"(?P<hour>[0-9]{2}):(?P<minute>[0-9]{2})"
)
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_ROW_KEYS = {
    "question_id",
    "question_type",
    "question",
    "answer",
    "question_date",
    "answer_session_ids",
    "haystack_session_ids",
    "haystack_dates",
    "haystack_sessions",
}
_MESSAGE_KEYS = {"role", "content", "has_answer"}

_ANSWER_TEMPLATE = (
    b"Answer the question using only the retrieved memory evidence below. "
    b"If the evidence is insufficient, say that the information is unavailable.\n\n"
    b"Retrieved memory evidence:\n{{retrieved_context}}\n\n"
    b"Question: {{question}}"
)
_GENERIC_JUDGE_TEMPLATE = (
    b"I will give you a question, a correct answer, and a response from a model. "
    b"Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    b"If the response is equivalent to the correct answer or contains all the intermediate "
    b"steps to get the correct answer, you should also answer yes. If the response only "
    b"contains a subset of the information required by the answer, answer no. \n\n"
    b"Question: {{question}}\n\nCorrect Answer: {{reference}}\n\n"
    b"Model Response: {{model_response}}\n\n"
    b"Is the model response correct? Answer yes or no only."
)
_TEMPORAL_JUDGE_TEMPLATE = (
    b"I will give you a question, a correct answer, and a response from a model. "
    b"Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    b"If the response is equivalent to the correct answer or contains all the intermediate "
    b"steps to get the correct answer, you should also answer yes. If the response only "
    b"contains a subset of the information required by the answer, answer no. In addition, "
    b"do not penalize off-by-one errors for the number of days. If the question asks for "
    b"the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., "
    b"predicting 19 days when the answer is 18), the model's response is still correct. "
    b"\n\nQuestion: {{question}}\n\nCorrect Answer: {{reference}}\n\n"
    b"Model Response: {{model_response}}\n\n"
    b"Is the model response correct? Answer yes or no only."
)
_KNOWLEDGE_UPDATE_JUDGE_TEMPLATE = (
    b"I will give you a question, a correct answer, and a response from a model. "
    b"Please answer yes if the response contains the correct answer. Otherwise, answer no. "
    b"If the response contains some previous information along with an updated answer, the "
    b"response should be considered as correct as long as the updated answer is the required "
    b"answer.\n\nQuestion: {{question}}\n\nCorrect Answer: {{reference}}\n\n"
    b"Model Response: {{model_response}}\n\n"
    b"Is the model response correct? Answer yes or no only."
)
_PREFERENCE_JUDGE_TEMPLATE = (
    b"I will give you a question, a rubric for desired personalized response, and a response "
    b"from a model. Please answer yes if the response satisfies the desired response. "
    b"Otherwise, answer no. The model does not need to reflect all the points in the rubric. "
    b"The response is correct as long as it recalls and utilizes the user's personal "
    b"information correctly.\n\nQuestion: {{question}}\n\nRubric: {{reference}}\n\n"
    b"Model Response: {{model_response}}\n\n"
    b"Is the model response correct? Answer yes or no only."
)
_ABS_JUDGE_TEMPLATE = (
    b"I will give you an unanswerable question, an explanation, and a response from a model. "
    b"Please answer yes if the model correctly identifies the question as unanswerable. The "
    b"model could say that the information is incomplete, or some other information is given "
    b"but the asked information is not.\n\nQuestion: {{question}}\n\n"
    b"Explanation: {{reference}}\n\nModel Response: {{model_response}}\n\n"
    b"Does the model correctly identify the question as unanswerable? Answer yes or no only."
)
_JUDGE_TEMPLATE_BY_TYPE = {
    "knowledge-update": "judge_knowledge_update",
    "multi-session": "judge_multi_session",
    "single-session-assistant": "judge_single_session_assistant",
    "single-session-preference": "judge_single_session_preference",
    "single-session-user": "judge_single_session_user",
    "temporal-reasoning": "judge_temporal_reasoning",
}
_JUDGE_TEMPLATES = {
    "judge_knowledge_update": _KNOWLEDGE_UPDATE_JUDGE_TEMPLATE,
    "judge_multi_session": _GENERIC_JUDGE_TEMPLATE,
    "judge_single_session_assistant": _GENERIC_JUDGE_TEMPLATE,
    "judge_single_session_preference": _PREFERENCE_JUDGE_TEMPLATE,
    "judge_single_session_user": _GENERIC_JUDGE_TEMPLATE,
    "judge_temporal_reasoning": _TEMPORAL_JUDGE_TEMPLATE,
    "judge_abs": _ABS_JUDGE_TEMPLATE,
}


@dataclass(frozen=True, slots=True)
class LongMemEvalMessage:
    role: str
    content: str
    has_answer: bool | None


@dataclass(frozen=True, slots=True)
class LongMemEvalSession:
    session_id: str
    raw_timestamp: str
    canonical_timestamp: str
    messages: tuple[LongMemEvalMessage, ...]


@dataclass(frozen=True, slots=True)
class LongMemEvalRow:
    source_row_number_1_indexed: int
    question_id: str
    question_type: str
    question: str
    answer: str | int | float
    raw_question_timestamp: str
    canonical_question_timestamp: str
    answer_session_ids: tuple[str, ...]
    message_has_answer_session_ids: tuple[str, ...]
    sessions: tuple[LongMemEvalSession, ...]

    @property
    def has_answer_label_mismatch(self) -> bool:
        return self.answer_session_ids != self.message_has_answer_session_ids


@dataclass(frozen=True, slots=True)
class LongMemEvalBundle:
    dataset_manifest: DatasetManifest
    case_manifest: CaseManifest
    selected_rows: tuple[LongMemEvalRow, ...]
    ingestion_plans: tuple[IngestionPlan, ...]
    case_plans: tuple[CasePlan, ...]

    @property
    def label_mismatch_question_ids(self) -> tuple[str, ...]:
        return tuple(row.question_id for row in self.selected_rows if row.has_answer_label_mismatch)


def parse_lme_timestamp(value: str) -> str:
    match = _RAW_TIMESTAMP.fullmatch(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError("LongMemEval timestamp does not match the required ASCII format")
    fields = {name: int(match.group(name)) for name in ("year", "month", "day", "hour", "minute")}
    try:
        parsed = datetime(**fields, tzinfo=UTC)
    except ValueError as exc:
        raise ValueError("LongMemEval timestamp contains an invalid date or time") from exc
    if match.group("weekday") != _WEEKDAYS[parsed.weekday()]:
        raise ValueError("LongMemEval timestamp weekday does not match its date")
    return parsed.isoformat(timespec="seconds")


def render_lme_session_document(
    messages: Sequence[Mapping[str, Any]] | Sequence[LongMemEvalMessage],
) -> bytes:
    rendered: list[dict[str, str]] = []
    for raw_message in messages:
        if isinstance(raw_message, LongMemEvalMessage):
            role_value: object = raw_message.role
            content_value: object = raw_message.content
        else:
            role_value = raw_message.get("role")
            content_value = raw_message.get("content")
        if role_value not in {"user", "assistant"} or not isinstance(content_value, str):
            raise ValueError("LongMemEval messages require a valid role and text content")
        rendered.append({"role": str(role_value), "content": content_value})
    return json.dumps(rendered, ensure_ascii=False, separators=(",", ":")).encode()


def load_longmemeval_rows(
    source_path: Path,
    *,
    expected_sha256: str = LME_SOURCE_SHA256,
) -> tuple[LongMemEvalRow, ...]:
    with source_path.open("rb") as source:
        actual_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("LongMemEval source checksum does not match the pinned SHA-256")
    with source_path.open(encoding="utf-8") as source:
        raw_rows = json.load(source)
    if not isinstance(raw_rows, list):
        raise ValueError("LongMemEval source must be a top-level array")

    rows = tuple(_parse_row(raw, ordinal) for ordinal, raw in enumerate(raw_rows, start=1))
    question_ids = tuple(row.question_id for row in rows)
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("LongMemEval source contains duplicate question IDs")
    return rows


def select_lme30(rows: Sequence[LongMemEvalRow]) -> tuple[LongMemEvalRow, ...]:
    selected: list[LongMemEvalRow] = []
    for question_type in QUESTION_TYPES:
        candidates = [row for row in rows if row.question_type == question_type]
        if len(candidates) < 5:
            raise ValueError(f"LongMemEval question type has fewer than five rows: {question_type}")
        candidates.sort(
            key=lambda row: (_selection_rank(row.question_id), row.question_id.encode())
        )
        selected.extend(candidates[:5])
    result = tuple(selected)
    if tuple(row.question_id for row in result) != LME30_EXPECTED_QUESTION_IDS:
        raise ValueError("LongMemEval selected IDs differ from the frozen LME-30 manifest")
    if sum(len(row.sessions) for row in result) != LME30_EXPECTED_SESSION_COUNT:
        raise ValueError("LongMemEval selected session count differs from 1,418")
    return result


def build_lme30_bundle(source_path: Path) -> LongMemEvalBundle:
    rows = load_longmemeval_rows(source_path)
    selected = select_lme30(rows)
    dataset = _dataset_manifest(source_path)
    bundle = _build_bundle(dataset, selected, workload_id=LME30_WORKLOAD_ID)
    if bundle.case_manifest.manifest_hash != LME30_CASE_MANIFEST_HASH:
        raise ValueError("LongMemEval case manifest differs from the frozen LME-30 hash")
    return bundle


def build_lme6_bundle(full: LongMemEvalBundle) -> LongMemEvalBundle:
    if full.case_manifest.workload_id != LME30_WORKLOAD_ID:
        raise ValueError("LME-6 must be filtered from the frozen LME-30 manifest")
    rows_by_id = {row.question_id: row for row in full.selected_rows}
    if not set(LME6_EXPECTED_QUESTION_IDS) <= set(rows_by_id):
        raise ValueError("LME-30 manifest does not contain every frozen LME-6 case")
    selected_ids = set(LME6_EXPECTED_QUESTION_IDS)
    selected_indices = tuple(
        index for index, row in enumerate(full.selected_rows) if row.question_id in selected_ids
    )
    selected = tuple(full.selected_rows[index] for index in selected_indices)
    if tuple(row.question_id for row in selected) != LME6_EXPECTED_QUESTION_IDS:
        raise ValueError("LME-6 cases do not preserve canonical LME-30 order")
    if sum(len(row.sessions) for row in selected) != LME6_EXPECTED_SESSION_COUNT:
        raise ValueError("LME-6 selected session count differs from 300")
    cases = tuple(full.case_manifest.cases[index] for index in selected_indices)
    plans = tuple(full.case_manifest.ingestion_plans[index] for index in selected_indices)
    contexts = tuple(full.case_manifest.logical_contexts[index] for index in selected_indices)
    manifest_fields = {
        "manifest_id": LME6_MANIFEST_ID,
        "workload_id": LME30_WORKLOAD_ID,
        "logical_contexts": contexts,
        "ingestion_plans": plans,
        "cases": cases,
    }
    manifest = CaseManifest(
        manifest_hash=case_manifest_hash(manifest_fields),
        manifest_id=LME6_MANIFEST_ID,
        workload_id=LME30_WORKLOAD_ID,
        logical_contexts=contexts,
        ingestion_plans=plans,
        cases=cases,
    )
    if manifest.manifest_hash != LME6_CASE_MANIFEST_HASH:
        raise ValueError("LongMemEval smoke manifest differs from the frozen LME-6 hash")
    return LongMemEvalBundle(
        dataset_manifest=full.dataset_manifest,
        case_manifest=manifest,
        selected_rows=selected,
        ingestion_plans=tuple(full.ingestion_plans[index] for index in selected_indices),
        case_plans=tuple(full.case_plans[index] for index in selected_indices),
    )


def render_lme_answer_prompt(*, question: str, evidence: bytes) -> RenderedPrompt:
    evidence_text = evidence.decode("utf-8", errors="strict")
    rendered = render_prompt(
        _ANSWER_PROMPT_PACK,
        "answer_user",
        {"question": question, "retrieved_context": evidence_text},
    )
    if evidence not in rendered.canonical_bytes:
        raise ValueError("rendered LongMemEval prompt changed the visible evidence bytes")
    return rendered


def render_lme_judge_prompt(
    *,
    question_type: str,
    question: str,
    reference: str,
    model_response: str,
    unanswerable: bool,
) -> RenderedPrompt:
    if question_type not in _JUDGE_TEMPLATE_BY_TYPE:
        raise ValueError(f"unsupported LongMemEval question type: {question_type}")
    template_name = "judge_abs" if unanswerable else _JUDGE_TEMPLATE_BY_TYPE[question_type]
    return render_prompt(
        _JUDGE_PROMPT_PACK,
        template_name,
        {"question": question, "reference": reference, "model_response": model_response},
    )


class LongMemEvalWorkload:
    """WorkloadPort implementation over one already-built LME bundle."""

    def __init__(self, bundle: LongMemEvalBundle) -> None:
        self._bundle = bundle
        self._rows_by_case = dict(
            zip(bundle.case_manifest.cases, bundle.selected_rows, strict=True)
        )

    def resolve_sources(self) -> DatasetManifest:
        return self._bundle.dataset_manifest

    def build_case_manifest(self, dataset_manifest: DatasetManifest) -> CaseManifest:
        if dataset_manifest != self._bundle.dataset_manifest:
            raise ValueError("LongMemEval workload received a different dataset manifest")
        return self._bundle.case_manifest

    def iter_ingestion_plans(self, case_manifest: CaseManifest) -> tuple[IngestionPlan, ...]:
        if case_manifest != self._bundle.case_manifest:
            raise ValueError("LongMemEval workload received a different case manifest")
        return self._bundle.ingestion_plans

    def iter_case_plans(self, case_manifest: CaseManifest) -> tuple[CasePlan, ...]:
        if case_manifest != self._bundle.case_manifest:
            raise ValueError("LongMemEval workload received a different case manifest")
        return self._bundle.case_plans

    def render_retrieval_query(self, case_plan: CasePlan) -> bytes:
        return case_plan.question_bytes

    def build_visible_evidence(
        self,
        native_batch: NativeEvidenceBatch,
        policy: VisibleEvidencePolicy,
    ) -> VisibleEvidence:
        if policy != LME_VISIBLE_EVIDENCE_POLICY:
            raise ValueError("LongMemEval visible-evidence policy must remain frozen")
        return build_lme_visible_evidence(native_batch, policy)

    def render_answer(
        self,
        case_plan: CasePlan,
        visible_evidence: VisibleEvidence,
    ) -> RenderedPrompt:
        verify_visible_evidence_hash(visible_evidence)
        if (
            case_plan.prompt_binding_id != LME_ANSWER_PROMPT_PACK_ID
            or case_plan.output_contract_id != LME_ANSWER_OUTPUT_CONTRACT_ID
            or case_plan.metric_id != LME_JUDGE_METRIC_ID
            or case_plan.judge_binding_id != LME_JUDGE_PROMPT_PACK_ID
            or case_plan.answer_max_output_tokens != LME_ANSWER_MAX_OUTPUT_TOKENS
        ):
            raise ValueError("LongMemEval answer binding or output ceiling has drifted")
        return render_lme_answer_prompt(
            question=case_plan.question_bytes.decode("utf-8"),
            evidence=visible_evidence.canonical_bytes,
        )

    def evaluate(self, case_plan: CasePlan, answer: AnswerValue) -> JudgeRequest:
        row = self._row_for_case_id(case_plan.case_manifest_entry_id)
        prompt = render_lme_judge_prompt(
            question_type=row.question_type,
            question=row.question,
            reference=_reference_text(row.answer),
            model_response=answer.parsed_value.decode("utf-8", errors="strict"),
            unanswerable=row.question_id.endswith("_abs"),
        )
        return JudgeRequest(prompt=prompt, output_contract_id=LME_JUDGE_OUTPUT_CONTRACT_ID)

    def finalize_judge(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
        judge_answer: AnswerValue,
    ) -> DeterministicEvaluation:
        if (
            case_plan.metric_id != LME_JUDGE_METRIC_ID
            or case_plan.judge_binding_id != LME_JUDGE_PROMPT_PACK_ID
        ):
            raise ValueError("LongMemEval judge binding has drifted")
        decision = judge_answer.parsed_value.decode("utf-8", errors="strict").strip().lower()
        if decision not in {"yes", "no"}:
            raise ValueError("LongMemEval judge output must be yes or no")
        trace = canonical_json_bytes(
            {
                "metric_id": LME_JUDGE_METRIC_ID,
                "numerator": int(decision == "yes"),
                "denominator": 1,
                "parsed_answer_sha256": answer.parsed_value_sha256,
                "judge_answer_sha256": judge_answer.parsed_value_sha256,
                "judge_raw_reference": judge_answer.raw_reference.sha256,
                "judge_decision": decision,
            }
        )
        return DeterministicEvaluation(
            metric_id=LME_JUDGE_METRIC_ID,
            result_sha256=hashlib.sha256(trace).hexdigest(),
            numerator=int(decision == "yes"),
            denominator=1,
            trace_bytes=trace,
        )

    def validate_records(self, records: WorkloadRecordSet) -> tuple[WorkloadRuleResult, ...]:
        expected_count = len(self._bundle.selected_rows)
        passed = (
            len(records.logical_context_records) == expected_count
            and len(records.ingestion_plan_records) == expected_count
            and len(records.case_records) == expected_count
        )
        return (
            WorkloadRuleResult(
                rule_id=f"{self._bundle.case_manifest.workload_id}-counts-v1",
                passed=passed,
                evidence_refs=tuple(record.case_occurrence_id for record in records.case_records),
            ),
        )

    def _row_for_case_id(self, case_id: str) -> LongMemEvalRow:
        for case, row in self._rows_by_case.items():
            if case.case_manifest_entry_id == case_id:
                return row
        raise ValueError("unknown LongMemEval case plan")


def _parse_row(raw: object, ordinal: int) -> LongMemEvalRow:
    if not isinstance(raw, dict) or set(raw) != _ROW_KEYS:
        raise ValueError("LongMemEval row has an invalid strict shape")
    question_id = _required_string(raw["question_id"], "question_id")
    question_type = _required_string(raw["question_type"], "question_type")
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"unknown LongMemEval question type: {question_type}")
    question = _required_string(raw["question"], "question")
    answer = raw["answer"]
    if isinstance(answer, bool) or not isinstance(answer, (str, int, float)):
        raise ValueError("LongMemEval answer must be a string or number")
    if isinstance(answer, float) and not math.isfinite(answer):
        raise ValueError("LongMemEval numeric answer must be finite")
    raw_question_timestamp = _required_string(raw["question_date"], "question_date")
    answer_session_ids = _string_tuple(raw["answer_session_ids"], "answer_session_ids")
    raw_session_ids = _string_tuple(raw["haystack_session_ids"], "haystack_session_ids")
    raw_dates = raw["haystack_dates"]
    raw_sessions = raw["haystack_sessions"]
    if not isinstance(raw_dates, list) or not isinstance(raw_sessions, list):
        raise ValueError("LongMemEval session fields must be arrays")
    if len(raw_session_ids) != len(raw_dates) or len(raw_dates) != len(raw_sessions):
        raise ValueError("LongMemEval session IDs, dates, and sessions must be aligned")
    if not raw_session_ids:
        raise ValueError("LongMemEval row requires at least one session")
    sessions: list[LongMemEvalSession] = []
    message_has_answer_session_ids: list[str] = []
    for session_id, raw_date, raw_session in zip(
        raw_session_ids,
        raw_dates,
        raw_sessions,
        strict=True,
    ):
        if not isinstance(raw_date, str):
            raise ValueError("LongMemEval session date must be text")
        messages = _parse_messages(raw_session)
        if any(message.has_answer is True for message in messages):
            message_has_answer_session_ids.append(session_id)
        sessions.append(
            LongMemEvalSession(
                session_id=session_id,
                raw_timestamp=raw_date,
                canonical_timestamp=parse_lme_timestamp(raw_date),
                messages=messages,
            )
        )
    return LongMemEvalRow(
        source_row_number_1_indexed=ordinal,
        question_id=question_id,
        question_type=question_type,
        question=question,
        answer=answer,
        raw_question_timestamp=raw_question_timestamp,
        canonical_question_timestamp=parse_lme_timestamp(raw_question_timestamp),
        answer_session_ids=answer_session_ids,
        message_has_answer_session_ids=tuple(message_has_answer_session_ids),
        sessions=tuple(sessions),
    )


def _parse_messages(raw_session: object) -> tuple[LongMemEvalMessage, ...]:
    if not isinstance(raw_session, list) or not raw_session:
        raise ValueError("LongMemEval session must be a non-empty message array")
    messages: list[LongMemEvalMessage] = []
    for raw_message in raw_session:
        if not isinstance(raw_message, dict):
            raise ValueError("LongMemEval message must be an object")
        if not {"role", "content"} <= set(raw_message) <= _MESSAGE_KEYS:
            raise ValueError("LongMemEval message has an invalid strict shape")
        role = raw_message["role"]
        content = raw_message["content"]
        has_answer = raw_message.get("has_answer")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            raise ValueError("LongMemEval message requires a valid role and text content")
        if has_answer is not None and not isinstance(has_answer, bool):
            raise ValueError("LongMemEval has_answer must be Boolean when present")
        messages.append(LongMemEvalMessage(role=role, content=content, has_answer=has_answer))
    return tuple(messages)


def _dataset_manifest(source_path: Path) -> DatasetManifest:
    if source_path.stat().st_size != LME_SOURCE_BYTE_COUNT:
        raise ValueError("LongMemEval source byte count differs from the frozen snapshot")
    source_file = DatasetFile(
        relative_path=LME_SOURCE_NAME,
        sha256=LME_SOURCE_SHA256,
        byte_count=LME_SOURCE_BYTE_COUNT,
        license_id=LME_SOURCE_LICENSE_ID,
    )
    manifest_hash = canonical_sha256(
        [
            "oamb-lme-dataset-manifest-v1",
            LME_DATASET_ID,
            LME_DATASET_REVISION,
            LME_DATASET_SPLIT,
            source_file,
        ]
    )
    if manifest_hash != LME_DATASET_MANIFEST_HASH:
        raise ValueError("LongMemEval dataset manifest differs from its frozen hash")
    return DatasetManifest(
        dataset_id=LME_DATASET_ID,
        revision=LME_DATASET_REVISION,
        split=LME_DATASET_SPLIT,
        manifest_hash=manifest_hash,
        source_files=(source_file,),
        payload_policy=LME_PAYLOAD_POLICY,
    )


def _build_bundle(
    dataset: DatasetManifest,
    selected_rows: tuple[LongMemEvalRow, ...],
    *,
    workload_id: str,
) -> LongMemEvalBundle:
    source_hash = dataset.source_files[0].sha256
    logical_contexts: list[LogicalContextManifestEntry] = []
    plan_manifests: list[IngestionPlanManifest] = []
    case_entries: list[CaseManifestEntry] = []
    ingestion_plans: list[IngestionPlan] = []
    case_plans: list[CasePlan] = []

    for row in selected_rows:
        source_documents = tuple(
            render_lme_session_document(session.messages) for session in row.sessions
        )
        source_hashes = tuple(hashlib.sha256(document).hexdigest() for document in source_documents)
        context_hash = canonical_sha256(
            [
                "oamb-lme-context-bytes-v1",
                tuple(
                    (session.session_id, session.canonical_timestamp, document_sha256)
                    for session, document_sha256 in zip(
                        row.sessions,
                        source_hashes,
                        strict=True,
                    )
                ),
            ]
        )
        content_id = context_content_id(
            dataset.revision,
            dataset.split,
            LME_SOURCE_NAME,
            context_hash,
        )
        context_id = context_manifest_entry_id(
            dataset.manifest_hash,
            source_hash,
            row.source_row_number_1_indexed,
            content_id,
        )
        logical_contexts.append(
            LogicalContextManifestEntry(
                context_content_id=content_id,
                context_manifest_entry_id=context_id,
                source_file_sha256=source_hash,
                source_row_number_1_indexed=row.source_row_number_1_indexed,
                context_bytes_sha256=context_hash,
            )
        )
        question_bytes = row.question.encode()
        question_hash = hashlib.sha256(question_bytes).hexdigest()
        case_id = case_manifest_entry_id(context_id, 1, question_hash, row.question_id)
        reference_payload = json.dumps(
            row.answer,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        reference_hash = hashlib.sha256(reference_payload).hexdigest()
        case_entries.append(
            CaseManifestEntry(
                case_manifest_entry_id=case_id,
                context_manifest_entry_id=context_id,
                source_question_number_1_indexed=1,
                question_bytes_sha256=question_hash,
                raw_question_id=row.question_id,
                answer_value_sha256=(reference_hash,),
            )
        )
        case_plans.append(
            CasePlan(
                case_manifest_entry_id=case_id,
                context_manifest_entry_id=context_id,
                source_question_number_1_indexed=1,
                question_bytes=question_bytes,
                reference_payload=reference_payload,
                reference_payload_sha256=reference_hash,
                prompt_binding_id=LME_ANSWER_PROMPT_PACK_ID,
                output_contract_id=LME_ANSWER_OUTPUT_CONTRACT_ID,
                metric_id=LME_JUDGE_METRIC_ID,
                judge_binding_id=LME_JUDGE_PROMPT_PACK_ID,
                answer_max_output_tokens=LME_ANSWER_MAX_OUTPUT_TOKENS,
                query_timestamp=row.canonical_question_timestamp,
            )
        )
        members = (context_id,)
        plan_manifest_id = plan_manifest_entry_id(workload_id, members)
        payload_hash = ingestion_payload_hash(source_hashes)
        physical_plan_id = ingestion_plan_id(plan_manifest_id, payload_hash)
        plan_manifests.append(
            IngestionPlanManifest(
                plan_manifest_entry_id=plan_manifest_id,
                ingestion_payload_hash=payload_hash,
                ingestion_plan_id=physical_plan_id,
                workload_id=workload_id,
                ordered_member_context_manifest_entry_ids=members,
                ordered_source_unit_bytes_sha256=source_hashes,
                ordered_case_manifest_entry_ids=(case_id,),
            )
        )
        source_units = tuple(
            SourceUnit(
                source_unit_id=canonical_sha256(
                    [
                        "oamb-lme-source-unit-v1",
                        LME30_WORKLOAD_ID,
                        row.question_id,
                        session.session_id,
                        ordinal,
                        payload_sha256,
                    ]
                ),
                context_manifest_entry_id=context_id,
                ordinal_1_indexed=ordinal,
                payload_sha256=payload_sha256,
                payload_bytes=payload,
                source_reference=session.session_id,
                occurred_at=session.canonical_timestamp,
                context_text=(
                    f"LongMemEval session {session.session_id} at {session.canonical_timestamp}"
                ),
                source_metadata=(
                    ("question_id", row.question_id),
                    ("session_id", session.session_id),
                ),
            )
            for ordinal, (session, payload_sha256, payload) in enumerate(
                zip(row.sessions, source_hashes, source_documents, strict=True),
                start=1,
            )
        )
        ingestion_plans.append(
            IngestionPlan(
                ingestion_plan_id=physical_plan_id,
                ordered_member_context_manifest_entry_ids=members,
                shared_context_sha256=context_hash,
                intended_source_count=len(source_units),
                ordered_source_units=source_units,
                ordered_case_manifest_entry_ids=(case_id,),
            )
        )

    manifest_id = f"{workload_id}-manifest-v1"
    manifest_fields = {
        "manifest_id": manifest_id,
        "workload_id": workload_id,
        "logical_contexts": tuple(logical_contexts),
        "ingestion_plans": tuple(plan_manifests),
        "cases": tuple(case_entries),
    }
    case_manifest = CaseManifest(
        manifest_id=manifest_id,
        manifest_hash=case_manifest_hash(manifest_fields),
        workload_id=workload_id,
        logical_contexts=tuple(logical_contexts),
        ingestion_plans=tuple(plan_manifests),
        cases=tuple(case_entries),
    )
    return LongMemEvalBundle(
        dataset_manifest=dataset,
        case_manifest=case_manifest,
        selected_rows=selected_rows,
        ingestion_plans=tuple(ingestion_plans),
        case_plans=tuple(case_plans),
    )


def _selection_rank(question_id: str) -> bytes:
    return hashlib.sha256(f"oamb-lme30-native-smoke-plus-v1\0{question_id}".encode()).digest()


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"LongMemEval {field} must be non-empty text")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"LongMemEval {field} must be an array of non-empty strings")
    return tuple(value)


def _reference_text(answer: str | int | float) -> str:
    return answer if isinstance(answer, str) else json.dumps(answer, separators=(",", ":"))


def _prompt_pack(
    *,
    prompt_pack_id: str,
    templates: Mapping[str, bytes],
    variables: tuple[str, ...],
    output_contract_id: str,
    origin: str,
    source_repository: str | None,
    source_revision: str | None,
    source_file_sha256: str | None,
    license_expression: str,
    rights_attestation: str,
) -> PromptPack:
    template_manifests = tuple(
        PromptTemplateManifest(
            template_name=name,
            relative_path=f"longmemeval/{name}.txt",
            content_sha256=hashlib.sha256(content).hexdigest(),
            byte_count=len(content),
            source_extracted_sha256=(
                hashlib.sha256(
                    content.replace(b"{{question}}", b"{}")
                    .replace(b"{{reference}}", b"{}")
                    .replace(b"{{model_response}}", b"{}")
                ).hexdigest()
                if origin == "attributed_source"
                else None
            ),
            adaptation_id=(
                "oamb-double-brace-variable-adaptation-v1"
                if origin == "attributed_source"
                else None
            ),
        )
        for name, content in templates.items()
    )
    fields: dict[str, Any] = {
        "prompt_pack_id": prompt_pack_id,
        "prompt_pack_version": "1",
        "workload_id": LME30_WORKLOAD_ID,
        "origin": origin,
        "source_repository": source_repository,
        "source_revision": source_revision,
        "source_file_sha256": source_file_sha256,
        "license_expression": license_expression,
        "rights_attestation": rights_attestation,
        "redistribution_allowed": True,
        "templates": template_manifests,
        "variables": variables,
        "output_contract_id": output_contract_id,
    }
    manifest = PromptPackManifest(
        **fields,
        manifest_sha256=prompt_pack_manifest_hash(fields),
    )
    return PromptPack(manifest=manifest, templates=templates)


_ANSWER_PROMPT_PACK = _prompt_pack(
    prompt_pack_id=LME_ANSWER_PROMPT_PACK_ID,
    templates={"answer_user": _ANSWER_TEMPLATE},
    variables=("question", "retrieved_context"),
    output_contract_id=LME_ANSWER_OUTPUT_CONTRACT_ID,
    origin="oamb_authored",
    source_repository=None,
    source_revision=None,
    source_file_sha256=None,
    license_expression="Apache-2.0",
    rights_attestation="oamb_authored",
)
_JUDGE_PROMPT_PACK = _prompt_pack(
    prompt_pack_id=LME_JUDGE_PROMPT_PACK_ID,
    templates=_JUDGE_TEMPLATES,
    variables=("question", "reference", "model_response"),
    output_contract_id=LME_JUDGE_OUTPUT_CONTRACT_ID,
    origin="attributed_source",
    source_repository="https://github.com/xiaowu0162/LongMemEval",
    source_revision="9e0b455f4ef0e2ab8f2e582289761153549043fc",
    source_file_sha256="ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251",
    license_expression="MIT",
    rights_attestation="project_licensed_source",
)
LME_PROMPT_PACKS = (_ANSWER_PROMPT_PACK, _JUDGE_PROMPT_PACK)
