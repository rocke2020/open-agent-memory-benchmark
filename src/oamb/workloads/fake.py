"""Deterministic generated workload for the credential-free vertical slice."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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
    case_manifest_hash,
)

FAKE_DATASET_ID = "oamb-generated-fake-v1"
FAKE_DATASET_REVISION = "generated-1"
FAKE_DATASET_SPLIT = "offline"
FAKE_WORKLOAD_ID = "oamb-fake-vertical-v1"
FAKE_SOURCE_PATH = "generated/fake-cases.json"
FAKE_JUDGE_MAX_OUTPUT_TOKENS = 1


@dataclass(frozen=True, slots=True)
class _GeneratedCase:
    raw_question_id: str
    question: bytes
    reference: bytes
    judge_binding_id: str | None


_READY_CASES = (
    _GeneratedCase("fake-q1", b"Which token is first?", b"alpha", None),
    _GeneratedCase("fake-q2", b"Which token is second?", b"beta", None),
    _GeneratedCase("fake-q3", b"Is the shared phrase grounded?", b"shared answer", "fake-judge-v1"),
)
_PARTIAL_CASES = (
    _GeneratedCase("fake-q4", b"What survived the partial ingest?", b"partial", None),
)
_CONTEXT_PAYLOADS = (
    (b"shared answer alpha beta",),
    (b"partial accepted", b"partial rejected"),
)


class GeneratedFakeWorkload:
    """A two-plan workload with one shared plan and one planted partial plan."""

    def __init__(self) -> None:
        self._dataset = _build_dataset_manifest()
        self._manifest, self._case_plans = _build_case_manifest(self._dataset)

    @property
    def case_plans(self) -> tuple[CasePlan, ...]:
        return self._case_plans

    def resolve_sources(self) -> DatasetManifest:
        return self._dataset

    def build_case_manifest(self, dataset_manifest: DatasetManifest) -> CaseManifest:
        if dataset_manifest != self._dataset:
            raise ValueError("fake workload received a different dataset manifest")
        return self._manifest

    def iter_ingestion_plans(self, case_manifest: CaseManifest) -> tuple[IngestionPlan, ...]:
        if case_manifest != self._manifest:
            raise ValueError("fake workload received a different case manifest")
        plans: list[IngestionPlan] = []
        for plan, payloads in zip(case_manifest.ingestion_plans, _CONTEXT_PAYLOADS, strict=True):
            source_units = tuple(
                SourceUnit(
                    source_unit_id=canonical_sha256(
                        ["oamb-fake-source-unit-v1", plan.ingestion_plan_id, ordinal]
                    ),
                    context_manifest_entry_id=plan.ordered_member_context_manifest_entry_ids[0],
                    ordinal_1_indexed=ordinal,
                    payload_sha256=hashlib.sha256(payload).hexdigest(),
                    payload_bytes=payload,
                )
                for ordinal, payload in enumerate(payloads, start=1)
            )
            plans.append(
                IngestionPlan(
                    ingestion_plan_id=plan.ingestion_plan_id,
                    ordered_member_context_manifest_entry_ids=(
                        plan.ordered_member_context_manifest_entry_ids
                    ),
                    shared_context_sha256=canonical_sha256(
                        [
                            "oamb-fake-shared-context-v1",
                            tuple(item.payload_sha256 for item in source_units),
                        ]
                    ),
                    intended_source_count=len(source_units),
                    ordered_source_units=source_units,
                    ordered_case_manifest_entry_ids=plan.ordered_case_manifest_entry_ids,
                )
            )
        return tuple(plans)

    def iter_case_plans(self, case_manifest: CaseManifest) -> tuple[CasePlan, ...]:
        if case_manifest != self._manifest:
            raise ValueError("fake workload received a different case manifest")
        return self._case_plans

    def render_retrieval_query(self, case_plan: CasePlan) -> bytes:
        return case_plan.question_bytes

    def build_visible_evidence(
        self,
        native_batch: NativeEvidenceBatch,
        policy: VisibleEvidencePolicy,
    ) -> VisibleEvidence:
        selected: list[tuple[str, str]] = []
        character_count = 0
        token_count = 0
        for candidate in native_batch.candidates:
            candidate_tokens = len(candidate.content.split())
            if policy.max_items is not None and len(selected) >= policy.max_items:
                break
            if (
                policy.max_characters is not None
                and character_count + len(candidate.content) > policy.max_characters
            ):
                break
            if policy.max_tokens is not None and token_count + candidate_tokens > policy.max_tokens:
                break
            selected.append((candidate.native_id, candidate.content))
            character_count += len(candidate.content)
            token_count += candidate_tokens
        payload = canonical_json_bytes(
            [{"native_id": native_id, "content": content} for native_id, content in selected]
        )
        return VisibleEvidence(
            canonical_bytes=payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            included_native_ids=tuple(native_id for native_id, _content in selected),
        )

    def render_answer(
        self,
        case_plan: CasePlan,
        visible_evidence: VisibleEvidence,
    ) -> RenderedPrompt:
        prompt = canonical_json_bytes(
            {
                "question": case_plan.question_bytes.decode("utf-8"),
                "visible_evidence_sha256": visible_evidence.sha256,
                "visible_evidence": visible_evidence.canonical_bytes.decode("utf-8"),
            }
        )
        return RenderedPrompt(canonical_bytes=prompt, sha256=hashlib.sha256(prompt).hexdigest())

    def evaluate(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
    ) -> DeterministicEvaluation | JudgeRequest:
        if case_plan.judge_binding_id is not None:
            prompt = canonical_json_bytes(
                {
                    "question": case_plan.question_bytes.decode("utf-8"),
                    "answer_sha256": answer.parsed_value_sha256,
                    "reference_sha256": case_plan.reference_payload_sha256,
                }
            )
            return JudgeRequest(
                prompt=RenderedPrompt(
                    canonical_bytes=prompt,
                    sha256=hashlib.sha256(prompt).hexdigest(),
                ),
                output_contract_id="fake-judge-json-v1",
                max_output_tokens=FAKE_JUDGE_MAX_OUTPUT_TOKENS,
            )
        trace = canonical_json_bytes(
            {
                "metric_id": case_plan.metric_id,
                "numerator": int(answer.parsed_value == case_plan.reference_payload),
                "denominator": 1,
                "parsed_answer_sha256": answer.parsed_value_sha256,
            }
        )
        return DeterministicEvaluation(
            metric_id=case_plan.metric_id,
            result_sha256=hashlib.sha256(trace).hexdigest(),
            numerator=int(answer.parsed_value == case_plan.reference_payload),
            denominator=1,
            trace_bytes=trace,
        )

    def finalize_judge(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
        judge_answer: AnswerValue,
    ) -> DeterministicEvaluation:
        decision = judge_answer.parsed_value.decode("utf-8", errors="strict").strip().lower()
        if decision not in {"yes", "no"}:
            raise ValueError("fake judge output must be yes or no")
        trace = canonical_json_bytes(
            {
                "metric_id": case_plan.metric_id,
                "numerator": int(decision == "yes"),
                "denominator": 1,
                "parsed_answer_sha256": answer.parsed_value_sha256,
                "judge_answer_sha256": judge_answer.parsed_value_sha256,
                "judge_raw_reference": judge_answer.raw_reference.sha256,
                "judge_decision": decision,
            }
        )
        return DeterministicEvaluation(
            metric_id=case_plan.metric_id,
            result_sha256=hashlib.sha256(trace).hexdigest(),
            numerator=int(decision == "yes"),
            denominator=1,
            trace_bytes=trace,
        )

    def validate_records(self, records: WorkloadRecordSet) -> tuple[WorkloadRuleResult, ...]:
        return (
            WorkloadRuleResult(
                rule_id="fake-workload-counts-v1",
                passed=(
                    len(records.logical_context_records) == 2
                    and len(records.ingestion_plan_records) == 2
                    and len(records.case_records) == 4
                ),
                evidence_refs=tuple(record.case_occurrence_id for record in records.case_records),
            ),
        )


def _build_dataset_manifest() -> DatasetManifest:
    source_bytes = canonical_json_bytes(
        [payload.decode("utf-8") for group in _CONTEXT_PAYLOADS for payload in group]
    )
    source = DatasetFile(
        relative_path=FAKE_SOURCE_PATH,
        sha256=hashlib.sha256(source_bytes).hexdigest(),
        byte_count=len(source_bytes),
        license_id="CC0-1.0",
    )
    manifest_hash = canonical_sha256(
        [
            "oamb-fake-dataset-manifest-v1",
            FAKE_DATASET_ID,
            FAKE_DATASET_REVISION,
            FAKE_DATASET_SPLIT,
            source,
        ]
    )
    return DatasetManifest(
        dataset_id=FAKE_DATASET_ID,
        revision=FAKE_DATASET_REVISION,
        split=FAKE_DATASET_SPLIT,
        manifest_hash=manifest_hash,
        source_files=(source,),
        payload_policy="generated-public",
    )


def _build_case_manifest(
    dataset: DatasetManifest,
) -> tuple[CaseManifest, tuple[CasePlan, ...]]:
    source_hash = dataset.source_files[0].sha256
    generated_groups = (_READY_CASES, _PARTIAL_CASES)
    logical_contexts: list[LogicalContextManifestEntry] = []
    cases: list[CaseManifestEntry] = []
    case_plans: list[CasePlan] = []
    ingestion_plans: list[IngestionPlanManifest] = []

    for row_number, (case_group, source_payloads) in enumerate(
        zip(generated_groups, _CONTEXT_PAYLOADS, strict=True),
        start=1,
    ):
        context_bytes_hash = canonical_sha256(
            [
                "oamb-fake-context-bytes-v1",
                tuple(hashlib.sha256(payload).hexdigest() for payload in source_payloads),
            ]
        )
        content_id = context_content_id(
            dataset.revision,
            dataset.split,
            FAKE_SOURCE_PATH,
            context_bytes_hash,
        )
        context_id = context_manifest_entry_id(
            dataset.manifest_hash,
            source_hash,
            row_number,
            content_id,
        )
        logical_contexts.append(
            LogicalContextManifestEntry(
                context_content_id=content_id,
                context_manifest_entry_id=context_id,
                source_file_sha256=source_hash,
                source_row_number_1_indexed=row_number,
                context_bytes_sha256=context_bytes_hash,
            )
        )
        group_case_ids: list[str] = []
        for question_number, generated_case in enumerate(case_group, start=1):
            question_hash = hashlib.sha256(generated_case.question).hexdigest()
            case_id = case_manifest_entry_id(
                context_id,
                question_number,
                question_hash,
                generated_case.raw_question_id,
            )
            group_case_ids.append(case_id)
            cases.append(
                CaseManifestEntry(
                    case_manifest_entry_id=case_id,
                    context_manifest_entry_id=context_id,
                    source_question_number_1_indexed=question_number,
                    question_bytes_sha256=question_hash,
                    raw_question_id=generated_case.raw_question_id,
                    answer_value_sha256=(hashlib.sha256(generated_case.reference).hexdigest(),),
                )
            )
            case_plans.append(
                CasePlan(
                    case_manifest_entry_id=case_id,
                    context_manifest_entry_id=context_id,
                    source_question_number_1_indexed=question_number,
                    question_bytes=generated_case.question,
                    reference_payload=generated_case.reference,
                    reference_payload_sha256=hashlib.sha256(generated_case.reference).hexdigest(),
                    prompt_binding_id="fake-answer-prompt-v1",
                    output_contract_id="fake-answer-text-v1",
                    metric_id="fake-exact-v1",
                    judge_binding_id=generated_case.judge_binding_id,
                )
            )
        member_ids = (context_id,)
        plan_manifest_id = plan_manifest_entry_id(FAKE_WORKLOAD_ID, member_ids)
        ordered_source_hashes = tuple(
            hashlib.sha256(payload).hexdigest() for payload in source_payloads
        )
        payload_hash = ingestion_payload_hash(ordered_source_hashes)
        ingestion_plans.append(
            IngestionPlanManifest(
                plan_manifest_entry_id=plan_manifest_id,
                ingestion_payload_hash=payload_hash,
                ingestion_plan_id=ingestion_plan_id(plan_manifest_id, payload_hash),
                workload_id=FAKE_WORKLOAD_ID,
                ordered_member_context_manifest_entry_ids=member_ids,
                ordered_source_unit_bytes_sha256=ordered_source_hashes,
                ordered_case_manifest_entry_ids=tuple(group_case_ids),
            )
        )

    manifest_fields = {
        "manifest_id": "generated-fake-manifest-v1",
        "workload_id": FAKE_WORKLOAD_ID,
        "logical_contexts": tuple(logical_contexts),
        "ingestion_plans": tuple(ingestion_plans),
        "cases": tuple(cases),
    }
    manifest_hash = case_manifest_hash(manifest_fields)
    return (
        CaseManifest(
            manifest_id="generated-fake-manifest-v1",
            manifest_hash=manifest_hash,
            workload_id=FAKE_WORKLOAD_ID,
            logical_contexts=tuple(logical_contexts),
            ingestion_plans=tuple(ingestion_plans),
            cases=tuple(cases),
        ),
        tuple(case_plans),
    )
