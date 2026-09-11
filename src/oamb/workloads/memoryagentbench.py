"""Frozen MemoryAgentBench MAB-65 manifest and ingestion construction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]

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
    EvidenceDecision,
    IngestionPlan,
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
from oamb.workloads.metrics import (
    MAB_DETECTIVEQA_MAX_OUTPUT_TOKENS,
    MAB_EVENTQA_MAX_OUTPUT_TOKENS,
    MAB_FACTCONSOLIDATION_MAX_OUTPUT_TOKENS,
    MAB_ICL_MAX_OUTPUT_TOKENS,
    MAB_REDIAL_MAX_OUTPUT_TOKENS,
    max_over_gold_answers,
)
from oamb.workloads.prompts import PromptPack, render_prompt
from oamb.workloads.redial import (
    MovieCatalog,
    resolve_ranked_movies,
    score_recall_at_5,
)
from oamb.workloads.visible_evidence import (
    EvidenceConflictError,
    count_o200k_tokens,
    o200k_encoding,
    tokenizer_fingerprint,
    verify_visible_evidence_hash,
)

MAB_CHUNK_MAX_TOKENS = 4_096
MAB_RETRIEVAL_TOP_K = 100
MAB_VISIBLE_EVIDENCE_MAX_ITEMS = MAB_RETRIEVAL_TOP_K
MAB_VISIBLE_EVIDENCE_MAX_CHARACTERS = 2_147_483_647
MAB_VISIBLE_EVIDENCE_MAX_TOKENS = 32_768
MAB_VISIBLE_EVIDENCE_POLICY = VisibleEvidencePolicy(
    max_items=MAB_VISIBLE_EVIDENCE_MAX_ITEMS,
    max_characters=MAB_VISIBLE_EVIDENCE_MAX_CHARACTERS,
    max_tokens=MAB_VISIBLE_EVIDENCE_MAX_TOKENS,
)
MAB65_WORKLOAD_ID = "mab65-v1"
MAB65_RANK_PREFIX = "oamb-mab65-v1"
MAB5_MANIFEST_ID = "mab5-live-smoke-v1"
MAB_DATASET_ID = "ai-hyz/MemoryAgentBench"
MAB_DATASET_REVISION = "7ea066982b140a19337e17e60d45d4076e042faf"
MAB_DATASET_SPLIT = "mab65-frozen-four-split-v1"
MAB_SOURCE_LICENSE_ID = "NOASSERTION"
MAB_PAYLOAD_POLICY = "local_only_no_redistribution"
MAB_DATASET_MANIFEST_HASH = "78011e09488f2ca88c13dd5518016b56668b1d49174fdec85a1db0eaccf00a2c"
MAB65_SELECTED_CASE_DIGEST = "e8526e11afbb2af5ffe5ccf35ccad20cd2c41f14fdc3e18aab62fb79417b97d0"
MAB65_SELECTED_PLAN_DIGEST = "85efa5ac9698bf512708cf93bcd011218578183e5f2fc86bbe8ed3f2eda00ef4"
MAB65_CR_32K_PLAN_ID = "0ebed1d647aa227f5026e71ca93a88fb5be2d0ef0e73140ce0caa8914c03e46f"
MAB65_CASE_MANIFEST_HASH = "d0dbbdf149e3a58d34265dbb6097a28b8990b585eaa9c8812893497a32c962f2"
MAB5_CASE_MANIFEST_HASH = "008770fc5e158fbf1f2bfb25a0f246988602a3128c204ba492d9f2b178828512"
MAB5_SELECTED_CASE_DIGEST = "c0c596bf1b4e9a8ab34284f1b9231cd89fead3b781893d49fcf917f426d6a7a3"
MAB5_SELECTED_PLAN_DIGEST = "0ee46f35ebb4917785e585994d53c3b0e68855d3575806740a8129b49e30a7c4"
MAB_PROMPT_SOURCE_SHA256 = "148c40d48d19f155ae845482c4417ba59cfa7ae4e194019509e023bd3a8755dd"
MAB_PROMPT_SOURCE_REVISION = "fe1735de8cf8b9908e1e3d3b5612afc815698062"
MAB_ANSWER_WRAPPER = (
    "Answer the user request using only the retrieved memory evidence below. Do not use outside knowledge.\n"
    "If the evidence is insufficient, follow the workload output contract without inventing facts.\n\n"
    "Retrieved memory evidence:\n<evidence>\n{{retrieved_context}}\n</evidence>"
)
MAB_ANSWER_WRAPPER_SHA256 = hashlib.sha256(MAB_ANSWER_WRAPPER.encode()).hexdigest()

MAB_PINNED_SOURCE_FILES = (
    (
        "data/Accurate_Retrieval-00000-of-00001.parquet",
        "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",
        "Accurate_Retrieval",
        20_024_386,
    ),
    (
        "data/Test_Time_Learning-00000-of-00001.parquet",
        "5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0",
        "Test_Time_Learning",
        3_947_476,
    ),
    (
        "data/Long_Range_Understanding-00000-of-00001.parquet",
        "5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539",
        "Long_Range_Understanding",
        49_342_452,
    ),
    (
        "data/Conflict_Resolution-00000-of-00001.parquet",
        "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",
        "Conflict_Resolution",
        1_491_588,
    ),
)
MAB_ENTITY_SOURCE_FILE = (
    "entity2id.json",
    "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5",
    1_758_081,
)
_PINNED_FILE_ORDER = {
    relative_path: ordinal
    for ordinal, (relative_path, _sha256, _split, _byte_count) in enumerate(MAB_PINNED_SOURCE_FILES)
}
_ICL_SOURCES = (
    "icl_banking77_5900shot_balance",
    "icl_clinic150_7050shot_balance",
    "icl_nlu_8296shot_balance",
    "icl_trec_coarse_6600shot_balance",
    "icl_trec_fine_6400shot_balance",
)
_SMOKE_SELECTION = (
    (
        "77742acb5c419f835a4c629a9a31b6a27faf1df718f9f6bd2c40438bc35de74f",
        "eventqa_65536@q6",
    ),
    (
        "8cb2e20f7bf5540ec5299cdad51e3823984a752826fce30d716bf7e77d8dedc4",
        "recsys_redial_full@q33",
    ),
    (
        "ed02d6d422b33f3bfe433f4559682205122d496e8daef1ea2a0695e9b530baa6",
        "icl_banking77_5900shot_balance@q56",
    ),
    (
        "e8281d5ca4588bda0bc94bbc281ddbad529fffae959bddee4f64bb6e8bedf91c",
        "detective_qa@q8",
    ),
    (
        "31e2161abe21a8f936d389e706270dd7ebfe1ed9302d4aa10cf3a0783340d9c1",
        "factconsolidation_mh_262k@q21",
    ),
)

_MAB_PROMPT_DEFINITIONS = (
    (
        "oamb-mab-eventqa-rag-v1",
        "mab-first-line-v1",
        MAB_EVENTQA_MAX_OUTPUT_TOKENS,
        "eventqa",
        "Based on the context you memorized, complete the task below:\n\n{question}\n\n The event that happens next is:",
        "c86b0dc5db229403976a20bcef1cc626ddc922569955a9d17c420171698ecb49",
    ),
    (
        "oamb-mab-icl-rag-v1",
        "mab-first-line-v1",
        MAB_ICL_MAX_OUTPUT_TOKENS,
        "icl",
        'Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {{label}}" and nothing else. \n\nQuestion:{question} \n\n label:',
        "e9df61cd16b5e95d08fe93b1bde82f2665b68ecff0bc6463fb43f69e3c570262",
    ),
    (
        "oamb-mab-redial-rag-v1",
        "mab-redial-ranked-movies-v1",
        MAB_REDIAL_MAX_OUTPUT_TOKENS,
        "redial",
        "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: {question} \n\n The recommendations are: \n",
        "96e3f5bae8307491abb3a85e0ddbebbd89a5d10d642f8721aa0433eb3203be5e",
    ),
    (
        "oamb-mab-detectiveqa-rag-v1",
        "mab-raw-or-first-line-max-v1",
        MAB_DETECTIVEQA_MAX_OUTPUT_TOKENS,
        "detectiveqa",
        "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
        "0117e352348af57cf8dd30a31286bfae1cd9f360a04d3d88eb5f6a041145a387",
    ),
    (
        "oamb-mab-factconsolidation-rag-v1",
        "mab-raw-or-first-line-max-v1",
        MAB_FACTCONSOLIDATION_MAX_OUTPUT_TOKENS,
        "factconsolidation",
        "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
        "ca4ea130355adab357ce8cd00bb11c1cdbe7cb945454764a66f1ac7f692f501d",
    ),
)


@dataclass(frozen=True, slots=True)
class MabDatasetRow:
    split: str
    source_relative_path: str
    source_file_sha256: str
    source_row_number_1_indexed: int
    source: str
    context: str
    questions: tuple[str, ...]
    answers: tuple[tuple[str, ...], ...]
    raw_question_ids: tuple[str, ...]
    context_entry: LogicalContextManifestEntry


@dataclass(frozen=True, slots=True)
class MabCase:
    entry: CaseManifestEntry
    case_plan: CasePlan
    component: str
    logical_case_label: str
    source_relative_path: str
    source_row_number_1_indexed: int


@dataclass(frozen=True, slots=True)
class MabPlan:
    manifest: IngestionPlanManifest
    runtime_plan: IngestionPlan
    member_labels: tuple[str, ...]
    component: str


@dataclass(frozen=True, slots=True)
class MabManifestBundle:
    dataset_manifest: DatasetManifest
    case_manifest: CaseManifest
    cases: tuple[MabCase, ...]
    plans: tuple[MabPlan, ...]
    movie_catalog: MovieCatalog
    entity_catalog_sha256: str
    unicode_fingerprint: str
    selected_case_digest: str
    selected_plan_digest: str


@dataclass(frozen=True, slots=True)
class MabPromptPackBinding:
    pack: PromptPack
    max_output_tokens: int
    retrieval_query_template: bytes

    @property
    def extracted_query_sha256(self) -> str:
        extracted = self.pack.manifest.templates[0].source_extracted_sha256
        if extracted is None:
            raise ValueError("MAB prompt binding lacks its source-extracted hash")
        return extracted

    def render_retrieval_query(self, question_bytes: bytes) -> bytes:
        question_bytes.decode("utf-8", errors="strict")
        return self.retrieval_query_template.replace(b"{{question}}", question_bytes)


def _build_mab_prompt_pack(
    prompt_pack_id: str,
    output_contract_id: str,
    max_output_tokens: int,
    short_name: str,
    extracted_query: str,
    extracted_query_sha256: str,
) -> MabPromptPackBinding:
    if hashlib.sha256(extracted_query.encode()).hexdigest() != extracted_query_sha256:
        raise ValueError(f"MAB extracted prompt hash mismatch: {short_name}")
    rendered_query = extracted_query.format(question="{{question}}")
    retrieval_query_template = rendered_query.encode()
    if retrieval_query_template.count(b"{{question}}") != 1:
        raise ValueError(f"MAB retrieval query placeholder mismatch: {short_name}")
    template = f"{MAB_ANSWER_WRAPPER}\n\n{rendered_query}".encode()
    template_manifest = PromptTemplateManifest(
        template_name="answer_user",
        relative_path=f"memoryagentbench/{short_name}-answer-user.txt",
        content_sha256=hashlib.sha256(template).hexdigest(),
        byte_count=len(template),
        source_extracted_sha256=extracted_query_sha256,
        adaptation_id="oamb-mab-answer-wrapper-v1",
    )
    values = dict(
        prompt_pack_id=prompt_pack_id,
        prompt_pack_version="1",
        workload_id=MAB65_WORKLOAD_ID,
        origin="attributed_source",
        source_repository="https://github.com/HUST-AI-HYZ/MemoryAgentBench",
        source_revision=MAB_PROMPT_SOURCE_REVISION,
        source_file_sha256=MAB_PROMPT_SOURCE_SHA256,
        license_expression="MIT",
        rights_attestation="pinned_upstream_mit_source",
        redistribution_allowed=True,
        templates=(template_manifest,),
        variables=("question", "retrieved_context"),
        output_contract_id=output_contract_id,
    )
    manifest = PromptPackManifest.model_validate(
        {**values, "manifest_sha256": prompt_pack_manifest_hash(values)}
    )
    return MabPromptPackBinding(
        pack=PromptPack(manifest=manifest, templates={"answer_user": template}),
        max_output_tokens=max_output_tokens,
        retrieval_query_template=retrieval_query_template,
    )


MAB_PROMPT_PACKS = tuple(
    _build_mab_prompt_pack(*definition) for definition in _MAB_PROMPT_DEFINITIONS
)
_MAB_PROMPT_BINDINGS_BY_ID = {
    binding.pack.manifest.prompt_pack_id: binding for binding in MAB_PROMPT_PACKS
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def mab_dataset_content_manifest_hash(files: Sequence[tuple[str, str]]) -> str:
    return canonical_sha256(
        [
            "oamb-mab65-dataset-manifest-v1",
            MAB_DATASET_ID,
            MAB_DATASET_REVISION,
            tuple(files),
        ]
    )


def _require_hash(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise ValueError(f"MemoryAgentBench checksum mismatch: {path.name}")


def _strict_strings(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"MemoryAgentBench {field} must be a list of non-empty strings")
    return tuple(value)


def read_aligned_parquet_rows(
    path: Path,
    *,
    split: str,
    expected_sha256: str,
    dataset_manifest_hash: str = MAB_DATASET_MANIFEST_HASH,
    source_relative_path: str | None = None,
) -> tuple[MabDatasetRow, ...]:
    """Read one hash-bound Parquet file and reject any aligned-list drift."""

    _require_hash(path, expected_sha256)
    table = pq.read_table(path)
    if table.column_names != ["context", "questions", "answers", "metadata"]:
        raise ValueError("MemoryAgentBench Parquet columns do not match the pinned schema")
    rows: list[MabDatasetRow] = []
    for row_number, document in enumerate(table.to_pylist(), start=1):
        context = document.get("context")
        metadata = document.get("metadata")
        if not isinstance(context, str) or not context or not isinstance(metadata, dict):
            raise ValueError("MemoryAgentBench row requires context and metadata")
        source = metadata.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("MemoryAgentBench row requires a source label")
        questions = _strict_strings(document.get("questions"), field="questions")
        raw_ids = _strict_strings(metadata.get("qa_pair_ids"), field="qa_pair_ids")
        answers_value = document.get("answers")
        if not isinstance(answers_value, list):
            raise ValueError("MemoryAgentBench answers must be a nested list")
        answers = tuple(
            _strict_strings(answer, field="answer alternatives") for answer in answers_value
        )
        if not questions or len(questions) != len(answers) or len(questions) != len(raw_ids):
            raise ValueError("MemoryAgentBench questions, answers, and IDs must align")
        if any(not answer for answer in answers) or any(not raw_id for raw_id in raw_ids):
            raise ValueError("MemoryAgentBench answers and raw IDs must be non-empty")
        context_bytes_hash = hashlib.sha256(context.encode("utf-8")).hexdigest()
        content_id = context_content_id(MAB_DATASET_REVISION, split, source, context_bytes_hash)
        context_entry_id = context_manifest_entry_id(
            dataset_manifest_hash,
            expected_sha256,
            row_number,
            content_id,
        )
        rows.append(
            MabDatasetRow(
                split=split,
                source_relative_path=source_relative_path or path.name,
                source_file_sha256=expected_sha256,
                source_row_number_1_indexed=row_number,
                source=source,
                context=context,
                questions=questions,
                answers=answers,
                raw_question_ids=raw_ids,
                context_entry=LogicalContextManifestEntry(
                    context_content_id=content_id,
                    context_manifest_entry_id=context_entry_id,
                    source_file_sha256=expected_sha256,
                    source_row_number_1_indexed=row_number,
                    context_bytes_sha256=context_bytes_hash,
                ),
            )
        )
    return tuple(rows)


def _rank(domain: str, identity: str) -> bytes:
    return hashlib.sha256(
        MAB65_RANK_PREFIX.encode() + b"\0" + domain.encode() + b"\0" + identity.encode("ascii")
    ).digest()


def _case_entry(row: MabDatasetRow, question_number: int) -> CaseManifestEntry:
    question = row.questions[question_number - 1]
    question_hash = hashlib.sha256(question.encode("utf-8")).hexdigest()
    raw_id = row.raw_question_ids[question_number - 1]
    entry_id = case_manifest_entry_id(
        row.context_entry.context_manifest_entry_id,
        question_number,
        question_hash,
        raw_id,
    )
    return CaseManifestEntry(
        case_manifest_entry_id=entry_id,
        context_manifest_entry_id=row.context_entry.context_manifest_entry_id,
        source_question_number_1_indexed=question_number,
        question_bytes_sha256=question_hash,
        raw_question_id=raw_id,
        answer_value_sha256=tuple(
            hashlib.sha256(answer.encode("utf-8")).hexdigest()
            for answer in row.answers[question_number - 1]
        ),
    )


def _ranked_questions(row: MabDatasetRow, domain: str, count: int) -> tuple[int, ...]:
    ranked = sorted(
        range(1, len(row.questions) + 1),
        key=lambda number: (
            _rank(domain, _case_entry(row, number).case_manifest_entry_id),
            _case_entry(row, number).case_manifest_entry_id,
        ),
    )
    return tuple(sorted(ranked[:count]))


def _component(source: str) -> str:
    if source == "eventqa_65536":
        return "ar"
    if source in _ICL_SOURCES:
        return "icl"
    if source == "recsys_redial_full":
        return "recsys"
    if source == "detective_qa":
        return "lru"
    if source.startswith("factconsolidation_"):
        return "cr_sf"
    raise ValueError(f"source is outside MAB-65: {source}")


def _interaction_ids(source: str) -> tuple[str, str, str]:
    if source == "eventqa_65536":
        return "oamb-mab-eventqa-rag-v1", "mab-first-line-v1", "mab-substring-em-v1"
    if source in _ICL_SOURCES:
        return "oamb-mab-icl-rag-v1", "mab-first-line-v1", "mab-exact-v1"
    if source == "recsys_redial_full":
        return (
            "oamb-mab-redial-rag-v1",
            "mab-redial-ranked-movies-v1",
            "mab-redial-recall-at-5-v1",
        )
    if source == "detective_qa":
        return (
            "oamb-mab-detectiveqa-rag-v1",
            "mab-raw-or-first-line-max-v1",
            "mab-exact-v1",
        )
    return (
        "oamb-mab-factconsolidation-rag-v1",
        "mab-raw-or-first-line-max-v1",
        "mab-substring-em-v1",
    )


def _canonical_rows(rows: Sequence[MabDatasetRow]) -> tuple[MabDatasetRow, ...]:
    try:
        return tuple(
            sorted(
                rows,
                key=lambda row: (
                    _PINNED_FILE_ORDER[row.source_relative_path],
                    row.source_row_number_1_indexed,
                    row.context_entry.context_manifest_entry_id,
                ),
            )
        )
    except KeyError as exc:
        raise ValueError(f"MAB-65 row has an unknown source path: {exc.args[0]}") from exc


def _selected_groups(
    rows: tuple[MabDatasetRow, ...],
) -> tuple[tuple[tuple[MabDatasetRow, ...], tuple[tuple[int, ...], ...]], ...]:
    rows = _canonical_rows(rows)
    groups: list[tuple[tuple[MabDatasetRow, ...], tuple[tuple[int, ...], ...]]] = []
    ar_rows = tuple(row for row in rows if row.source == "eventqa_65536")
    if len(ar_rows) != 5:
        raise ValueError("MAB-65 requires exactly five EventQA 64K rows")
    groups.extend(((row,), (_ranked_questions(row, "ar-case", 3),)) for row in ar_rows)

    redial = tuple(row for row in rows if row.source == "recsys_redial_full")
    if len(redial) != 1:
        raise ValueError("MAB-65 requires exactly one ReDial row")
    groups.append(((redial[0],), (_ranked_questions(redial[0], "redial-case", 10),)))
    for source in _ICL_SOURCES:
        matches = tuple(row for row in rows if row.source == source)
        if len(matches) != 1:
            raise ValueError(f"MAB-65 requires exactly one ICL row: {source}")
        groups.append(((matches[0],), (_ranked_questions(matches[0], "icl-case", 2),)))

    lru_rows = tuple(row for row in rows if row.source == "detective_qa")
    if len(lru_rows) != 10:
        raise ValueError("MAB-65 requires exactly ten DetectiveQA rows")
    extra_ids = {
        row.context_entry.context_manifest_entry_id
        for row in sorted(
            lru_rows,
            key=lambda row: (
                _rank("lru-extra-context", row.context_entry.context_manifest_entry_id),
                row.context_entry.context_manifest_entry_id,
            ),
        )[:5]
    }
    groups.extend(
        (
            (row,),
            (
                _ranked_questions(
                    row,
                    "lru-case",
                    2 if row.context_entry.context_manifest_entry_id in extra_ids else 1,
                ),
            ),
        )
        for row in lru_rows
    )

    cr_rows = tuple(row for row in rows if row.source.startswith("factconsolidation_"))
    cr_groups: list[tuple[tuple[MabDatasetRow, ...], str]] = []
    for length in ("6k", "32k", "64k", "262k"):
        sh = tuple(row for row in cr_rows if row.source == f"factconsolidation_sh_{length}")
        mh = tuple(row for row in cr_rows if row.source == f"factconsolidation_mh_{length}")
        if len(sh) != 1 or len(mh) != 1 or sh[0].context.encode() != mh[0].context.encode():
            raise ValueError(f"MAB-65 requires byte-identical SH/MH members at {length}")
        members: tuple[MabDatasetRow, ...] = (sh[0], mh[0])
        manifest_id = plan_manifest_entry_id(
            MAB65_WORKLOAD_ID,
            tuple(row.context_entry.context_manifest_entry_id for row in members),
        )
        cr_groups.append((members, manifest_id))

    three_manifest_id = min(
        (manifest_id for _members, manifest_id in cr_groups),
        key=lambda manifest_id: (
            _rank("cr-three-plan", manifest_id),
            manifest_id,
        ),
    )
    for members, manifest_id in cr_groups:
        if manifest_id == three_manifest_id:
            extra = min(
                members,
                key=lambda row: (
                    _rank("cr-extra-member", row.context_entry.context_manifest_entry_id),
                    row.context_entry.context_manifest_entry_id,
                ),
            )
            positions = tuple(
                _ranked_questions(row, "cr-case", 2 if row is extra else 1) for row in members
            )
        else:
            positions = tuple(_ranked_questions(row, "cr-case", 2) for row in members)
        groups.append((members, positions))
    return tuple(groups)


def _dataset_manifest(root: Path) -> DatasetManifest:
    revision_path = root / "REVISION"
    if revision_path.read_text(encoding="utf-8").strip() != MAB_DATASET_REVISION:
        raise ValueError("MemoryAgentBench revision mismatch")
    content_files = tuple(
        (path, sha256) for path, sha256, _split, _byte_count in MAB_PINNED_SOURCE_FILES
    ) + (MAB_ENTITY_SOURCE_FILE[:2],)
    computed_manifest_hash = mab_dataset_content_manifest_hash(content_files)
    if computed_manifest_hash != MAB_DATASET_MANIFEST_HASH:
        raise ValueError("MemoryAgentBench content manifest hash drift")
    files: list[DatasetFile] = []
    source_specs = (
        *MAB_PINNED_SOURCE_FILES,
        (
            MAB_ENTITY_SOURCE_FILE[0],
            MAB_ENTITY_SOURCE_FILE[1],
            "catalog",
            MAB_ENTITY_SOURCE_FILE[2],
        ),
    )
    for relative_path, sha256, _split, byte_count in source_specs:
        path = root / relative_path
        _require_hash(path, sha256)
        if path.stat().st_size != byte_count:
            raise ValueError(f"MemoryAgentBench byte count mismatch: {relative_path}")
        files.append(
            DatasetFile(
                relative_path=relative_path,
                sha256=sha256,
                byte_count=byte_count,
                license_id=MAB_SOURCE_LICENSE_ID,
            )
        )
    return DatasetManifest(
        dataset_id=MAB_DATASET_ID,
        revision=MAB_DATASET_REVISION,
        split=MAB_DATASET_SPLIT,
        manifest_hash=computed_manifest_hash,
        source_files=tuple(files),
        payload_policy=MAB_PAYLOAD_POLICY,
    )


def build_mab65_manifest(root: Path) -> MabManifestBundle:
    dataset = _dataset_manifest(root)
    entity_document: Any = json.loads(
        (root / MAB_ENTITY_SOURCE_FILE[0]).read_text(encoding="utf-8")
    )
    if not isinstance(entity_document, dict) or any(
        not isinstance(uri, str) or not isinstance(entity_id, int)
        for uri, entity_id in entity_document.items()
    ):
        raise ValueError("MemoryAgentBench entity catalog must map strings to integers")
    movie_catalog = MovieCatalog.from_entity_map(entity_document)
    unicode_fingerprint = canonical_sha256(
        ["oamb-mab-redial-unicode-v1", movie_catalog.unicode_version]
    )
    rows = tuple(
        row
        for relative_path, sha256, split, _byte_count in MAB_PINNED_SOURCE_FILES
        for row in read_aligned_parquet_rows(
            root / relative_path,
            split=split,
            expected_sha256=sha256,
            source_relative_path=relative_path,
        )
    )
    groups = _selected_groups(rows)
    selected_rows = {
        row.context_entry.context_manifest_entry_id: row for members, _ in groups for row in members
    }
    selected_positions = {
        row.context_entry.context_manifest_entry_id: positions
        for members, position_groups in groups
        for row, positions in zip(members, position_groups, strict=True)
    }
    ordered_rows = tuple(
        row
        for row in _canonical_rows(rows)
        if row.context_entry.context_manifest_entry_id in selected_rows
    )
    cases: list[MabCase] = []
    for row in ordered_rows:
        for question_number in selected_positions[row.context_entry.context_manifest_entry_id]:
            entry = _case_entry(row, question_number)
            prompt_id, output_id, metric_id = _interaction_ids(row.source)
            prompt_binding = _MAB_PROMPT_BINDINGS_BY_ID[prompt_id]
            reference = canonical_json_bytes(row.answers[question_number - 1])
            cases.append(
                MabCase(
                    entry=entry,
                    case_plan=CasePlan(
                        case_manifest_entry_id=entry.case_manifest_entry_id,
                        context_manifest_entry_id=entry.context_manifest_entry_id,
                        source_question_number_1_indexed=question_number,
                        question_bytes=row.questions[question_number - 1].encode("utf-8"),
                        reference_payload=reference,
                        reference_payload_sha256=hashlib.sha256(reference).hexdigest(),
                        prompt_binding_id=prompt_id,
                        output_contract_id=output_id,
                        metric_id=metric_id,
                        judge_binding_id=None,
                        answer_max_output_tokens=prompt_binding.max_output_tokens,
                    ),
                    component=_component(row.source),
                    logical_case_label=f"{row.source}@q{question_number}",
                    source_relative_path=row.source_relative_path,
                    source_row_number_1_indexed=row.source_row_number_1_indexed,
                )
            )
    plans: list[MabPlan] = []
    for members, _positions in groups:
        member_ids = tuple(row.context_entry.context_manifest_entry_id for row in members)
        manifest_id = plan_manifest_entry_id(MAB65_WORKLOAD_ID, member_ids)
        chunks = chunk_utf8_by_tokens(members[0].context)
        chunk_hashes = tuple(hashlib.sha256(chunk.encode("utf-8")).hexdigest() for chunk in chunks)
        payload_hash = ingestion_payload_hash(chunk_hashes)
        physical_plan_id = ingestion_plan_id(manifest_id, payload_hash)
        plan_case_ids = tuple(
            case.entry.case_manifest_entry_id
            for case in cases
            if case.entry.context_manifest_entry_id in set(member_ids)
        )
        source_units = tuple(
            SourceUnit(
                source_unit_id=canonical_sha256([physical_plan_id, ordinal, chunk_hash]),
                context_manifest_entry_id=member_ids[0],
                ordinal_1_indexed=ordinal,
                payload_sha256=chunk_hash,
                payload_bytes=chunk.encode("utf-8"),
            )
            for ordinal, (chunk, chunk_hash) in enumerate(zip(chunks, chunk_hashes, strict=True), 1)
        )
        manifest = IngestionPlanManifest(
            plan_manifest_entry_id=manifest_id,
            ingestion_payload_hash=payload_hash,
            ingestion_plan_id=physical_plan_id,
            workload_id=MAB65_WORKLOAD_ID,
            ordered_member_context_manifest_entry_ids=member_ids,
            ordered_source_unit_bytes_sha256=chunk_hashes,
            ordered_case_manifest_entry_ids=plan_case_ids,
        )
        plans.append(
            MabPlan(
                manifest=manifest,
                runtime_plan=IngestionPlan(
                    ingestion_plan_id=physical_plan_id,
                    ordered_member_context_manifest_entry_ids=member_ids,
                    shared_context_sha256=members[0].context_entry.context_bytes_sha256,
                    intended_source_count=len(source_units),
                    ordered_source_units=source_units,
                    ordered_case_manifest_entry_ids=plan_case_ids,
                ),
                member_labels=tuple(
                    f"{row.source}@{row.source_row_number_1_indexed}" for row in members
                ),
                component=_component(members[0].source),
            )
        )
    ordered_plan_ids = tuple(plan.manifest.plan_manifest_entry_id for plan in plans)
    ordered_case_ids = tuple(case.entry.case_manifest_entry_id for case in cases)
    case_digest = canonical_sha256(["oamb-mab65-selected-case-ids-v1", ordered_case_ids])
    plan_digest = canonical_sha256(["oamb-mab65-selected-plan-manifest-ids-v1", ordered_plan_ids])
    if len(ordered_rows) != 29 or len(plans) != 25 or len(cases) != 65:
        raise ValueError("MAB-65 selection counts do not close at 29/25/65")
    if case_digest != MAB65_SELECTED_CASE_DIGEST or plan_digest != MAB65_SELECTED_PLAN_DIGEST:
        raise ValueError("MAB-65 selected identity digest does not match the frozen snapshot")
    manifest_fields = {
        "manifest_id": MAB65_WORKLOAD_ID,
        "workload_id": MAB65_WORKLOAD_ID,
        "logical_contexts": tuple(row.context_entry for row in ordered_rows),
        "ingestion_plans": tuple(plan.manifest for plan in plans),
        "cases": tuple(case.entry for case in cases),
    }
    case_manifest = CaseManifest.model_validate(
        {
            **manifest_fields,
            "manifest_hash": case_manifest_hash(manifest_fields),
        }
    )
    if case_manifest.manifest_hash != MAB65_CASE_MANIFEST_HASH:
        raise ValueError("MAB-65 case manifest differs from its frozen hash")
    return MabManifestBundle(
        dataset_manifest=dataset,
        case_manifest=case_manifest,
        cases=tuple(cases),
        plans=tuple(plans),
        movie_catalog=movie_catalog,
        entity_catalog_sha256=MAB_ENTITY_SOURCE_FILE[1],
        unicode_fingerprint=unicode_fingerprint,
        selected_case_digest=case_digest,
        selected_plan_digest=plan_digest,
    )


def build_mab5_manifest(full: MabManifestBundle) -> MabManifestBundle:
    selected_cases_list: list[MabCase] = []
    for plan_id, label in _SMOKE_SELECTION:
        matching_plans = tuple(
            plan for plan in full.plans if plan.manifest.plan_manifest_entry_id == plan_id
        )
        if len(matching_plans) != 1:
            raise ValueError("MAB-5 smoke plan is missing or duplicated")
        plan_case_ids = set(matching_plans[0].manifest.ordered_case_manifest_entry_ids)
        matches = tuple(
            case
            for case in full.cases
            if case.entry.case_manifest_entry_id in plan_case_ids
            and case.logical_case_label == label
        )
        if len(matches) != 1:
            raise ValueError("MAB-5 smoke case is missing or duplicated within its plan")
        selected_cases_list.append(matches[0])
    selected_case_ids = {case.entry.case_manifest_entry_id for case in selected_cases_list}
    selected_cases = tuple(
        case for case in full.cases if case.entry.case_manifest_entry_id in selected_case_ids
    )
    expected_labels = tuple(label for _plan_id, label in _SMOKE_SELECTION)
    if tuple(case.logical_case_label for case in selected_cases) != expected_labels:
        raise ValueError("MAB-5 smoke cases do not match full-manifest order")
    selected_plans: list[MabPlan] = []
    selected_context_ids: set[str] = set()
    for plan in full.plans:
        case_ids = tuple(
            case_id
            for case_id in plan.manifest.ordered_case_manifest_entry_ids
            if case_id in selected_case_ids
        )
        if not case_ids:
            continue
        selected_context_ids.update(plan.manifest.ordered_member_context_manifest_entry_ids)
        manifest = plan.manifest.model_copy(update={"ordered_case_manifest_entry_ids": case_ids})
        selected_plans.append(
            MabPlan(
                manifest=manifest,
                runtime_plan=IngestionPlan(
                    ingestion_plan_id=plan.runtime_plan.ingestion_plan_id,
                    ordered_member_context_manifest_entry_ids=(
                        plan.runtime_plan.ordered_member_context_manifest_entry_ids
                    ),
                    shared_context_sha256=plan.runtime_plan.shared_context_sha256,
                    intended_source_count=plan.runtime_plan.intended_source_count,
                    ordered_source_units=plan.runtime_plan.ordered_source_units,
                    ordered_case_manifest_entry_ids=case_ids,
                ),
                member_labels=plan.member_labels,
                component=plan.component,
            )
        )
    logical_contexts = tuple(
        context
        for context in full.case_manifest.logical_contexts
        if context.context_manifest_entry_id in selected_context_ids
    )
    fields = {
        "manifest_id": MAB5_MANIFEST_ID,
        "workload_id": MAB65_WORKLOAD_ID,
        "logical_contexts": logical_contexts,
        "ingestion_plans": tuple(plan.manifest for plan in selected_plans),
        "cases": tuple(case.entry for case in selected_cases),
    }
    smoke_manifest = CaseManifest.model_validate(
        {
            **fields,
            "manifest_hash": case_manifest_hash(fields),
        }
    )
    if len(selected_plans) != 5 or len(selected_cases) != 5:
        raise ValueError("MAB-5 must contain exactly five plans and five cases")
    smoke_case_digest = canonical_sha256(
        [
            "oamb-mab5-selected-case-ids-v1",
            tuple(case.entry.case_manifest_entry_id for case in selected_cases),
        ]
    )
    smoke_plan_digest = canonical_sha256(
        [
            "oamb-mab5-selected-plan-manifest-ids-v1",
            tuple(plan.manifest.plan_manifest_entry_id for plan in selected_plans),
        ]
    )
    if (
        smoke_manifest.manifest_hash != MAB5_CASE_MANIFEST_HASH
        or smoke_case_digest != MAB5_SELECTED_CASE_DIGEST
        or smoke_plan_digest != MAB5_SELECTED_PLAN_DIGEST
    ):
        raise ValueError("MAB-5 manifest or selected identity digest differs from its freeze")
    return MabManifestBundle(
        dataset_manifest=full.dataset_manifest,
        case_manifest=smoke_manifest,
        cases=selected_cases,
        plans=tuple(selected_plans),
        movie_catalog=full.movie_catalog,
        entity_catalog_sha256=full.entity_catalog_sha256,
        unicode_fingerprint=full.unicode_fingerprint,
        selected_case_digest=smoke_case_digest,
        selected_plan_digest=smoke_plan_digest,
    )


def _mab_evidence_item_bytes(
    candidate: Any,
    identity: str,
    text: str,
) -> bytes:
    if not candidate.evidence_kind or not text:
        raise ValueError("MAB visible evidence requires non-empty kind and text")
    return json.dumps(
        {
            "provider_evidence_identity": identity,
            "source_unit_id": candidate.source_unit_id,
            "evidence_kind": candidate.evidence_kind,
            "text": text,
            "occurred_start": candidate.occurred_start,
            "occurred_end": candidate.occurred_end,
            "mentioned_at": candidate.mentioned_at,
            "native_reference": candidate.native_reference,
            "native_truncated": candidate.native_truncated,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _truncate_mab_item_to_tokens(
    candidate: Any,
    identity: str,
    lines: Sequence[bytes],
    max_tokens: int,
) -> bytes | None:
    encoding = o200k_encoding()
    content_tokens = encoding.encode_ordinary(candidate.content)
    existing = b"\n".join(lines)
    remaining = max_tokens - count_o200k_tokens(existing)
    for end in range(min(len(content_tokens), max(remaining, 0)), 0, -1):
        try:
            text = encoding.decode(content_tokens[:end], errors="strict")
        except UnicodeDecodeError:
            continue
        item = _mab_evidence_item_bytes(candidate, identity, text)
        proposed = b"\n".join((*lines, item))
        if count_o200k_tokens(proposed) <= max_tokens:
            return item
    return None


def build_mab_visible_evidence(
    native_batch: NativeEvidenceBatch,
    policy: VisibleEvidencePolicy = MAB_VISIBLE_EVIDENCE_POLICY,
) -> VisibleEvidence:
    """Preserve provider order and token-truncate only the final included item."""

    seen_content: dict[str, str] = {}
    lines: list[bytes] = []
    included_ids: list[str] = []
    decisions: list[EvidenceDecision] = []
    stopped = False
    first_exceeded: str | None = None
    for expected_rank, candidate in enumerate(native_batch.candidates, start=1):
        if candidate.native_rank_1_indexed != expected_rank:
            raise ValueError("native evidence ranks must preserve provider order")
        if candidate.native_truncated:
            raise ValueError("visible evidence rejects native truncation")
        identity = candidate.provider_evidence_identity
        if not identity:
            raise ValueError("provider evidence identity must not be empty")
        text_hash = hashlib.sha256(candidate.content.encode("utf-8")).hexdigest()
        previous_hash = seen_content.get(identity)
        if previous_hash is not None and previous_hash != text_hash:
            raise EvidenceConflictError(
                f"provider evidence identity has conflicting content: {identity}"
            )
        if previous_hash == text_hash:
            decisions.append(
                EvidenceDecision(
                    candidate.native_id,
                    identity,
                    text_hash,
                    "duplicate",
                    "duplicate_identity_and_text",
                )
            )
            continue
        seen_content[identity] = text_hash
        if stopped:
            decisions.append(
                EvidenceDecision(
                    candidate.native_id,
                    identity,
                    text_hash,
                    "budget_dropped",
                    first_exceeded,
                )
            )
            continue
        item = _mab_evidence_item_bytes(candidate, identity, candidate.content)
        proposed = b"\n".join((*lines, item))
        exceeded: str | None = None
        if policy.max_items is not None and len(lines) + 1 > policy.max_items:
            exceeded = "max_items"
        elif (
            policy.max_characters is not None
            and len(proposed.decode("utf-8", errors="strict")) > policy.max_characters
        ):
            exceeded = "max_characters"
        elif policy.max_tokens is not None and count_o200k_tokens(proposed) > policy.max_tokens:
            exceeded = "max_tokens"
        if exceeded is None:
            lines.append(item)
            included_ids.append(candidate.native_id)
            decisions.append(
                EvidenceDecision(candidate.native_id, identity, text_hash, "kept", None)
            )
            continue
        stopped = True
        first_exceeded = exceeded
        truncated = (
            _truncate_mab_item_to_tokens(candidate, identity, lines, policy.max_tokens)
            if exceeded == "max_tokens" and policy.max_tokens is not None
            else None
        )
        if truncated is not None:
            lines.append(truncated)
            included_ids.append(candidate.native_id)
            decisions.append(
                EvidenceDecision(
                    candidate.native_id,
                    identity,
                    text_hash,
                    "truncated",
                    "max_tokens",
                )
            )
        else:
            decisions.append(
                EvidenceDecision(
                    candidate.native_id,
                    identity,
                    text_hash,
                    "budget_dropped",
                    exceeded,
                )
            )
    payload = b"\n".join(lines)
    if native_batch.candidates and not payload:
        raise ValueError("non-empty native evidence produced an empty visible context")
    return VisibleEvidence(
        canonical_bytes=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        included_native_ids=tuple(included_ids),
        token_count=count_o200k_tokens(payload),
        candidate_count=len(native_batch.candidates),
        kept_count=len(lines),
        dropped_count=sum(
            decision.disposition in {"duplicate", "budget_dropped"} for decision in decisions
        ),
        truncated_count=sum(decision.disposition == "truncated" for decision in decisions),
        first_exceeded_limit=first_exceeded,
        decisions=tuple(decisions),
        payload_byte_count=len(payload),
        character_count=len(payload.decode("utf-8", errors="strict")),
        tokenizer_fingerprint=tokenizer_fingerprint(),
    )


class MemoryAgentBenchWorkload:
    """WorkloadPort implementation over one frozen MAB-65 or MAB-5 bundle."""

    def __init__(self, bundle: MabManifestBundle) -> None:
        self._bundle = bundle

    def resolve_sources(self) -> DatasetManifest:
        return self._bundle.dataset_manifest

    def build_case_manifest(self, dataset_manifest: DatasetManifest) -> CaseManifest:
        if dataset_manifest != self._bundle.dataset_manifest:
            raise ValueError("MemoryAgentBench workload received a different dataset manifest")
        return self._bundle.case_manifest

    def iter_ingestion_plans(self, case_manifest: CaseManifest) -> tuple[IngestionPlan, ...]:
        if case_manifest != self._bundle.case_manifest:
            raise ValueError("MemoryAgentBench workload received a different case manifest")
        return tuple(plan.runtime_plan for plan in self._bundle.plans)

    def iter_case_plans(self, case_manifest: CaseManifest) -> tuple[CasePlan, ...]:
        if case_manifest != self._bundle.case_manifest:
            raise ValueError("MemoryAgentBench workload received a different case manifest")
        return tuple(case.case_plan for case in self._bundle.cases)

    def render_retrieval_query(self, case_plan: CasePlan) -> bytes:
        try:
            binding = _MAB_PROMPT_BINDINGS_BY_ID[case_plan.prompt_binding_id]
        except KeyError as exc:
            raise ValueError("unknown MemoryAgentBench prompt binding") from exc
        return binding.render_retrieval_query(case_plan.question_bytes)

    def build_visible_evidence(
        self,
        native_batch: NativeEvidenceBatch,
        policy: VisibleEvidencePolicy,
    ) -> VisibleEvidence:
        if policy != MAB_VISIBLE_EVIDENCE_POLICY:
            raise ValueError("MemoryAgentBench visible-evidence policy must remain frozen")
        return build_mab_visible_evidence(native_batch, policy)

    def render_answer(
        self,
        case_plan: CasePlan,
        visible_evidence: VisibleEvidence,
    ) -> RenderedPrompt:
        verify_visible_evidence_hash(visible_evidence)
        try:
            binding = _MAB_PROMPT_BINDINGS_BY_ID[case_plan.prompt_binding_id]
        except KeyError as exc:
            raise ValueError("unknown MemoryAgentBench prompt binding") from exc
        pack = binding.pack
        if case_plan.output_contract_id != pack.manifest.output_contract_id:
            raise ValueError("MemoryAgentBench output contract differs from its PromptPack")
        if case_plan.answer_max_output_tokens != binding.max_output_tokens:
            raise ValueError("MemoryAgentBench answer output ceiling differs from its PromptPack")
        rendered = render_prompt(
            pack,
            "answer_user",
            {
                "question": case_plan.question_bytes.decode("utf-8", errors="strict"),
                "retrieved_context": visible_evidence.canonical_bytes.decode(
                    "utf-8", errors="strict"
                ),
            },
        )
        if visible_evidence.canonical_bytes not in rendered.canonical_bytes:
            raise ValueError("rendered MemoryAgentBench prompt changed visible evidence bytes")
        return rendered

    def evaluate(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
    ) -> DeterministicEvaluation:
        reference = json.loads(case_plan.reference_payload.decode("utf-8", errors="strict"))
        if (
            not isinstance(reference, list)
            or not reference
            or any(not isinstance(value, str) for value in reference)
        ):
            raise ValueError("MemoryAgentBench reference must be a non-empty string array")
        prediction = answer.parsed_value.decode("utf-8", errors="strict")
        if case_plan.metric_id == "mab-redial-recall-at-5-v1":
            parsed_titles = json.loads(prediction)
            if (
                not isinstance(parsed_titles, list)
                or not 1 <= len(parsed_titles) <= 20
                or any(not isinstance(title, str) or not title for title in parsed_titles)
            ):
                raise ValueError("parsed ReDial answer must be a non-empty title array")
            resolved = resolve_ranked_movies(tuple(parsed_titles), self._bundle.movie_catalog)
            recall = score_recall_at_5(
                resolved,
                tuple(int(value) for value in reference),
                self._bundle.movie_catalog,
            )
            numerator, denominator = recall.numerator, recall.denominator
            catalog_by_id = {movie.entity_id: movie for movie in self._bundle.movie_catalog.movies}
            related_titles = set(recall.gold_normalized_titles) | set(
                item.catalog_normalized_title for item in resolved
            )
            trace = {
                "metric_id": case_plan.metric_id,
                "parsed_titles": parsed_titles,
                "gold_entity_ids": list(recall.gold_entity_ids),
                "predicted_normalized_titles": list(recall.predicted_normalized_titles),
                "resolutions": [
                    {
                        "raw_title": item.raw_title,
                        "cleaned_title": item.cleaned_title,
                        "normalized_title": item.normalized_title,
                        "entity_id": item.entity_id,
                        "catalog_title": item.catalog_title,
                        "catalog_normalized_title": item.catalog_normalized_title,
                        "distance": item.distance,
                        "tied_entity_ids": list(item.tied_entity_ids),
                    }
                    for item in resolved
                ],
                "gold": [
                    {
                        "entity_id": entity_id,
                        "normalized_title": catalog_by_id[entity_id].normalized_title,
                    }
                    for entity_id in recall.gold_entity_ids
                ],
                "related_duplicate_title_groups": [
                    {"normalized_title": title, "entity_ids": list(entity_ids)}
                    for title, entity_ids in self._bundle.movie_catalog.duplicate_normalized_titles
                    if title in related_titles
                ],
                "numerator": numerator,
                "denominator": denominator,
            }
        elif case_plan.metric_id in {"mab-exact-v1", "mab-substring-em-v1"}:
            substring = case_plan.metric_id == "mab-substring-em-v1"
            if case_plan.output_contract_id == "mab-raw-or-first-line-max-v1":
                parsed_candidates = json.loads(prediction)
                if (
                    not isinstance(parsed_candidates, list)
                    or len(parsed_candidates) != 2
                    or any(not isinstance(value, str) for value in parsed_candidates)
                ):
                    raise ValueError("parsed raw-or-first-line answer must contain two strings")
                scored = any(
                    max_over_gold_answers(value, reference, substring=substring)
                    for value in parsed_candidates
                )
                trace_prediction: str | list[str] = parsed_candidates
            else:
                scored = max_over_gold_answers(prediction, reference, substring=substring)
                trace_prediction = prediction
            numerator, denominator = int(scored), 1
            trace = {
                "metric_id": case_plan.metric_id,
                "prediction": trace_prediction,
                "gold_answers": reference,
                "score": scored,
            }
        else:
            raise ValueError(f"unsupported MemoryAgentBench metric: {case_plan.metric_id}")
        trace_bytes = canonical_json_bytes(trace)
        return DeterministicEvaluation(
            metric_id=case_plan.metric_id,
            result_sha256=hashlib.sha256(trace_bytes).hexdigest(),
            numerator=numerator,
            denominator=denominator,
            trace_bytes=trace_bytes,
        )

    def finalize_judge(
        self,
        case_plan: CasePlan,
        answer: AnswerValue,
        judge_answer: AnswerValue,
    ) -> DeterministicEvaluation:
        del case_plan, answer, judge_answer
        raise ValueError("MemoryAgentBench does not use a judge model")

    def validate_records(self, records: WorkloadRecordSet) -> tuple[WorkloadRuleResult, ...]:
        passed = (
            len(records.logical_context_records) == len(self._bundle.case_manifest.logical_contexts)
            and len(records.ingestion_plan_records) == len(self._bundle.plans)
            and len(records.case_records) == len(self._bundle.cases)
        )
        return (
            WorkloadRuleResult(
                rule_id=f"{self._bundle.case_manifest.manifest_id}-counts-v1",
                passed=passed,
                evidence_refs=tuple(record.case_occurrence_id for record in records.case_records),
            ),
        )


def chunk_utf8_by_tokens(
    text: str,
    *,
    max_tokens: int = MAB_CHUNK_MAX_TOKENS,
) -> tuple[str, ...]:
    """Split ordinary UTF-8 text on strict token boundaries without overlap."""

    if not text:
        raise ValueError("MAB context must not be empty")
    if "\ufffd" in text:
        raise ValueError("MAB context must not contain a replacement character")
    if max_tokens <= 0:
        raise ValueError("MAB chunk token limit must be positive")
    encoding = o200k_encoding()
    tokens = encoding.encode_ordinary(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        while end > start:
            try:
                chunk = encoding.decode(tokens[start:end], errors="strict")
            except UnicodeDecodeError:
                end -= 1
                continue
            if 0 < len(encoding.encode_ordinary(chunk)) <= max_tokens:
                chunks.append(chunk)
                start = end
                break
            end -= 1
        else:
            raise ValueError("MAB chunk limit cannot contain one complete UTF-8 scalar")
    if "".join(chunks) != text:
        raise ValueError("MAB chunks do not reassemble the source text")
    return tuple(chunks)
