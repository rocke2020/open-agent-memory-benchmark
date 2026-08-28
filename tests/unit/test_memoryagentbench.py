from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
import tiktoken


def test_mab_chunker_reassembles_utf8_with_zero_overlap_and_token_bound() -> None:
    from oamb.workloads.memoryagentbench import chunk_utf8_by_tokens

    text = ("alpha 中文 👩‍💻 é <|endoftext|> " * 900).rstrip()
    chunks = chunk_utf8_by_tokens(text, max_tokens=97)
    encoding = tiktoken.get_encoding("o200k_base")

    assert len(chunks) > 1
    assert "".join(chunks) == text
    assert b"".join(chunk.encode("utf-8") for chunk in chunks) == text.encode("utf-8")
    assert all(1 <= len(encoding.encode_ordinary(chunk)) <= 97 for chunk in chunks)


def test_mab_chunker_rejects_empty_or_nonpositive_limit() -> None:
    from oamb.workloads.memoryagentbench import chunk_utf8_by_tokens

    for text, limit in (("", 4_096), ("payload", 0), ("replacement \ufffd", 4_096)):
        try:
            chunk_utf8_by_tokens(text, max_tokens=limit)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid chunk request must fail closed")


def test_mab_parquet_reader_rejects_misaligned_questions_answers_and_ids(
    tmp_path: Path,
) -> None:
    from oamb.workloads.memoryagentbench import read_aligned_parquet_rows

    path = tmp_path / "misaligned.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "context": "context",
                    "questions": ["q1", "q2"],
                    "answers": [["a1"]],
                    "metadata": {"source": "eventqa_65536", "qa_pair_ids": ["id1", "id2"]},
                }
            ]
        ),
        path,
    )

    with pytest.raises(ValueError, match="must align"):
        read_aligned_parquet_rows(
            path,
            split="Accurate_Retrieval",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


@pytest.mark.parametrize(
    ("questions", "answers"),
    (([""], [["answer"]]), (["question"], [[""]])),
)
def test_mab_parquet_reader_rejects_empty_question_or_answer_text(
    tmp_path: Path,
    questions: list[str],
    answers: list[list[str]],
) -> None:
    from oamb.workloads.memoryagentbench import read_aligned_parquet_rows

    path = tmp_path / "empty-text.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "context": "context",
                    "questions": questions,
                    "answers": answers,
                    "metadata": {"source": "eventqa_65536", "qa_pair_ids": ["id1"]},
                }
            ]
        ),
        path,
    )

    with pytest.raises(ValueError, match="non-empty"):
        read_aligned_parquet_rows(
            path,
            split="Accurate_Retrieval",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )


def test_five_mab_prompt_packs_bind_source_extraction_wrapper_and_output_contracts() -> None:
    from oamb.workloads.memoryagentbench import (
        MAB_ANSWER_WRAPPER_SHA256,
        MAB_PROMPT_PACKS,
        MAB_PROMPT_SOURCE_SHA256,
    )
    from oamb.workloads.prompts import render_prompt

    assert MAB_PROMPT_SOURCE_SHA256 == (
        "148c40d48d19f155ae845482c4417ba59cfa7ae4e194019509e023bd3a8755dd"
    )
    assert MAB_ANSWER_WRAPPER_SHA256 == (
        "c268f8751d27ec75fc7fe3b94d9d576a9a1d0bdaa70cb582a067c919e4e5edd0"
    )
    assert {
        binding.pack.manifest.prompt_pack_id: (
            binding.extracted_query_sha256,
            binding.pack.manifest.templates[0].content_sha256,
            binding.pack.manifest.output_contract_id,
        )
        for binding in MAB_PROMPT_PACKS
    } == {
        "oamb-mab-eventqa-rag-v1": (
            "c86b0dc5db229403976a20bcef1cc626ddc922569955a9d17c420171698ecb49",
            "a4cecd405eec83eb9602a8768678aa4a86c655df4875c381688248fb0dcbb3af",
            "mab-first-line-v1",
        ),
        "oamb-mab-icl-rag-v1": (
            "e9df61cd16b5e95d08fe93b1bde82f2665b68ecff0bc6463fb43f69e3c570262",
            "9d8df44d2ce398ad52a79c06f8294cff726461960903b5fda320041540358839",
            "mab-first-line-v1",
        ),
        "oamb-mab-redial-rag-v1": (
            "96e3f5bae8307491abb3a85e0ddbebbd89a5d10d642f8721aa0433eb3203be5e",
            "3617e4899bdeaea5b5dbad29e8a9c1baa12a8ea643af223c34979e97c31f253d",
            "mab-redial-ranked-movies-v1",
        ),
        "oamb-mab-detectiveqa-rag-v1": (
            "0117e352348af57cf8dd30a31286bfae1cd9f360a04d3d88eb5f6a041145a387",
            "856495424713cc3ef96ce84e33408cdeea687c2595c74d4a00c46aa6f3eef8bb",
            "mab-raw-or-first-line-max-v1",
        ),
        "oamb-mab-factconsolidation-rag-v1": (
            "ca4ea130355adab357ce8cd00bb11c1cdbe7cb945454764a66f1ac7f692f501d",
            "ed89c60fd7f8bec5835a0a1d57c625acc8854ca3836e54d2c5be28b795707db9",
            "mab-raw-or-first-line-max-v1",
        ),
    }
    assert all(
        binding.pack.manifest.templates[0].adaptation_id == "oamb-mab-answer-wrapper-v1"
        for binding in MAB_PROMPT_PACKS
    )
    assert {
        binding.pack.manifest.prompt_pack_id: binding.max_output_tokens
        for binding in MAB_PROMPT_PACKS
    } == {
        "oamb-mab-eventqa-rag-v1": 40,
        "oamb-mab-icl-rag-v1": 20,
        "oamb-mab-redial-rag-v1": 512,
        "oamb-mab-detectiveqa-rag-v1": 2_000,
        "oamb-mab-factconsolidation-rag-v1": 10,
    }
    rendered = render_prompt(
        MAB_PROMPT_PACKS[0].pack,
        "answer_user",
        {"question": "What happened?", "retrieved_context": "provider evidence"},
    ).canonical_bytes.decode("utf-8")
    assert rendered.startswith("Answer the user request using only the retrieved memory evidence")
    assert "<evidence>\nprovider evidence\n</evidence>" in rendered
    assert rendered.endswith("What happened?\n\n The event that happens next is:")


@pytest.mark.parametrize(
    ("prompt_binding_id", "expected_query"),
    (
        (
            "oamb-mab-eventqa-rag-v1",
            "Based on the context you memorized, complete the task below:\n\nWhat next?\n\n The event that happens next is:",
        ),
        (
            "oamb-mab-icl-rag-v1",
            'Use the provided mapping from the context to numerical label to assign a numerical label to the context. Only output "label: {label}" and nothing else. \n\nQuestion:What next? \n\n label:',
        ),
        (
            "oamb-mab-redial-rag-v1",
            "Pretend you are a movie recommender system. You need to recommend movies based on the dialogues you have memorized. Now I will give you a new conversation between a user and you (a recommender system). Based on the conversation, you reply me with 20 recommendations without extra sentences. \n\nFor Example:\n\n[Conversation]\n\nThe recommendations are: \n1.movie1\n2.movie2\n...\n\n Here is the conversation: What next? \n\n The recommendations are: \n",
        ),
        (
            "oamb-mab-detectiveqa-rag-v1",
            "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n What next? \n\n",
        ),
        (
            "oamb-mab-factconsolidation-rag-v1",
            "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, What next? \nAnswer:",
        ),
    ),
)
def test_mab_retrieval_query_uses_the_attributed_rag_agent_template(
    prompt_binding_id: str,
    expected_query: str,
) -> None:
    from oamb.contracts.ports import CasePlan
    from oamb.workloads.memoryagentbench import MemoryAgentBenchWorkload

    workload = MemoryAgentBenchWorkload(SimpleNamespace())  # type: ignore[arg-type]
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"What next?",
        reference_payload=b'["answer"]',
        reference_payload_sha256="c" * 64,
        prompt_binding_id=prompt_binding_id,
        output_contract_id="fixture-output-v1",
        metric_id="fixture-metric-v1",
        judge_binding_id=None,
    )

    query = workload.render_retrieval_query(case_plan)

    assert query == expected_query.encode("utf-8")
    assert query != case_plan.question_bytes
    assert b"Retrieved memory evidence" not in query


def test_dataset_content_manifest_recomputes_frozen_hash_and_binds_path_order() -> None:
    from oamb.workloads.memoryagentbench import mab_dataset_content_manifest_hash

    files = (
        (
            "data/Accurate_Retrieval-00000-of-00001.parquet",
            "56c3cd80fb6731a3e53cd1a6be3148f54df60ff2d290ee50e28f8acebf9655c1",
        ),
        (
            "data/Test_Time_Learning-00000-of-00001.parquet",
            "5338753be48f925d03318eed66117286e3489025fabe050a547bd086cd7d79c0",
        ),
        (
            "data/Long_Range_Understanding-00000-of-00001.parquet",
            "5ab175461954db67770d4a4cb69e569b513ebb96aceb9ee79b57f67488bcd539",
        ),
        (
            "data/Conflict_Resolution-00000-of-00001.parquet",
            "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45",
        ),
        ("entity2id.json", "63353aca481bc9558b502f91cb98f6fa26438796fdd7e0bc06b5a1532126e8b5"),
    )

    assert mab_dataset_content_manifest_hash(files) == (
        "78011e09488f2ca88c13dd5518016b56668b1d49174fdec85a1db0eaccf00a2c"
    )
    assert mab_dataset_content_manifest_hash(tuple(reversed(files))) != (
        "78011e09488f2ca88c13dd5518016b56668b1d49174fdec85a1db0eaccf00a2c"
    )
    changed_path = (("data/renamed.parquet", files[0][1]), *files[1:])
    assert mab_dataset_content_manifest_hash(changed_path) != (
        "78011e09488f2ca88c13dd5518016b56668b1d49174fdec85a1db0eaccf00a2c"
    )


def test_mab_visible_evidence_preserves_order_deduplicates_and_truncates_only_final_item() -> None:
    from oamb.contracts.ports import (
        NativeEvidenceBatch,
        NativeEvidenceCandidate,
        RawReferenceHandle,
        VisibleEvidencePolicy,
    )
    from oamb.workloads.memoryagentbench import build_mab_visible_evidence

    candidates = (
        NativeEvidenceCandidate("one", 1, "alpha", None, "identity-one"),
        NativeEvidenceCandidate("duplicate", 2, "alpha", None, "identity-one"),
        NativeEvidenceCandidate("two", 3, "beta " * 100, None, "identity-two"),
        NativeEvidenceCandidate("three", 4, "gamma", None, "identity-three"),
    )
    evidence = build_mab_visible_evidence(
        NativeEvidenceBatch(RawReferenceHandle("a" * 64), candidates),
        VisibleEvidencePolicy(max_items=10, max_characters=10_000, max_tokens=180),
    )

    lines = tuple(json.loads(line) for line in evidence.canonical_bytes.splitlines())
    assert evidence.included_native_ids == ("one", "two")
    assert tuple(line["provider_evidence_identity"] for line in lines) == (
        "identity-one",
        "identity-two",
    )
    assert lines[1]["text"] and ("beta " * 100).startswith(lines[1]["text"])
    assert tuple(decision.disposition for decision in evidence.decisions) == (
        "kept",
        "duplicate",
        "truncated",
        "budget_dropped",
    )
    assert evidence.truncated_count == 1
    assert evidence.token_count <= 180


def test_mab_visible_evidence_rejects_a_native_read_truncation() -> None:
    from oamb.contracts.ports import (
        NativeEvidenceBatch,
        NativeEvidenceCandidate,
        RawReferenceHandle,
    )
    from oamb.workloads.memoryagentbench import build_mab_visible_evidence

    with pytest.raises(ValueError, match="native truncation"):
        build_mab_visible_evidence(
            NativeEvidenceBatch(
                RawReferenceHandle("a" * 64),
                (
                    NativeEvidenceCandidate(
                        "partial",
                        1,
                        "partial text",
                        None,
                        "partial-identity",
                        native_truncated=True,
                    ),
                ),
            )
        )


def test_mab_visible_evidence_requires_an_explicit_provider_evidence_identity() -> None:
    from oamb.contracts.ports import (
        NativeEvidenceBatch,
        NativeEvidenceCandidate,
        RawReferenceHandle,
    )
    from oamb.workloads.memoryagentbench import build_mab_visible_evidence

    with pytest.raises(ValueError, match="provider evidence identity"):
        build_mab_visible_evidence(
            NativeEvidenceBatch(
                RawReferenceHandle("a" * 64),
                (NativeEvidenceCandidate("native-id", 1, "content", None),),
            )
        )


def test_mab_answer_renderer_rejects_a_visible_evidence_hash_mismatch() -> None:
    from oamb.contracts.ports import CasePlan, VisibleEvidence
    from oamb.workloads.memoryagentbench import MemoryAgentBenchWorkload

    workload = MemoryAgentBenchWorkload(SimpleNamespace())  # type: ignore[arg-type]
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"question",
        reference_payload=b'["answer"]',
        reference_payload_sha256="c" * 64,
        prompt_binding_id="oamb-mab-eventqa-rag-v1",
        output_contract_id="mab-first-line-v1",
        metric_id="mab-substring-em-v1",
        judge_binding_id=None,
        answer_max_output_tokens=40,
    )

    with pytest.raises(ValueError, match="hash"):
        workload.render_answer(
            case_plan,
            VisibleEvidence(
                canonical_bytes=b"{}",
                sha256="0" * 64,
                included_native_ids=(),
            ),
        )


def test_mab_answer_renderer_rejects_prompt_output_ceiling_drift() -> None:
    from dataclasses import replace

    from oamb.contracts.ports import CasePlan, VisibleEvidence
    from oamb.workloads.memoryagentbench import MemoryAgentBenchWorkload

    workload = MemoryAgentBenchWorkload(SimpleNamespace())  # type: ignore[arg-type]
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"question",
        reference_payload=b'["answer"]',
        reference_payload_sha256="c" * 64,
        prompt_binding_id="oamb-mab-eventqa-rag-v1",
        output_contract_id="mab-first-line-v1",
        metric_id="mab-substring-em-v1",
        judge_binding_id=None,
        answer_max_output_tokens=40,
    )
    evidence_bytes = b"{}"
    evidence = VisibleEvidence(
        canonical_bytes=evidence_bytes,
        sha256=hashlib.sha256(evidence_bytes).hexdigest(),
        included_native_ids=(),
    )

    with pytest.raises(ValueError, match="ceiling"):
        workload.render_answer(
            replace(case_plan, answer_max_output_tokens=41),
            evidence,
        )
    with pytest.raises(ValueError, match="output contract"):
        workload.render_answer(
            replace(case_plan, output_contract_id="mab-raw-or-first-line-max-v1"),
            evidence,
        )


def test_memoryagentbench_workload_is_port_and_exact_evaluation_has_rebuildable_trace() -> None:
    from oamb.contracts.ids import canonical_json_bytes
    from oamb.contracts.ports import (
        AnswerValue,
        CasePlan,
        RawReferenceHandle,
        WorkloadPort,
    )
    from oamb.workloads.memoryagentbench import MemoryAgentBenchWorkload

    workload = MemoryAgentBenchWorkload(SimpleNamespace())  # type: ignore[arg-type]
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"question",
        reference_payload=canonical_json_bytes(("The Answer",)),
        reference_payload_sha256="c" * 64,
        prompt_binding_id="oamb-mab-eventqa-rag-v1",
        output_contract_id="mab-first-line-v1",
        metric_id="mab-exact-v1",
        judge_binding_id=None,
    )
    answer = AnswerValue(
        raw_reference=RawReferenceHandle("d" * 64),
        raw_answer=b"Answer: the answer\nignored",
        parsed_value=b"the answer",
        parsed_value_sha256=hashlib.sha256(b"the answer").hexdigest(),
    )

    assert isinstance(workload, WorkloadPort)
    result = workload.evaluate(case_plan, answer)
    assert result.numerator == 1
    assert result.denominator == 1
    assert result.trace_bytes is not None
    assert json.loads(result.trace_bytes) == {
        "gold_answers": ["The Answer"],
        "metric_id": "mab-exact-v1",
        "prediction": "the answer",
        "score": True,
    }


def test_mab_default_visible_evidence_ceiling_is_exactly_32768_tokens() -> None:
    from oamb.contracts.ports import (
        NativeEvidenceBatch,
        NativeEvidenceCandidate,
        RawReferenceHandle,
    )
    from oamb.workloads.memoryagentbench import (
        MAB_VISIBLE_EVIDENCE_MAX_TOKENS,
        build_mab_visible_evidence,
    )

    evidence = build_mab_visible_evidence(
        NativeEvidenceBatch(
            RawReferenceHandle("e" * 64),
            (
                NativeEvidenceCandidate(
                    "oversized",
                    1,
                    "retrieved-memory " * 40_000,
                    None,
                    "oversized-identity",
                ),
            ),
        )
    )

    assert MAB_VISIBLE_EVIDENCE_MAX_TOKENS == 32_768
    assert evidence.token_count == MAB_VISIBLE_EVIDENCE_MAX_TOKENS
    assert evidence.truncated_count == 1
    assert evidence.decisions[0].disposition == "truncated"


def test_mab_redial_evaluation_consumes_parsed_title_array_not_raw_wire_text() -> None:
    from oamb.contracts.ids import canonical_json_bytes
    from oamb.contracts.ports import AnswerValue, CasePlan, RawReferenceHandle
    from oamb.workloads.memoryagentbench import MemoryAgentBenchWorkload
    from oamb.workloads.redial import MovieCatalog

    catalog = MovieCatalog.from_entity_map({"https://example.test/Toy_Story_(1995)": 1})
    workload = MemoryAgentBenchWorkload(SimpleNamespace(movie_catalog=catalog))  # type: ignore[arg-type]
    reference = canonical_json_bytes(("1",))
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"recommend",
        reference_payload=reference,
        reference_payload_sha256=hashlib.sha256(reference).hexdigest(),
        prompt_binding_id="oamb-mab-redial-rag-v1",
        output_contract_id="mab-redial-ranked-movies-v1",
        metric_id="mab-redial-recall-at-5-v1",
        judge_binding_id=None,
    )
    parsed = canonical_json_bytes(("Toy Story",))
    result = workload.evaluate(
        case_plan,
        AnswerValue(
            raw_reference=RawReferenceHandle("f" * 64),
            raw_answer=b"this raw wire is deliberately not a ranked list",
            parsed_value=parsed,
            parsed_value_sha256=hashlib.sha256(parsed).hexdigest(),
        ),
    )

    assert (result.numerator, result.denominator) == (1, 1)


def test_mab_redial_evaluation_trace_preserves_resolution_and_collision_evidence() -> None:
    from oamb.contracts.ids import canonical_json_bytes
    from oamb.contracts.ports import AnswerValue, CasePlan, RawReferenceHandle
    from oamb.workloads.memoryagentbench import MemoryAgentBenchWorkload
    from oamb.workloads.redial import MovieCatalog

    catalog = MovieCatalog.from_entity_map(
        {
            "movie/Alien_(1979)": 20,
            "movie/ALIEN_(1979)": 10,
        }
    )
    workload = MemoryAgentBenchWorkload(SimpleNamespace(movie_catalog=catalog))  # type: ignore[arg-type]
    reference = canonical_json_bytes(("10", "20"))
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"recommend",
        reference_payload=reference,
        reference_payload_sha256=hashlib.sha256(reference).hexdigest(),
        prompt_binding_id="oamb-mab-redial-rag-v1",
        output_contract_id="mab-redial-ranked-movies-v1",
        metric_id="mab-redial-recall-at-5-v1",
        judge_binding_id=None,
    )
    parsed = canonical_json_bytes(("alien",))

    result = workload.evaluate(
        case_plan,
        AnswerValue(
            raw_reference=RawReferenceHandle("f" * 64),
            raw_answer=b"1. alien",
            parsed_value=parsed,
            parsed_value_sha256=hashlib.sha256(parsed).hexdigest(),
        ),
    )
    assert result.trace_bytes is not None
    trace = json.loads(result.trace_bytes)

    assert trace["resolutions"] == [
        {
            "catalog_normalized_title": "alien",
            "catalog_title": "ALIEN",
            "cleaned_title": "alien",
            "distance": 0,
            "entity_id": 10,
            "normalized_title": "alien",
            "raw_title": "alien",
            "tied_entity_ids": [10, 20],
        }
    ]
    assert trace["gold"] == [
        {"entity_id": 10, "normalized_title": "alien"},
        {"entity_id": 20, "normalized_title": "alien"},
    ]
    assert trace["related_duplicate_title_groups"] == [
        {"entity_ids": [10, 20], "normalized_title": "alien"}
    ]


def test_mab_raw_or_first_line_evaluation_consumes_parsed_candidate_array() -> None:
    from oamb.contracts.ids import canonical_json_bytes
    from oamb.contracts.ports import AnswerValue, CasePlan, RawReferenceHandle
    from oamb.workloads.memoryagentbench import MemoryAgentBenchWorkload

    workload = MemoryAgentBenchWorkload(SimpleNamespace())  # type: ignore[arg-type]
    reference = canonical_json_bytes(("gold",))
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"question",
        reference_payload=reference,
        reference_payload_sha256=hashlib.sha256(reference).hexdigest(),
        prompt_binding_id="oamb-mab-detectiveqa-rag-v1",
        output_contract_id="mab-raw-or-first-line-max-v1",
        metric_id="mab-exact-v1",
        judge_binding_id=None,
    )
    parsed = canonical_json_bytes(("wrong raw", "gold"))
    result = workload.evaluate(
        case_plan,
        AnswerValue(
            raw_reference=RawReferenceHandle("f" * 64),
            raw_answer=b"wire content must not be reparsed here",
            parsed_value=parsed,
            parsed_value_sha256=hashlib.sha256(parsed).hexdigest(),
        ),
    )

    assert (result.numerator, result.denominator) == (1, 1)
