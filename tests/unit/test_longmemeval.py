from __future__ import annotations

import hashlib
import importlib
import json
import math
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest

QUESTION_TYPES = (
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
)


def require_longmemeval() -> ModuleType:
    try:
        return importlib.import_module("oamb.workloads.longmemeval")
    except ModuleNotFoundError:
        pytest.fail("oamb.workloads.longmemeval is not implemented", pytrace=False)


def _row(
    *, question_id: str = "example-q", question_type: str = "single-session-user"
) -> dict[str, object]:
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question": "What did I say?",
        "answer": "Café",
        "question_date": "2024/02/29 (Thu) 23:07",
        "answer_session_ids": ["session-1"],
        "haystack_session_ids": ["session-1"],
        "haystack_dates": ["2024/02/28 (Wed) 01:02"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "I chose Café.", "has_answer": True},
                {"role": "assistant", "content": "Noted."},
            ]
        ],
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> str:
    payload = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_timestamp_parser_checks_weekday_and_emits_utc_wall_time() -> None:
    lme = require_longmemeval()

    assert lme.parse_lme_timestamp("2024/02/29 (Thu) 23:07") == "2024-02-29T23:07:00+00:00"
    with pytest.raises(ValueError, match="weekday"):
        lme.parse_lme_timestamp("2024/02/29 (Fri) 23:07")
    with pytest.raises(ValueError, match="format"):
        lme.parse_lme_timestamp("2024-02-29 23:07")


def test_session_document_removes_has_answer_and_preserves_exact_utf8_bytes() -> None:
    lme = require_longmemeval()
    messages = cast(list[list[dict[str, Any]]], _row()["haystack_sessions"])[0]

    assert lme.render_lme_session_document(messages) == (
        b'[{"role":"user","content":"I chose Caf\xc3\xa9."},'
        b'{"role":"assistant","content":"Noted."}]'
    )


def test_loader_rejects_checksum_and_aligned_list_failures(tmp_path: Path) -> None:
    lme = require_longmemeval()
    source = tmp_path / "longmemeval.json"
    row = _row()
    expected_sha256 = _write_rows(source, [row])

    with pytest.raises(ValueError, match="checksum"):
        lme.load_longmemeval_rows(source, expected_sha256="0" * 64)

    row["haystack_dates"] = []
    expected_sha256 = _write_rows(source, [row])
    with pytest.raises(ValueError, match="aligned"):
        lme.load_longmemeval_rows(source, expected_sha256=expected_sha256)


def test_loader_preserves_both_answer_session_label_representations(tmp_path: Path) -> None:
    lme = require_longmemeval()
    source = tmp_path / "longmemeval.json"
    row = _row()
    row["answer_session_ids"] = ["different-session"]
    expected_sha256 = _write_rows(source, [row])

    loaded = lme.load_longmemeval_rows(source, expected_sha256=expected_sha256)

    assert loaded[0].answer_session_ids == ("different-session",)
    assert loaded[0].message_has_answer_session_ids == ("session-1",)
    assert loaded[0].has_answer_label_mismatch is True


def test_loader_rejects_non_finite_numeric_answers(tmp_path: Path) -> None:
    lme = require_longmemeval()
    source = tmp_path / "longmemeval.json"
    row = _row()
    row["answer"] = math.nan
    expected_sha256 = _write_rows(source, [row])

    with pytest.raises(ValueError, match="finite"):
        lme.load_longmemeval_rows(source, expected_sha256=expected_sha256)


@pytest.mark.parametrize(
    ("question_type", "expected_template"),
    (
        ("knowledge-update", "judge_knowledge_update"),
        ("multi-session", "judge_multi_session"),
        ("single-session-assistant", "judge_single_session_assistant"),
        ("single-session-preference", "judge_single_session_preference"),
        ("single-session-user", "judge_single_session_user"),
        ("temporal-reasoning", "judge_temporal_reasoning"),
    ),
)
def test_judge_renderer_selects_all_six_question_type_branches(
    question_type: str,
    expected_template: str,
) -> None:
    lme = require_longmemeval()

    rendered = lme.render_lme_judge_prompt(
        question_type=question_type,
        question="When?",
        reference="Yesterday",
        model_response="It was yesterday.",
        unanswerable=False,
    )

    assert rendered.prompt_pack_id == "oamb-lme-judge-v1"
    assert rendered.template_name == expected_template
    assert b"Question: When?" in rendered.canonical_bytes
    assert b"Model Response: It was yesterday." in rendered.canonical_bytes


def test_judge_renderer_uses_independent_unanswerable_branch() -> None:
    lme = require_longmemeval()

    rendered = lme.render_lme_judge_prompt(
        question_type="single-session-user",
        question="Unknown?",
        reference="The source does not say.",
        model_response="I cannot determine that.",
        unanswerable=True,
    )

    assert rendered.template_name == "judge_abs"
    assert b"unanswerable question" in rendered.canonical_bytes
    assert b"Explanation: The source does not say." in rendered.canonical_bytes


def test_workload_finalizes_strict_yes_no_judge_output_as_an_exact_fraction() -> None:
    lme = require_longmemeval()
    from types import SimpleNamespace

    from oamb.contracts.ports import AnswerValue, CasePlan, RawReferenceHandle

    workload = lme.LongMemEvalWorkload(
        SimpleNamespace(case_manifest=SimpleNamespace(cases=()), selected_rows=())
    )
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"question",
        reference_payload=b'"answer"',
        reference_payload_sha256="c" * 64,
        prompt_binding_id="oamb-lme-answer-v1",
        output_contract_id="lme-answer-text-v1",
        metric_id="lme-judged-accuracy-v1",
        judge_binding_id="oamb-lme-judge-v1",
        answer_max_output_tokens=8_192,
    )
    answer = AnswerValue(
        raw_reference=RawReferenceHandle("d" * 64),
        raw_answer=b"answer",
        parsed_value=b"answer",
        parsed_value_sha256=hashlib.sha256(b"answer").hexdigest(),
    )
    judge = AnswerValue(
        raw_reference=RawReferenceHandle("e" * 64),
        raw_answer=b"yes\n",
        parsed_value=b"yes\n",
        parsed_value_sha256=hashlib.sha256(b"yes\n").hexdigest(),
    )

    evaluation = workload.finalize_judge(case_plan, answer, judge)

    assert (evaluation.numerator, evaluation.denominator) == (1, 1)
    trace = json.loads(evaluation.trace_bytes)
    assert trace["judge_raw_reference"] == "e" * 64
    assert trace["parsed_answer_sha256"] == answer.parsed_value_sha256
    with pytest.raises(ValueError, match="yes or no"):
        workload.finalize_judge(
            case_plan,
            answer,
            AnswerValue(
                raw_reference=RawReferenceHandle("f" * 64),
                raw_answer=b"maybe",
                parsed_value=b"maybe",
                parsed_value_sha256=hashlib.sha256(b"maybe").hexdigest(),
            ),
        )


def test_answer_renderer_binds_exact_visible_evidence_bytes() -> None:
    lme = require_longmemeval()
    evidence = b'{"text":"literal <|endoftext|>"}'

    rendered = lme.render_lme_answer_prompt(question="What is literal?", evidence=evidence)

    assert rendered.prompt_pack_id == "oamb-lme-answer-v1"
    assert evidence in rendered.canonical_bytes
    assert rendered.sha256 == hashlib.sha256(rendered.canonical_bytes).hexdigest()


def test_workload_answer_renderer_rejects_a_visible_evidence_hash_mismatch() -> None:
    lme = require_longmemeval()
    from types import SimpleNamespace

    from oamb.contracts.ports import CasePlan, VisibleEvidence

    workload = lme.LongMemEvalWorkload(
        SimpleNamespace(
            case_manifest=SimpleNamespace(cases=()),
            selected_rows=(),
        )
    )
    case_plan = CasePlan(
        case_manifest_entry_id="a" * 64,
        context_manifest_entry_id="b" * 64,
        source_question_number_1_indexed=1,
        question_bytes=b"question",
        reference_payload=b'"answer"',
        reference_payload_sha256="c" * 64,
        prompt_binding_id="oamb-lme-answer-v1",
        output_contract_id="lme-answer-text-v1",
        metric_id="lme-judged-accuracy-v1",
        judge_binding_id="oamb-lme-judge-v1",
        answer_max_output_tokens=8_192,
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


def test_lme_workload_rejects_visible_evidence_policy_drift() -> None:
    lme = require_longmemeval()
    from types import SimpleNamespace

    from oamb.contracts.ports import NativeEvidenceBatch, RawReferenceHandle
    from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY

    workload = lme.LongMemEvalWorkload(
        SimpleNamespace(
            case_manifest=SimpleNamespace(cases=()),
            selected_rows=(),
        )
    )
    changed_policy = LME_VISIBLE_EVIDENCE_POLICY.__class__(
        max_items=LME_VISIBLE_EVIDENCE_POLICY.max_items - 1,
        max_characters=LME_VISIBLE_EVIDENCE_POLICY.max_characters,
        max_tokens=LME_VISIBLE_EVIDENCE_POLICY.max_tokens,
    )

    with pytest.raises(ValueError, match="frozen"):
        workload.build_visible_evidence(
            NativeEvidenceBatch(RawReferenceHandle("a" * 64), ()),
            changed_policy,
        )


def test_judge_prompt_manifest_binds_exact_source_extractions() -> None:
    lme = require_longmemeval()
    judge_pack = lme.LME_PROMPT_PACKS[1]

    assert judge_pack.manifest.source_file_sha256 == (
        "ecce9c4c79dc89d99534ac17b383a5cbb5b9f0c69ee98adaf0684742e3d95251"
    )
    assert {
        template.template_name: template.source_extracted_sha256
        for template in judge_pack.manifest.templates
    } == {
        "judge_knowledge_update": "183a9b3a6197ec620940f610cdc1207201ec98c1113dd633ea685cfc322fafac",
        "judge_multi_session": "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc",
        "judge_single_session_assistant": "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc",
        "judge_single_session_preference": "741ee3bcbea7ff5e8ed359acef61d2f8ded3de021bbcff6ee13de455f2e2aa9b",
        "judge_single_session_user": "fba020ba3d57982efdc9a937c1c01f897b789a608c7f88e60244121f6505e5bc",
        "judge_temporal_reasoning": "8d33a5fdd83afeeb4592454a965eab43d1fcb2dedc042d1d3892f4254be6c273",
        "judge_abs": "5c0b365a1e1d06db36377c735432b56e122ca3c428f89faf61d43a0d5a7e050b",
    }
