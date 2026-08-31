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


def _dataset_manifest_for_test(source: Path) -> Any:
    from oamb.contracts.specifications import DatasetFile, DatasetManifest

    source_file = DatasetFile(
        relative_path=source.name,
        sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        byte_count=source.stat().st_size,
        license_id="NOASSERTION",
    )
    return DatasetManifest(
        dataset_id="chronology-fixture",
        revision="chronology-v1",
        split="test",
        manifest_hash="a" * 64,
        source_files=(source_file,),
        payload_policy="generated-fixture",
    )


def _chronology_row() -> dict[str, object]:
    row = _row()
    row["answer_session_ids"] = ["z-tie-first"]
    row["haystack_session_ids"] = ["late", "z-tie-first", "a-tie-second", "middle"]
    row["haystack_dates"] = [
        "2024/03/01 (Fri) 00:00",
        "2024/02/27 (Tue) 08:00",
        "2024/02/27 (Tue) 08:00",
        "2024/02/28 (Wed) 01:02",
    ]
    row["haystack_sessions"] = [
        [
            {"role": "user", "content": "Late ☕."},
            {"role": "assistant", "content": "Still kept."},
        ],
        [
            {"role": "assistant", "content": "First tied assistant."},
            {"role": "user", "content": "First tied user.", "has_answer": True},
        ],
        [{"role": "user", "content": "Second tied."}],
        [{"role": "user", "content": "Middle."}],
    ]
    return row


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


def test_loader_stably_orders_session_occurrences_by_timestamp_then_source_ordinal(
    tmp_path: Path,
) -> None:
    """Catches source-array order or session ID being used as chronology."""

    lme = require_longmemeval()
    source = tmp_path / "longmemeval.json"
    expected_sha256 = _write_rows(source, [_chronology_row()])

    loaded = lme.load_longmemeval_rows(source, expected_sha256=expected_sha256)

    assert tuple(session.session_id for session in loaded[0].sessions) == (
        "z-tie-first",
        "a-tie-second",
        "middle",
        "late",
    )
    assert tuple(session.original_ordinal_1_indexed for session in loaded[0].sessions) == (
        2,
        3,
        4,
        1,
    )


def test_loader_keeps_turn_bytes_and_post_question_sessions_as_audit_facts(
    tmp_path: Path,
) -> None:
    """Catches sorting turns or filtering supplied sessions at question_date."""

    lme = require_longmemeval()
    source = tmp_path / "longmemeval.json"
    expected_sha256 = _write_rows(source, [_chronology_row()])

    row = lme.load_longmemeval_rows(source, expected_sha256=expected_sha256)[0]
    bundle = lme._build_bundle(
        _dataset_manifest_for_test(source),
        (row,),
        workload_id="chronology-v1",
    )
    units = bundle.ingestion_plans[0].ordered_source_units

    assert lme.render_lme_session_document(row.sessions[0].messages) == (
        b'[{"role":"assistant","content":"First tied assistant."},'
        b'{"role":"user","content":"First tied user."}]'
    )
    assert tuple(unit.source_reference for unit in units) == (
        "z-tie-first",
        "a-tie-second",
        "middle",
        "late",
    )
    late = units[-1]
    assert late.payload_bytes == (
        b'[{"role":"user","content":"Late \xe2\x98\x95."},'
        b'{"role":"assistant","content":"Still kept."}]'
    )
    assert dict(late.source_metadata) == {
        "question_id": "example-q",
        "session_id": "late",
        "original_ordinal_1_indexed": "1",
        "after_question_date": "true",
    }


def test_workload_owns_the_high_thinking_judge_output_ceiling(tmp_path: Path) -> None:
    lme = require_longmemeval()
    from oamb.contracts.ports import AnswerValue, RawReferenceHandle

    source = tmp_path / "longmemeval.json"
    expected_sha256 = _write_rows(source, [_row()])
    row = lme.load_longmemeval_rows(source, expected_sha256=expected_sha256)[0]
    bundle = lme._build_bundle(
        _dataset_manifest_for_test(source),
        (row,),
        workload_id="judge-ceiling-v1",
    )
    answer = AnswerValue(
        raw_reference=RawReferenceHandle("d" * 64),
        raw_answer=b"answer",
        parsed_value=b"answer",
        parsed_value_sha256=hashlib.sha256(b"answer").hexdigest(),
    )

    request = lme.LongMemEvalWorkload(bundle).evaluate(bundle.case_plans[0], answer)

    assert request.max_output_tokens == 1024


def test_bundle_derives_manifest_hashes_and_source_ids_after_chronology_sort(
    tmp_path: Path,
) -> None:
    """Catches hashing or identity derivation before chronological canonicalization."""

    lme = require_longmemeval()
    raw = _chronology_row()
    chronological_order = (1, 2, 3, 0)
    canonical = dict(raw)
    for key in ("haystack_session_ids", "haystack_dates", "haystack_sessions"):
        values = cast(list[object], raw[key])
        canonical[key] = [values[index] for index in chronological_order]
    shuffled_path = tmp_path / "shuffled.json"
    canonical_path = tmp_path / "canonical.json"
    shuffled_hash = _write_rows(shuffled_path, [raw])
    canonical_hash = _write_rows(canonical_path, [canonical])
    shuffled_row = lme.load_longmemeval_rows(shuffled_path, expected_sha256=shuffled_hash)[0]
    canonical_row = lme.load_longmemeval_rows(canonical_path, expected_sha256=canonical_hash)[0]
    dataset = _dataset_manifest_for_test(shuffled_path)

    shuffled = lme._build_bundle(dataset, (shuffled_row,), workload_id="chronology-v1")
    ordered = lme._build_bundle(dataset, (canonical_row,), workload_id="chronology-v1")

    assert shuffled.case_manifest == ordered.case_manifest
    assert tuple(
        unit.source_unit_id for unit in shuffled.ingestion_plans[0].ordered_source_units
    ) == tuple(unit.source_unit_id for unit in ordered.ingestion_plans[0].ordered_source_units)


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
