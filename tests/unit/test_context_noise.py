"""Token-weighted noise is independent of gold labels and whole-block sizes."""

import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).parents[2] / "src/oamb/workloads/context_noise.py"
TOKENIZER = {"name": "o200k_base", "version": "0.14.0", "method": "encode_ordinary"}


def module() -> dict[str, Any]:
    assert SCRIPT.exists(), "offline context-noise counter is missing"
    return runpy.run_path(str(SCRIPT))


def span(start: int, end: int, category: str) -> dict[str, Any]:
    return {"start": start, "end": end, "category": category, "reason": "Reviewed meaning"}


def test_mixed_context_counts_whole_encoding_once_and_keeps_metadata_separate():
    # Hand-checked: 3 header tokens, 3 direct tokens (including '.\n'), 4 unrelated tokens.
    result = module()["count_context"](
        "[1]\nUseful evidence.\nUnrelated noise.",
        [span(0, 4, "format"), span(4, 21, "direct"), span(21, 37, "unrelated")],
        TOKENIZER,
    )
    assert result["context_tokens"] == 10
    assert result["tokens_by_category"] == {
        "direct": 3,
        "related": 0,
        "unrelated": 4,
        "format": 3,
        "uncertain": 0,
        "unreviewed": 0,
    }


@pytest.mark.parametrize(
    "spans",
    [
        [span(0, 2, "direct"), span(1, 3, "unrelated")],
        [span(-1, 2, "direct")],
        [span(0, 100, "direct")],
        [span(0, 1, "invented")],
        [span(True, 2, "direct")],
    ],
)
def test_invalid_intervals_are_rejected(spans):
    with pytest.raises(ValueError, match="span"):
        module()["count_context"]("ABC", spans, TOKENIZER)


def test_unicode_offsets_must_preserve_utf8_and_uncovered_text_is_unreviewed():
    with pytest.raises(ValueError, match="UTF-8"):
        module()["count_context"]("你好", [span(1, 3, "direct")], TOKENIZER)
    result = module()["count_context"]("你好\n", [], TOKENIZER)
    assert result["tokens_by_category"]["unreviewed"] == 1
    assert result["tokens_by_category"]["format"] == 1


def test_tokenizer_drift_is_rejected():
    with pytest.raises(ValueError, match="tokenizer"):
        module()["count_context"]("text", [], {**TOKENIZER, "version": "0.0.0"})


def test_reference_aggregation_weights_tokens_and_reports_partial_coverage():
    rows = [
        {
            "noise_status": "reviewed",
            "context_tokens": 10000,
            "unrelated_text_tokens": 3000,
            "tokenizer": TOKENIZER,
        },
        {
            "noise_status": "reviewed",
            "context_tokens": 2000,
            "unrelated_text_tokens": 200,
            "tokenizer": TOKENIZER,
        },
        {
            "noise_status": "unreviewed",
            "context_tokens": 9000,
            "unrelated_text_tokens": 0,
            "tokenizer": TOKENIZER,
        },
    ]
    result = module()["aggregate_noise"](rows, expected_cases=4)
    assert result["noise_ratio"] == pytest.approx(0.26666666666666666)
    assert result["reviewed_cases"] == 2
    assert result["expected_cases"] == 4
    assert result["unreviewed_cases"] == 1
    assert result["missing_cases"] == 1


@pytest.mark.parametrize(
    "text,category,status",
    [
        ("", "direct", "empty_context"),
        ("Hello", "uncertain", "uncertain"),
        ("Hello", "unreviewed", "unreviewed"),
    ],
)
def test_empty_or_unresolved_review_cannot_publish_zero_noise(text, category, status):
    assessment = {
        "source_review_complete": True,
        "span_review_complete": True,
        "task_interpretation": "Reviewed the entire task",
        "spans": [span(0, len(text), category)] if text else [],
    }
    result = module()["review_noise"](text, assessment, TOKENIZER, [])
    assert result["noise_status"] == status
    assert result["noise_ratio"] is None


def test_uncovered_tail_of_one_token_keeps_review_incomplete():
    # 'Hello' is one token. Reviewing only its first byte does not review 'ello'.
    result = module()["review_noise"](
        "Hello",
        {
            "source_review_complete": True,
            "span_review_complete": True,
            "task_interpretation": "Question meaning",
            "spans": [span(0, 1, "direct")],
        },
        TOKENIZER,
        [],
    )
    assert result["noise_status"] == "unreviewed"
    assert result["noise_ratio"] is None


def test_conservative_review_reports_a_bound_without_an_exact_ratio():
    result = module()["review_noise"](
        "Hello",
        {
            "source_review_complete": True,
            "span_review_complete": False,
            "task_interpretation": "Task reviewed against sources",
            "measurement": "lower_bound",
            "spans": [span(0, 5, "unrelated")],
        },
        TOKENIZER,
        [],
    )
    assert result["noise_status"] == "lower_bound"
    assert result["noise_lower_bound"] == 1
    assert result["noise_ratio"] is None


def test_empty_tokenizer_cache_fails_without_http(tmp_path):
    code = f"""import runpy, socket
def forbidden(*args, **kwargs):
    raise AssertionError("HTTP attempted")
socket.socket.connect = forbidden
module = runpy.run_path({str(SCRIPT)!r})
try:
    module["encoder_for"]({TOKENIZER!r})
except ValueError as error:
    assert "offline tokenizer" in str(error), str(error)
else:
    raise AssertionError("missing tokenizer assets accepted")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "TIKTOKEN_CACHE_DIR": str(tmp_path / "empty-cache")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("quote", ["Useful", "invented"])
def test_evidence_requirements_validate_quotes_against_complete_sources(quote):
    assessment = {
        "source_review_complete": True,
        "span_review_complete": True,
        "task_interpretation": "Source-backed evidence",
        "spans": [span(0, 6, "direct")],
        "evidence_requirements": [
            {
                "description": "Required evidence",
                "status": "present",
                "context_quotes": ["Useful"],
                "source_quotes": [{"session_id": "s", "message_index": 1, "quote": quote}],
            }
        ],
    }
    sessions = [
        {"session_id": "s", "messages": [{"message_index": 1, "content": "Useful evidence"}]}
    ]
    if quote == "invented":
        with pytest.raises(ValueError, match="source quote"):
            module()["review_noise"]("Useful", assessment, TOKENIZER, sessions)
    else:
        result = module()["review_noise"]("Useful", assessment, TOKENIZER, sessions)
        assert result["evidence_coverage"]["present"] == 1
        assert result["noise_ratio"] == 0
