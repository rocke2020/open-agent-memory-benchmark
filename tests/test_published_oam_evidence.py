from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path

import pytest
import tiktoken

RESULTS = Path(__file__).resolve().parents[1] / "eval_results"


@pytest.mark.parametrize(
    ("dataset", "arm", "count", "first_score", "final_score", "first_passes", "final_passes"),
    [
        ("lme500", "oam", 500, 0.906, 0.94, 453, 470),
        ("beam10m", "oam", 200, 0.5989232503607503, 0.6155621392496391, 136, 142),
        (
            "beam10m",
            "hindsight-saved-context-replay",
            200,
            0.504018308080808,
            0.5215738636363636,
            108,
            112,
        ),
    ],
)
def test_public_question_evidence_reproduces_historical_scores(
    dataset, arm, count, first_score, final_score, first_passes, final_passes
):
    root = RESULTS / dataset
    manifest = json.loads((root / "manifest.json").read_bytes())
    records = []
    for relative in manifest["arms"][arm]["files"]:
        path = root / relative
        assert path.stat().st_size <= 16 * 1024 * 1024
        body = gzip.decompress(path.read_bytes())
        assert (b"/Users/" + b"rocke_dong") not in body
        document = json.loads(body)
        assert document["dataset"] == dataset and document["arm"] == arm
        records.extend(document["results"])

    assert len(records) == len({q["question_id"] for q in records}) == count
    for q in records:
        assert q["question"].strip() and q["context"].strip()
        assert q["gold_answers"] or q["rubric"]
        assert hashlib.sha256(q["context"].encode()).hexdigest() == q["context_sha256"]
        encoding = tiktoken.get_encoding(q["context_tokenizer"].split(":", 1)[1])
        assert len(encoding.encode_ordinary(q["context"])) == q["context_tokens"]
        assert len(q["source"]["record_sha256"]) == 64
        selected = q["retry"] if q["retry"] is not None else q["first_pass"]
        assert q["policy_final_attempt"] == ("retry" if q["retry"] is not None else "first_pass")
        assert q["answer"] == selected["answer"]
        assert q["score"] == selected["score"]
        assert q["correct"] == (q["score"] >= (1 if dataset == "lme500" else 0.5))
        for attempt in (q["first_pass"], q["retry"]):
            if attempt is None:
                continue
            assert attempt["answer"].strip()
            assert 0 <= attempt["score"] <= 1
            assert attempt["answer_effort"] in {"high", "max"}
            assert attempt["judge_effort"] == "high"
        if dataset == "beam10m":
            assert q["rubric"]
            for attempt in (q["first_pass"], q["retry"]):
                if attempt is None:
                    continue
                answer_call = attempt["answer_call"]
                assert answer_call["request"]["reasoning_effort"] == attempt["answer_effort"]
                assert q["context"] in "\n".join(
                    message["content"] for message in answer_call["request"]["messages"]
                )
                judgments = attempt["judgments"]
                assert len(judgments) == len(q["rubric"])
                assert [j["rubric_index"] for j in judgments] == list(range(len(q["rubric"])))
                assert math.isclose(
                    sum(j["score"] for j in judgments) / len(judgments), attempt["score"]
                )
                for judgment in judgments:
                    assert judgment["reason"].strip()
                    call = judgment["call"]
                    assert call["response"]["finish_reason"] == "stop"
                    assert call["request"]["reasoning_effort"] == "high"
                    assert len(call["source_receipt_sha256"]) == 64
                    visible = json.loads(call["response"]["content"])
                    assert visible["reason"] == judgment["reason"]
                    assert visible["score"] == judgment["raw_score"]
        if q["retry"] is not None:
            assert q["retry"]["context_sha256"] == q["context_sha256"]
            assert q["retry"]["answer_effort"] == "max"

    assert math.isclose(sum(q["first_pass"]["score"] for q in records) / count, first_score)
    assert math.isclose(sum(q["score"] for q in records) / count, final_score)
    assert sum(q["first_pass"]["correct"] for q in records) == first_passes
    assert sum(q["correct"] for q in records) == final_passes
    if dataset == "lme500":
        assert sum(q["context_tokens"] for q in records) == 6_201_308
    else:
        assert {str(q["conversation_id"]) for q in records} == {str(i) for i in range(1, 11)}
        assert sum(q["retry"] is not None for q in records) == (44 if arm == "oam" else 43)
