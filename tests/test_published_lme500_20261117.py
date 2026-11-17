"""Recompute the immutable 18k campaign without providers or private files."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
import statistics
from pathlib import Path

import tiktoken

from oamb.reporting.historical import _evidence

ROOT = Path(__file__).resolve().parents[1] / "eval_results/lme500-20261117"


def test_18k_evidence_preserves_scores_contexts_calls_and_source_identity():
    manifest = json.loads((ROOT / "manifest.json").read_bytes())
    revision = "38bd45d82f24aa33932a7d4b7afdf6ee27585f72"
    assert manifest["recall_product_revision"] == revision
    hashes = (ROOT / manifest["recall_source_hashes"]).read_bytes()
    assert hashlib.sha256(hashes).hexdigest() == manifest["recall_source_hashes_sha256"]
    sources = json.loads(hashes)
    assert len(sources) == 116
    assert all(path.startswith("src/oam/") and len(value) == 64 for path, value in sources.items())
    records = []
    for relative in manifest["arms"]["oam"]["files"]:
        body = (ROOT / relative).read_bytes()
        assert hashlib.sha256(body).hexdigest() == manifest["shard_sha256"][relative]
        document = json.loads(gzip.decompress(body))
        assert document["dataset"] == "lme500" and document["arm"] == "oam"
        records.extend(document["results"])
    assert len(records) == len({q["question_id"] for q in records}) == 500
    assert sum(q["correct"] for q in records) == 476
    assert sum(q["correct"] for q in records if q["cohort"] == "hard100") == 83
    assert sum(q["correct"] for q in records if q["cohort"] == "remaining400") == 393
    old = ROOT.parent / "lme500"
    prior_manifest = json.loads((old / "manifest.json").read_bytes())
    prior = {
        q["question_id"]: q
        for relative in prior_manifest["arms"]["oam"]["files"]
        for q in json.loads(gzip.decompress((old / relative).read_bytes()))["results"]
    }
    transitions: dict[tuple[bool, bool], int] = {}
    encoding = tiktoken.get_encoding("o200k_base")
    contexts = []
    timings = []
    carried = set()
    response_ids = set()
    failed = 0
    for q in records:
        previous = prior[q["question_id"]]
        for field in ("question", "question_type", "gold_answers", "rubric"):
            assert q[field] == previous[field]
        pair = (previous["correct"], q["correct"])
        transitions[pair] = transitions.get(pair, 0) + 1
        assert q["source"]["product_revision"] == revision
        if q.get("evaluation_status") == "carried_incorrect_gold_no":
            carried.add(q["question_id"])
            assert q["score"] == 0 and q["correct"] is False
            assert q["first_pass"] is None and q["retry"] is None
            assert not {"context", "context_tokens", "retrieve_time_ms", "answer"} & q.keys()
            continue
        context = q["context"]
        assert hashlib.sha256(context.encode()).hexdigest() == q["context_sha256"]
        assert len(encoding.encode_ordinary(context)) == q["context_tokens"] <= 18_000
        contexts.append(q["context_tokens"])
        assert len(q["retrieval_samples_ms"]) == 2
        assert math.isclose(statistics.mean(q["retrieval_samples_ms"]), q["retrieve_time_ms"])
        timings.extend(q["retrieval_samples_ms"])
        attempt = q["first_pass"]
        assert q["retry"] is None
        assert q["answer"] == attempt["answer"]
        assert q["score"] == attempt["score"] == int(q["correct"])
        for role in ("answer", "judge"):
            assert attempt[f"{role}_model"] == "minimax-m3.1-flash"
            assert attempt[f"{role}_effort"] == "adaptive"
            call = attempt[f"{role}_call"]
            request = call["request"]
            assert request["model"] == "minimax-m3.1-flash"
            assert request["thinking"] == {"type": "adaptive"}
            assert request["reasoning_split"] is True
            assert request["n"] == 1 and request["temperature"] == 0
            assert request["top_p"] == 1 and request["stop"] is None
            assert {"messages", "response_format", "reasoning_effort"}.isdisjoint(request)
            assert len(call["request_messages_sha256"]) == 64
            assert call["source_receipt_sha256"] == q["source"]["record_sha256"]
            assert call["status_code"] == 200
            response = call["response"]
            assert response["finish_reason"] == "stop" and response["content"].strip()
            assert response["usage"]["total_tokens"] > 0
            response_ids.add(response["id"])
            failed += len(call["failed_attempts"])
        assert attempt["answer_call"]["response"]["content"] == attempt["answer"]
        judge = attempt["judge_call"]["response"]["content"].strip()
        assert judge == attempt["judge_reason"]
        if judge.lower() in {"yes", "no"}:
            verdict = judge.lower() == "yes"
        else:
            match = re.search(r'["\']?correct["\']?\s*[:=]\s*(true|false)', judge, re.I)
            assert match is not None
            verdict = match[1].lower() == "true"
        assert verdict is q["correct"]
    assert carried == {"0a995998", "dd2973ad", "370a8ff4"}
    assert transitions == {
        (True, True): 454,
        (False, True): 22,
        (True, False): 12,
        (False, False): 12,
    }
    assert len(contexts) == 497 and sum(contexts) == 5_652_671
    assert len(timings) == len(response_ids) == 994
    assert failed == 21
    assert round(statistics.mean(timings), 2) == 332.95
    assert round(statistics.median(timings), 2) == 328.59
    assert round(sorted(timings)[math.ceil(0.95 * len(timings)) - 1], 2) == 428.25
    summary = _evidence(ROOT)["arms"]["oam"]["summary"]
    assert summary["count"] == 500 and summary["passes"] == 476
    assert summary["context_count"] == 497 and summary["carried_count"] == 3
    assert summary["retrieval_sample_count"] == 994
    assert summary["mean"] == 0.952
    assert summary["context_total"] == 5_652_671
