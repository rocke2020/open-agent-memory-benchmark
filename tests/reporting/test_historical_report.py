from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _payload(body: bytes) -> dict:
    match = re.search(
        rb'<script id="evidence" type="application/octet-stream">([^<]+)</script>', body
    )
    assert match is not None
    value = json.loads(gzip.decompress(base64.b64decode(match.group(1))))
    assert isinstance(value, dict)
    shared = {q["question_id"]: q.get("context") for q in value["arms"]["oam"]["results"]}
    for arm in value["arms"].values():
        for q in arm["results"]:
            if "context_ref" in q:
                q["context"] = shared[q["context_ref"]]
    return value


@pytest.mark.parametrize(
    ("dataset", "count", "mean", "passes"),
    [("lme500", 500, 0.932, 466), ("beam10m", 200, 0.6155621392496391, 142)],
)
def test_html_replays_published_questions_without_changing_context(dataset, count, mean, passes):
    from oamb.reporting.historical import render_historical_report

    root = ROOT / "eval_results" / dataset
    body = render_historical_report(root)
    assert body == render_historical_report(root)
    assert len(body) < 16 * 1024 * 1024
    payload = _payload(body)
    published = (root / "report.html").read_bytes()
    assert _payload(published) == payload
    # Compression headers may differ across supported Python/zlib platforms.
    for pattern in (rb"<style>.*?</style>", rb"<script>.*?</script>"):
        assert re.findall(pattern, published, re.S) == re.findall(pattern, body, re.S)
    records = payload["arms"]["oam"]["results"]
    assert len(records) == len({q["question_id"] for q in records}) == count
    assert payload["arms"]["oam"]["summary"]["mean"] == pytest.approx(mean)
    assert payload["arms"]["oam"]["summary"]["passes"] == passes
    for name, arm in payload["arms"].items():
        by_id = {q["question_id"]: q for q in arm["results"]}
        for relative in payload["manifest"]["arms"][name]["files"]:
            original = json.loads(gzip.decompress((root / relative).read_bytes()))
            for q in original["results"]:
                visible = by_id[q["question_id"]]
                assert visible["context"] == q["context"]
                assert visible["source_file"] == relative
                assert (
                    hashlib.sha256(visible["context"].encode()).hexdigest() == q["context_sha256"]
                )
                selected = q["retry"] or q["first_pass"]
                assert visible["selected_answer"]["answer"] == selected["answer"]
                assert visible["selected_answer"]["score"] == selected["score"]
                assert visible["selected_answer"]["answer_effort"] == selected["answer_effort"]
                assert "first_pass" not in visible and "retry" not in visible
                assert visible["selected_answer"]["judgments"] == [
                    {field: j[field] for field in ("rubric_index", "criterion", "score", "reason")}
                    for j in selected.get("judgments", [])
                ]
    if dataset == "beam10m":
        replay = payload["arms"]["hindsight-saved-context-replay"]
        assert replay["summary"]["mean"] == pytest.approx(0.5215738636363636)


@pytest.fixture
def saved_evidence(tmp_path):
    root = tmp_path / "lme500"
    root.mkdir()
    source = ROOT / "eval_results/lme500"
    manifest = json.loads((source / "manifest.json").read_bytes())
    relative = manifest["arms"]["oam"]["files"][0]
    document = json.loads(gzip.decompress((source / relative).read_bytes()))
    document["results"] = document["results"][:1]
    (root / "manifest.json").write_text(
        json.dumps(
            {"dataset": "lme500", "arms": {"oam": {"files": ["oam.json.gz"], "questions": 1}}}
        )
    )
    (root / "oam.json.gz").write_bytes(gzip.compress(json.dumps(document).encode()))
    return root


def test_report_builds_from_saved_evidence_without_a_checksum_inventory(saved_evidence):
    from oamb.reporting.historical import build_historical_report

    target = build_historical_report(saved_evidence)
    payload = _payload(target.read_bytes())
    original = json.loads(gzip.decompress((saved_evidence / "oam.json.gz").read_bytes()))
    assert payload["arms"]["oam"]["summary"]["count"] == 1
    assert payload["arms"]["oam"]["results"][0]["context"] == original["results"][0]["context"]
    before = target.read_bytes()
    assert build_historical_report(saved_evidence).read_bytes() == before
    assert {path.name for path in saved_evidence.iterdir()} == {
        "manifest.json",
        "oam.json.gz",
        "report.html",
    }


def test_report_rejects_a_context_that_differs_from_its_saved_hash(saved_evidence):
    from oamb.reporting.historical import render_historical_report

    shard = saved_evidence / "oam.json.gz"
    document = json.loads(gzip.decompress(shard.read_bytes()))
    document["results"][0]["context"] += "changed context"
    shard.write_bytes(gzip.compress(json.dumps(document).encode()))
    with pytest.raises(ValueError, match="saved context hash mismatch"):
        render_historical_report(saved_evidence)


def test_paired_lme_labels_preserve_model_identity():
    from oamb.reporting.historical import render_historical_report

    body = render_historical_report(ROOT / "eval_results/lme500")
    payload = _payload(body)
    assert payload["manifest"]["arms"]["oam"]["label"] == "OAM · MiniMax M3.1 Flash adaptive"
    assert payload["arms"]["deepseek-max"]["summary"]["passes"] == 463
    assert b"data.manifest.arms[name].label" in body


def test_report_summary_names_primary_model_and_recorded_evaluation():
    from oamb.reporting.historical import render_historical_report

    body = render_historical_report(ROOT / "eval_results/lme500")
    assert b'node("p", label("oam") + "' in body
    assert b"Recorded evaluation" in body
    assert b"Historical evaluation" not in body


def test_reports_show_selected_answers_without_retired_policy_controls():
    from oamb.reporting.historical import render_historical_report

    for dataset in ("lme500", "beam10m"):
        body = render_historical_report(ROOT / "eval_results" / dataset)
        script = re.findall(rb"<script>(.*?)</script>", body, re.S)[0]
        for label in (
            b"First pass",
            b"Policy final",
            b"Recorded retries",
            b"Outcome policy",
            b"Max-effort retry",
            b"With recorded retry",
            b"Retry score decreased",
        ):
            assert label not in script
        assert b"const attempt = q => q.selected_answer;" in script
        assert b'node("p", "Selected saved answer"' in script
        payload = _payload(body)
        for arm in payload["arms"].values():
            for q in arm["results"]:
                selected = q["selected_answer"]
                assert selected["answer"].strip()
                assert hashlib.sha256(q["context"].encode()).hexdigest() == q["context_sha256"]


@pytest.fixture
def carried_evidence(tmp_path):
    root = tmp_path / "carried"
    root.mkdir()
    context = "Complete saved context <script>quoted text</script>"
    records = []
    for index in range(500):
        record: dict[str, Any] = {
            "question_id": f"question-{index:03d}",
            "question": "What is remembered?",
            "gold_answers": ["A remembered fact"],
            "rubric": [],
            "question_type": "single-session-user",
            "source": {"record_sha256": "source-record-hash"},
            "first_pass": None,
            "retry": None,
        }
        if index < 497:
            correct = index < 476
            record.update(
                context=context,
                context_sha256=hashlib.sha256(context.encode()).hexdigest(),
                context_tokens=10,
                context_tokenizer="o200k_base",
                retrieve_time_ms=999,
                retrieval_samples_ms=[1, 9],
                first_pass={
                    "answer": "A remembered fact",
                    "correct": correct,
                    "score": int(correct),
                    "answer_model": "minimax-m3.1-flash",
                    "judge_model": "minimax-m3.1-flash",
                    "answer_effort": "adaptive",
                    "judge_effort": "adaptive",
                    "judge_reason": "A saved judgment",
                },
            )
        else:
            record.update(evaluation_status="carried_incorrect_gold_no", score=0, correct=False)
        records.append(record)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset": "lme500",
                "display_notes": ["497 current cases and three historical No outcomes."],
                "arms": {"oam": {"files": ["oam.json.gz"], "questions": 500}},
            }
        )
    )
    (root / "oam.json.gz").write_bytes(
        gzip.compress(json.dumps({"dataset": "lme500", "arm": "oam", "results": records}).encode())
    )
    return root


def test_carried_no_preserves_denominator_without_fabricating_current_evidence(carried_evidence):
    from oamb.reporting.historical import render_historical_report

    payload = _payload(render_historical_report(carried_evidence))
    arm = payload["arms"]["oam"]
    assert arm["summary"] == {
        "mean": 0.952,
        "passes": 476,
        "count": 500,
        "context_mean": 10,
        "context_total": 4970,
        "retrieval_median_ms": 5,
        "context_count": 497,
        "carried_count": 3,
        "retrieval_sample_count": 994,
    }
    assert payload["manifest"]["display_notes"] == [
        "497 current cases and three historical No outcomes."
    ]
    assert arm["results"][0]["retrieval_samples_ms"] == [1, 9]
    assert arm["results"][0]["retrieve_time_ms"] == 999
    assert arm["results"][0]["context"] == ("Complete saved context <script>quoted text</script>")
    for carried in arm["results"][-3:]:
        assert carried["evaluation_status"] == "carried_incorrect_gold_no"
        assert carried["score"] == 0
        assert carried["correct"] is False
        assert carried["selected_answer"] is None
        assert (
            not {
                "context",
                "context_sha256",
                "context_tokens",
                "retrieve_time_ms",
                "retrieval_samples_ms",
                "answer_call",
                "judge_call",
            }
            & carried.keys()
        )


@pytest.mark.parametrize(
    "tamper",
    [
        {"score": 1},
        {"score": False},
        {"correct": True},
        {"context": "fabricated"},
        {"context": None},
        {"context_sha256": "fabricated"},
        {"context_tokens": 0},
        {"context_tokenizer": "o200k_base"},
        {"retrieve_time_ms": 0},
        {"retrieval_samples_ms": []},
        {"first_pass": {"score": 0}},
        {"retry": {"score": 0}},
        {"answer_call": {}},
        {"judge_call": {}},
    ],
)
def test_carried_no_rejects_current_evidence_or_changed_outcome(carried_evidence, tamper):
    from oamb.reporting.historical import render_historical_report

    shard = carried_evidence / "oam.json.gz"
    document = json.loads(gzip.decompress(shard.read_bytes()))
    document["results"][-1].update(tamper)
    shard.write_bytes(gzip.compress(json.dumps(document).encode()))
    with pytest.raises(ValueError, match="invalid carried No"):
        render_historical_report(carried_evidence)


def test_carried_no_is_not_a_beam_result(carried_evidence):
    from oamb.reporting.historical import render_historical_report

    manifest_path = carried_evidence / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["dataset"] = "beam10m"
    manifest_path.write_text(json.dumps(manifest))
    shard = carried_evidence / "oam.json.gz"
    document = json.loads(gzip.decompress(shard.read_bytes()))
    document["dataset"] = "beam10m"
    shard.write_bytes(gzip.compress(json.dumps(document).encode()))
    with pytest.raises(ValueError, match="invalid carried No"):
        render_historical_report(carried_evidence)


def test_legacy_retrieval_summary_still_uses_saved_selected_timing(saved_evidence):
    from oamb.reporting.historical import render_historical_report

    summary = _payload(render_historical_report(saved_evidence))["arms"]["oam"]["summary"]
    document = json.loads(gzip.decompress((saved_evidence / "oam.json.gz").read_bytes()))
    assert summary["retrieval_median_ms"] == document["results"][0]["retrieve_time_ms"]
    assert "retrieval_sample_count" not in summary
