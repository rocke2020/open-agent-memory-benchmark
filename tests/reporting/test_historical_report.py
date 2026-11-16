from __future__ import annotations

import base64
import gzip
import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _payload(body: bytes) -> dict:
    match = re.search(
        rb'<script id="evidence" type="application/octet-stream">([^<]+)</script>', body
    )
    assert match is not None
    value = json.loads(gzip.decompress(base64.b64decode(match.group(1))))
    assert isinstance(value, dict)
    shared = {q["question_id"]: q["context"] for q in value["arms"]["oam"]["results"]}
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
