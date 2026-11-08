"""Offline review stays bound to scored inputs and never calls a provider."""

import hashlib
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from oamb import oam_workflow as workflow
from oamb.contracts.ids import canonical_sha256
from oamb.workloads.oam_inputs import load_histories


@pytest.fixture(params=["lme500", "beam10m"])
def saved_run(tmp_path, request):
    dataset = request.param
    source = Path(f"tests/fixtures/oam/{dataset}-smoke.json").resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    histories = load_histories(dataset, source, digest, [])
    config = workflow.OAMRunConfig(dataset=dataset, source=str(source), diagnostic=True)
    plan = {
        "profile": "oam-native-v1",
        "run_id": "review-fixture",
        "config": config.model_dump(),
        "source": str(source),
        "source_sha256": digest,
        "question_ids": [q.question_id for h in histories for q in h.questions],
        "bindings": {"answer": "deepseek-flash", "judge": "deepseek-flash"},
        "capabilities": {"index_generation_id": "generation"},
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    env = {"LLM_DEEP_MODEL": "deepseek-flash", "LLM_LIGHT_MODEL": "deepseek-flash"}
    context = "[1]\nUseful evidence.\nUnrelated noise."
    outputs = iter(
        ["green tea", "yes"]
        if dataset == "lme500"
        else ['{"reasoning":"evidence","answer":"red"}', '{"score":1,"reason":"red"}']
    )

    def respond(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": next(outputs)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        for history in histories:
            for question in history.questions:
                result = workflow.evaluate_question(tmp_path, config, env, cache, question, context)
                result["recall"] = {
                    "context": context,
                    "diagnostics": {
                        "generation_id": "generation",
                        "recall_level": "fast" if dataset == "lme500" else "normal",
                        "generative_calls": 0,
                        "degraded": [],
                        "partial": False,
                        "readiness": "ready",
                    },
                }
                path = tmp_path / "results" / (canonical_sha256(question.question_id) + ".json")
                path.write_text(json.dumps(result))
    workflow.validate_results(tmp_path, plan, histories)
    return tmp_path


def review_module() -> ModuleType:
    assert importlib.util.find_spec("oamb.workloads.oam_review"), "native offline review is missing"
    return importlib.import_module("oamb.workloads.oam_review")


def test_review_round_trip_preserves_scored_artifacts_and_reports_unreviewed(
    saved_run, monkeypatch
):
    monkeypatch.setattr(
        httpx.Client, "send", lambda *a, **k: pytest.fail("offline review used HTTP")
    )
    module = review_module()
    worksheet = saved_run / "review.json"
    before = {p: p.read_bytes() for p in saved_run.rglob("*.json")}
    module.export_review(saved_run, worksheet)
    report = module.summarize_review(saved_run, worksheet)
    assert report["noise"]["unreviewed_cases"] == 1
    assert report["noise"]["noise_ratio"] is None
    review = json.loads(worksheet.read_text())
    case = review["cases"][0]
    view = json.loads(Path(case["source_view"]["path"]).read_text())
    assert sum(len(s["messages"]) for s in view["sessions"]) >= 2
    case["assessment"] = {
        "source_review_complete": True,
        "span_review_complete": True,
        "task_interpretation": "Hand-reviewed fixture context",
        "spans": [
            {"start": a, "end": b, "category": category, "reason": "Fixture meaning"}
            for a, b, category in [(0, 4, "format"), (4, 21, "direct"), (21, 37, "unrelated")]
        ],
    }
    worksheet.write_text(json.dumps(review))
    report = module.summarize_review(saved_run, worksheet)
    assert report["noise"]["reviewed_cases"] == 1
    assert report["noise"]["noise_ratio"] == 0.4
    assert all(p.read_bytes() == data for p, data in before.items())
    with pytest.raises(FileExistsError):
        module.export_review(saved_run, worksheet)


@pytest.mark.parametrize("mutation", ["context", "question", "sources", "duplicate", "tokenizer"])
def test_review_rejects_tampering_against_frozen_inputs(saved_run, mutation):
    module = review_module()
    worksheet = saved_run / "review.json"
    module.export_review(saved_run, worksheet)
    review = json.loads(worksheet.read_text())
    case = review["cases"][0]
    if mutation == "sources":
        Path(case["source_view"]["path"]).write_text('{"sessions":[]}')
    elif mutation == "duplicate":
        review["cases"].append(case)
    elif mutation == "tokenizer":
        case["tokenizer"]["name"] = "cl100k_base"
    else:
        case[mutation] = "tampered"
    worksheet.write_text(json.dumps(review))
    with pytest.raises(ValueError):
        module.summarize_review(saved_run, worksheet)


def test_public_offline_modes_need_no_credentials_or_services(saved_run, monkeypatch):
    monkeypatch.setattr(workflow, "_environment", lambda *a: pytest.fail("loaded credentials"))
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("offline mode used HTTP"))
    worksheet = saved_run / "cli-review.json"
    for flag in ("--export-review", "--review-report"):
        monkeypatch.setattr(
            sys, "argv", ["oam_workflow", "run", str(saved_run), flag, str(worksheet)]
        )
        workflow.main()
    report = json.loads(worksheet.with_suffix(".report.json").read_text())
    assert report["status"] == "incomplete"
    assert report["noise"]["unreviewed_cases"] == 1


@pytest.mark.parametrize("role", ["answer", "judge"])
def test_offline_validation_does_not_gate_on_configured_model_names(saved_run, role):
    plan = json.loads((saved_run / "plan.json").read_text())
    plan["config"][role + "_model"] = "different-unrequested-model"
    assert len(workflow.validate_results(saved_run, plan)) == 1


@pytest.mark.parametrize("recorded_model", ["original-reported-model", "minimax-original-model"])
def test_offline_validation_keeps_model_names_as_provenance_without_comparing_them(
    saved_run,
    recorded_model,
):
    plan = json.loads((saved_run / "plan.json").read_text())
    for role in ("answer", "judge"):
        plan["config"][role + "_model"] = "user-selected-model"
        plan["bindings"][role] = recorded_model
    assert len(workflow.validate_results(saved_run, plan)) == 1
