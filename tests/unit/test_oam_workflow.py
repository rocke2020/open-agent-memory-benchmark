import json
import subprocess
from pathlib import Path
from typing import Any

import httpx
import pytest

from oamb.oam_workflow import CompletionCache, parse_completion, validate_results


@pytest.mark.parametrize(
    "patch",
    [
        {"finish_reason": "length"},
        {"finish_reason": "stop", "message": {"content": ""}},
        {"finish_reason": "stop", "message": {"content": "ok", "refusal": "refused"}},
    ],
)
def test_completion_rejects_truncated_empty_and_refused(patch):
    response = {
        "choices": [{"finish_reason": "stop", "message": {"content": "ok"}, **patch}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    with pytest.raises(ValueError):
        parse_completion(response)


def test_completion_resume_reuses_success_and_allows_unfinished_questions(tmp_path):
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "yes"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        cache = CompletionCache(tmp_path, http, "http://model/v1", "secret", timeout=1)
        assert cache.complete("model", "high", "prompt", "answer") == "yes"
        resumed = CompletionCache(tmp_path, http, "http://model/v1", "secret", timeout=1)
        assert resumed.complete("model", "high", "prompt", "answer") == "yes"
        assert resumed.complete("model", "high", "different", "answer") == "yes"
    assert len(requests) == 2
    assert all("max_tokens" not in request for request in requests)
    assert "secret" not in "".join(p.read_text() for p in tmp_path.rglob("*.json"))


def test_validator_rejects_incomplete_or_tampered_context(tmp_path):
    plan = {"question_ids": ["q"], "diagnostic": True}
    with pytest.raises(ValueError, match="coverage"):
        validate_results(tmp_path, plan)
    (tmp_path / "results").mkdir()
    (tmp_path / "results/q.json").write_text(
        json.dumps(
            {
                "question_id": "q",
                "context": "wrong",
                "context_sha256": "0" * 64,
                "answer": "ok",
                "score": 1.0,
            }
        )
    )
    with pytest.raises(ValueError):
        validate_results(tmp_path, plan)


def test_resume_retries_invalid_completion_and_preserves_failed_attempt(tmp_path):
    responses = iter(
        [
            {
                "choices": [{"finish_reason": "length", "message": {"content": "partial"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
            {
                "choices": [{"finish_reason": "stop", "message": {"content": "yes"}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ]
    )
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=next(responses)))
    ) as http:
        cache = CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        with pytest.raises(ValueError):
            cache.complete("model", "low", "prompt", "answer")
        assert cache.complete("model", "low", "prompt", "answer") == "yes"
    assert len(list((tmp_path / "model-calls").glob("*.json"))) == 2


def test_resume_does_not_reuse_semantically_invalid_judgment(tmp_path):
    contents = iter(["maybe", "yes"])

    def handle(request):
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": next(contents)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    def validate(text):
        if text not in {"yes", "no"}:
            raise ValueError("invalid judgment")

    with httpx.Client(transport=httpx.MockTransport(handle)) as http:
        cache = CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        with pytest.raises(ValueError):
            cache.complete("m", "low", "prompt", "judge", validator=validate)
        assert cache.complete("m", "low", "prompt", "judge", validator=validate) == "yes"


def test_public_scripts_route_explicit_oam_profile_without_starting_comparison(tmp_path):
    import os
    import subprocess

    root = Path(__file__).resolve().parents[2]
    fake_uv = tmp_path / "uv"
    fake_uv.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    fake_uv.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    for script, args, mode in [
        ("precheck.sh", ["--oam", "config.yml", "--output", "run"], "precheck"),
        ("run.sh", ["--oam", "run", "--resume"], "run"),
    ]:
        result = subprocess.run(
            [str(root / script), *args], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "oamb.oam_workflow" in result.stdout
        assert mode in result.stdout


def test_full_beam_rejects_200_questions_on_one_owner():
    from oamb.oam_workflow import validate_coverage
    from oamb.workloads.oam_inputs import History, Question

    histories = [
        History(
            "one",
            [],
            tuple(Question(str(i), "Q?", "information_extraction", "a") for i in range(200)),
        )
    ]
    with pytest.raises(ValueError, match="ten complete"):
        validate_coverage("beam10m", histories, diagnostic=False)


@pytest.mark.parametrize(
    "model,dimension,rejected", [("different-alias", 1024, False), ("same", 512, True)]
)
def test_embedding_resume_gate_checks_dimension_only(model, dimension, rejected):
    from oamb.oam_workflow import validate_embedding_resume

    previous = {"model": "same", "dimensions": 1024}
    current = {"model": model, "dimensions": dimension}
    if rejected:
        with pytest.raises(ValueError, match="dimension"):
            validate_embedding_resume(current, previous)
    else:
        validate_embedding_resume(current, previous)


@pytest.mark.parametrize("mutation", [None, "score", "context", "answer"])
def test_validation_binds_result_to_actual_model_input_and_judgment(tmp_path, mutation):
    import hashlib

    from oamb.workloads.longmemeval import render_lme_answer_prompt, render_lme_judge_prompt
    from oamb.workloads.oam_inputs import History, Question

    question = Question("q", "Drink?", "single-session-user", "tea")
    prompt = render_lme_answer_prompt(question="Drink?", evidence=b"tea").canonical_bytes.decode()
    judge = render_lme_judge_prompt(
        question_type="single-session-user",
        question="Drink?",
        reference="tea",
        model_response="tea",
        unanswerable=False,
    ).canonical_bytes.decode()
    (tmp_path / "model-calls").mkdir()
    for index, (stage, text, content) in enumerate(
        [("answer", prompt, "tea"), ("judge", judge, "yes")]
    ):
        (tmp_path / f"model-calls/{index}.json").write_text(
            json.dumps(
                {
                    "status": 200,
                    "stage": stage,
                    "request": {
                        "model": "m",
                        "reasoning_effort": "low",
                        "n": 1,
                        "temperature": 0,
                        "top_p": 1,
                        "stop": None,
                        "messages": [{"role": "user", "content": text}],
                    },
                    "response": {
                        "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    },
                }
            )
        )
    result: dict[str, Any] = {
        "question_id": "q",
        "question": "Drink?",
        "context": "tea",
        "answer": "tea",
        "answer_response": "tea",
        "score": 1.0,
        "judgments": ["yes"],
        "first_pass": {
            "answer": "tea",
            "answer_response": "tea",
            "score": 1.0,
            "judgments": ["yes"],
        },
        "context_sha256": hashlib.sha256(b"tea").hexdigest(),
        "recall": {
            "context": "tea",
            "diagnostics": {
                "generation_id": "g",
                "recall_level": "fast",
                "generative_calls": 0,
                "degraded": [],
                "partial": False,
                "readiness": "ready",
            },
        },
    }
    if mutation == "score":
        result["score"] = 0.0
    if mutation == "answer":
        result["answer"] = "coffee"
    if mutation == "context":
        result["context"] = "coffee"
        result["recall"]["context"] = "coffee"
        result["context_sha256"] = hashlib.sha256(b"coffee").hexdigest()
    (tmp_path / "results").mkdir()
    (tmp_path / "results/q.json").write_text(json.dumps(result))
    plan = {
        "question_ids": ["q"],
        "config": {"dataset": "lme500", "answer_effort": "low", "judge_effort": "low"},
        "bindings": {"answer": "m", "judge": "m"},
        "capabilities": {"index_generation_id": "g"},
    }
    histories = [History("q", [], (question,))]
    if mutation:
        with pytest.raises(ValueError):
            validate_results(tmp_path, plan, histories)
    else:
        assert validate_results(tmp_path, plan, histories)[0]["score"] == 1.0


@pytest.mark.parametrize(
    "scoring",
    [
        {},
        {"answer_model": "minimax-m3.1-flash", "answer_effort": "adaptive"},
        {"judge_model": "fixture-judge", "judge_effort": "none"},
    ],
)
def test_native_evaluation_and_resume_without_spending_or_sql_audits(
    tmp_path, monkeypatch, scoring
):
    from oamb import oam_workflow as workflow

    source = tmp_path / "input.json"
    source.write_bytes(Path("tests/fixtures/oam/lme500-smoke.json").read_bytes())
    config_path = tmp_path / "config.yml"
    config_path.write_text(f"dataset: lme500\nsource: {source}\ndiagnostic: true\n")
    with config_path.open("a") as stream:
        stream.write("".join(f"{key}: {value}\n" for key, value in scoring.items()))
    env = {
        "OAMB_OAM_BASE_URL": "http://oam",
        "LLM_BASE_URL": "http://model/v1",
        "LLM_API_KEY": "key",
        "OAMB_OAM_ROOT_KEY": "root-key",
        "LLM_DEEP_MODEL": "provider/user-selected-answer",
        "LLM_LIGHT_MODEL": "user-selected-extractor-and-judge",
        "OAMB_OAM_TENANT_ID": "benchmark",
        "OAMB_EMBEDDING_BASE_URL": "http://model/v1",
        "OAMB_EMBEDDING_MODEL": "provider/user-selected-embedding",
    }
    runtime: dict[str, Any] = {
        "config_sha256": "cfg",
        "models": {"extraction": "runtime-extractor-name", "embedding": "runtime-embedding-name"},
        "config": {
            "database": {"vector_index_mode": "small"},
            "index": {
                "insights_enabled": False,
                "embedding_dimensions": 1024,
                "generation_token_budget": None,
            },
            "profile": {"enabled": False},
            "models": {"reasoning_effort": {"extraction": "low"}},
        },
    }
    monkeypatch.setattr(workflow, "_environment", lambda *args: env)
    monkeypatch.setattr(workflow, "_runtime_config", lambda *args: runtime)
    responses = {
        "/health/ready": {"ready": True},
        "/v1/capabilities": {
            "contract_version": "0.0.1",
            "index_generation_id": "g",
            "config_sha256": "cfg",
            "limits": {"ingest": {"max_messages": 100, "max_bytes": 10000}},
        },
        "/v1/embeddings": {"data": [{"embedding": [0.0] * 1024}]},
    }
    calls = []

    def respond(request):
        path = request.url.path
        if path in responses:
            return httpx.Response(200, json=responses[path])
        calls.append(path)
        body = json.loads(request.content) if request.content else {}
        if path == "/v1/users":
            value = {"user_id": body["user_id"], "secret": "user-key"}
        elif path == "/v1/retain":
            value = {"operation_id": f"op-{len(calls)}", "source_id": f"s-{len(calls)}"}
        elif "/operations/" in path:
            value = {
                "operation_id": path.rsplit("/", 1)[-1],
                "status": "succeeded",
                "readiness": {
                    k: True
                    for k in (
                        "source_searchable",
                        "memory_ready",
                        "links_ready",
                        "temporal_links_ready",
                    )
                },
                "stage_state": {"extraction_coverage": "complete"},
            }
        elif path == "/v1/recall":
            value = {
                "context": "I prefer green tea.",
                "diagnostics": {
                    "generation_id": "g",
                    "recall_level": "fast",
                    "generative_calls": 0,
                    "degraded": [],
                    "partial": False,
                    "readiness": "ready",
                },
            }
        elif path == "/v1/chat/completions":
            content = "green tea" if calls.count(path) == 1 else "yes"
            value = {
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            }
        else:
            raise AssertionError(f"unexpected request: {path}")
        return httpx.Response(200, json=value)

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(respond))
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("evaluation must not inspect database query plans"),
    )
    output = tmp_path / "run"
    workflow.precheck(config_path, output, tmp_path / "unused.env")
    plan = json.loads((output / "plan.json").read_text())
    assert plan["question_ids"] == ["oam-smoke-lme"]
    assert plan["bindings"]["answer"] == scoring.get("answer_model", env["LLM_DEEP_MODEL"])
    assert plan["bindings"]["judge"] == scoring.get("judge_model", env["LLM_LIGHT_MODEL"])
    assert plan["runtime"]["config"]["index"]["generation_token_budget"] is None
    env["LLM_DEEP_MODEL"] = "minimax-operator-answer"
    env["LLM_LIGHT_MODEL"] = "minimax-operator-judge"
    workflow.run(output, resume=False)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["first_pass_mean_score"] == summary["policy_final_mean_score"] == 1.0
    assert "green tea" in (output / "report.html").read_text()
    assert calls.count("/v1/retain") == 2
    assert calls.count("/v1/chat/completions") == 2
    before = list(calls)
    workflow.run(output, resume=True)
    assert calls == before
    env["LLM_DEEP_MODEL"] = "operator-chosen-answer"
    env["LLM_LIGHT_MODEL"] = "operator-chosen-extraction-and-judge"
    env["LLM_BASE_URL"] = "http://another-provider/v1"
    env["LLM_API_KEY"] = "rotated-key"
    runtime["models"]["extraction"] = "operator-chosen-extractor"
    workflow.run(output, resume=True)
    assert calls == before
    plan["config"]["judge_model"] = "changed-judge"
    (output / "plan.json").write_text(json.dumps(plan))
    workflow.run(output, resume=True)
    assert calls == before
    plan["config"]["judge_model"] = scoring.get("judge_model")
    (output / "plan.json").write_text(json.dumps(plan))
    source.write_text("[]")
    with pytest.raises(ValueError, match="checksum"):
        workflow.run(output, resume=True)
    assert calls == before


def test_resume_prompt_identity_ignores_implementation_edits(tmp_path, monkeypatch):
    from oamb import oam_workflow as workflow

    code = tmp_path / "oam_workflow.py"
    code.write_text("first implementation")
    prompts = tmp_path / "workloads/prompt_templates"
    prompts.mkdir(parents=True)
    prompt = prompts / "answer.txt"
    prompt.write_text("published prompt")
    for name in ("workloads/oam_protocol.py", "workloads/oam_inputs.py", "memory_systems/oam.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("implementation")
    monkeypatch.setattr(workflow, "__file__", str(code))
    original = workflow._protocol_hash()
    code.write_text("harmless implementation fix")
    assert workflow._protocol_hash() == original
    prompt.write_text("different prompt")
    assert workflow._protocol_hash() != original
