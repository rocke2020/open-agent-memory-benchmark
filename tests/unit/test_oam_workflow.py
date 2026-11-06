import json
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


def test_completion_resume_reuses_success_but_counts_failed_paid_attempt(tmp_path):
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
        cache = CompletionCache(
            tmp_path, http, "http://model/v1", "secret", max_calls=1, max_tokens=10000, timeout=1
        )
        assert cache.complete("model", "high", "prompt", "answer", 1024) == "yes"
        resumed = CompletionCache(
            tmp_path, http, "http://model/v1", "secret", max_calls=1, max_tokens=10000, timeout=1
        )
        assert resumed.complete("model", "high", "prompt", "answer", 1024) == "yes"
        with pytest.raises(ValueError, match="budget"):
            resumed.complete("model", "high", "different", "answer", 1024)
    assert len(requests) == 1
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


def test_resume_retries_invalid_completion_within_total_attempt_budget(tmp_path):
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
        cache = CompletionCache(
            tmp_path, http, "http://model/v1", "key", max_calls=2, max_tokens=5000, timeout=1
        )
        with pytest.raises(ValueError):
            cache.complete("model", "low", "prompt", "answer", 1024)
        assert cache.complete("model", "low", "prompt", "answer", 1024) == "yes"
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
        cache = CompletionCache(
            tmp_path, http, "http://model/v1", "key", max_calls=2, max_tokens=5000, timeout=1
        )
        with pytest.raises(ValueError):
            cache.complete("m", "low", "prompt", "judge", 1024, validator=validate)
        assert cache.complete("m", "low", "prompt", "judge", 1024, validator=validate) == "yes"


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


def test_indexing_budget_counts_usage_and_reserves_unknown_sources(tmp_path):
    from oamb.oam_workflow import admit_indexing_source

    owner = tmp_path / "owners/u"
    owner.mkdir(parents=True)
    (owner / "source-000000.json").write_text('{"operation_id":"op"}')
    with pytest.raises(ValueError, match="indexing budget"):
        admit_indexing_source(tmp_path, source_ceiling=1000, total_ceiling=1500)
    (owner / "ready-000000.json").write_text(
        json.dumps({"usage": {"coverage": "complete", "total_tokens": 200}})
    )
    admit_indexing_source(tmp_path, source_ceiling=1000, total_ceiling=1500)
    with pytest.raises(ValueError, match="indexing budget"):
        admit_indexing_source(tmp_path, source_ceiling=1000, total_ceiling=1100)


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
        "score": 1.0,
        "judgments": ["yes"],
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
