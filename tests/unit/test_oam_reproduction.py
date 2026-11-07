import hashlib
import json

import httpx
import pytest

from oamb import oam_workflow as workflow
from oamb.workloads.oam_inputs import Question


def test_keyless_embedding_precheck_omits_empty_bearer_header():
    config = workflow.OAMRunConfig(
        dataset="lme500",
        source="unused",
    )

    def respond(request):
        assert "Authorization" not in request.headers
        return httpx.Response(200, json={"data": [{"embedding": [0.0] * 1024}]})

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        result = workflow._embedding_probe(
            http,
            {
                "OAMB_EMBEDDING_BASE_URL": "http://embedding/v1",
                "OAMB_EMBEDDING_API_KEY": "",
                "OAMB_EMBEDDING_MODEL": "qwen3-embedding:0.6b",
            },
            config,
        )
    assert result["dimensions"] == 1024


def test_beam_answer_uses_the_published_memories_prompt():
    prompt = workflow._answer_prompt(
        Question("q", "Color?", "information_extraction", "red"), "red", "beam10m"
    )
    assert prompt.startswith("You are an AI assistant with access to stored memories")
    assert (
        hashlib.sha256(prompt.encode()).hexdigest()
        == "c0bfe31f1b0f2adfe1808dfff2148fb2ce8f4a73703fe7c42c1ff0fc75e9c9fe"
    )
    assert "QUESTION: Color?\n\nRETRIEVED MEMORIES:\nred\n\nANSWER:" in prompt


def test_beam_judges_each_nugget_using_published_compliance_rules():
    question = Question("q", "Color?", "information_extraction", "red", ("Mention red",))
    prompt = workflow._judge_prompts(question, "red", "beam10m")[0]
    assert "For NEGATIVE constraints" in prompt
    assert "SCORING GUIDELINES:" in prompt
    assert "RUBRIC CRITERION:\nMention red" in prompt
    assert (
        hashlib.sha256(prompt.encode()).hexdigest()
        == "00d01b0f164cdec4326c108bd277fed952502840a5d479f4e513619e21a5019b"
    )


def test_beam_answer_requires_reasoning_and_nonempty_answer_json():
    assert hasattr(workflow, "answer_text"), "published structured answer parser is missing"
    assert workflow.answer_text('{"reasoning":"evidence","answer":"red"}', "beam10m") == "red"
    for content in ("red", '{"answer":"red"}', '{"reasoning":"ok","answer":""}'):
        with pytest.raises(ValueError):
            workflow.answer_text(content, "beam10m")


def test_reviewed_retry_is_context_bound_and_blocks_incorrect_gold():
    assert hasattr(workflow, "validate_retry_review"), "published retry review gate is missing"
    review = {
        "eligible": True,
        "coverage_review_complete": True,
        "gold_review_complete": True,
        "gold_status": "correct",
        "rationale": "Relevant source evidence is in context.",
        "context_sha256": hashlib.sha256(b"red").hexdigest(),
    }
    workflow.validate_retry_review(review, "red")
    for patch in (
        {"context_sha256": "0" * 64},
        {"gold_status": "incorrect"},
        {"coverage_review_complete": False},
        {"rationale": ""},
    ):
        with pytest.raises(ValueError):
            workflow.validate_retry_review({**review, **patch}, "red")


def test_retry_uses_same_prompt_max_effort_and_reuses_completed_work(tmp_path):
    assert hasattr(workflow, "evaluate_question"), "policy retry integration is missing"
    requests = []
    contents = iter(["red", "no", "blue", "no"])

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": next(contents)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    question = Question("q", "Color?", "single-session-user", "red")
    config = workflow.OAMRunConfig(
        dataset="lme500",
        source="unused",
    )
    env = {"LLM_DEEP_MODEL": "deepseek-flash", "LLM_LIGHT_MODEL": "deepseek-flash"}
    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        first = workflow.evaluate_question(tmp_path, config, env, cache, question, "red")
        assert first["score"] == 0
        review = {
            "eligible": True,
            "coverage_review_complete": True,
            "gold_review_complete": True,
            "gold_status": "correct",
            "rationale": "Source and context contain the answer.",
            "context_sha256": hashlib.sha256(b"red").hexdigest(),
        }
        (tmp_path / "retry-eligibility.json").write_text(json.dumps({"q": review}))
        final = workflow.evaluate_question(
            tmp_path, config, env, cache, question, "red", saved=first
        )
        assert final["first_pass"] == first["first_pass"]
        assert final["answer"] == "blue"
        assert final["retry"]["score"] == 0
        assert requests[0]["messages"] == requests[2]["messages"]
        assert requests[0]["reasoning_effort"] == "high"
        assert requests[2]["reasoning_effort"] == "max"
        assert (
            workflow.evaluate_question(tmp_path, config, env, cache, question, "red", saved=final)
            == final
        )
    assert len(requests) == 4


def test_lower_beam_retry_becomes_policy_result_and_reports_both_passes(tmp_path):
    question = Question("q", "Color?", "information_extraction", "red", ("red", "specific"))
    config = workflow.OAMRunConfig(
        dataset="beam10m",
        source="unused",
    )
    env = {"LLM_DEEP_MODEL": "deepseek-flash", "LLM_LIGHT_MODEL": "deepseek-flash"}
    contents = iter(
        [
            '{"reasoning":"seen","answer":"red"}',
            '{"score":0.5,"reason":"partial"}',
            '{"score":0,"reason":"missing"}',
            '{"reasoning":"seen","answer":"blue"}',
            '{"score":0,"reason":"wrong"}',
            '{"score":0,"reason":"missing"}',
        ]
    )
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": next(contents)}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        )
    ) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        first = workflow.evaluate_question(tmp_path, config, env, cache, question, "red")
        assert hasattr(workflow, "summarize_results"), (
            "first-pass/policy-final reporting is missing"
        )
        pending = workflow.summarize_results([first], config)
        assert pending["first_pass_mean_score"] == 0.25
        assert pending["policy_final_mean_score"] is None
        assert pending["pending_review"] == ["q"]
        review = {
            "eligible": True,
            "coverage_review_complete": True,
            "gold_review_complete": True,
            "gold_status": "ambiguous",
            "rationale": "Reviewed both source and retained context.",
            "context_sha256": hashlib.sha256(b"red").hexdigest(),
        }
        (tmp_path / "retry-eligibility.json").write_text(json.dumps({"q": review}))
        final = workflow.evaluate_question(
            tmp_path, config, env, cache, question, "red", saved=first
        )
        assert final["score"] == 0
        assert final["first_pass"]["score"] == 0.25
        summary = workflow.summarize_results([final], config)
        assert summary["policy_final_mean_score"] == 0
        assert summary["pending_review"] == []


def test_unknown_paid_outcome_never_redispatches_automatically(tmp_path):
    requests = []

    def timeout(request):
        requests.append(request)
        raise httpx.ReadTimeout("unknown remote outcome")

    with httpx.Client(transport=httpx.MockTransport(timeout)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        with pytest.raises(httpx.ReadTimeout):
            cache.complete("m", "high", "held prompt", "answer")
        with pytest.raises(ValueError, match="unknown model call outcome"):
            cache.complete("m", "high", "held prompt", "answer")
    assert len(requests) == 1


@pytest.mark.parametrize("draft_change", [None, "ineligible", "unreviewed", "deleted"])
def test_interrupted_retry_reuses_max_answer_before_resuming_judgment(tmp_path, draft_change):
    question = Question("q", "Color?", "single-session-user", "red")
    config = workflow.OAMRunConfig(
        dataset="lme500",
        source="unused",
    )
    env = {"LLM_DEEP_MODEL": "deepseek-flash", "LLM_LIGHT_MODEL": "deepseek-flash"}
    outputs = iter([(200, "red"), (200, "no"), (200, "blue"), (503, ""), (200, "yes")])
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        status, content = next(outputs)
        return httpx.Response(
            status,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        first = workflow.evaluate_question(tmp_path, config, env, cache, question, "red")
        review = {
            "eligible": True,
            "coverage_review_complete": True,
            "gold_review_complete": True,
            "gold_status": "correct",
            "rationale": "Source and held context checked.",
            "context_sha256": hashlib.sha256(b"red").hexdigest(),
        }
        (tmp_path / "retry-eligibility.json").write_text(json.dumps({"q": review}))
        with pytest.raises(ValueError, match="HTTP 503"):
            workflow.evaluate_question(tmp_path, config, env, cache, question, "red", saved=first)
        draft = tmp_path / "retry-eligibility.json"
        if draft_change == "deleted":
            draft.unlink()
        elif draft_change:
            changed = {**review, "eligible": False}
            if draft_change == "unreviewed":
                changed["gold_status"] = "unreviewed"
            draft.write_text(json.dumps({"q": changed}))
        final = workflow.evaluate_question(
            tmp_path, config, env, cache, question, "red", saved=first
        )
    assert final["first_pass"] == first["first_pass"]
    assert final["score"] == 1
    assert final["retry_review"] == review
    assert sum(request["reasoning_effort"] == "max" for request in requests) == 1
    assert len(requests) == 5


@pytest.mark.parametrize("tamper", ["system", "schema", "first_pass"])
def test_validation_binds_structured_beam_request_and_first_pass(tmp_path, tamper):
    from oamb.workloads.oam_inputs import History

    config = workflow.OAMRunConfig(
        dataset="beam10m",
        source="unused",
    )
    env = {"LLM_DEEP_MODEL": "deepseek-flash", "LLM_LIGHT_MODEL": "deepseek-flash"}
    question = Question("q", "Color?", "information_extraction", "red", ("red",))
    contents = iter(['{"reasoning":"seen","answer":"red"}', '{"score":1,"reason":"correct"}'])
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "stop", "message": {"content": next(contents)}}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                },
            )
        )
    ) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        result = workflow.evaluate_question(tmp_path, config, env, cache, question, "red")
    result["recall"] = {
        "context": "red",
        "diagnostics": {
            "generation_id": "g",
            "recall_level": "normal",
            "generative_calls": 0,
            "degraded": [],
            "partial": False,
            "readiness": "ready",
        },
    }
    path = next((tmp_path / "results").glob("*.json"))
    path.write_text(json.dumps(result))
    plan = {
        "question_ids": ["q"],
        "config": config.model_dump(),
        "bindings": {"answer": "deepseek-flash", "judge": "deepseek-flash"},
        "capabilities": {"index_generation_id": "g"},
    }
    histories = [History("c", [], (question,))]
    workflow.validate_results(tmp_path, plan, histories)
    if tamper == "first_pass":
        result["first_pass"]["score"] = 0
        path.write_text(json.dumps(result))
    else:
        call_path = sorted((tmp_path / "model-calls").glob("*.json"))[0]
        call = json.loads(call_path.read_text())
        if tamper == "system":
            call["request"]["messages"][0]["content"] = "different instructions"
        else:
            call["request"]["response_format"] = {"type": "text"}
        call_path.write_text(json.dumps(call))
    with pytest.raises(ValueError):
        workflow.validate_results(tmp_path, plan, histories)


def test_reproduction_rejects_different_role_models():
    assert hasattr(workflow, "validate_models"), "same-model reproduction gate is missing"
    config = workflow.OAMRunConfig(
        dataset="lme500",
        source="unused",
    )
    env = {
        "LLM_DEEP_MODEL": "deepseek-flash",
        "LLM_LIGHT_MODEL": "another-model",
        "OAMB_EMBEDDING_MODEL": "qwen3-embedding:0.6b",
    }
    with pytest.raises(ValueError, match="model"):
        workflow.validate_models(config, env, "deepseek-flash", "qwen3-embedding:0.6b")


def test_embedding_alias_changes_do_not_expand_the_resume_gate():
    config = workflow.OAMRunConfig(
        dataset="lme500",
        source="unused",
    )
    env = {
        "LLM_DEEP_MODEL": "deepseek-flash",
        "LLM_LIGHT_MODEL": "deepseek-flash",
        "OAMB_EMBEDDING_MODEL": "new-request-alias",
    }
    workflow.validate_models(config, env, "deepseek-flash", "new-runtime-alias", resumed=True)
    workflow.validate_embedding_resume(
        {"model": "new-request-alias", "dimensions": 1024},
        {"model": "old-alias", "dimensions": 1024},
    )
    with pytest.raises(ValueError, match="model"):
        workflow.validate_models(config, env, "deepseek-flash", "new-runtime-alias")
