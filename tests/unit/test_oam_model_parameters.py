"""Model policy reaches scoring requests and persisted completion identities."""

import json

import httpx
import pytest

from oamb import oam_workflow as workflow
from oamb.workloads.oam_inputs import Question
from oamb.workloads.oam_protocol import ANSWER_SCHEMA, completion_payload


@pytest.mark.parametrize("schema", [None, ANSWER_SCHEMA])
@pytest.mark.parametrize("effort,thinking", [("none", "disabled"), ("adaptive", "adaptive")])
def test_native_modes_omit_effort_and_override_schema_thinking(schema, effort, thinking):
    payload = completion_payload("deepseek-flash", effort, "Exact prompt", schema)
    assert payload["thinking"] == {"type": thinking}
    assert "reasoning_effort" not in payload


@pytest.mark.parametrize("effort", ["none", "low", "high", "max", "adaptive"])
def test_minimax_separates_reasoning_from_visible_content(effort):
    payload = completion_payload("provider/MiniMax-M3.1-flash", effort, "Exact prompt")
    assert payload["reasoning_split"] is True
    assert payload["thinking"] == {"type": "disabled" if effort == "none" else "adaptive"}
    if effort in {"none", "adaptive"}:
        assert "reasoning_effort" not in payload
    else:
        assert payload["reasoning_effort"] == effort


def test_unknown_effort_fails_before_dispatch():
    with pytest.raises(ValueError, match="reasoning"):
        completion_payload("deepseek-flash", "typo", "prompt")


@pytest.mark.parametrize("effort", ["low", "high", "max"])
@pytest.mark.parametrize("schema", [None, ANSWER_SCHEMA])
def test_explicit_efforts_preserve_the_published_wire_shape(effort, schema):
    payload = completion_payload("deepseek-flash", effort, "Exact prompt", schema)
    assert payload["reasoning_effort"] == effort
    assert payload.get("thinking") == ({"type": "enabled"} if schema else None)


def test_model_selection_uses_explicit_values_without_inferring_aliases():
    env = {"LLM_DEEP_MODEL": "provider/environment-answer", "LLM_LIGHT_MODEL": "environment-judge"}
    config = workflow.OAMRunConfig(dataset="lme500", source="unused")
    assert workflow.scoring_models(config, env) == {
        "answer": "provider/environment-answer",
        "judge": "environment-judge",
    }
    config.model = "user/common-model"
    config.answer_model = "different-provider/opaque-model:v7"
    assert workflow.scoring_models(config, env) == {
        "answer": "different-provider/opaque-model:v7",
        "judge": "user/common-model",
    }
    config.answer_model = "deepseek-flash"
    env["LLM_DEEP_MODEL"] = "deepseek/deepseek-flash"
    assert workflow.scoring_models(config, env)["answer"] == "deepseek-flash"


@pytest.mark.parametrize("dataset", ["lme500", "beam10m"])
def test_role_overrides_and_reasoning_reach_answer_judge_and_cache(tmp_path, dataset):
    config = workflow.OAMRunConfig(
        dataset=dataset,
        source="unused",
        answer_model="minimax-m3.1-flash",
        judge_model="fixture-judge",
        answer_effort="adaptive",
        judge_effort="none",
    )
    env = {
        "LLM_DEEP_MODEL": "deepseek-flash",
        "LLM_LIGHT_MODEL": "deepseek-flash",
        "OAMB_EMBEDDING_MODEL": "qwen3-embedding:0.6b",
    }
    question = Question("q", "Color?", "single-session-user", "red", ("Mention red",))
    outputs = iter(
        ["red", "yes"]
        if dataset == "lme500"
        else ['{"reasoning":"evidence","answer":"red"}', '{"score":1,"reason":"red"}']
    )
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": next(outputs),
                            "reasoning_content": "private reasoning",
                        },
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        result = workflow.evaluate_question(tmp_path, config, env, cache, question, "red")
        assert result["answer"] == "red"
        assert result["score"] == 1
        workflow.evaluate_question(tmp_path, config, env, cache, question, "red", saved=result)
    assert len(requests) == 2
    assert requests[0]["model"] == "minimax-m3.1-flash"
    assert requests[0]["thinking"] == {"type": "adaptive"}
    assert "reasoning_effort" not in requests[0]
    assert requests[1]["model"] == "fixture-judge"
    assert requests[1]["thinking"] == {"type": "disabled"}
    assert "private reasoning" not in json.dumps(requests[1])


def test_reasoning_change_cannot_reuse_a_different_paid_request(tmp_path):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "red"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        for effort in ("adaptive", "adaptive", "high"):
            assert cache.complete("deepseek-flash", effort, "Exact prompt", "answer") == "red"
    assert len(requests) == 2
    assert requests[0].get("thinking") == {"type": "adaptive"}
    assert requests[1]["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "first,second",
    [
        ("deepseek-flash", "deepseek/deepseek-flash"),
        ("deepseek-v4.1-flash", "deepseek/deepseek-flash"),
        ("provider/model-v2", "another/model-v2"),
        ("entirely-different-model", "operator-chosen-model"),
    ],
)
def test_model_names_do_not_gate_completed_call_reuse_or_rewrite_evidence(tmp_path, first, second):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "red"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://first/v1", "key1", timeout=1)
        assert cache.complete(first, "high", "Exact prompt", "answer") == "red"
        before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
        cache = workflow.CompletionCache(tmp_path, http, "http://second/v1", "key2", timeout=1)
        assert cache.complete(second, "high", "Exact prompt", "answer") == "red"
    assert len(requests) == 1
    assert requests[0]["model"] == first
    assert all(p.read_bytes() == data for p, data in before.items())
