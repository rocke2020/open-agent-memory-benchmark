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


@pytest.mark.parametrize("resumed_model", ["m", "another-provider/m"])
def test_unknown_paid_outcome_never_redispatches_automatically(tmp_path, resumed_model):
    requests = []

    def timeout(request):
        requests.append(request)
        raise httpx.ReadTimeout("unknown remote outcome")

    with httpx.Client(transport=httpx.MockTransport(timeout)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        with pytest.raises(httpx.ReadTimeout):
            cache.complete("m", "high", "held prompt", "answer")
        with pytest.raises(ValueError, match="unknown model call outcome"):
            cache.complete(resumed_model, "high", "held prompt", "answer")
    assert len(requests) == 1


@pytest.mark.parametrize("tamper", ["system", "schema", "score"])
def test_validation_binds_structured_beam_request_and_score(tmp_path, tamper):
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
    if tamper == "score":
        result["score"] = 0
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


@pytest.mark.parametrize("vector", [[0.1, 0.2, 0.3], [0.1, 0.2], [float("nan"), 0.2, 0.3]])
def test_embedding_accepts_user_model_names_but_requires_a_finite_vector(vector):
    config = workflow.OAMRunConfig(
        dataset="lme500",
        source="unused",
        embedding_model="user/embedding-model",
        embedding_dimensions=3,
    )

    def respond(request):
        assert json.loads(request.content)["model"] == "user/embedding-model"
        # NaN cannot be serialized by HTTPX's JSON helper; model input is intentionally malformed.
        return httpx.Response(200, content=json.dumps({"data": [{"embedding": vector}]}))

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        if len(vector) != 3 or any(value != value for value in vector):
            with pytest.raises(ValueError, match="finite vector"):
                workflow._embedding_probe(
                    http,
                    {
                        "OAMB_EMBEDDING_BASE_URL": "http://embedding/v1",
                        "OAMB_EMBEDDING_MODEL": "different-environment-name",
                    },
                    config,
                )
        else:
            result = workflow._embedding_probe(
                http,
                {
                    "OAMB_EMBEDDING_BASE_URL": "http://embedding/v1",
                    "OAMB_EMBEDDING_MODEL": "different-environment-name",
                },
                config,
            )
            assert result == {"model": "user/embedding-model", "dimensions": 3}


@pytest.mark.parametrize("answer_effort", ["max", "adaptive"])
def test_accepted_answer_survives_judge_output_recovery(tmp_path, answer_effort):
    question = Question("q", "Color?", "single-session-user", "red")
    config = workflow.OAMRunConfig(dataset="lme500", source="unused", answer_effort=answer_effort)
    env = {"LLM_DEEP_MODEL": "answer", "LLM_LIGHT_MODEL": "judge"}
    outputs = iter(["blue", "invalid judgment", "no"])
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": next(outputs)}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as http:
        cache = workflow.CompletionCache(tmp_path, http, "http://model/v1", "key", timeout=1)
        with pytest.raises(ValueError, match="yes or no"):
            workflow.evaluate_question(tmp_path, config, env, cache, question, "red")
        result = workflow.evaluate_question(tmp_path, config, env, cache, question, "red")
        before = {p: p.read_bytes() for p in tmp_path.rglob("*.json")}
        assert (
            workflow.evaluate_question(tmp_path, config, env, cache, question, "red", saved=result)
            == result
        )
        assert all(p.read_bytes() == content for p, content in before.items())
    assert result["answer"] == "blue"
    assert result["score"] == 0
    assert len(requests) == 3
    assert requests[0].get("reasoning_effort") == ("max" if answer_effort == "max" else None)
    assert workflow.summarize_results([result], config)["mean_score"] == 0
