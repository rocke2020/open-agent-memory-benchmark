"""Explicit native OAM workflow behind the public precheck/run scripts."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
import tiktoken
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from oamb.artifacts.atomic import atomic_replace_bytes, atomic_write_bytes
from oamb.contracts.ids import canonical_sha256
from oamb.memory_systems.oam import OAMAdapter, pack_messages
from oamb.runtime.provider_env import load_t10_provider_environment
from oamb.workloads.longmemeval import render_lme_answer_prompt, render_lme_judge_prompt
from oamb.workloads.model_parameters import ReasoningEffort
from oamb.workloads.oam_inputs import BEAM_CATEGORIES, History, Question, load_histories
from oamb.workloads.oam_protocol import (
    ANSWER_SCHEMA,
    JUDGE_SCHEMA,
    answer_text,
    completion_payload,
    validate_retry_review,
)

_CONTAINER_CONFIG = "/opt/oam-eval/config.yaml"


class OAMRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset: Literal["lme500", "beam10m"]
    source: str
    question_ids: list[str] = Field(default_factory=list)
    diagnostic: bool = False
    model: str | None = None
    embedding_model: str | None = None
    answer_model: str | None = None
    judge_model: str | None = None
    answer_effort: ReasoningEffort = "high"
    judge_effort: ReasoningEffort = "high"
    embedding_dimensions: int = Field(default=1024, ge=1)
    timeout_seconds: int = Field(default=900, ge=1)


def scoring_models(config: OAMRunConfig, env: dict[str, str]) -> dict[str, str]:
    selected = {}
    for role, variable in (("answer", "LLM_DEEP_MODEL"), ("judge", "LLM_LIGHT_MODEL")):
        requested = getattr(config, role + "_model")
        selected[role] = (
            requested
            if requested is not None
            else config.model
            if config.model is not None
            else env[variable]
        )
    return selected


def _request_identity(payload: dict[str, Any]) -> str:
    return canonical_sha256({k: v for k, v in payload.items() if k != "model"})


def _write(path: Path, value: Any, *, replace: bool = False) -> None:
    writer = atomic_replace_bytes if replace else atomic_write_bytes
    writer(path, _json_bytes(value))


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _read(path: Path) -> Any:
    return json.loads(path.read_bytes())


def _record(root: Path, kind: str, value: Any) -> None:
    _write(root / "raw" / f"{time.time_ns()}-{kind}-{uuid4().hex[:8]}.json", value)


def parse_completion(body: dict[str, Any]) -> str:
    choice = body["choices"][0]
    message = choice["message"]
    content = message.get("content")
    if (
        choice.get("finish_reason") != "stop"
        or message.get("refusal")
        or not isinstance(content, str)
        or not content.strip()
        or message.get("tool_calls")
    ):
        raise ValueError("model output is truncated, empty, refused or malformed")
    usage = body["usage"]
    if any(
        type(usage.get(key)) is not int or usage[key] < 0
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        raise ValueError("model usage is missing or malformed")
    return content.strip()


class CompletionCache:
    """One dispatch per saved attempt; successful complete bodies are reusable."""

    def __init__(
        self,
        root: Path,
        http: httpx.Client,
        base_url: str,
        key: str,
        *,
        timeout: float,
    ) -> None:
        self.root, self.http, self.url, self.key = root, http, base_url.rstrip("/"), key
        self.timeout = timeout

    def complete(
        self,
        model: str,
        effort: str,
        prompt: str,
        stage: str,
        *,
        validator: Callable[[str], object] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> str:
        payload = completion_payload(model, effort, prompt, schema)
        request_identity = _request_identity(payload)
        identity = hashlib.sha256(_json_bytes({"payload": payload, "stage": stage})).hexdigest()
        directory = self.root / "model-calls"
        saved = sorted(directory.glob("*.json"))
        for path in saved:
            attempt = _read(path)
            same_request = (
                attempt["stage"] == stage
                and _request_identity(attempt["request"]) == request_identity
            )
            if same_request and attempt.get("state") == "dispatched" and "status" not in attempt:
                raise ValueError(
                    "unknown model call outcome; reconcile the retained receipt before resuming"
                )
            if same_request and attempt.get("status") == 200:
                try:
                    content = parse_completion(attempt["response"])
                    if validator:
                        validator(content)
                    return content
                except (KeyError, TypeError, ValueError, IndexError):
                    pass
        path = directory / f"{len(saved):08d}-{identity}.json"
        attempt = {
            "identity": identity,
            "stage": stage,
            "request": payload,
            "state": "dispatched",
        }
        _write(path, attempt)
        started = time.monotonic()
        response = self.http.post(
            self.url + "/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.key}"},
            timeout=self.timeout,
        )
        attempt.update(status=response.status_code, elapsed_seconds=time.monotonic() - started)
        try:
            attempt["response"] = response.json()
        except ValueError:
            attempt["state"] = "invalid_json"
            _write(path, attempt, replace=True)
            raise ValueError("model returned non-JSON response") from None
        _write(path, attempt, replace=True)
        if not response.is_success:
            raise ValueError(f"model returned HTTP {response.status_code}; attempt preserved")
        content = parse_completion(attempt["response"])
        if validator:
            validator(content)
        return content


_ENV_KEYS = frozenset(
    {
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_LIGHT_MODEL",
        "LLM_DEEP_MODEL",
        "OAMB_EMBEDDING_BASE_URL",
        "OAMB_EMBEDDING_API_KEY",
        "OAMB_EMBEDDING_MODEL",
        "OAMB_OAM_BASE_URL",
        "OAMB_OAM_ROOT_KEY",
        "OAMB_OAM_CONTAINER",
        "OAMB_OAM_TENANT_ID",
    }
)


def _environment(path: Path) -> dict[str, str]:
    env = {**load_t10_provider_environment(path, expected_keys=_ENV_KEYS), **os.environ}
    required = _ENV_KEYS - {"OAMB_EMBEDDING_API_KEY"}
    if any(not env.get(k) or env[k].startswith("change-me") for k in required):
        raise ValueError("configure the OAM, embedding and model environment variables")
    return env


def _operator(env: dict[str, str], args: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        ["docker", "exec", env["OAMB_OAM_CONTAINER"], *args],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if result.returncode:
        raise ValueError("OAM operator command failed; inspect the isolated container logs")
    value: dict[str, Any] = json.loads(result.stdout)
    return value


def _runtime_config(env: dict[str, str]) -> dict[str, Any]:
    code = (
        "import json,os;from oam.config import load_config;"
        "from oam.models.bindings import resolve_model_binding;"
        f"r=load_config({_CONTAINER_CONFIG!r});"
        "b={k:resolve_model_binding(r.config.models,k,os.environ).model for k in ('extraction','embedding')};"
        "print(json.dumps({'config_sha256':r.sha256,'config':r.config.model_dump(mode='json'),'models':b}))"
    )
    return _operator(env, ["python", "-c", code])


def _embedding_probe(
    http: httpx.Client, env: dict[str, str], config: OAMRunConfig
) -> dict[str, Any]:
    key = env.get("OAMB_EMBEDDING_API_KEY", "")
    model = (
        config.embedding_model
        if config.embedding_model is not None
        else env["OAMB_EMBEDDING_MODEL"]
    )
    response = http.post(
        env["OAMB_EMBEDDING_BASE_URL"].rstrip("/") + "/embeddings",
        headers={"Authorization": f"Bearer {key}"} if key else {},
        json={
            "model": model,
            "input": ["OAMB readiness"],
        },
    )
    response.raise_for_status()
    body = response.json()
    vector = body["data"][0]["embedding"]
    if len(vector) != config.embedding_dimensions or not all(
        type(v) in (float, int) and math.isfinite(v) for v in vector
    ):
        raise ValueError("embedding must be one finite vector with the configured dimension")
    return {"model": model, "dimensions": len(vector)}


def _protocol_hash() -> str:
    root = Path(__file__).parent
    paths = sorted((root / "workloads/prompt_templates").rglob("*.txt"))
    return canonical_sha256(
        ["oam-native-v1", *[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]]
    )


def validate_embedding_resume(current: dict[str, Any], previous: dict[str, Any]) -> None:
    if current["dimensions"] != previous["dimensions"]:
        raise ValueError("embedding dimension changed")


def validate_coverage(dataset: str, histories: list[History], *, diagnostic: bool) -> None:
    if diagnostic:
        return
    if dataset == "lme500":
        if len(histories) != 500 or any(len(h.questions) != 1 for h in histories):
            raise ValueError("full LME requires 500 complete histories")
    elif len(histories) != 10 or any(
        any(sum(q.category == category for q in h.questions) != 2 for category in BEAM_CATEGORIES)
        for h in histories
    ):
        raise ValueError(
            "full BEAM requires ten complete conversation owners and two questions per category"
        )


def precheck(config_path: Path, output: Path, env_file: Path) -> None:
    config = OAMRunConfig.model_validate(yaml.safe_load(config_path.read_text()))
    env = _environment(env_file)
    source = Path(config.source).resolve()
    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    histories = load_histories(config.dataset, source, source_sha256, config.question_ids)
    validate_coverage(config.dataset, histories, diagnostic=config.diagnostic)
    if output.exists():
        raise ValueError("precheck output already exists; use a fresh directory")
    with httpx.Client(timeout=config.timeout_seconds, trust_env=False) as http:
        adapter = OAMAdapter(http, env["OAMB_OAM_BASE_URL"], lambda *a: None)
        capabilities = adapter.preflight()
        runtime = _runtime_config(env)
        effective = runtime["config"]
        expected_mode = "small" if config.dataset == "lme500" else "large"
        if (
            runtime["config_sha256"] != capabilities["config_sha256"]
            or effective["database"]["vector_index_mode"] != expected_mode
            or effective["index"]["insights_enabled"]
            or effective["profile"]["enabled"]
            or effective["models"]["reasoning_effort"]["extraction"] != "low"
            or effective["models"]["reasoning_effort"].get("planner") is not None
            or effective["index"]["embedding_dimensions"] != config.embedding_dimensions
        ):
            raise ValueError("OAM runtime does not match the dataset evaluation profile")
        embedding = _embedding_probe(http, env, config)
    question_ids = [q.question_id for h in histories for q in h.questions]
    run_id = uuid4().hex
    plan = {
        "profile": "oam-native-v1",
        "run_id": run_id,
        "config": config.model_dump(),
        "source": str(source),
        "source_sha256": source_sha256,
        "env_file": str(env_file.resolve()),
        "diagnostic": config.diagnostic,
        "question_ids": question_ids,
        "protocol_sha256": _protocol_hash(),
        "capabilities": capabilities,
        "runtime": runtime,
        "embedding": embedding,
        "bindings": {
            "oam": env["OAMB_OAM_BASE_URL"],
            "llm": env["LLM_BASE_URL"],
            **scoring_models(config, env),
            "tenant": env["OAMB_OAM_TENANT_ID"],
        },
    }
    _write(output / "plan.json", plan)
    print(f"precheck: PASS ({len(question_ids)} questions; no generation calls)")


def _answer_prompt(question: Question, context: str, dataset: str) -> str:
    if dataset == "lme500":
        return render_lme_answer_prompt(
            question=question.text,
            evidence=context.encode(),
            question_timestamp=question.reference_time,
        ).canonical_bytes.decode()
    return (
        (Path(__file__).parent / "workloads/prompt_templates/beam/answer.txt")
        .read_text()
        .format(query=question.text, context=context)
    )


def _judge_prompts(question: Question, answer: str, dataset: str) -> list[str]:
    if dataset == "lme500":
        return [
            render_lme_judge_prompt(
                question_type=question.category,
                question=question.text,
                reference=question.reference,
                model_response=answer,
                unanswerable=question.question_id.endswith("_abs"),
            ).canonical_bytes.decode()
        ]
    template = (Path(__file__).parent / "workloads/prompt_templates/beam/rubric.txt").read_text()
    return [
        template.format(
            query=question.text,
            answer=answer,
            rubric_item=(
                "LLM response should mention: " + criterion
                if question.category == "event_ordering" and "should" not in criterion.lower()
                else criterion
            ),
        )
        for criterion in question.rubric
    ]


def _score_judgments(judgments: list[str], dataset: str) -> float:
    scores = []
    for judgment in judgments:
        if dataset == "lme500":
            if judgment.lower() not in {"yes", "no"}:
                raise ValueError("LME judge must return yes or no")
            scores.append(float(judgment.lower() == "yes"))
        else:
            parsed = json.loads(judgment)
            if (
                type(parsed.get("score")) not in (int, float)
                or not math.isfinite(parsed["score"])
                or not 0 <= parsed["score"] <= 1
                or not parsed.get("reason")
            ):
                raise ValueError("BEAM rubric judge returned an invalid score")
            raw = float(parsed["score"])
            scores.append(1.0 if raw >= 0.75 else 0.5 if raw >= 0.25 else 0.0)
    if not scores:
        raise ValueError("missing judgments")
    return sum(scores) / len(scores)


def _review_for(root: Path, question_id: str, context: str) -> dict[str, Any] | None:
    path = root / "retry-eligibility.json"
    if not path.exists():
        return None
    reviews = _read(path)
    if not isinstance(reviews, dict):
        raise ValueError("retry eligibility must be a question-to-review mapping")
    review = reviews.get(question_id)
    if review is None:
        return None
    if not isinstance(review, dict):
        raise ValueError("invalid retry eligibility review")
    # A generated draft stays pending until the operator completes the review.
    if review.get("gold_status") == "unreviewed":
        if review.get("eligible") is True:
            raise ValueError("unreviewed gold cannot authorize a retry")
        return None
    validate_retry_review(review, context)
    return dict(review)


def evaluate_question(
    root: Path,
    config: OAMRunConfig,
    env: dict[str, str],
    cache: CompletionCache,
    question: Question,
    context: str,
    *,
    saved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One held-context first pass and at most one explicitly reviewed policy retry."""
    path = root / "results" / (canonical_sha256(question.question_id) + ".json")
    schema = ANSWER_SCHEMA if config.dataset == "beam10m" else None
    judge_schema = JUDGE_SCHEMA if config.dataset == "beam10m" else None
    prompt = _answer_prompt(question, context, config.dataset)
    models = scoring_models(config, env)

    def attempt(effort: str) -> dict[str, Any]:
        content = cache.complete(
            models["answer"],
            effort,
            prompt,
            "answer",
            schema=schema,
            validator=lambda text: answer_text(text, config.dataset),
        )
        answer = answer_text(content, config.dataset)
        judgments = [
            cache.complete(
                models["judge"],
                config.judge_effort,
                text,
                "judge",
                schema=judge_schema,
                validator=lambda text: _score_judgments([text], config.dataset),
            )
            for text in _judge_prompts(question, answer, config.dataset)
        ]
        return {
            "answer": answer,
            "answer_response": content,
            "judgments": judgments,
            "score": _score_judgments(judgments, config.dataset),
        }

    if saved is None:
        first = attempt(config.answer_effort)
        saved = {
            "question_id": question.question_id,
            "question": question.text,
            "context": context,
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
            "first_pass": first,
            **first,
        }
        _write(path, saved)
    if (
        saved["context"] != context
        or saved["context_sha256"] != hashlib.sha256(context.encode()).hexdigest()
    ):
        raise ValueError("retry context differs from the saved first pass")
    if saved.get("retry") is not None:
        return saved
    threshold = 1.0 if config.dataset == "lme500" else 0.5
    if saved["first_pass"]["score"] >= threshold:
        return saved
    review_path = root / "retry-reviews" / (canonical_sha256(question.question_id) + ".json")
    if review_path.exists():
        review = _read(review_path)
        validate_retry_review(review, context)
        if review["eligible"] is not True:
            raise ValueError("accepted retry review must remain eligible")
    else:
        review = _review_for(root, question.question_id, context)
    if review is None:
        return saved
    if not review["eligible"]:
        result = {**saved, "retry_review": review}
    else:
        # Admission owns an in-flight retry even if the editable draft later changes or disappears.
        if not review_path.exists():
            _write(review_path, review)
        retry = attempt("max")
        result = {**saved, **retry, "retry": retry, "retry_review": review}
    _write(path, result, replace=True)
    return result


def validate_results(
    root: Path, plan: dict[str, Any], histories: list[History] | None = None
) -> list[dict[str, Any]]:
    results = [_read(path) for path in sorted((root / "results").glob("*.json"))]
    if sorted(r["question_id"] for r in results) != sorted(plan["question_ids"]):
        raise ValueError("result coverage differs from the frozen selection")
    for result in results:
        if (
            not result["answer"].strip()
            or hashlib.sha256(result["context"].encode()).hexdigest() != result["context_sha256"]
            or not 0 <= result["score"] <= 1
            or not result.get("judgments")
            or not result.get("recall", {}).get("diagnostics")
        ):
            raise ValueError("saved result failed content/score/evidence validation")
    if histories is None:
        config = plan["config"]
        histories = load_histories(
            config["dataset"], Path(plan["source"]), plan["source_sha256"], config["question_ids"]
        )
    questions = {q.question_id: q for h in histories for q in h.questions}
    calls: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in (root / "model-calls").glob("*.json"):
        call = _read(path)
        if call.get("status") != 200:
            continue
        try:
            content = parse_completion(call["response"])
            request = call["request"]
            calls.setdefault((call["stage"], content), []).append(request)
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    dataset = plan["config"]["dataset"]
    for result in results:
        question = questions[result["question_id"]]
        diagnostics = result["recall"]["diagnostics"]
        if (
            result["question"] != question.text
            or result["recall"]["context"] != result["context"]
            or diagnostics.get("generative_calls") != 0
            or diagnostics.get("degraded") != []
            or diagnostics.get("partial") is not False
            or diagnostics.get("readiness") != "ready"
            or diagnostics.get("generation_id") != plan["capabilities"]["index_generation_id"]
            or diagnostics.get("recall_level") != ("fast" if dataset == "lme500" else "normal")
        ):
            raise ValueError("saved recall identity/readiness differs from frozen input")
        first = result["first_pass"]
        attempts = [(first, plan["config"]["answer_effort"])]
        retry = result.get("retry")
        if retry is not None:
            review = result["retry_review"]
            validate_retry_review(review, result["context"])
            if not review["eligible"] or first["score"] >= (1 if dataset == "lme500" else 0.5):
                raise ValueError("saved retry is not an eligible failed first pass")
            admitted = root / "retry-reviews" / (canonical_sha256(question.question_id) + ".json")
            if not admitted.exists() or _read(admitted) != review:
                raise ValueError("saved retry differs from its accepted review")
            attempts.append((retry, "max"))
        elif result.get("retry_review") is not None:
            validate_retry_review(result["retry_review"], result["context"])
            if result["retry_review"]["eligible"]:
                raise ValueError("eligible review has no completed retry")
        final = retry if retry is not None else first
        if any(
            result[name] != final[name]
            for name in ("answer", "answer_response", "judgments", "score")
        ):
            raise ValueError("policy result differs from the last completed attempt")
        for attempt, effort in attempts:
            if answer_text(attempt["answer_response"], dataset) != attempt["answer"]:
                raise ValueError("answer differs from the structured model response")
            judge_prompts = _judge_prompts(question, attempt["answer"], dataset)
            if len(judge_prompts) != len(attempt["judgments"]):
                raise ValueError("rubric judgment coverage is incomplete")
            expected = [
                (
                    "answer",
                    _answer_prompt(question, result["context"], dataset),
                    attempt["answer_response"],
                    effort,
                )
            ]
            expected.extend(
                ("judge", p, j, plan["config"]["judge_effort"])
                for p, j in zip(judge_prompts, attempt["judgments"], strict=True)
            )
            for stage, prompt, content, stage_effort in expected:
                schema = (
                    (ANSWER_SCHEMA if stage == "answer" else JUDGE_SCHEMA)
                    if dataset == "beam10m"
                    else None
                )
                if not any(
                    _request_identity(request)
                    == _request_identity(
                        completion_payload(request["model"], stage_effort, prompt, schema)
                    )
                    for request in calls.get((stage, content), [])
                ):
                    raise ValueError(
                        "saved result differs from the actual complete model input/output"
                    )
            if not math.isclose(
                _score_judgments(attempt["judgments"], dataset), attempt["score"], abs_tol=1e-12
            ):
                raise ValueError("saved score differs from the actual judgments")
    return results


def summarize_results(results: list[dict[str, Any]], config: OAMRunConfig) -> dict[str, Any]:
    threshold = 1 if config.dataset == "lme500" else 0.5
    pending = [
        r["question_id"]
        for r in results
        if r["first_pass"]["score"] < threshold and not r.get("retry_review")
    ]
    mean = sum(r["score"] for r in results) / len(results)
    return {
        "validated_questions": len(results),
        "diagnostic": config.diagnostic,
        "first_pass_mean_score": sum(r["first_pass"]["score"] for r in results) / len(results),
        "mean_score": mean,
        "policy_final_mean_score": None if pending else mean,
        "policy_final_complete": not pending,
        "pending_review": pending,
        "first_pass_threshold_passes": sum(r["first_pass"]["score"] >= threshold for r in results),
        "policy_threshold_passes": sum(r["score"] >= threshold for r in results),
        "max_effort_retries": sum(r.get("retry") is not None for r in results),
    }


def _report(root: Path, plan: dict[str, Any], results: list[dict[str, Any]]) -> None:
    summary = summarize_results(results, OAMRunConfig.model_validate(plan["config"]))
    rows = "".join(
        f"<article><h2>{html.escape(r['question_id'])}</h2>"
        f"<p>First pass: {r['first_pass']['score']}; current policy: {r['score']}</p><p>{html.escape(r['question'])}</p>"
        f"<p>{html.escape(r['answer'])}</p><details><summary>Exact retrieved context</summary>"
        f"<pre>{html.escape(r['context'])}</pre></details></article>"
        for r in results
    )
    title = "OAM diagnostic E2E" if plan["diagnostic"] else "OAM evaluation"
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="color-scheme" content="light dark"><title>{title}</title><style>
:root{{color-scheme:light dark;--bg:#fff;--fg:#18202a;--line:#ccc}}
@media(prefers-color-scheme:dark){{:root{{--bg:#14171b;--fg:#eee;--line:#555}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.6 system-ui;max-width:960px;margin:2rem auto;padding:1rem}}
article{{border-top:1px solid var(--line);padding:1rem 0}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
</style><h1>{title}</h1><p>{len(results)} unique questions. First-pass mean:
{summary["first_pass_mean_score"]:.3f}; policy-final mean: {summary["policy_final_mean_score"] if summary["policy_final_complete"] else "pending source/context/gold review"}.</p><p>Generation-only usage is in model-calls and
owners; embedding tokens are excluded. This run is separate from historical evidence.</p>{rows}</html>"""
    atomic_replace_bytes(root / "report.html", document.encode())


def run(root: Path, *, resume: bool) -> None:
    plan = _read(root / "plan.json")
    config = OAMRunConfig.model_validate(plan["config"])
    env = _environment(Path(plan["env_file"]))
    if plan["protocol_sha256"] != _protocol_hash():
        raise ValueError("evaluation prompts changed; use a fresh precheck scope")
    if (
        env["OAMB_OAM_BASE_URL"] != plan["bindings"]["oam"]
        or env["OAMB_OAM_TENANT_ID"] != plan["bindings"]["tenant"]
    ):
        raise ValueError("OAM scope changed")
    histories = load_histories(
        config.dataset, Path(plan["source"]), plan["source_sha256"], config.question_ids
    )
    if (root / "started.json").exists() and not resume:
        raise ValueError("run already started; use --resume")
    with httpx.Client(timeout=config.timeout_seconds, trust_env=False) as http:
        adapter = OAMAdapter(
            http,
            env["OAMB_OAM_BASE_URL"],
            lambda k, v: _record(root, k, v),
            timeout=config.timeout_seconds,
        )
        capabilities = adapter.preflight()
        if any(
            capabilities[k] != plan["capabilities"][k]
            for k in ("config_sha256", "index_generation_id")
        ):
            raise ValueError("OAM runtime changed since precheck")
        validate_embedding_resume(_embedding_probe(http, env, config), plan["embedding"])
        if not (root / "started.json").exists():
            _write(root / "started.json", {"started_at": time.time()})
        cache = CompletionCache(
            root,
            http,
            env["LLM_BASE_URL"],
            env["LLM_API_KEY"],
            timeout=config.timeout_seconds,
        )
        for history in histories:
            _run_history(root, plan, config, env, adapter, cache, history)
    results = validate_results(root, plan, histories)
    _report(root, plan, results)
    _write(
        root / "summary.json",
        summarize_results(results, config),
        replace=True,
    )
    review_path = root / "retry-eligibility.json"
    threshold = 1 if config.dataset == "lme500" else 0.5
    if not review_path.exists():
        _write(
            review_path,
            {
                r["question_id"]: {
                    "eligible": False,
                    "coverage_review_complete": False,
                    "gold_review_complete": False,
                    "gold_status": "unreviewed",
                    "rationale": "",
                    "context_sha256": r["context_sha256"],
                }
                for r in results
                if r["first_pass"]["score"] < threshold
            },
        )
    pending = summarize_results(results, config)["pending_review"]
    status = "FIRST PASS COMPLETE; REVIEW PENDING" if pending else "PASS"
    print(f"run: {status} ({len(results)} validated questions); report: {root / 'report.html'}")


def _run_history(
    root: Path,
    plan: dict[str, Any],
    config: OAMRunConfig,
    env: dict[str, str],
    adapter: OAMAdapter,
    cache: CompletionCache,
    history: History,
) -> None:
    owner = "oamb-" + plan["run_id"][:16] + "-" + canonical_sha256(history.history_id)[:16]
    directory = root / "owners" / owner
    key_path = directory / "user.key"
    if key_path.exists():
        key = key_path.read_text()
    else:
        key = adapter.create_owner(owner, env["OAMB_OAM_ROOT_KEY"])
        atomic_write_bytes(key_path, key.encode())
    limits = plan["capabilities"]["limits"]["ingest"]
    batches = [
        batch
        for source in history.sources
        for batch in pack_messages(
            source, max_messages=limits["max_messages"], max_bytes=limits["max_bytes"]
        )
    ]
    _write(
        directory / "input.json",
        {
            "history_id": history.history_id,
            "batches": len(batches),
            "date_provenance": history.date_provenance,
        },
    )
    for index, messages in enumerate(batches):
        path = directory / f"source-{index:06d}.json"
        if path.exists():
            receipt = _read(path)
        else:
            receipt = {"state": "dispatching", "payload_sha256": canonical_sha256(messages)}
            _write(path, receipt)
        if "operation_id" not in receipt:
            receipt = adapter.retain(owner, key, messages, occurrence_id=f"{owner}:{index}")
            _write(path, receipt, replace=True)
        ready_path = directory / f"ready-{index:06d}.json"
        if not ready_path.exists():
            ready = adapter.wait_ready(owner, key, receipt["operation_id"])
            _write(ready_path, ready)
        ready = _read(ready_path)
        adapter.validate_ready(ready)
        if ready.get("operation_id") != receipt["operation_id"]:
            raise ValueError("saved operation does not match the accepted source")
        print(f"ingestion: {history.history_id} {index + 1}/{len(batches)} ready", flush=True)
    if config.dataset == "beam10m":
        report = _operator(
            env,
            [
                "oam",
                "vector-indexes",
                "--config",
                _CONTAINER_CONFIG,
                "--once",
                "--tenant-id",
                env["OAMB_OAM_TENANT_ID"],
                "--user-id",
                owner,
            ],
        )
        owners = report.get("owners", [])
        if (
            report.get("busy")
            or len(owners) != 1
            or owners[0].get("state") != "ready"
            or owners[0].get("tenant_id") != env["OAMB_OAM_TENANT_ID"]
            or owners[0].get("user_id") != owner
        ):
            raise ValueError("BEAM dedicated owner index is not ready")
        _write(directory / "vector-index.json", report, replace=True)
    for question in history.questions:
        result_path = root / "results" / (canonical_sha256(question.question_id) + ".json")
        saved = _read(result_path) if result_path.exists() else None
        capture_path = directory / (canonical_sha256(question.question_id) + "-recall.json")
        if capture_path.exists():
            recall = _read(capture_path)
        else:
            recall = adapter.recall(
                owner,
                key,
                question.text,
                "fast" if config.dataset == "lme500" else "normal",
                plan["capabilities"]["index_generation_id"],
                question.reference_time,
            )
            _write(capture_path, recall)
        context = recall["context"]
        result = evaluate_question(root, config, env, cache, question, context, saved=saved)
        _write(
            result_path,
            {
                **result,
                "owner": owner,
                "context_tokens_o200k": len(
                    tiktoken.get_encoding("o200k_base").encode_ordinary(context)
                ),
                "recall": recall,
            },
            replace=True,
        )
        print(
            f"question: {question.question_id} first={result['first_pass']['score']:.3f} policy={result['score']:.3f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("precheck")
    check.add_argument("config", type=Path)
    check.add_argument("--output", required=True, type=Path)
    check.add_argument("--env-file", type=Path, default=Path(".env"))
    execute = sub.add_parser("run")
    execute.add_argument("directory", type=Path)
    execute.add_argument("--resume", action="store_true")
    offline = execute.add_mutually_exclusive_group()
    offline.add_argument("--export-review", type=Path)
    offline.add_argument("--review-report", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "precheck":
            precheck(args.config, args.output, args.env_file)
        elif args.export_review or args.review_report:
            from oamb.workloads.oam_review import export_review, summarize_review

            if args.export_review:
                exported = export_review(args.directory, args.export_review)
                print(f"review: exported {len(exported['cases'])} cases to {args.export_review}")
            else:
                report = summarize_review(args.directory, args.review_report)
                destination = args.review_report.with_suffix(".report.json")
                _write(destination, report, replace=True)
                print(f"review: {report['status']}; report: {destination}")
        else:
            run(args.directory, resume=args.resume)
    except (ValueError, KeyError, OSError, httpx.HTTPError, subprocess.SubprocessError) as error:
        # Supplier bodies and credentials belong only in private receipts, never console errors.
        print(
            f"{args.command}: FAIL ({type(error).__name__}): {str(error) if isinstance(error, ValueError) else 'inspect retained evidence'}"
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
