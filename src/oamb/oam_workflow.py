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
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx
import tiktoken
import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from oamb.artifacts.atomic import atomic_replace_bytes, atomic_write_bytes
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import BudgetScopeKind, BudgetSpec
from oamb.memory_systems.oam import OAMAdapter, pack_messages
from oamb.memory_systems.oam_index_plan import capture_index_plan
from oamb.runtime.provider_env import load_t10_provider_environment
from oamb.workloads.longmemeval import render_lme_answer_prompt, render_lme_judge_prompt
from oamb.workloads.oam_inputs import BEAM_CATEGORIES, History, Question, load_histories


class OAMRunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    dataset: Literal["lme500", "beam10m"]
    source: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    question_ids: list[str] = Field(default_factory=list)
    diagnostic: bool = False
    answer_model_env: str = "LLM_DEEP_MODEL"
    judge_model_env: str = "LLM_LIGHT_MODEL"
    answer_effort: Literal["low", "high", "max"] = "high"
    judge_effort: Literal["low", "high", "max"] = "high"
    embedding_model_env: str = "OAMB_EMBEDDING_MODEL"
    embedding_dimensions: int = Field(default=1024, ge=1)
    max_model_calls: int = Field(ge=1)
    max_model_tokens: int = Field(ge=1024)
    max_indexing_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(default=8192, ge=1024)
    timeout_seconds: int = Field(default=900, ge=1)
    tenant_id_env: str = "OAMB_OAM_TENANT_ID"
    container_config: str = "/opt/oam-eval/config.yaml"


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
        max_calls: int,
        max_tokens: int,
        timeout: float,
    ) -> None:
        self.root, self.http, self.url, self.key = root, http, base_url.rstrip("/"), key
        self.max_calls, self.max_tokens, self.timeout = max_calls, max_tokens, timeout

    def complete(
        self,
        model: str,
        effort: str,
        prompt: str,
        stage: str,
        max_output: int,
        *,
        validator: Callable[[str], object] | None = None,
    ) -> str:
        payload = {
            "model": model,
            "reasoning_effort": effort,
            "temperature": 0.0,
            "max_tokens": max_output,
            "messages": [{"role": "user", "content": prompt}],
        }
        identity = hashlib.sha256(_json_bytes({"payload": payload, "stage": stage})).hexdigest()
        directory = self.root / "model-calls"
        saved = sorted(directory.glob("*.json"))
        charged = 0
        for path in saved:
            attempt = _read(path)
            if attempt["identity"] == identity and attempt.get("status") == 200:
                try:
                    content = parse_completion(attempt["response"])
                    if validator:
                        validator(content)
                    return content
                except (KeyError, TypeError, ValueError, IndexError):
                    pass
            # Unknown and failed attempts retain their full reservation.
            body = attempt.get("response")
            usage = body.get("usage", {}) if isinstance(body, dict) else {}
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            charged += total if type(total) is int and total >= 0 else attempt["reserved_tokens"]
        # UTF-8 bytes bound prompt tokens without assuming a model's tokenizer.
        reservation = len(_json_bytes(payload)) + 2048 + max_output
        if len(saved) >= self.max_calls or charged + reservation > self.max_tokens:
            raise ValueError("model budget exhausted before dispatch")
        path = directory / f"{len(saved):08d}-{identity}.json"
        attempt = {
            "identity": identity,
            "stage": stage,
            "request": payload,
            "reserved_tokens": reservation,
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


def _environment(path: Path, config: OAMRunConfig) -> dict[str, str]:
    keys = _ENV_KEYS | {
        config.answer_model_env,
        config.judge_model_env,
        config.embedding_model_env,
        config.tenant_id_env,
    }
    env = {**load_t10_provider_environment(path, expected_keys=keys), **os.environ}
    required = keys - {
        "OAMB_EMBEDDING_API_KEY",
        "LLM_LIGHT_MODEL",
        "LLM_DEEP_MODEL",
        "OAMB_EMBEDDING_MODEL",
    }
    required |= {
        config.answer_model_env,
        config.judge_model_env,
        config.embedding_model_env,
        config.tenant_id_env,
    }
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


def _runtime_config(env: dict[str, str], config: OAMRunConfig) -> dict[str, Any]:
    code = (
        "import json;from oam.config import load_config;"
        f"r=load_config({config.container_config!r});"
        "print(json.dumps({'config_sha256':r.sha256,'config':r.config.model_dump(mode='json')}))"
    )
    return _operator(env, ["python", "-c", code])


def _embedding_probe(
    http: httpx.Client, env: dict[str, str], config: OAMRunConfig
) -> dict[str, Any]:
    response = http.post(
        env["OAMB_EMBEDDING_BASE_URL"].rstrip("/") + "/embeddings",
        headers={"Authorization": f"Bearer {env.get('OAMB_EMBEDDING_API_KEY', '')}"},
        json={"model": env[config.embedding_model_env], "input": ["OAMB readiness"]},
    )
    response.raise_for_status()
    body = response.json()
    vector = body["data"][0]["embedding"]
    if len(vector) != config.embedding_dimensions or not all(
        type(v) in (float, int) and math.isfinite(v) for v in vector
    ):
        raise ValueError("embedding must be one finite vector with the configured dimension")
    return {"model": env[config.embedding_model_env], "dimensions": len(vector)}


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


def admit_indexing_source(root: Path, *, source_ceiling: int, total_ceiling: int) -> None:
    charged = 0
    for receipt in (root / "owners").glob("*/source-*.json"):
        ready_path = receipt.with_name(receipt.name.replace("source-", "ready-"))
        usage = _read(ready_path).get("usage", {}) if ready_path.exists() else {}
        total = usage.get("total_tokens")
        charged += (
            total
            if usage.get("coverage") == "complete" and type(total) is int and total >= 0
            else source_ceiling
        )
    if charged + source_ceiling > total_ceiling:
        raise ValueError("indexing budget exhausted before source dispatch")


def precheck(config_path: Path, output: Path, env_file: Path) -> None:
    config = OAMRunConfig.model_validate(yaml.safe_load(config_path.read_text()))
    env = _environment(env_file, config)
    source = Path(config.source).resolve()
    histories = load_histories(config.dataset, source, config.source_sha256, config.question_ids)
    validate_coverage(config.dataset, histories, diagnostic=config.diagnostic)
    if output.exists():
        raise ValueError("precheck output already exists; use a fresh directory")
    with httpx.Client(timeout=config.timeout_seconds, trust_env=False) as http:
        adapter = OAMAdapter(http, env["OAMB_OAM_BASE_URL"], lambda *a: None)
        capabilities = adapter.preflight()
        runtime = _runtime_config(env, config)
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
            or type(effective["index"].get("generation_token_budget")) is not int
            or effective["index"]["generation_token_budget"] > config.max_indexing_tokens
        ):
            raise ValueError("OAM runtime does not match the dataset evaluation profile")
        embedding = _embedding_probe(http, env, config)
    question_ids = [q.question_id for h in histories for q in h.questions]
    run_id = uuid4().hex
    budget = BudgetSpec(
        budget_id="oam-answer-judge",
        scope_kind=BudgetScopeKind.RUN,
        scope_id=run_id,
        max_attempts=config.max_model_calls,
        max_input_tokens=config.max_model_tokens,
        max_output_tokens=config.max_model_tokens,
        max_wall_seconds=Decimal(config.timeout_seconds * config.max_model_calls),
        max_cost=None,
        currency=None,
    )
    plan = {
        "profile": "oam-native-v1",
        "run_id": run_id,
        "answer_judge_budget": budget.model_dump(mode="json"),
        "indexing_budget": {
            "max_generation_tokens": config.max_indexing_tokens,
            "per_source_ceiling": effective["index"]["generation_token_budget"],
        },
        "config": config.model_dump(),
        "source": str(source),
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
            "answer": env[config.answer_model_env],
            "judge": env[config.judge_model_env],
            "tenant": env[config.tenant_id_env],
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
                or parsed["score"] not in (0, 0.5, 1)
                or not parsed.get("reason")
            ):
                raise ValueError("BEAM rubric judge returned an invalid score")
            scores.append(float(parsed["score"]))
    if not scores:
        raise ValueError("missing judgments")
    return sum(scores) / len(scores)


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
            config["dataset"], Path(plan["source"]), config["source_sha256"], config["question_ids"]
        )
    questions = {q.question_id: q for h in histories for q in h.questions}
    calls = set()
    for path in (root / "model-calls").glob("*.json"):
        call = _read(path)
        if call.get("status") != 200:
            continue
        try:
            content = parse_completion(call["response"])
            request = call["request"]
            calls.add(
                (
                    call["stage"],
                    request["model"],
                    request["reasoning_effort"],
                    hashlib.sha256(request["messages"][0]["content"].encode()).hexdigest(),
                    content,
                )
            )
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
        answer_prompt = _answer_prompt(question, result["context"], dataset)
        expected = [("answer", answer_prompt, result["answer"])]
        judge_prompts = _judge_prompts(question, result["answer"], dataset)
        if len(judge_prompts) != len(result["judgments"]):
            raise ValueError("rubric judgment coverage is incomplete")
        expected.extend(
            ("judge", p, j) for p, j in zip(judge_prompts, result["judgments"], strict=True)
        )
        for stage, prompt, content in expected:
            key = (
                stage,
                plan["bindings"][stage],
                plan["config"][f"{stage}_effort"],
                hashlib.sha256(prompt.encode()).hexdigest(),
                content,
            )
            if key not in calls:
                raise ValueError("saved result differs from the actual complete model input/output")
        if not math.isclose(
            _score_judgments(result["judgments"], dataset), result["score"], abs_tol=1e-12
        ):
            raise ValueError("saved score differs from the actual judgments")
    return results


def _report(root: Path, plan: dict[str, Any], results: list[dict[str, Any]]) -> None:
    rows = "".join(
        f"<article><h2>{html.escape(r['question_id'])}</h2>"
        f"<p>Score: {r['score']}</p><p>{html.escape(r['question'])}</p>"
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
</style><h1>{title}</h1><p>{len(results)} unique questions. Mean score:
{sum(r["score"] for r in results) / len(results):.3f}.</p><p>Generation-only usage is in model-calls and
owners; embedding tokens are excluded. This run is separate from historical evidence.</p>{rows}</html>"""
    atomic_replace_bytes(root / "report.html", document.encode())


def run(root: Path, *, resume: bool) -> None:
    plan = _read(root / "plan.json")
    config = OAMRunConfig.model_validate(plan["config"])
    env = _environment(Path(plan["env_file"]), config)
    if plan["protocol_sha256"] != _protocol_hash():
        raise ValueError("workflow implementation changed; use a fresh precheck scope")
    current = {
        "oam": env["OAMB_OAM_BASE_URL"],
        "llm": env["LLM_BASE_URL"],
        "answer": env[config.answer_model_env],
        "judge": env[config.judge_model_env],
        "tenant": env[config.tenant_id_env],
    }
    if current != plan["bindings"]:
        raise ValueError("runtime/model binding changed")
    histories = load_histories(
        config.dataset, Path(plan["source"]), config.source_sha256, config.question_ids
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
            max_calls=config.max_model_calls,
            max_tokens=config.max_model_tokens,
            timeout=config.timeout_seconds,
        )
        for history in histories:
            _run_history(root, plan, config, env, adapter, cache, history)
    results = validate_results(root, plan, histories)
    _report(root, plan, results)
    _write(
        root / "summary.json",
        {
            "validated_questions": len(results),
            "diagnostic": config.diagnostic,
            "mean_score": sum(r["score"] for r in results) / len(results),
        },
        replace=True,
    )
    print(f"run: PASS ({len(results)} validated questions); report: {root / 'report.html'}")


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
            admit_indexing_source(
                root,
                source_ceiling=plan["indexing_budget"]["per_source_ceiling"],
                total_ceiling=config.max_indexing_tokens,
            )
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
                config.container_config,
                "--once",
                "--tenant-id",
                env[config.tenant_id_env],
                "--user-id",
                owner,
            ],
        )
        owners = report.get("owners", [])
        if (
            report.get("busy")
            or len(owners) != 1
            or owners[0].get("state") != "ready"
            or owners[0].get("tenant_id") != env[config.tenant_id_env]
            or owners[0].get("user_id") != owner
        ):
            raise ValueError("BEAM dedicated owner index is not ready")
        _write(directory / "vector-index.json", report, replace=True)
    index_plan = capture_index_plan(
        env["OAMB_OAM_CONTAINER"],
        config.container_config,
        env[config.tenant_id_env],
        owner,
        "small" if config.dataset == "lme500" else "large",
    )
    _write(directory / "index-plan.json", index_plan, replace=True)
    for question in history.questions:
        result_path = root / "results" / (canonical_sha256(question.question_id) + ".json")
        if result_path.exists():
            continue
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
        prompt = _answer_prompt(question, context, config.dataset)
        answer = cache.complete(
            env[config.answer_model_env],
            config.answer_effort,
            prompt,
            "answer",
            config.max_output_tokens,
        )
        judgments = [
            cache.complete(
                env[config.judge_model_env],
                config.judge_effort,
                prompt,
                "judge",
                config.max_output_tokens,
                validator=lambda text: _score_judgments([text], config.dataset),
            )
            for prompt in _judge_prompts(question, answer, config.dataset)
        ]
        score = _score_judgments(judgments, config.dataset)
        _write(
            result_path,
            {
                "question_id": question.question_id,
                "question": question.text,
                "owner": owner,
                "context": context,
                "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "context_tokens_o200k": len(
                    tiktoken.get_encoding("o200k_base").encode_ordinary(context)
                ),
                "answer": answer,
                "judgments": judgments,
                "score": score,
                "recall": recall,
            },
        )
        print(f"question: {question.question_id} scored {score:.3f}", flush=True)


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
    args = parser.parse_args()
    try:
        if args.command == "precheck":
            precheck(args.config, args.output, args.env_file)
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
