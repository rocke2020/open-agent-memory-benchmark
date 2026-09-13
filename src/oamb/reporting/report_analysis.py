"""Bounded, cached LLM analysis for the five comparison metrics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.runtime.provider_env import load_t10_provider_environment


def _load_report_analysis_prompt() -> tuple[str, str]:
    payload = files("oamb.reporting").joinpath("report_analysis_prompt.txt").read_bytes()
    try:
        prompt = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:  # pragma: no cover - invalid packaged resource
        raise RuntimeError("report analysis prompt must be UTF-8") from exc
    if not prompt.strip():  # pragma: no cover - invalid packaged resource
        raise RuntimeError("report analysis prompt must not be empty")
    return prompt, f"sha256:{hashlib.sha256(payload).hexdigest()}"


REPORT_ANALYSIS_NAME = "report-analysis.json"
REPORT_ANALYSIS_SYSTEM_PROMPT, REPORT_ANALYSIS_PROMPT_VERSION = _load_report_analysis_prompt()
REPORT_ANALYSIS_MAX_ATTEMPTS = 6
REPORT_ANALYSIS_MAX_OUTPUT_TOKENS = 8_192
REPORT_ANALYSIS_MAX_INSIGHT_CHARACTERS = 560
REPORT_ANALYSIS_METRICS = (
    "answer_accuracy",
    "context_tokens",
    "indexing_tokens",
    "retrieval_latency",
    "indexing_time",
)
REPORT_ANALYSIS_QUESTION_TYPES = (
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
)
REPORT_ANALYSIS_INVESTIGATION_ROOT = Path("docs/investigations")
_REPORT_ANALYSIS_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0)

_METRIC_DEFINITIONS = {
    "answer_accuracy": "Judged answer accuracy with its actual denominator.",
    "context_tokens": "Exact retrieved context tokens shown to the answer model.",
    "indexing_tokens": "Supplier-reported successful logical indexing producer tokens.",
    "retrieval_latency": "Provider retrieval request wall time; lower is faster.",
    "indexing_time": "First provider write through terminal indexing readiness.",
}
_METRIC_PRIORITIES = {
    "answer_accuracy": "co_primary",
    "context_tokens": "co_primary",
    "indexing_tokens": "important",
    "retrieval_latency": "important",
    "indexing_time": "secondary",
}
_PROVIDER_STAGE_ROLE_IDS = (
    "hindsight_extraction",
    "mem0_extraction",
    "openviking_semantic_understanding",
)
_RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 429})

Sleep = Callable[[float], None]
AnalysisGenerator = Callable[[dict[str, object]], bytes | None]


@dataclass(frozen=True, slots=True)
class ReportAnalysisBuildResult:
    analysis_path: Path
    cache_directory: Path
    cache_hit: bool


class _RetryableReportAnalysisError(ValueError):
    pass


def provider_stages_share_model_and_effort(models: object) -> bool:
    if not isinstance(models, (list, tuple)):
        return False
    bindings: dict[str, tuple[str, str]] = {}
    for item in models:
        if not isinstance(item, Mapping):
            continue
        role_id = item.get("role_id")
        if role_id not in _PROVIDER_STAGE_ROLE_IDS or role_id in bindings:
            continue
        model = item.get("model")
        effort = item.get("thinking_effort")
        if not isinstance(model, str) or not model or not isinstance(effort, str) or not effort:
            return False
        bindings[role_id] = (model, effort)
    return len(bindings) == len(_PROVIDER_STAGE_ROLE_IDS) and len(set(bindings.values())) == 1


def build_report_analysis_generator(
    *,
    plan: Any,
    model_env_path: Path,
    cache_root: Path,
    investigation_root: Path = REPORT_ANALYSIS_INVESTIGATION_ROOT,
    base_environment: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    sleep: Sleep = time.sleep,
) -> AnalysisGenerator:
    """Bind the report call to the frozen judge model and explicit model environment."""

    judge_roles = tuple(role for role in plan.model_roles if role.role_id == "judge")
    if len(judge_roles) != 1:
        raise ValueError("report analysis requires exactly one frozen judge model role")
    role = judge_roles[0]
    expected_keys = frozenset(
        {"LLM_URL_TYPE", role.endpoint_variable, role.credential_variable, role.model}
    )
    environment = dict(os.environ if base_environment is None else base_environment)
    environment.update(load_t10_provider_environment(model_env_path, expected_keys=expected_keys))
    if environment.get("LLM_URL_TYPE") != "openai_chat":
        raise ValueError("report analysis requires LLM_URL_TYPE=openai_chat")
    try:
        base_url = environment[role.endpoint_variable]
        api_key = environment[role.credential_variable]
    except KeyError as exc:
        raise ValueError("report analysis model environment is incomplete") from exc

    def generate(report: dict[str, object]) -> bytes | None:
        try:
            result = generate_report_analysis(
                report,
                model=environment[role.model],
                thinking_effort=role.thinking_effort,
                base_url=base_url,
                api_key=api_key,
                cache_root=cache_root,
                timeout_seconds=float(plan.execution.operation_timeout_seconds),
                investigation_root=investigation_root,
                transport=transport,
                sleep=sleep,
            )
            return read_regular_file(result.analysis_path) if result is not None else None
        except (OSError, ValueError):
            return None

    return generate


def generate_report_analysis(
    report: Mapping[str, object],
    *,
    model: str,
    thinking_effort: Literal["low", "high", "max"],
    base_url: str,
    api_key: str,
    cache_root: Path,
    timeout_seconds: float,
    investigation_root: Path = REPORT_ANALYSIS_INVESTIGATION_ROOT,
    transport: httpx.BaseTransport | None = None,
    sleep: Sleep = time.sleep,
) -> ReportAnalysisBuildResult | None:
    """Generate one validated analysis, retrying at most six total attempts."""

    if not model or thinking_effort not in {"low", "high", "max"}:
        raise ValueError("report analysis requires a generative model and thinking effort")
    if not base_url or not api_key:
        raise ValueError("report analysis endpoint and credential are required")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("report analysis timeout must be finite positive")

    report_bytes = canonical_json_bytes(report)
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    analysis_input = _analysis_input(report, investigation_root=investigation_root)
    analysis_input_bytes = canonical_json_bytes(analysis_input)
    analysis_input_sha256 = hashlib.sha256(analysis_input_bytes).hexdigest()
    endpoint_fingerprint = hashlib.sha256(base_url.rstrip("/").encode("utf-8")).hexdigest()
    cache_key = canonical_sha256(
        [
            "oamb-report-analysis-cache-v1",
            report_sha256,
            analysis_input_sha256,
            REPORT_ANALYSIS_PROMPT_VERSION,
            model,
            thinking_effort,
            endpoint_fingerprint,
        ]
    )
    cache_directory = Path(cache_root) / cache_key
    analysis_path = cache_directory / REPORT_ANALYSIS_NAME
    if analysis_path.exists() or analysis_path.is_symlink():
        payload = read_regular_file(analysis_path)
        parse_report_analysis(payload, report, investigation_root=investigation_root)
        return ReportAnalysisBuildResult(
            analysis_path=analysis_path,
            cache_directory=cache_directory,
            cache_hit=True,
        )

    investigations = analysis_input["investigations"]
    evaluation_evidence = {
        key: value for key, value in analysis_input.items() if key != "investigations"
    }
    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": REPORT_ANALYSIS_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": canonical_json_bytes(
                {
                    "task": "Read every investigation document before analyzing results.",
                    "investigations": investigations,
                }
            ).decode("utf-8"),
        },
        {
            "role": "user",
            "content": canonical_json_bytes(
                {
                    "task": (
                        "Now produce first-glance provider insights after checking every supplied "
                        "question result and question-type summary."
                    ),
                    "evidence": evaluation_evidence,
                }
            ).decode("utf-8"),
        },
    ]
    client = httpx.Client(
        base_url=base_url.rstrip("/"),
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=httpx.Timeout(timeout_seconds),
        transport=transport,
        trust_env=False,
        follow_redirects=False,
    )
    try:
        for attempt in range(1, REPORT_ANALYSIS_MAX_ATTEMPTS + 1):
            try:
                response = client.post(
                    "/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "n": 1,
                        "temperature": 0,
                        "top_p": 1,
                        "max_tokens": REPORT_ANALYSIS_MAX_OUTPUT_TOKENS,
                        "reasoning_effort": thinking_effort,
                    },
                )
            except (httpx.TimeoutException, httpx.TransportError):
                if attempt == REPORT_ANALYSIS_MAX_ATTEMPTS:
                    return None
                sleep(_retry_delay(attempt))
                continue

            raw_bytes = response.content
            raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            atomic_write_bytes(
                cache_directory / "attempts" / f"{attempt:02d}-{raw_sha256}-response.json",
                raw_bytes,
                trusted_root=Path(cache_root),
            )
            if not 200 <= response.status_code < 300:
                retryable = (
                    response.status_code in _RETRYABLE_HTTP_STATUS_CODES
                    or response.status_code >= 500
                )
                if not retryable or attempt == REPORT_ANALYSIS_MAX_ATTEMPTS:
                    return None
                sleep(_retry_delay(attempt))
                continue

            try:
                output, usage = _parse_completion_response(raw_bytes)
                overall, provider_insights = _parse_analysis_output(
                    output,
                    analysis_input=analysis_input,
                )
            except _RetryableReportAnalysisError as error:
                if attempt == REPORT_ANALYSIS_MAX_ATTEMPTS:
                    return None
                messages.extend(
                    (
                        {"role": "assistant", "content": _safe_correction_output(raw_bytes)},
                        {
                            "role": "user",
                            "content": (
                                "The previous response failed validation: "
                                f"{error}. Return corrected strict JSON only."
                            ),
                        },
                    )
                )
                sleep(_retry_delay(attempt))
                continue

            body: dict[str, object] = {
                "schema_name": "report_analysis",
                "schema_version": 1,
                "report_id": _required_text(report, "report_id"),
                "report_export_sha256": report_sha256,
                "analysis_input_sha256": analysis_input_sha256,
                "prompt_version": REPORT_ANALYSIS_PROMPT_VERSION,
                "model": model,
                "thinking_effort": thinking_effort,
                "endpoint_fingerprint": endpoint_fingerprint,
                "attempt_count": attempt,
                "raw_response_sha256": raw_sha256,
                "usage": usage,
                "overall": overall,
                "provider_insights": provider_insights,
            }
            document = {
                **body,
                "analysis_id": canonical_sha256(["oamb-report-analysis-v1", body]),
            }
            payload = canonical_json_bytes(document)
            parse_report_analysis(payload, report, investigation_root=investigation_root)
            atomic_write_bytes(
                analysis_path,
                payload,
                trusted_root=Path(cache_root),
            )
            return ReportAnalysisBuildResult(
                analysis_path=analysis_path,
                cache_directory=cache_directory,
                cache_hit=False,
            )
    finally:
        client.close()
    raise AssertionError("report analysis retry loop exceeded its bound")


def parse_report_analysis(
    payload: bytes,
    report: Mapping[str, object],
    *,
    investigation_root: Path = REPORT_ANALYSIS_INVESTIGATION_ROOT,
) -> dict[str, object]:
    """Parse and bind cached analysis to the exact deterministic report export."""

    try:
        value = json.loads(payload, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("report analysis is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("report analysis root must be an object")
    expected_keys = {
        "schema_name",
        "schema_version",
        "analysis_id",
        "report_id",
        "report_export_sha256",
        "analysis_input_sha256",
        "prompt_version",
        "model",
        "thinking_effort",
        "endpoint_fingerprint",
        "attempt_count",
        "raw_response_sha256",
        "usage",
        "overall",
        "provider_insights",
    }
    if set(value) != expected_keys:
        raise ValueError("report analysis fields do not match the closed schema")
    if value["schema_name"] != "report_analysis" or value["schema_version"] != 1:
        raise ValueError("report analysis schema is unsupported")
    report_bytes = canonical_json_bytes(report)
    if (
        value["report_id"] != _required_text(report, "report_id")
        or value["report_export_sha256"] != hashlib.sha256(report_bytes).hexdigest()
    ):
        raise ValueError("report analysis does not bind the exact report export")
    analysis_input = _analysis_input(report, investigation_root=investigation_root)
    expected_input_hash = hashlib.sha256(canonical_json_bytes(analysis_input)).hexdigest()
    if value["analysis_input_sha256"] != expected_input_hash:
        raise ValueError("report analysis compact input binding drifted")
    if value["prompt_version"] != REPORT_ANALYSIS_PROMPT_VERSION:
        raise ValueError("report analysis prompt version drifted")
    if value["thinking_effort"] not in {"low", "high", "max"}:
        raise ValueError("report analysis thinking effort is invalid")
    if not isinstance(value["attempt_count"], int) or not (
        1 <= value["attempt_count"] <= REPORT_ANALYSIS_MAX_ATTEMPTS
    ):
        raise ValueError("report analysis attempt count is invalid")
    for name in ("model", "endpoint_fingerprint", "raw_response_sha256"):
        if not isinstance(value[name], str) or not value[name]:
            raise ValueError(f"report analysis {name} is invalid")
    overall = _concise_sentence(value["overall"], "overall", maximum_characters=480)
    provider_ids = tuple(
        _required_text(cell, "provider_id") for cell in _required_report_cells(report)
    )
    provider_insights = _parse_provider_insights(
        value["provider_insights"],
        provider_ids=provider_ids,
    )
    _validate_visible_analysis(
        overall,
        provider_insights,
        analysis_input=analysis_input,
    )
    body = dict(value)
    analysis_id = body.pop("analysis_id")
    if analysis_id != canonical_sha256(["oamb-report-analysis-v1", body]):
        raise ValueError("report analysis identity does not close")
    value["overall"] = overall
    return cast(dict[str, object], value)


def _analysis_input(
    report: Mapping[str, object],
    *,
    investigation_root: Path = REPORT_ANALYSIS_INVESTIGATION_ROOT,
) -> dict[str, object]:
    raw_cells = report.get("cells")
    if not isinstance(raw_cells, (list, tuple)) or not raw_cells:
        raise ValueError("report analysis requires provider cells")
    cells: list[dict[str, object]] = []
    for raw_cell in raw_cells:
        if not isinstance(raw_cell, Mapping):
            raise ValueError("report analysis provider cell is invalid")
        accounting = _required_mapping(raw_cell, "accounting")
        tokens = _required_mapping(accounting, "tokens")
        indexing = _required_mapping(tokens, "indexing")
        observed_time = _required_mapping(raw_cell, "observed_time")
        cells.append(
            {
                "provider_id": _required_text(raw_cell, "provider_id"),
                "answer_accuracy": {
                    "numerator": raw_cell.get("judged_numerator", "unavailable"),
                    "denominator": raw_cell.get("judged_denominator", "unavailable"),
                },
                "context_tokens": accounting.get(
                    "answer_visible_context_tokens", {"status": "unavailable"}
                ),
                "indexing_tokens": {
                    "coverage": indexing.get("supplier_usage_coverage", {"status": "unavailable"}),
                    "supplier_reported_total_tokens": _required_mapping(indexing, "totals").get(
                        "supplier_reported_total_tokens", {"status": "unavailable"}
                    ),
                },
                "retrieval_latency": _analysis_time_seconds(
                    observed_time.get("provider_request", {"status": "unavailable"})
                ),
                "indexing_time": _analysis_time_seconds(
                    observed_time.get("indexing_ready", {"status": "unavailable"})
                ),
                "time_comparability": observed_time.get("comparability", "unavailable"),
                "limitations": raw_cell.get("limitations", ()),
            }
        )
    provider_ids = tuple(str(item["provider_id"]) for item in cells)
    question_evidence = _question_evidence(report, provider_ids=provider_ids)
    for cell in cells:
        cell["question_type_accuracy"] = _question_type_accuracy(
            str(cell["provider_id"]),
            question_evidence,
        )
    raw_retrieval = report.get("retrieval", ())
    if not isinstance(raw_retrieval, (list, tuple)):
        raise ValueError("report analysis retrieval inventory is invalid")
    retrieval_routes = tuple(
        {
            "provider_id": item.get("provider_id", "unavailable"),
            "route": item.get("route", "unavailable"),
        }
        for item in raw_retrieval
        if isinstance(item, Mapping)
    )
    return {
        "investigations": _load_investigations(investigation_root),
        "metric_order": REPORT_ANALYSIS_METRICS,
        "metric_definitions": tuple(
            {"metric_id": metric_id, "definition": _METRIC_DEFINITIONS[metric_id]}
            for metric_id in REPORT_ANALYSIS_METRICS
        ),
        "metric_priorities": tuple(
            {"metric_id": metric_id, "importance": _METRIC_PRIORITIES[metric_id]}
            for metric_id in REPORT_ANALYSIS_METRICS
        ),
        "model_effort_controls": _model_effort_control_evidence(
            report,
            provider_ids=provider_ids,
        ),
        "coverage": report.get("coverage", {"status": "unavailable"}),
        "cells": tuple(cells),
        "questions": question_evidence,
        "accuracy_decision": report.get("accuracy_decision", {"status": "unavailable"}),
        "pairwise_accuracy": report.get("comparisons", ()),
        "retrieval_controls": _retrieval_control_evidence(
            report,
            provider_ids=provider_ids,
        ),
        "retrieval_routes": retrieval_routes,
        "limitations": report.get("limitations", ()),
    }


def _model_effort_control_evidence(
    report: Mapping[str, object],
    *,
    provider_ids: tuple[str, ...],
) -> dict[str, object]:
    raw_models = report.get("models")
    if not isinstance(raw_models, (list, tuple)) or not raw_models:
        return {"status": "unavailable"}
    bindings: list[dict[str, str]] = []
    for item in raw_models:
        if not isinstance(item, Mapping):
            raise ValueError("report analysis model-effort controls are invalid")
        bindings.append(
            {
                "role_id": _required_text(item, "role_id"),
                "model": _required_text(item, "model"),
                "thinking_effort": _required_text(item, "thinking_effort"),
            }
        )
    return {
        "status": (
            "shared_across_providers"
            if provider_stages_share_model_and_effort(raw_models)
            else "provider_stages_differ"
        ),
        "providers": provider_ids,
        "bindings": tuple(bindings),
    }


def _retrieval_control_evidence(
    report: Mapping[str, object],
    *,
    provider_ids: tuple[str, ...],
) -> dict[str, object]:
    raw_sources = report.get("saved_result_sources")
    if not isinstance(raw_sources, (list, tuple)) or not raw_sources:
        return {"status": "unavailable"}
    sources = tuple(item for item in raw_sources if isinstance(item, Mapping))
    if len(sources) != len(raw_sources):
        raise ValueError("report analysis saved-result controls are invalid")
    by_provider = {_required_text(item, "provider_id"): item for item in sources}
    if len(by_provider) != len(sources) or set(by_provider) != set(provider_ids):
        raise ValueError("report analysis saved-result controls do not close")
    providers: list[dict[str, object]] = []
    for provider_id in provider_ids:
        source = by_provider[provider_id]
        top_k = source.get("retrieval_top_k", "unavailable")
        observed_max = source.get("observed_native_candidate_count_max", "unavailable")
        providers.append(
            {
                "provider_id": provider_id,
                "effective_top_k": top_k,
                "top_k_source": source.get("retrieval_top_k_source", "unavailable"),
                "source_plan_top_k": source.get("source_plan_retrieval_top_k", "unavailable"),
                "observed_native_candidate_count_max": observed_max,
            }
        )
    top_ks = tuple(item["effective_top_k"] for item in providers)
    all_share = all(type(value) is int and value > 0 for value in top_ks) and len(set(top_ks)) == 1
    shared_top_k_value = cast(int, top_ks[0]) if all_share else None
    observed_maxima = tuple(item["observed_native_candidate_count_max"] for item in providers)
    all_within = shared_top_k_value is not None and all(
        type(value) is int and 0 <= value <= shared_top_k_value for value in observed_maxima
    )
    comparisons = report.get("comparisons", ())
    expected_pair_count = len(provider_ids) * (len(provider_ids) - 1) // 2
    all_pairs_comparable = (
        isinstance(comparisons, (list, tuple))
        and len(comparisons) == expected_pair_count
        and all(
            isinstance(item, Mapping) and item.get("comparable") is True for item in comparisons
        )
    )
    return {
        "status": "available",
        "scope": "report_comparison_normalization_ceiling",
        "providers": tuple(providers),
        "shared_top_k": (shared_top_k_value if shared_top_k_value is not None else "unavailable"),
        "all_providers_share_top_k": all_share,
        "all_observed_candidates_within_top_k": all_within,
        "all_provider_pairs_comparable": all_pairs_comparable,
    }


def _load_investigations(root: Path) -> tuple[dict[str, str], ...]:
    selected_root = Path(root)
    if selected_root.is_symlink() or not selected_root.is_dir():
        raise ValueError("report analysis investigations must be a regular directory")
    paths = tuple(
        sorted(
            selected_root.rglob("*.md"),
            key=lambda path: path.relative_to(selected_root).as_posix(),
        )
    )
    if not paths:
        raise ValueError("report analysis investigations are empty")
    documents: list[dict[str, str]] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError("report analysis investigation must be a regular file")
        payload = read_regular_file(path)
        try:
            content = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("report analysis investigation must be UTF-8") from exc
        if not content.strip():
            raise ValueError("report analysis investigation must not be empty")
        documents.append(
            {
                "relative_path": path.relative_to(selected_root).as_posix(),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "content": content,
            }
        )
    return tuple(documents)


def _question_type_accuracy(
    provider_id: str,
    questions: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    summaries: list[dict[str, object]] = []
    for question_type in REPORT_ANALYSIS_QUESTION_TYPES:
        results = [
            next(
                result
                for result in cast(tuple[dict[str, str], ...], question["provider_results"])
                if result["provider_id"] == provider_id
            )
            for question in questions
            if question["question_type"] == question_type
        ]
        if not results:
            raise ValueError("report analysis question-type inventory is incomplete")
        correct = sum(result["result"] == "correct" for result in results)
        summaries.append(
            {
                "question_type": question_type,
                "correct": correct,
                "total": len(results),
                "incorrect": len(results) - correct,
            }
        )
    return tuple(summaries)


def _question_evidence(
    report: Mapping[str, object],
    *,
    provider_ids: tuple[str, ...],
) -> tuple[dict[str, object], ...]:
    questions = report.get("questions")
    if not isinstance(questions, (list, tuple)) or not questions:
        raise ValueError("report analysis requires question evidence")
    evidence: list[dict[str, object]] = []
    question_types: set[str] = set()
    for question in questions:
        if not isinstance(question, Mapping):
            raise ValueError("report analysis question evidence is invalid")
        question_type = _required_text(question, "question_type")
        question_types.add(question_type)
        raw_results = question.get("provider_results")
        if not isinstance(raw_results, (list, tuple)):
            raise ValueError("report analysis question provider results are invalid")
        provider_results: list[dict[str, str]] = []
        for result in raw_results:
            if not isinstance(result, Mapping):
                raise ValueError("report analysis question provider result is invalid")
            display_state = _required_text(result, "display_state")
            if display_state not in {"correct", "incorrect"}:
                raise ValueError("report analysis requires terminal judged question results")
            provider_results.append(
                {
                    "provider_id": _required_text(result, "provider_id"),
                    "result": display_state,
                    "model_answer": _required_text(result, "model_answer"),
                    "judge_decision": _required_text(result, "judge_decision"),
                }
            )
        if tuple(item["provider_id"] for item in provider_results) != provider_ids:
            raise ValueError("report analysis question provider inventory drifted")
        evidence.append(
            {
                "question_id": _required_text(question, "raw_question_id"),
                "question_type": question_type,
                "question": _required_text(question, "question"),
                "gold_answer": _gold_answer(question),
                "provider_results": tuple(provider_results),
            }
        )
    if question_types != set(REPORT_ANALYSIS_QUESTION_TYPES):
        raise ValueError("report analysis requires all six question types")
    return tuple(evidence)


def _gold_answer(question: Mapping[str, object]) -> str | int | float:
    value = question.get("gold_answer")
    if isinstance(value, str) and value:
        return value
    if type(value) is int or (type(value) is float and math.isfinite(value)):
        return value
    raise ValueError("report analysis gold answer is invalid")


def _parse_completion_response(raw_bytes: bytes) -> tuple[str, object]:
    try:
        value = json.loads(raw_bytes, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _RetryableReportAnalysisError("supplier response is not strict JSON") from exc
    if not isinstance(value, dict):
        raise _RetryableReportAnalysisError("supplier response root is not an object")
    choices = value.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise _RetryableReportAnalysisError("supplier response requires exactly one choice")
    choice = choices[0]
    if not isinstance(choice, dict) or choice.get("finish_reason") != "stop":
        raise _RetryableReportAnalysisError("supplier response did not finish normally")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise _RetryableReportAnalysisError("supplier response has no message")
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise _RetryableReportAnalysisError("supplier response has no visible content")
    return content, _usage_summary(value.get("usage"))


def _usage_summary(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"status": "unavailable"}
    prompt_tokens = value.get("prompt_tokens")
    completion_tokens = value.get("completion_tokens")
    total_tokens = value.get("total_tokens")
    if (
        type(prompt_tokens) is not int
        or prompt_tokens < 0
        or type(completion_tokens) is not int
        or completion_tokens < 0
        or type(total_tokens) is not int
        or total_tokens < 0
        or prompt_tokens + completion_tokens != total_tokens
    ):
        return {"status": "unavailable"}
    return {
        "status": "measured",
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _parse_analysis_output(
    output: str,
    *,
    analysis_input: Mapping[str, object],
) -> tuple[str, tuple[dict[str, str], ...]]:
    try:
        value = json.loads(output, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _RetryableReportAnalysisError("analysis content is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"overall", "provider_insights"}:
        raise _RetryableReportAnalysisError("analysis content has the wrong fields")
    provider_ids = tuple(
        _required_text(cell, "provider_id") for cell in _required_cells(analysis_input)
    )
    overall = _concise_sentence(value["overall"], "overall", maximum_characters=480)
    provider_insights = _parse_provider_insights(
        value["provider_insights"],
        provider_ids=provider_ids,
    )
    _validate_visible_analysis(
        overall,
        provider_insights,
        analysis_input=analysis_input,
    )
    return overall, provider_insights


def _parse_provider_insights(
    value: object,
    *,
    provider_ids: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list) or len(value) != len(provider_ids):
        raise _RetryableReportAnalysisError("analysis provider insight inventory is incomplete")
    insights: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {"provider_id", "insight"}:
            raise _RetryableReportAnalysisError("analysis provider insight is invalid")
        provider_id_value = item.get("provider_id")
        if not isinstance(provider_id_value, str) or not provider_id_value:
            raise _RetryableReportAnalysisError("analysis provider identity is invalid")
        provider_id = provider_id_value
        insight = _concise_sentence(
            item["insight"],
            f"{provider_id} insight",
            maximum_characters=REPORT_ANALYSIS_MAX_INSIGHT_CHARACTERS,
        )
        if not any(question_type in insight for question_type in REPORT_ANALYSIS_QUESTION_TYPES):
            raise _RetryableReportAnalysisError("each provider insight must name a question type")
        if re.search(r"\b\d{1,2}/\d{1,2}\b", insight) is None:
            raise _RetryableReportAnalysisError(
                "each provider insight must include correct/total evidence"
            )
        insights.append({"provider_id": provider_id, "insight": insight})
    if tuple(item["provider_id"] for item in insights) != provider_ids:
        raise _RetryableReportAnalysisError("analysis provider order or identity is invalid")
    return tuple(insights)


def _validate_visible_analysis(
    overall: str,
    provider_insights: Iterable[Mapping[str, str]],
    *,
    analysis_input: Mapping[str, object],
) -> None:
    cells = _required_cells(analysis_input)
    recorded_leader = _recorded_accuracy_leader(cells)
    normalized_overall = overall.casefold()
    overall_sentences = tuple(
        sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", overall) if sentence.strip()
    )
    model_effort_controls = analysis_input.get("model_effort_controls")
    shared_provider_stages = (
        isinstance(model_effort_controls, Mapping)
        and model_effort_controls.get("status") == "shared_across_providers"
    )
    required_opening = (
        "With the same models and thinking effort at each corresponding stage across providers,"
        if shared_provider_stages
        else "Across the evaluated providers,"
    )
    if len(overall_sentences) != 1 or not overall.startswith(required_opening):
        raise _RetryableReportAnalysisError(
            "overall must give accuracy and context equal space in one controlled sentence"
        )
    if recorded_leader is not None:
        provider_id, numerator, denominator = recorded_leader
        if provider_id.casefold() not in normalized_overall or (
            f"{numerator}/{denominator}" not in overall
        ):
            raise _RetryableReportAnalysisError(
                "overall must name the highest recorded provider and its exact result"
            )
        leader_fraction = f"{numerator}/{denominator}"
        if any(
            fraction != leader_fraction for fraction in re.findall(r"\b\d{1,3}/\d{1,3}\b", overall)
        ):
            raise _RetryableReportAnalysisError(
                "overall must not repeat nonleading provider results"
            )
    if re.search(
        r"\b(?:accuracy[_ ]decision|equal[-_ ]coverage|failed predicates?|threshold checks?)\b",
        overall,
        flags=re.IGNORECASE,
    ):
        raise _RetryableReportAnalysisError("overall must not expose internal decision jargon")
    context_leaders = _recorded_context_leaders(cells)
    context_sentence = overall_sentences[0]
    other_provider_ids = tuple(
        _required_text(cell, "provider_id")
        for cell in cells
        if _required_text(cell, "provider_id") not in context_leaders
    )
    if context_leaders and not any(
        provider_id.casefold() in context_sentence.casefold() for provider_id in context_leaders
    ):
        raise _RetryableReportAnalysisError("overall must name the lowest-context provider")
    if (
        "context" not in context_sentence.casefold()
        or re.search(r"\d", context_sentence) is None
        or not all(
            provider_id.casefold() in context_sentence.casefold()
            for provider_id in other_provider_ids
        )
    ):
        raise _RetryableReportAnalysisError(
            "overall must quantify the lowest-context provider against every peer"
        )
    insights = tuple(provider_insights)
    expected_by_provider = _question_type_accuracy_by_provider(cells)
    slowest_indexing_provider = _recorded_slowest_indexing_provider(cells)
    covered_operational_metrics: set[str] = set()
    for item in insights:
        provider_id = item["provider_id"]
        insight = item["insight"]
        mentioned_types = tuple(
            question_type
            for question_type in REPORT_ANALYSIS_QUESTION_TYPES
            if question_type in insight
        )
        sentences = tuple(
            sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", insight) if sentence.strip()
        )
        if len(sentences) != 2 or not 1 <= len(mentioned_types) <= 3:
            raise _RetryableReportAnalysisError(
                "each provider insight must use one to three question types in two sentences"
            )
        for question_type in mentioned_types:
            correct, total = expected_by_provider[provider_id][question_type]
            if f"{correct}/{total}" not in insight:
                raise _RetryableReportAnalysisError(
                    "provider insight question-type evidence does not match the report"
                )
        allowed_fractions = {
            f"{correct}/{total}" for correct, total in expected_by_provider[provider_id].values()
        }
        if any(
            fraction not in allowed_fractions
            for fraction in re.findall(r"\b\d{1,3}/\d{1,3}\b", insight)
        ):
            raise _RetryableReportAnalysisError(
                "provider insight contains a result outside the current report"
            )
        if re.search(
            r"\b(?:external|managed|official|public|self-evaluation|benchmark)\b|"
            r"\breference comparison\b",
            insight,
            flags=re.IGNORECASE,
        ):
            raise _RetryableReportAnalysisError(
                "provider insight must stay on the current report's evidence"
            )
        if re.search(
            r"\bvisible\s+evidence\b|\b(?:retrieved|injected)\s+(?:evidence|context)\b|"
            r"\bmemory\s+(?:contained|held|lacked|missed|omitted|retained|unused)\b|"
            r"\b(?:contained|held|lacked|missing|omitted|present|retained|unused)\s+"
            r"(?:in|from)\s+(?:the\s+)?memory\b",
            insight,
            flags=re.IGNORECASE,
        ):
            raise _RetryableReportAnalysisError(
                "provider insight cannot infer unsupplied retrieval-context contents"
            )
        other_provider_ids = tuple(
            candidate for candidate in expected_by_provider if candidate != provider_id
        )
        if not any(
            candidate.casefold() in sentences[1].casefold() for candidate in other_provider_ids
        ):
            raise _RetryableReportAnalysisError(
                "provider operational insight must compare another supplied provider"
            )
        operational_metrics = _mentioned_operational_metrics(sentences[1])
        if not operational_metrics:
            raise _RetryableReportAnalysisError(
                "each provider insight must include a non-accuracy metric comparison"
            )
        if "indexing_time" in operational_metrics and slowest_indexing_provider is not None:
            if provider_id != slowest_indexing_provider or not all(
                candidate.casefold() in sentences[1].casefold() for candidate in other_provider_ids
            ):
                raise _RetryableReportAnalysisError(
                    "indexing-time insight must lead with the largest cross-provider gap"
                )
        covered_operational_metrics.update(operational_metrics)
    if not {
        "context_tokens",
        "indexing_tokens",
        "retrieval_latency",
        "indexing_time",
    }.issubset(covered_operational_metrics):
        raise _RetryableReportAnalysisError(
            "provider insights must cover all four operational metrics"
        )
    visible_text = " ".join((overall, *(item["insight"] for item in insights)))
    if re.search(
        r"\btop[_ -]?k\b|\bcandidate[- ]ceilings?\b|\bretrieval[- ]control(?:s| alignment)?\b",
        visible_text,
        flags=re.IGNORECASE,
    ):
        raise _RetryableReportAnalysisError(
            "visible analysis must omit retrieval-control configuration"
        )
    if re.search(
        r"\b(?:evaluation|benchmark)[- ](?:process|configuration|design|controls?)\b|"
        r"\bcomparability\b|\bdecision[- ]thresholds?\b",
        visible_text,
        flags=re.IGNORECASE,
    ):
        raise _RetryableReportAnalysisError(
            "visible analysis must trust the validated evaluation and focus on results"
        )
    if re.search(
        r"\b(?:costs?|costly|cheaper|expensive)\b",
        visible_text,
        flags=re.IGNORECASE,
    ):
        raise _RetryableReportAnalysisError(
            "visible analysis must let token differences speak for themselves"
        )
    if re.search(r"\b(?:thousand|million|billion)\b", visible_text, flags=re.IGNORECASE):
        raise _RetryableReportAnalysisError(
            "visible analysis must use compact k, M, or B number suffixes"
        )
    if re.search(r"\d(?:\.\d+)?[kMB]\b", visible_text):
        raise _RetryableReportAnalysisError(
            "visible analysis must put a space before k, M, or B number suffixes"
        )
    if re.search(
        r"(?<![A-Za-z])(?:microseconds?|milliseconds?|[µμ]s|ms)(?![A-Za-z])",
        visible_text,
        flags=re.IGNORECASE,
    ):
        raise _RetryableReportAnalysisError("visible analysis durations must use seconds")
    for number in re.findall(r"\d[\d,.]*", visible_text):
        if len(re.sub(r"\D", "", number)) > 4:
            raise _RetryableReportAnalysisError(
                "visible analysis numbers must contain at most four digits"
            )


def _recorded_accuracy_leader(
    cells: tuple[Mapping[str, object], ...],
) -> tuple[str, int, int] | None:
    scored: list[tuple[Fraction, str, int, int]] = []
    for cell in cells:
        accuracy = cell.get("answer_accuracy")
        if not isinstance(accuracy, Mapping):
            return None
        numerator = accuracy.get("numerator")
        denominator = accuracy.get("denominator")
        if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
            return None
        scored.append(
            (
                Fraction(numerator, denominator),
                _required_text(cell, "provider_id"),
                numerator,
                denominator,
            )
        )
    highest = max(item[0] for item in scored)
    leaders = tuple(item for item in scored if item[0] == highest)
    if len(leaders) != 1:
        return None
    _, provider_id, numerator, denominator = leaders[0]
    return provider_id, numerator, denominator


def _recorded_context_leaders(
    cells: tuple[Mapping[str, object], ...],
) -> tuple[str, ...]:
    scored: list[tuple[Decimal, str]] = []
    for cell in cells:
        context = cell.get("context_tokens")
        if not isinstance(context, Mapping):
            return ()
        try:
            mean = Decimal(str(context.get("mean")))
        except InvalidOperation:
            return ()
        if not mean.is_finite() or mean < 0:
            return ()
        scored.append((mean, _required_text(cell, "provider_id")))
    if not scored:
        return ()
    lowest = min(item[0] for item in scored)
    return tuple(provider_id for mean, provider_id in scored if mean == lowest)


def _recorded_slowest_indexing_provider(
    cells: tuple[Mapping[str, object], ...],
) -> str | None:
    scored: list[tuple[Decimal, str]] = []
    for cell in cells:
        indexing_time = cell.get("indexing_time")
        if not isinstance(indexing_time, Mapping):
            return None
        try:
            median = Decimal(str(indexing_time.get("median_seconds")))
        except InvalidOperation:
            return None
        if not median.is_finite() or median < 0:
            return None
        scored.append((median, _required_text(cell, "provider_id")))
    if not scored:
        return None
    slowest = max(item[0] for item in scored)
    providers = tuple(provider_id for median, provider_id in scored if median == slowest)
    return providers[0] if len(providers) == 1 else None


def _mentioned_operational_metrics(text: str) -> set[str]:
    normalized = text.casefold()
    metrics: set[str] = set()
    if re.search(
        r"\b(?:answer[-\s]+)?context[-\s]+(?:tokens?|loads?)\b",
        normalized,
    ):
        metrics.add("context_tokens")
    if re.search(r"\bindexing[-\s]+tokens?\b", normalized):
        metrics.add("indexing_tokens")
    if re.search(
        r"\bretrieval[-\s]+(?:latency|median)\b|\bmedian[-\s]+retrieval\b",
        normalized,
    ):
        metrics.add("retrieval_latency")
    if re.search(r"\bindexing[-\s]+time\b", normalized):
        metrics.add("indexing_time")
    return metrics


def _question_type_accuracy_by_provider(
    cells: tuple[Mapping[str, object], ...],
) -> dict[str, dict[str, tuple[object, object]]]:
    expected: dict[str, dict[str, tuple[object, object]]] = {}
    for cell in cells:
        records = cell.get("question_type_accuracy")
        if not isinstance(records, (list, tuple)):
            raise ValueError("report analysis question-type evidence is invalid")
        expected[_required_text(cell, "provider_id")] = {
            _required_text(record, "question_type"): (
                record.get("correct"),
                record.get("total"),
            )
            for record in records
            if isinstance(record, Mapping)
        }
    return expected


def _analysis_time_seconds(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {"status": "unavailable"}
    return {
        "status": value.get("status", "unavailable"),
        "count": value.get("count", "unavailable"),
        "median_seconds": _microseconds_as_seconds(value.get("median_microseconds")),
        "p95_seconds": _microseconds_as_seconds(value.get("p95_microseconds")),
        "maximum_seconds": _microseconds_as_seconds(value.get("maximum_microseconds")),
    }


def _microseconds_as_seconds(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return "unavailable"
    try:
        seconds = Decimal(str(value)) / Decimal(1_000_000)
    except InvalidOperation:
        return "unavailable"
    return format(seconds, "f") if seconds.is_finite() else "unavailable"


def _required_cells(value: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    cells = value.get("cells")
    if not isinstance(cells, (list, tuple)) or any(not isinstance(cell, Mapping) for cell in cells):
        raise ValueError("report analysis requires provider cells")
    return cast(tuple[Mapping[str, object], ...], tuple(cells))


def _required_report_cells(report: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return _required_cells(report)


def _concise_sentence(value: object, label: str, *, maximum_characters: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _RetryableReportAnalysisError(f"{label} must be non-empty text")
    normalized = value.strip()
    if "\n" in normalized or "\r" in normalized or len(normalized) > maximum_characters:
        raise _RetryableReportAnalysisError(f"{label} must be compact single-line text")
    return normalized


def _safe_correction_output(raw_bytes: bytes) -> str:
    try:
        value = json.loads(raw_bytes)
        choices = value.get("choices") if isinstance(value, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        return content if isinstance(content, str) else ""
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""


def _required_text(value: Mapping[str, object], key: str) -> str:
    selected = value.get(key)
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"report analysis requires text field {key}")
    return selected


def _required_mapping(value: Mapping[str, object], key: str) -> Mapping[str, object]:
    selected = value.get(key)
    if not isinstance(selected, Mapping):
        raise ValueError(f"report analysis requires object field {key}")
    return selected


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _retry_delay(attempt: int) -> float:
    return _REPORT_ANALYSIS_RETRY_DELAYS[
        min(max(attempt, 1), len(_REPORT_ANALYSIS_RETRY_DELAYS)) - 1
    ]


__all__ = [
    "REPORT_ANALYSIS_MAX_ATTEMPTS",
    "REPORT_ANALYSIS_METRICS",
    "REPORT_ANALYSIS_NAME",
    "ReportAnalysisBuildResult",
    "build_report_analysis_generator",
    "generate_report_analysis",
    "parse_report_analysis",
    "provider_stages_share_model_and_effort",
]
