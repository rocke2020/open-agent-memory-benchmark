"""Bounded, cached LLM analysis for the five comparison metrics."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.runtime.provider_env import load_t10_provider_environment

REPORT_ANALYSIS_NAME = "report-analysis.json"
REPORT_ANALYSIS_PROMPT_VERSION = "oamb-five-metric-comparison-v1"
REPORT_ANALYSIS_MAX_ATTEMPTS = 6
REPORT_ANALYSIS_MAX_OUTPUT_TOKENS = 1_024
REPORT_ANALYSIS_METRICS = (
    "answer_accuracy",
    "context_tokens",
    "indexing_tokens",
    "retrieval_latency",
    "indexing_time",
)
_REPORT_ANALYSIS_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0)

_METRIC_DEFINITIONS = {
    "answer_accuracy": "Judged answer accuracy with its actual denominator.",
    "context_tokens": "Exact retrieved context tokens shown to the answer model.",
    "indexing_tokens": "Supplier-reported successful logical indexing producer tokens.",
    "retrieval_latency": "Provider retrieval request wall time; lower is faster.",
    "indexing_time": "First provider write through terminal indexing readiness.",
}
_PROFILE_PATH_EVIDENCE = {
    "hindsight-rest-v1": (
        "Each ordered source dispatch uses one synchronous Hindsight retain request."
    ),
    "mem0-rest-v1": "Each ordered source dispatch uses one synchronous Mem0 add request.",
    "openviking-session-rest-v1": (
        "Each ordered source dispatch checks and creates a native session, uploads messages, "
        "commits it, polls its task to terminal state, reads the archive, and captures a "
        "projection."
    ),
}
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


def build_report_analysis_generator(
    *,
    plan: Any,
    model_env_path: Path,
    cache_root: Path,
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
    analysis_input = _analysis_input(report)
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
        parse_report_analysis(payload, report)
        return ReportAnalysisBuildResult(
            analysis_path=analysis_path,
            cache_directory=cache_directory,
            cache_hit=True,
        )

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": (
                "Compare the three memory providers using only supplied evidence. Return strict "
                "JSON with exactly keys overall and metrics. metrics must contain exactly, in "
                f"this order: {', '.join(REPORT_ANALYSIS_METRICS)}. overall is one concise "
                "sentence; every metric value is one concise sentence. Explain observed reasons "
                "only when supplied evidence supports them; otherwise say the reason is unknown."
            ),
        },
        {
            "role": "user",
            "content": canonical_json_bytes(
                {
                    "task": "Produce a concise cross-provider comparison of all five metrics.",
                    "evidence": analysis_input,
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
                overall, metrics = _parse_analysis_output(output)
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
                "metrics": tuple(
                    {"metric_id": metric_id, "analysis": metrics[metric_id]}
                    for metric_id in REPORT_ANALYSIS_METRICS
                ),
            }
            document = {
                **body,
                "analysis_id": canonical_sha256(["oamb-report-analysis-v1", body]),
            }
            payload = canonical_json_bytes(document)
            parse_report_analysis(payload, report)
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
        "metrics",
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
    expected_input_hash = hashlib.sha256(canonical_json_bytes(_analysis_input(report))).hexdigest()
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
    metrics_value = value["metrics"]
    if not isinstance(metrics_value, list) or len(metrics_value) != len(REPORT_ANALYSIS_METRICS):
        raise ValueError("report analysis metric inventory is incomplete")
    metric_ids: list[str] = []
    for item in metrics_value:
        if not isinstance(item, dict) or set(item) != {"metric_id", "analysis"}:
            raise ValueError("report analysis metric record is invalid")
        metric_ids.append(_required_text(item, "metric_id"))
        _concise_sentence(item["analysis"], "metric analysis", maximum_characters=320)
    if tuple(metric_ids) != REPORT_ANALYSIS_METRICS:
        raise ValueError("report analysis metric order or identity drifted")
    body = dict(value)
    analysis_id = body.pop("analysis_id")
    if analysis_id != canonical_sha256(["oamb-report-analysis-v1", body]):
        raise ValueError("report analysis identity does not close")
    value["overall"] = overall
    return cast(dict[str, object], value)


def _analysis_input(report: Mapping[str, object]) -> dict[str, object]:
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
        profile = _required_text(raw_cell, "adapter_profile_id")
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
                "retrieval_latency": observed_time.get(
                    "provider_request", {"status": "unavailable"}
                ),
                "indexing_time": observed_time.get("indexing_ready", {"status": "unavailable"}),
                "time_comparability": observed_time.get("comparability", "unavailable"),
                "ingestion_path_evidence": _PROFILE_PATH_EVIDENCE.get(
                    profile, "Provider ingestion path evidence is unavailable."
                ),
                "limitations": raw_cell.get("limitations", ()),
            }
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
        "metric_order": REPORT_ANALYSIS_METRICS,
        "metric_definitions": tuple(
            {"metric_id": metric_id, "definition": _METRIC_DEFINITIONS[metric_id]}
            for metric_id in REPORT_ANALYSIS_METRICS
        ),
        "coverage": report.get("coverage", {"status": "unavailable"}),
        "cells": tuple(cells),
        "accuracy_decision": report.get("accuracy_decision", {"status": "unavailable"}),
        "pairwise_accuracy": report.get("comparisons", ()),
        "retrieval_routes": retrieval_routes,
        "limitations": report.get("limitations", ()),
    }


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


def _parse_analysis_output(output: str) -> tuple[str, dict[str, str]]:
    try:
        value = json.loads(output, object_pairs_hook=_unique_object)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise _RetryableReportAnalysisError("analysis content is not strict JSON") from exc
    if not isinstance(value, dict) or set(value) != {"overall", "metrics"}:
        raise _RetryableReportAnalysisError("analysis content has the wrong fields")
    overall = _concise_sentence(value["overall"], "overall", maximum_characters=480)
    metrics = value["metrics"]
    if not isinstance(metrics, dict) or tuple(metrics) != REPORT_ANALYSIS_METRICS:
        raise _RetryableReportAnalysisError("analysis metric order or identity is invalid")
    return overall, {
        metric_id: _concise_sentence(
            metrics[metric_id],
            f"{metric_id} analysis",
            maximum_characters=320,
        )
        for metric_id in REPORT_ANALYSIS_METRICS
    }


def _concise_sentence(value: object, label: str, *, maximum_characters: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _RetryableReportAnalysisError(f"{label} must be non-empty text")
    normalized = value.strip()
    if "\n" in normalized or "\r" in normalized or len(normalized) > maximum_characters:
        raise _RetryableReportAnalysisError(f"{label} must be one concise sentence")
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
]
