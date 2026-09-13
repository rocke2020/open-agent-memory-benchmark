from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

METRIC_IDS = (
    "answer_accuracy",
    "context_tokens",
    "indexing_tokens",
    "retrieval_latency",
    "indexing_time",
)


def _analysis_module() -> Any:
    try:
        return importlib.import_module("oamb.reporting.report_analysis")
    except ModuleNotFoundError:
        return None


def _report() -> dict[str, object]:
    cells = []
    providers = (
        ("hindsight", "hindsight-rest-v1"),
        ("mem0", "mem0-rest-v1"),
        ("openviking", "openviking-session-rest-v1"),
    )
    for index, (provider, profile) in enumerate(providers, start=1):
        cells.append(
            {
                "provider_id": provider,
                "adapter_profile_id": profile,
                "judged_numerator": 20 + index,
                "judged_denominator": 30,
                "accounting": {
                    "answer_visible_context_tokens": {
                        "status": "measured_complete",
                        "case_count": 30,
                        "measured_case_count": 30,
                        "total": 1000 * index,
                        "mean": str(100 * index),
                    },
                    "tokens": {
                        "indexing": {
                            "supplier_usage_coverage": {
                                "status": "measured_complete",
                                "record_count": 30,
                                "measured_record_count": 30,
                                "unavailable_record_count": 0,
                            },
                            "totals": {
                                "supplier_reported_total_tokens": {
                                    "status": "measured_complete",
                                    "value": 10_000 * index,
                                }
                            },
                        }
                    },
                },
                "observed_time": {
                    "comparability": "observed_only",
                    "provider_request": {
                        "status": "measured",
                        "count": 30,
                        "median_microseconds": str(100_000 * index),
                        "p95_microseconds": 200_000 * index,
                        "maximum_microseconds": 300_000 * index,
                    },
                    "indexing_ready": {
                        "status": "measured",
                        "count": 30,
                        "median_microseconds": str(1_000_000 * index),
                        "p95_microseconds": 2_000_000 * index,
                        "maximum_microseconds": 3_000_000 * index,
                    },
                },
                "limitations": (),
            }
        )
    return {
        "schema_name": "comparison_project_report",
        "schema_version": 1,
        "report_id": "a" * 64,
        "coverage": {
            "cell_count": 3,
            "unique_case_count": 30,
            "provider_specific_result_count": 90,
        },
        "cells": tuple(cells),
        "comparisons": (),
        "accuracy_decision": {"status": "no_clear_accuracy_leader"},
        "retrieval": (
            {"provider_id": "hindsight", "route": "recall"},
            {"provider_id": "mem0", "route": "/search"},
            {"provider_id": "openviking", "route": "/api/v1/search/find"},
        ),
        "questions": ({"private_question": "must-not-leave-the-report"},),
        "limitations": ("observed time is local run evidence",),
    }


def _valid_analysis_text() -> str:
    return json.dumps(
        {
            "overall": "No provider leads all five metrics in this observed run.",
            "metrics": {
                metric_id: f"Concise evidence-bounded comparison for {metric_id}."
                for metric_id in METRIC_IDS
            },
        }
    )


def _response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {"content": content, "reasoning_content": "internal"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        },
    )


def test_analysis_uses_six_total_attempts_then_seals_and_reuses_the_cache(
    tmp_path: Path,
) -> None:
    """Catches a retry bound below six or a matching cache that still dispatches."""

    module = _analysis_module()
    assert module is not None, "report analysis module is not implemented"
    request_bodies: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        if len(request_bodies) < 6:
            return _response('{"overall":"missing the five metric analyses"}')
        return _response(_valid_analysis_text())

    result = module.generate_report_analysis(
        _report(),
        model="deepseek-v4-flash",
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert result.cache_hit is False
    assert len(request_bodies) == 6
    document = json.loads(result.analysis_path.read_bytes())
    assert document["attempt_count"] == 6
    assert tuple(item["metric_id"] for item in document["metrics"]) == METRIC_IDS
    assert len(tuple(result.cache_directory.glob("attempts/*-response.json"))) == 6
    prompt = json.dumps(request_bodies[0], sort_keys=True)
    assert "must-not-leave-the-report" not in prompt
    assert "secret-test-key" not in prompt
    assert request_bodies[0]["max_tokens"] == 2_048
    assert "median_seconds" in prompt
    assert "median_microseconds" not in prompt

    def forbidden_handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("a valid matching analysis cache must prevent another paid call")

    cached = module.generate_report_analysis(
        _report(),
        model="deepseek-v4-flash",
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        transport=httpx.MockTransport(forbidden_handler),
        sleep=lambda _seconds: None,
    )

    assert cached is not None
    assert cached.cache_hit is True
    assert cached.analysis_path.read_bytes() == result.analysis_path.read_bytes()


def test_analysis_retries_provider_specific_overview_and_raw_subsecond_units(
    tmp_path: Path,
) -> None:
    """Catches plausible prose contradicting metrics or exposing raw timing units."""

    module = _analysis_module()
    assert module is not None
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            document["overall"] = "Hindsight leads the observed providers overall."
        elif calls == 2:
            document["metrics"]["retrieval_latency"] = (
                "Mem0 retrieval latency was 121414 microseconds."
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model="deepseek-v4-flash",
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 3
    assert json.loads(result.analysis_path.read_bytes())["attempt_count"] == 3


@pytest.mark.parametrize("malformed_envelope", [False, True])
def test_analysis_exhaustion_preserves_receipts_without_publishing_analysis(
    tmp_path: Path,
    malformed_envelope: bool,
) -> None:
    """Catches malformed output aborting fallback, being published, or a seventh attempt."""

    module = _analysis_module()
    assert module is not None, "report analysis module is not implemented"
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if malformed_envelope:
            return httpx.Response(200, json={"choices": [None]})
        return _response("not-json")

    result = module.generate_report_analysis(
        _report(),
        model="deepseek-v4-flash",
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is None
    assert calls == 6
    assert not tuple((tmp_path / "cache").glob("*/report-analysis.json"))
    assert len(tuple((tmp_path / "cache").glob("*/attempts/*-response.json"))) == 6


def test_analysis_publishes_only_closed_usage_totals(tmp_path: Path) -> None:
    """Catches supplier-specific or secret usage fields leaking into the report sidecar."""

    module = _analysis_module()
    assert module is not None

    def handler(_request: httpx.Request) -> httpx.Response:
        response = _response(_valid_analysis_text())
        document = json.loads(response.content)
        document["usage"]["private_trace"] = "must-not-be-published"
        return httpx.Response(200, json=document)

    result = module.generate_report_analysis(
        _report(),
        model="deepseek-v4-flash",
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    document = json.loads(result.analysis_path.read_bytes())
    assert document["usage"] == {
        "status": "measured",
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    assert b"must-not-be-published" not in result.analysis_path.read_bytes()
    assert (
        b"must-not-be-published"
        in next(result.cache_directory.glob("attempts/*-response.json")).read_bytes()
    )
