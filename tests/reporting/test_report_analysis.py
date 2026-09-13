from __future__ import annotations

import importlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import httpx
import pytest

PROVIDER_IDS = ("provider-a", "provider-b", "provider-c")
PROVIDER_STAGE_ROLE_IDS = (
    "hindsight_extraction",
    "mem0_extraction",
    "openviking_semantic_understanding",
)
QUESTION_TYPES = (
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
)


def _analysis_module() -> Any:
    try:
        return importlib.import_module("oamb.reporting.report_analysis")
    except ModuleNotFoundError:
        return None


def _report() -> dict[str, object]:
    cells = []
    shared_top_k = len(PROVIDER_IDS) * len(QUESTION_TYPES)
    providers = (
        (PROVIDER_IDS[0], "profile-a"),
        (PROVIDER_IDS[1], "profile-b"),
        (PROVIDER_IDS[2], "profile-c"),
    )
    for index, (provider, profile) in enumerate(providers, start=1):
        cells.append(
            {
                "provider_id": provider,
                "adapter_profile_id": profile,
                "judged_numerator": 20 + index,
                "judged_denominator": 30,
                "accuracy": {
                    "by_question_type": tuple(
                        {
                            "question_type": question_type,
                            "numerator": 1 if index == 1 or ordinal % 2 == 0 else 0,
                            "denominator": 1,
                        }
                        for ordinal, question_type in enumerate(QUESTION_TYPES, start=1)
                    )
                },
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
        "models": (
            *tuple(
                {
                    "role_id": role_id,
                    "model": __name__,
                    "thinking_effort": "low",
                }
                for role_id in PROVIDER_STAGE_ROLE_IDS
            ),
            {
                "role_id": "answer",
                "model": __name__,
                "thinking_effort": "low",
            },
            {
                "role_id": "judge",
                "model": __name__,
                "thinking_effort": "high",
            },
        ),
        "saved_result_sources": tuple(
            {
                "provider_id": provider_id,
                "source_plan_retrieval_top_k": ("unavailable" if index == 0 else shared_top_k),
                "retrieval_top_k": shared_top_k,
                "retrieval_top_k_source": (
                    "saved_report_control" if index == 0 else "source_resolved_plan"
                ),
                "observed_native_candidate_count_max": shared_top_k - index,
            }
            for index, provider_id in enumerate(PROVIDER_IDS, start=1)
        ),
        "comparisons": (),
        "accuracy_decision": {"status": "no_clear_accuracy_leader"},
        "retrieval": (
            {"provider_id": PROVIDER_IDS[0], "route": "/route-a"},
            {"provider_id": PROVIDER_IDS[1], "route": "/route-b"},
            {"provider_id": PROVIDER_IDS[2], "route": "/route-c"},
        ),
        "questions": tuple(
            {
                "raw_question_id": f"question-{ordinal}",
                "question_type": question_type,
                "question": f"Question text for {question_type}?",
                "gold_answer": (
                    42
                    if question_type == "temporal-reasoning"
                    else f"Gold answer for {question_type}."
                ),
                "answer_sessions": ({"private_session": "must-not-leave-the-report"},),
                "provider_results": tuple(
                    {
                        "provider_id": provider_id,
                        "display_state": (
                            "correct" if provider_index == 0 or ordinal % 2 == 0 else "incorrect"
                        ),
                        "model_answer": f"{provider_id} answer for {question_type}.",
                        "judge_decision": (
                            "yes" if provider_index == 0 or ordinal % 2 == 0 else "no"
                        ),
                        "injected_context": "must-not-leave-the-report",
                    }
                    for provider_index, provider_id in enumerate(PROVIDER_IDS)
                ),
            }
            for ordinal, question_type in enumerate(QUESTION_TYPES, start=1)
        ),
        "limitations": ("observed time is local run evidence",),
    }


def _valid_analysis_text(report: dict[str, object] | None = None) -> str:
    report = _report() if report is None else report
    cells = cast(tuple[dict[str, Any], ...], report["cells"])
    provider_ids = tuple(cell["provider_id"] for cell in cells)
    leader = max(
        cells,
        key=lambda cell: Fraction(cell["judged_numerator"], cell["judged_denominator"]),
    )
    context_leader = min(
        cells,
        key=lambda cell: int(cell["accounting"]["answer_visible_context_tokens"]["mean"]),
    )
    context_reductions = " and ".join(
        f"{round((1 - int(context_leader['accounting']['answer_visible_context_tokens']['mean']) / int(cell['accounting']['answer_visible_context_tokens']['mean'])) * 100)}% less context than {cell['provider_id']}"
        for cell in cells
        if cell is not context_leader
    )
    models = cast(tuple[dict[str, Any], ...], report["models"])
    provider_stage_bindings = tuple(
        (item["model"], item["thinking_effort"])
        for item in models
        if item["role_id"] in PROVIDER_STAGE_ROLE_IDS
    )
    opening = (
        "With the same models and thinking effort at each corresponding stage across providers,"
        if len(provider_stage_bindings) == len(PROVIDER_STAGE_ROLE_IDS)
        and len(set(provider_stage_bindings)) == 1
        else "Across the evaluated providers,"
    )

    def accuracy_sentence(cell: dict[str, object]) -> str:
        accuracy = cast(dict[str, Any], cell["accuracy"])
        records = cast(tuple[dict[str, Any], ...], accuracy["by_question_type"])
        ordered = sorted(
            records,
            key=lambda item: Fraction(item["numerator"], item["denominator"]),
            reverse=True,
        )
        strongest = ordered[0]
        weakest = ordered[-1]
        if strongest["question_type"] == weakest["question_type"]:
            weakest = ordered[1]
        no_errors = all(item["numerator"] == item["denominator"] for item in records)
        pattern = (
            "with no category error in the available cases"
            if no_errors
            else "with misses concentrated in the weaker category"
        )
        return (
            f"{cell['provider_id']} records {strongest['numerator']}/{strongest['denominator']} "
            f"on {strongest['question_type']} and {weakest['numerator']}/{weakest['denominator']} "
            f"on {weakest['question_type']}, {pattern}."
        )

    operational_sentences = (
        f"Compared with {provider_ids[1]}, its indexing-token total differs.",
        f"Compared with {provider_ids[2]}, its visible context tokens and retrieval latency differ.",
        f"Compared with {provider_ids[0]} and {provider_ids[1]}, its retrieval latency and "
        "indexing time differ.",
    )
    return json.dumps(
        {
            "overall": (
                f"{opening} {leader['provider_id']} has the highest recorded accuracy at "
                f"{leader['judged_numerator']}/{leader['judged_denominator']}, while "
                f"{context_leader['provider_id']} uses {context_reductions}."
            ),
            "provider_insights": [
                {
                    "provider_id": provider_id,
                    "insight": f"{accuracy_sentence(cell)} {operational}",
                }
                for provider_id, cell, operational in zip(
                    provider_ids,
                    cells,
                    operational_sentences,
                    strict=True,
                )
            ],
        }
    )


def _investigation_root(tmp_path: Path) -> Path:
    root = tmp_path / "investigations"
    root.mkdir(exist_ok=True)
    (root / "README.md").write_text("# Investigation index\n", encoding="utf-8")
    (root / "provider-study.md").write_text(
        "Provider C retained temporal facts but lost exact assistant-message details.\n",
        encoding="utf-8",
    )
    return root


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
    investigation_root = _investigation_root(tmp_path)

    def handler(request: httpx.Request) -> httpx.Response:
        request_bodies.append(json.loads(request.content))
        if len(request_bodies) < 6:
            return _response('{"overall":"missing the five metric analyses"}')
        return _response(_valid_analysis_text())

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert result.cache_hit is False
    assert len(request_bodies) == 6
    document = json.loads(result.analysis_path.read_bytes())
    assert document["attempt_count"] == 6
    assert tuple(item["provider_id"] for item in document["provider_insights"]) == PROVIDER_IDS
    assert len(tuple(result.cache_directory.glob("attempts/*-response.json"))) == 6
    assert request_bodies[0]["max_tokens"] == 8_192
    messages = request_bodies[0]["messages"]
    assert isinstance(messages, list)
    investigation_prompt = json.loads(messages[1]["content"])
    evaluation_prompt = json.loads(messages[2]["content"])
    assert [item["relative_path"] for item in investigation_prompt["investigations"]] == [
        "README.md",
        "provider-study.md",
    ]
    assert (
        "lost exact assistant-message details"
        in investigation_prompt["investigations"][1]["content"]
    )
    questions = evaluation_prompt["evidence"]["questions"]
    assert len(questions) == 6
    assert {item["question_type"] for item in questions} == set(QUESTION_TYPES)
    assert all(len(item["provider_results"]) == 3 for item in questions)
    prompt = json.dumps(request_bodies[0], sort_keys=True)
    assert "Question text for single-session-assistant?" in prompt
    assert "Gold answer for single-session-assistant." in prompt
    assert f"{PROVIDER_IDS[-1]} answer for single-session-assistant." in prompt
    assert "private_session" not in prompt
    assert "injected_context" not in prompt
    assert "must-not-leave-the-report" not in prompt
    assert "secret-test-key" not in prompt
    assert "median_seconds" in prompt
    assert "median_microseconds" not in prompt
    retrieval_controls = evaluation_prompt["evidence"]["retrieval_controls"]
    saved_sources = cast(tuple[dict[str, Any], ...], _report()["saved_result_sources"])
    assert retrieval_controls["shared_top_k"] == saved_sources[0]["retrieval_top_k"]
    assert retrieval_controls["all_providers_share_top_k"] is True
    assert retrieval_controls["all_observed_candidates_within_top_k"] is True
    assert tuple(item["top_k_source"] for item in retrieval_controls["providers"]) == tuple(
        item["retrieval_top_k_source"] for item in saved_sources
    )
    priorities = {
        item["metric_id"]: item["importance"]
        for item in evaluation_prompt["evidence"]["metric_priorities"]
    }
    assert priorities["answer_accuracy"] == priorities["context_tokens"] == "co_primary"
    assert priorities["indexing_tokens"] == priorities["retrieval_latency"] == "important"
    assert (
        evaluation_prompt["evidence"]["model_effort_controls"]["status"]
        == "shared_across_providers"
    )

    def forbidden_handler(_request: httpx.Request) -> httpx.Response:
        raise AssertionError("a valid matching analysis cache must prevent another paid call")

    cached = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(forbidden_handler),
        sleep=lambda _seconds: None,
    )

    assert cached is not None
    assert cached.cache_hit is True
    assert cached.analysis_path.read_bytes() == result.analysis_path.read_bytes()


def test_analysis_retries_accuracy_only_overall_summary(tmp_path: Path) -> None:
    module = _analysis_module()
    assert module is not None
    report = _report()
    cells = cast(tuple[dict[str, Any], ...], report["cells"])
    leader = max(
        cells,
        key=lambda cell: Fraction(cell["judged_numerator"], cell["judged_denominator"]),
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text(report))
        if calls == 1:
            document["overall"] = (
                f"{leader['provider_id']} has the highest recorded accuracy at "
                f"{leader['judged_numerator']}/{leader['judged_denominator']}."
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        report,
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=_investigation_root(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_summary_without_shared_model_effort_basis(
    tmp_path: Path,
) -> None:
    module = _analysis_module()
    assert module is not None
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            document["overall"] = document["overall"].replace(
                "With the same models and thinking effort at each corresponding stage across providers,",
                "Under comparable conditions,",
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=_investigation_root(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_uses_neutral_opening_when_provider_stage_effort_differs(
    tmp_path: Path,
) -> None:
    module = _analysis_module()
    assert module is not None
    report = _report()
    models = [dict(item) for item in cast(tuple[dict[str, Any], ...], report["models"])]
    models[1]["thinking_effort"] = "high"
    report["models"] = tuple(models)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(_valid_analysis_text(report))

    result = module.generate_report_analysis(
        report,
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=_investigation_root(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 1
    analysis = json.loads(result.analysis_path.read_bytes())
    assert analysis["overall"].startswith("Across the evaluated providers,")
    assert "same models" not in analysis["overall"]


def test_analysis_retries_malformed_provider_identity(tmp_path: Path) -> None:
    module = _analysis_module()
    assert module is not None
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            document["provider_insights"][0]["provider_id"] = None
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=_investigation_root(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_indexing_time_comparison_that_ignores_the_largest_gap(
    tmp_path: Path,
) -> None:
    module = _analysis_module()
    assert module is not None
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            document["provider_insights"][0]["insight"] = document["provider_insights"][0][
                "insight"
            ].replace("indexing-token total", "indexing-token total and indexing time")
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=_investigation_root(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_aligned_top_k_in_visible_summary(tmp_path: Path) -> None:
    module = _analysis_module()
    assert module is not None
    report = _report()
    saved_sources = cast(tuple[dict[str, Any], ...], report["saved_result_sources"])
    shared_top_k = saved_sources[0]["retrieval_top_k"]
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text(report))
        if calls == 1:
            document["overall"] = document["overall"].replace(
                "With the same models",
                f"Under a shared {shared_top_k}-candidate ceiling, with the same models",
                1,
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        report,
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=_investigation_root(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_shallow_or_hard_to_read_insights(
    tmp_path: Path,
) -> None:
    """Catches prose without category evidence or with hard-to-scan numbers."""

    module = _analysis_module()
    assert module is not None
    calls = 0
    investigation_root = _investigation_root(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            document["overall"] = "Insufficient evidence."
        elif calls == 2:
            document["provider_insights"][0]["insight"] = "Insufficient evidence."
        elif calls == 3:
            document["provider_insights"][2]["insight"] = (
                document["provider_insights"][2]["insight"].rsplit(".", 1)[0] + " at 12345 tokens."
            )
        elif calls == 4:
            document["provider_insights"][0]["insight"] = document["provider_insights"][0][
                "insight"
            ].replace(".", " at lower cost.", 1)
        elif calls == 5:
            first_sentence, second_sentence = document["provider_insights"][2]["insight"].split(
                ". ", 1
            )
            document["provider_insights"][2]["insight"] = (
                f"{first_sentence}. Official benchmark comparison: {second_sentence}"
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 6
    assert json.loads(result.analysis_path.read_bytes())["attempt_count"] == 6


def test_analysis_retries_overall_fraction_repeated_inside_provider_insight(
    tmp_path: Path,
) -> None:
    module = _analysis_module()
    assert module is not None
    report = _report()
    cells = cast(tuple[dict[str, Any], ...], report["cells"])
    leader = max(
        cells,
        key=lambda cell: Fraction(cell["judged_numerator"], cell["judged_denominator"]),
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text(report))
        if calls == 1:
            insight = document["provider_insights"][0]["insight"]
            document["provider_insights"][0]["insight"] = insight.replace(
                ", with",
                (
                    f", with an overall result of {leader['judged_numerator']}/"
                    f"{leader['judged_denominator']} and"
                ),
                1,
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        report,
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=_investigation_root(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_nonleading_overall_result_in_headline(tmp_path: Path) -> None:
    module = _analysis_module()
    assert module is not None
    report = _report()
    cells = cast(tuple[dict[str, Any], ...], report["cells"])
    nonleader = min(
        cells,
        key=lambda cell: Fraction(cell["judged_numerator"], cell["judged_denominator"]),
    )
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text(report))
        if calls == 1:
            document["overall"] = (
                document["overall"].removesuffix(".")
                + f", while {nonleader['provider_id']} recorded "
                + f"{nonleader['judged_numerator']}/{nonleader['judged_denominator']}."
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        report,
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=_investigation_root(tmp_path),
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_evaluation_process_commentary(tmp_path: Path) -> None:
    module = _analysis_module()
    assert module is not None
    calls = 0
    investigation_root = _investigation_root(tmp_path)
    report = _report()

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text(report))
        if calls == 1:
            document["overall"] = document["overall"].replace(
                "With the same models",
                "Although the evaluation configuration may be unreliable, with the same models",
                1,
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        report,
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_accepts_one_decision_relevant_type_and_metric_synonyms(
    tmp_path: Path,
) -> None:
    """Keeps the prose judge general instead of enforcing one wording template."""

    module = _analysis_module()
    assert module is not None
    report = _report()
    cells = cast(tuple[dict[str, Any], ...], report["cells"])
    cell = cells[2]
    record = cell["accuracy"]["by_question_type"][0]
    investigation_root = _investigation_root(tmp_path)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text(report))
        operational = document["provider_insights"][2]["insight"].split(". ", 1)[1]
        document["provider_insights"][2]["insight"] = (
            f"{cell['provider_id']} is weakest at {record['numerator']}/{record['denominator']} "
            f"on {record['question_type']}. {operational}"
        )
        document["provider_insights"][1]["insight"] = document["provider_insights"][1][
            "insight"
        ].replace("visible context tokens", "visible context load")
        document["provider_insights"][2]["insight"] = document["provider_insights"][2][
            "insight"
        ].replace("retrieval latency", "median retrieval")
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        report,
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 1


def test_analysis_retries_unsupported_retrieval_context_claim(tmp_path: Path) -> None:
    """Catches attributing an answer miss to context the analysis input does not contain."""

    module = _analysis_module()
    assert module is not None
    calls = 0
    investigation_root = _investigation_root(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            first_sentence, second_sentence = document["provider_insights"][1]["insight"].split(
                ". ", 1
            )
            document["provider_insights"][1]["insight"] = (
                f"{first_sentence}. From retrieved context, {second_sentence}"
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_insight_that_duplicates_all_category_scores(tmp_path: Path) -> None:
    """Catches a first-glance insight repeating the complete category table."""

    module = _analysis_module()
    assert module is not None
    calls = 0
    investigation_root = _investigation_root(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            report = _report()
            cells = cast(tuple[dict[str, Any], ...], report["cells"])
            cell = cells[0]
            records = cell["accuracy"]["by_question_type"]
            scores = ", ".join(
                f"{record['numerator']}/{record['denominator']} on {record['question_type']}"
                for record in records
            )
            operational = document["provider_insights"][0]["insight"].split(". ", 1)[1]
            document["provider_insights"][0]["insight"] = (
                f"{cell['provider_id']} scores {scores}, with no category error. {operational}"
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_when_non_accuracy_metrics_are_missing(tmp_path: Path) -> None:
    """Catches insights that ignore the four operational decision metrics."""

    module = _analysis_module()
    assert module is not None
    calls = 0
    investigation_root = _investigation_root(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            first_sentence = document["provider_insights"][0]["insight"].split(". ", 1)[0]
            document["provider_insights"][0]["insight"] = (
                f"{first_sentence}. Compared with {PROVIDER_IDS[1]}, its operational profile "
                "differs."
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_retries_when_one_of_five_metrics_is_omitted(tmp_path: Path) -> None:
    module = _analysis_module()
    assert module is not None
    calls = 0
    investigation_root = _investigation_root(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            for item in document["provider_insights"]:
                item["insight"] = item["insight"].replace("indexing time", "retrieval latency")
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


@pytest.mark.parametrize("malformed_suffix", ("10 thousand", "10k"))
def test_analysis_retries_non_compact_large_number_suffix(
    tmp_path: Path,
    malformed_suffix: str,
) -> None:
    """Catches analysis prose using longer units than the compact report table."""

    module = _analysis_module()
    assert module is not None
    calls = 0
    investigation_root = _investigation_root(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        document = json.loads(_valid_analysis_text())
        if calls == 1:
            document["provider_insights"][0]["insight"] = (
                document["provider_insights"][0]["insight"].rsplit(".", 1)[0]
                + f" by {malformed_suffix}."
            )
        return _response(json.dumps(document))

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
        transport=httpx.MockTransport(handler),
        sleep=lambda _seconds: None,
    )

    assert result is not None
    assert calls == 2


def test_analysis_cache_invalidates_when_an_investigation_changes(tmp_path: Path) -> None:
    """Catches refreshed investigation findings silently reusing stale commentary."""

    module = _analysis_module()
    assert module is not None
    investigation_root = _investigation_root(tmp_path)
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return _response(_valid_analysis_text())

    common = {
        "model": __name__,
        "thinking_effort": "high",
        "base_url": "https://models.example/v1",
        "api_key": "secret-test-key",
        "cache_root": tmp_path / "cache",
        "timeout_seconds": 30,
        "investigation_root": investigation_root,
        "transport": httpx.MockTransport(handler),
        "sleep": lambda _seconds: None,
    }
    first = module.generate_report_analysis(_report(), **common)
    assert first is not None
    (investigation_root / "provider-study.md").write_text(
        "Updated evidence about exact assistant-message detail failures.\n",
        encoding="utf-8",
    )
    second = module.generate_report_analysis(_report(), **common)

    assert second is not None
    assert calls == 2
    assert first.cache_directory != second.cache_directory


@pytest.mark.parametrize("malformed_envelope", [False, True])
def test_analysis_exhaustion_preserves_receipts_without_publishing_analysis(
    tmp_path: Path,
    malformed_envelope: bool,
) -> None:
    """Catches malformed output aborting fallback, being published, or a seventh attempt."""

    module = _analysis_module()
    assert module is not None, "report analysis module is not implemented"
    calls = 0
    investigation_root = _investigation_root(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if malformed_envelope:
            return httpx.Response(200, json={"choices": [None]})
        return _response("not-json")

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
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
    investigation_root = _investigation_root(tmp_path)

    def handler(_request: httpx.Request) -> httpx.Response:
        response = _response(_valid_analysis_text())
        document = json.loads(response.content)
        document["usage"]["private_trace"] = "must-not-be-published"
        return httpx.Response(200, json=document)

    result = module.generate_report_analysis(
        _report(),
        model=__name__,
        thinking_effort="high",
        base_url="https://models.example/v1",
        api_key="secret-test-key",
        cache_root=tmp_path / "cache",
        timeout_seconds=30,
        investigation_root=investigation_root,
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
