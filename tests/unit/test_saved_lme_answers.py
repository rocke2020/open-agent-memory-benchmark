from __future__ import annotations

import asyncio
import importlib
import json
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any

import httpx
import pytest

from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import build_resolved_plan
from oamb.workloads.longmemeval import (
    LME60_EXPECTED_QUESTION_IDS,
    QUESTION_TYPES,
    LongMemEvalRow,
    render_lme_judge_prompt,
)
from tests.benchmark_configuration import MODEL_ENVIRONMENT


def _study() -> ModuleType:
    try:
        return importlib.import_module("scripts.evaluate_saved_lme_answers")
    except ModuleNotFoundError:
        pytest.fail("saved-answer study is not implemented", pytrace=False)


def _inputs(tmp_path: Path) -> tuple[Any, tuple[LongMemEvalRow, ...], Path, dict[str, str]]:
    plan = build_resolved_plan(
        load_benchmark_configuration(
            Path(__file__).resolve().parents[2] / "configs/benchmark.yml",
            model_environment=MODEL_ENVIRONMENT,
        )
    )
    rows = tuple(
        LongMemEvalRow(
            source_row_number_1_indexed=index + 1,
            question_id=question_id,
            question_type=QUESTION_TYPES[index // 10],
            question=f"Question {index}?",
            answer=f"Reference {index}",
            raw_question_timestamp="2024/02/29 (Thu) 23:07",
            canonical_question_timestamp="2024-02-29T23:07:00+00:00",
            answer_session_ids=(),
            message_has_answer_session_ids=(),
            sessions=(),
        )
        for index, question_id in enumerate(LME60_EXPECTED_QUESTION_IDS)
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "query_id": row.question_id,
                        "query": row.question,
                        "gold_answers": [row.answer],
                        "answer": f"Saved answer {index}",
                        "correct": index < 56,
                        "judge_reason": "HISTORICAL_JUDGE_SECRET",
                    }
                    for index, row in enumerate(rows)
                ]
            }
        )
    )
    return (
        plan,
        rows,
        baseline,
        {
            "LLM_URL_TYPE": "openai_chat",
            "LLM_BASE_URL": "https://judge.example/v1",
            "LLM_API_KEY": "test-key",
            **MODEL_ENVIRONMENT,
        },
    )


@pytest.mark.parametrize("problem", ["duplicate", "missing", "question", "reference", "empty"])
def test_rejects_baseline_mismatch_before_creating_study(tmp_path: Path, problem: str) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    document = json.loads(baseline.read_bytes())
    if problem == "duplicate":
        document["results"].append(document["results"][0])
    elif problem == "missing":
        document["results"].pop()
    elif problem == "question":
        document["results"][0]["query"] = "Wrong question"
    elif problem == "reference":
        document["results"][0]["gold_answers"] = ["Wrong reference"]
    else:
        document["results"][0]["answer"] = "  "
    baseline.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        module.prepare_study(plan, rows, baseline, environment)


def _response(text: str, *, finish: str = "stop") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "fixture-completion",
            "model": "supplier-model-metadata",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
        },
    )


@pytest.mark.asyncio
async def test_frozen_wire_valid_no_and_reuse_preserve_baseline_and_completed_bytes(
    tmp_path: Path,
) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    before = baseline.read_bytes()
    inputs = module.prepare_study(plan, rows, baseline, environment)
    payloads: list[dict[str, Any]] = []

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://judge.example/v1/chat/completions"
        payloads.append(json.loads(request.content))
        return _response("no")

    output = tmp_path / "study"
    summary = await module.run_study(inputs, output, transport=httpx.MockTransport(respond))
    assert summary["baseline_common_judge"] == {"correct": 0, "judged": 60, "total": 60}
    assert summary["historical_baseline"]["correct"] == 56
    assert len(payloads) == 60
    first_prompt = render_lme_judge_prompt(
        question_type=rows[0].question_type,
        question=rows[0].question,
        reference=str(rows[0].answer),
        model_response="Saved answer 0",
        unanswerable=rows[0].question_id.endswith("_abs"),
    ).canonical_bytes.decode()
    judge = next(role for role in plan.model_roles if role.role_id == "judge")
    assert payloads[0] == {
        "model": MODEL_ENVIRONMENT["LLM_LIGHT_MODEL"],
        "messages": [{"role": "user", "content": first_prompt}],
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "stop": None,
        "reasoning_effort": judge.thinking_effort,
    }
    assert "HISTORICAL_JUDGE_SECRET" not in json.dumps(payloads)
    assert summary["reserved_output_tokens_per_call"] == 1024
    files = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
    repeated = await module.run_study(inputs, output, transport=httpx.MockTransport(respond))
    assert repeated == summary
    assert len(payloads) == 60
    assert all(path.read_bytes() == content for path, content in files.items())
    assert baseline.read_bytes() == before
    assert len(list(output.glob("questions/*/source/raw/*.json.gz"))) > 0


@pytest.mark.asyncio
async def test_changed_input_rejected_before_dispatch(tmp_path: Path) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    inputs = module.prepare_study(plan, rows, baseline, environment)
    output = tmp_path / "study"
    await module.run_study(
        inputs, output, transport=httpx.MockTransport(lambda _: _response("yes"))
    )
    document = json.loads(baseline.read_bytes())
    document["results"][0]["answer"] = "Changed saved answer"
    baseline.write_text(json.dumps(document))
    changed = module.prepare_study(plan, rows, baseline, environment)

    def forbidden(_: httpx.Request) -> httpx.Response:
        pytest.fail("changed input dispatched")

    with pytest.raises(ValueError, match="identity"):
        await module.run_study(changed, output, transport=httpx.MockTransport(forbidden))


@pytest.mark.asyncio
async def test_malformed_or_length_limited_output_exhausts_as_unjudged(tmp_path: Path) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    inputs = module.prepare_study(plan, rows, baseline, environment)
    calls = 0

    def respond(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        text = json.loads(request.content)["messages"][0]["content"]
        if "Saved answer 0" in text:
            return _response("yes", finish="length")
        return _response("no")

    summary = await module.run_study(
        inputs, tmp_path / "study", transport=httpx.MockTransport(respond)
    )
    assert calls == 65
    assert summary["baseline_common_judge"] == {"correct": 0, "judged": 59, "total": 60}
    assert summary["decision"] == "incomplete"
    assert summary["unjudged_question_ids"] == [rows[0].question_id]


@pytest.mark.parametrize(
    ("corrected", "decision"),
    [(56, "aggregate_parity_or_better"), (54, "near_parity"), (53, "stop_and_reassess")],
)
def test_paired_threshold_uses_actual_common_judge_counts(
    tmp_path: Path, corrected: int, decision: str
) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    inputs = module.prepare_study(plan, rows, baseline, environment)
    baseline_verdicts = {row.question_id: index < 56 for index, row in enumerate(rows)}
    corrected_verdicts = {row.question_id: index < corrected for index, row in enumerate(rows)}
    summary = module.summarize_study(inputs, baseline_verdicts, corrected_verdicts)
    assert summary["decision"] == decision
    assert summary["minimum_accepted_correct"] == 54
    assert len(summary["categories"]) == 6
    assert summary["paired"]["baseline_only"] == 56 - corrected
    assert len(summary["paired_verdict_flips"]) == 56 - corrected
    corrected_verdicts.pop(rows[-1].question_id)
    assert (
        module.summarize_study(inputs, baseline_verdicts, corrected_verdicts)["decision"]
        == "incomplete"
    )


@pytest.mark.asyncio
async def test_two_questions_overlap_and_cancellation_settles_without_new_admission(
    tmp_path: Path,
) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    inputs = module.prepare_study(plan, rows, baseline, environment)
    admitted = 0
    overlapping = asyncio.Event()
    hold = asyncio.Event()

    async def respond(_: httpx.Request) -> httpx.Response:
        nonlocal admitted
        admitted += 1
        if admitted == 2:
            overlapping.set()
        await hold.wait()
        return _response("yes")

    output = tmp_path / "study"
    task = asyncio.create_task(
        module.run_study(inputs, output, transport=httpx.MockTransport(respond))
    )
    await asyncio.wait_for(overlapping.wait(), timeout=3)
    assert admitted == 2
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert admitted == 2
    assert not list(output.glob("questions/*/result.json"))
    diagnostics = {
        path: path.read_bytes() for path in output.glob("questions/*/attempts/*/unjudged.json")
    }
    saved = [json.loads(content) for content in diagnostics.values()]
    assert len(saved) == 2
    assert all(item["status"] == "unjudged" for item in saved)
    summary = await module.run_study(
        inputs, output, transport=httpx.MockTransport(lambda _: _response("yes"))
    )
    assert summary["baseline_common_judge"] == {"correct": 60, "judged": 60, "total": 60}
    assert all(path.read_bytes() == content for path, content in diagnostics.items())


def test_changed_selection_rejected(tmp_path: Path) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    wrong = (replace(rows[0], question_id="wrong-id"), *rows[1:])
    with pytest.raises(ValueError, match="selection"):
        module.prepare_study(plan, wrong, baseline, environment)


def test_numeric_reference_matches_original_type_then_renders_native_text(tmp_path: Path) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    rows = (replace(rows[0], answer=2), *rows[1:])
    document = json.loads(baseline.read_bytes())
    document["results"][0]["gold_answers"] = [2]
    baseline.write_text(json.dumps(document))
    inputs = module.prepare_study(plan, rows, baseline, environment)
    assert "Correct Answer: 2" in inputs.answers[0].messages[0][1]


@pytest.mark.asyncio
async def test_tampered_completed_verdict_rejected_before_any_new_dispatch(tmp_path: Path) -> None:
    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    inputs = module.prepare_study(plan, rows, baseline, environment)
    output = tmp_path / "study"
    await module.run_study(inputs, output, transport=httpx.MockTransport(lambda _: _response("no")))
    saved_path = output / "questions" / rows[-1].question_id / "result.json"
    saved = json.loads(saved_path.read_bytes())
    saved["correct"] = True
    saved_path.write_text(json.dumps(saved))

    def forbidden(_: httpx.Request) -> httpx.Response:
        pytest.fail("corrupt completed judgment dispatched")

    with pytest.raises(ValueError, match="verdict"):
        await module.run_study(inputs, output, transport=httpx.MockTransport(forbidden))


def test_native_comparison_requires_exact_adjacent_plan_and_typed_results(tmp_path: Path) -> None:
    from oamb.config.doctor import resolved_plan_bytes
    from tests.unit.test_question_results import _judged_result

    module = _study()
    plan, rows, baseline, environment = _inputs(tmp_path)
    inputs = module.prepare_study(plan, rows, baseline, environment)
    native = tmp_path / "native"
    (native / "results").mkdir(parents=True)
    plan_path = native / "resolved-plan.json"
    plan_path.write_bytes(resolved_plan_bytes(plan))
    results_path = native / "results/hindsight.json"
    results_path.write_text(json.dumps({rows[0].question_id: _judged_result(rows[0].question_id)}))
    assert module.load_corrected_verdicts(inputs, results_path) == {rows[0].question_id: True}
    plan_path.write_bytes(resolved_plan_bytes(plan) + b"\n")
    with pytest.raises(ValueError, match="frozen plan"):
        module.load_corrected_verdicts(inputs, results_path)
