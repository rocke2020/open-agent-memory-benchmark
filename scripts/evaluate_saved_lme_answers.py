#!/usr/bin/env python3
"""Judge saved LME-60 answers without running a memory provider or generating answers."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from oamb.artifacts.atomic import atomic_replace_bytes, atomic_write_bytes, read_regular_file
from oamb.artifacts.store import ArtifactStore
from oamb.config.doctor import (
    ModelExecutionBinding,
    ResolvedPlan,
    load_resolved_plan_for_run,
    resolved_plan_bytes,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import ModelCompletion, ModelReceipt, ModelRequest
from oamb.contracts.specifications import ModelRoleBindingV2
from oamb.live import _role_bindings
from oamb.model_clients.openai_compatible import OpenAICompatibleModelClient
from oamb.runtime.model_completion_retry import execute_model_completion_retry
from oamb.runtime.provider_env import load_t10_provider_environment
from oamb.runtime.question_results import load_question_results
from oamb.workloads.longmemeval import (
    LME60_EXPECTED_QUESTION_IDS,
    LME_JUDGE_OUTPUT_CONTRACT_ID,
    QUESTION_TYPES,
    LongMemEvalRow,
    build_longmemeval_bundle,
    render_lme_judge_prompt,
)
from oamb.workloads.metrics import (
    LME_JUDGE_MAX_OUTPUT_TOKENS,
    parse_lme_judge_completion,
    parse_lme_judge_yes_no,
)

QUESTION_COUNT = 60
STUDY_CONCURRENCY = 2
NEAR_PARITY_TOLERANCE = 2


@dataclass(frozen=True)
class SavedAnswer:
    question_id: str
    question_type: str
    messages: tuple[tuple[str, str], ...]
    historical_correct: bool


@dataclass(frozen=True)
class StudyInputs:
    plan: ResolvedPlan
    judge: ModelExecutionBinding
    binding: ModelRoleBindingV2
    environment: Mapping[str, str]
    answers: tuple[SavedAnswer, ...]
    identity: str


def _json_object(path: Path) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(read_regular_file(path), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("study input must be a JSON object")
    return value


def prepare_study(
    plan: ResolvedPlan,
    rows: tuple[LongMemEvalRow, ...],
    baseline_path: Path,
    environment: Mapping[str, str],
) -> StudyInputs:
    """Validate the complete selected input before constructing any model client."""
    if (
        plan.dataset.selection != "lme60"
        or tuple(row.question_id for row in rows) != LME60_EXPECTED_QUESTION_IDS
    ):
        raise ValueError("study requires the frozen ordered LME-60 selection")
    if any(sum(row.question_type == category for row in rows) != 10 for category in QUESTION_TYPES):
        raise ValueError("study selection must contain ten questions in each category")
    judge = next(role for role in plan.model_roles if role.role_id == "judge")
    if (
        judge.maximum_output_tokens_per_call is not None
        or judge.temperature != "0"
        or judge.top_p != "1"
        or environment.get("LLM_URL_TYPE") != "openai_chat"
        or not environment.get(judge.endpoint_variable)
        or not environment.get(judge.credential_variable)
    ):
        raise ValueError("frozen judge settings or model environment differ from the native path")
    binding = _role_bindings((judge,), environment)[0]
    baseline = _json_object(baseline_path)
    results = baseline.get("results")
    if not isinstance(results, list):
        raise ValueError("baseline results must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        if not isinstance(result, dict) or not isinstance(result.get("query_id"), str):
            raise ValueError("baseline result requires a question ID")
        question_id = result["query_id"]
        if question_id in by_id:
            raise ValueError("baseline question IDs must be unique")
        by_id[question_id] = result
    answers: list[SavedAnswer] = []
    for row in rows:
        result = by_id.get(row.question_id)
        reference = str(row.answer)
        if (
            result is None
            or result.get("query") != row.question
            or result.get("gold_answers") != [row.answer]
            or not isinstance(result.get("answer"), str)
            or not result["answer"].strip()
            or type(result.get("correct")) is not bool
        ):
            raise ValueError(
                "selected baseline IDs, questions, references, or answers do not match"
            )
        prompt = render_lme_judge_prompt(
            question_type=row.question_type,
            question=row.question,
            reference=reference,
            model_response=result["answer"],
            unanswerable=row.question_id.endswith("_abs"),
        )
        answers.append(
            SavedAnswer(
                row.question_id,
                row.question_type,
                (("user", prompt.canonical_bytes.decode("utf-8")),),
                result["correct"],
            )
        )
    identity = canonical_sha256(
        [
            "saved-lme-answers",
            hashlib.sha256(read_regular_file(baseline_path)).hexdigest(),
            plan.resolved_plan_hash,
            binding.model_dump(mode="json"),
            [asdict(answer) for answer in answers],
            LME_JUDGE_OUTPUT_CONTRACT_ID,
            LME_JUDGE_MAX_OUTPUT_TOKENS,
        ]
    )
    return StudyInputs(plan, judge, binding, dict(environment), tuple(answers), identity)


def load_study(
    *, baseline: Path, resolved_plan: Path, dataset: Path, model_env: Path
) -> StudyInputs:
    plan = load_resolved_plan_for_run(resolved_plan)
    bundle = build_longmemeval_bundle(dataset, plan.dataset.selection)
    if (
        bundle.case_manifest.manifest_hash != plan.dataset.case_manifest_hash
        or bundle.case_manifest.workload_id != plan.dataset.workload_id
        or hashlib.sha256(read_regular_file(dataset)).hexdigest() != plan.dataset.source_sha256
    ):
        raise ValueError("study dataset differs from the frozen resolved plan")
    judge = next(role for role in plan.model_roles if role.role_id == "judge")
    environment = load_t10_provider_environment(
        model_env,
        expected_keys=frozenset(
            {"LLM_URL_TYPE", judge.endpoint_variable, judge.credential_variable}
        ),
    )
    return prepare_study(plan, bundle.selected_rows, baseline, environment)


def load_corrected_verdicts(inputs: StudyInputs, results_path: Path) -> dict[str, bool | None]:
    """Read ordinary results only beside their exact native frozen-plan bytes."""
    result_plan_path = results_path.parent.parent / "resolved-plan.json"
    if read_regular_file(result_plan_path) != resolved_plan_bytes(inputs.plan):
        raise ValueError("corrected results use a different frozen plan")
    results = load_question_results(
        results_path, ordered_question_ids=tuple(answer.question_id for answer in inputs.answers)
    )
    for result in results.values():
        if result.answer.status == "parsed" and not result.answer.parsed_answer.value.strip():
            raise ValueError("corrected answer is empty")
    return {
        question_id: result.evaluation.judge_decision == "yes"
        if result.terminal_status == "judged"
        else None
        for question_id, result in results.items()
    }


def _saved_verdict(inputs: StudyInputs, answer: SavedAnswer, directory: Path) -> bool | None:
    saved = _json_object(directory / "result.json")
    if (
        saved.get("input_identity") != inputs.identity
        or saved.get("question_id") != answer.question_id
    ):
        raise ValueError("saved judgment input identity differs")
    if saved.get("status") == "unjudged" and saved.get("correct") is None:
        return None
    if saved.get("status") != "judged" or type(saved.get("correct")) is not bool:
        raise ValueError("saved judgment is malformed")
    raw_hash = saved.get("raw_response_sha256")
    if (
        not isinstance(raw_hash, str)
        or len(raw_hash) != 64
        or set(raw_hash) - set("0123456789abcdef")
    ):
        raise ValueError("saved verdict has an invalid raw response hash")
    raw = gzip.decompress(read_regular_file(directory / "source/raw" / f"{raw_hash}.json.gz"))
    if hashlib.sha256(raw).hexdigest() != raw_hash:
        raise ValueError("saved verdict raw response hash differs")
    candidates, disposition = OpenAICompatibleModelClient._parse_choices(json.loads(raw))
    correct = parse_lme_judge_completion(ModelCompletion(disposition, candidates))
    if correct != saved["correct"]:
        raise ValueError("saved verdict differs from its raw response")
    return correct


async def _judge_answer(
    inputs: StudyInputs,
    answer: SavedAnswer,
    directory: Path,
    *,
    transport: httpx.AsyncBaseTransport | None,
) -> bool | None:
    store = ArtifactStore(directory)
    result_path = directory / "result.json"
    run_id = uuid.uuid4().hex
    client = OpenAICompatibleModelClient(
        store=store,
        base_url=inputs.environment[inputs.judge.endpoint_variable],
        api_key=inputs.environment[inputs.judge.credential_variable],
        role_binding=inputs.binding,
        usage_profile="openai-details-v3",
        transport=transport,
        read_timeout_seconds=float(inputs.plan.execution.operation_timeout_seconds),
        total_timeout_seconds=float(inputs.plan.execution.operation_timeout_seconds),
    )
    requests: dict[int, ModelRequest] = {}

    async def dispatch(ordinal: int, messages: tuple[tuple[str, str], ...]) -> ModelReceipt:
        messages_raw = store.write_raw(
            canonical_json_bytes(messages), media_type="application/json"
        )
        request = ModelRequest.for_attempt(
            ordinal=ordinal,
            parent_kind="case",
            parent_id=canonical_sha256([inputs.identity, answer.question_id, run_id]),
            stage="judge",
            role_binding_id=inputs.binding.binding_id,
            messages_sha256=messages_raw.reference.sha256,
            messages=messages,
            thinking_effort=client.thinking_effort_for(
                stage="judge", role_binding_id=inputs.binding.binding_id
            ),
            output_contract_id=LME_JUDGE_OUTPUT_CONTRACT_ID,
            max_output_tokens=None,
            temperature=inputs.judge.temperature,
            top_p=inputs.judge.top_p,
        )
        requests[ordinal] = request
        return await client.complete(request)

    def validate(receipt: ModelReceipt) -> None:
        parse_lme_judge_completion(ModelCompletion(receipt.finish_disposition, receipt.candidates))

    async def record(
        ordinal: int, receipt: ModelReceipt | None, error: BaseException | None
    ) -> None:
        raw = (
            receipt.raw_reference if receipt is not None else getattr(error, "raw_reference", None)
        )
        atomic_write_bytes(
            directory / "attempts" / run_id / f"{ordinal}.json",
            canonical_json_bytes(
                {
                    "request": asdict(requests[ordinal]) if ordinal in requests else None,
                    "reserved_output_tokens": LME_JUDGE_MAX_OUTPUT_TOKENS,
                    "raw_response_sha256": raw.sha256 if raw is not None else None,
                    "usage_reference_ids": receipt.usage_reference_ids
                    if receipt is not None
                    else getattr(error, "usage_reference_ids", ()),
                    "error_kind": type(error).__name__ if error is not None else None,
                }
            ),
            trusted_root=directory,
        )

    correct: bool | None = None
    failure: BaseException | None = None
    receipt: ModelReceipt | None = None
    try:
        receipt = await execute_model_completion_retry(
            messages=answer.messages,
            dispatch=dispatch,
            validate=validate,
            record=record,
            max_outer_attempts=inputs.plan.execution.model_max_attempts,
            max_transport_retries=inputs.plan.execution.model_transport_max_retries,
        )
        correct = parse_lme_judge_yes_no(receipt.output_text)
    except BaseException as error:
        failure = error
    finally:
        await client.close()
    interrupted = isinstance(failure, (asyncio.CancelledError, KeyboardInterrupt, SystemExit))
    atomic_write_bytes(
        directory / "attempts" / run_id / "unjudged.json" if interrupted else result_path,
        canonical_json_bytes(
            {
                "input_identity": inputs.identity,
                "question_id": answer.question_id,
                "status": "judged" if correct is not None else "unjudged",
                "correct": correct,
                "raw_response_sha256": receipt.raw_reference.sha256
                if receipt is not None
                else None,
                "error_kind": type(failure).__name__ if failure is not None else None,
                "attempts": len(requests),
            }
        ),
        trusted_root=directory,
    )
    if interrupted:
        assert failure is not None
        raise failure
    return correct


async def run_study(
    inputs: StudyInputs,
    output_dir: Path,
    *,
    corrected_verdicts: Mapping[str, bool | None] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Run two independent question writers; stop admission and settle both on cancellation."""
    identity_path = output_dir / "input-identity.json"
    expected = {"input_identity": inputs.identity}
    if output_dir.exists() and not identity_path.exists():
        raise ValueError("study output must be fresh or have a matching input identity")
    if identity_path.exists() and _json_object(identity_path) != expected:
        raise ValueError("study input identity differs from the saved study")
    atomic_write_bytes(identity_path, canonical_json_bytes(expected), trusted_root=output_dir)
    atomic_write_bytes(
        output_dir / "resolved-plan.json", resolved_plan_bytes(inputs.plan), trusted_root=output_dir
    )
    verdicts: dict[str, bool | None] = {}
    for answer in inputs.answers:
        directory = output_dir / "questions" / answer.question_id
        if (directory / "result.json").exists():
            verdicts[answer.question_id] = _saved_verdict(inputs, answer, directory)
    pending = iter(answer for answer in inputs.answers if answer.question_id not in verdicts)

    async def worker() -> None:
        for answer in pending:
            verdicts[answer.question_id] = await _judge_answer(
                inputs, answer, output_dir / "questions" / answer.question_id, transport=transport
            )

    async with asyncio.TaskGroup() as group:
        for _ in range(
            min(STUDY_CONCURRENCY, inputs.plan.execution.max_parallel_questions_per_provider)
        ):
            group.create_task(worker())
    summary = summarize_study(inputs, verdicts, corrected_verdicts)
    atomic_replace_bytes(
        output_dir / "summary.json", canonical_json_bytes(summary), trusted_root=output_dir
    )
    return summary


def summarize_study(
    inputs: StudyInputs,
    baseline_verdicts: Mapping[str, bool | None],
    corrected_verdicts: Mapping[str, bool | None] | None = None,
) -> dict[str, Any]:
    corrected = corrected_verdicts or {}
    ids = tuple(answer.question_id for answer in inputs.answers)
    for verdicts in (baseline_verdicts, corrected):
        if set(verdicts) - set(ids) or any(
            value is not None and type(value) is not bool for value in verdicts.values()
        ):
            raise ValueError("paired verdicts differ from the selected questions")
    historical = {answer.question_id: answer.historical_correct for answer in inputs.answers}

    def counts(
        verdicts: Mapping[str, bool | None], selected: tuple[str, ...] = ids
    ) -> dict[str, int]:
        return {
            "correct": sum(verdicts.get(question_id) is True for question_id in selected),
            "judged": sum(type(verdicts.get(question_id)) is bool for question_id in selected),
            "total": len(selected),
        }

    baseline_counts, corrected_counts = counts(baseline_verdicts), counts(corrected)
    complete = baseline_counts["judged"] == corrected_counts["judged"] == QUESTION_COUNT
    minimum = baseline_counts["correct"] - NEAR_PARITY_TOLERANCE
    decision = "incomplete"
    if complete:
        decision = (
            "aggregate_parity_or_better"
            if corrected_counts["correct"] >= baseline_counts["correct"]
            else "near_parity"
            if corrected_counts["correct"] >= minimum
            else "stop_and_reassess"
        )
    paired_ids = tuple(
        question_id
        for question_id in ids
        if type(baseline_verdicts.get(question_id)) is bool
        and type(corrected.get(question_id)) is bool
    )
    return {
        "input_identity": inputs.identity,
        "historical_baseline": counts(historical),
        "baseline_common_judge": baseline_counts,
        "corrected_oamb": corrected_counts,
        "reserved_output_tokens_per_call": LME_JUDGE_MAX_OUTPUT_TOKENS,
        "decision": decision,
        "minimum_accepted_correct": minimum
        if baseline_counts["judged"] == QUESTION_COUNT
        else None,
        "historical_successes_retained": all(
            corrected.get(question_id) is True for question_id in ids if historical[question_id]
        )
        if corrected_counts["judged"] == QUESTION_COUNT
        else None,
        "categories": {
            category: {
                name: counts(
                    verdicts,
                    tuple(
                        answer.question_id
                        for answer in inputs.answers
                        if answer.question_type == category
                    ),
                )
                for name, verdicts in (
                    ("historical_baseline", historical),
                    ("baseline_common_judge", baseline_verdicts),
                    ("corrected_oamb", corrected),
                )
            }
            for category in QUESTION_TYPES
        },
        "paired": {
            name: sum(
                baseline_verdicts[question_id] is left and corrected[question_id] is right
                for question_id in paired_ids
            )
            for name, left, right in (
                ("both_correct", True, True),
                ("baseline_only", True, False),
                ("corrected_only", False, True),
                ("both_incorrect", False, False),
            )
        },
        "paired_verdict_flips": [
            question_id
            for question_id in paired_ids
            if baseline_verdicts[question_id] != corrected[question_id]
        ],
        "historical_baseline_verdict_flips": [
            question_id
            for question_id in ids
            if type(baseline_verdicts.get(question_id)) is bool
            and baseline_verdicts[question_id] != historical[question_id]
        ],
        "unjudged_question_ids": [
            question_id
            for question_id in ids
            if type(baseline_verdicts.get(question_id)) is not bool
        ],
        "corrected_unjudged_question_ids": [
            question_id for question_id in ids if type(corrected.get(question_id)) is not bool
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "resolved-plan", "dataset", "output-dir", "model-env"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--oamb-results",
        type=Path,
        help="Native results/<provider>.json beside ../resolved-plan.json",
    )
    args = parser.parse_args()
    inputs = load_study(
        baseline=args.baseline,
        resolved_plan=args.resolved_plan,
        dataset=args.dataset,
        model_env=args.model_env,
    )
    corrected = load_corrected_verdicts(inputs, args.oamb_results) if args.oamb_results else None
    summary = asyncio.run(run_study(inputs, args.output_dir, corrected_verdicts=corrected))
    print(json.dumps(summary, indent=2))
    return 0 if summary["decision"] in {"aggregate_parity_or_better", "near_parity"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
