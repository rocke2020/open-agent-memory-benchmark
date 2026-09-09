from __future__ import annotations

import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType

import pytest


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


QUESTION_IDS = tuple(_sha(f"question-{index}") for index in range(60))


def _question_results_module() -> ModuleType:
    try:
        return importlib.import_module("oamb.runtime.question_results")
    except ModuleNotFoundError:
        pytest.fail("oamb.runtime.question_results is not implemented")


def _native_results_module() -> ModuleType:
    try:
        return importlib.import_module("oamb.runtime.native_results")
    except ModuleNotFoundError:
        pytest.fail("oamb.runtime.native_results is not implemented")


def _measured(value: int) -> dict[str, object]:
    return {
        "schema_name": "result_measurement",
        "schema_version": 1,
        "status": "measured",
        "value": value,
    }


def _available(value: str) -> dict[str, object]:
    return {
        "schema_name": "result_text",
        "schema_version": 1,
        "status": "available",
        "value": value,
    }


def _unavailable_measurement(reason: str) -> dict[str, object]:
    return {
        "schema_name": "result_measurement",
        "schema_version": 1,
        "status": "unavailable",
        "reason": reason,
    }


def _unavailable_text(reason: str) -> dict[str, object]:
    return {
        "schema_name": "result_text",
        "schema_version": 1,
        "status": "unavailable",
        "reason": reason,
    }


def _judged_result(question_id: str = QUESTION_IDS[0]) -> dict[str, object]:
    return {
        "schema_name": "question_result",
        "schema_version": 1,
        "terminal_status": "judged",
        "question_id": question_id,
        "ingestion": {
            "schema_name": "result_ingestion",
            "schema_version": 1,
            "status": "sealed",
            "intended_source_count": 2,
            "accepted_source_count": 2,
            "skipped_source_count": 0,
            "partial": False,
            "indexing_tokens": _measured(17),
            "indexing_token_coverage": "measured_complete",
            "indexing_ready_latency_microseconds": _measured(101),
        },
        "retrieval": {
            "schema_name": "result_retrieval",
            "schema_version": 1,
            "status": "succeeded",
            "visible_context": _available("remembered context"),
            "visible_context_byte_count": _measured(18),
            "visible_context_token_count": _measured(2),
            "native_candidate_count": _measured(3),
            "visible_kept_count": _measured(2),
            "visible_dropped_count": _measured(1),
            "visible_truncated_count": _measured(0),
            "request_latency_microseconds": _measured(202),
            "generation_proof_disposition": "runtime_verified",
        },
        "answer": {
            "schema_name": "result_answer",
            "schema_version": 1,
            "status": "parsed",
            "parsed_answer": _available("the answer"),
            "protocol_disposition": "normal_stop",
        },
        "evaluation": {
            "schema_name": "result_evaluation",
            "schema_version": 1,
            "disposition": "judged",
            "metric_id": "lme-judged-accuracy-v1",
            "numerator": 1,
            "denominator": 1,
            "judge_decision": "yes",
        },
        "attempts": {
            "schema_name": "result_attempt_counts",
            "schema_version": 1,
            "ingestion": 1,
            "retrieval": 1,
            "answer": 1,
            "judge": 1,
            "final_failure_class": "none",
        },
    }


def _unjudged_result(question_id: str = QUESTION_IDS[1]) -> dict[str, object]:
    result = deepcopy(_judged_result(question_id))
    reason = "judge output remained malformed after correction"
    result.update(
        {
            "terminal_status": "unjudged",
            "failure_stage": "judge",
            "failure_kind": "output_contract_error",
            "failure_reason": reason,
            "evaluation": {
                "schema_name": "result_evaluation",
                "schema_version": 1,
                "disposition": "unjudged",
                "reason": reason,
            },
        }
    )
    attempts = result["attempts"]
    assert isinstance(attempts, dict)
    attempts["judge"] = 6
    attempts["final_failure_class"] = "output_contract_error"
    return result


def _answer_failed_result(question_id: str = QUESTION_IDS[2]) -> dict[str, object]:
    result = deepcopy(_judged_result(question_id))
    reason = "answer output remained malformed after correction"
    result.update(
        {
            "terminal_status": "answer_failed",
            "failure_stage": "answer",
            "failure_kind": "output_contract_error",
            "failure_reason": reason,
            "answer": {
                "schema_name": "result_answer",
                "schema_version": 1,
                "status": "failed",
                "parsed_answer": _unavailable_text(reason),
                "protocol_disposition": "output_contract_error",
            },
            "evaluation": {
                "schema_name": "result_evaluation",
                "schema_version": 1,
                "disposition": "not_run",
                "reason": "answer did not satisfy its output contract",
            },
        }
    )
    attempts = result["attempts"]
    assert isinstance(attempts, dict)
    attempts["answer"] = 6
    attempts["judge"] = 0
    attempts["final_failure_class"] = "output_contract_error"
    return result


def test_provider_result_file_is_only_a_question_mapping_and_selects_missing_ids(
    tmp_path: Path,
) -> None:
    module = _question_results_module()
    path = tmp_path / "hindsight.json"
    path.write_text(
        json.dumps({QUESTION_IDS[0]: _judged_result()}),
        encoding="utf-8",
    )

    results = module.load_question_results(path, ordered_question_ids=QUESTION_IDS)

    assert tuple(results) == (QUESTION_IDS[0],)
    assert results[QUESTION_IDS[0]].question_id == QUESTION_IDS[0]
    assert module.remaining_question_ids(results, QUESTION_IDS) == QUESTION_IDS[1:]


def test_result_reader_accepts_only_reportable_terminal_results_and_preserves_unavailable(
    tmp_path: Path,
) -> None:
    module = _question_results_module()
    judged = _judged_result()
    ingestion = judged["ingestion"]
    assert isinstance(ingestion, dict)
    ingestion["indexing_tokens"] = _unavailable_measurement("supplier usage was not reported")
    ingestion["indexing_token_coverage"] = "unavailable"
    path = tmp_path / "hindsight.json"
    path.write_text(
        json.dumps(
            {
                QUESTION_IDS[0]: judged,
                QUESTION_IDS[1]: _unjudged_result(),
                QUESTION_IDS[2]: _answer_failed_result(),
            }
        ),
        encoding="utf-8",
    )

    results = module.load_question_results(path, ordered_question_ids=QUESTION_IDS)

    assert tuple(result.terminal_status for result in results.values()) == (
        "judged",
        "unjudged",
        "answer_failed",
    )
    assert results[QUESTION_IDS[0]].ingestion.indexing_tokens.status == "unavailable"


@pytest.mark.parametrize("invalid_kind", ["malformed", "duplicate", "unknown", "key-mismatch"])
def test_result_reader_rejects_invalid_files_without_changing_bytes(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    module = _question_results_module()
    path = tmp_path / "hindsight.json"
    documents = {
        "malformed": b'{"question":',
        "duplicate": (
            "{"
            + json.dumps(QUESTION_IDS[0])
            + ":"
            + json.dumps(_judged_result())
            + ","
            + json.dumps(QUESTION_IDS[0])
            + ":"
            + json.dumps(_judged_result())
            + "}"
        ).encode(),
        "unknown": json.dumps({_sha("unknown"): _judged_result(_sha("unknown"))}).encode(),
        "key-mismatch": json.dumps({QUESTION_IDS[0]: _judged_result(QUESTION_IDS[1])}).encode(),
    }
    original = documents[invalid_kind]
    path.write_bytes(original)

    with pytest.raises(module.QuestionResultsError):
        module.load_question_results(path, ordered_question_ids=QUESTION_IDS)

    assert path.read_bytes() == original


def test_result_file_initialization_is_empty_and_create_only(tmp_path: Path) -> None:
    module = _question_results_module()
    path = tmp_path / "results" / "hindsight.json"

    module.initialize_question_results(path)

    assert path.read_bytes() == b"{}"
    assert module.load_question_results(path, ordered_question_ids=QUESTION_IDS) == {}
    with pytest.raises(FileExistsError):
        module.initialize_question_results(path)
    assert path.read_bytes() == b"{}"


def test_add_result_orders_by_manifest_and_never_overwrites_paid_result(tmp_path: Path) -> None:
    module = _question_results_module()
    path = tmp_path / "hindsight.json"
    module.initialize_question_results(path)
    first = module.JudgedQuestionResult.model_validate(_judged_result(QUESTION_IDS[0]))
    second = module.JudgedQuestionResult.model_validate(_judged_result(QUESTION_IDS[1]))

    module.add_question_result(path, second, ordered_question_ids=QUESTION_IDS)
    module.add_question_result(path, first, ordered_question_ids=QUESTION_IDS)

    loaded = module.load_question_results(path, ordered_question_ids=QUESTION_IDS)
    assert tuple(loaded) == QUESTION_IDS[:2]
    before_duplicate = path.read_bytes()
    with pytest.raises(module.QuestionResultsError, match="already exists"):
        module.add_question_result(path, first, ordered_question_ids=QUESTION_IDS)
    assert path.read_bytes() == before_duplicate


def test_two_resumes_reuse_the_first_new_result_without_duplicate_entries(
    tmp_path: Path,
) -> None:
    module = _question_results_module()
    path = tmp_path / "hindsight.json"
    module.initialize_question_results(path)
    first = module.JudgedQuestionResult.model_validate(_judged_result(QUESTION_IDS[0]))
    second = module.JudgedQuestionResult.model_validate(_judged_result(QUESTION_IDS[1]))
    module.add_question_result(path, first, ordered_question_ids=QUESTION_IDS)

    first_resume = module.load_question_results(path, ordered_question_ids=QUESTION_IDS)
    assert module.remaining_question_ids(first_resume, QUESTION_IDS) == QUESTION_IDS[1:]
    module.add_question_result(path, second, ordered_question_ids=QUESTION_IDS)

    second_resume = module.load_question_results(path, ordered_question_ids=QUESTION_IDS)
    assert tuple(second_resume) == QUESTION_IDS[:2]
    assert module.remaining_question_ids(second_resume, QUESTION_IDS) == QUESTION_IDS[2:]


def test_atomic_interruption_before_replace_retries_and_after_replace_skips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _question_results_module()
    path = tmp_path / "hindsight.json"
    module.initialize_question_results(path)
    first = module.JudgedQuestionResult.model_validate(_judged_result(QUESTION_IDS[0]))
    real_replace = module.atomic_replace_bytes

    def interrupted_before_replace(*args: object, **kwargs: object) -> object:
        def fail(boundary: object, _target: Path) -> None:
            if str(boundary) == "after_file_fsync":
                raise RuntimeError("stopped before replace")

        return real_replace(*args, **kwargs, fault_hook=fail)

    monkeypatch.setattr(module, "atomic_replace_bytes", interrupted_before_replace)
    with pytest.raises(RuntimeError, match="before replace"):
        module.add_question_result(path, first, ordered_question_ids=QUESTION_IDS)
    assert (
        module.remaining_question_ids(
            module.load_question_results(path, ordered_question_ids=QUESTION_IDS),
            QUESTION_IDS,
        )
        == QUESTION_IDS
    )

    def interrupted_after_replace(*args: object, **kwargs: object) -> object:
        def fail(boundary: object, _target: Path) -> None:
            if str(boundary) == "after_target_replace":
                raise RuntimeError("stopped after replace")

        return real_replace(*args, **kwargs, fault_hook=fail)

    monkeypatch.setattr(module, "atomic_replace_bytes", interrupted_after_replace)
    with pytest.raises(RuntimeError, match="after replace"):
        module.add_question_result(path, first, ordered_question_ids=QUESTION_IDS)
    assert (
        module.remaining_question_ids(
            module.load_question_results(path, ordered_question_ids=QUESTION_IDS),
            QUESTION_IDS,
        )
        == QUESTION_IDS[1:]
    )


def test_complete_results_require_exact_manifest_keys(tmp_path: Path) -> None:
    module = _question_results_module()
    path = tmp_path / "hindsight.json"
    path.write_text(
        json.dumps({question_id: _judged_result(question_id) for question_id in QUESTION_IDS}),
        encoding="utf-8",
    )
    complete = module.load_question_results(path, ordered_question_ids=QUESTION_IDS)

    assert module.require_complete_question_results(complete, QUESTION_IDS) is complete
    with pytest.raises(module.QuestionResultsError, match="requires all manifest questions"):
        module.require_complete_question_results(
            {key: value for index, (key, value) in enumerate(complete.items()) if index},
            QUESTION_IDS,
        )


def test_provider_result_path_is_fixed_to_provider_names(tmp_path: Path) -> None:
    module = _question_results_module()

    assert module.provider_result_path(tmp_path, "hindsight") == tmp_path / "hindsight.json"
    with pytest.raises(module.QuestionResultsError, match="unsupported result provider"):
        module.provider_result_path(tmp_path, "../hindsight")


def test_native_result_projection_counts_model_outer_attempts_not_transport_calls() -> None:
    module = _native_results_module()
    attempts = (
        {"ordinal": 1, "request_messages_sha256": _sha("first")},
        {"ordinal": 2, "request_messages_sha256": _sha("first")},
        {"ordinal": 3, "request_messages_sha256": _sha("second")},
    )

    assert module._model_outer_attempt_count(attempts, max_transport_retries=2) == 2
