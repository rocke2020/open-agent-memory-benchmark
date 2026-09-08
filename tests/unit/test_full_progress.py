from __future__ import annotations

import hashlib
import importlib
import json
import multiprocessing
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


QUESTION_IDS = tuple(_sha(f"question-{index}") for index in range(60))
PLAN_HASH = _sha("resolved-plan")
MANIFEST_HASH = _sha("case-manifest")


def _try_full_resume_lock(path: str, queue: Any) -> None:
    module = _full_progress_module()
    try:
        with module.acquire_full_resume_lock(Path(path)):
            queue.put("acquired")
    except module.FullProgressError:
        queue.put("blocked")


def _full_progress_module() -> ModuleType:
    try:
        return importlib.import_module("oamb.runtime.full_progress")
    except ModuleNotFoundError:
        pytest.fail("oamb.runtime.full_progress is not implemented")


def _measured(value: int) -> dict[str, object]:
    return {
        "schema_name": "progress_measurement",
        "schema_version": 1,
        "status": "measured",
        "value": value,
    }


def _available(value: str) -> dict[str, object]:
    return {
        "schema_name": "progress_text",
        "schema_version": 1,
        "status": "available",
        "value": value,
    }


def _unavailable_text(reason: str) -> dict[str, object]:
    return {
        "schema_name": "progress_text",
        "schema_version": 1,
        "status": "unavailable",
        "reason": reason,
    }


def _unavailable(reason: str) -> dict[str, object]:
    return {
        "schema_name": "progress_measurement",
        "schema_version": 1,
        "status": "unavailable",
        "reason": reason,
    }


def _judged_entry(
    question_id: str = QUESTION_IDS[0],
    *,
    indexing_tokens: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_name": "full_progress_entry",
        "schema_version": 1,
        "terminal_status": "judged",
        "question_id": question_id,
        "ingestion": {
            "schema_name": "progress_ingestion",
            "schema_version": 1,
            "status": "sealed",
            "intended_source_count": 2,
            "accepted_source_count": 2,
            "skipped_source_count": 0,
            "partial": False,
            "indexing_tokens": indexing_tokens or _measured(17),
            "indexing_token_coverage": (
                "unavailable"
                if indexing_tokens is not None and indexing_tokens["status"] == "unavailable"
                else "measured_complete"
            ),
            "indexing_ready_latency_microseconds": _measured(101),
        },
        "retrieval": {
            "schema_name": "progress_retrieval",
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
            "schema_name": "progress_answer",
            "schema_version": 1,
            "status": "parsed",
            "parsed_answer": _available("the answer"),
            "protocol_disposition": "normal_stop",
        },
        "evaluation": {
            "schema_name": "progress_evaluation",
            "schema_version": 1,
            "disposition": "judged",
            "metric_id": "lme-judged-accuracy-v1",
            "numerator": 1,
            "denominator": 1,
            "judge_decision": "yes",
        },
        "attempts": {
            "schema_name": "progress_attempt_counts",
            "schema_version": 1,
            "ingestion": 1,
            "retrieval": 1,
            "answer": 1,
            "judge": 1,
            "final_failure_class": "none",
        },
    }


def _document(*results: dict[str, object]) -> dict[str, object]:
    return {
        "schema_name": "full_progress",
        "schema_version": 1,
        "resolved_plan_hash": PLAN_HASH,
        "cell_id": "hindsight-lme60",
        "provider_id": "hindsight",
        "workload_id": "longmemeval-v1",
        "case_manifest_hash": MANIFEST_HASH,
        "ordered_question_ids": list(QUESTION_IDS),
        "results": list(results),
    }


def _unjudged_entry(question_id: str = QUESTION_IDS[1]) -> dict[str, object]:
    entry = deepcopy(_judged_entry(question_id))
    entry.update(
        {
            "terminal_status": "unjudged",
            "failure_stage": "judge",
            "failure_kind": "output_contract_error",
            "failure_reason": "judge output remained malformed after correction",
            "evaluation": {
                "schema_name": "progress_evaluation",
                "schema_version": 1,
                "disposition": "unjudged",
                "reason": "judge output remained malformed after correction",
            },
        }
    )
    attempts = entry["attempts"]
    assert isinstance(attempts, dict)
    attempts["judge"] = 6
    attempts["final_failure_class"] = "output_contract_error"
    return entry


def _provider_failed_entry(question_id: str = QUESTION_IDS[2]) -> dict[str, object]:
    entry = deepcopy(_judged_entry(question_id))
    entry.update(
        {
            "terminal_status": "provider_failed",
            "failure_stage": "answer",
            "failure_kind": "output_contract_error",
            "failure_reason": "answer output remained malformed after correction",
            "answer": {
                "schema_name": "progress_answer",
                "schema_version": 1,
                "status": "failed",
                "parsed_answer": _unavailable_text(
                    "answer output remained malformed after correction"
                ),
                "protocol_disposition": "output_contract_error",
            },
            "evaluation": {
                "schema_name": "progress_evaluation",
                "schema_version": 1,
                "disposition": "not_run",
                "reason": "answer did not satisfy its output contract",
            },
        }
    )
    attempts = entry["attempts"]
    assert isinstance(attempts, dict)
    attempts["answer"] = 6
    attempts["judge"] = 0
    attempts["final_failure_class"] = "output_contract_error"
    return entry


def _load(module: ModuleType, path: Path) -> Any:
    return module.load_full_progress(
        path,
        expected_resolved_plan_hash=PLAN_HASH,
        expected_cell_id="hindsight-lme60",
        expected_provider_id="hindsight",
        expected_workload_id="longmemeval-v1",
        expected_case_manifest_hash=MANIFEST_HASH,
        expected_ordered_question_ids=QUESTION_IDS,
    )


def test_full_progress_loads_one_result_and_selects_remaining_in_manifest_order(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()
    path = tmp_path / "progress-hindsight.json"
    path.write_text(json.dumps(_document(_judged_entry())), encoding="utf-8")

    progress = _load(module, path)

    assert progress.completed_question_ids == (QUESTION_IDS[0],)
    assert progress.remaining_question_ids == QUESTION_IDS[1:]


def test_malformed_progress_fails_without_becoming_empty_or_changing_bytes(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()
    path = tmp_path / "progress-hindsight.json"
    malformed = b'{"schema_name":"full_progress","results":['
    path.write_bytes(malformed)

    with pytest.raises(module.FullProgressError, match="cannot parse"):
        _load(module, path)

    assert path.read_bytes() == malformed


def test_judged_progress_preserves_unavailable_indexing_tokens_instead_of_zero(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()
    path = tmp_path / "progress-mem0.json"
    path.write_text(
        json.dumps(
            {
                **_document(
                    _judged_entry(indexing_tokens=_unavailable("supplier usage was not reported"))
                ),
                "cell_id": "mem0-lme60",
                "provider_id": "mem0",
            }
        ),
        encoding="utf-8",
    )

    progress = module.load_full_progress(
        path,
        expected_resolved_plan_hash=PLAN_HASH,
        expected_cell_id="mem0-lme60",
        expected_provider_id="mem0",
        expected_workload_id="longmemeval-v1",
        expected_case_manifest_hash=MANIFEST_HASH,
        expected_ordered_question_ids=QUESTION_IDS,
    )

    assert progress.results[0].ingestion.indexing_tokens.status == "unavailable"
    assert progress.results[0].ingestion.indexing_tokens.reason == (
        "supplier usage was not reported"
    )


def test_progress_rejects_duplicate_json_keys_without_changing_bytes(tmp_path: Path) -> None:
    module = _full_progress_module()
    path = tmp_path / "progress-hindsight.json"
    raw = (
        json.dumps(_document(_judged_entry()))
        .replace(
            '"provider_id": "hindsight"',
            '"provider_id": "hindsight", "provider_id": "mem0"',
            1,
        )
        .encode("utf-8")
    )
    path.write_bytes(raw)

    with pytest.raises(module.FullProgressError, match="cannot parse"):
        _load(module, path)

    assert path.read_bytes() == raw


def test_resume_does_not_promote_a_stale_temporary_progress_file(tmp_path: Path) -> None:
    module = _full_progress_module()
    stale = tmp_path / ".progress-hindsight.json.tmp-stale"
    stale.write_text(json.dumps(_document(_judged_entry())), encoding="utf-8")
    canonical = tmp_path / "progress-hindsight.json"

    with pytest.raises(module.FullProgressError, match="cannot parse"):
        _load(module, canonical)

    assert not canonical.exists()
    assert stale.is_file()


@pytest.mark.parametrize(
    ("field_name", "changed", "message"),
    [
        ("resolved_plan_hash", _sha("other-plan"), "resolved_plan_hash"),
        ("cell_id", "other-cell", "cell_id"),
        ("provider_id", "mem0", "provider_id"),
        ("workload_id", "other-workload", "workload_id"),
        ("case_manifest_hash", _sha("other-manifest"), "case_manifest_hash"),
        ("ordered_question_ids", list(reversed(QUESTION_IDS)), "ordered_question_ids"),
    ],
)
def test_progress_rejects_runtime_identity_drift(
    tmp_path: Path,
    field_name: str,
    changed: object,
    message: str,
) -> None:
    module = _full_progress_module()
    path = tmp_path / "progress-hindsight.json"
    path.write_text(
        json.dumps({**_document(_judged_entry()), field_name: changed}),
        encoding="utf-8",
    )

    with pytest.raises(module.FullProgressError, match=message):
        _load(module, path)


@pytest.mark.parametrize("invalid_kind", ["duplicate", "unknown", "out-of-order"])
def test_progress_rejects_invalid_result_inventory(
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    module = _full_progress_module()
    second = _judged_entry(QUESTION_IDS[1])
    results = {
        "duplicate": (_judged_entry(), _judged_entry()),
        "unknown": (_judged_entry(_sha("unknown-question")),),
        "out-of-order": (second, _judged_entry()),
    }[invalid_kind]
    path = tmp_path / "progress-hindsight.json"
    path.write_text(json.dumps(_document(*results)), encoding="utf-8")

    with pytest.raises(module.FullProgressError, match="cannot parse"):
        _load(module, path)


def test_initialize_writes_one_empty_canonical_snapshot_and_never_overwrites_it(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()
    path = tmp_path / "results" / "progress-hindsight.json"
    progress = module.empty_full_progress(
        resolved_plan_hash=PLAN_HASH,
        cell_id="hindsight-lme60",
        provider_id="hindsight",
        workload_id="longmemeval-v1",
        case_manifest_hash=MANIFEST_HASH,
        ordered_question_ids=QUESTION_IDS,
    )

    module.initialize_full_progress(path, progress)

    expected = json.dumps(
        _document(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert path.read_bytes() == expected
    assert _load(module, path).remaining_question_ids == QUESTION_IDS
    with pytest.raises(FileExistsError):
        module.initialize_full_progress(path, progress)
    assert path.read_bytes() == expected


def test_provider_writer_merges_results_in_manifest_order_and_rejects_overwrite(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()
    path = tmp_path / "progress-hindsight.json"
    empty = module.empty_full_progress(
        resolved_plan_hash=PLAN_HASH,
        cell_id="hindsight-lme60",
        provider_id="hindsight",
        workload_id="longmemeval-v1",
        case_manifest_hash=MANIFEST_HASH,
        ordered_question_ids=QUESTION_IDS,
    )
    module.initialize_full_progress(path, empty)
    writer = module.ProviderProgressWriter(path, expected=empty)
    first = module.JudgedFullProgressEntry.model_validate(_judged_entry(QUESTION_IDS[0]))
    second = module.JudgedFullProgressEntry.model_validate(_judged_entry(QUESTION_IDS[1]))

    writer.publish(second)
    writer.publish(first)

    assert _load(module, path).completed_question_ids == QUESTION_IDS[:2]
    before_duplicate = path.read_bytes()
    with pytest.raises(module.FullProgressError, match="already complete"):
        writer.publish(first)
    assert path.read_bytes() == before_duplicate


def test_provider_writer_serializes_simultaneous_updates_without_losing_a_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _full_progress_module()
    path = tmp_path / "progress-hindsight.json"
    empty = module.empty_full_progress(
        resolved_plan_hash=PLAN_HASH,
        cell_id="hindsight-lme60",
        provider_id="hindsight",
        workload_id="longmemeval-v1",
        case_manifest_hash=MANIFEST_HASH,
        ordered_question_ids=QUESTION_IDS,
    )
    module.initialize_full_progress(path, empty)
    writer = module.ProviderProgressWriter(path, expected=empty)
    first = module.JudgedFullProgressEntry.model_validate(_judged_entry(QUESTION_IDS[0]))
    second = module.JudgedFullProgressEntry.model_validate(_judged_entry(QUESTION_IDS[1]))
    real_replace = module.atomic_replace_bytes
    first_replace_entered = threading.Event()
    release_first_replace = threading.Event()
    call_lock = threading.Lock()
    call_count = 0

    def delayed_first_replace(*args: object, **kwargs: object) -> object:
        nonlocal call_count
        with call_lock:
            call_count += 1
            current = call_count
        if current == 1:
            first_replace_entered.set()
            assert release_first_replace.wait(timeout=2)
        return real_replace(*args, **kwargs)

    monkeypatch.setattr(module, "atomic_replace_bytes", delayed_first_replace)
    second_done = threading.Event()

    def publish_second() -> object:
        try:
            return writer.publish(second)
        finally:
            second_done.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(writer.publish, first)
        assert first_replace_entered.wait(timeout=2)
        second_future = pool.submit(publish_second)
        time.sleep(0.05)
        release_first_replace.set()
        first_future.result(timeout=2)
        second_future.result(timeout=2)

    assert second_done.is_set()
    assert _load(module, path).completed_question_ids == QUESTION_IDS[:2]


def test_progress_accepts_only_the_two_bounded_model_failure_terminal_results(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()
    path = tmp_path / "progress-hindsight.json"
    path.write_text(
        json.dumps(_document(_judged_entry(), _unjudged_entry(), _provider_failed_entry())),
        encoding="utf-8",
    )

    progress = _load(module, path)

    assert tuple(item.terminal_status for item in progress.results) == (
        "judged",
        "unjudged",
        "provider_failed",
    )
    assert progress.completed_question_ids == QUESTION_IDS[:3]
    assert progress.remaining_question_ids == QUESTION_IDS[3:]


def test_final_progress_requires_the_exact_closed_sixty_result_inventory(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()
    incomplete_path = tmp_path / "incomplete.json"
    incomplete_path.write_text(json.dumps(_document(_judged_entry())), encoding="utf-8")
    incomplete = _load(module, incomplete_path)

    with pytest.raises(module.FullProgressError, match="requires 60 terminal results"):
        module.require_complete_full_progress(incomplete)

    complete_path = tmp_path / "complete.json"
    complete_path.write_text(
        json.dumps(_document(*(_judged_entry(question_id) for question_id in QUESTION_IDS))),
        encoding="utf-8",
    )
    complete = _load(module, complete_path)

    assert module.require_complete_full_progress(complete) is complete


def test_canonical_progress_path_is_fixed_to_the_three_provider_names(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()

    assert module.canonical_full_progress_path(tmp_path, "hindsight") == (
        tmp_path / "progress-hindsight.json"
    )
    with pytest.raises(module.FullProgressError, match="unsupported progress provider"):
        module.canonical_full_progress_path(tmp_path, "../hindsight")


def test_full_resume_process_lock_fails_fast_and_releases_with_its_owner(
    tmp_path: Path,
) -> None:
    module = _full_progress_module()
    context = multiprocessing.get_context("fork")
    lock_path = tmp_path / "full-test-resume.lock"

    with module.acquire_full_resume_lock(lock_path):
        queue = context.Queue()
        contender = context.Process(
            target=_try_full_resume_lock,
            args=(str(lock_path), queue),
        )
        contender.start()
        contender.join(timeout=2)
        assert contender.exitcode == 0
        assert queue.get(timeout=1) == "blocked"

    queue = context.Queue()
    successor = context.Process(
        target=_try_full_resume_lock,
        args=(str(lock_path), queue),
    )
    successor.start()
    successor.join(timeout=2)
    assert successor.exitcode == 0
    assert queue.get(timeout=1) == "acquired"
