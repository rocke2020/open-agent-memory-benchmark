from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from oamb.contracts.ids import canonical_sha256, ingestion_occurrence_id
from tests.contracts.test_history_rebuild_contracts import HASHES, _attempt

RUN_SCRIPT = Path(__file__).parents[2] / "run.sh"
RUN_ID = "capsule-run"
OCCURRENCE = ingestion_occurrence_id(RUN_ID, "hindsight", HASHES[0], history_attempt_ordinal=2)


def _write(root: Path, folder: str, name: str, value: object) -> None:
    path = root / "source" / folder / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "capsules" / "hindsight"
    _write(root, "specs", "run-spec", {"run_id": RUN_ID, "memory_system_id": "hindsight"})
    for index in range(2):
        _write(root, "cases", str(index), {})
    return root


def _event(**updates: object) -> dict[str, object]:
    return {
        "schema_name": "history_retry_event",
        "schema_version": 1,
        "history_retry_event_id": HASHES[5],
        "run_id": RUN_ID,
        "ingestion_plan_id": HASHES[0],
        "failed_history_attempt_id": HASHES[1],
        "failed_history_attempt_ordinal": 1,
        "retry_ordinal": 1,
        "retry_scheduled": True,
        "successor_ingestion_occurrence_id": OCCURRENCE,
        "successor_execution_run_id": RUN_ID,
        "successor_history_attempt_ordinal": 2,
        "retry_policy_hash": HASHES[2],
        "max_retries_per_operation": 2,
        "backoff_seconds": 1,
        "observed_at": "2026-09-06T00:00:00+00:00",
        **updates,
    }


def _intent(root: Path, *, stage: str = "scope_allocate", index: int = 1) -> None:
    identity = canonical_sha256(["progress-intent", stage, index])
    _write(
        root,
        "attempt-intents",
        identity,
        {
            "schema_name": "attempt_intent_record",
            "schema_version": 3,
            "attempt_id": identity,
            "scope_kind": "run",
            "scope_id": RUN_ID,
            "parent_kind": "ingestion_plan",
            "parent_id": OCCURRENCE,
            "stage": stage,
        },
    )


def _history(root: Path, *, status: str = "ready") -> None:
    record = _attempt(
        history_attempt_ordinal=2,
        run_id=RUN_ID,
        execution_run_id=RUN_ID,
        ingestion_occurrence_id=OCCURRENCE,
        status=status,
    )
    _write(root, "history-attempts", record.history_attempt_id, record.model_dump(mode="json"))


def _snapshot(root: Path, *, resume: bool = False) -> str:
    source = RUN_SCRIPT.read_text(encoding="utf-8")
    body = source.split("report_provider_progress() {\n", 1)[1].split("\nPY\n}\n", 1)[0] + "\nPY"
    command = "report_provider_progress() {\n" + body + '\n}\nreport_provider_progress "$@"'
    arguments = [str(root.parent), "60", "30", "true"]
    if resume:
        arguments.extend(("hindsight-lme60", "30", "30", "1"))
    result = subprocess.run(
        ["bash", "-c", command, "history-progress", *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return next(line for line in result.stdout.splitlines() if "hindsight" in line)


@pytest.mark.parametrize("resume", (False, True))
def test_fresh_progress_has_zero_rebuilds(tmp_path: Path, resume: bool) -> None:
    line = _snapshot(_root(tmp_path), resume=resume)
    assert "history_rebuild_attempts=0" in line


@pytest.mark.parametrize("resume", (False, True))
def test_scheduled_backoff_does_not_count_as_an_admitted_rebuild(
    tmp_path: Path,
    resume: bool,
) -> None:
    root = _root(tmp_path)
    _write(root, "history-retries", HASHES[5], _event())
    assert "history_rebuild_attempts=0" in _snapshot(root, resume=resume)


@pytest.mark.parametrize("resume", (False, True))
def test_scope_admission_counts_one_history_despite_multiple_source_attempts(
    tmp_path: Path,
    resume: bool,
) -> None:
    root = _root(tmp_path)
    _write(root, "history-retries", HASHES[5], _event())
    _intent(root)
    _intent(root, stage="memory_ingest", index=1)
    _intent(root, stage="memory_ingest", index=2)
    line = _snapshot(root, resume=resume)
    assert "history_rebuild_attempts=1" in line
    assert ("completed=2," if resume else "completed_questions=2 (2/60, 3%)") in line
    if resume:
        assert "question_progress=53% (32/60)" in line


@pytest.mark.parametrize("status", ("ready", "retryable_failed_settled"))
@pytest.mark.parametrize("resume", (False, True))
def test_terminal_history_is_retained_and_deduplicated_with_its_admission(
    tmp_path: Path,
    status: str,
    resume: bool,
) -> None:
    root = _root(tmp_path)
    _write(root, "history-retries", HASHES[5], _event())
    _intent(root)
    _history(root, status=status)
    assert "history_rebuild_attempts=1" in _snapshot(root, resume=resume)


def test_embedded_predecessor_does_not_count_in_local_progress(tmp_path: Path) -> None:
    root = _root(tmp_path)
    embedded = root / "source/parts/prior"
    _history(embedded)
    _write(embedded, "cases", "old-case", {})
    line = _snapshot(root)
    assert "history_rebuild_attempts=0" in line
    assert "completed_questions=2 (2/60, 3%)" in line
    _intent(root)
    assert "history_rebuild_attempts=unavailable" in _snapshot(root)
    _history(root)
    assert "history_rebuild_attempts=1" in _snapshot(root)


@pytest.mark.parametrize(
    "invalid", ("json", "missing", "bool", "retry_bool", "foreign", "duplicate")
)
def test_malformed_or_ambiguous_rebuild_evidence_is_unavailable(
    tmp_path: Path,
    invalid: str,
) -> None:
    root = _root(tmp_path)
    event: object = _event()
    if invalid == "json":
        event = "{"
    elif invalid == "missing":
        event = _event(successor_ingestion_occurrence_id=None)
    elif invalid == "bool":
        event = _event(successor_history_attempt_ordinal=True)
    elif invalid == "retry_bool":
        event = _event(retry_ordinal=True)
    elif invalid == "foreign":
        event = _event(run_id="other-run")
    _write(root, "history-retries", HASHES[5], event)
    _intent(root)
    if invalid == "duplicate":
        _intent(root, index=2)
    assert "history_rebuild_attempts=unavailable" in _snapshot(root)


def test_conflicting_history_ordinals_are_unavailable(tmp_path: Path) -> None:
    root = _root(tmp_path)
    _history(root)
    path = next((root / "source/history-attempts").glob("*.json"))
    record = json.loads(path.read_bytes())
    record["history_attempt_id"] = HASHES[10]
    record["ingestion_occurrence_id"] = HASHES[11]
    _write(root, "history-attempts", HASHES[10], record)
    assert "history_rebuild_attempts=unavailable" in _snapshot(root)
