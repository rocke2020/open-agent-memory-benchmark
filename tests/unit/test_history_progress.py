from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from oamb.contracts.ids import ingestion_occurrence_id
from tests.contracts.test_history_rebuild_contracts import HASHES, _attempt

RUN_SCRIPT = Path(__file__).parents[2] / "run.sh"
RUN_ID = "capsule-run"
OCCURRENCE = ingestion_occurrence_id(RUN_ID, "hindsight", HASHES[0])


def _write(root: Path, folder: str, name: str, value: object) -> None:
    path = root / "source" / folder / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def _root(tmp_path: Path, *, carried: bool = False) -> Path:
    root = tmp_path / "capsules" / "hindsight"
    _write(root, "specs", "run-spec", {"run_id": RUN_ID, "memory_system_id": "hindsight"})
    for index in range(2):
        _write(root, "cases", str(index), {})
    if carried:
        (root / "source" / "parts" / "prior").mkdir(parents=True)
    return root


def _history(root: Path, *, status: str = "ready") -> None:
    record = _attempt(
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
def test_carried_capsule_counts_one_terminal_fresh_history(
    tmp_path: Path,
    resume: bool,
) -> None:
    root = _root(tmp_path, carried=True)
    _history(root)
    line = _snapshot(root, resume=resume)
    assert "history_rebuild_attempts=1" in line
    assert ("completed=2," if resume else "completed_questions=2 (2/60, 3%)") in line
    if resume:
        assert "question_progress=53% (32/60)" in line


@pytest.mark.parametrize("status", ("ready", "retryable_failed_settled"))
@pytest.mark.parametrize("resume", (False, True))
def test_fresh_history_status_is_reported_once(
    tmp_path: Path,
    status: str,
    resume: bool,
) -> None:
    root = _root(tmp_path, carried=True)
    _history(root, status=status)
    assert "history_rebuild_attempts=1" in _snapshot(root, resume=resume)


def test_embedded_predecessor_does_not_count_in_local_progress(tmp_path: Path) -> None:
    root = _root(tmp_path, carried=True)
    embedded = root / "source" / "parts" / "prior"
    _history(embedded)
    _write(embedded, "cases", "old-case", {})
    line = _snapshot(root)
    assert "history_rebuild_attempts=0" in line
    assert "completed_questions=2 (2/60, 3%)" in line
    _history(root)
    assert "history_rebuild_attempts=1" in _snapshot(root)


@pytest.mark.parametrize(
    "invalid",
    ("json", "foreign", "ordinal", "predecessor", "claim", "duplicate"),
)
def test_malformed_or_nonfresh_history_is_unavailable(
    tmp_path: Path,
    invalid: str,
) -> None:
    root = _root(tmp_path, carried=True)
    record = _attempt(
        run_id=RUN_ID,
        execution_run_id=RUN_ID,
        ingestion_occurrence_id=OCCURRENCE,
    ).model_dump(mode="json")
    document: object = record
    if invalid == "json":
        document = "{"
    elif invalid == "foreign":
        record["execution_run_id"] = "other-run"
    elif invalid == "ordinal":
        record["history_attempt_ordinal"] = 2
    elif invalid == "predecessor":
        record["previous_retry_event_id"] = HASHES[5]
    elif invalid == "claim":
        record["admission_claim_raw_ref"] = HASHES[6]
    _write(root, "history-attempts", record["history_attempt_id"], document)
    if invalid == "duplicate":
        duplicate = dict(record)
        duplicate["history_attempt_id"] = HASHES[10]
        duplicate["ingestion_occurrence_id"] = HASHES[11]
        _write(root, "history-attempts", HASHES[10], duplicate)
    assert "history_rebuild_attempts=unavailable" in _snapshot(root)
