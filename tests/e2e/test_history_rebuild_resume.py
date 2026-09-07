from __future__ import annotations

from pathlib import Path

import pytest

import oamb.runtime.native_run as native
from tests.e2e.test_history_rebuild_vertical_slice import _run


def test_interrupted_batch_retry_cannot_resume_as_a_fresh_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    with pytest.raises(native.NativeRunInterrupted):
        _run(
            runs,
            monkeypatch,
            run_id="batch-retry-predecessor",
            fail_first_history=True,
            stop_during_backoff=True,
        )
    predecessor = runs / "batch-retry-predecessor"

    with pytest.raises(ValueError):
        _run(
            runs,
            monkeypatch,
            run_id="batch-retry-successor",
            fail_first_history=False,
            recovery_parts=(predecessor,),
        )

    assert not (runs / "batch-retry-successor-requests.jsonl").exists()
