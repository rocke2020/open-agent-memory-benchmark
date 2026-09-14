from __future__ import annotations

import signal
import time

import pytest

from oamb.contracts.ports import WorkerOutcome, WorkerReceipt, WorkerRequest
from oamb.runtime.worker import SupervisedWorker, WorkerRetiredError


def echo_handler(request: WorkerRequest) -> WorkerReceipt:
    return WorkerReceipt(
        request_id=request.request_id,
        outcome=WorkerOutcome.SUCCEEDED,
        raw_reference=None,
        canonical_bytes=request.canonical_bytes,
    )


def blocking_handler(request: WorkerRequest) -> WorkerReceipt:
    time.sleep(2)
    return echo_handler(request)


def sigterm_ignoring_handler(request: WorkerRequest) -> WorkerReceipt:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    time.sleep(2)
    return echo_handler(request)


def request(request_id: str) -> WorkerRequest:
    return WorkerRequest(
        request_id=request_id,
        operation="ingest",
        payload_sha256="a" * 64,
        canonical_bytes=b"payload",
    )


def test_supervised_worker_returns_typed_receipt_and_closes_cleanly() -> None:
    worker = SupervisedWorker(echo_handler, hard_timeout_seconds=5)

    receipt = worker.request(request("request-1"))
    worker.close()

    assert receipt.outcome is WorkerOutcome.SUCCEEDED
    assert receipt.canonical_bytes == b"payload"
    assert worker.is_alive is False


def test_hard_timeout_terminates_worker_and_permanently_retires_scope() -> None:
    worker = SupervisedWorker(blocking_handler, hard_timeout_seconds=0.05)

    receipt = worker.request(request("request-1"))

    assert receipt.outcome is WorkerOutcome.UNKNOWN_OUTCOME
    assert worker.is_alive is False
    with pytest.raises(WorkerRetiredError, match="retired"):
        worker.request(request("request-2"))
    worker.close()


def test_hard_timeout_kills_a_child_that_ignores_sigterm() -> None:
    worker = SupervisedWorker(sigterm_ignoring_handler, hard_timeout_seconds=0.3)

    try:
        receipt = worker.request(request("request-1"))

        assert receipt.outcome is WorkerOutcome.UNKNOWN_OUTCOME
        assert worker.is_alive is False
    finally:
        if worker.is_alive:
            worker._process.kill()
            worker._process.join(0.3)
        worker.close()


def test_hard_timeout_is_fatal_when_child_survives_kill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeConnection:
        def send(self, _value: object) -> None:
            pass

        def poll(self, _timeout: float) -> bool:
            return False

        def close(self) -> None:
            pass

    class UnkillableProcess:
        daemon = True

        def __init__(self) -> None:
            self.terminate_calls = 0
            self.kill_calls = 0
            self.join_calls = 0

        def start(self) -> None:
            pass

        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1

        def join(self, _timeout: float) -> None:
            self.join_calls += 1

    process = UnkillableProcess()
    parent = FakeConnection()
    child = FakeConnection()

    class FakeContext:
        def Pipe(self, *, duplex: bool) -> tuple[FakeConnection, FakeConnection]:
            assert duplex is True
            return parent, child

        def Process(self, **_kwargs: object) -> UnkillableProcess:
            return process

    monkeypatch.setattr(
        "oamb.runtime.worker.multiprocessing.get_context",
        lambda _method: FakeContext(),
    )
    worker = SupervisedWorker(echo_handler, hard_timeout_seconds=0.01)

    with pytest.raises(RuntimeError, match="remained alive after kill"):
        worker.request(request("request-1"))

    assert process.terminate_calls == 1
    assert process.kill_calls == 1
    assert process.join_calls == 2
