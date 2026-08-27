"""One supervised process for an uncancellable comparison-ineligible SDK call."""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from multiprocessing.connection import Connection
from typing import Protocol

from oamb.contracts.ports import WorkerOutcome, WorkerReceipt, WorkerRequest


class WorkerRetiredError(RuntimeError):
    """Raised when a terminated worker scope is used again."""


class WorkerTerminationError(RuntimeError):
    """Raised when a child remains alive after terminate and kill bounds."""


class StoppableProcess(Protocol):
    def is_alive(self) -> bool: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...

    def join(self, timeout: float | None = None) -> None: ...


WorkerHandler = Callable[[WorkerRequest], WorkerReceipt]


def _worker_loop(connection: Connection, handler: WorkerHandler) -> None:
    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            try:
                receipt = handler(request)
            except BaseException:
                receipt = WorkerReceipt(
                    request_id=request.request_id,
                    outcome=WorkerOutcome.FAILED,
                    raw_reference=None,
                    canonical_bytes=b"",
                )
            connection.send(receipt)
    finally:
        connection.close()


class SupervisedWorker:
    """Owns one child process and retires it after an unbounded-call timeout."""

    def __init__(self, handler: WorkerHandler, *, hard_timeout_seconds: float) -> None:
        if hard_timeout_seconds <= 0:
            raise ValueError("worker hard timeout must be positive")
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe(duplex=True)
        self._connection: Connection = parent
        self._process = context.Process(target=_worker_loop, args=(child, handler), daemon=True)
        self._timeout = hard_timeout_seconds
        self._retired = False
        self._process.start()
        child.close()

    @property
    def is_alive(self) -> bool:
        return self._process.is_alive()

    def request(self, request: WorkerRequest) -> WorkerReceipt:
        if self._retired or not self._process.is_alive():
            raise WorkerRetiredError("supervised worker scope is retired")
        try:
            self._connection.send(request)
        except (BrokenPipeError, EOFError, OSError):
            return self._retire_unknown(request.request_id)
        if not self._connection.poll(self._timeout):
            return self._retire_unknown(request.request_id)
        try:
            receipt = self._connection.recv()
        except (EOFError, OSError):
            return self._retire_unknown(request.request_id)
        if not isinstance(receipt, WorkerReceipt) or receipt.request_id != request.request_id:
            return self._retire_unknown(request.request_id)
        return receipt

    def close(self) -> None:
        try:
            if self._process.is_alive() and not self._retired:
                try:
                    self._connection.send(None)
                except (BrokenPipeError, EOFError, OSError):
                    pass
                self._process.join(self._timeout)
            if self._process.is_alive():
                _terminate_and_join(self._process, self._timeout)
        finally:
            self._connection.close()
            self._retired = True

    def _retire_unknown(self, request_id: str) -> WorkerReceipt:
        try:
            if self._process.is_alive():
                _terminate_and_join(self._process, self._timeout)
        finally:
            self._connection.close()
            self._retired = True
        return WorkerReceipt(
            request_id=request_id,
            outcome=WorkerOutcome.UNKNOWN_OUTCOME,
            raw_reference=None,
            canonical_bytes=b"",
        )


def _terminate_and_join(process: StoppableProcess, timeout: float) -> None:
    process.terminate()
    process.join(timeout)
    if not process.is_alive():
        return
    process.kill()
    process.join(timeout)
    if process.is_alive():
        raise WorkerTerminationError("supervised worker remained alive after kill")
