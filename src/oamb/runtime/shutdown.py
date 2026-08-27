"""Bounded shutdown ordering that preserves partial evidence."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol

CANCELLATION_SETTLEMENT_POLL_SECONDS = 0.01


class AsyncCloseable(Protocol):
    async def close(self) -> None: ...


class ShutdownSequenceError(RuntimeError):
    """Reports shutdown-stage failures after all later preservation steps run."""

    def __init__(self, errors: tuple[BaseException, ...]) -> None:
        self.errors = errors
        names = ", ".join(type(error).__name__ for error in errors)
        super().__init__(f"shutdown sequence failed: {names}")


class ActiveTaskSettlementError(RuntimeError):
    """The cancelled active task remained live beyond its settlement bound."""


class StopRequestTimeoutError(RuntimeError):
    """The cooperative stop request exceeded its hard timeout."""


@dataclass(frozen=True, slots=True)
class ShutdownHooks:
    seal_interruption: Callable[[bool], None]
    flush_records: Callable[[], None]
    write_manifest: Callable[[], None]
    record_close_error: Callable[[str, BaseException], None]


@dataclass(frozen=True, slots=True)
class ShutdownResult:
    unknown_outcome: bool
    close_error_owners: tuple[str, ...]
    active_error_type: str | None


class ShutdownCoordinator:
    """Stops scheduling, settles active work, then seals, closes, and checkpoints."""

    def __init__(self, hooks: ShutdownHooks) -> None:
        self._hooks = hooks
        self._active_task: asyncio.Task[None] | None = None
        self._request_stop: Callable[[], Awaitable[None]] | None = None
        self._shutdown_started = False

    @property
    def accepting_work(self) -> bool:
        return not self._shutdown_started

    def track_active(
        self,
        task: asyncio.Task[None],
        *,
        request_stop: Callable[[], Awaitable[None]],
    ) -> None:
        if self._shutdown_started or self._active_task is not None:
            raise RuntimeError("shutdown started or another call is already active")
        self._active_task = task
        self._request_stop = request_stop

    async def shutdown(
        self,
        *,
        clients: tuple[tuple[str, AsyncCloseable], ...],
        grace_seconds: float,
    ) -> ShutdownResult:
        if self._shutdown_started:
            raise RuntimeError("shutdown already started")
        if grace_seconds < 0:
            raise ValueError("shutdown grace must be non-negative")
        self._shutdown_started = True
        unknown_outcome = False
        active_task_settled = True
        active_error_type: str | None = None
        shutdown_errors: list[BaseException] = []
        if self._active_task is not None:
            if self._request_stop is None:
                raise RuntimeError("active task has no stop request")
            stop_task = asyncio.ensure_future(self._request_stop())
            completed_stop_tasks, _pending_stop_tasks = await asyncio.wait(
                (stop_task,), timeout=grace_seconds
            )
            if stop_task not in completed_stop_tasks:
                pending = await _cancel_and_settle(
                    (stop_task, self._active_task),
                    timeout=grace_seconds,
                )
                errors: list[BaseException] = [
                    StopRequestTimeoutError("request_stop exceeded shutdown grace")
                ]
                if self._active_task in pending:
                    errors.append(
                        ActiveTaskSettlementError(
                            "active task ignored cancellation after stop request timeout"
                        )
                    )
                for task in pending:
                    task.add_done_callback(_consume_task_result)
                raise ShutdownSequenceError(tuple(errors))
            try:
                stop_task.result()
            except BaseException as exc:
                shutdown_errors.append(exc)
            try:
                await asyncio.wait_for(asyncio.shield(self._active_task), timeout=grace_seconds)
            except TimeoutError:
                unknown_outcome = True
                self._active_task.cancel()
                try:
                    await asyncio.wait_for(asyncio.shield(self._active_task), timeout=grace_seconds)
                except asyncio.CancelledError:
                    pass
                except TimeoutError:
                    active_task_settled = self._active_task.done()
                    if not active_task_settled:
                        shutdown_errors.append(
                            ActiveTaskSettlementError(
                                "active task ignored cancellation beyond shutdown grace"
                            )
                        )
                except BaseException as exc:
                    active_error_type = type(exc).__name__
            except BaseException as exc:
                active_error_type = type(exc).__name__

        if not active_task_settled:
            raise ShutdownSequenceError(tuple(shutdown_errors))

        try:
            self._hooks.seal_interruption(unknown_outcome)
        except BaseException as exc:
            shutdown_errors.append(exc)
        try:
            self._hooks.flush_records()
        except BaseException as exc:
            shutdown_errors.append(exc)
        close_error_owners: list[str] = []
        if active_task_settled:
            for owner, client in clients:
                try:
                    await client.close()
                except BaseException as exc:
                    close_error_owners.append(owner)
                    try:
                        self._hooks.record_close_error(owner, exc)
                    except BaseException as record_error:
                        shutdown_errors.append(record_error)
        try:
            self._hooks.write_manifest()
        except BaseException as exc:
            shutdown_errors.append(exc)
        if shutdown_errors:
            raise ShutdownSequenceError(tuple(shutdown_errors))
        return ShutdownResult(unknown_outcome, tuple(close_error_owners), active_error_type)


def _consume_task_result(task: asyncio.Future[None]) -> None:
    try:
        task.result()
    except BaseException:
        pass


async def _cancel_and_settle(
    tasks: tuple[asyncio.Task[None], ...],
    *,
    timeout: float,
) -> set[asyncio.Task[None]]:
    pending = {task for task in tasks if not task.done()}
    deadline = asyncio.get_running_loop().time() + timeout
    while pending:
        for task in pending:
            task.cancel()
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            break
        _done, pending = await asyncio.wait(
            pending,
            timeout=min(remaining, CANCELLATION_SETTLEMENT_POLL_SECONDS),
        )
    return pending
