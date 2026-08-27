from __future__ import annotations

import asyncio

import pytest

from oamb.runtime.shutdown import ShutdownCoordinator, ShutdownHooks, ShutdownSequenceError


def test_shutdown_stops_active_call_before_seal_flush_close_and_checkpoint() -> None:
    events: list[str] = []
    release = asyncio.Event()

    async def scenario() -> None:
        async def active_call() -> None:
            await release.wait()
            events.append("active-stopped")

        async def request_stop() -> None:
            events.append("stop-requested")
            release.set()

        class Client:
            async def close(self) -> None:
                events.append("client-closed")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: events.append(f"sealed:{unknown}"),
                flush_records=lambda: events.append("flushed"),
                write_manifest=lambda: events.append("checkpoint"),
                record_close_error=lambda _owner, _error: events.append("close-error"),
            )
        )
        coordinator.track_active(asyncio.create_task(active_call()), request_stop=request_stop)
        await coordinator.shutdown(clients=(("memory", Client()),), grace_seconds=1)

    asyncio.run(scenario())

    assert events == [
        "stop-requested",
        "active-stopped",
        "sealed:False",
        "flushed",
        "client-closed",
        "checkpoint",
    ]


def test_shutdown_timeout_records_unknown_before_closing_client() -> None:
    events: list[str] = []

    async def scenario() -> None:
        async def active_call() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                events.append("active-cancelled")

        async def request_stop() -> None:
            events.append("stop-requested")

        class Client:
            async def close(self) -> None:
                events.append("client-closed")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: events.append(f"sealed:{unknown}"),
                flush_records=lambda: events.append("flushed"),
                write_manifest=lambda: events.append("checkpoint"),
                record_close_error=lambda _owner, _error: events.append("close-error"),
            )
        )
        coordinator.track_active(asyncio.create_task(active_call()), request_stop=request_stop)
        result = await coordinator.shutdown(
            clients=(("memory", Client()),),
            grace_seconds=DecimalSeconds("0.01"),
        )
        assert result.unknown_outcome is True

    asyncio.run(scenario())

    assert events == [
        "stop-requested",
        "active-cancelled",
        "sealed:True",
        "flushed",
        "client-closed",
        "checkpoint",
    ]


def test_shutdown_bounds_cancel_settlement_and_keeps_live_clients_open() -> None:
    events: list[str] = []

    async def scenario() -> None:
        release = asyncio.Event()

        async def cancellation_swallowing_call() -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                events.append("cancel-swallowed")
                await release.wait()

        async def request_stop() -> None:
            events.append("stop-requested")

        class Client:
            async def close(self) -> None:
                events.append("client-closed")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: events.append(f"sealed:{unknown}"),
                flush_records=lambda: events.append("flushed"),
                write_manifest=lambda: events.append("checkpoint"),
                record_close_error=lambda _owner, _error: events.append("close-error"),
            )
        )
        task = asyncio.create_task(cancellation_swallowing_call())
        coordinator.track_active(task, request_stop=request_stop)
        shutdown_task = asyncio.create_task(
            coordinator.shutdown(
                clients=(("memory", Client()),),
                grace_seconds=0.01,
            )
        )
        try:
            done, _pending = await asyncio.wait({shutdown_task}, timeout=0.1)
            if shutdown_task not in done:
                release.set()
                await shutdown_task
            assert shutdown_task in done, "cancel settlement exceeded its hard bound"
            with pytest.raises(ShutdownSequenceError, match="ActiveTaskSettlementError"):
                await shutdown_task
        finally:
            release.set()
            await asyncio.wait_for(task, timeout=0.25)

    asyncio.run(scenario())

    assert events == ["stop-requested", "cancel-swallowed"]


def test_shutdown_bounds_request_stop_and_runs_no_hooks_while_work_is_live() -> None:
    hook_events: list[str] = []

    async def scenario() -> None:
        release_stop = asyncio.Event()
        release_active = asyncio.Event()
        stop_finished = asyncio.Event()

        async def active_call() -> None:
            await release_active.wait()

        async def cancellation_swallowing_stop() -> None:
            try:
                await release_stop.wait()
            except asyncio.CancelledError:
                await release_stop.wait()
            finally:
                stop_finished.set()

        class Client:
            async def close(self) -> None:
                hook_events.append("client-closed")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: hook_events.append(f"sealed:{unknown}"),
                flush_records=lambda: hook_events.append("flushed"),
                write_manifest=lambda: hook_events.append("checkpoint"),
                record_close_error=lambda _owner, _error: hook_events.append("close-error"),
            )
        )
        active_task = asyncio.create_task(active_call())
        coordinator.track_active(active_task, request_stop=cancellation_swallowing_stop)
        shutdown_task = asyncio.create_task(
            coordinator.shutdown(
                clients=(("memory", Client()),),
                grace_seconds=0.01,
            )
        )
        try:
            done, _pending = await asyncio.wait({shutdown_task}, timeout=0.1)
            if shutdown_task not in done:
                release_stop.set()
                release_active.set()
                await shutdown_task
            assert shutdown_task in done, "request_stop exceeded its hard bound"
            with pytest.raises(ShutdownSequenceError, match="StopRequestTimeoutError"):
                await shutdown_task
            assert active_task.done()
            assert stop_finished.is_set()
        finally:
            release_stop.set()
            release_active.set()
            await asyncio.wait_for(stop_finished.wait(), timeout=0.25)
            await asyncio.wait_for(
                asyncio.gather(active_task, return_exceptions=True),
                timeout=0.25,
            )

    asyncio.run(scenario())

    assert hook_events == []


def test_shutdown_is_one_shot_and_never_flushes_or_closes_twice() -> None:
    events: list[str] = []

    async def scenario() -> None:
        class Client:
            async def close(self) -> None:
                events.append("client-closed")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: events.append(f"sealed:{unknown}"),
                flush_records=lambda: events.append("flushed"),
                write_manifest=lambda: events.append("checkpoint"),
                record_close_error=lambda _owner, _error: events.append("close-error"),
            )
        )
        clients = (("memory", Client()),)

        await coordinator.shutdown(clients=clients, grace_seconds=1)
        with pytest.raises(RuntimeError, match="shutdown already started"):
            await coordinator.shutdown(clients=clients, grace_seconds=1)

    asyncio.run(scenario())

    assert events == ["sealed:False", "flushed", "client-closed", "checkpoint"]


def test_close_failure_is_separate_and_does_not_skip_checkpoint() -> None:
    events: list[str] = []

    async def scenario() -> None:
        class BrokenClient:
            async def close(self) -> None:
                events.append("close-called")
                raise RuntimeError("close failed")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: events.append(f"sealed:{unknown}"),
                flush_records=lambda: events.append("flushed"),
                write_manifest=lambda: events.append("checkpoint"),
                record_close_error=lambda owner, error: events.append(
                    f"close-error:{owner}:{type(error).__name__}"
                ),
            )
        )
        result = await coordinator.shutdown(
            clients=(("memory", BrokenClient()),),
            grace_seconds=1,
        )
        assert result.close_error_owners == ("memory",)

    asyncio.run(scenario())

    assert events == [
        "sealed:False",
        "flushed",
        "close-called",
        "close-error:memory:RuntimeError",
        "checkpoint",
    ]


def test_active_call_exception_cannot_bypass_partial_seal_or_checkpoint() -> None:
    events: list[str] = []

    async def scenario() -> None:
        async def active_call() -> None:
            raise RuntimeError("operation failed")

        async def request_stop() -> None:
            events.append("stop-requested")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: events.append(f"sealed:{unknown}"),
                flush_records=lambda: events.append("flushed"),
                write_manifest=lambda: events.append("checkpoint"),
                record_close_error=lambda _owner, _error: events.append("close-error"),
            )
        )
        coordinator.track_active(asyncio.create_task(active_call()), request_stop=request_stop)

        result = await coordinator.shutdown(clients=(), grace_seconds=1)

        assert result.active_error_type == "RuntimeError"

    asyncio.run(scenario())

    assert events == ["stop-requested", "sealed:False", "flushed", "checkpoint"]


def test_flush_error_propagates_only_after_clients_close_and_checkpoint_runs() -> None:
    events: list[str] = []

    def broken_flush() -> None:
        events.append("flush-failed")
        raise OSError("disk full")

    async def scenario() -> None:
        class Client:
            async def close(self) -> None:
                events.append("client-closed")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: events.append(f"sealed:{unknown}"),
                flush_records=broken_flush,
                write_manifest=lambda: events.append("checkpoint"),
                record_close_error=lambda _owner, _error: events.append("close-error"),
            )
        )
        with pytest.raises(ShutdownSequenceError, match="OSError"):
            await coordinator.shutdown(clients=(("memory", Client()),), grace_seconds=1)

    asyncio.run(scenario())

    assert events == ["sealed:False", "flush-failed", "client-closed", "checkpoint"]


def test_stop_request_error_cannot_bypass_partial_seal_close_or_checkpoint() -> None:
    events: list[str] = []

    async def scenario() -> None:
        async def active_call() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                events.append("active-cancelled")

        async def broken_stop() -> None:
            events.append("stop-failed")
            raise RuntimeError("stop request failed")

        class Client:
            async def close(self) -> None:
                events.append("client-closed")

        coordinator = ShutdownCoordinator(
            ShutdownHooks(
                seal_interruption=lambda unknown: events.append(f"sealed:{unknown}"),
                flush_records=lambda: events.append("flushed"),
                write_manifest=lambda: events.append("checkpoint"),
                record_close_error=lambda _owner, _error: events.append("close-error"),
            )
        )
        coordinator.track_active(asyncio.create_task(active_call()), request_stop=broken_stop)

        with pytest.raises(ShutdownSequenceError, match="RuntimeError"):
            await coordinator.shutdown(clients=(("memory", Client()),), grace_seconds=0.01)

    asyncio.run(scenario())

    assert events == [
        "stop-failed",
        "active-cancelled",
        "sealed:True",
        "flushed",
        "client-closed",
        "checkpoint",
    ]


def DecimalSeconds(value: str) -> float:
    return float(value)
