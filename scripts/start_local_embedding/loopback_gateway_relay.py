#!/usr/bin/env python3
"""Relay the Linux Docker gateway endpoint to a loopback-only Ollama server."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import ipaddress
import os
import signal
from collections.abc import Sequence

MAX_ACTIVE_CONNECTIONS = 32
COPY_CHUNK_BYTES = 64 * 1024
SHUTDOWN_SECONDS = 1


def _ipv4(value: str) -> str:
    address = ipaddress.ip_address(value)
    if address.version != 4 or address.is_unspecified or address.is_multicast:
        raise argparse.ArgumentTypeError("relay hosts must be usable IPv4 addresses")
    return value


def _port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("relay ports must be from 1 through 65535")
    return port


async def _copy_stream(
    source: asyncio.StreamReader,
    destination: asyncio.StreamWriter,
) -> None:
    while content := await source.read(COPY_CHUNK_BYTES):
        destination.write(content)
        await destination.drain()
    if destination.can_write_eof():
        destination.write_eof()
        await destination.drain()


async def _serve(
    *,
    listen_host: str,
    listen_port: int,
    target_host: str,
    target_port: int,
) -> None:
    stop_requested = asyncio.Event()
    active_tasks: set[asyncio.Task[None]] = set()
    active_writers: set[asyncio.StreamWriter] = set()

    async def relay(
        source_reader: asyncio.StreamReader,
        source_writer: asyncio.StreamWriter,
    ) -> None:
        current = asyncio.current_task()
        if current is None or len(active_tasks) >= MAX_ACTIVE_CONNECTIONS:
            source_writer.close()
            await source_writer.wait_closed()
            return
        active_tasks.add(current)
        active_writers.add(source_writer)
        target_writer: asyncio.StreamWriter | None = None
        try:
            target_reader, target_writer = await asyncio.open_connection(
                target_host,
                target_port,
            )
            active_writers.add(target_writer)
            try:
                async with asyncio.TaskGroup() as copies:
                    copies.create_task(_copy_stream(source_reader, target_writer))
                    copies.create_task(_copy_stream(target_reader, source_writer))
            except* (BrokenPipeError, ConnectionError):
                pass
        except OSError:
            pass
        finally:
            for writer in (source_writer, target_writer):
                if writer is not None:
                    active_writers.discard(writer)
                    if stop_requested.is_set():
                        writer.transport.abort()
                    else:
                        writer.close()
                        with contextlib.suppress(OSError):
                            await writer.wait_closed()
            active_tasks.discard(current)

    server = await asyncio.start_server(relay, listen_host, listen_port)
    loop = asyncio.get_running_loop()
    forced_exit: asyncio.TimerHandle | None = None

    def request_stop() -> None:
        nonlocal forced_exit
        if stop_requested.is_set():
            return
        stop_requested.set()
        forced_exit = loop.call_later(SHUTDOWN_SECONDS, os._exit, 0)

    for signal_number in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signal_number, request_stop)
    async with server:
        await stop_requested.wait()
    for task in tuple(active_tasks):
        task.cancel()
    for writer in tuple(active_writers):
        writer.transport.abort()
    if active_tasks:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(
                asyncio.gather(*tuple(active_tasks), return_exceptions=True),
                timeout=SHUTDOWN_SECONDS,
            )
    if forced_exit is not None:
        forced_exit.cancel()


def main(arguments: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", required=True, type=_ipv4)
    parser.add_argument("--listen-port", required=True, type=_port)
    parser.add_argument("--target-host", required=True, type=_ipv4)
    parser.add_argument("--target-port", required=True, type=_port)
    parsed = parser.parse_args(arguments)
    asyncio.run(
        _serve(
            listen_host=parsed.listen_host,
            listen_port=parsed.listen_port,
            target_host=parsed.target_host,
            target_port=parsed.target_port,
        )
    )


if __name__ == "__main__":
    main()
