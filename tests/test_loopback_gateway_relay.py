from __future__ import annotations

import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

RELAY = (
    Path(__file__).parents[1] / "scripts" / "start_local_embedding" / "loopback_gateway_relay.py"
)
MAX_ACTIVE_CONNECTIONS = 32
RELAY_TEST_HOST = "127.0.0.1"


class _EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        self.request.sendall(b"relay:" + self.request.recv(1024))


class _HoldHandler(socketserver.BaseRequestHandler):
    active_condition = threading.Condition()
    active_count = 0

    def handle(self) -> None:
        with self.active_condition:
            type(self).active_count += 1
            self.active_condition.notify_all()
        try:
            while self.request.recv(1024):
                pass
        finally:
            with self.active_condition:
                type(self).active_count -= 1
                self.active_condition.notify_all()


class _StalledHandler(socketserver.BaseRequestHandler):
    accepted = threading.Event()
    release = threading.Event()

    def handle(self) -> None:
        self.accepted.set()
        self.release.wait(timeout=10)


class _ResetAwareHandler(socketserver.BaseRequestHandler):
    condition = threading.Condition()
    accepted_count = 0
    closed_count = 0

    def handle(self) -> None:
        with self.condition:
            type(self).accepted_count += 1
            self.condition.notify_all()
        try:
            content = self.request.recv(1024)
            if content == b"healthy":
                self.request.sendall(b"ok")
            else:
                while self.request.recv(1024):
                    pass
        except ConnectionError:
            pass
        finally:
            with self.condition:
                type(self).closed_count += 1
                self.condition.notify_all()


def _relay_port() -> int:
    with socket.socket() as listener:
        listener.bind((RELAY_TEST_HOST, 0))
        return int(listener.getsockname()[1])


def _start_relay(target_port: int, relay_port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(RELAY),
            "--listen-host",
            RELAY_TEST_HOST,
            "--listen-port",
            str(relay_port),
            "--target-host",
            "127.0.0.1",
            "--target-port",
            str(target_port),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _connect(relay_port: int, *, timeout: float = 5) -> socket.socket:
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection((RELAY_TEST_HOST, relay_port), timeout=1)
        except OSError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def test_relay_forwards_bytes_and_drains_on_termination() -> None:
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), _EchoHandler) as target:
        target_thread = threading.Thread(target=target.serve_forever)
        target_thread.start()
        relay_port = _relay_port()
        relay = _start_relay(int(target.server_address[1]), relay_port)
        try:
            with _connect(relay_port) as client:
                client.sendall(b"probe")
                assert client.recv(1024) == b"relay:probe"
        finally:
            relay.terminate()
            relay.wait(timeout=5)
            target.shutdown()
            target_thread.join(timeout=5)

    assert relay.returncode == 0


def test_relay_rejects_connections_above_its_active_cap() -> None:
    _HoldHandler.active_count = 0
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), _HoldHandler) as target:
        target_thread = threading.Thread(target=target.serve_forever)
        target_thread.start()
        relay_port = _relay_port()
        relay = _start_relay(int(target.server_address[1]), relay_port)
        clients: list[socket.socket] = []
        try:
            for _ in range(MAX_ACTIVE_CONNECTIONS):
                clients.append(_connect(relay_port))
            deadline = time.monotonic() + 5
            with _HoldHandler.active_condition:
                while _HoldHandler.active_count != MAX_ACTIVE_CONNECTIONS:
                    remaining = deadline - time.monotonic()
                    assert remaining > 0
                    _HoldHandler.active_condition.wait(timeout=remaining)
            with _connect(relay_port) as rejected:
                rejected.settimeout(2)
                assert rejected.recv(1) == b""
        finally:
            for client in clients:
                client.close()
            relay.terminate()
            relay.wait(timeout=5)
            target.shutdown()
            target_thread.join(timeout=5)

    assert relay.returncode == 0


def test_relay_termination_does_not_wait_for_a_stalled_backend() -> None:
    _StalledHandler.accepted.clear()
    _StalledHandler.release.clear()
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), _StalledHandler) as target:
        target_thread = threading.Thread(target=target.serve_forever)
        target_thread.start()
        relay_port = _relay_port()
        relay = _start_relay(int(target.server_address[1]), relay_port)
        client = _connect(relay_port)
        client.settimeout(0.1)
        try:
            assert _StalledHandler.accepted.wait(timeout=5)
            content = b"x" * (64 * 1024)
            sent = 0
            while sent < 8 * 1024 * 1024:
                try:
                    sent += client.send(content)
                except TimeoutError:
                    break
            assert sent > 0
            relay.terminate()
            relay.wait(timeout=3)
        finally:
            client.close()
            if relay.poll() is None:
                relay.kill()
                relay.wait(timeout=3)
            _StalledHandler.release.set()
            target.shutdown()
            target_thread.join(timeout=5)

    assert relay.returncode == 0


def test_client_resets_release_backend_connections_and_relay_slots() -> None:
    _ResetAwareHandler.accepted_count = 0
    _ResetAwareHandler.closed_count = 0
    with socketserver.ThreadingTCPServer(("127.0.0.1", 0), _ResetAwareHandler) as target:
        target_thread = threading.Thread(target=target.serve_forever)
        target_thread.start()
        relay_port = _relay_port()
        relay = _start_relay(int(target.server_address[1]), relay_port)
        try:
            for expected_count in range(1, MAX_ACTIVE_CONNECTIONS + 1):
                client = _connect(relay_port)
                client.sendall(b"partial-request")
                deadline = time.monotonic() + 5
                with _ResetAwareHandler.condition:
                    while _ResetAwareHandler.accepted_count < expected_count:
                        remaining = deadline - time.monotonic()
                        assert remaining > 0
                        _ResetAwareHandler.condition.wait(timeout=remaining)
                client.setsockopt(
                    socket.SOL_SOCKET,
                    socket.SO_LINGER,
                    struct.pack("ii", 1, 0),
                )
                client.close()

            deadline = time.monotonic() + 5
            with _ResetAwareHandler.condition:
                while _ResetAwareHandler.closed_count < MAX_ACTIVE_CONNECTIONS:
                    remaining = deadline - time.monotonic()
                    assert remaining > 0
                    _ResetAwareHandler.condition.wait(timeout=remaining)
            with _connect(relay_port) as healthy:
                healthy.sendall(b"healthy")
                assert healthy.recv(2) == b"ok"
        finally:
            relay.terminate()
            relay.wait(timeout=3)
            target.shutdown()
            target_thread.join(timeout=5)

    assert relay.returncode == 0
