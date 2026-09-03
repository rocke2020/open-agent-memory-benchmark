#!/usr/bin/env python3
"""Authenticated, fixed-table, read-only PostgreSQL projection service."""

from __future__ import annotations

import base64
import hmac
import json
import os
import re
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple
from urllib.parse import parse_qs, urlparse

RUN_ID_RE = re.compile(r"^[0-9a-f]{64}$")
COLLECTION_NAME = "oamb_memories"
TABLE_NAME = "public.oamb_memories"
MAX_INSPECTOR_RESPONSE_BYTES = 32 * 1024 * 1024
PAGE_SIZE = 256
PAGE_QUERY_LIMIT = PAGE_SIZE + 1
STATEMENT_TIMEOUT_MILLISECONDS = 15_000


class InspectorConfig(NamedTuple):
    listen_host: str
    listen_port: int
    api_key: str
    postgres_host: str
    postgres_port: int
    postgres_database: str
    postgres_user: str
    postgres_password: str


class BackendProtocolError(Exception):
    """The fixed PostgreSQL backend returned invalid projection data."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _parse_uuid(value: object) -> str:
    if not isinstance(value, str):
        raise BackendProtocolError("projection ID is not a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise BackendProtocolError("projection ID is not a UUID") from exc
    canonical = str(parsed)
    if value != canonical:
        raise BackendProtocolError("projection ID is not canonical")
    return canonical


def _decode_cursor(raw_cursor: str | None) -> str | None:
    if raw_cursor is None:
        return None
    try:
        padding = "=" * (-len(raw_cursor) % 4)
        decoded = base64.b64decode(
            raw_cursor + padding,
            altchars=b"-_",
            validate=True,
        )
        value = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cursor must be an OAMB inspector cursor") from exc
    if not isinstance(value, str):
        raise ValueError("cursor payload must be a UUID string")
    try:
        parsed = uuid.UUID(value)
    except ValueError as exc:
        raise ValueError("cursor payload must be a UUID string") from exc
    if value != str(parsed):
        raise ValueError("cursor payload must be a canonical UUID string")
    return value


def _encode_cursor(value: str | None) -> str | None:
    if value is None:
        return None
    return base64.urlsafe_b64encode(_json_bytes(value)).decode("ascii").rstrip("=")


def _postgres_connection(config: InspectorConfig):
    import psycopg

    return psycopg.connect(
        host=config.postgres_host,
        port=config.postgres_port,
        dbname=config.postgres_database,
        user=config.postgres_user,
        password=config.postgres_password,
        connect_timeout=5,
    )


def _read_projection(
    config: InspectorConfig,
    run_id: str,
    cursor_value: str | None,
    connection_factory,
) -> dict[str, object]:
    with connection_factory(config) as connection, connection.cursor() as cursor:
        cursor.execute("BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY")
        cursor.execute(
            "SELECT set_config('statement_timeout', %s, true)",
            (str(STATEMENT_TIMEOUT_MILLISECONDS),),
        )
        cursor.execute("SELECT to_regclass(%s)", (TABLE_NAME,))
        table_row = cursor.fetchone()
        if not isinstance(table_row, tuple) or len(table_row) != 1:
            raise BackendProtocolError("table lookup returned an invalid row")
        if table_row[0] is None:
            return {
                "collection": COLLECTION_NAME,
                "run_id": run_id,
                "count": 0,
                "points": [],
                "next_cursor": None,
            }

        cursor.execute(
            """
            SELECT count(*)
            FROM public.oamb_memories
            WHERE payload->>'run_id' = %s
            """,
            (run_id,),
        )
        count_row = cursor.fetchone()
        if (
            not isinstance(count_row, tuple)
            or len(count_row) != 1
            or type(count_row[0]) is not int
            or count_row[0] < 0
        ):
            raise BackendProtocolError("projection count is invalid")
        count = count_row[0]

        if cursor_value is None:
            cursor.execute(
                """
                SELECT id::text, payload
                FROM public.oamb_memories
                WHERE payload->>'run_id' = %s
                ORDER BY id
                LIMIT %s
                """,
                (run_id, PAGE_QUERY_LIMIT),
            )
        else:
            cursor.execute(
                """
                SELECT id::text, payload
                FROM public.oamb_memories
                WHERE payload->>'run_id' = %s AND id > %s::uuid
                ORDER BY id
                LIMIT %s
                """,
                (run_id, cursor_value, PAGE_QUERY_LIMIT),
            )
        rows = cursor.fetchall()
        if not isinstance(rows, list) or len(rows) > PAGE_QUERY_LIMIT:
            raise BackendProtocolError("projection page is invalid")

        points: list[dict[str, object]] = []
        previous_id = cursor_value
        for row in rows:
            if not isinstance(row, tuple) or len(row) != 2:
                raise BackendProtocolError("projection row shape is invalid")
            point_id = _parse_uuid(row[0])
            payload = row[1]
            if not isinstance(payload, dict) or payload.get("run_id") != run_id:
                raise BackendProtocolError("projection payload is invalid")
            if previous_id is not None and point_id <= previous_id:
                raise BackendProtocolError("projection IDs are not strictly ordered")
            previous_id = point_id
            points.append({"id": point_id, "payload": payload})

        if count < len(points):
            raise BackendProtocolError("projection count is smaller than its page")
        if cursor_value is None and len(points) != min(count, PAGE_QUERY_LIMIT):
            raise BackendProtocolError("projection count does not match its first page")
        has_next = len(points) > PAGE_SIZE
        points = points[:PAGE_SIZE]
        next_cursor = _encode_cursor(points[-1]["id"] if has_next else None)
        return {
            "collection": COLLECTION_NAME,
            "run_id": run_id,
            "count": count,
            "points": points,
            "next_cursor": next_cursor,
        }


def create_server(
    config: InspectorConfig,
    *,
    connection_factory=_postgres_connection,
) -> ThreadingHTTPServer:
    required_strings = (
        config.api_key,
        config.postgres_host,
        config.postgres_database,
        config.postgres_user,
        config.postgres_password,
    )
    if not all(required_strings):
        raise ValueError("inspector and PostgreSQL configuration values are required")
    if not 1 <= config.postgres_port <= 65_535:
        raise ValueError("invalid PostgreSQL port")

    class Handler(BaseHTTPRequestHandler):
        server_version = "OAMBMem0Inspector/1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _reply(self, status: int, value: object) -> None:
            payload = _json_bytes(value)
            if status == 200 and len(payload) > MAX_INSPECTOR_RESPONSE_BYTES:
                raise BackendProtocolError("projection response exceeds inspector limit")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {config.api_key}"
            return hmac.compare_digest(supplied, expected)

        def do_GET(self) -> None:
            if not self._authorized():
                self._reply(401, {"error": "unauthorized"})
                return
            parsed = urlparse(self.path)
            if parsed.path == "/health":
                if parsed.query:
                    self._reply(400, {"error": "health accepts no query"})
                    return
                self._reply(200, {"status": "ok", "mode": "read_only_projection"})
                return
            if parsed.path != "/v1/projection":
                self._reply(404, {"error": "not found"})
                return
            query = parse_qs(parsed.query, keep_blank_values=True, strict_parsing=False)
            if set(query) - {"run_id", "cursor"} or len(query.get("run_id", [])) != 1:
                self._reply(400, {"error": "exactly one run_id is required"})
                return
            run_id = query["run_id"][0]
            if not RUN_ID_RE.fullmatch(run_id):
                self._reply(400, {"error": "run_id must be 64 lowercase hex characters"})
                return
            try:
                cursor_values = query.get("cursor", [])
                if len(cursor_values) > 1:
                    raise ValueError("cursor may appear at most once")
                cursor_value = _decode_cursor(cursor_values[0] if cursor_values else None)
                result = _read_projection(
                    config,
                    run_id,
                    cursor_value,
                    connection_factory,
                )
                self._reply(200, result)
            except ValueError as exc:
                self._reply(400, {"error": str(exc)})
            except Exception:
                self._reply(502, {"error": "PostgreSQL projection unavailable"})

        def do_POST(self) -> None:
            self._reply(405, {"error": "method not allowed"})

        do_PUT = do_POST
        do_PATCH = do_POST
        do_DELETE = do_POST

    return ThreadingHTTPServer((config.listen_host, config.listen_port), Handler)


def _config_from_environment() -> InspectorConfig:
    return InspectorConfig(
        listen_host=os.environ.get("OAMB_INSPECTOR_LISTEN_HOST", "127.0.0.1"),
        listen_port=int(os.environ.get("OAMB_INSPECTOR_LISTEN_PORT", "6333")),
        api_key=os.environ["OAMB_MEM0_INSPECTOR_API_KEY"],
        postgres_host=os.environ["POSTGRES_HOST"],
        postgres_port=int(os.environ.get("POSTGRES_PORT", "5432")),
        postgres_database=os.environ["POSTGRES_DB"],
        postgres_user=os.environ["POSTGRES_USER"],
        postgres_password=os.environ["POSTGRES_PASSWORD"],
    )


if __name__ == "__main__":
    server = create_server(_config_from_environment())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
