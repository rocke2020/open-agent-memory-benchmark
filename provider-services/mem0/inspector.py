#!/usr/bin/env python3
"""Authenticated, fixed-collection, read-only Qdrant projection service."""

from __future__ import annotations

import base64
import hmac
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen


RUN_ID_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_QDRANT_RESPONSE_BYTES = 32 * 1024 * 1024
SCROLL_PAGE_SIZE = 256
QDRANT_TIMEOUT_SECONDS = 15


class InspectorConfig(NamedTuple):
    listen_host: str
    listen_port: int
    api_key: str
    qdrant_url: str
    qdrant_api_key: str
    collection: str


class BackendProtocolError(Exception):
    """The fixed Qdrant backend returned an invalid projection response."""


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _run_filter(run_id: str) -> dict[str, object]:
    return {"must": [{"key": "run_id", "match": {"value": run_id}}]}


def _decode_cursor(raw_cursor: str | None) -> object | None:
    if raw_cursor is None:
        return None
    try:
        padding = "=" * (-len(raw_cursor) % 4)
        decoded = base64.urlsafe_b64decode(raw_cursor + padding)
        value = json.loads(decoded)
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("cursor must be an OAMB inspector cursor") from exc
    if isinstance(value, (dict, list, bool)) or value is None:
        raise ValueError("cursor payload must be a string or number")
    return value


def _encode_cursor(value: object | None) -> str | None:
    if value is None:
        return None
    return base64.urlsafe_b64encode(_json_bytes(value)).decode("ascii").rstrip("=")


def _qdrant_json(
    config: InspectorConfig, method: str, path: str, body: dict[str, object] | None = None
) -> dict[str, object]:
    request = Request(
        config.qdrant_url.rstrip("/") + path,
        data=None if body is None else _json_bytes(body),
        method=method,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "api-key": config.qdrant_api_key,
        },
    )
    with urlopen(request, timeout=QDRANT_TIMEOUT_SECONDS) as response:
        payload = response.read(MAX_QDRANT_RESPONSE_BYTES + 1)
        if len(payload) > MAX_QDRANT_RESPONSE_BYTES:
            raise BackendProtocolError("Qdrant response exceeds inspector limit")
        try:
            decoded = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise BackendProtocolError("Qdrant returned invalid JSON") from exc
    if not isinstance(decoded, dict) or decoded.get("status") != "ok" or "result" not in decoded:
        raise BackendProtocolError("Qdrant returned an invalid response")
    return decoded


def create_server(config: InspectorConfig) -> ThreadingHTTPServer:
    if not config.api_key or not config.qdrant_api_key:
        raise ValueError("both inspector and Qdrant API keys are required")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,62}", config.collection):
        raise ValueError("invalid fixed collection name")

    class Handler(BaseHTTPRequestHandler):
        server_version = "OAMBMem0Inspector/1"

        def log_message(self, format: str, *args: object) -> None:
            return

        def _reply(self, status: int, value: object) -> None:
            payload = _json_bytes(value)
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
                cursor = _decode_cursor(cursor_values[0] if cursor_values else None)
                run_filter = _run_filter(run_id)
                base = f"/collections/{config.collection}/points"
                count_response = _qdrant_json(
                    config, "POST", base + "/count", {"filter": run_filter, "exact": True}
                )
                scroll_body: dict[str, object] = {
                    "filter": run_filter,
                    "limit": SCROLL_PAGE_SIZE,
                    "with_payload": True,
                    "with_vector": False,
                }
                if cursor is not None:
                    scroll_body["offset"] = cursor
                scroll_response = _qdrant_json(config, "POST", base + "/scroll", scroll_body)
                scroll_result = scroll_response["result"]
                if not isinstance(scroll_result, dict):
                    raise ValueError("Qdrant scroll result is not an object")
                result = {
                    "collection": config.collection,
                    "run_id": run_id,
                    "count": count_response["result"]["count"],
                    "points": scroll_result.get("points", []),
                    "next_cursor": _encode_cursor(scroll_result.get("next_page_offset")),
                }
                self._reply(200, result)
            except ValueError as exc:
                self._reply(400, {"error": str(exc)})
            except (BackendProtocolError, HTTPError, URLError, TimeoutError, KeyError, TypeError):
                self._reply(502, {"error": "Qdrant projection unavailable"})

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
        qdrant_url=os.environ["OAMB_QDRANT_URL"],
        qdrant_api_key=os.environ["OAMB_QDRANT_API_KEY"],
        collection=os.environ.get("OAMB_MEM0_COLLECTION", "oamb_memories"),
    )


if __name__ == "__main__":
    server = create_server(_config_from_environment())
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
