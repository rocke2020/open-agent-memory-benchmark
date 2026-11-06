"""Portable synchronous OAM HTTP adapter; no product checkout or SDK required."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

import httpx

from oamb.contracts.ids import canonical_sha256


def pack_messages(
    messages: list[dict[str, Any]], *, max_messages: int, max_bytes: int
) -> list[list[dict[str, Any]]]:
    """Preserve every UTF-8 character, role and timestamp under admission limits."""
    if max_messages < 1 or max_bytes < 4:
        raise ValueError("invalid OAM ingestion limits")
    batches: list[list[dict[str, Any]]] = []
    batch: list[dict[str, Any]] = []
    size = 0
    for message in messages:
        content = message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("empty source message")
        pieces: list[str] = []
        start, length = 0, 0
        for index, char in enumerate(content):
            width = len(char.encode())
            if length + width > max_bytes:
                pieces.append(content[start:index])
                start, length = index, 0
            length += width
        pieces.append(content[start:])
        for piece in pieces:
            width = len(piece.encode())
            if batch and (len(batch) == max_messages or size + width > max_bytes):
                batches.append(batch)
                batch, size = [], 0
            batch.append({**message, "content": piece})
            size += width
    if batch:
        batches.append(batch)
    return batches


class OAMAdapter:
    def __init__(
        self,
        http: httpx.Client,
        base_url: str,
        record: Callable[..., None],
        *,
        timeout: float = 900,
    ) -> None:
        self.http, self.base_url, self.record = http, base_url.rstrip("/"), record
        self.timeout = timeout

    def request(self, method: str, path: str, key: str = "", **kwargs: Any) -> dict[str, Any]:
        headers = kwargs.pop("headers", {})
        if key:
            headers["Authorization"] = f"Bearer {key}"
        started = time.monotonic()
        response = self.http.request(method, self.base_url + path, headers=headers, **kwargs)
        try:
            body = response.json()
        except ValueError:
            self.record("http_error", {"path": path, "status": response.status_code})
            raise ValueError(f"OAM returned non-JSON HTTP {response.status_code}") from None
        # User credentials are stored separately in owner-only files by the caller.
        safe_body = (
            {k: v for k, v in body.items() if k != "secret"} if isinstance(body, dict) else body
        )
        self.record(
            "http",
            {
                "method": method,
                "path": path,
                "request": kwargs.get("json"),
                "status": response.status_code,
                "response": safe_body,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        response.raise_for_status()
        if not isinstance(body, dict):
            raise ValueError("OAM response must be an object")
        return body

    def preflight(self) -> dict[str, Any]:
        ready = self.request("GET", "/health/ready")
        if ready.get("ready") is not True:
            raise ValueError("OAM is not ready")
        capabilities = self.request("GET", "/v1/capabilities")
        if capabilities.get("contract_version") != "0.0.1":
            raise ValueError("OAM contract version must be 0.0.1")
        if not capabilities.get("index_generation_id") or not capabilities.get("config_sha256"):
            raise ValueError("OAM runtime identity missing")
        return capabilities

    def create_owner(self, user_id: str, root_key: str) -> str:
        try:
            result = self.request("POST", "/v1/users", root_key, json={"user_id": user_id})
        except httpx.HTTPStatusError as error:
            detail = error.response.json().get("detail", {})
            if error.response.status_code != 422 or detail != {
                "code": "InvalidRequest",
                "message": "user_id already exists",
            }:
                raise
            # Creation may have committed before its response or local key was saved.
            result = self.request("POST", "/v1/credentials", root_key, json={"user_id": user_id})
            if result.get("principal_id") != f"user:{user_id}" or result.get("state") != "active":
                raise ValueError("OAM credential recovery returned an invalid identity") from None
        else:
            if result.get("user_id") != user_id:
                raise ValueError("OAM user creation returned an invalid identity")
        if not isinstance(result.get("secret"), str) or not result["secret"]:
            raise ValueError("OAM user creation returned an invalid identity")
        return str(result["secret"])

    def retain(
        self,
        user_id: str,
        key: str,
        messages: list[dict[str, Any]],
        *,
        occurrence_id: str = "single",
    ) -> dict[str, Any]:
        payload = {"input_type": "interaction", "user_id": user_id, "messages": messages}
        return self.request(
            "POST",
            "/v1/retain",
            key,
            json=payload,
            headers={"Idempotency-Key": "oamb-" + canonical_sha256([occurrence_id, payload])},
        )

    def wait_ready(self, user_id: str, key: str, operation_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        while True:
            result = self.request(
                "GET",
                f"/v1/users/{quote(user_id, safe='')}/operations/{quote(operation_id, safe='')}",
                key,
            )
            status = result.get("status")
            if status == "succeeded":
                self.validate_ready(result)
                return result
            if status not in {"pending", "processing"}:
                raise ValueError(f"OAM operation ended with {status}")
            if time.monotonic() >= deadline:
                raise TimeoutError("OAM operation readiness deadline exceeded")
            time.sleep(min(1, max(0, deadline - time.monotonic())))

    @staticmethod
    def validate_ready(result: dict[str, Any]) -> None:
        if result.get("status") != "succeeded" or not all(
            result.get("readiness", {}).get(name) is True
            for name in ("source_searchable", "memory_ready", "links_ready", "temporal_links_ready")
        ):
            raise ValueError("OAM operation lacks query readiness")
        if result.get("stage_state", {}).get("extraction_coverage") != "complete":
            raise ValueError("OAM operation has incomplete extraction coverage")

    def recall(
        self,
        user_id: str,
        key: str,
        query: str,
        mode: str,
        generation: str,
        reference_time: str | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "user_id": user_id,
            "query": query,
            "recall_level": mode,
            "include": {"chunks": True},
            "allow_partial": False,
        }
        if reference_time:
            payload.update(reference_time=reference_time, reference_timezone="UTC")
        result = self.request("POST", "/v1/recall", key, json=payload)
        diagnostics = result.get("diagnostics", {})
        if (
            diagnostics.get("generation_id") != generation
            or diagnostics.get("recall_level") != mode
            or diagnostics.get("generative_calls") != 0
            or diagnostics.get("degraded") != []
            or diagnostics.get("partial") is not False
            or diagnostics.get("readiness") != "ready"
            or not isinstance(result.get("context"), str)
        ):
            raise ValueError("OAM recall violated mode, generation or readiness contract")
        return result
