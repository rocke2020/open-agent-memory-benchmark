"""Shared single-dispatch REST transport for memory-system adapters."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Never

import httpx

from oamb.contracts.ports import (
    ArtifactStorePort,
    MemorySystemCallCancelledBeforeDispatch,
    MemorySystemCallCancelledUnknownOutcome,
    MemorySystemCallFailure,
    MemorySystemCallUnknownOutcome,
    MemorySystemReadCancelled,
    RawPayloadSealRequest,
    RawReferenceHandle,
)

_RECEIPT_SEAL_ERROR = "receipt_seal_error"
DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS = 120.0
DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True, slots=True)
class SealedRestResponse:
    status_code: int
    raw_bytes: bytes
    raw_reference: RawReferenceHandle
    request_reference: RawReferenceHandle | None = None


class SealedRestClient:
    """Dispatch admitted requests once within one event loop and seal responses."""

    def __init__(
        self,
        *,
        store: ArtifactStorePort,
        base_url: str,
        headers: Mapping[str, str],
        transport: httpx.AsyncBaseTransport | None = None,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS,
        write_timeout_seconds: float = 30.0,
        pool_timeout_seconds: float = 10.0,
        total_timeout_seconds: float = DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS,
    ) -> None:
        if not base_url:
            raise ValueError("memory-system base URL is required")
        if (
            min(
                connect_timeout_seconds,
                read_timeout_seconds,
                write_timeout_seconds,
                pool_timeout_seconds,
                total_timeout_seconds,
            )
            <= 0
        ):
            raise ValueError("memory-system timeouts must be positive")
        self._store = store
        self._base_url = base_url.rstrip("/")
        self._base_header_names = frozenset(name.lower() for name in headers)
        self._total_timeout_seconds = total_timeout_seconds
        self._transport = transport or httpx.AsyncHTTPTransport()
        self._client = httpx.AsyncClient(
            headers=dict(headers),
            timeout=httpx.Timeout(
                connect=connect_timeout_seconds,
                read=read_timeout_seconds,
                write=write_timeout_seconds,
                pool=pool_timeout_seconds,
            ),
            transport=self._transport,
            follow_redirects=False,
        )
        self._accepting_operations = True
        self._closed = False
        self._active_operation_count = 0
        self._active_operations_drained = asyncio.Event()
        self._active_operations_drained.set()
        self._close_lock = asyncio.Lock()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | int | bool] | None = None,
        json_payload: object | None = None,
        request_headers: Mapping[str, str] | None = None,
        write_intent: bool = False,
        request_evidence: bool = False,
        response_sanitizer: Callable[[bytes], bytes] | None = None,
        total_timeout_seconds: float | None = None,
    ) -> SealedRestResponse:
        request_headers = request_headers or {}
        if total_timeout_seconds is not None and (
            not math.isfinite(total_timeout_seconds) or total_timeout_seconds <= 0
        ):
            raise ValueError("request total timeout must be finite positive")
        if self._base_header_names.intersection(name.lower() for name in request_headers):
            raise ValueError("request headers cannot override client authority")
        self._admit_operation(write_intent=write_intent)
        try:
            return await self._request_once(
                method,
                path,
                params=params,
                json_payload=json_payload,
                request_headers=request_headers,
                write_intent=write_intent,
                request_evidence=request_evidence,
                response_sanitizer=response_sanitizer,
                total_timeout_seconds=total_timeout_seconds,
            )
        finally:
            self._release_operation()

    async def _request_once(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str | int | bool] | None,
        json_payload: object | None,
        request_headers: Mapping[str, str],
        write_intent: bool,
        request_evidence: bool,
        response_sanitizer: Callable[[bytes], bytes] | None,
        total_timeout_seconds: float | None,
    ) -> SealedRestResponse:
        url = f"{self._base_url}/{path.lstrip('/')}"
        request_reference: RawReferenceHandle | None = None
        if request_evidence:
            request_bytes = json.dumps(
                {
                    "schema_name": "oamb_rest_request_proof",
                    "schema_version": 1,
                    "method": method.upper(),
                    "path": path,
                    "params": dict(params or {}),
                    "json_payload": json_payload,
                    "request_header_names": sorted(name.lower() for name in request_headers),
                    "write_intent": write_intent,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            request_reference = self._store.seal_raw(
                RawPayloadSealRequest(
                    sha256=hashlib.sha256(request_bytes).hexdigest(),
                    media_type="application/vnd.oamb.request+json",
                    compression="gzip",
                    payload_bytes=request_bytes,
                )
            )
        try:
            async with asyncio.timeout(
                self._total_timeout_seconds
                if total_timeout_seconds is None
                else total_timeout_seconds
            ):
                response = await self._client.request(
                    method,
                    url,
                    params=params,
                    json=json_payload,
                    headers=dict(request_headers),
                )
        except asyncio.CancelledError as exc:
            if write_intent:
                raise MemorySystemCallCancelledUnknownOutcome(
                    "memory-system write was cancelled with unknown provider acceptance"
                ) from exc
            raise MemorySystemReadCancelled("memory-system read was cancelled") from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            if write_intent:
                raise MemorySystemCallUnknownOutcome(
                    "memory-system write timed out with unknown provider acceptance",
                    failure_kind="timeout",
                ) from exc
            raise MemorySystemCallFailure(
                "memory-system read timed out",
                failure_kind="timeout",
            ) from exc
        except httpx.TransportError as exc:
            if write_intent:
                raise MemorySystemCallUnknownOutcome(
                    "memory-system write transport failed with unknown provider acceptance",
                    failure_kind="transport_error",
                ) from exc
            raise MemorySystemCallFailure(
                "memory-system read transport failed",
                failure_kind="transport_error",
            ) from exc

        try:
            response_bytes = (
                response.content
                if response_sanitizer is None
                else response_sanitizer(response.content)
            )
            raw_reference = self._seal_raw(response_bytes)
        except Exception as exc:
            if write_intent:
                raise MemorySystemCallUnknownOutcome(
                    "memory-system write response could not be sealed",
                    failure_kind=_RECEIPT_SEAL_ERROR,
                ) from exc
            raise MemorySystemCallFailure(
                "memory-system read response could not be sealed",
                failure_kind=_RECEIPT_SEAL_ERROR,
            ) from exc
        sealed = SealedRestResponse(
            status_code=response.status_code,
            raw_bytes=response_bytes,
            raw_reference=raw_reference,
            request_reference=request_reference,
        )
        if not 200 <= response.status_code < 300:
            raise MemorySystemCallFailure(
                f"memory-system returned HTTP {response.status_code}",
                failure_kind="http_status",
                raw_reference=raw_reference,
                raw_response_bytes=response_bytes,
                status_code=response.status_code,
            )
        return sealed

    async def close(self) -> None:
        self.stop_accepting()
        await self._active_operations_drained.wait()
        async with self._close_lock:
            if not self._closed:
                if self._client.is_closed:
                    await self._transport.aclose()
                else:
                    await self._client.aclose()
                self._closed = True

    def stop_accepting(self) -> None:
        """Reject new work before an owner begins an ordered shutdown."""

        self._accepting_operations = False

    def _admit_operation(self, *, write_intent: bool) -> None:
        if self._closed:
            raise RuntimeError("memory-system REST client is closed")
        if not self._accepting_operations:
            self._raise_before_dispatch(write_intent=write_intent)
        self._active_operation_count += 1
        if self._active_operation_count == 1:
            self._active_operations_drained.clear()

    def _release_operation(self) -> None:
        self._active_operation_count -= 1
        if self._active_operation_count == 0:
            self._active_operations_drained.set()

    @staticmethod
    def _raise_before_dispatch(*, write_intent: bool) -> Never:
        if write_intent:
            raise MemorySystemCallCancelledBeforeDispatch(
                "memory-system write was cancelled before dispatch"
            )
        raise MemorySystemReadCancelled("memory-system read was cancelled before dispatch")

    def _seal_raw(self, raw_bytes: bytes) -> RawReferenceHandle:
        sha256 = hashlib.sha256(raw_bytes).hexdigest()
        return self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=sha256,
                media_type="application/json",
                compression="gzip",
                payload_bytes=raw_bytes,
            )
        )


def parse_exact_json_object(
    raw_bytes: bytes,
    *,
    expected_fields: frozenset[str],
) -> dict[str, Any]:
    """Parse a JSON object while rejecting duplicate, unknown, or missing fields."""

    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            raw_bytes,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("response is not valid JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("response root must be an object")
    actual_fields = frozenset(document)
    if actual_fields != expected_fields:
        raise ValueError(
            "response fields do not match the exact profile: "
            f"expected {sorted(expected_fields)}, got {sorted(actual_fields)}"
        )
    return document


def sealed_response_validation_failure(
    response: SealedRestResponse,
    *,
    message: str,
    supporting_raw_references: tuple[RawReferenceHandle, ...] = (),
) -> MemorySystemCallFailure:
    """Bind a response-validation error to the already sealed provider bytes."""

    return MemorySystemCallFailure(
        message,
        failure_kind="response_validation",
        raw_reference=response.raw_reference,
        raw_response_bytes=response.raw_bytes,
        supporting_raw_references=supporting_raw_references,
        status_code=response.status_code,
    )


def link_preceding_raw_references(
    failure: MemorySystemCallFailure,
    *,
    preceding_raw_references: tuple[RawReferenceHandle, ...],
) -> MemorySystemCallFailure:
    """Copy a call failure while linking earlier responses from one composite call."""

    return MemorySystemCallFailure(
        str(failure),
        failure_kind=failure.failure_kind,
        raw_reference=failure.raw_reference,
        raw_response_bytes=failure.raw_response_bytes,
        supporting_raw_references=(
            *preceding_raw_references,
            *failure.supporting_raw_references,
        ),
        status_code=failure.status_code,
    )


def link_preceding_read_cancellation(
    cancellation: MemorySystemReadCancelled,
    *,
    preceding_raw_references: tuple[RawReferenceHandle, ...],
) -> MemorySystemReadCancelled:
    """Copy a read cancellation while linking earlier composite responses."""

    return MemorySystemReadCancelled(
        str(cancellation),
        supporting_raw_references=(
            *preceding_raw_references,
            *cancellation.supporting_raw_references,
        ),
    )


def link_preceding_before_dispatch_cancellation(
    cancellation: MemorySystemCallCancelledBeforeDispatch,
    *,
    preceding_raw_references: tuple[RawReferenceHandle, ...],
) -> MemorySystemCallCancelledBeforeDispatch:
    """Copy an undispatched cancellation while linking completed preflight reads."""

    return MemorySystemCallCancelledBeforeDispatch(
        str(cancellation),
        supporting_raw_references=(
            *preceding_raw_references,
            *cancellation.supporting_raw_references,
        ),
    )


@contextmanager
def bind_sealed_response_validation(
    response: SealedRestResponse,
    *,
    message: str,
    supporting_raw_references: tuple[RawReferenceHandle, ...] = (),
) -> Iterator[None]:
    """Classify exact-profile validation against an already sealed response."""

    try:
        yield
    except ValueError as exc:
        raise sealed_response_validation_failure(
            response,
            message=message,
            supporting_raw_references=supporting_raw_references,
        ) from exc


__all__ = [
    "DEFAULT_MEMORY_SYSTEM_READ_TIMEOUT_SECONDS",
    "DEFAULT_MEMORY_SYSTEM_TOTAL_TIMEOUT_SECONDS",
    "MemorySystemCallCancelledBeforeDispatch",
    "MemorySystemCallCancelledUnknownOutcome",
    "MemorySystemCallFailure",
    "MemorySystemCallUnknownOutcome",
    "MemorySystemReadCancelled",
    "SealedRestClient",
    "SealedRestResponse",
    "bind_sealed_response_validation",
    "link_preceding_before_dispatch_cancellation",
    "link_preceding_read_cancellation",
    "link_preceding_raw_references",
    "parse_exact_json_object",
    "sealed_response_validation_failure",
]
