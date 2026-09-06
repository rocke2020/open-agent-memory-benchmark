"""Single-dispatch OpenAI Chat Completions transport with sealed evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from dataclasses import replace
from typing import Literal, cast

import httpx

from oamb.contracts.accounting import (
    ProofStatus,
    TokenDomain,
    TokenMeasurementSource,
    TokenStageV2,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
)
from oamb.contracts.evidence import infrastructure_supplier_call_id
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    ArtifactStorePort,
    ArtifactWriteRequest,
    FinishDisposition,
    ModelCallCancelledUnknownOutcome,
    ModelCallFailure,
    ModelCallUnknownOutcome,
    ModelCandidate,
    ModelReceipt,
    ModelRequest,
    ModelSupplierRateLimitRejection,
    RawPayloadSealRequest,
    RawReferenceHandle,
    ThinkingEffort,
)
from oamb.contracts.specifications import (
    BindingKind,
    ExecutionOwner,
    ModelRole,
    ModelRoleBindingV2,
    RoleBindingStatus,
)
from oamb.contracts.supplier_rejection import parse_structured_supplier_rejection

UsageProfile = Literal["strict-base-v2", "openai-details-v3"]
DEFAULT_MODEL_CALL_TIMEOUT_SECONDS = 120.0

_BASE_USAGE_FIELDS: frozenset[str] = frozenset(
    {"prompt_tokens", "completion_tokens", "total_tokens"}
)
_CACHE_ALIAS_USAGE_FIELDS: frozenset[str] = frozenset(
    {"prompt_cache_hit_tokens", "prompt_cache_miss_tokens"}
)
_OPENAI_DETAILS_USAGE_FIELDS: frozenset[str] = (
    _BASE_USAGE_FIELDS
    | {
        "prompt_tokens_details",
        "completion_tokens_details",
        "prompt_cached_tokens_details",
    }
    | _CACHE_ALIAS_USAGE_FIELDS
)
_OPENAI_DETAIL_FIELDS: dict[str, frozenset[str]] = {
    "prompt_tokens_details": frozenset({"audio_tokens", "cached_tokens"}),
    "completion_tokens_details": frozenset(
        {
            "accepted_prediction_tokens",
            "audio_tokens",
            "reasoning_tokens",
            "rejected_prediction_tokens",
        }
    ),
    "prompt_cached_tokens_details": frozenset({"audio_tokens"}),
}
_MODEL_PARENT_KIND_BY_STAGE = {
    "memory_ingest": "ingestion_plan",
    "answer": "case",
    "judge": "case",
    "model_readiness": "model_readiness",
}
_MODEL_STAGES_BY_ROLE = {
    ModelRole.MEMORY_EXTRACTION: frozenset({"memory_ingest", "model_readiness"}),
    ModelRole.ANSWER: frozenset({"answer"}),
    ModelRole.JUDGE: frozenset({"judge"}),
}


class _UsageProfileMismatch(ValueError):
    def __init__(self, message: str, *, usage_reference_ids: tuple[str, ...]) -> None:
        super().__init__(message)
        self.usage_reference_ids = usage_reference_ids


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


class OpenAICompatibleModelClient:
    """Dispatch exactly one `/chat/completions` request per runtime attempt."""

    def __init__(
        self,
        *,
        store: ArtifactStorePort,
        base_url: str,
        api_key: str,
        role_binding: ModelRoleBindingV2,
        usage_profile: UsageProfile = "strict-base-v2",
        transport: httpx.AsyncBaseTransport | None = None,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = DEFAULT_MODEL_CALL_TIMEOUT_SECONDS,
        write_timeout_seconds: float = 30.0,
        pool_timeout_seconds: float = 10.0,
        total_timeout_seconds: float | None = None,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("model endpoint and credential are required")
        total_timeout = (
            read_timeout_seconds if total_timeout_seconds is None else total_timeout_seconds
        )
        if not math.isfinite(total_timeout) or total_timeout <= 0:
            raise ValueError("model total timeout must be finite positive")
        role_binding = ModelRoleBindingV2.model_validate(role_binding.model_dump(mode="python"))
        if (
            role_binding.role_status != RoleBindingStatus.SELECTED
            or role_binding.execution_owner != ExecutionOwner.HARNESS
            or role_binding.binding_kind != BindingKind.MODEL_CLIENT
            or role_binding.role not in _MODEL_STAGES_BY_ROLE
            or role_binding.retry_policy_id != "no-retry-v1"
            or role_binding.credential_variable_name is None
            or role_binding.model is None
        ):
            raise ValueError("model client requires one selected no-retry harness role binding")
        if usage_profile not in {"strict-base-v2", "openai-details-v3"}:
            raise ValueError("unknown OpenAI-compatible usage profile")
        self._store = store
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._role_binding = role_binding
        self._model = role_binding.model
        if role_binding.thinking_effort == "not_applicable" or role_binding.thinking_effort is None:
            raise ValueError("model client requires a generative thinking effort")
        self._thinking_effort: ThinkingEffort = role_binding.thinking_effort
        self._usage_profile = usage_profile
        self._total_timeout_seconds = total_timeout
        self._transport = transport or httpx.AsyncHTTPTransport()
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
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

    def thinking_effort_for(self, *, stage: str, role_binding_id: str) -> ThinkingEffort:
        if role_binding_id != self._role_binding.binding_id:
            raise ValueError("model request role binding does not match the client instance")
        if stage not in _MODEL_STAGES_BY_ROLE[self._role_binding.role]:
            raise ValueError(
                f"model request stage {stage} is invalid for role {self._role_binding.role.value}"
            )
        return self._thinking_effort

    async def complete(self, request: ModelRequest) -> ModelReceipt:
        self._validate_request(request)
        self._admit_operation()
        try:
            return await self._complete_once(request)
        finally:
            self._release_operation()

    async def _complete_once(self, request: ModelRequest) -> ModelReceipt:
        payload: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": role, "content": content} for role, content in request.messages],
            "n": request.candidate_count,
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "max_tokens": request.max_output_tokens,
            "stop": list(request.stop) if request.stop is not None else None,
            "reasoning_effort": request.thinking_effort,
        }
        try:
            async with asyncio.timeout(self._total_timeout_seconds):
                response = await self._client.post(self._url, json=payload)
        except asyncio.CancelledError as exc:
            raise ModelCallCancelledUnknownOutcome(
                "model dispatch was cancelled with unknown supplier acceptance"
            ) from exc
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise ModelCallUnknownOutcome(
                "model dispatch timed out with unknown supplier acceptance",
                failure_kind="timeout",
            ) from exc
        except httpx.TransportError as exc:
            raise ModelCallUnknownOutcome(
                "model transport failed with unknown supplier acceptance",
                failure_kind="transport_error",
            ) from exc

        raw_bytes = response.content
        raw_reference = self._seal_raw(raw_bytes)
        if not 200 <= response.status_code < 300:
            rejection = parse_structured_supplier_rejection(
                raw_bytes,
                status_code=response.status_code,
            )
            usage_request = request
            if rejection is not None:
                usage_request = replace(
                    request,
                    attempt_id=infrastructure_supplier_call_id(
                        request.attempt_id,
                        request.supplier_call_ordinal,
                        raw_reference.sha256,
                    ),
                )
            usage_ids = self._seal_unavailable_usage(
                usage_request,
                raw_reference,
                reason=f"supplier_http_status_{response.status_code}",
            )
            if rejection is not None:
                raise ModelSupplierRateLimitRejection(
                    "model supplier returned a structured retry-safe 429 rejection",
                    classification=rejection,
                    raw_reference=raw_reference,
                    raw_response_bytes=raw_bytes,
                    usage_reference_ids=usage_ids,
                )
            raise ModelCallFailure(
                f"model supplier returned HTTP {response.status_code}",
                raw_reference=raw_reference,
                raw_response_bytes=raw_bytes,
                usage_reference_ids=usage_ids,
                retryable=response.status_code == 408 or response.status_code >= 500,
                failure_kind="supplier_error",
                supplier_status_code=response.status_code,
            )
        try:
            document = json.loads(raw_bytes, parse_constant=_reject_json_constant)
            if not isinstance(document, dict):
                raise ValueError("response root must be an object")
        except (TypeError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            usage_ids = self._seal_unavailable_usage(
                request,
                raw_reference,
                reason="malformed_supplier_response",
            )
            raise ModelCallFailure(
                f"model response parse failed: {exc}",
                raw_reference=raw_reference,
                raw_response_bytes=raw_bytes,
                usage_reference_ids=usage_ids,
                retryable=False,
                failure_kind="response_parse_error",
                supplier_status_code=response.status_code,
            ) from exc
        try:
            usage_ids, supplier_output_tokens = self._seal_usage(
                request,
                raw_reference,
                document.get("usage"),
            )
        except _UsageProfileMismatch as exc:
            raise ModelCallFailure(
                f"model usage profile failed: {exc}",
                raw_reference=raw_reference,
                raw_response_bytes=raw_bytes,
                usage_reference_ids=exc.usage_reference_ids,
                retryable=False,
                failure_kind="usage_parse_error",
                supplier_status_code=response.status_code,
            ) from exc
        except (TypeError, ValueError) as exc:
            usage_ids = self._seal_unavailable_usage(
                request,
                raw_reference,
                reason="malformed_supplier_usage",
            )
            raise ModelCallFailure(
                f"model usage parse failed: {exc}",
                raw_reference=raw_reference,
                raw_response_bytes=raw_bytes,
                usage_reference_ids=usage_ids,
                retryable=False,
                failure_kind="usage_parse_error",
                supplier_status_code=response.status_code,
            ) from exc
        if (
            supplier_output_tokens is not None
            and supplier_output_tokens > request.max_output_tokens
        ):
            raise ModelCallFailure(
                "supplier output usage exceeds the requested output ceiling",
                raw_reference=raw_reference,
                raw_response_bytes=raw_bytes,
                usage_reference_ids=usage_ids,
                retryable=False,
                failure_kind="output_contract_error",
                supplier_status_code=response.status_code,
            )
        try:
            candidates, disposition = self._parse_choices(document)
        except (KeyError, TypeError, ValueError) as exc:
            raise ModelCallFailure(
                f"model response parse failed: {exc}",
                raw_reference=raw_reference,
                raw_response_bytes=raw_bytes,
                usage_reference_ids=usage_ids,
                retryable=False,
                failure_kind="response_parse_error",
                supplier_status_code=response.status_code,
            ) from exc
        receipt = ModelReceipt(
            raw_reference=raw_reference,
            output_text=(
                candidates[0].content
                if len(candidates) == 1 and candidates[0].content is not None
                else ""
            ),
            usage_reference_ids=usage_ids,
            model=self._model,
            raw_response_bytes=raw_bytes,
            finish_disposition=disposition,
            candidates=candidates,
            supplier_status_code=response.status_code,
        )
        return receipt

    async def close(self) -> None:
        self._accepting_operations = False
        await self._active_operations_drained.wait()
        async with self._close_lock:
            if not self._closed:
                if self._client.is_closed:
                    await self._transport.aclose()
                else:
                    await self._client.aclose()
                self._closed = True

    def _admit_operation(self) -> None:
        if self._closed or not self._accepting_operations:
            raise RuntimeError("model client is closed")
        self._active_operation_count += 1
        if self._active_operation_count == 1:
            self._active_operations_drained.clear()

    def _release_operation(self) -> None:
        self._active_operation_count -= 1
        if self._active_operation_count == 0:
            self._active_operations_drained.set()

    def _validate_request(self, request: ModelRequest) -> None:
        try:
            expected_parent_kind = _MODEL_PARENT_KIND_BY_STAGE[request.stage]
        except KeyError as exc:
            raise ValueError(f"unsupported model request stage: {request.stage}") from exc
        expected_effort = self.thinking_effort_for(
            stage=request.stage,
            role_binding_id=request.role_binding_id,
        )
        if request.thinking_effort != expected_effort:
            raise ValueError("model request thinking effort differs from its role binding")
        if request.parent_kind != expected_parent_kind:
            raise ValueError(
                f"model request stage {request.stage} requires parent {expected_parent_kind}"
            )
        if request.candidate_count != 1:
            raise ValueError("model request requires exactly one candidate")
        if request.max_output_tokens < 1:
            raise ValueError("model request requires a positive output ceiling")
        if request.temperature != "0" or request.top_p != "1":
            raise ValueError("model request sampling must be temperature=0 and top_p=1")
        if canonical_sha256(request.messages) != request.messages_sha256:
            raise ValueError("model request message hash does not match exact messages")

    def _seal_raw(self, raw_bytes: bytes) -> RawReferenceHandle:
        raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()
        return self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=raw_sha256,
                media_type="application/json",
                compression="gzip",
                payload_bytes=raw_bytes,
            )
        )

    @staticmethod
    def _parse_choices(
        document: dict[str, object],
    ) -> tuple[tuple[ModelCandidate, ...], FinishDisposition]:
        raw_choices = document["choices"]
        if not isinstance(raw_choices, list) or not raw_choices:
            raise ValueError("response choices must be a non-empty list")
        candidates: list[ModelCandidate] = []
        dispositions: list[FinishDisposition] = []
        for raw_choice in raw_choices:
            if not isinstance(raw_choice, dict):
                raise ValueError("response choice must be an object")
            message = raw_choice.get("message")
            if not isinstance(message, dict):
                raise ValueError("response choice requires a message object")
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError("response message content must be text or null")
            tool_present = bool(message.get("tool_calls") or message.get("function_call"))
            reasoning = message.get("reasoning_content")
            if reasoning is not None and not isinstance(reasoning, str):
                raise ValueError("reasoning content must be text or null")
            raw_finish = raw_choice.get("finish_reason")
            if tool_present or raw_finish in {"tool_calls", "function_call"}:
                disposition = FinishDisposition.TOOL_CALL
            elif raw_finish == "stop":
                disposition = FinishDisposition.NORMAL_STOP
            elif raw_finish == "length":
                disposition = FinishDisposition.LENGTH_LIMIT
            elif raw_finish == "content_filter":
                disposition = FinishDisposition.CONTENT_FILTERED
            elif raw_finish is None:
                disposition = FinishDisposition.MISSING_FINISH_REASON
            else:
                raise ValueError(f"unknown finish reason: {raw_finish!r}")
            dispositions.append(disposition)
            candidates.append(
                ModelCandidate(
                    content=content,
                    tool_call_present=tool_present,
                    complete=True,
                    reasoning_content=reasoning,
                )
            )
        disposition = dispositions[0]
        if any(item != disposition for item in dispositions):
            raise ValueError("response choices have conflicting finish outcomes")
        return tuple(candidates), disposition

    def _seal_usage(
        self,
        request: ModelRequest,
        raw_reference: RawReferenceHandle,
        raw_usage: object,
    ) -> tuple[tuple[str, ...], int | None]:
        if raw_usage is None:
            return (
                self._seal_unavailable_usage(
                    request,
                    raw_reference,
                    reason="supplier_usage_missing",
                ),
                None,
            )
        if not isinstance(raw_usage, dict):
            raise ValueError("supplier usage must be an object")
        if self._usage_profile == "openai-details-v3":
            return self._seal_openai_details_usage(
                request,
                raw_reference,
                raw_usage,
            )
        raw_usage_fields = set(raw_usage)
        if not all(isinstance(field, str) for field in raw_usage_fields):
            raise ValueError("supplier usage field names must be strings")
        usage_fields = cast(set[str], raw_usage_fields)
        if not _BASE_USAGE_FIELDS.issubset(usage_fields):
            raise ValueError("supplier usage is missing frozen base fields")
        values = tuple(
            raw_usage.get(name)
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
            )
        )
        if not all(type(value) is int and value >= 0 for value in values):
            raise ValueError("supplier usage base token fields must be non-negative integers")
        input_tokens, output_tokens, total_tokens = values
        unknown_fields = tuple(sorted(usage_fields - _BASE_USAGE_FIELDS))
        if unknown_fields:
            reason = f"supplier_usage_unknown_fields:{','.join(unknown_fields)}"
            partial_usage = self._usage_record(
                request,
                raw_reference,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                proof_status=ProofStatus.MEASURED_PARTIAL,
                reason=reason,
            )
            usage_ids = (self._seal_usage_record(partial_usage),)
            raise _UsageProfileMismatch(
                "supplier usage fields differ from the frozen base profile",
                usage_reference_ids=usage_ids,
            )
        usage = self._usage_record(
            request,
            raw_reference,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            proof_status=ProofStatus.MEASURED_COMPLETE,
            reason=None,
        )
        return (self._seal_usage_record(usage),), output_tokens

    def _seal_openai_details_usage(
        self,
        request: ModelRequest,
        raw_reference: RawReferenceHandle,
        raw_usage: dict[object, object],
    ) -> tuple[tuple[str, ...], int | None]:
        raw_usage_fields = set(raw_usage)
        if not all(isinstance(field, str) for field in raw_usage_fields):
            raise ValueError("supplier usage field names must be strings")
        usage_fields = cast(set[str], raw_usage_fields)
        if not _BASE_USAGE_FIELDS.issubset(usage_fields):
            raise ValueError("supplier usage is missing frozen base fields")
        base_values = tuple(
            raw_usage.get(name) for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        )
        if not all(type(value) is int and value >= 0 for value in base_values):
            raise ValueError("supplier usage base token fields must be non-negative integers")
        input_tokens, output_tokens, total_tokens = cast(tuple[int, int, int], base_values)
        cache_alias_values: tuple[int, int] | None = None
        present_cache_aliases = usage_fields & _CACHE_ALIAS_USAGE_FIELDS
        if present_cache_aliases:
            if present_cache_aliases != _CACHE_ALIAS_USAGE_FIELDS:
                raise ValueError("supplier cache aliases must be provided together")
            raw_cache_alias_values = tuple(
                raw_usage.get(name)
                for name in ("prompt_cache_hit_tokens", "prompt_cache_miss_tokens")
            )
            if not all(type(value) is int and value >= 0 for value in raw_cache_alias_values):
                raise ValueError("supplier cache aliases must be non-negative integers")
            cache_alias_values = cast(tuple[int, int], raw_cache_alias_values)
            if sum(cache_alias_values) != input_tokens:
                raise ValueError("supplier cache aliases disagree with prompt tokens")
        detail_values: dict[str, int | None] = {
            "cached_input_tokens": None,
            "reasoning_tokens": None,
        }
        cached_input_raw_path: str | None = None
        unknown_paths: list[str] = [
            str(field) for field in sorted(usage_fields - _OPENAI_DETAILS_USAGE_FIELDS)
        ]
        for container_name, allowed_fields in _OPENAI_DETAIL_FIELDS.items():
            raw_container = raw_usage.get(container_name)
            if raw_container is None:
                continue
            if not isinstance(raw_container, dict) or not all(
                isinstance(field, str) for field in raw_container
            ):
                raise ValueError(f"supplier usage {container_name} must be an object")
            detail_container = cast(dict[str, object], raw_container)
            unknown_paths.extend(
                f"{container_name}.{field}"
                for field in sorted(set(detail_container) - allowed_fields)
            )
            for field, value in detail_container.items():
                if field in allowed_fields and (type(value) is not int or value < 0):
                    raise ValueError(
                        f"supplier usage {container_name}.{field} must be a non-negative integer"
                    )
                tracked = (container_name, field) in {
                    ("prompt_tokens_details", "cached_tokens"),
                    ("completion_tokens_details", "reasoning_tokens"),
                }
                if field in allowed_fields and not tracked and value != 0:
                    unknown_paths.append(f"unsupported_nonzero:{container_name}.{field}")
            if container_name == "prompt_tokens_details":
                cached = detail_container.get("cached_tokens")
                if type(cached) is int:
                    detail_values["cached_input_tokens"] = cached
                    cached_input_raw_path = "usage.prompt_tokens_details.cached_tokens"
            if container_name == "completion_tokens_details":
                reasoning = detail_container.get("reasoning_tokens")
                if type(reasoning) is int:
                    detail_values["reasoning_tokens"] = reasoning
        if cache_alias_values is not None:
            cache_hit_tokens, _ = cache_alias_values
            cached_input_tokens = detail_values["cached_input_tokens"]
            if cached_input_tokens is not None and cached_input_tokens != cache_hit_tokens:
                raise ValueError("supplier cache aliases disagree with nested cache details")
            if cached_input_tokens is None:
                detail_values["cached_input_tokens"] = cache_hit_tokens
                cached_input_raw_path = "usage.prompt_cache_hit_tokens"
        covered_dimensions = [
            "input_tokens",
            "visible_output_tokens",
            "supplier_reported_total_tokens",
        ]
        raw_field_paths = [
            ("input_tokens", "usage.prompt_tokens"),
            ("visible_output_tokens", "usage.completion_tokens"),
            ("supplier_reported_total_tokens", "usage.total_tokens"),
        ]
        for dimension, raw_path in (
            ("cached_input_tokens", cached_input_raw_path),
            ("reasoning_tokens", "usage.completion_tokens_details.reasoning_tokens"),
        ):
            if detail_values[dimension] is not None:
                if raw_path is None:
                    raise AssertionError("measured token detail requires a raw field path")
                covered_dimensions.append(dimension)
                raw_field_paths.append((dimension, raw_path))
        unavailable_dimensions = tuple(
            dimension
            for dimension in ("cached_input_tokens", "reasoning_tokens")
            if detail_values[dimension] is None
        )
        limitations = [
            *(f"unknown_field:{path}" for path in unknown_paths),
            *(f"unavailable:{dimension}" for dimension in unavailable_dimensions),
        ]
        proof_status = (
            ProofStatus.MEASURED_COMPLETE
            if not unavailable_dimensions and not unknown_paths
            else ProofStatus.MEASURED_PARTIAL
        )
        reason = ",".join(limitations) if limitations else None
        usage_id = canonical_sha256(
            [
                "oamb-openai-compatible-details-usage-v3",
                request.attempt_id,
                raw_reference.sha256,
                input_tokens,
                output_tokens,
                total_tokens,
                detail_values,
                proof_status,
                reason,
            ]
        )
        usage = TokenUsageRecordV3(
            usage_record_id=usage_id,
            attempt_id=request.attempt_id,
            parent_kind=request.parent_kind,
            parent_id=request.parent_id,
            stage=TokenStageV2(request.stage),
            operation_kind="openai_chat_completion",
            token_domain=TokenDomain.EXTERNAL_LLM,
            measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
            input_tokens=input_tokens,
            visible_output_tokens=output_tokens,
            supplier_reported_total_tokens=total_tokens,
            context_view_tokens=None,
            cached_input_tokens=detail_values["cached_input_tokens"],
            reasoning_tokens=detail_values["reasoning_tokens"],
            model=self._model,
            meter_schema_id="openai-chat-usage-details-v1",
            raw_field_paths=tuple(raw_field_paths),
            covered_dimensions=tuple(covered_dimensions),
            unavailable_dimensions=unavailable_dimensions,
            not_applicable_dimensions=(),
            inclusion_relationships=(),
            token_measurement_complete=not unavailable_dimensions,
            billing_complete=False,
            proof_status=proof_status,
            reason=reason,
            raw_response_ref=raw_reference.sha256,
        )
        usage_ids = (self._seal_usage_record(usage),)
        if unknown_paths:
            raise _UsageProfileMismatch(
                "supplier usage fields differ from the extended profile",
                usage_reference_ids=usage_ids,
            )
        return usage_ids, output_tokens

    def _seal_unavailable_usage(
        self,
        request: ModelRequest,
        raw_reference: RawReferenceHandle,
        *,
        reason: str,
    ) -> tuple[str, ...]:
        usage = self._usage_record(
            request,
            raw_reference,
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            proof_status=ProofStatus.UNAVAILABLE,
            reason=reason,
        )
        return (self._seal_usage_record(usage),)

    @staticmethod
    def _usage_record(
        request: ModelRequest,
        raw_reference: RawReferenceHandle,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        proof_status: ProofStatus,
        reason: str | None,
    ) -> TokenUsageRecordV2:
        usage_id = canonical_sha256(
            [
                "oamb-openai-compatible-usage-v1",
                request.attempt_id,
                raw_reference.sha256,
                input_tokens,
                output_tokens,
                total_tokens,
                proof_status,
                reason,
            ]
        )
        return TokenUsageRecordV2(
            usage_record_id=usage_id,
            attempt_id=request.attempt_id,
            parent_kind=request.parent_kind,
            parent_id=request.parent_id,
            stage=TokenStageV2(request.stage),
            operation_kind="openai_chat_completion",
            token_domain=TokenDomain.EXTERNAL_LLM,
            measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
            input_tokens=input_tokens,
            visible_output_tokens=output_tokens,
            supplier_reported_total_tokens=total_tokens,
            context_view_tokens=None,
            proof_status=proof_status,
            reason=reason,
            raw_response_ref=raw_reference.sha256,
        )

    def _seal_usage_record(self, usage: TokenUsageRecordV2 | TokenUsageRecordV3) -> str:
        canonical = canonical_json_bytes(usage)
        self._store.seal_source_record(
            ArtifactWriteRequest(
                record_id=usage.usage_record_id,
                relative_path=f"source/usage/{usage.usage_record_id}.json",
                canonical_sha256=hashlib.sha256(canonical).hexdigest(),
                canonical_bytes=canonical,
            )
        )
        return usage.usage_record_id
