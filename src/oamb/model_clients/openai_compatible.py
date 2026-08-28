"""Single-dispatch OpenAI Chat Completions transport with sealed evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from typing import Literal

import httpx

from oamb.contracts.accounting import (
    ProofStatus,
    TokenDomain,
    TokenMeasurementSource,
    TokenStageV2,
    TokenUsageRecordV2,
)
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
    RawPayloadSealRequest,
    RawReferenceHandle,
)
from oamb.contracts.specifications import (
    BindingKind,
    ExecutionOwner,
    ModelRole,
    ModelRoleBindingV2,
    RoleBindingStatus,
)

RuntimeModelPolicy = Literal["record", "require_match"]

_CORE_REQUEST_FIELDS = {
    "max_tokens",
    "messages",
    "model",
    "n",
    "response_format",
    "stop",
    "temperature",
    "tools",
    "top_p",
}
_BASE_USAGE_FIELDS = {"prompt_tokens", "completion_tokens", "total_tokens"}
_MODEL_PARENT_KIND_BY_STAGE = {
    "memory_ingest": "ingestion_plan",
    "answer": "case",
    "judge": "case",
    "quality_review": "phase_review",
    "model_readiness": "model_readiness",
}
_MODEL_STAGES_BY_ROLE = {
    ModelRole.MEMORY_EXTRACTION: frozenset({"memory_ingest", "model_readiness"}),
    ModelRole.ANSWER: frozenset({"answer"}),
    ModelRole.JUDGE: frozenset({"judge"}),
    ModelRole.QUALITY_REVIEW: frozenset({"quality_review"}),
}


class _RuntimeIdentityMismatch(ValueError):
    pass


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
        runtime_model_policy: RuntimeModelPolicy,
        reasoning_control: tuple[str, str],
        transport: httpx.AsyncBaseTransport | None = None,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 120.0,
        write_timeout_seconds: float = 30.0,
        pool_timeout_seconds: float = 10.0,
    ) -> None:
        if not base_url or not api_key:
            raise ValueError("model endpoint and credential are required")
        role_binding = ModelRoleBindingV2.model_validate(role_binding.model_dump(mode="python"))
        if (
            role_binding.role_status != RoleBindingStatus.SELECTED
            or role_binding.execution_owner != ExecutionOwner.HARNESS
            or role_binding.binding_kind != BindingKind.MODEL_CLIENT
            or role_binding.role not in _MODEL_STAGES_BY_ROLE
            or role_binding.retry_policy_id != "no-retry-v1"
            or role_binding.credential_variable_name is None
            or role_binding.configured_model is None
            or role_binding.resolved_model is None
        ):
            raise ValueError("model client requires one selected no-retry harness role binding")
        if runtime_model_policy not in {"record", "require_match"}:
            raise ValueError("runtime model policy must be record or require_match")
        if len(reasoning_control) != 2 or not reasoning_control[0] or not reasoning_control[1]:
            raise ValueError("reasoning control requires one explicit wire field and value")
        reasoning_field = reasoning_control[0]
        if (
            re.fullmatch(r"[a-z][a-z0-9_]*", reasoning_field) is None
            or reasoning_field in _CORE_REQUEST_FIELDS
        ):
            raise ValueError("reasoning control field is invalid or reserved")
        self._store = store
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._role_binding = role_binding
        self._configured_model = role_binding.configured_model
        self._expected_runtime_model = role_binding.resolved_model
        self._runtime_model_policy = runtime_model_policy
        self._reasoning_control = reasoning_control
        self._client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(
                connect=connect_timeout_seconds,
                read=read_timeout_seconds,
                write=write_timeout_seconds,
                pool=pool_timeout_seconds,
            ),
            transport=transport,
            follow_redirects=False,
        )
        self._accepting_operations = True
        self._closed = False
        self._operation_lock = asyncio.Lock()

    async def complete(self, request: ModelRequest) -> ModelReceipt:
        if not self._accepting_operations:
            raise RuntimeError("model client is closed")
        async with self._operation_lock:
            if not self._accepting_operations:
                raise RuntimeError("model client is closed")
            return await self._complete_once(request)

    async def _complete_once(self, request: ModelRequest) -> ModelReceipt:
        self._validate_request(request)
        reasoning_field, reasoning_value = self._reasoning_control
        payload: dict[str, object] = {
            "model": self._configured_model,
            "messages": [{"role": role, "content": content} for role, content in request.messages],
            "n": request.candidate_count,
            "temperature": float(request.temperature),
            "top_p": float(request.top_p),
            "max_tokens": request.max_output_tokens,
            "stop": list(request.stop) if request.stop is not None else None,
            reasoning_field: reasoning_value,
        }
        if request.stage == "quality_review":
            payload["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.post(self._url, json=payload)
        except asyncio.CancelledError as exc:
            raise ModelCallCancelledUnknownOutcome(
                "model dispatch was cancelled with unknown supplier acceptance"
            ) from exc
        except httpx.TimeoutException as exc:
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
            usage_ids = self._seal_unavailable_usage(
                request,
                raw_reference,
                reason=f"supplier_http_status_{response.status_code}",
            )
            raise ModelCallFailure(
                f"model supplier returned HTTP {response.status_code}",
                raw_reference=raw_reference,
                usage_reference_ids=usage_ids,
                retryable=response.status_code in {408, 429} or response.status_code >= 500,
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
                usage_reference_ids=usage_ids,
                retryable=False,
                failure_kind="response_parse_error",
                supplier_status_code=response.status_code,
            ) from exc
        try:
            runtime_model, runtime_status = self._runtime_identity(document)
        except _RuntimeIdentityMismatch as exc:
            raise ModelCallFailure(
                str(exc),
                raw_reference=raw_reference,
                usage_reference_ids=usage_ids,
                retryable=False,
                failure_kind="runtime_identity_error",
                supplier_status_code=response.status_code,
            ) from exc
        except (TypeError, ValueError) as exc:
            raise ModelCallFailure(
                f"model runtime identity parse failed: {exc}",
                raw_reference=raw_reference,
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
            raw_response_bytes=raw_bytes,
            finish_disposition=disposition,
            candidates=candidates,
            runtime_model=runtime_model,
            runtime_identity_status=runtime_status,
            supplier_status_code=response.status_code,
        )
        if request.stage == "quality_review":
            self._validate_quality_review(receipt)
        return receipt

    async def close(self) -> None:
        self._accepting_operations = False
        async with self._operation_lock:
            if not self._closed:
                await self._client.aclose()
                self._closed = True

    def _validate_request(self, request: ModelRequest) -> None:
        try:
            expected_parent_kind = _MODEL_PARENT_KIND_BY_STAGE[request.stage]
        except KeyError as exc:
            raise ValueError(f"unsupported model request stage: {request.stage}") from exc
        if request.role_binding_id != self._role_binding.binding_id:
            raise ValueError("model request role binding does not match the client instance")
        if request.stage not in _MODEL_STAGES_BY_ROLE[self._role_binding.role]:
            raise ValueError(
                f"model request stage {request.stage} is invalid for role "
                f"{self._role_binding.role.value}"
            )
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
        if not request.reasoning_disabled:
            raise ValueError("model request must explicitly disable reasoning")
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

    def _runtime_identity(
        self, document: dict[str, object]
    ) -> tuple[str | None, Literal["matched", "recorded", "unattested", "mismatch"]]:
        runtime_model = document.get("model")
        if runtime_model is None:
            if self._runtime_model_policy == "require_match":
                raise _RuntimeIdentityMismatch("required runtime model identity is missing")
            return None, "unattested"
        if not isinstance(runtime_model, str) or not runtime_model:
            raise ValueError("runtime model identity must be non-empty text")
        if runtime_model == self._expected_runtime_model:
            return runtime_model, "matched"
        if self._runtime_model_policy == "record":
            return runtime_model, "recorded"
        raise _RuntimeIdentityMismatch(
            f"runtime model identity mismatch: expected={self._expected_runtime_model!r}, "
            f"runtime={runtime_model!r}"
        )

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
        usage_fields = set(raw_usage)
        if not all(isinstance(field, str) for field in usage_fields):
            raise ValueError("supplier usage field names must be strings")
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

    def _seal_usage_record(self, usage: TokenUsageRecordV2) -> str:
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

    @staticmethod
    def _validate_quality_review(receipt: ModelReceipt) -> None:
        if receipt.finish_disposition != FinishDisposition.NORMAL_STOP:
            raise ModelCallFailure(
                "quality review requires a normal stop",
                raw_reference=receipt.raw_reference,
                usage_reference_ids=receipt.usage_reference_ids,
                retryable=False,
                failure_kind="output_contract_error",
                supplier_status_code=receipt.supplier_status_code,
            )
        if len(receipt.candidates) != 1:
            raise ModelCallFailure(
                "quality review requires exactly one candidate",
                raw_reference=receipt.raw_reference,
                usage_reference_ids=receipt.usage_reference_ids,
                retryable=False,
                failure_kind="output_contract_error",
                supplier_status_code=receipt.supplier_status_code,
            )
        candidate = receipt.candidates[0]
        if candidate.content is None or candidate.tool_call_present or not candidate.complete:
            raise ModelCallFailure(
                "quality review requires complete text without tool calls",
                raw_reference=receipt.raw_reference,
                usage_reference_ids=receipt.usage_reference_ids,
                retryable=False,
                failure_kind="output_contract_error",
                supplier_status_code=receipt.supplier_status_code,
            )
        try:
            document = json.loads(candidate.content, parse_constant=_reject_json_constant)
            if not isinstance(document, dict):
                raise ValueError("quality review requires a JSON object")
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ModelCallFailure(
                f"quality review requires a complete JSON object: {exc}",
                raw_reference=receipt.raw_reference,
                usage_reference_ids=receipt.usage_reference_ids,
                retryable=False,
                failure_kind="output_contract_error",
                supplier_status_code=receipt.supplier_status_code,
            ) from exc
