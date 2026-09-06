from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from typing import Any, cast

import httpx
import pytest

from oamb.contracts.ids import attempt_id, canonical_sha256
from oamb.contracts.ports import (
    ArtifactReadRequest,
    ArtifactSealReceipt,
    ArtifactWriteRequest,
    FinishDisposition,
    ModelCallFailure,
    ModelCallUnknownOutcome,
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
from oamb.model_clients.openai_compatible import (
    OpenAICompatibleModelClient,
    UsageProfile,
)


class CapturingStore:
    def __init__(self) -> None:
        self.raw: list[RawPayloadSealRequest] = []
        self.records: list[ArtifactWriteRequest] = []

    def seal_raw(self, request: RawPayloadSealRequest) -> RawReferenceHandle:
        self.raw.append(request)
        return RawReferenceHandle(request.sha256)

    def seal_source_record(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        self.records.append(request)
        return ArtifactSealReceipt(
            record_id=request.record_id,
            canonical_sha256=request.canonical_sha256,
        )

    def seal_checkpoint(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        return self.seal_source_record(request)

    def seal_source_manifest(self, request: ArtifactWriteRequest) -> ArtifactSealReceipt:
        return self.seal_source_record(request)

    def read_verified(self, request: ArtifactReadRequest) -> bytes:
        raise AssertionError(f"unexpected read: {request.relative_path}")


def _request(*, stage: str = "answer", thinking_effort: ThinkingEffort = "low") -> ModelRequest:
    messages = (("system", "follow the contract"), ("user", "hello"))
    return ModelRequest(
        attempt_id="a" * 64,
        parent_kind="case",
        parent_id="case-1",
        stage=stage,
        role_binding_id=f"{stage}-binding",
        messages_sha256=canonical_sha256(messages),
        messages=messages,
        output_contract_id="lme-answer-text-v1",
        max_output_tokens=None,
        candidate_count=1,
        temperature="0",
        top_p="1",
        thinking_effort=thinking_effort,
        stop=None,
    )


def test_model_request_fingerprint_binds_output_and_sampling_controls() -> None:
    request = _request()

    assert request.request_fingerprint != request.messages_sha256
    assert (
        request.request_fingerprint
        != replace(
            request,
            max_output_tokens=128,
        ).request_fingerprint
    )
    assert (
        request.request_fingerprint
        != replace(
            request,
            thinking_effort="high",
        ).request_fingerprint
    )


def test_model_request_attempt_identity_binds_the_full_request_fingerprint() -> None:
    messages = (("user", "hello"),)
    request = ModelRequest.for_attempt(
        ordinal=2,
        parent_kind="case",
        parent_id="case-1",
        stage="answer",
        role_binding_id="answer-binding",
        messages_sha256=canonical_sha256(messages),
        messages=messages,
        thinking_effort="low",
    )

    assert request.attempt_id == attempt_id(
        "case-1",
        "answer",
        2,
        request.request_fingerprint,
    )


@pytest.mark.parametrize("thinking_effort", ["low", "high", "max"])
def test_model_request_accepts_supported_thinking_effort(
    thinking_effort: ThinkingEffort,
) -> None:
    assert _request(thinking_effort=thinking_effort).thinking_effort == thinking_effort


@pytest.mark.parametrize("thinking_effort", ["none", "invalid"])
def test_model_request_rejects_unsupported_thinking_effort(thinking_effort: str) -> None:
    with pytest.raises(ValueError, match="thinking effort"):
        _request(thinking_effort=cast(ThinkingEffort, thinking_effort))


def _binding(
    *,
    role: ModelRole = ModelRole.ANSWER,
    binding_id: str | None = None,
    model: str = "answer-model",
    thinking_effort: ThinkingEffort = "low",
) -> ModelRoleBindingV2:
    default_binding_ids = {
        ModelRole.MEMORY_EXTRACTION: "memory_ingest-binding",
        ModelRole.ANSWER: "answer-binding",
        ModelRole.JUDGE: "judge-binding",
    }
    return ModelRoleBindingV2(
        binding_id=binding_id or default_binding_ids[role],
        role=role,
        role_status=RoleBindingStatus.SELECTED,
        execution_owner=ExecutionOwner.HARNESS,
        binding_kind=BindingKind.MODEL_CLIENT,
        provider="fixture-provider",
        endpoint_reference="fixture-model-endpoint",
        credential_variable_name="OAMB_FIXTURE_MODEL_API_KEY",
        model=model,
        thinking_effort=thinking_effort,
        parameters_fingerprint="1" * 64,
        retry_policy_id="no-retry-v1",
        configuration_fingerprint="2" * 64,
        redacted_endpoint_fingerprint="3" * 64,
    )


def _client(
    store: CapturingStore,
    handler: Any,
    *,
    model: str = "answer-model",
    role: ModelRole = ModelRole.ANSWER,
    binding_id: str | None = None,
    usage_profile: UsageProfile = "strict-base-v2",
    thinking_effort: ThinkingEffort = "low",
) -> OpenAICompatibleModelClient:
    return OpenAICompatibleModelClient(
        store=store,
        base_url="https://models.example/v1",
        api_key="secret",
        role_binding=_binding(
            role=role,
            binding_id=binding_id,
            model=model,
            thinking_effort=thinking_effort,
        ),
        usage_profile=usage_profile,
        transport=httpx.MockTransport(handler),
    )


def test_model_client_explicitly_disables_environment_proxy_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    original = httpx.AsyncClient

    class RecordingClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", RecordingClient)
    try:
        OpenAICompatibleModelClient(
            store=CapturingStore(),
            base_url="https://api.deepseek.com",
            api_key="secret",
            role_binding=_binding(),
        )
    finally:
        monkeypatch.setattr(httpx, "AsyncClient", original)
    assert captured["trust_env"] is False


@pytest.mark.asyncio
async def test_answer_omits_output_ceiling_and_seals_raw_usage() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "id": "response-1",
                "model": "answer-model@runtime",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 9000,
                    "total_tokens": 9011,
                },
            },
        )

    store = CapturingStore()
    client = OpenAICompatibleModelClient(
        store=store,
        base_url="https://models.example/v1",
        api_key="secret",
        role_binding=_binding(),
        transport=httpx.MockTransport(handler),
    )

    receipt = await client.complete(_request(thinking_effort="low"))
    sent = json.loads(calls[0].content)

    assert len(calls) == 1
    assert calls[0].url == httpx.URL("https://models.example/v1/chat/completions")
    assert sent["model"] == "answer-model"
    assert sent["n"] == 1
    assert "max_tokens" not in sent
    assert sent["reasoning_effort"] == "low"
    assert "tools" not in sent
    assert receipt.output_text == "answer"
    assert receipt.finish_disposition == FinishDisposition.NORMAL_STOP
    assert receipt.model == "answer-model"
    assert receipt.supplier_status_code == 200
    assert store.raw[0].sha256 == hashlib.sha256(store.raw[0].payload_bytes).hexdigest()
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["input_tokens"] == 11
    assert usage["visible_output_tokens"] == 9000
    assert usage["supplier_reported_total_tokens"] == 9011
    await client.close()


@pytest.mark.asyncio
async def test_judge_sends_request_bound_high_effort() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "judge-model",
                "choices": [{"index": 0, "message": {"content": "yes"}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2048,
                    "total_tokens": 2056,
                },
            },
        )

    client = _client(
        CapturingStore(),
        handler,
        model="judge-model",
        role=ModelRole.JUDGE,
        thinking_effort="high",
    )

    await client.complete(_request(stage="judge", thinking_effort="high"))

    sent = json.loads(calls[0].content)
    assert sent["reasoning_effort"] == "high"
    assert "max_tokens" not in sent
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stage", "role", "parent_kind", "thinking_effort"),
    (
        ("memory_ingest", ModelRole.MEMORY_EXTRACTION, "ingestion_plan", "low"),
        ("answer", ModelRole.ANSWER, "case", "low"),
        ("judge", ModelRole.JUDGE, "case", "high"),
    ),
)
async def test_non_probe_model_requests_reject_an_output_ceiling_before_dispatch(
    stage: str,
    role: ModelRole,
    parent_kind: str,
    thinking_effort: ThinkingEffort,
) -> None:
    calls = 0

    def handler(_request_value: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    binding_id = f"{stage}-binding"
    client = _client(
        CapturingStore(),
        handler,
        role=role,
        binding_id=binding_id,
        thinking_effort=thinking_effort,
    )
    request = replace(
        _request(stage=stage, thinking_effort=thinking_effort),
        parent_kind=cast(Any, parent_kind),
        parent_id="parent-1",
        role_binding_id=binding_id,
        max_output_tokens=128,
    )

    with pytest.raises(ValueError, match="non-probe model requests must omit"):
        await client.complete(request)
    assert calls == 0
    await client.close()


@pytest.mark.asyncio
async def test_request_effort_must_match_resolved_role_binding_before_dispatch() -> None:
    calls: list[httpx.Request] = []

    def capture_request(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200)

    client = _client(
        CapturingStore(),
        capture_request,
        thinking_effort="low",
    )

    with pytest.raises(ValueError, match="thinking effort.*role binding"):
        await client.complete(_request(thinking_effort="high"))

    assert calls == []
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("finish_reason", "message_extra", "expected"),
    [
        ("stop", {}, FinishDisposition.NORMAL_STOP),
        ("length", {}, FinishDisposition.LENGTH_LIMIT),
        ("content_filter", {}, FinishDisposition.CONTENT_FILTERED),
        (None, {}, FinishDisposition.MISSING_FINISH_REASON),
        ("tool_calls", {"tool_calls": [{"id": "tool-1"}]}, FinishDisposition.TOOL_CALL),
    ],
)
async def test_openai_finish_outcomes_are_closed_and_explicit(
    finish_reason: str | None,
    message_extra: dict[str, object],
    expected: FinishDisposition,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"content": "answer"} | message_extra,
                        "finish_reason": finish_reason,
                    }
                ],
            },
        )

    store = CapturingStore()
    client = _client(store, handler)

    receipt = await client.complete(_request())

    assert receipt.finish_disposition == expected
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["proof_status"] == "unavailable"
    assert usage["input_tokens"] is None
    assert usage["reason"] == "supplier_usage_missing"
    await client.close()


@pytest.mark.asyncio
async def test_supplier_error_is_single_dispatch_with_exact_error_body() -> None:
    calls = 0
    error_body = b'{"error":{"message":"busy"}}'

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(429, content=error_body)

    store = CapturingStore()
    client = _client(store, handler)

    with pytest.raises(ModelCallFailure) as failure:
        await client.complete(_request())

    assert calls == 1
    assert failure.value.retryable is False
    assert not isinstance(failure.value, ModelSupplierRateLimitRejection)
    assert failure.value.failure_kind == "supplier_error"
    assert failure.value.supplier_status_code == 429
    assert store.raw[0].payload_bytes == error_body
    await client.close()


@pytest.mark.asyncio
async def test_exact_structured_429_produces_retry_safe_typed_rejection() -> None:
    error_body = json.dumps(
        {
            "error": {
                "origin": "model_supplier",
                "failure_kind": "rate_limited",
                "status": 429,
                "acceptance": "not_accepted",
                "provider_mutation": "none",
                "retryable": True,
                "internal_retry_count": 0,
            }
        },
        separators=(",", ":"),
    ).encode()

    def handler(_request_value: httpx.Request) -> httpx.Response:
        return httpx.Response(429, content=error_body)

    store = CapturingStore()
    client = _client(store, handler)

    with pytest.raises(ModelSupplierRateLimitRejection) as failure:
        await client.complete(_request())

    assert failure.value.classification.provider_mutation == "none"
    assert failure.value.classification.internal_retry_count == 0
    assert failure.value.usage_reference_ids
    assert store.raw[0].payload_bytes == error_body
    await client.close()


@pytest.mark.asyncio
async def test_response_model_is_retained_only_in_raw_evidence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "different-runtime-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
            },
        )

    store = CapturingStore()
    client = _client(
        store,
        handler,
        model="configured-model",
    )

    receipt = await client.complete(_request())

    assert receipt.model == "configured-model"
    assert b'"model":"different-runtime-model"' in store.raw[0].payload_bytes
    await client.close()


@pytest.mark.asyncio
async def test_response_model_is_not_required() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}]
            },
        )

    client = _client(CapturingStore(), handler)

    receipt = await client.complete(_request())

    assert receipt.model == "answer-model"
    await client.close()


@pytest.mark.asyncio
async def test_invalid_stage_is_rejected_before_dispatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    client = _client(CapturingStore(), handler)

    with pytest.raises(ValueError, match="stage"):
        await client.complete(replace(_request(), stage="unknown-stage"))
    assert calls == 0
    await client.close()


@pytest.mark.asyncio
async def test_answer_client_rejects_a_judge_request_before_dispatch() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={})

    client = _client(CapturingStore(), handler)
    judge_request = replace(
        _request(),
        stage="judge",
        role_binding_id="answer-binding",
    )

    with pytest.raises(ValueError, match="role"):
        await client.complete(judge_request)
    assert calls == 0
    await client.close()


@pytest.mark.asyncio
async def test_model_readiness_usage_supports_its_explicit_parent_kind() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [{"index": 0, "message": {"content": "ready"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )

    store = CapturingStore()
    client = _client(
        store,
        handler,
        role=ModelRole.MEMORY_EXTRACTION,
        binding_id="model_readiness-binding",
    )
    request = replace(
        _request(),
        parent_kind="model_readiness",
        parent_id="readiness-1",
        stage="model_readiness",
        role_binding_id="model_readiness-binding",
        max_output_tokens=128,
    )

    await client.complete(request)

    assert json.loads(calls[0].content)["max_tokens"] == 128
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["parent_kind"] == "model_readiness"
    assert usage["stage"] == "model_readiness"
    await client.close()


@pytest.mark.asyncio
async def test_malformed_output_preserves_valid_supplier_usage_and_status() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            },
        )

    store = CapturingStore()
    client = _client(store, handler)

    with pytest.raises(ModelCallFailure) as failure:
        await client.complete(_request())

    assert failure.value.failure_kind == "response_parse_error"
    assert failure.value.supplier_status_code == 200
    assert len(store.records) == 1
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["proof_status"] == "measured_complete"
    assert usage["supplier_reported_total_tokens"] == 9
    await client.close()


@pytest.mark.asyncio
async def test_unknown_supplier_usage_fields_fail_closed_without_hiding_the_raw_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                    "reasoning_tokens": 1,
                },
            },
        )

    store = CapturingStore()
    client = _client(store, handler)

    with pytest.raises(ModelCallFailure) as failure:
        await client.complete(_request())

    assert failure.value.failure_kind == "usage_parse_error"
    assert store.raw
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["proof_status"] == "measured_partial"
    assert usage["input_tokens"] == 7
    assert usage["visible_output_tokens"] == 2
    assert usage["supplier_reported_total_tokens"] == 9
    assert usage["reason"] == "supplier_usage_unknown_fields:reasoning_tokens"
    await client.close()


@pytest.mark.asyncio
async def test_additive_openai_details_profile_preserves_cached_and_reasoning_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 87,
                    "completion_tokens": 11,
                    "total_tokens": 98,
                    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 7},
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 0,
                        "audio_tokens": 0,
                        "reasoning_tokens": 3,
                        "rejected_prediction_tokens": 0,
                    },
                    "prompt_cached_tokens_details": {"audio_tokens": 0},
                },
            },
        )

    store = CapturingStore()
    client = _client(store, handler, usage_profile="openai-details-v3")

    receipt = await client.complete(_request())

    assert receipt.usage_reference_ids
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["schema_version"] == 3
    assert usage["model"] == "answer-model"
    assert usage["cached_input_tokens"] == 7
    assert usage["reasoning_tokens"] == 3
    assert usage["covered_dimensions"] == [
        "input_tokens",
        "visible_output_tokens",
        "supplier_reported_total_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
    ]
    assert usage["raw_field_paths"][-2:] == [
        ["cached_input_tokens", "usage.prompt_tokens_details.cached_tokens"],
        ["reasoning_tokens", "usage.completion_tokens_details.reasoning_tokens"],
    ]
    assert usage["proof_status"] == "measured_complete"
    assert usage["billing_complete"] is False
    await client.close()


@pytest.mark.asyncio
async def test_additive_openai_details_profile_accepts_consistent_cache_aliases() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 87,
                    "completion_tokens": 11,
                    "total_tokens": 98,
                    "prompt_tokens_details": {"cached_tokens": 7},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                    "prompt_cache_hit_tokens": 7,
                    "prompt_cache_miss_tokens": 80,
                },
            },
        )

    store = CapturingStore()
    client = _client(store, handler, usage_profile="openai-details-v3")

    receipt = await client.complete(_request())

    assert receipt.usage_reference_ids
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["proof_status"] == "measured_complete"
    assert usage["cached_input_tokens"] == 7
    assert usage["reasoning_tokens"] == 3
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cache_hit_tokens", "cache_miss_tokens", "error"),
    (
        (7, 79, "supplier cache aliases disagree with prompt tokens"),
        (6, 81, "supplier cache aliases disagree with nested cache details"),
    ),
)
async def test_additive_openai_details_profile_rejects_inconsistent_cache_aliases(
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    error: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 87,
                    "completion_tokens": 11,
                    "total_tokens": 98,
                    "prompt_tokens_details": {"cached_tokens": 7},
                    "completion_tokens_details": {"reasoning_tokens": 3},
                    "prompt_cache_hit_tokens": cache_hit_tokens,
                    "prompt_cache_miss_tokens": cache_miss_tokens,
                },
            },
        )

    client = _client(CapturingStore(), handler, usage_profile="openai-details-v3")

    with pytest.raises(ModelCallFailure, match=error):
        await client.complete(_request())

    await client.close()


@pytest.mark.asyncio
async def test_additive_openai_details_profile_marks_missing_details_partial() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            },
        )

    store = CapturingStore()
    client = _client(store, handler, usage_profile="openai-details-v3")

    await client.complete(_request())

    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["proof_status"] == "measured_partial"
    assert usage["unavailable_dimensions"] == ["cached_input_tokens", "reasoning_tokens"]
    await client.close()


@pytest.mark.asyncio
async def test_additive_openai_details_profile_rejects_unknown_nested_extension() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                    "prompt_tokens_details": {"cached_tokens": 1, "future_tokens": 4},
                },
            },
        )

    store = CapturingStore()
    client = _client(store, handler, usage_profile="openai-details-v3")

    with pytest.raises(ModelCallFailure) as failure:
        await client.complete(_request())

    assert failure.value.failure_kind == "usage_parse_error"
    assert json.loads(store.records[0].canonical_bytes)["schema_version"] == 3
    await client.close()


@pytest.mark.asyncio
async def test_additive_openai_details_profile_rejects_nonzero_untracked_detail() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                    "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 1},
                    "completion_tokens_details": {
                        "accepted_prediction_tokens": 1,
                        "audio_tokens": 0,
                        "reasoning_tokens": 1,
                        "rejected_prediction_tokens": 0,
                    },
                    "prompt_cached_tokens_details": {"audio_tokens": 0},
                },
            },
        )

    client = _client(CapturingStore(), handler, usage_profile="openai-details-v3")

    with pytest.raises(ModelCallFailure) as failure:
        await client.complete(_request())

    assert failure.value.failure_kind == "usage_parse_error"
    await client.close()


@pytest.mark.asyncio
async def test_supplier_usage_cannot_exceed_the_requested_output_ceiling() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 129,
                    "total_tokens": 136,
                },
            },
        )

    store = CapturingStore()
    client = _client(
        store,
        handler,
        role=ModelRole.MEMORY_EXTRACTION,
        binding_id="model_readiness-binding",
    )
    request = replace(
        _request(),
        parent_kind="model_readiness",
        parent_id="readiness-1",
        stage="model_readiness",
        role_binding_id="model_readiness-binding",
        max_output_tokens=128,
    )

    with pytest.raises(ModelCallFailure) as failure:
        await client.complete(request)

    assert failure.value.failure_kind == "output_contract_error"
    assert failure.value.supplier_status_code == 200
    assert failure.value.raw_reference.sha256 == store.raw[0].sha256
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["proof_status"] == "measured_complete"
    assert usage["visible_output_tokens"] == 129
    await client.close()


@pytest.mark.asyncio
async def test_invalid_utf8_response_is_a_typed_sealed_parse_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"\xff")

    store = CapturingStore()
    client = _client(store, handler)

    with pytest.raises(ModelCallFailure) as failure:
        await client.complete(_request())

    assert failure.value.failure_kind == "response_parse_error"
    assert failure.value.supplier_status_code == 200
    assert store.raw[0].payload_bytes == b"\xff"
    usage = json.loads(store.records[0].canonical_bytes)
    assert usage["proof_status"] == "unavailable"
    await client.close()


@pytest.mark.asyncio
async def test_no_opaque_retry_and_timeout_is_unknown_outcome() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout("ambiguous timeout", request=request)

    client = OpenAICompatibleModelClient(
        store=CapturingStore(),
        base_url="https://models.example/v1",
        api_key="secret",
        role_binding=_binding(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelCallUnknownOutcome):
        await client.complete(_request())
    assert calls == 1
    await client.close()


@pytest.mark.asyncio
async def test_total_timeout_bounds_a_model_call_even_when_transport_never_returns() -> None:
    request_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    client = OpenAICompatibleModelClient(
        store=CapturingStore(),
        base_url="https://models.example/v1",
        api_key="secret",
        role_binding=_binding(),
        transport=httpx.MockTransport(handler),
        total_timeout_seconds=0.01,
    )

    with pytest.raises(ModelCallUnknownOutcome) as failure:
        await client.complete(_request())
    assert request_started.is_set()
    assert failure.value.failure_kind == "timeout"
    await client.close()


@pytest.mark.parametrize("total_timeout", [0.0, -1.0, float("inf"), float("nan")])
def test_total_timeout_requires_a_finite_positive_value(total_timeout: float) -> None:
    with pytest.raises(ValueError, match="total timeout"):
        OpenAICompatibleModelClient(
            store=CapturingStore(),
            base_url="https://models.example/v1",
            api_key="secret",
            role_binding=_binding(),
            transport=httpx.MockTransport(lambda _request: httpx.Response(200)),
            total_timeout_seconds=total_timeout,
        )


@pytest.mark.asyncio
async def test_inflight_cancellation_is_a_typed_unknown_outcome_and_remains_cancelled() -> None:
    request_started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    store = CapturingStore()
    client = _client(store, handler)
    completion = asyncio.create_task(client.complete(_request()))
    await request_started.wait()

    completion.cancel()
    with pytest.raises(asyncio.CancelledError) as cancelled:
        await completion

    assert isinstance(cancelled.value, ModelCallUnknownOutcome)
    assert cancelled.value.failure_kind == "cancelled"
    assert completion.cancelled() is True
    assert store.raw == []
    assert store.records == []
    await client.close()


@pytest.mark.asyncio
async def test_close_waits_for_the_single_inflight_dispatch_and_then_rejects_new_calls() -> None:
    request_started = asyncio.Event()
    release_response = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        request_started.set()
        await release_response.wait()
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
            },
        )

    client = _client(CapturingStore(), handler)
    completion = asyncio.create_task(client.complete(_request()))
    await request_started.wait()
    closing = asyncio.create_task(client.close())
    await asyncio.sleep(0)

    assert not closing.done()
    release_response.set()
    assert (await completion).output_text == "answer"
    await closing
    with pytest.raises(RuntimeError, match="closed"):
        await client.complete(_request())


@pytest.mark.asyncio
async def test_two_admitted_completions_dispatch_in_parallel_and_close_drains_both() -> None:
    both_requests_started = asyncio.Event()
    release_response = asyncio.Event()
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 2:
            both_requests_started.set()
        await release_response.wait()
        return httpx.Response(
            200,
            json={
                "model": "answer-model",
                "choices": [
                    {"index": 0, "message": {"content": "answer"}, "finish_reason": "stop"}
                ],
            },
        )

    client = _client(CapturingStore(), handler)
    completions = [asyncio.create_task(client.complete(_request())) for _ in range(2)]

    try:
        await asyncio.wait_for(both_requests_started.wait(), timeout=1.0)
        closing = asyncio.create_task(client.close())
        await asyncio.sleep(0)

        assert not closing.done()
        with pytest.raises(RuntimeError, match="closed"):
            await client.complete(_request())
    finally:
        release_response.set()

    assert [receipt.output_text for receipt in await asyncio.gather(*completions)] == [
        "answer",
        "answer",
    ]
    await closing
    assert calls == 2


@pytest.mark.asyncio
async def test_failed_transport_close_is_retried_before_client_reports_closed() -> None:
    class FailFirstCloseTransport(httpx.AsyncBaseTransport):
        def __init__(self) -> None:
            self.close_count = 0

        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            raise AssertionError(f"unexpected request while testing close: {request.url}")

        async def aclose(self) -> None:
            self.close_count += 1
            if self.close_count == 1:
                raise RuntimeError("fixture close failure")

    transport = FailFirstCloseTransport()
    client = OpenAICompatibleModelClient(
        store=CapturingStore(),
        base_url="https://models.example/v1",
        api_key="secret",
        role_binding=_binding(),
        transport=transport,
    )

    with pytest.raises(RuntimeError, match="fixture close failure"):
        await client.close()
    await client.close()
    await client.close()

    assert transport.close_count == 2
