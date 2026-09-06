"""Scripted model client with deterministic raw and usage evidence."""

from __future__ import annotations

import hashlib

from oamb.contracts.accounting import (
    ProofStatus,
    TokenDomain,
    TokenMeasurementSource,
    TokenStage,
    TokenUsageRecord,
    count_message_whitespace_tokens,
    count_whitespace_tokens,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    ArtifactStorePort,
    ArtifactWriteRequest,
    ModelCallFailure,
    ModelReceipt,
    ModelRequest,
    RawPayloadSealRequest,
    RawReferenceHandle,
    ThinkingEffort,
)

_THINKING_EFFORT_BY_ROLE: dict[str, ThinkingEffort] = {
    "fake-answer-v1": "low",
    "fake-judge-v1": "high",
}


class ScriptedFakeModelClient:
    """Fails one named answer once and leaves one named judge unjudged."""

    def __init__(self, store: ArtifactStorePort) -> None:
        self._store = store
        self._failed_message_hashes: set[str] = set()
        self._closed = False

    def thinking_effort_for(self, *, stage: str, role_binding_id: str) -> ThinkingEffort:
        expected_stage = "judge" if role_binding_id == "fake-judge-v1" else "answer"
        try:
            effort = _THINKING_EFFORT_BY_ROLE[role_binding_id]
        except KeyError as exc:
            raise ValueError("fake model request names an unknown role binding") from exc
        if stage != expected_stage:
            raise ValueError("fake model request stage differs from its role binding")
        return effort

    async def complete(self, request: ModelRequest) -> ModelReceipt:
        if self._closed:
            raise RuntimeError("fake model client is closed")
        expected_effort = self.thinking_effort_for(
            stage=request.stage,
            role_binding_id=request.role_binding_id,
        )
        if request.thinking_effort != expected_effort:
            raise ValueError("fake model request thinking effort differs from its role binding")
        text = "\n".join(content for _role, content in request.messages)
        if request.stage == "judge":
            raw_reference, usage_ids = self._seal_call(
                request,
                outcome="judge-unavailable",
                output_text="",
            )
            raise ModelCallFailure(
                "scripted judge remained unjudged",
                raw_reference=raw_reference,
                usage_reference_ids=usage_ids,
                retryable=False,
            )
        if (
            "Which token is second?" in text
            and request.messages_sha256 not in self._failed_message_hashes
        ):
            self._failed_message_hashes.add(request.messages_sha256)
            raw_reference, usage_ids = self._seal_call(
                request,
                outcome="billed-failure",
                output_text="",
            )
            raise ModelCallFailure(
                "scripted retryable answer failure",
                raw_reference=raw_reference,
                usage_reference_ids=usage_ids,
                retryable=True,
            )
        output_text = _answer_for(text)
        raw_reference, usage_ids = self._seal_call(
            request,
            outcome="success",
            output_text=output_text,
        )
        return ModelReceipt(
            raw_reference=raw_reference,
            output_text=output_text,
            usage_reference_ids=usage_ids,
            model="fake-model",
        )

    async def close(self) -> None:
        self._closed = True

    def _seal_call(
        self,
        request: ModelRequest,
        *,
        outcome: str,
        output_text: str,
    ) -> tuple[RawReferenceHandle, tuple[str, ...]]:
        if request.parent_kind == "model_readiness":
            raise ValueError("the T5 fake model client has no model-readiness role")
        input_tokens = count_message_whitespace_tokens(request.messages)
        output_tokens = count_whitespace_tokens(output_text)
        payload = canonical_json_bytes(
            {
                "attempt_id": request.attempt_id,
                "outcome": outcome,
                "parent_kind": request.parent_kind,
                "parent_id": request.parent_id,
                "stage": request.stage,
                "role_binding_id": request.role_binding_id,
                "messages_sha256": request.messages_sha256,
                "messages": request.messages,
                "thinking_effort": request.thinking_effort,
                "output_contract_id": request.output_contract_id,
                "max_output_tokens": request.max_output_tokens,
                "candidate_count": request.candidate_count,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "stop": request.stop,
                "request_fingerprint": request.request_fingerprint,
                "output_text": output_text,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                },
            }
        )
        raw_sha256 = hashlib.sha256(payload).hexdigest()
        raw_reference = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=raw_sha256,
                media_type="application/json",
                compression="gzip",
                payload_bytes=payload,
            )
        )
        usage_id = canonical_sha256(
            [
                "oamb-fake-model-usage-v1",
                request.attempt_id,
                input_tokens,
                output_tokens,
                raw_sha256,
            ]
        )
        usage = TokenUsageRecord(
            usage_record_id=usage_id,
            attempt_id=request.attempt_id,
            parent_kind=request.parent_kind,
            parent_id=request.parent_id,
            stage=TokenStage(request.stage),
            operation_kind=f"fake_{request.stage}_completion",
            token_domain=TokenDomain.EXTERNAL_LLM,
            measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
            input_tokens=input_tokens,
            visible_output_tokens=output_tokens,
            supplier_reported_total_tokens=input_tokens + output_tokens,
            context_view_tokens=None,
            proof_status=ProofStatus.MEASURED_COMPLETE,
            reason=None,
            raw_response_ref=raw_sha256,
        )
        usage_bytes = canonical_json_bytes(usage)
        self._store.seal_source_record(
            ArtifactWriteRequest(
                record_id=usage_id,
                relative_path=f"source/usage/{usage_id}.json",
                canonical_sha256=hashlib.sha256(usage_bytes).hexdigest(),
                canonical_bytes=usage_bytes,
            )
        )
        return raw_reference, (usage_id,)


def _answer_for(text: str) -> str:
    if "first" in text:
        return "alpha"
    if "second" in text:
        return "beta"
    if "grounded" in text:
        return "shared answer"
    return "partial"
