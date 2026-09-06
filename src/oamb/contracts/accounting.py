"""Typed token, resource, and money evidence without implicit zero values."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from .base import (
    NonEmptyStr,
    NonNegativeDecimal,
    NonNegativeInt,
    Sha256,
    StrictContract,
    UtcDateTime,
)
from .specifications import DispatchBudgetOwnerKind


def count_whitespace_tokens(text: str) -> int:
    """Return the deterministic fake meter's whitespace-delimited token count."""

    return len(text.split())


def count_message_whitespace_tokens(messages: tuple[tuple[str, str], ...]) -> int:
    return sum(count_whitespace_tokens(content) for _role, content in messages)


class ProofStatus(StrEnum):
    MEASURED_COMPLETE = "measured_complete"
    MEASURED_PARTIAL = "measured_partial"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class TokenStage(StrEnum):
    MEMORY_INGEST = "memory_ingest"
    MEMORY_QUERY = "memory_query"
    CONTEXT_VIEW = "context_view"
    ANSWER = "answer"
    JUDGE = "judge"


class TokenStageV2(StrEnum):
    MEMORY_INGEST = "memory_ingest"
    MEMORY_QUERY = "memory_query"
    CONTEXT_VIEW = "context_view"
    ANSWER = "answer"
    JUDGE = "judge"
    MODEL_READINESS = "model_readiness"


class TokenStageV3(StrEnum):
    MEMORY_INGEST = "memory_ingest"
    MEMORY_QUERY = "memory_query"
    CONTEXT_VIEW = "context_view"
    ANSWER = "answer"
    JUDGE = "judge"
    MODEL_READINESS = "model_readiness"
    MEMORY_CONFORMANCE = "memory_conformance"


class TokenDomain(StrEnum):
    LOCAL_CONTEXT_VIEW = "local_context_view"
    EXTERNAL_LLM = "external_llm"


class TokenMeasurementSource(StrEnum):
    LOCAL_TOKENIZER = "local_tokenizer"
    SUPPLIER_RESPONSE = "supplier_response"


class AggregationOperator(StrEnum):
    SUM = "sum"
    INTERVAL_UNION = "interval_union"
    MAXIMUM = "maximum"
    TERMINAL_SNAPSHOT = "terminal_snapshot"


class CostBasis(StrEnum):
    ACTUAL_SUPPLIER_CHARGE = "actual_supplier_charge"
    ESTIMATE_FROM_MEASURED_USAGE = "estimate_from_measured_usage"


class IndexingView(StrEnum):
    ATTEMPTED = "attempted"
    FINAL_CONTRIBUTION = "final_contribution"
    NOT_APPLICABLE = "not_applicable"


class MeasurementDimensionSpec(StrictContract):
    schema_name: Literal["measurement_dimension_spec"] = "measurement_dimension_spec"
    schema_version: Literal[1] = 1
    dimension_id: NonEmptyStr
    stage: NonEmptyStr
    operation_kind: NonEmptyStr
    parent_kind: NonEmptyStr
    unit: NonEmptyStr
    allowed_meter_sources: tuple[NonEmptyStr, ...]
    required: bool
    price_class: NonEmptyStr | None
    indexing_view_rule: NonEmptyStr
    aggregation_operator: AggregationOperator


class CostMeasurementSpec(StrictContract):
    schema_name: Literal["cost_measurement_spec"] = "cost_measurement_spec"
    schema_version: Literal[1] = 1
    measurement_spec_id: NonEmptyStr
    measurement_spec_version: NonEmptyStr
    dimensions: tuple[MeasurementDimensionSpec, ...]


class PriceSnapshot(StrictContract):
    schema_name: Literal["price_snapshot"] = "price_snapshot"
    schema_version: Literal[1] = 1
    price_snapshot_id: NonEmptyStr
    provider: NonEmptyStr
    effective_at: UtcDateTime
    currency: str
    price_class_ids: tuple[NonEmptyStr, ...]
    unit_prices: tuple[NonNegativeDecimal, ...]

    @model_validator(mode="after")
    def aligned_prices(self) -> Self:
        if len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        if len(self.price_class_ids) != len(self.unit_prices):
            raise ValueError("price classes and unit prices must align")
        return self


class TokenUsageRecord(StrictContract):
    schema_name: Literal["token_usage_record"] = "token_usage_record"
    schema_version: Literal[1] = 1
    usage_record_id: Sha256
    attempt_id: Sha256
    parent_kind: Literal["ingestion_plan", "case"]
    parent_id: NonEmptyStr
    stage: TokenStage
    operation_kind: NonEmptyStr
    token_domain: TokenDomain
    measurement_source: TokenMeasurementSource
    input_tokens: NonNegativeInt | None
    visible_output_tokens: NonNegativeInt | None
    supplier_reported_total_tokens: NonNegativeInt | None
    context_view_tokens: NonNegativeInt | None
    proof_status: ProofStatus
    reason: str | None
    raw_response_ref: Sha256 | None

    @model_validator(mode="after")
    def proof_shape(self) -> Self:
        values = (
            self.input_tokens,
            self.visible_output_tokens,
            self.supplier_reported_total_tokens,
            self.context_view_tokens,
        )
        if self.proof_status in {ProofStatus.UNAVAILABLE, ProofStatus.NOT_APPLICABLE}:
            if any(value is not None for value in values):
                raise ValueError(f"{self.proof_status.value} usage cannot contain token values")
            if not self.reason:
                raise ValueError(f"{self.proof_status.value} usage requires a reason")
        elif not any(value is not None for value in values):
            raise ValueError("measured usage requires at least one token value")
        unavailable = self.proof_status in {ProofStatus.UNAVAILABLE, ProofStatus.NOT_APPLICABLE}
        external_values = (
            self.input_tokens,
            self.visible_output_tokens,
            self.supplier_reported_total_tokens,
        )
        if self.token_domain == TokenDomain.LOCAL_CONTEXT_VIEW:
            if self.measurement_source != TokenMeasurementSource.LOCAL_TOKENIZER:
                raise ValueError("local context usage requires the local tokenizer")
            if any(value is not None for value in external_values):
                raise ValueError("local context usage cannot contain external LLM token values")
            if not unavailable and self.context_view_tokens is None:
                raise ValueError("measured local context usage requires context_view_tokens")
        else:
            if self.measurement_source != TokenMeasurementSource.SUPPLIER_RESPONSE:
                raise ValueError("external LLM usage requires a supplier response")
            if self.context_view_tokens is not None:
                raise ValueError("external LLM usage cannot contain context_view_tokens")
            if not unavailable and not any(value is not None for value in external_values):
                raise ValueError("measured external LLM usage requires supplier token values")
        return self


class TokenUsageRecordV2(StrictContract):
    schema_name: Literal["token_usage_record"] = "token_usage_record"
    schema_version: Literal[2] = 2
    usage_record_id: Sha256
    attempt_id: Sha256
    parent_kind: Literal["ingestion_plan", "case", "model_readiness"]
    parent_id: NonEmptyStr
    stage: TokenStageV2
    operation_kind: NonEmptyStr
    token_domain: TokenDomain
    measurement_source: TokenMeasurementSource
    input_tokens: NonNegativeInt | None
    visible_output_tokens: NonNegativeInt | None
    supplier_reported_total_tokens: NonNegativeInt | None
    context_view_tokens: NonNegativeInt | None
    proof_status: ProofStatus
    reason: str | None
    raw_response_ref: Sha256 | None

    @model_validator(mode="after")
    def proof_shape(self) -> Self:
        values = (
            self.input_tokens,
            self.visible_output_tokens,
            self.supplier_reported_total_tokens,
            self.context_view_tokens,
        )
        if self.proof_status in {ProofStatus.UNAVAILABLE, ProofStatus.NOT_APPLICABLE}:
            if any(value is not None for value in values):
                raise ValueError(f"{self.proof_status.value} usage cannot contain token values")
            if not self.reason:
                raise ValueError(f"{self.proof_status.value} usage requires a reason")
        elif not any(value is not None for value in values):
            raise ValueError("measured usage requires at least one token value")
        unavailable = self.proof_status in {ProofStatus.UNAVAILABLE, ProofStatus.NOT_APPLICABLE}
        external_values = (
            self.input_tokens,
            self.visible_output_tokens,
            self.supplier_reported_total_tokens,
        )
        if self.token_domain == TokenDomain.LOCAL_CONTEXT_VIEW:
            if self.measurement_source != TokenMeasurementSource.LOCAL_TOKENIZER:
                raise ValueError("local context usage requires the local tokenizer")
            if any(value is not None for value in external_values):
                raise ValueError("local context usage cannot contain external LLM token values")
            if not unavailable and self.context_view_tokens is None:
                raise ValueError("measured local context usage requires context_view_tokens")
        else:
            if self.measurement_source != TokenMeasurementSource.SUPPLIER_RESPONSE:
                raise ValueError("external LLM usage requires a supplier response")
            if self.context_view_tokens is not None:
                raise ValueError("external LLM usage cannot contain context_view_tokens")
            if not unavailable and not any(value is not None for value in external_values):
                raise ValueError("measured external LLM usage requires supplier token values")
        if self.parent_kind == "model_readiness":
            if self.stage != TokenStageV2.MODEL_READINESS:
                raise ValueError("model-readiness usage requires its matching token stage")
        elif self.stage == TokenStageV2.MODEL_READINESS:
            raise ValueError("model-readiness token stage requires its matching parent")
        return self


_EXTERNAL_TOKEN_DIMENSIONS = frozenset(
    {
        "input_tokens",
        "visible_output_tokens",
        "supplier_reported_total_tokens",
        "cached_input_tokens",
        "reasoning_tokens",
    }
)
_LOCAL_TOKEN_DIMENSIONS = frozenset({"context_view_tokens"})


class TokenUsageRecordV3(StrictContract):
    schema_name: Literal["token_usage_record"] = "token_usage_record"
    schema_version: Literal[3] = 3
    usage_record_id: Sha256
    attempt_id: Sha256
    parent_kind: Literal["ingestion_plan", "case", "model_readiness"]
    parent_id: NonEmptyStr
    stage: TokenStageV2
    operation_kind: NonEmptyStr
    token_domain: TokenDomain
    measurement_source: TokenMeasurementSource
    input_tokens: NonNegativeInt | None
    visible_output_tokens: NonNegativeInt | None
    supplier_reported_total_tokens: NonNegativeInt | None
    context_view_tokens: NonNegativeInt | None
    cached_input_tokens: NonNegativeInt | None
    reasoning_tokens: NonNegativeInt | None
    model: NonEmptyStr | None
    meter_schema_id: NonEmptyStr
    raw_field_paths: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    covered_dimensions: tuple[NonEmptyStr, ...]
    unavailable_dimensions: tuple[NonEmptyStr, ...]
    not_applicable_dimensions: tuple[NonEmptyStr, ...]
    inclusion_relationships: tuple[tuple[NonEmptyStr, NonEmptyStr], ...]
    token_measurement_complete: bool
    billing_complete: bool
    proof_status: ProofStatus
    reason: str | None
    raw_response_ref: Sha256 | None

    @model_validator(mode="after")
    def meter_coverage_is_closed(self) -> Self:
        groups = (
            self.covered_dimensions,
            self.unavailable_dimensions,
            self.not_applicable_dimensions,
        )
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("token meter coverage contains duplicate dimensions")
        covered, unavailable, not_applicable = (set(group) for group in groups)
        if covered & unavailable or covered & not_applicable or unavailable & not_applicable:
            raise ValueError("token meter coverage groups must be disjoint")

        values = {
            "input_tokens": self.input_tokens,
            "visible_output_tokens": self.visible_output_tokens,
            "supplier_reported_total_tokens": self.supplier_reported_total_tokens,
            "context_view_tokens": self.context_view_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }
        if self.token_domain == TokenDomain.EXTERNAL_LLM:
            expected_dimensions = _EXTERNAL_TOKEN_DIMENSIONS
            if self.measurement_source != TokenMeasurementSource.SUPPLIER_RESPONSE:
                raise ValueError("external LLM usage requires a supplier response")
            if self.context_view_tokens is not None:
                raise ValueError("external LLM usage cannot contain context-view tokens")
            if self.model is None:
                raise ValueError("external LLM usage requires a model")
            if self.raw_response_ref is None:
                raise ValueError("external LLM usage requires a raw response reference")
            raw_path_dimensions = tuple(item[0] for item in self.raw_field_paths)
            if len(set(raw_path_dimensions)) != len(raw_path_dimensions):
                raise ValueError("token usage raw field paths contain duplicate dimensions")
            if set(raw_path_dimensions) != covered:
                raise ValueError("measured supplier dimensions require exact raw field paths")
        else:
            expected_dimensions = _LOCAL_TOKEN_DIMENSIONS
            if self.measurement_source != TokenMeasurementSource.LOCAL_TOKENIZER:
                raise ValueError("local context usage requires the local tokenizer")
            if any(values[dimension] is not None for dimension in _EXTERNAL_TOKEN_DIMENSIONS):
                raise ValueError("local context usage cannot contain supplier token values")
            if self.model is not None:
                raise ValueError("local context usage cannot carry supplier model identity")
            if self.raw_field_paths:
                raise ValueError("local context usage cannot carry supplier raw field paths")

        if covered | unavailable | not_applicable != expected_dimensions:
            raise ValueError("token meter coverage must classify every profile dimension")
        for dimension in expected_dimensions:
            if (values[dimension] is not None) != (dimension in covered):
                raise ValueError("token values must exactly match measured coverage")
        relationship_dimensions = tuple(item[0] for item in self.inclusion_relationships)
        if len(set(relationship_dimensions)) != len(relationship_dimensions):
            raise ValueError("token inclusion relationships contain duplicate dimensions")
        if not set(relationship_dimensions) <= expected_dimensions:
            raise ValueError("token inclusion relationship names an unknown dimension")

        if self.token_measurement_complete != (not unavailable):
            raise ValueError("token measurement completeness disagrees with unavailable dimensions")
        if self.billing_complete and (unavailable or not self.token_measurement_complete):
            raise ValueError("billing completeness requires complete token measurement")
        if self.proof_status == ProofStatus.MEASURED_COMPLETE:
            if not self.token_measurement_complete or not covered:
                raise ValueError("measured-complete usage requires complete measured coverage")
            if self.reason is not None:
                raise ValueError("measured-complete usage cannot carry a limitation reason")
        elif self.proof_status == ProofStatus.MEASURED_PARTIAL:
            if not covered or not unavailable or not self.reason:
                raise ValueError(
                    "measured-partial usage requires measured and unavailable coverage"
                )
        elif self.proof_status == ProofStatus.UNAVAILABLE:
            if covered or unavailable != expected_dimensions or not self.reason:
                raise ValueError("unavailable usage requires every dimension unavailable")
        elif covered or not_applicable != expected_dimensions or not self.reason:
            raise ValueError("not-applicable usage requires every dimension not applicable")

        if self.parent_kind == "model_readiness":
            if self.stage != TokenStageV2.MODEL_READINESS:
                raise ValueError("model-readiness usage requires its matching token stage")
        elif self.stage == TokenStageV2.MODEL_READINESS:
            raise ValueError("model-readiness token stage requires its matching parent")
        return self


class TokenUsageRecordV4(TokenUsageRecordV3):
    schema_version: Literal[4] = 4  # type: ignore[assignment]
    parent_kind: Literal["ingestion_plan", "case", "model_readiness", "memory_conformance"]  # type: ignore[assignment]
    stage: TokenStageV3  # type: ignore[assignment]
    usage_owner_role_binding_id: NonEmptyStr | None

    @model_validator(mode="after")
    def conformance_parent_and_stage_match(self) -> Self:
        if self.parent_kind == "memory_conformance":
            if self.stage != TokenStageV3.MEMORY_CONFORMANCE:
                raise ValueError("memory-conformance usage requires its matching token stage")
            if self.usage_owner_role_binding_id is None:
                raise ValueError("memory-conformance usage requires its role owner")
        elif self.stage == TokenStageV3.MEMORY_CONFORMANCE:
            raise ValueError("memory-conformance token stage requires its matching parent")
        elif self.usage_owner_role_binding_id is not None:
            raise ValueError("only memory-conformance usage carries a role owner")
        return self


class TokenUsageRecordV5(TokenUsageRecordV3):
    schema_version: Literal[5] = 5  # type: ignore[assignment]
    attempt_id: Sha256
    dispatch_route_id: NonEmptyStr
    dispatch_route_hash: Sha256
    budget_owner_kind: DispatchBudgetOwnerKind
    budget_owner_id: NonEmptyStr
    parent_kind: Literal["ingestion_plan", "case"]
    stage: TokenStage  # type: ignore[assignment]


class ResourceUsageRecord(StrictContract):
    schema_name: Literal["resource_usage_record"] = "resource_usage_record"
    schema_version: Literal[1] = 1
    resource_record_id: Sha256
    parent_kind: NonEmptyStr
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    meter_boundary: NonEmptyStr
    dimension_id: NonEmptyStr
    value: NonNegativeDecimal | None
    unit: NonEmptyStr
    measurement_source: NonEmptyStr
    measurement_spec_id: NonEmptyStr
    environment_hash: Sha256
    started_at: UtcDateTime | None
    ended_at: UtcDateTime | None
    raw_telemetry_ref: Sha256 | None
    proof_status: ProofStatus
    reason: str | None

    @model_validator(mode="after")
    def proof_shape(self) -> Self:
        if self.proof_status in {ProofStatus.UNAVAILABLE, ProofStatus.NOT_APPLICABLE}:
            if self.value is not None:
                raise ValueError("unavailable resource evidence cannot contain a value")
            if not self.reason:
                raise ValueError("unavailable resource evidence requires a reason")
        elif self.value is None:
            raise ValueError("measured resource evidence requires a value")
        if (self.started_at is None) != (self.ended_at is None):
            raise ValueError("resource interval requires both timestamps")
        if self.started_at and self.ended_at and self.ended_at < self.started_at:
            raise ValueError("resource interval end precedes start")
        return self


class ResourceUsageRecordV2(ResourceUsageRecord):
    schema_version: Literal[2] = 2  # type: ignore[assignment]
    attempt_id: Sha256
    dispatch_route_id: NonEmptyStr
    dispatch_route_hash: Sha256
    budget_owner_kind: DispatchBudgetOwnerKind
    budget_owner_id: NonEmptyStr
    parent_kind: Literal["ingestion_plan", "case"]


class CostRecord(StrictContract):
    schema_name: Literal["cost_record"] = "cost_record"
    schema_version: Literal[1] = 1
    cost_record_id: Sha256
    parent_kind: NonEmptyStr
    parent_id: NonEmptyStr
    basis: CostBasis
    indexing_view: IndexingView
    amount: NonNegativeDecimal | None
    currency: str | None
    price_snapshot_id: str | None
    source_usage_record_ids: tuple[Sha256, ...]
    source_resource_record_ids: tuple[Sha256, ...]
    proof_status: ProofStatus
    reason: str | None

    @model_validator(mode="after")
    def proof_and_currency_shape(self) -> Self:
        unavailable = self.proof_status in {
            ProofStatus.UNAVAILABLE,
            ProofStatus.NOT_APPLICABLE,
        }
        if unavailable:
            if self.amount is not None or self.currency is not None:
                raise ValueError("unavailable cost cannot contain amount or currency")
            if not self.reason:
                raise ValueError("unavailable cost requires a reason")
        else:
            if self.amount is None or self.currency is None:
                raise ValueError("measured cost requires amount and currency")
            if len(self.currency) != 3:
                raise ValueError("currency must be an ISO 4217 code")
        return self


class CostRecordV2(CostRecord):
    schema_version: Literal[2] = 2  # type: ignore[assignment]
    attempt_id: Sha256
    dispatch_route_id: NonEmptyStr
    dispatch_route_hash: Sha256
    budget_owner_kind: DispatchBudgetOwnerKind
    budget_owner_id: NonEmptyStr
    parent_kind: Literal["ingestion_plan", "case"]

    @model_validator(mode="after")
    def measured_cost_has_a_source(self) -> Self:
        if self.proof_status in {
            ProofStatus.MEASURED_COMPLETE,
            ProofStatus.MEASURED_PARTIAL,
        } and not (self.source_usage_record_ids or self.source_resource_record_ids):
            raise ValueError("measured cost requires a usage or resource source")
        return self
