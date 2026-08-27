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
    QUALITY_REVIEW = "quality_review"


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
    parent_kind: Literal["ingestion_plan", "case", "phase_review"]
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
