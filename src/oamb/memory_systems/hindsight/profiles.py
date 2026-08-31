"""Pinned Hindsight v0.9.2 wire and runtime profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from oamb.contracts.accounting import (
    ProofStatus,
    TokenDomain,
    TokenMeasurementSource,
    TokenStageV2,
    TokenUsageRecordV3,
)
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import RawReferenceHandle
from oamb.memory_systems.rest import parse_exact_json_object

PROFILE_ID: Final = "hindsight-rest-v1"
MEMORY_SYSTEM_ID: Final = "hindsight"
API_VERSION: Final = "0.9.2"
SOURCE_REVISION: Final = "ebad478240d3171bb88201ececda5e8d9883d22d"
IMAGE_DIGEST: Final = "sha256:7635a15739361dbdf221ba796ad25a813f876144fe113022eea8e26cb6ee75e7"
EMBEDDING_MODEL: Final = "qwen3-embedding:0.6b"
EMBEDDING_DIMENSION: Final = 1024

VERSION_FIELDS: Final = frozenset({"api_version", "features"})
EXPECTED_FEATURES: Final[dict[str, bool]] = {
    "observations": True,
    "mcp": True,
    "worker": True,
    "bank_config_api": True,
    "bank_llm_health": False,
    "file_upload_api": True,
    "document_export_api": True,
    "document_import_api": True,
    "audit_log": False,
    "llm_trace": True,
    "store_document_text": True,
}
BANK_PAGE_FIELDS: Final = frozenset({"banks", "total", "limit", "offset"})
BANK_ITEM_BASE_FIELDS: Final = frozenset(
    {
        "bank_id",
        "name",
        "disposition",
        "mission",
        "created_at",
        "updated_at",
        "fact_count",
    }
)
BANK_ITEM_ACTIVITY_FIELDS: Final = frozenset({"last_document_at", "last_write_at"})
BANK_ITEM_FIELDS: Final = BANK_ITEM_BASE_FIELDS | BANK_ITEM_ACTIVITY_FIELDS
DISPOSITION_FIELDS: Final = frozenset({"skepticism", "literalism", "empathy"})
BANK_PROFILE_FIELDS: Final = frozenset({"bank_id", "name", "disposition", "mission", "background"})
BANK_CONFIG_FIELDS: Final = frozenset({"bank_id", "config", "overrides"})
EXPECTED_BANK_CONFIG: Final[dict[str, object]] = {
    "llm_gemini_safety_settings": None,
    "mcp_enabled_tools": None,
    "retain_chunk_size": 3000,
    "retain_structured_chunk_size": None,
    "retain_extraction_mode": "concise",
    "retain_mission": None,
    "retain_custom_instructions": None,
    "retain_default_strategy": None,
    "retain_strategies": None,
    "retain_chunk_batch_size": 100,
    "store_document_text": True,
    "enable_observations": False,
    "enable_auto_consolidation": True,
    "mental_model_min_refresh_interval_seconds": 0,
    "consolidation_max_memories_per_round": 100,
    "consolidation_llm_batch_size": 8,
    "consolidation_llm_parallelism": 4,
    "consolidation_source_facts_max_tokens": 4096,
    "consolidation_source_facts_max_tokens_per_observation": 256,
    "observations_mission": None,
    "max_observations_per_scope": -1,
    "observation_scope_limits": None,
    "entity_labels": None,
    "entities_allow_free_form": True,
    "memory_defense": None,
    "reflect_mission": None,
    "reflect_source_facts_max_tokens": -1,
    "enable_temporal_retrieval": True,
    "enable_graph_retrieval": True,
    "enable_reranking": False,
    "recall_include_chunks": True,
    "recall_max_tokens": 2048,
    "recall_chunks_max_tokens": 1000,
    "recall_budget_function": "fixed",
    "recall_budget_fixed_low": 100,
    "recall_budget_fixed_mid": 300,
    "recall_budget_fixed_high": 1000,
    "recall_budget_adaptive_low": 0.025,
    "recall_budget_adaptive_mid": 0.075,
    "recall_budget_adaptive_high": 0.25,
    "recall_budget_min": 20,
    "recall_budget_max": 2000,
    "disposition_skepticism": None,
    "disposition_literalism": None,
    "disposition_empathy": None,
    "audit_log_enabled": False,
}
RETAIN_SYNC_RESPONSE_FIELDS: Final = frozenset(
    {
        "success",
        "bank_id",
        "items_count",
        "async",
        "usage",
    }
)
RETAIN_RESPONSE_FIELDS: Final = RETAIN_SYNC_RESPONSE_FIELDS | frozenset(
    {
        "operation_id",
        "operation_ids",
    }
)
RETAIN_USAGE_FIELDS: Final = frozenset(
    {"input_tokens", "output_tokens", "total_tokens", "cached_tokens", "thoughts_tokens"}
)
EXTERNAL_TOKEN_DIMENSIONS: Final = (
    "input_tokens",
    "visible_output_tokens",
    "supplier_reported_total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)
BASE_TOKEN_DIMENSIONS: Final = EXTERNAL_TOKEN_DIMENSIONS[:3]
HIDDEN_TOKEN_DIMENSIONS: Final = EXTERNAL_TOKEN_DIMENSIONS[3:]
RETAIN_METER_SCHEMA_ID: Final = "hindsight-retain-usage-v0.9.2"


@dataclass(frozen=True, slots=True)
class HindsightVersion:
    api_version: str
    features: tuple[tuple[str, bool], ...]


@dataclass(frozen=True, slots=True)
class BankPage:
    bank_ids: tuple[str, ...]
    total: int
    limit: int
    offset: int


@dataclass(frozen=True, slots=True)
class RetainResult:
    usage: dict[str, int]


def parse_version_response(raw_bytes: bytes) -> HindsightVersion:
    document = parse_exact_json_object(raw_bytes, expected_fields=VERSION_FIELDS)
    if document["api_version"] != API_VERSION:
        raise ValueError("Hindsight API version does not match the exact profile")
    features = document["features"]
    if not isinstance(features, dict) or set(features) != set(EXPECTED_FEATURES):
        raise ValueError("Hindsight feature fields do not match the exact profile")
    if any(type(value) is not bool for value in features.values()):
        raise ValueError("Hindsight feature values must be booleans")
    if features != EXPECTED_FEATURES:
        raise ValueError("Hindsight feature values do not match the exact profile")
    return HindsightVersion(
        api_version=API_VERSION,
        features=tuple((name, features[name]) for name in EXPECTED_FEATURES),
    )


def _strict_non_negative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"Hindsight {field} must be a non-negative integer")
    return value


def _require_exact_object(value: object, fields: frozenset[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError(f"Hindsight {name} fields do not match the exact profile")
    return value


def parse_bank_page(raw_bytes: bytes, *, expected_limit: int, expected_offset: int) -> BankPage:
    document = parse_exact_json_object(raw_bytes, expected_fields=BANK_PAGE_FIELDS)
    banks = document["banks"]
    if not isinstance(banks, list):
        raise ValueError("Hindsight bank page banks must be an array")
    bank_ids: list[str] = []
    for item in banks:
        if not isinstance(item, dict) or set(item) not in {
            BANK_ITEM_BASE_FIELDS,
            BANK_ITEM_FIELDS,
        }:
            raise ValueError("Hindsight bank item fields do not match the exact profile")
        bank = item
        bank_id = bank["bank_id"]
        if not isinstance(bank_id, str) or not bank_id:
            raise ValueError("Hindsight bank ID must be non-empty")
        _require_exact_object(bank["disposition"], DISPOSITION_FIELDS, "disposition")
        bank_ids.append(bank_id)
    total = _strict_non_negative_int(document["total"], "bank total")
    limit = _strict_non_negative_int(document["limit"], "bank limit")
    offset = _strict_non_negative_int(document["offset"], "bank offset")
    if limit != expected_limit or offset != expected_offset:
        raise ValueError("Hindsight bank page did not echo the requested boundary")
    return BankPage(tuple(bank_ids), total, limit, offset)


def parse_bank_profile(raw_bytes: bytes, *, expected_bank_id: str) -> None:
    document = parse_exact_json_object(raw_bytes, expected_fields=BANK_PROFILE_FIELDS)
    if document["bank_id"] != expected_bank_id:
        raise ValueError("Hindsight bank response names a different bank")
    _require_exact_object(document["disposition"], DISPOSITION_FIELDS, "disposition")


def parse_bank_config(raw_bytes: bytes, *, expected_bank_id: str) -> None:
    document = parse_exact_json_object(raw_bytes, expected_fields=BANK_CONFIG_FIELDS)
    if document["bank_id"] != expected_bank_id:
        raise ValueError("Hindsight config response names a different bank")
    config = document["config"]
    overrides = document["overrides"]
    if not isinstance(config, dict) or not isinstance(overrides, dict):
        raise ValueError("Hindsight bank config values must be objects")
    if config != EXPECTED_BANK_CONFIG:
        raise ValueError("Hindsight score-affecting bank config does not match the profile")
    if overrides != {"enable_observations": False}:
        raise ValueError("Hindsight bank overrides do not match create-only allocation")


def parse_retain_response(
    raw_bytes: bytes,
    *,
    expected_bank_id: str,
    expected_items_count: int,
) -> RetainResult:
    try:
        document = parse_exact_json_object(raw_bytes, expected_fields=RETAIN_RESPONSE_FIELDS)
    except ValueError as response_error:
        try:
            document = parse_exact_json_object(
                raw_bytes,
                expected_fields=RETAIN_SYNC_RESPONSE_FIELDS,
            )
        except ValueError:
            raise response_error from None
    if document["success"] is not True:
        raise ValueError("Hindsight retain did not report success")
    if document["bank_id"] != expected_bank_id:
        raise ValueError("Hindsight retain response names a different bank")
    if (
        _strict_non_negative_int(document["items_count"], "retain item count")
        != expected_items_count
    ):
        raise ValueError("Hindsight retain response count does not match the dispatch")
    if document["async"] is not False:
        raise ValueError("Hindsight retain response is not synchronous")
    if document.get("operation_id") is not None or document.get("operation_ids") is not None:
        raise ValueError("Hindsight synchronous retain returned async operation identity")
    usage = _require_exact_object(document["usage"], RETAIN_USAGE_FIELDS, "retain usage")
    parsed_usage = {
        name: _strict_non_negative_int(value, f"retain usage {name}")
        for name, value in usage.items()
    }
    if parsed_usage["total_tokens"] != (
        parsed_usage["input_tokens"] + parsed_usage["output_tokens"]
    ):
        raise ValueError("Hindsight retain token total does not equal input plus output")
    if parsed_usage["cached_tokens"] != 0 or parsed_usage["thoughts_tokens"] != 0:
        raise ValueError("Hindsight v0.9.2 REST hidden usage sentinel drifted")
    return RetainResult(parsed_usage)


def retain_usage_record(
    result: RetainResult,
    *,
    attempt_id: str,
    parent_id: str,
    configured_model: str,
    runtime_model: str,
    raw_reference: RawReferenceHandle,
) -> TokenUsageRecordV3:
    usage = result.usage
    base_values = (
        usage["input_tokens"],
        usage["output_tokens"],
        usage["total_tokens"],
    )
    all_zero = base_values == (0, 0, 0)
    usage_record_id = canonical_sha256(
        ["oamb-hindsight-retain-usage-v1", attempt_id, parent_id, raw_reference.sha256]
    )
    common: dict[str, Any] = {
        "usage_record_id": usage_record_id,
        "attempt_id": attempt_id,
        "parent_kind": "ingestion_plan",
        "parent_id": parent_id,
        "stage": TokenStageV2.MEMORY_INGEST,
        "operation_kind": "retain_extraction",
        "token_domain": TokenDomain.EXTERNAL_LLM,
        "measurement_source": TokenMeasurementSource.SUPPLIER_RESPONSE,
        "context_view_tokens": None,
        "cached_input_tokens": None,
        "reasoning_tokens": None,
        "configured_model": configured_model,
        "runtime_model": runtime_model,
        "meter_schema_id": RETAIN_METER_SCHEMA_ID,
        "not_applicable_dimensions": (),
        "inclusion_relationships": (
            ("supplier_reported_total_tokens", "input_tokens+visible_output_tokens"),
            ("cached_input_tokens", "subset_of:input_tokens"),
            ("reasoning_tokens", "excluded_from:supplier_reported_total_tokens"),
        ),
        "token_measurement_complete": False,
        "billing_complete": False,
        "raw_response_ref": raw_reference.sha256,
    }
    if all_zero:
        return TokenUsageRecordV3(
            **common,
            input_tokens=None,
            visible_output_tokens=None,
            supplier_reported_total_tokens=None,
            raw_field_paths=(),
            covered_dimensions=(),
            unavailable_dimensions=EXTERNAL_TOKEN_DIMENSIONS,
            proof_status=ProofStatus.UNAVAILABLE,
            reason="provider_zero_usage_sentinel",
        )
    return TokenUsageRecordV3(
        **common,
        input_tokens=base_values[0],
        visible_output_tokens=base_values[1],
        supplier_reported_total_tokens=base_values[2],
        raw_field_paths=(
            ("input_tokens", "usage.input_tokens"),
            ("visible_output_tokens", "usage.output_tokens"),
            ("supplier_reported_total_tokens", "usage.total_tokens"),
        ),
        covered_dimensions=BASE_TOKEN_DIMENSIONS,
        unavailable_dimensions=HIDDEN_TOKEN_DIMENSIONS,
        proof_status=ProofStatus.MEASURED_PARTIAL,
        reason="cache_and_reasoning_usage_unavailable",
    )
