from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import ValidationError

from oamb.artifacts.store import ArtifactStore
from oamb.contracts import accounting, ports
from oamb.contracts.ids import canonical_sha256
from oamb.memory_systems.hindsight import HindsightAdapter
from oamb.memory_systems.mem0 import Mem0RestAdapter, Mem0SdkAdapter
from oamb.memory_systems.openviking import OpenVikingRestAdapter
from oamb.memory_systems.rest import (
    MemorySystemCallCancelledUnknownOutcome as RestCancelledUnknownOutcome,
)
from oamb.memory_systems.rest import (
    MemorySystemCallUnknownOutcome as RestUnknownOutcome,
)
from oamb.runtime.runner import UnknownExternalOutcome


def _port_type(name: str) -> Any:
    value = getattr(ports, name, None)
    assert value is not None, f"missing T7 port contract: {name}"
    return value


def _token_usage_v3_type() -> Any:
    value = getattr(accounting, "TokenUsageRecordV3", None)
    assert value is not None, "missing token_usage_record schema version 3"
    return value


def _source(source_id: str, ordinal: int) -> ports.SourceUnit:
    payload = f"source-{ordinal}".encode()
    return ports.SourceUnit(
        source_unit_id=source_id,
        context_manifest_entry_id=f"context-{ordinal}",
        ordinal_1_indexed=ordinal,
        payload_sha256=canonical_sha256([payload.decode()]),
        payload_bytes=payload,
    )


def _partial_hindsight_usage() -> Any:
    return _token_usage_v3_type()(
        usage_record_id="1" * 64,
        attempt_id="2" * 64,
        parent_kind="ingestion_plan",
        parent_id="occurrence-1",
        stage=accounting.TokenStageV2.MEMORY_INGEST,
        operation_kind="retain_extraction",
        token_domain=accounting.TokenDomain.EXTERNAL_LLM,
        measurement_source=accounting.TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=11,
        visible_output_tokens=2,
        supplier_reported_total_tokens=13,
        context_view_tokens=None,
        cached_input_tokens=None,
        reasoning_tokens=None,
        model="fixture-extractor",
        meter_schema_id="hindsight-retain-usage-v0.9.2",
        raw_field_paths=(
            ("input_tokens", "usage.input_tokens"),
            ("visible_output_tokens", "usage.output_tokens"),
            ("supplier_reported_total_tokens", "usage.total_tokens"),
        ),
        covered_dimensions=(
            "input_tokens",
            "visible_output_tokens",
            "supplier_reported_total_tokens",
        ),
        unavailable_dimensions=("cached_input_tokens", "reasoning_tokens"),
        not_applicable_dimensions=(),
        inclusion_relationships=(
            ("supplier_reported_total_tokens", "input_tokens+visible_output_tokens"),
            ("cached_input_tokens", "subset_of:input_tokens"),
            ("reasoning_tokens", "excluded_from:supplier_reported_total_tokens"),
        ),
        token_measurement_complete=False,
        billing_complete=False,
        proof_status=accounting.ProofStatus.MEASURED_PARTIAL,
        reason="cache_and_reasoning_usage_unavailable",
        raw_response_ref="3" * 64,
    )


def test_memory_unknown_outcome_has_one_contract_owner() -> None:
    unknown_type = _port_type("MemorySystemCallUnknownOutcome")
    cancelled_type = _port_type("MemorySystemCallCancelledUnknownOutcome")
    assert RestUnknownOutcome is unknown_type
    assert RestCancelledUnknownOutcome is cancelled_type
    assert UnknownExternalOutcome is unknown_type
    cancelled = cancelled_type("cancelled write")
    assert isinstance(cancelled, asyncio.CancelledError)
    assert isinstance(cancelled, unknown_type)


def test_unsupported_profile_verdict_is_typed_and_machine_readable() -> None:
    unsupported_type = _port_type("MemorySystemProfileUnsupported")
    unsupported = unsupported_type(
        "mem0-rest-v1",
        reason_codes=("implicit_entity_store", "incomplete_protected_projection"),
    )

    assert unsupported.profile_id == "mem0-rest-v1"
    assert unsupported.reason_codes == (
        "implicit_entity_store",
        "incomplete_protected_projection",
    )
    with pytest.raises(ValueError):
        unsupported_type("mem0-rest-v1", reason_codes=())
    with pytest.raises(ValueError):
        unsupported_type("mem0-rest-v1", reason_codes=("duplicate", "duplicate"))


def test_ingestion_batch_attempt_ordinal_is_bounded_and_skipped_sources_are_preserved() -> None:
    ingestion_dispatch_type = _port_type("IngestionDispatch")
    source = _source("source-1", 1)
    scope = ports.ScopeReceipt(
        ingestion_occurrence_id="a" * 64,
        scope_id="scope-1",
        raw_reference=ports.RawReferenceHandle("1" * 64),
    )
    dispatch = ingestion_dispatch_type(
        dispatch_ordinal_1_indexed=1,
        operation_kind="fixture_ingest",
        request_fingerprint="b" * 64,
        ordered_source_units=(source,),
    )

    request = ports.IngestionDispatchRequest(
        scope=scope,
        attempt_id="c" * 64,
        dispatch=dispatch,
        batch_attempt_ordinal=3,
    )
    receipt = ports.IngestionDispatchReceipt(
        attempt_id=request.attempt_id,
        dispatch=dispatch,
        accepted_source_unit_ids=(),
        rejected_source_unit_ids=(),
        skipped_source_unit_ids=("source-1",),
        raw_reference=ports.RawReferenceHandle("2" * 64),
        raw_response_bytes=b'{"error":"settled"}',
        usage_records=(),
    )
    ingestion = ports.IngestionReceipt(
        ingestion_occurrence_id=scope.ingestion_occurrence_id,
        accepted_source_unit_ids=(),
        rejected_source_unit_ids=(),
        skipped_source_unit_ids=("source-1",),
        raw_references=(receipt.raw_reference,),
        dispatch_receipts=(receipt,),
    )

    assert request.batch_attempt_ordinal == 3
    assert ingestion.skipped_source_unit_ids == ("source-1",)
    for invalid_ordinal in (0, 4, True, 1.0):
        with pytest.raises(ValueError, match="batch attempt ordinal"):
            ports.IngestionDispatchRequest(
                scope=scope,
                attempt_id="d" * 64,
                dispatch=dispatch,
                batch_attempt_ordinal=cast(Any, invalid_ordinal),
            )


@pytest.mark.asyncio
async def test_all_t7_adapters_satisfy_the_common_memory_system_port(tmp_path: Path) -> None:
    def reject_dispatch(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"common port check dispatched {request.method} {request.url}")

    transport = httpx.MockTransport(reject_dispatch)
    store = ArtifactStore(tmp_path / "capsule")
    adapters = (
        HindsightAdapter(
            store=store,
            base_url="https://hindsight.example",
            extraction_model="fixture-extractor",
            runtime_binding_hash="e" * 64,
            transport=transport,
        ),
        OpenVikingRestAdapter(
            store=store,
            base_url="https://openviking.example",
            api_key="fixture-key",
            benchmark_account="fixture-account",
            benchmark_user="fixture-user",
            runtime_binding_hash="f" * 64,
            transport=transport,
        ),
        Mem0RestAdapter(
            store=store,
            base_url="https://mem0.example",
            api_key="fixture-key",
            inspector_base_url="https://mem0-inspector.example",
            inspector_api_key="fixture-inspector-key",
            runtime_binding_hash="a" * 64,
            transport=transport,
            inspector_transport=transport,
        ),
        Mem0SdkAdapter(),
    )

    assert all(isinstance(adapter, ports.MemorySystemPort) for adapter in adapters)
    for adapter in adapters:
        await adapter.close()


def test_token_usage_v3_expresses_partial_supplier_meter_coverage() -> None:
    usage = _partial_hindsight_usage()

    assert usage.proof_status == accounting.ProofStatus.MEASURED_PARTIAL
    assert usage.token_measurement_complete is False
    assert usage.billing_complete is False
    assert usage.unavailable_dimensions == ("cached_input_tokens", "reasoning_tokens")


def test_token_usage_v3_allows_explicitly_complete_supplier_meter() -> None:
    partial = _partial_hindsight_usage().model_dump(mode="python")
    complete = _token_usage_v3_type().model_validate(
        partial
        | {
            "cached_input_tokens": 1,
            "reasoning_tokens": 0,
            "raw_field_paths": partial["raw_field_paths"]
            + (
                ("cached_input_tokens", "usage.cached_tokens"),
                ("reasoning_tokens", "usage.thoughts_tokens"),
            ),
            "covered_dimensions": partial["covered_dimensions"]
            + ("cached_input_tokens", "reasoning_tokens"),
            "unavailable_dimensions": (),
            "token_measurement_complete": True,
            "billing_complete": True,
            "proof_status": accounting.ProofStatus.MEASURED_COMPLETE,
            "reason": None,
        }
    )

    assert complete.billing_complete is True


@pytest.mark.parametrize(
    "change",
    [
        {"billing_complete": True},
        {"token_measurement_complete": True},
        {"raw_field_paths": (("input_tokens", "usage.input_tokens"),)},
        {"covered_dimensions": ("input_tokens", "input_tokens")},
        {"unavailable_dimensions": ("cached_input_tokens", "cached_input_tokens")},
    ],
)
def test_token_usage_v3_rejects_false_or_ambiguous_coverage(change: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        usage = _partial_hindsight_usage()
        _token_usage_v3_type().model_validate(usage.model_dump(mode="python") | change)
