from __future__ import annotations

import ast
import importlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError


def require(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} is not implemented", pytrace=False)


def protocol_spec() -> Any:
    specifications = require("oamb.contracts.specifications")
    return specifications.ProtocolSpec(
        protocol_id="oamb-v0.1",
        protocol_version="0.1.0",
        stages=("memory_ingest", "memory_query", "answer", "evaluate"),
        lifecycle_version="oamb-lifecycle-v1",
        context_policy="provider_order_no_rerank",
        output_contract_ids=("strict-answer-v1",),
        metric_ids=("exact-match-v1",),
        comparison_rule_ids=("same-controls-v1",),
    )


def test_public_contracts_are_strict_immutable_and_versioned() -> None:
    spec = protocol_spec()

    assert spec.schema_name == "protocol_spec"
    assert spec.schema_version == 1
    with pytest.raises(ValidationError):
        spec.protocol_id = "changed"
    with pytest.raises(ValidationError):
        type(spec)(**(spec.model_dump() | {"unknown": "field"}))
    bool_version = spec.model_dump(mode="json") | {"schema_version": True}
    with pytest.raises(ValidationError, match="schema_version"):
        type(spec).model_validate_json(json.dumps(bool_version))


def test_contract_json_round_trip_preserves_tuple_order() -> None:
    spec = protocol_spec()

    restored = type(spec).model_validate_json(spec.model_dump_json())

    assert restored == spec
    assert restored.stages == ("memory_ingest", "memory_query", "answer", "evaluate")


def test_case_manifest_closes_exact_parentage_and_identity_formulas() -> None:
    specifications = require("oamb.contracts.specifications")
    ids = require("oamb.contracts.ids")
    context_a = "a" * 64
    context_b = "b" * 64
    question_hash = "c" * 64
    answer_hashes = ("d" * 64,)
    case_id = ids.case_manifest_entry_id(context_a, 1, question_hash, "question-1")
    plan_manifest_id = ids.plan_manifest_entry_id("mab65-v1", (context_b,))
    payload_hash = ids.ingestion_payload_hash(("e" * 64,))
    plan_id = ids.ingestion_plan_id(plan_manifest_id, payload_hash)
    logical_contexts = (
        specifications.LogicalContextManifestEntry(
            context_content_id="1" * 64,
            context_manifest_entry_id=context_a,
            source_file_sha256="2" * 64,
            source_row_number_1_indexed=1,
            context_bytes_sha256="3" * 64,
        ),
        specifications.LogicalContextManifestEntry(
            context_content_id="4" * 64,
            context_manifest_entry_id=context_b,
            source_file_sha256="2" * 64,
            source_row_number_1_indexed=2,
            context_bytes_sha256="5" * 64,
        ),
    )
    case = specifications.CaseManifestEntry(
        case_manifest_entry_id=case_id,
        context_manifest_entry_id=context_a,
        source_question_number_1_indexed=1,
        question_bytes_sha256=question_hash,
        raw_question_id="question-1",
        answer_value_sha256=answer_hashes,
    )
    wrong_plan = specifications.IngestionPlanManifest(
        plan_manifest_entry_id=plan_manifest_id,
        ingestion_payload_hash=payload_hash,
        ingestion_plan_id=plan_id,
        workload_id="mab65-v1",
        ordered_member_context_manifest_entry_ids=(context_b,),
        ordered_source_unit_bytes_sha256=("e" * 64,),
        ordered_case_manifest_entry_ids=(case_id,),
    )

    with pytest.raises(ValidationError, match="case context.*plan"):
        specifications.CaseManifest(
            manifest_id="manifest-1",
            manifest_hash="f" * 64,
            workload_id="mab65-v1",
            logical_contexts=logical_contexts,
            ingestion_plans=(wrong_plan,),
            cases=(case,),
        )

    with pytest.raises(ValidationError, match="exactly once"):
        specifications.CaseManifest(
            manifest_id="manifest-1",
            manifest_hash="f" * 64,
            workload_id="mab65-v1",
            logical_contexts=logical_contexts,
            ingestion_plans=(
                wrong_plan.model_copy(update={"ordered_case_manifest_entry_ids": ()}),
            ),
            cases=(case,),
        )


def test_attempt_requires_timezone_and_one_index_contribution_shape() -> None:
    evidence = require("oamb.contracts.evidence")
    states = require("oamb.contracts.states")

    valid = evidence.AttemptRecord(
        attempt_id="a" * 64,
        parent_kind="case",
        parent_id="case-1",
        stage="answer",
        ordinal=1,
        request_fingerprint="b" * 64,
        started_at=datetime(2026, 8, 27, 12, 0, tzinfo=UTC),
        ended_at=datetime(2026, 8, 27, 12, 1, tzinfo=UTC),
        outcome=states.AttemptOutcome.SUCCEEDED,
        retry_of_attempt_id=None,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=None,
        raw_error_ref=None,
        index_contribution=states.IndexContribution.NOT_APPLICABLE,
        superseded_by_attempt_id=None,
    )
    assert valid.ordinal == 1

    with pytest.raises(ValidationError, match="timezone"):
        evidence.AttemptRecord(
            **(valid.model_dump() | {"started_at": datetime(2026, 8, 27, 12, 0)})
        )


def test_ready_ingestion_plan_requires_closed_counts_and_readiness_evidence() -> None:
    evidence = require("oamb.contracts.evidence")
    states = require("oamb.contracts.states")
    fields = {
        "ingestion_occurrence_id": "1" * 64,
        "run_id": "run-1",
        "memory_system_id": "fake-memory",
        "ingestion_plan_id": "2" * 64,
        "ordered_member_context_manifest_entry_ids": ("3" * 64,),
        "ordered_case_occurrence_ids": ("4" * 64,),
        "state": states.IngestionPlanState.READY,
        "intended_source_count": 1,
        "accepted_source_count": 0,
        "failed_source_count": 0,
        "readiness_evidence_refs": (),
        "attempt_ids": (),
        "usage_record_ids": (),
        "resource_record_ids": (),
        "cost_record_ids": (),
    }

    with pytest.raises(ValidationError, match="READY"):
        evidence.IngestionPlanRecord(**fields)

    ready = evidence.IngestionPlanRecord(
        **(
            fields
            | {
                "accepted_source_count": 1,
                "readiness_evidence_refs": ("5" * 64,),
            }
        )
    )
    assert ready.state == states.IngestionPlanState.READY


def test_unavailable_usage_is_not_represented_as_zero() -> None:
    accounting = require("oamb.contracts.accounting")

    unavailable = accounting.TokenUsageRecord(
        usage_record_id="c" * 64,
        attempt_id="a" * 64,
        parent_kind="case",
        parent_id="case-1",
        stage=accounting.TokenStage.ANSWER,
        operation_kind="answer_completion",
        token_domain=accounting.TokenDomain.EXTERNAL_LLM,
        measurement_source=accounting.TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=None,
        visible_output_tokens=None,
        supplier_reported_total_tokens=None,
        context_view_tokens=None,
        proof_status=accounting.ProofStatus.UNAVAILABLE,
        reason="supplier_usage_missing",
        raw_response_ref=None,
    )
    assert unavailable.input_tokens is None

    with pytest.raises(ValidationError, match="unavailable"):
        accounting.TokenUsageRecord(**(unavailable.model_dump() | {"input_tokens": 0}))


def test_token_domain_and_proof_status_shapes_are_closed() -> None:
    accounting = require("oamb.contracts.accounting")
    base = {
        "usage_record_id": "c" * 64,
        "attempt_id": "a" * 64,
        "parent_kind": "case",
        "parent_id": "case-1",
        "stage": accounting.TokenStage.CONTEXT_VIEW,
        "operation_kind": "visible_context",
        "token_domain": accounting.TokenDomain.LOCAL_CONTEXT_VIEW,
        "measurement_source": accounting.TokenMeasurementSource.LOCAL_TOKENIZER,
        "input_tokens": None,
        "visible_output_tokens": None,
        "supplier_reported_total_tokens": None,
        "context_view_tokens": None,
        "proof_status": accounting.ProofStatus.UNAVAILABLE,
        "reason": "tokenizer_not_available",
        "raw_response_ref": None,
    }

    unavailable_local = accounting.TokenUsageRecord(**base)
    assert unavailable_local.context_view_tokens is None

    with pytest.raises(ValidationError, match="external LLM"):
        accounting.TokenUsageRecord(
            **(
                base
                | {
                    "token_domain": accounting.TokenDomain.EXTERNAL_LLM,
                    "measurement_source": accounting.TokenMeasurementSource.SUPPLIER_RESPONSE,
                    "context_view_tokens": 10,
                    "proof_status": accounting.ProofStatus.MEASURED_PARTIAL,
                    "reason": None,
                }
            )
        )


def test_money_rejects_float_input() -> None:
    accounting = require("oamb.contracts.accounting")

    with pytest.raises(ValidationError):
        accounting.CostRecord(
            cost_record_id="d" * 64,
            parent_kind="run",
            parent_id="run-1",
            basis=accounting.CostBasis.ESTIMATE_FROM_MEASURED_USAGE,
            indexing_view=accounting.IndexingView.NOT_APPLICABLE,
            amount=0.25,
            currency="USD",
            price_snapshot_id="price-1",
            source_usage_record_ids=(),
            source_resource_record_ids=(),
            proof_status=accounting.ProofStatus.MEASURED_COMPLETE,
            reason=None,
        )


def test_money_json_accepts_only_canonical_decimal_strings() -> None:
    accounting = require("oamb.contracts.accounting")
    payload = {
        "schema_name": "cost_record",
        "schema_version": 1,
        "cost_record_id": "d" * 64,
        "parent_kind": "run",
        "parent_id": "run-1",
        "basis": "estimate_from_measured_usage",
        "indexing_view": "not_applicable",
        "amount": "0.25",
        "currency": "USD",
        "price_snapshot_id": "price-1",
        "source_usage_record_ids": [],
        "source_resource_record_ids": [],
        "proof_status": "measured_complete",
        "reason": None,
    }

    parsed = accounting.CostRecord.model_validate_json(json.dumps(payload))
    assert parsed.amount == Decimal("0.25")
    assert json.loads(parsed.model_dump_json())["amount"] == "0.25"

    with pytest.raises(ValidationError, match="decimal.*string"):
        accounting.CostRecord.model_validate_json(json.dumps(payload | {"amount": 0.25}))
    with pytest.raises(ValidationError, match="canonical decimal"):
        accounting.CostRecord.model_validate_json(json.dumps(payload | {"amount": "0.250"}))


def test_runtime_checkable_ports_accept_structural_fakes() -> None:
    ports = require("oamb.contracts.ports")

    assert {
        "resolve_sources",
        "build_case_manifest",
        "iter_ingestion_plans",
        "iter_case_plans",
        "render_retrieval_query",
        "build_visible_evidence",
        "render_answer",
        "evaluate",
        "finalize_judge",
        "validate_records",
    } <= set(vars(ports.WorkloadPort))
    assert {
        "resolve",
        "capabilities",
        "allocate_ingestion_scope",
        "ingest",
        "wait_ready",
        "inventory",
        "state_digest",
        "retrieve",
        "close",
    } <= set(vars(ports.MemorySystemPort))

    class FakeWorkload:
        def resolve_sources(self) -> Any:
            return object()

        def build_case_manifest(self, dataset_manifest: Any) -> Any:
            return object()

        def iter_ingestion_plans(self, case_manifest: Any) -> Any:
            return ()

        def iter_case_plans(self, case_manifest: Any) -> Any:
            return ()

        def render_retrieval_query(self, case_plan: Any) -> bytes:
            return b"query"

        def build_visible_evidence(self, native_batch: Any, policy: Any) -> Any:
            return object()

        def render_answer(self, case_plan: Any, visible_evidence: Any) -> Any:
            return object()

        def evaluate(self, case_plan: Any, answer: Any) -> Any:
            return object()

        def finalize_judge(self, case_plan: Any, answer: Any, judge_answer: Any) -> Any:
            return object()

        def validate_records(self, records: Any) -> Any:
            return ()

    assert isinstance(FakeWorkload(), ports.WorkloadPort)


def test_t5_implementation_imports_preserve_the_frozen_dependency_direction() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src" / "oamb"
    forbidden = {
        "memory_systems": ("oamb.runtime", "oamb.workloads", "oamb.model_clients"),
        "model_clients": ("oamb.runtime", "oamb.workloads", "oamb.memory_systems"),
        "workloads": ("oamb.runtime", "oamb.memory_systems", "oamb.model_clients"),
        "runtime": ("oamb.memory_systems", "oamb.model_clients", "oamb.artifacts.store"),
    }
    violations: list[str] = []
    for package, forbidden_prefixes in forbidden.items():
        for path in sorted((source_root / package).glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            imported = tuple(
                node.module
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom) and node.module is not None
            ) + tuple(
                alias.name
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
                for alias in node.names
            )
            for module in imported:
                if module.startswith(forbidden_prefixes):
                    violations.append(f"{path.relative_to(source_root)} imports {module}")

    assert violations == []


def test_workload_port_payloads_carry_inputs_needed_by_t5() -> None:
    ports = require("oamb.contracts.ports")

    assert {
        "intended_source_count",
        "ordered_source_units",
        "ordered_case_manifest_entry_ids",
    } <= set(ports.IngestionPlan.__dataclass_fields__)
    assert {
        "question_bytes",
        "reference_payload",
        "prompt_binding_id",
        "output_contract_id",
        "metric_id",
        "judge_binding_id",
    } <= set(ports.CasePlan.__dataclass_fields__)
    assert {"raw_answer", "parsed_value"} <= set(ports.AnswerValue.__dataclass_fields__)
    assert {
        "logical_context_records",
        "ingestion_plan_records",
        "case_records",
    } <= set(ports.WorkloadRecordSet.__dataclass_fields__)


def test_t5_case_and_summary_v2_distinguish_unjudged_from_incorrect() -> None:
    evidence = require("oamb.contracts.evidence")
    reporting = require("oamb.contracts.reporting")

    assert {
        "evaluation_disposition",
        "parsed_answer_sha256",
    } <= set(evidence.CaseRecordV2.model_fields)
    assert {
        "parsed_cases",
        "evaluated_cases",
        "judged_cases",
        "unjudged_cases",
    } <= set(reporting.RunSummaryV2.model_fields)
    assert reporting.RunReportModelV2.model_fields["summary"].annotation is reporting.RunSummaryV2


def test_t5_case_v2_closes_evaluation_disposition_against_terminal_state() -> None:
    evidence = require("oamb.contracts.evidence")
    states = require("oamb.contracts.states")
    evaluated = {
        "case_occurrence_id": "1" * 64,
        "run_id": "run-1",
        "ingestion_occurrence_id": "2" * 64,
        "case_manifest_entry_id": "3" * 64,
        "state": states.CaseState.COMPLETED,
        "retrieval_raw_ref": "4" * 64,
        "prompt_sha256": "5" * 64,
        "answer_raw_ref": "6" * 64,
        "parsed_answer_sha256": "7" * 64,
        "evaluation_raw_ref": "8" * 64,
        "evaluation_disposition": evidence.CaseEvaluationDisposition.DETERMINISTIC_EVALUATED,
        "attempt_ids": (),
        "error_stage": None,
    }
    assert evidence.CaseRecordV2(**evaluated).state == states.CaseState.COMPLETED

    with pytest.raises(ValidationError, match="completed case"):
        evidence.CaseRecordV2(**(evaluated | {"state": states.CaseState.ERROR}))
    with pytest.raises(ValidationError, match="completed case"):
        evidence.CaseRecordV2(
            **(
                evaluated
                | {
                    "retrieval_raw_ref": None,
                    "prompt_sha256": None,
                    "answer_raw_ref": None,
                    "parsed_answer_sha256": None,
                    "evaluation_raw_ref": None,
                    "evaluation_disposition": evidence.CaseEvaluationDisposition.NOT_RUN,
                }
            )
        )
