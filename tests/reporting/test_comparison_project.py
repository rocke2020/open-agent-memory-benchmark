from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import ResolvedPlan, build_resolved_plan
from oamb.contracts.evidence import ValidationResult
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256, openviking_session_id
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.comparison_project import ValidatedCellRoot
from oamb.workloads.visible_evidence import count_o200k_tokens, tokenizer_fingerprint
from tests.benchmark_configuration import load_lme6_configuration

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CASE_COUNT = 6


def _plan(tmp_path: Path) -> ResolvedPlan:
    configuration = load_lme6_configuration()
    dataset = replace(
        configuration.dataset,
        source_sha256=SHA_A,
        case_manifest_hash=SHA_B,
    )
    return build_resolved_plan(replace(configuration, dataset=dataset))


def _lme60_plan() -> ResolvedPlan:
    configuration = load_benchmark_configuration(Path("configs/benchmark.yml"))
    dataset = replace(
        configuration.dataset,
        source_sha256=SHA_A,
        case_manifest_hash=SHA_B,
    )
    return build_resolved_plan(replace(configuration, dataset=dataset))


def _validation(target_hash: str) -> ValidationResult:
    return ValidationResult(
        validation_profile_id="oamb-t8-native-evidence-v1",
        target_hash=target_hash,
        disposition=ValidationDisposition.VALIDATED,
        required_rule_ids=("fixture.valid",),
        executed_rule_ids=("fixture.valid",),
        passed_rule_ids=("fixture.valid",),
        failed_rule_ids=(),
        not_applicable_rule_ids=(),
        missing_rule_ids=(),
        implementation_versions=("fixture.valid@1",),
        issues=(),
    )


def _write_cell_root(
    root: Path,
    plan: ResolvedPlan,
    *,
    cell_index: int,
    metric_numerators: tuple[int, ...],
    case_manifest_hash: str = SHA_B,
    include_accounting: bool = False,
    include_case_content: bool = False,
    indexing_usage_mode: str = "complete",
    raw_question_ids: tuple[str, ...] | None = None,
    skipped_source_count: int = 0,
) -> ValidationResult:
    cell = plan.cells[cell_index]
    run_id = f"run-{cell.cell_id}"
    started_at = datetime(2026, 8, 30, 0, 0, tzinfo=UTC) + timedelta(minutes=cell_index)
    case_count = len(metric_numerators)
    manifest_case_count = len(raw_question_ids) if raw_question_ids is not None else case_count
    case_ids = tuple(f"{index:064x}" for index in range(1, manifest_case_count + 1))
    case_occurrence_ids = tuple(
        f"{index + 100 + cell_index * 10:064x}" for index in range(1, case_count + 1)
    )
    question_ids = raw_question_ids or tuple(
        f"question-{index}" for index in range(1, manifest_case_count + 1)
    )
    assert len(question_ids) == manifest_case_count
    records: list[tuple[str, str, dict[str, object]]] = [
        (
            "run_spec",
            "source/specs/run-spec.json",
            {
                "schema_name": "run_spec",
                "schema_version": 1,
                "run_id": run_id,
                "dataset_manifest_hash": SHA_C,
                "case_manifest_hash": case_manifest_hash,
                "workload_id": plan.dataset.workload_id,
                "memory_system_id": cell.provider_id,
                "code_revision": "fixture-revision",
            },
        ),
        (
            "run_preflight_record",
            "source/specs/run-preflight.json",
            {
                "schema_name": "run_preflight_record",
                "schema_version": 1,
                "run_id": run_id,
                "resolved_plan_hash": plan.resolved_plan_hash,
                "adapter_profile_id": cell.adapter_profile_id,
            },
        ),
        (
            "dataset_manifest",
            "source/specs/dataset-manifest.json",
            {
                "schema_name": "dataset_manifest",
                "schema_version": 1,
                "dataset_id": plan.dataset.dataset_id,
                "revision": plan.dataset.revision,
                "manifest_hash": SHA_C,
                "source_files": [{"sha256": plan.dataset.source_sha256}],
            },
        ),
        (
            "case_manifest",
            "source/specs/case-manifest.json",
            {
                "schema_name": "case_manifest",
                "schema_version": 1,
                "manifest_hash": case_manifest_hash,
                "workload_id": plan.dataset.workload_id,
                "cases": [
                    {
                        "case_manifest_entry_id": case_id,
                        "question_bytes_sha256": f"{index + 200:064x}",
                        "answer_value_sha256": [f"{index + 300:064x}"],
                        "raw_question_id": question_ids[index],
                    }
                    for index, case_id in enumerate(case_ids)
                ],
            },
        ),
        (
            "run_record",
            f"source/run/{run_id}.json",
            {
                "schema_name": "run_record",
                "schema_version": 1,
                "run_id": run_id,
                "started_at": started_at.isoformat(),
                "ended_at": (started_at + timedelta(seconds=30 + cell_index)).isoformat(),
            },
        ),
    ]
    raw_payloads: dict[str, bytes] = {}
    for index, (case_id, occurrence_id, numerator) in enumerate(
        zip(case_ids[:case_count], case_occurrence_ids, metric_numerators, strict=True)
    ):
        query_start = started_at + timedelta(seconds=index + 1)
        query_end = query_start + timedelta(milliseconds=100 + index + cell_index)
        answer_end = query_end + timedelta(milliseconds=50)
        query_attempt_id = f"{index + 400 + cell_index * 20:064x}"
        answer_attempt_id = f"{index + 500 + cell_index * 20:064x}"
        case_document: dict[str, object] = {
            "schema_name": "case_record",
            "schema_version": 3,
            "case_occurrence_id": occurrence_id,
            "run_id": run_id,
            "case_manifest_entry_id": case_id,
            "adapter_profile_id": cell.adapter_profile_id,
            "state": "completed",
            "metric_id": "longmemeval-judge",
            "metric_numerator": numerator,
            "metric_denominator": 1,
            "evaluation_disposition": "judged",
            "attempt_ids": [query_attempt_id, answer_attempt_id],
        }
        injected_context = canonical_json_bytes(
            {
                "evidence_kind": "memory",
                "provider_evidence_identity": f"memory-{cell_index}-{index}",
                "source_unit_id": f"source-{index}",
                "text": f"context for {cell.provider_id} case {index + 1}",
                "occurred_start": None,
                "occurred_end": None,
                "mentioned_at": None,
            }
        )
        visible_ref = hashlib.sha256(injected_context).hexdigest()
        raw_payloads[visible_ref] = injected_context
        case_document.update(
            {
                "visible_evidence_raw_ref": visible_ref,
                "visible_evidence_sha256": visible_ref,
                "visible_evidence_byte_count": len(injected_context),
                "visible_evidence_token_count": count_o200k_tokens(injected_context),
                "visible_evidence_tokenizer_fingerprint": tokenizer_fingerprint(),
            }
        )
        if include_case_content:
            injected_context = canonical_json_bytes(
                {
                    "evidence_kind": "memory",
                    "provider_evidence_identity": f"memory-{cell_index}-{index}",
                    "source_unit_id": (
                        None if cell.provider_id == "openviking" else f"source-{index}"
                    ),
                    "text": (
                        "context <img src=x> javascript: is text onerror=example "
                        f"https://example.invalid/ \u202e {cell.provider_id}"
                    ),
                    "occurred_start": None,
                    "occurred_end": None,
                    "mentioned_at": None,
                }
            )
            answer_text = f"answer <script>bad()</script> from {cell.provider_id}"
            answer_payload = canonical_json_bytes(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": answer_text, "role": "assistant"},
                        }
                    ]
                }
            )
            trace = canonical_json_bytes(
                {
                    "judge_decision": "yes" if numerator == 1 else "no",
                    "parsed_answer_sha256": hashlib.sha256(answer_text.encode()).hexdigest(),
                }
            )
            evaluation_payload = canonical_json_bytes(
                {
                    "denominator": 1,
                    "metric_id": "longmemeval-judge",
                    "numerator": numerator,
                    "parsed_answer_sha256": hashlib.sha256(answer_text.encode()).hexdigest(),
                    "result_sha256": hashlib.sha256(trace).hexdigest(),
                    "trace_hex": trace.hex(),
                }
            )
            visible_ref = hashlib.sha256(injected_context).hexdigest()
            answer_ref = hashlib.sha256(answer_payload).hexdigest()
            evaluation_ref = hashlib.sha256(evaluation_payload).hexdigest()
            raw_payloads.update(
                {
                    visible_ref: injected_context,
                    answer_ref: answer_payload,
                    evaluation_ref: evaluation_payload,
                }
            )
            case_document.update(
                {
                    "answer_raw_ref": answer_ref,
                    "evaluation_raw_ref": evaluation_ref,
                    "parsed_answer_sha256": hashlib.sha256(answer_text.encode()).hexdigest(),
                    "visible_evidence_raw_ref": visible_ref,
                    "visible_evidence_sha256": visible_ref,
                    "visible_evidence_byte_count": len(injected_context),
                    "visible_evidence_token_count": count_o200k_tokens(injected_context),
                    "visible_evidence_tokenizer_fingerprint": tokenizer_fingerprint(),
                }
            )
        records.extend(
            (
                (
                    "case_record",
                    f"source/cases/{occurrence_id}.json",
                    case_document,
                ),
                (
                    "attempt_record",
                    f"source/attempts/{query_attempt_id}.json",
                    {
                        "schema_name": "attempt_record",
                        "schema_version": 4,
                        "attempt_id": query_attempt_id,
                        "run_id": run_id,
                        "parent_kind": "case",
                        "parent_id": occurrence_id,
                        "stage": "memory_query",
                        "started_at": query_start.isoformat(),
                        "ended_at": query_end.isoformat(),
                        "outcome": "succeeded",
                        "raw_response_ref": f"{index + 600:064x}",
                    },
                ),
                (
                    "attempt_record",
                    f"source/attempts/{answer_attempt_id}.json",
                    {
                        "schema_name": "attempt_record",
                        "schema_version": 4,
                        "attempt_id": answer_attempt_id,
                        "run_id": run_id,
                        "parent_kind": "case",
                        "parent_id": occurrence_id,
                        "stage": "answer",
                        "started_at": query_end.isoformat(),
                        "ended_at": answer_end.isoformat(),
                        "outcome": "succeeded",
                        "raw_response_ref": f"{index + 700:064x}",
                    },
                ),
            )
        )

    if include_accounting:
        producer_binding_id = next(
            item.binding_hash for item in plan.model_roles if item.role_id == cell.producer_role_id
        )
        embedding_binding_id = next(
            item.binding_hash for item in plan.model_roles if item.role_id == cell.embedding_role_id
        )
        records.extend(
            _accounting_records(
                run_id,
                started_at,
                producer_binding_id=producer_binding_id,
                embedding_binding_id=embedding_binding_id,
                indexing_usage_mode=indexing_usage_mode,
            )
        )
        if skipped_source_count:
            ingestion_plan = next(
                document
                for record_kind, _relative_path, document in records
                if record_kind == "ingestion_plan_record"
            )
            ingestion_plan.update(
                ordered_source_unit_ids=[f"source-{index}" for index in range(2)],
                accepted_source_unit_ids=["source-0"],
                rejected_source_unit_ids=[],
                skipped_source_unit_ids=["source-1"],
            )

    entries: list[dict[str, object]] = []
    for ordinal, (record_kind, relative_path, document) in enumerate(records, start=1):
        content = canonical_json_bytes(document)
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            {
                "schema_name": "capsule_manifest_entry",
                "schema_version": 1,
                "record_kind": record_kind,
                "record_id": f"record-{ordinal}",
                "relative_path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    for raw_reference, payload in raw_payloads.items():
        relative_path = f"source/raw/{raw_reference}.json.gz"
        content = gzip.compress(payload, mtime=0)
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append(
            {
                "schema_name": "capsule_manifest_entry",
                "schema_version": 1,
                "record_kind": "raw_payload",
                "record_id": raw_reference,
                "relative_path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    source_manifest_hash = canonical_sha256([run_id, entries])
    manifest = {
        "schema_name": "capsule_manifest",
        "schema_version": 1,
        "capsule_id": canonical_sha256(["capsule", run_id, source_manifest_hash]),
        "run_id": run_id,
        "run_spec_hash": canonical_sha256(records[0][2]),
        "source_entries": entries,
        "source_manifest_hash": source_manifest_hash,
    }
    (root / "capsule-manifest.json").write_bytes(canonical_json_bytes(manifest))
    return _validation(source_manifest_hash)


def _accounting_records(
    run_id: str,
    started_at: datetime,
    *,
    producer_binding_id: str,
    embedding_binding_id: str,
    indexing_usage_mode: str,
) -> tuple[tuple[str, str, dict[str, object]], ...]:
    attempt_ids = {
        "indexing": "a" * 64,
        "indexing_failed": "1" * 64,
        "indexing_retry": "2" * 64,
        "retrieval": "b" * 64,
        "answer": "c" * 64,
        "judge": "d" * 64,
        "judge_retry": "e" * 64,
    }
    attempts = tuple(
        (
            "attempt_record",
            f"source/attempts/{attempt_id}.json",
            {
                "schema_name": "attempt_record",
                "schema_version": 4,
                "attempt_id": attempt_id,
                "run_id": run_id,
                "parent_kind": "case" if stage != "memory_ingest" else "ingestion_plan",
                "parent_id": "case-1" if stage != "memory_ingest" else "plan-1",
                "stage": stage,
                "started_at": started_at.isoformat(),
                "ended_at": (started_at + timedelta(milliseconds=100)).isoformat(),
                "outcome": outcome,
                "retry_of_attempt_id": retry_of,
                "index_contribution": (
                    "final" if stage == "memory_ingest" and outcome == "succeeded" else "none"
                ),
                "raw_response_ref": "f" * 64,
            },
        )
        for attempt_id, stage, outcome, retry_of in (
            (attempt_ids["indexing"], "memory_ingest", "succeeded", None),
            (attempt_ids["indexing_failed"], "memory_ingest", "failed", None),
            (
                attempt_ids["indexing_retry"],
                "memory_ingest",
                "succeeded",
                attempt_ids["indexing_failed"],
            ),
            (attempt_ids["retrieval"], "memory_query", "succeeded", None),
            (attempt_ids["judge"], "judge", "failed", None),
            (
                attempt_ids["judge_retry"],
                "judge",
                "succeeded",
                attempt_ids["judge"],
            ),
        )
    )

    ingestion_plan = (
        "ingestion_plan_record",
        "source/ingestion-plans/plan-1.json",
        {
            "schema_name": "ingestion_plan_record",
            "schema_version": 2,
            "ingestion_plan_id": "plan-1",
            "ingestion_occurrence_id": "plan-1",
            "state": "sealed",
            "ordered_dispatch_attempt_ids": [
                attempt_ids["indexing"],
                attempt_ids["indexing_retry"],
            ],
        },
    )

    def usage(
        label: str,
        attempt_key: str,
        stage: str,
        *,
        owner_kind: str = "model_role",
        owner_id: str,
        values: tuple[int | None, int | None, int | None, int | None, int | None],
        proof_status: str,
        billing_complete: bool,
    ) -> tuple[str, str, dict[str, object]]:
        names = (
            "input_tokens",
            "visible_output_tokens",
            "supplier_reported_total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        )
        covered = [name for name, value in zip(names, values, strict=True) if value is not None]
        unavailable = [name for name, value in zip(names, values, strict=True) if value is None]
        return (
            "token_usage_record",
            f"source/usage/{label}.json",
            {
                "schema_name": "token_usage_record",
                "schema_version": 5,
                "usage_record_id": canonical_sha256(["usage", run_id, label]),
                "attempt_id": attempt_ids[attempt_key],
                "stage": stage,
                "budget_owner_kind": owner_kind,
                "budget_owner_id": owner_id,
                "input_tokens": values[0],
                "visible_output_tokens": values[1],
                "supplier_reported_total_tokens": values[2],
                "cached_input_tokens": values[3],
                "reasoning_tokens": values[4],
                "covered_dimensions": covered,
                "unavailable_dimensions": unavailable,
                "not_applicable_dimensions": [],
                "proof_status": proof_status,
                "billing_complete": billing_complete,
            },
        )

    usage_records: tuple[tuple[str, str, dict[str, object]], ...] = (
        usage(
            "indexing-producer",
            "indexing",
            "memory_ingest",
            owner_id=producer_binding_id,
            values=(100, 20, 120, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
        usage(
            "indexing-embedding",
            "indexing",
            "memory_ingest",
            owner_id=embedding_binding_id,
            values=(1000, 200, 1200, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
        usage(
            "indexing-failed-producer",
            "indexing_failed",
            "memory_ingest",
            owner_id=producer_binding_id,
            values=(10000, 2000, 12000, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
        usage(
            "indexing-failed-embedding",
            "indexing_failed",
            "memory_ingest",
            owner_id=embedding_binding_id,
            values=(100000, 20000, 120000, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
        usage(
            "indexing-retry-producer",
            "indexing_retry",
            "memory_ingest",
            owner_id=producer_binding_id,
            values=(180, 40, 220, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
        usage(
            "indexing-retry-embedding",
            "indexing_retry",
            "memory_ingest",
            owner_id=embedding_binding_id,
            values=(1800, 400, 2200, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
        usage(
            "retrieval",
            "retrieval",
            "memory_query",
            owner_id="retrieval-role",
            values=(None, None, None, None, None),
            proof_status="unavailable",
            billing_complete=False,
        ),
        usage(
            "answer",
            "answer",
            "answer",
            owner_id="answer-role",
            values=(50, 5, 57, 0, 2),
            proof_status="measured_complete",
            billing_complete=True,
        ),
        usage(
            "judge",
            "judge",
            "judge",
            owner_id="judge-role",
            values=(None, None, None, None, None),
            proof_status="unavailable",
            billing_complete=False,
        ),
    )
    operation_usage_records = (
        usage(
            "indexing-operation",
            "indexing",
            "memory_ingest",
            owner_kind="provider_operation",
            owner_id="memory_ingest-operation",
            values=(7, 3, 10, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
        usage(
            "indexing-retry-operation",
            "indexing_retry",
            "memory_ingest",
            owner_kind="provider_operation",
            owner_id="memory_ingest-operation",
            values=(11, 4, 15, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
    )
    if indexing_usage_mode == "missing":
        usage_records = tuple(
            item
            for item in usage_records
            if item[2]["attempt_id"] != attempt_ids["indexing_retry"]
            or item[2]["budget_owner_id"] != producer_binding_id
        )
    elif indexing_usage_mode == "duplicate":
        duplicate = usage(
            "indexing-retry-producer-duplicate",
            "indexing_retry",
            "memory_ingest",
            owner_id=producer_binding_id,
            values=(1, 1, 2, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        )
        usage_records = (*usage_records, duplicate)
    elif indexing_usage_mode in {"operation_fallback", "operation_duplicate"}:
        usage_records = tuple(
            item
            for item in usage_records
            if item[2]["budget_owner_id"] != producer_binding_id
            or item[2]["attempt_id"]
            not in {
                attempt_ids["indexing"],
                attempt_ids["indexing_retry"],
            }
        )
        usage_records = (*usage_records, *operation_usage_records)
        if indexing_usage_mode == "operation_duplicate":
            duplicate = usage(
                "indexing-retry-operation-duplicate",
                "indexing_retry",
                "memory_ingest",
                owner_kind="provider_operation",
                owner_id="memory_ingest-operation",
                values=(1, 1, 2, None, None),
                proof_status="measured_partial",
                billing_complete=False,
            )
            usage_records = (*usage_records, duplicate)
    elif indexing_usage_mode == "model_and_operation":
        usage_records = (*usage_records, *operation_usage_records)
    elif indexing_usage_mode != "complete":
        raise ValueError("unknown indexing usage mode")
    resources = (
        (
            "resource_usage_record",
            "source/resources/request-wall.json",
            {
                "schema_name": "resource_usage_record",
                "schema_version": 2,
                "resource_record_id": canonical_sha256(["resource", run_id, "request-wall"]),
                "attempt_id": attempt_ids["retrieval"],
                "stage": "memory_query",
                "dimension_id": "provider_request_wall_seconds_v1",
                "value": Decimal("0.100"),
                "unit": "seconds",
                "proof_status": "measured_complete",
            },
        ),
        (
            "resource_usage_record",
            "source/resources/peak-memory.json",
            {
                "schema_name": "resource_usage_record",
                "schema_version": 2,
                "resource_record_id": canonical_sha256(["resource", run_id, "peak-memory"]),
                "attempt_id": attempt_ids["retrieval"],
                "stage": "memory_query",
                "dimension_id": "peak_memory_bytes",
                "value": 4096,
                "unit": "bytes",
                "proof_status": "measured_complete",
            },
        ),
    )
    costs = (
        (
            "cost_record",
            "source/costs/retrieval.json",
            {
                "schema_name": "cost_record",
                "schema_version": 2,
                "cost_record_id": canonical_sha256(["cost", run_id, "retrieval"]),
                "attempt_id": attempt_ids["retrieval"],
                "basis": "actual_supplier_charge",
                "amount": Decimal("0.0125"),
                "currency": "CNY",
                "proof_status": "measured_complete",
            },
        ),
        (
            "cost_record",
            "source/costs/indexing.json",
            {
                "schema_name": "cost_record",
                "schema_version": 2,
                "cost_record_id": canonical_sha256(["cost", run_id, "indexing"]),
                "attempt_id": attempt_ids["indexing"],
                "basis": "actual_supplier_charge",
                "amount": None,
                "currency": None,
                "proof_status": "unavailable",
            },
        ),
    )
    return (ingestion_plan, *attempts, *usage_records, *resources, *costs)


def _sources(tmp_path: Path, plan: ResolvedPlan) -> dict[str, ValidatedCellRoot]:
    numerators = ((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 0, 0), (1, 1, 0, 0, 0, 0))
    sources: dict[str, ValidatedCellRoot] = {}
    for index, cell in enumerate(plan.cells):
        root = tmp_path / "capsules" / cell.cell_id
        validation = _write_cell_root(
            root,
            plan,
            cell_index=index,
            metric_numerators=numerators[index],
        )
        sources[cell.cell_id] = ValidatedCellRoot(root=root, validation_result=validation)
    return sources


def _lme60_sources(
    tmp_path: Path,
    plan: ResolvedPlan,
    outcome_sets: tuple[tuple[int, ...], ...],
) -> dict[str, ValidatedCellRoot]:
    from oamb.workloads.longmemeval import LME60_EXPECTED_QUESTION_IDS

    sources: dict[str, ValidatedCellRoot] = {}
    for index, (cell, outcomes) in enumerate(zip(plan.cells, outcome_sets, strict=True)):
        root = tmp_path / "lme60-capsules" / cell.cell_id
        validation = _write_cell_root(
            root,
            plan,
            cell_index=index,
            metric_numerators=outcomes,
            raw_question_ids=LME60_EXPECTED_QUESTION_IDS,
        )
        sources[cell.cell_id] = ValidatedCellRoot(root=root, validation_result=validation)
    return sources


def _lme60_one_question_sources(
    tmp_path: Path,
    plan: ResolvedPlan,
) -> dict[str, ValidatedCellRoot]:
    from oamb.workloads.longmemeval import LME60_EXPECTED_QUESTION_IDS

    sources: dict[str, ValidatedCellRoot] = {}
    for index, cell in enumerate(plan.cells):
        root = tmp_path / "lme60-one-question-capsules" / cell.cell_id
        validation = _write_cell_root(
            root,
            plan,
            cell_index=index,
            metric_numerators=((1,), (0,), (1,))[index],
            raw_question_ids=LME60_EXPECTED_QUESTION_IDS,
        )
        sources[cell.cell_id] = ValidatedCellRoot(root=root, validation_result=validation)
    return sources


def _sources_with_accounting_mode(
    tmp_path: Path,
    plan: ResolvedPlan,
    indexing_usage_mode: str,
) -> dict[str, ValidatedCellRoot]:
    sources = _sources(tmp_path, plan)
    target_cell = plan.cells[0]
    target_root = tmp_path / f"accounting-{indexing_usage_mode}" / target_cell.cell_id
    target_validation = _write_cell_root(
        target_root,
        plan,
        cell_index=0,
        metric_numerators=(1, 1, 1, 1, 0, 0),
        include_accounting=True,
        indexing_usage_mode=indexing_usage_mode,
    )
    sources[target_cell.cell_id] = ValidatedCellRoot(
        root=target_root,
        validation_result=target_validation,
    )
    return sources


def _openviking_indexing_snapshot(
    plan: ResolvedPlan,
    *,
    raw_mode: str = "complete",
    canonical_measured: bool = False,
) -> SimpleNamespace:
    cell = plan.cells[2]
    attempt_ids = ("a" * 64, "2" * 64)
    source_ids = ("source-one", "source-two")
    ingestion_occurrence_id = "openviking-indexing-occurrence"
    values = ((11, 3, 14, 2, 1, 7, 21), (17, 5, 22, 4, 3, 9, 31))
    raw_payloads: dict[str, bytes] = {}
    readiness_refs: list[str] = []

    def add_completed_task(
        source_id: str,
        usage_values: tuple[int, int, int, int, int, int, int],
        *,
        task_suffix: str,
        accepted_task_suffix: str | None = None,
    ) -> str:
        prompt, completion, llm_total, cached, reasoning, embedding, combined = usage_values
        session_id = openviking_session_id(ingestion_occurrence_id, source_id)
        archive_uri = f"viking://user/test/sessions/{session_id}/history/archive_001"
        accepted_payload = canonical_json_bytes(
            {
                "status": "ok",
                "result": {
                    "session_id": session_id,
                    "status": "accepted",
                    "task_id": f"task-{accepted_task_suffix or task_suffix}",
                    "archive_uri": archive_uri,
                    "archived": True,
                },
                "error": None,
                "telemetry": None,
                "profile": None,
            }
        )
        accepted_reference = hashlib.sha256(accepted_payload).hexdigest()
        raw_payloads[accepted_reference] = accepted_payload
        readiness_refs.append(accepted_reference)
        payload = canonical_json_bytes(
            {
                "status": "ok",
                "result": {
                    "task_id": f"task-{task_suffix}",
                    "task_type": "session_commit",
                    "status": "completed",
                    "resource_id": session_id,
                    "result": {
                        "session_id": session_id,
                        "archive_uri": archive_uri,
                        "token_usage": {
                            "llm": {
                                "prompt_tokens": prompt,
                                "completion_tokens": completion,
                                "total_tokens": llm_total,
                                "cached_tokens": cached,
                                "reasoning_tokens": reasoning,
                            },
                            "embedding": {"total_tokens": embedding},
                            "total": {
                                "total_tokens": combined,
                                "cached_tokens": cached,
                                "reasoning_tokens": reasoning,
                            },
                        },
                    },
                },
                "error": None,
                "telemetry": None,
                "profile": None,
            }
        )
        reference = hashlib.sha256(payload).hexdigest()
        raw_payloads[reference] = payload
        readiness_refs.append(reference)
        return reference

    first_ref = add_completed_task(
        source_ids[0],
        values[0],
        task_suffix="one",
        accepted_task_suffix=(
            "one-mismatch"
            if raw_mode in {"mismatched_commit_task", "missing_mismatched_commit_task"}
            else None
        ),
    )
    if raw_mode == "same_ref_repeated":
        readiness_refs.append(first_ref)
    if raw_mode == "conflicting_snapshot":
        add_completed_task(source_ids[0], values[0], task_suffix="one-conflict")
    if raw_mode not in {"missing_snapshot", "missing_mismatched_commit_task"}:
        second_values = values[1]
        if raw_mode == "broken_equation":
            second_values = (17, 5, 23, 4, 3, 9, 32)
        add_completed_task(source_ids[1], second_values, task_suffix="two")
    if raw_mode == "unknown_session":
        add_completed_task("unknown-source", values[1], task_suffix="unknown")

    producer_binding_id = next(
        item.binding_hash for item in plan.model_roles if item.role_id == cell.producer_role_id
    )
    usage_records = tuple(
        {
            "attempt_id": attempt_id,
            "stage": "memory_ingest",
            "budget_owner_kind": "model_role",
            "budget_owner_id": producer_binding_id,
            "input_tokens": total - output if canonical_measured else None,
            "visible_output_tokens": output if canonical_measured else None,
            "supplier_reported_total_tokens": total if canonical_measured else None,
            "cached_input_tokens": None,
            "reasoning_tokens": None,
            "covered_dimensions": (
                (
                    "input_tokens",
                    "visible_output_tokens",
                    "supplier_reported_total_tokens",
                )
                if canonical_measured
                else ()
            ),
            "unavailable_dimensions": (
                ("cached_input_tokens", "reasoning_tokens")
                if canonical_measured
                else (
                    "input_tokens",
                    "visible_output_tokens",
                    "supplier_reported_total_tokens",
                    "cached_input_tokens",
                    "reasoning_tokens",
                )
            ),
            "proof_status": "measured_partial" if canonical_measured else "unavailable",
            "billing_complete": False,
        }
        for attempt_id, total, output in zip(attempt_ids, (120, 220), (20, 40), strict=True)
    )
    return SimpleNamespace(
        cell=cell,
        ingestion_plans=(
            {
                "adapter_profile_id": cell.adapter_profile_id,
                "ingestion_occurrence_id": ingestion_occurrence_id,
                "ordered_dispatch_attempt_ids": list(attempt_ids),
                "ordered_source_unit_ids": list(source_ids),
                "readiness_evidence_refs": readiness_refs,
            },
        ),
        token_usage=usage_records,
        raw_payloads=raw_payloads,
    )


def _content_sources(tmp_path: Path, plan: ResolvedPlan) -> dict[str, ValidatedCellRoot]:
    numerators = ((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 0, 0), (1, 1, 0, 0, 0, 0))
    sources: dict[str, ValidatedCellRoot] = {}
    for index, cell in enumerate(plan.cells):
        root = tmp_path / "content-capsules" / cell.cell_id
        validation = _write_cell_root(
            root,
            plan,
            cell_index=index,
            metric_numerators=numerators[index],
            include_case_content=True,
        )
        sources[cell.cell_id] = ValidatedCellRoot(root=root, validation_result=validation)
    return sources


class _TableHeaderCounter(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._in_table = False
        self._current = 0
        self.counts: list[int] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag == "table":
            self._in_table = True
            self._current = 0
        elif tag == "th" and self._in_table:
            self._current += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "table":
            self.counts.append(self._current)
            self._in_table = False


def _local_question_content() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "case_manifest_entry_id": f"{index:064x}",
            "raw_question_id": f"question-{index}",
            "question_type": "single-session-user",
            "question": (
                "What happened? <script>bad()</script> javascript: is text "
                "onerror=example https://example.invalid/ \u202e"
                if index == 1
                else f"Question {index}?"
            ),
            "gold_answer": f"Gold answer {index}",
            "answer_sessions": (
                {
                    "session_id": f"session-{index}",
                    "timestamp": "2026-08-30T00:00:00+00:00",
                    "messages": (
                        {
                            "role": "user",
                            "content": f"Source answer {index}",
                            "has_answer": True,
                        },
                    ),
                },
            ),
            "has_answer_label_mismatch": False,
        }
        for index in range(1, CASE_COUNT + 1)
    )


def test_project_emits_exact_three_pairs_and_eighteen_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources = _sources(tmp_path, plan)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "report-a",
    )
    export = json.loads(built.export_path.read_bytes())

    assert export["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 6,
        "provider_specific_result_count": 18,
    }
    assert [(cell["judged_numerator"], cell["judged_denominator"]) for cell in export["cells"]] == [
        (4, 6),
        (3, 6),
        (2, 6),
    ]
    assert [
        (item["left_cell_id"], item["right_cell_id"], item["signed_delta"])
        for item in export["comparisons"]
    ] == [
        ("hindsight-lme6", "mem0-lme6", {"denominator": 6, "numerator": 1}),
        ("hindsight-lme6", "openviking-lme6", {"denominator": 3, "numerator": 1}),
        ("mem0-lme6", "openviking-lme6", {"denominator": 6, "numerator": 1}),
    ]
    assert sum(len(cell["results"]) for cell in export["cells"]) == 18
    assert len(built.comparison_paths) == 3
    assert all(path.is_file() for path in built.comparison_paths)
    assert [cell["observed_time"]["provider_request"]["count"] for cell in export["cells"]] == [
        6,
        6,
        6,
    ]
    assert export["retrieval_generation"] == "disabled"
    assert all(item["runtime_proof_state"] == "unavailable" for item in export["retrieval"])
    assert any("runtime retrieval proof" in item for item in export["limitations"])


def test_project_displays_partial_ingestion_without_removing_judged_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources: dict[str, ValidatedCellRoot] = {}
    for index, cell in enumerate(plan.cells):
        root = tmp_path / "partial-capsules" / cell.cell_id
        validation = _write_cell_root(
            root,
            plan,
            cell_index=index,
            metric_numerators=(1, 1, 1, 1, 0, 0),
            include_accounting=True,
            skipped_source_count=1,
        )
        sources[cell.cell_id] = ValidatedCellRoot(root=root, validation_result=validation)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "partial-report",
    )
    export = json.loads(built.export_path.read_bytes())

    assert all(
        cell["ingestion"]
        == {
            "partial_history_count": 1,
            "skipped_source_count": 1,
        }
        for cell in export["cells"]
    )
    assert all(cell["judged_case_count"] == 6 for cell in export["cells"])
    assert "Partial ingestion" in built.html_path.read_text()


def test_project_reports_separated_accounting_and_preserves_unavailable_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources: dict[str, ValidatedCellRoot] = {}
    numerators = ((1, 1, 1, 1, 0, 0), (1, 1, 1, 0, 0, 0), (1, 1, 0, 0, 0, 0))
    for index, cell in enumerate(plan.cells):
        root = tmp_path / "accounting-capsules" / cell.cell_id
        validation = _write_cell_root(
            root,
            plan,
            cell_index=index,
            metric_numerators=numerators[index],
            include_accounting=True,
        )
        sources[cell.cell_id] = ValidatedCellRoot(root=root, validation_result=validation)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "accounting-report",
    )
    export = json.loads(built.export_path.read_bytes())
    accounting = export["cells"][0]["accounting"]

    assert accounting["attempts"] == {
        "attempt_count": 18,
        "budget_exceeded_count": 0,
        "cancelled_count": 0,
        "failed_count": 2,
        "measurement_coverage": "complete",
        "retry_count": 2,
        "succeeded_count": 16,
        "unknown_outcome_count": 0,
    }
    assert set(accounting["tokens"]) == {"indexing", "retrieval", "answer", "judge"}
    assert accounting["tokens"]["indexing"]["supplier_usage_coverage"] == {
        "billing_complete_record_count": 0,
        "measured_record_count": 2,
        "record_count": 2,
        "status": "measured_partial",
        "unavailable_record_count": 0,
    }
    assert accounting["tokens"]["indexing"]["totals"]["input_tokens"] == {
        "measured_record_count": 2,
        "status": "measured_complete",
        "unavailable_record_count": 0,
        "value": 280,
    }
    assert accounting["tokens"]["indexing"]["totals"]["supplier_reported_total_tokens"] == {
        "measured_record_count": 2,
        "status": "measured_complete",
        "unavailable_record_count": 0,
        "value": 340,
    }
    assert accounting["tokens"]["retrieval"]["totals"]["input_tokens"] == {
        "measured_record_count": 0,
        "status": "unavailable",
        "unavailable_record_count": 1,
        "value": "unavailable",
    }
    assert accounting["tokens"]["answer"]["totals"]["reasoning_tokens"]["value"] == 2
    assert accounting["tokens"]["judge"]["supplier_usage_coverage"]["status"] == ("unavailable")
    assert accounting["resources"]["peak_memory_bytes"]["value"] == "4096"
    assert accounting["resources"]["storage_bytes"] == {
        "aggregation": "terminal_snapshot",
        "measured_record_count": 0,
        "status": "unavailable",
        "unavailable_record_count": 0,
        "unit": "bytes",
        "value": "unavailable",
    }
    assert accounting["cost"] == {
        "actual_supplier_charge": {"CNY": "0.0125"},
        "billing_coverage": {
            "measured_record_count": 1,
            "record_count": 2,
            "status": "measured_partial",
            "unavailable_record_count": 1,
        },
        "currencies": ["CNY"],
    }
    assert accounting["measurement_coverage"] == {
        "attempts": "complete",
        "cost": "measured_partial",
        "resources": "measured_partial",
        "tokens": {
            "answer": "measured_complete",
            "indexing": "measured_partial",
            "judge": "unavailable",
            "retrieval": "unavailable",
        },
    }
    assert accounting["answer_visible_context_tokens"] == {
        "case_count": 6,
        "measured_case_count": 6,
        "mean": "52",
        "status": "measured_complete",
        "total": 312,
    }
    rendered = built.html_path.read_text(encoding="utf-8")
    assert "<th>Answer accuracy</th>" in rendered
    assert "<th>Context tokens</th>" in rendered
    assert "Five decision metrics" in rendered
    assert "312 total / 52 mean" in rendered
    assert "<th>Indexing tokens</th>" in rendered
    assert "<th>Retrieval latency (s)</th>" in rendered
    assert "<th>Indexing time (s)</th>" in rendered
    assert "/ coverage</th>" not in rendered
    assert "<th>Configured</th>" not in rendered
    assert "<th>Runtime</th>" in rendered
    assert "Secondary accounting" in rendered
    assert "Answer input" in rendered
    assert "Failures / retries" in rendered
    assert "CNY 0.0125" in rendered
    assert "Retrieval supplier usage" not in rendered
    assert "Judge supplier usage" not in rendered
    assert "Peak memory" not in rendered
    assert "Storage" not in rendered
    assert "Measurement coverage" not in rendered
    parser = _TableHeaderCounter()
    parser.feed(rendered)
    assert parser.counts
    assert max(parser.counts) <= 7


def test_indexing_measurement_note_explains_partial_and_unavailable_coverage() -> None:
    from oamb.reporting import comparison_project

    def cell(
        provider_id: str,
        measured: int,
        records: int,
        status: str,
        *,
        total_tokens: int | None,
        reasoning_tokens: int | None,
    ) -> dict[str, object]:
        return {
            "provider_id": provider_id,
            "accounting": {
                "tokens": {
                    "indexing": {
                        "supplier_usage_coverage": {
                            "measured_record_count": measured,
                            "record_count": records,
                            "status": status,
                        },
                        "totals": {
                            "supplier_reported_total_tokens": {
                                "status": (
                                    "measured_complete"
                                    if total_tokens is not None
                                    else "unavailable"
                                ),
                                "value": total_tokens
                                if total_tokens is not None
                                else "unavailable",
                            },
                            "reasoning_tokens": {
                                "status": (
                                    "measured_complete"
                                    if reasoning_tokens is not None
                                    else "unavailable"
                                ),
                                "value": (
                                    reasoning_tokens
                                    if reasoning_tokens is not None
                                    else "unavailable"
                                ),
                            },
                        },
                    }
                }
            },
        }

    note = comparison_project._indexing_measurement_note(
        (
            cell(
                "hindsight",
                300,
                300,
                "measured_partial",
                total_tokens=5_764_508,
                reasoning_tokens=None,
            ),
            cell(
                "mem0",
                0,
                300,
                "unavailable",
                total_tokens=None,
                reasoning_tokens=None,
            ),
            cell(
                "openviking",
                300,
                300,
                "measured_complete",
                total_tokens=6_876_300,
                reasoning_tokens=1_436_612,
            ),
        )
    )

    assert "supplier-reported total tokens" in note
    assert "provider-defined input plus output" in note
    assert "Reasoning is included once only when it is inside that supplier total" in note
    assert "reasoning subset is never added again" in note
    assert "reasoning is unavailable, the report does not infer or add it" in note
    assert "successful logical producer records" in note
    assert "Embedding and failed physical attempts are excluded" in note
    assert "hindsight: 300/300 metered" in note
    assert "displayed total covers all producer records" in note
    assert "hindsight reasoning breakdown is unavailable, not zero" in note
    assert "5,764,508 is the measured supplier total, with no inferred reasoning added" in note
    assert "measurement remains partial" in note
    assert "mem0: 0/300 metered" in note
    assert "openviking: 300/300 metered" in note
    assert "openviking reasoning: 1,436,612, already included in its total" in note
    assert "300/600" not in note
    assert "0/600" not in note
    assert "0/602" not in note
    assert note.count("supplier token totals are unavailable, not zero") == 1


@pytest.mark.parametrize(
    "indexing_usage_mode",
    ("missing", "duplicate", "operation_duplicate"),
)
def test_indexing_headline_rejects_missing_or_duplicate_producer_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    indexing_usage_mode: str,
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources = _sources_with_accounting_mode(tmp_path, plan, indexing_usage_mode)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    with pytest.raises(comparison_project.ComparisonProjectError, match="producer usage"):
        comparison_project.build_comparison_project(
            plan,
            sources,
            output_root=tmp_path / f"report-{indexing_usage_mode}",
        )


@pytest.mark.parametrize(
    ("indexing_usage_mode", "expected_total"),
    (("operation_fallback", 25), ("model_and_operation", 340)),
)
def test_indexing_headline_uses_unique_operation_fallback_but_prefers_model_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    indexing_usage_mode: str,
    expected_total: int,
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources = _sources_with_accounting_mode(tmp_path, plan, indexing_usage_mode)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / f"report-{indexing_usage_mode}",
    )
    export = json.loads(built.export_path.read_bytes())
    indexing = export["cells"][0]["accounting"]["tokens"]["indexing"]

    assert indexing["supplier_usage_coverage"]["record_count"] == 2
    assert indexing["supplier_usage_coverage"]["measured_record_count"] == 2
    assert indexing["totals"]["supplier_reported_total_tokens"]["value"] == expected_total


@pytest.mark.parametrize("raw_mode", ("complete", "same_ref_repeated"))
def test_openviking_indexing_headline_recovers_sealed_task_llm_usage(
    tmp_path: Path,
    raw_mode: str,
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    snapshot = _openviking_indexing_snapshot(plan, raw_mode=raw_mode)

    indexing = comparison_project._indexing_token_stage_document(plan, cast(Any, snapshot))

    assert indexing["supplier_usage_coverage"] == {
        "billing_complete_record_count": 0,
        "measured_record_count": 2,
        "record_count": 2,
        "status": "measured_complete",
        "unavailable_record_count": 0,
    }
    assert indexing["totals"]["input_tokens"]["value"] == 28
    assert indexing["totals"]["visible_output_tokens"]["value"] == 8
    assert indexing["totals"]["supplier_reported_total_tokens"]["value"] == 36
    assert indexing["totals"]["cached_input_tokens"]["value"] == 6
    assert indexing["totals"]["reasoning_tokens"]["value"] == 4


def test_openviking_indexing_headline_ignores_unrelated_task_snapshot(
    tmp_path: Path,
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    snapshot = _openviking_indexing_snapshot(plan, raw_mode="unknown_session")

    indexing = comparison_project._indexing_token_stage_document(plan, cast(Any, snapshot))

    assert indexing["supplier_usage_coverage"]["record_count"] == 2
    assert indexing["totals"]["supplier_reported_total_tokens"]["value"] == 36


def test_openviking_indexing_headline_keeps_unavailable_when_task_snapshot_is_missing(
    tmp_path: Path,
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    snapshot = _openviking_indexing_snapshot(plan, raw_mode="missing_snapshot")

    indexing = comparison_project._indexing_token_stage_document(plan, cast(Any, snapshot))

    assert indexing["supplier_usage_coverage"] == {
        "billing_complete_record_count": 0,
        "measured_record_count": 0,
        "record_count": 2,
        "status": "unavailable",
        "unavailable_record_count": 2,
    }
    assert indexing["totals"]["supplier_reported_total_tokens"]["value"] == "unavailable"


@pytest.mark.parametrize(
    "raw_mode",
    (
        "conflicting_snapshot",
        "broken_equation",
        "mismatched_commit_task",
        "missing_mismatched_commit_task",
    ),
)
def test_openviking_indexing_headline_rejects_incomplete_or_inconsistent_task_usage(
    tmp_path: Path,
    raw_mode: str,
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    snapshot = _openviking_indexing_snapshot(plan, raw_mode=raw_mode)

    with pytest.raises(
        comparison_project.ComparisonProjectError,
        match="OpenViking indexing usage",
    ):
        comparison_project._indexing_token_stage_document(plan, cast(Any, snapshot))


def test_openviking_indexing_headline_prefers_measured_canonical_usage(
    tmp_path: Path,
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    snapshot = _openviking_indexing_snapshot(
        plan,
        raw_mode="broken_equation",
        canonical_measured=True,
    )

    indexing = comparison_project._indexing_token_stage_document(plan, cast(Any, snapshot))

    assert indexing["supplier_usage_coverage"]["status"] == "measured_partial"
    assert indexing["totals"]["supplier_reported_total_tokens"]["value"] == 340


def test_project_revalidates_roots_rejects_manifest_drift_and_builds_offline_deterministically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources = _sources(tmp_path, plan)
    supplied_by_root = {source.root: source.validation_result for source in sources.values()}
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: supplied_by_root[root],
    )

    first = comparison_project.build_comparison_project(
        plan, sources, output_root=tmp_path / "report-a"
    )
    second = comparison_project.build_comparison_project(
        plan, sources, output_root=tmp_path / "report-b"
    )
    assert first.export_path.read_bytes() == second.export_path.read_bytes()
    assert first.html_path.read_bytes() == second.html_path.read_bytes()
    html = first.html_path.read_text(encoding="utf-8")
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "@media (prefers-color-scheme: dark)" in html
    assert "Content-Security-Policy" in html
    assert "connect-src 'none'" in html
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html
    assert "Provider A" in html
    assert "Provider B" in html
    assert "Accuracy delta (A − B)" in html
    assert "Positive favors Provider A; negative favors Provider B." in html
    assert "percentage points" in html
    assert "<th>Comparable</th>" not in html
    assert "Scale rank" not in html
    assert "low &lt; high &lt; max" in html
    assert "frozen T10" not in html
    assert (
        "Runtime models were verified against this comparison's frozen configured bindings" in html
    )
    assert "The complete bindings remain in report.json." in html
    assert re.search(r"\b\d+(?:\.\d+)? us\b", html) is None
    assert re.search(r"\b\d+(?:\.\d+)? ms\b", html) is None
    assert '<a href="report.json" download>Download report.json</a>' in html
    assert "<pre>{" not in html
    assert "Five decision metrics:" in html
    assert (
        "Answer accuracy · Context tokens · Indexing tokens · Retrieval latency · Indexing time"
        in html
    )
    assert html.index("<th>Retrieval latency (s)</th>") < html.index("<th>Indexing time (s)</th>")
    assert "Concise metric comparison" in html
    assert "Analysis unavailable" in html

    stale_root = next(iter(supplied_by_root))
    stale_validation = supplied_by_root[stale_root]
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: (
            stale_validation.model_copy(update={"target_hash": "f" * 64})
            if root == stale_root
            else supplied_by_root[root]
        ),
    )
    with pytest.raises(comparison_project.ComparisonProjectError, match="fresh validation"):
        comparison_project.build_comparison_project(
            plan, sources, output_root=tmp_path / "report-stale"
        )

    cross_dataset_sources = _sources(tmp_path / "cross", plan)
    first_cell = plan.cells[0]
    bad_root = cross_dataset_sources[first_cell.cell_id].root
    bad_validation = _write_cell_root(
        bad_root,
        plan,
        cell_index=0,
        metric_numerators=(1, 1, 1, 1, 0, 0),
        case_manifest_hash="d" * 64,
    )
    cross_dataset_sources[first_cell.cell_id] = comparison_project.ValidatedCellRoot(
        root=bad_root,
        validation_result=bad_validation,
    )
    cross_validations = {
        source.root: source.validation_result for source in cross_dataset_sources.values()
    }
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: cross_validations[root],
    )
    with pytest.raises(comparison_project.ComparisonProjectError, match="case manifest"):
        comparison_project.build_comparison_project(
            plan,
            cross_dataset_sources,
            output_root=tmp_path / "report-cross-dataset",
        )


def test_project_seals_and_embeds_analysis_bound_to_the_exact_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an analysis sidecar omitted from the manifest or blindly embedded."""

    from oamb.reporting import comparison_project
    from oamb.reporting.report_analysis import generate_report_analysis

    plan = _plan(tmp_path)
    sources = _sources(tmp_path, plan)
    supplied_by_root = {source.root: source.validation_result for source in sources.values()}
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: supplied_by_root[root],
    )
    metric_ids = (
        "answer_accuracy",
        "context_tokens",
        "indexing_tokens",
        "retrieval_latency",
        "indexing_time",
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        content = json.dumps(
            {
                "overall": "The observed providers trade quality, tokens, and time.",
                "metrics": {
                    metric_id: f"Evidence-bounded comparison for {metric_id}."
                    for metric_id in metric_ids
                },
            }
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
        )

    def generate(export: dict[str, object]) -> bytes | None:
        generated = generate_report_analysis(
            export,
            model="deepseek-v4-flash",
            thinking_effort="high",
            base_url="https://models.example/v1",
            api_key="secret-test-key",
            cache_root=tmp_path / "analysis-cache",
            timeout_seconds=30,
            transport=httpx.MockTransport(handler),
            sleep=lambda _seconds: None,
        )
        assert generated is not None
        return generated.analysis_path.read_bytes()

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "report",
        analysis_generator=generate,
    )

    assert built.analysis_path == built.output_root / "report-analysis.json"
    analysis_bytes = built.analysis_path.read_bytes()
    analysis = json.loads(analysis_bytes)
    assert (
        analysis["report_export_sha256"]
        == hashlib.sha256(built.export_path.read_bytes()).hexdigest()
    )
    html = built.html_path.read_text(encoding="utf-8")
    assert analysis["overall"] in html
    for item in analysis["metrics"]:
        assert item["analysis"] in html
    manifest = json.loads(built.manifest_path.read_bytes())
    assert manifest["analysis_sha256"] == hashlib.sha256(analysis_bytes).hexdigest()


def test_report_analysis_and_base_html_preparation_overlap_before_final_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches serial analysis/render preparation or publication before their join."""

    from oamb.reporting import comparison_project

    base_renderer = getattr(comparison_project, "_render_base_html", None)
    assert base_renderer is not None, "base HTML preparation is not independently runnable"
    base_renderer = cast(Callable[[dict[str, object]], bytes], base_renderer)
    plan = _plan(tmp_path)
    sources = _sources(tmp_path, plan)
    supplied_by_root = {source.root: source.validation_result for source in sources.values()}
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: supplied_by_root[root],
    )
    analysis_started = threading.Event()
    base_started = threading.Event()

    def blocked_base_renderer(export: dict[str, object]) -> bytes:
        base_started.set()
        assert analysis_started.wait(2), "analysis generation did not overlap base rendering"
        return base_renderer(export)

    def blocked_analysis(_export: dict[str, object]) -> None:
        analysis_started.set()
        assert base_started.wait(2), "base rendering did not overlap analysis generation"
        return None

    monkeypatch.setattr(comparison_project, "_render_base_html", blocked_base_renderer)
    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "report",
        analysis_generator=blocked_analysis,
    )

    assert base_started.is_set()
    assert analysis_started.is_set()
    assert built.analysis_path is None
    assert "Analysis unavailable" in built.html_path.read_text(encoding="utf-8")


def test_report_without_dataset_source_has_closed_absent_detail_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources = _content_sources(tmp_path, plan)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "summary-report",
    )
    export = json.loads(built.export_path.read_bytes())
    serialized = built.export_path.read_text(encoding="utf-8")

    assert export["dataset_details"] == {
        "reason": "dataset_source_not_supplied",
        "status": "absent",
    }
    assert len(export["questions"]) == CASE_COUNT
    assert all(len(item["provider_results"]) == 3 for item in export["questions"])
    assert "answer <script>bad()</script>" not in serialized
    assert "context <img src=x>" not in serialized
    assert all("question" not in item for item in export["questions"])
    assert all("gold_answer" not in item for item in export["questions"])


def test_detailed_report_joins_public_question_details_and_escapes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources = _content_sources(tmp_path, plan)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )
    dataset_source = tmp_path / "dataset.json"
    dataset_source.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        comparison_project,
        "_load_local_question_content",
        lambda resolved_plan, source, case_manifest_bytes: _local_question_content(),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "detailed-report",
        dataset_source=dataset_source,
    )
    export = json.loads(built.export_path.read_bytes())
    rendered = built.html_path.read_text(encoding="utf-8")

    assert export["dataset_details"]["status"] == "verified"
    assert export["dataset_details"]["license_id"] == "NOASSERTION"
    assert export["dataset_details"]["payload_policy"] == "download-required-not-redistributed"
    assert export["questions"][0]["question_type"] == "single-session-user"
    assert export["questions"][0]["gold_answer"] == "Gold answer 1"
    assert export["questions"][0]["answer_sessions"][0]["messages"][0]["has_answer"] is True
    assert len(export["questions"]) == CASE_COUNT
    assert all(len(item["provider_results"]) == 3 for item in export["questions"])
    assert "Question results" in rendered
    assert "Dataset provenance" in rendered
    assert "LongMemEval: Benchmarking Chat Assistants" in rendered
    assert "Original answer session" in rendered
    assert "Injected context" in rendered
    assert "&lt;script&gt;bad()&lt;/script&gt;" in rendered
    assert "&lt;img src=x&gt;" in rendered
    assert "javascript: is text" in rendered
    assert "onerror=example" in rendered
    assert "\\u202e" in rendered
    assert "<script>bad()" not in rendered
    assert "<img src=x>" not in rendered


@pytest.mark.parametrize(
    "fragment",
    (
        '<a href="javascript:alert(1)">unsafe</a>',
        '<p onerror="alert(1)">unsafe</p>',
        '<img src="report.json">',
    ),
)
def test_offline_html_inspector_rejects_unsafe_elements_and_attributes(fragment: str) -> None:
    from oamb.reporting.comparison_project import _OfflineHtmlInspector

    inspector = _OfflineHtmlInspector()
    inspector.feed(fragment)
    inspector.close()

    assert inspector.unsafe is True


def test_answer_session_documents_follow_authoritative_row_labels() -> None:
    from oamb.reporting import comparison_project

    authoritative = SimpleNamespace(
        session_id="authoritative-session",
        raw_timestamp="2026/08/30 (Sun) 00:00",
        messages=(
            SimpleNamespace(role="user", content="authoritative evidence", has_answer=False),
        ),
    )
    authoritative_and_flagged = SimpleNamespace(
        session_id="authoritative-and-flagged",
        raw_timestamp="2026/08/30 (Sun) 01:00",
        messages=(SimpleNamespace(role="user", content="also authoritative", has_answer=True),),
    )
    message_labeled_only = SimpleNamespace(
        session_id="message-labeled-only",
        raw_timestamp="2026/08/31 (Mon) 00:00",
        messages=(
            SimpleNamespace(role="user", content="non-authoritative evidence", has_answer=True),
        ),
    )
    row = SimpleNamespace(
        answer_session_ids=("authoritative-and-flagged", "authoritative-session"),
        sessions=(authoritative, authoritative_and_flagged, message_labeled_only),
    )

    assert comparison_project._answer_session_documents(row) == (
        {
            "session_id": "authoritative-session",
            "timestamp": "2026/08/30 (Sun) 00:00",
            "messages": (
                {
                    "role": "user",
                    "content": "authoritative evidence",
                    "has_answer": False,
                },
            ),
        },
        {
            "session_id": "authoritative-and-flagged",
            "timestamp": "2026/08/30 (Sun) 01:00",
            "messages": (
                {
                    "role": "user",
                    "content": "also authoritative",
                    "has_answer": True,
                },
            ),
        },
    )


def test_answer_session_documents_reject_missing_authoritative_session() -> None:
    from oamb.reporting import comparison_project

    row = SimpleNamespace(answer_session_ids=("missing-session",), sessions=())

    with pytest.raises(comparison_project.ComparisonProjectError, match="answer session"):
        comparison_project._answer_session_documents(row)


def test_export_validation_rejects_unsafe_html_before_publication_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from oamb.reporting import comparison_project

    plan = _plan(tmp_path)
    sources = _sources(tmp_path, plan)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )
    monkeypatch.setattr(
        comparison_project,
        "_embed_report_analysis",
        lambda _base, _analysis: b"<html><script>alert(1)</script></html>",
    )
    output_root = tmp_path / "unsafe-report"

    with pytest.raises(comparison_project.ComparisonProjectError, match="safe HTML"):
        comparison_project.build_comparison_project(
            plan,
            sources,
            output_root=output_root,
        )

    assert not (output_root / "report-manifest.json").exists()


@pytest.mark.parametrize(
    ("fraction", "expected"),
    (
        ({"numerator": 1, "denominator": 6}, "+16.7 pp"),
        ({"numerator": -1, "denominator": 6}, "-16.7 pp"),
        ({"numerator": 0, "denominator": 1}, "0.0 pp"),
    ),
)
def test_pairwise_delta_uses_signed_percentage_points(
    fraction: dict[str, int], expected: str
) -> None:
    from oamb.reporting.comparison_project import _delta_text

    assert _delta_text(fraction) == expected


def test_visible_time_is_decimal_seconds_only() -> None:
    from oamb.reporting.comparison_project import _time_text

    assert _time_text(1_234_567) == "1.235 s"


def test_accuracy_text_keeps_exact_denominator_and_human_percentage() -> None:
    from oamb.reporting.comparison_project import _accuracy_text

    assert (
        _accuracy_text(
            {
                "case_count": 6,
                "judged_case_count": 5,
                "judged_numerator": 4,
                "judged_denominator": 5,
            }
        )
        == "4/5 (80.0%); judged 5/6 cases"
    )


@pytest.mark.parametrize(
    ("numerator", "denominator", "lower", "upper"),
    (
        (0, 10, "0.000000", "0.277533"),
        (10, 10, "0.722467", "1.000000"),
        (30, 60, "0.377350", "0.622650"),
    ),
)
def test_wilson_accuracy_uses_the_frozen_95_percent_interval(
    numerator: int,
    denominator: int,
    lower: str,
    upper: str,
) -> None:
    from oamb.reporting.comparison_project import _wilson_accuracy_document

    assert _wilson_accuracy_document(numerator, denominator) == {
        "numerator": numerator,
        "denominator": denominator,
        "wilson_95": {
            "lower": lower,
            "upper": upper,
        },
    }


@pytest.mark.parametrize(
    ("left_only", "right_only", "exact_numerator", "exact_denominator", "display"),
    (
        (0, 0, 1, 1, "1.000000"),
        (6, 0, 1, 32, "0.031250"),
        (4, 1, 3, 8, "0.375000"),
        (1, 1, 1, 1, "1.000000"),
    ),
)
def test_exact_mcnemar_preserves_the_reduced_fraction_and_half_up_display(
    left_only: int,
    right_only: int,
    exact_numerator: int,
    exact_denominator: int,
    display: str,
) -> None:
    from oamb.reporting.comparison_project import _exact_mcnemar_document

    assert _exact_mcnemar_document(left_only, right_only) == {
        "numerator": exact_numerator,
        "denominator": exact_denominator,
        "display": display,
    }


def _lme60_results(outcomes: tuple[int, ...]) -> tuple[dict[str, object], ...]:
    assert len(outcomes) == 60
    return tuple(
        {
            "case_manifest_entry_id": f"{index:064x}",
            "state": "completed",
            "evaluation_disposition": "judged",
            "metric_id": "longmemeval-judge",
            "metric_numerator": outcome,
            "metric_denominator": 1,
        }
        for index, outcome in enumerate(outcomes, start=1)
    )


def _lme60_cell_document(
    plan: ResolvedPlan,
    cell_index: int,
    outcomes: tuple[int, ...],
) -> dict[str, object]:
    cell = plan.cells[cell_index]
    return {
        "cell_id": cell.cell_id,
        "provider_id": cell.provider_id,
        "case_count": 60,
        "metric_id": "longmemeval-judge",
        "completed_case_count": 60,
        "judged_case_count": 60,
        "judged_numerator": sum(outcomes),
        "judged_denominator": 60,
        "code_revisions": ("fixture-revision",),
        "results": _lme60_results(outcomes),
    }


def test_lme60_accuracy_reports_all_60_and_six_frozen_question_types() -> None:
    from oamb.reporting.comparison_project import _accuracy_document
    from oamb.workloads.longmemeval import LME60_EXPECTED_QUESTION_IDS, QUESTION_TYPES

    plan = _lme60_plan()
    outcomes = (
        (0,) * 10 + (1,) * 10 + (0, 1) * 5 + (1, 0) * 5 + (1,) * 6 + (0,) * 4 + (0,) * 6 + (1,) * 4
    )
    manifest_cases = [
        {
            "case_manifest_entry_id": f"{index:064x}",
            "raw_question_id": question_id,
        }
        for index, question_id in enumerate(LME60_EXPECTED_QUESTION_IDS, start=1)
    ]
    snapshot = SimpleNamespace(
        cases=tuple(reversed(_lme60_results(outcomes))),
        case_manifest={"cases": manifest_cases},
    )

    accuracy = _accuracy_document(plan, cast(Any, snapshot))
    by_question_type = cast(tuple[dict[str, object], ...], accuracy["by_question_type"])

    assert accuracy["all_60"] == {
        "numerator": 30,
        "denominator": 60,
        "wilson_95": {"lower": "0.377350", "upper": "0.622650"},
    }
    assert tuple(item["question_type"] for item in by_question_type) == QUESTION_TYPES
    assert [
        (item["numerator"], item["denominator"], item["wilson_95"]) for item in by_question_type
    ] == [
        (0, 10, {"lower": "0.000000", "upper": "0.277533"}),
        (10, 10, {"lower": "0.722467", "upper": "1.000000"}),
        (5, 10, {"lower": "0.236593", "upper": "0.763407"}),
        (5, 10, {"lower": "0.236593", "upper": "0.763407"}),
        (6, 10, {"lower": "0.312674", "upper": "0.831820"}),
        (4, 10, {"lower": "0.168180", "upper": "0.687326"}),
    ]


def test_pairwise_lme60_uses_matched_cases_when_completion_order_reverses() -> None:
    from oamb.reporting.comparison_project import _pair_document

    plan = _lme60_plan()
    left = _lme60_cell_document(plan, 0, (1,) * 36 + (0,) * 24)
    right = _lme60_cell_document(plan, 1, (1,) * 30 + (0,) * 30)
    right["results"] = tuple(reversed(cast(tuple[dict[str, object], ...], right["results"])))

    comparison = _pair_document(plan, cast(Any, left), cast(Any, right))

    assert comparison["paired_accuracy"] == {
        "left_correct_right_wrong": 6,
        "left_wrong_right_correct": 0,
        "exact_mcnemar_two_sided": {
            "numerator": 1,
            "denominator": 32,
            "display": "0.031250",
        },
    }
    assert comparison["accuracy_decision"] == {
        "status": "observed_accuracy_leader",
        "leader_cell_id": plan.cells[0].cell_id,
        "leader_provider_id": plan.cells[0].provider_id,
        "minimum_accuracy_delta": "0.05",
        "maximum_exact_mcnemar_p_value": "0.05",
        "failed_predicates": (),
    }


@pytest.mark.parametrize(
    ("left_correct", "right_correct", "mutate", "missing_field", "failed_predicate"),
    (
        (36, 30, "incomplete", None, "complete_equal_coverage"),
        (32, 30, None, None, "minimum_accuracy_delta"),
        (33, 30, None, None, "maximum_exact_mcnemar_p_value"),
        (36, 30, None, "minimum_accuracy_delta", "decision_policy_complete"),
        (36, 30, None, "maximum_exact_mcnemar_p_value", "decision_policy_complete"),
    ),
)
def test_lme60_accuracy_leader_fails_closed_for_each_required_predicate(
    left_correct: int,
    right_correct: int,
    mutate: str | None,
    missing_field: str | None,
    failed_predicate: str,
) -> None:
    from oamb.reporting.comparison_project import _pair_document

    plan = _lme60_plan()
    if missing_field == "minimum_accuracy_delta":
        plan = replace(
            plan,
            decision=cast(Any, SimpleNamespace(maximum_exact_mcnemar_p_value="0.05")),
        )
    elif missing_field == "maximum_exact_mcnemar_p_value":
        plan = replace(
            plan,
            decision=cast(Any, SimpleNamespace(minimum_accuracy_delta="0.05")),
        )
    left = _lme60_cell_document(plan, 0, (1,) * left_correct + (0,) * (60 - left_correct))
    right = _lme60_cell_document(plan, 1, (1,) * right_correct + (0,) * (60 - right_correct))
    if mutate == "incomplete":
        results = list(cast(tuple[dict[str, object], ...], left["results"]))
        results[-1] = {**results[-1], "state": "failed", "evaluation_disposition": "unjudged"}
        left["results"] = tuple(results)
        left["completed_case_count"] = 59
        left["judged_case_count"] = 59
        left["judged_denominator"] = 59

    comparison = _pair_document(plan, cast(Any, left), cast(Any, right))

    assert comparison["accuracy_decision"]["status"] == "no_clear_accuracy_leader"
    assert failed_predicate in comparison["accuracy_decision"]["failed_predicates"]


def test_report_names_one_accuracy_leader_only_after_it_clears_every_provider_pair() -> None:
    from itertools import combinations

    from oamb.reporting.comparison_project import _pair_document, _report_accuracy_decision

    plan = _lme60_plan()
    cells = (
        _lme60_cell_document(plan, 0, (1,) * 36 + (0,) * 24),
        _lme60_cell_document(plan, 1, (1,) * 30 + (0,) * 30),
        _lme60_cell_document(plan, 2, (1,) * 24 + (0,) * 36),
    )
    comparisons = tuple(
        _pair_document(plan, cast(Any, left), cast(Any, right))
        for left, right in combinations(cells, 2)
    )

    assert _report_accuracy_decision(plan, cast(Any, cells), comparisons) == {
        "status": "observed_accuracy_leader",
        "leader_cell_id": plan.cells[0].cell_id,
        "leader_provider_id": plan.cells[0].provider_id,
        "minimum_accuracy_delta": "0.05",
        "maximum_exact_mcnemar_p_value": "0.05",
        "failed_predicates": (),
    }

    blocked_cells = (
        cells[0],
        _lme60_cell_document(plan, 1, (1,) * 33 + (0,) * 27),
        cells[2],
    )
    blocked_pairs = tuple(
        _pair_document(plan, cast(Any, left), cast(Any, right))
        for left, right in combinations(blocked_cells, 2)
    )
    assert _report_accuracy_decision(plan, cast(Any, blocked_cells), blocked_pairs)["status"] == (
        "no_clear_accuracy_leader"
    )


def test_lme60_project_exports_accuracy_evidence_and_renders_the_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.reporting import comparison_project

    plan = _lme60_plan()
    outcome_sets = (
        (1,) * 36 + (0,) * 24,
        (1,) * 30 + (0,) * 30,
        (1,) * 24 + (0,) * 36,
    )
    sources = _lme60_sources(tmp_path, plan, outcome_sets)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "lme60-report",
    )
    export = json.loads(built.export_path.read_bytes())
    rendered = built.html_path.read_text(encoding="utf-8")

    assert export["cells"][0]["accuracy"]["all_60"]["wilson_95"] == {
        "lower": "0.473661",
        "upper": "0.714305",
    }
    assert len(export["cells"][0]["accuracy"]["by_question_type"]) == 6
    assert export["comparisons"][0]["paired_accuracy"]["exact_mcnemar_two_sided"] == {
        "numerator": 1,
        "denominator": 32,
        "display": "0.031250",
    }
    assert export["accuracy_decision"]["status"] == "observed_accuracy_leader"
    assert export["accuracy_decision"]["leader_provider_id"] == plan.cells[0].provider_id
    assert "95% Wilson" in rendered
    assert "Exact McNemar p" in rendered
    assert "Observed accuracy leader" in rendered
    assert "Accuracy by question type" in rendered


def test_lme60_normal_comparison_rejects_one_question_capsules(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.reporting import comparison_project

    plan = _lme60_plan()
    sources = _lme60_one_question_sources(tmp_path, plan)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    with pytest.raises(
        comparison_project.ComparisonProjectError,
        match="case records do not close|case/result coverage is incomplete",
    ):
        comparison_project.build_comparison_project(
            plan,
            sources,
            output_root=tmp_path / "normal-report",
        )


def test_lme60_diagnostic_comparison_reports_same_one_question_across_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.reporting import comparison_project

    plan = _lme60_plan()
    sources = _lme60_one_question_sources(tmp_path, plan)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "diagnostic-report",
        diagnostic=True,
    )
    export = json.loads(built.export_path.read_bytes())
    rendered = built.html_path.read_text(encoding="utf-8")

    assert export["diagnostic"] is True
    assert export["coverage"] == {
        "cell_count": 3,
        "unique_case_count": 1,
        "provider_specific_result_count": 3,
    }
    assert all("accuracy" not in cell for cell in export["cells"])
    assert export["accuracy_decision"]["status"] == "no_clear_accuracy_leader"
    assert export["accuracy_decision"]["failed_predicates"] == ["complete_equal_coverage"]
    assert "Diagnostic comparison: 1 of 60 frozen questions" in rendered


def test_lme60_diagnostic_pair_never_claims_a_partial_subset_leader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.reporting import comparison_project
    from oamb.workloads.longmemeval import LME60_EXPECTED_QUESTION_IDS

    plan = _lme60_plan()
    sources: dict[str, ValidatedCellRoot] = {}
    for index, cell in enumerate(plan.cells):
        root = tmp_path / "lme60-ten-question-capsules" / cell.cell_id
        validation = _write_cell_root(
            root,
            plan,
            cell_index=index,
            metric_numerators=((1,) * 10, (0,) * 10, (0,) * 10)[index],
            raw_question_ids=LME60_EXPECTED_QUESTION_IDS,
        )
        sources[cell.cell_id] = ValidatedCellRoot(root=root, validation_result=validation)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )

    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "diagnostic-ten-question-report",
        diagnostic=True,
    )
    export = json.loads(built.export_path.read_bytes())
    rendered = built.html_path.read_text(encoding="utf-8")

    assert all(
        pair["accuracy_decision"]["status"] == "no_clear_accuracy_leader"
        for pair in export["comparisons"]
    )
    assert all(
        "complete_equal_coverage" in pair["accuracy_decision"]["failed_predicates"]
        for pair in export["comparisons"]
    )
    assert "Observed leader:" not in rendered


@pytest.mark.parametrize("mutation", ("wilson", "mcnemar", "leader"))
def test_lme60_export_validation_rejects_derived_accuracy_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    from oamb.reporting import comparison_project

    plan = _lme60_plan()
    outcome_sets = (
        (1,) * 36 + (0,) * 24,
        (1,) * 30 + (0,) * 30,
        (1,) * 24 + (0,) * 36,
    )
    sources = _lme60_sources(tmp_path, plan, outcome_sets)
    monkeypatch.setattr(
        comparison_project,
        "validate_source_root",
        lambda root: next(
            source.validation_result for source in sources.values() if source.root == root
        ),
    )
    built = comparison_project.build_comparison_project(
        plan,
        sources,
        output_root=tmp_path / "valid-lme60-report",
    )
    mutated = json.loads(built.export_path.read_bytes())
    if mutation == "wilson":
        mutated["cells"][0]["accuracy"]["all_60"]["wilson_95"]["lower"] = "0.000000"
    elif mutation == "mcnemar":
        mutated["comparisons"][0]["paired_accuracy"]["exact_mcnemar_two_sided"]["numerator"] = 2
    else:
        mutated["accuracy_decision"]["leader_provider_id"] = plan.cells[1].provider_id

    with pytest.raises(comparison_project.ComparisonProjectError, match="accuracy evidence"):
        comparison_project._validate_accuracy_export(mutated)


def test_lme6_report_remains_descriptive_without_an_accuracy_leader_policy() -> None:
    from oamb.reporting.comparison_project import _pair_document

    plan = _plan(Path("."))
    left = {
        **_lme60_cell_document(_lme60_plan(), 0, (1,) * 36 + (0,) * 24),
        "cell_id": plan.cells[0].cell_id,
        "provider_id": plan.cells[0].provider_id,
        "case_count": 6,
        "completed_case_count": 6,
        "judged_case_count": 6,
        "judged_numerator": 4,
        "judged_denominator": 6,
        "results": _lme60_results((1, 1, 1, 1, 0, 0) + (0,) * 54)[:6],
    }
    right = {
        **left,
        "cell_id": plan.cells[1].cell_id,
        "provider_id": plan.cells[1].provider_id,
        "judged_numerator": 3,
        "results": _lme60_results((1, 1, 1, 0, 0, 0) + (0,) * 54)[:6],
    }

    comparison = _pair_document(plan, cast(Any, left), cast(Any, right))

    assert comparison["paired_accuracy"]["left_correct_right_wrong"] == 1
    assert comparison["accuracy_decision"]["status"] == "no_clear_accuracy_leader"
    assert comparison["accuracy_decision"]["failed_predicates"] == ("decision_policy_complete",)


def test_provider_headline_exposes_partial_context_and_latency_coverage() -> None:
    from oamb.reporting.comparison_project import _provider_summary_row

    row = _provider_summary_row(
        {
            "provider_id": "provider",
            "adapter_profile_id": "profile",
            "case_count": 3,
            "judged_case_count": 2,
            "judged_numerator": 1,
            "judged_denominator": 2,
            "accounting": {
                "answer_visible_context_tokens": {
                    "status": "measured_partial",
                    "case_count": 3,
                    "measured_case_count": 2,
                    "total": 100,
                    "mean": "50",
                },
                "tokens": {
                    "indexing": {
                        "supplier_usage_coverage": {
                            "status": "measured_partial",
                            "measured_record_count": 1,
                            "record_count": 2,
                        },
                        "totals": {"supplier_reported_total_tokens": {"value": 123}},
                    }
                },
            },
            "observed_time": {
                "indexing_ready": {
                    "status": "measured_partial",
                    "count": 1,
                    "median_microseconds": "1000000",
                    "p95_microseconds": 1_000_000,
                    "maximum_microseconds": 1_000_000,
                },
                "provider_request": {
                    "status": "measured",
                    "count": 3,
                    "median_microseconds": "100000",
                    "p95_microseconds": 200_000,
                    "maximum_microseconds": 200_000,
                },
            },
        }
    )

    assert "1/2 (50.0%); judged 2/3 cases" in row
    assert "100 total / 50 mean; 2/3 cases (measured_partial)" in row
    assert "n=1 (measured_partial)" in row
    assert "n=3 (measured)" in row


def test_indexing_ready_latency_spans_ingest_through_readiness_per_plan() -> None:
    from oamb.reporting.comparison_project import _indexing_ready_summary

    started = datetime(2026, 8, 31, tzinfo=UTC)

    def attempt(
        attempt_id: str,
        parent_id: str,
        stage: str,
        start_seconds: int,
        end_seconds: int,
        outcome: str = "succeeded",
        retry_of_attempt_id: str | None = None,
    ) -> dict[str, object]:
        return {
            "attempt_id": attempt_id,
            "parent_kind": "ingestion_plan",
            "parent_id": parent_id,
            "stage": stage,
            "started_at": (started + timedelta(seconds=start_seconds)).isoformat(),
            "ended_at": (started + timedelta(seconds=end_seconds)).isoformat(),
            "outcome": outcome,
            "retry_of_attempt_id": retry_of_attempt_id,
        }

    summary = _indexing_ready_summary(
        (
            attempt("a", "plan-a", "memory_ingest", 0, 2),
            attempt("b", "plan-a", "memory_ingest", 2, 4),
            attempt("c", "plan-a", "memory_readiness", 4, 10),
            attempt("d", "plan-a", "memory_projection", 10, 30),
            attempt("e", "plan-b", "memory_ingest", 100, 102),
            attempt("f", "plan-b", "memory_readiness", 102, 106),
        ),
        (
            {"ingestion_occurrence_id": "plan-a", "state": "sealed"},
            {"ingestion_occurrence_id": "plan-b", "state": "sealed"},
        ),
    )

    assert summary == {
        "count": 2,
        "maximum_microseconds": 10_000_000,
        "median_microseconds": "8000000",
        "p95_microseconds": 10_000_000,
        "status": "measured",
    }


def test_indexing_ready_latency_excludes_failed_plans_and_includes_successful_retry() -> None:
    from oamb.reporting.comparison_project import _indexing_ready_summary

    started = datetime(2026, 8, 31, tzinfo=UTC)

    def attempt(
        attempt_id: str,
        parent_id: str,
        stage: str,
        start_seconds: int,
        end_seconds: int,
        outcome: str,
        retry_of_attempt_id: str | None = None,
    ) -> dict[str, object]:
        return {
            "attempt_id": attempt_id,
            "parent_kind": "ingestion_plan",
            "parent_id": parent_id,
            "stage": stage,
            "started_at": (started + timedelta(seconds=start_seconds)).isoformat(),
            "ended_at": (started + timedelta(seconds=end_seconds)).isoformat(),
            "outcome": outcome,
            "retry_of_attempt_id": retry_of_attempt_id,
        }

    summary = _indexing_ready_summary(
        (
            attempt("a", "ready-plan", "memory_ingest", 0, 2, "succeeded"),
            attempt("b", "ready-plan", "memory_readiness", 2, 7, "failed"),
            attempt("c", "ready-plan", "memory_readiness", 7, 11, "succeeded", "b"),
            attempt("d", "failed-plan", "memory_ingest", 20, 22, "succeeded"),
            attempt("e", "failed-plan", "memory_readiness", 22, 30, "failed"),
        ),
        (
            {"ingestion_occurrence_id": "ready-plan", "state": "sealed"},
            {"ingestion_occurrence_id": "failed-plan", "state": "error"},
        ),
    )

    assert summary == {
        "count": 1,
        "maximum_microseconds": 11_000_000,
        "median_microseconds": "11000000",
        "p95_microseconds": 11_000_000,
        "status": "measured_partial",
    }


@pytest.mark.parametrize(
    "readiness_outcome",
    ("failed", "cancelled", "budget_exceeded", "unknown_outcome", None),
)
def test_indexing_ready_latency_rejects_terminal_plan_without_successful_final_readiness(
    readiness_outcome: str | None,
) -> None:
    from oamb.reporting.comparison_project import _indexing_ready_summary

    started = datetime(2026, 8, 31, tzinfo=UTC)
    summary = _indexing_ready_summary(
        (
            {
                "attempt_id": "ingest",
                "parent_kind": "ingestion_plan",
                "parent_id": "plan",
                "stage": "memory_ingest",
                "started_at": started.isoformat(),
                "ended_at": (started + timedelta(seconds=2)).isoformat(),
                "outcome": "succeeded",
            },
            {
                "attempt_id": "ready",
                "parent_kind": "ingestion_plan",
                "parent_id": "plan",
                "stage": "memory_readiness",
                "started_at": (started + timedelta(seconds=2)).isoformat(),
                "ended_at": (started + timedelta(seconds=7)).isoformat(),
                "outcome": readiness_outcome,
            },
        ),
        ({"ingestion_occurrence_id": "plan", "state": "sealed"},),
    )

    assert summary["status"] == "unavailable"
    assert summary["count"] == 0


def test_omitted_measurements_note_checks_resource_dimensions_not_aggregate() -> None:
    from oamb.reporting.comparison_project import _omitted_measurements_note

    cells = tuple(
        {
            "provider_id": provider_id,
            "accounting": {
                "tokens": {"retrieval": {"supplier_usage_coverage": {"status": "unavailable"}}},
                "resources": {
                    "peak_memory_bytes": {"status": "unavailable"},
                    "storage_bytes": {"status": "unavailable"},
                },
                "measurement_coverage": {"resources": "measured_partial"},
            },
        }
        for provider_id in ("hindsight", "mem0", "openviking")
    )

    note = _omitted_measurements_note(cast(Any, cells))

    assert "peak memory" in note
    assert "storage" in note


def test_model_output_parser_rejects_multiple_choices_and_hash_mismatch() -> None:
    from oamb.reporting import comparison_project

    multiple = canonical_json_bytes(
        {
            "choices": [
                {"message": {"content": "first"}},
                {"message": {"content": "second"}},
            ]
        }
    )
    with pytest.raises(comparison_project.ComparisonProjectError, match="exactly one"):
        comparison_project._model_output_text(
            multiple,
            expected_sha256=hashlib.sha256(b"first").hexdigest(),
        )

    one = canonical_json_bytes({"choices": [{"message": {"content": "answer"}}]})
    with pytest.raises(comparison_project.ComparisonProjectError, match="answer hash"):
        comparison_project._model_output_text(one, expected_sha256="f" * 64)


@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    (
        ("source_unit_id", "", False),
        ("provider_evidence_identity", "", False),
        ("evidence_kind", "", False),
        ("text", "", True),
        ("text", None, False),
        ("text", 42, False),
    ),
)
def test_visible_context_parser_preserves_empty_text_but_rejects_invalid_fields(
    field: str, value: object, accepted: bool
) -> None:
    from oamb.reporting import comparison_project

    payload = canonical_json_bytes(
        {
            "evidence_kind": "memory",
            "mentioned_at": None,
            "occurred_end": None,
            "occurred_start": None,
            "provider_evidence_identity": "provider-item",
            "source_unit_id": None,
            "text": "context",
            field: value,
        }
    )
    reference = hashlib.sha256(payload).hexdigest()
    snapshot = cast(Any, SimpleNamespace(raw_payloads={reference: payload}))
    case = {
        "visible_evidence_byte_count": len(payload),
        "visible_evidence_raw_ref": reference,
        "visible_evidence_sha256": reference,
        "visible_evidence_token_count": count_o200k_tokens(payload),
        "visible_evidence_tokenizer_fingerprint": tokenizer_fingerprint(),
    }

    if accepted:
        assert comparison_project._visible_context_text(snapshot, case) == payload.decode("utf-8")
    else:
        with pytest.raises(comparison_project.ComparisonProjectError, match="strict UTF-8 JSONL"):
            comparison_project._visible_context_text(snapshot, case)


def test_visible_context_parser_accepts_an_empty_evidence_sequence() -> None:
    from oamb.reporting import comparison_project

    payload = b""
    reference = hashlib.sha256(payload).hexdigest()
    snapshot = cast(Any, SimpleNamespace(raw_payloads={reference: payload}))
    case = {
        "visible_evidence_byte_count": 0,
        "visible_evidence_raw_ref": reference,
        "visible_evidence_sha256": reference,
        "visible_evidence_token_count": 0,
        "visible_evidence_tokenizer_fingerprint": tokenizer_fingerprint(),
    }

    assert comparison_project._visible_context_text(snapshot, case) == ""


def test_visible_context_parser_treats_unicode_line_separator_as_json_text() -> None:
    from oamb.reporting import comparison_project

    payload = canonical_json_bytes(
        {
            "evidence_kind": "memory",
            "mentioned_at": None,
            "occurred_end": None,
            "occurred_start": None,
            "provider_evidence_identity": "provider-item",
            "source_unit_id": None,
            "text": "before\u2028after",
        }
    )
    reference = hashlib.sha256(payload).hexdigest()
    snapshot = cast(Any, SimpleNamespace(raw_payloads={reference: payload}))
    case = {
        "visible_evidence_byte_count": len(payload),
        "visible_evidence_raw_ref": reference,
        "visible_evidence_sha256": reference,
        "visible_evidence_token_count": count_o200k_tokens(payload),
        "visible_evidence_tokenizer_fingerprint": tokenizer_fingerprint(),
    }

    assert comparison_project._visible_context_text(snapshot, case) == payload.decode("utf-8")


@pytest.mark.parametrize(
    ("cell_index", "path", "body"),
    (
        (
            0,
            "/v1/default/banks/bank/memories/recall",
            {
                "query": "question",
                "types": ["world", "experience"],
                "budget": "high",
                "max_tokens": 32768,
                "query_timestamp": None,
                "trace": True,
                "include": {"entities": None, "chunks": {}},
            },
        ),
        (
            1,
            "/search",
            {
                "query": "question",
                "filters": {"run_id": SHA_A},
                "top_k": 100,
                "threshold": 0.1,
            },
        ),
        (
            2,
            "/api/v1/search/find",
            {
                "query": "question",
                "target_uri": "viking://user/u/peers/p/memories",
                "context_type": "memory",
                "limit": 100,
            },
        ),
    ),
)
def test_exact_request_proof_closes_generation_free_retrieval(
    tmp_path: Path,
    cell_index: int,
    path: str,
    body: dict[str, object],
) -> None:
    from oamb.reporting.comparison_project import _retrieval_runtime_proof

    plan = _plan(tmp_path)
    proof = json.dumps(
        {
            "schema_name": "oamb_rest_request_proof",
            "schema_version": 1,
            "method": "POST",
            "path": path,
            "params": {},
            "json_payload": body,
            "request_header_names": [],
            "write_intent": False,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    proof_hash = hashlib.sha256(proof).hexdigest()
    cases = tuple({"retrieval_request_raw_ref": proof_hash} for _ in range(CASE_COUNT))

    assert (
        _retrieval_runtime_proof(plan.cells[cell_index], cases, {proof_hash: proof})
        == "runtime_verified"
    )

    mutated = json.loads(proof)
    mutated["json_payload"]["session_id"] = "forbidden"
    mutated_proof = json.dumps(
        mutated,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert (
        _retrieval_runtime_proof(
            plan.cells[cell_index],
            cases,
            {proof_hash: mutated_proof},
        )
        == "invalid"
    )
