from __future__ import annotations

import gzip
import hashlib
import json
import re
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import ResolvedPlan, build_resolved_plan
from oamb.contracts.evidence import ValidationResult
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.comparison_project import ValidatedCellRoot
from oamb.workloads.visible_evidence import count_o200k_tokens, tokenizer_fingerprint

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
CASE_COUNT = 6


def _plan(tmp_path: Path) -> ResolvedPlan:
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
) -> ValidationResult:
    cell = plan.cells[cell_index]
    run_id = f"run-{cell.cell_id}"
    started_at = datetime(2026, 8, 30, 0, 0, tzinfo=UTC) + timedelta(minutes=cell_index)
    case_ids = tuple(f"{index:064x}" for index in range(1, CASE_COUNT + 1))
    case_occurrence_ids = tuple(f"{index + 100 + cell_index * 10:064x}" for index in range(1, 7))
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
                        "raw_question_id": f"question-{index + 1}",
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
        zip(case_ids, case_occurrence_ids, metric_numerators, strict=True)
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
        records.extend(_accounting_records(run_id, started_at))

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
) -> tuple[tuple[str, str, dict[str, object]], ...]:
    attempt_ids = {
        "indexing": "a" * 64,
        "retrieval": "b" * 64,
        "answer": "c" * 64,
        "judge": "d" * 64,
        "retry": "e" * 64,
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
                "raw_response_ref": "f" * 64,
            },
        )
        for attempt_id, stage, outcome, retry_of in (
            (attempt_ids["indexing"], "memory_ingest", "succeeded", None),
            (attempt_ids["retrieval"], "memory_query", "succeeded", None),
            (attempt_ids["judge"], "judge", "failed", None),
            (attempt_ids["retry"], "judge", "succeeded", attempt_ids["judge"]),
        )
    )

    def usage(
        label: str,
        stage: str,
        *,
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
                "attempt_id": attempt_ids[label],
                "stage": stage,
                "budget_owner_kind": "model_role",
                "budget_owner_id": f"{label}-role",
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

    usage_records = (
        usage(
            "indexing",
            "memory_ingest",
            values=(100, 20, 120, None, None),
            proof_status="measured_partial",
            billing_complete=False,
        ),
        usage(
            "retrieval",
            "memory_query",
            values=(None, None, None, None, None),
            proof_status="unavailable",
            billing_complete=False,
        ),
        usage(
            "answer",
            "answer",
            values=(50, 5, 57, 0, 2),
            proof_status="measured_complete",
            billing_complete=True,
        ),
        usage(
            "judge",
            "judge",
            values=(None, None, None, None, None),
            proof_status="unavailable",
            billing_complete=False,
        ),
    )
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
    return attempts + usage_records + resources + costs


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
        "attempt_count": 16,
        "budget_exceeded_count": 0,
        "cancelled_count": 0,
        "failed_count": 1,
        "measurement_coverage": "complete",
        "retry_count": 1,
        "succeeded_count": 15,
        "unknown_outcome_count": 0,
    }
    assert set(accounting["tokens"]) == {"indexing", "retrieval", "answer", "judge"}
    assert accounting["tokens"]["indexing"]["supplier_usage_coverage"] == {
        "billing_complete_record_count": 0,
        "measured_record_count": 1,
        "record_count": 1,
        "status": "measured_partial",
        "unavailable_record_count": 0,
    }
    assert accounting["tokens"]["indexing"]["totals"]["input_tokens"] == {
        "measured_record_count": 1,
        "status": "measured_complete",
        "unavailable_record_count": 0,
        "value": 100,
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
    assert "Ctx tokens" in rendered
    assert "Four decision metrics" in rendered
    assert "312 total / 52 mean" in rendered
    assert "Indexing supplier tokens" in rendered
    assert "Index-ready latency / coverage (s)" in rendered
    assert "Recall latency / coverage (s)" in rendered
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
    assert "this comparison's frozen resolved-plan bindings" in html
    assert re.search(r"\b\d+(?:\.\d+)? us\b", html) is None
    assert re.search(r"\b\d+(?:\.\d+)? ms\b", html) is None
    assert '<a href="report.json" download>Download report.json</a>' in html
    assert "<pre>{" not in html

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
        "_render_html",
        lambda export: b"<html><script>alert(1)</script></html>",
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


def test_visible_context_parser_rejects_an_empty_source_unit_identity() -> None:
    from oamb.reporting import comparison_project

    payload = canonical_json_bytes(
        {
            "evidence_kind": "memory",
            "mentioned_at": None,
            "occurred_end": None,
            "occurred_start": None,
            "provider_evidence_identity": "provider-item",
            "source_unit_id": "",
            "text": "context",
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

    with pytest.raises(comparison_project.ComparisonProjectError, match="strict UTF-8 JSONL"):
        comparison_project._visible_context_text(snapshot, case)


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
