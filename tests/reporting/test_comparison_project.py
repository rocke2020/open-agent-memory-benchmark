from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from oamb.config.benchmark import load_benchmark_configuration
from oamb.config.doctor import ResolvedPlan, build_resolved_plan
from oamb.contracts.evidence import ValidationResult
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.comparison_project import ValidatedCellRoot

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
    for index, (case_id, occurrence_id, numerator) in enumerate(
        zip(case_ids, case_occurrence_ids, metric_numerators, strict=True)
    ):
        query_start = started_at + timedelta(seconds=index + 1)
        query_end = query_start + timedelta(milliseconds=100 + index + cell_index)
        answer_end = query_end + timedelta(milliseconds=50)
        query_attempt_id = f"{index + 400 + cell_index * 20:064x}"
        answer_attempt_id = f"{index + 500 + cell_index * 20:064x}"
        records.extend(
            (
                (
                    "case_record",
                    f"source/cases/{occurrence_id}.json",
                    {
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
                    },
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
    rendered = built.html_path.read_text(encoding="utf-8")
    assert "Indexing / retain supplier usage" in rendered
    assert "Retrieval supplier usage" in rendered
    assert "Answer supplier usage" in rendered
    assert "Judge supplier usage" in rendered
    assert "Peak memory" in rendered
    assert "Storage" in rendered
    assert "Failures / retries" in rendered
    assert "CNY 0.0125" in rendered


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
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html

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
