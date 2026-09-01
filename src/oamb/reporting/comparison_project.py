"""Generic multi-cell comparison and deterministic offline report export."""

from __future__ import annotations

import gzip
import hashlib
import html
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

from oamb.artifacts.atomic import read_regular_file
from oamb.artifacts.capsule import publish_with_last_marker, verify_published_directory
from oamb.artifacts.composition import inspect_embedded_composition
from oamb.artifacts.validation.composition import declares_composition
from oamb.artifacts.validation.retrieval_request import (
    retrieval_request_proves_generation_free,
)
from oamb.artifacts.validation.source_root import validate_source_root
from oamb.config.doctor import CellSpec, ModelExecutionBinding, ResolvedPlan
from oamb.contracts.evidence import CapsuleManifest, ValidationResult
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.states import ValidationDisposition

REPORT_EXPORT_NAME = "report.json"
REPORT_HTML_NAME = "report.html"
REPORT_MANIFEST_NAME = "report-manifest.json"
PAIR_DIRECTORY = "comparisons"
CONTROLLED_COMPARISON_WARNING = (
    "This is a controlled quality and token-efficiency comparison, not a deployment or "
    "provider-default-configuration comparison."
)
TOKEN_DIMENSIONS = (
    "input_tokens",
    "visible_output_tokens",
    "supplier_reported_total_tokens",
    "cached_input_tokens",
    "reasoning_tokens",
)
TOKEN_REPORT_STAGES = (
    ("indexing", "memory_ingest"),
    ("retrieval", "memory_query"),
    ("answer", "answer"),
    ("judge", "judge"),
)
RESOURCE_REPORT_DIMENSIONS = (
    ("provider_request_wall_seconds", "provider_request_wall_seconds_v1", "seconds", "sum"),
    ("peak_memory_bytes", "peak_memory_bytes", "bytes", "maximum"),
    ("storage_bytes", "storage_bytes", "bytes", "terminal_snapshot"),
)


class ComparisonProjectError(ValueError):
    """Validated cell evidence cannot close one resolved comparison plan."""


@dataclass(frozen=True, slots=True)
class ValidatedCellRoot:
    root: Path
    validation_result: ValidationResult


@dataclass(frozen=True, slots=True)
class ComparisonProjectBuildResult:
    output_root: Path
    export_path: Path
    html_path: Path
    comparison_paths: tuple[Path, ...]
    manifest_path: Path
    report_id: str


@dataclass(frozen=True, slots=True)
class _CellSnapshot:
    cell: CellSpec
    root: Path
    source_root_hash: str
    validation_hash: str
    run_id: str
    case_manifest_bytes: bytes
    case_manifest: dict[str, Any]
    cases: tuple[dict[str, Any], ...]
    attempts: tuple[dict[str, Any], ...]
    token_usage: tuple[dict[str, Any], ...]
    resources: tuple[dict[str, Any], ...]
    costs: tuple[dict[str, Any], ...]
    run_record: dict[str, Any]
    retrieval_runtime_proof_state: str
    infrastructure_retries: tuple[dict[str, Any], ...] = ()
    composition_part_states: tuple[str, ...] = ()


def build_comparison_project(
    plan: ResolvedPlan,
    sources: Mapping[str, ValidatedCellRoot],
    *,
    output_root: Path,
) -> ComparisonProjectBuildResult:
    """Revalidate all cells, derive every canonical pair, and publish one offline report."""

    if len(plan.cells) < 2:
        raise ComparisonProjectError("comparison requires at least two cells")
    expected_cell_ids = tuple(cell.cell_id for cell in plan.cells)
    if set(sources) != set(expected_cell_ids) or len(sources) != len(expected_cell_ids):
        raise ComparisonProjectError(
            "validated source roots do not close the resolved cell inventory"
        )

    snapshots = tuple(_load_cell_snapshot(plan, cell, sources[cell.cell_id]) for cell in plan.cells)
    _require_shared_case_manifest(plan, snapshots)
    cell_documents = tuple(_cell_document(plan, snapshot) for snapshot in snapshots)
    comparison_documents = tuple(
        _pair_document(plan, left, right) for left, right in combinations(cell_documents, 2)
    )
    if len(comparison_documents) != math.comb(len(plan.cells), 2):
        raise ComparisonProjectError("pairwise comparison inventory is incomplete")

    limitations = tuple(
        [
            f"runtime retrieval proof unavailable for {snapshot.cell.cell_id}; configured "
            "generation-free retrieval is not reported as a runtime-verified claim"
            for snapshot in snapshots
            if snapshot.retrieval_runtime_proof_state != "runtime_verified"
        ]
        + [
            "model and thinking-effort bindings are resolved-plan values; runtime proof remains "
            "bounded by each role's declared proof source",
            "observed time is local run evidence and is not an environment-independent latency score",
        ]
    )
    export_body: dict[str, Any] = {
        "schema_name": "comparison_project_report",
        "schema_version": 1,
        "comparison_id": plan.comparison_id,
        "resolved_plan_hash": plan.resolved_plan_hash,
        "dataset": {
            "dataset_id": plan.dataset.dataset_id,
            "workload_id": plan.dataset.workload_id,
            "selection": plan.dataset.selection,
            "revision": plan.dataset.revision,
            "source_sha256": plan.dataset.source_sha256,
            "case_manifest_hash": plan.dataset.case_manifest_hash,
        },
        "coverage": {
            "cell_count": len(cell_documents),
            "unique_case_count": len(snapshots[0].cases),
            "provider_specific_result_count": sum(len(item.cases) for item in snapshots),
        },
        "cells": cell_documents,
        "comparisons": comparison_documents,
        "models": tuple(_model_document(item) for item in plan.model_roles),
        "retrieval_generation": plan.retrieval.generation,
        "retrieval": tuple(
            _retrieval_document(plan, snapshot.cell, snapshot.retrieval_runtime_proof_state)
            for snapshot in snapshots
        ),
        "limitations": limitations,
        "controlled_comparison_warning": CONTROLLED_COMPARISON_WARNING,
    }
    report_id = canonical_sha256(["oamb-comparison-project-initial-v1", export_body])
    export = {**export_body, "report_id": report_id}
    export_bytes = canonical_json_bytes(export)
    html_bytes = _render_html(export)

    payloads: dict[str, bytes] = {
        REPORT_EXPORT_NAME: export_bytes,
        REPORT_HTML_NAME: html_bytes,
    }
    pair_relative_paths: list[str] = []
    pair_entries: list[dict[str, object]] = []
    for ordinal, comparison in enumerate(comparison_documents, start=1):
        relative_path = (
            f"{PAIR_DIRECTORY}/{ordinal:02d}-{comparison['left_cell_id']}--"
            f"{comparison['right_cell_id']}.json"
        )
        content = canonical_json_bytes(comparison)
        payloads[relative_path] = content
        pair_relative_paths.append(relative_path)
        pair_entries.append(
            {
                "relative_path": relative_path,
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    marker = canonical_json_bytes(
        {
            "schema_name": "comparison_project_manifest",
            "schema_version": 1,
            "report_id": report_id,
            "resolved_plan_hash": plan.resolved_plan_hash,
            "export_sha256": hashlib.sha256(export_bytes).hexdigest(),
            "html_sha256": hashlib.sha256(html_bytes).hexdigest(),
            "comparisons": pair_entries,
        }
    )
    root = Path(output_root)
    publication = publish_with_last_marker(
        root,
        payloads=payloads,
        marker_name=REPORT_MANIFEST_NAME,
        marker_bytes=marker,
    )
    verify_published_directory(root, REPORT_MANIFEST_NAME)
    return ComparisonProjectBuildResult(
        output_root=root,
        export_path=root / REPORT_EXPORT_NAME,
        html_path=root / REPORT_HTML_NAME,
        comparison_paths=tuple(root / item for item in pair_relative_paths),
        manifest_path=publication.marker_path,
        report_id=report_id,
    )


def _load_cell_snapshot(
    plan: ResolvedPlan,
    cell: CellSpec,
    source: ValidatedCellRoot,
) -> _CellSnapshot:
    root = Path(source.root)
    supplied_validation = source.validation_result
    if supplied_validation.disposition != ValidationDisposition.VALIDATED:
        raise ComparisonProjectError(f"cell {cell.cell_id} does not have VALIDATED evidence")
    fresh_validation = validate_source_root(root)
    if fresh_validation != supplied_validation:
        raise ComparisonProjectError(
            f"cell {cell.cell_id} requires fresh validation of the current source-root bytes"
        )

    try:
        manifest = CapsuleManifest.model_validate_json(
            read_regular_file(root / "capsule-manifest.json")
        )
    except Exception as exc:
        raise ComparisonProjectError(
            f"cell {cell.cell_id} capsule manifest cannot be reopened"
        ) from exc
    if supplied_validation.target_hash != manifest.source_manifest_hash:
        raise ComparisonProjectError(
            f"cell {cell.cell_id} validation target does not bind the root"
        )

    if declares_composition(root):
        return _load_composed_cell_snapshot(
            plan,
            cell,
            root,
            manifest,
            supplied_validation,
        )

    documents, raw_payloads = _read_source_documents(root, manifest, cell.cell_id)

    run_spec = _exact_document(documents, "run_spec", cell.cell_id)
    preflight = _exact_document(documents, "run_preflight_record", cell.cell_id)
    dataset_manifest = _exact_document(documents, "dataset_manifest", cell.cell_id)
    case_manifest_bytes, case_manifest = _exact_record(documents, "case_manifest", cell.cell_id)
    run_record = _exact_document(documents, "run_record", cell.cell_id)
    case_records = tuple(item[1] for item in documents.get("case_record", ()))
    attempts = tuple(item[1] for item in documents.get("attempt_record", ()))
    token_usage = tuple(item[1] for item in documents.get("token_usage_record", ()))
    resources = tuple(item[1] for item in documents.get("resource_usage_record", ()))
    costs = tuple(item[1] for item in documents.get("cost_record", ()))
    infrastructure_retries = tuple(
        item[1] for item in documents.get("infrastructure_retry_event", ())
    )

    if manifest.run_id != run_spec.get("run_id") or run_record.get("run_id") != manifest.run_id:
        raise ComparisonProjectError(f"cell {cell.cell_id} run identity does not close")
    if manifest.run_spec_hash != canonical_sha256(run_spec):
        raise ComparisonProjectError(f"cell {cell.cell_id} run spec hash does not close")
    if preflight.get("run_id") != manifest.run_id:
        raise ComparisonProjectError(f"cell {cell.cell_id} preflight run identity does not close")
    if preflight.get("resolved_plan_hash") != plan.resolved_plan_hash:
        raise ComparisonProjectError(f"cell {cell.cell_id} does not bind the resolved plan")
    if preflight.get("adapter_profile_id") != cell.adapter_profile_id:
        raise ComparisonProjectError(f"cell {cell.cell_id} adapter profile does not close")
    if (
        run_spec.get("memory_system_id") != cell.provider_id
        or run_spec.get("workload_id") != cell.workload_id
    ):
        raise ComparisonProjectError(f"cell {cell.cell_id} run spec does not close its cell")
    if run_spec.get("case_manifest_hash") != cell.case_manifest_hash:
        raise ComparisonProjectError(f"cell {cell.cell_id} case manifest does not close")
    if (
        dataset_manifest.get("dataset_id") != cell.dataset_id
        or dataset_manifest.get("revision") != plan.dataset.revision
        or run_spec.get("dataset_manifest_hash") != dataset_manifest.get("manifest_hash")
    ):
        raise ComparisonProjectError(f"cell {cell.cell_id} dataset manifest does not close")
    source_files = dataset_manifest.get("source_files")
    if not isinstance(source_files, list) or tuple(
        item.get("sha256") for item in source_files if isinstance(item, dict)
    ) != (cell.source_sha256,):
        raise ComparisonProjectError(f"cell {cell.cell_id} dataset source does not close")
    if (
        case_manifest.get("manifest_hash") != cell.case_manifest_hash
        or case_manifest.get("workload_id") != cell.workload_id
    ):
        raise ComparisonProjectError(f"cell {cell.cell_id} case manifest does not close")

    manifest_cases = case_manifest.get("cases")
    if not isinstance(manifest_cases, list):
        raise ComparisonProjectError(f"cell {cell.cell_id} case manifest inventory is invalid")
    expected_case_count = _expected_case_count(plan.dataset.selection)
    if expected_case_count is not None and len(manifest_cases) != expected_case_count:
        raise ComparisonProjectError(
            f"cell {cell.cell_id} case manifest requires {expected_case_count} cases"
        )
    manifest_case_ids = tuple(
        _required_text(item, "case_manifest_entry_id") for item in manifest_cases
    )
    cases_by_id = _unique_by(case_records, "case_manifest_entry_id", "case record")
    if tuple(cases_by_id) != manifest_case_ids:
        try:
            ordered_cases = tuple(cases_by_id[item] for item in manifest_case_ids)
        except KeyError as exc:
            raise ComparisonProjectError(
                f"cell {cell.cell_id} case records do not close the case manifest"
            ) from exc
        if len(cases_by_id) != len(manifest_case_ids):
            raise ComparisonProjectError(
                f"cell {cell.cell_id} case records do not close the case manifest"
            )
    else:
        ordered_cases = tuple(cases_by_id.values())
    if any(
        case_record.get("run_id") != manifest.run_id
        or case_record.get("adapter_profile_id") != cell.adapter_profile_id
        for case_record in ordered_cases
    ):
        raise ComparisonProjectError(f"cell {cell.cell_id} case record binding drifted")

    retrieval_proof_state = _retrieval_runtime_proof(cell, ordered_cases, raw_payloads)

    return _CellSnapshot(
        cell=cell,
        root=root,
        source_root_hash=manifest.source_manifest_hash,
        validation_hash=canonical_sha256(supplied_validation),
        run_id=manifest.run_id,
        case_manifest_bytes=case_manifest_bytes,
        case_manifest=case_manifest,
        cases=ordered_cases,
        attempts=attempts,
        token_usage=token_usage,
        resources=resources,
        costs=costs,
        run_record=run_record,
        retrieval_runtime_proof_state=retrieval_proof_state,
        infrastructure_retries=infrastructure_retries,
    )


def _load_composed_cell_snapshot(
    plan: ResolvedPlan,
    cell: CellSpec,
    root: Path,
    manifest: CapsuleManifest,
    validation: ValidationResult,
) -> _CellSnapshot:
    composition, embedded_roots = inspect_embedded_composition(root)
    if (
        composition.resolved_plan_hash != plan.resolved_plan_hash
        or composition.cell_spec_hash != cell.cell_spec_hash
        or composition.target_case_manifest_hash != cell.case_manifest_hash
        or composition.budget_policy_hash != cell.limits_hash
    ):
        raise ComparisonProjectError(f"cell {cell.cell_id} composition binding does not close")

    part_documents: dict[str, dict[str, list[tuple[bytes, dict[str, Any]]]]] = {}
    merged_raw_payloads: dict[str, bytes] = {}
    case_manifest_bytes: bytes | None = None
    case_manifest: dict[str, Any] | None = None
    starts: list[str] = []
    ends: list[str] = []
    for binding, embedded_root in zip(composition.ordered_parts, embedded_roots, strict=True):
        embedded_manifest = CapsuleManifest.model_validate_json(
            read_regular_file(embedded_root / "capsule-manifest.json")
        )
        if embedded_manifest.capsule_id != binding.capsule_id:
            raise ComparisonProjectError(
                f"cell {cell.cell_id} embedded part identity does not close"
            )
        documents, raw_payloads = _read_source_documents(
            embedded_root,
            embedded_manifest,
            cell.cell_id,
        )
        part_documents[binding.capsule_id] = documents
        for raw_id, payload in raw_payloads.items():
            previous = merged_raw_payloads.setdefault(raw_id, payload)
            if previous != payload:
                raise ComparisonProjectError(f"cell {cell.cell_id} embedded raw identity collides")
        current_manifest_bytes, current_manifest = _exact_record(
            documents, "case_manifest", cell.cell_id
        )
        if case_manifest_bytes is None:
            case_manifest_bytes = current_manifest_bytes
            case_manifest = current_manifest
        elif current_manifest_bytes != case_manifest_bytes:
            raise ComparisonProjectError(f"cell {cell.cell_id} embedded case manifests differ")
        run_spec = _exact_document(documents, "run_spec", cell.cell_id)
        preflight = _exact_document(documents, "run_preflight_record", cell.cell_id)
        dataset = _exact_document(documents, "dataset_manifest", cell.cell_id)
        part_run_record = _exact_document(documents, "run_record", cell.cell_id)
        if (
            run_spec.get("run_id") != binding.run_id
            or run_spec.get("memory_system_id") != cell.provider_id
            or run_spec.get("workload_id") != cell.workload_id
            or run_spec.get("case_manifest_hash") != cell.case_manifest_hash
            or preflight.get("run_id") != binding.run_id
            or preflight.get("resolved_plan_hash") != plan.resolved_plan_hash
            or preflight.get("adapter_profile_id") != cell.adapter_profile_id
            or dataset.get("dataset_id") != cell.dataset_id
            or dataset.get("revision") != plan.dataset.revision
        ):
            raise ComparisonProjectError(
                f"cell {cell.cell_id} embedded part does not close the resolved cell"
            )
        source_files = dataset.get("source_files")
        if not isinstance(source_files, list) or tuple(
            item.get("sha256") for item in source_files if isinstance(item, dict)
        ) != (cell.source_sha256,):
            raise ComparisonProjectError(
                f"cell {cell.cell_id} embedded dataset source does not close"
            )
        if isinstance(part_run_record.get("started_at"), str):
            starts.append(part_run_record["started_at"])
        if isinstance(part_run_record.get("ended_at"), str):
            ends.append(part_run_record["ended_at"])

    if case_manifest_bytes is None or case_manifest is None:
        raise ComparisonProjectError(f"cell {cell.cell_id} composition has no case manifest")
    manifest_cases = case_manifest.get("cases")
    if not isinstance(manifest_cases, list):
        raise ComparisonProjectError(f"cell {cell.cell_id} case manifest inventory is invalid")
    expected_case_ids = tuple(
        _required_text(item, "case_manifest_entry_id") for item in manifest_cases
    )
    expected_count = _expected_case_count(plan.dataset.selection)
    if expected_count is not None and len(expected_case_ids) != expected_count:
        raise ComparisonProjectError(
            f"cell {cell.cell_id} case manifest requires {expected_count} cases"
        )

    selected_case_records: list[dict[str, Any]] = []
    ingestion_occurrence_ids: list[str] = []
    case_occurrence_ids: list[str] = []
    for contribution in composition.ordered_contributions:
        documents = part_documents[contribution.source_capsule_id]
        cases_by_occurrence = _unique_by(
            tuple(item[1] for item in documents.get("case_record", ())),
            "case_occurrence_id",
            "case record",
        )
        contributed_cases = tuple(
            cases_by_occurrence[occurrence_id] for occurrence_id in contribution.case_occurrence_ids
        )
        if (
            tuple(_required_text(item, "case_manifest_entry_id") for item in contributed_cases)
            != contribution.case_manifest_entry_ids
        ):
            raise ComparisonProjectError(
                f"cell {cell.cell_id} composition case provenance does not close"
            )
        selected_case_records.extend(contributed_cases)
        ingestion_occurrence_ids.append(contribution.ingestion_occurrence_id)
        case_occurrence_ids.extend(contribution.case_occurrence_ids)

    all_attempts = tuple(
        item[1]
        for binding in composition.ordered_parts
        for item in part_documents[binding.capsule_id].get("attempt_record", ())
    )
    all_usage = tuple(
        item[1]
        for binding in composition.ordered_parts
        for item in part_documents[binding.capsule_id].get("token_usage_record", ())
    )
    all_resources = tuple(
        item[1]
        for binding in composition.ordered_parts
        for item in part_documents[binding.capsule_id].get("resource_usage_record", ())
    )
    all_costs = tuple(
        item[1]
        for binding in composition.ordered_parts
        for item in part_documents[binding.capsule_id].get("cost_record", ())
    )
    all_infrastructure_retries = tuple(
        item[1]
        for binding in composition.ordered_parts
        for item in part_documents[binding.capsule_id].get("infrastructure_retry_event", ())
    )
    for values, identity, label in (
        (all_attempts, "attempt_id", "composed attempt"),
        (all_usage, "usage_record_id", "composed token usage"),
        (all_resources, "resource_record_id", "composed resource usage"),
        (all_costs, "cost_record_id", "composed cost"),
        (all_infrastructure_retries, "retry_event_id", "composed retry event"),
    ):
        _unique_by(values, identity, label)

    cases_by_manifest_id = _unique_by(
        tuple(selected_case_records),
        "case_manifest_entry_id",
        "composed case record",
    )
    try:
        ordered_cases = tuple(cases_by_manifest_id[case_id] for case_id in expected_case_ids)
    except KeyError as exc:
        raise ComparisonProjectError(
            f"cell {cell.cell_id} composed cases do not close the target manifest"
        ) from exc
    if len(cases_by_manifest_id) != len(expected_case_ids) or any(
        case_record.get("adapter_profile_id") != cell.adapter_profile_id
        for case_record in ordered_cases
    ):
        raise ComparisonProjectError(f"cell {cell.cell_id} composed case record binding drifted")
    run_record: dict[str, Any] = {
        "schema_name": "run_record",
        "schema_version": 1,
        "run_id": composition.composition_id,
        "state": "finalized",
        "started_at": min(starts) if starts else None,
        "ended_at": max(ends) if ends else None,
        "ingestion_occurrence_ids": ingestion_occurrence_ids,
        "case_occurrence_ids": case_occurrence_ids,
    }
    retrieval_proof_state = _retrieval_runtime_proof(
        cell,
        ordered_cases,
        merged_raw_payloads,
    )
    return _CellSnapshot(
        cell=cell,
        root=root,
        source_root_hash=manifest.source_manifest_hash,
        validation_hash=canonical_sha256(validation),
        run_id=composition.composition_id,
        case_manifest_bytes=case_manifest_bytes,
        case_manifest=case_manifest,
        cases=ordered_cases,
        attempts=all_attempts,
        token_usage=all_usage,
        resources=all_resources,
        costs=all_costs,
        run_record=run_record,
        retrieval_runtime_proof_state=retrieval_proof_state,
        infrastructure_retries=all_infrastructure_retries,
        composition_part_states=tuple(
            binding.run_state.value for binding in composition.ordered_parts
        ),
    )


def _read_source_documents(
    root: Path,
    manifest: CapsuleManifest,
    cell_id: str,
) -> tuple[dict[str, list[tuple[bytes, dict[str, Any]]]], dict[str, bytes]]:
    documents: dict[str, list[tuple[bytes, dict[str, Any]]]] = {}
    raw_payloads: dict[str, bytes] = {}
    for entry in manifest.source_entries:
        try:
            content = read_regular_file(root / entry.relative_path)
        except Exception as exc:
            raise ComparisonProjectError(f"cell {cell_id} source entry cannot be reopened") from exc
        if hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ComparisonProjectError(f"cell {cell_id} source entry hash drifted")
        if entry.record_kind == "raw_payload":
            try:
                payload = (
                    gzip.decompress(content) if entry.relative_path.endswith(".gz") else content
                )
            except (OSError, EOFError) as exc:
                raise ComparisonProjectError(
                    f"cell {cell_id} raw request evidence cannot be reopened"
                ) from exc
            if hashlib.sha256(payload).hexdigest() != entry.record_id:
                raise ComparisonProjectError(f"cell {cell_id} raw payload identity drifted")
            raw_payloads[entry.record_id] = payload
            continue
        try:
            document = json.loads(content, object_pairs_hook=_unique_json_object)
        except (TypeError, ValueError) as exc:
            raise ComparisonProjectError(
                f"cell {cell_id} source record is not strict JSON"
            ) from exc
        if not isinstance(document, dict) or document.get("schema_name") != entry.record_kind:
            raise ComparisonProjectError(f"cell {cell_id} source record kind drifted")
        documents.setdefault(entry.record_kind, []).append((content, document))
    return documents, raw_payloads


def _require_shared_case_manifest(
    plan: ResolvedPlan,
    snapshots: tuple[_CellSnapshot, ...],
) -> None:
    if not snapshots:
        raise ComparisonProjectError("comparison has no cell snapshots")
    manifest_bytes = snapshots[0].case_manifest_bytes
    if any(item.case_manifest_bytes != manifest_bytes for item in snapshots[1:]):
        raise ComparisonProjectError("cells do not share the exact case manifest bytes")
    case_count = len(snapshots[0].cases)
    if any(len(item.cases) != case_count for item in snapshots):
        raise ComparisonProjectError("provider-specific result coverage is not closed")
    expected = _expected_case_count(plan.dataset.selection)
    if expected is not None and (
        case_count != expected
        or sum(len(item.cases) for item in snapshots) != expected * len(plan.cells)
    ):
        raise ComparisonProjectError("comparison case/result coverage is incomplete")


def _cell_document(plan: ResolvedPlan, snapshot: _CellSnapshot) -> dict[str, Any]:
    judged = tuple(
        item
        for item in snapshot.cases
        if item.get("evaluation_disposition") == "judged"
        and isinstance(item.get("metric_numerator"), int)
        and isinstance(item.get("metric_denominator"), int)
    )
    metric_ids = tuple(dict.fromkeys(_required_text(item, "metric_id") for item in judged))
    if len(metric_ids) > 1:
        raise ComparisonProjectError(f"cell {snapshot.cell.cell_id} mixes metric policies")
    numerator = sum(int(item["metric_numerator"]) for item in judged)
    denominator = sum(int(item["metric_denominator"]) for item in judged)
    limitations: list[str] = []
    if len(judged) != len(snapshot.cases):
        limitations.append(
            f"{len(snapshot.cases) - len(judged)} of {len(snapshot.cases)} cases are outside the "
            "judged accuracy denominator"
        )
    if snapshot.composition_part_states:
        limitations.append(
            f"composed from {len(snapshot.composition_part_states)} immutable part capsules"
        )
        limitations.append(
            "accounting includes all recovery-part work; cell elapsed spans inter-part gaps"
        )
        nonfinal = tuple(
            state for state in snapshot.composition_part_states if state != "finalized"
        )
        if nonfinal:
            limitations.append(
                "resume source part states "
                f"{','.join(nonfinal)}; only terminal-success whole-plan groups contributed"
            )
    return {
        "cell_id": snapshot.cell.cell_id,
        "cell_spec_hash": snapshot.cell.cell_spec_hash,
        "provider_id": snapshot.cell.provider_id,
        "adapter_profile_id": snapshot.cell.adapter_profile_id,
        "run_id": snapshot.run_id,
        "source_root_hash": snapshot.source_root_hash,
        "validation_result_hash": snapshot.validation_hash,
        "case_count": len(snapshot.cases),
        "metric_id": metric_ids[0] if metric_ids else "unavailable",
        "judged_case_count": len(judged),
        "judged_numerator": numerator,
        "judged_denominator": denominator,
        "results": tuple(
            {
                "case_manifest_entry_id": item["case_manifest_entry_id"],
                "case_occurrence_id": item.get("case_occurrence_id", "unavailable"),
                "state": item.get("state", "unavailable"),
                "evaluation_disposition": item.get("evaluation_disposition", "unavailable"),
                "metric_id": item.get("metric_id", "unavailable"),
                "metric_numerator": item.get("metric_numerator", "unavailable"),
                "metric_denominator": item.get("metric_denominator", "unavailable"),
            }
            for item in snapshot.cases
        ),
        "model_role_ids": {
            "producer": snapshot.cell.producer_role_id,
            "embedding": snapshot.cell.embedding_role_id,
            "answer": snapshot.cell.answer_role_id,
            "judge": snapshot.cell.judge_role_id,
        },
        "retrieval_binding_id": snapshot.cell.retrieval_binding_id,
        "observed_time": _observed_time(snapshot),
        "accounting": _accounting_document(snapshot),
        "limitations": limitations,
    }


def _accounting_document(snapshot: _CellSnapshot) -> dict[str, object]:
    token_stages = {
        label: _token_stage_document(snapshot.token_usage, stage)
        for label, stage in TOKEN_REPORT_STAGES
    }
    resource_document = _resource_document(snapshot.resources)
    cost_document = _cost_document(snapshot.costs)
    return {
        "attempts": _attempt_document(snapshot.attempts),
        "infrastructure_retries": {
            "rejection_count": len(snapshot.infrastructure_retries),
            "internal_retry_count": sum(
                int(item.get("internal_retry_count", 0)) for item in snapshot.infrastructure_retries
            ),
            "scheduled_retry_count": sum(
                item.get("retry_scheduled") is True for item in snapshot.infrastructure_retries
            ),
            "total_retry_count": sum(
                int(item.get("internal_retry_count", 0)) + int(item.get("retry_scheduled") is True)
                for item in snapshot.infrastructure_retries
            ),
            "backoff_seconds": sum(
                int(item["backoff_seconds"])
                for item in snapshot.infrastructure_retries
                if isinstance(item.get("backoff_seconds"), int)
            ),
            "measurement_coverage": "complete",
        },
        "tokens": token_stages,
        "resources": resource_document,
        "cost": cost_document,
        "measurement_coverage": {
            "attempts": "complete",
            "tokens": {
                label: value["supplier_usage_coverage"]["status"]
                for label, value in token_stages.items()
            },
            "resources": _resource_coverage(resource_document),
            "cost": cost_document["billing_coverage"]["status"],
        },
    }


def _attempt_document(attempts: tuple[dict[str, Any], ...]) -> dict[str, object]:
    outcomes = tuple(item.get("outcome") for item in attempts)
    return {
        "attempt_count": len(attempts),
        "succeeded_count": outcomes.count("succeeded"),
        "failed_count": outcomes.count("failed"),
        "cancelled_count": outcomes.count("cancelled"),
        "budget_exceeded_count": outcomes.count("budget_exceeded"),
        "unknown_outcome_count": outcomes.count("unknown_outcome"),
        "retry_count": sum(item.get("retry_of_attempt_id") is not None for item in attempts),
        "measurement_coverage": "complete",
    }


def _token_stage_document(
    records: tuple[dict[str, Any], ...],
    stage: str,
) -> dict[str, Any]:
    selected = tuple(item for item in records if item.get("stage") == stage)
    measured = tuple(
        item
        for item in selected
        if item.get("proof_status") in {"measured_complete", "measured_partial"}
    )
    unavailable = tuple(item for item in selected if item.get("proof_status") == "unavailable")
    partial = any(item.get("proof_status") == "measured_partial" for item in selected)
    if not measured:
        coverage_status = "unavailable"
    elif unavailable or partial:
        coverage_status = "measured_partial"
    else:
        coverage_status = "measured_complete"
    return {
        "stage": stage,
        "supplier_usage_coverage": {
            "status": coverage_status,
            "record_count": len(selected),
            "measured_record_count": len(measured),
            "unavailable_record_count": len(unavailable),
            "billing_complete_record_count": sum(
                item.get("billing_complete") is True for item in selected
            ),
        },
        "totals": {
            dimension: _token_dimension_total(selected, dimension) for dimension in TOKEN_DIMENSIONS
        },
        "by_budget_owner": tuple(
            {
                "budget_owner_kind": item.get("budget_owner_kind", "unavailable"),
                "budget_owner_id": item.get("budget_owner_id", "unavailable"),
                "proof_status": item.get("proof_status", "unavailable"),
                "billing_complete": item.get("billing_complete", "unavailable"),
            }
            for item in selected
        ),
    }


def _token_dimension_total(
    records: tuple[dict[str, Any], ...],
    dimension: str,
) -> dict[str, object]:
    values = tuple(
        value
        for item in records
        if isinstance((value := item.get(dimension)), int) and not isinstance(value, bool)
    )
    unavailable_count = sum(dimension in item.get("unavailable_dimensions", ()) for item in records)
    if not values:
        status = "unavailable"
        total: int | str = "unavailable"
    else:
        status = "measured_partial" if unavailable_count else "measured_complete"
        total = sum(values)
    return {
        "status": status,
        "value": total,
        "measured_record_count": len(values),
        "unavailable_record_count": unavailable_count,
    }


def _resource_document(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return {
        label: _resource_dimension(records, dimension, unit, aggregation)
        for label, dimension, unit, aggregation in RESOURCE_REPORT_DIMENSIONS
    }


def _resource_dimension(
    records: tuple[dict[str, Any], ...],
    dimension: str,
    unit: str,
    aggregation: str,
) -> dict[str, object]:
    selected = tuple(item for item in records if item.get("dimension_id") == dimension)
    values = tuple(
        parsed
        for item in selected
        if item.get("proof_status") in {"measured_complete", "measured_partial"}
        and (parsed := _decimal_value(item.get("value"))) is not None
    )
    unavailable_count = sum(item.get("proof_status") == "unavailable" for item in selected)
    if not values:
        status = "unavailable"
        value = "unavailable"
    else:
        status = "measured_partial" if unavailable_count else "measured_complete"
        if aggregation == "sum":
            aggregate = sum(values, Decimal(0))
        elif aggregation == "maximum":
            aggregate = max(values)
        else:
            aggregate = values[-1]
        value = _decimal_text(aggregate)
    return {
        "status": status,
        "value": value,
        "unit": unit,
        "aggregation": aggregation,
        "measured_record_count": len(values),
        "unavailable_record_count": unavailable_count,
    }


def _resource_coverage(document: Mapping[str, object]) -> str:
    statuses = tuple(item["status"] for item in document.values() if isinstance(item, Mapping))
    if not any(status != "unavailable" for status in statuses):
        return "unavailable"
    if any(status != "measured_complete" for status in statuses):
        return "measured_partial"
    return "measured_complete"


def _cost_document(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    measured = tuple(
        item
        for item in records
        if item.get("basis") == "actual_supplier_charge"
        and item.get("proof_status") in {"measured_complete", "measured_partial"}
        and isinstance(item.get("currency"), str)
        and _decimal_value(item.get("amount")) is not None
    )
    unavailable_count = sum(item.get("proof_status") == "unavailable" for item in records)
    totals: dict[str, Decimal] = {}
    for item in measured:
        currency = str(item["currency"])
        amount = _decimal_value(item.get("amount"))
        assert amount is not None
        totals[currency] = totals.get(currency, Decimal(0)) + amount
    if not measured:
        coverage_status = "unavailable"
    elif unavailable_count or len(measured) != len(records):
        coverage_status = "measured_partial"
    else:
        coverage_status = "measured_complete"
    return {
        "actual_supplier_charge": (
            {currency: _decimal_text(amount) for currency, amount in sorted(totals.items())}
            if totals
            else "unavailable"
        ),
        "currencies": sorted(totals) if totals else "unavailable",
        "billing_coverage": {
            "status": coverage_status,
            "record_count": len(records),
            "measured_record_count": len(measured),
            "unavailable_record_count": unavailable_count,
        },
    }


def _decimal_value(value: object) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def _pair_document(
    plan: ResolvedPlan,
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, Any]:
    limitations: list[str] = []
    comparable = True
    if left["metric_id"] == "unavailable" or left["metric_id"] != right["metric_id"]:
        comparable = False
        limitations.append("metric policy or judged evidence is unavailable")
    if (
        left["judged_case_count"] != left["case_count"]
        or right["judged_case_count"] != right["case_count"]
    ):
        comparable = False
        limitations.append("pair lacks complete judged case coverage")
    left_denominator = _required_int(left, "judged_denominator")
    right_denominator = _required_int(right, "judged_denominator")
    if not left_denominator or not right_denominator:
        comparable = False
        limitations.append("pair has an empty judged denominator")
    signed_delta: dict[str, int] | str
    if comparable:
        delta = Fraction(_required_int(left, "judged_numerator"), left_denominator) - Fraction(
            _required_int(right, "judged_numerator"), right_denominator
        )
        signed_delta = {"numerator": delta.numerator, "denominator": delta.denominator}
    else:
        signed_delta = "unavailable"
    body: dict[str, object] = {
        "schema_name": "pairwise_comparison",
        "schema_version": 1,
        "comparison_project_id": plan.comparison_id,
        "resolved_plan_hash": plan.resolved_plan_hash,
        "dataset_id": plan.dataset.dataset_id,
        "case_manifest_hash": plan.dataset.case_manifest_hash,
        "left_cell_id": left["cell_id"],
        "right_cell_id": right["cell_id"],
        "left_provider_id": left["provider_id"],
        "right_provider_id": right["provider_id"],
        "case_count": left["case_count"],
        "metric_id": left["metric_id"],
        "left_judged": {
            "case_count": left["judged_case_count"],
            "numerator": left["judged_numerator"],
            "denominator": left["judged_denominator"],
        },
        "right_judged": {
            "case_count": right["judged_case_count"],
            "numerator": right["judged_numerator"],
            "denominator": right["judged_denominator"],
        },
        "signed_delta": signed_delta,
        "comparable": comparable,
        "limitations": limitations,
    }
    return {
        **body,
        "comparison_id": canonical_sha256(["oamb-pairwise-comparison-initial-v1", body]),
    }


def _observed_time(snapshot: _CellSnapshot) -> dict[str, object]:
    attempts_by_id = _unique_by(snapshot.attempts, "attempt_id", "attempt")
    case_timings: list[dict[str, object]] = []
    provider_request_durations: list[int] = []
    for case in snapshot.cases:
        attempt_ids = case.get("attempt_ids")
        selected = (
            tuple(attempts_by_id[item] for item in attempt_ids if item in attempts_by_id)
            if isinstance(attempt_ids, list)
            else ()
        )
        attempt_intervals = tuple(
            interval for item in selected if (interval := _duration_interval(item)) is not None
        )
        query_durations = tuple(
            interval[2]
            for item in selected
            if item.get("stage") == "memory_query"
            and (interval := _duration_interval(item)) is not None
        )
        provider_request_durations.extend(query_durations)
        case_wall = (
            _microseconds(
                min(item[0] for item in attempt_intervals),
                max(item[1] for item in attempt_intervals),
            )
            if attempt_intervals
            else "unavailable"
        )
        case_timings.append(
            {
                "case_manifest_entry_id": case["case_manifest_entry_id"],
                "case_wall_microseconds": case_wall,
                "provider_request_wall_microseconds": query_durations or "unavailable",
            }
        )
    run_interval = _duration_interval(snapshot.run_record)
    return {
        "run_wall_microseconds": run_interval[2] if run_interval is not None else "unavailable",
        "provider_request": _duration_summary(tuple(provider_request_durations)),
        "cases": case_timings,
        "comparability": "observed_only",
    }


def _duration_interval(document: Mapping[str, object]) -> tuple[datetime, datetime, int] | None:
    started = document.get("started_at")
    ended = document.get("ended_at")
    if not isinstance(started, str) or not isinstance(ended, str):
        return None
    try:
        start_time = datetime.fromisoformat(started)
        end_time = datetime.fromisoformat(ended)
    except ValueError:
        return None
    if (
        start_time.tzinfo is None
        or end_time.tzinfo is None
        or start_time.utcoffset() is None
        or end_time.utcoffset() is None
        or end_time < start_time
    ):
        return None
    return start_time, end_time, _microseconds(start_time, end_time)


def _microseconds(started: datetime, ended: datetime) -> int:
    delta = ended - started
    return (delta.days * 86_400 + delta.seconds) * 1_000_000 + delta.microseconds


def _duration_summary(values: tuple[int, ...]) -> dict[str, object]:
    if not values:
        return {
            "status": "unavailable",
            "count": 0,
            "median_microseconds": "unavailable",
            "p95_microseconds": "unavailable",
            "maximum_microseconds": "unavailable",
        }
    ordered = tuple(sorted(values))
    midpoint = len(ordered) // 2
    median = (
        str(ordered[midpoint])
        if len(ordered) % 2
        else _half_integer_text(ordered[midpoint - 1] + ordered[midpoint])
    )
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "status": "measured",
        "count": len(ordered),
        "median_microseconds": median,
        "p95_microseconds": ordered[p95_index],
        "maximum_microseconds": ordered[-1],
    }


def _half_integer_text(doubled_value: int) -> str:
    return str(doubled_value // 2) if doubled_value % 2 == 0 else f"{doubled_value // 2}.5"


def _model_document(binding: ModelExecutionBinding) -> dict[str, object]:
    return {
        "role_id": binding.role_id,
        "configured_model": binding.configured_model,
        "runtime_model": binding.runtime_model,
        "thinking_effort": binding.thinking_effort,
        "thinking_effort_scale": binding.thinking_effort_scale,
        "thinking_effort_rank_1_indexed": binding.thinking_effort_rank_1_indexed,
        "execution_owner": binding.execution_owner,
        "proof_kind": binding.proof_kind,
        "proof_reference": binding.proof_reference,
        "usage_coverage": binding.usage_coverage,
        "binding_source": "resolved_plan",
    }


def _retrieval_document(
    plan: ResolvedPlan,
    cell: CellSpec,
    runtime_proof_state: str,
) -> dict[str, object]:
    by_id = {item.binding_id: item for item in plan.retrieval.bindings}
    binding = by_id[cell.retrieval_binding_id]
    return {
        "cell_id": cell.cell_id,
        "provider_id": cell.provider_id,
        "binding_id": binding.binding_id,
        "binding_hash": binding.binding_hash,
        "route": binding.route,
        "query_embedding_role": binding.query_embedding_role,
        "disabled_feature": binding.disabled_feature,
        "disabled_value": binding.disabled_value,
        "request_constraint": binding.request_constraint,
        "proof_kind": binding.proof_kind,
        "configured_generation": plan.retrieval.generation,
        "runtime_proof_state": runtime_proof_state,
    }


def _retrieval_runtime_proof(
    cell: CellSpec,
    cases: tuple[dict[str, Any], ...],
    raw_payloads: Mapping[str, bytes],
) -> str:
    proofs: list[bytes] = []
    for case in cases:
        reference = case.get("retrieval_request_raw_ref")
        if not isinstance(reference, str):
            return "unavailable"
        payload = raw_payloads.get(reference)
        if payload is None:
            return "unavailable"
        proofs.append(payload)
    if not proofs:
        return "unavailable"
    if all(
        retrieval_request_proves_generation_free(cell.adapter_profile_id, proof) for proof in proofs
    ):
        return "runtime_verified"
    return "invalid"


def _render_html(export: Mapping[str, Any]) -> bytes:
    cells = export["cells"]
    comparisons = export["comparisons"]
    models = export["models"]
    retrieval = export["retrieval"]
    limitations = export["limitations"]
    assert isinstance(cells, tuple)
    assert isinstance(comparisons, tuple)
    assert isinstance(models, tuple)
    assert isinstance(retrieval, tuple)
    assert isinstance(limitations, tuple)
    cell_rows = "".join(
        "<tr>"
        f"<td>{_escape(item['provider_id'])}</td>"
        f"<td>{_escape(item['adapter_profile_id'])}</td>"
        f"<td>{item['judged_numerator']}/{item['judged_denominator']}</td>"
        f"<td>{_escape(_time_text(item['observed_time']['run_wall_microseconds']))}</td>"
        f"<td>{_escape(_request_time_text(item['observed_time']['provider_request']))}</td>"
        "</tr>"
        for item in cells
    )
    comparison_rows = "".join(
        "<tr>"
        f"<td>{_escape(item['left_provider_id'])} / {_escape(item['right_provider_id'])}</td>"
        f"<td>{_escape(_delta_text(item['signed_delta']))}</td>"
        f"<td>{str(item['comparable']).lower()}</td>"
        "</tr>"
        for item in comparisons
    )
    model_rows = "".join(
        "<tr>"
        f"<td>{_escape(item['role_id'])}</td>"
        f"<td>{_escape(item['configured_model'])}</td>"
        f"<td>{_escape(item['runtime_model'])}</td>"
        f"<td>{_escape(item['thinking_effort'])}</td>"
        f"<td>{_escape(_scale_text(item))}</td>"
        f"<td>{_escape(item['proof_kind'])}</td>"
        "</tr>"
        for item in models
    )
    retrieval_rows = "".join(
        "<tr>"
        f"<td>{_escape(item['provider_id'])}</td>"
        f"<td><code>{_escape(item['route'])}</code></td>"
        f"<td>{_escape(item['disabled_feature'])}={_escape(item['disabled_value'])}</td>"
        f'<td><span class="badge warning">{_escape(item["runtime_proof_state"])}</span></td>'
        "</tr>"
        for item in retrieval
    )
    accounting_rows = "".join(
        "<tr>"
        f"<td>{_escape(item['provider_id'])}</td>"
        f"<td>{_escape(_token_usage_text(item['accounting']['tokens']['indexing']))}</td>"
        f"<td>{_escape(_token_usage_text(item['accounting']['tokens']['retrieval']))}</td>"
        f"<td>{_escape(_token_usage_text(item['accounting']['tokens']['answer']))}</td>"
        f"<td>{_escape(_token_usage_text(item['accounting']['tokens']['judge']))}</td>"
        f"<td>{_escape(_failure_retry_text(item['accounting']['attempts']))}</td>"
        f"<td>{_escape(_resource_text(item['accounting']['resources']['peak_memory_bytes']))}</td>"
        f"<td>{_escape(_resource_text(item['accounting']['resources']['storage_bytes']))}</td>"
        f"<td>{_escape(_cost_text(item['accounting']['cost']))}</td>"
        f"<td>{_escape(_coverage_text(item['accounting']['measurement_coverage']))}</td>"
        "</tr>"
        for item in cells
    )
    limitation_items = "".join(f"<li>{_escape(item)}</li>" for item in limitations)
    embedded_json = html.escape(canonical_json_bytes(export).decode("utf-8"))
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light dark">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>OAMB comparison report</title>
<style>
:root {{ color-scheme: light dark; --bg: #ffffff; --fg: #17202a; --muted: #5d6d7e; --panel: #f5f7f9; --border: #ccd1d1; --warning-bg: #fff3cd; --warning-fg: #664d03; --pre-bg: #f1f3f5; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #111418; --fg: #edf2f7; --muted: #aab4bf; --panel: #1b2026; --border: #47515c; --warning-bg: #4b3b00; --warning-fg: #ffe69c; --pre-bg: #0b0d10; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }}
main {{ max-width: 1120px; margin: auto; padding: 2rem; }}
section {{ margin: 1.5rem 0; padding: 1rem; background: var(--panel); border: 1px solid var(--border); border-radius: .5rem; }}
table {{ width: 100%; border-collapse: collapse; }} th, td {{ padding: .5rem; text-align: left; border-bottom: 1px solid var(--border); }}
code, pre {{ background: var(--pre-bg); }} code {{ padding: .1rem .25rem; }} pre {{ padding: 1rem; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; }}
.muted {{ color: var(--muted); }} .badge {{ padding: .15rem .4rem; border-radius: .3rem; }} .warning {{ background: var(--warning-bg); color: var(--warning-fg); }}
</style>
</head>
<body><main>
<h1>OAMB comparison report</h1>
<p><strong>{_escape(export["controlled_comparison_warning"])}</strong></p>
<p class="muted">Report ID: <code>{_escape(export["report_id"])}</code></p>
<section><h2>Coverage and judged accuracy</h2><p>{export["coverage"]["unique_case_count"]} unique cases; {export["coverage"]["provider_specific_result_count"]} provider-specific results.</p>
<table><thead><tr><th>Provider</th><th>Profile</th><th>Judged</th><th>Run wall</th><th>Provider request</th></tr></thead><tbody>{cell_rows}</tbody></table></section>
<section><h2>Pairwise accuracy deltas</h2><table><thead><tr><th>Pair</th><th>Left minus right</th><th>Comparable</th></tr></thead><tbody>{comparison_rows}</tbody></table></section>
<section><h2>Model and thinking-effort bindings</h2><table><thead><tr><th>Role</th><th>Configured</th><th>Runtime</th><th>Effort</th><th>Scale rank</th><th>Proof</th></tr></thead><tbody>{model_rows}</tbody></table></section>
<section><h2>Generation-free retrieval</h2><table><thead><tr><th>Provider</th><th>Route</th><th>Disabled setting</th><th>Runtime proof</th></tr></thead><tbody>{retrieval_rows}</tbody></table></section>
<section><h2>Accounting and measurement coverage</h2><table><thead><tr><th>Provider</th><th>Indexing / retain supplier usage</th><th>Retrieval supplier usage</th><th>Answer supplier usage</th><th>Judge supplier usage</th><th>Failures / retries</th><th>Peak memory</th><th>Storage</th><th>Cost / billing</th><th>Measurement coverage</th></tr></thead><tbody>{accounting_rows}</tbody></table></section>
<section><h2>Limitations</h2><ul>{limitation_items}</ul></section>
<section><h2>Deterministic export</h2><pre>{embedded_json}</pre></section>
</main></body></html>
"""
    return document.encode("utf-8")


def _scale_text(item: Mapping[str, Any]) -> str:
    scale = item["thinking_effort_scale"]
    rank = item["thinking_effort_rank_1_indexed"]
    if not scale:
        return "not applicable"
    return f"{rank}/{len(scale)} ({' < '.join(str(value) for value in scale)})"


def _time_text(value: object) -> str:
    return "unavailable" if value == "unavailable" else f"{value} us"


def _request_time_text(value: Mapping[str, object]) -> str:
    if value["status"] == "unavailable":
        return "unavailable"
    return f"n={value['count']}, median={value['median_microseconds']} us"


def _delta_text(value: object) -> str:
    if not isinstance(value, dict):
        return "unavailable"
    return f"{value['numerator']}/{value['denominator']}"


def _token_usage_text(value: Mapping[str, Any]) -> str:
    coverage = value["supplier_usage_coverage"]
    totals = value["totals"]
    dimensions = (
        ("input", "input_tokens"),
        ("output", "visible_output_tokens"),
        ("total", "supplier_reported_total_tokens"),
        ("cached", "cached_input_tokens"),
        ("reasoning", "reasoning_tokens"),
    )
    rendered = ", ".join(f"{label}={totals[dimension]['value']}" for label, dimension in dimensions)
    return f"{coverage['status']} (n={coverage['record_count']}; {rendered})"


def _failure_retry_text(value: Mapping[str, object]) -> str:
    return (
        f"failed={value['failed_count']}, retry={value['retry_count']}, "
        f"cancelled={value['cancelled_count']}, unknown={value['unknown_outcome_count']}"
    )


def _resource_text(value: Mapping[str, object]) -> str:
    if value["status"] == "unavailable":
        return "unavailable"
    return f"{value['value']} {value['unit']} ({value['status']})"


def _cost_text(value: Mapping[str, Any]) -> str:
    totals = value["actual_supplier_charge"]
    rendered = (
        ", ".join(f"{currency} {amount}" for currency, amount in totals.items())
        if isinstance(totals, Mapping)
        else "unavailable"
    )
    return f"{rendered}; billing={value['billing_coverage']['status']}"


def _coverage_text(value: Mapping[str, Any]) -> str:
    tokens = value["tokens"]
    return (
        f"attempts={value['attempts']}; tokens=indexing:{tokens['indexing']},"
        f"retrieval:{tokens['retrieval']},answer:{tokens['answer']},judge:{tokens['judge']}; "
        f"resources={value['resources']}; cost={value['cost']}"
    )


def _escape(value: object) -> str:
    return html.escape(str(value))


def _expected_case_count(selection: str) -> int | None:
    if selection == "lme6":
        return 6
    if selection == "lme30":
        return 30
    return None


def _exact_document(
    documents: Mapping[str, list[tuple[bytes, dict[str, Any]]]],
    kind: str,
    cell_id: str,
) -> dict[str, Any]:
    return _exact_record(documents, kind, cell_id)[1]


def _exact_record(
    documents: Mapping[str, list[tuple[bytes, dict[str, Any]]]],
    kind: str,
    cell_id: str,
) -> tuple[bytes, dict[str, Any]]:
    records = documents.get(kind, ())
    if len(records) != 1:
        raise ComparisonProjectError(f"cell {cell_id} requires exactly one {kind}")
    return records[0]


def _unique_by(
    values: tuple[dict[str, Any], ...],
    field: str,
    label: str,
) -> dict[str, dict[str, Any]]:
    keyed: dict[str, dict[str, Any]] = {}
    for item in values:
        key = _required_text(item, field)
        if key in keyed:
            raise ComparisonProjectError(f"duplicate {label} identity")
        keyed[key] = item
    return keyed


def _required_text(document: Mapping[str, Any], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ComparisonProjectError(f"source record requires non-empty {field}")
    return value


def _required_int(document: Mapping[str, Any], field: str) -> int:
    value = document.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ComparisonProjectError(f"derived comparison requires integer {field}")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


__all__ = [
    "ComparisonProjectBuildResult",
    "ComparisonProjectError",
    "ValidatedCellRoot",
    "build_comparison_project",
]
