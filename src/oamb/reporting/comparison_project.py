"""Generic multi-cell comparison and deterministic offline report export."""

from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from fractions import Fraction
from html.parser import HTMLParser
from itertools import combinations
from pathlib import Path
from typing import Any

from oamb.artifacts.atomic import read_regular_file, sha256_file
from oamb.artifacts.capsule import publish_with_last_marker, verify_published_directory
from oamb.artifacts.composition import inspect_embedded_composition
from oamb.artifacts.validation.composition import declares_composition
from oamb.artifacts.validation.openviking_session_evidence import (
    reconstruct_openviking_session_indexing_usage,
)
from oamb.artifacts.validation.retrieval_request import (
    retrieval_request_proves_generation_free,
)
from oamb.artifacts.validation.source_root import validate_source_root
from oamb.config.doctor import CellSpec, ModelExecutionBinding, ResolvedPlan
from oamb.contracts.evidence import CapsuleManifest, ValidationResult
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.states import ValidationDisposition
from oamb.memory_systems.openviking.session_adapter import OPENVIKING_SESSION_PROFILE_ID

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
VISIBLE_EVIDENCE_KEYS = frozenset(
    {
        "provider_evidence_identity",
        "source_unit_id",
        "evidence_kind",
        "text",
        "occurred_start",
        "occurred_end",
        "mentioned_at",
    }
)
_BIDI_CONTROL_PATTERN = re.compile("[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_UNSAFE_HTML_TAGS = frozenset(
    {"script", "img", "link", "iframe", "object", "embed", "audio", "video", "source"}
)


class _OfflineHtmlInspector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.unsafe = False
        self.references: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        normalized_tag = tag.casefold()
        if normalized_tag in _UNSAFE_HTML_TAGS:
            self.unsafe = True
        for name, value in attrs:
            normalized_name = name.casefold()
            if normalized_name.startswith("on"):
                self.unsafe = True
            if normalized_name in {"href", "src"}:
                reference = value or ""
                self.references.append(reference)
                if reference.lstrip().casefold().startswith("javascript:"):
                    self.unsafe = True

    def handle_startendtag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        self.handle_starttag(tag, attrs)


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
    ingestion_plans: tuple[dict[str, Any], ...]
    attempts: tuple[dict[str, Any], ...]
    token_usage: tuple[dict[str, Any], ...]
    resources: tuple[dict[str, Any], ...]
    costs: tuple[dict[str, Any], ...]
    run_record: dict[str, Any]
    retrieval_runtime_proof_state: str
    raw_payloads: Mapping[str, bytes]
    infrastructure_retries: tuple[dict[str, Any], ...] = ()
    composition_part_states: tuple[str, ...] = ()


def build_comparison_project(
    plan: ResolvedPlan,
    sources: Mapping[str, ValidatedCellRoot],
    *,
    output_root: Path,
    dataset_source: Path | None = None,
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
    local_question_content = (
        _load_local_question_content(plan, Path(dataset_source), snapshots[0].case_manifest_bytes)
        if dataset_source is not None
        else ()
    )
    question_documents = _question_documents(
        snapshots,
        local_question_content=local_question_content,
    )
    dataset_details = _dataset_details_document(
        plan,
        included=bool(local_question_content),
    )
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
        "dataset_details": dataset_details,
        "coverage": {
            "cell_count": len(cell_documents),
            "unique_case_count": len(snapshots[0].cases),
            "provider_specific_result_count": sum(len(item.cases) for item in snapshots),
        },
        "cells": cell_documents,
        "questions": question_documents,
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
    _validate_comparison_export(export, payloads, tuple(pair_relative_paths))
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
    ingestion_plans = tuple(item[1] for item in documents.get("ingestion_plan_record", ()))
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
        ingestion_plans=ingestion_plans,
        attempts=attempts,
        token_usage=token_usage,
        resources=resources,
        costs=costs,
        run_record=run_record,
        retrieval_runtime_proof_state=retrieval_proof_state,
        raw_payloads=raw_payloads,
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

    selected_ingestion_plans: list[dict[str, Any]] = []
    selected_case_records: list[dict[str, Any]] = []
    ingestion_occurrence_ids: list[str] = []
    case_occurrence_ids: list[str] = []
    for contribution in composition.ordered_contributions:
        documents = part_documents[contribution.source_capsule_id]
        ingestion_plans_by_occurrence = _unique_by(
            tuple(item[1] for item in documents.get("ingestion_plan_record", ())),
            "ingestion_occurrence_id",
            "ingestion occurrence",
        )
        ingestion_plan = ingestion_plans_by_occurrence[contribution.ingestion_occurrence_id]
        cases_by_occurrence = _unique_by(
            tuple(item[1] for item in documents.get("case_record", ())),
            "case_occurrence_id",
            "case record",
        )
        contributed_cases = tuple(
            cases_by_occurrence[occurrence_id] for occurrence_id in contribution.case_occurrence_ids
        )
        if (
            ingestion_plan.get("ingestion_plan_id") != contribution.ingestion_plan_id
            or ingestion_plan.get("run_id") != contribution.source_run_id
            or tuple(ingestion_plan.get("ordered_case_occurrence_ids", ()))
            != contribution.case_occurrence_ids
        ):
            raise ComparisonProjectError(
                f"cell {cell.cell_id} composition ingestion provenance does not close"
            )
        if (
            tuple(_required_text(item, "case_manifest_entry_id") for item in contributed_cases)
            != contribution.case_manifest_entry_ids
        ):
            raise ComparisonProjectError(
                f"cell {cell.cell_id} composition case provenance does not close"
            )
        selected_ingestion_plans.append(ingestion_plan)
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
        (
            tuple(selected_ingestion_plans),
            "ingestion_occurrence_id",
            "composed ingestion occurrence",
        ),
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
        ingestion_plans=tuple(selected_ingestion_plans),
        attempts=all_attempts,
        token_usage=all_usage,
        resources=all_resources,
        costs=all_costs,
        run_record=run_record,
        retrieval_runtime_proof_state=retrieval_proof_state,
        raw_payloads=merged_raw_payloads,
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
        "accounting": _accounting_document(plan, snapshot),
        "limitations": limitations,
    }


def _accounting_document(plan: ResolvedPlan, snapshot: _CellSnapshot) -> dict[str, object]:
    token_stages = {
        label: (
            _indexing_token_stage_document(plan, snapshot)
            if label == "indexing"
            else _token_stage_document(snapshot.token_usage, stage)
        )
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
        "answer_visible_context_tokens": _answer_visible_context_document(snapshot),
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


def _indexing_token_stage_document(
    plan: ResolvedPlan,
    snapshot: _CellSnapshot,
) -> dict[str, Any]:
    producer_roles = tuple(
        item for item in plan.model_roles if item.role_id == snapshot.cell.producer_role_id
    )
    if len(producer_roles) != 1:
        raise ComparisonProjectError("cell requires exactly one producer model role")
    producer_binding_id = producer_roles[0].binding_hash

    logical_attempt_ids: list[str] = []
    for ingestion_plan in snapshot.ingestion_plans:
        attempt_ids = ingestion_plan.get("ordered_dispatch_attempt_ids")
        if not isinstance(attempt_ids, list) or any(
            not isinstance(attempt_id, str) or not attempt_id for attempt_id in attempt_ids
        ):
            raise ComparisonProjectError(
                "ingestion plan requires ordered logical dispatch attempt IDs"
            )
        logical_attempt_ids.extend(attempt_ids)
    if len(set(logical_attempt_ids)) != len(logical_attempt_ids):
        raise ComparisonProjectError("logical indexing dispatch attempt is duplicated")

    model_usage_by_attempt: dict[str, list[dict[str, Any]]] = {
        attempt_id: [] for attempt_id in logical_attempt_ids
    }
    operation_usage_by_attempt: dict[str, list[dict[str, Any]]] = {
        attempt_id: [] for attempt_id in logical_attempt_ids
    }
    for record in snapshot.token_usage:
        attempt_id = record.get("attempt_id")
        if record.get("stage") != "memory_ingest" or not isinstance(attempt_id, str):
            continue
        if attempt_id not in model_usage_by_attempt:
            continue
        if (
            record.get("budget_owner_kind") == "model_role"
            and record.get("budget_owner_id") == producer_binding_id
        ):
            model_usage_by_attempt[attempt_id].append(record)
        elif record.get("budget_owner_kind") == "provider_operation":
            operation_usage_by_attempt[attempt_id].append(record)

    selected: list[dict[str, Any]] = []
    for attempt_id in logical_attempt_ids:
        model_records = model_usage_by_attempt[attempt_id]
        operation_records = operation_usage_by_attempt[attempt_id]
        if len(model_records) == 1:
            selected.append(model_records[0])
        elif not model_records and len(operation_records) == 1:
            selected.append(operation_records[0])
        else:
            raise ComparisonProjectError(
                "indexing producer usage requires exactly one record per logical dispatch"
            )
    if (
        snapshot.cell.adapter_profile_id == OPENVIKING_SESSION_PROFILE_ID
        and selected
        and all(item.get("proof_status") == "unavailable" for item in selected)
    ):
        try:
            recovered = tuple(
                usage
                for ingestion_plan in snapshot.ingestion_plans
                for usage in reconstruct_openviking_session_indexing_usage(
                    raw_payloads=snapshot.raw_payloads,
                    plan=ingestion_plan,
                )
            )
        except ValueError as exc:
            raise ComparisonProjectError("OpenViking indexing usage evidence is invalid") from exc
        if tuple(item.attempt_id for item in recovered) != tuple(logical_attempt_ids):
            raise ComparisonProjectError("OpenViking indexing usage ledger does not close")
        selected = [
            {
                "attempt_id": item.attempt_id,
                "stage": "memory_ingest",
                "budget_owner_kind": "model_role",
                "budget_owner_id": producer_binding_id,
                "input_tokens": item.prompt_tokens,
                "visible_output_tokens": item.completion_tokens,
                "supplier_reported_total_tokens": item.total_tokens,
                "cached_input_tokens": item.cached_tokens,
                "reasoning_tokens": item.reasoning_tokens,
                "covered_dimensions": TOKEN_DIMENSIONS,
                "unavailable_dimensions": (),
                "proof_status": "measured_complete",
                "billing_complete": False,
                "measurement_source": "sealed_openviking_task_snapshot",
                "raw_response_ref": item.raw_response_ref,
            }
            for item in recovered
        ]
    return _token_stage_document(tuple(selected), "memory_ingest")


def _answer_visible_context_document(snapshot: _CellSnapshot) -> dict[str, object]:
    measured: list[int] = []
    for case in snapshot.cases:
        count = case.get("visible_evidence_token_count")
        if count is None:
            continue
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ComparisonProjectError("answer-visible context token count is invalid")
        _visible_context_text(snapshot, case)
        measured.append(count)
    if not measured:
        return {
            "status": "unavailable",
            "case_count": len(snapshot.cases),
            "measured_case_count": 0,
            "total": "unavailable",
            "mean": "unavailable",
        }
    status = "measured_complete" if len(measured) == len(snapshot.cases) else "measured_partial"
    total = sum(measured)
    return {
        "status": status,
        "case_count": len(snapshot.cases),
        "measured_case_count": len(measured),
        "total": total,
        "mean": _mean_text(total, len(measured)),
    }


def _mean_text(total: int, count: int) -> str:
    value = (Decimal(total) / Decimal(count)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    rendered = format(value, "f")
    return rendered[:-2] if rendered.endswith(".0") else rendered


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


def _load_local_question_content(
    plan: ResolvedPlan,
    dataset_source: Path,
    case_manifest_bytes: bytes,
) -> tuple[dict[str, object], ...]:
    from oamb.workloads.longmemeval import build_lme6_bundle, build_lme30_bundle

    try:
        before_hash = sha256_file(dataset_source)
    except OSError as exc:
        raise ComparisonProjectError("dataset source is not one readable regular file") from exc
    if before_hash != plan.dataset.source_sha256:
        raise ComparisonProjectError("dataset source hash differs from the resolved plan")
    try:
        full = build_lme30_bundle(dataset_source)
        if plan.dataset.selection == "lme6":
            bundle = build_lme6_bundle(full)
        elif plan.dataset.selection == "lme30":
            bundle = full
        else:
            raise ComparisonProjectError(
                "question-detail report supports only the frozen LongMemEval selections"
            )
    except ComparisonProjectError:
        raise
    except (OSError, TypeError, ValueError) as exc:
        raise ComparisonProjectError("dataset source cannot rebuild the frozen workload") from exc
    try:
        after_hash = sha256_file(dataset_source)
    except OSError as exc:
        raise ComparisonProjectError("dataset source changed while the report was built") from exc
    if after_hash != before_hash:
        raise ComparisonProjectError("dataset source changed while the report was built")
    if (
        bundle.dataset_manifest.dataset_id != plan.dataset.dataset_id
        or bundle.dataset_manifest.revision != plan.dataset.revision
        or bundle.case_manifest.workload_id != plan.dataset.workload_id
        or bundle.case_manifest.manifest_hash != plan.dataset.case_manifest_hash
        or canonical_json_bytes(bundle.case_manifest) != case_manifest_bytes
    ):
        raise ComparisonProjectError("dataset source does not match the frozen case manifest")

    content: list[dict[str, object]] = []
    for manifest_case, row in zip(
        bundle.case_manifest.cases,
        bundle.selected_rows,
        strict=True,
    ):
        if manifest_case.raw_question_id != row.question_id:
            raise ComparisonProjectError("dataset question identity differs from the case manifest")
        answer_sessions = _answer_session_documents(row)
        content.append(
            {
                "case_manifest_entry_id": manifest_case.case_manifest_entry_id,
                "raw_question_id": row.question_id,
                "question_type": row.question_type,
                "question": row.question,
                "gold_answer": row.answer,
                "answer_sessions": answer_sessions,
                "has_answer_label_mismatch": row.has_answer_label_mismatch,
            }
        )
    return tuple(content)


def _answer_session_documents(row: Any) -> tuple[dict[str, object], ...]:
    answer_session_ids = frozenset(row.answer_session_ids)
    if len(answer_session_ids) != len(row.answer_session_ids):
        raise ComparisonProjectError("dataset answer session identity is duplicated")
    selected: list[dict[str, object]] = []
    resolved_ids: set[str] = set()
    for session in row.sessions:
        if session.session_id not in answer_session_ids:
            continue
        if session.session_id in resolved_ids:
            raise ComparisonProjectError("dataset answer session identity is duplicated")
        resolved_ids.add(session.session_id)
        selected.append(
            {
                "session_id": session.session_id,
                "timestamp": session.raw_timestamp,
                "messages": tuple(
                    {
                        "role": message.role,
                        "content": message.content,
                        "has_answer": message.has_answer,
                    }
                    for message in session.messages
                ),
            }
        )
    if resolved_ids != answer_session_ids:
        raise ComparisonProjectError("dataset answer session identity cannot be resolved")
    return tuple(selected)


def _dataset_details_document(
    plan: ResolvedPlan,
    *,
    included: bool,
) -> dict[str, object]:
    if not included:
        return {
            "status": "absent",
            "reason": "dataset_source_not_supplied",
        }
    from oamb.workloads.longmemeval import (
        LME_DATASET_CITATION,
        LME_DATASET_SOURCE_ID,
        LME_PAYLOAD_POLICY,
        LME_SOURCE_LICENSE_ID,
    )

    return {
        "status": "verified",
        "source_id": LME_DATASET_SOURCE_ID,
        "revision": plan.dataset.revision,
        "source_sha256": plan.dataset.source_sha256,
        "case_manifest_hash": plan.dataset.case_manifest_hash,
        "distribution_scope": "local_only",
        "license_id": LME_SOURCE_LICENSE_ID,
        "payload_policy": LME_PAYLOAD_POLICY,
        "citation": LME_DATASET_CITATION,
    }


def _question_documents(
    snapshots: tuple[_CellSnapshot, ...],
    *,
    local_question_content: tuple[dict[str, object], ...],
) -> tuple[dict[str, object], ...]:
    manifest_cases = snapshots[0].case_manifest.get("cases")
    if not isinstance(manifest_cases, list):
        raise ComparisonProjectError("question report requires a case manifest inventory")
    manifest_case_ids = tuple(
        _required_text(item, "case_manifest_entry_id") for item in manifest_cases
    )
    content_by_id: dict[str, dict[str, object]] = {}
    if local_question_content:
        for item in local_question_content:
            identity = item.get("case_manifest_entry_id")
            if not isinstance(identity, str) or identity in content_by_id:
                raise ComparisonProjectError("question content is missing or duplicated")
            content_by_id[identity] = item
        if tuple(content_by_id) != manifest_case_ids:
            raise ComparisonProjectError("question content order differs from the manifest")

    questions: list[dict[str, object]] = []
    for index, case_id in enumerate(manifest_case_ids):
        provider_results = tuple(
            _question_provider_result(
                snapshot,
                snapshot.cases[index],
                include_content=bool(local_question_content),
            )
            for snapshot in snapshots
        )
        document: dict[str, object] = {
            "case_manifest_entry_id": case_id,
            "ordinal_1_indexed": index + 1,
            "provider_results": provider_results,
        }
        if local_question_content:
            document.update(content_by_id[case_id])
        questions.append(document)
    return tuple(questions)


def _question_provider_result(
    snapshot: _CellSnapshot,
    case: dict[str, Any],
    *,
    include_content: bool,
) -> dict[str, object]:
    state = _case_display_state(case)
    result: dict[str, object] = {
        "provider_id": snapshot.cell.provider_id,
        "display_state": state,
        "evaluation_disposition": case.get("evaluation_disposition", "unavailable"),
        "metric_numerator": case.get("metric_numerator", "unavailable"),
        "metric_denominator": case.get("metric_denominator", "unavailable"),
        "context_tokens": case.get("visible_evidence_token_count", "unavailable"),
    }
    if not include_content:
        return result
    if state not in {"correct", "incorrect", "unjudged"}:
        result["detail_unavailable_reason"] = f"case state is {state}"
        return result
    answer_reference = case.get("answer_raw_ref")
    answer_hash = case.get("parsed_answer_sha256")
    evaluation_reference = case.get("evaluation_raw_ref")
    if not all(isinstance(item, str) for item in (answer_reference, answer_hash)):
        result["detail_unavailable_reason"] = "answer evidence is unavailable"
        return result
    answer_payload = snapshot.raw_payloads.get(str(answer_reference))
    if answer_payload is None:
        raise ComparisonProjectError("answer raw reference is absent from the capsule")
    result["model_answer"] = _model_output_text(
        answer_payload,
        expected_sha256=str(answer_hash),
    )
    result["injected_context"] = _visible_context_text(snapshot, case)
    if isinstance(evaluation_reference, str):
        evaluation_payload = snapshot.raw_payloads.get(evaluation_reference)
        if evaluation_payload is None:
            raise ComparisonProjectError("evaluation raw reference is absent from the capsule")
        result["judge_decision"] = _judge_decision(evaluation_payload, case)
    else:
        result["judge_decision"] = "unavailable"
    return result


def _case_display_state(case: Mapping[str, Any]) -> str:
    state = case.get("state")
    disposition = case.get("evaluation_disposition")
    numerator = case.get("metric_numerator")
    denominator = case.get("metric_denominator")
    if state == "completed" and disposition == "judged":
        if (
            isinstance(numerator, int)
            and not isinstance(numerator, bool)
            and isinstance(denominator, int)
            and not isinstance(denominator, bool)
            and denominator > 0
        ):
            return "correct" if numerator == denominator else "incorrect"
        return "unavailable"
    if state == "completed":
        return "unjudged"
    error_stage = str(case.get("error_stage", ""))
    if "unknown" in error_stage:
        return "unknown"
    if "timeout" in error_stage:
        return "timeout"
    if state == "error":
        return "failed"
    if state in {"unsupported", "budget_exceeded"}:
        return "blocked"
    return "unavailable"


def _model_output_text(payload: bytes, *, expected_sha256: str) -> str:
    try:
        document = json.loads(payload, object_pairs_hook=_unique_json_object)
    except (TypeError, ValueError) as exc:
        raise ComparisonProjectError("answer evidence is not strict JSON") from exc
    choices = document.get("choices") if isinstance(document, dict) else None
    if not isinstance(choices, list) or len(choices) != 1:
        raise ComparisonProjectError("answer evidence must contain exactly one choice")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise ComparisonProjectError("answer evidence choice does not contain text")
    try:
        encoded = content.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ComparisonProjectError("answer evidence is not valid UTF-8 text") from exc
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ComparisonProjectError("answer hash differs from the parsed model output")
    return content


def _judge_decision(payload: bytes, case: Mapping[str, Any]) -> str:
    try:
        evaluation = json.loads(payload, object_pairs_hook=_unique_json_object)
        trace_hex = evaluation.get("trace_hex") if isinstance(evaluation, dict) else None
        trace_bytes = bytes.fromhex(trace_hex) if isinstance(trace_hex, str) else b""
        trace = json.loads(trace_bytes, object_pairs_hook=_unique_json_object)
    except (TypeError, ValueError) as exc:
        raise ComparisonProjectError("judge evaluation trace is invalid") from exc
    if (
        not isinstance(evaluation, dict)
        or hashlib.sha256(trace_bytes).hexdigest() != evaluation.get("result_sha256")
        or evaluation.get("metric_id") != case.get("metric_id")
        or evaluation.get("numerator") != case.get("metric_numerator")
        or evaluation.get("denominator") != case.get("metric_denominator")
        or evaluation.get("parsed_answer_sha256") != case.get("parsed_answer_sha256")
    ):
        raise ComparisonProjectError("judge evaluation trace differs from the case result")
    decision = trace.get("judge_decision") if isinstance(trace, dict) else None
    expected = "yes" if case.get("metric_numerator") == case.get("metric_denominator") else "no"
    if not isinstance(decision, str) or decision not in {"yes", "no"} or decision != expected:
        raise ComparisonProjectError("judge decision differs from the case metric")
    return decision


def _visible_context_text(snapshot: _CellSnapshot, case: Mapping[str, Any]) -> str:
    from oamb.workloads.visible_evidence import count_o200k_tokens, tokenizer_fingerprint

    reference = case.get("visible_evidence_raw_ref")
    expected_hash = case.get("visible_evidence_sha256")
    byte_count = case.get("visible_evidence_byte_count")
    token_count = case.get("visible_evidence_token_count")
    fingerprint = case.get("visible_evidence_tokenizer_fingerprint")
    if not isinstance(reference, str):
        raise ComparisonProjectError("answer-visible context raw reference is unavailable")
    payload = snapshot.raw_payloads.get(reference)
    if payload is None:
        raise ComparisonProjectError("answer-visible context raw reference is absent")
    if (
        reference != expected_hash
        or hashlib.sha256(payload).hexdigest() != reference
        or len(payload) != byte_count
        or fingerprint != tokenizer_fingerprint()
        or count_o200k_tokens(payload) != token_count
    ):
        raise ComparisonProjectError("answer-visible context hash, token, or fingerprint drifted")
    try:
        text = payload.decode("utf-8", errors="strict")
        lines = text.splitlines()
        for line in lines:
            item = json.loads(line, object_pairs_hook=_unique_json_object)
            if not isinstance(item, dict) or set(item) != VISIBLE_EVIDENCE_KEYS:
                raise ValueError("visible evidence line has the wrong shape")
            if not all(
                isinstance(item[field], str) and item[field]
                for field in (
                    "provider_evidence_identity",
                    "evidence_kind",
                    "text",
                )
            ):
                raise ValueError("visible evidence line has empty identity or text")
            source_unit_id = item["source_unit_id"]
            if source_unit_id is not None and (
                not isinstance(source_unit_id, str) or not source_unit_id
            ):
                raise ValueError("visible evidence line has an invalid source unit")
            if any(
                item[field] is not None and not isinstance(item[field], str)
                for field in ("occurred_start", "occurred_end", "mentioned_at")
            ):
                raise ValueError("visible evidence line has an invalid timestamp")
    except (UnicodeDecodeError, TypeError, ValueError) as exc:
        raise ComparisonProjectError("answer-visible context is not strict UTF-8 JSONL") from exc
    return text


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
        "indexing_ready": _indexing_ready_summary(snapshot.attempts, snapshot.ingestion_plans),
        "provider_request": _duration_summary(tuple(provider_request_durations)),
        "cases": case_timings,
        "comparability": "observed_only",
    }


def _indexing_ready_summary(
    attempts: tuple[Mapping[str, object], ...],
    ingestion_plans: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for attempt in attempts:
        if (
            attempt.get("parent_kind") == "ingestion_plan"
            and attempt.get("stage") in {"memory_ingest", "memory_readiness"}
            and isinstance(attempt.get("parent_id"), str)
        ):
            grouped.setdefault(str(attempt["parent_id"]), []).append(attempt)
    plan_by_id = _unique_by(
        tuple(dict(item) for item in ingestion_plans),
        "ingestion_occurrence_id",
        "ingestion occurrence",
    )
    spans: list[int] = []
    for plan_id, plan in plan_by_id.items():
        if plan.get("state") not in {"ready", "sealed"}:
            continue
        selected = grouped.get(plan_id, ())
        ingest_intervals = tuple(
            interval
            for item in selected
            if item.get("stage") == "memory_ingest"
            and (interval := _duration_interval(item)) is not None
        )
        readiness_attempts = tuple(
            (item, interval)
            for item in selected
            if item.get("stage") == "memory_readiness"
            and (interval := _duration_interval(item)) is not None
        )
        if not ingest_intervals or not readiness_attempts:
            continue
        final_readiness, final_interval = max(
            readiness_attempts,
            key=lambda item: (
                item[1][1],
                item[1][0],
                str(item[0].get("attempt_id", "")),
            ),
        )
        if final_readiness.get("outcome") != "succeeded":
            continue
        started = min(item[0] for item in ingest_intervals)
        ready = final_interval[1]
        if any(item[1] > ready for item in ingest_intervals):
            raise ComparisonProjectError("indexing readiness ended before ingestion settled")
        spans.append(_microseconds(started, ready))
    summary = _duration_summary(tuple(spans))
    if spans and len(spans) != len(plan_by_id):
        summary["status"] = "measured_partial"
    return summary


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


def _validate_comparison_export(
    export: Mapping[str, Any],
    payloads: Mapping[str, bytes],
    pair_relative_paths: tuple[str, ...],
) -> None:
    body = dict(export)
    report_id = body.pop("report_id", None)
    if report_id != canonical_sha256(["oamb-comparison-project-initial-v1", body]):
        raise ComparisonProjectError("comparison export report identity drifted")
    expected_paths = {REPORT_EXPORT_NAME, REPORT_HTML_NAME, *pair_relative_paths}
    if set(payloads) != expected_paths:
        raise ComparisonProjectError("comparison export payload inventory is incomplete")
    if payloads[REPORT_EXPORT_NAME] != canonical_json_bytes(export):
        raise ComparisonProjectError("comparison export JSON is not canonical")
    comparisons = export.get("comparisons")
    if not isinstance(comparisons, tuple) or len(comparisons) != len(pair_relative_paths):
        raise ComparisonProjectError("comparison export pair inventory is incomplete")
    for comparison, relative_path in zip(comparisons, pair_relative_paths, strict=True):
        if payloads[relative_path] != canonical_json_bytes(comparison):
            raise ComparisonProjectError("comparison export pair payload drifted")

    questions = export.get("questions")
    cells = export.get("cells")
    details = export.get("dataset_details")
    if (
        not isinstance(questions, tuple)
        or not isinstance(cells, tuple)
        or not isinstance(details, dict)
    ):
        raise ComparisonProjectError("comparison export dataset detail shape is invalid")
    detail_status = details.get("status")
    content_keys = {
        "raw_question_id",
        "question_type",
        "question",
        "gold_answer",
        "answer_sessions",
    }
    if detail_status == "absent":
        if details != {"status": "absent", "reason": "dataset_source_not_supplied"}:
            raise ComparisonProjectError("comparison export absent dataset detail is invalid")
        if any(content_keys.intersection(question) for question in questions):
            raise ComparisonProjectError("comparison export has partial dataset detail")
    elif detail_status == "verified":
        required_detail_keys = {
            "status",
            "source_id",
            "revision",
            "source_sha256",
            "case_manifest_hash",
            "distribution_scope",
            "license_id",
            "payload_policy",
            "citation",
        }
        if (
            set(details) != required_detail_keys
            or details.get("distribution_scope") != "local_only"
            or any(not content_keys.issubset(question) for question in questions)
        ):
            raise ComparisonProjectError("comparison export verified dataset detail is incomplete")
    else:
        raise ComparisonProjectError("comparison export dataset detail status is invalid")
    expected_providers = {item.get("provider_id") for item in cells}
    allowed_states = {
        "correct",
        "incorrect",
        "unjudged",
        "failed",
        "blocked",
        "timeout",
        "unknown",
        "unavailable",
    }
    for question in questions:
        results = question.get("provider_results")
        if (
            not isinstance(results, tuple)
            or {item.get("provider_id") for item in results if isinstance(item, dict)}
            != expected_providers
        ):
            raise ComparisonProjectError("comparison export question/provider matrix is incomplete")
        if any(item.get("display_state") not in allowed_states for item in results):
            raise ComparisonProjectError("comparison export question state is invalid")

    html_bytes = payloads[REPORT_HTML_NAME]
    try:
        rendered = html_bytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ComparisonProjectError("comparison export does not contain safe HTML") from exc
    if html_bytes != _render_html(export):
        raise ComparisonProjectError("comparison export HTML re-render differs")
    lowered = rendered.lower()
    inspector = _OfflineHtmlInspector()
    inspector.feed(rendered)
    inspector.close()
    unsafe = (
        inspector.unsafe
        or '<meta http-equiv="content-security-policy"' not in lowered
        or "default-src 'none'" not in lowered
        or "connect-src 'none'" not in lowered
        or "object-src 'none'" not in lowered
        or "base-uri 'none'" not in lowered
    )
    if unsafe or inspector.references != [REPORT_EXPORT_NAME]:
        raise ComparisonProjectError("comparison export does not contain safe HTML")
    if rendered.count('<a href="report.json" download>') != 1:
        raise ComparisonProjectError("comparison export JSON download target is invalid")


def _render_html(export: Mapping[str, Any]) -> bytes:
    cells = export["cells"]
    comparisons = export["comparisons"]
    models = export["models"]
    retrieval = export["retrieval"]
    questions = export["questions"]
    limitations = export["limitations"]
    assert isinstance(cells, tuple)
    assert isinstance(comparisons, tuple)
    assert isinstance(models, tuple)
    assert isinstance(retrieval, tuple)
    assert isinstance(questions, tuple)
    assert isinstance(limitations, tuple)
    cell_rows = "".join(_provider_summary_row(item) for item in cells)
    secondary_accounting = _secondary_accounting_html(cells)
    comparison_rows = "".join(
        "<tr>"
        f"<td>{_escape(item['left_provider_id'])}</td>"
        f"<td>{_escape(item['right_provider_id'])}</td>"
        f"<td>{_escape(_delta_text(item['signed_delta']))}</td>"
        "</tr>"
        for item in comparisons
        if item["comparable"] is True
    )
    incomparable_count = sum(item["comparable"] is not True for item in comparisons)
    comparison_note = (
        f'<p class="muted">{incomparable_count} incomparable pair(s) are omitted here and '
        "retained with reasons in report.json.</p>"
        if incomparable_count
        else ""
    )
    model_rows = "".join(
        "<tr>"
        f"<td>{_escape(item['role_id'])}</td>"
        f"<td>{_escape(item['runtime_model'])}</td>"
        f"<td>{_escape(item['thinking_effort'])}</td>"
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
    question_rows = _question_matrix_rows(questions, cells)
    question_details = _question_detail_html(questions)
    provider_headings = "".join(f"<th>{_escape(item['provider_id'])}</th>" for item in cells)
    limitation_items = "".join(f"<li>{_escape(item)}</li>" for item in limitations)
    dataset_notice = _dataset_notice(export["dataset_details"])
    style = """
:root { color-scheme: light dark; --bg:#fff; --fg:#17202a; --muted:#5d6d7e; --panel:#f5f7f9; --border:#ccd1d1; --warning-bg:#fff3cd; --warning-fg:#664d03; --pre-bg:#f1f3f5; --good-bg:#d1e7dd; --good-fg:#0f5132; --bad-bg:#f8d7da; --bad-fg:#842029; }
@media (prefers-color-scheme: dark) { :root { --bg:#111418; --fg:#edf2f7; --muted:#aab4bf; --panel:#1b2026; --border:#47515c; --warning-bg:#4b3b00; --warning-fg:#ffe69c; --pre-bg:#0b0d10; --good-bg:#123c2d; --good-fg:#a3e9c4; --bad-bg:#4a1d24; --bad-fg:#ffb3bd; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 system-ui,sans-serif; }
main { max-width:1180px; margin:auto; padding:2rem; }
section { margin:1.5rem 0; padding:1rem; background:var(--panel); border:1px solid var(--border); border-radius:.6rem; }
.table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; }
th,td { padding:.55rem; text-align:left; vertical-align:top; border-bottom:1px solid var(--border); }
th { white-space:nowrap; }
code,pre { background:var(--pre-bg); }
code { padding:.1rem .25rem; overflow-wrap:anywhere; word-break:break-word; }
pre { max-height:26rem; padding:1rem; overflow:auto; white-space:pre-wrap; overflow-wrap:anywhere; unicode-bidi:plaintext; }
details { margin:.7rem 0; }
summary { cursor:pointer; font-weight:650; }
.muted { color:var(--muted); }
.badge { display:inline-block; padding:.12rem .42rem; border-radius:.35rem; white-space:nowrap; }
.correct { background:var(--good-bg); color:var(--good-fg); }
.incorrect,.failed,.timeout,.unknown { background:var(--bad-bg); color:var(--bad-fg); }
.unjudged,.blocked,.unavailable,.warning,.dataset-notice { background:var(--warning-bg); color:var(--warning-fg); }
.dataset-notice { padding:.8rem 1rem; border-radius:.5rem; }
.provider-detail,.session { border-left:3px solid var(--border); padding-left:1rem; }
.metric-key { padding:.75rem 1rem; border-left:4px solid #2f6fdd; background:var(--pre-bg); }
a { color:inherit; }
""".strip()
    style_hash = base64.b64encode(hashlib.sha256(style.encode()).digest()).decode("ascii")
    csp = (
        "default-src 'none'; "
        f"style-src 'sha256-{style_hash}'; "
        "connect-src 'none'; img-src 'none'; font-src 'none'; media-src 'none'; "
        "object-src 'none'; frame-src 'none'; form-action 'none'; base-uri 'none'"
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light dark">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<title>OAMB comparison report</title>
<style>{style}</style>
</head>
<body><main>
<h1>OAMB comparison report</h1>
<p><strong>{_escape(export["controlled_comparison_warning"])}</strong></p>
{dataset_notice}
<p class="muted">Report ID: <code>{_escape(export["report_id"])}</code></p>
<section><h2>Provider decision summary</h2><p class="metric-key"><strong>Four decision metrics:</strong> Accuracy · Ctx tokens · Indexing tokens · Index / recall latency</p><p>{export["coverage"]["unique_case_count"]} unique cases; {export["coverage"]["provider_specific_result_count"]} provider-specific results. Ctx tokens are the exact retrieval context shown to the answer model.</p>
<div class="table-wrap"><table><thead><tr><th>Provider / profile</th><th>Judged accuracy</th><th>Ctx tokens</th><th>Indexing tokens</th><th>Index-ready latency (s)</th><th>Recall latency (s)</th></tr></thead><tbody>{cell_rows}</tbody></table></div>
{_indexing_measurement_note(cells)}{_omitted_measurements_note(cells)}<p class="muted">Ctx tokens are the exact context shown to the answer model; they are not provider-internal retrieval supplier usage. Index-ready latency spans first ingest through readiness per isolated context; recall latency is the provider memory-query request. Both show median / p95 / max observed seconds.</p>{secondary_accounting}</section>
<section><h2>Pairwise accuracy deltas</h2><p>Compares two providers' judged accuracy on the same questions. Positive favors Provider A; negative favors Provider B. Values are percentage points.</p><div class="table-wrap"><table><thead><tr><th>Provider A</th><th>Provider B</th><th>Accuracy delta (A − B)</th></tr></thead><tbody>{comparison_rows}</tbody></table></div>{comparison_note}</section>
<section><h2>Question results</h2><p>Each row shows one frozen question across all providers; expand the evidence-backed details below when available.</p><div class="table-wrap"><table><thead><tr><th>Question</th><th>Type</th>{provider_headings}</tr></thead><tbody>{question_rows}</tbody></table></div>{question_details}</section>
<section><h2>Model and thinking-effort bindings</h2><p>For generative roles, effort follows <code>low &lt; high &lt; max</code>; embedding is not applicable. Runtime models were verified against this comparison's frozen configured bindings before dispatch; any mismatch fails the run. The complete bindings remain in report.json.</p><div class="table-wrap"><table><thead><tr><th>Role</th><th>Runtime</th><th>Effort</th><th>Proof</th></tr></thead><tbody>{model_rows}</tbody></table></div></section>
<section><h2>Generation-free retrieval</h2><div class="table-wrap"><table><thead><tr><th>Provider</th><th>Route</th><th>Disabled setting</th><th>Runtime proof</th></tr></thead><tbody>{retrieval_rows}</tbody></table></div></section>
<section><h2>Limitations</h2><ul>{limitation_items}</ul></section>
<section><h2>Deterministic export</h2><p><a href="report.json" download>Download report.json</a> for the complete machine-readable evidence and unavailable-measurement detail.</p></section>
</main></body></html>
"""
    return document.encode("utf-8")


def _provider_summary_row(item: Mapping[str, Any]) -> str:
    return (
        "<tr>"
        f"<td><strong>{_escape(item['provider_id'])}</strong><br>"
        f'<span class="muted">{_escape(item["adapter_profile_id"])}</span></td>'
        f"<td>{_escape(_accuracy_text(item))}</td>"
        f"<td>{_escape(_context_text(item))}</td>"
        f"<td>{_escape(_supplier_tokens_text(item['accounting']['tokens']['indexing']))}</td>"
        f"<td>{_escape(_request_time_text(item['observed_time']['indexing_ready']))}</td>"
        f"<td>{_escape(_request_time_text(item['observed_time']['provider_request']))}</td>"
        "</tr>"
    )


def _accuracy_text(item: Mapping[str, Any]) -> str:
    numerator = item["judged_numerator"]
    denominator = item["judged_denominator"]
    judged_case_count = item.get("judged_case_count")
    case_count = item.get("case_count")
    if (
        isinstance(judged_case_count, bool)
        or not isinstance(judged_case_count, int)
        or judged_case_count < 0
        or isinstance(case_count, bool)
        or not isinstance(case_count, int)
        or case_count < judged_case_count
    ):
        return "unavailable"
    coverage = f"judged {judged_case_count}/{case_count} cases"
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        return f"unavailable; {coverage}"
    percentage = (Decimal(numerator) * Decimal(100) / Decimal(denominator)).quantize(
        Decimal("0.1"),
        rounding=ROUND_HALF_UP,
    )
    return f"{numerator}/{denominator} ({format(percentage, 'f')}%); {coverage}"


def _context_text(item: Mapping[str, Any]) -> str:
    accounting = item["accounting"]
    context = accounting["answer_visible_context_tokens"]
    coverage = (
        f"{context['measured_case_count']}/{context['case_count']} cases ({context['status']})"
    )
    if context["status"] == "unavailable":
        return f"unavailable; {coverage}"
    return (
        f"{_number_text(context['total'])} total / {_number_text(context['mean'])} mean; {coverage}"
    )


def _secondary_accounting_html(cells: tuple[Mapping[str, Any], ...]) -> str:
    items = "".join(
        "<li>"
        f"<strong>{_escape(item['provider_id'])}</strong>: Answer input "
        f"{_escape(_number_text(item['accounting']['tokens']['answer']['totals']['input_tokens']['value']))}; "
        f"Failures / retries {_escape(_failure_retry_text(item['accounting']['attempts']))}; "
        f"cost {_escape(_cost_text(item['accounting']['cost']))}</li>"
        for item in cells
    )
    return (
        "<details><summary>Secondary accounting</summary>"
        f'<ul>{items}</ul><p class="muted">Lower Ctx tokens usually reduce Answer input, '
        "but this report does not infer a provider's internal retrieval strategy from that "
        "correlation.</p></details>"
    )


def _supplier_tokens_text(stage: Mapping[str, Any]) -> str:
    coverage = stage["supplier_usage_coverage"]
    total = stage["totals"]["supplier_reported_total_tokens"]["value"]
    metered = coverage["measured_record_count"]
    records = coverage["record_count"]
    if total == "unavailable":
        return f"unavailable ({metered}/{records} metered)"
    return f"{_number_text(total)}; {metered}/{records} metered ({coverage['status']})"


def _number_text(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, str) and re.fullmatch(r"-?\d+(?:\.\d+)?", value):
        return format(Decimal(value), ",f")
    return str(value)


def _omitted_measurements_note(
    cells: tuple[Mapping[str, Any], ...],
) -> str:
    omitted: list[str] = []
    if all(
        item["accounting"]["tokens"]["retrieval"]["supplier_usage_coverage"]["status"]
        == "unavailable"
        for item in cells
    ):
        omitted.append("provider-internal retrieval/query supplier tokens")
    if all(
        item["accounting"]["resources"]["peak_memory_bytes"]["status"] == "unavailable"
        for item in cells
    ):
        omitted.append("peak memory")
    if all(
        item["accounting"]["resources"]["storage_bytes"]["status"] == "unavailable"
        for item in cells
    ):
        omitted.append("storage")
    if not omitted:
        return ""
    return (
        '<p class="muted">Omitted because unavailable across all providers: '
        f"{_escape(', '.join(omitted))}. Full measurement states remain in report.json.</p>"
    )


def _indexing_measurement_note(cells: tuple[Mapping[str, Any], ...]) -> str:
    parts: list[str] = []
    reasoning_parts: list[str] = []
    for item in cells:
        indexing = item["accounting"]["tokens"]["indexing"]
        coverage = indexing["supplier_usage_coverage"]
        measured = coverage["measured_record_count"]
        records = coverage["record_count"]
        prefix = f"{item['provider_id']}: {measured}/{records} metered"
        if measured == 0:
            parts.append(f"{prefix}, so supplier token totals are unavailable, not zero")
        elif measured < records:
            parts.append(
                f"{prefix}; the displayed total sums only those metered records and is incomplete"
            )
        else:
            parts.append(f"{prefix}; the displayed total covers all producer records")
        totals = indexing["totals"]
        supplier_total = totals["supplier_reported_total_tokens"]
        reasoning = totals["reasoning_tokens"]
        if supplier_total["value"] != "unavailable":
            if reasoning["value"] == "unavailable":
                reasoning_parts.append(
                    f"{item['provider_id']} reasoning breakdown is unavailable, not zero; "
                    f"{_number_text(supplier_total['value'])} is the measured supplier total, "
                    "with no inferred reasoning added, so the measurement remains partial"
                )
            else:
                reasoning_parts.append(
                    f"{item['provider_id']} reasoning: {_number_text(reasoning['value'])}, "
                    "already included in its total"
                )
    return (
        '<p class="muted"><strong>Indexing token definition:</strong> Indexing tokens are '
        "supplier-reported total tokens for successful logical producer records "
        "(provider-defined input plus output). Reasoning is included once only when it is "
        "inside that supplier total; a reasoning subset is never added again. If reasoning "
        "is unavailable, the report does not infer or add it. Embedding and failed physical "
        f"attempts are excluded. Meter coverage: {_escape('; '.join(parts))}. "
        f"Reasoning coverage: {_escape('; '.join(reasoning_parts))}.</p>"
    )


def _dataset_notice(details: Mapping[str, object]) -> str:
    if details.get("status") != "verified":
        return (
            '<p class="muted">Question text, gold answers, and source sessions are absent '
            "because no verified dataset source was supplied.</p>"
        )
    return (
        '<details class="dataset-notice"><summary>Personal/offline detail report · '
        "Dataset provenance</summary>"
        "<p>This content-bearing artifact is local-only under the frozen dataset policy. "
        f"Source: {_escape(details['source_id'])}; revision: "
        f"<code>{_escape(details['revision'])}</code>; source SHA-256: "
        f"<code>{_escape(details['source_sha256'])}</code>; license metadata: "
        f"{_escape(details['license_id'])}; payload policy: "
        f"{_escape(details['payload_policy'])}. Citation: "
        f"{_escape(details['citation'])}.</p></details>"
    )


def _question_matrix_rows(
    questions: tuple[Mapping[str, Any], ...],
    cells: tuple[Mapping[str, Any], ...],
) -> str:
    provider_ids = tuple(str(item["provider_id"]) for item in cells)
    rows: list[str] = []
    for question in questions:
        by_provider = {str(item["provider_id"]): item for item in question["provider_results"]}
        label = question.get("raw_question_id", f"Case {question['ordinal_1_indexed']}")
        question_type = question.get("question_type", "details absent")
        states = "".join(_state_badge(by_provider[provider_id]) for provider_id in provider_ids)
        rows.append(f"<tr><td>{_escape(label)}</td><td>{_escape(question_type)}</td>{states}</tr>")
    return "".join(rows)


def _state_badge(result: Mapping[str, Any]) -> str:
    state = str(result["display_state"])
    return f'<td><span class="badge {state}">{_escape(state)}</span></td>'


def _question_detail_html(questions: tuple[Mapping[str, Any], ...]) -> str:
    if not questions or "question" not in questions[0]:
        return ""
    return "".join(_one_question_detail(question) for question in questions)


def _one_question_detail(question: Mapping[str, Any]) -> str:
    mismatch = (
        '<p class="muted">Dataset note: turn-level has_answer labels differ from '
        "answer_session_ids; sessions shown follow answer_session_ids.</p>"
        if question.get("has_answer_label_mismatch") is True
        else ""
    )
    sessions = "".join(_answer_session_html(item) for item in question["answer_sessions"])
    providers = "".join(_provider_detail_html(item) for item in question["provider_results"])
    return (
        "<details>"
        f"<summary>{_escape(question['raw_question_id'])} — "
        f"{_escape(question['question_type'])}</summary>"
        f"<h3>Question</h3><pre>{_escape(question['question'])}</pre>"
        f"<h3>Gold answer</h3><pre>{_escape(question['gold_answer'])}</pre>"
        f"{mismatch}<h3>Original answer sessions</h3>{sessions}"
        f"<h3>Provider evidence</h3>{providers}</details>"
    )


def _answer_session_html(session: Mapping[str, Any]) -> str:
    messages = "".join(
        "<p>"
        f"<strong>{_escape(message['role'])}</strong> "
        f'<span class="muted">has_answer={_escape(message["has_answer"])}</span>'
        f"</p><pre>{_escape(message['content'])}</pre>"
        for message in session["messages"]
    )
    return (
        '<details class="session"><summary>Original answer session '
        f"{_escape(session['session_id'])}</summary>"
        f'<p class="muted">{_escape(session["timestamp"])}</p>{messages}</details>'
    )


def _provider_detail_html(result: Mapping[str, Any]) -> str:
    heading = (
        '<details class="provider-detail"><summary>'
        f"{_escape(result['provider_id'])}: {_escape(result['display_state'])}</summary>"
    )
    reason = result.get("detail_unavailable_reason")
    if reason is not None:
        return f"{heading}<p>{_escape(reason)}</p></details>"
    return (
        f"{heading}<p>Judge decision: <strong>{_escape(result['judge_decision'])}</strong>; "
        f"Ctx tokens: {_escape(result['context_tokens'])}</p>"
        f"<h4>Model answer</h4><pre>{_escape(result['model_answer'])}</pre>"
        "<details><summary>Injected context</summary>"
        f"<pre>{_escape(result['injected_context'])}</pre></details></details>"
    )


def _time_text(value: object) -> str:
    if value == "unavailable":
        return "unavailable"
    decimal = (Decimal(str(value)) / Decimal(1_000_000)).quantize(
        Decimal("0.001"),
        rounding=ROUND_HALF_UP,
    )
    return f"{format(decimal, 'f')} s"


def _request_time_text(value: Mapping[str, object]) -> str:
    coverage = f"n={value['count']} ({value['status']})"
    if value["status"] == "unavailable":
        return f"unavailable; {coverage}"
    return (
        f"median {_time_text(value['median_microseconds'])}; "
        f"p95 {_time_text(value['p95_microseconds'])}; "
        f"max {_time_text(value['maximum_microseconds'])}; {coverage}"
    )


def _delta_text(value: object) -> str:
    if not isinstance(value, dict):
        return "unavailable"
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if (
        isinstance(numerator, bool)
        or not isinstance(numerator, int)
        or isinstance(denominator, bool)
        or not isinstance(denominator, int)
        or denominator <= 0
    ):
        return "unavailable"
    points = (Decimal(numerator) * Decimal(100) / Decimal(denominator)).quantize(
        Decimal("0.1"),
        rounding=ROUND_HALF_UP,
    )
    sign = "+" if points > 0 else ""
    return f"{sign}{format(points, 'f')} pp"


def _failure_retry_text(value: Mapping[str, object]) -> str:
    return (
        f"failed={value['failed_count']}, retry={value['retry_count']}, "
        f"cancelled={value['cancelled_count']}, unknown={value['unknown_outcome_count']}"
    )


def _cost_text(value: Mapping[str, Any]) -> str:
    totals = value["actual_supplier_charge"]
    rendered = (
        ", ".join(f"{currency} {amount}" for currency, amount in totals.items())
        if isinstance(totals, Mapping)
        else "unavailable"
    )
    return f"{rendered}; billing={value['billing_coverage']['status']}"


def _escape(value: object) -> str:
    visible = _BIDI_CONTROL_PATTERN.sub(
        lambda match: f"\\u{ord(match.group(0)):04x}",
        str(value),
    )
    return html.escape(visible)


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
