"""Report assembly for separately preserved provider result snapshots."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oamb.artifacts.atomic import read_regular_file
from oamb.config.doctor import ResolvedPlan, ResolvedPlanError, load_resolved_plan_for_run
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.reporting.comparison_project import (
    ComparisonProjectBuildResult,
    ComparisonProjectError,
    build_question_results_comparison_project,
)
from oamb.reporting.report_analysis import build_report_analysis_generator
from oamb.runtime.question_results import (
    RESULT_PROVIDER_IDS,
    QuestionResult,
    load_question_results,
)
from oamb.workloads.longmemeval import build_longmemeval_bundle


class SavedResultsError(ValueError):
    """Saved provider results do not form one truthful offline report input."""


_RESOLVED_PLAN_HASH_DOMAIN = "oamb-resolved-plan-initial-v1"


@dataclass(frozen=True, slots=True)
class _SavedProviderInput:
    provider_id: str
    snapshot_name: str
    plan_path: Path
    result_path: Path
    plan_bytes: bytes
    plan_document: dict[str, Any]
    loaded_plan: ResolvedPlan | None


def build_saved_results_report(
    *,
    result_root: Path,
    output_root: Path,
    dataset_source: Path | None = None,
    analysis_model_env: Path | None = None,
    analysis_cache_root: Path | None = None,
) -> ComparisonProjectBuildResult:
    """Build one report from exactly three preserved provider snapshots."""

    sources = _discover_saved_provider_inputs(Path(result_root))
    reference = _select_reference_plan(sources)
    reference_document = reference.plan_document
    plan = reference.loaded_plan
    if plan is None:  # pragma: no cover - guarded by _select_reference_plan
        raise SavedResultsError("saved results do not contain a current readable resolved plan")
    _require_shared_report_policy(sources, reference_document)
    if analysis_model_env is None and analysis_cache_root is not None:
        raise SavedResultsError("analysis cache requires an analysis model environment")
    analysis_generator = (
        build_report_analysis_generator(
            plan=plan,
            model_env_path=Path(analysis_model_env),
            cache_root=(
                Path(analysis_cache_root)
                if analysis_cache_root is not None
                else Path(output_root).parent / "report-analysis-cache"
            ),
        )
        if analysis_model_env is not None
        else None
    )

    source_path = Path(dataset_source) if dataset_source is not None else Path(plan.dataset.path)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    try:
        manifest = build_longmemeval_bundle(source_path, plan.dataset.selection).case_manifest
    except Exception as exc:
        raise SavedResultsError("saved result dataset cannot be loaded") from exc
    if (
        manifest.manifest_hash != plan.dataset.case_manifest_hash
        or manifest.workload_id != plan.dataset.workload_id
    ):
        raise SavedResultsError("saved result dataset differs from the resolved plans")
    ordered_question_ids = tuple(item.raw_question_id for item in manifest.cases)

    by_provider = {source.provider_id: source for source in sources}
    results_by_cell: dict[str, Mapping[str, QuestionResult]] = {}
    metadata: list[dict[str, object]] = []
    top_k_by_provider: dict[str, object] = {}
    reference_execution = reference_document.get("execution")
    for cell in plan.cells:
        source = by_provider[cell.provider_id]
        try:
            results = load_question_results(
                source.result_path,
                ordered_question_ids=ordered_question_ids,
            )
        except Exception as exc:
            raise SavedResultsError(
                f"saved result cannot parse or validate: {source.provider_id}"
            ) from exc
        results_by_cell[cell.cell_id] = results
        source_cell = _provider_cell(source.plan_document, source.provider_id)
        retrieval = _mapping(source.plan_document.get("retrieval"), "saved retrieval")
        top_k = retrieval.get("top_k", "unavailable")
        top_k_by_provider[source.provider_id] = top_k
        metadata.append(
            {
                "provider_id": source.provider_id,
                "snapshot_name": source.snapshot_name,
                "resolved_plan_hash": _sha256_text(
                    source.plan_document.get("resolved_plan_hash"),
                    "saved resolved plan hash",
                ),
                "resolved_plan_file_sha256": hashlib.sha256(source.plan_bytes).hexdigest(),
                "result_sha256": hashlib.sha256(read_regular_file(source.result_path)).hexdigest(),
                "cell_spec_hash": _sha256_text(
                    source_cell.get("cell_spec_hash"),
                    "saved cell spec hash",
                ),
                "retrieval_top_k": top_k,
                "execution_matches_reference": source.plan_document.get("execution")
                == reference_execution,
            }
        )

    pair_limitations: dict[frozenset[str], tuple[str, ...]] = {}
    for left_index, left in enumerate(plan.cells):
        for right in plan.cells[left_index + 1 :]:
            if top_k_by_provider[left.provider_id] != top_k_by_provider[right.provider_id]:
                pair_limitations[frozenset((left.provider_id, right.provider_id))] = (
                    "saved source plans do not prove one shared retrieval top_k candidate ceiling",
                )

    try:
        return build_question_results_comparison_project(
            plan,
            results_by_cell,
            case_manifest=manifest,
            output_root=Path(output_root),
            dataset_source=source_path,
            saved_result_sources=tuple(metadata),
            pair_limitations=pair_limitations,
            analysis_generator=analysis_generator,
        )
    except ComparisonProjectError as exc:
        raise SavedResultsError(str(exc)) from exc


def _discover_saved_provider_inputs(result_root: Path) -> tuple[_SavedProviderInput, ...]:
    if result_root.is_symlink() or not result_root.is_dir():
        raise SavedResultsError("saved result directory must be a regular directory")
    discovered: dict[str, _SavedProviderInput] = {}
    try:
        snapshots = tuple(sorted(result_root.iterdir(), key=lambda item: item.name))
    except OSError as exc:
        raise SavedResultsError("saved result directory cannot be read") from exc
    for snapshot in snapshots:
        if snapshot.is_symlink() or not snapshot.is_dir():
            continue
        results_root = snapshot / "results"
        plan_path = snapshot / "resolved-plan.json"
        if not results_root.is_dir() or results_root.is_symlink() or not plan_path.exists():
            continue
        candidates = tuple(
            results_root / f"{provider_id}.json"
            for provider_id in RESULT_PROVIDER_IDS
            if (results_root / f"{provider_id}.json").exists()
        )
        if len(candidates) != 1:
            raise SavedResultsError(
                f"saved snapshot must contain exactly one provider result: {snapshot.name}"
            )
        provider_id = candidates[0].stem
        if provider_id in discovered:
            raise SavedResultsError(f"duplicate saved result provider: {provider_id}")
        plan_bytes, plan_document = _load_plan_document(plan_path)
        try:
            loaded_plan = load_resolved_plan_for_run(plan_path)
        except ResolvedPlanError:
            loaded_plan = None
        discovered[provider_id] = _SavedProviderInput(
            provider_id=provider_id,
            snapshot_name=snapshot.name,
            plan_path=plan_path,
            result_path=candidates[0],
            plan_bytes=plan_bytes,
            plan_document=plan_document,
            loaded_plan=loaded_plan,
        )
    missing = tuple(provider for provider in RESULT_PROVIDER_IDS if provider not in discovered)
    if missing:
        raise SavedResultsError(f"missing saved result providers: {', '.join(missing)}")
    return tuple(discovered[provider] for provider in RESULT_PROVIDER_IDS)


def _select_reference_plan(
    sources: tuple[_SavedProviderInput, ...],
) -> _SavedProviderInput:
    readable = tuple(source for source in sources if source.loaded_plan is not None)
    if not readable:
        raise SavedResultsError("saved results do not contain a current readable resolved plan")
    counts = Counter(str(source.plan_document.get("resolved_plan_hash")) for source in readable)
    selected_hash = min(counts, key=lambda value: (-counts[value], value))
    return min(
        (
            source
            for source in readable
            if source.plan_document.get("resolved_plan_hash") == selected_hash
        ),
        key=lambda source: source.snapshot_name,
    )


def _require_shared_report_policy(
    sources: tuple[_SavedProviderInput, ...],
    reference: Mapping[str, object],
) -> None:
    for source in sources:
        document = source.plan_document
        for key in ("comparison_id", "dataset", "decision", "model_roles"):
            if document.get(key) != reference.get(key):
                raise SavedResultsError(f"saved provider plans differ in report policy: {key}")
        retrieval = _mapping(document.get("retrieval"), "saved retrieval")
        reference_retrieval = _mapping(reference.get("retrieval"), "reference retrieval")
        if retrieval.get("generation") != reference_retrieval.get("generation"):
            raise SavedResultsError("saved provider plans differ in retrieval generation policy")
        binding = _provider_retrieval_binding(document, source.provider_id)
        reference_binding = _provider_retrieval_binding(reference, source.provider_id)
        if binding != reference_binding:
            raise SavedResultsError(
                f"saved provider retrieval binding differs: {source.provider_id}"
            )
        cell = dict(_provider_cell(document, source.provider_id))
        reference_cell = dict(_provider_cell(reference, source.provider_id))
        for ignored in ("cell_spec_hash", "execution_hash"):
            cell.pop(ignored, None)
            reference_cell.pop(ignored, None)
        if cell != reference_cell:
            raise SavedResultsError(f"saved provider cell binding differs: {source.provider_id}")


def _load_plan_document(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        content = read_regular_file(path)
        document = json.loads(
            content,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except Exception as exc:
        raise SavedResultsError("saved resolved plan cannot parse") from exc
    if not isinstance(document, dict) or canonical_json_bytes(document) != content:
        raise SavedResultsError("saved resolved plan must be canonical JSON")
    if document.get("schema_name") != "resolved_plan" or document.get("schema_version") != 1:
        raise SavedResultsError("saved resolved plan schema is unsupported")
    plan_hash = _sha256_text(document.get("resolved_plan_hash"), "saved resolved plan hash")
    payload = dict(document)
    payload.pop("resolved_plan_hash")
    if plan_hash != canonical_sha256([_RESOLVED_PLAN_HASH_DOMAIN, payload]):
        raise SavedResultsError("saved resolved plan hash does not match its content")
    return content, document


def _provider_cell(document: Mapping[str, object], provider_id: str) -> Mapping[str, object]:
    cells = document.get("cells")
    if not isinstance(cells, list):
        raise SavedResultsError("saved resolved plan cell inventory is invalid")
    matches = tuple(
        item for item in cells if isinstance(item, dict) and item.get("provider_id") == provider_id
    )
    if len(matches) != 1:
        raise SavedResultsError(f"saved resolved plan does not contain provider: {provider_id}")
    return matches[0]


def _provider_retrieval_binding(
    document: Mapping[str, object], provider_id: str
) -> Mapping[str, object]:
    retrieval = _mapping(document.get("retrieval"), "saved retrieval")
    bindings = retrieval.get("bindings")
    if not isinstance(bindings, list):
        raise SavedResultsError("saved retrieval binding inventory is invalid")
    matches = tuple(
        item
        for item in bindings
        if isinstance(item, dict) and item.get("provider_id") == provider_id
    )
    if len(matches) != 1:
        raise SavedResultsError(f"saved retrieval binding is missing: {provider_id}")
    return matches[0]


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SavedResultsError(f"{label} is invalid")
    return value


def _sha256_text(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SavedResultsError(f"{label} is invalid")
    return value


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key: {key}")
        document[key] = value
    return document


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"invalid JSON constant: {value}")


__all__ = ["SavedResultsError", "build_saved_results_report"]
