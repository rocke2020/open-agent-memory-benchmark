#!/usr/bin/env python3
"""Transfer retained results into an already-prechecked new-plan checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from oamb.config.benchmark import MODEL_PROFILE_ENVIRONMENT_KEYS, load_benchmark_configuration
from oamb.config.doctor import build_resolved_plan, resolved_plan_bytes
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.live import (
    load_live_environment,
    validate_live_readiness_receipt,
)
from oamb.runtime.provider_env import load_t10_provider_environment
from oamb.runtime.question_results import load_question_results, remaining_question_ids
from oamb.workloads.longmemeval import build_longmemeval_bundle

_PROVIDERS = ("hindsight", "mem0", "openviking")
_PROGRESS_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "resolved_plan_hash",
        "cell_id",
        "provider_id",
        "workload_id",
        "case_manifest_hash",
        "ordered_question_ids",
        "results",
    }
)
_PROGRESS_RESULT_SCHEMA_NAMES = {
    "full_progress_entry": "question_result",
    "progress_answer": "result_answer",
    "progress_attempt_counts": "result_attempt_counts",
    "progress_evaluation": "result_evaluation",
    "progress_ingestion": "result_ingestion",
    "progress_measurement": "result_measurement",
    "progress_retrieval": "result_retrieval",
    "progress_text": "result_text",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"required migration input is not a regular file: {path}")
    return path.read_bytes()


def _load_canonical_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    content = _read_regular(path)
    try:
        document = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"migration input is not valid JSON: {path}") from exc
    if not isinstance(document, dict) or canonical_json_bytes(document) != content:
        raise ValueError(f"migration input is not a canonical JSON object: {path}")
    return content, document


def _load_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    content = _read_regular(path)
    try:
        document = json.loads(content)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"migration input is not valid JSON: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"migration input is not a JSON object: {path}")
    return content, document


def _resolve_inside(root: Path, selected: str, label: str) -> Path:
    candidate = Path(selected)
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"{label} is outside the selected repository")
    return resolved


def _validate_label(label: str) -> None:
    if (
        not label
        or label in {".", ".."}
        or any(
            character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
            for character in label
        )
    ):
        raise ValueError("target migration label is invalid")


def _validate_legacy_plan(
    legacy_plan: Mapping[str, Any],
    current_plan: Mapping[str, Any],
) -> None:
    if legacy_plan.get("schema_name") != "resolved_plan" or legacy_plan.get("schema_version") != 1:
        raise ValueError("legacy resolved plan schema is invalid")
    legacy_payload = dict(legacy_plan)
    legacy_hash = legacy_payload.pop("resolved_plan_hash", None)
    if legacy_hash != canonical_sha256(["oamb-resolved-plan-initial-v1", legacy_payload]):
        raise ValueError("legacy resolved plan hash does not match its content")
    if "embedding_endpoint" in legacy_payload:
        raise ValueError("source resolved plan is not the legacy managed-local format")

    for field in ("comparison_id", "dataset", "retrieval", "execution", "decision"):
        if legacy_plan.get(field) != current_plan.get(field):
            raise ValueError(f"legacy and current resolved plans differ at {field}")

    legacy_roles = legacy_plan.get("model_roles")
    current_roles = current_plan.get("model_roles")
    if not isinstance(legacy_roles, list) or not isinstance(current_roles, list):
        raise ValueError("resolved model roles are invalid")
    legacy_by_id = {item.get("role_id"): item for item in legacy_roles if isinstance(item, dict)}
    current_by_id = {item.get("role_id"): item for item in current_roles if isinstance(item, dict)}
    if set(legacy_by_id) != set(current_by_id):
        raise ValueError("legacy and current model role inventories differ")
    for role_id in legacy_by_id:
        legacy_role = dict(legacy_by_id[role_id])
        current_role = dict(current_by_id[role_id])
        ignored = {"binding_hash"}
        if role_id == "embedding":
            ignored.update({"credential_variable", "recipient"})
        for field in ignored:
            legacy_role.pop(field, None)
            current_role.pop(field, None)
        if legacy_role != current_role:
            raise ValueError(f"legacy and current model role differ: {role_id}")

    legacy_cells = legacy_plan.get("cells")
    current_cells = current_plan.get("cells")
    if not isinstance(legacy_cells, list) or not isinstance(current_cells, list):
        raise ValueError("resolved cells are invalid")
    if len(legacy_cells) != len(current_cells):
        raise ValueError("legacy and current cell inventories differ")
    for legacy_cell, current_cell in zip(legacy_cells, current_cells, strict=True):
        if not isinstance(legacy_cell, dict) or not isinstance(current_cell, dict):
            raise ValueError("resolved cell entry is invalid")
        legacy_identity = dict(legacy_cell)
        current_identity = dict(current_cell)
        for field in ("cell_spec_hash", "model_role_binding_hashes"):
            legacy_identity.pop(field, None)
            current_identity.pop(field, None)
        if legacy_identity != current_identity:
            raise ValueError("legacy and current resolved cell identities differ")


def _normalize_progress_result(value: object) -> object:
    if isinstance(value, dict):
        normalized = {key: _normalize_progress_result(item) for key, item in value.items()}
        schema_name = normalized.get("schema_name")
        if isinstance(schema_name, str) and schema_name in _PROGRESS_RESULT_SCHEMA_NAMES:
            normalized["schema_name"] = _PROGRESS_RESULT_SCHEMA_NAMES[schema_name]
        return normalized
    if isinstance(value, list):
        return [_normalize_progress_result(item) for item in value]
    return value


def _validate_progress(
    *,
    document: Mapping[str, Any],
    provider: str,
    cell_id: str,
    legacy_plan_hash: str,
    workload_id: str,
    case_manifest_hash: str,
    ordered_question_ids: tuple[str, ...],
    ordinary_results: Mapping[str, Any],
) -> None:
    if frozenset(document) != _PROGRESS_KEYS:
        raise ValueError(f"legacy progress keys are invalid: {provider}")
    if (
        document["schema_name"] != "full_progress"
        or document["schema_version"] != 1
        or document["resolved_plan_hash"] != legacy_plan_hash
        or document["cell_id"] != cell_id
        or document["provider_id"] != provider
        or document["workload_id"] != workload_id
        or document["case_manifest_hash"] != case_manifest_hash
        or document["ordered_question_ids"] != list(ordered_question_ids)
    ):
        raise ValueError(f"legacy progress identity is invalid: {provider}")
    progress_results = document["results"]
    if not isinstance(progress_results, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("question_id"), str)
        for item in progress_results
    ):
        raise ValueError(f"legacy progress results are invalid: {provider}")
    progress_by_id = {item["question_id"]: item for item in progress_results}
    if len(progress_by_id) != len(progress_results) or tuple(progress_by_id) != tuple(
        question_id for question_id in ordered_question_ids if question_id in progress_by_id
    ):
        raise ValueError(f"legacy progress result order is invalid: {provider}")
    if set(progress_by_id) != set(ordinary_results):
        raise ValueError(f"progress and ordinary result IDs differ: {provider}")
    normalized = {
        question_id: _normalize_progress_result(progress_by_id[question_id])
        for question_id in progress_by_id
    }
    if normalized != ordinary_results:
        raise ValueError(f"progress and ordinary result content differs: {provider}")


def _write_create_only(path: Path, content: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _replace_pointer(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.migration-{os.getpid()}")
    _write_create_only(temporary, content)
    os.replace(temporary, path)


def migrate_resume_data(
    *,
    source_repository: Path,
    target_repository: Path,
    benchmark_config: Path,
    dataset_source: Path,
    target_label: str,
    model_environment: Mapping[str, str],
) -> dict[str, Any]:
    """Transfer one selected legacy resume into the current create-only format."""

    _validate_label(target_label)
    source_repository = source_repository.resolve()
    target_repository = target_repository.resolve()
    source_state_path = source_repository / "outputs" / "tmp" / "quick-start-current.json"
    source_selector_path = source_repository / "outputs" / "tmp" / "full-test-current"
    _source_state_bytes, source_state = _load_json_object(source_state_path)
    source_label = _read_regular(source_selector_path).decode("utf-8").removesuffix("\n")
    _validate_label(source_label)
    source_precheck_plan = _resolve_inside(
        source_repository, source_state.get("resolved_plan", ""), "legacy precheck plan"
    )
    source_full_root = source_repository / "outputs" / "full-test" / source_label
    source_full_plan = source_full_root / "resolved-plan.json"
    legacy_plan_bytes, legacy_plan = _load_canonical_object(source_precheck_plan)
    if _read_regular(source_full_plan) != legacy_plan_bytes:
        raise ValueError("legacy precheck and full-run plans differ")

    dataset_bytes = _read_regular(dataset_source)
    state_dataset = _resolve_inside(
        source_repository, source_state.get("dataset_source", ""), "legacy dataset source"
    )
    if _sha256(_read_regular(state_dataset)) != _sha256(dataset_bytes):
        raise ValueError("legacy and selected dataset sources differ")

    configuration = load_benchmark_configuration(
        benchmark_config,
        model_environment=model_environment,
    )
    current_plan = build_resolved_plan(configuration)
    current_plan_bytes = resolved_plan_bytes(current_plan)
    current_plan_document = json.loads(current_plan_bytes)
    _validate_legacy_plan(legacy_plan, current_plan_document)
    if _sha256(dataset_bytes) != current_plan.dataset.source_sha256:
        raise ValueError("selected dataset source hash differs from the current plan")
    case_manifest = build_longmemeval_bundle(dataset_source, "lme60").case_manifest
    ordered_question_ids = tuple(item.raw_question_id for item in case_manifest.cases)
    if case_manifest.manifest_hash != current_plan.dataset.case_manifest_hash:
        raise ValueError("selected dataset manifest differs from the current plan")

    legacy_plan_hash = legacy_plan["resolved_plan_hash"]
    if not isinstance(legacy_plan_hash, str):
        raise ValueError("legacy resolved plan hash is invalid")
    result_bytes_by_provider: dict[str, bytes] = {}
    completed_counts: dict[str, int] = {}
    remaining_counts: dict[str, int] = {}
    source_result_hashes: dict[str, str] = {}
    source_progress_hashes: dict[str, str] = {}
    cell_by_provider = {cell.provider_id: cell.cell_id for cell in current_plan.cells}
    for provider in _PROVIDERS:
        result_path = source_full_root / "results" / f"{provider}.json"
        result_bytes = _read_regular(result_path)
        results = load_question_results(
            result_path,
            ordered_question_ids=ordered_question_ids,
        )
        progress_path = source_full_root / "results" / f"progress-{provider}.json"
        progress_bytes, progress = _load_canonical_object(progress_path)
        ordinary_document = json.loads(result_bytes)
        _validate_progress(
            document=progress,
            provider=provider,
            cell_id=cell_by_provider[provider],
            legacy_plan_hash=legacy_plan_hash,
            workload_id=current_plan.dataset.workload_id,
            case_manifest_hash=current_plan.dataset.case_manifest_hash,
            ordered_question_ids=ordered_question_ids,
            ordinary_results=ordinary_document,
        )
        result_bytes_by_provider[provider] = result_bytes
        completed_counts[provider] = len(results)
        remaining_counts[provider] = len(remaining_question_ids(results, ordered_question_ids))
        source_result_hashes[provider] = _sha256(result_bytes)
        source_progress_hashes[provider] = _sha256(progress_bytes)

    target_outputs = target_repository / "outputs"
    target_tmp = target_outputs / "tmp"
    target_full_root = target_outputs / "full-test" / target_label
    if target_full_root.exists() or target_full_root.is_symlink():
        raise FileExistsError(f"migration target already exists: {target_full_root}")

    target_state_path = target_tmp / "quick-start-current.json"
    target_state_bytes, target_state = _load_json_object(target_state_path)
    target_precheck_label = target_state.get("run_label")
    if not isinstance(target_precheck_label, str):
        raise ValueError("target precheck run label is invalid")
    _validate_label(target_precheck_label)
    target_work_dir = _resolve_inside(
        target_repository,
        target_state.get("work_dir", ""),
        "target precheck work directory",
    )
    if target_work_dir != target_tmp / "precheck" / target_precheck_label:
        raise ValueError("target precheck work directory is invalid")
    target_plan_path = _resolve_inside(
        target_repository,
        target_state.get("resolved_plan", ""),
        "target precheck plan",
    )
    if target_plan_path != target_work_dir / "plan" / "resolved-plan.json":
        raise ValueError("target precheck plan path is invalid")
    if _read_regular(target_plan_path) != current_plan_bytes:
        raise ValueError("target precheck plan differs from the current resolved plan")
    target_dataset = _resolve_inside(
        target_repository,
        target_state.get("dataset_source", ""),
        "target dataset source",
    )
    if _sha256(_read_regular(target_dataset)) != current_plan.dataset.source_sha256:
        raise ValueError("target dataset source differs from the current resolved plan")
    if target_state.get("embedding_ownership") != "embedding_local_fallback":
        raise ValueError("target precheck embedding ownership is invalid")
    try:
        target_environment = load_live_environment(
            plan=current_plan,
            provider_env_path=target_repository / ".env",
            model_env_path=target_repository / ".env",
            provider_runtime_directory=target_repository / "provider-services" / ".runtime",
            base_environment=os.environ,
        )
        validate_live_readiness_receipt(
            plan=current_plan,
            provider_runtime_directory=target_repository / "provider-services" / ".runtime",
            environment=target_environment,
        )
    except (OSError, ValueError) as exc:
        raise ValueError("target new-plan readiness is missing or invalid") from exc

    target_tmp.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".managed-local-migration-", dir=target_tmp))
    staged_full = staging / "full"
    receipt: dict[str, Any] = {
        "schema_name": "managed_local_resume_migration",
        "schema_version": 1,
        "source_plan_file_sha256": _sha256(legacy_plan_bytes),
        "source_resolved_plan_hash": legacy_plan_hash,
        "target_resolved_plan_hash": current_plan.resolved_plan_hash,
        "source_result_sha256": source_result_hashes,
        "source_progress_sha256": source_progress_hashes,
        "completed_result_counts": completed_counts,
        "remaining_result_counts": remaining_counts,
        "source_full_run_preserved_in_place": True,
        "target_precheck_state_sha256": _sha256(target_state_bytes),
        "external_calls_dispatched": 0,
    }
    try:
        _write_create_only(staged_full / "resolved-plan.json", current_plan_bytes)
        for provider, content in result_bytes_by_provider.items():
            _write_create_only(staged_full / "results" / f"{provider}.json", content)
        _write_create_only(
            staged_full / "migration-receipt.json",
            canonical_json_bytes(receipt),
        )
        for provider in _PROVIDERS:
            staged_result = staged_full / "results" / f"{provider}.json"
            if _sha256(_read_regular(staged_result)) != source_result_hashes[provider]:
                raise ValueError(f"migrated result hash differs: {provider}")
            rehearsed = load_question_results(
                staged_result,
                ordered_question_ids=ordered_question_ids,
            )
            if (
                len(remaining_question_ids(rehearsed, ordered_question_ids))
                != remaining_counts[provider]
            ):
                raise ValueError(f"zero-dispatch rehearsal differs: {provider}")

        target_full_root.parent.mkdir(parents=True, exist_ok=True)
        staged_full.rename(target_full_root)
        staging.rmdir()
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if _read_regular(target_full_root / "resolved-plan.json") != _read_regular(target_plan_path):
        raise ValueError("published migration plans differ")
    _replace_pointer(
        target_tmp / "full-test-current",
        f"{target_label}\n".encode(),
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer one retained managed-local resume into a target whose current "
            "precheck and readiness evidence already bind the new plan."
        )
    )
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument(
        "--target-repository",
        type=Path,
        required=True,
        help="Target checkout after a successful new-plan precheck.",
    )
    parser.add_argument("--benchmark-config", type=Path, required=True)
    parser.add_argument("--dataset-source", type=Path, required=True)
    parser.add_argument("--target-label", required=True)
    arguments = parser.parse_args()
    model_environment = load_t10_provider_environment(
        arguments.target_repository / ".env",
        expected_keys=MODEL_PROFILE_ENVIRONMENT_KEYS,
    )
    receipt = migrate_resume_data(
        source_repository=arguments.source_repository,
        target_repository=arguments.target_repository,
        benchmark_config=arguments.benchmark_config,
        dataset_source=arguments.dataset_source,
        target_label=arguments.target_label,
        model_environment=model_environment,
    )
    print(
        "migration: PASS "
        f"completed={receipt['completed_result_counts']} "
        f"remaining={receipt['remaining_result_counts']} "
        "external_calls=0"
    )


if __name__ == "__main__":
    main()
