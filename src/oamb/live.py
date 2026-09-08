"""Generic live LongMemEval cell composition from one frozen resolved plan."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import socket
import stat
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any, Literal

from oamb.artifacts.atomic import read_regular_file
from oamb.config.benchmark import MODEL_ROLE_IDS, ModelRoleId
from oamb.config.doctor import CellSpec, ModelExecutionBinding, ResolvedPlan
from oamb.config.provider_services import (
    ProviderServiceBindingError,
    load_provider_service_bindings,
    validate_provider_service_receipt,
)
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    BindingKind,
    BudgetScopeKindV3,
    BudgetSpecV4,
    CaseManifest,
    DatasetManifest,
    DispatchBudgetOwnerKind,
    DispatchBudgetRoute,
    ExecutionOwner,
    ModelRole,
    ModelRoleBindingV2,
    ProviderBudgetCap,
    ProviderOperationBudgetCeiling,
    ResourceBudgetCeiling,
    RoleBindingStatus,
    RoleBudgetCeiling,
    RunPreflightRecord,
    RunSpec,
    SourceEvidenceBinding,
    SourceEvidenceKind,
    budget_spec_v4_hash,
    dispatch_budget_route_hash,
    provider_operation_budget_ceiling_hash,
    run_preflight_record_hash,
)
from oamb.runtime.full_progress import (
    FullProgress,
    canonical_full_progress_path,
    load_full_progress,
)
from oamb.runtime.native_run import NativeRunArtifacts, NativeRunControl
from oamb.runtime.preflight import (
    CONTROLLED_EMBEDDING_DIMENSION,
    CONTROLLED_EMBEDDING_MODEL,
    ControlledEmbeddingDescriptor,
)
from oamb.runtime.provider_env import load_t10_provider_environment
from oamb.workloads.longmemeval import (
    LME6_EXPECTED_QUESTION_IDS,
    LME6_EXPECTED_SESSION_COUNT,
    LME30_EXPECTED_QUESTION_IDS,
    LME30_EXPECTED_SESSION_COUNT,
    LME60_EXPECTED_QUESTION_IDS,
    LME60_EXPECTED_SESSION_COUNT,
    LME_DATASET_MANIFEST_HASH,
    LongMemEvalWorkload,
    build_longmemeval_bundle,
)
from oamb.workloads.metrics import (
    LME_ANSWER_MAX_OUTPUT_TOKENS,
    LME_JUDGE_MAX_OUTPUT_TOKENS,
)
from oamb.workloads.visible_evidence import LME_VISIBLE_EVIDENCE_POLICY

_MEMORY_STAGES = (
    "runtime_resolve",
    "scope_allocate",
    "memory_ingest",
    "memory_readiness",
    "memory_projection",
    "pre_query_projection",
    "memory_query",
    "post_query_projection",
)
_MEMORY_SYSTEM_BY_PROVIDER = {
    "hindsight": "hindsight",
    "mem0": "mem0",
    "openviking": "openviking",
}
_EXTRA_ENVIRONMENT_BY_PROVIDER = {
    "hindsight": (),
    "mem0": ("OAMB_MEM0_INSPECTOR_BASE_URL", "OAMB_MEM0_INSPECTOR_API_KEY"),
    "openviking": ("OAMB_OPENVIKING_ACCOUNT_ID", "OAMB_OPENVIKING_ADMIN_USER_ID"),
}
_SERVICE_RECEIPT_POINTER = "service-verification-current.sha256"
_LLM_URL_TYPE_VARIABLE = "LLM_URL_TYPE"
_SUPPORTED_LLM_URL_TYPE = "openai_chat"


class LiveConfigurationError(ValueError):
    """The frozen cell cannot be safely composed before any client exists."""


class LiveCellExecutionError(RuntimeError):
    """One or more isolated provider cells failed after the shared barrier."""

    def __init__(
        self,
        message: str,
        *,
        outcomes: tuple[LiveCellOutcome, ...] = (),
        errors: tuple[BaseException, ...] = (),
    ) -> None:
        super().__init__(message)
        self.outcomes = outcomes
        self.errors = errors


@dataclass(frozen=True, slots=True)
class LiveCell:
    plan: ResolvedPlan
    cell: CellSpec
    run_id: str
    role_ids: tuple[str, ...]
    output_root: Path
    capsule_root: Path
    dataset_path: Path
    control: NativeRunControl
    environment: Mapping[str, str] = field(repr=False, compare=False)
    requested_case_manifest_entry_ids: tuple[str, ...] = ()
    progress_path: Path | None = None
    expected_progress: FullProgress | None = None


@dataclass(frozen=True, slots=True)
class LiveCellCompletion:
    cell_id: str
    capsule_root: Path


LiveCellStatus = Literal["completed", "failed", "not_started"]


@dataclass(frozen=True, slots=True)
class LiveCellOutcome:
    cell_id: str
    status: LiveCellStatus
    capsule_root: Path | None
    detail: str | None


@dataclass(frozen=True, slots=True)
class FullResumeSelection:
    progress_by_cell: Mapping[str, FullProgress]
    remaining_case_manifest_entry_ids: Mapping[str, tuple[str, ...]]
    case_manifest: CaseManifest


def load_live_environment(
    *,
    provider_env_path: Path,
    model_env_path: Path,
    provider_runtime_directory: Path,
    base_environment: Mapping[str, str],
) -> dict[str, str]:
    """Load local runtime values as inert data and derive loopback service URLs."""

    environment = dict(base_environment)
    environment.update(load_t10_provider_environment(provider_env_path))
    environment.update(
        load_t10_provider_environment(
            model_env_path,
            expected_keys=frozenset({_LLM_URL_TYPE_VARIABLE, "LLM_BASE_URL", "LLM_API_KEY"}),
        )
    )
    if environment.get(_LLM_URL_TYPE_VARIABLE) != _SUPPORTED_LLM_URL_TYPE:
        raise LiveConfigurationError("LLM_URL_TYPE must be openai_chat in OAMB v0.1.0")
    aliases = {
        "OAMB_HINDSIGHT_LLM_BASE_URL": "LLM_BASE_URL",
        "OAMB_HINDSIGHT_LLM_API_KEY": "LLM_API_KEY",
        "OAMB_MEM0_LLM_BASE_URL": "LLM_BASE_URL",
        "OAMB_MEM0_LLM_API_KEY": "LLM_API_KEY",
        "OAMB_OPENVIKING_VLM_BASE_URL": "LLM_BASE_URL",
        "OAMB_OPENVIKING_VLM_API_KEY": "LLM_API_KEY",
    }
    for target, source in aliases.items():
        if source in environment:
            environment[target] = environment[source]
    for target, port_name in (
        ("OAMB_HINDSIGHT_BASE_URL", "OAMB_HINDSIGHT_PORT"),
        ("OAMB_MEM0_BASE_URL", "OAMB_MEM0_PORT"),
        ("OAMB_MEM0_INSPECTOR_BASE_URL", "OAMB_MEM0_INSPECTOR_PORT"),
        ("OAMB_OPENVIKING_BASE_URL", "OAMB_OPENVIKING_PORT"),
    ):
        port = environment.get(port_name)
        if target not in environment and port:
            environment[target] = f"http://127.0.0.1:{port}"
    user_key_path = provider_runtime_directory / "openviking-user-key"
    if "OAMB_OPENVIKING_USER_API_KEY" not in environment and user_key_path.is_file():
        user_key = user_key_path.read_text(encoding="utf-8").strip()
        if user_key:
            environment["OAMB_OPENVIKING_USER_API_KEY"] = user_key
    return environment


def validate_lme60_mem0_input_encoding(dataset_path: Path) -> int:
    """Encode every frozen LME-60 source through the real Mem0 request boundary."""

    from oamb.memory_systems.mem0.adapter import _source_messages
    from oamb.memory_systems.mem0.wire import Mem0SourceMetadata, build_add_http_request

    bundle = build_longmemeval_bundle(dataset_path, "lme60")
    synthetic_run_id = canonical_sha256(["oamb-lme60-input-encoding-precheck-v1"])
    checked = 0
    for plan in bundle.ingestion_plans:
        for source in plan.ordered_source_units:
            request = build_add_http_request(
                messages=_source_messages(source),
                run_id=synthetic_run_id,
                metadata=Mem0SourceMetadata(
                    ingestion_occurrence_id=synthetic_run_id,
                    ingestion_plan_id=plan.ingestion_plan_id,
                    source_unit_id=source.source_unit_id,
                    source_ordinal=source.ordinal_1_indexed,
                ),
            )
            encoded_messages = json.loads(request.body)["messages"]
            if encoded_messages != json.loads(source.payload_bytes):
                raise ValueError("Mem0 input precheck changed source message values or order")
            encoded_message_bytes = json.dumps(
                encoded_messages,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
            if encoded_message_bytes != source.payload_bytes:
                raise ValueError("Mem0 input precheck changed canonical source payload bytes")
            checked += 1
    if checked != LME60_EXPECTED_SESSION_COUNT:
        raise ValueError("Mem0 input precheck did not cover every frozen LME-60 source")
    return checked


def load_live_provider_evidence(
    *,
    plan: ResolvedPlan,
    provider_runtime_directory: Path,
    environment: Mapping[str, str],
    service_receipt_path: Path | None = None,
) -> tuple[str, dict[str, SourceEvidenceBinding]]:
    """Reopen one explicitly selected provider-service proof before composition."""

    if service_receipt_path is None:
        receipt_path = resolve_service_verification_receipt(provider_runtime_directory)
    else:
        receipt_path = resolve_service_verification_receipt(
            provider_runtime_directory,
            expected_sha256=service_receipt_path.stem,
        )
        if receipt_path != service_receipt_path.absolute():
            raise LiveConfigurationError("provider verification receipt path is not canonical")
    provider_project, attestation_hash = _live_service_identity(
        provider_runtime_directory,
    )
    try:
        embedding_artifact_hash = hashlib.sha256(
            read_regular_file(provider_runtime_directory / "embedding-ready-response.json")
        ).hexdigest()
        embedding_endpoint = environment["OAMB_EMBEDDING_BASE_URL"]
    except (KeyError, OSError) as exc:
        raise LiveConfigurationError(
            "provider runtime embedding or attestation proof is missing"
        ) from exc
    embedding = ControlledEmbeddingDescriptor(
        endpoint_fingerprint=canonical_sha256(
            ["oamb-controlled-embedding-endpoint-initial-v1", embedding_endpoint]
        ),
        model=CONTROLLED_EMBEDDING_MODEL,
        artifact_fingerprint=embedding_artifact_hash,
        dimension=CONTROLLED_EMBEDDING_DIMENSION,
        input_adaptation_fingerprint=canonical_sha256(
            ["oamb-controlled-embedding-input-initial-v1", "exact-query-bytes"]
        ),
    )
    profile_ids = ("hindsight-rest-v1", "mem0-rest-v1", "openviking-rest-v1")
    bindings = load_provider_service_bindings(
        receipt_path,
        expected_project=provider_project,
        expected_project_attestation_sha256=attestation_hash,
        controlled_embeddings={profile_id: embedding for profile_id in profile_ids},
        expected_provider_models=_expected_provider_models(plan),
        expected_provider_thinking_efforts=_expected_provider_thinking_efforts(plan),
    )
    evidence_by_provider: dict[str, SourceEvidenceBinding] = {}
    for binding in bindings:
        fields = {
            "source_kind": SourceEvidenceKind.PROVIDER_SERVICE,
            "source_identity": f"{provider_project}:{binding.exact_profile.profile_id}",
            "source_root_hash": binding.profile_proof_manifest_sha256,
            "validation_result_hash": binding.receipt_sha256,
            "source_schema_versions": ("provider_service_evidence_manifest@1",),
        }
        evidence_by_provider[binding.exact_profile.memory_system_id] = SourceEvidenceBinding(
            binding_id=canonical_sha256(
                ["oamb-provider-evidence-initial-v1", provider_project, fields]
            ),
            **fields,  # type: ignore[arg-type]
        )
    return provider_project, evidence_by_provider


def validate_live_service_verification_receipt(
    *,
    plan: ResolvedPlan,
    provider_runtime_directory: Path,
    environment: Mapping[str, str],
    service_receipt_path: Path,
) -> None:
    """Validate one selected service receipt and all referenced proof bytes."""

    provider_project, attestation_hash = _live_service_identity(
        provider_runtime_directory,
    )
    try:
        validate_provider_service_receipt(
            service_receipt_path,
            expected_project=provider_project,
            expected_project_attestation_sha256=attestation_hash,
            expected_provider_models=_expected_provider_models(plan),
            expected_provider_thinking_efforts=_expected_provider_thinking_efforts(plan),
        )
    except (OSError, ProviderServiceBindingError) as exc:
        raise LiveConfigurationError(
            "selected service verification receipt or proof is malformed"
        ) from exc


def _live_service_identity(
    provider_runtime_directory: Path,
) -> tuple[str, str]:
    try:
        attestation = read_regular_file(provider_runtime_directory / "provider-project.attestation")
        project_lines = tuple(
            line.removeprefix(b"project=")
            for line in attestation.splitlines()
            if line.startswith(b"project=")
        )
        if len(project_lines) != 1:
            raise ValueError("provider project attestation has no unique project")
        provider_project = project_lines[0].decode("ascii")
        attestation_hash = hashlib.sha256(attestation).hexdigest()
    except (OSError, UnicodeError, ValueError) as exc:
        raise LiveConfigurationError(
            "provider runtime project identity or attestation is missing"
        ) from exc
    if not provider_project:
        raise LiveConfigurationError("provider runtime project identity is malformed")
    return provider_project, attestation_hash


def _expected_provider_models(plan: ResolvedPlan) -> dict[str, str]:
    return {
        "hindsight-rest-v1": _model_plan(plan, "hindsight_extraction").model,
        "mem0-rest-v1": _model_plan(plan, "mem0_extraction").model,
        "openviking-rest-v1": _model_plan(plan, "openviking_semantic_understanding").model,
    }


def _expected_provider_thinking_efforts(plan: ResolvedPlan) -> dict[str, str]:
    return {
        "hindsight-rest-v1": _model_plan(plan, "hindsight_extraction").thinking_effort,
        "mem0-rest-v1": _model_plan(plan, "mem0_extraction").thinking_effort,
        "openviking-rest-v1": _model_plan(
            plan, "openviking_semantic_understanding"
        ).thinking_effort,
    }


def _validate_live_models(
    plan: ResolvedPlan,
    environment: Mapping[str, str],
) -> None:
    model_variables: tuple[tuple[ModelRoleId, str], ...] = (
        ("hindsight_extraction", "OAMB_HINDSIGHT_LLM_MODEL"),
        ("mem0_extraction", "OAMB_MEM0_LLM_MODEL"),
        ("openviking_semantic_understanding", "OAMB_OPENVIKING_VLM_MODEL"),
        ("embedding", "OAMB_EMBEDDING_MODEL"),
    )
    for role_id, variable in model_variables:
        if environment.get(variable) != _model_plan(plan, role_id).model:
            raise LiveConfigurationError(f"{role_id} model differs from the plan")


def resolve_service_verification_receipt(
    provider_runtime_directory: Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """Resolve and content-validate one service receipt without enumerating history."""

    runtime = Path(provider_runtime_directory).absolute()
    receipts = runtime / "service-verification-receipts"
    _require_real_directory(runtime, label="provider runtime")
    _require_real_directory(receipts, label="service verification receipt directory")
    if expected_sha256 is None:
        try:
            pointer_bytes = read_regular_file(receipts / _SERVICE_RECEIPT_POINTER)
        except OSError as exc:
            raise LiveConfigurationError(
                "current service verification receipt selection is missing or malformed"
            ) from exc
        if (
            len(pointer_bytes) != 65
            or pointer_bytes[-1:] != b"\n"
            or any(byte not in b"0123456789abcdef" for byte in pointer_bytes[:-1])
        ):
            raise LiveConfigurationError(
                "current service verification receipt selection is missing or malformed"
            )
        receipt_hash = pointer_bytes[:-1].decode("ascii")
    else:
        receipt_hash = expected_sha256
        if not _is_sha256(receipt_hash):
            raise LiveConfigurationError(
                "bound service verification receipt hash is missing or malformed"
            )
    receipt_path = receipts / f"{receipt_hash}.json"
    try:
        receipt_bytes = read_regular_file(receipt_path)
    except OSError as exc:
        raise LiveConfigurationError(
            "service verification receipt is missing or malformed"
        ) from exc
    if hashlib.sha256(receipt_bytes).hexdigest() != receipt_hash:
        raise LiveConfigurationError("service verification receipt content hash differs")
    return receipt_path


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise LiveConfigurationError(f"{label} is missing or malformed") from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise LiveConfigurationError(f"{label} is a symbolic link or non-directory")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_live_readiness_receipt(
    *,
    plan: ResolvedPlan,
    provider_runtime_directory: Path,
    environment: Mapping[str, str],
) -> Path:
    """Reopen the all-role preflight receipt and its durable dispatch ledger."""

    _validate_live_models(plan, environment)

    receipt_path = provider_runtime_directory / "model-readiness-receipt.json"
    attempt_path = provider_runtime_directory / "model-readiness-attempt.json"
    try:
        receipt_bytes = read_regular_file(receipt_path)
        receipt = json.loads(receipt_bytes)
        attempt_bytes = read_regular_file(attempt_path)
        attempt = json.loads(attempt_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveConfigurationError("live readiness evidence is missing or malformed") from exc
    receipt_keys = {
        "schema_version",
        "provider_project",
        "resolved_plan_hash",
        "service_verification_receipt_sha256",
        "attempt_sha256",
        "completed_at_utc",
        "operation_timeout_seconds",
        "model_calls_dispatched",
        "role_ids",
        "provider_internal_retries",
        "environment_hash",
        "billing_complete",
        "cost_usd",
        "memory_state_created",
    }
    if not isinstance(receipt, dict) or set(receipt) != receipt_keys:
        raise LiveConfigurationError("live readiness receipt schema is invalid")
    service_receipt_path = resolve_service_verification_receipt(
        provider_runtime_directory,
        expected_sha256=receipt["service_verification_receipt_sha256"],
    )
    try:
        service_receipt = json.loads(read_regular_file(service_receipt_path))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LiveConfigurationError("live readiness evidence is missing or malformed") from exc
    if (
        receipt["schema_version"] != "oamb-provider-model-readiness-receipt-v1"
        or not isinstance(service_receipt, dict)
        or receipt["provider_project"] != service_receipt.get("provider_project")
        or receipt["resolved_plan_hash"] != plan.resolved_plan_hash
        or not _is_sha256(receipt["environment_hash"])
        or receipt["attempt_sha256"] != hashlib.sha256(attempt_bytes).hexdigest()
        or receipt["operation_timeout_seconds"] != plan.execution.operation_timeout_seconds
        or receipt["model_calls_dispatched"] != 7
        or receipt["role_ids"] != list(MODEL_ROLE_IDS)
        or type(receipt["provider_internal_retries"]) is not int
        or receipt["provider_internal_retries"] != 0
        or receipt["billing_complete"] is not False
        or receipt["cost_usd"] is not None
        or receipt["memory_state_created"] is not False
    ):
        raise LiveConfigurationError("live readiness receipt does not bind the resolved plan")
    attempt_keys = {
        "schema_version",
        "provider_project",
        "resolved_plan_hash",
        "started_at_utc",
        "calls_reserved",
        "calls_dispatched",
        "operation_timeout_seconds",
        "attempts",
        "billing_complete",
        "cost_usd",
        "memory_state_created",
    }
    expected_attempt_roles = (
        "embedding",
        "hindsight_extraction",
        "mem0_extraction",
        "openviking_semantic_understanding",
        "answer",
        "judge",
        "openviking-ready",
    )
    attempts = attempt.get("attempts") if isinstance(attempt, dict) else None
    if (
        not isinstance(attempt, dict)
        or set(attempt) != attempt_keys
        or attempt["schema_version"] != "oamb-provider-model-readiness-attempt-v1"
        or attempt["provider_project"] != receipt["provider_project"]
        or attempt["resolved_plan_hash"] != plan.resolved_plan_hash
        or attempt["calls_reserved"] != 7
        or attempt["calls_dispatched"] != 7
        or attempt["operation_timeout_seconds"] != plan.execution.operation_timeout_seconds
        or attempt["billing_complete"] is not False
        or attempt["cost_usd"] is not None
        or attempt["memory_state_created"] is not False
        or not isinstance(attempts, list)
        or tuple(item.get("role") for item in attempts if isinstance(item, dict))
        != expected_attempt_roles
        or any(
            not isinstance(item, dict)
            or item.get("status") != "succeeded"
            or not isinstance(item.get("dispatched_at_utc"), str)
            or not isinstance(item.get("completed_at_utc"), str)
            for item in attempts
        )
    ):
        raise LiveConfigurationError("live readiness attempt ledger is incomplete")
    validate_live_service_verification_receipt(
        plan=plan,
        provider_runtime_directory=provider_runtime_directory,
        environment=environment,
        service_receipt_path=service_receipt_path,
    )
    return service_receipt_path


def live_readiness_environment_hash(
    plan: ResolvedPlan,
    environment: Mapping[str, str],
) -> str:
    """Bind the secret-safe endpoint and credential values used by all model roles."""

    required_names: list[str] = [_LLM_URL_TYPE_VARIABLE]
    for role in plan.model_roles:
        required_names.append(role.endpoint_variable)
        if role.credential_variable != "not_applicable":
            required_names.append(role.credential_variable)
    unique_names = tuple(dict.fromkeys(required_names))
    resolved = _resolve_environment(environment, unique_names)
    return _environment_hash(unique_names, resolved)


def _resolve_live_workload(
    cell: LiveCell,
) -> tuple[LongMemEvalWorkload, DatasetManifest, CaseManifest]:
    source_path = cell.dataset_path
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    workload = LongMemEvalWorkload(build_longmemeval_bundle(source_path, cell.cell.selection))
    dataset_manifest = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset_manifest)
    return workload, dataset_manifest, case_manifest


def load_full_resume_selection(
    *,
    plan: ResolvedPlan,
    progress_root: Path,
) -> FullResumeSelection:
    """Load three strict snapshots and map remaining question IDs before construction."""

    if plan.dataset.selection != "lme60" or tuple(cell.provider_id for cell in plan.cells) != (
        "hindsight",
        "mem0",
        "openviking",
    ):
        raise LiveConfigurationError("full progress resume requires the frozen LME-60 cells")
    source_path = Path(plan.dataset.path)
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    bundle = build_longmemeval_bundle(source_path, plan.dataset.selection)
    manifest = bundle.case_manifest
    if (
        manifest.manifest_hash != plan.dataset.case_manifest_hash
        or manifest.workload_id != plan.dataset.workload_id
    ):
        raise LiveConfigurationError("resume dataset differs from the resolved plan")
    ordered_question_ids = tuple(item.raw_question_id for item in manifest.cases)
    case_id_by_question = {
        item.raw_question_id: item.case_manifest_entry_id for item in manifest.cases
    }
    progress_by_cell: dict[str, FullProgress] = {}
    remaining_by_cell: dict[str, tuple[str, ...]] = {}
    for cell in plan.cells:
        progress = load_full_progress(
            canonical_full_progress_path(progress_root, cell.provider_id),
            expected_resolved_plan_hash=plan.resolved_plan_hash,
            expected_cell_id=cell.cell_id,
            expected_provider_id=cell.provider_id,
            expected_workload_id=cell.workload_id,
            expected_case_manifest_hash=cell.case_manifest_hash,
            expected_ordered_question_ids=ordered_question_ids,
        )
        progress_by_cell[cell.cell_id] = progress
        remaining_by_cell[cell.cell_id] = tuple(
            case_id_by_question[question_id] for question_id in progress.remaining_question_ids
        )
    return FullResumeSelection(
        progress_by_cell=progress_by_cell,
        remaining_case_manifest_entry_ids=remaining_by_cell,
        case_manifest=manifest,
    )


def _validated_cell_internal_retry_count(cell: LiveCell) -> int:
    provider_runtime_directory = cell.control.provider_lifecycle_coordination_directory
    if provider_runtime_directory is None:
        raise LiveConfigurationError("provider retry proof directory is unavailable")
    expected_hash = cell.control.preflight_record.provider_service_evidence.validation_result_hash
    service_receipt_path = resolve_service_verification_receipt(
        provider_runtime_directory,
        expected_sha256=expected_hash,
    )
    validate_live_service_verification_receipt(
        plan=cell.plan,
        provider_runtime_directory=provider_runtime_directory,
        environment=cell.environment,
        service_receipt_path=service_receipt_path,
    )
    return cell.plan.execution.extraction_max_retries


def execute_live_cell(
    cell: LiveCell,
    *,
    stop_event: Any | None = None,
) -> NativeRunArtifacts:
    """Run one already-closed cell through the generic native vertical slice."""

    from oamb.artifacts.store import ArtifactStore
    from oamb.memory_systems.hindsight.adapter import HindsightAdapter
    from oamb.memory_systems.mem0.adapter import Mem0RestAdapter
    from oamb.memory_systems.openviking.session_adapter import (
        OpenVikingSessionAdapter,
        maximum_task_polls_for_timeout,
    )
    from oamb.model_clients.openai_compatible import OpenAICompatibleModelClient
    from oamb.runtime.native_run import run_native_vertical_slice

    workload, _dataset_manifest, case_manifest = _resolve_live_workload(cell)
    bindings_by_role = dict(zip(cell.role_ids, cell.control.role_bindings, strict=True))
    answer_plan = _model_plan(cell.plan, "answer")
    judge_plan = _model_plan(cell.plan, "judge")
    operation_timeout_seconds = cell.plan.execution.operation_timeout_seconds
    memory_timeout_seconds = float(operation_timeout_seconds)
    model_timeout_seconds = float(operation_timeout_seconds)

    def memory_factory(store: object, _plans: object) -> object:
        environment = cell.environment
        provider = cell.cell.provider_id
        internal_retry_count = _validated_cell_internal_retry_count(cell)
        if provider == "hindsight":
            producer = _model_plan(cell.plan, "hindsight_extraction")
            return HindsightAdapter(
                store=store,  # type: ignore[arg-type]
                base_url=environment[cell.cell.endpoint_variable],
                authorization=None,
                extraction_model=producer.model,
                runtime_binding_hash=cell.control.run_spec.runtime_binding_hash,
                internal_retry_count=internal_retry_count,
                read_timeout_seconds=memory_timeout_seconds,
                total_timeout_seconds=memory_timeout_seconds,
            )
        if provider == "mem0":
            return Mem0RestAdapter(
                store=store,  # type: ignore[arg-type]
                base_url=environment[cell.cell.endpoint_variable],
                api_key=environment[cell.cell.credential_variable],
                inspector_base_url=environment["OAMB_MEM0_INSPECTOR_BASE_URL"],
                inspector_api_key=environment["OAMB_MEM0_INSPECTOR_API_KEY"],
                runtime_binding_hash=cell.control.run_spec.runtime_binding_hash,
                internal_retry_count=internal_retry_count,
                read_timeout_seconds=memory_timeout_seconds,
                total_timeout_seconds=memory_timeout_seconds,
            )
        if provider == "openviking":
            return OpenVikingSessionAdapter(
                store=store,  # type: ignore[arg-type]
                base_url=environment[cell.cell.endpoint_variable],
                api_key=environment[cell.cell.credential_variable],
                benchmark_account=environment["OAMB_OPENVIKING_ACCOUNT_ID"],
                benchmark_user=environment["OAMB_OPENVIKING_ADMIN_USER_ID"],
                runtime_binding_hash=cell.control.run_spec.runtime_binding_hash,
                internal_retry_count=internal_retry_count,
                maximum_task_polls=maximum_task_polls_for_timeout(operation_timeout_seconds),
                task_poll_timeout_seconds=memory_timeout_seconds,
                read_timeout_seconds=memory_timeout_seconds,
                total_timeout_seconds=memory_timeout_seconds,
            )
        raise LiveConfigurationError(f"unsupported live provider: {provider}")

    def model_factory(store: object) -> OpenAICompatibleModelClient:
        return OpenAICompatibleModelClient(
            store=store,  # type: ignore[arg-type]
            base_url=cell.environment[answer_plan.endpoint_variable],
            api_key=cell.environment[answer_plan.credential_variable],
            role_binding=bindings_by_role["answer"],
            usage_profile="openai-details-v3",
            read_timeout_seconds=model_timeout_seconds,
            total_timeout_seconds=model_timeout_seconds,
        )

    def judge_factory(store: object) -> OpenAICompatibleModelClient:
        return OpenAICompatibleModelClient(
            store=store,  # type: ignore[arg-type]
            base_url=cell.environment[judge_plan.endpoint_variable],
            api_key=cell.environment[judge_plan.credential_variable],
            role_binding=bindings_by_role["judge"],
            usage_profile="openai-details-v3",
            read_timeout_seconds=model_timeout_seconds,
            total_timeout_seconds=model_timeout_seconds,
        )

    terminal_case_publisher = None
    if cell.progress_path is not None and cell.expected_progress is not None:
        from oamb.runtime.full_progress import ProviderProgressWriter
        from oamb.runtime.native_progress import project_native_progress_entry

        writer = ProviderProgressWriter(cell.progress_path, expected=cell.expected_progress)

        def publish_terminal_case(
            plan_record: object,
            case_record: object,
            attempts: object,
            history_attempts: object,
        ) -> None:
            entry = project_native_progress_entry(
                plan=cell.plan,
                cell=cell.cell,
                capsule_root=cell.capsule_root,
                case_manifest=case_manifest,
                plan_record=plan_record,  # type: ignore[arg-type]
                case_record=case_record,  # type: ignore[arg-type]
                attempts=attempts,  # type: ignore[arg-type]
                history_attempts=history_attempts,  # type: ignore[arg-type]
            )
            if entry is not None:
                writer.publish(entry)

        terminal_case_publisher = publish_terminal_case

    return run_native_vertical_slice(
        output_root=cell.output_root,
        run_id=cell.run_id,
        adapter_profile_id=cell.cell.adapter_profile_id,
        workload=workload,
        visible_evidence_policy=LME_VISIBLE_EVIDENCE_POLICY,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,  # type: ignore[arg-type]
        model_factory=model_factory,
        answer_role_binding_id=bindings_by_role["answer"].binding_id,
        judge_model_factory=judge_factory,
        judge_role_binding_id=bindings_by_role["judge"].binding_id,
        control=cell.control,
        requested_case_manifest_entry_ids=cell.requested_case_manifest_entry_ids,
        terminal_case_publisher=terminal_case_publisher,
        stop_event=stop_event,
    )


def execute_live_cells(cells: tuple[LiveCell, ...]) -> tuple[LiveCellCompletion, ...]:
    """Admit isolated provider cells together and return canonical-order completions."""

    if not cells:
        raise LiveConfigurationError("live execution requires at least one cell")
    plan_hashes = {cell.plan.resolved_plan_hash for cell in cells}
    provider_ids = tuple(cell.cell.provider_id for cell in cells)
    run_ids = tuple(cell.run_id for cell in cells)
    capsule_roots = tuple(cell.capsule_root for cell in cells)
    lifecycle_domains = tuple(cell.control.provider_runtime_directory for cell in cells)
    if len(plan_hashes) != 1:
        raise LiveConfigurationError("parallel live cells must share one resolved plan")
    if len(cells) > cells[0].plan.execution.max_parallel_providers_per_dataset:
        raise LiveConfigurationError("selected cells exceed the provider concurrency limit")
    for label, values in (
        ("provider IDs", provider_ids),
        ("run IDs", run_ids),
        ("capsule roots", capsule_roots),
        ("lifecycle domains", lifecycle_domains),
    ):
        if len(set(values)) != len(values):
            raise LiveConfigurationError(f"parallel live cells require distinct {label}")

    if len(cells) == 1:
        cell = cells[0]
        try:
            completed = execute_live_cell(cell)
        except BaseException as exc:
            detail = f"{type(exc).__name__}: {exc}"
            raise LiveCellExecutionError(
                f"cell {cell.cell.cell_id} failed with {detail}",
                outcomes=(
                    LiveCellOutcome(
                        cell.cell.cell_id,
                        "failed",
                        cell.capsule_root if cell.capsule_root.exists() else None,
                        detail,
                    ),
                ),
                errors=(exc,),
            ) from exc
        if completed.capsule_root != cell.capsule_root:
            raise LiveCellExecutionError(
                f"cell {cell.cell.cell_id} returned a mismatched completion identity",
                outcomes=(
                    LiveCellOutcome(
                        cell.cell.cell_id,
                        "failed",
                        cell.capsule_root if cell.capsule_root.exists() else None,
                        "mismatched completion identity",
                    ),
                ),
            )
        return (LiveCellCompletion(cell.cell.cell_id, completed.capsule_root),)

    context = multiprocessing.get_context("fork")
    stop_events = tuple(context.Event() for _cell in cells)
    previous_signal_handlers = _install_live_stop_handlers(stop_events)
    try:
        return _execute_parallel_live_cells(cells, context, stop_events)
    finally:
        _restore_live_stop_handlers(previous_signal_handlers)


def _execute_parallel_live_cells(
    cells: tuple[LiveCell, ...],
    context: Any,
    stop_events: tuple[Any, ...],
) -> tuple[LiveCellCompletion, ...]:
    workers: list[tuple[LiveCell, BaseProcess, Connection]] = []
    errors: list[BaseException] = []
    outcomes = {
        cell.cell.cell_id: LiveCellOutcome(cell.cell.cell_id, "not_started", None, None)
        for cell in cells
    }
    for cell, stop_event in zip(cells, stop_events, strict=True):
        receiver: Connection | None = None
        sender: Connection | None = None
        created_process: BaseProcess | None = None
        try:
            receiver, sender = context.Pipe(duplex=False)
            created_process = context.Process(
                target=_execute_live_cell_worker,
                args=(cell, sender, stop_event),
                name=f"oamb-cell-{cell.cell.provider_id}",
                daemon=False,
            )
            created_process.start()
        except BaseException as exc:
            error = LiveCellExecutionError(
                f"cell {cell.cell.cell_id} admission failed with {type(exc).__name__}: {exc}"
            )
            errors.append(error)
            outcomes[cell.cell.cell_id] = LiveCellOutcome(
                cell.cell.cell_id,
                "failed",
                cell.capsule_root if cell.capsule_root.exists() else None,
                str(error),
            )
            if created_process is not None and created_process.pid is not None:
                assert receiver is not None and sender is not None
                sender.close()
                workers.append((cell, created_process, receiver))
            else:
                if sender is not None:
                    sender.close()
                if receiver is not None:
                    receiver.close()
                if created_process is not None:
                    created_process.close()
            break
        assert receiver is not None and sender is not None
        sender.close()
        workers.append((cell, created_process, receiver))

    completions: dict[str, LiveCellCompletion] = {}
    for cell, worker_process, receiver in workers:
        message: object | None = None
        try:
            message = receiver.recv()
        except EOFError:
            message = ("failed", cell.cell.cell_id, "WorkerExit", "no result message")
        except BaseException as exc:
            error = LiveCellExecutionError(
                f"cell {cell.cell.cell_id} supervision failed with {type(exc).__name__}: {exc}"
            )
            errors.append(error)
            outcomes[cell.cell.cell_id] = LiveCellOutcome(
                cell.cell.cell_id,
                "failed",
                cell.capsule_root if cell.capsule_root.exists() else None,
                str(error),
            )
        finally:
            receiver.close()
        while True:
            try:
                worker_process.join()
            except BaseException as exc:
                error = LiveCellExecutionError(
                    f"cell {cell.cell.cell_id} join was interrupted by {type(exc).__name__}: {exc}"
                )
                errors.append(error)
                outcomes[cell.cell.cell_id] = LiveCellOutcome(
                    cell.cell.cell_id,
                    "failed",
                    cell.capsule_root if cell.capsule_root.exists() else None,
                    str(error),
                )
                continue
            break
        kind = message[0] if isinstance(message, tuple) and message else None
        if worker_process.exitcode != 0 and kind == "completed":
            message = (
                "failed",
                cell.cell.cell_id,
                "WorkerExit",
                f"worker exited with status {worker_process.exitcode}",
            )
            kind = "failed"
        try:
            worker_process.close()
        except BaseException as exc:
            error = LiveCellExecutionError(
                f"cell {cell.cell.cell_id} worker close failed with {type(exc).__name__}: {exc}"
            )
            errors.append(error)
            current = outcomes[cell.cell.cell_id]
            outcomes[cell.cell.cell_id] = LiveCellOutcome(
                current.cell_id,
                current.status,
                current.capsule_root,
                str(error) if current.detail is None else f"{current.detail}; {error}",
            )
        if kind == "completed" and isinstance(message, tuple) and len(message) == 3:
            _kind, cell_id, capsule_root = message
            if cell_id != cell.cell.cell_id or Path(capsule_root) != cell.capsule_root:
                error = LiveCellExecutionError(
                    f"cell {cell.cell.cell_id} returned a mismatched completion identity"
                )
                errors.append(error)
                outcomes[cell.cell.cell_id] = LiveCellOutcome(
                    cell.cell.cell_id,
                    "failed",
                    cell.capsule_root if cell.capsule_root.exists() else None,
                    str(error),
                )
            else:
                completions[cell_id] = LiveCellCompletion(cell_id, Path(capsule_root))
                prior_detail = outcomes[cell_id].detail
                outcomes[cell_id] = LiveCellOutcome(
                    cell_id,
                    "completed",
                    Path(capsule_root),
                    prior_detail,
                )
        elif kind == "failed" and isinstance(message, tuple) and len(message) == 4:
            _kind, cell_id, error_type, error_message = message
            error = LiveCellExecutionError(
                f"cell {cell_id} failed with {error_type}: {error_message}"
            )
            errors.append(error)
            if cell_id in outcomes:
                outcomes[cell_id] = LiveCellOutcome(
                    cell_id,
                    "failed",
                    cell.capsule_root if cell.capsule_root.exists() else None,
                    f"{error_type}: {error_message}",
                )
        else:
            error = LiveCellExecutionError(
                f"cell {cell.cell.cell_id} returned a malformed worker result"
            )
            errors.append(error)
            outcomes[cell.cell.cell_id] = LiveCellOutcome(
                cell.cell.cell_id,
                "failed",
                cell.capsule_root if cell.capsule_root.exists() else None,
                str(error),
            )
    if errors:
        raise LiveCellExecutionError(
            "; ".join(str(error) for error in errors),
            outcomes=tuple(outcomes[cell.cell.cell_id] for cell in cells),
            errors=tuple(errors),
        )
    return tuple(completions[cell.cell.cell_id] for cell in cells)


def _install_live_stop_handlers(
    stop_events: tuple[Any, ...],
) -> dict[signal.Signals, Any]:
    if threading.current_thread() is not threading.main_thread():
        return {}

    def request_stop(_signum: int, _frame: object) -> None:
        for stop_event in stop_events:
            stop_event.set()

    previous: dict[signal.Signals, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    return previous


def _restore_live_stop_handlers(previous: dict[signal.Signals, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _execute_live_cell_worker(
    cell: LiveCell,
    sender: Connection,
    stop_event: Any,
) -> None:
    try:
        completed = execute_live_cell(cell, stop_event=stop_event)
        sender.send(("completed", cell.cell.cell_id, str(completed.capsule_root)))
    except BaseException as exc:
        sender.send(("failed", cell.cell.cell_id, type(exc).__name__, str(exc)))
    finally:
        sender.close()


def select_live_cells(
    plan: ResolvedPlan, selected_cell_ids: tuple[str, ...]
) -> tuple[CellSpec, ...]:
    """Select an ordered subset without changing any frozen cell bytes."""

    if not selected_cell_ids:
        return plan.cells
    if len(set(selected_cell_ids)) != len(selected_cell_ids):
        raise LiveConfigurationError("live cell selection contains a duplicate")
    cells_by_id = {cell.cell_id: cell for cell in plan.cells}
    unknown = tuple(cell_id for cell_id in selected_cell_ids if cell_id not in cells_by_id)
    if unknown:
        raise LiveConfigurationError(f"unknown live cell: {unknown[0]}")
    canonical = tuple(cell.cell_id for cell in plan.cells if cell.cell_id in selected_cell_ids)
    if canonical != selected_cell_ids:
        raise LiveConfigurationError("live cells must remain in canonical order")
    return tuple(cells_by_id[cell_id] for cell_id in selected_cell_ids)


def resolve_live_question_case_ids(
    plan: ResolvedPlan,
    dataset_path: Path,
    raw_question_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Resolve public raw question IDs to frozen internal case identities."""

    if not raw_question_ids:
        return ()
    if len(set(raw_question_ids)) != len(raw_question_ids):
        raise LiveConfigurationError("live question selection contains a duplicate")
    bundle = build_longmemeval_bundle(Path(dataset_path), plan.dataset.selection)
    cases_by_raw_id = {
        case.raw_question_id: case.case_manifest_entry_id for case in bundle.case_manifest.cases
    }
    unknown = tuple(
        question_id for question_id in raw_question_ids if question_id not in cases_by_raw_id
    )
    if unknown:
        raise LiveConfigurationError(f"unknown frozen question: {unknown[0]}")
    canonical = tuple(
        case.raw_question_id
        for case in bundle.case_manifest.cases
        if case.raw_question_id in raw_question_ids
    )
    if canonical != raw_question_ids:
        raise LiveConfigurationError("live questions must remain in frozen manifest order")
    return tuple(cases_by_raw_id[question_id] for question_id in raw_question_ids)


def build_live_cell(
    *,
    plan: ResolvedPlan,
    cell_id: str,
    output_root: Path,
    provider_runtime_directory: Path,
    provider_project_id: str,
    provider_evidence: SourceEvidenceBinding,
    environment: Mapping[str, str],
    run_label: str,
    observed_at: datetime,
    code_revision: str,
    requested_case_manifest_entry_ids: tuple[str, ...] = (),
    progress_path: Path | None = None,
    expected_progress: FullProgress | None = None,
) -> LiveCell:
    """Close one live cell without constructing provider or model clients."""

    selected = select_live_cells(plan, (cell_id,))
    cell = selected[0]
    if (progress_path is None) != (expected_progress is None):
        raise LiveConfigurationError(
            "resume progress path and expected snapshot must be supplied together"
        )
    if expected_progress is not None:
        if not requested_case_manifest_entry_ids:
            raise LiveConfigurationError("resume cell requires at least one remaining case")
        if (
            expected_progress.resolved_plan_hash != plan.resolved_plan_hash
            or expected_progress.cell_id != cell.cell_id
            or expected_progress.provider_id != cell.provider_id
            or expected_progress.workload_id != cell.workload_id
            or expected_progress.case_manifest_hash != cell.case_manifest_hash
        ):
            raise LiveConfigurationError("resume progress identity differs from the live cell")
        assert progress_path is not None
        if progress_path != canonical_full_progress_path(
            progress_path.parent,
            cell.provider_id,
        ):
            raise LiveConfigurationError("resume progress path is not canonical")
    if not run_label or run_label.strip() != run_label:
        raise LiveConfigurationError("live run label must be non-empty canonical text")
    if not provider_project_id or provider_project_id.strip() != provider_project_id:
        raise LiveConfigurationError("provider project ID is required")
    if not provider_runtime_directory.is_absolute():
        raise LiveConfigurationError("provider runtime directory must be absolute")
    if observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise LiveConfigurationError("live observed_at requires an explicit timezone")
    if provider_evidence.source_kind != SourceEvidenceKind.PROVIDER_SERVICE:
        raise LiveConfigurationError("provider evidence must be provider-service evidence")
    if "provider_service_evidence_manifest@1" not in provider_evidence.source_schema_versions:
        raise LiveConfigurationError("provider evidence schema is not the verified service profile")

    roles_by_id = {role.role_id: role for role in plan.model_roles}
    role_ids = (
        cell.producer_role_id,
        cell.embedding_role_id,
        cell.answer_role_id,
        cell.judge_role_id,
    )
    selected_roles = tuple(roles_by_id[role_id] for role_id in role_ids)
    required_environment = _required_environment(cell, selected_roles)
    resolved_environment = _resolve_environment(environment, required_environment)
    run_id = canonical_sha256(
        ["oamb-live-run-initial-v1", plan.resolved_plan_hash, cell.cell_spec_hash, run_label]
    )
    frozen_output_root = Path(output_root).absolute()
    capsule_root = frozen_output_root / run_id
    if capsule_root.exists() or capsule_root.is_symlink():
        raise LiveConfigurationError(f"live capsule already exists: {capsule_root}")
    lifecycle_domain = provider_runtime_directory / "lifecycle-domains" / cell.provider_id
    for pointer_name in ("active-operation", "active-provider-attempt"):
        if (provider_runtime_directory / pointer_name).exists():
            raise LiveConfigurationError(
                f"legacy provider lifecycle pointer is active before client construction: "
                f"{pointer_name}"
            )
        if (lifecycle_domain / pointer_name).exists():
            raise LiveConfigurationError(
                f"provider lifecycle domain is active before client construction: "
                f"{cell.provider_id}/{pointer_name}"
            )
    attempts_directory = lifecycle_domain / "active-provider-attempts"
    if attempts_directory.is_dir() and any(attempts_directory.iterdir()):
        raise LiveConfigurationError(
            f"provider lifecycle domain has active attempts: {cell.provider_id}"
        )

    bindings = _role_bindings(selected_roles, resolved_environment)
    binding_by_role_id: dict[str, ModelRoleBindingV2] = {
        role_id: binding for role_id, binding in zip(role_ids, bindings, strict=True)
    }
    budget = _budget(
        run_id=run_id,
        cell=cell,
        plan=plan,
        binding_by_role_id=binding_by_role_id,
    )
    runtime_binding_hash = canonical_sha256(
        [
            "oamb-live-runtime-binding-initial-v1",
            cell.cell_spec_hash,
            provider_evidence.binding_id,
            _endpoint_fingerprint(cell.endpoint_variable, resolved_environment),
        ]
    )
    run_spec = RunSpec(
        run_id=run_id,
        protocol_id="oamb-initial-v1",
        dataset_manifest_hash=LME_DATASET_MANIFEST_HASH,
        case_manifest_hash=plan.dataset.case_manifest_hash,
        workload_id=cell.workload_id,
        memory_system_id=_MEMORY_SYSTEM_BY_PROVIDER[cell.provider_id],
        runtime_binding_hash=runtime_binding_hash,
        environment_hash=_environment_hash(required_environment, resolved_environment),
        model_role_binding_ids=tuple(binding.binding_id for binding in bindings),
        budget_id=budget.budget_id,
        code_revision=code_revision,
        normalizer_fingerprint=canonical_sha256(
            ["oamb-live-normalizer-initial-v1", cell.provider_id, cell.retrieval_binding_hash]
        ),
    )
    artifact_fingerprint = canonical_sha256(
        ["oamb-artifact-root-initial-v1", str(frozen_output_root.resolve(strict=False))]
    )
    preflight_fields = {
        "run_id": run_id,
        "observed_at": observed_at.astimezone(UTC),
        "resolved_plan_hash": plan.resolved_plan_hash,
        "run_spec_hash": canonical_sha256(run_spec),
        "dataset_manifest_hash": run_spec.dataset_manifest_hash,
        "subset_manifest_hash": run_spec.case_manifest_hash,
        "adapter_profile_id": cell.adapter_profile_id,
        "adapter_profile_hash": cell.cell_spec_hash,
        "provider_project_id": provider_project_id,
        "provider_profile_id": cell.adapter_profile_id,
        "runtime_binding_hash": runtime_binding_hash,
        "provider_service_evidence": provider_evidence,
        "provider_profile_evidence": provider_evidence,
        "role_binding_ids": run_spec.model_role_binding_ids,
        "dispatch_routes": budget.dispatch_routes,
        "budget_hash": canonical_sha256(budget),
        "redacted_endpoint_fingerprints": tuple(
            binding.redacted_endpoint_fingerprint for binding in bindings
        ),
        "credential_reference_fingerprints": tuple(
            canonical_sha256(
                [
                    "oamb-credential-reference-initial-v1",
                    role_id,
                    role.credential_variable,
                ]
            )
            for role_id, role in zip(role_ids, selected_roles, strict=True)
        ),
        "artifact_repository_fingerprint": artifact_fingerprint,
        "artifact_durability_proof_hash": canonical_sha256(
            ["oamb-artifact-durability-initial-v1", artifact_fingerprint, "create-only"]
        ),
    }
    preflight = RunPreflightRecord.model_validate(
        {
            "preflight_record_hash": run_preflight_record_hash(preflight_fields),
            **preflight_fields,
        }
    )
    control = NativeRunControl(
        run_spec=run_spec,
        preflight_record=preflight,
        budget=budget,
        role_bindings=bindings,
        owner_id=f"oamb-live-{os.getpid()}",
        host_fingerprint=canonical_sha256(["oamb-live-host-initial-v1", socket.gethostname()]),
        process_id=os.getpid(),
        provider_runtime_directory=lifecycle_domain,
        wall_clock=lambda: datetime.now(UTC),
        monotonic_clock=time.monotonic,
        max_parallel_history_ingestions=(
            plan.execution.max_parallel_history_ingestions_per_provider
        ),
        max_parallel_questions=plan.execution.max_parallel_questions_per_provider,
        max_retries_per_operation=plan.execution.max_retries_per_operation,
        extraction_max_retries=plan.execution.extraction_max_retries,
        model_max_attempts=plan.execution.model_max_attempts,
        model_transport_max_retries=plan.execution.model_transport_max_retries,
        provider_lifecycle_coordination_directory=provider_runtime_directory,
    )
    return LiveCell(
        plan=plan,
        cell=cell,
        run_id=run_id,
        role_ids=role_ids,
        output_root=frozen_output_root,
        capsule_root=capsule_root,
        dataset_path=Path(plan.dataset.path),
        control=control,
        environment=resolved_environment,
        requested_case_manifest_entry_ids=requested_case_manifest_entry_ids,
        progress_path=progress_path,
        expected_progress=expected_progress,
    )


def _required_environment(
    cell: CellSpec,
    roles: tuple[ModelExecutionBinding, ...],
) -> tuple[str, ...]:
    names = [_LLM_URL_TYPE_VARIABLE, cell.endpoint_variable]
    if cell.credential_variable != "not_applicable":
        names.append(cell.credential_variable)
    for role in roles:
        names.append(role.endpoint_variable)
        if role.credential_variable != "not_applicable":
            names.append(role.credential_variable)
    names.extend(_EXTRA_ENVIRONMENT_BY_PROVIDER[cell.provider_id])
    return tuple(dict.fromkeys(names))


def _model_plan(plan: ResolvedPlan, role_id: ModelRoleId) -> ModelExecutionBinding:
    matches = tuple(role for role in plan.model_roles if role.role_id == role_id)
    if len(matches) != 1:
        raise LiveConfigurationError(f"resolved plan does not contain one model role: {role_id}")
    return matches[0]


def _resolve_environment(
    environment: Mapping[str, str], required_names: tuple[str, ...]
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for name in required_names:
        value = environment.get(name)
        if not isinstance(value, str) or not value:
            raise LiveConfigurationError(f"missing live environment reference: {name}")
        resolved[name] = value
    return resolved


def _endpoint_fingerprint(name: str, environment: Mapping[str, str]) -> str:
    return canonical_sha256(["oamb-live-endpoint-initial-v1", name, environment[name]])


def _environment_hash(
    required_names: tuple[str, ...],
    environment: Mapping[str, str],
) -> str:
    return canonical_sha256(
        [
            "oamb-live-environment-initial-v1",
            tuple(
                (
                    name,
                    canonical_sha256(["redacted-environment-value", environment[name]]),
                )
                for name in required_names
            ),
        ]
    )


def _role_bindings(
    roles: tuple[ModelExecutionBinding, ...],
    environment: Mapping[str, str],
) -> tuple[ModelRoleBindingV2, ...]:
    result: list[ModelRoleBindingV2] = []
    for role in roles:
        model_role = {
            "hindsight_extraction": ModelRole.MEMORY_EXTRACTION,
            "mem0_extraction": ModelRole.MEMORY_EXTRACTION,
            "openviking_semantic_understanding": ModelRole.MEMORY_EXTRACTION,
            "embedding": ModelRole.EMBEDDING,
            "answer": ModelRole.ANSWER,
            "judge": ModelRole.JUDGE,
        }[role.role_id]
        binding_id = {
            "answer": "oamb-lme-answer-v1",
            "judge": "oamb-lme-judge-v1",
        }.get(role.role_id, role.binding_hash)
        result.append(
            ModelRoleBindingV2(
                binding_id=binding_id,
                role=model_role,
                role_status=RoleBindingStatus.SELECTED,
                execution_owner=(
                    ExecutionOwner.HARNESS
                    if role.execution_owner == "harness"
                    else ExecutionOwner.MEMORY_SYSTEM
                ),
                binding_kind=(
                    BindingKind.MODEL_CLIENT
                    if role.execution_owner == "harness"
                    else BindingKind.NATIVE
                ),
                provider=("vllm-metal" if role.role_id == "embedding" else _SUPPORTED_LLM_URL_TYPE),
                endpoint_reference=role.endpoint_variable,
                credential_variable_name=(
                    None
                    if role.credential_variable == "not_applicable"
                    else role.credential_variable
                ),
                model=role.model,
                thinking_effort=role.thinking_effort,
                parameters_fingerprint=canonical_sha256(
                    [
                        "oamb-live-model-parameters-initial-v1",
                        role.binding_hash,
                        role.temperature,
                        role.top_p,
                        role.maximum_output_tokens_per_call,
                    ]
                ),
                retry_policy_id="no-retry-v1",
                configuration_fingerprint=role.binding_hash,
                redacted_endpoint_fingerprint=canonical_sha256(
                    [
                        "oamb-live-role-endpoint-initial-v1",
                        role.role_id,
                        _endpoint_fingerprint(role.endpoint_variable, environment),
                    ]
                ),
            )
        )
    return tuple(result)


def _budget(
    *,
    run_id: str,
    cell: CellSpec,
    plan: ResolvedPlan,
    binding_by_role_id: Mapping[str, ModelRoleBindingV2],
) -> BudgetSpecV4:
    case_count, source_count = _workload_counts(cell.selection)
    provider_role_id = cell.producer_role_id
    ingest_attempts = plan.execution.max_retries_per_operation + 1
    model_attempts = plan.execution.model_max_attempts * (
        plan.execution.model_transport_max_retries + 1
    )
    operation_timeout_seconds = plan.execution.operation_timeout_seconds
    role_attempts: dict[str, int] = {
        provider_role_id: source_count * ingest_attempts,
        "embedding": source_count * ingest_attempts + case_count,
        "answer": case_count * model_attempts,
        "judge": case_count * model_attempts,
    }
    output_tokens_per_attempt: dict[str, int] = {
        provider_role_id: 0,
        "embedding": 0,
        "answer": LME_ANSWER_MAX_OUTPUT_TOKENS,
        "judge": LME_JUDGE_MAX_OUTPUT_TOKENS,
    }
    role_ceilings_list: list[RoleBudgetCeiling] = []
    for role_id, binding in binding_by_role_id.items():
        max_attempts = role_attempts[role_id]
        owns_wall = role_id in {"answer", "judge"}
        max_wall = Decimal(max_attempts * operation_timeout_seconds if owns_wall else 0)
        output_tokens = output_tokens_per_attempt[role_id] * max_attempts
        role_ceilings_list.append(
            RoleBudgetCeiling(
                role_binding_id=binding.binding_id,
                max_attempts=max_attempts,
                max_input_tokens=0,
                max_output_tokens=output_tokens,
                max_dispatch_wall_seconds=max_wall,
                max_cost=None,
                currency=None,
                price_snapshot_id=None,
                resource_ceilings=(
                    (
                        ResourceBudgetCeiling(
                            dimension_id="provider_request_wall_seconds_v1",
                            maximum=max_wall,
                            unit="seconds",
                        ),
                    )
                    if owns_wall
                    else ()
                ),
                provider_budget_cap=ProviderBudgetCap(
                    provider=binding.provider or "provider",
                    operation_kind=(
                        "embeddings" if binding.role == ModelRole.EMBEDDING else "chat_completion"
                    ),
                    billing_unit="request",
                    maximum_accepted_units=Decimal(max_attempts),
                ),
            )
        )
    role_ceilings = tuple(role_ceilings_list)
    operation_attempts = {
        "runtime_resolve": 1,
        "scope_allocate": case_count,
        "memory_ingest": source_count * ingest_attempts,
        "memory_readiness": case_count,
        "memory_projection": case_count,
        "pre_query_projection": case_count,
        "memory_query": case_count,
        "post_query_projection": case_count,
    }
    operations: list[ProviderOperationBudgetCeiling] = []
    routes: list[DispatchBudgetRoute] = []
    producer_id = binding_by_role_id[provider_role_id].binding_id
    embedding_id = binding_by_role_id["embedding"].binding_id
    for stage in _MEMORY_STAGES:
        operation_id = f"{cell.adapter_profile_id}:{stage}"
        max_attempts = operation_attempts[stage]
        max_wall = Decimal(max_attempts * operation_timeout_seconds)
        request_seconds = ResourceBudgetCeiling(
            dimension_id="provider_request_wall_seconds_v1",
            maximum=max_wall,
            unit="seconds",
        )
        operation_fields = {
            "provider_operation_ceiling_id": operation_id,
            "adapter_profile_id": cell.adapter_profile_id,
            "operation_kind": stage,
            "billing_unit": "request",
            "maximum_accepted_units": Decimal(max_attempts),
            "max_attempts": max_attempts,
            "max_dispatch_wall_seconds": max_wall,
            "resource_ceilings": (request_seconds,),
        }
        operations.append(
            ProviderOperationBudgetCeiling.model_validate(
                {
                    "provider_operation_ceiling_hash": (
                        provider_operation_budget_ceiling_hash(operation_fields)
                    ),
                    **operation_fields,
                }
            )
        )
        internal_roles = (
            (producer_id, embedding_id)
            if stage == "memory_ingest"
            else ((embedding_id,) if stage == "memory_query" else ())
        )
        route_fields = {
            "route_id": stage,
            "stage": stage,
            "dispatch_owner_kind": DispatchBudgetOwnerKind.PROVIDER_OPERATION,
            "dispatch_model_role_binding_id": None,
            "provider_operation_ceiling_id": operation_id,
            "adapter_profile_id": cell.adapter_profile_id,
            "operation_kind": stage,
            "billing_unit": "request",
            "internal_usage_role_binding_ids": internal_roles,
        }
        routes.append(
            DispatchBudgetRoute.model_validate(
                {"route_hash": dispatch_budget_route_hash(route_fields), **route_fields}
            )
        )
    for stage in ("answer", "judge"):
        binding = binding_by_role_id[stage]
        route_fields = {
            "route_id": stage,
            "stage": stage,
            "dispatch_owner_kind": DispatchBudgetOwnerKind.MODEL_ROLE,
            "dispatch_model_role_binding_id": binding.binding_id,
            "provider_operation_ceiling_id": None,
            "adapter_profile_id": None,
            "operation_kind": "chat_completion",
            "billing_unit": "request",
            "internal_usage_role_binding_ids": (),
        }
        routes.append(
            DispatchBudgetRoute.model_validate(
                {"route_hash": dispatch_budget_route_hash(route_fields), **route_fields}
            )
        )
    maximum_operation_attempts = (
        sum(operation_attempts.values()) + role_attempts["answer"] + role_attempts["judge"]
    )
    maximum_owner_authorizations = sum(operation_attempts.values()) + sum(role_attempts.values())
    if (
        maximum_operation_attempts != plan.execution.per_cell_max_operation_attempt_count
        or maximum_owner_authorizations != plan.execution.per_cell_max_owner_authorization_count
    ):
        raise LiveConfigurationError(
            "derived live operation authorization differs from the resolved plan"
        )
    maximum_output_tokens = sum(
        output_tokens_per_attempt[role_id] * max_attempts
        for role_id, max_attempts in role_attempts.items()
    )
    maximum_dispatch_wall_seconds = Decimal(maximum_operation_attempts * operation_timeout_seconds)
    budget_fields = {
        "budget_id": canonical_sha256(["oamb-live-operation-authorization-initial-v1", run_id]),
        "scope_kind": BudgetScopeKindV3.RUN,
        "scope_id": run_id,
        "max_attempts": maximum_owner_authorizations,
        "max_input_tokens": 0,
        "max_output_tokens": maximum_output_tokens,
        "max_dispatch_wall_seconds": maximum_dispatch_wall_seconds,
        "max_cost": None,
        "currency": None,
        "resource_ceilings": (
            ResourceBudgetCeiling(
                dimension_id="provider_request_wall_seconds_v1",
                maximum=maximum_dispatch_wall_seconds,
                unit="seconds",
            ),
        ),
        "role_ceilings": role_ceilings,
        "provider_operation_ceilings": tuple(operations),
        "dispatch_routes": tuple(routes),
        "stop_condition_ids": (
            "operation_authorization_exhausted",
            "identity_drift",
            "unknown_outcome",
        ),
    }
    return BudgetSpecV4.model_validate(
        {"budget_hash": budget_spec_v4_hash(budget_fields), **budget_fields}
    )


def _workload_counts(selection: str) -> tuple[int, int]:
    if selection == "lme6":
        return len(LME6_EXPECTED_QUESTION_IDS), LME6_EXPECTED_SESSION_COUNT
    if selection == "lme30":
        return len(LME30_EXPECTED_QUESTION_IDS), LME30_EXPECTED_SESSION_COUNT
    if selection == "lme60":
        return len(LME60_EXPECTED_QUESTION_IDS), LME60_EXPECTED_SESSION_COUNT
    raise LiveConfigurationError(f"unsupported live workload selection: {selection}")


__all__ = [
    "LiveCell",
    "LiveCellCompletion",
    "LiveCellExecutionError",
    "LiveCellOutcome",
    "LiveConfigurationError",
    "build_live_cell",
    "execute_live_cell",
    "execute_live_cells",
    "load_live_environment",
    "load_live_provider_evidence",
    "live_readiness_environment_hash",
    "resolve_live_question_case_ids",
    "select_live_cells",
    "validate_live_readiness_receipt",
    "validate_live_service_verification_receipt",
]
