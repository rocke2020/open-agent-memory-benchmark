"""Generic live LongMemEval cell composition from one frozen resolved plan."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path

from oamb.config.benchmark import ModelRoleId
from oamb.config.doctor import CellSpec, ModelExecutionBinding, ResolvedPlan
from oamb.config.provider_services import load_provider_service_bindings
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    BindingKind,
    BudgetScopeKindV3,
    BudgetSpecV4,
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
from oamb.runtime.native_continuation import InitializedContinuation
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
    LME30_WORKLOAD_ID,
    LME_DATASET_MANIFEST_HASH,
    LongMemEvalWorkload,
    build_lme6_bundle,
    build_lme30_bundle,
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
_PROVIDER_ROLE_BY_CELL: dict[str, ModelRoleId] = {
    "hindsight-lme6": "hindsight_extraction",
    "mem0-lme6": "mem0_extraction",
    "openviking-lme6": "openviking_semantic_understanding",
}
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


class LiveConfigurationError(ValueError):
    """The frozen cell cannot be safely composed before any client exists."""


class LiveCellExecutionError(RuntimeError):
    """One admitted isolated provider cell failed after the shared barrier."""


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
    continuation: InitializedContinuation | None = None


@dataclass(frozen=True, slots=True)
class LiveCellCompletion:
    cell_id: str
    capsule_root: Path


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
            expected_keys=frozenset({"DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"}),
        )
    )
    aliases = {
        "OAMB_DEEPSEEK_BASE_URL": "DEEPSEEK_BASE_URL",
        "OAMB_DEEPSEEK_API_KEY": "DEEPSEEK_API_KEY",
    }
    for target, source in aliases.items():
        if target not in environment and source in environment:
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


def load_live_provider_evidence(
    *,
    plan: ResolvedPlan,
    provider_runtime_directory: Path,
    environment: Mapping[str, str],
) -> tuple[str, dict[str, SourceEvidenceBinding]]:
    """Reopen the current provider-service proof store before live composition."""

    receipt_paths = tuple(
        sorted((provider_runtime_directory / "service-verification-receipts").glob("*.json"))
    )
    if len(receipt_paths) != 1:
        raise LiveConfigurationError(
            "provider runtime must contain exactly one verification receipt"
        )
    receipt_path = receipt_paths[0]
    try:
        receipt = json.loads(receipt_path.read_bytes())
        provider_project = receipt["provider_project"]
        attestation_hash = receipt["project_attestation_sha256"]
    except (KeyError, TypeError, json.JSONDecodeError, UnicodeError) as exc:
        raise LiveConfigurationError("provider verification receipt is malformed") from exc
    if not isinstance(provider_project, str) or not isinstance(attestation_hash, str):
        raise LiveConfigurationError("provider verification receipt identity is malformed")
    attestation_path = provider_runtime_directory / "provider-project.attestation"
    try:
        actual_attestation_hash = hashlib.sha256(attestation_path.read_bytes()).hexdigest()
        embedding_artifact_hash = hashlib.sha256(
            (provider_runtime_directory / "embedding-ready-response.json").read_bytes()
        ).hexdigest()
        embedding_endpoint = environment["OAMB_EMBEDDING_BASE_URL"]
    except (KeyError, OSError) as exc:
        raise LiveConfigurationError(
            "provider runtime embedding or attestation proof is missing"
        ) from exc
    if actual_attestation_hash != attestation_hash:
        raise LiveConfigurationError("provider project attestation content changed")
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
        expected_provider_models={
            "hindsight-rest-v1": _model_plan(plan, "hindsight_extraction").configured_model,
            "mem0-rest-v1": _model_plan(plan, "mem0_extraction").configured_model,
            "openviking-rest-v1": _model_plan(
                plan, "openviking_semantic_understanding"
            ).configured_model,
        },
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


def execute_live_cell(cell: LiveCell) -> NativeRunArtifacts:
    """Run one already-closed cell through the generic native vertical slice."""

    from oamb.artifacts.store import ArtifactStore
    from oamb.contracts.specifications import INFRASTRUCTURE_RETRY_POLICY_HASH
    from oamb.memory_systems.hindsight.adapter import HindsightAdapter
    from oamb.memory_systems.mem0.adapter import Mem0RestAdapter
    from oamb.memory_systems.openviking.session_adapter import (
        OpenVikingSessionAdapter,
        maximum_task_polls_for_timeout,
    )
    from oamb.model_clients.openai_compatible import OpenAICompatibleModelClient
    from oamb.runtime.case_partition import build_case_partition_spec
    from oamb.runtime.native_run import run_native_vertical_slice

    source_path = cell.dataset_path
    if not source_path.is_absolute():
        source_path = Path.cwd() / source_path
    bundle = build_lme6_bundle(build_lme30_bundle(source_path))
    workload = LongMemEvalWorkload(bundle)
    partition = None
    if cell.requested_case_manifest_entry_ids:
        dataset_manifest = workload.resolve_sources()
        case_manifest = workload.build_case_manifest(dataset_manifest)
        partition = build_case_partition_spec(
            run_id=cell.run_id,
            resolved_plan_hash=cell.plan.resolved_plan_hash,
            cell_spec_hash=cell.cell.cell_spec_hash,
            dataset_manifest_hash=dataset_manifest.manifest_hash,
            case_manifest=case_manifest,
            case_plans=workload.iter_case_plans(case_manifest),
            requested_case_manifest_entry_ids=cell.requested_case_manifest_entry_ids,
            budget_policy_hash=cell.cell.limits_hash,
            retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
        )
    continuation = None
    if cell.continuation is not None:
        from oamb.runtime.native_continuation import load_openviking_continuation

        case_manifest = workload.build_case_manifest(workload.resolve_sources())
        continuation = load_openviking_continuation(
            cell.continuation,
            plans=workload.iter_ingestion_plans(case_manifest),
            case_plans=workload.iter_case_plans(case_manifest),
        )
    bindings_by_role = dict(zip(cell.role_ids, cell.control.role_bindings, strict=True))
    answer_plan = _model_plan(cell.plan, "answer")
    judge_plan = _model_plan(cell.plan, "judge")
    memory_timeout_seconds = float(cell.plan.limits.memory_operation_timeout_seconds)
    model_timeout_seconds = float(cell.plan.limits.model_call_timeout_seconds)

    def memory_factory(store: object, _plans: object) -> object:
        environment = cell.environment
        provider = cell.cell.provider_id
        if provider == "hindsight":
            producer = _model_plan(cell.plan, "hindsight_extraction")
            return HindsightAdapter(
                store=store,  # type: ignore[arg-type]
                base_url=environment[cell.cell.endpoint_variable],
                authorization=None,
                configured_extraction_model=producer.configured_model,
                runtime_extraction_model=producer.runtime_model,
                runtime_binding_hash=cell.control.run_spec.runtime_binding_hash,
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
                maximum_task_polls=maximum_task_polls_for_timeout(
                    cell.plan.limits.memory_operation_timeout_seconds
                ),
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
            runtime_model_policy="require_match",
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
            runtime_model_policy="require_match",
            usage_profile="openai-details-v3",
            read_timeout_seconds=model_timeout_seconds,
            total_timeout_seconds=model_timeout_seconds,
        )

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
        continuation=continuation,
        partition=partition,
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

    context = multiprocessing.get_context("fork")
    workers: list[tuple[LiveCell, BaseProcess, Connection]] = []
    errors: list[BaseException] = []
    for cell in cells:
        receiver: Connection | None = None
        sender: Connection | None = None
        created_process: BaseProcess | None = None
        try:
            receiver, sender = context.Pipe(duplex=False)
            created_process = context.Process(
                target=_execute_live_cell_worker,
                args=(cell, sender),
                name=f"oamb-cell-{cell.cell.provider_id}",
                daemon=False,
            )
            created_process.start()
        except BaseException as exc:
            errors.append(
                LiveCellExecutionError(
                    f"cell {cell.cell.cell_id} admission failed with {type(exc).__name__}: {exc}"
                )
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
            errors.append(
                LiveCellExecutionError(
                    f"cell {cell.cell.cell_id} supervision failed with {type(exc).__name__}: {exc}"
                )
            )
        finally:
            receiver.close()
        while True:
            try:
                worker_process.join()
            except BaseException as exc:
                errors.append(
                    LiveCellExecutionError(
                        f"cell {cell.cell.cell_id} join was interrupted by "
                        f"{type(exc).__name__}: {exc}"
                    )
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
            errors.append(
                LiveCellExecutionError(
                    f"cell {cell.cell.cell_id} worker close failed with {type(exc).__name__}: {exc}"
                )
            )
        if kind == "completed" and isinstance(message, tuple) and len(message) == 3:
            _kind, cell_id, capsule_root = message
            if cell_id != cell.cell.cell_id or Path(capsule_root) != cell.capsule_root:
                errors.append(
                    LiveCellExecutionError(
                        f"cell {cell.cell.cell_id} returned a mismatched completion identity"
                    )
                )
            else:
                completions[cell_id] = LiveCellCompletion(cell_id, Path(capsule_root))
        elif kind == "failed" and isinstance(message, tuple) and len(message) == 4:
            _kind, cell_id, error_type, error_message = message
            errors.append(
                LiveCellExecutionError(f"cell {cell_id} failed with {error_type}: {error_message}")
            )
        else:
            errors.append(
                LiveCellExecutionError(
                    f"cell {cell.cell.cell_id} returned a malformed worker result"
                )
            )
    if len(errors) == 1:
        raise errors[0]
    if errors:
        raise BaseExceptionGroup("parallel live cells failed", errors)
    return tuple(completions[cell.cell.cell_id] for cell in cells)


def _execute_live_cell_worker(cell: LiveCell, sender: Connection) -> None:
    try:
        completed = execute_live_cell(cell)
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
) -> LiveCell:
    """Close one live cell without constructing provider or model clients."""

    selected = select_live_cells(plan, (cell_id,))
    cell = selected[0]
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
        _PROVIDER_ROLE_BY_CELL[cell.cell_id],
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
        workload_id=LME30_WORKLOAD_ID,
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
    )


def build_live_continuation_cell(
    *,
    plan: ResolvedPlan,
    cell_id: str,
    base_capsule_root: Path,
    output_root: Path,
    provider_runtime_directory: Path,
    provider_project_id: str,
    provider_evidence: SourceEvidenceBinding,
    environment: Mapping[str, str],
) -> LiveCell:
    """Reopen one aborted OpenViking cell without discarding confirmed work."""

    from oamb.contracts.evidence import RunRecord
    from oamb.runtime.native_continuation import initialize_continuation_root

    cell = select_live_cells(plan, (cell_id,))[0]
    if cell.provider_id != "openviking":
        raise LiveConfigurationError("current continuation supports only OpenViking")
    if not provider_runtime_directory.is_absolute():
        raise LiveConfigurationError("provider runtime directory must be absolute")
    if (
        provider_evidence.source_kind != SourceEvidenceKind.PROVIDER_SERVICE
        or "provider_service_evidence_manifest@1" not in provider_evidence.source_schema_versions
    ):
        raise LiveConfigurationError("provider evidence is not the verified service profile")
    roles_by_id = {role.role_id: role for role in plan.model_roles}
    role_ids = (
        _PROVIDER_ROLE_BY_CELL[cell.cell_id],
        cell.embedding_role_id,
        cell.answer_role_id,
        cell.judge_role_id,
    )
    selected_roles = tuple(roles_by_id[role_id] for role_id in role_ids)
    required_environment = _required_environment(cell, selected_roles)
    resolved_environment = _resolve_environment(environment, required_environment)
    current_bindings = _role_bindings(selected_roles, resolved_environment)
    current_binding_by_role_id: dict[str, ModelRoleBindingV2] = {
        role_id: binding for role_id, binding in zip(role_ids, current_bindings, strict=True)
    }
    base = Path(base_capsule_root).resolve(strict=True)
    try:
        run_spec = RunSpec.model_validate_json((base / "source/specs/run-spec.json").read_bytes())
        preflight = RunPreflightRecord.model_validate_json(
            (base / "source/specs/run-preflight.json").read_bytes()
        )
        budget = BudgetSpecV4.model_validate_json((base / "source/specs/budget.json").read_bytes())
        old_run = RunRecord.model_validate_json(
            (base / f"source/run/{run_spec.run_id}.json").read_bytes()
        )
        bindings_by_id = {
            binding.binding_id: binding
            for path in (base / "source/model-role-bindings").glob("*.json")
            for binding in (ModelRoleBindingV2.model_validate_json(path.read_bytes()),)
        }
        bindings = tuple(
            bindings_by_id[binding_id] for binding_id in run_spec.model_role_binding_ids
        )
    except (KeyError, OSError, ValueError) as exc:
        raise LiveConfigurationError("continuation base control evidence is incomplete") from exc
    expected_runtime_binding_hash = canonical_sha256(
        [
            "oamb-live-runtime-binding-initial-v1",
            cell.cell_spec_hash,
            provider_evidence.binding_id,
            _endpoint_fingerprint(cell.endpoint_variable, resolved_environment),
        ]
    )
    expected_budget = _budget(
        run_id=run_spec.run_id,
        cell=cell,
        plan=plan,
        binding_by_role_id=current_binding_by_role_id,
    )
    if (
        old_run.state.value != "aborted"
        or preflight.resolved_plan_hash != plan.resolved_plan_hash
        or preflight.adapter_profile_hash != cell.cell_spec_hash
        or preflight.run_id != run_spec.run_id
        or preflight.run_spec_hash != canonical_sha256(run_spec)
        or preflight.provider_project_id != provider_project_id
        or preflight.adapter_profile_id != cell.adapter_profile_id
        or preflight.provider_service_evidence != provider_evidence
        or preflight.provider_profile_evidence != provider_evidence
        or preflight.runtime_binding_hash != expected_runtime_binding_hash
        or run_spec.runtime_binding_hash != expected_runtime_binding_hash
        or run_spec.environment_hash
        != _environment_hash(required_environment, resolved_environment)
        or run_spec.dataset_manifest_hash != LME_DATASET_MANIFEST_HASH
        or run_spec.case_manifest_hash != plan.dataset.case_manifest_hash
        or run_spec.workload_id != LME30_WORKLOAD_ID
        or budget != expected_budget
        or bindings != current_bindings
        or tuple(binding.binding_id for binding in bindings) != run_spec.model_role_binding_ids
    ):
        raise LiveConfigurationError(
            "continuation resolved plan or runtime binding differs from the frozen OpenViking cell"
        )
    lifecycle_domain = provider_runtime_directory / "lifecycle-domains" / cell.provider_id
    for pointer_name in ("active-operation", "active-provider-attempt"):
        if (lifecycle_domain / pointer_name).exists():
            raise LiveConfigurationError(
                f"provider lifecycle domain is active before continuation: {pointer_name}"
            )
    attempts_directory = lifecycle_domain / "active-provider-attempts"
    if attempts_directory.is_dir() and any(attempts_directory.iterdir()):
        raise LiveConfigurationError("provider lifecycle domain has active attempts")

    initialized = initialize_continuation_root(base, Path(output_root))
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
        provider_lifecycle_coordination_directory=provider_runtime_directory,
    )
    return LiveCell(
        plan=plan,
        cell=cell,
        run_id=run_spec.run_id,
        role_ids=role_ids,
        output_root=initialized.target_root.parent,
        capsule_root=initialized.target_root,
        dataset_path=Path(plan.dataset.path),
        control=control,
        environment=resolved_environment,
        continuation=initialized,
    )


def _required_environment(
    cell: CellSpec,
    roles: tuple[ModelExecutionBinding, ...],
) -> tuple[str, ...]:
    names = [cell.endpoint_variable]
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
                provider=("vllm-metal" if role.role_id == "embedding" else "deepseek"),
                endpoint_reference=role.endpoint_variable,
                credential_variable_name=(
                    None
                    if role.credential_variable == "not_applicable"
                    else role.credential_variable
                ),
                configured_model=role.configured_model,
                resolved_model=role.runtime_model,
                thinking_effort=role.thinking_effort,
                parameters_fingerprint=canonical_sha256(
                    [
                        "oamb-live-model-parameters-initial-v1",
                        role.binding_hash,
                        role.temperature,
                        role.top_p,
                        role.max_output_tokens,
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
    limits = plan.limits
    case_count, source_count = _workload_counts(cell.selection)
    provider_role_id = _PROVIDER_ROLE_BY_CELL[cell.cell_id]
    role_attempts: dict[str, int] = {
        provider_role_id: source_count,
        "embedding": source_count + case_count,
        "answer": case_count,
        "judge": case_count,
    }
    input_weights: dict[str, int] = {
        provider_role_id: 50,
        "embedding": 25,
        "answer": 15,
        "judge": 10,
    }
    output_weights: dict[str, int] = {
        provider_role_id: 50,
        "embedding": 0,
        "answer": 25,
        "judge": 25,
    }
    cost_per_budgeted_attempt = Decimal(limits.max_cost) / limits.max_budgeted_attempts
    role_ceilings_list: list[RoleBudgetCeiling] = []
    for role_id, binding in binding_by_role_id.items():
        max_attempts = role_attempts[role_id]
        role_cost = cost_per_budgeted_attempt * max_attempts
        owns_wall = role_id in {"answer", "judge"}
        max_wall = Decimal(max_attempts * limits.model_call_timeout_seconds if owns_wall else 0)
        role_ceilings_list.append(
            RoleBudgetCeiling(
                role_binding_id=binding.binding_id,
                max_attempts=max_attempts,
                max_input_tokens=limits.max_input_tokens * input_weights[role_id] // 100,
                max_output_tokens=limits.max_output_tokens * output_weights[role_id] // 100,
                max_dispatch_wall_seconds=max_wall,
                max_cost=role_cost,
                currency=limits.currency,
                price_snapshot_id=None,
                resource_ceilings=(
                    ResourceBudgetCeiling(
                        dimension_id="provider_request_wall_seconds_v1",
                        maximum=max_wall,
                        unit="seconds",
                    ),
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
        "memory_ingest": source_count,
        "memory_readiness": case_count,
        "memory_projection": case_count,
        "pre_query_projection": case_count,
        "memory_query": case_count,
        "post_query_projection": case_count,
    }
    operations: list[ProviderOperationBudgetCeiling] = []
    routes: list[DispatchBudgetRoute] = []
    producer_id = binding_by_role_id[_PROVIDER_ROLE_BY_CELL[cell.cell_id]].binding_id
    embedding_id = binding_by_role_id["embedding"].binding_id
    for stage in _MEMORY_STAGES:
        operation_id = f"{cell.adapter_profile_id}:{stage}"
        max_attempts = operation_attempts[stage]
        max_wall = Decimal(max_attempts * limits.memory_operation_timeout_seconds)
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
    budget_fields = {
        "budget_id": canonical_sha256(["oamb-live-budget-initial-v1", run_id]),
        "scope_kind": BudgetScopeKindV3.RUN,
        "scope_id": run_id,
        "max_attempts": limits.max_budgeted_attempts,
        "max_input_tokens": limits.max_input_tokens,
        "max_output_tokens": limits.max_output_tokens,
        "max_dispatch_wall_seconds": Decimal(limits.total_wall_time_seconds),
        "max_cost": Decimal(limits.max_cost),
        "currency": limits.currency,
        "resource_ceilings": (
            ResourceBudgetCeiling(
                dimension_id="provider_request_wall_seconds_v1",
                maximum=Decimal(limits.total_wall_time_seconds),
                unit="seconds",
            ),
        ),
        "role_ceilings": role_ceilings,
        "provider_operation_ceilings": tuple(operations),
        "dispatch_routes": tuple(routes),
        "stop_condition_ids": ("budget_exhausted", "identity_drift", "unknown_outcome"),
    }
    return BudgetSpecV4.model_validate(
        {"budget_hash": budget_spec_v4_hash(budget_fields), **budget_fields}
    )


def _workload_counts(selection: str) -> tuple[int, int]:
    if selection == "lme6":
        return len(LME6_EXPECTED_QUESTION_IDS), LME6_EXPECTED_SESSION_COUNT
    if selection == "lme30":
        return len(LME30_EXPECTED_QUESTION_IDS), LME30_EXPECTED_SESSION_COUNT
    raise LiveConfigurationError(f"unsupported live workload selection: {selection}")


__all__ = [
    "LiveCell",
    "LiveCellCompletion",
    "LiveCellExecutionError",
    "LiveConfigurationError",
    "build_live_cell",
    "execute_live_cell",
    "execute_live_cells",
    "load_live_environment",
    "load_live_provider_evidence",
    "select_live_cells",
]
