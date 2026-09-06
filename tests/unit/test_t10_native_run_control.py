from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

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
from oamb.runtime.native_run import NativeRunControl
from oamb.runtime.resume import ProviderLifecycleBridge

NOW = datetime(2026, 8, 30, 8, 0, tzinfo=UTC)


def _sha(label: str) -> str:
    return canonical_sha256([label])


def _source(
    identity: str,
    *,
    source_schema_versions: tuple[str, ...] = ("provider_service_evidence_manifest@1",),
) -> SourceEvidenceBinding:
    return SourceEvidenceBinding(
        binding_id=_sha(f"binding:{identity}"),
        source_kind=SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity=identity,
        source_root_hash=_sha(f"root:{identity}"),
        validation_result_hash=_sha(f"validation:{identity}"),
        source_schema_versions=source_schema_versions,
    )


def _control(
    *,
    run_id: str = "live-run",
    dataset_manifest_hash: str | None = None,
    case_manifest_hash: str | None = None,
    workload_id: str = "fixture-workload-v1",
    memory_system_id: str = "fixture-memory-v1",
    runtime_binding_hash: str | None = None,
    adapter_profile_id: str = "fixture-rest-v1",
    adapter_profile_hash: str | None = None,
    answer_role_binding_id: str = "answer-binding",
    judge_role_binding_id: str | None = None,
    provider_runtime_directory: Path | None = None,
    bounded_cost: bool = False,
    code_revision: str = "fixture-revision",
) -> NativeRunControl:
    role = ModelRoleBindingV2(
        binding_id=answer_role_binding_id,
        role=ModelRole.ANSWER,
        role_status=RoleBindingStatus.SELECTED,
        execution_owner=ExecutionOwner.HARNESS,
        binding_kind=BindingKind.MODEL_CLIENT,
        provider="openai_chat",
        endpoint_reference="LLM_BASE_URL",
        credential_variable_name="LLM_API_KEY",
        model="deepseek-chat",
        thinking_effort="low",
        parameters_fingerprint=_sha("parameters"),
        retry_policy_id="no-retry-v1",
        configuration_fingerprint=_sha("role-configuration"),
        redacted_endpoint_fingerprint=_sha("endpoint"),
    )
    judge_role = (
        role.model_copy(
            update={
                "binding_id": judge_role_binding_id,
                "role": ModelRole.JUDGE,
                "thinking_effort": "high",
                "configuration_fingerprint": _sha("judge-role-configuration"),
                "redacted_endpoint_fingerprint": _sha("judge-endpoint"),
            }
        )
        if judge_role_binding_id is not None
        else None
    )
    role_bindings = (role, *((judge_role,) if judge_role is not None else ()))
    memory_stages = (
        "runtime_resolve",
        "scope_allocate",
        "memory_ingest",
        "memory_readiness",
        "memory_projection",
        "pre_query_projection",
        "memory_query",
        "post_query_projection",
    )
    provider_operations = tuple(
        ProviderOperationBudgetCeiling.model_validate(
            {
                "provider_operation_ceiling_hash": provider_operation_budget_ceiling_hash(
                    provider_operation_fields
                ),
                **provider_operation_fields,
            }
        )
        for stage in memory_stages
        for provider_operation_fields in (
            dict(
                provider_operation_ceiling_id=f"{stage}-operation",
                adapter_profile_id=adapter_profile_id,
                operation_kind=stage,
                billing_unit="request",
                maximum_accepted_units=Decimal("20"),
                max_attempts=20,
                max_dispatch_wall_seconds=Decimal("600"),
                resource_ceilings=(
                    ResourceBudgetCeiling(
                        dimension_id="provider_request_wall_seconds_v1",
                        maximum=Decimal("600"),
                        unit="seconds",
                    ),
                ),
            ),
        )
    )
    provider_routes = tuple(
        DispatchBudgetRoute.model_validate(
            {
                "route_hash": dispatch_budget_route_hash(provider_route_fields),
                **provider_route_fields,
            }
        )
        for stage, operation in zip(memory_stages, provider_operations, strict=True)
        for provider_route_fields in (
            dict(
                route_id=stage,
                stage=stage,
                dispatch_owner_kind=DispatchBudgetOwnerKind.PROVIDER_OPERATION,
                dispatch_model_role_binding_id=None,
                provider_operation_ceiling_id=operation.provider_operation_ceiling_id,
                adapter_profile_id=adapter_profile_id,
                operation_kind=stage,
                billing_unit="request",
                internal_usage_role_binding_ids=(),
            ),
        )
    )
    answer_route_fields = dict(
        route_id="answer",
        stage="answer",
        dispatch_owner_kind=DispatchBudgetOwnerKind.MODEL_ROLE,
        dispatch_model_role_binding_id=role.binding_id,
        provider_operation_ceiling_id=None,
        adapter_profile_id=None,
        operation_kind="chat_completion",
        billing_unit="request",
        internal_usage_role_binding_ids=(),
    )
    direct_routes: tuple[DispatchBudgetRoute, ...] = (
        DispatchBudgetRoute.model_validate(
            {
                "route_hash": dispatch_budget_route_hash(answer_route_fields),
                **answer_route_fields,
            }
        ),
    )
    if judge_role is not None:
        judge_route_fields = dict(
            route_id="judge",
            stage="judge",
            dispatch_owner_kind=DispatchBudgetOwnerKind.MODEL_ROLE,
            dispatch_model_role_binding_id=judge_role.binding_id,
            provider_operation_ceiling_id=None,
            adapter_profile_id=None,
            operation_kind="chat_completion",
            billing_unit="request",
            internal_usage_role_binding_ids=(),
        )
        direct_routes = (
            *direct_routes,
            DispatchBudgetRoute.model_validate(
                {
                    "route_hash": dispatch_budget_route_hash(judge_route_fields),
                    **judge_route_fields,
                }
            ),
        )
    budget_fields = dict(
        budget_id="run-budget",
        scope_kind=BudgetScopeKindV3.RUN,
        scope_id=run_id,
        max_attempts=100,
        max_input_tokens=100_000,
        max_output_tokens=10_000,
        max_dispatch_wall_seconds=Decimal("3000"),
        max_cost=Decimal("20") if bounded_cost else None,
        currency="USD" if bounded_cost else None,
        resource_ceilings=(
            ResourceBudgetCeiling(
                dimension_id="provider_request_wall_seconds_v1",
                maximum=Decimal("3000"),
                unit="seconds",
            ),
        ),
        role_ceilings=tuple(
            RoleBudgetCeiling(
                role_binding_id=selected_role.binding_id,
                max_attempts=10,
                max_input_tokens=100_000,
                max_output_tokens=10_000,
                max_dispatch_wall_seconds=Decimal("600"),
                max_cost=Decimal("10") if bounded_cost else None,
                currency="USD" if bounded_cost else None,
                price_snapshot_id=None,
                resource_ceilings=(
                    ResourceBudgetCeiling(
                        dimension_id="provider_request_wall_seconds_v1",
                        maximum=Decimal("600"),
                        unit="seconds",
                    ),
                ),
                provider_budget_cap=ProviderBudgetCap(
                    provider="openai_chat",
                    operation_kind="chat_completion",
                    billing_unit="request",
                    maximum_accepted_units=Decimal("10"),
                ),
            )
            for selected_role in role_bindings
        ),
        provider_operation_ceilings=provider_operations,
        dispatch_routes=(*provider_routes, *direct_routes),
        stop_condition_ids=("identity_drift", "budget_exhausted"),
    )
    budget = BudgetSpecV4.model_validate(
        {
            "budget_hash": budget_spec_v4_hash(budget_fields),
            **budget_fields,
        }
    )
    run_spec = RunSpec(
        run_id=run_id,
        protocol_id="oamb-native-v1",
        dataset_manifest_hash=dataset_manifest_hash or _sha("dataset"),
        case_manifest_hash=case_manifest_hash or _sha("subset"),
        workload_id=workload_id,
        memory_system_id=memory_system_id,
        runtime_binding_hash=runtime_binding_hash or _sha("runtime"),
        environment_hash=_sha("environment"),
        model_role_binding_ids=tuple(item.binding_id for item in role_bindings),
        budget_id=budget.budget_id,
        code_revision=code_revision,
        normalizer_fingerprint=_sha("normalizer"),
    )
    preflight_fields = dict(
        run_id=run_spec.run_id,
        observed_at=NOW + timedelta(minutes=1),
        resolved_plan_hash=_sha("resolved-plan"),
        run_spec_hash=canonical_sha256(run_spec),
        dataset_manifest_hash=run_spec.dataset_manifest_hash,
        subset_manifest_hash=run_spec.case_manifest_hash,
        adapter_profile_id=adapter_profile_id,
        adapter_profile_hash=adapter_profile_hash or _sha("adapter-profile"),
        provider_project_id="oamb-providers-fixture1",
        provider_profile_id=adapter_profile_id,
        runtime_binding_hash=run_spec.runtime_binding_hash,
        provider_service_evidence=_source("fixture-model-readiness"),
        provider_profile_evidence=_source("fixture-provider-profile"),
        role_binding_ids=tuple(item.binding_id for item in role_bindings),
        dispatch_routes=budget.dispatch_routes,
        budget_hash=canonical_sha256(budget),
        redacted_endpoint_fingerprints=tuple(
            item.redacted_endpoint_fingerprint for item in role_bindings
        ),
        credential_reference_fingerprints=tuple(
            _sha(f"LLM_API_KEY:{item.binding_id}") for item in role_bindings
        ),
        artifact_repository_fingerprint=_sha("artifact-root"),
        artifact_durability_proof_hash=_sha("durability"),
    )
    preflight = RunPreflightRecord.model_validate(
        {
            "preflight_record_hash": run_preflight_record_hash(preflight_fields),
            **preflight_fields,
        }
    )
    return NativeRunControl(
        run_spec=run_spec,
        preflight_record=preflight,
        budget=budget,
        role_bindings=role_bindings,
        owner_id="operator-process-1",
        host_fingerprint=_sha("host"),
        process_id=1234,
        provider_runtime_directory=(
            provider_runtime_directory or Path("/tmp/oamb-provider-runtime-fixture")
        ),
        wall_clock=lambda: NOW + timedelta(minutes=2),
        monotonic_clock=lambda: 1.0,
    )


def test_live_native_control_closes_all_durable_identities() -> None:
    control = _control()

    assert control.preflight_record.run_spec_hash == canonical_sha256(control.run_spec)
    assert control.preflight_record.budget_hash == canonical_sha256(control.budget)
    assert control.budget.schema_version == 4


def test_live_cell_deadline_cancels_the_active_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from oamb.runtime import native_run

    cancelled = asyncio.Event()

    async def never_finishes(**_kwargs: Any) -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(native_run, "_run_native_vertical_slice", never_finishes)
    request = type(
        "DeadlineRequest",
        (),
        {
            "control": type(
                "DeadlineControl",
                (),
                {
                    "budget": type(
                        "DeadlineBudget",
                        (),
                        {"max_dispatch_wall_seconds": Decimal("0.01")},
                    )()
                },
            )(),
            "output_root": Path("/tmp/not-used"),
            "run_id": "run",
            "adapter_profile_id": "adapter",
            "workload": object(),
            "visible_evidence_policy": object(),
            "artifact_store_factory": object(),
            "memory_factory": object(),
            "model_factory": object(),
            "answer_role_binding_id": "answer",
            "judge_model_factory": None,
            "judge_role_binding_id": None,
            "close_timeout_seconds": 1.0,
            "lease": None,
            "provider_lifecycle": None,
        },
    )()

    async def scenario() -> None:
        with pytest.raises(TimeoutError):
            await native_run._run_native_with_cell_deadline(request, None)
        assert cancelled.is_set()

    asyncio.run(scenario())


def test_live_reservation_derives_token_time_resource_and_cost_from_budget() -> None:
    from oamb.runtime.native_run import _reservation_allocations_for_route

    control = _control(bounded_cost=True)
    route = next(item for item in control.budget.dispatch_routes if item.stage == "answer")

    evidence, runtime, maximum = _reservation_allocations_for_route(
        control.budget,
        route,
        maximum_output_tokens=512,
    )

    assert len(evidence) == len(runtime) == 1
    assert evidence[0].allocated_input_tokens == 10_000
    assert evidence[0].allocated_output_tokens == 512
    assert evidence[0].allocated_dispatch_wall_seconds == Decimal("60")
    assert evidence[0].allocated_cost == Decimal("1")
    assert evidence[0].allocated_resource_ceilings[0].maximum == Decimal("60")
    assert maximum == runtime[0].maximum


def test_live_native_control_rejects_role_or_scope_drift() -> None:
    control = _control()

    with pytest.raises(ValueError, match="role inventory"):
        replace(control, role_bindings=())


def test_run_release_verifies_terminal_manifest_before_clearing_pointer(
    tmp_path: Path,
) -> None:
    """Catches clearing lifecycle ownership before the terminal capsule is durable."""

    bridge = ProviderLifecycleBridge(tmp_path)
    authority = bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )
    observed_pointer_states: list[bool] = []

    bridge.release_run(
        authority,
        seal_and_verify_terminal_manifest=lambda: observed_pointer_states.append(
            (tmp_path / "active-operation").is_file()
        ),
    )

    assert observed_pointer_states == [True]
    assert not (tmp_path / "active-operation").exists()


def test_run_release_keeps_pointer_when_terminal_manifest_verification_fails(
    tmp_path: Path,
) -> None:
    """Catches losing recovery ownership when final capsule verification fails."""

    bridge = ProviderLifecycleBridge(tmp_path)
    authority = bridge.acquire_run(
        run_id="run-a",
        provider_project="oamb-providers-test-a",
        profile_id="mem0-rest-v1",
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )

    def reject_terminal() -> None:
        raise RuntimeError("terminal capsule is not durable")

    with pytest.raises(RuntimeError, match="not durable"):
        bridge.release_run(
            authority,
            seal_and_verify_terminal_manifest=reject_terminal,
        )

    assert (tmp_path / "active-operation").is_file()


def _prepared_live_provider_attempt(
    tmp_path: Path, *, stage: str
) -> tuple[Any, Any, ProviderLifecycleBridge, Any, Path]:
    from oamb.artifacts.store import ArtifactStore
    from oamb.runtime.native_run import (
        _live_budget_ledger,
        _NativeExecutionState,
        _prepare_native_attempt,
    )

    runtime = tmp_path / "provider-runtime"
    control = _control(provider_runtime_directory=runtime)
    lifecycle = ProviderLifecycleBridge(runtime)
    authority = lifecycle.acquire_run(
        run_id=control.run_spec.run_id,
        provider_project=control.preflight_record.provider_project_id,
        profile_id=control.preflight_record.adapter_profile_id,
        lease_epoch=1,
        lease_record_hash="a" * 64,
    )
    state = _NativeExecutionState(
        store=ArtifactStore(tmp_path / "capsule"),
        run_id=control.run_spec.run_id,
        lease_record_hash="a" * 64,
        close_timeout_seconds=1,
        owner_id=control.owner_id,
        budget_id=control.budget.budget_id,
        budget_scope_id=control.run_spec.run_id,
        started_at=NOW,
        control=control,
        budget_ledger=_live_budget_ledger(control),
        provider_lifecycle=lifecycle,
    )
    attempt_identity = _sha(f"attempt:{stage}")
    prepared = _prepare_native_attempt(
        state,
        attempt_identity=attempt_identity,
        parent_kind="ingestion_plan",
        parent_id=_sha(f"parent:{stage}"),
        stage=stage,
        ordinal=1,
        request_fingerprint=_sha(f"request:{stage}"),
        role_binding_id=control.preflight_record.adapter_profile_id,
    )
    pointer = runtime / "active-provider-attempts" / f"{attempt_identity}.json"
    assert pointer.is_file()
    return state, prepared, lifecycle, authority, pointer


def test_live_success_accounting_clears_the_exact_provider_attempt_pointer(
    tmp_path: Path,
) -> None:
    """Catches clearing a live success with an obsolete lifecycle-call signature."""

    from oamb.contracts.accounting import IndexingView
    from oamb.runtime.native_run import _seal_live_native_attempt_accounting

    state, prepared, lifecycle, authority, pointer = _prepared_live_provider_attempt(
        tmp_path,
        stage="scope_allocate",
    )

    _seal_live_native_attempt_accounting(
        state,
        prepared,
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
        raw_response_ref=_sha("scope-response"),
        usage_record_ids=(),
        inline_usage_records=(),
        indexing_view=IndexingView.NOT_APPLICABLE,
    )

    assert not pointer.exists()
    lifecycle.release_run(authority)


def test_live_cancelled_before_dispatch_clears_the_exact_provider_attempt_pointer(
    tmp_path: Path,
) -> None:
    """Catches leaving a guarded attempt active after proven pre-dispatch cancellation."""

    from oamb.contracts.ports import MemorySystemCallCancelledBeforeDispatch
    from oamb.contracts.states import AttemptOutcome
    from oamb.runtime.native_run import _seal_native_failure

    state, prepared, lifecycle, authority, pointer = _prepared_live_provider_attempt(
        tmp_path,
        stage="scope_allocate",
    )

    terminal = _seal_native_failure(
        state,
        prepared,
        started_at=NOW,
        error=MemorySystemCallCancelledBeforeDispatch("cancelled before provider dispatch"),
    )

    assert terminal.outcome == AttemptOutcome.CANCELLED
    assert not pointer.exists()
    lifecycle.release_run(authority)
