from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from oamb.artifacts.store import ArtifactStore
from oamb.config.load import EnvironmentReference
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    BudgetScopeKindV2,
    DispatchBudgetOwnerKind,
    DispatchBudgetRoute,
    SourceEvidenceBinding,
    SourceEvidenceKind,
    dispatch_budget_route_hash,
)
from oamb.runtime.preflight import (
    AdapterProfileDescriptor,
    ArtifactDurabilityPreflight,
    GateStatus,
    OperationKind,
    ProviderGateClosure,
    ResolutionStatus,
    ResolvedRunPlan,
    RoleSlot,
    RoleSlotName,
    TransportKind,
    build_run_preflight_record,
    seal_run_preflight_record,
)

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


class _Contract:
    schema_version = 2

    def __init__(self, **values: object) -> None:
        self.__dict__.update(values)

    def model_dump(self, *, mode: str = "python") -> dict[str, object]:
        del mode
        return dict(self.__dict__)


def _source(identity: str, root_hash: str) -> SourceEvidenceBinding:
    return SourceEvidenceBinding(
        binding_id=canonical_sha256([identity, root_hash]),
        source_kind=SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity=identity,
        source_root_hash=root_hash,
        validation_result_hash=THIRD_HASH,
        source_schema_versions=("provider_service_evidence_manifest@1",),
    )


def _route() -> DispatchBudgetRoute:
    values = dict(
        route_id="mem0-ingest-route",
        stage="memory_ingest",
        dispatch_owner_kind=DispatchBudgetOwnerKind.PROVIDER_OPERATION,
        dispatch_model_role_binding_id=None,
        provider_operation_ceiling_id="mem0-ingest-cap",
        adapter_profile_id="mem0-rest-v2.0.19",
        operation_kind="memory_ingest",
        billing_unit="request",
        internal_usage_role_binding_ids=("mem0-extraction", "controlled-embedding"),
    )
    return DispatchBudgetRoute.model_validate(
        {
            "route_hash": dispatch_budget_route_hash(values),
            **values,
        }
    )


def test_build_and_seal_run_preflight_record_binds_resolved_live_closure(
    tmp_path: Path,
) -> None:
    roles = (
        RoleSlot(
            role=RoleSlotName.EXTRACTION,
            status=ResolutionStatus.RESOLVED,
            binding=_Contract(binding_id="mem0-extraction"),
            credential_reference=EnvironmentReference("MEM0_LLM_API_KEY"),
            evidence_reference="role:mem0-extraction",
        ),
        RoleSlot(
            role=RoleSlotName.EMBEDDING,
            status=ResolutionStatus.RESOLVED,
            binding=_Contract(binding_id="controlled-embedding"),
            credential_reference=None,
            evidence_reference="role:controlled-embedding",
        ),
        RoleSlot(
            role=RoleSlotName.ANSWER,
            status=ResolutionStatus.RESOLVED,
            binding=_Contract(binding_id="answer"),
            credential_reference=EnvironmentReference("DEEPSEEK_API_KEY"),
            evidence_reference="role:answer",
        ),
        RoleSlot(
            role=RoleSlotName.JUDGE,
            status=ResolutionStatus.RESOLVED,
            binding=_Contract(binding_id="judge"),
            credential_reference=EnvironmentReference("DEEPSEEK_API_KEY"),
            evidence_reference="role:judge",
        ),
        RoleSlot(
            role=RoleSlotName.QUALITY_REVIEW,
            status=ResolutionStatus.UNSELECTED,
            binding=None,
            credential_reference=None,
            evidence_reference="not selected",
        ),
    )
    plan = ResolvedRunPlan(
        operation_kind=OperationKind.BENCHMARK_RUN,
        budget_spec=_Contract(
            budget_id="run-budget",
            scope_kind=BudgetScopeKindV2.RUN,
        ),
        role_slots=roles,
        adapter_profile=AdapterProfileDescriptor(
            profile_id="mem0-rest-v2.0.19",
            memory_system_id="mem0",
            transport_kind=TransportKind.REST_API,
            controlled_embedding=None,
            native_reranking_disabled=True,
            oamb_reranker_configured=False,
            provider_project_id="oamb-t10-mem0-lme6",
            release_version="2.0.19",
            source_revision="dc82354",
            build_artifact_sha256=HASH,
        ),
        provider_gates=ProviderGateClosure(*(GateStatus.PASS,) * 5),
        artifact_durability=ArtifactDurabilityPreflight(
            artifact_root_fingerprint=OTHER_HASH,
            available_bytes=1_000_000,
            lease_supported=True,
            file_fsync_supported=True,
            directory_fsync_supported=True,
            no_replace_supported=True,
        ),
        runtime_binding=_Contract(runtime_binding_hash=THIRD_HASH),
        provider_runtime_attestation=None,
        approval=_Contract(approval_hash=HASH),
        cost_measurement_spec=_Contract(measurement_spec_id="cost-v1"),
        price_snapshot=None,
        execution_environment_binding=_Contract(environment_hash=OTHER_HASH),
        schema_versions=(1, 2),
        plan_hash=HASH,
    )

    record = build_run_preflight_record(
        plan=plan,
        run_id="t10-mem0-lme6-run",
        observed_at=NOW,
        run_spec_hash=OTHER_HASH,
        dataset_manifest_hash=THIRD_HASH,
        subset_manifest_hash=HASH,
        adapter_profile_hash=OTHER_HASH,
        provider_service_evidence=_source("mem0-model-readiness", OTHER_HASH),
        memory_conformance_evidence=_source("mem0-memory-conformance", HASH),
        dispatch_routes=(_route(),),
        redacted_endpoint_fingerprints=(HASH,),
        credential_reference_fingerprints=(OTHER_HASH,),
    )
    store = ArtifactStore(tmp_path / "capsule")

    seal_run_preflight_record(store, record)

    sealed = (store.root / "source/specs/run-preflight.json").read_bytes()
    assert record.preflight_record_hash.encode() in sealed
