"""Bounded execution for an approved phase AI review plan."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import NoReturn

from oamb.artifacts.atomic import read_regular_file
from oamb.artifacts.store import ArtifactStore
from oamb.config.load import EnvironmentReference
from oamb.contracts.accounting import (
    AggregationOperator,
    CostBasis,
    CostMeasurementSpec,
    CostRecord,
    IndexingView,
    PriceSnapshot,
    ProofStatus,
    ResourceUsageRecord,
    TokenDomain,
    TokenMeasurementSource,
    TokenStageV2,
    TokenUsageRecordV2,
)
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptReceiptKind,
    AttemptReceiptRecord,
    AttemptRecordV2,
    BudgetReservationRecord,
    CloseErrorRecord,
    OccurrenceClaimRecord,
    PhaseReviewOccurrenceRecordV2,
    RunLeaseRecord,
    phase_review_occurrence_id_v2,
)
from oamb.contracts.ids import attempt_id as derive_attempt_id
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import (
    ArtifactWriteRequest,
    ModelCallFailure,
    ModelClientPort,
    ModelReceipt,
    ModelRequest,
    RawPayloadSealRequest,
)
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    AIReviewBatchResult,
    AIReviewIntegrityResult,
    EvaluationReviewBundle,
)
from oamb.contracts.specifications import (
    AIReviewPlan,
    BudgetScopeKindV2,
    BudgetSpecV2,
    ExecutionEnvironmentBinding,
    ExternalCallApprovalRecord,
    ModelRoleBindingV2,
    ResourceBudgetCeiling,
)
from oamb.contracts.states import AttemptOutcome, IndexContribution
from oamb.model_clients.openai_compatible import OpenAICompatibleModelClient
from oamb.phase_review_profiles import (
    FAKE_PHASE_REVIEW_CLIENT_KIND,
    FAKE_PHASE_REVIEW_CONFIGURED_MODEL,
    FAKE_PHASE_REVIEW_COST_REASON,
    FAKE_PHASE_REVIEW_CREDENTIAL_VARIABLE,
    FAKE_PHASE_REVIEW_ENDPOINT,
    FAKE_PHASE_REVIEW_PROVIDER,
    FAKE_PHASE_REVIEW_RESOLVED_MODEL,
    FAKE_PHASE_REVIEW_WALL_MEASUREMENT_SOURCE,
    OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND,
    OPENAI_PHASE_REVIEW_WALL_MEASUREMENT_SOURCE,
    PHASE_REVIEW_CLIENT_KINDS,
    PHASE_REVIEW_INPUT_TOKEN_PRICE_CLASS,
    PHASE_REVIEW_OUTPUT_TOKEN_PRICE_CLASS,
    PHASE_REVIEW_TOKEN_PRICE_UNIT_SCALE,
    PHASE_REVIEW_WALL_DIMENSION_ID,
    PHASE_REVIEW_WALL_UNIT,
)
from oamb.phase_review_profiles import (
    phase_review_runtime_hash as phase_review_profile_runtime_hash,
)
from oamb.phase_review_repository import (
    phase_review_occurrence_root,
    phase_review_runtime_directory,
    prepare_artifact_repository,
)
from oamb.reporting.human_review import validate_ai_review_history
from oamb.reporting.review import (
    ai_review_request_fingerprint,
    parse_ai_review_batch_output,
    parse_ai_review_integrity_output,
    reduce_ai_quality_review,
)
from oamb.runtime.attempts import AttemptCoordinator
from oamb.runtime.budget import BudgetAmount, BudgetCeiling, BudgetLedger
from oamb.runtime.fake_run import _probe_artifact_durability
from oamb.runtime.preflight import (
    OperationKind,
    ResolutionStatus,
    RoleSlot,
    RoleSlotName,
    RunPreflightRequest,
    resolve_run_plan,
)
from oamb.runtime.resume import (
    LeaseJournal,
    ProviderLifecycleBridge,
    host_identity_fingerprint,
)
from oamb.workloads.visible_evidence import o200k_encoding

_FAKE_ENVIRONMENT_VALUE = "offline-fixture-selection"
_PHASE_REVIEW_OWNER_ID = "oamb-phase-review-runner-v1"
_QUALITY_REVIEW_STAGE = "quality_review"


@dataclass(frozen=True, slots=True)
class PhaseReviewRunResult:
    lease_record: RunLeaseRecord
    occurrence_claims: tuple[OccurrenceClaimRecord, ...]
    budget_reservations: tuple[BudgetReservationRecord, ...]
    attempt_intents: tuple[AttemptIntentRecord, ...]
    attempt_receipts: tuple[AttemptReceiptRecord, ...]
    occurrence: PhaseReviewOccurrenceRecordV2
    batch_results: tuple[AIReviewBatchResult, ...]
    integrity_result: AIReviewIntegrityResult
    attempts: tuple[AttemptRecordV2, ...]
    usage_records: tuple[TokenUsageRecordV2, ...]
    resource_records: tuple[ResourceUsageRecord, ...]
    cost_records: tuple[CostRecord, ...]
    ai_record: AIQualityReviewRecord


@dataclass(frozen=True, slots=True)
class PartialPhaseReviewRunEvidence:
    occurrence: PhaseReviewOccurrenceRecordV2
    batch_results: tuple[AIReviewBatchResult, ...]
    integrity_result: AIReviewIntegrityResult | None
    attempts: tuple[AttemptRecordV2, ...]
    usage_records: tuple[TokenUsageRecordV2, ...]
    resource_records: tuple[ResourceUsageRecord, ...]
    cost_records: tuple[CostRecord, ...]


class PhaseReviewRunFailure(RuntimeError):
    def __init__(self, message: str, partial: PartialPhaseReviewRunEvidence) -> None:
        super().__init__(message)
        self.partial = partial


class _FakePhaseReviewModelClient:
    def __init__(self, store: ArtifactStore, responses: tuple[str, ...]) -> None:
        self._store = store
        self._responses = responses
        self._dispatched = 0
        self._closed = False
        self.usage_records: list[TokenUsageRecordV2] = []

    async def complete(self, request: ModelRequest) -> ModelReceipt:
        if self._closed:
            raise RuntimeError("fake phase-review client is closed")
        if self._dispatched >= len(self._responses):
            raise RuntimeError("fake phase-review response inventory is exhausted")
        output = self._responses[self._dispatched]
        self._dispatched += 1
        raw_payload = canonical_json_bytes(
            {
                "attempt_id": request.attempt_id,
                "messages": request.messages,
                "output_text": output,
                "runtime_model": FAKE_PHASE_REVIEW_RESOLVED_MODEL,
            }
        )
        raw_sha256 = hashlib.sha256(raw_payload).hexdigest()
        raw_reference = self._store.seal_raw(
            RawPayloadSealRequest(
                sha256=raw_sha256,
                media_type="application/json",
                compression="gzip",
                payload_bytes=raw_payload,
            )
        )
        request_text = canonical_json_bytes({"messages": request.messages}).decode("utf-8")
        input_tokens = len(o200k_encoding().encode_ordinary(request_text))
        output_tokens = len(o200k_encoding().encode_ordinary(output))
        usage_id = canonical_sha256(
            [
                "oamb-fake-phase-review-usage-v1",
                request.attempt_id,
                input_tokens,
                output_tokens,
                raw_sha256,
            ]
        )
        usage = TokenUsageRecordV2(
            usage_record_id=usage_id,
            attempt_id=request.attempt_id,
            parent_kind="phase_review",
            parent_id=request.parent_id,
            stage=TokenStageV2.QUALITY_REVIEW,
            operation_kind=_QUALITY_REVIEW_STAGE,
            token_domain=TokenDomain.EXTERNAL_LLM,
            measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
            input_tokens=input_tokens,
            visible_output_tokens=output_tokens,
            supplier_reported_total_tokens=input_tokens + output_tokens,
            context_view_tokens=None,
            proof_status=ProofStatus.MEASURED_COMPLETE,
            reason=None,
            raw_response_ref=raw_sha256,
        )
        _seal_source_record(self._store, "usage", usage_id, usage)
        self.usage_records.append(usage)
        return ModelReceipt(
            raw_reference=raw_reference,
            output_text=output,
            usage_reference_ids=(usage_id,),
            raw_response_bytes=raw_payload,
            runtime_model=FAKE_PHASE_REVIEW_RESOLVED_MODEL,
            runtime_identity_status="recorded",
        )

    async def close(self) -> None:
        self._closed = True


def phase_review_model_hash(role: ModelRoleBindingV2) -> str:
    return canonical_sha256(
        [
            "oamb-phase-review-model-binding-v1",
            role.configured_model,
            role.resolved_model,
            role.parameters_fingerprint,
        ]
    )


def phase_review_runtime_hash(
    role: ModelRoleBindingV2,
    environment: ExecutionEnvironmentBinding,
    *,
    client_kind: str = FAKE_PHASE_REVIEW_CLIENT_KIND,
) -> str:
    return phase_review_profile_runtime_hash(
        role.redacted_endpoint_fingerprint,
        environment.environment_hash,
        client_kind=client_kind,
    )


def run_phase_review(
    *,
    client_kind: str,
    bundle: EvaluationReviewBundle,
    plan: AIReviewPlan,
    occurrence: PhaseReviewOccurrenceRecordV2,
    approval: ExternalCallApprovalRecord,
    budget: BudgetSpecV2,
    role: ModelRoleBindingV2,
    cost_measurement_spec: CostMeasurementSpec,
    execution_environment: ExecutionEnvironmentBinding,
    request_directory: Path,
    fake_responses: tuple[str, ...],
    artifact_repository: Path,
    started_at: datetime | None,
    ended_at: datetime | None,
    price_snapshot: PriceSnapshot | None = None,
    previous_ai_history: tuple[AIQualityReviewRecord, ...] = (),
) -> PhaseReviewRunResult:
    """Preflight the complete phase graph before credentials or client construction."""

    observed_at = started_at if started_at is not None else _utc_now()
    repository, repository_fingerprint = prepare_artifact_repository(artifact_repository)
    _validate_fixed_inputs(
        bundle=bundle,
        plan=plan,
        occurrence=occurrence,
        approval=approval,
        budget=budget,
        role=role,
        client_kind=client_kind,
        cost_measurement_spec=cost_measurement_spec,
        execution_environment=execution_environment,
        price_snapshot=price_snapshot,
        artifact_repository_fingerprint=repository_fingerprint,
        observed_at=observed_at,
        started_at=started_at,
        ended_at=ended_at,
        previous_ai_history=previous_ai_history,
    )
    requests = _load_requests(plan, request_directory)
    if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND:
        if len(fake_responses) != plan.expected_attempt_count:
            raise ValueError("fake response count does not match the approved AI review plan")
    elif fake_responses:
        raise ValueError("OpenAI-compatible review cannot consume fake response files")
    durability = _probe_artifact_durability(repository)
    resolve_run_plan(
        RunPreflightRequest(
            operation_kind=OperationKind.PHASE_REVIEW,
            budget_spec=budget,
            role_slots=_phase_role_slots(role),
            adapter_profile=None,
            provider_gates=None,
            artifact_durability=durability,
            runtime_binding=None,
            provider_runtime_attestation=None,
            approval=approval,
            cost_measurement_spec=cost_measurement_spec,
            price_snapshot=price_snapshot,
            environment={str(role.credential_variable_name): _FAKE_ENVIRONMENT_VALUE},
            observed_at=observed_at,
            execution_environment_binding=execution_environment,
        )
    )

    occurrence_root = phase_review_occurrence_root(
        repository, occurrence.phase_review_occurrence_id
    )
    store = ArtifactStore(occurrence_root)
    lifecycle = ProviderLifecycleBridge(phase_review_runtime_directory(repository))
    lease_journal = LeaseJournal(store, lifecycle)
    lease = _phase_review_lease(
        occurrence=occurrence,
        role=role,
        client_kind=client_kind,
        acquired_at=observed_at,
    )
    lease_journal.acquire(lease)
    budget_ledger = _phase_review_budget_ledger(budget)
    coordinator = AttemptCoordinator(
        store,
        budget_ledger,
        provider_lifecycle=lifecycle,
    )
    release_lease = True
    try:
        if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND:
            client: ModelClientPort = _build_fake_client(store, fake_responses)
        else:
            credential_name = role.credential_variable_name
            if credential_name is None:
                raise ValueError("OpenAI-compatible review role has no credential reference")
            api_key = os.environ.get(credential_name)
            if not api_key:
                raise ValueError(
                    "OpenAI-compatible review requires non-empty environment variable "
                    f"{credential_name}"
                )
            client = _build_openai_client(store, role, api_key)
        return asyncio.run(
            _execute_review(
                client=client,
                client_kind=client_kind,
                store=store,
                plan=plan,
                occurrence=occurrence,
                approval=approval,
                budget=budget,
                requests=requests,
                cost_measurement_spec=cost_measurement_spec,
                execution_environment=execution_environment,
                price_snapshot=price_snapshot,
                fake_started_at=started_at,
                fake_ended_at=ended_at,
                coordinator=coordinator,
                budget_ledger=budget_ledger,
                lease_record=lease,
                previous_ai_history=previous_ai_history,
            )
        )
    except PhaseReviewRunFailure as exc:
        if exc.partial.occurrence.state == "interrupted_unknown_outcome":
            release_lease = False
        raise
    finally:
        if release_lease:
            primary_error = sys.exception()
            if not _phase_review_release_ready(store, occurrence.phase_review_occurrence_id):
                message = (
                    "phase-review lease retained because terminal source evidence is incomplete"
                )
                if primary_error is None:
                    raise RuntimeError(message)
                primary_error.add_note(message)
            else:
                try:
                    lease_journal.release()
                except BaseException as release_error:
                    if primary_error is None:
                        raise
                    primary_error.add_note(
                        "phase-review lease release also failed: "
                        f"{type(release_error).__name__}: {release_error}"
                    )


def _validate_fixed_inputs(
    *,
    bundle: EvaluationReviewBundle,
    plan: AIReviewPlan,
    occurrence: PhaseReviewOccurrenceRecordV2,
    approval: ExternalCallApprovalRecord,
    budget: BudgetSpecV2,
    role: ModelRoleBindingV2,
    client_kind: str,
    cost_measurement_spec: CostMeasurementSpec,
    execution_environment: ExecutionEnvironmentBinding,
    price_snapshot: PriceSnapshot | None,
    artifact_repository_fingerprint: str,
    observed_at: datetime,
    started_at: datetime | None,
    ended_at: datetime | None,
    previous_ai_history: tuple[AIQualityReviewRecord, ...],
) -> None:
    if plan.review_bundle_hash != bundle.bundle_id:
        raise ValueError("AI review plan does not bind the supplied review bundle")
    _validate_previous_ai_history(bundle, plan, occurrence, previous_ai_history)
    if (
        occurrence.phase_id != bundle.phase_id
        or occurrence.review_bundle_hash != bundle.bundle_id
        or occurrence.reviewer_role_binding_hash != plan.reviewer_role_binding_hash
        or occurrence.approval_record_id != approval.approval_hash
        or occurrence.budget_id != budget.budget_id
        or occurrence.artifact_repository_fingerprint != artifact_repository_fingerprint
        or occurrence.state != "planned"
        or occurrence.started_at is not None
        or occurrence.ended_at is not None
    ):
        raise ValueError("planned phase-review occurrence does not bind its exact inputs")
    if client_kind not in PHASE_REVIEW_CLIENT_KINDS:
        raise ValueError("unknown phase-review client kind")
    common_role_mismatch = (
        role.binding_id != plan.reviewer_role_binding_hash
        or plan.reviewer_model_hash != phase_review_model_hash(role)
        or plan.reviewer_runtime_hash
        != phase_review_runtime_hash(role, execution_environment, client_kind=client_kind)
        or plan.reviewer_configuration_hash != role.configuration_fingerprint
    )
    fake_role_mismatch = client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND and (
        role.provider != FAKE_PHASE_REVIEW_PROVIDER
        or role.endpoint_reference != FAKE_PHASE_REVIEW_ENDPOINT
        or role.credential_variable_name != FAKE_PHASE_REVIEW_CREDENTIAL_VARIABLE
        or role.configured_model != FAKE_PHASE_REVIEW_CONFIGURED_MODEL
        or role.resolved_model != FAKE_PHASE_REVIEW_RESOLVED_MODEL
    )
    openai_role_mismatch = client_kind == OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND and (
        role.endpoint_reference is None
        or not role.endpoint_reference.startswith(("https://", "http://127.0.0.1:"))
        or role.credential_variable_name is None
    )
    if common_role_mismatch or fake_role_mismatch or openai_role_mismatch:
        raise ValueError("AI review plan does not bind the selected role/model/runtime")
    if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND:
        if (
            started_at is None
            or ended_at is None
            or started_at.tzinfo is None
            or ended_at.tzinfo is None
            or ended_at <= started_at
        ):
            raise ValueError("fake review requires an increasing timezone-aware fixture interval")
        if not (approval.approved_at <= started_at < ended_at <= approval.expires_at):
            raise ValueError("fake review interval is outside its approval window")
    elif started_at is not None or ended_at is not None:
        raise ValueError("OpenAI-compatible review captures timing and rejects supplied timestamps")
    if (
        len(cost_measurement_spec.dimensions) != 1
        or len(budget.role_ceilings) != 1
        or tuple(item.dimension_id for item in cost_measurement_spec.dimensions)
        != tuple(item.dimension_id for item in budget.resource_ceilings)
        or tuple(item.dimension_id for item in cost_measurement_spec.dimensions)
        != tuple(item.dimension_id for item in budget.role_ceilings[0].resource_ceilings)
    ):
        raise ValueError("cost measurement dimensions do not bind the phase-review budget")
    if any(
        item.stage != _QUALITY_REVIEW_STAGE
        or item.operation_kind != _QUALITY_REVIEW_STAGE
        or item.parent_kind != "phase_review"
        for item in cost_measurement_spec.dimensions
    ):
        raise ValueError("cost measurement spec is not phase-review scoped")
    measurement_dimension = cost_measurement_spec.dimensions[0]
    expected_measurement_source = (
        FAKE_PHASE_REVIEW_WALL_MEASUREMENT_SOURCE
        if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND
        else OPENAI_PHASE_REVIEW_WALL_MEASUREMENT_SOURCE
    )
    if (
        measurement_dimension.dimension_id != PHASE_REVIEW_WALL_DIMENSION_ID
        or measurement_dimension.unit != PHASE_REVIEW_WALL_UNIT
        or measurement_dimension.allowed_meter_sources != (expected_measurement_source,)
        or not measurement_dimension.required
        or measurement_dimension.price_class is not None
        or measurement_dimension.indexing_view_rule != IndexingView.NOT_APPLICABLE
        or measurement_dimension.aggregation_operator != AggregationOperator.INTERVAL_UNION
    ):
        raise ValueError("phase review requires its exact repository-owned wall-time meter")
    role_price_ids = {item.price_snapshot_id for item in budget.role_ceilings}
    if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND:
        if price_snapshot is not None or role_price_ids != {None}:
            raise ValueError("fake phase review cannot claim a supplier price snapshot")
    elif (
        price_snapshot is None
        or len(budget.role_ceilings) != 1
        or role_price_ids != {price_snapshot.price_snapshot_id}
        or price_snapshot.provider != role.provider
        or price_snapshot.provider != budget.role_ceilings[0].provider_budget_cap.provider
        or price_snapshot.currency != budget.currency
        or price_snapshot.currency != budget.role_ceilings[0].currency
        or price_snapshot.price_class_ids
        != (
            PHASE_REVIEW_INPUT_TOKEN_PRICE_CLASS,
            PHASE_REVIEW_OUTPUT_TOKEN_PRICE_CLASS,
        )
        or len(cost_measurement_spec.dimensions) != 1
        or cost_measurement_spec.dimensions[0].price_class is not None
    ):
        raise ValueError(
            "OpenAI-compatible review requires exact per-million input/output token prices"
        )
    elif price_snapshot.effective_at > observed_at:
        raise ValueError("phase-review price snapshot is not effective at dispatch preflight")
    _validate_plan_budget(plan, budget, price_snapshot=price_snapshot)


def _validate_plan_budget(
    plan: AIReviewPlan,
    budget: BudgetSpecV2,
    *,
    price_snapshot: PriceSnapshot | None,
) -> None:
    if len(budget.role_ceilings) != 1:
        raise ValueError("phase review requires exactly one role budget ceiling")
    role = budget.role_ceilings[0]
    planned_inputs = tuple(batch.input_tokens for batch in plan.case_batches) + (
        plan.phase_integrity_input_tokens,
    )
    planned_outputs = tuple(batch.maximal_output_tokens for batch in plan.case_batches) + (
        plan.phase_integrity_maximal_output_tokens,
    )
    planned_cost = sum(
        (
            _phase_review_planned_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                price_snapshot=price_snapshot,
            )
            for input_tokens, output_tokens in zip(
                planned_inputs,
                planned_outputs,
                strict=True,
            )
        ),
        Decimal("0"),
    )
    if (
        budget.max_attempts < plan.expected_attempt_count
        or role.max_attempts < plan.expected_attempt_count
        or sum(planned_inputs) > budget.max_input_tokens
        or sum(planned_inputs) > role.max_input_tokens
        or sum(planned_outputs) > budget.max_output_tokens
        or sum(planned_outputs) > role.max_output_tokens
        or budget.max_cost is None
        or role.max_cost is None
        or budget.currency is None
        or role.provider_budget_cap.operation_kind != _QUALITY_REVIEW_STAGE
        or role.provider_budget_cap.billing_unit != "request"
        or role.provider_budget_cap.maximum_accepted_units < Decimal(plan.expected_attempt_count)
        or planned_cost > budget.max_cost
        or planned_cost > role.max_cost
    ):
        raise ValueError("AI review plan exceeds or does not close its phase budget")


def _validate_previous_ai_history(
    bundle: EvaluationReviewBundle,
    plan: AIReviewPlan,
    occurrence: PhaseReviewOccurrenceRecordV2,
    previous_ai_history: tuple[AIQualityReviewRecord, ...],
) -> None:
    if occurrence.ordinal == 1:
        if previous_ai_history:
            raise ValueError("first phase-review occurrence cannot consume predecessor history")
        return
    if len(previous_ai_history) != occurrence.ordinal - 1:
        raise ValueError("phase-review predecessor history has an ordinal gap")
    validate_ai_review_history(bundle.bundle_id, previous_ai_history)
    if any(
        record.occurrence_id
        != phase_review_occurrence_id_v2(
            phase_id=occurrence.phase_id,
            review_bundle_hash=occurrence.review_bundle_hash,
            reviewer_role_binding_hash=occurrence.reviewer_role_binding_hash,
            artifact_repository_fingerprint=occurrence.artifact_repository_fingerprint,
            ordinal=record.ordinal,
        )
        for record in previous_ai_history
    ):
        raise ValueError(
            "phase-review predecessor history does not belong to the bound artifact repository"
        )
    head = previous_ai_history[-1]
    if (
        head.status != "inconclusive"
        or head.review_outcome_kind != "operational_inconclusive"
        or head.review_plan_hash != plan.plan_hash
        or head.reviewer_role_binding_hash != plan.reviewer_role_binding_hash
        or head.reviewer_model_hash != plan.reviewer_model_hash
        or head.reviewer_runtime_hash != plan.reviewer_runtime_hash
        or head.reviewer_configuration_hash != plan.reviewer_configuration_hash
    ):
        raise ValueError(
            "phase-review supersession requires the matching operational INCONCLUSIVE head"
        )


def _phase_review_lease(
    *,
    occurrence: PhaseReviewOccurrenceRecordV2,
    role: ModelRoleBindingV2,
    client_kind: str,
    acquired_at: datetime,
) -> RunLeaseRecord:
    host = socket.gethostname()
    candidate = RunLeaseRecord(
        lease_record_hash="0" * 64,
        run_id=occurrence.phase_review_occurrence_id,
        provider_project_id=str(role.provider),
        provider_profile_id=canonical_sha256(
            ["oamb-phase-review-provider-profile-v1", client_kind, role.binding_id]
        ),
        lease_epoch=1,
        owner_id=_PHASE_REVIEW_OWNER_ID,
        host_fingerprint=host_identity_fingerprint(host),
        process_id=os.getpid(),
        predecessor_lease_record_hash=None,
        acquired_at=acquired_at,
    )
    return candidate.model_copy(
        update={
            "lease_record_hash": canonical_sha256(
                candidate.model_dump(mode="python", exclude={"lease_record_hash"})
            )
        }
    )


def _phase_review_budget_ledger(budget: BudgetSpecV2) -> BudgetLedger:
    role = budget.role_ceilings[0]
    provider_units = _provider_budget_units(role.provider_budget_cap.maximum_accepted_units)
    parent_amount = BudgetAmount(
        attempts=budget.max_attempts,
        input_tokens=budget.max_input_tokens,
        output_tokens=budget.max_output_tokens,
        wall_seconds=budget.max_dispatch_wall_seconds,
        cost=budget.max_cost or Decimal("0"),
        resources=tuple(
            (item.dimension_id, item.maximum, item.unit) for item in budget.resource_ceilings
        ),
        provider_units=(
            (
                role.provider_budget_cap.provider,
                role.provider_budget_cap.operation_kind,
                role.provider_budget_cap.billing_unit,
                provider_units,
            ),
        ),
    )
    role_amount = BudgetAmount(
        attempts=role.max_attempts,
        input_tokens=role.max_input_tokens,
        output_tokens=role.max_output_tokens,
        wall_seconds=role.max_dispatch_wall_seconds,
        cost=role.max_cost or Decimal("0"),
        resources=tuple(
            (item.dimension_id, item.maximum, item.unit) for item in role.resource_ceilings
        ),
        provider_units=(
            (
                role.provider_budget_cap.provider,
                role.provider_budget_cap.operation_kind,
                role.provider_budget_cap.billing_unit,
                provider_units,
            ),
        ),
    )
    return BudgetLedger(
        BudgetCeiling(maximum=parent_amount, currency=budget.currency),
        role_ceilings={
            role.role_binding_id: BudgetCeiling(maximum=role_amount, currency=role.currency)
        },
    )


def _provider_budget_units(value: Decimal) -> int:
    units = int(value)
    if Decimal(units) != value:
        raise ValueError("phase-review provider request budget must be an integer")
    return units


@dataclass(frozen=True, slots=True)
class _PreparedPhaseAttempt:
    reservation_id: str
    maximum: BudgetAmount
    claim: OccurrenceClaimRecord
    reservation: BudgetReservationRecord
    intent: AttemptIntentRecord


def _prepare_phase_attempt(
    *,
    coordinator: AttemptCoordinator,
    budget_ledger: BudgetLedger,
    lease_record: RunLeaseRecord,
    plan: AIReviewPlan,
    occurrence: PhaseReviewOccurrenceRecordV2,
    budget: BudgetSpecV2,
    request: ModelRequest,
    ordinal: int,
    price_snapshot: PriceSnapshot | None,
    reserved_at: datetime,
) -> _PreparedPhaseAttempt:
    role = budget.role_ceilings[0]
    snapshot = budget_ledger.snapshot()
    consumed = snapshot.committed.add(snapshot.reserved)
    consumed_resources = {dimension_id: value for dimension_id, value, _unit in consumed.resources}
    consumed_wall_resource = consumed_resources.get(PHASE_REVIEW_WALL_DIMENSION_ID, Decimal("0"))
    parent_resource_remaining = budget.resource_ceilings[0].maximum - consumed_wall_resource
    role_resource_remaining = role.resource_ceilings[0].maximum - consumed_wall_resource
    remaining_wall = min(
        budget.max_dispatch_wall_seconds - consumed.wall_seconds,
        role.max_dispatch_wall_seconds - consumed.wall_seconds,
        parent_resource_remaining,
        role_resource_remaining,
    )
    if remaining_wall <= 0:
        raise ValueError("phase-review budget has no remaining dispatch wall time")
    planned_input_tokens = (
        plan.case_batches[ordinal - 1].input_tokens
        if ordinal <= len(plan.case_batches)
        else plan.phase_integrity_input_tokens
    )
    reserved_cost = _phase_review_planned_cost(
        input_tokens=planned_input_tokens,
        output_tokens=request.max_output_tokens,
        price_snapshot=price_snapshot,
    )
    resource_ceiling = ResourceBudgetCeiling(
        dimension_id=PHASE_REVIEW_WALL_DIMENSION_ID,
        maximum=remaining_wall,
        unit=PHASE_REVIEW_WALL_UNIT,
    )
    maximum = BudgetAmount(
        attempts=1,
        input_tokens=planned_input_tokens,
        output_tokens=request.max_output_tokens,
        wall_seconds=remaining_wall,
        cost=reserved_cost,
        resources=((PHASE_REVIEW_WALL_DIMENSION_ID, remaining_wall, PHASE_REVIEW_WALL_UNIT),),
        provider_units=(
            (
                role.provider_budget_cap.provider,
                role.provider_budget_cap.operation_kind,
                role.provider_budget_cap.billing_unit,
                1,
            ),
        ),
    )
    claim_fields = {
        "occurrence_id": occurrence.phase_review_occurrence_id,
        "lease_record_hash": lease_record.lease_record_hash,
        "lease_epoch": 1,
        "owner_id": _PHASE_REVIEW_OWNER_ID,
        "stage": _QUALITY_REVIEW_STAGE,
        "request_fingerprint": request.messages_sha256,
        "reconciliation_capability": "none",
        "claimed_at": reserved_at,
    }
    claim_id = canonical_sha256(["oamb-phase-review-occurrence-claim-v1", claim_fields])
    claim = OccurrenceClaimRecord.model_validate({"claim_id": claim_id, **claim_fields})
    reservation_fields = {
        "budget_id": budget.budget_id,
        "scope_kind": BudgetScopeKindV2.PHASE_REVIEW,
        "scope_id": occurrence.phase_review_occurrence_id,
        "role_binding_id": role.role_binding_id,
        "attempt_id": request.attempt_id,
        "reserved_attempts": 1,
        "reserved_input_tokens": planned_input_tokens,
        "reserved_output_tokens": request.max_output_tokens,
        "reserved_dispatch_wall_seconds": remaining_wall,
        "reserved_cost": reserved_cost,
        "currency": budget.currency,
        "reserved_resource_ceilings": (resource_ceiling,),
        "reserved_provider_units": Decimal("1"),
        "reserved_at": reserved_at,
    }
    reservation_id = canonical_sha256(
        ["oamb-phase-review-budget-reservation-v1", reservation_fields]
    )
    reservation = BudgetReservationRecord.model_validate(
        {"reservation_id": reservation_id, **reservation_fields}
    )
    intent = AttemptIntentRecord(
        attempt_id=request.attempt_id,
        claim_id=claim_id,
        reservation_id=reservation_id,
        parent_kind="phase_review",
        parent_id=occurrence.phase_review_occurrence_id,
        role_binding_id=role.role_binding_id,
        stage=_QUALITY_REVIEW_STAGE,
        request_fingerprint=request.messages_sha256,
        reconciliation_capability="none",
        idempotency_key_hash=None,
        sealed_at=reserved_at,
    )
    coordinator.prepare(
        claim=claim,
        reservation=reservation,
        intent=intent,
        maximum=maximum,
    )
    return _PreparedPhaseAttempt(
        reservation_id=reservation_id,
        maximum=maximum,
        claim=claim,
        reservation=reservation,
        intent=intent,
    )


def _phase_review_planned_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    price_snapshot: PriceSnapshot | None,
) -> Decimal:
    if price_snapshot is None:
        return Decimal("0")
    input_price, output_price = price_snapshot.unit_prices
    return (Decimal(input_tokens) * input_price + Decimal(output_tokens) * output_price) / Decimal(
        PHASE_REVIEW_TOKEN_PRICE_UNIT_SCALE
    )


def _observed_phase_budget(
    *,
    usage_records: tuple[TokenUsageRecordV2, ...],
    resource: ResourceUsageRecord,
    costs: tuple[CostRecord, ...],
    maximum: BudgetAmount,
    budget: BudgetSpecV2,
) -> BudgetAmount:
    if any(item.proof_status != ProofStatus.MEASURED_COMPLETE for item in usage_records):
        return maximum
    cost = sum((item.amount or Decimal("0") for item in costs), Decimal("0"))
    role = budget.role_ceilings[0]
    return BudgetAmount(
        attempts=1,
        input_tokens=min(
            sum(item.input_tokens or 0 for item in usage_records), maximum.input_tokens
        ),
        output_tokens=min(
            sum(item.visible_output_tokens or 0 for item in usage_records), maximum.output_tokens
        ),
        wall_seconds=min(resource.value or Decimal("0"), maximum.wall_seconds),
        cost=min(cost, maximum.cost),
        resources=(
            (
                resource.dimension_id,
                min(resource.value or Decimal("0"), maximum.resources[0][1]),
                resource.unit,
            ),
        ),
        provider_units=(
            (
                role.provider_budget_cap.provider,
                role.provider_budget_cap.operation_kind,
                role.provider_budget_cap.billing_unit,
                1,
            ),
        ),
    )


def _finalize_phase_attempt(
    *,
    coordinator: AttemptCoordinator,
    prepared: _PreparedPhaseAttempt,
    attempt: AttemptRecordV2,
    usage_records: tuple[TokenUsageRecordV2, ...],
    resource: ResourceUsageRecord,
    costs: tuple[CostRecord, ...],
    budget: BudgetSpecV2,
    store: ArtifactStore,
    receipt_kind: AttemptReceiptKind | None,
    raw_payload: bytes | None,
    wall_seconds: Decimal,
) -> AttemptReceiptRecord | None:
    if receipt_kind is None:
        coordinator.mark_unknown(
            reservation_id=prepared.reservation_id,
            attempt=attempt,
        )
        durable_receipt = None
    else:
        if raw_payload is None:
            raise ValueError("received phase-review attempt is missing its raw payload")
        durable_receipt = _attempt_receipt(
            attempt=attempt,
            wall_seconds=wall_seconds,
            receipt_kind=receipt_kind,
        )
        coordinator.record_receipt(
            receipt=durable_receipt,
            raw_payload=raw_payload,
            media_type="application/json",
            observed=_observed_phase_budget(
                usage_records=usage_records,
                resource=resource,
                costs=costs,
                maximum=prepared.maximum,
                budget=budget,
            ),
        )
        coordinator.seal_terminal(attempt)
    _seal_accounting_evidence(store, resource, costs)
    return durable_receipt


def _load_requests(plan: AIReviewPlan, root: Path) -> tuple[str, ...]:
    requests = tuple(
        _read_canonical_request(root / "case-requests" / f"{index:04d}.json")
        for index in range(1, len(plan.case_batches) + 1)
    ) + (_read_canonical_request(root / "integrity-request.json"),)
    expected = tuple(batch.request_fingerprint for batch in plan.case_batches) + (
        plan.phase_integrity_request_fingerprint,
    )
    if tuple(ai_review_request_fingerprint(item) for item in requests) != expected:
        raise ValueError("AI review request bytes do not bind the approved plan")
    return requests


def _read_canonical_request(path: Path) -> str:
    content = path.read_bytes()
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} is not one strict UTF-8 JSON request") from exc
    if not isinstance(value, dict) or content != canonical_json_bytes(value):
        raise ValueError(f"{path} is not a canonical request object")
    return content.decode("utf-8")


def _phase_role_slots(role: ModelRoleBindingV2) -> tuple[RoleSlot, ...]:
    return tuple(
        RoleSlot(
            role=slot,
            status=(
                ResolutionStatus.RESOLVED
                if slot == RoleSlotName.QUALITY_REVIEW
                else ResolutionStatus.UNSELECTED
            ),
            binding=role if slot == RoleSlotName.QUALITY_REVIEW else None,
            credential_reference=(
                EnvironmentReference(str(role.credential_variable_name))
                if slot == RoleSlotName.QUALITY_REVIEW
                else None
            ),
            evidence_reference=(
                "phase-review-role-binding" if slot == RoleSlotName.QUALITY_REVIEW else "unselected"
            ),
        )
        for slot in RoleSlotName
    )


def _build_fake_client(
    store: ArtifactStore,
    responses: tuple[str, ...],
) -> _FakePhaseReviewModelClient:
    return _FakePhaseReviewModelClient(store, responses)


def _build_openai_client(
    store: ArtifactStore,
    role: ModelRoleBindingV2,
    api_key: str,
) -> OpenAICompatibleModelClient:
    assert role.endpoint_reference is not None
    return OpenAICompatibleModelClient(
        store=store,
        base_url=role.endpoint_reference,
        api_key=api_key,
        role_binding=role,
        runtime_model_policy="require_match",
        reasoning_control=("reasoning_effort", "none"),
    )


async def _execute_review(
    *,
    client: ModelClientPort,
    client_kind: str,
    store: ArtifactStore,
    plan: AIReviewPlan,
    occurrence: PhaseReviewOccurrenceRecordV2,
    approval: ExternalCallApprovalRecord,
    budget: BudgetSpecV2,
    requests: tuple[str, ...],
    cost_measurement_spec: CostMeasurementSpec,
    execution_environment: ExecutionEnvironmentBinding,
    price_snapshot: PriceSnapshot | None,
    fake_started_at: datetime | None,
    fake_ended_at: datetime | None,
    coordinator: AttemptCoordinator,
    budget_ledger: BudgetLedger,
    lease_record: RunLeaseRecord,
    previous_ai_history: tuple[AIQualityReviewRecord, ...],
) -> PhaseReviewRunResult:
    attempts: list[AttemptRecordV2] = []
    usage_records: list[TokenUsageRecordV2] = []
    resources: list[ResourceUsageRecord] = []
    costs: list[CostRecord] = []
    occurrence_claims: list[OccurrenceClaimRecord] = []
    budget_reservations: list[BudgetReservationRecord] = []
    attempt_intents: list[AttemptIntentRecord] = []
    attempt_receipts: list[AttemptReceiptRecord] = []
    batch_results: list[AIReviewBatchResult] = []
    integrity_result: AIReviewIntegrityResult | None = None
    fingerprints = tuple(batch.request_fingerprint for batch in plan.case_batches) + (
        plan.phase_integrity_request_fingerprint,
    )
    execution_started_at = (
        fake_started_at if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND else _utc_now()
    )
    assert execution_started_at is not None
    try:
        for ordinal, (request_text, fingerprint) in enumerate(
            zip(requests, fingerprints, strict=True),
            1,
        ):
            dispatch_preflight_at = (
                execution_started_at + timedelta(microseconds=ordinal)
                if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND
                else _utc_now()
            )
            try:
                _require_dispatch_allowed(
                    ordinal=ordinal,
                    approval=approval,
                    budget=budget,
                    attempts=tuple(attempts),
                    current_time=dispatch_preflight_at,
                )
            except ValueError as exc:
                _raise_partial_failure(
                    str(exc),
                    store=store,
                    occurrence=occurrence,
                    state="budget_exceeded",
                    started_at=execution_started_at,
                    ended_at=_utc_now(),
                    batch_results=tuple(batch_results),
                    integrity_result=integrity_result,
                    attempts=tuple(attempts),
                    usage_records=tuple(usage_records),
                    resources=tuple(resources),
                    costs=tuple(costs),
                )
            request = _model_request(
                plan,
                occurrence,
                request_text=request_text,
                request_fingerprint=fingerprint,
                ordinal=ordinal,
            )
            prepared = _prepare_phase_attempt(
                coordinator=coordinator,
                budget_ledger=budget_ledger,
                lease_record=lease_record,
                plan=plan,
                occurrence=occurrence,
                budget=budget,
                request=request,
                ordinal=ordinal,
                price_snapshot=price_snapshot,
                reserved_at=dispatch_preflight_at,
            )
            occurrence_claims.append(prepared.claim)
            budget_reservations.append(prepared.reservation)
            attempt_intents.append(prepared.intent)
            if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND:
                assert fake_started_at is not None
                dispatch_started_at = fake_started_at + timedelta(microseconds=ordinal)
                monotonic_started = Decimal("0")
            else:
                dispatch_started_at = _utc_now()
                monotonic_started = Decimal(str(_monotonic_now()))
            coordinator.mark_dispatched(request.attempt_id)
            try:
                if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND:
                    receipt = await client.complete(request)
                else:
                    receipt = await asyncio.wait_for(
                        client.complete(request),
                        timeout=float(prepared.maximum.wall_seconds),
                    )
            except BaseException as exc:
                dispatch_ended_at, wall_seconds = _dispatch_end(
                    client_kind=client_kind,
                    fake_started_at=fake_started_at,
                    ordinal=ordinal,
                    monotonic_started=monotonic_started,
                )
                known_receipt = isinstance(exc, ModelCallFailure) and bool(exc.raw_response_bytes)
                failed_attempt = _failure_attempt(
                    request,
                    ordinal=ordinal,
                    request_fingerprint=fingerprint,
                    started_at=dispatch_started_at,
                    ended_at=dispatch_ended_at,
                    error=exc,
                    known_receipt=known_receipt,
                )
                failed_usage = _failure_usage_records(store, request, exc)
                failed_resource = _build_resource(
                    failed_attempt,
                    wall_seconds=wall_seconds,
                    cost_measurement_spec=cost_measurement_spec,
                    execution_environment=execution_environment,
                )
                failed_costs = _build_costs(
                    failed_usage,
                    (failed_resource,),
                    client_kind=client_kind,
                    price_snapshot=price_snapshot,
                    failure=True,
                )
                attempts.append(failed_attempt)
                usage_records.extend(failed_usage)
                resources.append(failed_resource)
                costs.extend(failed_costs)
                try:
                    durable_receipt = _finalize_phase_attempt(
                        coordinator=coordinator,
                        prepared=prepared,
                        attempt=failed_attempt,
                        usage_records=failed_usage,
                        resource=failed_resource,
                        costs=failed_costs,
                        budget=budget,
                        store=store,
                        receipt_kind=(AttemptReceiptKind.ERROR if known_receipt else None),
                        raw_payload=(
                            exc.raw_response_bytes if isinstance(exc, ModelCallFailure) else None
                        ),
                        wall_seconds=wall_seconds,
                    )
                    if durable_receipt is not None:
                        attempt_receipts.append(durable_receipt)
                except BaseException as finalization_error:
                    _raise_partial_failure(
                        "phase-review dispatch failed and durable attempt finalization failed: "
                        f"{type(exc).__name__}; {type(finalization_error).__name__}",
                        store=store,
                        occurrence=occurrence,
                        state=("error" if known_receipt else "interrupted_unknown_outcome"),
                        started_at=execution_started_at,
                        ended_at=dispatch_ended_at,
                        batch_results=tuple(batch_results),
                        integrity_result=integrity_result,
                        attempts=tuple(attempts),
                        usage_records=tuple(usage_records),
                        resources=tuple(resources),
                        costs=tuple(costs),
                    )
                _raise_partial_failure(
                    f"phase-review dispatch {ordinal} failed: {type(exc).__name__}",
                    store=store,
                    occurrence=occurrence,
                    state="error" if known_receipt else "interrupted_unknown_outcome",
                    started_at=execution_started_at,
                    ended_at=dispatch_ended_at,
                    batch_results=tuple(batch_results),
                    integrity_result=integrity_result,
                    attempts=tuple(attempts),
                    usage_records=tuple(usage_records),
                    resources=tuple(resources),
                    costs=tuple(costs),
                )
            dispatch_ended_at, wall_seconds = _dispatch_end(
                client_kind=client_kind,
                fake_started_at=fake_started_at,
                ordinal=ordinal,
                monotonic_started=monotonic_started,
            )
            attempt = _success_attempt(
                request,
                receipt,
                ordinal=ordinal,
                request_fingerprint=fingerprint,
                started_at=dispatch_started_at,
                ended_at=dispatch_ended_at,
            )
            attempt_usage = _load_usage_records(store, (receipt,))
            resource = _build_resource(
                attempt,
                wall_seconds=wall_seconds,
                cost_measurement_spec=cost_measurement_spec,
                execution_environment=execution_environment,
            )
            attempt_costs = _build_costs(
                attempt_usage,
                (resource,),
                client_kind=client_kind,
                price_snapshot=price_snapshot,
            )
            attempts.append(attempt)
            usage_records.extend(attempt_usage)
            resources.append(resource)
            costs.extend(attempt_costs)
            try:
                durable_receipt = _finalize_phase_attempt(
                    coordinator=coordinator,
                    prepared=prepared,
                    attempt=attempt,
                    usage_records=attempt_usage,
                    resource=resource,
                    costs=attempt_costs,
                    budget=budget,
                    store=store,
                    receipt_kind=AttemptReceiptKind.RESPONSE,
                    raw_payload=receipt.raw_response_bytes,
                    wall_seconds=wall_seconds,
                )
                if durable_receipt is not None:
                    attempt_receipts.append(durable_receipt)
            except BaseException as finalization_error:
                _raise_partial_failure(
                    "phase-review durable attempt finalization failed: "
                    f"{type(finalization_error).__name__}",
                    store=store,
                    occurrence=occurrence,
                    state="error",
                    started_at=execution_started_at,
                    ended_at=dispatch_ended_at,
                    batch_results=tuple(batch_results),
                    integrity_result=integrity_result,
                    attempts=tuple(attempts),
                    usage_records=tuple(usage_records),
                    resources=tuple(resources),
                    costs=tuple(costs),
                )
            if any(item.proof_status != ProofStatus.MEASURED_COMPLETE for item in attempt_usage):
                _raise_partial_failure(
                    "phase-review supplier usage is unavailable for budget and cost closure",
                    store=store,
                    occurrence=occurrence,
                    state="evidence_inconclusive",
                    started_at=execution_started_at,
                    ended_at=dispatch_ended_at,
                    batch_results=tuple(batch_results),
                    integrity_result=integrity_result,
                    attempts=tuple(attempts),
                    usage_records=tuple(usage_records),
                    resources=tuple(resources),
                    costs=tuple(costs),
                )
            try:
                _require_observed_budget(
                    budget=budget,
                    attempts=tuple(attempts),
                    usage_records=tuple(usage_records),
                    resources=tuple(resources),
                    costs=tuple(costs),
                )
                if ordinal <= len(plan.case_batches):
                    batch_results.append(
                        parse_ai_review_batch_output(
                            plan.case_batches[ordinal - 1], receipt.output_text
                        )
                    )
                else:
                    integrity_result = parse_ai_review_integrity_output(
                        plan.phase_integrity_id,
                        receipt.output_text,
                    )
            except ValueError as exc:
                _raise_partial_failure(
                    str(exc),
                    store=store,
                    occurrence=occurrence,
                    state=(
                        "budget_exceeded"
                        if "budget" in str(exc) or "ceiling" in str(exc)
                        else "error"
                    ),
                    started_at=execution_started_at,
                    ended_at=dispatch_ended_at,
                    batch_results=tuple(batch_results),
                    integrity_result=integrity_result,
                    attempts=tuple(attempts),
                    usage_records=tuple(usage_records),
                    resources=tuple(resources),
                    costs=tuple(costs),
                )
    except BaseException as primary_error:
        try:
            await client.close()
        except BaseException as close_error:
            try:
                _seal_phase_close_error(
                    store,
                    occurrence_id=occurrence.phase_review_occurrence_id,
                    client_profile_id=client_kind,
                    error=close_error,
                    occurred_at=_utc_now(),
                )
            except BaseException as diagnostic_error:
                primary_error.add_note(
                    "phase-review close diagnostic sealing also failed: "
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                )
        raise
    else:
        try:
            await client.close()
        except BaseException as close_error:
            try:
                _seal_phase_close_error(
                    store,
                    occurrence_id=occurrence.phase_review_occurrence_id,
                    client_profile_id=client_kind,
                    error=close_error,
                    occurred_at=_utc_now(),
                )
                _raise_partial_failure(
                    f"phase-review client close failed: {type(close_error).__name__}",
                    store=store,
                    occurrence=occurrence,
                    state="error",
                    started_at=execution_started_at,
                    ended_at=_utc_now(),
                    batch_results=tuple(batch_results),
                    integrity_result=integrity_result,
                    attempts=tuple(attempts),
                    usage_records=tuple(usage_records),
                    resources=tuple(resources),
                    costs=tuple(costs),
                )
            except PhaseReviewRunFailure:
                raise
            except BaseException as diagnostic_error:
                close_error.add_note(
                    "phase-review close diagnostic terminalization also failed: "
                    f"{type(diagnostic_error).__name__}: {diagnostic_error}"
                )
                raise close_error from diagnostic_error

    execution_ended_at = (
        fake_ended_at if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND else _utc_now()
    )
    assert execution_ended_at is not None
    try:
        if integrity_result is None:
            raise ValueError("complete phase review is missing its integrity result")
        record = reduce_ai_quality_review(
            plan,
            batch_results=tuple(batch_results),
            integrity_result=integrity_result,
            occurrence_id=occurrence.phase_review_occurrence_id,
            ordinal=occurrence.ordinal,
            previous_ai_review_record_hash=(
                None if not previous_ai_history else previous_ai_history[-1].ai_review_record_id
            ),
            previous_history_root_hash=(
                None if not previous_ai_history else previous_ai_history[-1].history_root_hash
            ),
            attempt_ids=tuple(item.attempt_id for item in attempts),
            usage_record_ids=tuple(item.usage_record_id for item in usage_records),
            resource_record_ids=tuple(item.resource_record_id for item in resources),
            cost_record_ids=tuple(item.cost_record_id for item in costs),
            accounting_closed=all(
                item.proof_status in {ProofStatus.MEASURED_COMPLETE, ProofStatus.NOT_APPLICABLE}
                for item in costs
            ),
            created_at=execution_ended_at,
        )
        for batch_result in batch_results:
            _seal_source_record(
                store,
                "phase-review-batch-results",
                canonical_sha256(batch_result),
                batch_result,
            )
        _seal_source_record(
            store,
            "phase-review-integrity-results",
            canonical_sha256(integrity_result),
            integrity_result,
        )
        _seal_source_record(
            store,
            "ai-review-records",
            record.ai_review_record_id,
            record,
        )
        _seal_source_record(
            store,
            "ai-review-histories",
            record.history_root_hash,
            (*previous_ai_history, record),
        )
        sealed_occurrence = _occurrence_with_state(
            occurrence,
            state="sealed",
            started_at=execution_started_at,
            ended_at=execution_ended_at,
        )
        _seal_source_record(
            store,
            "phase-review-occurrences",
            sealed_occurrence.phase_review_occurrence_id,
            sealed_occurrence,
        )
    except BaseException as completion_error:
        _raise_partial_failure(
            f"phase-review durable completion failed: {type(completion_error).__name__}",
            store=store,
            occurrence=occurrence,
            state="error",
            started_at=execution_started_at,
            ended_at=execution_ended_at,
            batch_results=tuple(batch_results),
            integrity_result=integrity_result,
            attempts=tuple(attempts),
            usage_records=tuple(usage_records),
            resources=tuple(resources),
            costs=tuple(costs),
        )
    return PhaseReviewRunResult(
        lease_record=lease_record,
        occurrence_claims=tuple(occurrence_claims),
        budget_reservations=tuple(budget_reservations),
        attempt_intents=tuple(attempt_intents),
        attempt_receipts=tuple(attempt_receipts),
        occurrence=sealed_occurrence,
        batch_results=tuple(batch_results),
        integrity_result=integrity_result,
        attempts=tuple(attempts),
        usage_records=tuple(usage_records),
        resource_records=tuple(resources),
        cost_records=tuple(costs),
        ai_record=record,
    )


def _load_usage_records(
    store: ArtifactStore,
    receipts: tuple[ModelReceipt, ...],
) -> tuple[TokenUsageRecordV2, ...]:
    records: list[TokenUsageRecordV2] = []
    for receipt in receipts:
        if len(receipt.usage_reference_ids) != 1:
            raise ValueError("phase-review model receipt requires one usage record")
        usage_id = receipt.usage_reference_ids[0]
        path = store.root / "source" / "usage" / f"{usage_id}.json"
        record = TokenUsageRecordV2.model_validate_json(read_regular_file(path))
        if record.usage_record_id != usage_id:
            raise ValueError("phase-review usage record identity does not match its receipt")
        records.append(record)
    return tuple(records)


def _model_request(
    plan: AIReviewPlan,
    occurrence: PhaseReviewOccurrenceRecordV2,
    *,
    request_text: str,
    request_fingerprint: str,
    ordinal: int,
) -> ModelRequest:
    document = json.loads(request_text)
    messages = tuple((str(role), str(content)) for role, content in document["messages"])
    return ModelRequest(
        attempt_id=derive_attempt_id(
            occurrence.phase_review_occurrence_id,
            _QUALITY_REVIEW_STAGE,
            ordinal,
            request_fingerprint,
        ),
        parent_kind="phase_review",
        parent_id=occurrence.phase_review_occurrence_id,
        stage=_QUALITY_REVIEW_STAGE,
        role_binding_id=plan.reviewer_role_binding_hash,
        messages_sha256=request_fingerprint,
        messages=messages,
        output_contract_id="ai-quality-review-json-v1",
        max_output_tokens=(
            plan.case_batches[ordinal - 1].maximal_output_tokens
            if ordinal <= len(plan.case_batches)
            else plan.phase_integrity_maximal_output_tokens
        ),
    )


def _dispatch_end(
    *,
    client_kind: str,
    fake_started_at: datetime | None,
    ordinal: int,
    monotonic_started: Decimal,
) -> tuple[datetime, Decimal]:
    if client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND:
        assert fake_started_at is not None
        return fake_started_at + timedelta(microseconds=ordinal), Decimal("0")
    ended_at = _utc_now()
    elapsed = Decimal(str(_monotonic_now())) - monotonic_started
    if elapsed < 0:
        raise ValueError("phase-review monotonic clock moved backwards")
    return ended_at, elapsed


def _success_attempt(
    request: ModelRequest,
    receipt: ModelReceipt,
    *,
    ordinal: int,
    request_fingerprint: str,
    started_at: datetime,
    ended_at: datetime,
) -> AttemptRecordV2:
    return AttemptRecordV2(
        attempt_id=request.attempt_id,
        parent_kind="phase_review",
        parent_id=request.parent_id,
        stage=_QUALITY_REVIEW_STAGE,
        ordinal=ordinal,
        request_fingerprint=request_fingerprint,
        started_at=started_at,
        ended_at=ended_at,
        outcome=AttemptOutcome.SUCCEEDED,
        retry_of_attempt_id=None,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=receipt.raw_reference.sha256,
        raw_error_ref=None,
        index_contribution=IndexContribution.NONE,
        superseded_by_attempt_id=None,
    )


def _failure_attempt(
    request: ModelRequest,
    *,
    ordinal: int,
    request_fingerprint: str,
    started_at: datetime,
    ended_at: datetime,
    error: BaseException,
    known_receipt: bool,
) -> AttemptRecordV2:
    unknown = not known_receipt
    raw_handle = getattr(error, "raw_reference", None)
    raw_error_ref = None if unknown else getattr(raw_handle, "sha256", None)
    return AttemptRecordV2(
        attempt_id=request.attempt_id,
        parent_kind="phase_review",
        parent_id=request.parent_id,
        stage=_QUALITY_REVIEW_STAGE,
        ordinal=ordinal,
        request_fingerprint=request_fingerprint,
        started_at=started_at,
        ended_at=ended_at,
        outcome=AttemptOutcome.UNKNOWN_OUTCOME if unknown else AttemptOutcome.FAILED,
        retry_of_attempt_id=None,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=None,
        raw_error_ref=raw_error_ref,
        index_contribution=IndexContribution.NONE,
        superseded_by_attempt_id=None,
    )


def _attempt_receipt(
    *,
    attempt: AttemptRecordV2,
    wall_seconds: Decimal,
    receipt_kind: AttemptReceiptKind,
) -> AttemptReceiptRecord:
    return AttemptReceiptRecord(
        attempt_id=attempt.attempt_id,
        receipt_kind=receipt_kind,
        raw_response_ref=attempt.raw_response_ref,
        raw_error_ref=attempt.raw_error_ref,
        dispatch_started_at=attempt.started_at,
        receipt_observed_at=attempt.ended_at,
        provider_request_wall_seconds=wall_seconds,
    )


def _failure_usage_records(
    store: ArtifactStore,
    request: ModelRequest,
    error: BaseException,
) -> tuple[TokenUsageRecordV2, ...]:
    usage_ids = error.usage_reference_ids if isinstance(error, ModelCallFailure) else ()
    if usage_ids and isinstance(error, ModelCallFailure):
        receipts = tuple(
            ModelReceipt(
                raw_reference=error.raw_reference,
                output_text="",
                usage_reference_ids=(str(usage_id),),
            )
            for usage_id in usage_ids
        )
        return _load_usage_records(store, receipts)
    usage = TokenUsageRecordV2(
        usage_record_id=canonical_sha256(
            ["oamb-phase-review-unavailable-usage-v1", request.attempt_id, type(error).__name__]
        ),
        attempt_id=request.attempt_id,
        parent_kind="phase_review",
        parent_id=request.parent_id,
        stage=TokenStageV2.QUALITY_REVIEW,
        operation_kind=_QUALITY_REVIEW_STAGE,
        token_domain=TokenDomain.EXTERNAL_LLM,
        measurement_source=TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=None,
        visible_output_tokens=None,
        supplier_reported_total_tokens=None,
        context_view_tokens=None,
        proof_status=ProofStatus.UNAVAILABLE,
        reason="dispatch ended without complete supplier usage",
        raw_response_ref=None,
    )
    _seal_source_record(store, "usage", usage.usage_record_id, usage)
    return (usage,)


def _build_resource(
    attempt: AttemptRecordV2,
    *,
    wall_seconds: Decimal,
    cost_measurement_spec: CostMeasurementSpec,
    execution_environment: ExecutionEnvironmentBinding,
) -> ResourceUsageRecord:
    dimension = cost_measurement_spec.dimensions[0]
    value = Decimal("1") if dimension.unit == "request" else wall_seconds
    return ResourceUsageRecord(
        resource_record_id=canonical_sha256(
            ["oamb-phase-review-resource-v1", attempt.attempt_id, dimension.dimension_id, value]
        ),
        parent_kind="phase_review",
        parent_id=attempt.parent_id,
        stage=_QUALITY_REVIEW_STAGE,
        meter_boundary=dimension.dimension_id,
        dimension_id=dimension.dimension_id,
        value=value,
        unit=dimension.unit,
        measurement_source=dimension.allowed_meter_sources[0],
        measurement_spec_id=cost_measurement_spec.measurement_spec_id,
        environment_hash=execution_environment.environment_hash,
        started_at=attempt.started_at,
        ended_at=attempt.ended_at,
        raw_telemetry_ref=attempt.raw_response_ref or attempt.raw_error_ref,
        proof_status=ProofStatus.MEASURED_COMPLETE,
        reason=None,
    )


def _seal_accounting_evidence(
    store: ArtifactStore,
    resource: ResourceUsageRecord,
    costs: tuple[CostRecord, ...],
) -> None:
    _seal_source_record(store, "resources", resource.resource_record_id, resource)
    for cost in costs:
        _seal_source_record(store, "costs", cost.cost_record_id, cost)


def _require_dispatch_allowed(
    *,
    ordinal: int,
    approval: ExternalCallApprovalRecord,
    budget: BudgetSpecV2,
    attempts: tuple[AttemptRecordV2, ...],
    current_time: datetime,
) -> None:
    role = budget.role_ceilings[0]
    wall_seconds = sum(
        (_exact_seconds(item.ended_at - item.started_at) for item in attempts),
        Decimal("0"),
    )
    if current_time >= approval.expires_at:
        raise ValueError("phase-review approval expired before the next dispatch")
    if (
        ordinal > budget.max_attempts
        or ordinal > role.max_attempts
        or Decimal(ordinal) > role.provider_budget_cap.maximum_accepted_units
        or wall_seconds >= budget.max_dispatch_wall_seconds
        or wall_seconds >= role.max_dispatch_wall_seconds
        or len(attempts) != ordinal - 1
    ):
        raise ValueError("phase-review budget blocks the next dispatch")


def _require_observed_budget(
    *,
    budget: BudgetSpecV2,
    attempts: tuple[AttemptRecordV2, ...],
    usage_records: tuple[TokenUsageRecordV2, ...],
    resources: tuple[ResourceUsageRecord, ...],
    costs: tuple[CostRecord, ...],
) -> None:
    role = budget.role_ceilings[0]
    input_tokens = sum(item.input_tokens or 0 for item in usage_records)
    output_tokens = sum(item.visible_output_tokens or 0 for item in usage_records)
    wall_seconds = sum(
        (_exact_seconds(item.ended_at - item.started_at) for item in attempts),
        Decimal("0"),
    )
    cost = sum((item.amount or Decimal("0") for item in costs), Decimal("0"))
    parent_resource_ceilings = {
        item.dimension_id: (item.maximum, item.unit) for item in budget.resource_ceilings
    }
    role_resource_ceilings = {
        item.dimension_id: (item.maximum, item.unit) for item in role.resource_ceilings
    }
    resource_totals: dict[str, Decimal] = {}
    resource_evidence_invalid = False
    for record in resources:
        if (
            record.value is None
            or record.dimension_id not in parent_resource_ceilings
            or record.dimension_id not in role_resource_ceilings
            or record.unit != parent_resource_ceilings[record.dimension_id][1]
            or record.unit != role_resource_ceilings[record.dimension_id][1]
        ):
            resource_evidence_invalid = True
            break
        resource_totals[record.dimension_id] = (
            resource_totals.get(record.dimension_id, Decimal("0")) + record.value
        )
    resource_budget_exceeded = (
        resource_evidence_invalid
        or set(resource_totals) != set(parent_resource_ceilings)
        or set(resource_totals) != set(role_resource_ceilings)
        or any(
            value > parent_resource_ceilings[dimension_id][0]
            or value > role_resource_ceilings[dimension_id][0]
            for dimension_id, value in resource_totals.items()
        )
    )
    if (
        len(attempts) > budget.max_attempts
        or len(attempts) > role.max_attempts
        or input_tokens > budget.max_input_tokens
        or input_tokens > role.max_input_tokens
        or output_tokens > budget.max_output_tokens
        or output_tokens > role.max_output_tokens
        or wall_seconds > budget.max_dispatch_wall_seconds
        or wall_seconds > role.max_dispatch_wall_seconds
        or budget.max_cost is None
        or role.max_cost is None
        or cost > budget.max_cost
        or cost > role.max_cost
        or resource_budget_exceeded
    ):
        raise ValueError("observed phase-review evidence exceeds a budget ceiling")


def _occurrence_with_state(
    occurrence: PhaseReviewOccurrenceRecordV2,
    *,
    state: str,
    started_at: datetime,
    ended_at: datetime,
) -> PhaseReviewOccurrenceRecordV2:
    return PhaseReviewOccurrenceRecordV2.model_validate(
        occurrence.model_copy(
            update={"state": state, "started_at": started_at, "ended_at": ended_at}
        ).model_dump(mode="python")
    )


def _phase_review_release_ready(store: ArtifactStore, occurrence_id: str) -> bool:
    path = store.root / "source" / "phase-review-occurrences" / f"{occurrence_id}.json"
    try:
        occurrence = PhaseReviewOccurrenceRecordV2.model_validate_json(read_regular_file(path))
    except (OSError, ValueError):
        return False
    if occurrence.state == "sealed":
        ai_records = tuple((store.root / "source" / "ai-review-records").glob("*.json"))
        histories = tuple((store.root / "source" / "ai-review-histories").glob("*.json"))
        return len(ai_records) == 1 and len(histories) == 1
    return occurrence.state in {
        "error",
        "cancelled",
        "budget_exceeded",
        "evidence_inconclusive",
    }


def _raise_partial_failure(
    message: str,
    *,
    store: ArtifactStore,
    occurrence: PhaseReviewOccurrenceRecordV2,
    state: str,
    started_at: datetime,
    ended_at: datetime,
    batch_results: tuple[AIReviewBatchResult, ...],
    integrity_result: AIReviewIntegrityResult | None,
    attempts: tuple[AttemptRecordV2, ...],
    usage_records: tuple[TokenUsageRecordV2, ...],
    resources: tuple[ResourceUsageRecord, ...],
    costs: tuple[CostRecord, ...],
) -> NoReturn:
    failed_occurrence = _occurrence_with_state(
        occurrence,
        state=state,
        started_at=started_at,
        ended_at=max(started_at, ended_at),
    )
    _seal_source_record(
        store,
        "phase-review-occurrences",
        failed_occurrence.phase_review_occurrence_id,
        failed_occurrence,
    )
    raise PhaseReviewRunFailure(
        message,
        PartialPhaseReviewRunEvidence(
            occurrence=failed_occurrence,
            batch_results=batch_results,
            integrity_result=integrity_result,
            attempts=attempts,
            usage_records=usage_records,
            resource_records=resources,
            cost_records=costs,
        ),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _monotonic_now() -> float:
    return time.monotonic()


def _exact_seconds(duration: timedelta) -> Decimal:
    return Decimal(duration.days * 86_400 + duration.seconds) + (
        Decimal(duration.microseconds) / Decimal(1_000_000)
    )


def _build_costs(
    usage_records: tuple[TokenUsageRecordV2, ...],
    resources: tuple[ResourceUsageRecord, ...],
    *,
    client_kind: str,
    price_snapshot: PriceSnapshot | None,
    failure: bool = False,
) -> tuple[CostRecord, ...]:
    records: list[CostRecord] = []
    for usage, resource in zip(usage_records, resources, strict=True):
        estimated = (
            client_kind == OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND
            and usage.proof_status == ProofStatus.MEASURED_COMPLETE
        )
        if estimated:
            assert price_snapshot is not None
            input_unit_price, output_unit_price = price_snapshot.unit_prices
            currency = price_snapshot.currency
            price_snapshot_id = price_snapshot.price_snapshot_id
            proof_status = ProofStatus.MEASURED_COMPLETE
            reason = None
        elif client_kind == FAKE_PHASE_REVIEW_CLIENT_KIND:
            input_unit_price = None
            output_unit_price = None
            currency = None
            price_snapshot_id = None
            proof_status = ProofStatus.NOT_APPLICABLE
            reason = FAKE_PHASE_REVIEW_COST_REASON
        else:
            input_unit_price = None
            output_unit_price = None
            currency = None
            price_snapshot_id = (
                price_snapshot.price_snapshot_id
                if price_snapshot is not None and not failure
                else None
            )
            proof_status = ProofStatus.UNAVAILABLE
            reason = (
                "supplier token usage unavailable for price estimation"
                if not failure
                else "failed OpenAI-compatible dispatch lacks complete billing evidence"
            )
        records.append(
            CostRecord(
                cost_record_id=canonical_sha256(
                    [
                        "oamb-phase-review-cost-v1",
                        usage.usage_record_id,
                        resource.resource_record_id,
                    ]
                ),
                parent_kind="phase_review",
                parent_id=usage.parent_id,
                basis=(
                    CostBasis.ESTIMATE_FROM_MEASURED_USAGE
                    if estimated
                    else CostBasis.ACTUAL_SUPPLIER_CHARGE
                ),
                indexing_view=IndexingView.NOT_APPLICABLE,
                amount=(
                    None
                    if input_unit_price is None
                    or output_unit_price is None
                    or usage.input_tokens is None
                    or usage.visible_output_tokens is None
                    else (
                        Decimal(usage.input_tokens) * input_unit_price
                        + Decimal(usage.visible_output_tokens) * output_unit_price
                    )
                    / Decimal(PHASE_REVIEW_TOKEN_PRICE_UNIT_SCALE)
                ),
                currency=currency,
                price_snapshot_id=price_snapshot_id,
                source_usage_record_ids=(usage.usage_record_id,),
                source_resource_record_ids=(resource.resource_record_id,),
                proof_status=proof_status,
                reason=reason,
            )
        )
    return tuple(records)


def _seal_source_record(
    store: ArtifactStore,
    collection: str,
    record_id: str,
    record: object,
) -> None:
    content = canonical_json_bytes(record)
    store.seal_source_record(
        ArtifactWriteRequest(
            record_id=record_id,
            relative_path=f"source/{collection}/{record_id}.json",
            canonical_sha256=hashlib.sha256(content).hexdigest(),
            canonical_bytes=content,
        )
    )


def _seal_phase_close_error(
    store: ArtifactStore,
    *,
    occurrence_id: str,
    client_profile_id: str,
    error: BaseException,
    occurred_at: datetime,
) -> None:
    payload = canonical_json_bytes(
        {
            "operation": "phase_review_client_close",
            "error_type": type(error).__name__,
            "message": str(error),
        }
    )
    error_ref = store.seal_raw(
        RawPayloadSealRequest(
            sha256=hashlib.sha256(payload).hexdigest(),
            media_type="application/json",
            compression="gzip",
            payload_bytes=payload,
        )
    ).sha256
    fields = {
        "owner_kind": "phase_review",
        "owner_id": occurrence_id,
        "client_profile_id": client_profile_id,
        "error_ref": error_ref,
        "shutdown_stage": "model_client_close",
        "occurred_at": occurred_at,
    }
    close_error_id = canonical_sha256(["oamb-close-error-v1", fields])
    record = CloseErrorRecord.model_validate({"close_error_id": close_error_id, **fields})
    _seal_source_record(store, "close-errors", close_error_id, record)


__all__ = [
    "FAKE_PHASE_REVIEW_CLIENT_KIND",
    "OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND",
    "PHASE_REVIEW_CLIENT_KINDS",
    "PhaseReviewRunResult",
    "phase_review_model_hash",
    "phase_review_runtime_hash",
    "run_phase_review",
]
