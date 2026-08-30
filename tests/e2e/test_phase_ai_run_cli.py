from __future__ import annotations

import asyncio
import importlib
import json
import runpy
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from oamb.artifacts.validation.phase import phase_ai_review_evidence_closes
from oamb.cli import app
from oamb.contracts.accounting import (
    AggregationOperator,
    CostMeasurementSpec,
    MeasurementDimensionSpec,
    PriceSnapshot,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import ArtifactStorePort
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    AIReviewIntegrityResult,
    EvaluationReviewBundle,
    QualityReviewStatus,
)
from oamb.contracts.specifications import (
    BindingKind,
    BudgetScopeKindV2,
    BudgetSpecV2,
    ComparabilityStatus,
    ExecutionEnvironmentBinding,
    ExecutionOwner,
    ExternalCallApprovalRecord,
    ModelRole,
    ModelRoleBindingV2,
    ProviderBudgetCap,
    ResourceBudgetCeiling,
    RoleBindingStatus,
    RoleBudgetCeiling,
    external_call_approval_hash,
)
from oamb.model_clients.openai_compatible import OpenAICompatibleModelClient
from oamb.phase_cli_io import load_phase_review_evidence
from oamb.phase_review_repository import prepare_artifact_repository
from oamb.reporting.human_review import (
    create_human_quality_review_record,
    derive_evaluation_phase_gate,
)
from oamb.reporting.review import (
    build_ai_review_plan,
    build_evaluation_review_bundle,
    reduce_ai_quality_review,
)
from oamb.workloads.visible_evidence import o200k_encoding, tokenizer_fingerprint

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
NOW = datetime(2026, 8, 29, 1, tzinfo=UTC)
_TEST_ROOT = Path(__file__).parents[1]
_AI_FIXTURES = runpy.run_path(str(_TEST_ROOT / "unit" / "test_t8_ai_review.py"))
_bundle = _AI_FIXTURES["_bundle"]
_integrity_projection = _AI_FIXTURES["_integrity_projection"]
_projections = _AI_FIXTURES["_projections"]


@dataclass(frozen=True)
class PhaseRunFixture:
    arguments: list[str]
    approval_path: Path
    artifact_repository: Path
    budget_path: Path
    bundle_path: Path
    cost_measurement_spec_path: Path
    occurrence_path: Path
    price_snapshot_path: Path | None
    output_directory: Path
    plan_path: Path
    expected_attempts: int
    response_texts: tuple[str, ...]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def _token_counter(text: str) -> int:
    return len(o200k_encoding().encode_ordinary(text))


def _phase_run_fixture(tmp_path: Path, *, client_kind: str = "fake") -> PhaseRunFixture:
    is_openai = client_kind == "openai-compatible"
    review = importlib.import_module("oamb.reporting.review")
    projections = tuple(
        sorted(
            _projections(review),
            key=lambda item: (
                item.memory_system_id,
                item.workload_id,
                item.manifest_ordinal,
                item.case_occurrence_id,
            ),
        )
    )
    base_bundle = _bundle()
    bundle = build_evaluation_review_bundle(
        phase_id=base_bundle.phase_id,
        ordered_capsule_hashes=base_bundle.ordered_capsule_hashes,
        ordered_validation_hashes=base_bundle.ordered_validation_hashes,
        ordered_case_occurrence_ids=tuple(item.case_occurrence_id for item in projections),
        unique_case_manifest_entry_ids=base_bundle.unique_case_manifest_entry_ids,
        ordinary_derivation_hashes=base_bundle.ordinary_derivation_hashes,
        report_model_hash=base_bundle.report_model_hash,
        report_html_hash=base_bundle.report_html_hash,
        export_validation_hash=base_bundle.export_validation_hash,
        limitations=base_bundle.limitations,
    )
    role = ModelRoleBindingV2(
        binding_id=SHA_B,
        role=ModelRole.QUALITY_REVIEW,
        role_status=RoleBindingStatus.SELECTED,
        execution_owner=ExecutionOwner.HARNESS,
        binding_kind=BindingKind.MODEL_CLIENT,
        provider="openai-compatible-fixture" if is_openai else "oamb-fake-reviewer",
        endpoint_reference=("https://models.example/v1" if is_openai else "offline://phase-review"),
        credential_variable_name=(
            "OAMB_QUALITY_REVIEW_API_KEY" if is_openai else "OAMB_FAKE_PHASE_REVIEW_TOKEN"
        ),
        configured_model="review-model" if is_openai else "fake-review-v1",
        resolved_model=("review-model@runtime" if is_openai else "fake-review-v1@recorded"),
        parameters_fingerprint=SHA_A,
        retry_policy_id="no-retry-v1",
        configuration_fingerprint=SHA_C,
        redacted_endpoint_fingerprint=SHA_A,
    )
    environment = ExecutionEnvironmentBinding(
        environment_hash=SHA_B,
        operating_system="fixture-os",
        architecture="fixture-arch",
        python_version="3.11",
        cpu_description="fixture-cpu",
        memory_bytes=1_000_000,
        comparability_status=ComparabilityStatus.COMPARABLE,
    )
    reviewer_model_hash = canonical_sha256(
        [
            "oamb-phase-review-model-binding-v1",
            role.configured_model,
            role.resolved_model,
            role.parameters_fingerprint,
        ]
    )
    reviewer_runtime_hash = canonical_sha256(
        [
            "oamb-phase-review-runtime-binding-v1",
            client_kind,
            role.redacted_endpoint_fingerprint,
            environment.environment_hash,
        ]
    )
    plan_build = build_ai_review_plan(
        bundle,
        projections,
        _integrity_projection(review),
        projection_spec_hash=SHA_A,
        prompt_pack_hash=SHA_B,
        output_contract_hash=SHA_C,
        parser_hash=SHA_A,
        reviewer_role_binding_hash=role.binding_id,
        reviewer_model_hash=reviewer_model_hash,
        reviewer_runtime_hash=reviewer_runtime_hash,
        reviewer_configuration_hash=role.configuration_fingerprint,
        token_counter=_token_counter,
        counter_fingerprint=tokenizer_fingerprint(),
        model_context_window_tokens=32_768,
        aggregate_version="ai-quality-aggregate-v1",
    )
    plan = plan_build.plan
    evidence_contracts = importlib.import_module("oamb.contracts.evidence")
    artifact_repository, artifact_repository_fingerprint = prepare_artifact_repository(
        tmp_path / "artifacts"
    )
    occurrence_id = evidence_contracts.phase_review_occurrence_id_v2(
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        reviewer_role_binding_hash=role.binding_id,
        artifact_repository_fingerprint=artifact_repository_fingerprint,
        ordinal=1,
    )
    price_snapshot = (
        PriceSnapshot(
            price_snapshot_id="phase-review-price-v1",
            provider=role.provider or "",
            effective_at=NOW,
            currency="USD",
            price_class_ids=(
                "quality_review_input_tokens_per_million",
                "quality_review_output_tokens_per_million",
            ),
            unit_prices=(Decimal("1"), Decimal("2")),
        )
        if is_openai
        else None
    )
    resource_ceiling = ResourceBudgetCeiling(
        dimension_id="provider_request_wall_seconds_v1",
        maximum=Decimal("30"),
        unit="seconds",
    )
    planned_input_tokens = sum(batch.input_tokens for batch in plan.case_batches) + (
        plan.phase_integrity_input_tokens
    )
    planned_output_tokens = sum(batch.maximal_output_tokens for batch in plan.case_batches) + (
        plan.phase_integrity_maximal_output_tokens
    )
    role_ceiling = RoleBudgetCeiling(
        role_binding_id=role.binding_id,
        max_attempts=plan.expected_attempt_count,
        max_input_tokens=planned_input_tokens,
        max_output_tokens=planned_output_tokens,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=Decimal("1"),
        currency="USD",
        price_snapshot_id=(price_snapshot.price_snapshot_id if price_snapshot else None),
        resource_ceilings=(resource_ceiling,),
        provider_budget_cap=ProviderBudgetCap(
            provider=role.provider or "",
            operation_kind="quality_review",
            billing_unit="request",
            maximum_accepted_units=Decimal(plan.expected_attempt_count),
        ),
    )
    budget = BudgetSpecV2(
        budget_id="phase-review-budget",
        scope_kind=BudgetScopeKindV2.PHASE_REVIEW,
        scope_id=occurrence_id,
        approval_id="phase-review-approval",
        max_attempts=plan.expected_attempt_count,
        max_input_tokens=planned_input_tokens,
        max_output_tokens=planned_output_tokens,
        max_dispatch_wall_seconds=Decimal("30"),
        max_cost=Decimal("1"),
        currency="USD",
        resource_ceilings=(resource_ceiling,),
        role_ceilings=(role_ceiling,),
        stop_condition_ids=("budget_exhausted",),
    )
    approval_fields: dict[str, Any] = {
        "approval_id": budget.approval_id,
        "operation_kind": "phase_review",
        "scope_kind": BudgetScopeKindV2.PHASE_REVIEW,
        "scope_id": occurrence_id,
        "runtime_binding_hash": None,
        "provider_runtime_profile_attestation_hash": None,
        "role_binding_ids": (role.binding_id,),
        "budget_hash": canonical_sha256(budget),
        "approved_at": NOW,
        "expires_at": NOW + timedelta(hours=1),
        "unmetered_cost_acknowledged": False,
        "stop_condition_ids": budget.stop_condition_ids,
    }
    approval = ExternalCallApprovalRecord.model_validate(
        {"approval_hash": external_call_approval_hash(approval_fields), **approval_fields}
    )
    occurrence = evidence_contracts.PhaseReviewOccurrenceRecordV2(
        phase_review_occurrence_id=occurrence_id,
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        reviewer_role_binding_hash=role.binding_id,
        artifact_repository_fingerprint=artifact_repository_fingerprint,
        ordinal=1,
        approval_record_id=approval.approval_hash,
        budget_id=budget.budget_id,
        state="planned",
        started_at=None,
        ended_at=None,
    )
    measurement = CostMeasurementSpec(
        measurement_spec_id="phase-review-measurement-v1",
        measurement_spec_version="1",
        dimensions=(
            MeasurementDimensionSpec(
                dimension_id="provider_request_wall_seconds_v1",
                stage="quality_review",
                operation_kind="quality_review",
                parent_kind="phase_review",
                unit="seconds",
                allowed_meter_sources=("process_meter" if is_openai else "fixture_clock_v1",),
                required=True,
                price_class=None,
                indexing_view_rule="not_applicable",
                aggregation_operator=AggregationOperator.INTERVAL_UNION,
            ),
        ),
    )

    inputs = tmp_path / "inputs"
    requests = tmp_path / "requests"
    responses = tmp_path / "responses"
    paths = {
        "bundle": inputs / "bundle.json",
        "plan": inputs / "plan.json",
        "occurrence": inputs / "occurrence.json",
        "approval": inputs / "approval.json",
        "budget": inputs / "budget.json",
        "role": inputs / "role.json",
        "measurement": inputs / "measurement.json",
        "environment": inputs / "environment.json",
        "price": inputs / "price.json",
    }
    for label, value in (
        ("bundle", bundle),
        ("plan", plan),
        ("occurrence", occurrence),
        ("approval", approval),
        ("budget", budget),
        ("role", role),
        ("measurement", measurement),
        ("environment", environment),
    ):
        _write_json(paths[label], value)
    if price_snapshot is not None:
        _write_json(paths["price"], price_snapshot)
    for index, request in enumerate(plan_build.case_requests, 1):
        request_path = requests / "case-requests" / f"{index:04d}.json"
        request_path.parent.mkdir(parents=True, exist_ok=True)
        request_path.write_text(request, encoding="utf-8")
    (requests / "integrity-request.json").parent.mkdir(parents=True, exist_ok=True)
    (requests / "integrity-request.json").write_text(plan_build.integrity_request, encoding="utf-8")
    response_paths: list[Path] = []
    for index, batch in enumerate(plan.case_batches, 1):
        response_path = responses / f"batch-{index}.json"
        _write_json(
            response_path,
            {
                "batch_id": batch.batch_id,
                "results": [
                    {"case_occurrence_id": case_id, "status": "pass", "findings": []}
                    for case_id in batch.ordered_case_occurrence_ids
                ],
                "status": "pass",
            },
        )
        response_paths.append(response_path)
    integrity_response = responses / "integrity.json"
    _write_json(
        integrity_response,
        {"integrity_id": plan.phase_integrity_id, "status": "pass", "findings": []},
    )
    response_paths.append(integrity_response)

    arguments = [
        "phase",
        "ai-review",
        "run",
        "--client",
        client_kind,
        "--bundle",
        str(paths["bundle"]),
        "--plan",
        str(paths["plan"]),
        "--occurrence",
        str(paths["occurrence"]),
        "--approval",
        str(paths["approval"]),
        "--budget",
        str(paths["budget"]),
        "--role-binding",
        str(paths["role"]),
        "--cost-measurement-spec",
        str(paths["measurement"]),
        "--execution-environment",
        str(paths["environment"]),
        "--request-directory",
        str(requests),
        "--artifact-repository",
        str(artifact_repository),
        "--output-directory",
        str(tmp_path / "run-output"),
    ]
    if is_openai:
        arguments.extend(("--price-snapshot", str(paths["price"])))
    else:
        arguments.extend(
            (
                "--started-at",
                (NOW + timedelta(minutes=1)).isoformat(),
                "--ended-at",
                (NOW + timedelta(minutes=2)).isoformat(),
            )
        )
        for response_path in response_paths:
            arguments.extend(("--fake-response", str(response_path)))
    return PhaseRunFixture(
        arguments=arguments,
        approval_path=paths["approval"],
        artifact_repository=artifact_repository,
        budget_path=paths["budget"],
        bundle_path=paths["bundle"],
        cost_measurement_spec_path=paths["measurement"],
        occurrence_path=paths["occurrence"],
        price_snapshot_path=paths["price"] if is_openai else None,
        output_directory=tmp_path / "run-output",
        plan_path=paths["plan"],
        expected_attempts=plan.expected_attempt_count,
        response_texts=tuple(path.read_text(encoding="utf-8") for path in response_paths),
    )


def _rewrite_phase_budget(
    fixture: PhaseRunFixture,
    *,
    max_cost: Decimal | None = None,
    max_wall_seconds: Decimal | None = None,
    max_resource_seconds: Decimal | None = None,
) -> None:
    budget = BudgetSpecV2.model_validate_json(fixture.budget_path.read_bytes())
    role = budget.role_ceilings[0]
    role_updates: dict[str, object] = {}
    budget_updates: dict[str, object] = {}
    if max_cost is not None:
        role_updates["max_cost"] = max_cost
        budget_updates["max_cost"] = max_cost
    if max_wall_seconds is not None:
        role_updates["max_dispatch_wall_seconds"] = max_wall_seconds
        role_updates["resource_ceilings"] = (
            role.resource_ceilings[0].model_copy(update={"maximum": max_wall_seconds}),
        )
        budget_updates["max_dispatch_wall_seconds"] = max_wall_seconds
        budget_updates["resource_ceilings"] = (
            budget.resource_ceilings[0].model_copy(update={"maximum": max_wall_seconds}),
        )
    if max_resource_seconds is not None:
        role_updates["resource_ceilings"] = (
            role.resource_ceilings[0].model_copy(update={"maximum": max_resource_seconds}),
        )
        budget_updates["resource_ceilings"] = (
            budget.resource_ceilings[0].model_copy(update={"maximum": max_resource_seconds}),
        )
    changed_role = role.model_copy(update=role_updates)
    changed_budget = budget.model_copy(update={**budget_updates, "role_ceilings": (changed_role,)})
    _write_json(fixture.budget_path, changed_budget)
    approval = ExternalCallApprovalRecord.model_validate_json(fixture.approval_path.read_bytes())
    approval_fields = approval.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "approval_hash"},
    )
    approval_fields["budget_hash"] = canonical_sha256(changed_budget)
    approval_hash = external_call_approval_hash(approval_fields)
    changed_approval = ExternalCallApprovalRecord.model_validate(
        {"approval_hash": approval_hash, **approval_fields}
    )
    _write_json(fixture.approval_path, changed_approval)
    evidence_contracts = importlib.import_module("oamb.contracts.evidence")
    occurrence = evidence_contracts.PhaseReviewOccurrenceRecordV2.model_validate_json(
        fixture.occurrence_path.read_bytes()
    )
    _write_json(
        fixture.occurrence_path,
        occurrence.model_copy(update={"approval_record_id": approval_hash}),
    )


def _rewrite_phase_ordinal(fixture: PhaseRunFixture, ordinal: int) -> None:
    evidence_contracts = importlib.import_module("oamb.contracts.evidence")
    occurrence = evidence_contracts.PhaseReviewOccurrenceRecordV2.model_validate_json(
        fixture.occurrence_path.read_bytes()
    )
    occurrence_id = evidence_contracts.phase_review_occurrence_id_v2(
        phase_id=occurrence.phase_id,
        review_bundle_hash=occurrence.review_bundle_hash,
        reviewer_role_binding_hash=occurrence.reviewer_role_binding_hash,
        artifact_repository_fingerprint=occurrence.artifact_repository_fingerprint,
        ordinal=ordinal,
    )
    budget = BudgetSpecV2.model_validate_json(fixture.budget_path.read_bytes()).model_copy(
        update={"scope_id": occurrence_id}
    )
    _write_json(fixture.budget_path, budget)
    approval = ExternalCallApprovalRecord.model_validate_json(fixture.approval_path.read_bytes())
    approval_fields = approval.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "approval_hash"},
    )
    approval_fields.update({"scope_id": occurrence_id, "budget_hash": canonical_sha256(budget)})
    approval_hash = external_call_approval_hash(approval_fields)
    _write_json(
        fixture.approval_path,
        ExternalCallApprovalRecord.model_validate(
            {"approval_hash": approval_hash, **approval_fields}
        ),
    )
    _write_json(
        fixture.occurrence_path,
        occurrence.model_copy(
            update={
                "phase_review_occurrence_id": occurrence_id,
                "ordinal": ordinal,
                "approval_record_id": approval_hash,
            }
        ),
    )


def _operational_inconclusive_history(
    fixture: PhaseRunFixture,
    *,
    occurrence_id: str | None = None,
) -> tuple[AIQualityReviewRecord, ...]:
    plan = importlib.import_module(
        "oamb.contracts.specifications"
    ).AIReviewPlan.model_validate_json(fixture.plan_path.read_bytes())
    occurrence = json.loads(fixture.occurrence_path.read_bytes())
    record = reduce_ai_quality_review(
        plan,
        batch_results=(),
        integrity_result=AIReviewIntegrityResult(
            integrity_id=plan.phase_integrity_id,
            status=QualityReviewStatus.PASS,
            findings=(),
        ),
        occurrence_id=occurrence_id or occurrence["phase_review_occurrence_id"],
        ordinal=1,
        previous_ai_review_record_hash=None,
        previous_history_root_hash=None,
        attempt_ids=(SHA_A,),
        usage_record_ids=(),
        resource_record_ids=(),
        cost_record_ids=(),
        accounting_closed=False,
        created_at=NOW + timedelta(minutes=3),
    )
    return (record,)


def test_phase_ai_review_run_uses_the_bounded_fake_client_and_retains_evidence(
    tmp_path: Path,
) -> None:
    fixture = _phase_run_fixture(tmp_path)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code == 0, result.output
    record = AIQualityReviewRecord.model_validate_json(
        (fixture.output_directory / "ai-review-record.json").read_bytes()
    )
    assert record.status == "pass"
    assert len(json.loads((fixture.output_directory / "attempts.json").read_bytes())) == (
        fixture.expected_attempts
    )
    assert len(json.loads((fixture.output_directory / "usage-records.json").read_bytes())) == (
        fixture.expected_attempts
    )
    assert len(json.loads((fixture.output_directory / "resource-records.json").read_bytes())) == (
        fixture.expected_attempts
    )
    assert len(json.loads((fixture.output_directory / "cost-records.json").read_bytes())) == (
        fixture.expected_attempts
    )
    costs = json.loads((fixture.output_directory / "cost-records.json").read_bytes())
    assert {item["proof_status"] for item in costs} == {"not_applicable"}
    bundle = EvaluationReviewBundle.model_validate_json(fixture.bundle_path.read_bytes())
    retained_evidence = load_phase_review_evidence(fixture.output_directory)
    assert phase_ai_review_evidence_closes(bundle, retained_evidence, record)
    assert all(item["amount"] is None and item["currency"] is None for item in costs)
    assert all(
        item["reason"] == "credential-free fake review has no supplier billing" for item in costs
    )
    assert {
        "plan.json",
        "role-binding.json",
        "approval.json",
        "budget.json",
        "cost-measurement-spec.json",
        "execution-environment.json",
    }.issubset(path.name for path in fixture.output_directory.iterdir())
    assert not (fixture.output_directory / "price-snapshot.json").exists()
    assert "fake client completed" in result.output


def test_fake_runner_outputs_validate_through_the_t10_phase_gate(
    tmp_path: Path,
) -> None:
    fixture = _phase_run_fixture(tmp_path)
    runner = CliRunner()

    run_result = runner.invoke(app, fixture.arguments)

    assert run_result.exit_code == 0, run_result.output
    bundle = EvaluationReviewBundle.model_validate_json(fixture.bundle_path.read_bytes())
    ai_record = AIQualityReviewRecord.model_validate_json(
        (fixture.output_directory / "ai-review-record.json").read_bytes()
    )
    human_record = create_human_quality_review_record(
        canonical_ai_record=ai_record,
        status="pass",
        nonce="phase-runner-e2e-001",
        operator_id="fixture-operator",
        finding_codes=(),
        evidence_references=(),
        used_nonces=(),
        created_at=NOW + timedelta(minutes=4),
    )
    gate = derive_evaluation_phase_gate(
        phase_id=bundle.phase_id,
        review_bundle_hash=bundle.bundle_id,
        ordered_ai_history=(ai_record,),
        human_records=(human_record,),
    )
    assert json.loads((fixture.output_directory / "ai-history.json").read_bytes()) == [
        json.loads(canonical_json_bytes(ai_record))
    ]
    gate_path = tmp_path / "gate.json"
    validation_path = tmp_path / "gate-validation.json"
    _write_json(gate_path, gate)
    _write_json(fixture.output_directory / "human-review.json", human_record)

    validated = runner.invoke(
        app,
        [
            "phase",
            "gate",
            "validate",
            "--bundle",
            str(fixture.bundle_path),
            "--gate",
            str(gate_path),
            "--review-evidence-directory",
            str(fixture.output_directory),
            "--output",
            str(validation_path),
        ],
    )

    assert validated.exit_code == 0, validated.output
    assert json.loads(validation_path.read_bytes())["disposition"] == "validated"


def test_phase_ai_review_run_rejects_approval_drift_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path)
    approval = json.loads(fixture.approval_path.read_bytes())
    approval["role_binding_ids"] = [SHA_A]
    approval["approval_hash"] = external_call_approval_hash(
        {
            key: value
            for key, value in approval.items()
            if key not in {"schema_name", "schema_version", "approval_hash"}
        }
    )
    fixture.approval_path.write_bytes(canonical_json_bytes(approval))
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed before preflight closed")

    monkeypatch.setattr(run_module, "_build_fake_client", forbidden_factory)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert not fixture.output_directory.exists()
    assert "planned phase-review occurrence" in result.output


def _install_openai_mock(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request, int], httpx.Response],
) -> list[httpx.Request]:
    run_module = importlib.import_module("oamb.phase_review_run")
    calls: list[httpx.Request] = []

    def capturing_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return handler(request, len(calls))

    def client_factory(
        store: ArtifactStorePort,
        role: ModelRoleBindingV2,
        api_key: str,
    ) -> OpenAICompatibleModelClient:
        return OpenAICompatibleModelClient(
            store=store,
            base_url=role.endpoint_reference or "",
            api_key=api_key,
            role_binding=role,
            runtime_model_policy="require_match",
            reasoning_control=("reasoning_effort", "none"),
            transport=httpx.MockTransport(capturing_handler),
        )

    monkeypatch.setattr(run_module, "_build_openai_client", client_factory)
    utc_tick = 0
    monotonic_tick = 0

    def utc_now() -> datetime:
        nonlocal utc_tick
        utc_tick += 1
        return NOW + timedelta(minutes=1, milliseconds=utc_tick)

    def monotonic_now() -> float:
        nonlocal monotonic_tick
        monotonic_tick += 1
        return monotonic_tick * 0.25

    monkeypatch.setattr(run_module, "_utc_now", utc_now)
    monkeypatch.setattr(run_module, "_monotonic_now", monotonic_now)
    monkeypatch.setenv("OAMB_QUALITY_REVIEW_API_KEY", "fixture-secret")
    return calls


def test_phase_ai_review_run_uses_openai_transport_and_measures_each_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(_request: httpx.Request, ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code == 0, result.output
    assert len(calls) == fixture.expected_attempts
    request_payload = json.loads(calls[0].content)
    assert request_payload["response_format"] == {"type": "json_object"}
    resources = json.loads((fixture.output_directory / "resource-records.json").read_bytes())
    assert all(Decimal(item["value"]) > 0 for item in resources)
    attempts = json.loads((fixture.output_directory / "attempts.json").read_bytes())
    assert all(item["started_at"] < item["ended_at"] for item in attempts)
    costs = json.loads((fixture.output_directory / "cost-records.json").read_bytes())
    assert {item["basis"] for item in costs} == {"estimate_from_measured_usage"}
    assert {item["proof_status"] for item in costs} == {"measured_complete"}
    assert {item["amount"] for item in costs} == {"0.000003"}
    assert (fixture.output_directory / "price-snapshot.json").is_file()


def test_phase_ai_review_run_seals_intent_before_each_dispatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    occurrence = json.loads(fixture.occurrence_path.read_bytes())
    source = (
        fixture.artifact_repository
        / "phase-reviews"
        / occurrence["phase_review_occurrence_id"]
        / "source"
    )
    observed: list[tuple[int, int, int, bool]] = []

    def handler(_request: httpx.Request, ordinal: int) -> httpx.Response:
        observed.append(
            (
                len(tuple((source / "occurrence-claims").glob("*.json"))),
                len(tuple((source / "budget-reservations").glob("*.json"))),
                len(tuple((source / "attempt-intents").glob("*.json"))),
                (
                    fixture.artifact_repository
                    / ".runtime"
                    / "phase-review"
                    / "active-provider-attempt"
                ).is_file(),
            )
        )
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code == 0, result.output
    assert len(calls) == fixture.expected_attempts
    assert observed == [
        (ordinal, ordinal, ordinal, True) for ordinal in range(1, fixture.expected_attempts + 1)
    ]
    occurrence = json.loads(fixture.occurrence_path.read_bytes())
    source = (
        fixture.artifact_repository
        / "phase-reviews"
        / occurrence["phase_review_occurrence_id"]
        / "source"
    )
    assert len(tuple((source / "ai-review-records").glob("*.json"))) == 1
    assert len(tuple((source / "ai-review-histories").glob("*.json"))) == 1


def test_phase_ai_review_run_preserves_success_with_unavailable_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(_request: httpx.Request, ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert len(calls) == 1
    occurrence = json.loads((fixture.output_directory / "occurrence.json").read_bytes())
    attempts = json.loads((fixture.output_directory / "attempts.json").read_bytes())
    usage = json.loads((fixture.output_directory / "usage-records.json").read_bytes())
    resources = json.loads((fixture.output_directory / "resource-records.json").read_bytes())
    costs = json.loads((fixture.output_directory / "cost-records.json").read_bytes())
    assert occurrence["state"] == "evidence_inconclusive"
    assert [item["outcome"] for item in attempts] == ["succeeded"]
    assert [item["proof_status"] for item in usage] == ["unavailable"]
    assert [item["proof_status"] for item in resources] == ["measured_complete"]
    assert [item["proof_status"] for item in costs] == ["unavailable"]
    assert costs[0]["amount"] is None
    assert costs[0]["currency"] is None
    assert not (fixture.output_directory / "ai-review-record.json").exists()


def test_phase_ai_review_run_preserves_failed_attempt_measured_cost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(_request: httpx.Request, _ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "invalid-quality-response",
                "model": "review-model@runtime",
                "choices": [{"message": {"content": "not-json"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert len(calls) == 1
    attempts = json.loads((fixture.output_directory / "attempts.json").read_bytes())
    costs = json.loads((fixture.output_directory / "cost-records.json").read_bytes())
    assert [item["outcome"] for item in attempts] == ["failed"]
    assert [item["proof_status"] for item in costs] == ["measured_complete"]
    assert costs[0]["amount"] == "0.00002"
    assert costs[0]["price_snapshot_id"] == "phase-review-price-v1"


def test_phase_ai_review_run_preserves_primary_failure_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(_request: httpx.Request, _ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "invalid-quality-response",
                "model": "review-model@runtime",
                "choices": [{"message": {"content": "not-json"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)
    original_close = OpenAICompatibleModelClient.close

    async def failing_close(client: OpenAICompatibleModelClient) -> None:
        await original_close(client)
        raise RuntimeError("planted close failure")

    monkeypatch.setattr(OpenAICompatibleModelClient, "close", failing_close)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert len(calls) == 1
    assert "dispatch 1 failed" in result.output
    occurrence = json.loads(fixture.occurrence_path.read_bytes())
    close_errors = tuple(
        (
            fixture.artifact_repository
            / "phase-reviews"
            / occurrence["phase_review_occurrence_id"]
            / "source"
            / "close-errors"
        ).glob("*.json")
    )
    assert len(close_errors) == 1
    retained_occurrence = json.loads((fixture.output_directory / "occurrence.json").read_bytes())
    assert retained_occurrence["state"] == "error"


def test_phase_ai_review_run_retains_complete_attempts_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(_request: httpx.Request, ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)
    original_close = OpenAICompatibleModelClient.close

    async def failing_close(client: OpenAICompatibleModelClient) -> None:
        await original_close(client)
        raise RuntimeError("planted close failure")

    monkeypatch.setattr(OpenAICompatibleModelClient, "close", failing_close)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert len(calls) == fixture.expected_attempts
    attempts = json.loads((fixture.output_directory / "attempts.json").read_bytes())
    occurrence = json.loads((fixture.output_directory / "occurrence.json").read_bytes())
    assert len(attempts) == fixture.expected_attempts
    assert occurrence["state"] == "error"
    assert not (fixture.output_directory / "ai-review-record.json").exists()


def test_phase_ai_review_run_terminalizes_post_receipt_accounting_seal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(_request: httpx.Request, ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)
    run_module = importlib.import_module("oamb.phase_review_run")
    original_seal = run_module._seal_accounting_evidence
    invocations = 0

    def failing_seal(*args: object, **kwargs: object) -> None:
        nonlocal invocations
        invocations += 1
        if invocations == 1:
            raise OSError("planted accounting seal failure")
        original_seal(*args, **kwargs)

    monkeypatch.setattr(run_module, "_seal_accounting_evidence", failing_seal)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert len(calls) == 1
    retained_occurrence = json.loads((fixture.output_directory / "occurrence.json").read_bytes())
    assert retained_occurrence["state"] == "error"
    occurrence = json.loads(fixture.occurrence_path.read_bytes())
    source = (
        fixture.artifact_repository
        / "phase-reviews"
        / occurrence["phase_review_occurrence_id"]
        / "source"
    )
    assert len(tuple((source / "attempt-receipts").glob("*.json"))) == 1
    assert len(tuple((source / "attempts").glob("*.json"))) == 1
    assert len(tuple((source / "phase-review-occurrences").glob("*.json"))) == 1
    assert not (
        fixture.artifact_repository / ".runtime" / "phase-review" / "active-run-lease"
    ).exists()


def test_phase_ai_review_run_missing_credential_constructs_no_openai_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed without its credential")

    monkeypatch.setattr(run_module, "_build_openai_client", forbidden_factory)
    monkeypatch.delenv("OAMB_QUALITY_REVIEW_API_KEY", raising=False)
    monkeypatch.setattr(run_module, "_utc_now", lambda: NOW + timedelta(minutes=1))

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert "OAMB_QUALITY_REVIEW_API_KEY" in result.output
    assert not fixture.output_directory.exists()


def test_phase_ai_review_run_rejects_future_price_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    assert fixture.price_snapshot_path is not None
    price_snapshot = json.loads(fixture.price_snapshot_path.read_bytes())
    price_snapshot["effective_at"] = (NOW + timedelta(days=30)).isoformat()
    fixture.price_snapshot_path.write_bytes(canonical_json_bytes(price_snapshot))
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed before price preflight closed")

    monkeypatch.setattr(run_module, "_build_openai_client", forbidden_factory)
    monkeypatch.setattr(run_module, "_utc_now", lambda: NOW + timedelta(minutes=1))
    monkeypatch.setenv("OAMB_QUALITY_REVIEW_API_KEY", "fixture-secret")

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert "price snapshot is not effective" in result.output
    assert not fixture.output_directory.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (("provider", "different-provider"), ("currency", "EUR")),
)
def test_phase_ai_review_run_rejects_price_scope_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    assert fixture.price_snapshot_path is not None
    price_snapshot = json.loads(fixture.price_snapshot_path.read_bytes())
    price_snapshot[field] = value
    fixture.price_snapshot_path.write_bytes(canonical_json_bytes(price_snapshot))
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed before price preflight closed")

    monkeypatch.setattr(run_module, "_build_openai_client", forbidden_factory)
    monkeypatch.setattr(run_module, "_utc_now", lambda: NOW + timedelta(minutes=1))
    monkeypatch.setenv("OAMB_QUALITY_REVIEW_API_KEY", "fixture-secret")

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert not fixture.output_directory.exists()


def test_phase_ai_review_run_rejects_underfunded_planned_cost_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    _rewrite_phase_budget(fixture, max_cost=Decimal("0.000001"))
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed for an underfunded phase plan")

    monkeypatch.setattr(run_module, "_build_openai_client", forbidden_factory)
    monkeypatch.setattr(run_module, "_utc_now", lambda: NOW + timedelta(minutes=1))
    monkeypatch.setenv("OAMB_QUALITY_REVIEW_API_KEY", "fixture-secret")

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert not fixture.output_directory.exists()


def test_phase_ai_review_run_rejects_unowned_resource_meter_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    measurement = json.loads(fixture.cost_measurement_spec_path.read_bytes())
    measurement["dimensions"][0]["allowed_meter_sources"] = ["process-meter"]
    fixture.cost_measurement_spec_path.write_bytes(canonical_json_bytes(measurement))
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed before meter preflight closed")

    monkeypatch.setattr(run_module, "_build_openai_client", forbidden_factory)
    monkeypatch.setattr(run_module, "_utc_now", lambda: NOW + timedelta(minutes=1))
    monkeypatch.setenv("OAMB_QUALITY_REVIEW_API_KEY", "fixture-secret")

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert not fixture.output_directory.exists()


def test_phase_ai_review_run_preserves_attempt_two_timeout_and_stops_attempt_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(request: httpx.Request, ordinal: int) -> httpx.Response:
        if ordinal == 2:
            raise httpx.ReadTimeout("fixture timeout", request=request)
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert len(calls) == 2
    attempts = json.loads((fixture.output_directory / "attempts.json").read_bytes())
    assert [item["outcome"] for item in attempts] == ["succeeded", "unknown_outcome"]
    occurrence = json.loads((fixture.output_directory / "occurrence.json").read_bytes())
    assert occurrence["state"] == "interrupted_unknown_outcome"
    assert (fixture.output_directory / "cost-measurement-spec.json").is_file()
    assert (fixture.output_directory / "execution-environment.json").is_file()
    assert (fixture.output_directory / "price-snapshot.json").is_file()
    assert not (fixture.output_directory / "ai-review-record.json").exists()


def test_phase_ai_review_run_stops_before_next_dispatch_when_approval_expires(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(_request: httpx.Request, ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)
    run_module = importlib.import_module("oamb.phase_review_run")
    instants = iter(
        (
            NOW + timedelta(minutes=1),
            NOW + timedelta(minutes=1, milliseconds=1),
            NOW + timedelta(minutes=1, milliseconds=2),
            NOW + timedelta(minutes=1, milliseconds=3),
            NOW + timedelta(minutes=1, milliseconds=4),
            NOW + timedelta(hours=2),
            NOW + timedelta(hours=2, milliseconds=1),
        )
    )
    monkeypatch.setattr(run_module, "_utc_now", lambda: next(instants))

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert len(calls) == 1
    assert "approval expired" in result.output
    attempts = json.loads((fixture.output_directory / "attempts.json").read_bytes())
    assert [item["outcome"] for item in attempts] == ["succeeded"]
    occurrence = json.loads((fixture.output_directory / "occurrence.json").read_bytes())
    assert occurrence["state"] == "budget_exceeded"


def test_phase_ai_review_run_stops_on_measured_wall_time_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")

    def handler(_request: httpx.Request, ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)
    monotonic_instants = iter((0.0, 31.0, 31.0, 62.0, 62.0, 93.0))
    run_module = importlib.import_module("oamb.phase_review_run")
    monkeypatch.setattr(run_module, "_monotonic_now", lambda: next(monotonic_instants))

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert len(calls) == 1
    assert "budget ceiling" in result.output
    resource_records = json.loads((fixture.output_directory / "resource-records.json").read_bytes())
    assert [item["value"] for item in resource_records] == ["31"]
    occurrence = json.loads((fixture.output_directory / "occurrence.json").read_bytes())
    assert occurrence["state"] == "budget_exceeded"


def test_phase_ai_review_run_enforces_remaining_wall_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    _rewrite_phase_budget(fixture, max_wall_seconds=Decimal("0.01"))
    run_module = importlib.import_module("oamb.phase_review_run")

    class BlockingClient:
        calls = 0

        async def complete(self, _request: object) -> object:
            self.calls += 1
            await asyncio.sleep(60)
            raise AssertionError("phase-review deadline did not cancel the dispatch")

        async def close(self) -> None:
            return None

    client = BlockingClient()
    monkeypatch.setattr(run_module, "_build_openai_client", lambda *_args, **_kwargs: client)
    monkeypatch.setattr(run_module, "_utc_now", lambda: NOW + timedelta(minutes=1))
    monkeypatch.setenv("OAMB_QUALITY_REVIEW_API_KEY", "fixture-secret")

    started = time.monotonic()
    result = CliRunner().invoke(app, fixture.arguments)
    elapsed = time.monotonic() - started

    assert result.exit_code != 0
    assert client.calls == 1
    assert elapsed < 1
    occurrence = json.loads((fixture.output_directory / "occurrence.json").read_bytes())
    assert occurrence["state"] == "interrupted_unknown_outcome"


def test_phase_ai_review_run_reserves_the_smaller_resource_wall_ceiling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    _rewrite_phase_budget(fixture, max_resource_seconds=Decimal("10"))

    def handler(_request: httpx.Request, ordinal: int) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": f"response-{ordinal}",
                "model": "review-model@runtime",
                "choices": [
                    {
                        "message": {"content": fixture.response_texts[ordinal - 1]},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    calls = _install_openai_mock(monkeypatch, handler)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code == 0, result.output
    assert len(calls) == fixture.expected_attempts


def test_phase_ai_review_run_rejects_occurrence_replay_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path)
    run_module = importlib.import_module("oamb.phase_review_run")
    original_factory = run_module._build_fake_client
    constructions = 0

    def counting_factory(*args: object, **kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        return original_factory(*args, **kwargs)

    monkeypatch.setattr(run_module, "_build_fake_client", counting_factory)
    first = CliRunner().invoke(app, fixture.arguments)
    second_arguments = list(fixture.arguments)
    output_index = second_arguments.index("--output-directory") + 1
    second_arguments[output_index] = str(tmp_path / "second-run-output")

    second = CliRunner().invoke(app, second_arguments)

    assert first.exit_code == 0, first.output
    assert second.exit_code != 0
    assert constructions == 1
    occurrence = json.loads(fixture.occurrence_path.read_bytes())
    lease = (
        fixture.artifact_repository
        / "phase-reviews"
        / occurrence["phase_review_occurrence_id"]
        / "source"
        / "run-leases"
        / "1.json"
    )
    assert lease.is_file()
    assert not (tmp_path / "second-run-output").exists()


def test_phase_ai_review_run_rejects_ordinal_two_without_history_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path)
    _rewrite_phase_ordinal(fixture, 2)
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed before predecessor history closed")

    monkeypatch.setattr(run_module, "_build_fake_client", forbidden_factory)

    result = CliRunner().invoke(app, fixture.arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert not fixture.output_directory.exists()


def test_phase_ai_review_run_supersedes_operational_inconclusive_history(
    tmp_path: Path,
) -> None:
    fixture = _phase_run_fixture(tmp_path)
    history = _operational_inconclusive_history(fixture)
    history_path = tmp_path / "ai-history.json"
    _write_json(history_path, history)
    _rewrite_phase_ordinal(fixture, 2)
    arguments = [*fixture.arguments, "--previous-ai-history", str(history_path)]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code == 0, result.output
    record = AIQualityReviewRecord.model_validate_json(
        (fixture.output_directory / "ai-review-record.json").read_bytes()
    )
    retained_history = json.loads((fixture.output_directory / "ai-history.json").read_bytes())
    assert record.ordinal == 2
    assert record.previous_ai_review_record_hash == history[0].ai_review_record_id
    assert record.previous_history_root_hash == history[0].history_root_hash
    assert len(retained_history) == 2


def test_phase_ai_review_run_rejects_foreign_predecessor_lineage_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path)
    evidence_contracts = importlib.import_module("oamb.contracts.evidence")
    occurrence = evidence_contracts.PhaseReviewOccurrenceRecordV2.model_validate_json(
        fixture.occurrence_path.read_bytes()
    )
    foreign_occurrence_id = evidence_contracts.phase_review_occurrence_id_v2(
        phase_id=occurrence.phase_id,
        review_bundle_hash=occurrence.review_bundle_hash,
        reviewer_role_binding_hash=occurrence.reviewer_role_binding_hash,
        artifact_repository_fingerprint=SHA_B,
        ordinal=1,
    )
    history = _operational_inconclusive_history(
        fixture,
        occurrence_id=foreign_occurrence_id,
    )
    history_path = tmp_path / "foreign-ai-history.json"
    _write_json(history_path, history)
    _rewrite_phase_ordinal(fixture, 2)
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed before predecessor lineage closed")

    monkeypatch.setattr(run_module, "_build_fake_client", forbidden_factory)
    arguments = [*fixture.arguments, "--previous-ai-history", str(history_path)]

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert not fixture.output_directory.exists()


def test_phase_ai_review_run_rejects_different_bound_repository_before_client_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _phase_run_fixture(tmp_path, client_kind="openai-compatible")
    arguments = list(fixture.arguments)
    repository_index = arguments.index("--artifact-repository") + 1
    arguments[repository_index] = str(tmp_path / "different-artifacts")
    run_module = importlib.import_module("oamb.phase_review_run")
    constructions = 0

    def forbidden_factory(*_args: object, **_kwargs: object) -> object:
        nonlocal constructions
        constructions += 1
        raise AssertionError("client constructed for an unbound artifact repository")

    monkeypatch.setattr(run_module, "_build_openai_client", forbidden_factory)
    monkeypatch.setattr(run_module, "_utc_now", lambda: NOW + timedelta(minutes=1))
    monkeypatch.setenv("OAMB_QUALITY_REVIEW_API_KEY", "fixture-secret")

    result = CliRunner().invoke(app, arguments)

    assert result.exit_code != 0
    assert constructions == 0
    assert not fixture.output_directory.exists()
