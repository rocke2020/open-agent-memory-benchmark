from __future__ import annotations

import hashlib
import importlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

HASH = "a" * 64
OTHER_HASH = "b" * 64
THIRD_HASH = "c" * 64
NOW = datetime(2026, 8, 29, 12, 0, tzinfo=UTC)


def require(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as exc:
        pytest.fail(f"{module_name} T10 live contracts are not implemented: {exc}", pytrace=False)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _resource_ceiling(specifications: ModuleType) -> Any:
    return specifications.ResourceBudgetCeiling(
        dimension_id="provider_request_wall_seconds_v1",
        maximum=Decimal("30"),
        unit="seconds",
    )


def _role_ceiling(specifications: ModuleType, role_binding_id: str) -> Any:
    return specifications.RoleBudgetCeiling(
        role_binding_id=role_binding_id,
        max_attempts=2,
        max_input_tokens=4096,
        max_output_tokens=4096,
        max_dispatch_wall_seconds=Decimal("60"),
        max_cost=None,
        currency=None,
        price_snapshot_id=None,
        resource_ceilings=(_resource_ceiling(specifications),),
        provider_budget_cap=specifications.ProviderBudgetCap(
            provider="deepseek",
            operation_kind="provider_internal_model_usage",
            billing_unit="request",
            maximum_accepted_units=Decimal("2"),
        ),
    )


def _provider_operation_ceiling(specifications: ModuleType) -> Any:
    values = dict(
        provider_operation_ceiling_id="mem0-add-call-cap",
        adapter_profile_id="mem0-rest-v2.0.19",
        operation_kind="memory_ingest",
        billing_unit="request",
        maximum_accepted_units=Decimal("1"),
        max_attempts=1,
        max_dispatch_wall_seconds=Decimal("30"),
        resource_ceilings=(_resource_ceiling(specifications),),
    )
    return specifications.ProviderOperationBudgetCeiling(
        provider_operation_ceiling_hash=(
            specifications.provider_operation_budget_ceiling_hash(values)
        ),
        **values,
    )


def _dispatch_route(specifications: ModuleType) -> Any:
    values = dict(
        route_id="mem0-ingest-route",
        stage="memory_ingest",
        dispatch_owner_kind=specifications.DispatchBudgetOwnerKind.PROVIDER_OPERATION,
        dispatch_model_role_binding_id=None,
        provider_operation_ceiling_id="mem0-add-call-cap",
        adapter_profile_id="mem0-rest-v2.0.19",
        operation_kind="memory_ingest",
        billing_unit="request",
        internal_usage_role_binding_ids=("mem0-extraction", "controlled-embedding"),
    )
    return specifications.DispatchBudgetRoute(
        route_hash=specifications.dispatch_budget_route_hash(values),
        **values,
    )


def _source_binding(specifications: ModuleType, *, identity: str, root_hash: str) -> Any:
    return specifications.SourceEvidenceBinding(
        binding_id=HASH,
        source_kind=specifications.SourceEvidenceKind.PROVIDER_SERVICE,
        source_identity=identity,
        source_root_hash=root_hash,
        validation_result_hash=THIRD_HASH,
        source_schema_versions=("provider_service_evidence_manifest@1",),
    )


def test_conformance_budget_uses_new_scope_and_discriminated_dispatch_owner() -> None:
    specifications = require("oamb.contracts.specifications")

    provider_operation = _provider_operation_ceiling(specifications)
    route = _dispatch_route(specifications)
    budget = specifications.BudgetSpecV3(
        budget_id="mem0-conformance-budget",
        scope_kind=specifications.BudgetScopeKindV3.MEMORY_CONFORMANCE,
        scope_id="mem0-conformance-occurrence",
        approval_id="mem0-conformance-approval",
        max_attempts=3,
        max_input_tokens=8192,
        max_output_tokens=8192,
        max_dispatch_wall_seconds=Decimal("120"),
        max_cost=None,
        currency=None,
        resource_ceilings=(_resource_ceiling(specifications),),
        role_ceilings=(
            _role_ceiling(specifications, "mem0-extraction"),
            _role_ceiling(specifications, "controlled-embedding"),
        ),
        provider_operation_ceilings=(provider_operation,),
        dispatch_routes=(route,),
        stop_condition_ids=("unknown_outcome", "identity_drift"),
    )

    assert budget.scope_kind == specifications.BudgetScopeKindV3.MEMORY_CONFORMANCE
    assert budget.dispatch_routes[0].internal_usage_role_binding_ids == (
        "mem0-extraction",
        "controlled-embedding",
    )

    with pytest.raises(ValidationError):
        specifications.BudgetSpecV2(
            **(
                budget.model_dump()
                | {
                    "schema_version": 2,
                    "scope_kind": "memory_conformance",
                }
            )
        )
    with pytest.raises(ValidationError, match="provider-operation ceiling"):
        specifications.BudgetSpecV3(**(budget.model_dump() | {"provider_operation_ceilings": ()}))


def test_conformance_route_must_exactly_match_its_provider_operation_ceiling() -> None:
    specifications = require("oamb.contracts.specifications")
    route_values = _dispatch_route(specifications).model_dump(exclude={"route_hash"})
    route_values["billing_unit"] = "document"
    mismatched_route = specifications.DispatchBudgetRoute(
        route_hash=specifications.dispatch_budget_route_hash(route_values),
        **route_values,
    )

    with pytest.raises(ValidationError, match="route and provider-operation ceiling"):
        specifications.BudgetSpecV3(
            budget_id="mem0-conformance-budget",
            scope_kind=specifications.BudgetScopeKindV3.MEMORY_CONFORMANCE,
            scope_id="mem0-conformance-occurrence",
            approval_id="mem0-conformance-approval",
            max_attempts=3,
            max_input_tokens=8192,
            max_output_tokens=8192,
            max_dispatch_wall_seconds=Decimal("120"),
            max_cost=None,
            currency=None,
            resource_ceilings=(_resource_ceiling(specifications),),
            role_ceilings=(
                _role_ceiling(specifications, "mem0-extraction"),
                _role_ceiling(specifications, "controlled-embedding"),
            ),
            provider_operation_ceilings=(_provider_operation_ceiling(specifications),),
            dispatch_routes=(mismatched_route,),
            stop_condition_ids=("unknown_outcome",),
        )


def test_conformance_transaction_versions_reject_old_wire_forms() -> None:
    accounting = require("oamb.contracts.accounting")
    evidence = require("oamb.contracts.evidence")
    specifications = require("oamb.contracts.specifications")
    states = require("oamb.contracts.states")

    reservation = evidence.BudgetReservationRecordV2(
        reservation_id=HASH,
        budget_id="mem0-conformance-budget",
        scope_kind=specifications.BudgetScopeKindV3.MEMORY_CONFORMANCE,
        scope_id="mem0-conformance-occurrence",
        dispatch_owner_kind=specifications.DispatchBudgetOwnerKind.PROVIDER_OPERATION,
        role_binding_id=None,
        provider_operation_ceiling_id="mem0-add-call-cap",
        internal_usage_role_binding_ids=("mem0-extraction", "controlled-embedding"),
        attempt_id=OTHER_HASH,
        reserved_attempts=1,
        reserved_input_tokens=4096,
        reserved_output_tokens=4096,
        reserved_dispatch_wall_seconds=Decimal("30"),
        reserved_cost=None,
        currency=None,
        reserved_resource_ceilings=(_resource_ceiling(specifications),),
        reserved_provider_units=Decimal("1"),
        reserved_at=NOW,
    )
    intent = evidence.AttemptIntentRecordV2(
        attempt_id=OTHER_HASH,
        claim_id=HASH,
        reservation_id=reservation.reservation_id,
        parent_kind="memory_conformance",
        parent_id="mem0-conformance-occurrence",
        dispatch_route_id="mem0-ingest-route",
        dispatch_owner_kind=specifications.DispatchBudgetOwnerKind.PROVIDER_OPERATION,
        role_binding_id=None,
        provider_operation_ceiling_id="mem0-add-call-cap",
        internal_usage_role_binding_ids=("mem0-extraction", "controlled-embedding"),
        stage="memory_ingest",
        request_fingerprint=THIRD_HASH,
        reconciliation_capability="none",
        idempotency_key_hash=None,
        sealed_at=NOW,
    )
    attempt = evidence.AttemptRecordV3(
        attempt_id=OTHER_HASH,
        parent_kind="memory_conformance",
        parent_id="mem0-conformance-occurrence",
        stage="memory_ingest",
        ordinal=1,
        request_fingerprint=THIRD_HASH,
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
        outcome=states.AttemptOutcome.SUCCEEDED,
        retry_of_attempt_id=None,
        idempotency_key_hash=None,
        reconciliation_capability="none",
        raw_response_ref=HASH,
        raw_error_ref=None,
        index_contribution=states.IndexContribution.FINAL,
        superseded_by_attempt_id=None,
    )
    usage = accounting.TokenUsageRecordV4(
        usage_record_id=THIRD_HASH,
        attempt_id=attempt.attempt_id,
        parent_kind="memory_conformance",
        parent_id="mem0-conformance-occurrence",
        stage=accounting.TokenStageV3.MEMORY_CONFORMANCE,
        operation_kind="memory_ingest",
        usage_owner_role_binding_id="mem0-extraction",
        token_domain=accounting.TokenDomain.EXTERNAL_LLM,
        measurement_source=accounting.TokenMeasurementSource.SUPPLIER_RESPONSE,
        input_tokens=None,
        visible_output_tokens=None,
        supplier_reported_total_tokens=None,
        context_view_tokens=None,
        cached_input_tokens=None,
        reasoning_tokens=None,
        configured_model="deepseek-chat",
        runtime_model="deepseek-chat",
        meter_schema_id="openai-compatible-usage-v1",
        raw_field_paths=(),
        covered_dimensions=(),
        unavailable_dimensions=(
            "input_tokens",
            "visible_output_tokens",
            "supplier_reported_total_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
        ),
        not_applicable_dimensions=(),
        inclusion_relationships=(),
        token_measurement_complete=False,
        billing_complete=False,
        proof_status=accounting.ProofStatus.UNAVAILABLE,
        reason="provider_does_not_expose_usage",
        raw_response_ref=HASH,
    )

    assert intent.parent_kind == "memory_conformance"
    assert usage.usage_owner_role_binding_id == "mem0-extraction"
    assert usage.stage == accounting.TokenStageV3.MEMORY_CONFORMANCE
    with pytest.raises(ValidationError):
        evidence.BudgetReservationRecord(**(reservation.model_dump(exclude={"schema_version"})))
    with pytest.raises(ValidationError):
        evidence.AttemptIntentRecord(**intent.model_dump(exclude={"schema_version"}))
    with pytest.raises(ValidationError):
        evidence.AttemptRecordV2(**attempt.model_dump(exclude={"schema_version"}))
    with pytest.raises(ValidationError):
        accounting.TokenUsageRecordV3(**usage.model_dump(exclude={"schema_version"}))


def test_memory_conformance_occurrence_and_approval_are_independent_from_run() -> None:
    evidence = require("oamb.contracts.evidence")
    specifications = require("oamb.contracts.specifications")

    approval_values = dict(
        approval_id="mem0-conformance-approval",
        operation_kind="memory_conformance",
        scope_kind=specifications.BudgetScopeKindV3.MEMORY_CONFORMANCE,
        scope_id="mem0-conformance-occurrence",
        runtime_binding_hash=HASH,
        provider_runtime_profile_attestation_hash=None,
        role_binding_ids=("mem0-extraction", "controlled-embedding"),
        budget_hash=OTHER_HASH,
        approved_at=NOW,
        expires_at=NOW + timedelta(hours=1),
        unmetered_cost_acknowledged=True,
        stop_condition_ids=("unknown_outcome",),
    )
    approval = specifications.ExternalCallApprovalRecordV2(
        approval_hash=specifications.external_call_approval_v2_hash(approval_values),
        **approval_values,
    )
    occurrence = evidence.MemoryConformanceOccurrenceRecord(
        occurrence_id="mem0-conformance-occurrence",
        provider="mem0",
        provider_project_id="oamb-mem0-conformance-1",
        provider_profile_id="mem0-rest-v2.0.19",
        provider_scope_id="oamb-conformance-scope-1",
        runtime_binding_hash=HASH,
        approval_id=approval.approval_id,
        budget_id="mem0-conformance-budget",
        state=evidence.MemoryConformanceOccurrenceState.SEALED,
        operation_claim_ids=(HASH,),
        dispatch_route_ids=("mem0-ingest-route",),
        attempt_ids=(OTHER_HASH,),
        usage_record_ids=(THIRD_HASH,),
        resource_record_ids=(),
        cost_record_ids=(),
        protected_state_before_hash=HASH,
        protected_state_after_hash=HASH,
        started_at=NOW,
        ended_at=NOW + timedelta(seconds=1),
    )

    assert occurrence.provider_scope_id != occurrence.provider_project_id
    with pytest.raises(ValidationError, match="protected state"):
        evidence.MemoryConformanceOccurrenceRecord(
            **(occurrence.model_dump() | {"protected_state_after_hash": OTHER_HASH})
        )


def test_run_preflight_persists_exact_conformance_and_dispatch_routing() -> None:
    schema = require("oamb.contracts.schema")
    specifications = require("oamb.contracts.specifications")

    readiness = _source_binding(
        specifications,
        identity="mem0-model-readiness",
        root_hash=OTHER_HASH,
    )
    conformance = _source_binding(
        specifications,
        identity="mem0-memory-conformance",
        root_hash=HASH,
    )
    route = _dispatch_route(specifications)
    values = dict(
        run_id="t10-mem0-lme6-run",
        observed_at=NOW,
        resolved_plan_hash=HASH,
        run_spec_hash=OTHER_HASH,
        dataset_manifest_hash=THIRD_HASH,
        subset_manifest_hash=HASH,
        adapter_profile_id="mem0-rest-v2.0.19",
        adapter_profile_hash=OTHER_HASH,
        provider_project_id="oamb-t10-mem0-lme6",
        provider_profile_id="mem0-rest-v2.0.19",
        runtime_binding_hash=THIRD_HASH,
        provider_service_evidence=readiness,
        memory_conformance_evidence=conformance,
        role_binding_ids=("mem0-extraction", "controlled-embedding", "answer", "judge"),
        dispatch_routes=(route,),
        approval_hash=HASH,
        budget_hash=OTHER_HASH,
        redacted_endpoint_fingerprints=(HASH,),
        credential_reference_fingerprints=(OTHER_HASH,),
        artifact_repository_fingerprint=THIRD_HASH,
        artifact_durability_proof_hash=HASH,
    )
    record = specifications.RunPreflightRecord(
        preflight_record_hash=specifications.run_preflight_record_hash(values),
        **values,
    )

    parsed = schema.parse_contract(
        json.loads(json.dumps(record.model_dump(mode="json"), sort_keys=True))
    )
    assert parsed == record

    drifted = record.model_dump()
    drifted["subset_manifest_hash"] = OTHER_HASH
    with pytest.raises(ValidationError, match="preflight record hash"):
        specifications.RunPreflightRecord(**drifted)


def test_evaluation_report_spec_uses_a_new_version_without_expanding_v1() -> None:
    specifications = require("oamb.contracts.specifications")
    fields = dict(
        report_kind="evaluation",
        audience="public",
        preview_max_field_bytes=1024,
        preview_total_bytes=4096,
        display_field_ids=("phase", "runs", "comparisons", "cases"),
        renderer_hash=HASH,
        asset_hashes=(OTHER_HASH,),
        browser_contract_hash=THIRD_HASH,
        performance_contract_hash=HASH,
        export_profile_selector_id="public-evaluation-v1",
        export_profile_selector_version=1,
    )
    spec = specifications.ReportSpecV2(
        report_spec_id=specifications.report_spec_v2_id(**fields),
        **fields,
    )

    assert spec.report_kind == "evaluation"
    with pytest.raises(ValidationError):
        specifications.ReportSpec(
            report_spec_id=HASH,
            **fields,
        )


def _evaluation_identity_chain(specifications: ModuleType) -> tuple[Any, ...]:
    ids = require("oamb.contracts.ids")
    spec_fields = dict(
        report_kind="evaluation",
        audience="public",
        preview_max_field_bytes=1024,
        preview_total_bytes=4096,
        display_field_ids=("phase", "runs", "comparisons", "cases"),
        renderer_hash=HASH,
        asset_hashes=(OTHER_HASH,),
        browser_contract_hash=THIRD_HASH,
        performance_contract_hash=HASH,
        export_profile_selector_id="public-evaluation-v1",
        export_profile_selector_version=1,
    )
    spec = specifications.ReportSpecV2(
        report_spec_id=specifications.report_spec_v2_id(**spec_fields),
        **spec_fields,
    )
    binding_fields = dict(
        spec_kind="benchmark_report",
        spec_schema_name=spec.schema_name,
        spec_schema_version=spec.schema_version,
        spec_id=spec.report_spec_id,
        spec_hash=ids.canonical_sha256(spec),
    )
    binding = specifications.ReportIdentitySpecBindingV2(
        binding_id=specifications.report_identity_spec_binding_v2_id(**binding_fields),
        **binding_fields,
    )
    runs = tuple(
        specifications.EvaluationModelClosureEntry(
            model_kind="run",
            report_model_id=_sha(f"run-id-{index}"),
            canonical_bytes_hash=_sha(f"run-bytes-{index}"),
            ordered_source_root_hash=_sha(f"run-roots-{index}"),
            evidence_validation_result_hash=_sha(f"run-validation-{index}"),
            coverage_hash=_sha(f"run-coverage-{index}"),
        )
        for index in range(4)
    )
    comparisons = (
        specifications.EvaluationModelClosureEntry(
            model_kind="comparison",
            report_model_id=_sha("comparison-id"),
            canonical_bytes_hash=_sha("comparison-bytes"),
            ordered_source_root_hash=_sha("comparison-roots"),
            evidence_validation_result_hash=_sha("comparison-validation"),
            coverage_hash=_sha("comparison-coverage"),
        ),
    )
    closure = specifications.EvaluationModelClosure(
        ordered_run_entries=runs,
        eligible_comparison_entries=comparisons,
        model_closure_hash=specifications.evaluation_model_closure_hash(
            ordered_run_entries=runs,
            eligible_comparison_entries=comparisons,
        ),
    )
    source = specifications.SourceEvidenceBinding(
        binding_id=_sha("source-binding"),
        source_kind=specifications.SourceEvidenceKind.DERIVATION,
        source_identity="evaluation-source",
        source_root_hash=_sha("source-root"),
        validation_result_hash=_sha("source-validation"),
        source_schema_versions=("capsule_manifest@1",),
    )
    derivation_fields = dict(
        derivation_kind="evaluation_report",
        ordered_source_bindings=(source,),
        ordered_source_root_hash=ids.canonical_sha256(
            ["oamb-ordered-source-roots-v1", (source.source_root_hash,)]
        ),
        evidence_validation_result_hash=_sha("ordered-validations"),
        transform_spec_hash=_sha("transform"),
        report_identity_spec_binding=binding,
        evaluation_model_closure=closure,
        reducer_and_renderer_input_hashes=(_sha("reducer"), _sha("renderer")),
    )
    derivation = specifications.DerivationSpecV3(
        derivation_input_hash=specifications.derivation_spec_v3_input_hash(**derivation_fields),
        **derivation_fields,
    )
    return spec, binding, closure, source, derivation


def test_evaluation_identity_chain_round_trips_through_exact_registry_versions() -> None:
    specifications = require("oamb.contracts.specifications")
    schema = require("oamb.contracts.schema")
    spec, binding, closure, _, derivation = _evaluation_identity_chain(specifications)

    for value in (spec, binding, derivation):
        parsed = schema.parse_contract(value.model_dump(mode="json"))
        assert parsed == value
    assert tuple(entry.model_kind for entry in closure.ordered_run_entries) == ("run",) * 4

    old_fields = dict(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=1024,
        preview_total_bytes=4096,
        display_field_ids=("summary", "cases"),
        renderer_hash=HASH,
        asset_hashes=(OTHER_HASH,),
        browser_contract_hash=THIRD_HASH,
        performance_contract_hash="d" * 64,
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )
    assert specifications.report_spec_id(**old_fields) == (
        "adae2094a5c03f62d2e2776a767c1bf460efaceb939b19cef970afa33c84bd01"
    )


def test_evaluation_identity_chain_rejects_cross_versions_and_cycle_fields() -> None:
    specifications = require("oamb.contracts.specifications")
    _, binding, closure, source, derivation = _evaluation_identity_chain(specifications)

    with pytest.raises(ValidationError):
        specifications.ReportSpecV2(
            report_spec_id=HASH,
            report_kind="run",
            audience="public",
            preview_max_field_bytes=1,
            preview_total_bytes=1,
            display_field_ids=("summary",),
            renderer_hash=HASH,
            asset_hashes=(),
            browser_contract_hash=OTHER_HASH,
            performance_contract_hash=THIRD_HASH,
            export_profile_selector_id="public-run-v1",
            export_profile_selector_version=1,
        )
    invalid = derivation.model_dump()
    invalid["final_derivation_id"] = HASH
    with pytest.raises(ValidationError):
        specifications.DerivationSpecV3(**invalid)
    invalid = derivation.model_dump()
    invalid["report_identity_spec_binding"] = specifications.ReportIdentitySpecBinding(
        binding_id=specifications.report_identity_spec_binding_id(
            spec_kind="benchmark_report",
            spec_schema_name="report_spec",
            spec_schema_version=1,
            spec_id=HASH,
            spec_hash=OTHER_HASH,
        ),
        spec_kind="benchmark_report",
        spec_schema_name="report_spec",
        spec_schema_version=1,
        spec_id=HASH,
        spec_hash=OTHER_HASH,
    )
    with pytest.raises(ValidationError):
        specifications.DerivationSpecV3(**invalid)

    changed_runs = list(closure.ordered_run_entries)
    changed_runs[0] = changed_runs[0].model_copy(update={"coverage_hash": _sha("changed")})
    changed_hash = specifications.evaluation_model_closure_hash(
        ordered_run_entries=tuple(changed_runs),
        eligible_comparison_entries=closure.eligible_comparison_entries,
    )
    assert changed_hash != closure.model_closure_hash

    derivation_fields = derivation.model_dump(
        exclude={"schema_name", "schema_version", "derivation_input_hash"}
    )
    derivation_fields["evaluation_model_closure"] = closure.model_copy(
        update={"ordered_run_entries": tuple(changed_runs), "model_closure_hash": changed_hash}
    )
    changed_derivation_hash = specifications.derivation_spec_v3_input_hash(**derivation_fields)
    assert changed_derivation_hash != derivation.derivation_input_hash


def test_evaluation_artifact_manifest_v3_repeats_the_exact_model_closure() -> None:
    specifications = require("oamb.contracts.specifications")
    reporting = require("oamb.contracts.reporting")
    schema = require("oamb.contracts.schema")
    _, binding, closure, source, _ = _evaluation_identity_chain(specifications)
    fields = dict(
        report_id=_sha("evaluation-report"),
        report_kind="evaluation",
        report_identity_spec_binding=binding,
        ordered_source_bindings=(source,),
        ordered_evidence_validation_hashes=(_sha("source-validation"),),
        report_model_hash=_sha("evaluation-model"),
        evaluation_model_closure=closure,
        renderer_hash=HASH,
        asset_hashes=(OTHER_HASH,),
        browser_contract_hash=THIRD_HASH,
        performance_contract_hash=HASH,
        export_profile_selector_id="public-evaluation-v1",
        export_profile_selector_version=1,
        export_profile_hash=_sha("export-profile"),
        audience="public",
        schema_versions=("evaluation_report_model@1", "report_artifact_manifest@3"),
        limitations=("supplier billing unavailable",),
    )
    render_input_hash = reporting.report_artifact_manifest_v3_render_input_hash(**fields)
    manifest = reporting.ReportArtifactManifestV3(
        artifact_manifest_id=reporting.report_artifact_manifest_v3_id(
            **fields, render_input_hash=render_input_hash
        ),
        render_input_hash=render_input_hash,
        **fields,
    )

    assert schema.parse_contract(manifest.model_dump(mode="json")) == manifest
    changed = manifest.model_dump(
        exclude={"schema_name", "schema_version", "artifact_manifest_id", "render_input_hash"}
    )
    changed["evaluation_model_closure"] = closure.model_copy(
        update={"model_closure_hash": _sha("different-model-closure")}
    )
    assert reporting.report_artifact_manifest_v3_render_input_hash(**changed) != (
        manifest.render_input_hash
    )
    cyclic = manifest.model_dump()
    cyclic["report_html_hash"] = _sha("html")
    with pytest.raises(ValidationError):
        reporting.ReportArtifactManifestV3(**cyclic)
