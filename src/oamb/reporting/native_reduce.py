"""Pure native run reduction from a freshly validated sealed capsule root."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import TypeVar, cast

from oamb.artifacts.atomic import read_regular_file
from oamb.artifacts.capsule_graph import read_capsule_graph
from oamb.artifacts.validation.native import validate_native_capsule
from oamb.artifacts.validation.reduction import (
    AccountingValidationInput,
    accounting_validation_input,
)
from oamb.contracts.accounting import (
    CostRecord,
    ProofStatus,
    ResourceUsageRecord,
    TokenUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
)
from oamb.contracts.evidence import (
    AttemptRecordV2,
    CapsuleManifest,
    CaseEvaluationDisposition,
    CaseRecordV3,
    HistoryAttemptRecord,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
    ValidationResult,
)
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import (
    CompletionSummaryV3,
    DisplayPreview,
    Mab65ReportReduction,
    MabPlanEvidenceBinding,
    MetricSummary,
    ReportRecordProjection,
    RunReportModelV3,
)
from oamb.contracts.specifications import (
    CaseManifest,
    ReportSpec,
    RunSpec,
    SourceEvidenceBinding,
    SourceEvidenceKind,
)
from oamb.contracts.states import (
    AttemptOutcome,
    CaseState,
    IngestionPlanState,
    ValidationDisposition,
)
from oamb.reporting.accounting import (
    AccountingOwner,
    AccountingRecordView,
    AccountingReduction,
    reduce_accounting_records,
)
from oamb.reporting.previews import build_display_preview
from oamb.reporting.public import (
    build_mab65_report_reduction,
    build_metric_summary,
    build_report_record_projection,
    build_run_report_model,
    build_validation_claim_boundary,
)
from oamb.workloads.mab65_reduction import reduce_mab65
from oamb.workloads.memoryagentbench import MAB65_WORKLOAD_ID, MabManifestBundle

TokenRecord = TokenUsageRecord | TokenUsageRecordV2 | TokenUsageRecordV3
NativePlanRecord = IngestionPlanRecordV2 | IngestionPlanRecordV3
RecordT = TypeVar("RecordT")


class NativeReportReductionError(ValueError):
    """A capsule cannot authorize a normal native report reduction."""


@dataclass(frozen=True, slots=True)
class _NativeReportRecords:
    case_manifest: CaseManifest
    run_spec: RunSpec | None
    plans: tuple[NativePlanRecord, ...]
    cases: tuple[CaseRecordV3, ...]
    attempts: tuple[AttemptRecordV2, ...]
    history_attempts: tuple[HistoryAttemptRecord, ...]
    tokens: tuple[TokenRecord, ...]
    resources: tuple[ResourceUsageRecord, ...]
    costs: tuple[CostRecord, ...]
    raw_paths: dict[str, Path]
    source_schema_versions: tuple[str, ...]


NATIVE_REPORT_REDUCER_CONTRACT = {
    "reducer_id": "native-completion-accounting-v1",
    "version": 1,
    "input_schemas": (
        "capsule_manifest@1",
        "dataset_manifest@1",
        "case_manifest@1",
        "ingestion_plan_record@2|3",
        "case_record@3",
        "attempt_record@2",
        "token_usage_record@1|2|3",
        "resource_usage_record@1",
        "cost_record@1",
    ),
    "output_schema": "run_report_model@3",
    "plan_order": "case_manifest.ingestion_plans",
    "case_order": "case_manifest.cases",
    "attempt_order": "started_at,attempt_id",
    "completion_algorithm": "terminal-state exact counts",
    "metric_algorithm": "exact mean over sealed per-case fractions",
    "accounting_algorithm": "owner scoped attempted/final/not-applicable reduction",
}

MAB65_REPORT_REDUCER_CONTRACT = {
    "reducer_id": "mab65-capability-balanced-index-v1",
    "version": 1,
    "input_schemas": (
        "mab_manifest_bundle",
        "ingestion_plan_record@2|3",
        "case_record@3",
    ),
    "output_schema": "mab65_report_reduction@1",
    "plan_order": "component ar,icl,recsys,lru,cr_sf then frozen manifest order",
    "algorithm": "complete-only plan macro and exact-rational capability weighting",
    "control_boundary": "sealed comparison controls required",
}


def native_report_reducer_implementation_hash() -> str:
    return canonical_sha256(
        [
            NATIVE_REPORT_REDUCER_CONTRACT,
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        ]
    )


def mab65_report_reducer_implementation_hash() -> str:
    reporting_root = Path(__file__).parent
    package_root = reporting_root.parent
    return canonical_sha256(
        [
            MAB65_REPORT_REDUCER_CONTRACT,
            hashlib.sha256(
                (package_root / "workloads" / "mab65_reduction.py").read_bytes()
            ).hexdigest(),
            hashlib.sha256((reporting_root / "public.py").read_bytes()).hexdigest(),
        ]
    )


def reduce_native_run_report(
    capsule_root: Path,
    validation_result: ValidationResult,
    *,
    report_spec: ReportSpec,
    validation_target: object | None = None,
) -> RunReportModelV3:
    capsule_root = Path(capsule_root)
    if validation_result.disposition != ValidationDisposition.VALIDATED:
        raise NativeReportReductionError("native report requires evidence validation PASS")
    if validation_result.validation_profile_id == "oamb-t8-native-run-evidence-v1":
        from oamb.artifacts.validation.run_evidence import (
            NativeRunEvidenceValidationInput,
            validate_native_run_evidence,
        )

        if not isinstance(validation_target, NativeRunEvidenceValidationInput):
            raise NativeReportReductionError(
                "composite native report validation requires its complete validation target"
            )
        if validation_target.capsule_root != capsule_root:
            raise NativeReportReductionError(
                "composite native report validation names a different capsule root"
            )
        fresh = validate_native_run_evidence(validation_target)
    else:
        fresh = validate_native_capsule(capsule_root)
    if fresh != validation_result:
        raise NativeReportReductionError(
            "native report requires fresh validation of the current capsule bytes"
        )
    manifest = CapsuleManifest.model_validate_json(
        read_regular_file(capsule_root / "capsule-manifest.json")
    )
    if validation_result.target_hash != manifest.source_manifest_hash:
        raise NativeReportReductionError("native validation does not bind this capsule root")
    records = _load_report_records(capsule_root, manifest)
    case_manifest = records.case_manifest
    plans = _ordered_plans(records.plans, case_manifest)
    cases = _ordered_cases(records.cases, case_manifest)
    attempts = tuple(sorted(records.attempts, key=lambda item: (item.started_at, item.attempt_id)))
    summary = _completion_summary(manifest.run_id, case_manifest, plans, cases)
    accounting = _accounting_summary(
        records.tokens,
        records.resources,
        records.costs,
        plans,
        records.history_attempts,
    )
    metrics = _metric_summaries(cases)
    mab65_reduction = _build_native_mab65_report_reduction(
        workload_id=case_manifest.workload_id,
        manifest_id=case_manifest.manifest_id,
        validation_target=validation_target,
        plans=plans,
        cases=cases,
    )
    record_projections = _record_projections(
        records,
        plans,
        cases,
        attempts,
        capsule_root=capsule_root,
        report_spec=report_spec,
    )
    source_fields = {
        "source_kind": SourceEvidenceKind.RUN,
        "source_identity": manifest.run_id,
        "source_root_hash": manifest.source_manifest_hash,
        "validation_result_hash": canonical_sha256(validation_result),
        "source_schema_versions": records.source_schema_versions,
    }
    source_binding = SourceEvidenceBinding.model_validate(
        {
            "binding_id": canonical_sha256(["oamb-source-evidence-binding-v1", source_fields]),
            **source_fields,
        }
    )
    memory_system_ids = tuple(dict.fromkeys(item.memory_system_id for item in plans))
    if len(memory_system_ids) != 1:
        raise NativeReportReductionError("native report requires one memory-system identity")
    limitations = (
        ["live provider execution; supplier billing is reported only when evidenced"]
        if records.run_spec is not None
        else ["fixture-only native evidence; no live provider call was made"]
    )
    if not accounting.billing_complete:
        limitations.append("supplier billing completeness is unavailable")
    if not accounting.cost_complete:
        limitations.append("complete cost evidence is unavailable")
    return build_run_report_model(
        report_spec_hash=canonical_sha256(report_spec),
        source_binding=source_binding,
        evidence_validation_profile_hash=canonical_sha256(
            [
                "oamb-validation-profile-binding-v1",
                validation_result.validation_profile_id,
                validation_result.required_rule_ids,
                validation_result.implementation_versions,
            ]
        ),
        evidence_validation_result_hash=canonical_sha256(validation_result),
        reducer_bindings=(
            (
                "native-completion-accounting-v1",
                1,
                native_report_reducer_implementation_hash(),
            ),
            *(
                (
                    (
                        "mab65-capability-balanced-index-v1",
                        1,
                        mab65_report_reducer_implementation_hash(),
                    ),
                )
                if mab65_reduction is not None
                else ()
            ),
        ),
        audience=report_spec.audience,
        origin_kind="native",
        capsule_id=manifest.capsule_id,
        protocol_id=(
            records.run_spec.protocol_id
            if records.run_spec is not None
            else "native-fixture-protocol-v1"
        ),
        workload_id=case_manifest.workload_id,
        memory_system_id=memory_system_ids[0],
        claim_boundary=build_validation_claim_boundary(
            status="pass",
            applicable_rule_count=len(validation_result.required_rule_ids),
            executed_rule_count=len(validation_result.executed_rule_ids),
            passed_rule_count=len(validation_result.passed_rule_ids),
            failed_rule_count=len(validation_result.failed_rule_ids),
            not_applicable_rule_count=len(validation_result.not_applicable_rule_ids),
            missing_rule_count=len(validation_result.missing_rule_ids),
            comparison_eligible=True,
            billing_complete=accounting.billing_complete,
            cost_complete=accounting.cost_complete,
        ),
        summary=summary,
        metric_summaries=metrics,
        mab65_reduction=mab65_reduction,
        measurement_lines=accounting.lines,
        logical_context_ids=tuple(
            item.context_manifest_entry_id for item in case_manifest.logical_contexts
        ),
        ingestion_occurrence_ids=tuple(item.ingestion_occurrence_id for item in plans),
        case_occurrence_ids=tuple(item.case_occurrence_id for item in cases),
        attempt_ids=tuple(item.attempt_id for item in attempts),
        record_projections=record_projections,
        limitations=tuple(limitations),
    )


def _build_native_mab65_report_reduction(
    *,
    workload_id: str = MAB65_WORKLOAD_ID,
    manifest_id: str = MAB65_WORKLOAD_ID,
    validation_target: object | None,
    plans: tuple[NativePlanRecord, ...],
    cases: tuple[CaseRecordV3, ...],
) -> Mab65ReportReduction | None:
    if workload_id != MAB65_WORKLOAD_ID or manifest_id != MAB65_WORKLOAD_ID:
        return None

    from oamb.artifacts.validation.run_evidence import (
        NativeRunEvidenceValidationInput,
    )

    if not isinstance(validation_target, NativeRunEvidenceValidationInput):
        raise NativeReportReductionError(
            "MAB-65 report reduction requires the composite validation target"
        )
    if validation_target.workload_profile_id != "oamb-t8-workload-mab65-v1":
        raise NativeReportReductionError(
            "MAB-65 report reduction requires the exact workload profile"
        )
    bundle = cast(MabManifestBundle, validation_target.workload_target)
    if bundle.case_manifest.workload_id != MAB65_WORKLOAD_ID:
        raise NativeReportReductionError("MAB-65 validation target names another workload")

    plan_records_by_id = _unique_records_by(
        plans,
        key=lambda item: item.ingestion_plan_id,
        label="MAB-65 physical ingestion plan",
    )
    case_records_by_id = _unique_records_by(
        cases,
        key=lambda item: item.case_manifest_entry_id,
        label="MAB-65 case",
    )
    expected_plan_ids = tuple(plan.manifest.ingestion_plan_id for plan in bundle.plans)
    expected_case_ids = tuple(case.entry.case_manifest_entry_id for case in bundle.cases)
    if set(plan_records_by_id) != set(expected_plan_ids) or set(case_records_by_id) != set(
        expected_case_ids
    ):
        raise NativeReportReductionError(
            "MAB-65 sealed record inventory does not close the validated workload"
        )

    case_metrics = {
        case_id: Fraction(record.metric_numerator, record.metric_denominator)
        for case_id, record in case_records_by_id.items()
        if (
            record.state == CaseState.COMPLETED
            and record.evaluation_disposition == CaseEvaluationDisposition.DETERMINISTIC_EVALUATED
            and record.metric_numerator is not None
            and record.metric_denominator is not None
        )
    }
    ready_plan_manifest_ids = frozenset(
        plan.manifest.plan_manifest_entry_id
        for plan in bundle.plans
        if plan_records_by_id[plan.manifest.ingestion_plan_id].state == IngestionPlanState.SEALED
    )
    query_state_unchanged = all(
        record.query_mutation_status == "unchanged"
        and record.pre_query_state_sha256 is not None
        and record.pre_query_state_sha256 == record.post_query_state_sha256
        for record in case_records_by_id.values()
    )
    reduction = reduce_mab65(
        bundle,
        case_metrics=case_metrics,
        ready_plan_manifest_ids=ready_plan_manifest_ids,
        capsule_valid=True,
        query_state_unchanged=query_state_unchanged,
        expected_catalog_sha256=bundle.entity_catalog_sha256,
        expected_unicode_fingerprint=bundle.unicode_fingerprint,
        expected_interaction_fingerprint=canonical_sha256(
            ["oamb-mab65-sealed-interaction-controls-required-v1"]
        ),
        observed_interaction_fingerprint=canonical_sha256(
            ["oamb-mab65-sealed-interaction-controls-unavailable-v1"]
        ),
        comparison_controls_closed=False,
    )
    if not reduction.available:
        return build_mab65_report_reduction(reduction)

    plan_evidence_bindings: list[MabPlanEvidenceBinding] = []
    for plan in bundle.plans:
        plan_record = plan_records_by_id[plan.manifest.ingestion_plan_id]
        expected_case_occurrence_ids = tuple(
            case_records_by_id[case_id].case_occurrence_id
            for case_id in plan.manifest.ordered_case_manifest_entry_ids
        )
        if (
            plan_record.ordered_member_context_manifest_entry_ids
            != plan.manifest.ordered_member_context_manifest_entry_ids
            or plan_record.ordered_case_occurrence_ids != expected_case_occurrence_ids
        ):
            raise NativeReportReductionError(
                "MAB-65 plan evidence does not close its sealed membership"
            )
        plan_cases = tuple(
            case_records_by_id[case_id] for case_id in plan.manifest.ordered_case_manifest_entry_ids
        )
        metric_ids = tuple(dict.fromkeys(case.metric_id for case in plan_cases))
        if len(metric_ids) != 1 or metric_ids[0] is None:
            raise NativeReportReductionError("MAB-65 plan evidence does not name one exact metric")
        plan_evidence_bindings.append(
            MabPlanEvidenceBinding(
                plan_manifest_entry_id=plan.manifest.plan_manifest_entry_id,
                ingestion_plan_id=plan.manifest.ingestion_plan_id,
                ingestion_occurrence_id=plan_record.ingestion_occurrence_id,
                ordered_logical_context_ids=(plan_record.ordered_member_context_manifest_entry_ids),
                member_labels=plan.member_labels,
                ordered_case_occurrence_ids=plan_record.ordered_case_occurrence_ids,
                metric_id=metric_ids[0],
                resolution_evidence_references=(
                    tuple(cast(str, case.evaluation_raw_ref) for case in plan_cases)
                    if plan.component == "recsys"
                    else ()
                ),
            )
        )
    return build_mab65_report_reduction(
        reduction,
        plan_evidence_bindings=tuple(plan_evidence_bindings),
    )


def _unique_records_by(
    values: tuple[RecordT, ...],
    *,
    key: Callable[[RecordT], str],
    label: str,
) -> dict[str, RecordT]:
    keyed = {key(item): item for item in values}
    if len(keyed) != len(values):
        raise NativeReportReductionError(f"duplicate {label} identity")
    return keyed


def _load_report_records(
    root: Path,
    manifest: CapsuleManifest,
) -> _NativeReportRecords:
    case_manifests: list[CaseManifest] = []
    run_specs: list[RunSpec] = []
    plans: list[NativePlanRecord] = []
    cases: list[CaseRecordV3] = []
    attempts: list[AttemptRecordV2] = []
    history_attempts: list[HistoryAttemptRecord] = []
    tokens: list[TokenRecord] = []
    resources: list[ResourceUsageRecord] = []
    costs: list[CostRecord] = []
    raw_paths: dict[str, Path] = {}
    source_schema_versions = ["capsule_manifest@1"]
    graph = read_capsule_graph(root)
    for graph_index, (capsule_root, capsule_manifest) in enumerate(graph):
        for entry in capsule_manifest.source_entries:
            path = capsule_root / entry.relative_path
            content = read_regular_file(path)
            if entry.record_kind == "embedded_part_file":
                continue
            if entry.record_kind == "raw_payload":
                previous = raw_paths.setdefault(entry.record_id, path)
                if read_regular_file(previous) != content:
                    raise NativeReportReductionError(
                        "raw payload identity collides across capsules"
                    )
                continue
            try:
                document = json.loads(content)
                schema_name = document["schema_name"]
                schema_version = document["schema_version"]
                if not isinstance(schema_name, str) or not isinstance(schema_version, int):
                    raise TypeError
            except (KeyError, TypeError, ValueError) as exc:
                raise NativeReportReductionError(
                    "native report source record has no exact schema identity"
                ) from exc
            source_schema_versions.append(f"{schema_name}@{schema_version}")
            if graph_index == 0 and entry.record_kind == "case_manifest":
                case_manifests.append(CaseManifest.model_validate_json(content))
            elif graph_index == 0 and entry.record_kind == "run_spec":
                run_specs.append(RunSpec.model_validate_json(content))
            elif graph_index == 0 and entry.record_kind == "ingestion_plan_record":
                if document["schema_version"] == 2:
                    plans.append(IngestionPlanRecordV2.model_validate_json(content))
                elif document["schema_version"] == 3:
                    plans.append(IngestionPlanRecordV3.model_validate_json(content))
                else:
                    raise NativeReportReductionError(
                        "unsupported native ingestion-plan record version"
                    )
            elif graph_index == 0 and entry.record_kind == "case_record":
                cases.append(CaseRecordV3.model_validate_json(content))
            elif entry.record_kind == "attempt_record":
                attempts.append(AttemptRecordV2.model_validate_json(content))
            elif entry.record_kind == "history_attempt_record":
                history_attempts.append(HistoryAttemptRecord.model_validate_json(content))
            elif entry.record_kind == "token_usage_record":
                tokens.append(_parse_token_usage(content))
            elif entry.record_kind == "resource_usage_record":
                resources.append(ResourceUsageRecord.model_validate_json(content))
            elif entry.record_kind == "cost_record":
                costs.append(CostRecord.model_validate_json(content))
    if len(case_manifests) != 1 or len(run_specs) > 1 or not plans or not cases:
        raise NativeReportReductionError("native capsule report inventory is incomplete")
    return _NativeReportRecords(
        case_manifest=case_manifests[0],
        run_spec=run_specs[0] if run_specs else None,
        plans=tuple(plans),
        cases=tuple(cases),
        attempts=tuple(attempts),
        history_attempts=tuple(history_attempts),
        tokens=tuple(tokens),
        resources=tuple(resources),
        costs=tuple(costs),
        raw_paths=raw_paths,
        source_schema_versions=tuple(dict.fromkeys(source_schema_versions)),
    )


def _ordered_plans(
    values: tuple[NativePlanRecord, ...],
    manifest: CaseManifest,
) -> tuple[NativePlanRecord, ...]:
    by_plan = {item.ingestion_plan_id: item for item in values}
    try:
        return tuple(by_plan[item.ingestion_plan_id] for item in manifest.ingestion_plans)
    except KeyError as exc:
        raise NativeReportReductionError("native plan records do not close the manifest") from exc


def _ordered_cases(
    values: tuple[CaseRecordV3, ...],
    manifest: CaseManifest,
) -> tuple[CaseRecordV3, ...]:
    by_case = {item.case_manifest_entry_id: item for item in values}
    try:
        return tuple(by_case[item.case_manifest_entry_id] for item in manifest.cases)
    except KeyError as exc:
        raise NativeReportReductionError("native case records do not close the manifest") from exc


def _completion_summary(
    run_id: str,
    manifest: CaseManifest,
    plans: tuple[NativePlanRecord, ...],
    cases: tuple[CaseRecordV3, ...],
) -> CompletionSummaryV3:
    terminal_states = {
        CaseState.COMPLETED,
        CaseState.ERROR,
        CaseState.UNSUPPORTED,
        CaseState.CANCELLED,
        CaseState.BUDGET_EXCEEDED,
    }
    return CompletionSummaryV3(
        run_id=run_id,
        intended_logical_contexts=len(manifest.logical_contexts),
        intended_ingestion_plans=len(manifest.ingestion_plans),
        ready_ingestion_plans=sum(item.state == IngestionPlanState.SEALED for item in plans),
        intended_cases=len(manifest.cases),
        terminal_cases=sum(item.state in terminal_states for item in cases),
        completed_cases=sum(item.state == CaseState.COMPLETED for item in cases),
        errored_cases=sum(item.state == CaseState.ERROR for item in cases),
        unsupported_cases=sum(item.state == CaseState.UNSUPPORTED for item in cases),
        cancelled_cases=sum(item.state == CaseState.CANCELLED for item in cases),
        budget_exceeded_cases=sum(item.state == CaseState.BUDGET_EXCEEDED for item in cases),
        parsed_cases=sum(item.parsed_answer_sha256 is not None for item in cases),
        evaluated_cases=sum(
            item.evaluation_disposition
            in {
                CaseEvaluationDisposition.DETERMINISTIC_EVALUATED,
                CaseEvaluationDisposition.JUDGED,
            }
            for item in cases
        ),
        judged_cases=sum(
            item.evaluation_disposition == CaseEvaluationDisposition.JUDGED for item in cases
        ),
        unjudged_cases=sum(
            item.evaluation_disposition == CaseEvaluationDisposition.UNJUDGED for item in cases
        ),
        metric_eligible_cases=sum(
            item.metric_numerator is not None and item.metric_denominator is not None
            for item in cases
        ),
    )


def _accounting_summary(
    tokens: tuple[TokenRecord, ...],
    resources: tuple[ResourceUsageRecord, ...],
    costs: tuple[CostRecord, ...],
    plans: tuple[NativePlanRecord, ...],
    history_attempts: tuple[HistoryAttemptRecord, ...],
) -> AccountingReduction:
    final_plan_ids = frozenset(item.ingestion_occurrence_id for item in plans)
    views = _accounting_record_views(tokens, resources, costs, final_plan_ids=final_plan_ids)
    return reduce_accounting_records(
        token_records=tokens,
        resource_records=resources,
        cost_records=costs,
        record_views=views,
        expected_plan_ids=frozenset(
            item.ingestion_occurrence_id for item in (*plans, *history_attempts)
        ),
    )


def build_native_accounting_validation_input(
    capsule_root: Path,
) -> AccountingValidationInput:
    capsule_root = Path(capsule_root)
    manifest = CapsuleManifest.model_validate_json(
        read_regular_file(capsule_root / "capsule-manifest.json")
    )
    records = _load_report_records(capsule_root, manifest)
    plans = _ordered_plans(records.plans, records.case_manifest)
    return accounting_validation_input(
        token_records=records.tokens,
        resource_records=records.resources,
        cost_records=records.costs,
        record_views=_accounting_record_views(
            records.tokens,
            records.resources,
            records.costs,
            final_plan_ids=frozenset(item.ingestion_occurrence_id for item in plans),
        ),
        expected_plan_ids=frozenset(
            item.ingestion_occurrence_id for item in (*plans, *records.history_attempts)
        ),
        expected_attempt_ids=frozenset(item.attempt_id for item in records.attempts),
        attempts=records.attempts,
        require_attempt_accounting_closure=True,
    )


def _accounting_record_views(
    tokens: tuple[TokenRecord, ...],
    resources: tuple[ResourceUsageRecord, ...],
    costs: tuple[CostRecord, ...],
    *,
    final_plan_ids: frozenset[str],
) -> tuple[AccountingRecordView, ...]:
    views = [
        AccountingRecordView(
            record_id=item.usage_record_id,
            owner_kind=_accounting_owner(item.parent_kind),
            indexing_view=(
                ("final_contribution" if item.parent_id in final_plan_ids else "attempted")
                if item.parent_kind == "ingestion_plan"
                else "not_applicable"
            ),
        )
        for item in tokens
    ]
    views.extend(
        AccountingRecordView(
            record_id=item.resource_record_id,
            owner_kind=_accounting_owner(item.parent_kind),
            indexing_view=(
                ("final_contribution" if item.parent_id in final_plan_ids else "attempted")
                if item.parent_kind == "ingestion_plan"
                else "not_applicable"
            ),
        )
        for item in resources
    )
    views.extend(
        AccountingRecordView(
            record_id=item.cost_record_id,
            owner_kind=_accounting_owner(item.parent_kind),
            indexing_view=item.indexing_view.value,
        )
        for item in costs
    )
    return tuple(views)


def _parse_token_usage(content: bytes) -> TokenRecord:
    try:
        schema_version = int(json.loads(content)["schema_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise NativeReportReductionError("native token usage has no schema version") from exc
    if schema_version == 1:
        return TokenUsageRecord.model_validate_json(content)
    if schema_version == 2:
        return TokenUsageRecordV2.model_validate_json(content)
    if schema_version == 3:
        return TokenUsageRecordV3.model_validate_json(content)
    raise NativeReportReductionError(
        f"unsupported native token usage schema version: {schema_version}"
    )


def _accounting_owner(value: str) -> AccountingOwner:
    if value not in {
        "ingestion_plan",
        "case",
        "model_readiness",
        "run",
    }:
        raise NativeReportReductionError(f"unsupported native accounting owner: {value}")
    return cast(AccountingOwner, value)


def _record_projections(
    records: _NativeReportRecords,
    plans: tuple[NativePlanRecord, ...],
    cases: tuple[CaseRecordV3, ...],
    attempts: tuple[AttemptRecordV2, ...],
    *,
    capsule_root: Path,
    report_spec: ReportSpec,
) -> tuple[ReportRecordProjection, ...]:
    manifest = records.case_manifest
    projections: list[ReportRecordProjection] = []
    preview_bytes_remaining = report_spec.preview_total_bytes
    previewed_raw_references: set[str] = set()

    def display_previews(
        raw_references: tuple[tuple[str, str], ...],
    ) -> tuple[DisplayPreview, ...]:
        nonlocal preview_bytes_remaining
        if report_spec.audience != "local":
            return ()
        previews = []
        for raw_reference, media_type in raw_references:
            if raw_reference in previewed_raw_references or preview_bytes_remaining < 1:
                continue
            path = records.raw_paths.get(raw_reference)
            if path is None:
                raise NativeReportReductionError(
                    "native display preview cannot locate sealed raw evidence"
                )
            relative_path = path.relative_to(capsule_root.resolve()).as_posix()
            try:
                preview = build_display_preview(
                    path,
                    source_reference=relative_path,
                    media_type=media_type,
                    max_bytes=min(
                        report_spec.preview_max_field_bytes,
                        preview_bytes_remaining,
                    ),
                    compression="gzip",
                )
            except (OSError, ValueError) as exc:
                raise NativeReportReductionError(
                    "native display preview cannot reopen sealed raw evidence"
                ) from exc
            if preview.sha256 != raw_reference:
                raise NativeReportReductionError(
                    "native display preview identity disagrees with its raw reference"
                )
            previewed_raw_references.add(raw_reference)
            preview_bytes_remaining -= preview.shown_bytes
            previews.append(preview)
        return tuple(previews)

    for index, context in enumerate(manifest.logical_contexts, 1):
        member_plans = tuple(
            plan.ingestion_occurrence_id
            for plan in plans
            if context.context_manifest_entry_id in plan.ordered_member_context_manifest_entry_ids
        )
        projections.append(
            build_report_record_projection(
                record_id=context.context_manifest_entry_id,
                axis="logical-context",
                label=f"Logical context {index}",
                status="included",
                failure_stage=None,
                evaluation_status=None,
                verdict=None,
                capabilities_or_types=(),
                metric_ids=(),
                proof_statuses=(),
                raw_evidence_present=False,
                latency_microseconds=None,
                context_view_tokens=None,
                declared_usage=None,
                detail_items=(
                    ("source_row", str(context.source_row_number_1_indexed)),
                    ("context_sha256", context.context_bytes_sha256),
                    ("ingestion_occurrence_ids", ",".join(member_plans)),
                ),
            )
        )

    for index, plan in enumerate(plans, 1):
        proofs, context_tokens, declared_usage = _projection_accounting(
            (*plan.usage_record_ids, *plan.resource_record_ids, *plan.cost_record_ids),
            records,
        )
        projections.append(
            build_report_record_projection(
                record_id=plan.ingestion_occurrence_id,
                axis="plan",
                label=f"Ingestion plan {index}",
                status=plan.state.value,
                failure_stage=None,
                evaluation_status=None,
                verdict=None,
                capabilities_or_types=(),
                metric_ids=(),
                proof_statuses=proofs,
                raw_evidence_present=bool(
                    plan.scope_raw_refs or plan.readiness_evidence_refs or plan.projection_raw_refs
                ),
                display_previews=display_previews(
                    tuple(
                        (raw_reference, "application/json")
                        for raw_reference in (
                            *plan.scope_raw_refs,
                            *plan.readiness_evidence_refs,
                            *plan.projection_raw_refs,
                        )
                    )
                ),
                latency_microseconds=None,
                context_view_tokens=context_tokens,
                declared_usage=declared_usage,
                detail_items=(
                    ("ingestion_plan_id", plan.ingestion_plan_id),
                    (
                        "logical_context_ids",
                        ",".join(plan.ordered_member_context_manifest_entry_ids),
                    ),
                    ("case_occurrence_ids", ",".join(plan.ordered_case_occurrence_ids)),
                    ("source_unit_count", str(len(plan.ordered_source_unit_ids))),
                ),
            )
        )

    manifest_cases = {item.case_manifest_entry_id: item for item in manifest.cases}
    for index, case in enumerate(cases, 1):
        proofs, context_tokens, declared_usage = _projection_accounting(
            (*case.usage_record_ids, *case.resource_record_ids, *case.cost_record_ids),
            records,
        )
        manifest_case = manifest_cases[case.case_manifest_entry_id]
        verdict = None
        if case.metric_numerator is not None and case.metric_denominator is not None:
            verdict = "pass" if case.metric_numerator == case.metric_denominator else "fail"
        projections.append(
            build_report_record_projection(
                record_id=case.case_occurrence_id,
                axis="case",
                label=f"Case {index}: {manifest_case.raw_question_id}",
                status=case.state.value,
                failure_stage=case.error_stage,
                evaluation_status=case.evaluation_disposition.value,
                verdict=verdict,
                capabilities_or_types=(),
                metric_ids=((case.metric_id,) if case.metric_id is not None else ()),
                proof_statuses=proofs,
                raw_evidence_present=any(
                    (
                        case.retrieval_raw_ref,
                        case.retrieval_supporting_raw_refs,
                        case.visible_evidence_raw_ref,
                        case.visible_decision_ledger_raw_ref,
                        case.pre_query_projection_raw_refs,
                        case.post_query_projection_raw_refs,
                        case.prompt_raw_ref,
                        case.judge_prompt_raw_ref,
                        case.answer_raw_ref,
                        case.evaluation_raw_ref,
                    )
                ),
                display_previews=display_previews(
                    tuple(
                        (raw_reference, media_type)
                        for raw_reference, media_type in (
                            *(
                                (ref, "application/json")
                                for ref in case.retrieval_supporting_raw_refs
                            ),
                            *(
                                (ref, "application/json")
                                for ref in case.pre_query_projection_raw_refs
                            ),
                            *(
                                (ref, "application/json")
                                for ref in case.post_query_projection_raw_refs
                            ),
                            (case.retrieval_raw_ref, "application/json"),
                            (
                                case.visible_evidence_raw_ref,
                                "application/vnd.oamb.visible-evidence",
                            ),
                            (case.visible_decision_ledger_raw_ref, "application/json"),
                            (case.prompt_raw_ref, "text/plain; charset=utf-8"),
                            (case.judge_prompt_raw_ref, "text/plain; charset=utf-8"),
                            (case.answer_raw_ref, "application/json"),
                            (case.evaluation_raw_ref, "application/json"),
                        )
                        if raw_reference is not None
                    )
                ),
                latency_microseconds=None,
                context_view_tokens=context_tokens,
                declared_usage=declared_usage,
                detail_items=(
                    ("case_manifest_entry_id", case.case_manifest_entry_id),
                    ("ingestion_occurrence_id", case.ingestion_occurrence_id),
                    ("attempt_ids", ",".join(case.attempt_ids)),
                    ("parsed_answer_sha256", case.parsed_answer_sha256 or "unavailable"),
                ),
            )
        )

    usage_by_attempt: dict[str, list[TokenRecord]] = defaultdict(list)
    for usage in records.tokens:
        usage_by_attempt[usage.attempt_id].append(usage)
    for index, attempt in enumerate(attempts, 1):
        attempt_usage = tuple(usage_by_attempt[attempt.attempt_id])
        proofs = tuple(dict.fromkeys(item.proof_status for item in attempt_usage))
        context_tokens = sum(
            (item.context_view_tokens or 0 for item in attempt_usage),
            0,
        )
        declared_usage = sum(
            (
                (item.input_tokens or 0)
                + (item.visible_output_tokens or 0)
                + (item.context_view_tokens or 0)
                for item in attempt_usage
            ),
            0,
        )
        latency = int((attempt.ended_at - attempt.started_at).total_seconds() * 1_000_000)
        projections.append(
            build_report_record_projection(
                record_id=attempt.attempt_id,
                axis="attempt",
                label=f"Attempt {index}: {attempt.stage}",
                status=attempt.outcome.value,
                failure_stage=(
                    attempt.stage if attempt.outcome != AttemptOutcome.SUCCEEDED else None
                ),
                evaluation_status=None,
                verdict=None,
                capabilities_or_types=(),
                metric_ids=(),
                proof_statuses=proofs,
                raw_evidence_present=bool(attempt.raw_response_ref or attempt.raw_error_ref),
                display_previews=display_previews(
                    tuple(
                        (raw_reference, "application/json")
                        for raw_reference in (
                            attempt.raw_response_ref,
                            attempt.raw_error_ref,
                        )
                        if raw_reference is not None
                    )
                ),
                latency_microseconds=latency,
                context_view_tokens=context_tokens or None,
                declared_usage=declared_usage or None,
                detail_items=(
                    ("parent_kind", attempt.parent_kind),
                    ("parent_id", attempt.parent_id),
                    ("stage", attempt.stage),
                    ("request_fingerprint", attempt.request_fingerprint),
                ),
            )
        )
    return tuple(projections)


def _projection_accounting(
    record_ids: tuple[str, ...],
    records: _NativeReportRecords,
) -> tuple[tuple[ProofStatus, ...], int | None, int | None]:
    token_by_id = {item.usage_record_id: item for item in records.tokens}
    resource_by_id = {item.resource_record_id: item for item in records.resources}
    cost_by_id = {item.cost_record_id: item for item in records.costs}
    selected = tuple(
        token_by_id.get(record_id) or resource_by_id.get(record_id) or cost_by_id.get(record_id)
        for record_id in record_ids
    )
    proofs = tuple(dict.fromkeys(item.proof_status for item in selected if item is not None))
    tokens = tuple(
        item
        for item in selected
        if isinstance(item, (TokenUsageRecord, TokenUsageRecordV2, TokenUsageRecordV3))
    )
    context_tokens = sum((item.context_view_tokens or 0 for item in tokens), 0)
    declared_usage = sum(
        (
            (item.input_tokens or 0)
            + (item.visible_output_tokens or 0)
            + (item.context_view_tokens or 0)
            for item in tokens
        ),
        0,
    )
    return proofs, context_tokens or None, declared_usage or None


def _metric_summaries(cases: tuple[CaseRecordV3, ...]) -> tuple[MetricSummary, ...]:
    grouped: dict[str, list[CaseRecordV3]] = defaultdict(list)
    for case in cases:
        if case.metric_id is not None:
            grouped[case.metric_id].append(case)
    output: list[MetricSummary] = []
    for metric_id, members in sorted(grouped.items()):
        fractions = tuple(
            Fraction(item.metric_numerator, item.metric_denominator)
            for item in members
            if item.metric_numerator is not None and item.metric_denominator is not None
        )
        if len(fractions) != len(members):
            raise NativeReportReductionError("native metric inventory contains a missing fraction")
        aggregate = sum(fractions, Fraction()) / len(fractions)
        output.append(
            build_metric_summary(
                metric_id=metric_id,
                metric_version=1,
                stratum_id="all",
                score_numerator=aggregate.numerator,
                score_denominator=aggregate.denominator,
                input_count=len(members),
                case_occurrence_ids=tuple(item.case_occurrence_id for item in members),
                claim_note="exact mean over sealed per-case fractions",
            )
        )
    return tuple(output)


__all__ = [
    "NativeReportReductionError",
    "build_native_accounting_validation_input",
    "reduce_native_run_report",
]
