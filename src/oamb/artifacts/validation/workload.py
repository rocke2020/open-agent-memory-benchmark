"""Production semantic rules for the frozen T8 workload profiles."""

from __future__ import annotations

import hashlib
from typing import Any

from oamb.artifacts.validation.registry import ValidationRule
from oamb.contracts.evidence import ValidationIssue, ValidationSeverity
from oamb.contracts.ids import (
    canonical_sha256,
    context_content_id,
    context_manifest_entry_id,
    ingestion_payload_hash,
    ingestion_plan_id,
)
from oamb.contracts.ports import CasePlan, IngestionPlan
from oamb.contracts.specifications import (
    CaseManifest,
    CaseManifestEntry,
    DatasetFile,
    DatasetManifest,
    IngestionPlanManifest,
)
from oamb.workloads.longmemeval import (
    LME6_CASE_MANIFEST_HASH,
    LME6_EXPECTED_QUESTION_IDS,
    LME6_EXPECTED_SESSION_COUNT,
    LME6_MANIFEST_ID,
    LME30_CASE_MANIFEST_HASH,
    LME30_EXPECTED_QUESTION_IDS,
    LME30_EXPECTED_SESSION_COUNT,
    LME30_WORKLOAD_ID,
    LME60_CASE_MANIFEST_HASH,
    LME60_EXPECTED_QUESTION_IDS,
    LME60_EXPECTED_SESSION_COUNT,
    LME60_MANIFEST_ID,
    LME60_WORKLOAD_ID,
    LME_ANSWER_OUTPUT_CONTRACT_ID,
    LME_ANSWER_PROMPT_PACK_ID,
    LME_DATASET_ID,
    LME_DATASET_MANIFEST_HASH,
    LME_DATASET_REVISION,
    LME_DATASET_SPLIT,
    LME_JUDGE_METRIC_ID,
    LME_JUDGE_PROMPT_PACK_ID,
    LME_PAYLOAD_POLICY,
    LME_SOURCE_BYTE_COUNT,
    LME_SOURCE_LICENSE_ID,
    LME_SOURCE_NAME,
    LME_SOURCE_SHA256,
    QUESTION_TYPES,
    LongMemEvalBundle,
    render_lme_session_document,
)


def _lme_source_manifest_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme.source-manifest.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    dataset = target.dataset_manifest
    expected_dataset = DatasetManifest(
        dataset_id=LME_DATASET_ID,
        revision=LME_DATASET_REVISION,
        split=LME_DATASET_SPLIT,
        manifest_hash=LME_DATASET_MANIFEST_HASH,
        source_files=(
            DatasetFile(
                relative_path=LME_SOURCE_NAME,
                sha256=LME_SOURCE_SHA256,
                byte_count=LME_SOURCE_BYTE_COUNT,
                license_id=LME_SOURCE_LICENSE_ID,
            ),
        ),
        payload_policy=LME_PAYLOAD_POLICY,
    )
    exact = bool(
        _canonical_dataset_manifest(dataset) == expected_dataset
        and _canonical_case_manifest(target.case_manifest) is not None
        and target.case_manifest.workload_id in {LME30_WORKLOAD_ID, LME60_WORKLOAD_ID}
        and _lme_runtime_sources_close(target)
    )
    return () if exact else (_issue(rule_id, dataset.manifest_hash, "lme-source-drift"),)


def _lme_prompt_answer_parser_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme.prompt-answer-parser.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    cases = target.case_manifest.cases
    exact = bool(
        len(cases) == len(target.case_plans) == len(target.selected_rows)
        and all(
            _case_plan_closes(entry, plan)
            and plan.reference_payload_sha256 in entry.answer_value_sha256
            and plan.prompt_binding_id == LME_ANSWER_PROMPT_PACK_ID
            and plan.output_contract_id == LME_ANSWER_OUTPUT_CONTRACT_ID
            and plan.query_timestamp == row.canonical_question_timestamp
            for entry, plan, row in zip(
                cases,
                target.case_plans,
                target.selected_rows,
                strict=True,
            )
        )
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "lme-prompt-parser-drift"),)
    )


def _lme_judge_metric_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme.judge-metric.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    exact = bool(
        target.case_plans
        and all(
            plan.metric_id == LME_JUDGE_METRIC_ID
            and plan.judge_binding_id == LME_JUDGE_PROMPT_PACK_ID
            for plan in target.case_plans
        )
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "lme-judge-metric-drift"),)
    )


def _lme6_selector_membership_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme6.selector-membership.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    exact = bool(
        target.case_manifest.manifest_id == LME6_MANIFEST_ID
        and target.case_manifest.manifest_hash == LME6_CASE_MANIFEST_HASH
        and tuple(row.question_id for row in target.selected_rows) == LME6_EXPECTED_QUESTION_IDS
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "lme6-selector-drift"),)
    )


def _lme6_denominator_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme6.denominator.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    exact = bool(
        len(target.case_manifest.logical_contexts) == 6
        and len(target.case_manifest.ingestion_plans) == 6
        and len(target.case_manifest.cases) == 6
        and sum(plan.intended_source_count for plan in target.ingestion_plans)
        == LME6_EXPECTED_SESSION_COUNT
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "lme6-denominator-drift"),)
    )


def _lme30_selector_coverage_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme30.selector-coverage.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    exact = bool(
        target.case_manifest.manifest_id == f"{LME30_WORKLOAD_ID}-manifest-v1"
        and target.case_manifest.manifest_hash == LME30_CASE_MANIFEST_HASH
        and tuple(row.question_id for row in target.selected_rows) == LME30_EXPECTED_QUESTION_IDS
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "lme30-selector-drift"),)
    )


def _lme30_denominator_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme30.denominator.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    exact = bool(
        len(target.case_manifest.logical_contexts) == 30
        and len(target.case_manifest.ingestion_plans) == 30
        and len(target.case_manifest.cases) == 30
        and sum(plan.intended_source_count for plan in target.ingestion_plans)
        == LME30_EXPECTED_SESSION_COUNT
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "lme30-denominator-drift"),)
    )


def _lme60_selector_membership_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme60.selector-membership.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    exact = bool(
        target.case_manifest.manifest_id == LME60_MANIFEST_ID
        and target.case_manifest.workload_id == LME60_WORKLOAD_ID
        and target.case_manifest.manifest_hash == LME60_CASE_MANIFEST_HASH
        and tuple(row.question_id for row in target.selected_rows) == LME60_EXPECTED_QUESTION_IDS
        and all(
            all(
                row.question_type == question_type
                for row in target.selected_rows[index * 10 : (index + 1) * 10]
            )
            for index, question_type in enumerate(QUESTION_TYPES)
        )
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "lme60-selector-drift"),)
    )


def _lme60_denominator_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.lme60.denominator.v1"
    if not isinstance(target, LongMemEvalBundle):
        return _wrong_target(rule_id)
    exact = bool(
        len(target.case_manifest.logical_contexts) == 60
        and len(target.case_manifest.ingestion_plans) == 60
        and len(target.case_manifest.cases) == 60
        and sum(plan.intended_source_count for plan in target.ingestion_plans)
        == LME60_EXPECTED_SESSION_COUNT
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "lme60-denominator-drift"),)
    )


def _canonical_dataset_manifest(dataset: object) -> DatasetManifest | None:
    try:
        if not isinstance(dataset, DatasetManifest):
            return None
        return DatasetManifest.model_validate(dataset.model_dump(mode="python"))
    except Exception:
        return None


def _canonical_case_manifest(case_manifest: object) -> CaseManifest | None:
    try:
        if not isinstance(case_manifest, CaseManifest):
            return None
        return CaseManifest.model_validate(case_manifest.model_dump(mode="python"))
    except Exception:
        return None


def _case_plan_closes(entry: CaseManifestEntry, plan: CasePlan) -> bool:
    reference_hash = hashlib.sha256(plan.reference_payload).hexdigest()
    return bool(
        plan.case_manifest_entry_id == entry.case_manifest_entry_id
        and plan.context_manifest_entry_id == entry.context_manifest_entry_id
        and plan.source_question_number_1_indexed == entry.source_question_number_1_indexed
        and hashlib.sha256(plan.question_bytes).hexdigest() == entry.question_bytes_sha256
        and reference_hash == plan.reference_payload_sha256
    )


def _runtime_plan_closes(
    manifest: IngestionPlanManifest,
    runtime: IngestionPlan,
    *,
    expected_source_unit_ids: tuple[str, ...],
    expected_shared_context_sha256: str,
) -> bool:
    try:
        payload_hashes = tuple(
            hashlib.sha256(unit.payload_bytes).hexdigest() for unit in runtime.ordered_source_units
        )
        recomputed_payload_hash = ingestion_payload_hash(payload_hashes)
        recomputed_plan_id = ingestion_plan_id(
            manifest.plan_manifest_entry_id,
            recomputed_payload_hash,
        )
    except Exception:
        return False
    return bool(
        runtime.ingestion_plan_id == manifest.ingestion_plan_id == recomputed_plan_id
        and runtime.ordered_member_context_manifest_entry_ids
        == manifest.ordered_member_context_manifest_entry_ids
        and runtime.ordered_case_manifest_entry_ids == manifest.ordered_case_manifest_entry_ids
        and runtime.shared_context_sha256 == expected_shared_context_sha256
        and runtime.intended_source_count == len(runtime.ordered_source_units)
        and payload_hashes == manifest.ordered_source_unit_bytes_sha256
        and recomputed_payload_hash == manifest.ingestion_payload_hash
        and len(expected_source_unit_ids) == len(runtime.ordered_source_units)
        and all(
            unit.ordinal_1_indexed == ordinal
            and unit.context_manifest_entry_id
            == manifest.ordered_member_context_manifest_entry_ids[0]
            and unit.payload_sha256 == payload_hashes[ordinal - 1]
            and unit.source_unit_id == expected_source_unit_ids[ordinal - 1]
            for ordinal, unit in enumerate(runtime.ordered_source_units, start=1)
        )
    )


def _lme_runtime_sources_close(target: LongMemEvalBundle) -> bool:
    try:
        if not (
            len(target.selected_rows)
            == len(target.case_manifest.logical_contexts)
            == len(target.case_manifest.ingestion_plans)
            == len(target.ingestion_plans)
        ):
            return False
        source_file = target.dataset_manifest.source_files[0]
        for row, context, manifest, runtime in zip(
            target.selected_rows,
            target.case_manifest.logical_contexts,
            target.case_manifest.ingestion_plans,
            target.ingestion_plans,
            strict=True,
        ):
            documents = tuple(
                render_lme_session_document(session.messages) for session in row.sessions
            )
            document_hashes = tuple(hashlib.sha256(document).hexdigest() for document in documents)
            context_hash = canonical_sha256(
                [
                    "oamb-lme-context-bytes-v1",
                    tuple(
                        (session.session_id, session.canonical_timestamp, document_hash)
                        for session, document_hash in zip(
                            row.sessions,
                            document_hashes,
                            strict=True,
                        )
                    ),
                ]
            )
            content_id = context_content_id(
                target.dataset_manifest.revision,
                target.dataset_manifest.split,
                LME_SOURCE_NAME,
                context_hash,
            )
            expected_context_id = context_manifest_entry_id(
                target.dataset_manifest.manifest_hash,
                source_file.sha256,
                row.source_row_number_1_indexed,
                content_id,
            )
            expected_source_ids = tuple(
                canonical_sha256(
                    [
                        "oamb-lme-source-unit-v1",
                        target.case_manifest.workload_id,
                        row.question_id,
                        session.session_id,
                        ordinal,
                        document_hash,
                    ]
                )
                for ordinal, (session, document_hash) in enumerate(
                    zip(row.sessions, document_hashes, strict=True),
                    start=1,
                )
            )
            if not (
                context.context_content_id == content_id
                and context.context_manifest_entry_id == expected_context_id
                and context.source_file_sha256 == source_file.sha256
                and context.source_row_number_1_indexed == row.source_row_number_1_indexed
                and context.context_bytes_sha256 == context_hash
                and tuple(unit.payload_bytes for unit in runtime.ordered_source_units) == documents
                and tuple(unit.source_reference for unit in runtime.ordered_source_units)
                == tuple(session.session_id for session in row.sessions)
                and tuple(unit.occurred_at for unit in runtime.ordered_source_units)
                == tuple(session.canonical_timestamp for session in row.sessions)
                and _runtime_plan_closes(
                    manifest,
                    runtime,
                    expected_source_unit_ids=expected_source_ids,
                    expected_shared_context_sha256=context_hash,
                )
            ):
                return False
        return True
    except Exception:
        return False


def _wrong_target(rule_id: str) -> tuple[ValidationIssue, ...]:
    return (_issue(rule_id, "target", "validation-target-type-mismatch"),)


def _issue(rule_id: str, evidence_ref: str, code: str) -> ValidationIssue:
    return ValidationIssue(
        rule_id=rule_id,
        code=code,
        severity=ValidationSeverity.ERROR,
        evidence_ref=evidence_ref,
        json_pointer=None,
        remediation_code=f"repair-{code}",
    )


WORKLOAD_RULES: tuple[ValidationRule, ...] = (
    ValidationRule("workload.lme.source-manifest.v1", 1, _lme_source_manifest_rule),
    ValidationRule("workload.lme.prompt-answer-parser.v1", 1, _lme_prompt_answer_parser_rule),
    ValidationRule("workload.lme.judge-metric.v1", 1, _lme_judge_metric_rule),
    ValidationRule("workload.lme6.selector-membership.v1", 1, _lme6_selector_membership_rule),
    ValidationRule("workload.lme6.denominator.v1", 1, _lme6_denominator_rule),
    ValidationRule("workload.lme30.selector-coverage.v1", 1, _lme30_selector_coverage_rule),
    ValidationRule("workload.lme30.denominator.v1", 1, _lme30_denominator_rule),
    ValidationRule("workload.lme60.selector-membership.v1", 1, _lme60_selector_membership_rule),
    ValidationRule("workload.lme60.denominator.v1", 1, _lme60_denominator_rule),
)


__all__ = ["WORKLOAD_RULES"]
