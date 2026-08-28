"""Production semantic rules for the frozen T8 workload profiles."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from typing import Any

from oamb.artifacts.validation.registry import ValidationRule
from oamb.contracts.evidence import ValidationIssue, ValidationSeverity
from oamb.contracts.ids import (
    canonical_json_bytes,
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
    LongMemEvalBundle,
    render_lme_session_document,
)
from oamb.workloads.mab65_reduction import Mab65Reduction, reduce_mab65
from oamb.workloads.memoryagentbench import (
    MAB5_CASE_MANIFEST_HASH,
    MAB5_MANIFEST_ID,
    MAB5_SELECTED_CASE_DIGEST,
    MAB5_SELECTED_PLAN_DIGEST,
    MAB65_CASE_MANIFEST_HASH,
    MAB65_SELECTED_CASE_DIGEST,
    MAB65_SELECTED_PLAN_DIGEST,
    MAB65_WORKLOAD_ID,
    MAB_DATASET_ID,
    MAB_DATASET_MANIFEST_HASH,
    MAB_DATASET_REVISION,
    MAB_DATASET_SPLIT,
    MAB_ENTITY_SOURCE_FILE,
    MAB_PAYLOAD_POLICY,
    MAB_PINNED_SOURCE_FILES,
    MAB_PROMPT_PACKS,
    MAB_SOURCE_LICENSE_ID,
    MabManifestBundle,
)
from oamb.workloads.redial import MovieCatalog

_MAB_METRIC_BY_COMPONENT = {
    "ar": "mab-substring-em-v1",
    "icl": "mab-exact-v1",
    "recsys": "mab-redial-recall-at-5-v1",
    "lru": "mab-exact-v1",
    "cr_sf": "mab-substring-em-v1",
}
_MAB5_LABELS = (
    "eventqa_65536@q6",
    "recsys_redial_full@q33",
    "icl_banking77_5900shot_balance@q56",
    "detective_qa@q8",
    "factconsolidation_mh_262k@q21",
)
_MAB65_COMPONENT_COUNTS = {
    "ar": 15,
    "icl": 10,
    "recsys": 10,
    "lru": 15,
    "cr_sf": 15,
}
_MAB65_VALIDATION_SCORE_BY_COMPONENT = {
    "ar": Fraction(1),
    "icl": Fraction(0),
    "recsys": Fraction(1, 2),
    "lru": Fraction(1, 4),
    "cr_sf": Fraction(3, 4),
}
_MAB65_VALIDATION_INDEX = Fraction(225, 4)
_MAB_ENTITY_CATALOG_CONTENT_DIGEST = (
    "eb4b6336039b0d812e97ad4ed169bcbb2f169856378649b17fae8e878d67dc81"
)
_MAB_UNICODE_FINGERPRINT = "1a846dffe314101b43cb68132979e8a1f052b7d2115532981d9866db1eeb352a"


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
        and target.case_manifest.workload_id == LME30_WORKLOAD_ID
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


def _mab_source_manifest_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab.source-manifest.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    dataset = target.dataset_manifest
    source_files = tuple(
        DatasetFile(
            relative_path=relative_path,
            sha256=sha256,
            byte_count=byte_count,
            license_id=MAB_SOURCE_LICENSE_ID,
        )
        for relative_path, sha256, _split, byte_count in (
            *MAB_PINNED_SOURCE_FILES,
            (
                MAB_ENTITY_SOURCE_FILE[0],
                MAB_ENTITY_SOURCE_FILE[1],
                "catalog",
                MAB_ENTITY_SOURCE_FILE[2],
            ),
        )
    )
    expected_dataset = DatasetManifest(
        dataset_id=MAB_DATASET_ID,
        revision=MAB_DATASET_REVISION,
        split=MAB_DATASET_SPLIT,
        manifest_hash=MAB_DATASET_MANIFEST_HASH,
        source_files=source_files,
        payload_policy=MAB_PAYLOAD_POLICY,
    )
    exact = bool(
        _canonical_dataset_manifest(dataset) == expected_dataset
        and _canonical_case_manifest(target.case_manifest) is not None
        and target.case_manifest.workload_id == MAB65_WORKLOAD_ID
        and _mab_catalog_closes(target)
        and _mab_case_provenance_closes(target)
        and _mab_runtime_sources_close(target)
    )
    return () if exact else (_issue(rule_id, dataset.manifest_hash, "mab-source-drift"),)


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


def _mab_prompt_answer_parser_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab.prompt-answer-parser.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    bindings = {
        item.pack.manifest.prompt_pack_id: (
            item.pack.manifest.output_contract_id,
            item.max_output_tokens,
        )
        for item in MAB_PROMPT_PACKS
    }
    exact = bool(
        target.cases
        and all(
            _case_plan_closes(case.entry, case.case_plan)
            and _mab_reference_payload_closes(case.entry, case.case_plan)
            and bindings.get(case.case_plan.prompt_binding_id)
            == (case.case_plan.output_contract_id, case.case_plan.answer_max_output_tokens)
            for case in target.cases
        )
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "mab-prompt-parser-drift"),)
    )


def _mab_metric_strata_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab.metric-strata.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    exact = bool(
        target.cases
        and all(
            case.case_plan.metric_id == _MAB_METRIC_BY_COMPONENT.get(case.component)
            and case.case_plan.judge_binding_id is None
            for case in target.cases
        )
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "mab-metric-strata-drift"),)
    )


def _mab5_selector_grouping_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab5.selector-grouping.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    exact = bool(
        target.case_manifest.manifest_id == MAB5_MANIFEST_ID
        and target.case_manifest.manifest_hash == MAB5_CASE_MANIFEST_HASH
        and target.selected_case_digest == MAB5_SELECTED_CASE_DIGEST
        and target.selected_plan_digest == MAB5_SELECTED_PLAN_DIGEST
        and target.selected_case_digest
        == canonical_sha256(
            [
                "oamb-mab5-selected-case-ids-v1",
                tuple(case.entry.case_manifest_entry_id for case in target.cases),
            ]
        )
        and target.selected_plan_digest
        == canonical_sha256(
            [
                "oamb-mab5-selected-plan-manifest-ids-v1",
                tuple(plan.manifest.plan_manifest_entry_id for plan in target.plans),
            ]
        )
        and tuple(case.logical_case_label for case in target.cases) == _MAB5_LABELS
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "mab5-selector-drift"),)
    )


def _mab5_smoke_isolation_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab5.smoke-isolation.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    exact = bool(
        len(target.plans) == len(target.cases) == 5
        and all(len(plan.manifest.ordered_case_manifest_entry_ids) == 1 for plan in target.plans)
        and len(target.plans[-1].manifest.ordered_member_context_manifest_entry_ids) == 2
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "mab5-isolation-drift"),)
    )


def _mab5_no_aggregate_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab5.no-aggregate.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    exact = bool(
        target.case_manifest.manifest_id == MAB5_MANIFEST_ID
        and target.case_manifest.manifest_hash != MAB65_CASE_MANIFEST_HASH
        and len(target.cases) == 5
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "mab5-aggregate-boundary-drift"),)
    )


def _mab65_selector_grouping_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab65.selector-grouping.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    actual_counts = {
        component: sum(case.component == component for case in target.cases)
        for component in _MAB65_COMPONENT_COUNTS
    }
    exact = bool(
        target.case_manifest.manifest_id == MAB65_WORKLOAD_ID
        and target.case_manifest.manifest_hash == MAB65_CASE_MANIFEST_HASH
        and target.selected_case_digest == MAB65_SELECTED_CASE_DIGEST
        and target.selected_plan_digest == MAB65_SELECTED_PLAN_DIGEST
        and target.selected_case_digest
        == canonical_sha256(
            [
                "oamb-mab65-selected-case-ids-v1",
                tuple(case.entry.case_manifest_entry_id for case in target.cases),
            ]
        )
        and target.selected_plan_digest
        == canonical_sha256(
            [
                "oamb-mab65-selected-plan-manifest-ids-v1",
                tuple(plan.manifest.plan_manifest_entry_id for plan in target.plans),
            ]
        )
        and len(target.case_manifest.logical_contexts) == 29
        and len(target.plans) == 25
        and len(target.cases) == 65
        and actual_counts == _MAB65_COMPONENT_COUNTS
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "mab65-selector-drift"),)
    )


def _mab65_reduction(target: MabManifestBundle) -> Mab65Reduction:
    return reduce_mab65(
        target,
        case_metrics={
            case.entry.case_manifest_entry_id: _MAB65_VALIDATION_SCORE_BY_COMPONENT[case.component]
            for case in target.cases
        },
        ready_plan_manifest_ids=frozenset(
            plan.manifest.plan_manifest_entry_id for plan in target.plans
        ),
        capsule_valid=True,
        query_state_unchanged=True,
        expected_catalog_sha256=target.entity_catalog_sha256,
        expected_unicode_fingerprint=target.unicode_fingerprint,
        expected_interaction_fingerprint="f" * 64,
        observed_interaction_fingerprint="f" * 64,
        comparison_controls_closed=True,
    )


def _mab65_complete_reducer_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab65.complete-reducer.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    reduced = _mab65_reduction(target)
    exact = bool(
        reduced.available
        and len(reduced.plans) == 25
        and len(reduced.components) == 5
        and tuple(item.score for item in reduced.components)
        == tuple(
            _MAB65_VALIDATION_SCORE_BY_COMPONENT[component]
            for component in ("ar", "icl", "recsys", "lru", "cr_sf")
        )
        and reduced.index_value == _MAB65_VALIDATION_INDEX
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "mab65-reducer-unavailable"),)
    )


def _mab65_capability_index_rule(target: Any) -> tuple[ValidationIssue, ...]:
    rule_id = "workload.mab65.capability-index.v1"
    if not isinstance(target, MabManifestBundle):
        return _wrong_target(rule_id)
    reduced = _mab65_reduction(target)
    exact = bool(
        reduced.available
        and tuple(item.capability for item in reduced.capabilities) == ("ar", "ttl", "lru", "cr_sf")
        and all(item.weight == Fraction(1, 4) for item in reduced.capabilities)
        and reduced.ttl_score == Fraction(1, 4)
        and reduced.index_value == _MAB65_VALIDATION_INDEX
    )
    return (
        ()
        if exact
        else (_issue(rule_id, target.case_manifest.manifest_hash, "mab65-capability-index-drift"),)
    )


def _case_plan_closes(entry: CaseManifestEntry, plan: CasePlan) -> bool:
    reference_hash = hashlib.sha256(plan.reference_payload).hexdigest()
    return bool(
        plan.case_manifest_entry_id == entry.case_manifest_entry_id
        and plan.context_manifest_entry_id == entry.context_manifest_entry_id
        and plan.source_question_number_1_indexed == entry.source_question_number_1_indexed
        and hashlib.sha256(plan.question_bytes).hexdigest() == entry.question_bytes_sha256
        and reference_hash == plan.reference_payload_sha256
    )


def _mab_reference_payload_closes(entry: CaseManifestEntry, plan: CasePlan) -> bool:
    try:
        values = json.loads(plan.reference_payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return bool(
        isinstance(values, list)
        and values
        and all(isinstance(value, str) and value for value in values)
        and canonical_json_bytes(values) == plan.reference_payload
        and tuple(hashlib.sha256(value.encode("utf-8")).hexdigest() for value in values)
        == entry.answer_value_sha256
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
                        LME30_WORKLOAD_ID,
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


def _mab_case_provenance_closes(target: MabManifestBundle) -> bool:
    try:
        source_sha256_by_path = {
            source.relative_path: source.sha256 for source in target.dataset_manifest.source_files
        }
        contexts_by_id = {
            context.context_manifest_entry_id: context
            for context in target.case_manifest.logical_contexts
        }
        if len(target.cases) != len(target.case_manifest.cases):
            return False
        for manifest_case, case in zip(
            target.case_manifest.cases,
            target.cases,
            strict=True,
        ):
            context = contexts_by_id[manifest_case.context_manifest_entry_id]
            source_sha256 = source_sha256_by_path[case.source_relative_path]
            expected_context_id = context_manifest_entry_id(
                target.dataset_manifest.manifest_hash,
                source_sha256,
                case.source_row_number_1_indexed,
                context.context_content_id,
            )
            if not (
                case.entry == manifest_case
                and context.source_file_sha256 == source_sha256
                and context.source_row_number_1_indexed == case.source_row_number_1_indexed
                and context.context_manifest_entry_id == expected_context_id
            ):
                return False
        return True
    except Exception:
        return False


def _mab_catalog_closes(target: MabManifestBundle) -> bool:
    try:
        entity_map = {movie.source_uri: movie.entity_id for movie in target.movie_catalog.movies}
        if len(entity_map) != len(target.movie_catalog.movies):
            return False
        rebuilt_catalog = MovieCatalog.from_entity_map(entity_map)
        content_digest = canonical_sha256(
            [
                "oamb-mab-entity-catalog-content-v1",
                tuple(
                    sorted(
                        entity_map.items(),
                        key=lambda item: (item[0].encode("utf-8"), item[1]),
                    )
                ),
            ]
        )
        unicode_fingerprint = canonical_sha256(
            ["oamb-mab-redial-unicode-v1", rebuilt_catalog.unicode_version]
        )
        return bool(
            target.entity_catalog_sha256 == MAB_ENTITY_SOURCE_FILE[1]
            and content_digest == _MAB_ENTITY_CATALOG_CONTENT_DIGEST
            and target.movie_catalog == rebuilt_catalog
            and unicode_fingerprint == _MAB_UNICODE_FINGERPRINT
            and target.unicode_fingerprint == unicode_fingerprint
        )
    except Exception:
        return False


def _mab_runtime_sources_close(target: MabManifestBundle) -> bool:
    try:
        contexts_by_id = {
            context.context_manifest_entry_id: context
            for context in target.case_manifest.logical_contexts
        }
        if len(target.plans) != len(target.case_manifest.ingestion_plans):
            return False
        for manifest, plan in zip(
            target.case_manifest.ingestion_plans,
            target.plans,
            strict=True,
        ):
            runtime = plan.runtime_plan
            payload_hashes = tuple(
                hashlib.sha256(unit.payload_bytes).hexdigest()
                for unit in runtime.ordered_source_units
            )
            expected_source_ids = tuple(
                canonical_sha256([manifest.ingestion_plan_id, ordinal, payload_hash])
                for ordinal, payload_hash in enumerate(payload_hashes, start=1)
            )
            member_context_hashes = {
                contexts_by_id[context_id].context_bytes_sha256
                for context_id in manifest.ordered_member_context_manifest_entry_ids
            }
            if not (
                plan.manifest == manifest
                and len(member_context_hashes) == 1
                and _runtime_plan_closes(
                    manifest,
                    runtime,
                    expected_source_unit_ids=expected_source_ids,
                    expected_shared_context_sha256=next(iter(member_context_hashes)),
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
    ValidationRule("workload.mab.source-manifest.v1", 1, _mab_source_manifest_rule),
    ValidationRule("workload.mab.prompt-answer-parser.v1", 1, _mab_prompt_answer_parser_rule),
    ValidationRule("workload.mab.metric-strata.v1", 1, _mab_metric_strata_rule),
    ValidationRule("workload.mab5.selector-grouping.v1", 1, _mab5_selector_grouping_rule),
    ValidationRule("workload.mab5.smoke-isolation.v1", 1, _mab5_smoke_isolation_rule),
    ValidationRule("workload.mab5.no-aggregate.v1", 1, _mab5_no_aggregate_rule),
    ValidationRule("workload.mab65.selector-grouping.v1", 1, _mab65_selector_grouping_rule),
    ValidationRule("workload.mab65.complete-reducer.v1", 1, _mab65_complete_reducer_rule),
    ValidationRule("workload.mab65.capability-index.v1", 1, _mab65_capability_index_rule),
)


__all__ = ["WORKLOAD_RULES"]
