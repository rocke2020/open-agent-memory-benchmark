"""Immutable source evidence and structural validation records."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal, Self

from pydantic import model_validator

from .base import (
    NonEmptyStr,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictContract,
    UtcDateTime,
)
from .states import (
    AttemptOutcome,
    CaseState,
    IndexContribution,
    IngestionPlanState,
    ResumeDisposition,
    RunState,
    ValidationDisposition,
)


class OriginClass(StrEnum):
    OAMB_NATIVE = "oamb_native"
    EXTERNAL_AMB_GENERATED = "external_amb_generated"
    IMPORTED_THIRD_PARTY = "imported_third_party"


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class RawReference(StrictContract):
    schema_name: Literal["raw_reference"] = "raw_reference"
    schema_version: Literal[1] = 1
    sha256: Sha256
    media_type: NonEmptyStr
    byte_count: NonNegativeInt
    compression: Literal["none", "gzip"]


class OriginRecord(StrictContract):
    schema_name: Literal["origin_record"] = "origin_record"
    schema_version: Literal[1] = 1
    origin_id: NonEmptyStr
    origin_class: OriginClass
    producer: NonEmptyStr
    source_sha256: Sha256
    license_id: NonEmptyStr


class RunRecord(StrictContract):
    schema_name: Literal["run_record"] = "run_record"
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    run_spec_hash: Sha256
    state: RunState
    resume_disposition: ResumeDisposition
    started_at: UtcDateTime | None
    ended_at: UtcDateTime | None
    ingestion_occurrence_ids: tuple[Sha256, ...]
    case_occurrence_ids: tuple[Sha256, ...]


class AttemptRecord(StrictContract):
    schema_name: Literal["attempt_record"] = "attempt_record"
    schema_version: Literal[1] = 1
    attempt_id: Sha256
    parent_kind: Literal["ingestion_plan", "case", "phase_review"]
    parent_id: NonEmptyStr
    stage: NonEmptyStr
    ordinal: PositiveInt
    request_fingerprint: Sha256
    started_at: UtcDateTime
    ended_at: UtcDateTime
    outcome: AttemptOutcome
    retry_of_attempt_id: Sha256 | None
    idempotency_key_hash: Sha256 | None
    reconciliation_capability: Literal["none", "idempotency_key", "receipt_lookup"]
    raw_response_ref: Sha256 | None
    raw_error_ref: Sha256 | None
    index_contribution: IndexContribution
    superseded_by_attempt_id: Sha256 | None

    @model_validator(mode="after")
    def coherent_times_and_contribution(self) -> Self:
        if self.ended_at < self.started_at:
            raise ValueError("attempt end precedes start")
        if self.index_contribution == IndexContribution.SUPERSEDED:
            if self.superseded_by_attempt_id is None:
                raise ValueError("superseded contribution requires successor attempt")
        elif self.superseded_by_attempt_id is not None:
            raise ValueError("only superseded contribution names a successor")
        return self


class LogicalContextRecord(StrictContract):
    schema_name: Literal["logical_context_record"] = "logical_context_record"
    schema_version: Literal[1] = 1
    context_manifest_entry_id: Sha256
    run_id: NonEmptyStr
    context_content_id: Sha256
    source_file_sha256: Sha256
    source_row_number_1_indexed: PositiveInt
    context_bytes_sha256: Sha256
    ordered_case_manifest_entry_ids: tuple[Sha256, ...]
    origin_id: NonEmptyStr


class IngestionPlanRecord(StrictContract):
    schema_name: Literal["ingestion_plan_record"] = "ingestion_plan_record"
    schema_version: Literal[1] = 1
    ingestion_occurrence_id: Sha256
    run_id: NonEmptyStr
    memory_system_id: NonEmptyStr
    ingestion_plan_id: Sha256
    ordered_member_context_manifest_entry_ids: tuple[Sha256, ...]
    ordered_case_occurrence_ids: tuple[Sha256, ...]
    state: IngestionPlanState
    intended_source_count: NonNegativeInt
    accepted_source_count: NonNegativeInt
    failed_source_count: NonNegativeInt
    readiness_evidence_refs: tuple[Sha256, ...]
    attempt_ids: tuple[Sha256, ...]
    usage_record_ids: tuple[Sha256, ...]
    resource_record_ids: tuple[Sha256, ...]
    cost_record_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def source_counts_close(self) -> Self:
        if self.accepted_source_count + self.failed_source_count > self.intended_source_count:
            raise ValueError("accepted plus failed sources exceeds intended sources")
        if self.state in {IngestionPlanState.READY, IngestionPlanState.SEALED}:
            if self.accepted_source_count + self.failed_source_count != self.intended_source_count:
                raise ValueError("READY ingestion plan requires closed source counts")
            if not self.readiness_evidence_refs:
                raise ValueError("READY ingestion plan requires readiness evidence")
        return self


class CaseRecord(StrictContract):
    schema_name: Literal["case_record"] = "case_record"
    schema_version: Literal[1] = 1
    case_occurrence_id: Sha256
    run_id: NonEmptyStr
    ingestion_occurrence_id: Sha256
    case_manifest_entry_id: Sha256
    state: CaseState
    retrieval_raw_ref: Sha256 | None
    prompt_sha256: Sha256 | None
    answer_raw_ref: Sha256 | None
    evaluation_raw_ref: Sha256 | None
    attempt_ids: tuple[Sha256, ...]
    error_stage: str | None


class CapsuleManifestEntry(StrictContract):
    schema_name: Literal["capsule_manifest_entry"] = "capsule_manifest_entry"
    schema_version: Literal[1] = 1
    record_kind: NonEmptyStr
    record_id: NonEmptyStr
    relative_path: NonEmptyStr
    sha256: Sha256

    @model_validator(mode="after")
    def relative_safe_path(self) -> Self:
        components = self.relative_path.split("/")
        if self.relative_path.startswith("/") or ".." in components:
            raise ValueError("capsule path must be relative and non-escaping")
        return self


class CapsuleManifest(StrictContract):
    schema_name: Literal["capsule_manifest"] = "capsule_manifest"
    schema_version: Literal[1] = 1
    capsule_id: Sha256
    run_id: NonEmptyStr
    run_spec_hash: Sha256
    source_entries: tuple[CapsuleManifestEntry, ...]
    source_manifest_hash: Sha256

    @model_validator(mode="after")
    def excludes_self_and_checkpoints(self) -> Self:
        paths = tuple(item.relative_path for item in self.source_entries)
        if len(set(paths)) != len(paths):
            raise ValueError("capsule manifest contains duplicate paths")
        if any(
            path == "capsule-manifest.json" or path.startswith("checkpoints/") for path in paths
        ):
            raise ValueError("capsule manifest cannot index itself or checkpoints")
        return self


class ValidationIssue(StrictContract):
    schema_name: Literal["validation_issue"] = "validation_issue"
    schema_version: Literal[1] = 1
    rule_id: NonEmptyStr
    code: NonEmptyStr
    severity: ValidationSeverity
    evidence_ref: NonEmptyStr
    json_pointer: str | None
    remediation_code: NonEmptyStr


class ValidationResult(StrictContract):
    schema_name: Literal["validation_result"] = "validation_result"
    schema_version: Literal[1] = 1
    validation_profile_id: NonEmptyStr
    target_hash: Sha256
    disposition: ValidationDisposition
    required_rule_ids: tuple[NonEmptyStr, ...]
    executed_rule_ids: tuple[NonEmptyStr, ...]
    passed_rule_ids: tuple[NonEmptyStr, ...]
    failed_rule_ids: tuple[NonEmptyStr, ...]
    not_applicable_rule_ids: tuple[NonEmptyStr, ...]
    missing_rule_ids: tuple[NonEmptyStr, ...]
    implementation_versions: tuple[NonEmptyStr, ...]
    issues: tuple[ValidationIssue, ...]

    @model_validator(mode="after")
    def rule_partitions_and_disposition_close(self) -> Self:
        categories = {
            "required": self.required_rule_ids,
            "executed": self.executed_rule_ids,
            "passed": self.passed_rule_ids,
            "failed": self.failed_rule_ids,
            "not_applicable": self.not_applicable_rule_ids,
            "missing": self.missing_rule_ids,
        }
        if any(len(set(values)) != len(values) for values in categories.values()):
            raise ValueError("validation rule partitions cannot contain duplicates")
        required = set(self.required_rule_ids)
        executed = set(self.executed_rule_ids)
        passed = set(self.passed_rule_ids)
        failed = set(self.failed_rule_ids)
        failed_required = failed & required
        not_applicable = set(self.not_applicable_rule_ids)
        missing = set(self.missing_rule_ids)
        if executed != passed | failed_required or passed & failed_required:
            raise ValueError("executed rules must partition into passed and failed rules")
        if any(
            left & right
            for left, right in (
                (executed, not_applicable),
                (executed, missing),
                (not_applicable, missing),
            )
        ):
            raise ValueError("required rule partitions must be disjoint")
        if required != executed | not_applicable | missing:
            raise ValueError(
                "required rules must have an executed, not-applicable, or missing result"
            )
        implemented_rule_ids = tuple(
            item.rsplit("@", 1)[0] for item in self.implementation_versions
        )
        if implemented_rule_ids != self.executed_rule_ids:
            raise ValueError("implementation versions must align with executed rules")
        error_issue_rule_ids = {
            issue.rule_id for issue in self.issues if issue.severity == ValidationSeverity.ERROR
        }
        if error_issue_rule_ids != failed:
            raise ValueError("error issues must correspond exactly to failed rules")
        if any(issue.rule_id not in executed | failed for issue in self.issues):
            raise ValueError("validation issues must correspond to executed or failed rules")
        if self.disposition == ValidationDisposition.VALIDATED:
            if failed or missing:
                raise ValueError("validated result cannot contain failed or missing rules")
        elif not failed and not missing:
            raise ValueError("invalid result requires a failed or missing rule")
        return self
