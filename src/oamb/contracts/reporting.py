"""Minimal deterministic report contracts required by the future fake slice."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from .base import NonEmptyStr, NonNegativeInt, Sha256, StrictContract


class RunSummary(StrictContract):
    schema_name: Literal["run_summary"] = "run_summary"
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    intended_logical_contexts: NonNegativeInt
    intended_ingestion_plans: NonNegativeInt
    ready_ingestion_plans: NonNegativeInt
    intended_cases: NonNegativeInt
    terminal_cases: NonNegativeInt
    completed_cases: NonNegativeInt
    errored_cases: NonNegativeInt
    unsupported_cases: NonNegativeInt
    cancelled_cases: NonNegativeInt
    budget_exceeded_cases: NonNegativeInt
    billing_complete: bool
    cost_complete: bool


class RunSummaryV2(StrictContract):
    schema_name: Literal["run_summary"] = "run_summary"
    schema_version: Literal[2] = 2
    run_id: NonEmptyStr
    intended_logical_contexts: NonNegativeInt
    intended_ingestion_plans: NonNegativeInt
    ready_ingestion_plans: NonNegativeInt
    intended_cases: NonNegativeInt
    terminal_cases: NonNegativeInt
    completed_cases: NonNegativeInt
    errored_cases: NonNegativeInt
    unsupported_cases: NonNegativeInt
    cancelled_cases: NonNegativeInt
    budget_exceeded_cases: NonNegativeInt
    parsed_cases: NonNegativeInt
    evaluated_cases: NonNegativeInt
    judged_cases: NonNegativeInt
    unjudged_cases: NonNegativeInt
    billing_complete: bool
    cost_complete: bool

    @model_validator(mode="after")
    def completion_counts_are_bounded(self) -> RunSummaryV2:
        terminal_parts = (
            self.completed_cases
            + self.errored_cases
            + self.unsupported_cases
            + self.cancelled_cases
            + self.budget_exceeded_cases
        )
        if self.ready_ingestion_plans > self.intended_ingestion_plans:
            raise ValueError("ready ingestion plans exceed intended plans")
        if self.terminal_cases > self.intended_cases or terminal_parts != self.terminal_cases:
            raise ValueError("terminal case counts do not close")
        if self.parsed_cases > self.terminal_cases:
            raise ValueError("parsed cases exceed terminal cases")
        if self.judged_cases > self.evaluated_cases:
            raise ValueError("judged cases exceed evaluated cases")
        if self.evaluated_cases + self.unjudged_cases > self.parsed_cases:
            raise ValueError("evaluation dispositions exceed parsed cases")
        if self.unjudged_cases > self.errored_cases:
            raise ValueError("unjudged cases require terminal errors")
        return self


class RunReportModel(StrictContract):
    schema_name: Literal["run_report_model"] = "run_report_model"
    schema_version: Literal[1] = 1
    report_id: Sha256
    source_manifest_hash: Sha256
    evidence_validation_hash: Sha256
    summary: RunSummary
    limitations: tuple[NonEmptyStr, ...]


class RunReportModelV2(StrictContract):
    schema_name: Literal["run_report_model"] = "run_report_model"
    schema_version: Literal[2] = 2
    report_id: Sha256
    source_manifest_hash: Sha256
    evidence_validation_hash: Sha256
    summary: RunSummaryV2
    limitations: tuple[NonEmptyStr, ...]


class ReportArtifactManifest(StrictContract):
    schema_name: Literal["report_artifact_manifest"] = "report_artifact_manifest"
    schema_version: Literal[1] = 1
    report_id: Sha256
    ordered_source_root_hashes: tuple[Sha256, ...]
    evidence_validation_hash: Sha256
    report_model_hash: Sha256
    renderer_hash: Sha256
    asset_hashes: tuple[Sha256, ...]
    audience: Literal["local", "public"]
    limitations: tuple[NonEmptyStr, ...]
