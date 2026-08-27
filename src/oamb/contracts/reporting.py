"""Minimal deterministic report contracts required by the future fake slice."""

from __future__ import annotations

from typing import Literal

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


class RunReportModel(StrictContract):
    schema_name: Literal["run_report_model"] = "run_report_model"
    schema_version: Literal[1] = 1
    report_id: Sha256
    source_manifest_hash: Sha256
    evidence_validation_hash: Sha256
    summary: RunSummary
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
