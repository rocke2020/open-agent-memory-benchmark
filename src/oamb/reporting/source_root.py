"""Generic offline report construction from a sealed source root."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, TypeAlias

from oamb.artifacts.validation.fake import EvidenceNotValidatedError
from oamb.artifacts.validation.native import NATIVE_EVIDENCE_PROFILE_ID
from oamb.artifacts.validation.profiles import fake_evidence_profile
from oamb.artifacts.validation.source_root import validate_source_root
from oamb.contracts.evidence import ValidationResult
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.states import ValidationDisposition

from .html import ReportBuildResult, build_fake_report
from .native_reduce import (
    native_report_reducer_implementation_hash,
    reduce_native_run_report,
)
from .offline_renderer import offline_asset_hashes, offline_renderer_hash
from .publication import ReportBuildResultV2, build_report_derivation
from .roots import build_report_spec

SOURCE_ROOT_REPORT_COMMITTED_AT = datetime(2026, 1, 2, tzinfo=UTC)
SOURCE_ROOT_REPORT_PREVIEW_MAX_FIELD_BYTES = 4_096
SOURCE_ROOT_REPORT_PREVIEW_TOTAL_BYTES = 65_536
SOURCE_ROOT_REPORT_DISPLAY_FIELDS = (
    "identity",
    "completion",
    "quality",
    "usage",
    "limitations",
)
SourceRootReportBuildResult: TypeAlias = ReportBuildResult | ReportBuildResultV2


def build_source_root_report(
    source_root: Path,
    evidence_validation: ValidationResult,
    *,
    output_root: Path,
    audience: Literal["local", "public"],
    diagnostic: bool = False,
) -> SourceRootReportBuildResult:
    """Revalidate and publish one report from a supported sealed source root."""

    if evidence_validation.disposition != ValidationDisposition.VALIDATED:
        raise EvidenceNotValidatedError("report requires a VALIDATED evidence result")
    root = Path(source_root)
    if validate_source_root(root) != evidence_validation:
        raise EvidenceNotValidatedError(
            "report requires fresh validation of the current source-root bytes"
        )
    if evidence_validation.validation_profile_id == fake_evidence_profile().profile_id:
        return build_fake_report(
            root,
            evidence_validation,
            output_root=output_root,
            audience=audience,
            diagnostic=diagnostic,
        )
    if evidence_validation.validation_profile_id != NATIVE_EVIDENCE_PROFILE_ID:
        raise EvidenceNotValidatedError("report does not support this validation profile")
    if diagnostic:
        raise EvidenceNotValidatedError("native diagnostic reporting is not supported")

    report_spec = build_report_spec(
        report_kind="run",
        audience=audience,
        preview_max_field_bytes=SOURCE_ROOT_REPORT_PREVIEW_MAX_FIELD_BYTES,
        preview_total_bytes=SOURCE_ROOT_REPORT_PREVIEW_TOTAL_BYTES,
        display_field_ids=SOURCE_ROOT_REPORT_DISPLAY_FIELDS,
        renderer_hash=offline_renderer_hash(),
        asset_hashes=offline_asset_hashes(),
        export_profile_selector_id=f"{audience}-run-v1",
        export_profile_selector_version=1,
    )
    model = reduce_native_run_report(root, evidence_validation, report_spec=report_spec)
    return build_report_derivation(
        model=model,
        report_spec=report_spec,
        ordered_source_bindings=model.ordered_source_bindings,
        evidence_validations=(evidence_validation,),
        evidence_validation_targets=(root,),
        transform_spec_hash=canonical_sha256(
            [
                "oamb-source-root-report-transform-v1",
                native_report_reducer_implementation_hash(),
            ]
        ),
        schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
        output_root=output_root,
        committed_at=SOURCE_ROOT_REPORT_COMMITTED_AT,
    )


__all__ = ["SourceRootReportBuildResult", "build_source_root_report"]
