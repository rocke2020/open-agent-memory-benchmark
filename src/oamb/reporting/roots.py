"""Canonical report-spec and derivation-root identity builders."""

from __future__ import annotations

from oamb.contracts.ids import canonical_sha256
from oamb.contracts.specifications import (
    AcceptanceReportSpec,
    DerivationSpecV2,
    ReportIdentitySpecBinding,
    ReportSpec,
    SourceEvidenceBinding,
    acceptance_report_spec_id,
    derivation_spec_v2_input_hash,
    report_identity_spec_binding_id,
    report_spec_id,
)
from oamb.reporting.acceptance_contracts import (
    browser_acceptance_contract_hash,
    performance_acceptance_contract_hash,
)


def build_report_spec(
    *,
    report_kind: str,
    audience: str,
    preview_max_field_bytes: int,
    preview_total_bytes: int,
    display_field_ids: tuple[str, ...],
    renderer_hash: str,
    asset_hashes: tuple[str, ...],
    browser_contract_hash: str | None = None,
    performance_contract_hash: str | None = None,
    export_profile_selector_id: str,
    export_profile_selector_version: int,
) -> ReportSpec:
    from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash

    expected_browser_hash = browser_acceptance_contract_hash()
    expected_performance_hash = performance_acceptance_contract_hash()
    if renderer_hash != offline_renderer_hash() or asset_hashes != offline_asset_hashes():
        raise ValueError("report spec renderer identity is not repository-owned")
    if browser_contract_hash not in {None, expected_browser_hash}:
        raise ValueError("report spec browser contract is not repository-owned")
    if performance_contract_hash not in {None, expected_performance_hash}:
        raise ValueError("report spec performance contract is not repository-owned")
    fields = {
        "report_kind": report_kind,
        "audience": audience,
        "preview_max_field_bytes": preview_max_field_bytes,
        "preview_total_bytes": preview_total_bytes,
        "display_field_ids": display_field_ids,
        "renderer_hash": renderer_hash,
        "asset_hashes": asset_hashes,
        "browser_contract_hash": expected_browser_hash,
        "performance_contract_hash": expected_performance_hash,
        "export_profile_selector_id": export_profile_selector_id,
        "export_profile_selector_version": export_profile_selector_version,
    }
    return ReportSpec.model_validate(
        {
            "report_spec_id": report_spec_id(**fields),
            **fields,
        }
    )


def build_acceptance_report_spec(
    *,
    audience: str,
    evaluation_report_hash: str,
    evaluation_export_validation_hash: str,
    review_bundle_hash: str,
    ai_review_record_hash: str,
    human_review_record_hash: str,
    phase_gate_hash: str,
    renderer_hash: str,
    asset_hashes: tuple[str, ...],
    browser_contract_hash: str | None = None,
    performance_contract_hash: str | None = None,
    export_profile_selector_id: str,
    export_profile_selector_version: int,
) -> AcceptanceReportSpec:
    from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash

    expected_browser_hash = browser_acceptance_contract_hash()
    expected_performance_hash = performance_acceptance_contract_hash()
    if renderer_hash != offline_renderer_hash() or asset_hashes != offline_asset_hashes():
        raise ValueError("acceptance report renderer identity is not repository-owned")
    if browser_contract_hash not in {None, expected_browser_hash}:
        raise ValueError("acceptance report browser contract is not repository-owned")
    if performance_contract_hash not in {None, expected_performance_hash}:
        raise ValueError("acceptance report performance contract is not repository-owned")
    fields = {
        "audience": audience,
        "evaluation_report_hash": evaluation_report_hash,
        "evaluation_export_validation_hash": evaluation_export_validation_hash,
        "review_bundle_hash": review_bundle_hash,
        "ai_review_record_hash": ai_review_record_hash,
        "human_review_record_hash": human_review_record_hash,
        "phase_gate_hash": phase_gate_hash,
        "renderer_hash": renderer_hash,
        "asset_hashes": asset_hashes,
        "browser_contract_hash": expected_browser_hash,
        "performance_contract_hash": expected_performance_hash,
        "export_profile_selector_id": export_profile_selector_id,
        "export_profile_selector_version": export_profile_selector_version,
    }
    return AcceptanceReportSpec.model_validate(
        {
            "acceptance_report_spec_id": acceptance_report_spec_id(**fields),
            **fields,
        }
    )


def build_report_identity_spec_binding(
    spec: ReportSpec | AcceptanceReportSpec,
) -> ReportIdentitySpecBinding:
    if isinstance(spec, ReportSpec):
        spec_kind = "benchmark_report"
        spec_id = spec.report_spec_id
    else:
        spec_kind = "acceptance_report"
        spec_id = spec.acceptance_report_spec_id
    fields = {
        "spec_kind": spec_kind,
        "spec_schema_name": spec.schema_name,
        "spec_schema_version": spec.schema_version,
        "spec_id": spec_id,
        "spec_hash": canonical_sha256(spec),
    }
    return ReportIdentitySpecBinding.model_validate(
        {
            "binding_id": report_identity_spec_binding_id(**fields),
            **fields,
        }
    )


def build_derivation_spec_v2(
    *,
    derivation_kind: str,
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...],
    evidence_validation_result_hash: str,
    transform_spec_hash: str,
    report_identity_spec_binding: ReportIdentitySpecBinding | None,
    reducer_and_renderer_input_hashes: tuple[str, ...],
) -> DerivationSpecV2:
    ordered_source_root_hash = canonical_sha256(
        [
            "oamb-ordered-source-roots-v1",
            tuple(item.source_root_hash for item in ordered_source_bindings),
        ]
    )
    fields = {
        "derivation_kind": derivation_kind,
        "ordered_source_bindings": ordered_source_bindings,
        "ordered_source_root_hash": ordered_source_root_hash,
        "evidence_validation_result_hash": evidence_validation_result_hash,
        "transform_spec_hash": transform_spec_hash,
        "report_identity_spec_binding": report_identity_spec_binding,
        "reducer_and_renderer_input_hashes": reducer_and_renderer_input_hashes,
    }
    return DerivationSpecV2.model_validate(
        {
            "derivation_input_hash": derivation_spec_v2_input_hash(**fields),
            **fields,
        }
    )


__all__ = [
    "build_acceptance_report_spec",
    "build_derivation_spec_v2",
    "build_report_identity_spec_binding",
    "build_report_spec",
]
