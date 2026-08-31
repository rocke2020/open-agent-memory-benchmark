from __future__ import annotations

import asyncio
import hashlib
import importlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

import pytest

from oamb.artifacts.validation.adapter import (
    Mem0AdapterValidationInput,
    mem0_adapter_validation_input,
)
from oamb.artifacts.validation.catalog import validate_catalog_profile
from oamb.artifacts.validation.reduction import comparison_validation_input
from oamb.contracts.evidence import ValidationResult
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.ports import MemorySystemProfileUnsupported
from oamb.contracts.reporting import (
    ComparisonControlBinding,
    ComparisonControlSnapshot,
    CompletionSummaryV3,
    DisplayPreview,
    ExactRational,
    PairedMetricDelta,
    RunReportModelV3,
    comparison_report_model_id,
    evaluation_report_model_id,
    run_report_model_v3_id,
)
from oamb.contracts.specifications import (
    ComparisonCostView,
    ReportSpec,
    SourceEvidenceBinding,
    SourceEvidenceKind,
)
from oamb.contracts.states import ValidationDisposition
from oamb.memory_systems.mem0 import Mem0ReferenceNegativeAdapter
from oamb.reporting.compare import (
    REQUIRED_COMPARISON_CONTROL_IDS,
    build_comparison_pair_binding,
    build_comparison_spec,
)
from oamb.reporting.public import (
    build_comparison_report_model,
    build_diagnostic_run_report_model,
    build_evaluation_report_model,
    build_run_report_model,
)
from oamb.reporting.roots import (
    build_evaluation_report_spec,
    build_report_spec,
)
from tests.reporting.test_t8_offline_renderer import (
    _run_report,
    build_claim_boundary,
    build_record_projections,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
COMMITTED_AT = datetime(2026, 8, 28, tzinfo=UTC)


def _publication() -> ModuleType:
    try:
        return importlib.import_module("oamb.reporting.publication")
    except ModuleNotFoundError:
        pytest.fail("generic T8 report publication is not implemented", pytrace=False)


def _renderer() -> ModuleType:
    return importlib.import_module("oamb.reporting.offline_renderer")


def _validation(
    *,
    profile_id: str = "oamb-t8-adapter-mem0-rest-v1",
) -> ValidationResult:
    production = validate_catalog_profile(
        "oamb-t8-adapter-mem0-rest-v1",
        _validation_target(),
    )
    return production.model_copy(update={"validation_profile_id": profile_id})


def _validation_target(*, resolve_attempts: int = 1) -> Mem0AdapterValidationInput:
    adapter = Mem0ReferenceNegativeAdapter()
    for _index in range(resolve_attempts):
        with pytest.raises(MemorySystemProfileUnsupported):
            asyncio.run(adapter.resolve())
    target = mem0_adapter_validation_input(adapter)
    asyncio.run(adapter.close())
    return target


def _model_and_spec(
    audience: Literal["local", "public"] = "public",
) -> tuple[RunReportModelV3, ReportSpec, SourceEvidenceBinding]:
    renderer = _renderer()
    validation = _validation()
    validation_hash = canonical_sha256(validation)
    validation_profile_hash = canonical_sha256(
        [
            "oamb-validation-profile-binding-v1",
            validation.validation_profile_id,
            validation.required_rule_ids,
            validation.implementation_versions,
        ]
    )
    source = SourceEvidenceBinding(
        binding_id=SHA_A,
        source_kind=SourceEvidenceKind.EXTERNAL,
        source_identity="run-publication-fixture",
        source_root_hash=validation.target_hash,
        validation_result_hash=validation_hash,
        source_schema_versions=("capsule_manifest@1",),
    )
    summary = CompletionSummaryV3(
        run_id="run-publication-fixture",
        intended_logical_contexts=1,
        intended_ingestion_plans=1,
        ready_ingestion_plans=1,
        intended_cases=1,
        terminal_cases=1,
        completed_cases=1,
        errored_cases=0,
        unsupported_cases=0,
        cancelled_cases=0,
        budget_exceeded_cases=0,
        parsed_cases=1,
        evaluated_cases=1,
        judged_cases=1,
        unjudged_cases=0,
        metric_eligible_cases=1,
    )
    model = build_run_report_model(
        report_spec_hash=SHA_A,
        source_binding=source,
        evidence_validation_profile_hash=validation_profile_hash,
        evidence_validation_result_hash=validation_hash,
        reducer_bindings=(("completion-v1", 1, SHA_D),),
        audience=audience,
        origin_kind="external",
        capsule_id=None,
        protocol_id="fixture-protocol-v1",
        workload_id="fixture-workload-v1",
        memory_system_id="fixture-memory-v1",
        claim_boundary=build_claim_boundary(rule_count=len(validation.required_rule_ids)),
        summary=summary,
        metric_summaries=(),
        measurement_lines=(),
        logical_context_ids=(SHA_A,),
        ingestion_occurrence_ids=(SHA_B,),
        case_occurrence_ids=(SHA_C,),
        attempt_ids=(SHA_D,),
        record_projections=build_record_projections(
            logical_ids=(SHA_A,),
            plan_ids=(SHA_B,),
            case_ids=(SHA_C,),
            attempt_ids=(SHA_D,),
        ),
        limitations=("fixture-only",),
    )
    css_hash, script_hash = renderer.offline_asset_hashes()
    spec = build_report_spec(
        report_kind="run",
        audience=audience,
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
        renderer_hash=renderer.offline_renderer_hash(),
        asset_hashes=(css_hash, script_hash),
        export_profile_selector_id=f"{audience}-run-v1",
        export_profile_selector_version=1,
    )
    model = _bind_run_model_to_spec(model, spec)
    return model, spec, source


def _bind_run_model_to_spec(model: RunReportModelV3, spec: ReportSpec) -> RunReportModelV3:
    model = model.model_copy(update={"report_spec_hash": canonical_sha256(spec)})
    fields = model.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "report_id"},
    )
    return type(model).model_validate(
        {
            **model.model_dump(mode="python"),
            "report_id": importlib.import_module("oamb.contracts.reporting").run_report_model_v3_id(
                **fields
            ),
        }
    )


def _evaluation_model_spec_and_sources() -> tuple[
    Any,
    Any,
    tuple[SourceEvidenceBinding, ...],
    tuple[ValidationResult, ...],
    tuple[Any, ...],
]:
    renderer = _renderer()
    spec = build_evaluation_report_spec(
        audience="public",
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("phase", "runs", "comparisons", "cases", "limitations"),
        renderer_hash=renderer.offline_renderer_hash(),
        asset_hashes=renderer.offline_asset_hashes(),
        export_profile_selector_id="public-evaluation-v1",
        export_profile_selector_version=1,
    )
    spec_hash = canonical_sha256(spec)
    base = _run_report()
    targets = tuple(_validation_target(resolve_attempts=index) for index in range(1, 5))
    validations = tuple(
        validate_catalog_profile("oamb-t8-adapter-mem0-rest-v1", target) for target in targets
    )
    validation_profile_hash = canonical_sha256(
        [
            "oamb-validation-profile-binding-v1",
            validations[0].validation_profile_id,
            validations[0].required_rule_ids,
            validations[0].implementation_versions,
        ]
    )
    runs = []
    run_case_manifest_ids = (SHA_A, SHA_A, SHA_B, SHA_B)
    for index, character in enumerate("abcd"):
        validation = validations[index]
        validation_hash = canonical_sha256(validation)
        case_occurrence_id = hashlib.sha256(f"case-occurrence-{character}".encode()).hexdigest()
        source = SourceEvidenceBinding(
            binding_id=hashlib.sha256(f"binding-{character}".encode()).hexdigest(),
            source_kind=SourceEvidenceKind.RUN,
            source_identity=f"evaluation-run-{character}",
            source_root_hash=validation.target_hash,
            validation_result_hash=validation_hash,
            source_schema_versions=("capsule_manifest@1",),
        )
        payload = base.model_dump(mode="python")
        payload["summary"]["run_id"] = f"evaluation-run-{character}"
        payload["report_spec_hash"] = spec_hash
        payload["ordered_source_bindings"] = (source.model_dump(mode="python"),)
        payload["evidence_validation_profile_hash"] = validation_profile_hash
        payload["evidence_validation_result_hash"] = validation_hash
        payload["claim_boundary"] = build_claim_boundary(
            rule_count=len(validation.required_rule_ids)
        ).model_dump(mode="python")
        payload["case_occurrence_ids"] = (case_occurrence_id,)
        case_projection = next(
            item for item in payload["record_projections"] if item["axis"] == "case"
        )
        case_projection["record_id"] = case_occurrence_id
        case_projection["detail_items"] = (
            ("case_manifest_entry_id", run_case_manifest_ids[index]),
            *tuple(case_projection["detail_items"]),
        )
        fields = {
            key: value
            for key, value in payload.items()
            if key not in {"schema_name", "schema_version", "report_id"}
        }
        payload["report_id"] = run_report_model_v3_id(**fields)
        runs.append(RunReportModelV3.model_validate(payload))
    controls = tuple(
        ComparisonControlBinding(control_id=control_id, value_hash=SHA_C)
        for control_id in REQUIRED_COMPARISON_CONTROL_IDS
    )
    left = ComparisonControlSnapshot(
        run_id=runs[0].summary.run_id,
        source_root_hash=runs[0].ordered_source_bindings[0].source_root_hash,
        memory_system_id="memory-left",
        provider_native_profile_hash=SHA_C,
        controls=controls,
    )
    right = ComparisonControlSnapshot(
        run_id=runs[1].summary.run_id,
        source_root_hash=runs[1].ordered_source_bindings[0].source_root_hash,
        memory_system_id="memory-right",
        provider_native_profile_hash=SHA_D,
        controls=controls,
    )
    pair = build_comparison_pair_binding(
        case_manifest_entry_id=SHA_A,
        left_case_occurrence_id=runs[0].case_occurrence_ids[0],
        right_case_occurrence_id=runs[1].case_occurrence_ids[0],
        metric_id="fixture-exact-v1",
    )
    spec_target = build_comparison_spec(
        left_run_id=left.run_id,
        right_run_id=right.run_id,
        cost_view=ComparisonCostView.NONE,
        ordered_pair_bindings=(pair,),
        required_control_ids=REQUIRED_COMPARISON_CONTROL_IDS,
        comparison_policy_hash=SHA_D,
        winner_reducer=None,
        cost_control=None,
    )
    delta = PairedMetricDelta(
        case_manifest_entry_id=SHA_A,
        left_case_occurrence_id=runs[0].case_occurrence_ids[0],
        right_case_occurrence_id=runs[1].case_occurrence_ids[0],
        metric_id="fixture-exact-v1",
        left=ExactRational(numerator=1, denominator=1),
        right=ExactRational(numerator=0, denominator=1),
        signed_delta=ExactRational(numerator=1, denominator=1),
        absolute_delta=ExactRational(numerator=1, denominator=1),
    )
    comparison_target = comparison_validation_input(
        spec=spec_target,
        left=left,
        right=right,
        paired_metric_deltas=(delta,),
    )
    comparison_validation = validate_catalog_profile(
        "oamb-t8-comparison-paired-native-v1", comparison_target
    )
    comparison = build_comparison_report_model(
        report_spec_hash=spec_hash,
        ordered_source_bindings=(
            runs[0].ordered_source_bindings[0],
            runs[1].ordered_source_bindings[0],
        ),
        ordered_evidence_validation_hashes=(
            runs[0].evidence_validation_result_hash,
            runs[1].evidence_validation_result_hash,
        ),
        left_run_report_hash=canonical_sha256(runs[0]),
        right_run_report_hash=canonical_sha256(runs[1]),
        claim_boundary=build_claim_boundary(
            rule_count=len(comparison_validation.required_rule_ids)
        ),
        comparison=comparison_target.report,
        limitations=("fixture comparison",),
    )
    model = build_evaluation_report_model(
        phase_id="fixture-evaluation",
        report_spec_hash=spec_hash,
        ordered_run_models=(runs[0], runs[1], runs[2], runs[3]),
        eligible_comparison_models=(comparison,),
        unique_case_count=2,
        limitations=("fixture only",),
    )
    comparison_source = SourceEvidenceBinding(
        binding_id=hashlib.sha256(b"comparison-source-binding").hexdigest(),
        source_kind=SourceEvidenceKind.DERIVATION,
        source_identity=comparison.report_id,
        source_root_hash=comparison_validation.target_hash,
        validation_result_hash=canonical_sha256(comparison_validation),
        source_schema_versions=("comparison_report_model@1",),
    )
    sources = (
        *(run.ordered_source_bindings[0] for run in runs),
        comparison_source,
    )
    return (
        model,
        spec,
        sources,
        (*validations, comparison_validation),
        (*targets, comparison_target),
    )


def test_evaluation_publication_uses_v2_v2_v3_v3_and_offline_html(tmp_path: Path) -> None:
    publication = _publication()
    model, spec, sources, validations, validation_targets = _evaluation_model_spec_and_sources()

    built = publication.build_report_derivation(
        model=model,
        report_spec=spec,
        ordered_source_bindings=sources,
        evidence_validations=validations,
        evidence_validation_targets=validation_targets,
        transform_spec_hash=SHA_D,
        schema_versions=("evaluation_report_model@1", "report_artifact_manifest@3"),
        output_root=tmp_path,
        committed_at=COMMITTED_AT,
    )

    derivation = (built.final_directory / "derivation-spec.json").read_text()
    artifact = (built.final_directory / "report-artifact-manifest.json").read_text()
    html = built.report_path.read_text()
    assert '"schema_version":3' in derivation
    assert '"schema_version":3' in artifact
    assert '<meta name="color-scheme" content="light dark">' in html
    assert "https://" not in html and "http://" not in html


def test_evaluation_export_rejects_self_hashed_comparison_with_foreign_run_hashes(
    tmp_path: Path,
) -> None:
    publication = _publication()
    model, spec, sources, validations, validation_targets = _evaluation_model_spec_and_sources()
    comparison_payload = model.eligible_comparison_models[0].model_dump(mode="python")
    comparison_payload["left_run_report_hash"] = hashlib.sha256(b"foreign-left").hexdigest()
    comparison_payload["right_run_report_hash"] = hashlib.sha256(b"foreign-right").hexdigest()
    comparison_fields = {
        key: value
        for key, value in comparison_payload.items()
        if key not in {"schema_name", "schema_version", "report_id"}
    }
    comparison_payload["report_id"] = comparison_report_model_id(**comparison_fields)
    foreign_comparison = type(model.eligible_comparison_models[0]).model_validate(
        comparison_payload
    )
    foreign_model = build_evaluation_report_model(
        phase_id=model.phase_id,
        report_spec_hash=model.report_spec_hash,
        ordered_run_models=model.ordered_run_models,
        eligible_comparison_models=(foreign_comparison,),
        unique_case_count=model.unique_case_count,
        limitations=model.limitations,
    )

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=foreign_model,
            report_spec=spec,
            ordered_source_bindings=sources,
            evidence_validations=validations,
            evidence_validation_targets=validation_targets,
            transform_spec_hash=SHA_D,
            schema_versions=("evaluation_report_model@1", "report_artifact_manifest@3"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}


def test_evaluation_export_requires_exact_run_and_comparison_source_validation_closure(
    tmp_path: Path,
) -> None:
    publication = _publication()
    model, spec, sources, validations, validation_targets = _evaluation_model_spec_and_sources()

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=sources[:-1],
            evidence_validations=validations[:-1],
            evidence_validation_targets=validation_targets[:-1],
            transform_spec_hash=SHA_D,
            schema_versions=("evaluation_report_model@1", "report_artifact_manifest@3"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}


def test_evaluation_export_rederives_unique_case_coverage_from_display_projections(
    tmp_path: Path,
) -> None:
    publication = _publication()
    model, spec, sources, validations, validation_targets = _evaluation_model_spec_and_sources()
    payload = model.model_dump(mode="python")
    payload["unique_case_count"] = 1
    fields = {
        key: value
        for key, value in payload.items()
        if key not in {"schema_name", "schema_version", "report_id"}
    }
    payload["report_id"] = evaluation_report_model_id(**fields)
    forged_model = type(model).model_validate(payload)

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=forged_model,
            report_spec=spec,
            ordered_source_bindings=sources,
            evidence_validations=validations,
            evidence_validation_targets=validation_targets,
            transform_spec_hash=SHA_D,
            schema_versions=("evaluation_report_model@1", "report_artifact_manifest@3"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}


def test_evaluation_export_rejects_duplicate_system_case_display_coverage(
    tmp_path: Path,
) -> None:
    publication = _publication()
    model, spec, sources, validations, validation_targets = _evaluation_model_spec_and_sources()
    duplicated_run_payload = model.ordered_run_models[3].model_dump(mode="python")
    duplicated_case_id = model.ordered_run_models[2].case_occurrence_ids[0]
    duplicated_run_payload["case_occurrence_ids"] = (duplicated_case_id,)
    case_projection = next(
        item for item in duplicated_run_payload["record_projections"] if item["axis"] == "case"
    )
    case_projection["record_id"] = duplicated_case_id
    run_fields = {
        key: value
        for key, value in duplicated_run_payload.items()
        if key not in {"schema_name", "schema_version", "report_id"}
    }
    duplicated_run_payload["report_id"] = run_report_model_v3_id(**run_fields)
    duplicated_run = RunReportModelV3.model_validate(duplicated_run_payload)
    duplicated_model = build_evaluation_report_model(
        phase_id=model.phase_id,
        report_spec_hash=model.report_spec_hash,
        ordered_run_models=(
            model.ordered_run_models[0],
            model.ordered_run_models[1],
            model.ordered_run_models[2],
            duplicated_run,
        ),
        eligible_comparison_models=model.eligible_comparison_models,
        unique_case_count=model.unique_case_count,
        limitations=model.limitations,
    )

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=duplicated_model,
            report_spec=spec,
            ordered_source_bindings=sources,
            evidence_validations=validations,
            evidence_validation_targets=validation_targets,
            transform_spec_hash=SHA_D,
            schema_versions=("evaluation_report_model@1", "report_artifact_manifest@3"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}


def test_external_run_publication_fails_closed_until_the_t9_importer(
    tmp_path: Path,
) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}
    assert not (tmp_path / "derivations").exists()


def test_public_export_failure_is_retained_only_as_an_attempt(tmp_path: Path) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
            extra_publication_payloads={
                "copied/operator-note.txt": b"credential sk-fixture-secret-must-not-publish"
            },
        )

    error = captured.value
    assert error.attempt_directory.is_dir()
    assert (error.attempt_directory / "export-validation.json").is_file()
    assert not (error.attempt_directory / "derived-manifest.json").exists()
    assert not (tmp_path / "derivations").exists()
    assert "public-sensitive-content" in {issue.code for issue in error.result.issues}


def test_export_scan_inventory_cannot_omit_a_publication_payload(tmp_path: Path) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
            omitted_scan_paths=frozenset({"outputs/report-model.json"}),
        )

    assert "publication-scan-coverage-mismatch" in {
        issue.code for issue in captured.value.result.issues
    }


def test_export_rejects_unfrozen_selector_and_schema_inventory(tmp_path: Path) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()
    bad_spec = build_report_spec(
        report_kind="run",
        audience=spec.audience,
        preview_max_field_bytes=spec.preview_max_field_bytes,
        preview_total_bytes=spec.preview_total_bytes,
        display_field_ids=spec.display_field_ids,
        renderer_hash=spec.renderer_hash,
        asset_hashes=spec.asset_hashes,
        browser_contract_hash=spec.browser_contract_hash,
        performance_contract_hash=spec.performance_contract_hash,
        export_profile_selector_id="attacker-controlled",
        export_profile_selector_version=999,
    )
    bad_model = _bind_run_model_to_spec(model, bad_spec)

    with pytest.raises(publication.ReportExportError) as selector:
        publication.build_report_derivation(
            model=bad_model,
            report_spec=bad_spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path / "selector",
            committed_at=COMMITTED_AT,
        )
    with pytest.raises(publication.ReportExportError) as schema:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("garbage@999",),
            output_root=tmp_path / "schema",
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in selector.value.result.issues}
    assert "report-binding-mismatch" in {issue.code for issue in schema.value.result.issues}


def test_report_spec_builder_rejects_planted_renderer_and_acceptance_hashes() -> None:
    _model, spec, _source = _model_and_spec()

    def build_with_hashes(
        *,
        renderer_hash: str = spec.renderer_hash,
        browser_contract_hash: str = spec.browser_contract_hash,
        performance_contract_hash: str = spec.performance_contract_hash,
    ) -> ReportSpec:
        return build_report_spec(
            report_kind="run",
            audience=spec.audience,
            preview_max_field_bytes=spec.preview_max_field_bytes,
            preview_total_bytes=spec.preview_total_bytes,
            display_field_ids=spec.display_field_ids,
            renderer_hash=renderer_hash,
            asset_hashes=spec.asset_hashes,
            browser_contract_hash=browser_contract_hash,
            performance_contract_hash=performance_contract_hash,
            export_profile_selector_id="public-run-v1",
            export_profile_selector_version=1,
        )

    with pytest.raises(ValueError, match="renderer identity"):
        build_with_hashes(renderer_hash=SHA_A)
    with pytest.raises(ValueError, match="browser contract"):
        build_with_hashes(browser_contract_hash=SHA_A)
    with pytest.raises(ValueError, match="performance contract"):
        build_with_hashes(performance_contract_hash=SHA_A)


def test_export_rejects_network_active_markup_in_any_copied_payload(tmp_path: Path) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
            extra_publication_payloads={
                "copied/active.html": (
                    b'<script src="https://attacker.invalid/payload.js"></script>'
                )
            },
        )

    assert "network-active-publication-payload" in {
        issue.code for issue in captured.value.result.issues
    }
    assert not (tmp_path / "derivations").exists()


@pytest.mark.parametrize(
    "bidi_control",
    tuple(chr(codepoint) for codepoint in (*range(0x202A, 0x202F), *range(0x2066, 0x206A)))
    + ("\u200e", "\u200f"),
)
def test_public_export_rejects_every_frozen_bidi_control(
    tmp_path: Path,
    bidi_control: str,
) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
            extra_publication_payloads={"copied/note.txt": f"safe{bidi_control}spoof".encode()},
        )

    assert "public-sensitive-content" in {issue.code for issue in captured.value.result.issues}


def test_export_rejects_an_arbitrary_shape_valid_upstream_validation(tmp_path: Path) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()
    arbitrary = _validation(profile_id="arbitrary-always-pass-v1")
    source = source.model_copy(update={"validation_result_hash": canonical_sha256(arbitrary)})
    model = model.model_copy(update={"ordered_source_bindings": (source,)})

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(arbitrary,),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}


def test_export_replays_upstream_validation_against_the_supplied_source_target(
    tmp_path: Path,
) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()
    original_target = _validation_target()
    first_event = original_target.audit.ordered_events[0]
    invalid_target = replace(
        original_target,
        audit=replace(
            original_target.audit,
            ordered_events=(
                replace(first_event, dispatcher_count_after=1),
                *original_target.audit.ordered_events[1:],
            ),
        ),
    )

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(invalid_target,),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}


def test_export_rejects_display_previews_outside_the_bound_report_spec(
    tmp_path: Path,
) -> None:
    publication = _publication()
    model, spec, source = _model_and_spec()
    oversized_text = "x" * (spec.preview_max_field_bytes + 1)
    encoded = oversized_text.encode()
    oversized = DisplayPreview(
        text=oversized_text,
        shown_bytes=len(encoded),
        total_bytes=len(encoded),
        sha256=hashlib.sha256(encoded).hexdigest(),
        media_type="application/json",
        truncated=False,
        source_reference="source/raw/oversized.json",
        limitation=None,
    )
    projections = tuple(
        projection.model_copy(update={"display_previews": (oversized,)})
        if projection.axis == "case"
        else projection
        for projection in model.record_projections
    )
    oversized_model = _bind_run_model_to_spec(
        model.model_copy(update={"record_projections": projections}),
        spec,
    )

    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(
            model=oversized_model,
            report_spec=spec,
            ordered_source_bindings=(source,),
            evidence_validations=(_validation(),),
            evidence_validation_targets=(_validation_target(),),
            transform_spec_hash=SHA_A,
            schema_versions=("run_report_model@3", "report_artifact_manifest@2"),
            output_root=tmp_path,
            committed_at=COMMITTED_AT,
        )

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}


def test_external_diagnostic_publication_fails_closed_until_the_t9_importer(
    tmp_path: Path,
) -> None:
    publication = _publication()
    target = _validation_target()
    first_event = target.audit.ordered_events[0]
    invalid_target = replace(
        target,
        audit=replace(
            target.audit,
            ordered_events=(
                replace(first_event, dispatcher_count_after=1),
                *target.audit.ordered_events[1:],
            ),
        ),
    )
    validation = validate_catalog_profile("oamb-t8-adapter-mem0-rest-v1", invalid_target)
    assert validation.disposition == ValidationDisposition.INVALID
    validation_hash = canonical_sha256(validation)
    source = SourceEvidenceBinding(
        binding_id=SHA_A,
        source_kind=SourceEvidenceKind.EXTERNAL,
        source_identity="invalid-diagnostic-fixture",
        source_root_hash=validation.target_hash,
        validation_result_hash=validation_hash,
        source_schema_versions=("mem0_zero_dispatch_audit@1",),
    )
    css_hash, script_hash = _renderer().offline_asset_hashes()
    spec = build_report_spec(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "validation", "limitations"),
        renderer_hash=_renderer().offline_renderer_hash(),
        asset_hashes=(css_hash, script_hash),
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )
    profile_hash = canonical_sha256(
        [
            "oamb-validation-profile-binding-v1",
            validation.validation_profile_id,
            validation.required_rule_ids,
            validation.implementation_versions,
        ]
    )
    with pytest.raises(ValueError, match="run identity"):
        build_diagnostic_run_report_model(
            report_spec_hash=canonical_sha256(spec),
            source_binding=source,
            evidence_validation_profile_hash=profile_hash,
            evidence_validation_result_hash=validation_hash,
            origin_kind="external",
            run_id="spoofed-unrelated-run",
            validation_issue_codes=tuple(dict.fromkeys(issue.code for issue in validation.issues)),
            limitations=("diagnostic-only",),
        )
    model = build_diagnostic_run_report_model(
        report_spec_hash=canonical_sha256(spec),
        source_binding=source,
        evidence_validation_profile_hash=profile_hash,
        evidence_validation_result_hash=validation_hash,
        origin_kind="external",
        run_id="invalid-diagnostic-fixture",
        validation_issue_codes=tuple(dict.fromkeys(issue.code for issue in validation.issues)),
        limitations=("diagnostic-only; no benchmark quality or cost claims",),
    )
    arguments = {
        "model": model,
        "report_spec": spec,
        "ordered_source_bindings": (source,),
        "evidence_validations": (validation,),
        "evidence_validation_targets": (invalid_target,),
        "transform_spec_hash": SHA_A,
        "schema_versions": (
            "diagnostic_run_report_model@1",
            "report_artifact_manifest@2",
        ),
        "output_root": tmp_path,
        "committed_at": COMMITTED_AT,
    }

    with pytest.raises(ValueError, match="explicit opt-in"):
        publication.build_report_derivation(**arguments)
    with pytest.raises(publication.ReportExportError) as captured:
        publication.build_report_derivation(**arguments, diagnostic=True)

    assert "report-binding-mismatch" in {issue.code for issue in captured.value.result.issues}
    assert not (tmp_path / "derivations").exists()
