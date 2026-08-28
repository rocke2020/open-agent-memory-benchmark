"""Credential-free phase review and acceptance CLI composition."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from oamb import phase_cli_io as _phase_io
from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
from oamb.artifacts.validation.phase import (
    PhaseGateValidationInput,
    phase_ai_review_evidence_closes,
    validate_t10_phase_gate,
)
from oamb.contracts.accounting import CostMeasurementSpec, PriceSnapshot
from oamb.contracts.evidence import PhaseReviewOccurrenceRecordV2
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    AIReviewCaseProjection,
    AIReviewIntegrityProjection,
    EvaluationPhaseGate,
    EvaluationReviewBundle,
    HumanQualityReviewRecord,
    QualityReviewStatus,
)
from oamb.contracts.specifications import (
    AIReviewPlan,
    BudgetSpecV2,
    ExecutionEnvironmentBinding,
    ExternalCallApprovalRecord,
    HumanReviewKeyBinding,
    ModelRoleBindingV2,
    SourceEvidenceBinding,
    SourceEvidenceKind,
)
from oamb.contracts.states import ValidationDisposition
from oamb.phase_review_profiles import PHASE_REVIEW_CLIENT_KINDS
from oamb.reporting.human_review import (
    derive_evaluation_phase_gate,
    import_human_review_decision,
    prepare_human_review_decision,
)
from oamb.reporting.review import (
    build_ai_review_plan,
    build_evaluation_review_bundle,
    parse_ai_review_batch_output,
    parse_ai_review_integrity_output,
    reduce_ai_quality_review,
)
from oamb.workloads.visible_evidence import o200k_encoding, tokenizer_fingerprint

if TYPE_CHECKING:
    from oamb.phase_review_run import PartialPhaseReviewRunEvidence

phase_app = typer.Typer(
    help="Build and validate immutable evaluation-phase review artifacts offline.",
    no_args_is_help=True,
)
bundle_app = typer.Typer(help="Build a canonical evaluation review bundle.")
ai_review_app = typer.Typer(
    help="Plan, execute an approved reviewer, or reduce recorded review outputs.",
    no_args_is_help=True,
)
human_review_app = typer.Typer(
    help="Prepare or verify externally signed human decisions; OAMB never signs.",
    no_args_is_help=True,
)
gate_app = typer.Typer(help="Build or validate the immutable T10 phase gate.")
acceptance_report_app = typer.Typer(help="Build an export-validated acceptance report.")
phase_app.add_typer(bundle_app, name="bundle")
phase_app.add_typer(ai_review_app, name="ai-review")
phase_app.add_typer(human_review_app, name="human-review")
phase_app.add_typer(gate_app, name="gate")
phase_app.add_typer(acceptance_report_app, name="acceptance-report")

_BUNDLE_INPUT_FIELDS = {
    "phase_id",
    "ordered_capsule_hashes",
    "ordered_validation_hashes",
    "ordered_case_occurrence_ids",
    "unique_case_manifest_entry_ids",
    "ordinary_derivation_hashes",
    "report_model_hash",
    "report_html_hash",
    "export_validation_hash",
    "limitations",
}
_AI_REVIEW_CONFIG_FIELDS = {
    "projection_spec_hash",
    "prompt_pack_hash",
    "output_contract_hash",
    "parser_hash",
    "reviewer_role_binding_hash",
    "reviewer_model_hash",
    "reviewer_runtime_hash",
    "reviewer_configuration_hash",
    "model_context_window_tokens",
    "aggregate_version",
}
_AI_REDUCTION_INPUT_FIELDS = {
    "occurrence_id",
    "ordinal",
    "previous_ai_review_record_hash",
    "previous_history_root_hash",
    "attempt_ids",
    "usage_record_ids",
    "resource_record_ids",
    "cost_record_ids",
    "accounting_closed",
    "created_at",
}
_PHASE_ACCEPTANCE_TRANSFORM_VERSION = "oamb-phase-acceptance-report-cli-v1"
_PHASE_ACCEPTANCE_SCHEMAS = (
    "phase_acceptance_report@1",
    "report_artifact_manifest@2",
)
_BUNDLE_INPUT_HELP = (
    "Canonical JSON keys: phase_id, ordered_capsule_hashes, ordered_validation_hashes, "
    "ordered_case_occurrence_ids, unique_case_manifest_entry_ids, "
    "ordinary_derivation_hashes, report_model_hash, report_html_hash, "
    "export_validation_hash, limitations."
)
_AI_REVIEW_CONFIG_HELP = (
    "Canonical JSON keys: projection_spec_hash, prompt_pack_hash, output_contract_hash, "
    "parser_hash, reviewer_role_binding_hash, reviewer_model_hash, "
    "reviewer_runtime_hash, reviewer_configuration_hash, model_context_window_tokens, "
    "aggregate_version. Token counts use the repository-pinned o200k_base counter."
)
_AI_REDUCTION_INPUT_HELP = (
    "Canonical JSON keys: occurrence_id, ordinal, previous_ai_review_record_hash, "
    "previous_history_root_hash, attempt_ids, usage_record_ids, resource_record_ids, "
    "cost_record_ids, accounting_closed, created_at."
)
_PHASE_REVIEW_EVIDENCE_HELP = (
    "Directory containing canonical plan.json, role-binding.json, occurrence.json, "
    "approval.json, budget.json, cost-measurement-spec.json, execution-environment.json, "
    "optional live-only price-snapshot.json, batch-results.json, integrity-result.json, "
    "ai-history.json, attempts.json, usage-records.json, resource-records.json, "
    "cost-records.json, human-key-binding.json, human-decision.json, and human-signature.txt."
)


@bundle_app.command("build")
def bundle_build(
    input_path: Annotated[
        Path,
        typer.Option("--input", help=_BUNDLE_INPUT_HELP),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only EvaluationReviewBundle JSON path."),
    ],
) -> None:
    """Build one canonical review bundle without reading credentials or providers."""

    try:
        fields = _phase_io.load_exact_object(input_path, _BUNDLE_INPUT_FIELDS)
        bundle = build_evaluation_review_bundle(
            phase_id=_phase_io.string_value(fields, "phase_id"),
            ordered_capsule_hashes=_phase_io.string_tuple(fields, "ordered_capsule_hashes"),
            ordered_validation_hashes=_phase_io.string_tuple(fields, "ordered_validation_hashes"),
            ordered_case_occurrence_ids=_phase_io.string_tuple(
                fields, "ordered_case_occurrence_ids"
            ),
            unique_case_manifest_entry_ids=_phase_io.string_tuple(
                fields, "unique_case_manifest_entry_ids"
            ),
            ordinary_derivation_hashes=_phase_io.string_tuple(fields, "ordinary_derivation_hashes"),
            report_model_hash=_phase_io.string_value(fields, "report_model_hash"),
            report_html_hash=_phase_io.string_value(fields, "report_html_hash"),
            export_validation_hash=_phase_io.string_value(fields, "export_validation_hash"),
            limitations=_phase_io.string_tuple(fields, "limitations"),
        )
        _phase_io.write_contract(output, bundle)
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"review bundle: {output}")


@ai_review_app.command("plan")
def ai_review_plan(
    bundle_path: Annotated[Path, typer.Option("--bundle", help="EvaluationReviewBundle JSON.")],
    case_projections: Annotated[
        Path,
        typer.Option("--case-projections", help="Canonical array of case projections."),
    ],
    integrity_projection: Annotated[
        Path,
        typer.Option("--integrity-projection", help="AIReviewIntegrityProjection JSON."),
    ],
    review_config: Annotated[
        Path,
        typer.Option("--review-config", help=_AI_REVIEW_CONFIG_HELP),
    ],
    output_directory: Annotated[
        Path,
        typer.Option(
            "--output-directory",
            help="Create-only directory for plan.json and exact request payloads.",
        ),
    ],
) -> None:
    """Freeze a bounded AI review plan and request bytes; make no model call."""

    try:
        bundle = _phase_io.load_contract(bundle_path, EvaluationReviewBundle)
        projections = _phase_io.load_contract_sequence(case_projections, AIReviewCaseProjection)
        integrity = _phase_io.load_contract(integrity_projection, AIReviewIntegrityProjection)
        config = _phase_io.load_exact_object(review_config, _AI_REVIEW_CONFIG_FIELDS)
        build = build_ai_review_plan(
            bundle,
            projections,
            integrity,
            projection_spec_hash=_phase_io.string_value(config, "projection_spec_hash"),
            prompt_pack_hash=_phase_io.string_value(config, "prompt_pack_hash"),
            output_contract_hash=_phase_io.string_value(config, "output_contract_hash"),
            parser_hash=_phase_io.string_value(config, "parser_hash"),
            reviewer_role_binding_hash=_phase_io.string_value(config, "reviewer_role_binding_hash"),
            reviewer_model_hash=_phase_io.string_value(config, "reviewer_model_hash"),
            reviewer_runtime_hash=_phase_io.string_value(config, "reviewer_runtime_hash"),
            reviewer_configuration_hash=_phase_io.string_value(
                config, "reviewer_configuration_hash"
            ),
            token_counter=_count_review_tokens,
            counter_fingerprint=tokenizer_fingerprint(),
            model_context_window_tokens=_phase_io.integer_value(
                config, "model_context_window_tokens"
            ),
            aggregate_version=_phase_io.string_value(config, "aggregate_version"),
        )
        _phase_io.write_contract(
            output_directory / "plan.json", build.plan, trusted_root=output_directory
        )
        for index, request in enumerate(build.case_requests, 1):
            atomic_write_bytes(
                output_directory / "case-requests" / f"{index:04d}.json",
                request.encode("utf-8"),
                trusted_root=output_directory,
            )
        atomic_write_bytes(
            output_directory / "integrity-request.json",
            build.integrity_request.encode("utf-8"),
            trusted_root=output_directory,
        )
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"AI review plan: {output_directory / 'plan.json'}")
    typer.echo("status: planned offline; no model call was made")


@ai_review_app.command("run")
def ai_review_run(
    client: Annotated[
        str,
        typer.Option(
            "--client",
            help="Model client profile: fake or openai-compatible.",
        ),
    ],
    bundle_path: Annotated[Path, typer.Option("--bundle", help="EvaluationReviewBundle JSON.")],
    plan_path: Annotated[Path, typer.Option("--plan", help="Canonical AIReviewPlan JSON.")],
    occurrence_path: Annotated[
        Path,
        typer.Option("--occurrence", help="Planned PhaseReviewOccurrenceRecord v2 JSON."),
    ],
    approval_path: Annotated[
        Path,
        typer.Option("--approval", help="Exact phase-review approval record JSON."),
    ],
    budget_path: Annotated[
        Path,
        typer.Option("--budget", help="Exact occurrence-scoped BudgetSpec v2 JSON."),
    ],
    role_binding_path: Annotated[
        Path,
        typer.Option("--role-binding", help="Selected quality-review ModelRoleBinding v2 JSON."),
    ],
    cost_measurement_spec_path: Annotated[
        Path,
        typer.Option(
            "--cost-measurement-spec",
            help="Phase-review CostMeasurementSpec JSON.",
        ),
    ],
    execution_environment_path: Annotated[
        Path,
        typer.Option(
            "--execution-environment",
            help="Exact ExecutionEnvironmentBinding JSON.",
        ),
    ],
    request_directory: Annotated[
        Path,
        typer.Option(
            "--request-directory",
            help="Create-only plan output containing case-requests/ and integrity-request.json.",
        ),
    ],
    artifact_repository: Annotated[
        Path,
        typer.Option(
            "--artifact-repository",
            help="Bound repository containing the derived phase-review occurrence root.",
        ),
    ],
    output_directory: Annotated[
        Path,
        typer.Option("--output-directory", help="New create-only review result directory."),
    ],
    started_at: Annotated[
        str | None,
        typer.Option(
            "--started-at",
            help="Fake only: deterministic start with explicit UTC offset.",
        ),
    ] = None,
    ended_at: Annotated[
        str | None,
        typer.Option(
            "--ended-at",
            help="Fake only: deterministic end with explicit UTC offset.",
        ),
    ] = None,
    fake_responses: Annotated[
        list[Path] | None,
        typer.Option(
            "--fake-response",
            help="Fake only: recorded response, once per planned attempt in plan order.",
        ),
    ] = None,
    price_snapshot_path: Annotated[
        Path | None,
        typer.Option(
            "--price-snapshot",
            help="OpenAI-compatible only: exact measured-usage price snapshot JSON.",
        ),
    ] = None,
    previous_ai_history_path: Annotated[
        Path | None,
        typer.Option(
            "--previous-ai-history",
            help="Required for ordinal 2+: complete canonical AI history JSON array.",
        ),
    ] = None,
) -> None:
    """Run one approved reviewer and retain its full attempt/accounting evidence."""

    from oamb.phase_review_run import PhaseReviewRunFailure, run_phase_review

    try:
        if client not in PHASE_REVIEW_CLIENT_KINDS:
            raise ValueError("unknown phase AI review client profile")
        if output_directory.exists():
            raise ValueError("phase AI review output directory already exists")
        bundle = _phase_io.load_contract(bundle_path, EvaluationReviewBundle)
        plan = _phase_io.load_contract(plan_path, AIReviewPlan)
        occurrence = _phase_io.load_contract(occurrence_path, PhaseReviewOccurrenceRecordV2)
        approval = _phase_io.load_contract(approval_path, ExternalCallApprovalRecord)
        budget = _phase_io.load_contract(budget_path, BudgetSpecV2)
        role = _phase_io.load_contract(role_binding_path, ModelRoleBindingV2)
        measurement = _phase_io.load_contract(cost_measurement_spec_path, CostMeasurementSpec)
        environment = _phase_io.load_contract(
            execution_environment_path, ExecutionEnvironmentBinding
        )
        price_snapshot = (
            None
            if price_snapshot_path is None
            else _phase_io.load_contract(price_snapshot_path, PriceSnapshot)
        )
        previous_ai_history = (
            ()
            if previous_ai_history_path is None
            else _phase_io.load_contract_sequence(
                previous_ai_history_path,
                AIQualityReviewRecord,
            )
        )
        result = run_phase_review(
            client_kind=client,
            bundle=bundle,
            plan=plan,
            occurrence=occurrence,
            approval=approval,
            budget=budget,
            role=role,
            cost_measurement_spec=measurement,
            execution_environment=environment,
            request_directory=request_directory,
            fake_responses=tuple(_phase_io.read_utf8(path) for path in fake_responses or ()),
            artifact_repository=artifact_repository,
            started_at=(None if started_at is None else _phase_io.parse_utc_datetime(started_at)),
            ended_at=None if ended_at is None else _phase_io.parse_utc_datetime(ended_at),
            price_snapshot=price_snapshot,
            previous_ai_history=previous_ai_history,
        )
        _write_phase_review_inputs(
            output_directory,
            plan=plan,
            role=role,
            approval=approval,
            budget=budget,
            cost_measurement_spec=measurement,
            execution_environment=environment,
            price_snapshot=price_snapshot,
        )
        _phase_io.write_contract(output_directory / "lease-record.json", result.lease_record)
        _write_contract_array(
            output_directory / "occurrence-claims.json",
            result.occurrence_claims,
            output_directory,
        )
        _write_contract_array(
            output_directory / "budget-reservations.json",
            result.budget_reservations,
            output_directory,
        )
        _write_contract_array(
            output_directory / "attempt-intents.json",
            result.attempt_intents,
            output_directory,
        )
        _write_contract_array(
            output_directory / "attempt-receipts.json",
            result.attempt_receipts,
            output_directory,
        )
        _phase_io.write_contract(output_directory / "occurrence.json", result.occurrence)
        _write_contract_array(
            output_directory / "batch-results.json", result.batch_results, output_directory
        )
        _phase_io.write_contract(
            output_directory / "integrity-result.json", result.integrity_result
        )
        _write_contract_array(output_directory / "attempts.json", result.attempts, output_directory)
        _write_contract_array(
            output_directory / "usage-records.json", result.usage_records, output_directory
        )
        _write_contract_array(
            output_directory / "resource-records.json", result.resource_records, output_directory
        )
        _write_contract_array(
            output_directory / "cost-records.json", result.cost_records, output_directory
        )
        _phase_io.write_contract(output_directory / "ai-review-record.json", result.ai_record)
        _write_contract_array(
            output_directory / "ai-history.json",
            (*previous_ai_history, result.ai_record),
            output_directory,
        )
    except PhaseReviewRunFailure as exc:
        _write_phase_review_inputs(
            output_directory,
            plan=plan,
            role=role,
            approval=approval,
            budget=budget,
            cost_measurement_spec=measurement,
            execution_environment=environment,
            price_snapshot=price_snapshot,
        )
        _write_partial_phase_review(output_directory, exc.partial)
        if previous_ai_history:
            _write_contract_array(
                output_directory / "ai-history.json",
                previous_ai_history,
                output_directory,
            )
        typer.echo(f"partial phase-review evidence: {output_directory}")
        _raise_cli_error(exc)
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"AI review record: {output_directory / 'ai-review-record.json'}")
    typer.echo(f"status: {client} client completed and retained accounting evidence")


@ai_review_app.command("reduce-recorded")
def ai_review_reduce_recorded(
    plan_path: Annotated[Path, typer.Option("--plan", help="Canonical AIReviewPlan JSON.")],
    batch_outputs: Annotated[
        list[Path],
        typer.Option(
            "--batch-output",
            help="Recorded strict JSON output, once per planned case batch in plan order.",
        ),
    ],
    integrity_output: Annotated[
        Path,
        typer.Option("--integrity-output", help="Recorded strict phase-integrity JSON output."),
    ],
    reduction_input: Annotated[
        Path,
        typer.Option("--reduction-input", help=_AI_REDUCTION_INPUT_HELP),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only AIQualityReviewRecord JSON path."),
    ],
) -> None:
    """Strictly parse recorded/fake outputs and reduce them without dispatching a model."""

    try:
        plan = _phase_io.load_contract(plan_path, AIReviewPlan)
        if len(batch_outputs) != len(plan.case_batches):
            raise ValueError("recorded batch output count does not match the review plan")
        batch_results = tuple(
            parse_ai_review_batch_output(batch, _phase_io.read_utf8(output_path))
            for batch, output_path in zip(plan.case_batches, batch_outputs, strict=True)
        )
        integrity_result = parse_ai_review_integrity_output(
            plan.phase_integrity_id,
            _phase_io.read_utf8(integrity_output),
        )
        reduction = _phase_io.load_exact_object(reduction_input, _AI_REDUCTION_INPUT_FIELDS)
        record = reduce_ai_quality_review(
            plan,
            batch_results=batch_results,
            integrity_result=integrity_result,
            occurrence_id=_phase_io.string_value(reduction, "occurrence_id"),
            ordinal=_phase_io.integer_value(reduction, "ordinal"),
            previous_ai_review_record_hash=_phase_io.optional_string_value(
                reduction, "previous_ai_review_record_hash"
            ),
            previous_history_root_hash=_phase_io.optional_string_value(
                reduction, "previous_history_root_hash"
            ),
            attempt_ids=_phase_io.string_tuple(reduction, "attempt_ids"),
            usage_record_ids=_phase_io.string_tuple(reduction, "usage_record_ids"),
            resource_record_ids=_phase_io.string_tuple(reduction, "resource_record_ids"),
            cost_record_ids=_phase_io.string_tuple(reduction, "cost_record_ids"),
            accounting_closed=_phase_io.boolean_value(reduction, "accounting_closed"),
            created_at=_phase_io.utc_datetime_value(reduction, "created_at"),
        )
        _phase_io.write_contract(output, record)
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"AI review record: {output}")
    typer.echo("status: reduced from recorded outputs; no model call was made")


@human_review_app.command("decision-prepare")
def human_review_decision_prepare(
    bundle_path: Annotated[Path, typer.Option("--bundle", help="EvaluationReviewBundle JSON.")],
    ai_review: Annotated[
        Path,
        typer.Option("--ai-review", help="Canonical AI PASS record JSON."),
    ],
    review_evidence_directory: Annotated[
        Path,
        typer.Option(
            "--review-evidence-directory",
            help=_PHASE_REVIEW_EVIDENCE_HELP,
        ),
    ],
    status: Annotated[str, typer.Option("--status", help="Human decision: pass or fail.")],
    decided_at: Annotated[
        str,
        typer.Option("--decided-at", help="Decision timestamp with an explicit UTC offset."),
    ],
    nonce: Annotated[str, typer.Option("--nonce", help="Fresh human-decision nonce.")],
    reviewer_label: Annotated[
        str,
        typer.Option("--reviewer-label", help="Public-safe reviewer label."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only unsigned canonical decision JSON."),
    ],
    finding_codes: Annotated[
        list[str] | None,
        typer.Option("--finding-code", help="Repeat for each public finding code."),
    ] = None,
    evidence_references: Annotated[
        list[str] | None,
        typer.Option("--evidence-reference", help="Repeat for each public evidence reference."),
    ] = None,
) -> None:
    """Prepare exact unsigned bytes after interactive bundle-hash confirmation."""

    try:
        bundle = _phase_io.load_contract(bundle_path, EvaluationReviewBundle)
        ai_record = _phase_io.load_contract(ai_review, AIQualityReviewRecord)
        _require_validator_bound_unique_ai_pass_head(
            bundle,
            ai_record,
            review_evidence_directory,
        )
        _confirm_bundle_hash(bundle)
        prepared = prepare_human_review_decision(
            review_bundle_hash=bundle.bundle_id,
            ai_review_record_hash=ai_record.ai_review_record_id,
            status=status,
            decided_at=_phase_io.parse_utc_datetime(decided_at),
            nonce=nonce,
            reviewer_label=reviewer_label,
            finding_codes=tuple(finding_codes or ()),
            evidence_references=tuple(evidence_references or ()),
        )
        atomic_write_bytes(output, prepared.canonical_bytes, trusted_root=output.parent)
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"unsigned human decision: {output}")
    typer.echo("sign these exact bytes outside OAMB with the pinned Ed25519 private key")


@human_review_app.command("record")
def human_review_record(
    bundle_path: Annotated[Path, typer.Option("--bundle", help="EvaluationReviewBundle JSON.")],
    ai_review: Annotated[
        Path,
        typer.Option("--ai-review", help="Canonical AI PASS record JSON."),
    ],
    review_evidence_directory: Annotated[
        Path,
        typer.Option(
            "--review-evidence-directory",
            help=_PHASE_REVIEW_EVIDENCE_HELP,
        ),
    ],
    decision_file: Annotated[
        Path,
        typer.Option("--decision-file", help="Canonical unsigned HumanReviewDecision bytes."),
    ],
    signature_file: Annotated[
        Path,
        typer.Option("--signature-file", help="Detached canonical-base64 Ed25519 signature."),
    ],
    key_binding_path: Annotated[
        Path,
        typer.Option("--key-binding", help="Pinned HumanReviewKeyBinding JSON."),
    ],
    imported_at: Annotated[
        str,
        typer.Option("--imported-at", help="Import timestamp with an explicit UTC offset."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only verified HumanQualityReviewRecord JSON."),
    ],
    prior_human_reviews: Annotated[
        list[Path] | None,
        typer.Option(
            "--prior-human-review",
            help="Repeat for append-only prior records whose nonces cannot be reused.",
        ),
    ] = None,
) -> None:
    """Verify a detached signature under the pinned public key and import the record."""

    try:
        bundle = _phase_io.load_contract(bundle_path, EvaluationReviewBundle)
        ai_record = _phase_io.load_contract(ai_review, AIQualityReviewRecord)
        _require_validator_bound_unique_ai_pass_head(
            bundle,
            ai_record,
            review_evidence_directory,
        )
        key_binding = _phase_io.load_contract(key_binding_path, HumanReviewKeyBinding)
        prior_records = tuple(
            _phase_io.load_contract(path, HumanQualityReviewRecord)
            for path in prior_human_reviews or ()
        )
        _confirm_bundle_hash(bundle)
        record = import_human_review_decision(
            decision_bytes=read_regular_file(decision_file),
            signature_base64=_phase_io.read_signature(signature_file),
            key_binding=key_binding,
            canonical_ai_record=ai_record,
            used_nonces=tuple(item.decision_nonce for item in prior_records),
            imported_at=_phase_io.parse_utc_datetime(imported_at),
        )
        _phase_io.write_contract(output, record)
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"verified human review record: {output}")


@gate_app.command("build")
def gate_build(
    bundle_path: Annotated[Path, typer.Option("--bundle", help="EvaluationReviewBundle JSON.")],
    ai_reviews: Annotated[
        list[Path],
        typer.Option("--ai-review", help="Repeat in complete append-only AI history order."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only EvaluationPhaseGate JSON path."),
    ],
    human_review: Annotated[
        Path | None,
        typer.Option("--human-review", help="Optional verified human review record JSON."),
    ] = None,
) -> None:
    """Derive the immutable gate from the complete ordered review history."""

    try:
        bundle = _phase_io.load_contract(bundle_path, EvaluationReviewBundle)
        ai_history = tuple(
            _phase_io.load_contract(path, AIQualityReviewRecord) for path in ai_reviews
        )
        human_records = (
            ()
            if human_review is None
            else (_phase_io.load_contract(human_review, HumanQualityReviewRecord),)
        )
        gate = derive_evaluation_phase_gate(
            phase_id=bundle.phase_id,
            review_bundle_hash=bundle.bundle_id,
            ordered_ai_history=ai_history,
            human_records=human_records,
        )
        _phase_io.write_contract(output, gate)
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"phase gate: {output}")


@gate_app.command("validate")
def gate_validate(
    bundle_path: Annotated[Path, typer.Option("--bundle", help="EvaluationReviewBundle JSON.")],
    gate_path: Annotated[Path, typer.Option("--gate", help="EvaluationPhaseGate JSON.")],
    review_evidence_directory: Annotated[
        Path,
        typer.Option(
            "--review-evidence-directory",
            help=_PHASE_REVIEW_EVIDENCE_HELP,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only phase ValidationResult JSON path."),
    ],
) -> None:
    """Run the closed T10 validator over the gate and full immutable review evidence."""

    try:
        bundle = _phase_io.load_contract(bundle_path, EvaluationReviewBundle)
        gate = _phase_io.load_contract(gate_path, EvaluationPhaseGate)
        evidence = _phase_io.load_phase_review_evidence(review_evidence_directory)
        result = validate_t10_phase_gate(bundle, gate, review_evidence=evidence)
        _phase_io.write_contract(output, result)
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"{result.disposition.value}: {output}")
    if result.disposition != ValidationDisposition.VALIDATED:
        raise typer.Exit(code=1)


@acceptance_report_app.command("build")
def acceptance_report_build(
    bundle_path: Annotated[Path, typer.Option("--bundle", help="EvaluationReviewBundle JSON.")],
    gate_path: Annotated[Path, typer.Option("--gate", help="EvaluationPhaseGate JSON.")],
    review_evidence_directory: Annotated[
        Path,
        typer.Option(
            "--review-evidence-directory",
            help=_PHASE_REVIEW_EVIDENCE_HELP,
        ),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Create-only derivation parent directory."),
    ],
    committed_at: Annotated[
        str,
        typer.Option("--committed-at", help="Derivation timestamp with an explicit UTC offset."),
    ],
    audience: Annotated[
        str,
        typer.Option("--audience", help="Acceptance report audience: public or local."),
    ] = "public",
    limitations: Annotated[
        list[str] | None,
        typer.Option("--limitation", help="Repeat for each explicit report limitation."),
    ] = None,
) -> None:
    """Fresh-validate the T10 gate, then render, export-validate, and seal its report."""

    from oamb.reporting.offline_renderer import offline_asset_hashes, offline_renderer_hash
    from oamb.reporting.public import build_phase_acceptance_report
    from oamb.reporting.publication import build_report_derivation
    from oamb.reporting.roots import build_acceptance_report_spec

    try:
        if audience not in {"public", "local"}:
            raise ValueError("audience must be 'public' or 'local'")
        bundle = _phase_io.load_contract(bundle_path, EvaluationReviewBundle)
        gate = _phase_io.load_contract(gate_path, EvaluationPhaseGate)
        evidence = _phase_io.load_phase_review_evidence(review_evidence_directory)
        validation = validate_t10_phase_gate(bundle, gate, review_evidence=evidence)
        if validation.disposition != ValidationDisposition.VALIDATED:
            raise ValueError("phase gate is not VALIDATED; acceptance report was not built")
        human = gate.human_record
        if human is None:
            raise ValueError("phase acceptance report requires a verified human review")
        spec = build_acceptance_report_spec(
            audience=audience,
            evaluation_report_hash=bundle.report_model_hash,
            evaluation_export_validation_hash=bundle.export_validation_hash,
            review_bundle_hash=bundle.bundle_id,
            ai_review_record_hash=gate.canonical_ai_review_record_hash,
            human_review_record_hash=human.human_review_record_id,
            phase_gate_hash=gate.gate_id,
            renderer_hash=offline_renderer_hash(),
            asset_hashes=offline_asset_hashes(),
            export_profile_selector_id=f"{audience}-phase-acceptance-v1",
            export_profile_selector_version=1,
        )
        model = build_phase_acceptance_report(
            acceptance_report_spec_hash=canonical_sha256(spec),
            phase_id=bundle.phase_id,
            evaluation_report_hash=bundle.report_model_hash,
            evaluation_export_validation_hash=bundle.export_validation_hash,
            review_bundle_hash=bundle.bundle_id,
            phase_gate_hash=gate.gate_id,
            ai_review_record_hash=gate.canonical_ai_review_record_hash,
            human_review_record_hash=human.human_review_record_id,
            gate_review_bundle_hash=gate.review_bundle_hash,
            gate_passed_by_ai=gate.passed_by_ai,
            gate_passed_by_human=gate.passed_by_human,
            finding_codes=human.finding_codes,
            evidence_references=human.evidence_references,
            limitations=tuple(limitations or ()),
        )
        validation_hash = canonical_sha256(validation)
        source_binding = SourceEvidenceBinding(
            binding_id=canonical_sha256(
                ["oamb-phase-gate-source-binding-v1", gate.gate_id, validation_hash]
            ),
            source_kind=SourceEvidenceKind.DERIVATION,
            source_identity=gate.gate_id,
            source_root_hash=validation.target_hash,
            validation_result_hash=validation_hash,
            source_schema_versions=("evaluation_phase_gate@1",),
        )
        target = PhaseGateValidationInput(bundle=bundle, gate=gate, review_evidence=evidence)
        built = build_report_derivation(
            model=model,
            report_spec=spec,
            ordered_source_bindings=(source_binding,),
            evidence_validations=(validation,),
            evidence_validation_targets=(target,),
            transform_spec_hash=canonical_sha256(
                [_PHASE_ACCEPTANCE_TRANSFORM_VERSION, spec.acceptance_report_spec_id]
            ),
            schema_versions=_PHASE_ACCEPTANCE_SCHEMAS,
            output_root=output_root,
            committed_at=_phase_io.parse_utc_datetime(committed_at),
        )
    except Exception as exc:
        _raise_cli_error(exc)
    typer.echo(f"acceptance report: {built.report_path}")


def _count_review_tokens(text: str) -> int:
    return len(o200k_encoding().encode_ordinary(text))


def _write_contract_array(path: Path, values: tuple[object, ...], trusted_root: Path) -> None:
    atomic_write_bytes(path, canonical_json_bytes(values), trusted_root=trusted_root)


def _write_phase_review_inputs(
    output_directory: Path,
    *,
    plan: AIReviewPlan,
    role: ModelRoleBindingV2,
    approval: ExternalCallApprovalRecord,
    budget: BudgetSpecV2,
    cost_measurement_spec: CostMeasurementSpec,
    execution_environment: ExecutionEnvironmentBinding,
    price_snapshot: PriceSnapshot | None,
) -> None:
    _phase_io.write_contract(output_directory / "plan.json", plan)
    _phase_io.write_contract(output_directory / "role-binding.json", role)
    _phase_io.write_contract(output_directory / "approval.json", approval)
    _phase_io.write_contract(output_directory / "budget.json", budget)
    _phase_io.write_contract(output_directory / "cost-measurement-spec.json", cost_measurement_spec)
    _phase_io.write_contract(output_directory / "execution-environment.json", execution_environment)
    if price_snapshot is not None:
        _phase_io.write_contract(output_directory / "price-snapshot.json", price_snapshot)


def _write_partial_phase_review(
    output_directory: Path,
    partial: PartialPhaseReviewRunEvidence,
) -> None:
    _phase_io.write_contract(output_directory / "occurrence.json", partial.occurrence)
    _write_contract_array(
        output_directory / "batch-results.json", partial.batch_results, output_directory
    )
    if partial.integrity_result is not None:
        _phase_io.write_contract(
            output_directory / "integrity-result.json", partial.integrity_result
        )
    _write_contract_array(output_directory / "attempts.json", partial.attempts, output_directory)
    _write_contract_array(
        output_directory / "usage-records.json", partial.usage_records, output_directory
    )
    _write_contract_array(
        output_directory / "resource-records.json", partial.resource_records, output_directory
    )
    _write_contract_array(
        output_directory / "cost-records.json", partial.cost_records, output_directory
    )


def _require_validator_bound_unique_ai_pass_head(
    bundle: EvaluationReviewBundle,
    ai_record: AIQualityReviewRecord,
    review_evidence_directory: Path,
) -> None:
    evidence = _phase_io.load_phase_review_evidence(review_evidence_directory)
    evidence_head = evidence.ordered_ai_history[-1]
    if not phase_ai_review_evidence_closes(bundle, evidence, ai_record):
        raise ValueError("complete phase review evidence is not freshly VALIDATED")
    matching_records = tuple(
        record
        for record in evidence.ordered_ai_history
        if record.ai_review_record_id == ai_record.ai_review_record_id
    )
    if (
        len(matching_records) != 1
        or matching_records[0] != ai_record
        or evidence_head != ai_record
        or ai_record.status != QualityReviewStatus.PASS
    ):
        raise ValueError("human review requires the validator-bound unique canonical AI PASS head")


def _confirm_bundle_hash(bundle: EvaluationReviewBundle) -> None:
    typer.echo(f"review bundle: {bundle.bundle_id}")
    typer.echo(f"evaluation report model: {bundle.report_model_hash}")
    if not _stdin_is_interactive():
        raise ValueError("bundle-hash confirmation requires an interactive TTY")
    confirmed = typer.prompt("Retype the exact review bundle hash")
    if confirmed != bundle.bundle_id:
        raise ValueError("interactive review bundle hash confirmation did not match")


def _stdin_is_interactive() -> bool:
    return sys.stdin.isatty()


def _raise_cli_error(exc: Exception) -> None:
    if isinstance(exc, typer.Exit):
        raise exc
    raise typer.BadParameter(str(exc)) from exc


__all__ = ["phase_app"]
