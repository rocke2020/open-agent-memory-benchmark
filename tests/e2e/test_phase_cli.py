from __future__ import annotations

import json
import runpy
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from oamb import phase_cli
from oamb.artifacts.validation.phase import (
    PhaseReviewEvidence,
    phase_ai_review_evidence_closes,
)
from oamb.cli import app
from oamb.contracts.ids import canonical_json_bytes
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    EvaluationPhaseGate,
    EvaluationReviewBundle,
    HumanQualityReviewRecord,
    PhaseAcceptanceReport,
)
from oamb.contracts.specifications import AIReviewPlan
from oamb.phase_cli_io import load_phase_review_evidence
from oamb.reporting.review import build_evaluation_review_bundle

_TEST_ROOT = Path(__file__).parents[1]
_PHASE_FIXTURES = runpy.run_path(str(_TEST_ROOT / "contracts" / "test_t8_phase_validation.py"))
_AI_FIXTURES = runpy.run_path(str(_TEST_ROOT / "unit" / "test_t8_ai_review.py"))
_accepted_gate_with_evidence = _PHASE_FIXTURES["_accepted_gate_with_evidence"]
_bundle = _AI_FIXTURES["_bundle"]
_integrity_projection = _AI_FIXTURES["_integrity_projection"]
_plan_build = _AI_FIXTURES["_plan_build"]
_projections = _AI_FIXTURES["_projections"]

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
CREATED_AT = datetime(2026, 8, 29, tzinfo=UTC)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value))


def test_phase_cli_help_exposes_explicit_review_and_human_boundaries() -> None:
    runner = CliRunner()

    phase = runner.invoke(app, ["phase", "--help"])
    ai_review = runner.invoke(app, ["phase", "ai-review", "--help"])
    human_review = runner.invoke(app, ["phase", "human-review", "--help"])
    bundle_build = runner.invoke(app, ["phase", "bundle", "build", "--help"])
    review_plan = runner.invoke(app, ["phase", "ai-review", "plan", "--help"])
    review_run = runner.invoke(app, ["phase", "ai-review", "run", "--help"])
    review_reduce = runner.invoke(app, ["phase", "ai-review", "reduce-recorded", "--help"])
    gate_validate = runner.invoke(app, ["phase", "gate", "validate", "--help"])

    assert phase.exit_code == 0, phase.output
    assert ai_review.exit_code == 0, ai_review.output
    assert human_review.exit_code == 0, human_review.output
    assert bundle_build.exit_code == 0, bundle_build.output
    assert review_plan.exit_code == 0, review_plan.output
    assert review_run.exit_code == 0, review_run.output
    assert review_reduce.exit_code == 0, review_reduce.output
    assert gate_validate.exit_code == 0, gate_validate.output
    assert "bundle" in phase.output
    assert "gate" in phase.output
    assert "acceptance-report" in phase.output
    assert "plan" in ai_review.output
    assert "run" in ai_review.output
    assert "reduce-recorded" in ai_review.output
    assert "Model client profile: fake" in review_run.output
    assert "openai-compatible" in review_run.output
    assert "without dispatching" in review_reduce.output
    assert "record" in human_review.output
    assert "decision-prepare" not in human_review.output
    assert "sign" not in human_review.output.lower()
    assert "ordered_capsule_hashes" in bundle_build.output
    assert "projection_spec_hash" in review_plan.output
    assert "occurrence_id" in review_reduce.output
    assert "HumanQualityReviewRecord" in human_review.output


def test_core_cli_and_help_do_not_import_model_clients_or_concrete_adapters() -> None:
    script = (
        "import sys; import oamb.cli; "
        "forbidden=('oamb.model_clients', 'oamb.memory_systems.hindsight', "
        "'oamb.memory_systems.mem0', 'oamb.memory_systems.openviking'); "
        "loaded=sorted(name for name in sys.modules "
        "if any(name == item or name.startswith(item + '.') for item in forbidden)); "
        "assert not loaded, loaded"
    )
    imported = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    root_help = subprocess.run(
        [sys.executable, "-m", "oamb.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    phase_help = subprocess.run(
        [sys.executable, "-m", "oamb.cli", "phase", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert imported.returncode == 0, imported.stderr
    assert root_help.returncode == 0, root_help.stderr
    assert phase_help.returncode == 0, phase_help.stderr
    assert "ai-review" in phase_help.stdout


def test_phase_bundle_plan_and_recorded_ai_reduction_are_create_only(tmp_path: Path) -> None:
    runner = CliRunner()
    bundle_input = tmp_path / "bundle-input.json"
    bundle_path = tmp_path / "bundle.json"
    expected_bundle = _bundle()
    _write_json(
        bundle_input,
        expected_bundle.model_dump(
            mode="json",
            exclude={"schema_name", "schema_version", "bundle_id", "ordered_review_input_hash"},
        ),
    )

    bundled = runner.invoke(
        app,
        [
            "phase",
            "bundle",
            "build",
            "--input",
            str(bundle_input),
            "--output",
            str(bundle_path),
        ],
    )

    assert bundled.exit_code == 0, bundled.output
    bundle = EvaluationReviewBundle.model_validate_json(bundle_path.read_bytes())
    assert bundle == expected_bundle

    projections_path = tmp_path / "case-projections.json"
    integrity_path = tmp_path / "integrity-projection.json"
    config_path = tmp_path / "review-config.json"
    plan_root = tmp_path / "review-plan"
    ordered_projections = tuple(
        sorted(
            _projections(__import__("oamb.reporting.review", fromlist=["review"])),
            key=lambda item: (
                item.memory_system_id,
                item.workload_id,
                item.manifest_ordinal,
                item.case_occurrence_id,
            ),
        )
    )
    bundle = build_evaluation_review_bundle(
        phase_id=bundle.phase_id,
        ordered_capsule_hashes=bundle.ordered_capsule_hashes,
        ordered_validation_hashes=bundle.ordered_validation_hashes,
        ordered_case_occurrence_ids=tuple(item.case_occurrence_id for item in ordered_projections),
        unique_case_manifest_entry_ids=bundle.unique_case_manifest_entry_ids,
        ordinary_derivation_hashes=bundle.ordinary_derivation_hashes,
        report_model_hash=bundle.report_model_hash,
        report_html_hash=bundle.report_html_hash,
        export_validation_hash=bundle.export_validation_hash,
        limitations=bundle.limitations,
    )
    plan_bundle_path = tmp_path / "plan-bundle.json"
    _write_json(plan_bundle_path, bundle)
    _write_json(projections_path, ordered_projections)
    _write_json(
        integrity_path,
        _integrity_projection(__import__("oamb.reporting.review", fromlist=["review"])),
    )
    _write_json(
        config_path,
        {
            "projection_spec_hash": SHA_A,
            "prompt_pack_hash": SHA_B,
            "output_contract_hash": SHA_C,
            "parser_hash": SHA_D,
            "reviewer_role_binding_hash": SHA_A,
            "reviewer_model_hash": SHA_B,
            "reviewer_runtime_hash": SHA_C,
            "reviewer_configuration_hash": SHA_D,
            "model_context_window_tokens": 32768,
            "aggregate_version": "ai-quality-aggregate-v1",
        },
    )

    planned = runner.invoke(
        app,
        [
            "phase",
            "ai-review",
            "plan",
            "--bundle",
            str(plan_bundle_path),
            "--case-projections",
            str(projections_path),
            "--integrity-projection",
            str(integrity_path),
            "--review-config",
            str(config_path),
            "--output-directory",
            str(plan_root),
        ],
    )

    assert planned.exit_code == 0, planned.output
    plan = AIReviewPlan.model_validate_json((plan_root / "plan.json").read_bytes())
    assert plan.reviewer_counter_fingerprint != SHA_D
    case_outputs: list[Path] = []
    for index, batch in enumerate(plan.case_batches, 1):
        output_path = tmp_path / f"batch-{index}.json"
        _write_json(
            output_path,
            {
                "batch_id": batch.batch_id,
                "results": [
                    {"case_occurrence_id": case_id, "status": "pass", "findings": []}
                    for case_id in batch.ordered_case_occurrence_ids
                ],
                "status": "pass",
            },
        )
        case_outputs.append(output_path)
    integrity_output = tmp_path / "integrity-output.json"
    _write_json(
        integrity_output,
        {"integrity_id": plan.phase_integrity_id, "status": "pass", "findings": []},
    )
    reduction_input = tmp_path / "reduction-input.json"
    _write_json(
        reduction_input,
        {
            "occurrence_id": SHA_C,
            "ordinal": 1,
            "previous_ai_review_record_hash": None,
            "previous_history_root_hash": None,
            "attempt_ids": [f"{index:064x}" for index in range(701, 704)],
            "usage_record_ids": [SHA_A],
            "resource_record_ids": [SHA_B],
            "cost_record_ids": [SHA_C],
            "accounting_closed": True,
            "created_at": CREATED_AT.isoformat(),
        },
    )
    ai_record_path = tmp_path / "ai-review-record.json"
    arguments = [
        "phase",
        "ai-review",
        "reduce-recorded",
        "--plan",
        str(plan_root / "plan.json"),
    ]
    for output_path in case_outputs:
        arguments.extend(("--batch-output", str(output_path)))
    arguments.extend(
        (
            "--integrity-output",
            str(integrity_output),
            "--reduction-input",
            str(reduction_input),
            "--output",
            str(ai_record_path),
        )
    )

    reduced = runner.invoke(app, arguments)

    assert reduced.exit_code == 0, reduced.output
    ai_record = AIQualityReviewRecord.model_validate_json(ai_record_path.read_bytes())
    assert ai_record.status == "pass"
    duplicate = runner.invoke(app, arguments)
    assert duplicate.exit_code == 0, duplicate.output
    ai_record_path.write_text("{}", encoding="utf-8")
    collision = runner.invoke(app, arguments)
    assert collision.exit_code != 0
    assert "different bytes" in collision.output


def _write_review_evidence(root: Path, evidence: PhaseReviewEvidence) -> None:
    _write_json(root / "plan.json", evidence.plan)
    _write_json(root / "role-binding.json", evidence.reviewer_role_binding)
    _write_json(root / "cost-measurement-spec.json", evidence.cost_measurement_spec)
    _write_json(root / "execution-environment.json", evidence.execution_environment)
    if evidence.price_snapshot is not None:
        _write_json(root / "price-snapshot.json", evidence.price_snapshot)
    _write_json(root / "lease-record.json", evidence.lease_record)
    _write_json(root / "occurrence-claims.json", evidence.occurrence_claims)
    _write_json(root / "budget-reservations.json", evidence.budget_reservations)
    _write_json(root / "attempt-intents.json", evidence.attempt_intents)
    _write_json(root / "attempt-receipts.json", evidence.attempt_receipts)
    if evidence.close_errors:
        _write_json(root / "close-errors.json", evidence.close_errors)
    _write_json(root / "occurrence.json", evidence.occurrence)
    _write_json(root / "approval.json", evidence.approval)
    _write_json(root / "budget.json", evidence.budget)
    _write_json(root / "batch-results.json", evidence.batch_results)
    _write_json(root / "integrity-result.json", evidence.integrity_result)
    _write_json(root / "ai-history.json", evidence.ordered_ai_history)
    _write_json(root / "attempts.json", evidence.attempts)
    _write_json(root / "usage-records.json", evidence.usage_records)
    _write_json(root / "resource-records.json", evidence.resource_records)
    _write_json(root / "cost-records.json", evidence.cost_records)


def _reduce_shape_only_pass(root: Path, evidence: PhaseReviewEvidence) -> Path:
    runner = CliRunner()
    plan = evidence.plan
    plan_path = root / "plan.json"
    integrity_output = root / "integrity-output.json"
    reduction_input = root / "reduction-input.json"
    output = root / "shape-only-pass.json"
    _write_json(plan_path, plan)
    batch_outputs: list[Path] = []
    for index, batch in enumerate(plan.case_batches, 1):
        batch_output = root / f"batch-{index}.json"
        _write_json(
            batch_output,
            {
                "batch_id": batch.batch_id,
                "results": [
                    {"case_occurrence_id": case_id, "status": "pass", "findings": []}
                    for case_id in batch.ordered_case_occurrence_ids
                ],
                "status": "pass",
            },
        )
        batch_outputs.append(batch_output)
    _write_json(
        integrity_output,
        {"integrity_id": plan.phase_integrity_id, "status": "pass", "findings": []},
    )
    _write_json(
        reduction_input,
        {
            "occurrence_id": SHA_C,
            "ordinal": 1,
            "previous_ai_review_record_hash": None,
            "previous_history_root_hash": None,
            "attempt_ids": [
                f"{index + 20_000:064x}" for index in range(plan.expected_attempt_count)
            ],
            "usage_record_ids": [SHA_A],
            "resource_record_ids": [SHA_B],
            "cost_record_ids": [SHA_C],
            "accounting_closed": True,
            "created_at": CREATED_AT.isoformat(),
        },
    )
    arguments = [
        "phase",
        "ai-review",
        "reduce-recorded",
        "--plan",
        str(plan_path),
    ]
    for batch_output in batch_outputs:
        arguments.extend(("--batch-output", str(batch_output)))
    arguments.extend(
        (
            "--integrity-output",
            str(integrity_output),
            "--reduction-input",
            str(reduction_input),
            "--output",
            str(output),
        )
    )

    reduced = runner.invoke(app, arguments)

    assert reduced.exit_code == 0, reduced.output
    assert AIQualityReviewRecord.model_validate_json(output.read_bytes()).status == "pass"
    return output


def test_phase_human_gate_validation_and_acceptance_report_offline_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phase_cli, "_stdin_is_interactive", lambda: True)
    runner = CliRunner()
    bundle, accepted_gate, evidence = _accepted_gate_with_evidence()
    assert accepted_gate.human_record is not None
    bundle_path = tmp_path / "bundle.json"
    ai_path = tmp_path / "ai.json"
    gate_path = tmp_path / "gate.json"
    review_evidence_root = tmp_path / "review-evidence"
    human_path = review_evidence_root / "human-review.json"
    validation_path = tmp_path / "phase-validation.json"
    _write_json(bundle_path, bundle)
    _write_json(ai_path, accepted_gate.ai_record)
    _write_review_evidence(review_evidence_root, evidence)

    recorded = runner.invoke(
        app,
        [
            "phase",
            "human-review",
            "record",
            "--bundle",
            str(bundle_path),
            "--ai-review",
            str(ai_path),
            "--review-evidence-directory",
            str(review_evidence_root),
            "--status",
            accepted_gate.human_record.status,
            "--nonce",
            accepted_gate.human_record.confirmation_nonce,
            "--operator-id",
            accepted_gate.human_record.operator_id,
            "--created-at",
            accepted_gate.human_record.created_at.isoformat(),
        ],
        input=(
            f"PASS {bundle.bundle_id} {accepted_gate.ai_record.ai_review_record_id} "
            f"{accepted_gate.human_record.confirmation_nonce}\n"
        ),
    )

    assert recorded.exit_code == 0, recorded.output
    human = HumanQualityReviewRecord.model_validate_json(human_path.read_bytes())
    assert human == accepted_gate.human_record

    built_gate = runner.invoke(
        app,
        [
            "phase",
            "gate",
            "build",
            "--bundle",
            str(bundle_path),
            "--ai-review",
            str(ai_path),
            "--human-review",
            str(human_path),
            "--output",
            str(gate_path),
        ],
    )

    assert built_gate.exit_code == 0, built_gate.output
    gate = EvaluationPhaseGate.model_validate_json(gate_path.read_bytes())
    assert gate == accepted_gate
    validated = runner.invoke(
        app,
        [
            "phase",
            "gate",
            "validate",
            "--bundle",
            str(bundle_path),
            "--gate",
            str(gate_path),
            "--review-evidence-directory",
            str(review_evidence_root),
            "--output",
            str(validation_path),
        ],
    )

    assert validated.exit_code == 0, validated.output
    assert json.loads(validation_path.read_bytes())["disposition"] == "validated"

    acceptance_root = tmp_path / "acceptance"
    accepted = runner.invoke(
        app,
        [
            "phase",
            "acceptance-report",
            "build",
            "--bundle",
            str(bundle_path),
            "--gate",
            str(gate_path),
            "--review-evidence-directory",
            str(review_evidence_root),
            "--output-root",
            str(acceptance_root),
            "--committed-at",
            CREATED_AT.isoformat(),
            "--limitation",
            "fixture-only offline phase chain",
        ],
    )

    assert accepted.exit_code == 0, accepted.output
    report_path = Path(accepted.output.strip().split("acceptance report: ", 1)[1])
    assert report_path.is_file()
    model = PhaseAcceptanceReport.model_validate_json(
        (report_path.parent / "report-model.json").read_bytes()
    )
    assert model.passed_by_ai is True
    assert model.passed_by_human is True


def test_human_record_rejects_noninteractive_confirmation_before_write(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    bundle, gate, evidence = _accepted_gate_with_evidence()
    bundle_path = tmp_path / "bundle.json"
    ai_path = tmp_path / "ai.json"
    review_evidence_root = tmp_path / "review-evidence"
    output = review_evidence_root / "human-review.json"
    _write_json(bundle_path, bundle)
    _write_json(ai_path, gate.ai_record)
    _write_review_evidence(review_evidence_root, evidence)

    rejected = runner.invoke(
        app,
        [
            "phase",
            "human-review",
            "record",
            "--bundle",
            str(bundle_path),
            "--ai-review",
            str(ai_path),
            "--review-evidence-directory",
            str(review_evidence_root),
            "--status",
            "pass",
            "--nonce",
            gate.human_record.confirmation_nonce if gate.human_record else "fixture-nonce",
            "--operator-id",
            gate.human_record.operator_id if gate.human_record else "fixture-operator",
            "--created-at",
            gate.human_record.created_at.isoformat() if gate.human_record else "",
        ],
        input=f"{bundle.bundle_id}\n",
    )

    assert rejected.exit_code != 0
    assert "interactive TTY" in rejected.output
    assert not output.exists()


def test_human_record_uses_one_canonical_create_only_path_per_review_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phase_cli, "_stdin_is_interactive", lambda: True)
    bundle, gate, evidence = _accepted_gate_with_evidence()
    assert gate.human_record is not None
    bundle_path = tmp_path / "bundle.json"
    ai_path = tmp_path / "ai.json"
    review_root = tmp_path / "review-evidence"
    _write_json(bundle_path, bundle)
    _write_json(ai_path, gate.ai_record)
    _write_review_evidence(review_root, evidence)
    common = [
        "phase",
        "human-review",
        "record",
        "--bundle",
        str(bundle_path),
        "--ai-review",
        str(ai_path),
        "--review-evidence-directory",
        str(review_root),
        "--status",
        "pass",
        "--operator-id",
        gate.human_record.operator_id,
        "--created-at",
        gate.human_record.created_at.isoformat(),
    ]
    runner = CliRunner()
    first_nonce = "canonical-review-1"
    first = runner.invoke(
        app,
        [*common, "--nonce", first_nonce],
        input=f"PASS {bundle.bundle_id} {gate.ai_record.ai_review_record_id} {first_nonce}\n",
    )

    assert first.exit_code == 0, first.output
    canonical_path = review_root / "human-review.json"
    first_bytes = canonical_path.read_bytes()

    second = runner.invoke(
        app,
        [*common, "--nonce", "canonical-review-2"],
        input=(
            f"PASS {bundle.bundle_id} {gate.ai_record.ai_review_record_id} canonical-review-2\n"
        ),
    )

    assert second.exit_code != 0
    assert "already has a human quality review record" in second.output
    assert canonical_path.read_bytes() == first_bytes


def test_human_record_rejects_shape_only_recorded_pass_before_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(phase_cli, "_stdin_is_interactive", lambda: True)
    bundle, gate, evidence = _accepted_gate_with_evidence()
    forged_ai_path = _reduce_shape_only_pass(tmp_path / "forged", evidence)
    assert gate.human_record is not None
    bundle_path = tmp_path / "bundle.json"
    review_evidence_root = tmp_path / "review-evidence"
    output = review_evidence_root / "human-review.json"
    _write_json(bundle_path, bundle)
    _write_review_evidence(review_evidence_root, evidence)

    result = CliRunner().invoke(
        app,
        [
            "phase",
            "human-review",
            "record",
            "--bundle",
            str(bundle_path),
            "--ai-review",
            str(forged_ai_path),
            "--review-evidence-directory",
            str(review_evidence_root),
            "--status",
            "pass",
            "--nonce",
            "forged-ai-confirmation",
            "--operator-id",
            "fixture-operator",
            "--created-at",
            gate.human_record.created_at.isoformat(),
        ],
        input=(
            f"PASS {bundle.bundle_id} {gate.ai_record.ai_review_record_id} forged-ai-confirmation\n"
        ),
    )

    assert result.exit_code != 0
    assert "not freshly VALIDATED" in result.output
    assert "Traceback" not in result.output
    assert not output.exists()


def test_phase_ai_evidence_validation_requires_no_prior_human_artifact(
    tmp_path: Path,
) -> None:
    bundle, gate, evidence = _accepted_gate_with_evidence()
    review_evidence_root = tmp_path / "review-evidence"
    _write_review_evidence(review_evidence_root, evidence)
    loaded = load_phase_review_evidence(review_evidence_root)

    assert phase_ai_review_evidence_closes(bundle, loaded, gate.ai_record)


def test_phase_ai_reduction_rejects_malformed_recorded_output_before_write(
    tmp_path: Path,
) -> None:
    runner = CliRunner()
    plan = _plan_build(__import__("oamb.reporting.review", fromlist=["review"])).plan
    plan_path = tmp_path / "plan.json"
    bad_output = tmp_path / "bad-output.json"
    integrity_output = tmp_path / "integrity-output.json"
    reduction_input = tmp_path / "reduction-input.json"
    output = tmp_path / "record.json"
    _write_json(plan_path, plan)
    bad_output.write_text('{"batch_id":"wrong","results":[],"status":"pass"}', encoding="utf-8")
    _write_json(
        integrity_output,
        {"integrity_id": plan.phase_integrity_id, "status": "pass", "findings": []},
    )
    _write_json(
        reduction_input,
        {
            "occurrence_id": SHA_C,
            "ordinal": 1,
            "previous_ai_review_record_hash": None,
            "previous_history_root_hash": None,
            "attempt_ids": [SHA_A],
            "usage_record_ids": [SHA_A],
            "resource_record_ids": [SHA_B],
            "cost_record_ids": [SHA_C],
            "accounting_closed": True,
            "created_at": CREATED_AT.isoformat(),
        },
    )

    batch_outputs = [bad_output]
    for index, batch in enumerate(plan.case_batches[1:], 2):
        batch_output = tmp_path / f"batch-{index}.json"
        _write_json(
            batch_output,
            {
                "batch_id": batch.batch_id,
                "results": [
                    {"case_occurrence_id": case_id, "status": "pass", "findings": []}
                    for case_id in batch.ordered_case_occurrence_ids
                ],
                "status": "pass",
            },
        )
        batch_outputs.append(batch_output)
    arguments = [
        "phase",
        "ai-review",
        "reduce-recorded",
        "--plan",
        str(plan_path),
    ]
    for batch_output in batch_outputs:
        arguments.extend(("--batch-output", str(batch_output)))
    arguments.extend(
        (
            "--integrity-output",
            str(integrity_output),
            "--reduction-input",
            str(reduction_input),
            "--output",
            str(output),
        )
    )
    rejected = runner.invoke(
        app,
        arguments,
    )

    assert rejected.exit_code != 0
    assert not output.exists()
    assert "wrong batch" in rejected.output
