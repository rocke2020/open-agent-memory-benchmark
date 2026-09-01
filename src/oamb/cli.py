"""Offline-first OAMB command composition root."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import typer

from .artifacts.atomic import atomic_write_bytes
from .contracts.evidence import ValidationResult
from .contracts.ids import canonical_json_bytes, canonical_sha256
from .contracts.specifications import RunSpec
from .contracts.states import ValidationDisposition
from .historical_cli import external_app

if TYPE_CHECKING:
    from .contracts.ports import ArtifactStorePort, IngestionPlan, MemorySystemPort
    from .runtime.fake_run import FakeRunArtifacts, FakeRunScenario
    from .runtime.preflight import ArtifactDurabilityPreflight

app = typer.Typer(
    name="oamb",
    help="Open Agent Memory Benchmark evidence and comparison tooling.",
    no_args_is_help=True,
)
manifest_app = typer.Typer(help="Build deterministic workload manifests.")
capsule_app = typer.Typer(help="Validate immutable source capsules.")
report_app = typer.Typer(help="Build validated offline reports.")
app.add_typer(manifest_app, name="manifest")
app.add_typer(capsule_app, name="capsule")
app.add_typer(report_app, name="report")
app.add_typer(external_app, name="external")


@app.command("doctor")
def doctor_command(
    config: Annotated[
        Path,
        typer.Argument(help="Strict benchmark YAML configuration.", metavar="CONFIG"),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Comparison output directory."),
    ],
) -> None:
    """Resolve generic benchmark configuration without constructing providers."""

    from .config.benchmark import BenchmarkConfigurationError, load_benchmark_configuration
    from .config.doctor import ResolvedPlanError, build_resolved_plan, resolved_plan_bytes

    try:
        configuration = load_benchmark_configuration(config)
        plan = build_resolved_plan(configuration)
    except (BenchmarkConfigurationError, ResolvedPlanError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    destination = output / "resolved-plan.json"
    if destination.exists():
        raise typer.BadParameter("resolved plan output already exists")
    content = resolved_plan_bytes(plan)
    atomic_write_bytes(destination, content, trusted_root=output)
    typer.echo(f"comparison: {plan.comparison_id}")
    typer.echo(f"cells: {len(plan.cells)}")
    typer.echo(f"retrieval generation: {plan.retrieval.generation}")
    for role in plan.model_roles:
        rank = (
            "not applicable"
            if role.thinking_effort_rank_1_indexed is None
            else f"rank {role.thinking_effort_rank_1_indexed}/{len(role.thinking_effort_scale)}"
        )
        typer.echo(
            f"model {role.role_id}: {role.configured_model} / "
            f"{role.thinking_effort} ({rank}); recipient={role.recipient}"
        )
    typer.echo("credential values: [REDACTED]")
    typer.echo(f"resolved plan: {destination}")
    typer.echo(f"resolved plan sha256: {hashlib.sha256(content).hexdigest()}")


@manifest_app.command("build")
def manifest_build(
    workload: Annotated[str, typer.Option("--workload", help="Workload ID or 'fake'.")],
    run_id: Annotated[str, typer.Option("--run-id", help="Immutable run identity.")],
    output: Annotated[Path, typer.Option("--output", help="Manifest bundle directory.")],
) -> None:
    """Build the generated credential-free T5 input bundle."""

    from .runtime.fake_run import (
        FAKE_RUNTIME_BINDING_HASH,
        build_fake_budget,
        build_fake_run_spec,
    )
    from .workloads.fake import GeneratedFakeWorkload

    if workload != "fake":
        raise typer.BadParameter("T5 implements only the generated 'fake' workload")
    generated = GeneratedFakeWorkload()
    dataset = generated.resolve_sources()
    case_manifest = generated.build_case_manifest(dataset)
    run_spec = build_fake_run_spec(
        run_id=run_id,
        dataset_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        runtime_binding_hash=FAKE_RUNTIME_BINDING_HASH,
        workload_id=case_manifest.workload_id,
    )
    bundle = {
        "dataset-manifest.json": canonical_json_bytes(dataset),
        "case-manifest.json": canonical_json_bytes(case_manifest),
        "run-spec.json": canonical_json_bytes(run_spec),
        "budget.json": canonical_json_bytes(build_fake_budget(run_id)),
    }
    for name, content in bundle.items():
        atomic_write_bytes(output / name, content, trusted_root=output)
    typer.echo(f"manifest bundle: {output}")


@app.command("preflight")
def preflight_command(
    run_spec: Annotated[Path, typer.Option("--run-spec", help="RunSpec JSON path.")],
    artifact_root: Annotated[
        Path,
        typer.Option("--artifact-root", help="Create-only capsule parent directory."),
    ],
    output: Annotated[Path, typer.Option("--output", help="Resolved fake plan path.")],
) -> None:
    """Resolve the zero-external fake graph before execution."""

    from .runtime.fake_run import resolve_fake_preflight

    run_spec_bytes = run_spec.read_bytes()
    parsed = RunSpec.model_validate_json(run_spec_bytes)
    if parsed != _expected_fake_run_spec(parsed.run_id):
        raise typer.BadParameter("run spec does not exactly match the generated fake workload")
    resolved = resolve_fake_preflight(artifact_root / parsed.run_id, parsed.run_id)
    document = {
        "schema_name": "fake_resolved_plan",
        "schema_version": 1,
        "operation_kind": "fake_run",
        "run_id": parsed.run_id,
        "run_spec_path": str(run_spec.resolve()),
        "run_spec_sha256": hashlib.sha256(run_spec_bytes).hexdigest(),
        "artifact_root": str(artifact_root.resolve()),
        "artifact_durability": _artifact_durability_document(resolved.artifact_durability),
        "plan_hash": resolved.plan_hash,
    }
    content = canonical_json_bytes(document)
    atomic_write_bytes(output, content, trusted_root=output.parent)
    typer.echo(f"resolved plan: {output}")


@app.command("run")
def run_command(
    resolved_plan: Annotated[
        Path,
        typer.Argument(help="Canonical resolved-plan JSON path.", metavar="RESOLVED_PLAN"),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Create-only live capsule parent."),
    ] = None,
    cell: Annotated[
        list[str] | None,
        typer.Option("--cell", help="Ordered frozen cell ID; repeat to select multiple."),
    ] = None,
    case: Annotated[
        list[str] | None,
        typer.Option("--case", help="Ordered case ID; repeat to select whole case-plan groups."),
    ] = None,
    run_label: Annotated[
        str | None,
        typer.Option("--run-label", help="Fresh operator-chosen run label."),
    ] = None,
    continue_from: Annotated[
        Path | None,
        typer.Option(
            "--continue-from",
            help="Aborted OpenViking capsule whose confirmed work must be reused.",
        ),
    ] = None,
    provider_runtime: Annotated[
        Path,
        typer.Option("--provider-runtime", help="Verified provider runtime directory."),
    ] = Path("provider-services/.runtime"),
    provider_env: Annotated[
        Path,
        typer.Option("--provider-env", help="Provider service dotenv file."),
    ] = Path("provider-services/.env"),
    model_env: Annotated[
        Path,
        typer.Option("--model-env", help="Answer and judge dotenv file."),
    ] = Path(".env"),
    scenario: Annotated[
        str,
        typer.Option("--scenario", hidden=True),
    ] = "standard",
) -> None:
    """Execute frozen cells and seal source capsules."""

    document = _load_object(resolved_plan)
    if document.get("schema_name") == "fake_resolved_plan":
        if case:
            raise typer.BadParameter("--case requires a live resolved plan")
        _run_fake_resolved_plan(resolved_plan, scenario=scenario)
        return

    from .config.doctor import ResolvedPlanError, load_resolved_plan_for_run
    from .live import (
        LiveCellExecutionError,
        LiveConfigurationError,
        build_live_cell,
        build_live_continuation_cell,
        execute_live_cells,
        load_live_environment,
        load_live_provider_evidence,
        select_live_cells,
    )

    if output_root is None:
        raise typer.BadParameter("live run requires --output-root")
    if continue_from is None and run_label is None:
        raise typer.BadParameter("fresh live run requires --run-label")
    if continue_from is not None and run_label is not None:
        raise typer.BadParameter("--continue-from and --run-label are mutually exclusive")
    if continue_from is not None and case:
        raise typer.BadParameter("--continue-from and --case are mutually exclusive")
    try:
        plan = load_resolved_plan_for_run(resolved_plan)
        environment = load_live_environment(
            provider_env_path=provider_env,
            model_env_path=model_env,
            provider_runtime_directory=provider_runtime,
            base_environment=os.environ,
        )
        provider_project, evidence_by_provider = load_live_provider_evidence(
            plan=plan,
            provider_runtime_directory=provider_runtime,
            environment=environment,
        )
        selected_cells = select_live_cells(plan, tuple(cell or ()))
        if case and len(selected_cells) != 1:
            raise LiveConfigurationError("case partition requires exactly one selected cell")
        code_revision = _live_source_revision()
        observed_at = datetime.now(UTC)
        built_cells = []
        if continue_from is not None:
            if len(selected_cells) != 1:
                raise LiveConfigurationError("continuation requires exactly one selected cell")
            frozen_cell = selected_cells[0]
            built_cells.append(
                build_live_continuation_cell(
                    plan=plan,
                    cell_id=frozen_cell.cell_id,
                    base_capsule_root=continue_from,
                    output_root=output_root,
                    provider_runtime_directory=provider_runtime.resolve(),
                    provider_project_id=provider_project,
                    provider_evidence=evidence_by_provider[frozen_cell.provider_id],
                    environment=environment,
                )
            )
        else:
            assert run_label is not None
            for frozen_cell in selected_cells:
                built_cells.append(
                    build_live_cell(
                        plan=plan,
                        cell_id=frozen_cell.cell_id,
                        output_root=output_root,
                        provider_runtime_directory=provider_runtime.resolve(),
                        provider_project_id=provider_project,
                        provider_evidence=evidence_by_provider[frozen_cell.provider_id],
                        environment=environment,
                        run_label=run_label,
                        observed_at=observed_at,
                        code_revision=code_revision,
                        requested_case_manifest_entry_ids=tuple(case or ()),
                    )
                )
        for completed in execute_live_cells(tuple(built_cells)):
            typer.echo(f"cell {completed.cell_id}: {completed.capsule_root}")
    except (
        KeyError,
        LiveCellExecutionError,
        LiveConfigurationError,
        ResolvedPlanError,
        ValueError,
    ) as exc:
        raise typer.BadParameter(str(exc)) from exc
    except BaseExceptionGroup as exc:
        raise typer.BadParameter(str(exc)) from exc


def _run_fake_resolved_plan(resolved_plan: Path, *, scenario: str) -> None:
    """Retain the generated zero-external fixture through the generic command."""

    from .runtime.fake_run import FakeRunScenario, resolve_fake_preflight

    try:
        parsed_scenario = FakeRunScenario(scenario)
    except ValueError as exc:
        choices = ", ".join(item.value for item in FakeRunScenario)
        raise typer.BadParameter(f"scenario must be one of: {choices}") from exc

    document = _load_object(resolved_plan)
    _require_exact_keys(
        document,
        {
            "schema_name",
            "schema_version",
            "operation_kind",
            "run_id",
            "run_spec_path",
            "run_spec_sha256",
            "artifact_root",
            "artifact_durability",
            "plan_hash",
        },
    )
    if (
        document["schema_name"] != "fake_resolved_plan"
        or document["schema_version"] != 1
        or document["operation_kind"] != "fake_run"
    ):
        raise typer.BadParameter("unsupported resolved plan")
    run_spec_path = Path(str(document["run_spec_path"]))
    run_spec_bytes = run_spec_path.read_bytes()
    if hashlib.sha256(run_spec_bytes).hexdigest() != document["run_spec_sha256"]:
        raise typer.BadParameter("resolved plan RunSpec hash mismatch")
    run_spec = RunSpec.model_validate_json(run_spec_bytes)
    if run_spec.run_id != document["run_id"]:
        raise typer.BadParameter("resolved plan run identity mismatch")
    if run_spec != _expected_fake_run_spec(run_spec.run_id):
        raise typer.BadParameter("resolved plan RunSpec is not the closed fake configuration")
    artifact_root = Path(str(document["artifact_root"]))
    capsule_root = artifact_root / run_spec.run_id
    captured_durability = _parse_artifact_durability(document["artifact_durability"])
    bound_plan = resolve_fake_preflight(
        capsule_root,
        run_spec.run_id,
        artifact_durability=captured_durability,
    )
    if bound_plan.plan_hash != document["plan_hash"]:
        raise typer.BadParameter("resolved plan hash does not bind its captured proof")
    current = resolve_fake_preflight(capsule_root, run_spec.run_id)
    if (
        current.artifact_durability.artifact_root_fingerprint
        != captured_durability.artifact_root_fingerprint
    ):
        raise typer.BadParameter("artifact root identity changed after preflight")
    completed = run_generated_fake_vertical_slice(
        output_root=artifact_root,
        run_id=run_spec.run_id,
        scenario=parsed_scenario,
    )
    typer.echo(f"capsule: {completed.capsule_root}")


def _live_source_revision() -> str:
    entries = []
    root = Path(__file__).resolve().parents[2]
    for relative_root in (Path("src/oamb"), Path("configs")):
        for path in sorted((root / relative_root).rglob("*")):
            if path.is_file() and path.suffix in {".py", ".yml", ".yaml"}:
                entries.append(
                    (
                        path.relative_to(root).as_posix(),
                        hashlib.sha256(path.read_bytes()).hexdigest(),
                    )
                )
    return canonical_sha256(["oamb-live-source-revision-initial-v1", entries])


@capsule_app.command("validate")
def capsule_validate(
    capsule: Annotated[Path, typer.Argument(help="Sealed source root directory.")],
    output: Annotated[Path, typer.Option("--output", help="Validation result path.")],
    diagnostic: Annotated[
        bool,
        typer.Option("--diagnostic", help="Retain INVALID result without a success exit."),
    ] = False,
) -> None:
    """Reopen and validate one sealed source root."""

    from .artifacts.validation.source_root import validate_source_root

    result = validate_source_root(capsule)
    atomic_write_bytes(output, canonical_json_bytes(result), trusted_root=output.parent)
    typer.echo(f"{result.disposition.value}: {output}")
    if result.disposition != ValidationDisposition.VALIDATED and not diagnostic:
        raise typer.Exit(code=1)


@capsule_app.command("compose")
def capsule_compose(
    plan: Annotated[
        Path,
        typer.Option("--plan", help="Canonical resolved-plan JSON path."),
    ],
    cell: Annotated[
        str,
        typer.Option("--cell", help="Frozen target cell ID."),
    ],
    part: Annotated[
        list[Path],
        typer.Option("--part", help="Sealed part capsule root; repeat for every part."),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", help="Create-only composed capsule root."),
    ],
) -> None:
    """Compose compatible immutable part capsules into one complete cell root."""

    from .artifacts.composition import (
        CapsuleCompositionError,
        CapsuleCompositionTarget,
        compose_capsules,
    )
    from .config.doctor import ResolvedPlanError, load_resolved_plan_for_run
    from .contracts.specifications import INFRASTRUCTURE_RETRY_POLICY_HASH

    try:
        resolved = load_resolved_plan_for_run(plan)
        cells = tuple(item for item in resolved.cells if item.cell_id == cell)
        if len(cells) != 1:
            raise CapsuleCompositionError("unknown composition target cell")
        selected = cells[0]
        composed = compose_capsules(
            tuple(part),
            output,
            target=CapsuleCompositionTarget(
                resolved_plan_hash=resolved.resolved_plan_hash,
                cell_spec_hash=selected.cell_spec_hash,
                target_case_manifest_hash=selected.case_manifest_hash,
                budget_policy_hash=selected.limits_hash,
                retry_policy_hash=INFRASTRUCTURE_RETRY_POLICY_HASH,
            ),
        )
    except (OSError, CapsuleCompositionError, ResolvedPlanError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"capsule: {composed.capsule_root}")


@app.command("compare")
def compare_command(
    resolved_plan: Annotated[
        Path,
        typer.Argument(help="Canonical resolved-plan JSON path.", metavar="RESOLVED_PLAN"),
    ],
    cell_root: Annotated[
        list[str],
        typer.Option("--cell-root", help="CELL_ID=sealed source-root; repeat per cell."),
    ],
    validation: Annotated[
        list[str],
        typer.Option("--validation", help="CELL_ID=validation JSON; repeat per cell."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Create-only comparison report directory."),
    ],
    dataset_source: Annotated[
        Path | None,
        typer.Option(
            "--dataset-source",
            help="Exact frozen dataset file used to add local-only question and answer details.",
        ),
    ] = None,
) -> None:
    """Freshly validate frozen cells and build every pair plus offline report."""

    from .config.doctor import ResolvedPlanError, load_resolved_plan_for_run
    from .reporting.comparison_project import (
        ComparisonProjectError,
        ValidatedCellRoot,
        build_comparison_project,
    )

    try:
        plan = load_resolved_plan_for_run(resolved_plan)
        roots = _named_paths(cell_root, label="cell root")
        validations = _named_paths(validation, label="validation")
        expected = {cell.cell_id for cell in plan.cells}
        if set(roots) != expected or set(validations) != expected:
            raise typer.BadParameter("cell roots and validations must name every frozen cell")
        sources = {
            cell.cell_id: ValidatedCellRoot(
                root=roots[cell.cell_id],
                validation_result=ValidationResult.model_validate_json(
                    validations[cell.cell_id].read_bytes()
                ),
            )
            for cell in plan.cells
        }
        built = build_comparison_project(
            plan,
            sources,
            output_root=output_root,
            dataset_source=dataset_source,
        )
    except (OSError, ComparisonProjectError, ResolvedPlanError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    for comparison_path in built.comparison_paths:
        typer.echo(f"comparison: {comparison_path}")
    typer.echo(f"report export: {built.export_path}")
    typer.echo(f"offline report: {built.html_path}")


def _named_paths(values: list[str], *, label: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values:
        if value.count("=") != 1:
            raise typer.BadParameter(f"{label} must use CELL_ID=PATH")
        identity, raw_path = value.split("=", 1)
        if not identity or not raw_path or identity in paths:
            raise typer.BadParameter(f"{label} identity/path is empty or duplicated")
        paths[identity] = Path(raw_path)
    return paths


@app.command("summarize")
def summarize_command(
    capsule: Annotated[Path, typer.Argument(help="Sealed capsule directory.")],
    validation: Annotated[
        Path,
        typer.Option("--validation", help="EvidenceValidationResult JSON path."),
    ],
    output: Annotated[Path, typer.Option("--output", help="RunSummary JSON path.")],
) -> None:
    """Build a completion-first summary from validated evidence."""

    from .reporting.reduce import reduce_run_summary

    result = ValidationResult.model_validate_json(validation.read_bytes())
    summary = reduce_run_summary(capsule, result)
    atomic_write_bytes(output, canonical_json_bytes(summary), trusted_root=output.parent)
    typer.echo(f"summary: {output}")


@report_app.command("build")
def report_build(
    capsule: Annotated[Path, typer.Argument(help="Sealed source root directory.")],
    validation: Annotated[
        Path,
        typer.Option("--validation", help="EvidenceValidationResult JSON path."),
    ],
    output_root: Annotated[
        Path,
        typer.Option("--output-root", help="Derivation parent directory."),
    ],
    audience: Annotated[
        str,
        typer.Option("--audience", help="Report audience: public or local."),
    ] = "public",
    diagnostic: Annotated[
        bool,
        typer.Option("--diagnostic", help="Build a restricted diagnostic report."),
    ] = False,
) -> None:
    """Build one offline report from a freshly validated source root."""

    from .artifacts.atomic import read_regular_file
    from .artifacts.validation.fake import EvidenceNotValidatedError
    from .reporting.source_root import build_source_root_report

    if audience not in {"public", "local"}:
        raise typer.BadParameter("audience must be 'public' or 'local'")
    result = ValidationResult.model_validate_json(read_regular_file(validation))
    try:
        built = build_source_root_report(
            capsule,
            result,
            output_root=output_root,
            audience=cast(Literal["local", "public"], audience),
            diagnostic=diagnostic,
        )
    except EvidenceNotValidatedError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"report: {built.report_path}")


def _load_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{path} must contain a JSON object")
    return value


def run_generated_fake_vertical_slice(
    *,
    output_root: Path,
    run_id: str,
    scenario: FakeRunScenario | None = None,
) -> FakeRunArtifacts:
    """Compose the concrete generated fakes only at the CLI boundary."""

    from .artifacts.store import ArtifactStore
    from .memory_systems.fake import ScriptedFakeMemorySystem
    from .model_clients.fake import ScriptedFakeModelClient
    from .runtime.fake_run import FakeRunScenario, run_fake_vertical_slice
    from .workloads.fake import GeneratedFakeWorkload

    selected_scenario = FakeRunScenario.STANDARD if scenario is None else scenario

    workload = GeneratedFakeWorkload()

    def memory_factory(
        store: ArtifactStorePort,
        plans: tuple[IngestionPlan, ...],
    ) -> MemorySystemPort:
        return ScriptedFakeMemorySystem(
            store=store,
            delayed_readiness_plan_ids=(plans[0].ingestion_plan_id,),
            partial_ingestion_plan_ids=(plans[1].ingestion_plan_id,),
        )

    return run_fake_vertical_slice(
        output_root=output_root,
        run_id=run_id,
        workload=workload,
        artifact_store_factory=ArtifactStore,
        memory_factory=memory_factory,
        model_factory=ScriptedFakeModelClient,
        scenario=selected_scenario,
    )


def _expected_fake_run_spec(run_id: str) -> RunSpec:
    from .runtime.fake_run import FAKE_RUNTIME_BINDING_HASH, build_fake_run_spec
    from .workloads.fake import GeneratedFakeWorkload

    workload = GeneratedFakeWorkload()
    dataset = workload.resolve_sources()
    case_manifest = workload.build_case_manifest(dataset)
    return build_fake_run_spec(
        run_id=run_id,
        dataset_hash=dataset.manifest_hash,
        case_manifest_hash=case_manifest.manifest_hash,
        runtime_binding_hash=FAKE_RUNTIME_BINDING_HASH,
        workload_id=case_manifest.workload_id,
    )


def _artifact_durability_document(
    value: ArtifactDurabilityPreflight,
) -> dict[str, object]:
    return {
        "artifact_root_fingerprint": value.artifact_root_fingerprint,
        "available_bytes": value.available_bytes,
        "lease_supported": value.lease_supported,
        "file_fsync_supported": value.file_fsync_supported,
        "directory_fsync_supported": value.directory_fsync_supported,
        "no_replace_supported": value.no_replace_supported,
    }


def _parse_artifact_durability(value: object) -> ArtifactDurabilityPreflight:
    from .runtime.preflight import ArtifactDurabilityPreflight

    if not isinstance(value, dict):
        raise typer.BadParameter("resolved plan artifact proof must be an object")
    expected_keys = {
        "artifact_root_fingerprint",
        "available_bytes",
        "lease_supported",
        "file_fsync_supported",
        "directory_fsync_supported",
        "no_replace_supported",
    }
    if set(value) != expected_keys:
        raise typer.BadParameter("resolved plan artifact proof fields do not match")
    fingerprint = value["artifact_root_fingerprint"]
    available_bytes = value["available_bytes"]
    booleans = tuple(
        value[key]
        for key in sorted(expected_keys - {"artifact_root_fingerprint", "available_bytes"})
    )
    if (
        not isinstance(fingerprint, str)
        or len(fingerprint) != 64
        or any(character not in "0123456789abcdef" for character in fingerprint)
        or type(available_bytes) is not int
        or available_bytes <= 0
        or any(type(item) is not bool for item in booleans)
    ):
        raise typer.BadParameter("resolved plan artifact proof has invalid values")
    return ArtifactDurabilityPreflight(
        artifact_root_fingerprint=fingerprint,
        available_bytes=available_bytes,
        lease_supported=value["lease_supported"],
        file_fsync_supported=value["file_fsync_supported"],
        directory_fsync_supported=value["directory_fsync_supported"],
        no_replace_supported=value["no_replace_supported"],
    )


def _require_exact_keys(document: dict[str, object], expected: set[str]) -> None:
    if set(document) != expected:
        raise typer.BadParameter("resolved plan fields do not match the closed T5 format")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
