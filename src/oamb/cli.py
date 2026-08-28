"""Offline-first OAMB command composition root."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Literal, cast

import typer

from .artifacts.atomic import atomic_write_bytes
from .contracts.evidence import ValidationResult
from .contracts.ids import canonical_json_bytes
from .contracts.specifications import RunSpec
from .contracts.states import ValidationDisposition
from .historical_cli import external_app
from .phase_cli import phase_app

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
app.add_typer(phase_app, name="phase")
app.add_typer(external_app, name="external")


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
        typer.Option("--resolved-plan", help="Resolved fake plan JSON path."),
    ],
    scenario: Annotated[
        str,
        typer.Option("--scenario", help="Deterministic fake execution scenario."),
    ] = "standard",
) -> None:
    """Execute the resolved fake plan and seal a source capsule."""

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


@capsule_app.command("validate")
def capsule_validate(
    capsule: Annotated[Path, typer.Argument(help="Sealed capsule directory.")],
    output: Annotated[Path, typer.Option("--output", help="Validation result path.")],
    diagnostic: Annotated[
        bool,
        typer.Option("--diagnostic", help="Retain INVALID result without a success exit."),
    ] = False,
) -> None:
    """Execute the closed fake evidence profile."""

    from .artifacts.validation.fake import validate_fake_capsule

    result = validate_fake_capsule(capsule)
    atomic_write_bytes(output, canonical_json_bytes(result), trusted_root=output.parent)
    typer.echo(f"{result.disposition.value}: {output}")
    if result.disposition != ValidationDisposition.VALIDATED and not diagnostic:
        raise typer.Exit(code=1)


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
    capsule: Annotated[Path, typer.Argument(help="Sealed capsule directory.")],
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
    """Render, export-validate, and seal one offline report."""

    from .reporting.html import build_fake_report

    if audience not in {"public", "local"}:
        raise typer.BadParameter("audience must be 'public' or 'local'")
    result = ValidationResult.model_validate_json(validation.read_bytes())
    built = build_fake_report(
        capsule,
        result,
        output_root=output_root,
        audience=cast(Literal["local", "public"], audience),
        diagnostic=diagnostic,
    )
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
