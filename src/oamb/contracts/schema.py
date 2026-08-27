"""Explicit public-contract registry and deterministic JSON Schema generation."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .accounting import (
    CostMeasurementSpec,
    CostRecord,
    MeasurementDimensionSpec,
    PriceSnapshot,
    ResourceUsageRecord,
    TokenUsageRecord,
    TokenUsageRecordV2,
)
from .evidence import (
    AttemptIntentRecord,
    AttemptReceiptRecord,
    AttemptRecord,
    AttemptRecordV2,
    BudgetReservationRecord,
    CapsuleManifest,
    CapsuleManifestEntry,
    CaseRecord,
    CheckpointManifest,
    CloseErrorRecord,
    DerivationManifest,
    IngestionPlanRecord,
    LogicalContextRecord,
    ModelReadinessOccurrenceRecord,
    OccurrenceClaimRecord,
    OriginRecord,
    ProviderServiceEvidenceManifest,
    RawReference,
    RecoveryDecisionRecord,
    RunLeaseHeartbeatRecord,
    RunLeaseRecord,
    RunRecord,
    ValidationIssue,
    ValidationResult,
)
from .reporting import ReportArtifactManifest, RunReportModel, RunSummary
from .specifications import (
    BudgetSpec,
    BudgetSpecV2,
    CaseManifest,
    CaseManifestEntry,
    DatasetFile,
    DatasetManifest,
    DerivationSpec,
    ExecutionEnvironmentBinding,
    ExternalCallApprovalRecord,
    IngestionPlanManifest,
    InteractionSpec,
    LogicalContextManifestEntry,
    MemorySystemRuntimeBinding,
    MemorySystemRuntimeBindingV2,
    MemorySystemSpec,
    ModelRoleBinding,
    ModelRoleBindingV2,
    ProtocolSpec,
    ProviderBudgetCap,
    ProviderRuntimeProfileAttestation,
    ResourceBudgetCeiling,
    RoleBudgetCeiling,
    RunSpec,
    SourceEvidenceBinding,
    ValidationProfile,
    ValidationRuleRequirement,
    WorkloadSpec,
)

PUBLIC_CONTRACTS: tuple[type[BaseModel], ...] = (
    ProtocolSpec,
    DatasetFile,
    DatasetManifest,
    LogicalContextManifestEntry,
    CaseManifestEntry,
    IngestionPlanManifest,
    CaseManifest,
    InteractionSpec,
    WorkloadSpec,
    MemorySystemSpec,
    MemorySystemRuntimeBinding,
    MemorySystemRuntimeBindingV2,
    ExecutionEnvironmentBinding,
    ModelRoleBinding,
    ModelRoleBindingV2,
    BudgetSpec,
    BudgetSpecV2,
    ResourceBudgetCeiling,
    ProviderBudgetCap,
    RoleBudgetCeiling,
    ProviderRuntimeProfileAttestation,
    ExternalCallApprovalRecord,
    SourceEvidenceBinding,
    DerivationSpec,
    RunSpec,
    ValidationRuleRequirement,
    ValidationProfile,
    RawReference,
    OriginRecord,
    RunRecord,
    AttemptRecord,
    AttemptRecordV2,
    RunLeaseRecord,
    RunLeaseHeartbeatRecord,
    OccurrenceClaimRecord,
    ModelReadinessOccurrenceRecord,
    BudgetReservationRecord,
    AttemptIntentRecord,
    AttemptReceiptRecord,
    LogicalContextRecord,
    IngestionPlanRecord,
    CaseRecord,
    CapsuleManifestEntry,
    CapsuleManifest,
    CheckpointManifest,
    ProviderServiceEvidenceManifest,
    RecoveryDecisionRecord,
    CloseErrorRecord,
    DerivationManifest,
    ValidationIssue,
    ValidationResult,
    MeasurementDimensionSpec,
    CostMeasurementSpec,
    PriceSnapshot,
    TokenUsageRecord,
    TokenUsageRecordV2,
    ResourceUsageRecord,
    CostRecord,
    RunSummary,
    RunReportModel,
    ReportArtifactManifest,
)


def _registry_key(model: type[BaseModel]) -> tuple[str, int]:
    name_value = model.model_fields["schema_name"].default
    version_value = model.model_fields["schema_version"].default
    if not isinstance(name_value, str) or not isinstance(version_value, int):
        raise TypeError(f"invalid public contract identity: {model.__name__}")
    return name_value, version_value


CONTRACT_REGISTRY: dict[tuple[str, int], type[BaseModel]] = {
    _registry_key(model): model for model in PUBLIC_CONTRACTS
}
if len(CONTRACT_REGISTRY) != len(PUBLIC_CONTRACTS):
    raise RuntimeError("duplicate public contract schema identity")


class UnknownContractError(ValueError):
    pass


def parse_contract(document: dict[str, Any]) -> BaseModel:
    name = document.get("schema_name")
    version = document.get("schema_version")
    if not isinstance(name, str) or type(version) is not int:
        raise UnknownContractError(f"unsupported contract: {name!r} v{version!r}")
    try:
        model = CONTRACT_REGISTRY[(name, version)]
    except KeyError as exc:
        raise UnknownContractError(f"unsupported contract: {name!r} v{version!r}") from exc
    return model.model_validate_json(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    )


def schema_filename(model: type[BaseModel]) -> str:
    name, version = _registry_key(model)
    return f"{name}.v{version}.schema.json"


def schema_bytes(model: type[BaseModel]) -> bytes:
    name, version = _registry_key(model)
    document = model.model_json_schema(mode="validation")
    _require_discriminators(document)
    document["$id"] = f"https://open-agent-memory-benchmark.dev/schemas/{name}/v{version}"
    document["title"] = name
    return (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def _require_discriminators(value: Any) -> None:
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict) and {"schema_name", "schema_version"} <= set(properties):
            required = set(value.get("required", ()))
            required.update(("schema_name", "schema_version"))
            value["required"] = sorted(required)
        for nested in value.values():
            _require_discriminators(nested)
    elif isinstance(value, list):
        for nested in value:
            _require_discriminators(nested)


def expected_schema_files() -> dict[str, bytes]:
    return {schema_filename(model): schema_bytes(model) for model in PUBLIC_CONTRACTS}


def generate_schemas(output_directory: Path) -> tuple[Path, ...]:
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in sorted(expected_schema_files().items()):
        path = output_directory / name
        path.write_bytes(content)
        written.append(path)
    return tuple(written)


def schema_drift(schema_directory: Path) -> tuple[str, ...]:
    expected = expected_schema_files()
    actual_names = (
        {path.name for path in schema_directory.iterdir() if path.is_file()}
        if schema_directory.is_dir()
        else set()
    )
    drift: list[str] = []
    for name, content in sorted(expected.items()):
        path = schema_directory / name
        if not path.exists():
            drift.append(f"missing:{name}")
        elif path.read_bytes() != content:
            drift.append(f"changed:{name}")
    for name in sorted(actual_names - set(expected)):
        drift.append(f"extra:{name}")
    return tuple(drift)


def packaged_schema_names() -> tuple[str, ...]:
    package_directory = resources.files("oamb").joinpath("schemas")
    if package_directory.is_dir():
        return tuple(sorted(item.name for item in package_directory.iterdir() if item.is_file()))
    source_directory = Path(__file__).resolve().parents[3] / "schemas"
    if source_directory.is_dir():
        return tuple(sorted(path.name for path in source_directory.iterdir() if path.is_file()))
    return ()
