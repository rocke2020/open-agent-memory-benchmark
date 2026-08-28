"""Explicit public-contract registry and deterministic JSON Schema generation."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, RootModel

from .accounting import (
    CostMeasurementSpec,
    CostRecord,
    MeasurementDimensionSpec,
    PriceSnapshot,
    ResourceUsageRecord,
    TokenUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
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
    CaseRecordV2,
    CaseRecordV3,
    CheckpointManifest,
    CloseErrorRecord,
    DerivationManifest,
    IngestionPlanRecord,
    IngestionPlanRecordV2,
    LogicalContextRecord,
    ModelReadinessOccurrenceRecord,
    OccurrenceClaimRecord,
    OriginRecord,
    PhaseReviewOccurrenceRecord,
    ProviderServiceEvidenceManifest,
    RawReference,
    RecoveryDecisionRecord,
    RunLeaseHeartbeatRecord,
    RunLeaseRecord,
    RunRecord,
    ValidationIssue,
    ValidationResult,
)
from .reporting import (
    AggregateMetricDelta,
    AIQualityReviewRecord,
    AIReviewBatchResult,
    AIReviewCaseProjection,
    AIReviewCaseResult,
    AIReviewFinding,
    AIReviewIntegrityProjection,
    AIReviewIntegrityResult,
    ComparisonControlBinding,
    ComparisonControlSnapshot,
    ComparisonCostDelta,
    ComparisonPredicateResult,
    ComparisonReportModel,
    ComparisonReportPayload,
    CompletionSummaryV3,
    DiagnosticRunReportModel,
    DisplayPreview,
    EvaluationPhaseGate,
    EvaluationReviewBundle,
    ExactRational,
    HumanQualityReviewRecord,
    HumanReviewDecision,
    Mab65ReportReduction,
    MabCapabilityMetricSummary,
    MabComponentMetricSummary,
    MabPlanEvidenceBinding,
    MabPlanMetricSummary,
    MeasurementSummaryLine,
    MetricSummary,
    PairedMetricDelta,
    PhaseAcceptanceReport,
    ReducerBinding,
    ReleaseReportModel,
    ReportArtifactManifest,
    ReportArtifactManifestV2,
    ReportRecordProjection,
    RunReportModel,
    RunReportModelV2,
    RunReportModelV3,
    RunSummary,
    RunSummaryV2,
    SignatureVerificationRecord,
    ValidationClaimBoundary,
)
from .specifications import (
    AcceptanceReportSpec,
    AIReviewBatch,
    AIReviewPlan,
    BudgetSpec,
    BudgetSpecV2,
    CaseManifest,
    CaseManifestEntry,
    ComparisonCostControl,
    ComparisonPairBinding,
    ComparisonSpec,
    ComparisonWinnerReducer,
    DatasetFile,
    DatasetManifest,
    DerivationSpec,
    DerivationSpecV2,
    ExecutionEnvironmentBinding,
    ExternalCallApprovalRecord,
    HumanReviewKeyBinding,
    IngestionPlanManifest,
    InteractionSpec,
    LogicalContextManifestEntry,
    MemorySystemRuntimeBinding,
    MemorySystemRuntimeBindingV2,
    MemorySystemSpec,
    MetricSpec,
    ModelRoleBinding,
    ModelRoleBindingV2,
    OutputContract,
    PromptPackManifest,
    ProtocolSpec,
    ProviderBudgetCap,
    ProviderRuntimeProfileAttestation,
    ReportIdentitySpecBinding,
    ReportSpec,
    ResourceBudgetCeiling,
    RoleBudgetCeiling,
    RunSpec,
    SourceEvidenceBinding,
    ValidationProfile,
    ValidationRuleRequirement,
    WorkloadSpec,
)


class ComparisonReportContract(RootModel[ComparisonReportPayload]):
    """Registry-only root preserving both comparison-report wire variants."""

    public_schema_name: ClassVar[str] = "comparison_report"
    public_schema_version: ClassVar[int] = 1


PUBLIC_CONTRACTS: tuple[type[BaseModel], ...] = (
    ProtocolSpec,
    DatasetFile,
    DatasetManifest,
    PromptPackManifest,
    OutputContract,
    MetricSpec,
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
    HumanReviewKeyBinding,
    SourceEvidenceBinding,
    DerivationSpec,
    DerivationSpecV2,
    ReportSpec,
    AcceptanceReportSpec,
    ReportIdentitySpecBinding,
    ComparisonPairBinding,
    ComparisonCostControl,
    ComparisonWinnerReducer,
    ComparisonSpec,
    AIReviewBatch,
    AIReviewPlan,
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
    IngestionPlanRecordV2,
    CaseRecord,
    CaseRecordV2,
    CaseRecordV3,
    PhaseReviewOccurrenceRecord,
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
    TokenUsageRecordV3,
    ResourceUsageRecord,
    CostRecord,
    RunSummary,
    RunSummaryV2,
    ComparisonControlBinding,
    ComparisonControlSnapshot,
    DisplayPreview,
    ExactRational,
    PairedMetricDelta,
    ComparisonPredicateResult,
    ComparisonCostDelta,
    AggregateMetricDelta,
    ComparisonReportContract,
    EvaluationReviewBundle,
    AIReviewCaseProjection,
    AIReviewIntegrityProjection,
    AIReviewFinding,
    AIReviewCaseResult,
    AIReviewBatchResult,
    AIReviewIntegrityResult,
    AIQualityReviewRecord,
    HumanReviewDecision,
    SignatureVerificationRecord,
    HumanQualityReviewRecord,
    EvaluationPhaseGate,
    CompletionSummaryV3,
    ValidationClaimBoundary,
    ReportRecordProjection,
    MetricSummary,
    ReducerBinding,
    MabPlanEvidenceBinding,
    MabPlanMetricSummary,
    MabComponentMetricSummary,
    MabCapabilityMetricSummary,
    Mab65ReportReduction,
    MeasurementSummaryLine,
    RunReportModel,
    RunReportModelV2,
    RunReportModelV3,
    DiagnosticRunReportModel,
    ComparisonReportModel,
    ReleaseReportModel,
    PhaseAcceptanceReport,
    ReportArtifactManifest,
    ReportArtifactManifestV2,
)


def _registry_key(model: type[BaseModel]) -> tuple[str, int]:
    name_field = model.model_fields.get("schema_name")
    version_field = model.model_fields.get("schema_version")
    name_value = (
        name_field.default if name_field is not None else getattr(model, "public_schema_name", None)
    )
    version_value = (
        version_field.default
        if version_field is not None
        else getattr(model, "public_schema_version", None)
    )
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
    parsed = model.model_validate_json(
        json.dumps(document, ensure_ascii=False, separators=(",", ":"))
    )
    if isinstance(parsed, ComparisonReportContract):
        return parsed.root
    return parsed


def schema_filename(model: type[BaseModel]) -> str:
    name, version = _registry_key(model)
    return f"{name}.v{version}.schema.json"


def schema_bytes(model: type[BaseModel]) -> bytes:
    name, version = _registry_key(model)
    document = model.model_json_schema(mode="validation")
    _require_discriminators(document)
    if "properties" not in document:
        document["type"] = "object"
        document["properties"] = {
            "schema_name": {"const": name, "default": name, "type": "string"},
            "schema_version": {"const": version, "default": version, "type": "integer"},
        }
        document["required"] = ["schema_name", "schema_version"]
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
