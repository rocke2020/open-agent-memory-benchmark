"""Explicit registry and fail-closed parser for versioned persisted contracts."""

from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import BaseModel, RootModel

from .accounting import (
    CostMeasurementSpec,
    CostRecord,
    CostRecordV2,
    MeasurementDimensionSpec,
    PriceSnapshot,
    ResourceUsageRecord,
    ResourceUsageRecordV2,
    TokenUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
    TokenUsageRecordV4,
    TokenUsageRecordV5,
)
from .evidence import (
    AttemptIntentRecord,
    AttemptIntentRecordV2,
    AttemptIntentRecordV3,
    AttemptReceiptRecord,
    AttemptRecord,
    AttemptRecordV2,
    AttemptRecordV3,
    AttemptRecordV4,
    BudgetOwnerAllocation,
    BudgetReservationRecord,
    BudgetReservationRecordV2,
    BudgetReservationRecordV3,
    CapsuleManifest,
    CapsuleManifestEntry,
    CaseRecord,
    CaseRecordV2,
    CaseRecordV3,
    CheckpointManifest,
    CloseErrorRecord,
    DerivationManifest,
    HistoryAttemptRecord,
    InfrastructureRetryEvent,
    IngestionPlanRecord,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
    LogicalContextRecord,
    MemoryConformanceEvidenceManifest,
    MemoryConformanceOccurrenceRecord,
    ModelReadinessOccurrenceRecord,
    OccurrenceClaimRecord,
    OriginRecord,
    ProviderServiceEvidenceManifest,
    RawReference,
    RunLeaseHeartbeatRecord,
    RunLeaseRecord,
    RunRecord,
    ValidationIssue,
    ValidationResult,
)
from .external import (
    ExternalCompatibilityAssessment,
    ExternalHistoricalAggregate,
    ExternalHistoricalCase,
    ExternalHistoricalCategoryAggregate,
    ExternalHistoricalEvidence,
    ExternalHistoricalEvidenceReport,
    ExternalTransformationRecord,
)
from .reporting import (
    AggregateMetricDelta,
    ComparisonControlBinding,
    ComparisonControlProvenanceBinding,
    ComparisonControlSnapshot,
    ComparisonControlSnapshotV2,
    ComparisonControlSourceReference,
    ComparisonCostDelta,
    ComparisonPredicateResult,
    ComparisonReportModel,
    ComparisonReportPayload,
    CompletionSummaryV3,
    ControlledEmbeddingComparisonProjection,
    DiagnosticRunReportModel,
    DisplayPreview,
    EvaluationReportModel,
    ExactRational,
    Mab65ReportReduction,
    MabCapabilityMetricSummary,
    MabComponentMetricSummary,
    MabPlanEvidenceBinding,
    MabPlanMetricSummary,
    MeasurementSummaryLine,
    MetricSummary,
    PairedMetricDelta,
    ReducerBinding,
    ReleaseReportModel,
    ReportArtifactManifest,
    ReportArtifactManifestV2,
    ReportArtifactManifestV3,
    ReportRecordProjection,
    RunComparisonControlBasisRecord,
    RunReportModel,
    RunReportModelV2,
    RunReportModelV3,
    RunSummary,
    RunSummaryV2,
    RuntimeMeasurementControlRecord,
    ValidationClaimBoundary,
    WorkloadExecutionControlRecord,
)
from .specifications import (
    BudgetSpec,
    BudgetSpecV2,
    BudgetSpecV3,
    BudgetSpecV4,
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
    DerivationSpecV3,
    DispatchBudgetRoute,
    ExecutionEnvironmentBinding,
    IngestionPlanManifest,
    InteractionSpec,
    LogicalContextManifestEntry,
    MemoryConformanceSpec,
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
    ProviderOperationBudgetCeiling,
    ProviderRuntimeProfileAttestation,
    ReportIdentitySpecBinding,
    ReportIdentitySpecBindingV2,
    ReportSpec,
    ReportSpecV2,
    ResourceBudgetCeiling,
    RoleBudgetCeiling,
    RunPreflightRecord,
    RunPreflightRecordV2,
    RunSpec,
    SourceEvidenceBinding,
    ValidationProfile,
    ValidationRuleRequirement,
    WorkloadSpec,
)


class ComparisonReportContract(RootModel[ComparisonReportPayload]):
    """Registry-only root preserving both comparison-report wire variants."""

    contract_name: ClassVar[str] = "comparison_report"
    contract_version: ClassVar[int] = 1


VERSIONED_CONTRACTS: tuple[type[BaseModel], ...] = (
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
    BudgetSpecV3,
    BudgetSpecV4,
    ResourceBudgetCeiling,
    ProviderBudgetCap,
    ProviderOperationBudgetCeiling,
    DispatchBudgetRoute,
    RoleBudgetCeiling,
    ProviderRuntimeProfileAttestation,
    SourceEvidenceBinding,
    DerivationSpec,
    DerivationSpecV2,
    DerivationSpecV3,
    ReportSpec,
    ReportSpecV2,
    ReportIdentitySpecBinding,
    ReportIdentitySpecBindingV2,
    ComparisonPairBinding,
    ComparisonCostControl,
    ComparisonWinnerReducer,
    ComparisonSpec,
    RunSpec,
    RunPreflightRecord,
    RunPreflightRecordV2,
    ValidationRuleRequirement,
    ValidationProfile,
    RawReference,
    OriginRecord,
    RunRecord,
    AttemptRecord,
    AttemptRecordV2,
    AttemptRecordV3,
    AttemptRecordV4,
    InfrastructureRetryEvent,
    HistoryAttemptRecord,
    RunLeaseRecord,
    RunLeaseHeartbeatRecord,
    OccurrenceClaimRecord,
    ModelReadinessOccurrenceRecord,
    MemoryConformanceOccurrenceRecord,
    MemoryConformanceSpec,
    BudgetReservationRecord,
    BudgetReservationRecordV2,
    BudgetOwnerAllocation,
    BudgetReservationRecordV3,
    AttemptIntentRecord,
    AttemptIntentRecordV2,
    AttemptIntentRecordV3,
    AttemptReceiptRecord,
    LogicalContextRecord,
    IngestionPlanRecord,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
    CaseRecord,
    CaseRecordV2,
    CaseRecordV3,
    CapsuleManifestEntry,
    CapsuleManifest,
    CheckpointManifest,
    ProviderServiceEvidenceManifest,
    MemoryConformanceEvidenceManifest,
    CloseErrorRecord,
    DerivationManifest,
    ValidationIssue,
    ValidationResult,
    ExternalHistoricalCase,
    ExternalHistoricalCategoryAggregate,
    ExternalHistoricalAggregate,
    ExternalCompatibilityAssessment,
    ExternalTransformationRecord,
    ExternalHistoricalEvidence,
    ExternalHistoricalEvidenceReport,
    MeasurementDimensionSpec,
    CostMeasurementSpec,
    PriceSnapshot,
    TokenUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
    TokenUsageRecordV4,
    TokenUsageRecordV5,
    ResourceUsageRecord,
    ResourceUsageRecordV2,
    CostRecord,
    CostRecordV2,
    RunSummary,
    RunSummaryV2,
    ComparisonControlBinding,
    ComparisonControlSourceReference,
    ComparisonControlProvenanceBinding,
    WorkloadExecutionControlRecord,
    ControlledEmbeddingComparisonProjection,
    RuntimeMeasurementControlRecord,
    RunComparisonControlBasisRecord,
    ComparisonControlSnapshot,
    ComparisonControlSnapshotV2,
    DisplayPreview,
    EvaluationReportModel,
    ExactRational,
    PairedMetricDelta,
    ComparisonPredicateResult,
    ComparisonCostDelta,
    AggregateMetricDelta,
    ComparisonReportContract,
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
    ReportArtifactManifest,
    ReportArtifactManifestV2,
    ReportArtifactManifestV3,
)


def _registry_key(model: type[BaseModel]) -> tuple[str, int]:
    name_field = model.model_fields.get("schema_name")
    version_field = model.model_fields.get("schema_version")
    name_value = (
        name_field.default if name_field is not None else getattr(model, "contract_name", None)
    )
    version_value = (
        version_field.default
        if version_field is not None
        else getattr(model, "contract_version", None)
    )
    if not isinstance(name_value, str) or not isinstance(version_value, int):
        raise TypeError(f"invalid versioned contract identity: {model.__name__}")
    return name_value, version_value


CONTRACT_REGISTRY: dict[tuple[str, int], type[BaseModel]] = {
    _registry_key(model): model for model in VERSIONED_CONTRACTS
}
if len(CONTRACT_REGISTRY) != len(VERSIONED_CONTRACTS):
    raise RuntimeError("duplicate versioned contract identity")


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
