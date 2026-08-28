"""Explicit registry and fail-closed parser for versioned persisted contracts."""

from __future__ import annotations

import json
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
