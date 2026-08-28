"""Strict control-plane and manifest contracts used by the offline kernel."""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal, Self

from pydantic import model_validator

from oamb.constants import (
    AI_REVIEW_MAX_CASES_PER_BATCH,
    AI_REVIEW_MAX_INPUT_BYTES,
    AI_REVIEW_MAX_INPUT_TOKENS,
    AI_REVIEW_MAX_OUTPUT_BYTES,
    AI_REVIEW_MAX_OUTPUT_TOKENS,
)

from .base import (
    NonEmptyStr,
    NonNegativeDecimal,
    NonNegativeInt,
    PositiveInt,
    Sha256,
    StrictContract,
    UtcDateTime,
)
from .ids import (
    canonical_sha256,
    case_manifest_entry_id,
    ingestion_payload_hash,
    ingestion_plan_id,
    plan_manifest_entry_id,
)


class ValidationStage(StrEnum):
    EVIDENCE = "evidence"
    EXPORT = "export"


class BudgetScopeKind(StrEnum):
    RUN = "run"
    PHASE_REVIEW = "phase_review"


class BudgetScopeKindV2(StrEnum):
    RUN = "run"
    PHASE_REVIEW = "phase_review"
    MODEL_READINESS = "model_readiness"


class TransportProfile(StrEnum):
    REST = "rest"
    SDK = "sdk"
    FAKE = "fake"


class AttestationStatus(StrEnum):
    ATTESTED = "attested"
    UNATTESTED = "unattested"
    UNSUPPORTED = "unsupported"


class ComparabilityStatus(StrEnum):
    COMPARABLE = "comparable"
    NOT_COMPARABLE = "not_comparable"
    UNKNOWN = "unknown"


class ModelRole(StrEnum):
    MEMORY_EXTRACTION = "memory_extraction"
    EMBEDDING = "embedding"
    ANSWER = "answer"
    JUDGE = "judge"
    QUALITY_REVIEW = "quality_review"


class RoleBindingStatus(StrEnum):
    SELECTED = "selected"
    DISABLED = "disabled"
    NOT_APPLICABLE = "not_applicable"


class ExecutionOwner(StrEnum):
    MEMORY_SYSTEM = "memory_system"
    HARNESS = "harness"
    DETERMINISTIC_METRIC = "deterministic_metric"


class BindingKind(StrEnum):
    NATIVE = "native"
    MODEL_CLIENT = "model_client"
    DISABLED = "disabled"
    NOT_APPLICABLE = "not_applicable"


class RuntimeAttestationStatus(StrEnum):
    RUNTIME_VERIFIED = "runtime_verified"
    BUILD_PROVENANCE_VERIFIED = "build_provenance_verified"
    UNATTESTED = "unattested"
    UNSUPPORTED = "unsupported"


class ProviderGateStatus(StrEnum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    UNSUPPORTED = "unsupported"


class SourceEvidenceKind(StrEnum):
    RUN = "run"
    EXTERNAL = "external"
    DERIVATION = "derivation"
    PROVIDER_SERVICE = "provider_service"


class ComparisonCostView(StrEnum):
    NONE = "none"
    RAW_RESOURCE = "raw_resource"
    ACTUAL_CHARGE = "actual_charge"
    DECLARED_RATE_ESTIMATE = "declared_rate_estimate"


class HumanReviewKeyBinding(StrictContract):
    schema_name: Literal["human_review_key_binding"] = "human_review_key_binding"
    schema_version: Literal[1] = 1
    key_binding_id: Sha256
    algorithm: Literal["ed25519"] = "ed25519"
    source_kind: Literal["raw", "file"]
    source_reference: NonEmptyStr
    public_key_base64: NonEmptyStr
    public_key_sha256: Sha256
    fingerprint: NonEmptyStr

    @model_validator(mode="after")
    def public_key_and_binding_identity_are_exact(self) -> Self:
        try:
            public_key = base64.b64decode(self.public_key_base64, validate=True)
        except Exception as exc:
            raise ValueError("human review public key is not canonical base64") from exc
        if len(public_key) != 32:
            raise ValueError("Ed25519 public key must contain exactly 32 bytes")
        if base64.b64encode(public_key).decode("ascii") != self.public_key_base64:
            raise ValueError("human review public key must use canonical base64")
        public_key_sha256 = hashlib.sha256(public_key).hexdigest()
        if self.public_key_sha256 != public_key_sha256:
            raise ValueError("human review public-key hash does not match its bytes")
        if self.fingerprint != f"SHA256:{public_key_sha256}":
            raise ValueError("human review key fingerprint does not match its bytes")
        expected = human_review_key_binding_id(
            source_kind=self.source_kind,
            source_reference=self.source_reference,
            public_key_base64=self.public_key_base64,
            public_key_sha256=self.public_key_sha256,
            fingerprint=self.fingerprint,
        )
        if self.key_binding_id != expected:
            raise ValueError("human review key binding identity does not match its payload")
        return self


def human_review_key_binding_id(
    *,
    source_kind: str,
    source_reference: str,
    public_key_base64: str,
    public_key_sha256: str,
    fingerprint: str,
) -> str:
    return canonical_sha256(
        [
            "oamb-human-review-key-binding-v1",
            source_kind,
            source_reference,
            public_key_base64,
            public_key_sha256,
            fingerprint,
        ]
    )


class ComparisonPairBinding(StrictContract):
    schema_name: Literal["comparison_pair_binding"] = "comparison_pair_binding"
    schema_version: Literal[1] = 1
    pair_id: Sha256
    case_manifest_entry_id: Sha256
    left_case_occurrence_id: Sha256
    right_case_occurrence_id: Sha256
    metric_id: NonEmptyStr

    @model_validator(mode="after")
    def pair_identity_is_canonical(self) -> Self:
        expected = comparison_pair_id(
            case_manifest_entry_id=self.case_manifest_entry_id,
            left_case_occurrence_id=self.left_case_occurrence_id,
            right_case_occurrence_id=self.right_case_occurrence_id,
            metric_id=self.metric_id,
        )
        if self.pair_id != expected:
            raise ValueError("comparison pair identity does not match its canonical payload")
        return self


class ComparisonCostControl(StrictContract):
    schema_name: Literal["comparison_cost_control"] = "comparison_cost_control"
    schema_version: Literal[1] = 1
    view: ComparisonCostView
    dimension_id: NonEmptyStr
    unit: NonEmptyStr
    basis: Literal["raw_resource", "actual_supplier_charge", "estimate_from_measured_usage"]
    currency: str | None
    environment_hash: Sha256
    measurement_spec_hash: Sha256
    price_policy_hash: Sha256 | None
    fx_policy_hash: Sha256 | None

    @model_validator(mode="after")
    def selected_cost_view_has_one_compatible_basis(self) -> Self:
        if self.view == ComparisonCostView.NONE:
            raise ValueError("none cost view cannot contain a cost control")
        if self.view == ComparisonCostView.RAW_RESOURCE:
            if (
                self.basis != "raw_resource"
                or self.currency is not None
                or self.price_policy_hash is not None
                or self.fx_policy_hash is not None
            ):
                raise ValueError("raw-resource comparison cannot contain monetary policy")
        else:
            expected_basis = (
                "actual_supplier_charge"
                if self.view == ComparisonCostView.ACTUAL_CHARGE
                else "estimate_from_measured_usage"
            )
            if self.basis != expected_basis:
                raise ValueError("cost view and monetary basis do not match")
            if self.currency is None or len(self.currency) != 3 or self.price_policy_hash is None:
                raise ValueError("monetary comparison requires currency and price policy")
        return self


class ComparisonWinnerReducer(StrictContract):
    schema_name: Literal["comparison_winner_reducer"] = "comparison_winner_reducer"
    schema_version: Literal[1] = 1
    reducer_id: NonEmptyStr
    reducer_version: PositiveInt
    metric_id: NonEmptyStr
    denominator: PositiveInt
    aggregate_contract_hash: Sha256


class ComparisonSpec(StrictContract):
    schema_name: Literal["comparison_spec"] = "comparison_spec"
    schema_version: Literal[1] = 1
    comparison_spec_id: Sha256
    left_run_id: NonEmptyStr
    right_run_id: NonEmptyStr
    cost_view: ComparisonCostView
    ordered_pair_bindings: tuple[ComparisonPairBinding, ...]
    required_control_ids: tuple[NonEmptyStr, ...]
    comparison_policy_hash: Sha256
    winner_reducer: ComparisonWinnerReducer | None
    cost_control: ComparisonCostControl | None

    @model_validator(mode="after")
    def comparison_scope_and_identity_are_closed(self) -> Self:
        if self.left_run_id == self.right_run_id:
            raise ValueError("comparison requires two distinct runs")
        if not self.ordered_pair_bindings:
            raise ValueError("comparison spec requires an ordered pair inventory")
        pair_ids = tuple(item.pair_id for item in self.ordered_pair_bindings)
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("duplicate comparison pair identity")
        if not self.required_control_ids or len(set(self.required_control_ids)) != len(
            self.required_control_ids
        ):
            raise ValueError("comparison spec requires a unique closed control inventory")
        if (self.cost_view == ComparisonCostView.NONE) != (self.cost_control is None):
            raise ValueError("selected comparison cost view requires exactly one cost control")
        if self.cost_control is not None and self.cost_control.view != self.cost_view:
            raise ValueError("comparison cost control does not match the selected view")
        expected = comparison_spec_id(
            left_run_id=self.left_run_id,
            right_run_id=self.right_run_id,
            cost_view=self.cost_view,
            ordered_pair_bindings=self.ordered_pair_bindings,
            required_control_ids=self.required_control_ids,
            comparison_policy_hash=self.comparison_policy_hash,
            winner_reducer=self.winner_reducer,
            cost_control=self.cost_control,
        )
        if self.comparison_spec_id != expected:
            raise ValueError("comparison spec identity does not match its canonical payload")
        return self


class AIReviewBatch(StrictContract):
    schema_name: Literal["ai_review_batch"] = "ai_review_batch"
    schema_version: Literal[1] = 1
    batch_id: Sha256
    ordered_case_occurrence_ids: tuple[Sha256, ...]
    payload_hash: Sha256
    request_fingerprint: Sha256
    input_bytes: PositiveInt
    input_tokens: PositiveInt
    maximal_output_bytes: PositiveInt
    maximal_output_tokens: PositiveInt

    @model_validator(mode="after")
    def batch_is_nonempty_and_unique(self) -> Self:
        if not self.ordered_case_occurrence_ids:
            raise ValueError("AI review batch cannot be empty")
        if len(set(self.ordered_case_occurrence_ids)) != len(self.ordered_case_occurrence_ids):
            raise ValueError("AI review batch contains duplicate case coverage")
        if len(self.ordered_case_occurrence_ids) > AI_REVIEW_MAX_CASES_PER_BATCH:
            raise ValueError("AI review batch exceeds the whole-case limit")
        if self.input_bytes > AI_REVIEW_MAX_INPUT_BYTES:
            raise ValueError("AI review batch exceeds the input-byte limit")
        if self.input_tokens > AI_REVIEW_MAX_INPUT_TOKENS:
            raise ValueError("AI review batch exceeds the input-token limit")
        if self.maximal_output_bytes > AI_REVIEW_MAX_OUTPUT_BYTES:
            raise ValueError("AI review batch exceeds the output-byte limit")
        if self.maximal_output_tokens > AI_REVIEW_MAX_OUTPUT_TOKENS:
            raise ValueError("AI review batch exceeds the output-token limit")
        return self


class AIReviewPlan(StrictContract):
    schema_name: Literal["ai_review_plan"] = "ai_review_plan"
    schema_version: Literal[1] = 1
    review_bundle_hash: Sha256
    projection_spec_hash: Sha256
    finding_registry_hash: Sha256
    prompt_pack_hash: Sha256
    output_contract_hash: Sha256
    parser_hash: Sha256
    reviewer_role_binding_hash: Sha256
    reviewer_model_hash: Sha256
    reviewer_runtime_hash: Sha256
    reviewer_configuration_hash: Sha256
    reviewer_counter_fingerprint: Sha256
    model_context_window_tokens: PositiveInt
    ordered_case_occurrence_ids: tuple[Sha256, ...]
    case_batches: tuple[AIReviewBatch, ...]
    case_coverage_hash: Sha256
    phase_integrity_id: Sha256
    phase_integrity_payload_hash: Sha256
    phase_integrity_request_fingerprint: Sha256
    phase_integrity_input_bytes: PositiveInt
    phase_integrity_input_tokens: PositiveInt
    phase_integrity_maximal_output_bytes: PositiveInt
    phase_integrity_maximal_output_tokens: PositiveInt
    expected_attempt_count: PositiveInt
    aggregate_version: NonEmptyStr
    plan_hash: Sha256

    @model_validator(mode="after")
    def plan_identity_coverage_and_capacity_close(self) -> Self:
        flattened = tuple(
            case_id for batch in self.case_batches for case_id in batch.ordered_case_occurrence_ids
        )
        if (
            not flattened
            or flattened != self.ordered_case_occurrence_ids
            or len(set(flattened)) != len(flattened)
        ):
            raise ValueError("AI review plan case coverage is missing, duplicated, or reordered")
        expected_coverage_hash = canonical_sha256(
            ["oamb-ai-review-case-coverage-v1", self.ordered_case_occurrence_ids]
        )
        if self.case_coverage_hash != expected_coverage_hash:
            raise ValueError("AI review plan coverage hash does not match its case inventory")
        for batch in self.case_batches:
            expected_batch_id = canonical_sha256(
                [
                    "oamb-ai-review-batch-v1",
                    self.review_bundle_hash,
                    batch.ordered_case_occurrence_ids,
                    batch.payload_hash,
                ]
            )
            if batch.batch_id != expected_batch_id:
                raise ValueError("AI review batch identity does not match its canonical payload")
            if batch.input_tokens + batch.maximal_output_tokens > self.model_context_window_tokens:
                raise ValueError("AI review batch exceeds the reviewer context window")
        expected_integrity_id = canonical_sha256(
            [
                "oamb-ai-review-integrity-v1",
                self.review_bundle_hash,
                self.phase_integrity_payload_hash,
            ]
        )
        if self.phase_integrity_id != expected_integrity_id:
            raise ValueError("phase-integrity identity does not match its canonical payload")
        if (
            self.phase_integrity_input_tokens + self.phase_integrity_maximal_output_tokens
            > self.model_context_window_tokens
        ):
            raise ValueError("phase integrity exceeds the reviewer context window")
        if self.phase_integrity_input_bytes > AI_REVIEW_MAX_INPUT_BYTES:
            raise ValueError("phase integrity exceeds the input-byte limit")
        if self.phase_integrity_input_tokens > AI_REVIEW_MAX_INPUT_TOKENS:
            raise ValueError("phase integrity exceeds the input-token limit")
        if self.phase_integrity_maximal_output_bytes > AI_REVIEW_MAX_OUTPUT_BYTES:
            raise ValueError("phase integrity exceeds the output-byte limit")
        if self.phase_integrity_maximal_output_tokens > AI_REVIEW_MAX_OUTPUT_TOKENS:
            raise ValueError("phase integrity exceeds the output-token limit")
        if self.expected_attempt_count != len(self.case_batches) + 1:
            raise ValueError(
                "AI review expected attempt count must include every batch and integrity"
            )
        expected_plan_hash = ai_review_plan_hash(
            review_bundle_hash=self.review_bundle_hash,
            projection_spec_hash=self.projection_spec_hash,
            finding_registry_hash=self.finding_registry_hash,
            prompt_pack_hash=self.prompt_pack_hash,
            output_contract_hash=self.output_contract_hash,
            parser_hash=self.parser_hash,
            reviewer_role_binding_hash=self.reviewer_role_binding_hash,
            reviewer_model_hash=self.reviewer_model_hash,
            reviewer_runtime_hash=self.reviewer_runtime_hash,
            reviewer_configuration_hash=self.reviewer_configuration_hash,
            reviewer_counter_fingerprint=self.reviewer_counter_fingerprint,
            model_context_window_tokens=self.model_context_window_tokens,
            ordered_case_occurrence_ids=self.ordered_case_occurrence_ids,
            case_batches=self.case_batches,
            case_coverage_hash=self.case_coverage_hash,
            phase_integrity_id=self.phase_integrity_id,
            phase_integrity_payload_hash=self.phase_integrity_payload_hash,
            phase_integrity_request_fingerprint=self.phase_integrity_request_fingerprint,
            phase_integrity_input_bytes=self.phase_integrity_input_bytes,
            phase_integrity_input_tokens=self.phase_integrity_input_tokens,
            phase_integrity_maximal_output_bytes=self.phase_integrity_maximal_output_bytes,
            phase_integrity_maximal_output_tokens=self.phase_integrity_maximal_output_tokens,
            expected_attempt_count=self.expected_attempt_count,
            aggregate_version=self.aggregate_version,
        )
        if self.plan_hash != expected_plan_hash:
            raise ValueError("AI review plan hash does not match its canonical payload")
        return self


def ai_review_plan_hash(
    *,
    review_bundle_hash: str,
    projection_spec_hash: str,
    finding_registry_hash: str,
    prompt_pack_hash: str,
    output_contract_hash: str,
    parser_hash: str,
    reviewer_role_binding_hash: str,
    reviewer_model_hash: str,
    reviewer_runtime_hash: str,
    reviewer_configuration_hash: str,
    reviewer_counter_fingerprint: str,
    model_context_window_tokens: int,
    ordered_case_occurrence_ids: tuple[str, ...],
    case_batches: tuple[AIReviewBatch, ...],
    case_coverage_hash: str,
    phase_integrity_id: str,
    phase_integrity_payload_hash: str,
    phase_integrity_request_fingerprint: str,
    phase_integrity_input_bytes: int,
    phase_integrity_input_tokens: int,
    phase_integrity_maximal_output_bytes: int,
    phase_integrity_maximal_output_tokens: int,
    expected_attempt_count: int,
    aggregate_version: str,
) -> str:
    return canonical_sha256(
        [
            "oamb-ai-review-plan-v1",
            review_bundle_hash,
            projection_spec_hash,
            finding_registry_hash,
            prompt_pack_hash,
            output_contract_hash,
            parser_hash,
            reviewer_role_binding_hash,
            reviewer_model_hash,
            reviewer_runtime_hash,
            reviewer_configuration_hash,
            reviewer_counter_fingerprint,
            model_context_window_tokens,
            ordered_case_occurrence_ids,
            case_batches,
            case_coverage_hash,
            phase_integrity_id,
            phase_integrity_payload_hash,
            phase_integrity_request_fingerprint,
            phase_integrity_input_bytes,
            phase_integrity_input_tokens,
            phase_integrity_maximal_output_bytes,
            phase_integrity_maximal_output_tokens,
            expected_attempt_count,
            aggregate_version,
        ]
    )


def comparison_pair_id(
    *,
    case_manifest_entry_id: str,
    left_case_occurrence_id: str,
    right_case_occurrence_id: str,
    metric_id: str,
) -> str:
    return canonical_sha256(
        [
            "oamb-comparison-pair-v1",
            case_manifest_entry_id,
            left_case_occurrence_id,
            right_case_occurrence_id,
            metric_id,
        ]
    )


def comparison_spec_id(
    *,
    left_run_id: str,
    right_run_id: str,
    cost_view: ComparisonCostView,
    ordered_pair_bindings: tuple[ComparisonPairBinding, ...],
    required_control_ids: tuple[str, ...],
    comparison_policy_hash: str,
    winner_reducer: ComparisonWinnerReducer | None,
    cost_control: ComparisonCostControl | None,
) -> str:
    return canonical_sha256(
        [
            "oamb-comparison-spec-v1",
            left_run_id,
            right_run_id,
            cost_view,
            ordered_pair_bindings,
            required_control_ids,
            comparison_policy_hash,
            winner_reducer,
            cost_control,
        ]
    )


class SourceEvidenceBinding(StrictContract):
    schema_name: Literal["source_evidence_binding"] = "source_evidence_binding"
    schema_version: Literal[1] = 1
    binding_id: Sha256
    source_kind: SourceEvidenceKind
    source_identity: NonEmptyStr
    source_root_hash: Sha256
    validation_result_hash: Sha256
    source_schema_versions: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def schema_versions_are_unique(self) -> Self:
        if not self.source_schema_versions:
            raise ValueError("source evidence binding requires schema versions")
        if len(set(self.source_schema_versions)) != len(self.source_schema_versions):
            raise ValueError("source evidence binding contains duplicate schema versions")
        return self


class ProtocolSpec(StrictContract):
    schema_name: Literal["protocol_spec"] = "protocol_spec"
    schema_version: Literal[1] = 1
    protocol_id: NonEmptyStr
    protocol_version: NonEmptyStr
    stages: tuple[NonEmptyStr, ...]
    lifecycle_version: NonEmptyStr
    context_policy: NonEmptyStr
    output_contract_ids: tuple[NonEmptyStr, ...]
    metric_ids: tuple[NonEmptyStr, ...]
    comparison_rule_ids: tuple[NonEmptyStr, ...]


class DatasetFile(StrictContract):
    schema_name: Literal["dataset_file"] = "dataset_file"
    schema_version: Literal[1] = 1
    relative_path: NonEmptyStr
    sha256: Sha256
    byte_count: NonNegativeInt
    license_id: NonEmptyStr


class DatasetManifest(StrictContract):
    schema_name: Literal["dataset_manifest"] = "dataset_manifest"
    schema_version: Literal[1] = 1
    dataset_id: NonEmptyStr
    revision: NonEmptyStr
    split: NonEmptyStr
    manifest_hash: Sha256
    source_files: tuple[DatasetFile, ...]
    payload_policy: NonEmptyStr


class PromptTemplateManifest(StrictContract):
    schema_name: Literal["prompt_template_manifest"] = "prompt_template_manifest"
    schema_version: Literal[1] = 1
    template_name: NonEmptyStr
    relative_path: NonEmptyStr
    content_sha256: Sha256
    byte_count: PositiveInt
    source_extracted_sha256: Sha256 | None = None
    adaptation_id: NonEmptyStr | None = None

    @model_validator(mode="after")
    def source_extraction_and_adaptation_are_paired(self) -> Self:
        if (self.source_extracted_sha256 is None) != (self.adaptation_id is None):
            raise ValueError("source-extracted prompt hash and adaptation ID must be paired")
        return self


def prompt_pack_manifest_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "prompt_pack_manifest")
    payload.setdefault("schema_version", 1)
    payload.pop("manifest_sha256", None)
    return canonical_sha256(payload)


class PromptPackManifest(StrictContract):
    schema_name: Literal["prompt_pack_manifest"] = "prompt_pack_manifest"
    schema_version: Literal[1] = 1
    prompt_pack_id: NonEmptyStr
    prompt_pack_version: NonEmptyStr
    workload_id: NonEmptyStr
    origin: Literal["oamb_authored", "attributed_source", "user_supplied", "generated"]
    source_repository: NonEmptyStr | None
    source_revision: NonEmptyStr | None
    source_file_sha256: Sha256 | None
    license_expression: NonEmptyStr
    rights_attestation: NonEmptyStr
    redistribution_allowed: bool
    templates: tuple[PromptTemplateManifest, ...]
    variables: tuple[NonEmptyStr, ...]
    output_contract_id: NonEmptyStr
    manifest_sha256: Sha256

    @model_validator(mode="after")
    def identity_and_members_are_closed(self) -> Self:
        if not self.templates:
            raise ValueError("prompt pack requires at least one template")
        names = tuple(template.template_name for template in self.templates)
        paths = tuple(template.relative_path for template in self.templates)
        if len(set(names)) != len(names) or len(set(paths)) != len(paths):
            raise ValueError("prompt pack contains duplicate template names or paths")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("prompt pack contains a duplicate variable")
        if not self.variables:
            raise ValueError("prompt pack requires declared variables")
        for variable in self.variables:
            if re.fullmatch(r"[a-z][a-z0-9_]*", variable) is None:
                raise ValueError("prompt variable names require lower snake case")
            if any(
                sensitive in variable
                for sensitive in ("api_key", "secret", "password", "credential", "access_token")
            ):
                raise ValueError("secret values are not valid prompt variables")
        source_values = (
            self.source_repository,
            self.source_revision,
            self.source_file_sha256,
        )
        if self.origin == "attributed_source" and any(value is None for value in source_values):
            raise ValueError("attributed PromptPack requires complete source provenance")
        if self.origin == "attributed_source" and any(
            template.source_extracted_sha256 is None for template in self.templates
        ):
            raise ValueError(
                "attributed PromptPack templates require source-extracted hashes and adaptations"
            )
        if any(value is not None for value in source_values) and any(
            value is None for value in source_values
        ):
            raise ValueError("prompt source repository, revision, and hash must be complete")
        expected = prompt_pack_manifest_hash(
            self.model_dump(mode="python", exclude={"manifest_sha256"})
        )
        if self.manifest_sha256 != expected:
            raise ValueError("prompt pack manifest hash does not match its content")
        return self


class OutputContract(StrictContract):
    schema_name: Literal["output_contract"] = "output_contract"
    schema_version: Literal[1] = 1
    output_contract_id: NonEmptyStr
    representation: Literal["text", "boolean", "ranked_text_list"]
    parser_id: NonEmptyStr
    max_output_tokens: PositiveInt
    required_candidate_count: PositiveInt
    accepted_finish_dispositions: tuple[
        Literal[
            "normal_stop",
            "length_limit",
            "content_filtered",
            "tool_call",
            "missing_finish_reason",
            "cancelled",
            "incomplete_stream",
            "transport_error",
        ],
        ...,
    ]
    parse_failure_policy: Literal["terminal_error", "unjudged"]

    @model_validator(mode="after")
    def finish_dispositions_are_closed(self) -> Self:
        if not self.accepted_finish_dispositions:
            raise ValueError("output contract requires a finish disposition")
        if len(set(self.accepted_finish_dispositions)) != len(self.accepted_finish_dispositions):
            raise ValueError("output contract contains duplicate finish dispositions")
        return self


class MetricSpec(StrictContract):
    schema_name: Literal["metric_spec"] = "metric_spec"
    schema_version: Literal[1] = 1
    metric_id: NonEmptyStr
    output_contract_id: NonEmptyStr
    compatible_output_contract_ids: tuple[NonEmptyStr, ...] = ()
    normalizer_id: NonEmptyStr
    scorer_id: NonEmptyStr
    answer_set_policy: NonEmptyStr
    input_fields: tuple[NonEmptyStr, ...]
    judge_prompt_pack_id: NonEmptyStr | None
    failure_semantics: Literal["unavailable", "terminal_error"]
    unjudged_semantics: Literal["unavailable", "not_applicable", "unjudged"]

    @model_validator(mode="after")
    def evaluation_ownership_is_explicit(self) -> Self:
        if not self.input_fields or len(set(self.input_fields)) != len(self.input_fields):
            raise ValueError("metric input fields must be non-empty and unique")
        if self.judge_prompt_pack_id is not None and "judge" not in self.scorer_id:
            raise ValueError("judge prompt requires a judge scorer")
        output_ids = (self.output_contract_id, *self.compatible_output_contract_ids)
        if len(set(output_ids)) != len(output_ids):
            raise ValueError("metric contains duplicate output contract IDs")
        return self


class LogicalContextManifestEntry(StrictContract):
    schema_name: Literal["logical_context_manifest_entry"] = "logical_context_manifest_entry"
    schema_version: Literal[1] = 1
    context_content_id: Sha256
    context_manifest_entry_id: Sha256
    source_file_sha256: Sha256
    source_row_number_1_indexed: PositiveInt
    context_bytes_sha256: Sha256


class CaseManifestEntry(StrictContract):
    schema_name: Literal["case_manifest_entry"] = "case_manifest_entry"
    schema_version: Literal[1] = 1
    case_manifest_entry_id: Sha256
    context_manifest_entry_id: Sha256
    source_question_number_1_indexed: PositiveInt
    question_bytes_sha256: Sha256
    raw_question_id: NonEmptyStr
    answer_value_sha256: tuple[Sha256, ...]

    @model_validator(mode="after")
    def identity_matches_content(self) -> Self:
        expected = case_manifest_entry_id(
            self.context_manifest_entry_id,
            self.source_question_number_1_indexed,
            self.question_bytes_sha256,
            self.raw_question_id,
        )
        if self.case_manifest_entry_id != expected:
            raise ValueError("case manifest entry identity does not match its content")
        return self


class IngestionPlanManifest(StrictContract):
    schema_name: Literal["ingestion_plan_manifest"] = "ingestion_plan_manifest"
    schema_version: Literal[1] = 1
    plan_manifest_entry_id: Sha256
    ingestion_payload_hash: Sha256
    ingestion_plan_id: Sha256
    workload_id: NonEmptyStr
    ordered_member_context_manifest_entry_ids: tuple[Sha256, ...]
    ordered_source_unit_bytes_sha256: tuple[Sha256, ...]
    ordered_case_manifest_entry_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def non_empty_members(self) -> Self:
        if not self.ordered_member_context_manifest_entry_ids:
            raise ValueError("ingestion plan requires a logical member")
        if not self.ordered_source_unit_bytes_sha256:
            raise ValueError("ingestion plan requires a source unit")
        expected_manifest_id = plan_manifest_entry_id(
            self.workload_id, self.ordered_member_context_manifest_entry_ids
        )
        expected_payload_hash = ingestion_payload_hash(self.ordered_source_unit_bytes_sha256)
        expected_plan_id = ingestion_plan_id(expected_manifest_id, expected_payload_hash)
        if self.plan_manifest_entry_id != expected_manifest_id:
            raise ValueError("plan manifest entry identity does not match its members")
        if self.ingestion_payload_hash != expected_payload_hash:
            raise ValueError("ingestion payload hash does not match its source units")
        if self.ingestion_plan_id != expected_plan_id:
            raise ValueError("ingestion plan identity does not match its manifest and payload")
        return self


def case_manifest_hash(fields: Mapping[str, Any]) -> str:
    payload = {
        field: fields[field]
        for field in (
            "manifest_id",
            "workload_id",
            "logical_contexts",
            "ingestion_plans",
            "cases",
        )
    }
    return canonical_sha256(["oamb-case-manifest-v1", payload])


class CaseManifest(StrictContract):
    schema_name: Literal["case_manifest"] = "case_manifest"
    schema_version: Literal[1] = 1
    manifest_id: NonEmptyStr
    manifest_hash: Sha256
    workload_id: NonEmptyStr
    logical_contexts: tuple[LogicalContextManifestEntry, ...]
    ingestion_plans: tuple[IngestionPlanManifest, ...]
    cases: tuple[CaseManifestEntry, ...]

    @model_validator(mode="after")
    def unique_and_closed_parentage(self) -> Self:
        context_ids = tuple(item.context_manifest_entry_id for item in self.logical_contexts)
        case_ids = tuple(item.case_manifest_entry_id for item in self.cases)
        plan_ids = tuple(item.ingestion_plan_id for item in self.ingestion_plans)
        if len(set(context_ids)) != len(context_ids):
            raise ValueError("duplicate logical context identity")
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("duplicate case identity")
        if len(set(plan_ids)) != len(plan_ids):
            raise ValueError("duplicate ingestion plan identity")
        context_set = set(context_ids)
        case_set = set(case_ids)
        flattened_context_ids: list[str] = []
        flattened_case_ids: list[str] = []
        cases_by_id = {item.case_manifest_entry_id: item for item in self.cases}
        for plan in self.ingestion_plans:
            if plan.workload_id != self.workload_id:
                raise ValueError("ingestion plan belongs to a different workload")
            if not set(plan.ordered_member_context_manifest_entry_ids) <= context_set:
                raise ValueError("ingestion plan references an unknown logical context")
            if not set(plan.ordered_case_manifest_entry_ids) <= case_set:
                raise ValueError("ingestion plan references an unknown case")
            member_ids = set(plan.ordered_member_context_manifest_entry_ids)
            for referenced_case_id in plan.ordered_case_manifest_entry_ids:
                if cases_by_id[referenced_case_id].context_manifest_entry_id not in member_ids:
                    raise ValueError("case context does not belong to its ingestion plan")
            flattened_context_ids.extend(plan.ordered_member_context_manifest_entry_ids)
            flattened_case_ids.extend(plan.ordered_case_manifest_entry_ids)
        if (
            len(flattened_context_ids) != len(context_ids)
            or set(flattened_context_ids) != context_set
        ):
            raise ValueError("every logical context must belong to exactly once ingestion plan")
        if len(flattened_case_ids) != len(case_ids) or set(flattened_case_ids) != case_set:
            raise ValueError("every case must belong to exactly once ingestion plan")
        expected_hash = case_manifest_hash(self.model_dump(mode="python"))
        if self.manifest_hash != expected_hash:
            raise ValueError("case manifest hash does not match its exact members")
        return self


class InteractionSpec(StrictContract):
    schema_name: Literal["interaction_spec"] = "interaction_spec"
    schema_version: Literal[1] = 1
    interaction_id: NonEmptyStr
    interaction_version: NonEmptyStr
    chunking_id: NonEmptyStr
    ingestion_mapping_id: NonEmptyStr
    retrieval_top_k: PositiveInt
    normalizer_id: NonEmptyStr
    query_effect_policy: NonEmptyStr
    prompt_pack_ids: tuple[NonEmptyStr, ...]
    output_contract_ids: tuple[NonEmptyStr, ...]


class WorkloadSpec(StrictContract):
    schema_name: Literal["workload_spec"] = "workload_spec"
    schema_version: Literal[1] = 1
    workload_id: NonEmptyStr
    workload_version: NonEmptyStr
    dataset_manifest_hash: Sha256
    case_manifest_hash: Sha256
    interaction_spec_id: NonEmptyStr
    prompt_pack_ids: tuple[NonEmptyStr, ...]
    metric_ids: tuple[NonEmptyStr, ...]


class MemorySystemSpec(StrictContract):
    schema_name: Literal["memory_system_spec"] = "memory_system_spec"
    schema_version: Literal[1] = 1
    memory_system_id: NonEmptyStr
    adapter_id: NonEmptyStr
    adapter_revision: NonEmptyStr
    transport_profile: TransportProfile
    capabilities: tuple[NonEmptyStr, ...]
    native_configuration_fingerprint: Sha256


class MemorySystemRuntimeBinding(StrictContract):
    schema_name: Literal["memory_system_runtime_binding"] = "memory_system_runtime_binding"
    schema_version: Literal[1] = 1
    memory_system_id: NonEmptyStr
    runtime_binding_hash: Sha256
    release_version: NonEmptyStr
    artifact_sha256: Sha256
    storage_engine: NonEmptyStr
    model_role_binding_ids: tuple[NonEmptyStr, ...]
    attestation_status: AttestationStatus


class MemorySystemRuntimeBindingV2(StrictContract):
    schema_name: Literal["memory_system_runtime_binding"] = "memory_system_runtime_binding"
    schema_version: Literal[2] = 2
    memory_system_id: NonEmptyStr
    runtime_binding_hash: Sha256
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    edition: NonEmptyStr
    distribution_channel: NonEmptyStr
    api_version: NonEmptyStr
    release_version: NonEmptyStr
    source_revision: NonEmptyStr
    artifact_kind: NonEmptyStr
    artifact_sha256: Sha256
    endpoint_fingerprint: Sha256
    deployment_configuration_sha256: Sha256
    storage_engine: NonEmptyStr
    storage_engine_version: NonEmptyStr
    schema_revision: NonEmptyStr
    vector_index_type: NonEmptyStr
    distance_metric: NonEmptyStr
    vector_dimension: PositiveInt
    index_configuration_sha256: Sha256
    model_role_binding_ids: tuple[NonEmptyStr, ...]
    native_feature_flags_fingerprint: Sha256
    native_reranking_status: Literal["disabled"]
    attestation_method: NonEmptyStr
    attestation_status: RuntimeAttestationStatus
    raw_proof_refs: tuple[Sha256, ...]
    model_readiness_required: bool
    model_readiness_evidence: SourceEvidenceBinding | None

    @model_validator(mode="after")
    def model_readiness_binding_is_closed(self) -> Self:
        if not self.model_role_binding_ids:
            raise ValueError("runtime binding requires model role bindings")
        if len(set(self.model_role_binding_ids)) != len(self.model_role_binding_ids):
            raise ValueError("runtime binding contains duplicate model role bindings")
        if not self.raw_proof_refs:
            raise ValueError("runtime binding requires raw proof references")
        if self.model_readiness_required:
            if self.model_readiness_evidence is None:
                raise ValueError("required model readiness evidence is missing")
            if self.model_readiness_evidence.source_kind != SourceEvidenceKind.PROVIDER_SERVICE:
                raise ValueError("model readiness evidence must bind provider-service evidence")
        elif self.model_readiness_evidence is not None:
            raise ValueError("optional model readiness cannot carry an evidence binding")
        expected_hash = memory_system_runtime_binding_hash(
            self.model_dump(mode="python", exclude={"runtime_binding_hash"})
        )
        if self.runtime_binding_hash != expected_hash:
            raise ValueError("runtime binding hash does not match its canonical fields")
        return self


def memory_system_runtime_binding_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "memory_system_runtime_binding")
    payload.setdefault("schema_version", 2)
    payload.pop("runtime_binding_hash", None)
    return canonical_sha256(payload)


class ExecutionEnvironmentBinding(StrictContract):
    schema_name: Literal["execution_environment_binding"] = "execution_environment_binding"
    schema_version: Literal[1] = 1
    environment_hash: Sha256
    operating_system: NonEmptyStr
    architecture: NonEmptyStr
    python_version: NonEmptyStr
    cpu_description: NonEmptyStr
    memory_bytes: PositiveInt
    comparability_status: ComparabilityStatus


class ModelRoleBinding(StrictContract):
    schema_name: Literal["model_role_binding"] = "model_role_binding"
    schema_version: Literal[1] = 1
    binding_id: NonEmptyStr
    role: NonEmptyStr
    owner: NonEmptyStr
    endpoint_fingerprint: Sha256
    configured_model: NonEmptyStr
    resolved_model: NonEmptyStr
    parameters_fingerprint: Sha256
    budget_role: NonEmptyStr


class ModelRoleBindingV2(StrictContract):
    schema_name: Literal["model_role_binding"] = "model_role_binding"
    schema_version: Literal[2] = 2
    binding_id: NonEmptyStr
    role: ModelRole
    role_status: RoleBindingStatus
    execution_owner: ExecutionOwner
    binding_kind: BindingKind
    provider: NonEmptyStr | None
    endpoint_reference: NonEmptyStr | None
    credential_variable_name: NonEmptyStr | None
    configured_model: NonEmptyStr | None
    resolved_model: NonEmptyStr | None
    parameters_fingerprint: Sha256 | None
    retry_policy_id: NonEmptyStr | None
    configuration_fingerprint: Sha256
    redacted_endpoint_fingerprint: Sha256 | None

    @model_validator(mode="after")
    def binding_status_matches_shape(self) -> Self:
        selected_fields = (
            self.provider,
            self.endpoint_reference,
            self.configured_model,
            self.resolved_model,
            self.parameters_fingerprint,
            self.retry_policy_id,
            self.redacted_endpoint_fingerprint,
        )
        if self.role_status == RoleBindingStatus.SELECTED:
            if self.binding_kind not in {BindingKind.NATIVE, BindingKind.MODEL_CLIENT}:
                raise ValueError("selected binding requires native or model_client kind")
            if any(value is None for value in selected_fields):
                raise ValueError("selected binding requires resolved provider and model fields")
        else:
            expected_kind = (
                BindingKind.DISABLED
                if self.role_status == RoleBindingStatus.DISABLED
                else BindingKind.NOT_APPLICABLE
            )
            if self.binding_kind != expected_kind:
                raise ValueError("unselected binding status and kind must match")
            if any(value is not None for value in selected_fields):
                raise ValueError("unselected binding cannot carry provider or model fields")
            if self.credential_variable_name is not None:
                raise ValueError("unselected binding cannot name a credential variable")
        if self.execution_owner == ExecutionOwner.DETERMINISTIC_METRIC:
            if self.role != ModelRole.JUDGE or self.role_status != RoleBindingStatus.NOT_APPLICABLE:
                raise ValueError("deterministic metric is only valid for a not-applicable judge")
        return self


class ResourceBudgetCeiling(StrictContract):
    schema_name: Literal["resource_budget_ceiling"] = "resource_budget_ceiling"
    schema_version: Literal[1] = 1
    dimension_id: NonEmptyStr
    maximum: NonNegativeDecimal
    unit: NonEmptyStr


class ProviderBudgetCap(StrictContract):
    schema_name: Literal["provider_budget_cap"] = "provider_budget_cap"
    schema_version: Literal[1] = 1
    provider: NonEmptyStr
    operation_kind: NonEmptyStr
    billing_unit: NonEmptyStr
    maximum_accepted_units: NonNegativeDecimal


class RoleBudgetCeiling(StrictContract):
    schema_name: Literal["role_budget_ceiling"] = "role_budget_ceiling"
    schema_version: Literal[1] = 1
    role_binding_id: NonEmptyStr
    max_attempts: PositiveInt
    max_input_tokens: NonNegativeInt
    max_output_tokens: NonNegativeInt
    max_dispatch_wall_seconds: NonNegativeDecimal
    max_cost: NonNegativeDecimal | None
    currency: str | None
    price_snapshot_id: NonEmptyStr | None
    resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    provider_budget_cap: ProviderBudgetCap

    @model_validator(mode="after")
    def currency_and_resources_close(self) -> Self:
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        if self.price_snapshot_id is not None and self.max_cost is None:
            raise ValueError("price snapshot requires a cost ceiling")
        resource_ids = tuple(item.dimension_id for item in self.resource_ceilings)
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("role ceiling contains duplicate resource dimensions")
        return self


class BudgetSpec(StrictContract):
    schema_name: Literal["budget_spec"] = "budget_spec"
    schema_version: Literal[1] = 1
    budget_id: NonEmptyStr
    scope_kind: BudgetScopeKind
    scope_id: NonEmptyStr
    approval_id: NonEmptyStr | None
    max_attempts: NonNegativeInt
    max_input_tokens: NonNegativeInt
    max_output_tokens: NonNegativeInt
    max_wall_seconds: NonNegativeDecimal
    max_cost: NonNegativeDecimal | None
    currency: str | None

    @model_validator(mode="after")
    def currency_matches_cost(self) -> Self:
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        return self


class BudgetSpecV2(StrictContract):
    schema_name: Literal["budget_spec"] = "budget_spec"
    schema_version: Literal[2] = 2
    budget_id: NonEmptyStr
    scope_kind: BudgetScopeKindV2
    scope_id: NonEmptyStr
    approval_id: NonEmptyStr
    max_attempts: PositiveInt
    max_input_tokens: NonNegativeInt
    max_output_tokens: NonNegativeInt
    max_dispatch_wall_seconds: NonNegativeDecimal
    max_cost: NonNegativeDecimal | None
    currency: str | None
    resource_ceilings: tuple[ResourceBudgetCeiling, ...]
    role_ceilings: tuple[RoleBudgetCeiling, ...]
    stop_condition_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def parent_and_role_ceilings_close(self) -> Self:
        if (self.max_cost is None) != (self.currency is None):
            raise ValueError("max_cost and currency must be present together")
        if self.currency is not None and len(self.currency) != 3:
            raise ValueError("currency must be an ISO 4217 code")
        if not self.role_ceilings:
            raise ValueError("version 2 budget requires at least one role ceiling")
        role_ids = tuple(item.role_binding_id for item in self.role_ceilings)
        if len(set(role_ids)) != len(role_ids):
            raise ValueError("duplicate role ceiling")
        resource_ids = tuple(item.dimension_id for item in self.resource_ceilings)
        if len(set(resource_ids)) != len(resource_ids):
            raise ValueError("budget contains duplicate resource dimensions")
        if len(set(self.stop_condition_ids)) != len(self.stop_condition_ids):
            raise ValueError("budget contains duplicate stop conditions")
        if any(ceiling.currency != self.currency for ceiling in self.role_ceilings):
            raise ValueError("role budget currency must match the parent currency")
        if self.max_cost is not None and any(
            ceiling.max_cost is None or ceiling.max_cost > self.max_cost
            for ceiling in self.role_ceilings
        ):
            raise ValueError("role cost ceiling must fit the parent cost ceiling")
        return self


class ProviderRuntimeProfileAttestation(StrictContract):
    schema_name: Literal["provider_runtime_profile_attestation"] = (
        "provider_runtime_profile_attestation"
    )
    schema_version: Literal[1] = 1
    provider: NonEmptyStr
    provider_project_id: NonEmptyStr
    provider_profile_id: NonEmptyStr
    attestation_hash: Sha256
    transport_profile: TransportProfile
    release_version: NonEmptyStr
    source_revision: NonEmptyStr
    build_artifact_sha256: Sha256
    redacted_configuration_sha256: Sha256
    redacted_endpoint_fingerprint: Sha256
    auth_configuration_sha256: Sha256
    storage_configuration_sha256: Sha256
    model_role_binding_ids: tuple[NonEmptyStr, ...]
    native_reranking_status: Literal["disabled"]
    liveness_status: ProviderGateStatus
    storage_configuration_status: ProviderGateStatus
    runtime_identity_status: ProviderGateStatus
    model_readiness_status: ProviderGateStatus
    memory_conformance_status: ProviderGateStatus
    raw_proof_refs: tuple[Sha256, ...]

    @model_validator(mode="after")
    def remains_pre_readiness(self) -> Self:
        expected_hash = provider_runtime_profile_attestation_hash(
            self.model_dump(mode="python", exclude={"attestation_hash"})
        )
        if self.attestation_hash != expected_hash:
            raise ValueError("attestation hash does not match its canonical fields")
        if self.model_readiness_status != ProviderGateStatus.NOT_RUN:
            raise ValueError("pre-readiness attestation requires model readiness NOT_RUN")
        if self.memory_conformance_status != ProviderGateStatus.NOT_RUN:
            raise ValueError("pre-readiness attestation requires memory conformance NOT_RUN")
        if not self.model_role_binding_ids:
            raise ValueError("pre-readiness attestation requires selected model roles")
        if len(set(self.model_role_binding_ids)) != len(self.model_role_binding_ids):
            raise ValueError("pre-readiness attestation contains duplicate model roles")
        if not self.raw_proof_refs:
            raise ValueError("pre-readiness attestation requires raw proof references")
        return self


class ExternalCallApprovalRecord(StrictContract):
    schema_name: Literal["external_call_approval_record"] = "external_call_approval_record"
    schema_version: Literal[1] = 1
    approval_id: NonEmptyStr
    approval_hash: Sha256
    operation_kind: NonEmptyStr
    scope_kind: BudgetScopeKindV2
    scope_id: NonEmptyStr
    runtime_binding_hash: Sha256 | None
    provider_runtime_profile_attestation_hash: Sha256 | None
    role_binding_ids: tuple[NonEmptyStr, ...]
    budget_hash: Sha256
    approved_at: UtcDateTime
    expires_at: UtcDateTime
    unmetered_cost_acknowledged: bool
    stop_condition_ids: tuple[NonEmptyStr, ...]

    @model_validator(mode="after")
    def scope_runtime_and_expiry_close(self) -> Self:
        expected_hash = external_call_approval_hash(
            self.model_dump(mode="python", exclude={"approval_hash"})
        )
        if self.approval_hash != expected_hash:
            raise ValueError("approval hash does not match its canonical fields")
        if self.expires_at <= self.approved_at:
            raise ValueError("approval expiry must follow approval time")
        if not self.role_binding_ids:
            raise ValueError("approval requires at least one role binding")
        if len(set(self.role_binding_ids)) != len(self.role_binding_ids):
            raise ValueError("approval contains duplicate role bindings")
        if len(set(self.stop_condition_ids)) != len(self.stop_condition_ids):
            raise ValueError("approval contains duplicate stop conditions")
        if self.scope_kind == BudgetScopeKindV2.RUN:
            if self.runtime_binding_hash is None:
                raise ValueError("run approval requires a runtime binding")
            if self.provider_runtime_profile_attestation_hash is not None:
                raise ValueError("run approval cannot use a pre-readiness attestation")
        elif self.scope_kind == BudgetScopeKindV2.MODEL_READINESS:
            if self.provider_runtime_profile_attestation_hash is None:
                raise ValueError("model-readiness approval requires an attestation")
            if self.runtime_binding_hash is not None:
                raise ValueError("model-readiness approval cannot use a runtime binding")
        elif (
            self.runtime_binding_hash is not None
            or self.provider_runtime_profile_attestation_hash is not None
        ):
            raise ValueError("phase-review approval has no memory-system runtime binding")
        return self


def provider_runtime_profile_attestation_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "provider_runtime_profile_attestation")
    payload.setdefault("schema_version", 1)
    payload.pop("attestation_hash", None)
    return canonical_sha256(payload)


def external_call_approval_hash(fields: Mapping[str, Any]) -> str:
    payload = dict(fields)
    payload.setdefault("schema_name", "external_call_approval_record")
    payload.setdefault("schema_version", 1)
    payload.pop("approval_hash", None)
    return canonical_sha256(payload)


class DerivationSpec(StrictContract):
    schema_name: Literal["derivation_spec"] = "derivation_spec"
    schema_version: Literal[1] = 1
    derivation_kind: NonEmptyStr
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    ordered_source_root_hash: Sha256
    transform_spec_hash: Sha256
    report_spec_hash: Sha256 | None
    reducer_and_renderer_input_hashes: tuple[Sha256, ...]
    derivation_input_hash: Sha256

    @model_validator(mode="after")
    def source_bindings_are_non_empty_and_unique(self) -> Self:
        if not self.ordered_source_bindings:
            raise ValueError("derivation requires at least one source binding")
        binding_ids = tuple(item.binding_id for item in self.ordered_source_bindings)
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("derivation contains duplicate source bindings")
        return self


class ReportSpec(StrictContract):
    schema_name: Literal["report_spec"] = "report_spec"
    schema_version: Literal[1] = 1
    report_spec_id: Sha256
    report_kind: Literal["run", "comparison", "release"]
    audience: Literal["local", "public"]
    preview_max_field_bytes: PositiveInt
    preview_total_bytes: PositiveInt
    display_field_ids: tuple[NonEmptyStr, ...]
    renderer_hash: Sha256
    asset_hashes: tuple[Sha256, ...]
    browser_contract_hash: Sha256
    performance_contract_hash: Sha256
    export_profile_selector_id: NonEmptyStr
    export_profile_selector_version: PositiveInt

    @model_validator(mode="after")
    def report_spec_identity_and_limits_are_canonical(self) -> Self:
        if self.preview_max_field_bytes > self.preview_total_bytes:
            raise ValueError("report field preview exceeds the total preview budget")
        if not self.display_field_ids or len(set(self.display_field_ids)) != len(
            self.display_field_ids
        ):
            raise ValueError("report spec display fields must be nonempty and unique")
        if len(set(self.asset_hashes)) != len(self.asset_hashes):
            raise ValueError("report spec contains duplicate asset hashes")
        expected = report_spec_id(
            report_kind=self.report_kind,
            audience=self.audience,
            preview_max_field_bytes=self.preview_max_field_bytes,
            preview_total_bytes=self.preview_total_bytes,
            display_field_ids=self.display_field_ids,
            renderer_hash=self.renderer_hash,
            asset_hashes=self.asset_hashes,
            browser_contract_hash=self.browser_contract_hash,
            performance_contract_hash=self.performance_contract_hash,
            export_profile_selector_id=self.export_profile_selector_id,
            export_profile_selector_version=self.export_profile_selector_version,
        )
        if self.report_spec_id != expected:
            raise ValueError("report spec identity does not match its canonical payload")
        return self


class AcceptanceReportSpec(StrictContract):
    schema_name: Literal["acceptance_report_spec"] = "acceptance_report_spec"
    schema_version: Literal[1] = 1
    acceptance_report_spec_id: Sha256
    audience: Literal["local", "public"]
    evaluation_report_hash: Sha256
    evaluation_export_validation_hash: Sha256
    review_bundle_hash: Sha256
    ai_review_record_hash: Sha256
    human_review_record_hash: Sha256
    phase_gate_hash: Sha256
    renderer_hash: Sha256
    asset_hashes: tuple[Sha256, ...]
    browser_contract_hash: Sha256
    performance_contract_hash: Sha256
    export_profile_selector_id: NonEmptyStr
    export_profile_selector_version: PositiveInt

    @model_validator(mode="after")
    def acceptance_report_spec_identity_is_canonical(self) -> Self:
        if len(set(self.asset_hashes)) != len(self.asset_hashes):
            raise ValueError("acceptance report spec contains duplicate asset hashes")
        expected = acceptance_report_spec_id(
            audience=self.audience,
            evaluation_report_hash=self.evaluation_report_hash,
            evaluation_export_validation_hash=self.evaluation_export_validation_hash,
            review_bundle_hash=self.review_bundle_hash,
            ai_review_record_hash=self.ai_review_record_hash,
            human_review_record_hash=self.human_review_record_hash,
            phase_gate_hash=self.phase_gate_hash,
            renderer_hash=self.renderer_hash,
            asset_hashes=self.asset_hashes,
            browser_contract_hash=self.browser_contract_hash,
            performance_contract_hash=self.performance_contract_hash,
            export_profile_selector_id=self.export_profile_selector_id,
            export_profile_selector_version=self.export_profile_selector_version,
        )
        if self.acceptance_report_spec_id != expected:
            raise ValueError("acceptance report spec identity does not match its canonical payload")
        return self


class ReportIdentitySpecBinding(StrictContract):
    schema_name: Literal["report_identity_spec_binding"] = "report_identity_spec_binding"
    schema_version: Literal[1] = 1
    binding_id: Sha256
    spec_kind: Literal["benchmark_report", "acceptance_report"]
    spec_schema_name: Literal["report_spec", "acceptance_report_spec"]
    spec_schema_version: Literal[1] = 1
    spec_id: Sha256
    spec_hash: Sha256

    @model_validator(mode="after")
    def binding_discriminator_and_identity_are_canonical(self) -> Self:
        expected_schema_name = (
            "report_spec" if self.spec_kind == "benchmark_report" else "acceptance_report_spec"
        )
        if self.spec_schema_name != expected_schema_name:
            raise ValueError("report identity binding discriminator does not match its schema")
        expected = report_identity_spec_binding_id(
            spec_kind=self.spec_kind,
            spec_schema_name=self.spec_schema_name,
            spec_schema_version=self.spec_schema_version,
            spec_id=self.spec_id,
            spec_hash=self.spec_hash,
        )
        if self.binding_id != expected:
            raise ValueError("report identity binding does not match its canonical payload")
        return self


RENDERING_DERIVATION_KINDS = frozenset(
    {
        "run_report",
        "diagnostic_run_report",
        "comparison_report",
        "release_report",
        "phase_acceptance_report",
    }
)


class DerivationSpecV2(StrictContract):
    schema_name: Literal["derivation_spec"] = "derivation_spec"
    schema_version: Literal[2] = 2
    derivation_kind: NonEmptyStr
    ordered_source_bindings: tuple[SourceEvidenceBinding, ...]
    ordered_source_root_hash: Sha256
    evidence_validation_result_hash: Sha256
    transform_spec_hash: Sha256
    report_identity_spec_binding: ReportIdentitySpecBinding | None
    reducer_and_renderer_input_hashes: tuple[Sha256, ...]
    derivation_input_hash: Sha256

    @model_validator(mode="after")
    def source_report_cardinality_and_identity_close(self) -> Self:
        if not self.ordered_source_bindings:
            raise ValueError("derivation requires at least one source binding")
        binding_ids = tuple(item.binding_id for item in self.ordered_source_bindings)
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("derivation contains duplicate source bindings")
        expected_root_hash = canonical_sha256(
            [
                "oamb-ordered-source-roots-v1",
                tuple(item.source_root_hash for item in self.ordered_source_bindings),
            ]
        )
        if self.ordered_source_root_hash != expected_root_hash:
            raise ValueError("derivation ordered source-root hash does not match its bindings")
        rendering = self.derivation_kind in RENDERING_DERIVATION_KINDS
        if rendering and self.report_identity_spec_binding is None:
            raise ValueError("rendering derivation requires exactly one report identity binding")
        if not rendering and self.report_identity_spec_binding is not None:
            raise ValueError("non-rendering derivation cannot carry a report identity binding")
        expected_input_hash = derivation_spec_v2_input_hash(
            derivation_kind=self.derivation_kind,
            ordered_source_bindings=self.ordered_source_bindings,
            ordered_source_root_hash=self.ordered_source_root_hash,
            evidence_validation_result_hash=self.evidence_validation_result_hash,
            transform_spec_hash=self.transform_spec_hash,
            report_identity_spec_binding=self.report_identity_spec_binding,
            reducer_and_renderer_input_hashes=self.reducer_and_renderer_input_hashes,
        )
        if self.derivation_input_hash != expected_input_hash:
            raise ValueError("derivation input hash does not match its canonical payload")
        return self


def report_spec_id(**fields: object) -> str:
    return canonical_sha256(["oamb-report-spec-v1", fields])


def acceptance_report_spec_id(**fields: object) -> str:
    return canonical_sha256(["oamb-acceptance-report-spec-v1", fields])


def report_identity_spec_binding_id(**fields: object) -> str:
    return canonical_sha256(["oamb-report-identity-spec-binding-v1", fields])


def derivation_spec_v2_input_hash(**fields: object) -> str:
    return canonical_sha256(["oamb-derivation-input-v2", fields])


class RunSpec(StrictContract):
    schema_name: Literal["run_spec"] = "run_spec"
    schema_version: Literal[1] = 1
    run_id: NonEmptyStr
    protocol_id: NonEmptyStr
    dataset_manifest_hash: Sha256
    case_manifest_hash: Sha256
    workload_id: NonEmptyStr
    memory_system_id: NonEmptyStr
    runtime_binding_hash: Sha256
    environment_hash: Sha256
    model_role_binding_ids: tuple[NonEmptyStr, ...]
    budget_id: NonEmptyStr
    code_revision: NonEmptyStr
    normalizer_fingerprint: Sha256


class ValidationRuleRequirement(StrictContract):
    schema_name: Literal["validation_rule_requirement"] = "validation_rule_requirement"
    schema_version: Literal[1] = 1
    rule_id: NonEmptyStr
    minimum_version: PositiveInt


class ValidationProfile(StrictContract):
    schema_name: Literal["validation_profile"] = "validation_profile"
    schema_version: Literal[1] = 1
    profile_id: NonEmptyStr
    stage: ValidationStage
    required_rules: tuple[ValidationRuleRequirement, ...]
    applicability: tuple[NonEmptyStr, ...]
    required_rule_inventory_hash: Sha256

    @classmethod
    def create(
        cls,
        *,
        profile_id: str,
        stage: ValidationStage,
        required_rules: tuple[ValidationRuleRequirement, ...],
        applicability: tuple[str, ...],
    ) -> Self:
        inventory = tuple(
            (requirement.rule_id, requirement.minimum_version) for requirement in required_rules
        )
        return cls(
            profile_id=profile_id,
            stage=stage,
            required_rules=required_rules,
            applicability=applicability,
            required_rule_inventory_hash=canonical_sha256(
                ["oamb-required-rule-inventory-v1", inventory]
            ),
        )

    @model_validator(mode="after")
    def unique_requirements(self) -> Self:
        rule_ids = tuple(item.rule_id for item in self.required_rules)
        if len(set(rule_ids)) != len(rule_ids):
            raise ValueError("validation profile contains duplicate rule IDs")
        return self
