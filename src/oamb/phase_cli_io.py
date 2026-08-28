"""Strict file inputs for credential-free phase CLI composition."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar, cast

from pydantic import BaseModel

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
from oamb.artifacts.validation.phase import PhaseReviewEvidence
from oamb.contracts.accounting import (
    CostMeasurementSpec,
    CostRecord,
    PriceSnapshot,
    ResourceUsageRecord,
    TokenUsageRecordV2,
    TokenUsageRecordV3,
)
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptReceiptRecord,
    AttemptRecordV2,
    BudgetReservationRecord,
    CloseErrorRecord,
    OccurrenceClaimRecord,
    PhaseReviewOccurrenceRecordV2,
    RunLeaseRecord,
)
from oamb.contracts.ids import canonical_json_bytes
from oamb.contracts.reporting import (
    AIQualityReviewRecord,
    AIReviewBatchResult,
    AIReviewIntegrityResult,
)
from oamb.contracts.schema import parse_contract
from oamb.contracts.specifications import (
    AIReviewPlan,
    BudgetSpecV2,
    ExecutionEnvironmentBinding,
    ExternalCallApprovalRecord,
    HumanReviewKeyBinding,
    ModelRoleBindingV2,
)
from oamb.phase_review_profiles import (
    FAKE_PHASE_REVIEW_CONFIGURED_MODEL,
    FAKE_PHASE_REVIEW_CREDENTIAL_VARIABLE,
    FAKE_PHASE_REVIEW_ENDPOINT,
    FAKE_PHASE_REVIEW_PROVIDER,
    FAKE_PHASE_REVIEW_RESOLVED_MODEL,
)

_ModelT = TypeVar("_ModelT", bound=BaseModel)


def load_phase_review_evidence(root: Path) -> PhaseReviewEvidence:
    reviewer_role = load_contract(root / "role-binding.json", ModelRoleBindingV2)
    price_snapshot_path = root / "price-snapshot.json"
    if _is_exact_fake_role(reviewer_role):
        if price_snapshot_path.exists() or price_snapshot_path.is_symlink():
            raise ValueError("exact fake review evidence cannot contain price-snapshot.json")
        price_snapshot = None
    else:
        price_snapshot = load_contract(price_snapshot_path, PriceSnapshot)
    human_paths = (
        root / "human-key-binding.json",
        root / "human-decision.json",
        root / "human-signature.txt",
    )
    present_human_paths = tuple(path for path in human_paths if path.exists() or path.is_symlink())
    if present_human_paths and len(present_human_paths) != len(human_paths):
        raise ValueError("phase review evidence contains an incomplete human evidence set")
    human_key_binding = (
        None if not present_human_paths else load_contract(human_paths[0], HumanReviewKeyBinding)
    )
    human_decision_bytes = None if not present_human_paths else read_regular_file(human_paths[1])
    human_signature_base64 = None if not present_human_paths else read_signature(human_paths[2])
    return PhaseReviewEvidence(
        plan=load_contract(root / "plan.json", AIReviewPlan),
        reviewer_role_binding=reviewer_role,
        occurrence=load_contract(root / "occurrence.json", PhaseReviewOccurrenceRecordV2),
        approval=load_contract(root / "approval.json", ExternalCallApprovalRecord),
        budget=load_contract(root / "budget.json", BudgetSpecV2),
        cost_measurement_spec=load_contract(
            root / "cost-measurement-spec.json", CostMeasurementSpec
        ),
        execution_environment=load_contract(
            root / "execution-environment.json", ExecutionEnvironmentBinding
        ),
        price_snapshot=price_snapshot,
        lease_record=load_contract(root / "lease-record.json", RunLeaseRecord),
        occurrence_claims=load_contract_sequence(
            root / "occurrence-claims.json", OccurrenceClaimRecord
        ),
        budget_reservations=load_contract_sequence(
            root / "budget-reservations.json", BudgetReservationRecord
        ),
        attempt_intents=load_contract_sequence(root / "attempt-intents.json", AttemptIntentRecord),
        attempt_receipts=load_contract_sequence(
            root / "attempt-receipts.json", AttemptReceiptRecord
        ),
        close_errors=(
            ()
            if not (root / "close-errors.json").exists()
            else load_contract_sequence(root / "close-errors.json", CloseErrorRecord)
        ),
        batch_results=load_contract_sequence(root / "batch-results.json", AIReviewBatchResult),
        integrity_result=load_contract(root / "integrity-result.json", AIReviewIntegrityResult),
        ordered_ai_history=load_contract_sequence(root / "ai-history.json", AIQualityReviewRecord),
        attempts=load_contract_sequence(root / "attempts.json", AttemptRecordV2),
        usage_records=_load_contract_sequence_union(
            root / "usage-records.json",
            (TokenUsageRecordV2, TokenUsageRecordV3),
        ),
        resource_records=load_contract_sequence(
            root / "resource-records.json", ResourceUsageRecord
        ),
        cost_records=load_contract_sequence(root / "cost-records.json", CostRecord),
        human_key_binding=human_key_binding,
        human_decision_bytes=human_decision_bytes,
        human_signature_base64=human_signature_base64,
    )


def _is_exact_fake_role(role: ModelRoleBindingV2) -> bool:
    return bool(
        role.provider == FAKE_PHASE_REVIEW_PROVIDER
        and role.endpoint_reference == FAKE_PHASE_REVIEW_ENDPOINT
        and role.credential_variable_name == FAKE_PHASE_REVIEW_CREDENTIAL_VARIABLE
        and role.configured_model == FAKE_PHASE_REVIEW_CONFIGURED_MODEL
        and role.resolved_model == FAKE_PHASE_REVIEW_RESOLVED_MODEL
    )


def load_contract(path: Path, expected_type: type[_ModelT]) -> _ModelT:
    content = read_regular_file(path)
    document = _load_json(content, label=str(path))
    if not isinstance(document, dict):
        raise ValueError(f"{path} must contain one versioned contract object")
    parsed = parse_contract(document)
    if not isinstance(parsed, expected_type):
        raise ValueError(f"{path} contains {type(parsed).__name__}, not {expected_type.__name__}")
    if content != canonical_json_bytes(parsed):
        raise ValueError(f"{path} is not canonical contract JSON")
    return parsed


def load_contract_sequence(path: Path, expected_type: type[_ModelT]) -> tuple[_ModelT, ...]:
    return cast(
        tuple[_ModelT, ...],
        _load_contract_sequence_union(path, (expected_type,)),
    )


def _load_contract_sequence_union(
    path: Path,
    expected_types: tuple[type[BaseModel], ...],
) -> tuple[Any, ...]:
    content = read_regular_file(path)
    document = _load_json(content, label=str(path))
    if not isinstance(document, list):
        raise ValueError(f"{path} must contain a canonical array of versioned contracts")
    parsed: list[BaseModel] = []
    for item in document:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains a non-object contract item")
        contract = parse_contract(item)
        if not isinstance(contract, expected_types):
            names = ", ".join(item_type.__name__ for item_type in expected_types)
            raise ValueError(f"{path} contains {type(contract).__name__}, expected {names}")
        parsed.append(contract)
    values = tuple(parsed)
    if content != canonical_json_bytes(values):
        raise ValueError(f"{path} is not a canonical contract array")
    return values


def load_exact_object(path: Path, expected_fields: set[str]) -> dict[str, object]:
    document = _load_json(read_regular_file(path), label=str(path))
    if not isinstance(document, dict) or set(document) != expected_fields:
        raise ValueError(f"{path} fields do not match the closed CLI input format")
    return cast(dict[str, object], document)


def _load_json(content: bytes, *, label: str) -> object:
    try:
        return json.loads(content, parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict UTF-8 JSON") from exc


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def write_contract(
    output: Path,
    contract: BaseModel,
    *,
    trusted_root: Path | None = None,
) -> None:
    atomic_write_bytes(
        output,
        canonical_json_bytes(contract),
        trusted_root=trusted_root or output.parent,
    )


def read_utf8(path: Path) -> str:
    try:
        return read_regular_file(path).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not strict UTF-8") from exc


def read_signature(path: Path) -> str:
    value = read_utf8(path)
    if value.endswith("\n"):
        value = value[:-1]
    if not value or value != value.strip() or "\n" in value or "\r" in value:
        raise ValueError("detached signature file must contain one canonical-base64 line")
    return value


def string_value(document: dict[str, object], key: str) -> str:
    value = document[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def optional_string_value(document: dict[str, object], key: str) -> str | None:
    value = document[key]
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be null or a non-empty string")
    return value


def string_tuple(document: dict[str, object], key: str) -> tuple[str, ...]:
    value = document[key]
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return tuple(value)


def integer_value(document: dict[str, object], key: str) -> int:
    value = document[key]
    if type(value) is not int:
        raise ValueError(f"{key} must be an integer")
    return value


def boolean_value(document: dict[str, object], key: str) -> bool:
    value = document[key]
    if type(value) is not bool:
        raise ValueError(f"{key} must be a boolean")
    return value


def utc_datetime_value(document: dict[str, object], key: str) -> datetime:
    return parse_utc_datetime(string_value(document, key))


def parse_utc_datetime(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("timestamp is not ISO 8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp requires an explicit UTC offset")
    return parsed


__all__ = [
    "boolean_value",
    "integer_value",
    "load_contract",
    "load_contract_sequence",
    "load_exact_object",
    "load_phase_review_evidence",
    "optional_string_value",
    "parse_utc_datetime",
    "read_signature",
    "read_utf8",
    "string_tuple",
    "string_value",
    "utc_datetime_value",
    "write_contract",
]
