"""Create-only composition of compatible immutable case-partition capsules."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file, sha256_file
from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.native import validate_partition_capsule
from oamb.contracts.evidence import (
    AttemptIntentRecord,
    AttemptIntentRecordV3,
    AttemptReceiptKind,
    AttemptReceiptRecord,
    AttemptRecordV2,
    AttemptRecordV4,
    CapsuleCompositionContribution,
    CapsuleCompositionPartBinding,
    CapsuleCompositionRecord,
    CapsuleManifest,
    CapsuleManifestEntry,
    CaseRecordV3,
    HistoryAttemptRecord,
    HistoryRetryCarryRecord,
    HistoryRetryEvent,
    IngestionPlanRecordV2,
    IngestionPlanRecordV3,
    RunRecord,
    capsule_composition_contribution_hash,
    capsule_composition_id,
    capsule_composition_part_binding_hash,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.schema import parse_contract
from oamb.contracts.specifications import (
    BudgetSpecV4,
    CaseManifest,
    CasePartitionSpec,
    DatasetManifest,
    ModelRoleBindingV2,
    RunPreflightRecord,
    RunPreflightRecordV2,
    RunSpec,
)
from oamb.contracts.states import (
    AttemptOutcome,
    CaseState,
    IngestionPlanState,
    RunState,
    ValidationDisposition,
)

from .recovery_safety import history_group_is_terminal_known

COMPOSITION_VALIDATION_PROFILE_ID = "oamb-capsule-composition-v1"


class CapsuleCompositionError(ValueError):
    """Immutable parts cannot form one exact compatible target cell."""


@dataclass(frozen=True, slots=True)
class CapsuleCompositionArtifacts:
    capsule_root: Path
    manifest: CapsuleManifest
    composition: CapsuleCompositionRecord


@dataclass(frozen=True, slots=True)
class CapsuleRecoveryPlan:
    """Deterministic whole-group reuse and fresh-scope recovery selection."""

    source_capsule_ids: tuple[str, ...]
    source_manifest_sha256s: tuple[str, ...]
    reusable_ingestion_plan_ids: tuple[str, ...]
    quarantined_ingestion_plan_ids: tuple[str, ...]
    remaining_ingestion_plan_ids: tuple[str, ...]
    remaining_case_manifest_entry_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CapsuleCompositionTarget:
    resolved_plan_hash: str
    cell_spec_hash: str
    target_case_manifest_hash: str
    budget_policy_hash: str
    retry_policy_hash: str
    execution_configuration_hash: str | None = None
    execution_configuration_family_hash: str | None = None


@dataclass(frozen=True, slots=True)
class _PartSnapshot:
    root: Path
    manifest: CapsuleManifest
    manifest_bytes: bytes
    partition: CasePartitionSpec
    case_manifest: CaseManifest
    dataset_manifest: DatasetManifest
    run: RunRecord
    plans: tuple[IngestionPlanRecordV2 | IngestionPlanRecordV3, ...]
    cases: tuple[CaseRecordV3, ...]
    attempts: tuple[AttemptRecordV2 | AttemptRecordV4, ...]
    intents: tuple[AttemptIntentRecord | AttemptIntentRecordV3, ...]
    receipts: tuple[AttemptReceiptRecord, ...]
    history_attempts: tuple[HistoryAttemptRecord, ...]
    history_retry_events: tuple[HistoryRetryEvent, ...]
    history_retry_carries: tuple[HistoryRetryCarryRecord, ...]
    run_spec: RunSpec | None
    preflight: RunPreflightRecord | RunPreflightRecordV2 | None
    budget: BudgetSpecV4 | None
    role_bindings: tuple[ModelRoleBindingV2, ...]


def compose_capsules(
    part_roots: tuple[Path, ...],
    output_root: Path,
    *,
    target: CapsuleCompositionTarget | None = None,
) -> CapsuleCompositionArtifacts:
    """Validate exact immutable parts, then publish one deterministic outer capsule."""

    output = Path(output_root)
    if output.exists() or output.is_symlink():
        raise CapsuleCompositionError("composition output already exists")
    if not part_roots:
        raise CapsuleCompositionError("missing composition parts")
    resolved_roots = tuple(Path(root).resolve(strict=True) for root in part_roots)
    resolved_output = output.resolve(strict=False)
    if any(
        resolved_output == root
        or resolved_output.is_relative_to(root)
        or root.is_relative_to(resolved_output)
        for root in resolved_roots
    ):
        raise CapsuleCompositionError("composition output overlaps a source part")
    if len(set(resolved_roots)) != len(resolved_roots):
        raise CapsuleCompositionError("duplicate composition part")
    parts = tuple(_load_part(root) for root in resolved_roots)
    if len({part.manifest.capsule_id for part in parts}) != len(parts):
        raise CapsuleCompositionError("duplicate composition capsule")
    composition = _compose_record(parts, target=target)
    if target is not None and (
        composition.resolved_plan_hash != target.resolved_plan_hash
        or composition.cell_spec_hash != target.cell_spec_hash
        or composition.target_case_manifest_hash != target.target_case_manifest_hash
        or composition.budget_policy_hash != target.budget_policy_hash
        or composition.retry_policy_hash != target.retry_policy_hash
    ):
        raise CapsuleCompositionError("composition parts do not match the requested target")
    entries = _publish_composition(parts, composition, output)
    run_spec_hash = canonical_sha256(["oamb-composed-run-spec-v1", composition.composition_id])
    source_manifest_hash = canonical_sha256(
        [
            "oamb-source-manifest-v1",
            tuple(entry.model_dump(mode="python") for entry in entries),
        ]
    )
    manifest = CapsuleManifest(
        capsule_id=canonical_sha256(
            [
                "oamb-capsule-v1",
                composition.composition_id,
                run_spec_hash,
                source_manifest_hash,
            ]
        ),
        run_id=composition.composition_id,
        run_spec_hash=run_spec_hash,
        source_entries=entries,
        source_manifest_hash=source_manifest_hash,
    )
    ArtifactStore(output).seal_capsule(manifest)
    return CapsuleCompositionArtifacts(output, manifest, composition)


def analyze_capsule_recovery(
    part_roots: tuple[Path, ...],
    *,
    target: CapsuleCompositionTarget | None = None,
) -> CapsuleRecoveryPlan:
    """Validate immutable parts and derive the target-order fresh-scope complement."""

    if not part_roots:
        raise CapsuleCompositionError("missing recovery parts")
    resolved_roots = tuple(Path(root).resolve(strict=True) for root in part_roots)
    if len(set(resolved_roots)) != len(resolved_roots):
        raise CapsuleCompositionError("duplicate recovery part")
    parts = tuple(_load_part(root) for root in resolved_roots)
    if len({part.manifest.capsule_id for part in parts}) != len(parts):
        raise CapsuleCompositionError("duplicate recovery capsule")
    _require_compatible_parts(parts, target=target)
    contributions_by_plan = _contributions_by_plan(parts)
    first = parts[0]
    selected = {plan_id for part in parts for plan_id in part.partition.selected_ingestion_plan_ids}
    target_plans = tuple(
        plan for plan in first.case_manifest.ingestion_plans if plan.ingestion_plan_id in selected
    )
    target_plan_ids = tuple(plan.ingestion_plan_id for plan in target_plans)
    if selected != set(target_plan_ids):
        raise CapsuleCompositionError("unknown selected recovery group")
    if not set(contributions_by_plan) <= set(target_plan_ids):
        raise CapsuleCompositionError("unknown recovery contribution")
    _require_attempted_groups_recoverable(parts, contributions_by_plan)
    reusable = tuple(plan_id for plan_id in target_plan_ids if plan_id in contributions_by_plan)
    quarantined = tuple(
        plan_id
        for plan_id in target_plan_ids
        if plan_id in selected and plan_id not in contributions_by_plan
    )
    remaining_plans = tuple(
        plan for plan in target_plans if plan.ingestion_plan_id not in contributions_by_plan
    )
    ordered_parts = tuple(sorted(parts, key=lambda part: part.manifest.capsule_id))
    return CapsuleRecoveryPlan(
        source_capsule_ids=tuple(part.manifest.capsule_id for part in ordered_parts),
        source_manifest_sha256s=tuple(
            hashlib.sha256(part.manifest_bytes).hexdigest() for part in ordered_parts
        ),
        reusable_ingestion_plan_ids=reusable,
        quarantined_ingestion_plan_ids=quarantined,
        remaining_ingestion_plan_ids=tuple(plan.ingestion_plan_id for plan in remaining_plans),
        remaining_case_manifest_entry_ids=tuple(
            case_id for plan in remaining_plans for case_id in plan.ordered_case_manifest_entry_ids
        ),
    )


def _require_attempted_groups_recoverable(
    parts: tuple[_PartSnapshot, ...],
    contributions_by_plan: dict[str, CapsuleCompositionContribution],
) -> None:
    attempts = tuple(item for part in parts for item in part.history_attempts)
    if (
        any(item.status == "unknown" for item in attempts)
        or any(
            item.outcome == AttemptOutcome.UNKNOWN_OUTCOME
            for part in parts
            for item in part.attempts
        )
        or any(
            item.receipt_kind == AttemptReceiptKind.UNKNOWN_OUTCOME
            for part in parts
            for item in part.receipts
        )
    ):
        raise CapsuleCompositionError("attempted history group has unknown outcome")
    parts_by_capsule_id = {part.manifest.capsule_id: part for part in parts}
    for plan_id, contribution in contributions_by_plan.items():
        contributor = parts_by_capsule_id[contribution.source_capsule_id]
        carried_capsule_ids = {
            binding.capsule_id
            for carry in contributor.history_retry_carries
            for binding in carry.source_part_bindings
        }
        attempted_predecessor_capsule_ids = {
            part.manifest.capsule_id
            for part in parts
            if part is not contributor
            and any(item.ingestion_plan_id == plan_id for item in part.history_attempts)
        }
        if not attempted_predecessor_capsule_ids <= carried_capsule_ids:
            raise CapsuleCompositionError(
                "completed history group was attempted outside its validated recovery chain"
            )
    selected_plan_ids = {
        plan_id for part in parts for plan_id in part.partition.selected_ingestion_plan_ids
    }
    for plan_id in selected_plan_ids - set(contributions_by_plan):
        owning_parts = tuple(
            part for part in parts if plan_id in part.partition.selected_ingestion_plan_ids
        )
        if not owning_parts or any(part.run.state != RunState.ABORTED for part in owning_parts):
            raise CapsuleCompositionError(
                "incomplete history group is not owned only by aborted parts"
            )
    for plan_id in {item.ingestion_plan_id for item in attempts} - set(contributions_by_plan):
        plan_attempts = tuple(item for item in attempts if item.ingestion_plan_id == plan_id)
        if _aborted_group_is_terminal_known(parts, plan_id, plan_attempts):
            continue
        raise CapsuleCompositionError(
            "attempted history group is not eligible for terminal-known fresh-scope rebuild"
        )


def _aborted_group_is_terminal_known(
    parts: tuple[_PartSnapshot, ...],
    plan_id: str,
    history_attempts: tuple[HistoryAttemptRecord, ...],
) -> bool:
    owning_parts = tuple(
        part
        for part in parts
        if any(item.ingestion_plan_id == plan_id for item in part.history_attempts)
    )
    if not owning_parts or any(part.run.state != RunState.ABORTED for part in owning_parts):
        return False
    return history_group_is_terminal_known(
        history_attempts,
        tuple(attempt for part in owning_parts for attempt in part.attempts),
        tuple(receipt for part in owning_parts for receipt in part.receipts),
    )


def load_composition_record(root: Path) -> CapsuleCompositionRecord:
    manifest = CapsuleManifest.model_validate_json(
        read_regular_file(Path(root) / "capsule-manifest.json")
    )
    entries = tuple(
        entry
        for entry in manifest.source_entries
        if entry.record_kind == "capsule_composition_record"
    )
    if len(entries) != 1:
        raise CapsuleCompositionError("composition record inventory is not singular")
    return CapsuleCompositionRecord.model_validate_json(
        read_regular_file(Path(root) / entries[0].relative_path)
    )


def inspect_embedded_composition(root: Path) -> tuple[CapsuleCompositionRecord, tuple[Path, ...]]:
    composition = load_composition_record(root)
    resolved_root = Path(root).resolve(strict=True)
    embedded_list: list[Path] = []
    for part in composition.ordered_parts:
        expected = f"source/parts/{part.capsule_id}"
        if part.embedded_root != expected:
            raise CapsuleCompositionError("composition part embedded root is not canonical")
        candidate = resolved_root / part.embedded_root
        if any(
            current.is_symlink()
            for current in (
                resolved_root / "source",
                resolved_root / "source" / "parts",
                candidate,
            )
        ):
            raise CapsuleCompositionError("composition part embedded root uses a symbolic link")
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_relative_to(resolved_root):
            raise CapsuleCompositionError("composition part embedded root escapes the capsule")
        embedded_list.append(resolved_candidate)
    embedded = tuple(embedded_list)
    return composition, embedded


def _load_part(root: Path) -> _PartSnapshot:
    manifest_path = root / "capsule-manifest.json"
    manifest_bytes = read_regular_file(manifest_path)
    manifest = CapsuleManifest.model_validate_json(manifest_bytes)
    validation = validate_partition_capsule(root)
    if validation.disposition != ValidationDisposition.VALIDATED:
        raise CapsuleCompositionError("unvalidated composition part")
    contracts: list[BaseModel] = []
    for entry in manifest.source_entries:
        path = root / entry.relative_path
        if path.is_symlink() or sha256_file(path) != entry.sha256:
            raise CapsuleCompositionError("modified composition part")
        if entry.record_kind in {"raw_payload", "embedded_part_file"}:
            continue
        content = read_regular_file(path)
        document = json.loads(content)
        if not isinstance(document, dict):
            raise CapsuleCompositionError("malformed composition part")
        contracts.append(parse_contract(document))
    partition = _one(contracts, CasePartitionSpec, "partition")
    case_manifest = _one(contracts, CaseManifest, "case manifest")
    dataset_manifest = _one(contracts, DatasetManifest, "dataset manifest")
    run = _one(contracts, RunRecord, "run")
    plans = tuple(
        item
        for item in contracts
        if isinstance(item, (IngestionPlanRecordV2, IngestionPlanRecordV3))
    )
    cases = tuple(item for item in contracts if isinstance(item, CaseRecordV3))
    attempts = tuple(
        item for item in contracts if isinstance(item, (AttemptRecordV2, AttemptRecordV4))
    )
    intents = tuple(
        item for item in contracts if isinstance(item, (AttemptIntentRecord, AttemptIntentRecordV3))
    )
    receipts = tuple(item for item in contracts if isinstance(item, AttemptReceiptRecord))
    history_attempts = tuple(item for item in contracts if isinstance(item, HistoryAttemptRecord))
    history_retry_events = tuple(item for item in contracts if isinstance(item, HistoryRetryEvent))
    history_retry_carries = tuple(
        item for item in contracts if isinstance(item, HistoryRetryCarryRecord)
    )
    run_specs = tuple(item for item in contracts if isinstance(item, RunSpec))
    preflights = tuple(
        item for item in contracts if isinstance(item, (RunPreflightRecord, RunPreflightRecordV2))
    )
    budgets = tuple(item for item in contracts if isinstance(item, BudgetSpecV4))
    role_bindings = tuple(item for item in contracts if isinstance(item, ModelRoleBindingV2))
    control_counts = (len(run_specs), len(preflights), len(budgets))
    if any(control_counts) and control_counts != (1, 1, 1):
        raise CapsuleCompositionError("composition part live control inventory is incomplete")
    return _PartSnapshot(
        root=root,
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        partition=partition,
        case_manifest=case_manifest,
        dataset_manifest=dataset_manifest,
        run=run,
        plans=plans,
        cases=cases,
        attempts=attempts,
        intents=intents,
        receipts=receipts,
        history_attempts=history_attempts,
        history_retry_events=history_retry_events,
        history_retry_carries=history_retry_carries,
        run_spec=run_specs[0] if run_specs else None,
        preflight=preflights[0] if preflights else None,
        budget=budgets[0] if budgets else None,
        role_bindings=role_bindings,
    )


def _compose_record(
    parts: tuple[_PartSnapshot, ...],
    *,
    target: CapsuleCompositionTarget | None = None,
) -> CapsuleCompositionRecord:
    first = parts[0]
    compatibility = _require_compatible_parts(parts, target=target)
    operation_attempt_ids = tuple(attempt.attempt_id for part in parts for attempt in part.attempts)
    if len(set(operation_attempt_ids)) != len(operation_attempt_ids):
        raise CapsuleCompositionError("duplicate operation attempt across composition parts")
    part_bindings = tuple(
        sorted((_part_binding(part) for part in parts), key=lambda item: item.capsule_id)
    )
    contributions_by_plan = _contributions_by_plan(parts)
    _require_attempted_groups_recoverable(parts, contributions_by_plan)
    target_plan_ids = tuple(plan.ingestion_plan_id for plan in first.case_manifest.ingestion_plans)
    missing = tuple(plan_id for plan_id in target_plan_ids if plan_id not in contributions_by_plan)
    if missing:
        raise CapsuleCompositionError("missing composition contributions")
    if set(contributions_by_plan) != set(target_plan_ids):
        raise CapsuleCompositionError("unknown composition contribution")
    contributions = tuple(contributions_by_plan[plan_id] for plan_id in target_plan_ids)
    fields = {
        "schema_name": "capsule_composition_record",
        "schema_version": 1,
        "resolved_plan_hash": first.partition.resolved_plan_hash,
        "cell_spec_hash": first.partition.cell_spec_hash,
        "dataset_manifest_hash": first.partition.dataset_manifest_hash,
        "target_case_manifest_hash": first.partition.target_case_manifest_hash,
        "target_case_execution_bindings_hash": (
            first.partition.target_case_execution_bindings_hash
        ),
        "budget_policy_hash": first.partition.budget_policy_hash,
        "retry_policy_hash": first.partition.retry_policy_hash,
        "execution_configuration_hash": compatibility,
        "ordered_parts": part_bindings,
        "ordered_contributions": contributions,
        "exact_union_hash": canonical_sha256(
            ["oamb-capsule-composition-exact-union-v1", contributions]
        ),
    }
    if target is not None and compatibility not in {_compatibility_key(part) for part in parts}:
        fields.update(
            {
                "target_execution_configuration_hash": (target.execution_configuration_hash),
                "target_execution_configuration_family_hash": (
                    target.execution_configuration_family_hash
                ),
            }
        )
    return CapsuleCompositionRecord.model_validate(
        {"composition_id": capsule_composition_id(fields), **fields}
    )


def _require_compatible_parts(
    parts: tuple[_PartSnapshot, ...],
    *,
    target: CapsuleCompositionTarget | None = None,
) -> str:
    first = parts[0]
    compatibility_keys = tuple(_compatibility_key(part) for part in parts)
    compatibility = compatibility_keys[0]
    if target is not None and (target.execution_configuration_hash is None) != (
        target.execution_configuration_family_hash is None
    ):
        raise CapsuleCompositionError(
            "recovery execution configuration hashes must be supplied together"
        )
    if target is not None and (
        first.partition.resolved_plan_hash != target.resolved_plan_hash
        or first.partition.cell_spec_hash != target.cell_spec_hash
        or first.partition.target_case_manifest_hash != target.target_case_manifest_hash
        or first.partition.budget_policy_hash != target.budget_policy_hash
        or first.partition.retry_policy_hash != target.retry_policy_hash
    ):
        raise CapsuleCompositionError("recovery parts do not match the requested target")
    family_keys: tuple[str, ...] | None = None
    if target is not None and target.execution_configuration_family_hash is not None:
        family_keys = tuple(_compatibility_family_key(part) for part in parts)
        if (
            len(set(family_keys)) != 1
            or family_keys[0] != target.execution_configuration_family_hash
        ):
            raise CapsuleCompositionError(
                "recovery parts do not match the requested execution configuration family"
            )
    target_configuration_matches = bool(
        target is None
        or target.execution_configuration_hash is None
        or compatibility == target.execution_configuration_hash
    )
    if len(set(compatibility_keys)) == 1 and target_configuration_matches:
        return compatibility
    if target is None:
        raise CapsuleCompositionError("incompatible composition parts")
    if (
        target.execution_configuration_hash is None
        or target.execution_configuration_family_hash is None
    ):
        raise CapsuleCompositionError(
            "recovery parts do not match the requested execution configuration"
        )
    if any(
        compatibility_key != target.execution_configuration_hash
        and part.run.state != RunState.ABORTED
        for part, compatibility_key in zip(parts, compatibility_keys, strict=True)
    ):
        raise CapsuleCompositionError(
            "revision-drifted recovery part is not aborted and does not match the requested "
            "execution configuration"
        )
    assert family_keys is not None
    return canonical_sha256(
        [
            "oamb-mixed-code-revision-execution-configuration-v1",
            target.execution_configuration_hash,
            target.execution_configuration_family_hash,
            tuple(sorted(compatibility_keys)),
        ]
    )


def _contributions_by_plan(
    parts: tuple[_PartSnapshot, ...],
) -> dict[str, CapsuleCompositionContribution]:
    contributions: dict[str, CapsuleCompositionContribution] = {}
    for part in parts:
        for contribution in _terminal_contributions(part):
            if contribution.ingestion_plan_id in contributions:
                raise CapsuleCompositionError("overlap between composition contributions")
            contributions[contribution.ingestion_plan_id] = contribution
    return contributions


def _compatibility_key(part: _PartSnapshot) -> str:
    return _part_execution_configuration_hash(part, include_code_revision=True)


def _compatibility_family_key(part: _PartSnapshot) -> str:
    return _part_execution_configuration_hash(part, include_code_revision=False)


def _part_execution_configuration_hash(
    part: _PartSnapshot,
    *,
    include_code_revision: bool,
) -> str:
    runtime_bindings = tuple(
        (
            plan.memory_system_id,
            plan.adapter_profile_id,
            plan.runtime_binding_hash,
        )
        for plan in part.plans
    )
    if part.run_spec is not None and part.preflight is not None:
        controlled_runtime_binding = (
            part.run_spec.memory_system_id,
            part.preflight.adapter_profile_id,
            part.run_spec.runtime_binding_hash,
        )
        if runtime_bindings and any(
            runtime_binding != controlled_runtime_binding for runtime_binding in runtime_bindings
        ):
            raise CapsuleCompositionError("runtime binding inventory is inconsistent")
        runtime_bindings = (controlled_runtime_binding,)
    builder = (
        build_composition_execution_configuration_hash
        if include_code_revision
        else build_composition_execution_configuration_family_hash
    )
    return builder(
        partition=part.partition,
        case_manifest=part.case_manifest,
        dataset_manifest=part.dataset_manifest,
        runtime_bindings=runtime_bindings,
        run_spec=part.run_spec,
        preflight=part.preflight,
        budget=part.budget,
        role_bindings=part.role_bindings,
    )


def build_composition_execution_configuration_hash(
    *,
    partition: CasePartitionSpec,
    case_manifest: CaseManifest,
    dataset_manifest: DatasetManifest,
    runtime_bindings: tuple[tuple[str, str, str], ...],
    run_spec: RunSpec | None,
    preflight: RunPreflightRecord | RunPreflightRecordV2 | None,
    budget: BudgetSpecV4 | None,
    role_bindings: tuple[ModelRoleBindingV2, ...],
) -> str:
    return _build_composition_execution_configuration_hash(
        partition=partition,
        case_manifest=case_manifest,
        dataset_manifest=dataset_manifest,
        runtime_bindings=runtime_bindings,
        run_spec=run_spec,
        preflight=preflight,
        budget=budget,
        role_bindings=role_bindings,
        include_code_revision=True,
    )


def build_composition_execution_configuration_family_hash(
    *,
    partition: CasePartitionSpec,
    case_manifest: CaseManifest,
    dataset_manifest: DatasetManifest,
    runtime_bindings: tuple[tuple[str, str, str], ...],
    run_spec: RunSpec | None,
    preflight: RunPreflightRecord | RunPreflightRecordV2 | None,
    budget: BudgetSpecV4 | None,
    role_bindings: tuple[ModelRoleBindingV2, ...],
) -> str:
    """Bind recovery execution controls while excluding only the source revision."""

    return _build_composition_execution_configuration_hash(
        partition=partition,
        case_manifest=case_manifest,
        dataset_manifest=dataset_manifest,
        runtime_bindings=runtime_bindings,
        run_spec=run_spec,
        preflight=preflight,
        budget=budget,
        role_bindings=role_bindings,
        include_code_revision=False,
    )


def _build_composition_execution_configuration_hash(
    *,
    partition: CasePartitionSpec,
    case_manifest: CaseManifest,
    dataset_manifest: DatasetManifest,
    runtime_bindings: tuple[tuple[str, str, str], ...],
    run_spec: RunSpec | None,
    preflight: RunPreflightRecord | RunPreflightRecordV2 | None,
    budget: BudgetSpecV4 | None,
    role_bindings: tuple[ModelRoleBindingV2, ...],
    include_code_revision: bool,
) -> str:
    live_configuration: object = "fixture"
    control_records = (run_spec, preflight, budget)
    if all(record is not None for record in control_records):
        assert run_spec is not None and preflight is not None and budget is not None
        run_spec_exclusions = {"run_id", "budget_id"}
        if not include_code_revision:
            run_spec_exclusions.add("code_revision")
        live_configuration = {
            "run_spec": run_spec.model_dump(
                mode="python",
                exclude=run_spec_exclusions,
            ),
            "preflight": preflight.model_dump(
                mode="python",
                exclude={
                    "preflight_record_hash",
                    "run_id",
                    "observed_at",
                    "run_spec_hash",
                    "budget_hash",
                    "artifact_repository_fingerprint",
                    "artifact_durability_proof_hash",
                },
            ),
            "budget": budget.model_dump(
                mode="python",
                exclude={"budget_id", "budget_hash", "scope_id"},
            ),
            "role_bindings": tuple(
                binding.model_dump(mode="python")
                for binding in sorted(role_bindings, key=lambda item: item.binding_id)
            ),
        }
    elif any(record is not None for record in control_records):
        raise CapsuleCompositionError("execution configuration control inventory is incomplete")
    return canonical_sha256(
        [
            (
                "oamb-composition-execution-configuration-v1"
                if include_code_revision
                else "oamb-composition-execution-configuration-family-v1"
            ),
            partition.resolved_plan_hash,
            partition.cell_spec_hash,
            partition.dataset_manifest_hash,
            partition.target_case_manifest_hash,
            partition.target_case_execution_bindings,
            partition.target_case_execution_bindings_hash,
            partition.budget_policy_hash,
            partition.retry_policy_hash,
            case_manifest,
            dataset_manifest,
            tuple(dict.fromkeys(runtime_bindings)),
            live_configuration,
        ]
    )


def capsule_execution_configuration_hash(root: Path) -> str:
    """Reopen one validated immutable capsule and return its composition key."""

    return _compatibility_key(_load_part(Path(root).resolve(strict=True)))


def capsule_execution_configuration_family_hash(root: Path) -> str:
    """Reopen one validated capsule and return its revision-excluding family key."""

    return _compatibility_family_key(_load_part(Path(root).resolve(strict=True)))


def _part_binding(part: _PartSnapshot) -> CapsuleCompositionPartBinding:
    fields = {
        "schema_name": "capsule_composition_part_binding",
        "schema_version": 1,
        "capsule_id": part.manifest.capsule_id,
        "run_id": part.manifest.run_id,
        "manifest_sha256": hashlib.sha256(part.manifest_bytes).hexdigest(),
        "partition_id": part.partition.partition_id,
        "embedded_root": f"source/parts/{part.manifest.capsule_id}",
        "run_state": part.run.state,
    }
    return CapsuleCompositionPartBinding.model_validate(
        {"part_binding_hash": capsule_composition_part_binding_hash(fields), **fields}
    )


def _terminal_contributions(
    part: _PartSnapshot,
) -> tuple[CapsuleCompositionContribution, ...]:
    if part.run.state not in {
        RunState.FINALIZED,
        RunState.ABORTED,
        RunState.INFRASTRUCTURE_BLOCKED,
    }:
        raise CapsuleCompositionError("unsealed composition part")
    cases_by_occurrence = {case.case_occurrence_id: case for case in part.cases}
    contributions: list[CapsuleCompositionContribution] = []
    for plan in part.plans:
        if plan.state != IngestionPlanState.SEALED:
            continue
        physical_history = tuple(
            item
            for item in part.history_attempts
            if item.ingestion_plan_id == plan.ingestion_plan_id
        )
        if physical_history:
            selected_ready = tuple(item for item in physical_history if item.status == "ready")
            if (
                len(selected_ready) != 1
                or selected_ready[0].ingestion_occurrence_id != plan.ingestion_occurrence_id
                or selected_ready[0].ingestion_plan_record_hash != canonical_sha256(plan)
            ):
                raise CapsuleCompositionError("history ready selection does not close")
        cases = tuple(
            cases_by_occurrence.get(case_id) for case_id in plan.ordered_case_occurrence_ids
        )
        if any(case is None or case.state != CaseState.COMPLETED for case in cases):
            continue
        # Fresh part validation has already checked every retry chain. Failed
        # predecessors do not invalidate a sealed plan with completed cases.
        concrete_cases = tuple(case for case in cases if case is not None)
        fields = {
            "schema_name": "capsule_composition_contribution",
            "schema_version": 1,
            "source_capsule_id": part.manifest.capsule_id,
            "source_run_id": part.manifest.run_id,
            "ingestion_plan_id": plan.ingestion_plan_id,
            "ingestion_occurrence_id": plan.ingestion_occurrence_id,
            "case_manifest_entry_ids": tuple(
                case.case_manifest_entry_id for case in concrete_cases
            ),
            "case_occurrence_ids": tuple(case.case_occurrence_id for case in concrete_cases),
        }
        contributions.append(
            CapsuleCompositionContribution.model_validate(
                {
                    "contribution_hash": capsule_composition_contribution_hash(fields),
                    **fields,
                }
            )
        )
    return tuple(contributions)


def _publish_composition(
    parts: tuple[_PartSnapshot, ...],
    composition: CapsuleCompositionRecord,
    output: Path,
) -> tuple[CapsuleManifestEntry, ...]:
    payloads: list[tuple[str, str, str, bytes]] = []
    composition_path = f"source/composition/{composition.composition_id}.json"
    payloads.append(
        (
            "capsule_composition_record",
            composition.composition_id,
            composition_path,
            canonical_json_bytes(composition),
        )
    )
    for part in sorted(parts, key=lambda item: item.manifest.capsule_id):
        prefix = f"source/parts/{part.manifest.capsule_id}"
        payloads.append(
            (
                "embedded_part_file",
                canonical_sha256(
                    ["oamb-embedded-part-file-v1", part.manifest.capsule_id, "manifest"]
                ),
                f"{prefix}/capsule-manifest.json",
                part.manifest_bytes,
            )
        )
        for entry in part.manifest.source_entries:
            content = read_regular_file(part.root / entry.relative_path)
            if hashlib.sha256(content).hexdigest() != entry.sha256:
                raise CapsuleCompositionError("modified composition part")
            payloads.append(
                (
                    "embedded_part_file",
                    canonical_sha256(
                        [
                            "oamb-embedded-part-file-v1",
                            part.manifest.capsule_id,
                            entry.relative_path,
                        ]
                    ),
                    f"{prefix}/{entry.relative_path}",
                    content,
                )
            )
    entries: list[CapsuleManifestEntry] = []
    for record_kind, record_id, relative_path, content in sorted(
        payloads, key=lambda item: item[2]
    ):
        atomic_write_bytes(output / relative_path, content, trusted_root=output)
        entries.append(
            CapsuleManifestEntry(
                record_kind=record_kind,
                record_id=record_id,
                relative_path=relative_path,
                sha256=hashlib.sha256(content).hexdigest(),
            )
        )
    return tuple(entries)


def _one(contracts: list[BaseModel], kind: type[BaseModel], label: str):  # type: ignore[no-untyped-def]
    matches = tuple(item for item in contracts if isinstance(item, kind))
    if len(matches) != 1:
        raise CapsuleCompositionError(f"composition part {label} inventory is not singular")
    return matches[0]


__all__ = [
    "COMPOSITION_VALIDATION_PROFILE_ID",
    "CapsuleCompositionArtifacts",
    "CapsuleCompositionError",
    "CapsuleCompositionTarget",
    "build_composition_execution_configuration_hash",
    "build_composition_execution_configuration_family_hash",
    "capsule_execution_configuration_family_hash",
    "capsule_execution_configuration_hash",
    "compose_capsules",
    "inspect_embedded_composition",
    "load_composition_record",
]
