"""Prepare immutable whole-history retry carry evidence before provider construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
from oamb.contracts.evidence import (
    HistoryRetryAllowance,
    HistoryRetryCarryRecord,
    history_retry_carry_id,
)

from .composition import (
    CapsuleCompositionError,
    _aborted_group_is_terminal_known,
    _compatibility_family_key,
    _contributions_by_plan,
    _load_part,
    _part_binding,
    _PartSnapshot,
    _require_attempted_groups_recoverable,
)


def prepare_history_recovery(
    *,
    part_roots: tuple[Path, ...],
    capsule_root: Path,
    run_id: str,
    selected_ingestion_plan_ids: tuple[str, ...],
    max_retries_per_operation: int,
    memory_system_id: str | None,
    execution_configuration_family_hash: str,
) -> HistoryRetryCarryRecord:
    """Embed terminal-known ABORTED predecessors for fresh-scope reconstruction."""

    if not part_roots:
        raise CapsuleCompositionError("history recovery requires predecessor parts")
    if not run_id:
        raise CapsuleCompositionError("history recovery identity is incomplete")
    if max_retries_per_operation not in {0, 1, 2}:
        raise CapsuleCompositionError("history recovery retry limit must be 0, 1, or 2")
    if not selected_ingestion_plan_ids or len(set(selected_ingestion_plan_ids)) != len(
        selected_ingestion_plan_ids
    ):
        raise CapsuleCompositionError("history recovery plans must be non-empty and unique")
    resolved = tuple(Path(root).resolve(strict=True) for root in part_roots)
    if len(set(resolved)) != len(resolved):
        raise CapsuleCompositionError("history recovery parts must be unique")
    parts = tuple(_load_part(root) for root in resolved)
    if len({part.manifest.capsule_id for part in parts}) != len(parts):
        raise CapsuleCompositionError("history recovery capsule identities must be unique")

    evidence_parts = _history_part_graph(parts)
    if memory_system_id is None:
        derived_memory_ids = {
            *(item.memory_system_id for part in evidence_parts for item in part.plans),
            *(item.memory_system_id for part in evidence_parts for item in part.history_attempts),
            *(
                part.run_spec.memory_system_id
                for part in evidence_parts
                if part.run_spec is not None
            ),
        }
        if len(derived_memory_ids) != 1:
            raise CapsuleCompositionError("history recovery memory-system identity is not singular")
        memory_system_id = next(iter(derived_memory_ids))
    target_execution_configuration_family_hash = execution_configuration_family_hash
    _require_recovery_compatibility(
        evidence_parts,
        selected_ingestion_plan_ids,
        memory_system_id,
        target_execution_configuration_family_hash,
    )
    _require_closed_recovery_graph(evidence_parts, selected_ingestion_plan_ids)
    attempts = tuple(item for part in evidence_parts for item in part.history_attempts)
    if len({item.history_attempt_id for item in attempts}) != len(attempts):
        raise CapsuleCompositionError("duplicate history attempt evidence")
    allowances: list[HistoryRetryAllowance] = []
    for plan_id in sorted(selected_ingestion_plan_ids):
        plan_attempts = tuple(item for item in attempts if item.ingestion_plan_id == plan_id)
        if plan_attempts and not _aborted_group_is_terminal_known(
            evidence_parts,
            plan_id,
            plan_attempts,
        ):
            raise CapsuleCompositionError(
                "history group is not eligible for terminal-known fresh-scope rebuild"
            )
        allowances.append(
            HistoryRetryAllowance(
                ingestion_plan_id=plan_id,
                next_history_attempt_ordinal=1,
                execution_run_id=run_id,
                previous_retry_event_id=None,
                predecessor_history_attempt_id=None,
                consumed_retries=0,
            )
        )

    ordered_parts = tuple(sorted(parts, key=lambda item: item.manifest.capsule_id))
    bindings = tuple(_part_binding(part) for part in ordered_parts)
    fields = {
        "schema_name": "history_retry_carry_record",
        "schema_version": 1,
        "run_id": run_id,
        "source_part_bindings": bindings,
        "allowances": tuple(allowances),
        "retry_policy_hash": ordered_parts[0].partition.retry_policy_hash,
        "max_retries_per_operation": max_retries_per_operation,
        "execution_configuration_family_hash": (target_execution_configuration_family_hash),
    }
    carry = HistoryRetryCarryRecord.model_validate(
        {"carry_record_id": history_retry_carry_id(fields), **fields}
    )
    _embed_parts(ordered_parts, Path(capsule_root))
    return carry


def _require_closed_recovery_graph(
    evidence_parts: tuple[_PartSnapshot, ...],
    selected_ingestion_plan_ids: tuple[str, ...],
) -> None:
    contributions_by_plan = _contributions_by_plan(evidence_parts)
    _require_attempted_groups_recoverable(evidence_parts, contributions_by_plan)
    selected_plan_ids = set(selected_ingestion_plan_ids)
    if selected_plan_ids.intersection(contributions_by_plan):
        raise CapsuleCompositionError("completed history group cannot be rebuilt")
    source_selected_plan_ids = {
        plan_id for part in evidence_parts for plan_id in part.partition.selected_ingestion_plan_ids
    }
    if source_selected_plan_ids - selected_plan_ids - set(contributions_by_plan):
        raise CapsuleCompositionError(
            "history recovery source contains an incomplete unselected group"
        )


def _require_recovery_compatibility(
    parts: tuple[_PartSnapshot, ...],
    selected_plan_ids: tuple[str, ...],
    memory_system_id: str,
    execution_configuration_family_hash: str,
) -> None:
    if any(
        _compatibility_family_key(part) != execution_configuration_family_hash for part in parts
    ):
        raise CapsuleCompositionError("history recovery execution configuration family drifted")
    retry_hashes = {part.partition.retry_policy_hash for part in parts}
    if len(retry_hashes) != 1:
        raise CapsuleCompositionError("history recovery retry policy drifted")
    retry_policy_hash = parts[0].partition.retry_policy_hash
    known_plan_ids = {
        plan.ingestion_plan_id for part in parts for plan in part.case_manifest.ingestion_plans
    }
    if not set(selected_plan_ids) <= known_plan_ids:
        raise CapsuleCompositionError("unknown selected history recovery group")
    for part in parts:
        bound_memory_ids = {
            *(item.memory_system_id for item in part.plans),
            *(item.memory_system_id for item in part.history_attempts),
        }
        if part.run_spec is not None:
            bound_memory_ids.add(part.run_spec.memory_system_id)
        if bound_memory_ids and bound_memory_ids != {memory_system_id}:
            raise CapsuleCompositionError("history recovery memory-system binding drifted")
        if any(
            item.retry_policy_hash != retry_policy_hash for item in part.history_attempts
        ) or any(item.retry_policy_hash != retry_policy_hash for item in part.history_retry_events):
            raise CapsuleCompositionError("history evidence retry policy drifted")


def _history_part_graph(parts: tuple[_PartSnapshot, ...]) -> tuple[_PartSnapshot, ...]:
    discovered: dict[str, _PartSnapshot] = {}

    def visit(part: _PartSnapshot) -> None:
        existing = discovered.get(part.manifest.capsule_id)
        if existing is not None:
            if existing.manifest_bytes != part.manifest_bytes:
                raise CapsuleCompositionError("history source capsule identity collides")
            return
        discovered[part.manifest.capsule_id] = part
        for carry in part.history_retry_carries:
            for binding in carry.source_part_bindings:
                nested = _load_part(part.root / binding.embedded_root)
                if (
                    nested.manifest.capsule_id != binding.capsule_id
                    or hashlib.sha256(nested.manifest_bytes).hexdigest() != binding.manifest_sha256
                ):
                    raise CapsuleCompositionError("history source binding does not close")
                visit(nested)

    for part in parts:
        visit(part)
    return tuple(discovered[item] for item in sorted(discovered))


def _embed_parts(parts: tuple[_PartSnapshot, ...], capsule_root: Path) -> None:
    for part in parts:
        manifest = part.manifest
        prefix = capsule_root / "source" / "parts" / manifest.capsule_id
        atomic_write_bytes(
            prefix / "capsule-manifest.json",
            part.manifest_bytes,
            trusted_root=capsule_root,
        )
        for entry in manifest.source_entries:
            content = read_regular_file(part.root / entry.relative_path)
            if hashlib.sha256(content).hexdigest() != entry.sha256:
                raise CapsuleCompositionError("modified history recovery part")
            atomic_write_bytes(
                prefix / entry.relative_path,
                content,
                trusted_root=capsule_root,
            )


__all__ = ["prepare_history_recovery"]
