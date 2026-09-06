"""Prepare immutable whole-history retry carry evidence before provider construction."""

from __future__ import annotations

import hashlib
from pathlib import Path

from oamb.artifacts.atomic import atomic_write_bytes, read_regular_file
from oamb.contracts.evidence import (
    HistoryAttemptRecord,
    HistoryRetryAllowance,
    HistoryRetryCarryRecord,
    HistoryRetryEvent,
    history_retry_carry_id,
)

from .composition import CapsuleCompositionError, _load_part, _part_binding, _PartSnapshot


def prepare_history_recovery(
    *,
    part_roots: tuple[Path, ...],
    capsule_root: Path,
    run_id: str,
    selected_ingestion_plan_ids: tuple[str, ...],
    max_retries_per_operation: int,
    memory_system_id: str | None,
) -> tuple[HistoryRetryCarryRecord, tuple[HistoryRetryEvent, ...]]:
    """Embed validated predecessors and derive the only resumable retry allowances."""

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
    _require_recovery_compatibility(evidence_parts, selected_ingestion_plan_ids, memory_system_id)
    attempts = tuple(item for part in evidence_parts for item in part.history_attempts)
    events = tuple(item for part in evidence_parts for item in part.history_retry_events)
    if len({item.history_attempt_id for item in attempts}) != len(attempts):
        raise CapsuleCompositionError("duplicate history attempt evidence")
    if len({item.history_retry_event_id for item in events}) != len(events):
        raise CapsuleCompositionError("duplicate history retry event evidence")
    pending: list[HistoryRetryEvent] = []
    allowances: list[HistoryRetryAllowance] = []
    for plan_id in sorted(selected_ingestion_plan_ids):
        plan_attempts = tuple(
            sorted(
                (item for item in attempts if item.ingestion_plan_id == plan_id),
                key=lambda item: item.history_attempt_ordinal,
            )
        )
        if not plan_attempts:
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
            continue
        _require_history_chain(plan_attempts, events, max_retries_per_operation)
        last = plan_attempts[-1]
        if last.status == "ready":
            continue
        if last.status != "retryable_failed_settled":
            raise CapsuleCompositionError("history group has no retryable settled predecessor")
        matches = tuple(
            item
            for item in events
            if item.ingestion_plan_id == plan_id
            and item.failed_history_attempt_id == last.history_attempt_id
        )
        if len(matches) != 1 or not matches[0].retry_scheduled:
            raise CapsuleCompositionError("history group has no single pending scheduled retry")
        event = matches[0]
        if event.max_retries_per_operation != max_retries_per_operation:
            raise CapsuleCompositionError("history retry allowance differs from current policy")
        if any(
            item.history_attempt_ordinal == event.successor_history_attempt_ordinal
            for item in plan_attempts
        ):
            raise CapsuleCompositionError("scheduled history successor was already admitted")
        if any(
            intent.parent_id == event.successor_ingestion_occurrence_id
            for part in evidence_parts
            for intent in part.intents
        ):
            raise CapsuleCompositionError("scheduled history successor has dispatch intent")
        assert event.successor_execution_run_id is not None
        assert event.successor_history_attempt_ordinal is not None
        allowances.append(
            HistoryRetryAllowance(
                ingestion_plan_id=plan_id,
                next_history_attempt_ordinal=event.successor_history_attempt_ordinal,
                execution_run_id=event.successor_execution_run_id,
                previous_retry_event_id=event.history_retry_event_id,
                predecessor_history_attempt_id=last.history_attempt_id,
                consumed_retries=event.retry_ordinal,
            )
        )
        pending.append(event)

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
    }
    carry = HistoryRetryCarryRecord.model_validate(
        {"carry_record_id": history_retry_carry_id(fields), **fields}
    )
    _embed_parts(ordered_parts, Path(capsule_root))
    return carry, tuple(sorted(pending, key=lambda item: item.ingestion_plan_id))


def _require_history_chain(
    attempts: tuple[HistoryAttemptRecord, ...],
    events: tuple[HistoryRetryEvent, ...],
    max_retries_per_operation: int,
) -> None:
    ordinals = tuple(item.history_attempt_ordinal for item in attempts)
    if ordinals != tuple(range(1, len(attempts) + 1)):
        raise CapsuleCompositionError("history attempt ordinals do not form one chain")
    first = attempts[0]
    if any(
        item.max_retries_per_operation != max_retries_per_operation
        or item.ingestion_plan_id != first.ingestion_plan_id
        or item.memory_system_id != first.memory_system_id
        or item.runtime_binding_hash != first.runtime_binding_hash
        or item.retry_policy_hash != first.retry_policy_hash
        or item.history_input_hash != first.history_input_hash
        for item in attempts
    ):
        raise CapsuleCompositionError("history attempt chain identity drifted")
    events_by_id = {item.history_retry_event_id: item for item in events}
    for previous, current in zip(attempts, attempts[1:], strict=False):
        event = events_by_id.get(current.previous_retry_event_id or "")
        if (
            event is None
            or not event.retry_scheduled
            or event.failed_history_attempt_id != previous.history_attempt_id
            or event.successor_ingestion_occurrence_id != current.ingestion_occurrence_id
            or event.successor_execution_run_id != current.execution_run_id
            or event.successor_history_attempt_ordinal != current.history_attempt_ordinal
        ):
            raise CapsuleCompositionError("history attempt chain omits its exact retry event")


def _require_recovery_compatibility(
    parts: tuple[_PartSnapshot, ...],
    selected_plan_ids: tuple[str, ...],
    memory_system_id: str,
) -> None:
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
