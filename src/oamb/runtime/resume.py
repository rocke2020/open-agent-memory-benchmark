"""Fingerprint closure, crash classification, and provider lease bridging."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import cast

from oamb.artifacts.atomic import ArtifactCollisionError, read_regular_file
from oamb.contracts.evidence import (
    MemoryConformanceEvidenceManifest,
    MemoryConformanceOccurrenceRecord,
    MemoryConformanceOccurrenceState,
    RecoveryDecisionRecord,
    RecoveryDisposition,
    RunLeaseHeartbeatRecord,
    RunLeaseRecord,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import ArtifactReadRequest, ArtifactStorePort
from oamb.contracts.specifications import MemoryConformanceSpec
from oamb.contracts.states import AttemptOutcome

from .source_records import seal_source_contract


class ResumeRejectedError(RuntimeError):
    """Raised when safe ownership or replay-free resume cannot be proven."""


@dataclass(frozen=True, slots=True)
class StaleLeaseOwner:
    host: str
    host_fingerprint: str
    process_id: int


@dataclass(frozen=True, slots=True)
class StaleLeaseAuthorization:
    reason: str


def host_identity_fingerprint(host: str) -> str:
    if not host:
        raise ResumeRejectedError("host identity must not be empty")
    return canonical_sha256(["oamb-host-identity-v1", host])


def authorize_stale_lease_recovery(
    owner: StaleLeaseOwner,
    *,
    current_host: str,
    process_is_alive: Callable[[int], bool],
    explicitly_authorized: bool,
) -> StaleLeaseAuthorization:
    if owner.host == current_host:
        if process_is_alive(owner.process_id):
            raise ResumeRejectedError("lease owner process is still alive")
        return StaleLeaseAuthorization(reason="same_host_process_absent")
    if not explicitly_authorized:
        raise ResumeRejectedError("foreign lease recovery requires explicit authorization")
    return StaleLeaseAuthorization(reason="foreign_owner_explicitly_authorized")


def _validate_stale_recovery(
    *,
    store: ArtifactStorePort,
    previous: RunLeaseRecord,
    successor: RunLeaseRecord,
    decision: RecoveryDecisionRecord,
    owner: StaleLeaseOwner,
    current_host: str,
    process_is_alive: Callable[[int], bool],
    explicitly_authorized: bool,
) -> None:
    try:
        stored_previous_bytes = store.read_verified(
            ArtifactReadRequest(
                relative_path=f"source/run-leases/{previous.lease_epoch}.json",
                expected_sha256=canonical_sha256(previous),
            )
        )
    except Exception as exc:
        raise ResumeRejectedError(
            "previous lease does not match canonical stored evidence"
        ) from exc
    try:
        stored_previous = RunLeaseRecord.model_validate_json(stored_previous_bytes)
    except ValueError as exc:
        raise ResumeRejectedError("stored previous lease is invalid") from exc
    if (
        stored_previous_bytes != canonical_json_bytes(stored_previous)
        or stored_previous != previous
    ):
        raise ResumeRejectedError("previous lease does not match canonical stored evidence")
    previous_hash = canonical_sha256(
        stored_previous.model_dump(mode="python", exclude={"lease_record_hash"})
    )
    if previous_hash != previous.lease_record_hash:
        raise ResumeRejectedError("previous lease record hash does not match its fields")
    if (
        owner.host_fingerprint != host_identity_fingerprint(owner.host)
        or owner.host_fingerprint != previous.host_fingerprint
        or owner.process_id != previous.process_id
    ):
        raise ResumeRejectedError("stale owner does not match the previous lease")
    authorization = authorize_stale_lease_recovery(
        owner,
        current_host=current_host,
        process_is_alive=process_is_alive,
        explicitly_authorized=explicitly_authorized,
    )
    if decision.authorization_id != authorization.reason:
        raise ResumeRejectedError("recovery decision authorization does not match")
    if (
        successor.run_id != previous.run_id
        or successor.provider_project_id != previous.provider_project_id
        or successor.provider_profile_id != previous.provider_profile_id
        or successor.lease_epoch != previous.lease_epoch + 1
        or successor.predecessor_lease_record_hash != previous.lease_record_hash
    ):
        raise ResumeRejectedError("successor lease does not extend the stale lease")
    if (
        successor.host_fingerprint != host_identity_fingerprint(current_host)
        or successor.process_id != os.getpid()
        or successor.acquired_at <= previous.acquired_at
    ):
        raise ResumeRejectedError("successor lease does not identify the current owner")
    successor_hash = canonical_sha256(
        successor.model_dump(mode="python", exclude={"lease_record_hash"})
    )
    if successor.lease_record_hash != successor_hash:
        raise ResumeRejectedError("successor lease record hash does not match its fields")
    decision_hash = canonical_sha256(
        decision.model_dump(mode="python", exclude={"recovery_decision_id"})
    )
    if decision.recovery_decision_id != decision_hash:
        raise ResumeRejectedError("recovery decision hash does not match its fields")
    if (
        decision.disposition != RecoveryDisposition.RESUME_SAFE
        or decision.run_id != previous.run_id
        or decision.previous_lease_record_hash != previous.lease_record_hash
        or decision.new_lease_record_hash != successor.lease_record_hash
        or decision.previous_lease_epoch != previous.lease_epoch
        or decision.new_lease_epoch != successor.lease_epoch
        or decision.decided_at <= previous.acquired_at
        or decision.decided_at > successor.acquired_at
    ):
        raise ResumeRejectedError("recovery decision does not bind the lease chain")


@dataclass(frozen=True, slots=True)
class ResumeFingerprint:
    protocol: str
    dataset: str
    case_manifest: str
    prompt_pack: str
    metric: str
    code_revision: str
    memory_runtime: str
    model_roles: str
    budget: str
    normalizer: str


def require_matching_fingerprint(
    expected: ResumeFingerprint,
    actual: ResumeFingerprint,
) -> None:
    for field_name in expected.__dataclass_fields__:
        if getattr(expected, field_name) != getattr(actual, field_name):
            raise ResumeRejectedError(f"resume fingerprint mismatch: {field_name}")


@dataclass(frozen=True, slots=True)
class AttemptEvidenceState:
    claim: bool
    intent: bool
    receipt: bool
    terminal_outcome: AttemptOutcome | None


class RecoveryAction(StrEnum):
    REACQUIRE = "reacquire"
    FINISH_LOCAL = "finish_local"
    UNKNOWN_OUTCOME = "unknown_outcome"
    NO_ACTION = "no_action"


def classify_attempt_recovery(state: AttemptEvidenceState) -> RecoveryAction:
    if state.terminal_outcome == AttemptOutcome.UNKNOWN_OUTCOME:
        if not (state.claim and state.intent) or state.receipt:
            raise ResumeRejectedError(
                "unknown-outcome terminal cannot have receipt or incomplete intent evidence"
            )
        return RecoveryAction.NO_ACTION
    if state.terminal_outcome is not None:
        if not (state.claim and state.intent and state.receipt):
            raise ResumeRejectedError("terminal attempt evidence is incomplete")
        return RecoveryAction.NO_ACTION
    if state.receipt:
        if not (state.claim and state.intent):
            raise ResumeRejectedError("attempt receipt lacks claim or intent")
        return RecoveryAction.FINISH_LOCAL
    if state.intent:
        if not state.claim:
            raise ResumeRejectedError("attempt intent lacks occurrence claim")
        return RecoveryAction.UNKNOWN_OUTCOME
    if state.claim:
        return RecoveryAction.REACQUIRE
    raise ResumeRejectedError("attempt has no recoverable evidence")


class ProviderLifecycleBridge:
    """Shares run ownership and per-attempt crash pointers with provider services."""

    _LIFECYCLE_LOCK_NAME = "provider-lifecycle.lock"
    _LEGACY_ACTIVE_RUN_NAME = "active-run-lease"
    _LEGACY_ACTIVE_ATTEMPT_NAME = "active-provider-attempt"
    _ACTIVE_OPERATION_NAME = "active-operation"
    _ACTIVE_ATTEMPTS_DIRECTORY_NAME = "active-provider-attempts"
    _CONFORMANCE_SPEC_PATH = Path("source/specs/memory-conformance-spec.json")

    def __init__(
        self,
        provider_runtime_directory: Path,
        *,
        coordination_directory: Path | None = None,
    ) -> None:
        self._runtime_directory = provider_runtime_directory
        self._coordination_directory = coordination_directory or provider_runtime_directory
        self._active_authority: _RunLeaseAuthority | _MemoryConformanceAuthority | None = None

    def acquire_run(
        self,
        *,
        run_id: str,
        provider_project: str,
        profile_id: str,
        lease_epoch: int,
        lease_record_hash: str,
        durable_lease: Callable[[], None] | None = None,
    ) -> _RunLeaseAuthority:
        if lease_epoch < 1:
            raise ResumeRejectedError("lease epoch must be positive")
        _require_sha256(lease_record_hash)
        document = {
            "schema_name": "oamb_provider_active_operation_pointer",
            "schema_version": 1,
            "kind": "benchmark_run",
            "owner": run_id,
            "run_id": run_id,
            "provider_project": provider_project,
            "profile_id": profile_id,
            "lease_epoch": lease_epoch,
            "lease_record_sha256": lease_record_hash,
        }
        with self._lifecycle_lock():
            self._require_operation_slot_available()
            if durable_lease is not None:
                durable_lease()
            self._write_create_only(
                self._runtime_directory / self._ACTIVE_OPERATION_NAME,
                canonical_json_bytes(document),
            )
        authority = _RunLeaseAuthority(lease_record_hash)
        self._active_authority = authority
        return authority

    def acquire_memory_conformance(
        self,
        *,
        evidence_root: Path,
        owner: str,
    ) -> _MemoryConformanceAuthority:
        if not owner:
            raise ResumeRejectedError("memory conformance owner must not be empty")
        evidence_root = Path(evidence_root)
        with self._lifecycle_lock():
            self._require_operation_slot_available()
            spec = _read_registered_conformance_spec(
                evidence_root / self._CONFORMANCE_SPEC_PATH,
                evidence_root=evidence_root,
            )
            document = {
                "schema_name": "oamb_provider_active_operation_pointer",
                "schema_version": 1,
                "kind": "memory_conformance",
                "owner": owner,
                "occurrence_id": spec.occurrence_id,
                "conformance_spec_sha256": spec.conformance_spec_hash,
                "provider_project": spec.provider_project_id,
                "profile_id": spec.provider_profile_id,
            }
            self._write_create_only(
                self._runtime_directory / self._ACTIVE_OPERATION_NAME,
                canonical_json_bytes(document),
            )
        authority = _MemoryConformanceAuthority(
            conformance_spec_hash=spec.conformance_spec_hash,
            occurrence_id=spec.occurrence_id,
            provider_project=spec.provider_project_id,
            profile_id=spec.provider_profile_id,
            evidence_root=evidence_root,
        )
        self._active_authority = authority
        return authority

    def release_run(
        self,
        authority: _RunLeaseAuthority,
        *,
        seal_and_verify_terminal_manifest: Callable[[], None] | None = None,
    ) -> None:
        if authority is not self._active_authority:
            raise ResumeRejectedError("run release requires the current lease authority")
        with self._lifecycle_lock():
            if self._active_attempts_present():
                raise ResumeRejectedError("active provider attempt blocks run release")
            path = self._runtime_directory / self._ACTIVE_OPERATION_NAME
            document = self._read_pointer(self._ACTIVE_OPERATION_NAME, "active operation")
            if document.get("kind") != "benchmark_run":
                raise ResumeRejectedError("active operation is not a benchmark run")
            if document.get("lease_record_sha256") != authority.lease_record_hash:
                raise ResumeRejectedError("active run lease hash does not match")
            if seal_and_verify_terminal_manifest is not None:
                seal_and_verify_terminal_manifest()
            path.unlink()
            _fsync_directory(self._runtime_directory)
        self._active_authority = None

    def release_supervised_aborted_run(
        self,
        authority: _RunLeaseAuthority,
        *,
        verify_terminal_abort: Callable[[tuple[dict[str, object], ...]], None],
    ) -> None:
        """Release only this process's verified terminal abort and its attempts."""
        if authority is not self._active_authority:
            raise ResumeRejectedError("run release requires the current lease authority")
        with self._lifecycle_lock():
            operation = self._read_pointer(self._ACTIVE_OPERATION_NAME, "active operation")
            if operation.get("kind") != "benchmark_run":
                raise ResumeRejectedError("active operation is not a benchmark run")
            if operation.get("lease_record_sha256") != authority.lease_record_hash:
                raise ResumeRejectedError("active run lease hash does not match")
            attempts = self._validated_active_run_attempts(operation, authority)
            verify_terminal_abort(attempts)
            attempts_directory = self._runtime_directory / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
            for attempt in attempts:
                attempt_id = cast(str, attempt["attempt_id"])
                (attempts_directory / f"{attempt_id}.json").unlink()
            if attempts:
                _fsync_directory(attempts_directory)
            (self._runtime_directory / self._ACTIVE_OPERATION_NAME).unlink()
            _fsync_directory(self._runtime_directory)
        self._active_authority = None

    def release_memory_conformance(
        self,
        authority: _MemoryConformanceAuthority,
    ) -> None:
        if authority is not self._active_authority:
            raise ResumeRejectedError(
                "memory conformance release requires the current operation authority"
            )
        with self._lifecycle_lock():
            if self._active_attempts_present():
                raise ResumeRejectedError(
                    "active provider attempt blocks memory conformance release"
                )
            pointer = self._read_pointer(self._ACTIVE_OPERATION_NAME, "active operation")
            _operation_authority_fields(pointer, authority)
            if (
                pointer.get("provider_project") != authority.provider_project
                or pointer.get("profile_id") != authority.profile_id
            ):
                raise ResumeRejectedError("memory conformance pointer identity does not match")
            _validate_terminal_conformance_evidence(authority)
            path = self._runtime_directory / self._ACTIVE_OPERATION_NAME
            path.unlink()
            _fsync_directory(self._runtime_directory)
        self._active_authority = None

    def _require_operation_slot_available(self) -> None:
        if (self._runtime_directory / self._LEGACY_ACTIVE_RUN_NAME).exists():
            raise ResumeRejectedError("a legacy active run lease already exists")
        if (self._runtime_directory / self._ACTIVE_OPERATION_NAME).exists():
            raise ResumeRejectedError("an active provider operation already exists")
        if self._active_attempts_present():
            raise ResumeRejectedError("an active provider attempt already exists")

    def recover_stale_run(
        self,
        *,
        store: ArtifactStorePort,
        previous: RunLeaseRecord,
        successor: RunLeaseRecord,
        decision: RecoveryDecisionRecord,
        owner: StaleLeaseOwner,
        current_host: str,
        process_is_alive: Callable[[int], bool],
        explicitly_authorized: bool,
    ) -> _RunLeaseAuthority:
        if self._active_authority is not None:
            raise ResumeRejectedError("current process already owns a run lease")
        successor_document = {
            "schema_name": "oamb_provider_active_operation_pointer",
            "schema_version": 1,
            "kind": "benchmark_run",
            "owner": successor.run_id,
            "run_id": successor.run_id,
            "provider_project": successor.provider_project_id,
            "profile_id": successor.provider_profile_id,
            "lease_epoch": successor.lease_epoch,
            "lease_record_sha256": successor.lease_record_hash,
        }
        with self._lifecycle_lock():
            if self._active_attempts_present():
                raise ResumeRejectedError("active provider attempt blocks stale-lease recovery")
            current = self._read_pointer(self._ACTIVE_OPERATION_NAME, "active operation")
            if current.get("kind") != "benchmark_run":
                raise ResumeRejectedError("stale recovery requires a benchmark operation")
            previous_document = {
                "run_id": previous.run_id,
                "provider_project": previous.provider_project_id,
                "profile_id": previous.provider_profile_id,
                "lease_epoch": previous.lease_epoch,
                "lease_record_sha256": previous.lease_record_hash,
            }
            successor_identity = {
                "run_id": successor.run_id,
                "provider_project": successor.provider_project_id,
                "profile_id": successor.provider_profile_id,
                "lease_epoch": successor.lease_epoch,
                "lease_record_sha256": successor.lease_record_hash,
            }
            points_to_previous = all(
                current.get(key) == value for key, value in previous_document.items()
            )
            points_to_successor = all(
                current.get(key) == value for key, value in successor_identity.items()
            )
            if not (points_to_previous or points_to_successor):
                raise ResumeRejectedError("stale lease pointer does not match recovery source")
            _validate_stale_recovery(
                store=store,
                previous=previous,
                successor=successor,
                decision=decision,
                owner=owner,
                current_host=current_host,
                process_is_alive=process_is_alive,
                explicitly_authorized=explicitly_authorized,
            )
            seal_source_contract(
                store,
                relative_path=f"source/recovery-decisions/{decision.recovery_decision_id}.json",
                record_id=decision.recovery_decision_id,
                record=decision,
            )
            seal_source_contract(
                store,
                relative_path=f"source/run-leases/{successor.lease_epoch}.json",
                record_id=successor.lease_record_hash,
                record=successor,
            )
            if points_to_previous:
                self._write_replace(
                    self._runtime_directory / self._ACTIVE_OPERATION_NAME,
                    canonical_json_bytes(successor_document),
                )
        authority = _RunLeaseAuthority(successor.lease_record_hash)
        self._active_authority = authority
        return authority

    def mark_attempt_dispatched(self, *, attempt_id: str, intent_record_hash: str) -> None:
        _require_sha256(attempt_id)
        _require_sha256(intent_record_hash)
        authority = self._active_authority
        if authority is None:
            raise ResumeRejectedError("provider dispatch requires the current operation authority")
        with self._lifecycle_lock():
            operation = self._read_pointer(self._ACTIVE_OPERATION_NAME, "active operation")
            operation_kind, operation_id, operation_hash = _operation_authority_fields(
                operation, authority
            )
            attempts_directory = self._ensure_active_attempts_directory()
            self._write_create_only(
                attempts_directory / f"{attempt_id}.json",
                canonical_json_bytes(
                    {
                        "schema_name": "oamb_provider_active_attempt_pointer",
                        "schema_version": 1,
                        "attempt_id": attempt_id,
                        "intent_record_sha256": intent_record_hash,
                        "operation_kind": operation_kind,
                        "operation_id": operation_id,
                        "operation_record_sha256": operation_hash,
                    }
                ),
            )

    def clear_attempt_after_receipt(
        self,
        *,
        attempt_id: str,
        expected_intent_record_hash: str,
    ) -> None:
        _require_sha256(attempt_id)
        _require_sha256(expected_intent_record_hash)
        authority = self._active_authority
        if authority is None:
            raise ResumeRejectedError("provider receipt requires the current operation authority")
        with self._lifecycle_lock():
            operation = self._read_pointer(self._ACTIVE_OPERATION_NAME, "active operation")
            _, _, operation_hash = _operation_authority_fields(operation, authority)
            path = (
                self._runtime_directory
                / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
                / f"{attempt_id}.json"
            )
            document = self._read_attempt_pointer(path)
            if (
                document.get("attempt_id") != attempt_id
                or document.get("intent_record_sha256") != expected_intent_record_hash
                or document.get("operation_record_sha256") != operation_hash
            ):
                raise ResumeRejectedError("active provider attempt intent hash does not match")
            path.unlink()
            _fsync_directory(path.parent)

    def _active_attempts_present(self) -> bool:
        if (self._runtime_directory / self._LEGACY_ACTIVE_ATTEMPT_NAME).exists():
            return True
        directory = self._runtime_directory / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
        if not directory.exists():
            return False
        if directory.is_symlink() or not directory.is_dir():
            raise ResumeRejectedError("active provider attempts path is unsafe")
        try:
            return next(directory.iterdir(), None) is not None
        except OSError as exc:
            raise ResumeRejectedError("active provider attempts cannot be inspected") from exc

    def _validated_active_run_attempts(
        self,
        operation: dict[str, object],
        authority: _RunLeaseAuthority,
    ) -> tuple[dict[str, object], ...]:
        legacy = self._runtime_directory / self._LEGACY_ACTIVE_ATTEMPT_NAME
        if legacy.exists() or legacy.is_symlink():
            raise ResumeRejectedError("legacy active provider attempt blocks run release")
        directory = self._runtime_directory / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
        if not directory.exists():
            return ()
        if directory.is_symlink() or not directory.is_dir():
            raise ResumeRejectedError("active provider attempts path is unsafe")
        try:
            paths = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
        except OSError as exc:
            raise ResumeRejectedError("active provider attempts cannot be inspected") from exc
        attempts: list[dict[str, object]] = []
        for path in paths:
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise ResumeRejectedError(
                    "active provider attempt pointer cannot be inspected"
                ) from exc
            if not stat.S_ISREG(mode):
                raise ResumeRejectedError("active provider attempt pointer is unsafe")
            document = self._read_attempt_pointer(path)
            attempt_id = cast(str, document["attempt_id"])
            if path.name != f"{attempt_id}.json":
                raise ResumeRejectedError("active provider attempt filename does not match")
            if (
                document.get("operation_kind") != "benchmark_run"
                or document.get("operation_id") != operation.get("run_id")
                or document.get("operation_record_sha256") != authority.lease_record_hash
            ):
                raise ResumeRejectedError("active provider attempt does not bind the run")
            attempts.append(document)
        return tuple(attempts)

    def _ensure_active_attempts_directory(self) -> Path:
        directory = self._runtime_directory / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
        try:
            directory.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise ResumeRejectedError("active provider attempts directory is unavailable") from exc
        if directory.is_symlink() or not directory.is_dir():
            raise ResumeRejectedError("active provider attempts path is unsafe")
        return directory

    def _read_attempt_pointer(self, path: Path) -> dict[str, object]:
        if path.is_symlink():
            raise ResumeRejectedError("active provider attempt pointer is symbolic")
        try:
            document = json.loads(path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ResumeRejectedError(
                "active provider attempt pointer is absent or malformed"
            ) from exc
        expected_keys = {
            "schema_name",
            "schema_version",
            "attempt_id",
            "intent_record_sha256",
            "operation_kind",
            "operation_id",
            "operation_record_sha256",
        }
        valid = (
            isinstance(document, dict)
            and set(document) == expected_keys
            and document.get("schema_name") == "oamb_provider_active_attempt_pointer"
            and document.get("schema_version") == 1
            and document.get("operation_kind") in {"benchmark_run", "memory_conformance"}
            and isinstance(document.get("operation_id"), str)
            and bool(document.get("operation_id"))
            and _is_sha256(document.get("attempt_id"))
            and _is_sha256(document.get("intent_record_sha256"))
            and _is_sha256(document.get("operation_record_sha256"))
        )
        if not valid:
            raise ResumeRejectedError("active provider attempt pointer is malformed")
        return cast(dict[str, object], document)

    def _read_pointer(self, name: str, label: str) -> dict[str, object]:
        path = self._runtime_directory / name
        if path.is_symlink():
            raise ResumeRejectedError(f"{label} pointer is a symbolic link")
        try:
            document = json.loads(path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ResumeRejectedError(f"{label} pointer is absent or malformed") from exc
        if not isinstance(document, dict):
            raise ResumeRejectedError(f"{label} pointer is malformed")
        if name == self._ACTIVE_OPERATION_NAME:
            common_valid = (
                document.get("schema_name") == "oamb_provider_active_operation_pointer"
                and document.get("schema_version") == 1
                and all(
                    isinstance(document.get(key), str) and bool(document.get(key))
                    for key in ("owner", "provider_project", "profile_id")
                )
            )
            if document.get("kind") == "benchmark_run":
                expected_keys = {
                    "schema_name",
                    "schema_version",
                    "kind",
                    "owner",
                    "run_id",
                    "provider_project",
                    "profile_id",
                    "lease_epoch",
                    "lease_record_sha256",
                }
                valid = (
                    common_valid
                    and set(document) == expected_keys
                    and isinstance(document.get("run_id"), str)
                    and bool(document.get("run_id"))
                    and isinstance(document.get("lease_epoch"), int)
                    and not isinstance(document.get("lease_epoch"), bool)
                    and int(document["lease_epoch"]) >= 1
                    and _is_sha256(document.get("lease_record_sha256"))
                )
            elif document.get("kind") == "memory_conformance":
                expected_keys = {
                    "schema_name",
                    "schema_version",
                    "kind",
                    "owner",
                    "occurrence_id",
                    "conformance_spec_sha256",
                    "provider_project",
                    "profile_id",
                }
                valid = (
                    common_valid
                    and set(document) == expected_keys
                    and isinstance(document.get("occurrence_id"), str)
                    and bool(document.get("occurrence_id"))
                    and _is_sha256(document.get("conformance_spec_sha256"))
                )
            else:
                valid = False
        else:
            valid = False
        if not valid:
            raise ResumeRejectedError(f"{label} pointer is malformed")
        return document

    @contextmanager
    def _lifecycle_lock(self) -> Iterator[None]:
        self._require_safe_lifecycle_paths()
        if self._coordination_directory == self._runtime_directory:
            with self._directory_lifecycle_lock(self._runtime_directory):
                yield
            return
        self._require_coordination_gate_available()
        with self._directory_lifecycle_lock(self._runtime_directory):
            self._require_coordination_gate_available()
            yield

    def _require_safe_lifecycle_paths(self) -> None:
        coordination = self._coordination_directory
        domain = self._runtime_directory
        if coordination.is_symlink():
            raise ResumeRejectedError("provider lifecycle coordination path is symbolic and unsafe")
        if coordination == domain:
            return
        try:
            relative_domain = domain.relative_to(coordination)
        except ValueError as exc:
            raise ResumeRejectedError(
                "provider lifecycle domain is outside its coordination directory"
            ) from exc
        if (
            len(relative_domain.parts) != 2
            or relative_domain.parts[0] != "lifecycle-domains"
            or not relative_domain.parts[1]
        ):
            raise ResumeRejectedError("provider lifecycle domain layout is invalid")
        domains_parent = coordination / relative_domain.parts[0]
        if domains_parent.is_symlink() or domain.is_symlink():
            raise ResumeRejectedError("provider lifecycle domain path is symbolic and unsafe")

    def _require_coordination_gate_available(self) -> None:
        lock_path = self._coordination_directory / self._LIFECYCLE_LOCK_NAME
        if lock_path.exists() or lock_path.is_symlink():
            raise ResumeRejectedError("provider lifecycle operation is active")

    @contextmanager
    def _directory_lifecycle_lock(self, directory: Path) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / self._LIFECYCLE_LOCK_NAME
        try:
            lock_path.mkdir()
        except FileExistsError as exc:
            raise ResumeRejectedError("provider lifecycle operation is active") from exc
        try:
            yield
        finally:
            try:
                lock_path.rmdir()
            finally:
                _fsync_directory(directory)

    @staticmethod
    def _write_create_only(path: Path, content: bytes) -> None:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while creating provider lifecycle pointer")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)

    @staticmethod
    def _write_replace(path: Path, content: bytes) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.next-", dir=path.parent)
        temporary = Path(temporary_name)
        try:
            view = memoryview(content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short write while replacing provider lifecycle pointer")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()
                _fsync_directory(path.parent)


@dataclass(frozen=True, slots=True)
class _RunLeaseAuthority:
    lease_record_hash: str


@dataclass(frozen=True, slots=True)
class _MemoryConformanceAuthority:
    conformance_spec_hash: str
    occurrence_id: str
    provider_project: str
    profile_id: str
    evidence_root: Path


def _read_registered_conformance_spec(
    path: Path,
    *,
    evidence_root: Path,
) -> MemoryConformanceSpec:
    if evidence_root.is_symlink() or any(
        component.is_symlink()
        for component in (
            evidence_root / "source",
            evidence_root / "source/specs",
            path,
        )
    ):
        raise ResumeRejectedError("registered specification path is symbolic")
    try:
        content = read_regular_file(path)
    except (FileNotFoundError, ArtifactCollisionError, OSError) as exc:
        raise ResumeRejectedError("registered specification is absent or unsafe") from exc
    try:
        spec = MemoryConformanceSpec.model_validate_json(content)
    except ValueError as exc:
        raise ResumeRejectedError("registered specification is invalid") from exc
    if content != canonical_json_bytes(spec):
        raise ResumeRejectedError("registered specification bytes are not canonical")
    return spec


def _validate_terminal_conformance_evidence(
    authority: _MemoryConformanceAuthority,
) -> None:
    root = authority.evidence_root
    manifest_path = root / "memory-conformance-manifest.json"
    manifest_bytes = _read_conformance_evidence_file(
        root,
        manifest_path,
        label="terminal manifest",
    )
    try:
        manifest = MemoryConformanceEvidenceManifest.model_validate_json(manifest_bytes)
    except ValueError as exc:
        raise ResumeRejectedError("terminal manifest is invalid") from exc
    if manifest_bytes != canonical_json_bytes(manifest):
        raise ResumeRejectedError("terminal manifest bytes are not canonical")
    if (
        manifest.conformance_spec_hash != authority.conformance_spec_hash
        or manifest.occurrence_id != authority.occurrence_id
    ):
        raise ResumeRejectedError("terminal manifest does not match operation authority")
    if manifest.terminal_state == MemoryConformanceOccurrenceState.INTERRUPTED_UNKNOWN_OUTCOME:
        raise ResumeRejectedError("memory conformance unknown outcome retains pointers")

    source_entries = tuple(manifest.source_entries)
    occurrence_entries = tuple(
        entry
        for entry in source_entries
        if entry.record_kind == "memory_conformance_occurrence_record"
    )
    spec_entries = tuple(
        entry for entry in source_entries if entry.record_kind == "memory_conformance_spec"
    )
    if len(occurrence_entries) != 1 or len(spec_entries) != 1:
        raise ResumeRejectedError(
            "terminal manifest requires one occurrence and one registered specification"
        )
    spec_entry = spec_entries[0]
    if spec_entry.relative_path != ProviderLifecycleBridge._CONFORMANCE_SPEC_PATH.as_posix():
        raise ResumeRejectedError("terminal manifest names a different registered specification")

    for entry in (*source_entries, *manifest.raw_entries):
        content = _read_conformance_evidence_file(
            root,
            root / entry.relative_path,
            label=f"terminal manifest entry {entry.relative_path}",
        )
        if hashlib.sha256(content).hexdigest() != entry.sha256:
            raise ResumeRejectedError("terminal manifest entry hash does not match")

    spec = _read_registered_conformance_spec(
        root / ProviderLifecycleBridge._CONFORMANCE_SPEC_PATH,
        evidence_root=root,
    )
    if (
        spec.conformance_spec_id != manifest.conformance_spec_id
        or spec.conformance_spec_hash != authority.conformance_spec_hash
        or spec.occurrence_id != authority.occurrence_id
        or spec.provider_project_id != authority.provider_project
        or spec.provider_profile_id != authority.profile_id
    ):
        raise ResumeRejectedError("terminal manifest specification closure does not match")

    occurrence_entry = occurrence_entries[0]
    occurrence_bytes = _read_conformance_evidence_file(
        root,
        root / occurrence_entry.relative_path,
        label="terminal occurrence",
    )
    try:
        occurrence = MemoryConformanceOccurrenceRecord.model_validate_json(occurrence_bytes)
    except ValueError as exc:
        raise ResumeRejectedError("terminal occurrence is invalid") from exc
    if occurrence_bytes != canonical_json_bytes(occurrence):
        raise ResumeRejectedError("terminal occurrence bytes are not canonical")
    occurrence_hash = hashlib.sha256(occurrence_bytes).hexdigest()
    if (
        occurrence_hash != manifest.terminal_occurrence_hash
        or occurrence_entry.sha256 != occurrence_hash
        or occurrence.occurrence_id != spec.occurrence_id
        or occurrence.provider != spec.provider
        or occurrence.provider_project_id != spec.provider_project_id
        or occurrence.provider_profile_id != spec.provider_profile_id
        or occurrence.runtime_binding_hash != spec.runtime_binding_hash
        or occurrence.budget_id != spec.budget_id
        or occurrence.dispatch_route_ids != tuple(route.route_id for route in spec.dispatch_routes)
        or occurrence.state != manifest.terminal_state
    ):
        raise ResumeRejectedError("terminal occurrence closure does not match its specification")


def _read_conformance_evidence_file(root: Path, path: Path, *, label: str) -> bytes:
    if root.is_symlink():
        raise ResumeRejectedError(f"{label} root is symbolic")
    try:
        relative_path = path.relative_to(root)
    except ValueError as exc:
        raise ResumeRejectedError(f"{label} escapes the evidence root") from exc
    current = root
    for component in relative_path.parts:
        current = current / component
        if current.is_symlink():
            raise ResumeRejectedError(f"{label} path is symbolic")
    try:
        return read_regular_file(path)
    except (FileNotFoundError, ArtifactCollisionError, OSError) as exc:
        raise ResumeRejectedError(f"{label} is absent or unsafe") from exc


def _operation_authority_fields(
    document: dict[str, object],
    authority: _RunLeaseAuthority | _MemoryConformanceAuthority,
) -> tuple[str, str, str]:
    if isinstance(authority, _RunLeaseAuthority):
        if document.get("kind") != "benchmark_run":
            raise ResumeRejectedError("operation authority kind does not match benchmark run")
        record_hash = document.get("lease_record_sha256")
        operation_id = document.get("run_id")
        expected_hash = authority.lease_record_hash
    else:
        if document.get("kind") != "memory_conformance":
            raise ResumeRejectedError("operation authority kind does not match conformance")
        record_hash = document.get("conformance_spec_sha256")
        operation_id = document.get("occurrence_id")
        expected_hash = authority.conformance_spec_hash
    if record_hash != expected_hash or not isinstance(operation_id, str):
        raise ResumeRejectedError("provider operation authority does not match")
    return str(document["kind"]), operation_id, expected_hash


def _require_sha256(value: str) -> None:
    if not _is_sha256(value):
        raise ResumeRejectedError("lease record hash must be lowercase SHA-256")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class LeaseJournal:
    """Seals immutable lease/heartbeat evidence before operational ownership."""

    def __init__(
        self, store: ArtifactStorePort, provider_lifecycle: ProviderLifecycleBridge
    ) -> None:
        self._store = store
        self._provider_lifecycle = provider_lifecycle
        self._current_lease: RunLeaseRecord | None = None
        self._last_heartbeat: RunLeaseHeartbeatRecord | None = None
        self._lease_authority: _RunLeaseAuthority | None = None

    @property
    def current_lease_hash(self) -> str | None:
        return None if self._current_lease is None else self._current_lease.lease_record_hash

    def acquire(self, record: RunLeaseRecord) -> None:
        if self._current_lease is not None:
            raise ResumeRejectedError("lease journal already owns a run")
        if record.lease_epoch != 1 or record.predecessor_lease_record_hash is not None:
            raise ResumeRejectedError(
                "ordinary acquisition requires the first lease epoch; recovery creates successors"
            )
        expected_hash = canonical_sha256(
            record.model_dump(mode="python", exclude={"lease_record_hash"})
        )
        if record.lease_record_hash != expected_hash:
            raise ResumeRejectedError("lease record hash does not match its fields")

        def seal_lease() -> None:
            receipt = seal_source_contract(
                self._store,
                relative_path=f"source/run-leases/{record.lease_epoch}.json",
                record_id=record.lease_record_hash,
                record=record,
            )
            if not receipt.created:
                raise ResumeRejectedError("lease acquisition record is already sealed")

        authority = self._provider_lifecycle.acquire_run(
            run_id=record.run_id,
            provider_project=record.provider_project_id,
            profile_id=record.provider_profile_id,
            lease_epoch=record.lease_epoch,
            lease_record_hash=record.lease_record_hash,
            durable_lease=seal_lease,
        )
        self._current_lease = record
        self._lease_authority = authority

    def recover_stale(
        self,
        *,
        previous: RunLeaseRecord,
        successor: RunLeaseRecord,
        decision: RecoveryDecisionRecord,
        owner: StaleLeaseOwner,
        current_host: str,
        process_is_alive: Callable[[int], bool],
        explicitly_authorized: bool,
    ) -> None:
        if self._current_lease is not None:
            raise ResumeRejectedError("lease journal already owns a run")
        authority = self._provider_lifecycle.recover_stale_run(
            store=self._store,
            previous=previous,
            successor=successor,
            decision=decision,
            owner=owner,
            current_host=current_host,
            process_is_alive=process_is_alive,
            explicitly_authorized=explicitly_authorized,
        )
        self._current_lease = successor
        self._lease_authority = authority

    def append_heartbeat(self, record: RunLeaseHeartbeatRecord) -> None:
        lease = self._current_lease
        if lease is None or record.lease_record_hash != lease.lease_record_hash:
            raise ResumeRejectedError("heartbeat belongs to a different lease")
        expected_sequence = (
            1 if self._last_heartbeat is None else self._last_heartbeat.heartbeat_sequence + 1
        )
        expected_predecessor = (
            None if self._last_heartbeat is None else self._last_heartbeat.heartbeat_record_hash
        )
        if record.heartbeat_sequence != expected_sequence:
            raise ResumeRejectedError("heartbeat sequence is not append-only")
        if record.predecessor_heartbeat_hash != expected_predecessor:
            raise ResumeRejectedError("heartbeat predecessor does not match")
        expected_hash = canonical_sha256(
            record.model_dump(mode="python", exclude={"heartbeat_record_hash"})
        )
        if record.heartbeat_record_hash != expected_hash:
            raise ResumeRejectedError("heartbeat record hash does not match its fields")
        seal_source_contract(
            self._store,
            relative_path=(
                f"source/run-lease-heartbeats/{lease.lease_epoch}-{record.heartbeat_sequence}.json"
            ),
            record_id=record.heartbeat_record_hash,
            record=record,
        )
        self._last_heartbeat = record

    def release(self) -> None:
        lease = self._current_lease
        authority = self._lease_authority
        if lease is None or authority is None:
            raise ResumeRejectedError("lease journal does not own a run")
        self._provider_lifecycle.release_run(authority)
        self._current_lease = None
        self._last_heartbeat = None
        self._lease_authority = None
