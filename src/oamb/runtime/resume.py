"""Fingerprint closure, crash classification, and provider lease bridging."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from oamb.contracts.evidence import (
    RecoveryDecisionRecord,
    RecoveryDisposition,
    RunLeaseHeartbeatRecord,
    RunLeaseRecord,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.ports import ArtifactReadRequest, ArtifactStorePort
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
    """Shares the provider-services lifecycle lock and active ownership pointers."""

    _LIFECYCLE_LOCK_NAME = "provider-lifecycle.lock"
    _ACTIVE_RUN_NAME = "active-run-lease"
    _ACTIVE_ATTEMPT_NAME = "active-provider-attempt"

    def __init__(self, provider_runtime_directory: Path) -> None:
        self._runtime_directory = provider_runtime_directory
        self._active_authority: _RunLeaseAuthority | None = None

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
            "schema_name": "oamb_provider_active_run_pointer",
            "schema_version": 1,
            "run_id": run_id,
            "provider_project": provider_project,
            "profile_id": profile_id,
            "lease_epoch": lease_epoch,
            "lease_record_sha256": lease_record_hash,
        }
        with self._lifecycle_lock():
            if (self._runtime_directory / self._ACTIVE_RUN_NAME).exists():
                raise ResumeRejectedError("an active run lease already exists")
            if (self._runtime_directory / self._ACTIVE_ATTEMPT_NAME).exists():
                raise ResumeRejectedError("an active provider attempt already exists")
            if durable_lease is not None:
                durable_lease()
            self._write_create_only(
                self._runtime_directory / self._ACTIVE_RUN_NAME,
                canonical_json_bytes(document),
            )
        authority = _RunLeaseAuthority(lease_record_hash)
        self._active_authority = authority
        return authority

    def release_run(self, authority: _RunLeaseAuthority) -> None:
        if authority is not self._active_authority:
            raise ResumeRejectedError("run release requires the current lease authority")
        with self._lifecycle_lock():
            if (self._runtime_directory / self._ACTIVE_ATTEMPT_NAME).exists():
                raise ResumeRejectedError("active provider attempt blocks run release")
            path = self._runtime_directory / self._ACTIVE_RUN_NAME
            document = self._read_pointer(self._ACTIVE_RUN_NAME, "active run")
            if document.get("lease_record_sha256") != authority.lease_record_hash:
                raise ResumeRejectedError("active run lease hash does not match")
            path.unlink()
            _fsync_directory(self._runtime_directory)
        self._active_authority = None

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
            "schema_name": "oamb_provider_active_run_pointer",
            "schema_version": 1,
            "run_id": successor.run_id,
            "provider_project": successor.provider_project_id,
            "profile_id": successor.provider_profile_id,
            "lease_epoch": successor.lease_epoch,
            "lease_record_sha256": successor.lease_record_hash,
        }
        with self._lifecycle_lock():
            if (self._runtime_directory / self._ACTIVE_ATTEMPT_NAME).exists():
                raise ResumeRejectedError("active provider attempt blocks stale-lease recovery")
            current = self._read_pointer(self._ACTIVE_RUN_NAME, "active run")
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
                    self._runtime_directory / self._ACTIVE_RUN_NAME,
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
            raise ResumeRejectedError("provider dispatch requires the current lease authority")
        with self._lifecycle_lock():
            run_document = self._read_pointer(self._ACTIVE_RUN_NAME, "active run")
            if run_document.get("lease_record_sha256") != authority.lease_record_hash:
                raise ResumeRejectedError("provider dispatch lease authority does not match")
            if (self._runtime_directory / self._ACTIVE_ATTEMPT_NAME).exists():
                raise ResumeRejectedError("an active provider attempt already exists")
            self._write_create_only(
                self._runtime_directory / self._ACTIVE_ATTEMPT_NAME,
                canonical_json_bytes(
                    {
                        "schema_name": "oamb_provider_active_attempt_pointer",
                        "schema_version": 1,
                        "attempt_id": attempt_id,
                        "intent_record_sha256": intent_record_hash,
                        "run_id": run_document["run_id"],
                        "lease_record_sha256": run_document["lease_record_sha256"],
                    }
                ),
            )

    def clear_attempt_after_receipt(self, expected_intent_record_hash: str) -> None:
        _require_sha256(expected_intent_record_hash)
        authority = self._active_authority
        if authority is None:
            raise ResumeRejectedError("provider receipt requires the current lease authority")
        with self._lifecycle_lock():
            run_document = self._read_pointer(self._ACTIVE_RUN_NAME, "active run")
            if run_document.get("lease_record_sha256") != authority.lease_record_hash:
                raise ResumeRejectedError("provider receipt lease authority does not match")
            document = self._read_pointer(self._ACTIVE_ATTEMPT_NAME, "active provider attempt")
            if (
                document.get("intent_record_sha256") != expected_intent_record_hash
                or document.get("lease_record_sha256") != authority.lease_record_hash
            ):
                raise ResumeRejectedError("active provider attempt intent hash does not match")
            path = self._runtime_directory / self._ACTIVE_ATTEMPT_NAME
            path.unlink()
            _fsync_directory(self._runtime_directory)

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
        if name == self._ACTIVE_RUN_NAME:
            expected_keys = {
                "schema_name",
                "schema_version",
                "run_id",
                "provider_project",
                "profile_id",
                "lease_epoch",
                "lease_record_sha256",
            }
            valid = (
                set(document) == expected_keys
                and document.get("schema_name") == "oamb_provider_active_run_pointer"
                and document.get("schema_version") == 1
                and all(
                    isinstance(document.get(key), str) and bool(document.get(key))
                    for key in ("run_id", "provider_project", "profile_id")
                )
                and isinstance(document.get("lease_epoch"), int)
                and not isinstance(document.get("lease_epoch"), bool)
                and int(document["lease_epoch"]) >= 1
                and _is_sha256(document.get("lease_record_sha256"))
            )
        else:
            expected_keys = {
                "schema_name",
                "schema_version",
                "attempt_id",
                "intent_record_sha256",
                "run_id",
                "lease_record_sha256",
            }
            valid = (
                set(document) == expected_keys
                and document.get("schema_name") == "oamb_provider_active_attempt_pointer"
                and document.get("schema_version") == 1
                and isinstance(document.get("run_id"), str)
                and bool(document.get("run_id"))
                and _is_sha256(document.get("attempt_id"))
                and _is_sha256(document.get("intent_record_sha256"))
                and _is_sha256(document.get("lease_record_sha256"))
            )
        if not valid:
            raise ResumeRejectedError(f"{label} pointer is malformed")
        return document

    @contextmanager
    def _lifecycle_lock(self) -> Iterator[None]:
        self._runtime_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self._runtime_directory / self._LIFECYCLE_LOCK_NAME
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
                _fsync_directory(self._runtime_directory)

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
