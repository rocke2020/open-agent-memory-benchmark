"""Active provider-operation ownership and per-attempt lifecycle pointers."""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from oamb.contracts.ids import canonical_json_bytes


class ProviderLifecycleError(RuntimeError):
    """Raised when provider lifecycle ownership cannot be changed safely."""


class ProviderLifecyclePort(Protocol):
    """The attempt-ledger boundary for provider dispatch ownership."""

    def mark_attempt_dispatched(self, *, attempt_id: str, intent_record_hash: str) -> None: ...

    def clear_attempt_after_receipt(
        self,
        *,
        attempt_id: str,
        expected_intent_record_hash: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _RunLeaseAuthority:
    lease_record_hash: str


class ProviderLifecycleBridge:
    """Shares active run ownership and attempt pointers with provider services."""

    _LIFECYCLE_LOCK_NAME = "provider-lifecycle.lock"
    _LEGACY_ACTIVE_RUN_NAME = "active-run-lease"
    _LEGACY_ACTIVE_ATTEMPT_NAME = "active-provider-attempt"
    _ACTIVE_OPERATION_NAME = "active-operation"
    _ACTIVE_ATTEMPTS_DIRECTORY_NAME = "active-provider-attempts"

    def __init__(
        self,
        provider_runtime_directory: Path,
        *,
        coordination_directory: Path | None = None,
    ) -> None:
        self._runtime_directory = provider_runtime_directory
        self._coordination_directory = coordination_directory or provider_runtime_directory
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
            raise ProviderLifecycleError("lease epoch must be positive")
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

    def release_run(
        self,
        authority: _RunLeaseAuthority,
        *,
        seal_and_verify_terminal_manifest: Callable[[], None] | None = None,
    ) -> None:
        if authority is not self._active_authority:
            raise ProviderLifecycleError("run release requires the current lease authority")
        with self._lifecycle_lock():
            if self._active_attempts_present():
                raise ProviderLifecycleError("active provider attempt blocks run release")
            path = self._runtime_directory / self._ACTIVE_OPERATION_NAME
            document = self._read_operation_pointer()
            if document.get("lease_record_sha256") != authority.lease_record_hash:
                raise ProviderLifecycleError("active run lease hash does not match")
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
        """Release this process's verified terminal abort and its attempt pointers."""

        if authority is not self._active_authority:
            raise ProviderLifecycleError("run release requires the current lease authority")
        with self._lifecycle_lock():
            operation = self._read_operation_pointer()
            if operation.get("lease_record_sha256") != authority.lease_record_hash:
                raise ProviderLifecycleError("active run lease hash does not match")
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

    def mark_attempt_dispatched(self, *, attempt_id: str, intent_record_hash: str) -> None:
        _require_sha256(attempt_id)
        _require_sha256(intent_record_hash)
        authority = self._active_authority
        if authority is None:
            raise ProviderLifecycleError(
                "provider dispatch requires the current operation authority"
            )
        with self._lifecycle_lock():
            operation = self._read_operation_pointer()
            self._require_run_authority(operation, authority)
            attempts_directory = self._ensure_active_attempts_directory()
            self._write_create_only(
                attempts_directory / f"{attempt_id}.json",
                canonical_json_bytes(
                    {
                        "schema_name": "oamb_provider_active_attempt_pointer",
                        "schema_version": 1,
                        "attempt_id": attempt_id,
                        "intent_record_sha256": intent_record_hash,
                        "operation_kind": "benchmark_run",
                        "operation_id": cast(str, operation["run_id"]),
                        "operation_record_sha256": authority.lease_record_hash,
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
            raise ProviderLifecycleError(
                "provider receipt requires the current operation authority"
            )
        with self._lifecycle_lock():
            operation = self._read_operation_pointer()
            self._require_run_authority(operation, authority)
            path = (
                self._runtime_directory
                / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
                / f"{attempt_id}.json"
            )
            document = self._read_attempt_pointer(path)
            if (
                document.get("attempt_id") != attempt_id
                or document.get("intent_record_sha256") != expected_intent_record_hash
                or document.get("operation_record_sha256") != authority.lease_record_hash
            ):
                raise ProviderLifecycleError("active provider attempt intent hash does not match")
            path.unlink()
            _fsync_directory(path.parent)

    def _require_operation_slot_available(self) -> None:
        if (self._runtime_directory / self._LEGACY_ACTIVE_RUN_NAME).exists():
            raise ProviderLifecycleError("a legacy active run lease already exists")
        if (self._runtime_directory / self._ACTIVE_OPERATION_NAME).exists():
            raise ProviderLifecycleError("an active provider operation already exists")
        if self._active_attempts_present():
            raise ProviderLifecycleError("an active provider attempt already exists")

    def _active_attempts_present(self) -> bool:
        if (self._runtime_directory / self._LEGACY_ACTIVE_ATTEMPT_NAME).exists():
            return True
        directory = self._runtime_directory / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
        if not directory.exists():
            return False
        if directory.is_symlink() or not directory.is_dir():
            raise ProviderLifecycleError("active provider attempts path is unsafe")
        try:
            return next(directory.iterdir(), None) is not None
        except OSError as exc:
            raise ProviderLifecycleError("active provider attempts cannot be inspected") from exc

    def _validated_active_run_attempts(
        self,
        operation: dict[str, object],
        authority: _RunLeaseAuthority,
    ) -> tuple[dict[str, object], ...]:
        legacy = self._runtime_directory / self._LEGACY_ACTIVE_ATTEMPT_NAME
        if legacy.exists() or legacy.is_symlink():
            raise ProviderLifecycleError("legacy active provider attempt blocks run release")
        directory = self._runtime_directory / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
        if not directory.exists():
            return ()
        if directory.is_symlink() or not directory.is_dir():
            raise ProviderLifecycleError("active provider attempts path is unsafe")
        try:
            paths = tuple(sorted(directory.iterdir(), key=lambda path: path.name))
        except OSError as exc:
            raise ProviderLifecycleError("active provider attempts cannot be inspected") from exc
        attempts: list[dict[str, object]] = []
        for path in paths:
            try:
                mode = path.lstat().st_mode
            except OSError as exc:
                raise ProviderLifecycleError(
                    "active provider attempt pointer cannot be inspected"
                ) from exc
            if not stat.S_ISREG(mode):
                raise ProviderLifecycleError("active provider attempt pointer is unsafe")
            document = self._read_attempt_pointer(path)
            attempt_id = cast(str, document["attempt_id"])
            if path.name != f"{attempt_id}.json":
                raise ProviderLifecycleError("active provider attempt filename does not match")
            if (
                document.get("operation_kind") != "benchmark_run"
                or document.get("operation_id") != operation.get("run_id")
                or document.get("operation_record_sha256") != authority.lease_record_hash
            ):
                raise ProviderLifecycleError("active provider attempt does not bind the run")
            attempts.append(document)
        return tuple(attempts)

    def _ensure_active_attempts_directory(self) -> Path:
        directory = self._runtime_directory / self._ACTIVE_ATTEMPTS_DIRECTORY_NAME
        try:
            directory.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise ProviderLifecycleError(
                "active provider attempts directory is unavailable"
            ) from exc
        if directory.is_symlink() or not directory.is_dir():
            raise ProviderLifecycleError("active provider attempts path is unsafe")
        return directory

    def _read_attempt_pointer(self, path: Path) -> dict[str, object]:
        if path.is_symlink():
            raise ProviderLifecycleError("active provider attempt pointer is symbolic")
        try:
            document = json.loads(path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ProviderLifecycleError(
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
            and document.get("operation_kind") == "benchmark_run"
            and isinstance(document.get("operation_id"), str)
            and bool(document.get("operation_id"))
            and _is_sha256(document.get("attempt_id"))
            and _is_sha256(document.get("intent_record_sha256"))
            and _is_sha256(document.get("operation_record_sha256"))
        )
        if not valid:
            raise ProviderLifecycleError("active provider attempt pointer is malformed")
        return cast(dict[str, object], document)

    def _read_operation_pointer(self) -> dict[str, object]:
        path = self._runtime_directory / self._ACTIVE_OPERATION_NAME
        if path.is_symlink():
            raise ProviderLifecycleError("active operation pointer is a symbolic link")
        try:
            document = json.loads(path.read_bytes())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise ProviderLifecycleError("active operation pointer is absent or malformed") from exc
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
            isinstance(document, dict)
            and set(document) == expected_keys
            and document.get("schema_name") == "oamb_provider_active_operation_pointer"
            and document.get("schema_version") == 1
            and document.get("kind") == "benchmark_run"
            and all(
                isinstance(document.get(key), str) and bool(document.get(key))
                for key in ("owner", "run_id", "provider_project", "profile_id")
            )
            and isinstance(document.get("lease_epoch"), int)
            and not isinstance(document.get("lease_epoch"), bool)
            and int(document["lease_epoch"]) >= 1
            and _is_sha256(document.get("lease_record_sha256"))
        )
        if not valid:
            raise ProviderLifecycleError("active operation pointer is malformed")
        return cast(dict[str, object], document)

    @staticmethod
    def _require_run_authority(document: dict[str, object], authority: _RunLeaseAuthority) -> None:
        if document.get("lease_record_sha256") != authority.lease_record_hash:
            raise ProviderLifecycleError("provider operation authority does not match")

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
            raise ProviderLifecycleError(
                "provider lifecycle coordination path is symbolic and unsafe"
            )
        if coordination == domain:
            return
        try:
            relative_domain = domain.relative_to(coordination)
        except ValueError as exc:
            raise ProviderLifecycleError(
                "provider lifecycle domain is outside its coordination directory"
            ) from exc
        if (
            len(relative_domain.parts) != 2
            or relative_domain.parts[0] != "lifecycle-domains"
            or not relative_domain.parts[1]
        ):
            raise ProviderLifecycleError("provider lifecycle domain layout is invalid")
        domains_parent = coordination / relative_domain.parts[0]
        if domains_parent.is_symlink() or domain.is_symlink():
            raise ProviderLifecycleError("provider lifecycle domain path is symbolic and unsafe")

    def _require_coordination_gate_available(self) -> None:
        lock_path = self._coordination_directory / self._LIFECYCLE_LOCK_NAME
        if lock_path.exists() or lock_path.is_symlink():
            raise ProviderLifecycleError("provider lifecycle operation is active")

    @contextmanager
    def _directory_lifecycle_lock(self, directory: Path) -> Iterator[None]:
        directory.mkdir(parents=True, exist_ok=True)
        lock_path = directory / self._LIFECYCLE_LOCK_NAME
        try:
            lock_path.mkdir()
        except FileExistsError as exc:
            raise ProviderLifecycleError("provider lifecycle operation is active") from exc
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


def _require_sha256(value: str) -> None:
    if not _is_sha256(value):
        raise ProviderLifecycleError("lease record hash must be lowercase SHA-256")


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


__all__ = [
    "ProviderLifecycleBridge",
    "ProviderLifecycleError",
    "ProviderLifecyclePort",
]
