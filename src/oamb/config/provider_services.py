"""Bind secret-free provider-service receipts to preflight descriptors."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oamb.config.redact import is_sensitive_key
from oamb.runtime.preflight import (
    AdapterProfileDescriptor,
    ControlledEmbeddingDescriptor,
    GateStatus,
    ProviderGateClosure,
    TransportKind,
    validate_adapter_profile,
)


class ProviderServiceBindingError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ExactAdapterProfile:
    profile_id: str
    memory_system_id: str
    release_version: str
    source_revision: str
    build_artifact_sha256: str
    transport_kind: TransportKind


@dataclass(frozen=True, slots=True)
class ProviderServiceProfileBinding:
    receipt_sha256: str
    provider_project: str
    project_attestation_sha256: str
    profile_proof_manifest_sha256: str
    verified_at_utc: datetime
    exact_profile: ExactAdapterProfile
    adapter_profile: AdapterProfileDescriptor
    provider_gates: ProviderGateClosure


HINDSIGHT_REST_PROFILE = ExactAdapterProfile(
    profile_id="hindsight-rest-v1",
    memory_system_id="hindsight",
    release_version="0.9.2",
    source_revision="ebad478240d3171bb88201ececda5e8d9883d22d",
    build_artifact_sha256="7635a15739361dbdf221ba796ad25a813f876144fe113022eea8e26cb6ee75e7",
    transport_kind=TransportKind.REST_API,
)
HINDSIGHT_RETAIN_BATCH_LIMIT = 1
MEM0_REST_PROFILE = ExactAdapterProfile(
    profile_id="mem0-rest-v1",
    memory_system_id="mem0",
    release_version="2.0.19",
    source_revision="dc82354e143c2581d505d581a00286d6ef8c3605",
    build_artifact_sha256="5443d9dd99196e33fdde31ef663518c022a8f4a86ef032e7b4bd7b00285705c2",
    transport_kind=TransportKind.REST_API,
)
OPENVIKING_REST_PROFILE = ExactAdapterProfile(
    profile_id="openviking-rest-v1",
    memory_system_id="openviking",
    release_version="0.4.16",
    source_revision="499995f3ed2e7f551a715179c4053772c51ff819",
    build_artifact_sha256="46f9e34cd37238c28cbd9535033773d179006bdf7f3e528dd1c46567abce7701",
    transport_kind=TransportKind.REST_API,
)
MEM0_SDK_PROFILE = ExactAdapterProfile(
    profile_id="mem0-sdk-v1",
    memory_system_id="mem0",
    release_version="2.0.19",
    source_revision="dc82354e143c2581d505d581a00286d6ef8c3605",
    build_artifact_sha256="5443d9dd99196e33fdde31ef663518c022a8f4a86ef032e7b4bd7b00285705c2",
    transport_kind=TransportKind.PYTHON_SDK,
)

DEFAULT_REST_PROFILES = (
    HINDSIGHT_REST_PROFILE,
    MEM0_REST_PROFILE,
    OPENVIKING_REST_PROFILE,
)
PROFILE_PROOF_FILES = {
    "hindsight-rest-v1": (
        "hindsight-health.json",
        "hindsight-version.json",
        "hindsight-model-config.json",
        "hindsight-retry-config.json",
    ),
    "mem0-rest-v1": (
        "mem0-openapi.json",
        "mem0-empty-projection.json",
        "mem0-config-redacted.json",
        "mem0-retry-config.json",
    ),
    "openviking-rest-v1": (
        "openviking-health.json",
        "openviking-auth-identity.json",
        "openviking-storage.json",
        "openviking-model-config.json",
        "openviking-retry-config.json",
    ),
}

_DEFAULT_REST_PROFILE_IDS = tuple(profile.profile_id for profile in DEFAULT_REST_PROFILES)
_REDACTED_VALUE = "[redacted]"
_RECEIPT_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "provider_project",
        "project_attestation_sha256",
        "verified_at_utc",
        "profiles",
    }
)
_PROFILE_KEYS = frozenset(
    {
        "profile_id",
        "proof_manifest_sha256",
        "liveness",
        "storage_configuration",
        "runtime_identity",
        "model_readiness",
        "memory_conformance",
    }
)
_PROOF_MANIFEST_KEYS = frozenset({"schema_name", "schema_version", "profile_id", "files"})
_PROOF_FILE_KEYS = frozenset({"relative_path", "sha256", "byte_count"})
_MEM0_VECTOR_STORE_KEYS = frozenset({"provider", "config"})
_MEM0_PGVECTOR_CONFIG_KEYS = frozenset(
    {
        "host",
        "port",
        "dbname",
        "user",
        "password",
        "collection_name",
        "embedding_model_dims",
        "diskann",
        "hnsw",
    }
)
_PROJECT_PATTERN = re.compile(r"^oamb-providers-[a-z0-9-]{8,48}$")
_OPENAPI_SENSITIVE_SCHEMA_KEYS = frozenset(
    {
        "$ref",
        "allOf",
        "anyOf",
        "deprecated",
        "format",
        "items",
        "nullable",
        "oneOf",
        "readOnly",
        "title",
        "type",
        "writeOnly",
    }
)
_OPENAPI_JSON_TYPES = frozenset(
    {"array", "boolean", "integer", "null", "number", "object", "string"}
)
_OPENAPI_SAFE_FORMATS = frozenset(
    {
        "binary",
        "byte",
        "date",
        "date-time",
        "email",
        "hostname",
        "ipv4",
        "ipv6",
        "password",
        "uuid",
    }
)
_OPENAPI_LOCAL_SCHEMA_REF = re.compile(r"^#/components/schemas/[A-Za-z0-9._-]+$")


def build_adapter_profile_descriptor(
    profile: ExactAdapterProfile,
    controlled_embedding: ControlledEmbeddingDescriptor,
    *,
    provider_project_id: str,
) -> AdapterProfileDescriptor:
    return validate_adapter_profile(
        AdapterProfileDescriptor(
            profile_id=profile.profile_id,
            memory_system_id=profile.memory_system_id,
            transport_kind=profile.transport_kind,
            controlled_embedding=controlled_embedding,
            native_reranking_disabled=True,
            oamb_reranker_configured=False,
            provider_project_id=provider_project_id,
            release_version=profile.release_version,
            source_revision=profile.source_revision,
            build_artifact_sha256=profile.build_artifact_sha256,
        )
    )


def require_default_comparison_profiles(
    profiles: tuple[AdapterProfileDescriptor, ...],
) -> tuple[AdapterProfileDescriptor, ...]:
    expected_identity = tuple(
        (
            profile.profile_id,
            profile.memory_system_id,
            profile.transport_kind,
            profile.release_version,
            profile.source_revision,
            profile.build_artifact_sha256,
        )
        for profile in DEFAULT_REST_PROFILES
    )
    actual_identity = tuple(
        (
            profile.profile_id,
            profile.memory_system_id,
            profile.transport_kind,
            profile.release_version,
            profile.source_revision,
            profile.build_artifact_sha256,
        )
        for profile in profiles
    )
    if actual_identity != expected_identity:
        raise ProviderServiceBindingError(
            "default comparison requires exactly the Hindsight, Mem0, and OpenViking REST profiles"
        )
    return profiles


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _require_exact_keys(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ProviderServiceBindingError(f"{label} has extra or missing fields")
    return value


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProviderServiceBindingError(f"{label} must be a lowercase SHA-256")
    return value


def _require_embedding_mapping(
    value: Mapping[str, object],
    label: str,
) -> dict[str, object]:
    if set(value) != set(_DEFAULT_REST_PROFILE_IDS):
        raise ProviderServiceBindingError(f"{label} must name exactly the three REST profiles")
    return dict(value)


def _require_provider_model_mapping(value: Mapping[str, object]) -> dict[str, str]:
    if set(value) != set(_DEFAULT_REST_PROFILE_IDS):
        raise ProviderServiceBindingError(
            "expected provider models must name exactly the three REST profiles"
        )
    models: dict[str, str] = {}
    for profile_id in _DEFAULT_REST_PROFILE_IDS:
        model = value[profile_id]
        if not isinstance(model, str) or not model:
            raise ProviderServiceBindingError(
                f"expected provider model is missing for {profile_id}"
            )
        models[profile_id] = model
    return models


def _read_regular_proof_file(path: Path, label: str, *, trusted_root: Path) -> bytes:
    path = path.absolute()
    trusted_root = trusted_root.absolute()
    _require_no_symlink_directory_components(path.parent, label, trusted_root=trusted_root)
    try:
        path_stat = os.lstat(path)
    except FileNotFoundError as exc:
        raise ProviderServiceBindingError(f"missing {label}") from exc
    if not stat.S_ISREG(path_stat.st_mode):
        raise ProviderServiceBindingError(f"{label} must be a regular proof file, not a symlink")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise ProviderServiceBindingError(
            f"cannot open {label} without following symlinks"
        ) from exc
    try:
        opened_stat = os.fstat(descriptor)
        if (path_stat.st_dev, path_stat.st_ino) != (opened_stat.st_dev, opened_stat.st_ino):
            raise ProviderServiceBindingError(f"{label} changed while opening")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _require_no_symlink_directory_components(
    directory: Path,
    label: str,
    *,
    trusted_root: Path,
) -> None:
    if not directory.is_relative_to(trusted_root):
        raise ProviderServiceBindingError(f"{label} path is outside its trusted root")
    cursor = trusted_root
    for component in (None, *directory.relative_to(trusted_root).parts):
        if component is not None:
            cursor /= component
        try:
            metadata = cursor.lstat()
        except FileNotFoundError as exc:
            raise ProviderServiceBindingError(f"missing parent directory for {label}") from exc
        if not stat.S_ISDIR(metadata.st_mode):
            raise ProviderServiceBindingError(
                f"{label} path contains a symbolic link or non-directory component"
            )


def _is_openapi_schema_property(path: tuple[str, ...]) -> bool:
    return (
        len(path) >= 5
        and path[0] == "components"
        and path[1] == "schemas"
        and path[3] == "properties"
    )


def _is_openapi_structural_name(path: tuple[str, ...]) -> bool:
    return (
        (len(path) == 2 and path[0] == "paths")
        or (
            len(path) == 3 and path[0] == "components" and path[1] in {"schemas", "securitySchemes"}
        )
        or (len(path) >= 3 and path[-2].isdigit() and path[-3] == "security")
    )


def _validate_sensitive_openapi_schema(
    schema: dict[object, object],
    *,
    property_name: str | None,
) -> None:
    if not all(isinstance(key, str) for key in schema) or not set(schema).issubset(
        _OPENAPI_SENSITIVE_SCHEMA_KEYS
    ):
        raise ProviderServiceBindingError(
            "provider proof contains value-bearing sensitive OpenAPI metadata"
        )
    for field, field_value in schema.items():
        if field == "type":
            values = field_value if isinstance(field_value, list) else [field_value]
            if not values or any(value not in _OPENAPI_JSON_TYPES for value in values):
                raise ProviderServiceBindingError("provider proof contains unsafe OpenAPI type")
        elif field == "format":
            if field_value not in _OPENAPI_SAFE_FORMATS:
                raise ProviderServiceBindingError("provider proof contains unsafe OpenAPI format")
        elif field == "$ref":
            if not isinstance(field_value, str) or not _OPENAPI_LOCAL_SCHEMA_REF.fullmatch(
                field_value
            ):
                raise ProviderServiceBindingError(
                    "provider proof contains unsafe OpenAPI reference"
                )
        elif field == "title":
            expected_title = property_name.replace("_", " ").title() if property_name else None
            if field_value != expected_title:
                raise ProviderServiceBindingError("provider proof contains unsafe OpenAPI title")
        elif field in {"deprecated", "nullable", "readOnly", "writeOnly"}:
            if type(field_value) is not bool:
                raise ProviderServiceBindingError("provider proof contains unsafe OpenAPI flag")
        elif field == "items":
            if not isinstance(field_value, dict):
                raise ProviderServiceBindingError("provider proof contains unsafe OpenAPI items")
            _validate_sensitive_openapi_schema(field_value, property_name=None)
        else:
            if not isinstance(field_value, list) or not field_value:
                raise ProviderServiceBindingError("provider proof contains unsafe OpenAPI variants")
            for variant in field_value:
                if not isinstance(variant, dict):
                    raise ProviderServiceBindingError(
                        "provider proof contains unsafe OpenAPI variants"
                    )
                _validate_sensitive_openapi_schema(variant, property_name=None)


def _validate_secret_free_value(
    value: object,
    *,
    path: tuple[str, ...] = (),
    sensitive_container: bool = False,
    allow_openapi_schema_metadata: bool = False,
) -> None:
    key = path[-1] if path else None
    sensitive_key = (
        key is not None
        and is_sensitive_key(key)
        and not (allow_openapi_schema_metadata and _is_openapi_structural_name(path))
    )
    schema_metadata = (
        sensitive_key
        and _is_openapi_schema_property(path)
        and allow_openapi_schema_metadata
        and isinstance(value, dict)
    )
    if schema_metadata:
        assert isinstance(value, dict) and key is not None
        _validate_sensitive_openapi_schema(value, property_name=key)
    sensitive_container = sensitive_container or (sensitive_key and not schema_metadata)
    if sensitive_container and not isinstance(value, (dict, list)):
        if value is not None and value != _REDACTED_VALUE:
            raise ProviderServiceBindingError("provider proof contains an unredacted secret")
        return
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _validate_secret_free_value(
                child_value,
                path=(*path, str(child_key)),
                sensitive_container=sensitive_container,
                allow_openapi_schema_metadata=allow_openapi_schema_metadata,
            )
    elif isinstance(value, list):
        for child_value in value:
            _validate_secret_free_value(
                child_value,
                path=path,
                sensitive_container=sensitive_container,
                allow_openapi_schema_metadata=allow_openapi_schema_metadata,
            )


def _parse_secret_free_proof(content: bytes, *, filename: str) -> object:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderServiceBindingError("provider proof blob is not valid UTF-8 JSON") from exc
    _validate_secret_free_value(
        value,
        allow_openapi_schema_metadata=filename == "mem0-openapi.json",
    )
    return value


def _semantic_error(filename: str) -> ProviderServiceBindingError:
    return ProviderServiceBindingError(f"provider proof semantic/model check failed: {filename}")


def _validate_proof_semantics(
    filename: str,
    value: object,
    *,
    expected_model: str,
) -> None:
    if not isinstance(value, dict):
        raise _semantic_error(filename)
    valid = False
    if filename == "hindsight-health.json":
        valid = value.get("status") == "healthy" and value.get("database") == "connected"
    elif filename == "hindsight-version.json":
        valid = value.get("api_version") == "0.9.2" and isinstance(value.get("features"), dict)
    elif filename == "hindsight-model-config.json":
        valid = value.get("model") == expected_model and value.get("reasoning_effort") == "low"
    elif filename == "hindsight-retry-config.json":
        valid = value == {
            "llm_max_retries": 0,
            "openai_sdk_max_retries": 0,
            "retain_llm_max_retries": 0,
            "worker_max_retries": 0,
        }
    elif filename == "mem0-openapi.json":
        paths = value.get("paths")
        valid = isinstance(paths, dict) and all(path in paths for path in ("/memories", "/search"))
    elif filename == "mem0-empty-projection.json":
        valid = (
            value.get("count") == 0
            and value.get("points") == []
            and value.get("next_cursor") is None
        )
    elif filename == "mem0-config-redacted.json":
        vector_store = value.get("vector_store")
        vector_config = vector_store.get("config") if isinstance(vector_store, dict) else None
        embedder = value.get("embedder")
        embedder_config = embedder.get("config") if isinstance(embedder, dict) else None
        llm = value.get("llm")
        llm_config = llm.get("config") if isinstance(llm, dict) else None
        valid = (
            value.get("version") == "v1.1"
            and isinstance(vector_store, dict)
            and set(vector_store) == _MEM0_VECTOR_STORE_KEYS
            and vector_store.get("provider") == "pgvector"
            and isinstance(vector_config, dict)
            and set(vector_config) == _MEM0_PGVECTOR_CONFIG_KEYS
            and vector_config.get("host") == "mem0-postgres"
            and vector_config.get("port") == 5432
            and vector_config.get("dbname") == "postgres"
            and vector_config.get("user") == "oamb_mem0"
            and vector_config.get("password") == _REDACTED_VALUE
            and vector_config.get("collection_name") == "oamb_memories"
            and vector_config.get("embedding_model_dims") == 1024
            and vector_config.get("diskann") is False
            and vector_config.get("hnsw") is True
            and isinstance(embedder_config, dict)
            and embedder_config.get("embedding_dims") == 1024
            and isinstance(llm_config, dict)
            and llm_config.get("model") == expected_model
            and llm_config.get("reasoning_effort") == "low"
            and llm_config.get("is_reasoning_model") is True
            and value.get("reranker") is None
        )
    elif filename == "mem0-retry-config.json":
        valid = value == {"openai_sdk_max_retries": 0}
    elif filename == "openviking-health.json":
        valid = (
            value.get("status") == "ok"
            and value.get("healthy") is True
            and value.get("version") == "v0.4.16"
            and value.get("auth_mode") == "api_key"
        )
    elif filename == "openviking-auth-identity.json":
        valid = (
            isinstance(value.get("account_id"), str)
            and bool(value["account_id"])
            and isinstance(value.get("user_id"), str)
            and bool(value["user_id"])
            and value.get("user_role") == "admin"
            and value.get("provisioning_role") == "root"
            and value.get("initialized") is True
        )
    elif filename == "openviking-storage.json":
        valid = (
            value.get("status") == "ok"
            and value.get("openviking_version") == "0.4.16"
            and value.get("mode") == "read_only_storage"
            and value.get("dimension") == 1024
            and value.get("collection") == "context"
            and type(value.get("files_verified")) is int
            and int(value["files_verified"]) > 0
            and type(value.get("agfs_entries")) is int
            and int(value["agfs_entries"]) >= 0
        )
    elif filename == "openviking-model-config.json":
        valid = (
            value.get("provider") == "openai"
            and value.get("model") == expected_model
            and value.get("reasoning_effort") == "low"
        )
    elif filename == "openviking-retry-config.json":
        valid = value == {
            "embedding_max_retries": 0,
            "memory_extraction_max_retries": 0,
            "openai_sdk_max_retries": 0,
            "vlm_max_retries": 0,
        }
    if not valid:
        raise _semantic_error(filename)


def _validate_profile_proof_store(
    receipt_path: Path,
    profile: ExactAdapterProfile,
    manifest_hash: str,
    *,
    expected_model: str,
) -> None:
    proof_store = receipt_path.parent / "proofs"
    manifest_path = proof_store / "manifests" / f"{manifest_hash}.json"
    manifest_bytes = _read_regular_proof_file(
        manifest_path,
        "provider proof manifest",
        trusted_root=receipt_path.parent,
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != manifest_hash:
        raise ProviderServiceBindingError("provider proof manifest content hash does not match")
    try:
        value = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderServiceBindingError(
            "provider proof manifest is not valid UTF-8 JSON"
        ) from exc
    manifest = _require_exact_keys(value, _PROOF_MANIFEST_KEYS, "provider proof manifest")
    if _canonical_json_bytes(manifest) != manifest_bytes:
        raise ProviderServiceBindingError("provider proof manifest is not canonical JSON")
    if (
        manifest["schema_name"] != "oamb_provider_profile_proof_manifest"
        or type(manifest["schema_version"]) is not int
        or manifest["schema_version"] != 1
        or manifest["profile_id"] != profile.profile_id
    ):
        raise ProviderServiceBindingError(
            "provider proof manifest profile or schema does not match"
        )
    files = manifest["files"]
    if not isinstance(files, list):
        raise ProviderServiceBindingError("provider proof manifest file set is invalid")
    entries = tuple(
        _require_exact_keys(item, _PROOF_FILE_KEYS, "provider proof manifest entry")
        for item in files
    )
    expected_filenames = PROFILE_PROOF_FILES[profile.profile_id]
    actual_filenames = tuple(entry["relative_path"] for entry in entries)
    if actual_filenames != expected_filenames:
        raise ProviderServiceBindingError(
            "provider proof manifest file set must contain exact relative filenames"
        )
    for entry in entries:
        blob_hash = _require_sha256(entry["sha256"], "provider proof blob hash")
        byte_count = entry["byte_count"]
        if type(byte_count) is not int or byte_count < 0:
            raise ProviderServiceBindingError(
                "provider proof manifest byte count must be a non-negative integer"
            )
        blob_path = proof_store / "blobs" / blob_hash
        blob_bytes = _read_regular_proof_file(
            blob_path,
            "provider proof blob",
            trusted_root=receipt_path.parent,
        )
        if hashlib.sha256(blob_bytes).hexdigest() != blob_hash:
            raise ProviderServiceBindingError("provider proof blob content hash does not match")
        if len(blob_bytes) != byte_count:
            raise ProviderServiceBindingError("provider proof blob byte count does not match")
        filename = str(entry["relative_path"])
        proof = _parse_secret_free_proof(blob_bytes, filename=filename)
        _validate_proof_semantics(filename, proof, expected_model=expected_model)


def _parse_canonical_receipt(content: bytes) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProviderServiceBindingError(
            "provider service receipt is not valid UTF-8 JSON"
        ) from exc
    receipt = _require_exact_keys(value, _RECEIPT_KEYS, "provider service receipt")
    if _canonical_json_bytes(receipt) != content:
        raise ProviderServiceBindingError("provider service receipt is not canonical JSON")
    if (
        receipt["schema_name"] != "oamb_provider_service_verification"
        or type(receipt["schema_version"]) is not int
        or receipt["schema_version"] != 1
    ):
        raise ProviderServiceBindingError("unsupported provider service receipt schema")
    return receipt


def _parse_verified_at(value: object) -> datetime:
    if not isinstance(value, str):
        raise ProviderServiceBindingError("verified_at_utc must be a UTC timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    except ValueError as exc:
        raise ProviderServiceBindingError("verified_at_utc must be a UTC timestamp") from exc


def _validated_profile_records(value: object) -> tuple[dict[str, Any], ...]:
    if not isinstance(value, list) or len(value) != len(DEFAULT_REST_PROFILES):
        raise ProviderServiceBindingError("receipt requires exactly the three REST profiles")
    records = tuple(
        _require_exact_keys(item, _PROFILE_KEYS, "provider profile receipt") for item in value
    )
    if tuple(record["profile_id"] for record in records) != _DEFAULT_REST_PROFILE_IDS:
        raise ProviderServiceBindingError(
            "receipt profiles must be the ordered Hindsight, Mem0, and OpenViking REST profiles"
        )
    expected_gate_values = {
        "liveness": "passed",
        "storage_configuration": "passed",
        "runtime_identity": "passed",
        "model_readiness": "not_run",
        "memory_conformance": "not_run",
    }
    for record in records:
        if any(record[field] != expected for field, expected in expected_gate_values.items()):
            raise ProviderServiceBindingError(
                "service receipt requires PASS liveness/storage/runtime identity and "
                "NOT_RUN model readiness/memory conformance"
            )
        _require_sha256(record["proof_manifest_sha256"], "profile proof manifest hash")
    return records


def load_provider_service_bindings(
    receipt_path: Path,
    *,
    expected_project: str,
    expected_project_attestation_sha256: str,
    controlled_embeddings: Mapping[str, ControlledEmbeddingDescriptor],
    expected_provider_models: Mapping[str, str],
) -> tuple[ProviderServiceProfileBinding, ...]:
    content = _read_regular_proof_file(
        receipt_path,
        "provider service receipt",
        trusted_root=receipt_path.parent,
    )
    receipt_sha256 = hashlib.sha256(content).hexdigest()
    if receipt_path.name != f"{receipt_sha256}.json":
        raise ProviderServiceBindingError(
            "provider service receipt content hash does not match its name"
        )
    receipt = _parse_canonical_receipt(content)
    if not _PROJECT_PATTERN.fullmatch(expected_project):
        raise ProviderServiceBindingError("expected provider project has an invalid identity")
    if receipt["provider_project"] != expected_project:
        raise ProviderServiceBindingError("provider service receipt project does not match")
    attestation_hash = _require_sha256(
        receipt["project_attestation_sha256"],
        "provider project attestation hash",
    )
    _require_sha256(
        expected_project_attestation_sha256,
        "expected provider project attestation hash",
    )
    if attestation_hash != expected_project_attestation_sha256:
        raise ProviderServiceBindingError("provider project attestation hash does not match")

    verified_at = _parse_verified_at(receipt["verified_at_utc"])
    profile_records = _validated_profile_records(receipt["profiles"])
    embeddings = _require_embedding_mapping(
        controlled_embeddings,
        "controlled embedding descriptors",
    )
    provider_models = _require_provider_model_mapping(expected_provider_models)
    bindings: list[ProviderServiceProfileBinding] = []
    for exact_profile, profile_record in zip(
        DEFAULT_REST_PROFILES,
        profile_records,
        strict=True,
    ):
        profile_id = exact_profile.profile_id
        proof_manifest_sha256 = _require_sha256(
            profile_record["proof_manifest_sha256"],
            "profile proof manifest hash",
        )
        _validate_profile_proof_store(
            receipt_path,
            exact_profile,
            proof_manifest_sha256,
            expected_model=provider_models[profile_id],
        )
        embedding = embeddings[profile_id]
        if not isinstance(embedding, ControlledEmbeddingDescriptor):
            raise ProviderServiceBindingError(
                f"controlled embedding descriptor is not typed for {profile_id}"
            )
        gates = ProviderGateClosure(
            liveness=GateStatus.PASS,
            storage_configuration=GateStatus.PASS,
            runtime_identity=GateStatus.PASS,
            model_readiness=GateStatus.NOT_RUN,
            memory_conformance=GateStatus.NOT_RUN,
        )
        bindings.append(
            ProviderServiceProfileBinding(
                receipt_sha256=receipt_sha256,
                provider_project=expected_project,
                project_attestation_sha256=attestation_hash,
                profile_proof_manifest_sha256=proof_manifest_sha256,
                verified_at_utc=verified_at,
                exact_profile=exact_profile,
                adapter_profile=build_adapter_profile_descriptor(
                    exact_profile,
                    embedding,
                    provider_project_id=expected_project,
                ),
                provider_gates=gates,
            )
        )
    require_default_comparison_profiles(tuple(binding.adapter_profile for binding in bindings))
    return tuple(bindings)
