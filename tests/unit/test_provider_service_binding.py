from __future__ import annotations

import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from oamb.config.benchmark import load_benchmark_configuration
from tests.benchmark_configuration import MODEL_ENVIRONMENT

if TYPE_CHECKING:
    from oamb.config.provider_services import ProviderServiceProfileBinding
    from oamb.runtime.preflight import ControlledEmbeddingDescriptor

HASH_A = "a" * 64
HASH_E = "e" * 64
HASH_F = "f" * 64

PROJECT = "oamb-providers-test-alpha"
BENCHMARK_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "benchmark.yml"
BENCHMARK_MODELS = load_benchmark_configuration(
    BENCHMARK_CONFIG, model_environment=MODEL_ENVIRONMENT
).models
EXPECTED_PROVIDER_MODELS = {
    "hindsight-rest-v1": BENCHMARK_MODELS.hindsight_extraction.model,
    "mem0-rest-v1": BENCHMARK_MODELS.mem0_extraction.model,
    "openviking-rest-v1": BENCHMARK_MODELS.openviking_semantic_understanding.model,
}
EXPECTED_EMBEDDING_MODEL = BENCHMARK_MODELS.embedding.model
EXPECTED_PROVIDER_THINKING_EFFORTS = {
    "hindsight-rest-v1": "low",
    "mem0-rest-v1": "low",
    "openviking-rest-v1": "low",
}
PROFILE_FILES = {
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


def test_provider_and_runner_proof_inventories_match() -> None:
    producer_path = (
        Path(__file__).resolve().parents[2] / "provider-services" / "lib" / "service_receipt.py"
    )
    spec = importlib.util.spec_from_file_location("provider_service_receipt", producer_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.PROFILE_PROOF_FILES == PROFILE_FILES


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _proof_bytes(filename: str) -> bytes:
    documents: dict[str, object] = {
        "hindsight-health.json": {"database": "connected", "status": "healthy"},
        "hindsight-version.json": {"api_version": "0.9.2", "features": {}},
        "hindsight-model-config.json": {
            "model": EXPECTED_PROVIDER_MODELS["hindsight-rest-v1"],
            "reasoning_effort": "low",
        },
        "hindsight-retry-config.json": {
            "extraction_max_retries": 10,
            "llm_base_url_host": "api.deepseek.com",
            "llm_max_retries": 0,
            "NO_PROXY": "127.0.0.1,localhost,host.docker.internal,api.deepseek.com",
            "no_proxy": "127.0.0.1,localhost,host.docker.internal,api.deepseek.com",
            "openai_sdk_max_retries": 0,
            "retain_llm_max_retries": 10,
            "worker_max_retries": 0,
        },
        "mem0-openapi.json": {
            "components": {"schemas": {"Login": {"properties": {"password": {"type": "string"}}}}},
            "paths": {"/memories": {}, "/search": {}},
        },
        "mem0-empty-projection.json": {"count": 0, "next_cursor": None, "points": []},
        "mem0-config-redacted.json": {
            "api_key": "[redacted]",
            "embedder": {"config": {"embedding_dims": 1024}},
            "llm": {
                "config": {
                    "is_reasoning_model": True,
                    "model": EXPECTED_PROVIDER_MODELS["mem0-rest-v1"],
                    "reasoning_effort": "low",
                }
            },
            "reranker": None,
            "vector_store": {
                "config": {
                    "collection_name": "oamb_memories",
                    "dbname": "postgres",
                    "diskann": False,
                    "embedding_model_dims": 1024,
                    "hnsw": True,
                    "host": "mem0-postgres",
                    "password": "[redacted]",
                    "port": 5432,
                    "user": "oamb_mem0",
                },
                "provider": "pgvector",
            },
            "version": "v1.1",
        },
        "mem0-retry-config.json": {
            "extraction_max_retries": 10,
            "llm_base_url_host": "api.deepseek.com",
            "NO_PROXY": "127.0.0.1,localhost,host.docker.internal,api.deepseek.com",
            "no_proxy": "127.0.0.1,localhost,host.docker.internal,api.deepseek.com",
            "openai_sdk_max_retries": 0,
        },
        "openviking-health.json": {
            "auth_mode": "api_key",
            "healthy": True,
            "status": "ok",
            "version": "v0.4.16",
        },
        "openviking-auth-identity.json": {
            "account_id": "oamb-benchmark",
            "initialized": True,
            "provisioning_role": "root",
            "user_id": "oamb-admin",
            "user_role": "admin",
        },
        "openviking-storage.json": {
            "agfs_entries": 0,
            "collection": "context",
            "dimension": 1024,
            "files_verified": 1,
            "mode": "read_only_storage",
            "openviking_version": "0.4.16",
            "status": "ok",
        },
        "openviking-model-config.json": {
            "model": EXPECTED_PROVIDER_MODELS["openviking-rest-v1"],
            "provider": "openai",
            "reasoning_effort": "low",
        },
        "openviking-retry-config.json": {
            "embedding_max_retries": 0,
            "extraction_max_retries": 10,
            "llm_base_url_host": "api.deepseek.com",
            "memory_extraction_max_retries": 10,
            "NO_PROXY": "127.0.0.1,localhost,host.docker.internal,api.deepseek.com",
            "no_proxy": "127.0.0.1,localhost,host.docker.internal,api.deepseek.com",
            "openai_sdk_max_retries": 0,
            "vlm_max_retries": 0,
        },
    }
    return _canonical_bytes(documents.get(filename, {"filename": filename, "status": "ok"}))


def _write_profile_manifest(
    receipt_directory: Path,
    profile_id: str,
    *,
    filenames: tuple[str, ...] | None = None,
    payload_overrides: dict[str, bytes] | None = None,
) -> str:
    proof_store = receipt_directory / "proofs"
    manifests = proof_store / "manifests"
    blobs = proof_store / "blobs"
    manifests.mkdir(parents=True, exist_ok=True)
    blobs.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    for filename in filenames or PROFILE_FILES[profile_id]:
        content = (payload_overrides or {}).get(filename, _proof_bytes(filename))
        content_hash = hashlib.sha256(content).hexdigest()
        (blobs / content_hash).write_bytes(content)
        entries.append(
            {
                "relative_path": filename,
                "sha256": content_hash,
                "byte_count": len(content),
            }
        )
    manifest_bytes = _canonical_bytes(
        {
            "schema_name": "oamb_provider_profile_proof_manifest",
            "schema_version": 1,
            "profile_id": profile_id,
            "files": entries,
        }
    )
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    (manifests / f"{manifest_hash}.json").write_bytes(manifest_bytes)
    return manifest_hash


def _write_service_artifacts(
    root: Path,
    *,
    filenames_by_profile: dict[str, tuple[str, ...]] | None = None,
    payload_overrides: dict[str, bytes] | None = None,
) -> Path:
    receipt_directory = root / "receipts"
    receipt_directory.mkdir(parents=True, exist_ok=True)
    manifest_hashes = {
        profile_id: _write_profile_manifest(
            receipt_directory,
            profile_id,
            filenames=(filenames_by_profile or {}).get(profile_id),
            payload_overrides=payload_overrides,
        )
        for profile_id in PROFILE_FILES
    }
    receipt = {
        "schema_name": "oamb_provider_service_verification",
        "schema_version": 1,
        "provider_project": PROJECT,
        "project_attestation_sha256": HASH_A,
        "verified_at_utc": "2026-08-27T12:00:00Z",
        "profiles": [
            {
                "profile_id": profile_id,
                "proof_manifest_sha256": manifest_hashes[profile_id],
                "liveness": "passed",
                "storage_configuration": "passed",
                "runtime_identity": "passed",
                "model_readiness": "not_run",
                "memory_conformance": "not_run",
            }
            for profile_id in PROFILE_FILES
        ],
    }
    return _write_receipt(receipt_directory, receipt)


def _write_receipt(receipt_directory: Path, receipt: dict[str, object]) -> Path:
    content = _canonical_bytes(receipt)
    path = receipt_directory / f"{hashlib.sha256(content).hexdigest()}.json"
    path.write_bytes(content)
    return path


def _controlled_embeddings() -> dict[str, ControlledEmbeddingDescriptor]:
    from oamb.runtime.preflight import ControlledEmbeddingDescriptor

    return {
        "hindsight-rest-v1": ControlledEmbeddingDescriptor(
            endpoint_fingerprint=HASH_E,
            model=EXPECTED_EMBEDDING_MODEL,
            artifact_fingerprint=HASH_F,
            dimension=1024,
            input_adaptation_fingerprint="1" * 64,
        ),
        "mem0-rest-v1": ControlledEmbeddingDescriptor(
            endpoint_fingerprint=HASH_E,
            model=EXPECTED_EMBEDDING_MODEL,
            artifact_fingerprint=HASH_F,
            dimension=1024,
            input_adaptation_fingerprint="2" * 64,
        ),
        "openviking-rest-v1": ControlledEmbeddingDescriptor(
            endpoint_fingerprint=HASH_E,
            model=EXPECTED_EMBEDDING_MODEL,
            artifact_fingerprint=HASH_F,
            dimension=1024,
            input_adaptation_fingerprint="3" * 64,
        ),
    }


def _load(
    receipt_path: Path,
    *,
    expected_provider_thinking_efforts: dict[str, str] | None = None,
) -> tuple[ProviderServiceProfileBinding, ...]:
    from oamb.config.provider_services import load_provider_service_bindings

    return load_provider_service_bindings(
        receipt_path,
        expected_project=PROJECT,
        expected_project_attestation_sha256=HASH_A,
        controlled_embeddings=_controlled_embeddings(),
        expected_provider_models=EXPECTED_PROVIDER_MODELS,
        expected_provider_thinking_efforts=(
            expected_provider_thinking_efforts or EXPECTED_PROVIDER_THINKING_EFFORTS
        ),
    )


def _read_receipt(receipt_path: Path) -> dict[str, object]:
    value = json.loads(receipt_path.read_bytes())
    assert isinstance(value, dict)
    return value


def test_profile_bindings_carry_strict_extraction_retry_count_from_validated_proofs(
    tmp_path: Path,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    valid = _write_service_artifacts(tmp_path / "valid")
    assert tuple(binding.internal_retry_count for binding in _load(valid)) == (10, 10, 10)

    invalid_payloads = (
        b'{"NO_PROXY":"127.0.0.1,localhost,host.docker.internal,api.deepseek.com",'
        b'"llm_base_url_host":"api.deepseek.com","no_proxy":"127.0.0.1,localhost,'
        b'host.docker.internal,api.deepseek.com","openai_sdk_max_retries":false}',
        b'{"NO_PROXY":"127.0.0.1,localhost,host.docker.internal,api.deepseek.com",'
        b'"llm_base_url_host":"api.deepseek.com","openai_sdk_max_retries":0}',
        b'{"NO_PROXY":"127.0.0.1,localhost,host.docker.internal,other.example",'
        b'"llm_base_url_host":"api.deepseek.com","no_proxy":"127.0.0.1,localhost,'
        b'host.docker.internal,other.example","openai_sdk_max_retries":0}',
    )
    for index, payload in enumerate(invalid_payloads):
        invalid = _write_service_artifacts(
            tmp_path / f"invalid-{index}",
            payload_overrides={"mem0-retry-config.json": payload},
        )
        with pytest.raises(ProviderServiceBindingError):
            _load(invalid)


def _rewrite_profile_manifest(
    receipt_path: Path,
    profile_index: int,
    manifest: dict[str, object],
) -> Path:
    manifest_bytes = _canonical_bytes(manifest)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    manifests = receipt_path.parent / "proofs" / "manifests"
    (manifests / f"{manifest_hash}.json").write_bytes(manifest_bytes)
    receipt = _read_receipt(receipt_path)
    profiles = receipt["profiles"]
    assert isinstance(profiles, list)
    profile = profiles[profile_index]
    assert isinstance(profile, dict)
    profile["proof_manifest_sha256"] = manifest_hash
    return _write_receipt(receipt_path.parent, receipt)


def test_exact_profiles_build_preflight_descriptors_and_keep_sdk_separate() -> None:
    from oamb.config.provider_services import (
        HINDSIGHT_REST_PROFILE,
        MEM0_REST_PROFILE,
        MEM0_SDK_PROFILE,
        OPENVIKING_REST_PROFILE,
        build_adapter_profile_descriptor,
        require_default_comparison_profiles,
    )
    from oamb.runtime.preflight import ControlledEmbeddingDescriptor, TransportKind

    embedding = ControlledEmbeddingDescriptor(
        endpoint_fingerprint=HASH_E,
        model=EXPECTED_EMBEDDING_MODEL,
        artifact_fingerprint=HASH_F,
        dimension=1024,
        input_adaptation_fingerprint="1" * 64,
    )
    exact_profiles = (
        HINDSIGHT_REST_PROFILE,
        MEM0_REST_PROFILE,
        OPENVIKING_REST_PROFILE,
        MEM0_SDK_PROFILE,
    )

    assert tuple(profile.profile_id for profile in exact_profiles) == (
        "hindsight-rest-v1",
        "mem0-rest-v1",
        "openviking-rest-v1",
        "mem0-sdk-v1",
    )
    assert tuple(profile.release_version for profile in exact_profiles) == (
        "0.9.2",
        "2.0.19",
        "0.4.16",
        "2.0.19",
    )
    assert tuple(profile.source_revision for profile in exact_profiles) == (
        "ebad478240d3171bb88201ececda5e8d9883d22d",
        "dc82354e143c2581d505d581a00286d6ef8c3605",
        "499995f3ed2e7f551a715179c4053772c51ff819",
        "dc82354e143c2581d505d581a00286d6ef8c3605",
    )
    assert tuple(profile.transport_kind for profile in exact_profiles) == (
        TransportKind.REST_API,
        TransportKind.REST_API,
        TransportKind.REST_API,
        TransportKind.PYTHON_SDK,
    )

    rest_descriptors = tuple(
        build_adapter_profile_descriptor(
            profile,
            embedding,
            provider_project_id=PROJECT,
        )
        for profile in exact_profiles[:3]
    )
    assert require_default_comparison_profiles(rest_descriptors) == rest_descriptors

    sdk_descriptor = build_adapter_profile_descriptor(
        MEM0_SDK_PROFILE,
        embedding,
        provider_project_id=PROJECT,
    )
    assert sdk_descriptor.default_comparison_eligible is False
    with pytest.raises(ValueError, match="default comparison.*REST profiles"):
        require_default_comparison_profiles(
            (rest_descriptors[0], sdk_descriptor, rest_descriptors[2])
        )


def test_consumer_independently_validates_proof_store_and_builds_bindings(
    tmp_path: Path,
) -> None:
    from oamb.runtime.preflight import GateStatus, TransportKind

    receipt_path = _write_service_artifacts(tmp_path)

    bindings = _load(receipt_path)

    assert tuple(binding.adapter_profile.profile_id for binding in bindings) == (
        "hindsight-rest-v1",
        "mem0-rest-v1",
        "openviking-rest-v1",
    )
    assert all(
        binding.adapter_profile.transport_kind == TransportKind.REST_API for binding in bindings
    )
    assert all(binding.provider_project == PROJECT for binding in bindings)
    assert all(binding.project_attestation_sha256 == HASH_A for binding in bindings)
    assert all(len(binding.profile_proof_manifest_sha256) == 64 for binding in bindings)
    assert all(binding.receipt_sha256 == receipt_path.stem for binding in bindings)
    for binding in bindings:
        assert binding.provider_gates.liveness == GateStatus.PASS
        assert binding.provider_gates.storage_configuration == GateStatus.PASS
        assert binding.provider_gates.runtime_identity == GateStatus.PASS
        assert binding.provider_gates.model_readiness == GateStatus.NOT_RUN
        assert binding.provider_gates.memory_conformance == GateStatus.NOT_RUN


@pytest.mark.parametrize(
    ("filename", "model_path"),
    (
        ("hindsight-model-config.json", ("model",)),
        ("mem0-config-redacted.json", ("llm", "config", "model")),
        ("openviking-model-config.json", ("model",)),
    ),
)
def test_consumer_rejects_provider_model_proof_that_differs_from_resolved_plan(
    tmp_path: Path,
    filename: str,
    model_path: tuple[str, ...],
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    document = json.loads(_proof_bytes(filename))
    target = document
    for key in model_path[:-1]:
        target = target[key]
    target[model_path[-1]] = "different-model"
    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={filename: _canonical_bytes(document)},
    )

    with pytest.raises(ProviderServiceBindingError, match="model"):
        _load(receipt_path)


@pytest.mark.parametrize(
    "filenames",
    (
        ("hindsight-health.json",),
        (
            "hindsight-health.json",
            "hindsight-version.json",
            "unexpected.json",
        ),
        ("/tmp/hindsight-health.json", "hindsight-version.json"),
    ),
)
def test_consumer_rejects_nonclosed_or_absolute_manifest_file_sets(
    tmp_path: Path,
    filenames: tuple[str, ...],
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(
        tmp_path,
        filenames_by_profile={"hindsight-rest-v1": filenames},
    )

    with pytest.raises(ProviderServiceBindingError, match="file set|relative"):
        _load(receipt_path)


def test_consumer_rejects_a_blob_whose_bytes_do_not_match_manifest(tmp_path: Path) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(tmp_path)
    receipt = _read_receipt(receipt_path)
    profiles = receipt["profiles"]
    assert isinstance(profiles, list)
    first_profile = profiles[0]
    assert isinstance(first_profile, dict)
    manifest_hash = first_profile["proof_manifest_sha256"]
    assert isinstance(manifest_hash, str)
    manifest_path = receipt_path.parent / "proofs" / "manifests" / f"{manifest_hash}.json"
    manifest = json.loads(manifest_path.read_bytes())
    blob_hash = manifest["files"][0]["sha256"]
    (receipt_path.parent / "proofs" / "blobs" / blob_hash).write_bytes(b"tampered")

    with pytest.raises(ProviderServiceBindingError, match="proof blob.*hash"):
        _load(receipt_path)


def test_consumer_rejects_a_manifest_byte_count_mismatch(tmp_path: Path) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(tmp_path)
    receipt = _read_receipt(receipt_path)
    profiles = receipt["profiles"]
    assert isinstance(profiles, list)
    first_profile = profiles[0]
    assert isinstance(first_profile, dict)
    manifest_hash = first_profile["proof_manifest_sha256"]
    assert isinstance(manifest_hash, str)
    manifest_path = receipt_path.parent / "proofs" / "manifests" / f"{manifest_hash}.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["files"][0]["byte_count"] += 1
    changed_receipt = _rewrite_profile_manifest(receipt_path, 0, manifest)

    with pytest.raises(ProviderServiceBindingError, match="byte count"):
        _load(changed_receipt)


def test_consumer_rejects_a_proof_blob_symlink(tmp_path: Path) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(tmp_path)
    receipt = _read_receipt(receipt_path)
    profiles = receipt["profiles"]
    assert isinstance(profiles, list)
    first_profile = profiles[0]
    assert isinstance(first_profile, dict)
    manifest_hash = first_profile["proof_manifest_sha256"]
    assert isinstance(manifest_hash, str)
    manifest_path = receipt_path.parent / "proofs" / "manifests" / f"{manifest_hash}.json"
    manifest = json.loads(manifest_path.read_bytes())
    blob_hash = manifest["files"][0]["sha256"]
    blob_path = receipt_path.parent / "proofs" / "blobs" / blob_hash
    content = blob_path.read_bytes()
    external = tmp_path / "external.json"
    external.write_bytes(content)
    blob_path.unlink()
    blob_path.symlink_to(external)

    with pytest.raises(ProviderServiceBindingError, match="regular proof file|symlink"):
        _load(receipt_path)


def test_consumer_rejects_a_receipt_leaf_symlink(tmp_path: Path) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(tmp_path)
    content = receipt_path.read_bytes()
    external = tmp_path / "external-receipt.json"
    external.write_bytes(content)
    receipt_path.unlink()
    receipt_path.symlink_to(external)

    with pytest.raises(ProviderServiceBindingError, match="regular proof file|symlink"):
        _load(receipt_path)


def test_consumer_rejects_a_symlink_in_receipt_ancestors(tmp_path: Path) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(tmp_path)
    linked_parent = tmp_path / "linked-receipts"
    linked_parent.symlink_to(receipt_path.parent, target_is_directory=True)
    linked_receipt = linked_parent / receipt_path.name

    with pytest.raises(ProviderServiceBindingError, match="symbolic link|symlink"):
        _load(linked_receipt)


@pytest.mark.parametrize(
    "sensitive_key",
    (
        "api_key",
        "apiKey",
        "api-key",
        "clientSecret",
        "accessToken",
        "clientSecretValue",
        "oauthAccessTokenValue",
        "myApiKeyValue",
        "nestedCredentialValue",
    ),
)
def test_consumer_rejects_secret_proof_bytes_even_when_hashes_close(
    tmp_path: Path,
    sensitive_key: str,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={
            "hindsight-health.json": _canonical_bytes(
                {
                    sensitive_key: "plaintext-secret",
                    "database": "connected",
                    "status": "healthy",
                }
            )
        },
    )

    with pytest.raises(ProviderServiceBindingError, match="secret"):
        _load(receipt_path)


def test_consumer_allows_sensitive_field_names_inside_openapi_schema(
    tmp_path: Path,
) -> None:
    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={
            "mem0-openapi.json": _canonical_bytes(
                {
                    "components": {
                        "schemas": {
                            "LoginRequest": {
                                "properties": {"password": {"type": "string", "format": "password"}}
                            }
                        }
                    },
                    "paths": {"/memories": {}, "/search": {}},
                }
            )
        },
    )

    assert len(_load(receipt_path)) == 3


def test_consumer_rejects_a_scalar_secret_at_an_openapi_schema_property(
    tmp_path: Path,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={
            "mem0-openapi.json": _canonical_bytes(
                {
                    "components": {
                        "schemas": {"Login": {"properties": {"password": "plaintext-secret"}}}
                    },
                    "paths": {"/memories": {}, "/search": {}},
                }
            )
        },
    )

    with pytest.raises(ProviderServiceBindingError, match="secret"):
        _load(receipt_path)


@pytest.mark.parametrize("value_key", ("default", "example", "examples", "const", "enum"))
def test_consumer_rejects_value_bearing_sensitive_openapi_metadata(
    tmp_path: Path,
    value_key: str,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    value: object = (
        ["plaintext-secret"] if value_key in {"examples", "enum"} else "plaintext-secret"
    )
    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={
            "mem0-openapi.json": _canonical_bytes(
                {
                    "components": {
                        "schemas": {
                            "Login": {
                                "properties": {"password": {"type": "string", value_key: value}}
                            }
                        }
                    },
                    "paths": {"/memories": {}, "/search": {}},
                }
            )
        },
    )

    with pytest.raises(ProviderServiceBindingError, match="OpenAPI metadata"):
        _load(receipt_path)


@pytest.mark.parametrize(
    "schema",
    (
        {"type": "string", "format": "plaintext-secret"},
        {"$ref": "#/plaintext-secret"},
    ),
)
def test_consumer_rejects_unsafe_sensitive_openapi_structural_values(
    tmp_path: Path,
    schema: dict[str, object],
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={
            "mem0-openapi.json": _canonical_bytes(
                {
                    "components": {"schemas": {"Login": {"properties": {"password": schema}}}},
                    "paths": {"/memories": {}, "/search": {}},
                }
            )
        },
    )

    with pytest.raises(ProviderServiceBindingError, match="unsafe OpenAPI"):
        _load(receipt_path)


def test_consumer_rejects_plaintext_nested_under_a_sensitive_container(
    tmp_path: Path,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={
            "hindsight-health.json": _canonical_bytes(
                {
                    "authorization": {"value": "Bearer plaintext-secret"},
                    "database": "connected",
                    "status": "healthy",
                }
            )
        },
    )

    with pytest.raises(ProviderServiceBindingError, match="secret"):
        _load(receipt_path)


@pytest.mark.parametrize(
    ("filename", "document"),
    [
        (
            "hindsight-health.json",
            {"status": "unhealthy", "database": "connected"},
        ),
        (
            "hindsight-version.json",
            {"api_version": "99.0.0", "features": {}},
        ),
    ],
)
def test_consumer_rejects_self_consistent_but_semantically_failed_proofs(
    tmp_path: Path,
    filename: str,
    document: dict[str, object],
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={filename: _canonical_bytes(document)},
    )

    with pytest.raises(ProviderServiceBindingError, match="semantic"):
        _load(receipt_path)


@pytest.mark.parametrize(
    ("path", "value"),
    (
        (("vector_store", "provider"), "qdrant"),
        (("vector_store", "config", "dbname"), "mem0_app"),
        (("vector_store", "config", "collection_name"), "other"),
        (("vector_store", "config", "embedding_model_dims"), 768),
        (("vector_store", "config", "hnsw"), False),
        (("vector_store", "config", "diskann"), True),
        (("vector_store", "config", "url"), "http://mem0-qdrant:6333"),
    ),
)
def test_consumer_rejects_wrong_mem0_pgvector_proof(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    filename = "mem0-config-redacted.json"
    document = json.loads(_proof_bytes(filename))
    target = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={filename: _canonical_bytes(document)},
    )

    with pytest.raises(ProviderServiceBindingError, match="semantic"):
        _load(receipt_path)


@pytest.mark.parametrize(
    "filename",
    (
        "hindsight-model-config.json",
        "mem0-config-redacted.json",
        "openviking-model-config.json",
    ),
)
def test_consumer_rejects_wrong_model_proofs(
    tmp_path: Path,
    filename: str,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    document = json.loads(_proof_bytes(filename))
    if filename == "mem0-config-redacted.json":
        document["llm"]["config"]["model"] = "different-model"
    else:
        document["model"] = "different-model"
    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={filename: _canonical_bytes(document)},
    )

    with pytest.raises(ProviderServiceBindingError, match="semantic"):
        _load(receipt_path)


def test_consumer_uses_plan_expected_provider_thinking_effort(tmp_path: Path) -> None:
    filename = "mem0-config-redacted.json"
    document = json.loads(_proof_bytes(filename))
    document["llm"]["config"]["reasoning_effort"] = "high"
    receipt_path = _write_service_artifacts(
        tmp_path,
        payload_overrides={filename: _canonical_bytes(document)},
    )
    expected_efforts = {**EXPECTED_PROVIDER_THINKING_EFFORTS, "mem0-rest-v1": "high"}

    bindings = _load(
        receipt_path,
        expected_provider_thinking_efforts=expected_efforts,
    )

    assert len(bindings) == 3


@pytest.mark.parametrize(
    "case",
    (
        "wrong_schema",
        "unknown_profile",
        "duplicate_profile",
        "wrong_project",
        "invalid_project",
        "wrong_attestation_hash",
        "failed_liveness",
        "promoted_model_readiness",
        "promoted_memory_conformance",
        "invalid_manifest_hash",
        "missing_manifest",
        "invalid_attestation_hash",
        "extra_top_level_field",
        "missing_top_level_field",
        "extra_profile_field",
        "missing_profile_field",
    ),
)
def test_receipt_rejects_invalid_identity_gate_hash_and_shape(
    tmp_path: Path,
    case: str,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    original_path = _write_service_artifacts(tmp_path)
    receipt = deepcopy(_read_receipt(original_path))
    expected_project = PROJECT
    expected_attestation = HASH_A
    profiles = receipt["profiles"]
    assert isinstance(profiles, list)
    first_profile = profiles[0]
    assert isinstance(first_profile, dict)

    if case == "wrong_schema":
        receipt["schema_version"] = 2
    elif case == "unknown_profile":
        first_profile["profile_id"] = "unknown-rest-v1"
    elif case == "duplicate_profile":
        first_profile["profile_id"] = "mem0-rest-v1"
    elif case == "wrong_project":
        receipt["provider_project"] = "oamb-providers-other-alpha"
    elif case == "invalid_project":
        receipt["provider_project"] = "not-a-provider-project"
        expected_project = "not-a-provider-project"
    elif case == "wrong_attestation_hash":
        receipt["project_attestation_sha256"] = "9" * 64
    elif case == "failed_liveness":
        first_profile["liveness"] = "failed"
    elif case == "promoted_model_readiness":
        first_profile["model_readiness"] = "passed"
    elif case == "promoted_memory_conformance":
        first_profile["memory_conformance"] = "passed"
    elif case == "invalid_manifest_hash":
        first_profile["proof_manifest_sha256"] = "not-a-sha256"
    elif case == "missing_manifest":
        first_profile["proof_manifest_sha256"] = "9" * 64
    elif case == "invalid_attestation_hash":
        receipt["project_attestation_sha256"] = "not-a-sha256"
        expected_attestation = "not-a-sha256"
    elif case == "extra_top_level_field":
        receipt["secret"] = "must-not-be-accepted"
    elif case == "missing_top_level_field":
        del receipt["verified_at_utc"]
    elif case == "extra_profile_field":
        first_profile["release"] = "floating"
    elif case == "missing_profile_field":
        del first_profile["runtime_identity"]
    else:
        raise AssertionError(f"unhandled test case: {case}")

    receipt_path = _write_receipt(original_path.parent, receipt)
    with pytest.raises(ProviderServiceBindingError):
        from oamb.config.provider_services import load_provider_service_bindings

        load_provider_service_bindings(
            receipt_path,
            expected_project=expected_project,
            expected_project_attestation_sha256=expected_attestation,
            controlled_embeddings=_controlled_embeddings(),
            expected_provider_models=EXPECTED_PROVIDER_MODELS,
            expected_provider_thinking_efforts=EXPECTED_PROVIDER_THINKING_EFFORTS,
        )


def test_receipt_rejects_a_filename_that_does_not_match_content(tmp_path: Path) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    valid_path = _write_service_artifacts(tmp_path)
    wrong_path = valid_path.parent / f"{'0' * 64}.json"
    valid_path.rename(wrong_path)

    with pytest.raises(ProviderServiceBindingError, match="content hash"):
        _load(wrong_path)


def test_receipt_rejects_noncanonical_json_even_when_its_filename_matches(
    tmp_path: Path,
) -> None:
    from oamb.config.provider_services import ProviderServiceBindingError

    valid_path = _write_service_artifacts(tmp_path)
    content = json.dumps(_read_receipt(valid_path), indent=2).encode("utf-8")
    path = valid_path.parent / f"{hashlib.sha256(content).hexdigest()}.json"
    path.write_bytes(content)

    with pytest.raises(ProviderServiceBindingError, match="canonical"):
        _load(path)
