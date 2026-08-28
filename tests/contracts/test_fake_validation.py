from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from oamb.artifacts.store import ArtifactStore
from oamb.artifacts.validation.fake import fake_evidence_registry, validate_fake_capsule
from oamb.artifacts.validation.profiles import fake_evidence_profile, fake_export_profile
from oamb.cli import run_generated_fake_vertical_slice
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.reporting import ReportArtifactManifest, RunReportModelV2
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.html import build_fake_report
from oamb.reporting.renderer import load_fake_report_css, render_fake_report_html
from oamb.reporting.validation import (
    FAKE_EXPORT_REQUIRED_PAYLOADS,
    fake_export_registry,
    validate_fake_export,
)
from oamb.runtime.fake_run import FakeRunScenario


def _rewrite_source_contract(
    capsule_root: Path,
    relative_path: str,
    mutate: Callable[[dict[str, object]], None],
) -> None:
    path = capsule_root / relative_path
    document = json.loads(path.read_bytes())
    assert isinstance(document, dict)
    mutate(document)
    path.write_bytes(canonical_json_bytes(document))
    _reseal_manifest(capsule_root, relative_path)


def _reseal_manifest(capsule_root: Path, changed_path: str) -> None:
    manifest_path = capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert isinstance(manifest, dict)
    entries = manifest["source_entries"]
    assert isinstance(entries, list)
    for entry in entries:
        assert isinstance(entry, dict)
        if entry["relative_path"] == changed_path:
            entry["sha256"] = hashlib.sha256((capsule_root / changed_path).read_bytes()).hexdigest()
    manifest["source_manifest_hash"] = canonical_sha256(["oamb-source-manifest-v1", tuple(entries)])
    manifest["capsule_id"] = canonical_sha256(
        [
            "oamb-capsule-v1",
            manifest["run_id"],
            manifest["run_spec_hash"],
            manifest["source_manifest_hash"],
        ]
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def _reseal_manifest_identity(capsule_root: Path) -> None:
    manifest_path = capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert isinstance(manifest, dict)
    entries = manifest["source_entries"]
    assert isinstance(entries, list)
    manifest["source_manifest_hash"] = canonical_sha256(["oamb-source-manifest-v1", tuple(entries)])
    manifest["capsule_id"] = canonical_sha256(
        [
            "oamb-capsule-v1",
            manifest["run_id"],
            manifest["run_spec_hash"],
            manifest["source_manifest_hash"],
        ]
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))


def _add_source_document(
    capsule_root: Path,
    relative_path: str,
    document: dict[str, object],
) -> None:
    path = capsule_root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    content = canonical_json_bytes(document)
    path.write_bytes(content)
    manifest_path = capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["source_entries"].append(
        {
            "schema_name": "capsule_manifest_entry",
            "schema_version": 1,
            "record_kind": document["schema_name"],
            "record_id": path.stem,
            "relative_path": relative_path,
            "sha256": hashlib.sha256(content).hexdigest(),
        }
    )
    manifest["source_entries"] = sorted(
        manifest["source_entries"], key=lambda entry: entry["relative_path"]
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(capsule_root)


def _replace_string(value: object, old: str, new: str) -> object:
    if isinstance(value, dict):
        return {key: _replace_string(child, old, new) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_string(child, old, new) for child in value]
    return new if value == old else value


def _rewrite_raw_document(
    capsule_root: Path,
    select: Callable[[dict[str, object]], bool],
    mutate: Callable[[dict[str, object]], None],
) -> tuple[str, str]:
    raw_path: Path | None = None
    document: dict[str, object] | None = None
    for candidate in sorted((capsule_root / "source" / "raw").glob("*.json.gz")):
        candidate_document = json.loads(gzip.decompress(candidate.read_bytes()))
        if isinstance(candidate_document, dict) and select(candidate_document):
            raw_path = candidate
            document = candidate_document
            break
    assert raw_path is not None and document is not None
    old_id = raw_path.name.removesuffix(".json.gz")
    mutate(document)
    payload = canonical_json_bytes(document)
    new_id = hashlib.sha256(payload).hexdigest()
    new_path = raw_path.with_name(f"{new_id}.json.gz")
    raw_path.rename(capsule_root.parent / f"{old_id}.removed")
    new_path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))

    for source_path in sorted((capsule_root / "source").rglob("*.json")):
        source_document = json.loads(source_path.read_bytes())
        rewritten = _replace_string(source_document, old_id, new_id)
        if rewritten != source_document:
            source_path.write_bytes(canonical_json_bytes(rewritten))

    manifest_path = capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for entry in manifest["source_entries"]:
        if entry["record_id"] == old_id and entry["relative_path"].startswith("source/raw/"):
            entry["record_id"] = new_id
            entry["relative_path"] = new_path.relative_to(capsule_root).as_posix()
            entry["sha256"] = hashlib.sha256(new_path.read_bytes()).hexdigest()
        elif entry["relative_path"].endswith(".json"):
            entry["sha256"] = hashlib.sha256(
                (capsule_root / entry["relative_path"]).read_bytes()
            ).hexdigest()
    manifest["source_entries"] = sorted(
        manifest["source_entries"], key=lambda entry: entry["relative_path"]
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(capsule_root)
    return old_id, new_id


def _replace_source_references(capsule_root: Path, old: str, new: str) -> None:
    for source_path in sorted((capsule_root / "source").rglob("*.json")):
        source_document = json.loads(source_path.read_bytes())
        rewritten = _replace_string(source_document, old, new)
        if rewritten != source_document:
            source_path.write_bytes(canonical_json_bytes(rewritten))
    manifest_path = capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for entry in manifest["source_entries"]:
        relative_path = entry["relative_path"]
        if relative_path.endswith(".json"):
            entry["sha256"] = hashlib.sha256(
                (capsule_root / relative_path).read_bytes()
            ).hexdigest()
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(capsule_root)


def _add_raw_payload(capsule_root: Path, document: dict[str, object]) -> str:
    stored = ArtifactStore(capsule_root).write_raw(
        canonical_json_bytes(document),
        media_type="application/json",
    )
    relative_path = stored.path.relative_to(capsule_root).as_posix()
    manifest_path = capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["source_entries"].append(
        {
            "schema_name": "capsule_manifest_entry",
            "schema_version": 1,
            "record_kind": "raw_payload",
            "record_id": stored.reference.sha256,
            "relative_path": relative_path,
            "sha256": hashlib.sha256(stored.path.read_bytes()).hexdigest(),
        }
    )
    manifest["source_entries"] = sorted(
        manifest["source_entries"], key=lambda entry: entry["relative_path"]
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(capsule_root)
    return stored.reference.sha256


def _remove_source_documents(capsule_root: Path, paths: list[Path]) -> None:
    relative_paths = {path.relative_to(capsule_root).as_posix() for path in paths}
    removed_root = capsule_root.parent / f"{capsule_root.name}-removed-source"
    removed_root.mkdir(parents=True, exist_ok=True)
    for ordinal, path in enumerate(paths, start=1):
        path.rename(removed_root / f"{ordinal}-{path.name}")
    manifest_path = capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["source_entries"] = [
        entry
        for entry in manifest["source_entries"]
        if entry["relative_path"] not in relative_paths
    ]
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(capsule_root)


def _first_source_path(capsule_root: Path, collection: str) -> str:
    return (
        next((capsule_root / "source" / collection).glob("*.json"))
        .relative_to(capsule_root)
        .as_posix()
    )


@pytest.mark.parametrize(
    ("expected_rule", "plant"),
    (
        (
            "fake.schema.v1",
            lambda root: _rewrite_source_contract(
                root,
                _first_source_path(root, "run"),
                lambda document: document.__setitem__("unknown", "planted"),
            ),
        ),
        (
            "fake.manifest-closure.v1",
            lambda root: (root / _first_source_path(root, "run")).write_bytes(b"{}"),
        ),
        (
            "fake.parentage.v1",
            lambda root: _rewrite_source_contract(
                root,
                _first_source_path(root, "cases"),
                lambda document: document.__setitem__("ingestion_occurrence_id", "f" * 64),
            ),
        ),
        (
            "fake.execution.v1",
            lambda root: _rewrite_source_contract(
                root,
                _first_source_path(root, "cases"),
                lambda document: document.__setitem__("state", "retrieving"),
            ),
        ),
        (
            "fake.raw-reference.v1",
            lambda root: _rewrite_source_contract(
                root,
                _first_source_path(root, "cases"),
                lambda document: document.__setitem__("retrieval_raw_ref", "f" * 64),
            ),
        ),
        (
            "fake.index-accounting.v1",
            lambda root: _plant_duplicate_plan_usage(root),
        ),
    ),
)
def test_each_fake_evidence_rule_has_an_independent_planted_failure(
    tmp_path: Path,
    expected_rule: str,
    plant: Callable[[Path], object],
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / expected_rule,
        run_id="fake-rule-negative",
    )
    plant(completed.capsule_root)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert expected_rule in result.failed_rule_ids


def test_fake_evidence_profile_fails_closed_for_missing_rule_or_inventory_drift(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-profile-closure",
    )
    missing = validate_fake_capsule(
        completed.capsule_root,
        registry=fake_evidence_registry(exclude={"fake.execution.v1"}),
    )
    profile = fake_evidence_profile().model_copy(update={"required_rule_inventory_hash": "0" * 64})
    drifted = validate_fake_capsule(completed.capsule_root, profile=profile)

    assert missing.disposition == ValidationDisposition.INVALID
    assert missing.missing_rule_ids == ("fake.execution.v1",)
    assert drifted.disposition == ValidationDisposition.INVALID
    assert drifted.failed_rule_ids == ("validation.profile-inventory.v1",)


def test_fake_manifest_rejects_source_symlink_even_when_target_bytes_match(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-source-symlink",
    )
    source_path = completed.capsule_root / _first_source_path(completed.capsule_root, "run")
    external_copy = tmp_path / "external-run.json"
    source_path.rename(external_copy)
    source_path.symlink_to(external_copy)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.manifest-closure.v1" in result.failed_rule_ids


def test_fake_schema_rejects_manifest_record_kind_drift(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-record-kind-drift",
    )
    manifest_path = completed.capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    assert isinstance(manifest, dict)
    entries = manifest["source_entries"]
    assert isinstance(entries, list)
    case_entry = next(
        entry
        for entry in entries
        if isinstance(entry, dict) and str(entry["relative_path"]).startswith("source/cases/")
    )
    case_entry["record_kind"] = "logical_context_record"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(completed.capsule_root)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.schema.v1" in result.failed_rule_ids


def test_fake_raw_rule_rejects_payload_whose_content_hash_no_longer_matches_reference(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-raw-identity",
    )
    raw_path = next((completed.capsule_root / "source" / "raw").glob("*.json.gz"))
    original = gzip.decompress(raw_path.read_bytes())
    raw_path.write_bytes(gzip.compress(original + b" ", mtime=0))
    _reseal_manifest(
        completed.capsule_root,
        raw_path.relative_to(completed.capsule_root).as_posix(),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.raw-reference.v1" in result.failed_rule_ids


def test_fake_raw_rule_rejects_unreferenced_payload(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-orphan-raw",
    )
    stored = ArtifactStore(completed.capsule_root).write_raw(
        b'{"operation":"unreferenced"}',
        media_type="application/json",
    )
    manifest_path = completed.capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    relative_path = stored.path.relative_to(completed.capsule_root).as_posix()
    manifest["source_entries"].append(
        {
            "schema_name": "capsule_manifest_entry",
            "schema_version": 1,
            "record_kind": "raw_payload",
            "record_id": stored.reference.sha256,
            "relative_path": relative_path,
            "sha256": hashlib.sha256(stored.path.read_bytes()).hexdigest(),
        }
    )
    manifest["source_entries"] = sorted(
        manifest["source_entries"], key=lambda entry: entry["relative_path"]
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(completed.capsule_root)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.raw-reference.v1" in result.failed_rule_ids


@pytest.mark.parametrize(
    "mismatch",
    ("ingest", "readiness", "retrieval", "model_output", "model_usage", "evaluation"),
)
def test_fake_raw_rule_rejects_raw_to_normalized_mapping_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=f"fake-raw-mapping-{mismatch}",
    )
    selectors: dict[str, Callable[[dict[str, object]], bool]] = {
        "ingest": lambda document: document.get("operation") == "ingest",
        "readiness": lambda document: (
            document.get("operation") == "wait_ready" and document.get("ready") is True
        ),
        "retrieval": lambda document: document.get("operation") == "retrieve",
        "model_output": lambda document: (
            document.get("outcome") == "success" and document.get("output_text") == "alpha"
        ),
        "model_usage": lambda document: (
            document.get("outcome") == "success" and document.get("output_text") == "beta"
        ),
        "evaluation": lambda document: document.get("metric_id") == "fake-exact-v1",
    }

    def mutate(document: dict[str, object]) -> None:
        if mismatch == "ingest":
            document["accepted_source_unit_ids"] = []
        elif mismatch == "readiness":
            document["ready"] = False
        elif mismatch == "retrieval":
            candidates = document["candidates"]
            assert isinstance(candidates, list)
            document["candidates"] = list(reversed(candidates))
        elif mismatch == "model_output":
            document["output_text"] = "wrong"
        elif mismatch == "model_usage":
            usage = document["usage"]
            assert isinstance(usage, dict)
            usage["input_tokens"] = int(usage["input_tokens"]) + 1
        else:
            document["result_sha256"] = "f" * 64

    _rewrite_raw_document(completed.capsule_root, selectors[mismatch], mutate)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.raw-reference.v1" in result.failed_rule_ids


@pytest.mark.parametrize("operation", ("ingest", "wait_ready"))
def test_fake_raw_rule_rejects_same_count_wrong_source_unit_ids(
    tmp_path: Path,
    operation: str,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=f"fake-wrong-{operation}-source-ids",
    )

    def mutate(document: dict[str, object]) -> None:
        field = "accepted_source_unit_ids" if operation == "ingest" else "expected_source_unit_ids"
        source_ids = document[field]
        assert isinstance(source_ids, list)
        document[field] = ["f" * 64 for _source_id in source_ids]

    _rewrite_raw_document(
        completed.capsule_root,
        lambda document: document.get("operation") == operation,
        mutate,
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.raw-reference.v1" in result.failed_rule_ids


def test_fake_raw_rule_binds_retrieval_to_the_case(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-case-prompt-chain",
    )
    answered_paths = [
        path
        for path in (completed.capsule_root / "source" / "cases").glob("*.json")
        if json.loads(path.read_bytes())["retrieval_raw_ref"] is not None
    ]
    assert len(answered_paths) >= 2
    other_case = json.loads(answered_paths[1].read_bytes())
    first_relative = answered_paths[0].relative_to(completed.capsule_root).as_posix()
    _rewrite_source_contract(
        completed.capsule_root,
        first_relative,
        lambda document: document.__setitem__(
            "retrieval_raw_ref",
            other_case["retrieval_raw_ref"],
        ),
    )

    retrieval_result = validate_fake_capsule(completed.capsule_root)

    assert retrieval_result.disposition == ValidationDisposition.INVALID
    assert "fake.raw-reference.v1" in retrieval_result.failed_rule_ids


def test_fake_raw_rule_binds_prompt_to_the_model_request(tmp_path: Path) -> None:
    prompt_completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "prompt-capsules",
        run_id="fake-case-prompt-hash",
    )
    prompt_case_path = next(
        path
        for path in (prompt_completed.capsule_root / "source" / "cases").glob("*.json")
        if json.loads(path.read_bytes())["prompt_sha256"] is not None
    )
    _rewrite_source_contract(
        prompt_completed.capsule_root,
        prompt_case_path.relative_to(prompt_completed.capsule_root).as_posix(),
        lambda document: document.__setitem__("prompt_sha256", "f" * 64),
    )

    prompt_result = validate_fake_capsule(prompt_completed.capsule_root)

    assert prompt_result.disposition == ValidationDisposition.INVALID
    assert "fake.raw-reference.v1" in prompt_result.failed_rule_ids


def test_fake_parentage_rejects_usage_attached_to_the_wrong_case(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-usage-parentage",
    )
    usage_path = _first_source_path(completed.capsule_root, "usage")
    _rewrite_source_contract(
        completed.capsule_root,
        usage_path,
        lambda document: document.__setitem__("parent_id", "f" * 64),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


def test_fake_parentage_binds_run_spec_bytes_to_the_run_record(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-run-spec-binding",
    )
    candidates = sorted((completed.capsule_root / "source" / "specs").glob("*.json"))
    run_spec_path = next(
        path.relative_to(completed.capsule_root).as_posix()
        for path in candidates
        if json.loads(path.read_bytes())["schema_name"] == "run_spec"
    )
    _rewrite_source_contract(
        completed.capsule_root,
        run_spec_path,
        lambda document: document.__setitem__("protocol_id", "drifted-protocol"),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


def test_fake_parentage_rejects_nonzero_or_nonfake_execution_budget(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-budget-boundary",
    )
    budget_path = next(
        path.relative_to(completed.capsule_root).as_posix()
        for path in sorted((completed.capsule_root / "source" / "specs").glob("*.json"))
        if json.loads(path.read_bytes())["schema_name"] == "budget_spec"
    )
    _rewrite_source_contract(
        completed.capsule_root,
        budget_path,
        lambda document: document.__setitem__("max_attempts", 1),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


@pytest.mark.parametrize("extra_kind", ("origin", "attempt", "foreign_close_error"))
def test_fake_parentage_rejects_extra_or_foreign_source_records(
    tmp_path: Path,
    extra_kind: str,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=f"fake-extra-{extra_kind}",
    )
    contracts = {
        path.parent.name: json.loads(path.read_bytes())
        for path in completed.capsule_root.glob("source/*/*.json")
    }
    if extra_kind == "origin":
        original = contracts["origins"]
        document = original | {"origin_id": "extra-origin"}
        relative_path = "source/origins/extra-origin.json"
    elif extra_kind == "attempt":
        original = contracts["attempts"]
        document = original | {"attempt_id": "f" * 64}
        relative_path = f"source/attempts/{'f' * 64}.json"
    else:
        raw_id = next(
            path.name.removesuffix(".json.gz")
            for path in (completed.capsule_root / "source" / "raw").glob("*.json.gz")
        )
        document = {
            "schema_name": "close_error_record",
            "schema_version": 1,
            "close_error_id": "e" * 64,
            "owner_kind": "run",
            "owner_id": "foreign-run",
            "client_profile_id": "fake-model",
            "error_ref": raw_id,
            "shutdown_stage": "client_close",
            "occurred_at": "2026-01-01T00:00:01Z",
        }
        relative_path = f"source/close-errors/{'e' * 64}.json"
    _add_source_document(completed.capsule_root, relative_path, document)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


def test_fake_parentage_rejects_a_duplicate_logical_context_source_record(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-duplicate-logical-context",
    )
    context_path = next((completed.capsule_root / "source" / "logical-contexts").glob("*.json"))
    context_document = json.loads(context_path.read_bytes())
    context_id = context_document["context_manifest_entry_id"]
    _add_source_document(
        completed.capsule_root,
        f"source/duplicate-logical-contexts/{context_id}.json",
        context_document,
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


def test_fake_parentage_binds_logical_context_to_its_ordered_cases(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-logical-case-membership",
    )
    context_path = _first_source_path(completed.capsule_root, "logical-contexts")
    _rewrite_source_contract(
        completed.capsule_root,
        context_path,
        lambda document: document.__setitem__("ordered_case_manifest_entry_ids", []),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


@pytest.mark.parametrize("field", ("ingestion_occurrence_ids", "case_occurrence_ids"))
def test_fake_parentage_rejects_duplicate_run_occurrence_references(
    tmp_path: Path,
    field: str,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=f"fake-duplicate-{field}",
    )
    run_path = _first_source_path(completed.capsule_root, "run")

    def duplicate_first_reference(document: dict[str, object]) -> None:
        references = document[field]
        assert isinstance(references, list)
        document[field] = [*references, references[0]]

    _rewrite_source_contract(
        completed.capsule_root,
        run_path,
        duplicate_first_reference,
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


def test_fake_index_accounting_rejects_missing_billed_retry_usage(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-missing-retry-usage",
    )
    attempt_documents = tuple(
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source" / "attempts").glob("*.json")
    )
    failed_answer = next(
        document
        for document in attempt_documents
        if document["stage"] == "answer" and document["outcome"] == "failed"
    )
    usage_path = next(
        path
        for path in (completed.capsule_root / "source" / "usage").glob("*.json")
        if json.loads(path.read_bytes())["attempt_id"] == failed_answer["attempt_id"]
    )
    relative_path = usage_path.relative_to(completed.capsule_root).as_posix()
    usage_path.rename(tmp_path / usage_path.name)
    manifest_path = completed.capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["source_entries"] = [
        entry for entry in manifest["source_entries"] if entry["relative_path"] != relative_path
    ]
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(completed.capsule_root)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.index-accounting.v1" in result.failed_rule_ids


def test_fake_index_accounting_rejects_contribution_tampering(
    tmp_path: Path,
) -> None:
    contribution = run_generated_fake_vertical_slice(
        output_root=tmp_path / "contribution",
        run_id="fake-index-contribution",
    )
    sealed_plan = next(
        json.loads(path.read_bytes())
        for path in (contribution.capsule_root / "source" / "ingestion-plans").glob("*.json")
        if json.loads(path.read_bytes())["state"] == "sealed"
    )
    ingest_attempt_path = next(
        path
        for path in (contribution.capsule_root / "source" / "attempts").glob("*.json")
        if json.loads(path.read_bytes())["attempt_id"] == sealed_plan["attempt_ids"][0]
    )
    _rewrite_source_contract(
        contribution.capsule_root,
        ingest_attempt_path.relative_to(contribution.capsule_root).as_posix(),
        lambda document: document.__setitem__("index_contribution", "none"),
    )

    contribution_result = validate_fake_capsule(contribution.capsule_root)

    assert contribution_result.disposition == ValidationDisposition.INVALID
    assert "fake.index-accounting.v1" in contribution_result.failed_rule_ids


def test_fake_index_accounting_rejects_ingest_meter_tampering(tmp_path: Path) -> None:
    metering = run_generated_fake_vertical_slice(
        output_root=tmp_path / "metering",
        run_id="fake-ingest-metering",
    )
    usage_path = next(
        path
        for path in (metering.capsule_root / "source" / "usage").glob("*.json")
        if json.loads(path.read_bytes())["stage"] == "memory_ingest"
    )

    def inflate_meter(document: dict[str, object]) -> None:
        input_tokens = document["input_tokens"]
        total_tokens = document["supplier_reported_total_tokens"]
        assert isinstance(input_tokens, int)
        assert isinstance(total_tokens, int)
        document["input_tokens"] = input_tokens + 100
        document["supplier_reported_total_tokens"] = total_tokens + 100

    _rewrite_source_contract(
        metering.capsule_root,
        usage_path.relative_to(metering.capsule_root).as_posix(),
        inflate_meter,
    )

    metering_result = validate_fake_capsule(metering.capsule_root)

    assert metering_result.disposition == ValidationDisposition.INVALID
    assert "fake.index-accounting.v1" in metering_result.failed_rule_ids


def test_fake_index_accounting_recomputes_model_meter_from_request_and_output(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-coordinated-model-meter",
    )
    selected_attempt_id: str | None = None

    def select(document: dict[str, object]) -> bool:
        nonlocal selected_attempt_id
        if document.get("outcome") == "success" and document.get("output_text") == "alpha":
            attempt_id = document.get("attempt_id")
            assert isinstance(attempt_id, str)
            selected_attempt_id = attempt_id
            return True
        return False

    def inflate_raw_meter(document: dict[str, object]) -> None:
        usage = document["usage"]
        assert isinstance(usage, dict)
        input_tokens = usage["input_tokens"]
        assert isinstance(input_tokens, int)
        usage["input_tokens"] = input_tokens + 100

    _old_raw_id, new_raw_id = _rewrite_raw_document(
        completed.capsule_root,
        select,
        inflate_raw_meter,
    )
    assert selected_attempt_id is not None
    usage_path = next(
        path
        for path in (completed.capsule_root / "source" / "usage").glob("*.json")
        if json.loads(path.read_bytes())["attempt_id"] == selected_attempt_id
    )
    usage_document = json.loads(usage_path.read_bytes())
    input_tokens = usage_document["input_tokens"]
    output_tokens = usage_document["visible_output_tokens"]
    assert isinstance(input_tokens, int)
    assert isinstance(output_tokens, int)
    input_tokens += 100
    usage_document["input_tokens"] = input_tokens
    usage_document["supplier_reported_total_tokens"] = input_tokens + output_tokens
    new_usage_id = canonical_sha256(
        [
            "oamb-fake-model-usage-v1",
            selected_attempt_id,
            input_tokens,
            output_tokens,
            new_raw_id,
        ]
    )
    usage_document["usage_record_id"] = new_usage_id
    new_usage_path = usage_path.with_name(f"{new_usage_id}.json")
    usage_path.rename(new_usage_path)
    new_usage_path.write_bytes(canonical_json_bytes(usage_document))
    manifest_path = completed.capsule_root / "capsule-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    for entry in manifest["source_entries"]:
        if entry["relative_path"] == usage_path.relative_to(completed.capsule_root).as_posix():
            entry["record_id"] = new_usage_id
            entry["relative_path"] = new_usage_path.relative_to(completed.capsule_root).as_posix()
            entry["sha256"] = hashlib.sha256(new_usage_path.read_bytes()).hexdigest()
    manifest["source_entries"] = sorted(
        manifest["source_entries"], key=lambda entry: entry["relative_path"]
    )
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    _reseal_manifest_identity(completed.capsule_root)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.index-accounting.v1" in result.failed_rule_ids


def test_fake_parentage_rejects_answered_case_without_its_attempt_ledger(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-missing-case-attempt-ledger",
    )
    case_path = next(
        path
        for path in (completed.capsule_root / "source" / "cases").glob("*.json")
        if json.loads(path.read_bytes())["evaluation_disposition"] == "deterministic_evaluated"
    )
    case_document = json.loads(case_path.read_bytes())
    attempt_ids = set(case_document["attempt_ids"])
    source_paths = [
        path
        for collection in ("attempts", "usage")
        for path in (completed.capsule_root / "source" / collection).glob("*.json")
        if (
            json.loads(path.read_bytes()).get("attempt_id") in attempt_ids
            or json.loads(path.read_bytes()).get("usage_record_id") in attempt_ids
        )
    ]
    _remove_source_documents(completed.capsule_root, source_paths)
    _rewrite_source_contract(
        completed.capsule_root,
        case_path.relative_to(completed.capsule_root).as_posix(),
        lambda document: document.__setitem__("attempt_ids", []),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.execution.v1" in result.failed_rule_ids


def test_fake_parentage_requires_plan_usage_refs_in_cancelled_capsule(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-cancelled-plan-usage",
        scenario=FakeRunScenario.CANCELLED,
    )
    plan_path = next(
        path
        for path in (completed.capsule_root / "source" / "ingestion-plans").glob("*.json")
        if json.loads(path.read_bytes())["usage_record_ids"]
    )
    _rewrite_source_contract(
        completed.capsule_root,
        plan_path.relative_to(completed.capsule_root).as_posix(),
        lambda document: document.__setitem__("usage_record_ids", []),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


def test_fake_execution_rejects_plan_counts_without_an_ingest_attempt(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-plan-count-without-attempt",
        scenario=FakeRunScenario.CANCELLED,
    )
    plan_path = next(
        path
        for path in (completed.capsule_root / "source" / "ingestion-plans").glob("*.json")
        if not json.loads(path.read_bytes())["attempt_ids"]
    )

    def plant_unproven_counts(document: dict[str, object]) -> None:
        document["state"] = "error"
        document["accepted_source_count"] = 1

    _rewrite_source_contract(
        completed.capsule_root,
        plan_path.relative_to(completed.capsule_root).as_posix(),
        plant_unproven_counts,
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.execution.v1" in result.failed_rule_ids


def test_fake_parentage_rejects_unresolved_resource_and_cost_refs(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-resource-cost-refs",
    )
    plan_path = _first_source_path(completed.capsule_root, "ingestion-plans")

    def plant_refs(document: dict[str, object]) -> None:
        document["resource_record_ids"] = ["e" * 64]
        document["cost_record_ids"] = ["f" * 64]

    _rewrite_source_contract(completed.capsule_root, plan_path, plant_refs)

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


def test_fake_execution_rejects_success_with_both_response_and_error_raw(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-success-with-error-raw",
    )
    error_raw_id = _add_raw_payload(
        completed.capsule_root,
        {"error_type": "RuntimeError", "message": "planted impossible error"},
    )
    attempt_path = next(
        path
        for path in (completed.capsule_root / "source" / "attempts").glob("*.json")
        if json.loads(path.read_bytes())["stage"] == "answer"
        and json.loads(path.read_bytes())["outcome"] == "succeeded"
    )
    _rewrite_source_contract(
        completed.capsule_root,
        attempt_path.relative_to(completed.capsule_root).as_posix(),
        lambda document: document.__setitem__("raw_error_ref", error_raw_id),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.execution.v1" in result.failed_rule_ids


def test_fake_execution_rejects_a_retry_link_to_the_wrong_attempt(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-retry-edge",
    )
    retry_path = next(
        path.relative_to(completed.capsule_root).as_posix()
        for path in (completed.capsule_root / "source" / "attempts").glob("*.json")
        if json.loads(path.read_bytes())["retry_of_attempt_id"] is not None
    )
    _rewrite_source_contract(
        completed.capsule_root,
        retry_path,
        lambda document: document.__setitem__("retry_of_attempt_id", "f" * 64),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.execution.v1" in result.failed_rule_ids


def test_fake_execution_rejects_a_parent_referenced_unknown_attempt_stage(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-unknown-attempt-stage",
    )
    case_path = next((completed.capsule_root / "source" / "cases").glob("*.json"))
    case_document = json.loads(case_path.read_bytes())
    original_attempt = next(
        json.loads(path.read_bytes())
        for path in (completed.capsule_root / "source" / "attempts").glob("*.json")
        if json.loads(path.read_bytes())["parent_id"] == case_document["case_occurrence_id"]
    )
    planted_attempt_id = "f" * 64
    planted_attempt = original_attempt | {
        "attempt_id": planted_attempt_id,
        "stage": "unexpected_stage",
    }
    _add_source_document(
        completed.capsule_root,
        f"source/attempts/{planted_attempt_id}.json",
        planted_attempt,
    )
    relative_case_path = case_path.relative_to(completed.capsule_root).as_posix()

    def append_attempt_id(document: dict[str, object]) -> None:
        attempt_ids = document["attempt_ids"]
        assert isinstance(attempt_ids, list)
        document["attempt_ids"] = [*attempt_ids, planted_attempt_id]

    _rewrite_source_contract(
        completed.capsule_root,
        relative_case_path,
        append_attempt_id,
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.execution.v1" in result.failed_rule_ids


def test_fake_parentage_rejects_duplicate_attempt_edges(
    tmp_path: Path,
) -> None:
    duplicated = run_generated_fake_vertical_slice(
        output_root=tmp_path / "duplicated",
        run_id="fake-duplicate-attempt-edge",
    )
    case_path = next(
        path
        for path in (duplicated.capsule_root / "source" / "cases").glob("*.json")
        if json.loads(path.read_bytes())["attempt_ids"]
    )

    def duplicate_attempt_edge(document: dict[str, object]) -> None:
        attempt_ids = document["attempt_ids"]
        assert isinstance(attempt_ids, list)
        document["attempt_ids"] = [*attempt_ids, attempt_ids[0]]

    _rewrite_source_contract(
        duplicated.capsule_root,
        case_path.relative_to(duplicated.capsule_root).as_posix(),
        duplicate_attempt_edge,
    )

    duplicate_result = validate_fake_capsule(duplicated.capsule_root)

    assert duplicate_result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in duplicate_result.failed_rule_ids


def test_fake_parentage_rejects_noncanonical_attempt_identity(tmp_path: Path) -> None:
    identity = run_generated_fake_vertical_slice(
        output_root=tmp_path / "identity",
        run_id="fake-noncanonical-attempt-id",
    )
    attempt_path = next(
        path
        for path in (identity.capsule_root / "source" / "attempts").glob("*.json")
        if json.loads(path.read_bytes())["stage"] == "memory_ingest"
    )
    original_attempt_id = json.loads(attempt_path.read_bytes())["attempt_id"]
    _replace_source_references(identity.capsule_root, original_attempt_id, "f" * 64)

    identity_result = validate_fake_capsule(identity.capsule_root)

    assert identity_result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in identity_result.failed_rule_ids


def test_fake_parentage_rejects_multiple_close_errors_for_one_client(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-duplicate-close-client",
    )
    for ordinal in (1, 2):
        raw_id = _add_raw_payload(
            completed.capsule_root,
            {"error_type": "RuntimeError", "message": f"planted close {ordinal}"},
        )
        close_error_id = canonical_sha256(
            [
                "oamb-fake-close-error-v1",
                completed.run_record.run_id,
                "fake-model",
                raw_id,
            ]
        )
        _add_source_document(
            completed.capsule_root,
            f"source/close-errors/{close_error_id}.json",
            {
                "schema_name": "close_error_record",
                "schema_version": 1,
                "close_error_id": close_error_id,
                "owner_kind": "run",
                "owner_id": completed.run_record.run_id,
                "client_profile_id": "fake-model",
                "error_ref": raw_id,
                "shutdown_stage": "client_close",
                "occurred_at": f"2026-01-01T00:00:0{ordinal}Z",
            },
        )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.parentage.v1" in result.failed_rule_ids


@pytest.mark.parametrize("broken_edge", ("plan_state", "resume_disposition"))
def test_fake_execution_rejects_broken_unknown_outcome_propagation(
    tmp_path: Path,
    broken_edge: str,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id=f"fake-unknown-{broken_edge}",
        scenario=FakeRunScenario.UNKNOWN_OUTCOME,
    )
    if broken_edge == "plan_state":
        path = next(
            candidate.relative_to(completed.capsule_root).as_posix()
            for candidate in (completed.capsule_root / "source" / "ingestion-plans").glob("*.json")
            if json.loads(candidate.read_bytes())["state"] == "interrupted_unknown_outcome"
        )
        field = "state"
        value = "error"
    else:
        path = _first_source_path(completed.capsule_root, "run")
        field = "resume_disposition"
        value = "not_applicable"
    _rewrite_source_contract(
        completed.capsule_root,
        path,
        lambda document: document.__setitem__(field, value),
    )

    result = validate_fake_capsule(completed.capsule_root)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.execution.v1" in result.failed_rule_ids


def _plant_duplicate_plan_usage(capsule_root: Path) -> None:
    paths = sorted((capsule_root / "source" / "ingestion-plans").glob("*.json"))
    first = json.loads(paths[0].read_bytes())
    second_relative = paths[1].relative_to(capsule_root).as_posix()
    _rewrite_source_contract(
        capsule_root,
        second_relative,
        lambda document: document.__setitem__("usage_record_ids", first["usage_record_ids"]),
    )


@pytest.mark.parametrize(
    ("expected_rule", "mutate"),
    (
        (
            "fake.export.payload-closure.v1",
            lambda payloads: payloads.pop("input-specs/report-spec.json"),
        ),
        (
            "fake.export.safe-html.v1",
            lambda payloads: payloads.__setitem__(
                "outputs/report.html",
                payloads["outputs/report.html"]
                + b'<img src="x" onerror="alert(1)"><script>bad()</script>',
            ),
        ),
        (
            "fake.export.report-binding.v1",
            lambda payloads: payloads.__setitem__(
                "outputs/report-model.json",
                payloads["outputs/report-model.json"].replace(
                    b"generated fake evidence",
                    b"changed fake evidence",
                ),
            ),
        ),
    ),
)
def test_each_fake_export_rule_has_an_independent_planted_failure(
    tmp_path: Path,
    expected_rule: str,
    mutate: Callable[[dict[str, bytes]], object],
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-export",
    )
    evidence = validate_fake_capsule(completed.capsule_root)
    report = build_fake_report(
        completed.capsule_root,
        evidence,
        output_root=tmp_path / "reports",
        audience="public",
    )
    payloads = {
        relative_path: (report.attempt_directory / relative_path).read_bytes()
        for relative_path in FAKE_EXPORT_REQUIRED_PAYLOADS
    }
    mutate(payloads)

    result = validate_fake_export(payloads)

    assert result.disposition == ValidationDisposition.INVALID
    assert expected_rule in result.failed_rule_ids


def test_fake_export_rejects_safe_visible_html_drift(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-visible-html-drift",
    )
    evidence = validate_fake_capsule(completed.capsule_root)
    report = build_fake_report(
        completed.capsule_root,
        evidence,
        output_root=tmp_path / "reports",
        audience="public",
    )
    payloads = {
        relative_path: (report.attempt_directory / relative_path).read_bytes()
        for relative_path in FAKE_EXPORT_REQUIRED_PAYLOADS
    }
    payloads["outputs/report.html"] = payloads["outputs/report.html"].replace(
        b"<dt>Completed cases</dt><dd>2</dd>",
        b"<dt>Completed cases</dt><dd>999</dd>",
    )

    result = validate_fake_export(payloads)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.export.report-binding.v1" in result.failed_rule_ids


def test_fake_export_recomputes_report_id_and_required_limitations(tmp_path: Path) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-report-claim-boundary",
    )
    evidence = validate_fake_capsule(completed.capsule_root)
    report = build_fake_report(
        completed.capsule_root,
        evidence,
        output_root=tmp_path / "reports",
        audience="public",
    )
    payloads = {
        relative_path: (report.attempt_directory / relative_path).read_bytes()
        for relative_path in FAKE_EXPORT_REQUIRED_PAYLOADS
    }
    model = RunReportModelV2.model_validate_json(payloads["outputs/report-model.json"])
    limitations = tuple(
        item
        for item in model.limitations
        if item != "synthetic usage is not a real supplier charge"
    )
    drifted_model = model.model_copy(update={"limitations": limitations})
    model_bytes = canonical_json_bytes(drifted_model)
    artifact = ReportArtifactManifest.model_validate_json(
        payloads["report-artifact-manifest.json"]
    ).model_copy(
        update={
            "report_model_hash": hashlib.sha256(model_bytes).hexdigest(),
            "limitations": limitations,
        }
    )
    payloads["outputs/report-model.json"] = model_bytes
    payloads["report-artifact-manifest.json"] = canonical_json_bytes(artifact)
    payloads["outputs/report.html"] = render_fake_report_html(
        drifted_model,
        load_fake_report_css(),
        diagnostic=False,
    )

    result = validate_fake_export(payloads)

    assert result.disposition == ValidationDisposition.INVALID
    assert "fake.export.report-binding.v1" in result.failed_rule_ids


def test_fake_export_profile_fails_closed_for_missing_rule_or_inventory_drift(
    tmp_path: Path,
) -> None:
    completed = run_generated_fake_vertical_slice(
        output_root=tmp_path / "capsules",
        run_id="fake-export-profile",
    )
    evidence = validate_fake_capsule(completed.capsule_root)
    report = build_fake_report(
        completed.capsule_root,
        evidence,
        output_root=tmp_path / "reports",
        audience="public",
    )
    payloads = {
        relative_path: (report.attempt_directory / relative_path).read_bytes()
        for relative_path in FAKE_EXPORT_REQUIRED_PAYLOADS
    }
    missing = validate_fake_export(
        payloads,
        registry=fake_export_registry(exclude={"fake.export.safe-html.v1"}),
    )
    profile = fake_export_profile().model_copy(update={"required_rule_inventory_hash": "0" * 64})
    drifted = validate_fake_export(payloads, profile=profile)

    assert missing.disposition == ValidationDisposition.INVALID
    assert missing.missing_rule_ids == ("fake.export.safe-html.v1",)
    assert drifted.disposition == ValidationDisposition.INVALID
    assert drifted.failed_rule_ids == ("validation.profile-inventory.v1",)
