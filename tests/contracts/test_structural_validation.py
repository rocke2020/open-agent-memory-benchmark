from __future__ import annotations

import hashlib
import importlib
import json
from copy import deepcopy
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError

FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "validation"


def require(module_name: str) -> ModuleType:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        pytest.fail(f"{module_name} is not implemented", pytrace=False)


def load_fixture(name: str) -> Any:
    core = require("oamb.artifacts.validation.core")
    payload = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    if "base_fixture" in payload:
        base = json.loads((FIXTURE_ROOT / payload["base_fixture"]).read_text(encoding="utf-8"))
        payload = apply_mutation(base, payload["mutation"])
    return core.StructuralValidationInput.model_validate_json(json.dumps(payload))


def apply_mutation(payload: dict[str, Any], mutation: dict[str, Any]) -> dict[str, Any]:
    mutated = deepcopy(payload)
    kind = mutation["kind"]
    if kind == "schema":
        mutated["documents"][0]["document"]["unknown_field"] = mutation["unknown_field"]
    elif kind == "identity":
        mutated["identity_claims"][0]["actual_id"] = mutation["actual_id"]
    elif kind == "reference":
        mutated["reference_claims"][0]["target_record_id"] = mutation["target_record_id"]
    elif kind == "count":
        mutated["count_claims"][0]["declared_count"] = mutation["declared_count"]
    elif kind == "hash":
        mutated["hash_claims"][0]["actual_sha256"] = mutation["actual_sha256"]
    elif kind == "transition":
        mutated["transition_claims"][0]["claimed_next_state"] = mutation["claimed_next_state"]
    elif kind == "control":
        mutated["control_field_claims"][0]["control_values"] = mutation["control_values"]
    else:
        raise AssertionError(f"unknown fixture mutation: {kind}")
    return mutated


def validate(name: str) -> Any:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    return core.validate_structural_input(
        load_fixture(name), profiles.structural_evidence_profile(), registry.structural_registry()
    )


def test_valid_structural_fixture_executes_closed_rule_inventory() -> None:
    result = validate("valid.json")

    assert result.disposition.value == "validated"
    assert result.failed_rule_ids == ()
    assert result.missing_rule_ids == ()
    assert result.required_rule_ids == result.executed_rule_ids


@pytest.mark.parametrize(
    ("fixture", "rule_id"),
    (
        ("invalid-schema.json", "structural.schema.v1"),
        ("invalid-identity.json", "structural.identity.v1"),
        ("invalid-reference.json", "structural.reference.v1"),
        ("invalid-count.json", "structural.count.v1"),
        ("invalid-hash.json", "structural.hash.v1"),
        ("invalid-transition.json", "structural.transition.v1"),
        ("invalid-control.json", "structural.control-value.v1"),
    ),
)
def test_each_early_rule_has_a_planted_failure(fixture: str, rule_id: str) -> None:
    result = validate(fixture)

    assert result.disposition.value == "invalid"
    assert result.failed_rule_ids == (rule_id,)
    assert {issue.rule_id for issue in result.issues} == {rule_id}
    assert "secret-token" not in json.dumps(result.model_dump(mode="json"))


def test_missing_and_incompatible_required_rules_fail_closed() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    specifications = require("oamb.contracts.specifications")

    profile = profiles.structural_evidence_profile()
    missing_registry = registry.structural_registry(exclude={"structural.hash.v1"})
    missing = core.validate_structural_input(load_fixture("valid.json"), profile, missing_registry)
    assert missing.disposition.value == "invalid"
    assert missing.missing_rule_ids == ("structural.hash.v1",)

    requirements = tuple(
        specifications.ValidationRuleRequirement(
            rule_id=item.rule_id,
            minimum_version=99 if item.rule_id == "structural.hash.v1" else item.minimum_version,
        )
        for item in profile.required_rules
    )
    incompatible_profile = specifications.ValidationProfile.create(
        profile_id="structural-evidence-incompatible-v1",
        stage=profile.stage,
        required_rules=requirements,
        applicability=("t3",),
    )
    incompatible = core.validate_structural_input(
        load_fixture("valid.json"), incompatible_profile, registry.structural_registry()
    )
    assert incompatible.disposition.value == "invalid"
    assert incompatible.missing_rule_ids == ("structural.hash.v1",)


def test_tampered_required_rule_inventory_hash_fails_before_rules_execute() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    profile = profiles.structural_evidence_profile().model_copy(
        update={"required_rule_inventory_hash": "0" * 64}
    )

    result = core.validate_structural_input(
        load_fixture("valid.json"), profile, registry.structural_registry()
    )

    assert result.disposition.value == "invalid"
    assert result.executed_rule_ids == ()
    assert result.failed_rule_ids == ("validation.profile-inventory.v1",)


def test_omitted_required_rule_and_empty_input_fail_before_rules_execute() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    specifications = require("oamb.contracts.specifications")
    profile = profiles.structural_evidence_profile()
    omitted_profile = specifications.ValidationProfile.create(
        profile_id="omitted-structural-rule-v1",
        stage=profile.stage,
        required_rules=tuple(
            requirement
            for requirement in profile.required_rules
            if requirement.rule_id != "structural.hash.v1"
        ),
        applicability=profile.applicability,
    )

    omitted = core.validate_structural_input(
        load_fixture("valid.json"), omitted_profile, registry.structural_registry()
    )
    assert omitted.disposition.value == "invalid"
    assert omitted.executed_rule_ids == ()
    assert omitted.failed_rule_ids == ("validation.profile-coverage.v1",)

    valid = load_fixture("valid.json")
    empty = valid.model_copy(
        update={
            "documents": (),
            "identity_claims": (),
            "reference_claims": (),
            "count_claims": (),
            "hash_claims": (),
            "transition_claims": (),
            "control_field_claims": (),
        }
    )
    empty_result = core.validate_structural_input(empty, profile, registry.structural_registry())
    assert empty_result.disposition.value == "invalid"
    assert empty_result.executed_rule_ids == ()
    assert empty_result.failed_rule_ids == ("validation.input-coverage.v1",)


def test_required_structural_rules_cannot_be_caller_declared_not_applicable() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    valid = load_fixture("valid.json")
    skipped_rule_ids = {
        "structural.identity.v1",
        "structural.reference.v1",
        "structural.count.v1",
        "structural.hash.v1",
        "structural.transition.v1",
        "structural.control-value.v1",
    }
    skipped = valid.model_copy(
        update={
            "identity_claims": (),
            "reference_claims": (),
            "count_claims": (),
            "hash_claims": (),
            "transition_claims": (),
            "control_field_claims": (),
            "rule_coverage_claims": tuple(
                claim
                if claim.rule_id not in skipped_rule_ids
                else claim.model_copy(
                    update={
                        "target_evidence_refs": (),
                        "not_applicable_reason": "caller-says-skip",
                    }
                )
                for claim in valid.rule_coverage_claims
            ),
        }
    )

    result = core.validate_structural_input(
        skipped, profiles.structural_evidence_profile(), registry.structural_registry()
    )

    assert result.disposition.value == "invalid"
    assert result.executed_rule_ids == ()
    assert result.failed_rule_ids == ("validation.input-coverage.v1",)


def test_artifact_kind_derives_required_identity_reference_and_hash_coverage() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    valid = load_fixture("valid.json")
    unchecked_context = core.ArtifactDocument(
        record_id="ctx-unchecked",
        record_kind="logical_context_manifest_entry",
        document={
            "schema_name": "logical_context_manifest_entry",
            "schema_version": 1,
            "context_content_id": "1" * 64,
            "context_manifest_entry_id": "2" * 64,
            "source_file_sha256": "3" * 64,
            "source_row_number_1_indexed": 1,
            "context_bytes_sha256": "4" * 64,
        },
    )
    coverage = tuple(
        claim.model_copy(
            update={"target_evidence_refs": (*claim.target_evidence_refs, "ctx-unchecked")}
        )
        if claim.rule_id == "structural.schema.v1"
        else claim
        for claim in valid.rule_coverage_claims
    )
    unchecked = valid.model_copy(
        update={
            "documents": (*valid.documents, unchecked_context),
            "rule_coverage_claims": coverage,
        }
    )

    result = core.validate_structural_input(
        unchecked, profiles.structural_evidence_profile(), registry.structural_registry()
    )

    assert result.disposition.value == "invalid"
    assert result.executed_rule_ids == ()
    assert result.failed_rule_ids == ("validation.input-coverage.v1",)


def test_identity_claim_cannot_choose_its_own_domain_for_a_known_artifact() -> None:
    core = require("oamb.artifacts.validation.core")
    ids = require("oamb.contracts.ids")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    valid = load_fixture("valid.json")
    dataset_hash = "5" * 64
    source_hash = "6" * 64
    content_id = "7" * 64
    identity_parts = (dataset_hash, source_hash, 1, content_id)
    self_certified_id = ids.sha256_identity("attacker-selected-domain", identity_parts)
    context_payload = {"payload": "context"}
    context_hash = ids.canonical_sha256(context_payload)
    dataset = core.ArtifactDocument(
        record_id="dataset-1",
        record_kind="dataset_manifest",
        document={
            "schema_name": "dataset_manifest",
            "schema_version": 1,
            "dataset_id": "dataset-1",
            "revision": "r1",
            "split": "test",
            "manifest_hash": dataset_hash,
            "source_files": [],
            "payload_policy": "fixture-only",
        },
    )
    context = core.ArtifactDocument(
        record_id="ctx-self-certified",
        record_kind="logical_context_manifest_entry",
        document={
            "schema_name": "logical_context_manifest_entry",
            "schema_version": 1,
            "context_content_id": content_id,
            "context_manifest_entry_id": self_certified_id,
            "source_file_sha256": source_hash,
            "source_row_number_1_indexed": 1,
            "context_bytes_sha256": context_hash,
        },
    )
    added_refs = {
        "structural.schema.v1": ("dataset-1", "ctx-self-certified"),
        "structural.identity.v1": ("ctx-self-certified",),
        "structural.reference.v1": ("ctx-self-certified",),
        "structural.hash.v1": ("ctx-self-certified",),
    }
    coverage = tuple(
        claim.model_copy(
            update={
                "target_evidence_refs": (
                    *claim.target_evidence_refs,
                    *added_refs.get(claim.rule_id, ()),
                )
            }
        )
        for claim in valid.rule_coverage_claims
    )
    self_certified = valid.model_copy(
        update={
            "documents": (*valid.documents, dataset, context),
            "identity_claims": (
                *valid.identity_claims,
                core.IdentityClaim(
                    evidence_ref="ctx-self-certified",
                    identity_prefix="attacker-selected-domain",
                    identity_parts=identity_parts,
                    actual_id=self_certified_id,
                ),
            ),
            "reference_claims": (
                *valid.reference_claims,
                core.ReferenceClaim(
                    evidence_ref="ctx-self-certified",
                    target_record_id="dataset-1",
                    expected_record_kind="dataset_manifest",
                ),
            ),
            "hash_claims": (
                *valid.hash_claims,
                core.HashClaim(
                    evidence_ref="ctx-self-certified",
                    value=context_payload,
                    actual_sha256=context_hash,
                ),
            ),
            "rule_coverage_claims": coverage,
        }
    )

    result = core.validate_structural_input(
        self_certified,
        profiles.structural_evidence_profile(),
        registry.structural_registry(),
    )

    assert result.disposition.value == "invalid"
    assert "structural.identity.v1" in result.failed_rule_ids


def test_logical_context_validates_both_id_layers_and_raw_byte_hash() -> None:
    core = require("oamb.artifacts.validation.core")
    ids = require("oamb.contracts.ids")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    valid = load_fixture("valid.json")
    dataset_hash = "5" * 64
    source_hash = "6" * 64
    raw_context = b"hello"
    context_hash = hashlib.sha256(raw_context).hexdigest()
    content_parts = ("r1", "test", "source.json", context_hash)
    content_id = ids.sha256_identity("oamb-context-content-v1", content_parts)
    manifest_parts = (dataset_hash, source_hash, 1, content_id)
    manifest_entry_id = ids.sha256_identity("oamb-context-manifest-entry-v1", manifest_parts)
    dataset = core.ArtifactDocument(
        record_id="dataset-1",
        record_kind="dataset_manifest",
        document={
            "schema_name": "dataset_manifest",
            "schema_version": 1,
            "dataset_id": "dataset-1",
            "revision": "r1",
            "split": "test",
            "manifest_hash": dataset_hash,
            "source_files": [
                {
                    "schema_name": "dataset_file",
                    "schema_version": 1,
                    "relative_path": "source.json",
                    "sha256": source_hash,
                    "byte_count": len(raw_context),
                    "license_id": "Apache-2.0",
                }
            ],
            "payload_policy": "fixture-only",
        },
    )
    context = core.ArtifactDocument(
        record_id="ctx-valid",
        record_kind="logical_context_manifest_entry",
        document={
            "schema_name": "logical_context_manifest_entry",
            "schema_version": 1,
            "context_content_id": content_id,
            "context_manifest_entry_id": manifest_entry_id,
            "source_file_sha256": source_hash,
            "source_row_number_1_indexed": 1,
            "context_bytes_sha256": context_hash,
        },
    )
    added_refs = {
        "structural.schema.v1": ("dataset-1", "ctx-valid"),
        "structural.identity.v1": ("ctx-valid",),
        "structural.reference.v1": ("ctx-valid",),
        "structural.hash.v1": ("ctx-valid",),
    }
    candidate = valid.model_copy(
        update={
            "documents": (*valid.documents, dataset, context),
            "identity_claims": (
                *valid.identity_claims,
                core.IdentityClaim(
                    evidence_ref="ctx-valid",
                    identity_prefix="oamb-context-content-v1",
                    identity_parts=content_parts,
                    actual_id=content_id,
                ),
                core.IdentityClaim(
                    evidence_ref="ctx-valid",
                    identity_prefix="oamb-context-manifest-entry-v1",
                    identity_parts=manifest_parts,
                    actual_id=manifest_entry_id,
                ),
            ),
            "reference_claims": (
                *valid.reference_claims,
                core.ReferenceClaim(
                    evidence_ref="ctx-valid",
                    target_record_id="dataset-1",
                    expected_record_kind="dataset_manifest",
                ),
            ),
            "hash_claims": (
                *valid.hash_claims,
                core.HashClaim(
                    evidence_ref="ctx-valid",
                    hash_kind=core.HashKind.RAW_BYTES,
                    value=raw_context,
                    actual_sha256=context_hash,
                ),
            ),
            "rule_coverage_claims": tuple(
                claim.model_copy(
                    update={
                        "target_evidence_refs": (
                            *claim.target_evidence_refs,
                            *added_refs.get(claim.rule_id, ()),
                        )
                    }
                )
                for claim in valid.rule_coverage_claims
            ),
        }
    )

    result = core.validate_structural_input(
        candidate, profiles.structural_evidence_profile(), registry.structural_registry()
    )

    assert result.disposition.value == "validated"

    truncated = candidate.model_copy(
        update={
            "identity_claims": tuple(
                claim.model_copy(update={"identity_parts": ("r1", "test")})
                if claim.evidence_ref == "ctx-valid"
                and claim.identity_prefix == "oamb-context-content-v1"
                else claim
                for claim in candidate.identity_claims
            )
        }
    )
    truncated_result = core.validate_structural_input(
        truncated, profiles.structural_evidence_profile(), registry.structural_registry()
    )
    assert truncated_result.disposition.value == "invalid"
    assert "structural.identity.v1" in truncated_result.failed_rule_ids


def test_schema_failure_short_circuits_rules_that_need_typed_fields() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    valid = load_fixture("valid.json")
    broken_case = core.ArtifactDocument(
        record_id="case-broken",
        record_kind="case_manifest_entry",
        document={
            "schema_name": "case_manifest_entry",
            "schema_version": 1,
            "case_manifest_entry_id": "1" * 64,
            "context_manifest_entry_id": "2" * 64,
            "source_question_number_1_indexed": 1,
            "question_bytes_sha256": "3" * 64,
            "answer_value_sha256": ["4" * 64],
        },
    )
    added_refs = {
        "structural.schema.v1": ("case-broken",),
        "structural.identity.v1": ("case-broken",),
        "structural.reference.v1": ("case-broken",),
        "structural.hash.v1": ("case-broken",),
    }
    candidate = valid.model_copy(
        update={
            "documents": (*valid.documents, broken_case),
            "identity_claims": (
                *valid.identity_claims,
                core.IdentityClaim(
                    evidence_ref="case-broken",
                    identity_prefix="oamb-case-manifest-entry-v1",
                    identity_parts=("2" * 64, 1, "3" * 64, "missing"),
                    actual_id="1" * 64,
                ),
            ),
            "reference_claims": (
                *valid.reference_claims,
                core.ReferenceClaim(
                    evidence_ref="case-broken",
                    target_record_id="protocol-1",
                    expected_record_kind="protocol_spec",
                ),
            ),
            "hash_claims": (
                *valid.hash_claims,
                core.HashClaim(
                    evidence_ref="case-broken",
                    value={"question": "missing"},
                    actual_sha256="3" * 64,
                ),
            ),
            "rule_coverage_claims": tuple(
                claim.model_copy(
                    update={
                        "target_evidence_refs": (
                            *claim.target_evidence_refs,
                            *added_refs.get(claim.rule_id, ()),
                        )
                    }
                )
                for claim in valid.rule_coverage_claims
            ),
        }
    )

    result = core.validate_structural_input(
        candidate, profiles.structural_evidence_profile(), registry.structural_registry()
    )

    assert result.disposition.value == "invalid"
    assert result.failed_rule_ids == ("structural.schema.v1",)
    assert set(result.missing_rule_ids) == set(result.required_rule_ids) - {"structural.schema.v1"}

    missing_schema_result = core.validate_structural_input(
        candidate,
        profiles.structural_evidence_profile(),
        registry.structural_registry(exclude={"structural.schema.v1"}),
    )
    assert missing_schema_result.disposition.value == "invalid"
    assert set(missing_schema_result.missing_rule_ids) == set(
        missing_schema_result.required_rule_ids
    )


def test_schema_invalid_float_returns_invalid_instead_of_breaking_target_hash() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    valid = load_fixture("valid.json")
    malformed_document = valid.documents[0].model_copy(
        update={"document": valid.documents[0].document | {"unexpected_float": 0.5}}
    )
    malformed = valid.model_copy(update={"documents": (malformed_document, *valid.documents[1:])})

    result = core.validate_structural_input(
        malformed, profiles.structural_evidence_profile(), registry.structural_registry()
    )

    assert result.disposition.value == "invalid"
    assert result.failed_rule_ids == ("structural.schema.v1",)


def test_schema_version_boolean_is_not_integer_version_one() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    payload = json.loads((FIXTURE_ROOT / "valid.json").read_text(encoding="utf-8"))
    payload["documents"][0]["document"]["schema_version"] = True
    candidate = core.StructuralValidationInput.model_validate_json(json.dumps(payload))

    result = core.validate_structural_input(
        candidate, profiles.structural_evidence_profile(), registry.structural_registry()
    )

    assert result.disposition.value == "invalid"
    assert result.failed_rule_ids == ("structural.schema.v1",)


def test_validation_target_hash_type_tags_cannot_collide_with_user_values() -> None:
    core = require("oamb.artifacts.validation.core")
    valid = load_fixture("valid.json")
    float_value = valid.documents[0].model_copy(
        update={"document": valid.documents[0].document | {"extra": 0.5}}
    )
    marker_shaped_value = valid.documents[0].model_copy(
        update={
            "document": valid.documents[0].document | {"extra": {"invalid_float_hex": (0.5).hex()}}
        }
    )
    float_input = valid.model_copy(update={"documents": (float_value,)})
    marker_input = valid.model_copy(update={"documents": (marker_shaped_value,)})

    assert core._validation_input_hash(float_input) != core._validation_input_hash(marker_input)
    raw_bytes = b"hello"
    marker = {
        "invalid_or_raw_bytes_sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "byte_count": len(raw_bytes),
    }
    assert core._validation_hash_value(raw_bytes) != core._validation_hash_value(marker)


def test_invalid_identity_parts_return_mismatch_instead_of_raising() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    valid = load_fixture("valid.json")
    invalid_identity = valid.identity_claims[0].model_copy(update={"identity_parts": (0.5,)})
    malformed = valid.model_copy(update={"identity_claims": (invalid_identity,)})

    result = core.validate_structural_input(
        malformed, profiles.structural_evidence_profile(), registry.structural_registry()
    )

    assert result.disposition.value == "invalid"
    assert result.failed_rule_ids == ("structural.identity.v1",)

    mixed_key_payload = valid.model_dump(mode="python")
    mixed_key_payload["identity_claims"][0]["identity_parts"] = ({"a": 1, 2: 3},)
    with pytest.raises(ValidationError):
        core.StructuralValidationInput.model_validate(mixed_key_payload)


@pytest.mark.parametrize("invalid_count", (True, 1.0, "1"))
def test_structural_envelope_does_not_coerce_count_types(invalid_count: object) -> None:
    core = require("oamb.artifacts.validation.core")
    valid = load_fixture("valid.json")
    payload = valid.model_dump(mode="python")
    payload["count_claims"][0]["declared_count"] = invalid_count

    with pytest.raises(ValidationError):
        core.StructuralValidationInput.model_validate(payload)


def test_structural_envelope_rejects_empty_references_and_non_finite_floats() -> None:
    core = require("oamb.artifacts.validation.core")

    with pytest.raises(ValidationError):
        core.ArtifactDocument(record_id="", record_kind="protocol_spec", document={})
    with pytest.raises(ValidationError):
        core.IdentityClaim(
            evidence_ref="artifact-1",
            identity_prefix="oamb-test-v1",
            identity_parts=(float("nan"),),
            actual_id="a" * 64,
        )


def test_ingestion_plan_reference_groups_require_their_exact_record_kinds() -> None:
    core = require("oamb.artifacts.validation.core")
    valid = load_fixture("valid.json")
    logical = core.ArtifactDocument(
        record_id="logical-target",
        record_kind="logical_context_manifest_entry",
        document={},
    )
    case = core.ArtifactDocument(
        record_id="case-target",
        record_kind="case_manifest_entry",
        document={},
    )
    swapped_plan = core.ArtifactDocument(
        record_id="plan-swapped",
        record_kind="ingestion_plan_manifest",
        document={
            "ordered_member_context_manifest_entry_ids": ["case-target"],
            "ordered_case_manifest_entry_ids": ["logical-target"],
        },
    )
    swapped = valid.model_copy(
        update={
            "documents": (*valid.documents, logical, case, swapped_plan),
            "reference_claims": (
                *valid.reference_claims,
                core.ReferenceClaim(
                    evidence_ref="plan-swapped",
                    target_record_id="case-target",
                    expected_record_kind="case_manifest_entry",
                ),
                core.ReferenceClaim(
                    evidence_ref="plan-swapped",
                    target_record_id="logical-target",
                    expected_record_kind="logical_context_manifest_entry",
                ),
            ),
        }
    )

    issues = core._reference_rule(swapped)

    assert any(
        issue.evidence_ref == "plan-swapped" and issue.code == "reference-shape-invalid"
        for issue in issues
    )


def test_validation_result_cannot_claim_validated_with_missing_rules() -> None:
    core = require("oamb.artifacts.validation.core")
    evidence = require("oamb.contracts.evidence")
    states = require("oamb.contracts.states")

    with pytest.raises(ValidationError, match="validated.*missing"):
        result = evidence.ValidationResult(
            validation_profile_id="profile-v1",
            target_hash="a" * 64,
            disposition=states.ValidationDisposition.VALIDATED,
            required_rule_ids=("required-rule",),
            executed_rule_ids=(),
            passed_rule_ids=(),
            failed_rule_ids=(),
            not_applicable_rule_ids=(),
            missing_rule_ids=("required-rule",),
            implementation_versions=(),
            issues=(),
        )
        core.require_validated(result)


def test_validation_result_cannot_claim_validated_with_error_issue() -> None:
    evidence = require("oamb.contracts.evidence")
    states = require("oamb.contracts.states")
    issue = evidence.ValidationIssue(
        rule_id="required-rule",
        code="planted-error",
        severity=evidence.ValidationSeverity.ERROR,
        evidence_ref="artifact-1",
        json_pointer=None,
        remediation_code="repair-planted-error",
    )

    with pytest.raises(ValidationError, match="error issues.*failed rules"):
        evidence.ValidationResult(
            validation_profile_id="profile-v1",
            target_hash="a" * 64,
            disposition=states.ValidationDisposition.VALIDATED,
            required_rule_ids=("required-rule",),
            executed_rule_ids=("required-rule",),
            passed_rule_ids=("required-rule",),
            failed_rule_ids=(),
            not_applicable_rule_ids=(),
            missing_rule_ids=(),
            implementation_versions=("required-rule@1",),
            issues=(issue,),
        )


def test_structural_engine_rejects_an_export_stage_profile() -> None:
    core = require("oamb.artifacts.validation.core")
    profiles = require("oamb.artifacts.validation.profiles")
    registry = require("oamb.artifacts.validation.registry")
    specifications = require("oamb.contracts.specifications")
    profile = profiles.structural_evidence_profile()
    export_profile = specifications.ValidationProfile.create(
        profile_id="wrong-stage-v1",
        stage=specifications.ValidationStage.EXPORT,
        required_rules=profile.required_rules,
        applicability=profile.applicability,
    )

    result = core.validate_structural_input(
        load_fixture("valid.json"), export_profile, registry.structural_registry()
    )

    assert result.disposition.value == "invalid"
    assert result.executed_rule_ids == ()
    assert result.failed_rule_ids == ("validation.stage.v1",)


def test_duplicate_rule_registration_and_invalid_reducer_input_are_rejected() -> None:
    core = require("oamb.artifacts.validation.core")
    registry = require("oamb.artifacts.validation.registry")

    rules = registry.RuleRegistry()
    rule = registry.structural_registry().get("structural.hash.v1")
    rules.register(rule)
    with pytest.raises(registry.DuplicateRuleError):
        rules.register(rule)

    with pytest.raises(core.EvidenceNotValidatedError):
        core.require_validated(validate("invalid-hash.json"))
