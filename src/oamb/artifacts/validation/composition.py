"""Fresh recursive validation for immutable composed capsules."""

from __future__ import annotations

import hashlib
from pathlib import Path

from oamb.artifacts.atomic import read_regular_file, sha256_file
from oamb.artifacts.composition import (
    COMPOSITION_VALIDATION_PROFILE_ID,
    CapsuleCompositionError,
    CapsuleCompositionTarget,
    _compose_record,
    _load_part,
    inspect_embedded_composition,
)
from oamb.contracts.evidence import (
    CapsuleManifest,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
)
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.states import ValidationDisposition

COMPOSITION_RULE_ID = "composition.recursive-exact-union.v1"


def validate_composed_capsule(capsule_root: Path) -> ValidationResult:
    """Reopen outer and embedded bytes and recompute the exact composition."""

    root = Path(capsule_root)
    try:
        manifest = CapsuleManifest.model_validate_json(
            read_regular_file(root / "capsule-manifest.json")
        )
        _verify_outer_manifest(root, manifest)
        composition, embedded_roots = inspect_embedded_composition(root)
        parts = tuple(_load_part(embedded_root) for embedded_root in embedded_roots)
        target = None
        if composition.target_execution_configuration_hash is not None:
            target = CapsuleCompositionTarget(
                resolved_plan_hash=composition.resolved_plan_hash,
                cell_spec_hash=composition.cell_spec_hash,
                target_case_manifest_hash=composition.target_case_manifest_hash,
                budget_policy_hash=composition.budget_policy_hash,
                retry_policy_hash=composition.retry_policy_hash,
                execution_configuration_hash=(composition.target_execution_configuration_hash),
                execution_configuration_family_hash=(
                    composition.target_execution_configuration_family_hash
                ),
            )
        recomputed = _compose_record(parts, target=target)
        if recomputed != composition:
            raise CapsuleCompositionError("composition record differs from embedded parts")
        expected_run_spec_hash = canonical_sha256(
            ["oamb-composed-run-spec-v1", composition.composition_id]
        )
        if (
            manifest.run_id != composition.composition_id
            or manifest.run_spec_hash != expected_run_spec_hash
        ):
            raise CapsuleCompositionError("composition outer identity mismatch")
    except (OSError, ValueError):
        target_hash = _target_hash(root)
        issue = ValidationIssue(
            rule_id=COMPOSITION_RULE_ID,
            code="composition-invalid",
            severity=ValidationSeverity.ERROR,
            evidence_ref=target_hash,
            json_pointer=None,
            remediation_code="rebuild-composition-from-valid-parts",
        )
        return ValidationResult(
            validation_profile_id=COMPOSITION_VALIDATION_PROFILE_ID,
            target_hash=target_hash,
            disposition=ValidationDisposition.INVALID,
            required_rule_ids=(COMPOSITION_RULE_ID,),
            executed_rule_ids=(COMPOSITION_RULE_ID,),
            passed_rule_ids=(),
            failed_rule_ids=(COMPOSITION_RULE_ID,),
            not_applicable_rule_ids=(),
            missing_rule_ids=(),
            implementation_versions=(f"{COMPOSITION_RULE_ID}@1",),
            issues=(issue,),
        )
    return ValidationResult(
        validation_profile_id=COMPOSITION_VALIDATION_PROFILE_ID,
        target_hash=manifest.source_manifest_hash,
        disposition=ValidationDisposition.VALIDATED,
        required_rule_ids=(COMPOSITION_RULE_ID,),
        executed_rule_ids=(COMPOSITION_RULE_ID,),
        passed_rule_ids=(COMPOSITION_RULE_ID,),
        failed_rule_ids=(),
        not_applicable_rule_ids=(),
        missing_rule_ids=(),
        implementation_versions=(f"{COMPOSITION_RULE_ID}@1",),
        issues=(),
    )


def declares_composition(root: Path) -> bool:
    try:
        manifest = CapsuleManifest.model_validate_json(
            read_regular_file(Path(root) / "capsule-manifest.json")
        )
    except (OSError, ValueError):
        return False
    return any(
        entry.record_kind == "capsule_composition_record" for entry in manifest.source_entries
    )


def _verify_outer_manifest(root: Path, manifest: CapsuleManifest) -> None:
    actual_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "source").rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    expected_paths = {entry.relative_path for entry in manifest.source_entries}
    if actual_paths != expected_paths:
        raise CapsuleCompositionError("composition source inventory mismatch")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise CapsuleCompositionError("composition contains a symbolic link")
    for entry in manifest.source_entries:
        if sha256_file(root / entry.relative_path) != entry.sha256:
            raise CapsuleCompositionError("composition source hash mismatch")
    source_manifest_hash = canonical_sha256(
        [
            "oamb-source-manifest-v1",
            tuple(entry.model_dump(mode="python") for entry in manifest.source_entries),
        ]
    )
    capsule_id = canonical_sha256(
        ["oamb-capsule-v1", manifest.run_id, manifest.run_spec_hash, source_manifest_hash]
    )
    if manifest.source_manifest_hash != source_manifest_hash or manifest.capsule_id != capsule_id:
        raise CapsuleCompositionError("composition manifest identity mismatch")


def _target_hash(root: Path) -> str:
    try:
        return hashlib.sha256(read_regular_file(root / "capsule-manifest.json")).hexdigest()
    except OSError:
        return hashlib.sha256(b"").hexdigest()


__all__ = ["declares_composition", "validate_composed_capsule"]
