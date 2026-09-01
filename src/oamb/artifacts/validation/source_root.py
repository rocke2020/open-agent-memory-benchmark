"""Generic sealed source-root validation dispatch."""

from __future__ import annotations

import json
from pathlib import Path

from oamb.artifacts.atomic import read_regular_file
from oamb.contracts.evidence import CapsuleManifest, ValidationResult
from oamb.contracts.states import ValidationDisposition
from oamb.workloads.fake import FAKE_WORKLOAD_ID

from .composition import declares_composition, validate_composed_capsule
from .fake import validate_fake_capsule
from .native import validate_native_capsule, validate_partition_capsule


def validate_source_root(source_root: Path) -> ValidationResult:
    """Reopen one sealed source root and select its closed evidence profile."""

    root = Path(source_root)
    if declares_composition(root):
        return validate_composed_capsule(root)
    fake_result = validate_fake_capsule(root)
    if fake_result.disposition == ValidationDisposition.VALIDATED:
        return fake_result
    native_result = validate_native_capsule(root)
    if native_result.disposition == ValidationDisposition.VALIDATED:
        return native_result
    partition_result = validate_partition_capsule(root)
    if partition_result.disposition == ValidationDisposition.VALIDATED:
        return partition_result
    return fake_result if _declares_fake_workload(root) else native_result


def _declares_fake_workload(source_root: Path) -> bool:
    try:
        manifest = CapsuleManifest.model_validate_json(
            read_regular_file(source_root / "capsule-manifest.json")
        )
        run_specs = tuple(
            entry for entry in manifest.source_entries if entry.record_kind == "run_spec"
        )
        if len(run_specs) != 1:
            return False
        document = json.loads(read_regular_file(source_root / run_specs[0].relative_path))
    except (OSError, TypeError, ValueError):
        return False
    return isinstance(document, dict) and document.get("workload_id") == FAKE_WORKLOAD_ID


__all__ = ["validate_source_root"]
