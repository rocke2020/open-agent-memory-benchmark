"""Pure fake-report identity and claim-boundary formulas."""

from __future__ import annotations

from typing import Literal

from oamb.contracts.evidence import ValidationResult
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import RunSummaryV2

NORMAL_FAKE_REPORT_LIMITATIONS = (
    "generated fake evidence only; no live provider support claim",
    "one scripted judge is unjudged and excluded from judged quality",
    "synthetic usage is not a real supplier charge",
)


def fake_report_limitations(
    validation: ValidationResult,
    *,
    diagnostic: bool,
) -> tuple[str, ...]:
    if diagnostic:
        codes = tuple(sorted({issue.code for issue in validation.issues}))
        return (
            "diagnostic-only: evidence validation did not pass",
            *(f"validation issue: {code}" for code in codes),
        )
    return NORMAL_FAKE_REPORT_LIMITATIONS


def fake_report_id(
    *,
    source_manifest_hash: str,
    evidence_validation_hash: str,
    summary: RunSummaryV2,
    audience: Literal["local", "public"],
    diagnostic: bool,
    limitations: tuple[str, ...],
) -> str:
    return canonical_sha256(
        [
            "oamb-fake-run-report-v1",
            source_manifest_hash,
            evidence_validation_hash,
            summary,
            audience,
            diagnostic,
            limitations,
        ]
    )


__all__ = [
    "NORMAL_FAKE_REPORT_LIMITATIONS",
    "fake_report_id",
    "fake_report_limitations",
]
