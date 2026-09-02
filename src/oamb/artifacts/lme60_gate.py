"""Shared proof marker check for complete LME-60 artifacts."""

from __future__ import annotations

BOUNDED_PROVIDER_PROFILE_EVIDENCE_VERSION = "bounded_provider_profile_evidence@1"


def has_bounded_provider_profile_evidence(preflight: object | None) -> bool:
    profile = getattr(preflight, "provider_profile_evidence", None)
    versions = getattr(profile, "source_schema_versions", ())
    return BOUNDED_PROVIDER_PROFILE_EVIDENCE_VERSION in versions


__all__ = [
    "BOUNDED_PROVIDER_PROFILE_EVIDENCE_VERSION",
    "has_bounded_provider_profile_evidence",
]
