"""Lightweight phase-review constants shared without concrete client imports."""

from oamb.contracts.ids import canonical_sha256

FAKE_PHASE_REVIEW_CLIENT_KIND = "fake"
OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND = "openai-compatible"
FAKE_PHASE_REVIEW_PROVIDER = "oamb-fake-reviewer"
FAKE_PHASE_REVIEW_ENDPOINT = "offline://phase-review"
FAKE_PHASE_REVIEW_CREDENTIAL_VARIABLE = "OAMB_FAKE_PHASE_REVIEW_TOKEN"
FAKE_PHASE_REVIEW_CONFIGURED_MODEL = "fake-review-v1"
FAKE_PHASE_REVIEW_RESOLVED_MODEL = "fake-review-v1@recorded"
FAKE_PHASE_REVIEW_COST_REASON = "credential-free fake review has no supplier billing"
PHASE_REVIEW_INPUT_TOKEN_PRICE_CLASS = "quality_review_input_tokens_per_million"
PHASE_REVIEW_OUTPUT_TOKEN_PRICE_CLASS = "quality_review_output_tokens_per_million"
PHASE_REVIEW_TOKEN_PRICE_UNIT_SCALE = 1_000_000
PHASE_REVIEW_WALL_DIMENSION_ID = "provider_request_wall_seconds_v1"
PHASE_REVIEW_WALL_UNIT = "seconds"
FAKE_PHASE_REVIEW_WALL_MEASUREMENT_SOURCE = "fixture_clock_v1"
OPENAI_PHASE_REVIEW_WALL_MEASUREMENT_SOURCE = "process_meter"
PHASE_REVIEW_CLIENT_KINDS = frozenset(
    {FAKE_PHASE_REVIEW_CLIENT_KIND, OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND}
)


def phase_review_runtime_hash(
    role_redacted_endpoint_fingerprint: str | None,
    environment_hash: str,
    *,
    client_kind: str,
) -> str:
    if client_kind not in PHASE_REVIEW_CLIENT_KINDS:
        raise ValueError("unknown phase-review client kind")
    return canonical_sha256(
        [
            "oamb-phase-review-runtime-binding-v1",
            client_kind,
            role_redacted_endpoint_fingerprint,
            environment_hash,
        ]
    )


__all__ = [
    "FAKE_PHASE_REVIEW_CLIENT_KIND",
    "FAKE_PHASE_REVIEW_CONFIGURED_MODEL",
    "FAKE_PHASE_REVIEW_COST_REASON",
    "FAKE_PHASE_REVIEW_CREDENTIAL_VARIABLE",
    "FAKE_PHASE_REVIEW_ENDPOINT",
    "FAKE_PHASE_REVIEW_PROVIDER",
    "FAKE_PHASE_REVIEW_RESOLVED_MODEL",
    "FAKE_PHASE_REVIEW_WALL_MEASUREMENT_SOURCE",
    "OPENAI_COMPATIBLE_PHASE_REVIEW_CLIENT_KIND",
    "OPENAI_PHASE_REVIEW_WALL_MEASUREMENT_SOURCE",
    "PHASE_REVIEW_CLIENT_KINDS",
    "PHASE_REVIEW_INPUT_TOKEN_PRICE_CLASS",
    "PHASE_REVIEW_OUTPUT_TOKEN_PRICE_CLASS",
    "PHASE_REVIEW_TOKEN_PRICE_UNIT_SCALE",
    "PHASE_REVIEW_WALL_DIMENSION_ID",
    "PHASE_REVIEW_WALL_UNIT",
    "phase_review_runtime_hash",
]
