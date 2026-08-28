"""Repository-owned browser and performance acceptance identities."""

from __future__ import annotations

from oamb.constants import (
    REPORT_BUILD_MAX_RSS_BYTES,
    REPORT_BUILD_MAX_SECONDS,
    REPORT_DETAIL_NAV_MAX_MILLISECONDS,
    REPORT_FILTER_SORT_MAX_MILLISECONDS,
    REPORT_INTERACTIVE_MAX_MILLISECONDS,
    REPORT_MAX_HTML_BYTES,
    REPORT_SCALE_MAX_RATIO,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256

BROWSER_ACCEPTANCE_CONTRACT = {
    "contract_id": "oamb-browser-acceptance-v1",
    "playwright_version": "1.55.0",
    "engines": (
        {"name": "chromium", "revision": "1187", "version": "140.0.7339.16"},
        {"name": "firefox", "revision": "1490", "version": "141.0"},
        {"name": "webkit", "revision": "2203", "version": "26.0"},
    ),
    "platform_matrix": (
        "linux-x86_64:chromium,firefox,webkit",
        "macos:chromium,webkit",
    ),
    "transport": "file://",
    "network_requests": 0,
}

PERFORMANCE_ACCEPTANCE_CONTRACT = {
    "contract_id": "oamb-report-performance-v1",
    "measurement_schema": "report_performance_measurement@1",
    "case_counts": (50, 500, 5_000),
    "attempts_per_case": 2,
    "evidence_preview_bytes": 512,
    "max_html_bytes": REPORT_MAX_HTML_BYTES,
    "max_build_seconds": REPORT_BUILD_MAX_SECONDS,
    "max_peak_rss_bytes": REPORT_BUILD_MAX_RSS_BYTES,
    "max_interactive_milliseconds": REPORT_INTERACTIVE_MAX_MILLISECONDS,
    "max_filter_sort_milliseconds": REPORT_FILTER_SORT_MAX_MILLISECONDS,
    "max_detail_navigation_milliseconds": REPORT_DETAIL_NAV_MAX_MILLISECONDS,
    "max_10x_growth_ratio": REPORT_SCALE_MAX_RATIO,
    "reference_job": "linux-x86_64-4cpu-16gib",
}


def browser_acceptance_contract_bytes() -> bytes:
    return canonical_json_bytes(BROWSER_ACCEPTANCE_CONTRACT)


def browser_acceptance_contract_hash() -> str:
    return canonical_sha256(BROWSER_ACCEPTANCE_CONTRACT)


def performance_acceptance_contract_bytes() -> bytes:
    return canonical_json_bytes(PERFORMANCE_ACCEPTANCE_CONTRACT)


def performance_acceptance_contract_hash() -> str:
    return canonical_sha256(PERFORMANCE_ACCEPTANCE_CONTRACT)


__all__ = [
    "BROWSER_ACCEPTANCE_CONTRACT",
    "PERFORMANCE_ACCEPTANCE_CONTRACT",
    "browser_acceptance_contract_bytes",
    "browser_acceptance_contract_hash",
    "performance_acceptance_contract_bytes",
    "performance_acceptance_contract_hash",
]
