"""Fixed public text for external historical limitation codes."""

from __future__ import annotations

from oamb.contracts.external import HistoricalLimitationCode

_LIMITATION_TEXT: dict[HistoricalLimitationCode, str] = {
    "no-oamb-attempt-ledger": "No OAMB attempt ledger exists for this external producer run.",
    "indexing-usage-unavailable": (
        "Exact canonical indexing usage is unavailable; no zero value is inferred."
    ),
    "external-llm-usage-unavailable": (
        "Complete external-LLM usage is unavailable; no zero value is inferred."
    ),
    "prompt-compatibility-unknown": (
        "Prompt compatibility is unknown because the OAMB comparison predicate was not run."
    ),
    "amb-context-view-not-oamb-context-view": (
        "Historical Ctx tokens count the AMB formatted retrieval view, not OAMB context_view."
    ),
    "no-causal-attribution": (
        "The preserved verdicts do not establish model-only or memory-only causal attribution."
    ),
}


def historical_limitation_texts(
    codes: tuple[HistoricalLimitationCode, ...],
) -> tuple[str, ...]:
    return tuple(_LIMITATION_TEXT[code] for code in codes)


__all__ = ["historical_limitation_texts"]
