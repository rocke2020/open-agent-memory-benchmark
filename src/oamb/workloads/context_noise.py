"""Offline reviewed-span accounting, ported from the OAM evaluator.

Copyright (c) 2026 Rocke Dong. MIT; see LICENSE.OAM-eval.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import tempfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from typing import Any

import tiktoken
import tiktoken.load
import tiktoken.registry

from oamb.artifacts.atomic import atomic_write_bytes

CATEGORIES = ("direct", "related", "unrelated", "uncertain", "unreviewed", "format")
STATUSES = ("reviewed", "unreviewed", "uncertain", "lower_bound", "empty_context")


def digest_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def write_json_new(path: Path, value: Any) -> None:
    atomic_write_bytes(Path(path), _json_bytes(value))


def encoder_for(tokenizer: dict[str, Any]) -> tiktoken.Encoding:
    if (
        not isinstance(tokenizer, dict)
        or tokenizer.get("version") != version("tiktoken")
        or tokenizer.get("method") not in {"encode", "encode_ordinary"}
        or not isinstance(tokenizer.get("name"), str)
    ):
        raise ValueError(
            "tokenizer identity is missing, unsupported or differs from installed version"
        )
    # Constructors run under tiktoken's own reentrant registry lock. Substitute
    # only their cache reader, so missing/corrupt assets cannot trigger a download.
    with tiktoken.registry._lock:
        original = tiktoken.load.read_file_cached
        tiktoken.load.read_file_cached = offline_tokenizer_asset
        try:
            return tiktoken.get_encoding(tokenizer["name"])
        finally:
            tiktoken.load.read_file_cached = original


def offline_tokenizer_asset(blobpath: str, expected_hash: str | None = None) -> bytes:
    """Read the dependency's content-addressed cache without fetch or replacement."""
    cache_dir = os.environ.get(
        "TIKTOKEN_CACHE_DIR",
        os.environ.get("DATA_GYM_CACHE_DIR", os.path.join(tempfile.gettempdir(), "data-gym-cache")),
    )
    cache_path = Path(cache_dir) / hashlib.sha1(blobpath.encode()).hexdigest()
    if not cache_dir or not cache_path.is_file():
        raise ValueError(
            "offline tokenizer assets unavailable; prepare the frozen tokenizer cache separately"
        )
    data = cache_path.read_bytes()
    if expected_hash and hashlib.sha256(data).hexdigest() != expected_hash:
        raise ValueError("offline tokenizer asset hash differs; repair the local cache separately")
    return data


def count_context(
    context: str, spans: list[dict[str, Any]], tokenizer: dict[str, Any]
) -> dict[str, Any]:
    encoder = encoder_for(tokenizer)
    if not isinstance(context, str) or not isinstance(spans, list):
        raise ValueError("context text and span list are required")
    raw = context.encode("utf-8")
    boundaries = {0}
    offset = 0
    for char in context:
        offset += len(char.encode("utf-8"))
        boundaries.add(offset)
    intervals = []
    for span in spans:
        if not isinstance(span, dict):
            raise ValueError("invalid context span")
        start, end, category = span.get("start"), span.get("end"), span.get("category")
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= len(raw)
            or category not in CATEGORIES
            or (
                category not in {"format", "unreviewed"} and not str(span.get("reason", "")).strip()
            )
        ):
            raise ValueError("invalid context span bounds, category or reason")
        if start not in boundaries or end not in boundaries:
            raise ValueError("span offset splits a UTF-8 character")
        intervals.append((start, end, category))
    intervals.sort()
    if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:], strict=False)):
        raise ValueError("context spans overlap")
    starts = [span[0] for span in intervals]
    gaps = []
    cursor = 0
    for start, end, _ in intervals:
        gaps.append(raw[cursor:start])
        cursor = end
    gaps.append(raw[cursor:])
    unresolved = any(gap.strip(b" \t\r\n") for gap in gaps) or any(
        category == "unreviewed" and raw[start:end].strip(b" \t\r\n")
        for start, end, category in intervals
    )
    uncertain = any(
        category == "uncertain" and raw[start:end].strip(b" \t\r\n")
        for start, end, category in intervals
    )
    tokens = getattr(encoder, tokenizer["method"])(context)
    counts = dict.fromkeys(CATEGORIES, 0)
    offset = 0
    for token in tokens:
        data = encoder.decode_single_token_bytes(token)
        stripped = data.lstrip(b" \t\r\n")
        point = offset + len(data) - len(stripped)
        category = "format" if not stripped else "unreviewed"
        index = bisect.bisect_right(starts, point) - 1
        if stripped and index >= 0 and point < intervals[index][1]:
            category = intervals[index][2]
        counts[category] += 1
        offset += len(data)
    return {
        "context_tokens": len(tokens),
        "unrelated_text_tokens": counts["unrelated"],
        "tokens_by_category": counts,
        "tokenizer": tokenizer,
        "span_coverage_complete": not unresolved,
        "has_uncertain_spans": bool(uncertain),
    }


def default_assessment() -> dict[str, Any]:
    return {
        "source_review_complete": False,
        "span_review_complete": False,
        "task_interpretation": "",
        "spans": [],
        "evidence_requirements": [],
        "measurement": "exact",
    }


def review_noise(
    context: str,
    assessment: dict[str, Any] | None,
    tokenizer: dict[str, Any],
    sessions: list[dict[str, Any]],
) -> dict[str, Any]:
    assessment = assessment if assessment is not None else default_assessment()
    if not isinstance(assessment, dict):
        raise ValueError("noise assessment must be an object")
    for flag in ("source_review_complete", "span_review_complete"):
        if type(assessment.get(flag, False)) is not bool:
            raise ValueError(f"noise {flag} must be boolean")
    measurement = assessment.get("measurement", "exact")
    if measurement not in {"exact", "lower_bound"}:
        raise ValueError("invalid noise measurement method")
    result = count_context(context, assessment.get("spans", []), tokenizer)
    requirements = assessment.get("evidence_requirements", [])
    if not isinstance(requirements, list):
        raise ValueError("noise evidence requirements must be a list")
    coverage: dict[str, Any] = {
        "present": 0,
        "missing": 0,
        "uncertain": 0,
        "requirements": requirements,
    }
    for requirement in requirements:
        status = requirement.get("status")
        quotes, sources = requirement.get("context_quotes"), requirement.get("source_quotes")
        if (
            status not in {"present", "missing", "uncertain"}
            or not requirement.get("description")
            or not isinstance(sources, list)
            or not sources
        ):
            raise ValueError("invalid source-backed evidence requirement")
        for source in sources:
            quote = source.get("quote")
            originals = [
                m["content"]
                for s in sessions
                for m in s["messages"]
                if s["session_id"] == source.get("session_id")
                and m["message_index"] == source.get("message_index")
                and (
                    source.get("session_index") is None
                    or s.get("session_index") == source["session_index"]
                )
            ]
            if (
                len(originals) != 1
                or not isinstance(quote, str)
                or not quote.strip()
                or quote not in originals[0]
            ):
                raise ValueError("nonexistent source quote in noise requirement")
        if not isinstance(quotes, list) or any(
            not isinstance(q, str) or not q.strip() or q not in context for q in quotes
        ):
            raise ValueError("nonexistent context quote in noise requirement")
        if (status == "present" and not quotes) or (status == "missing" and quotes):
            raise ValueError("evidence status disagrees with context quotes")
        coverage[status] += 1
    reasons = []
    if not assessment.get("source_review_complete", False):
        reasons.append("source_review_incomplete")
    if not assessment.get("span_review_complete", False) or not result["span_coverage_complete"]:
        reasons.append("span_review_incomplete")
    if (
        not isinstance(assessment.get("task_interpretation"), str)
        or not assessment["task_interpretation"].strip()
    ):
        reasons.append("task_interpretation_missing")
    if result["has_uncertain_spans"]:
        reasons.append("uncertain_relevance")
    total = result["context_tokens"]
    conservative = (
        measurement == "lower_bound"
        and assessment.get("source_review_complete") is True
        and "task_interpretation_missing" not in reasons
    )
    status = (
        "empty_context"
        if not total
        else "lower_bound"
        if conservative
        else "uncertain"
        if result["has_uncertain_spans"]
        else "unreviewed"
        if reasons
        else "lower_bound"
        if measurement == "lower_bound"
        else "reviewed"
    )
    result.update(
        noise_status=status,
        reasons=reasons,
        noise_ratio=result["unrelated_text_tokens"] / total if status == "reviewed" else None,
        noise_lower_bound=result["unrelated_text_tokens"] / total
        if status == "lower_bound"
        else None,
        source_review_complete=assessment.get("source_review_complete", False),
        span_review_complete=assessment.get("span_review_complete", False),
        evidence_coverage=coverage,
    )
    return result


def aggregate_noise(rows: list[dict[str, Any]], *, expected_cases: int) -> dict[str, Any]:
    if type(expected_cases) is not int or expected_cases < len(rows):
        raise ValueError("invalid expected noise case count")
    reviewed = [r for r in rows if r["noise_status"] == "reviewed"]
    identities = {digest_json(r["tokenizer"]) for r in reviewed}
    if len(identities) > 1:
        raise ValueError("cannot aggregate different tokenizer identities")
    counts = Counter(r["noise_status"] for r in rows)
    total = sum(r["context_tokens"] for r in reviewed)
    noise = sum(r["unrelated_text_tokens"] for r in reviewed)
    return {
        "expected_cases": expected_cases,
        "saved_cases": len(rows),
        "missing_cases": expected_cases - len(rows),
        **{f"{status}_cases": counts[status] for status in STATUSES},
        "context_tokens": total,
        "unrelated_text_tokens": noise,
        "noise_ratio": noise / total if total else None,
        "tokenizer": reviewed[0]["tokenizer"] if reviewed else None,
    }
