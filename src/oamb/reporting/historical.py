"""Single-file, offline question browser for published historical evidence."""

from __future__ import annotations

import base64
import gzip
import hashlib
import html
import json
import math
import statistics
from importlib import resources
from pathlib import Path
from typing import Any

from oamb.artifacts.atomic import atomic_replace_bytes

_MAX_REPORT_BYTES = 16 * 1024 * 1024
_ATTEMPT_FIELDS = (
    "answer",
    "score",
    "correct",
    "answer_effort",
    "judge_effort",
    "judge_reason",
    "answer_model",
    "judge_model",
)
_QUESTION_FIELDS = (
    "question_id",
    "question",
    "gold_answers",
    "rubric",
    "question_type",
    "context",
    "context_sha256",
    "context_tokens",
    "context_tokenizer",
    "retrieve_time_ms",
    "retrieval_samples_ms",
    "source",
    "cohort",
    "conversation_id",
)
_CARRIED_STATUS = "carried_incorrect_gold_no"
_CARRIED_FIELDS = {
    "question_id",
    "question",
    "gold_answers",
    "rubric",
    "question_type",
    "source",
    "cohort",
    "conversation_id",
    "evaluation_status",
    "score",
    "correct",
    "first_pass",
    "retry",
}


def _digest(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _summary(records: list[dict[str, Any]], dataset: str) -> dict[str, Any]:
    scores = [
        q["selected_answer"]["score"] if q["selected_answer"] else q["score"] for q in records
    ]
    measured = [q for q in records if q["selected_answer"] is not None]
    samples = [
        sample
        for q in measured
        for sample in q.get("retrieval_samples_ms", [q["retrieve_time_ms"]])
    ]
    summary = {
        "mean": statistics.mean(scores),
        "passes": sum(score >= (1 if dataset == "lme500" else 0.5) for score in scores),
        "count": len(records),
        "context_mean": statistics.mean(q["context_tokens"] for q in measured)
        if measured
        else None,
        "context_total": sum(q["context_tokens"] for q in measured) if measured else None,
        "retrieval_median_ms": statistics.median(samples) if samples else None,
    }
    if len(measured) != len(records) or any("retrieval_samples_ms" in q for q in records):
        summary.update(
            context_count=len(measured),
            carried_count=len(records) - len(measured),
            retrieval_sample_count=len(samples),
        )
    return summary


def _evidence(root: Path) -> dict[str, Any]:
    def read(relative: str) -> bytes:
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("evidence path escapes result directory")
        return path.read_bytes()

    manifest = json.loads(read("manifest.json"))
    dataset = manifest["dataset"]
    if dataset not in {"lme500", "beam10m"}:
        raise ValueError("unsupported historical dataset")
    if "oam" not in manifest["arms"]:
        raise ValueError("historical report requires OAM evidence")
    arms: dict[str, Any] = {}
    for name in ["oam", *(key for key in manifest["arms"] if key != "oam")]:
        arm = manifest["arms"][name]
        records = []
        for relative in arm["files"]:
            document = json.loads(gzip.decompress(read(relative)))
            if document["dataset"] != dataset or document["arm"] != name:
                raise ValueError("shard differs from manifest dataset or arm")
            for original in document["results"]:
                q = {key: original[key] for key in _QUESTION_FIELDS if key in original}
                q["source_file"] = relative
                if original.get("evaluation_status") == _CARRIED_STATUS:
                    if (
                        dataset != "lme500"
                        or type(original.get("score")) not in (int, float)
                        or original["score"] != 0
                        or original.get("correct") is not False
                        or original.get("first_pass") is not None
                        or original.get("retry") is not None
                        or original.keys() - _CARRIED_FIELDS
                    ):
                        raise ValueError(
                            "invalid carried No: requires fixed LME No without current evidence"
                        )
                    q.update(
                        evaluation_status=_CARRIED_STATUS,
                        score=0,
                        correct=False,
                        selected_answer=None,
                    )
                    records.append(q)
                    continue
                if _digest(q["context"].encode()) != q["context_sha256"]:
                    raise ValueError("saved context hash mismatch")
                if "retrieval_samples_ms" in q:
                    samples = q["retrieval_samples_ms"]
                    if (
                        not isinstance(samples, list)
                        or not samples
                        or any(
                            type(sample) not in (int, float)
                            or not math.isfinite(sample)
                            or sample < 0
                            for sample in samples
                        )
                    ):
                        raise ValueError("invalid saved retrieval samples")
                for key in ("first_pass", "retry"):
                    attempt = original[key]
                    q[key] = (
                        {field: attempt[field] for field in _ATTEMPT_FIELDS if field in attempt}
                        if attempt is not None
                        else None
                    )
                    if attempt is None:
                        continue
                    if not math.isfinite(attempt["score"]) or not 0 <= attempt["score"] <= 1:
                        raise ValueError("invalid saved score")
                    q[key]["judgments"] = [
                        {
                            field: j[field]
                            for field in ("rubric_index", "criterion", "score", "reason")
                        }
                        for j in attempt.get("judgments", [])
                    ]
                    q[key]["metadata_projection"] = bool(
                        attempt.get("answer_call", {}).get("metadata_projection")
                    )
                first = q.pop("first_pass")
                recorded = q.pop("retry")
                q["selected_answer"] = recorded or first
                records.append(q)
        if len(records) != arm["questions"] or len({q["question_id"] for q in records}) != len(
            records
        ):
            raise ValueError("question coverage is incomplete or duplicated")
        arms[name] = {
            "results": records,
            "summary": _summary(records, dataset),
        }
    primary = {q["question_id"]: q for q in arms["oam"]["results"]}
    for arm in arms.values():
        other = {q["question_id"]: q for q in arm["results"]}
        if set(other) != set(primary) or any(
            any(q[field] != other[qid][field] for field in ("question", "gold_answers", "rubric"))
            for qid, q in primary.items()
        ):
            raise ValueError("comparison question, gold or rubric differs")
    return {"dataset": dataset, "manifest": manifest, "arms": arms}


def render_historical_report(root: Path) -> bytes:
    """Render all question evidence inline; no server or network fetch is required."""
    data = _evidence(Path(root))
    assets = resources.files("oamb.reporting").joinpath("assets")
    css = assets.joinpath("report.css").read_text() + assets.joinpath("historical.css").read_text()
    script = assets.joinpath("historical.js").read_text()
    # Paired arms share exact contexts; embed each context once and restore in browser.
    if data["dataset"] == "lme500" and len(data["arms"]) > 1:
        primary = {q["question_id"]: q for q in data["arms"]["oam"]["results"]}
        for name, arm in data["arms"].items():
            if name == "oam":
                continue
            for q in arm["results"]:
                if "context" in q and q["context"] == primary[q["question_id"]].get("context"):
                    del q["context"]
                    q["context_ref"] = q["question_id"]
    css_hash = base64.b64encode(hashlib.sha256(css.encode()).digest()).decode()
    script_hash = base64.b64encode(hashlib.sha256(script.encode()).digest()).decode()
    payload = base64.b64encode(
        gzip.compress(
            json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(),
            compresslevel=9,
            mtime=0,
        )
    ).decode()
    title = "LME-500" if data["dataset"] == "lme500" else "BEAM-10M"
    mean = data["arms"]["oam"]["summary"]["mean"]
    csp = (
        "default-src 'none'; connect-src 'none'; object-src 'none'; base-uri 'none'; "
        f"script-src 'sha256-{script_hash}'; style-src 'sha256-{css_hash}'; form-action 'none'"
    )
    page = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark"><meta http-equiv="Content-Security-Policy" content="{html.escape(csp, quote=True)}">
<title>{title} · OAM evaluation report</title><style>{css}</style></head>
<body class="historical"><header class="topbar"><a href="../README.md">OAMB</a><span>/</span><strong>{title}</strong><span class="top-note">Recorded evaluation</span><a class="report-link" href="README.md">Report &amp; provenance ↗</a></header>
<div id="app"><aside><div class="side-head"><p class="eyebrow">OAM · {title}</p><h1>{mean:.2%}</h1><p id="loading" role="status">Opening saved question evidence…</p></div></aside><main><noscript>Enable JavaScript to browse the embedded evidence. The complete Markdown report and compressed result files remain available beside this file.</noscript></main></div>
<script id="evidence" type="application/octet-stream">{payload}</script><script>{script}</script></body></html>'''
    body = page.encode()
    if len(body) > _MAX_REPORT_BYTES:
        raise ValueError("historical HTML exceeds the 16 MiB artifact limit")
    return body


def build_historical_report(root: Path) -> Path:
    """Update the derived HTML from saved public evidence."""
    root = Path(root)
    body = render_historical_report(root)
    target = root / "report.html"
    atomic_replace_bytes(target, body)
    return target
