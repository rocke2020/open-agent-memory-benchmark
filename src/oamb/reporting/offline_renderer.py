"""Deterministic single-file renderer for typed offline report models."""

from __future__ import annotations

import base64
import hashlib
import html
import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from importlib import resources
from typing import TypeAlias, cast

from pydantic import BaseModel, RootModel

from oamb.contracts.base import canonical_decimal_text
from oamb.contracts.external import ExternalHistoricalEvidenceReport
from oamb.contracts.ids import canonical_sha256
from oamb.contracts.reporting import (
    ComparisonReportModel,
    DiagnosticRunReportModel,
    EvaluationReportModel,
    ReleaseReportModel,
    RunReportModelV3,
)
from oamb.external_evidence.limitations import historical_limitation_texts
from oamb.reporting.acceptance_contracts import (
    BROWSER_ACCEPTANCE_CONTRACT,
    PERFORMANCE_ACCEPTANCE_CONTRACT,
    browser_acceptance_contract_hash,
    performance_acceptance_contract_hash,
)

OfflineReportModel: TypeAlias = (
    RunReportModelV3
    | ExternalHistoricalEvidenceReport
    | DiagnosticRunReportModel
    | ComparisonReportModel
    | EvaluationReportModel
    | ReleaseReportModel
)


def render_offline_report(model: OfflineReportModel) -> bytes:
    """Render one immutable typed model without recomputing any claim."""

    return _render_offline_report(
        model,
        canonical_model_bytes=_canonical_display_model_bytes(model),
    )


def _render_offline_report(
    model: OfflineReportModel,
    *,
    canonical_model_bytes: bytes,
) -> bytes:
    """Render with already canonicalized bytes for bounded performance measurement."""

    css = _asset_text("report-v1.css")
    script = _asset_text("report-v1.js")
    origin: str
    if isinstance(model, (RunReportModelV3, DiagnosticRunReportModel)):
        origin = model.origin_kind
    elif isinstance(model, ExternalHistoricalEvidenceReport):
        origin = model.origin_class.value
    else:
        origin = "derived"
    diagnostic_banner = (
        '<section class="diagnostic" role="alert">DIAGNOSTIC — INVALID EVIDENCE. '
        "No final quality or cost claims are present.</section>"
        if isinstance(model, DiagnosticRunReportModel)
        else ""
    )
    quality_summary = (
        "Quality claims are unavailable because evidence validation failed."
        if isinstance(model, DiagnosticRunReportModel)
        else "Quality state is read from the typed model and is never derived here."
    )
    usage_summary = (
        "Usage and cost claims are unavailable because evidence validation failed."
        if isinstance(model, DiagnosticRunReportModel)
        else "Measurement domains and proof states remain separate."
    )
    validation_summary = _static_validation_summary(model)
    identity_summary = _static_identity_summary(model)
    completion_summary = _static_completion_summary(model)
    quality_items = _static_quality_items(model)
    measurement_items = _static_measurement_items(model)
    limitations = (
        "".join(f"<li>{html.escape(limitation)}</li>" for limitation in _limitations(model))
        or "<li>None declared.</li>"
    )
    csp = "; ".join(
        (
            "default-src 'none'",
            "connect-src 'none'",
            "object-src 'none'",
            "base-uri 'none'",
            "form-action 'none'",
            "font-src 'none'",
            "img-src data:",
            f"style-src 'sha256-{_csp_hash(css)}'",
            f"script-src 'sha256-{_csp_hash(script)}'",
        )
    )
    report_json = _safe_json(canonical_model_bytes)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light dark">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<title>OAMB offline evidence report</title>
<style id="oamb-report-style">{css}</style>
</head>
<body>
<main>
<h1>OAMB offline evidence report</h1>
{diagnostic_banner}
<section id="identity" aria-labelledby="identity-heading">
<h2 id="identity-heading">Identity and origin</h2>
<p>Typed report: <code id="oamb-report-kind">{html.escape(model.schema_name)}</code></p>
<p>Report ID: <code id="oamb-report-id">{model.report_id}</code></p>
<p>Evidence origin: <strong id="oamb-origin-kind">{origin}</strong></p>
<div id="oamb-identity-summary">{identity_summary}</div>
</section>
<section id="validation" aria-labelledby="validation-heading">
<h2 id="validation-heading">Validation and claim boundary</h2>
<p id="oamb-validation-summary">{validation_summary}</p>
<p id="oamb-browser-contract">{_browser_contract_summary()}</p>
<p id="oamb-performance-contract">{_performance_contract_summary()}</p>
<p class="muted">All displayed claims are precomputed in the sealed report model.</p>
</section>
<section id="completion" aria-labelledby="completion-heading">
<h2 id="completion-heading">Execution completeness</h2>
<p id="oamb-completion-summary">{completion_summary}</p>
<div class="controls" aria-label="Report navigation controls">
<label>Filter visible records<input id="oamb-filter" type="search" autocomplete="off"></label>
<label>Record axis<select id="oamb-axis-filter"><option value="all">All axes</option><option value="report">Report</option><option value="logical-context">Logical context</option><option value="plan">Ingestion plan</option><option value="case">Case</option><option value="attempt">Attempt</option></select></label>
<label>Terminal status<select id="oamb-status-filter" aria-label="Terminal status" multiple size="1"><option value="all" selected>All statuses</option></select></label>
<label>Failure stage<select id="oamb-failure-filter" aria-label="Failure stage" multiple size="1"><option value="all" selected>All failure stages</option></select></label>
<label>Evaluation status<select id="oamb-evaluation-filter" aria-label="Evaluation status" multiple size="1"><option value="all" selected>All evaluation states</option></select></label>
<label>Verdict<select id="oamb-verdict-filter" aria-label="Verdict" multiple size="1"><option value="all" selected>All verdicts</option></select></label>
<label>Capability or type<select id="oamb-capability-filter" aria-label="Capability or type" multiple size="1"><option value="all" selected>All capabilities and types</option></select></label>
<label>Metric<select id="oamb-metric-filter" aria-label="Metric" multiple size="1"><option value="all" selected>All metrics</option></select></label>
<label>Proof status<select id="oamb-proof-filter" aria-label="Proof status" multiple size="1"><option value="all" selected>All proof states</option></select></label>
<label>Raw evidence<select id="oamb-raw-filter" aria-label="Raw evidence" multiple size="1"><option value="all" selected>Any raw evidence</option><option value="present">Raw present</option><option value="absent">Raw absent</option></select></label>
<label>Stable order<select id="oamb-sort"><option value="source">Manifest/source order</option><option value="label">Label</option><option value="status">Terminal status</option><option value="metric">Metric</option><option value="latency">Latency</option><option value="context">Ctx tokens</option><option value="usage">Declared usage</option></select></label>
<button id="oamb-previous" type="button">Previous visible record</button>
<button id="oamb-next" type="button">Next visible record</button>
<button id="oamb-copy" type="button">Copy visible record</button>
</div>
<p id="oamb-copy-status" class="copy-status" role="status"></p>
<div id="oamb-records" class="records" aria-live="polite"></div>
</section>
<section id="quality" aria-labelledby="quality-heading">
<h2 id="quality-heading">Quality and review</h2>
<p class="muted">{quality_summary}</p>
<ul id="oamb-quality-summaries">{quality_items}</ul>
<div id="oamb-mab65" hidden>
<h3>MAB-65 exact reduction</h3>
<p id="oamb-mab65-boundary" class="muted"></p>
<h4>Primary components</h4>
<ol id="oamb-mab65-components"></ol>
<h4>Capabilities</h4>
<ol id="oamb-mab65-capabilities"></ol>
<p id="oamb-mab65-index"></p>
</div>
</section>
<section id="usage-cost" aria-labelledby="usage-cost-heading">
<h2 id="usage-cost-heading">Usage and cost</h2>
<p class="muted">{usage_summary}</p>
<ul id="oamb-measurement-summaries">{measurement_items}</ul>
</section>
<section id="limitations" aria-labelledby="limitations-heading">
<h2 id="limitations-heading">Limitations</h2>
<ul id="oamb-static-limitations">{limitations}</ul>
</section>
<noscript><p id="oamb-nojs-explanation">JavaScript is disabled; this typed evidence summary remains complete.</p></noscript>
<script type="application/json" id="oamb-report-data">{report_json}</script>
<script id="oamb-report-script">{script}</script>
</main>
</body>
</html>
"""
    return document.encode("utf-8")


def _static_validation_summary(model: OfflineReportModel) -> str:
    validation_hashes: tuple[str, ...]
    boundary = None
    if isinstance(model, RunReportModelV3):
        validation_hashes = (model.evidence_validation_result_hash,)
        boundary = model.claim_boundary
    elif isinstance(model, DiagnosticRunReportModel):
        return (
            "Validation: INVALID; diagnostic-only evidence: "
            f"{model.evidence_validation_result_hash}; issue codes: "
            f"{', '.join(model.validation_issue_codes)}."
        )
    elif isinstance(model, ExternalHistoricalEvidenceReport):
        validation_hashes = (model.evidence_validation_result_hash,)
        return (
            "Validation: PASS; external factual profile; comparison eligible: false; "
            "billing complete: false; cost complete: false. Validation evidence: "
            f"{model.evidence_validation_result_hash}."
        )
    elif isinstance(model, (ComparisonReportModel, ReleaseReportModel)):
        validation_hashes = model.ordered_evidence_validation_hashes
        boundary = model.claim_boundary
    elif isinstance(model, EvaluationReportModel):
        validation_hashes = tuple(
            item.evidence_validation_result_hash for item in model.ordered_run_models
        )
    else:
        validation_hashes = (model.evaluation_export_validation_hash,)
    joined = ", ".join(validation_hashes)
    if boundary is None:
        return f"Validation evidence: {joined}."
    return (
        f"Validation: {boundary.status.upper()}; applicable rules: "
        f"{boundary.applicable_rule_count}; executed: {boundary.executed_rule_count}; "
        f"passed: {boundary.passed_rule_count}; failed: {boundary.failed_rule_count}; "
        f"not applicable: {boundary.not_applicable_rule_count}; "
        f"missing: {boundary.missing_rule_count}; comparison eligible: "
        f"{str(boundary.comparison_eligible).lower()}; billing complete: "
        f"{str(boundary.billing_complete).lower()}; cost complete: "
        f"{str(boundary.cost_complete).lower()}. Validation evidence: {joined}."
    )


def _static_identity_summary(model: OfflineReportModel) -> str:
    rows: tuple[tuple[str, str], ...]
    if isinstance(model, RunReportModelV3):
        source = model.ordered_source_bindings[0]
        rows = (
            ("Run ID", model.summary.run_id),
            ("Protocol", model.protocol_id),
            ("Workload", model.workload_id),
            ("Memory system", model.memory_system_id),
            ("Capsule ID", model.capsule_id or "unavailable"),
            ("Source identity", source.source_identity),
            ("Source root", source.source_root_hash),
            ("Source binding", source.binding_id),
            ("Validation result", model.evidence_validation_result_hash),
            ("Report spec", model.report_spec_hash),
        )
    elif isinstance(model, DiagnosticRunReportModel):
        source = model.ordered_source_bindings[0]
        rows = (
            ("Run ID", model.run_id),
            ("Source identity", source.source_identity),
            ("Source root", source.source_root_hash),
            ("Source binding", source.binding_id),
            ("Validation result", model.evidence_validation_result_hash),
            ("Report spec", model.report_spec_hash),
        )
    elif isinstance(model, ExternalHistoricalEvidenceReport):
        source = model.ordered_source_bindings[0]
        rows = (
            ("External evidence ID", model.external_evidence_id),
            ("Producer repository", model.producer_repository),
            ("Producer code revision", model.producer_code_revision),
            ("Producer base revision", model.producer_base_revision),
            ("Producer protocol", model.producer_protocol),
            ("Raw source SHA-256", model.source_sha256),
            ("Raw source bytes", str(model.source_byte_count)),
            ("Source attestation SHA-256", model.source_attestation_sha256),
            ("Source attestation revision", model.source_attestation_revision),
            ("Importer implementation", model.importer_implementation_hash),
            ("Compatibility", model.compatibility_status),
            ("Source root", source.source_root_hash),
            ("Source binding", source.binding_id),
            ("Validation result", model.evidence_validation_result_hash),
            ("Report spec", model.report_spec_hash),
        )
    elif isinstance(model, ComparisonReportModel):
        rows = (
            ("Left source root", model.ordered_source_bindings[0].source_root_hash),
            ("Right source root", model.ordered_source_bindings[1].source_root_hash),
            ("Left run report", model.left_run_report_hash),
            ("Right run report", model.right_run_report_hash),
            ("Report spec", model.report_spec_hash),
        )
    elif isinstance(model, ReleaseReportModel):
        rows = (
            (
                "Ordered source roots",
                ", ".join(binding.source_root_hash for binding in model.ordered_source_bindings),
            ),
            ("Run report roots", ", ".join(model.run_report_hashes)),
            ("Comparison report roots", ", ".join(model.comparison_report_hashes)),
            ("Report spec", model.report_spec_hash),
        )
    elif isinstance(model, EvaluationReportModel):
        rows = (
            ("Phase", model.phase_id),
            ("Run report roots", ", ".join(model.ordered_run_model_hashes)),
            (
                "Comparison report roots",
                ", ".join(model.eligible_comparison_model_hashes),
            ),
            ("Report spec", model.report_spec_hash),
        )
    else:
        raise TypeError(f"unsupported offline report model: {type(model).__name__}")
    return "".join(
        f"<p>{html.escape(label)}: <code>{html.escape(value)}</code></p>" for label, value in rows
    )


def _static_completion_summary(model: OfflineReportModel) -> str:
    if isinstance(model, RunReportModelV3):
        summary = model.summary
        return (
            f"Intended logical contexts: {summary.intended_logical_contexts}; "
            f"intended ingestion plans: {summary.intended_ingestion_plans}; "
            f"ready ingestion plans: {summary.ready_ingestion_plans}; "
            f"intended cases: {summary.intended_cases}; "
            f"terminal: {summary.terminal_cases}; completed: {summary.completed_cases}; "
            f"errored: {summary.errored_cases}; unsupported: {summary.unsupported_cases}; "
            f"cancelled: {summary.cancelled_cases}; budget exceeded: "
            f"{summary.budget_exceeded_cases}; parsed: {summary.parsed_cases}; "
            f"evaluated: {summary.evaluated_cases}; judged: {summary.judged_cases}; "
            f"unjudged: {summary.unjudged_cases}; metric eligible: "
            f"{summary.metric_eligible_cases}."
        )
    if isinstance(model, DiagnosticRunReportModel):
        return "Diagnostic-only run; completion, quality, metrics, usage, and cost are omitted."
    if isinstance(model, ExternalHistoricalEvidenceReport):
        return (
            f"External producer cases: {model.total_cases}; terminal factual results: "
            f"{model.total_cases}; judged verdicts: {model.total_cases}; no OAMB-native "
            "capsule, ingestion plan, or attempt ledger is claimed."
        )
    if isinstance(model, ComparisonReportModel):
        return (
            f"Comparison compatible: {str(model.comparison.comparable).lower()}; "
            f"Predicate count: {len(model.comparison.predicates)}."
        )
    if isinstance(model, ReleaseReportModel):
        return (
            f"Run reports: {len(model.run_report_hashes)}; "
            f"Comparison reports: {len(model.comparison_report_hashes)}."
        )
    if isinstance(model, EvaluationReportModel):
        return (
            f"Run reports: {len(model.ordered_run_models)}; "
            f"unique cases: {model.unique_case_count}; "
            f"system results: {model.system_result_count}; "
            f"eligible comparisons: {len(model.eligible_comparison_models)}."
        )
    raise TypeError(f"unsupported offline report model: {type(model).__name__}")


def _static_quality_items(model: OfflineReportModel) -> str:
    if isinstance(model, DiagnosticRunReportModel):
        return "<li>Unavailable in diagnostic output.</li>"
    if isinstance(model, ExternalHistoricalEvidenceReport):
        rows = [
            "<li>AMB historical judged accuracy: "
            f"{model.accuracy.numerator}/{model.accuracy.denominator}; "
            f"{model.correct_cases}/{model.total_cases}.</li>"
        ]
        rows.extend(
            f"<li>{html.escape(item.category)}: {item.correct_cases}/{item.total_cases}.</li>"
            for item in model.category_aggregates
        )
        return "".join(rows)
    if isinstance(model, EvaluationReportModel):
        return "".join(
            f"<li>{html.escape(run.memory_system_id)} / "
            f"{html.escape(run.workload_id)}: {len(run.metric_summaries)} metric strata.</li>"
            for run in model.ordered_run_models
        )
    if not isinstance(model, RunReportModelV3) or not model.metric_summaries:
        return "<li>No ordinary metric summaries.</li>"
    return "".join(
        "<li>"
        f"{html.escape(item.metric_id)}@{item.metric_version} "
        f"[{html.escape(item.stratum_id)}]: "
        f"{item.value.numerator}/{item.value.denominator}; "
        f"inputs: {item.input_count}; "
        f"{html.escape(item.claim_note)}"
        "</li>"
        for item in model.metric_summaries
    )


def _static_measurement_items(model: OfflineReportModel) -> str:
    if isinstance(model, DiagnosticRunReportModel):
        return "<li>Unavailable in diagnostic output.</li>"
    if isinstance(model, ExternalHistoricalEvidenceReport):
        return (
            "<li>AMB formatted retrieval-view tokens: "
            f"{model.amb_formatted_view_context_tokens_total} total; exact mean "
            f"{model.amb_formatted_view_context_tokens_mean.numerator}/"
            f"{model.amb_formatted_view_context_tokens_mean.denominator}. "
            "This is not OAMB context_view usage.</li>"
            f"<li>Producer retrieval timing: {model.retrieval_time_ms_total} ms total; "
            f"{model.retrieval_time_ms_mean} ms mean.</li>"
            "<li>Indexing usage: unavailable.</li>"
            "<li>Complete external-LLM usage: unavailable.</li>"
        )
    if isinstance(model, EvaluationReportModel):
        count = sum(len(run.measurement_lines) for run in model.ordered_run_models)
        return f"<li>Bound run measurement lines: {count}.</li>"
    if not isinstance(model, RunReportModelV3) or not model.measurement_lines:
        return "<li>No measurement summaries.</li>"
    rows = []
    for item in model.measurement_lines:
        value = (
            f"{item.value.numerator}/{item.value.denominator} {html.escape(item.unit)}"
            if item.value is not None
            else "unavailable"
        )
        currency = f" {html.escape(item.currency)}" if item.currency is not None else ""
        reason = f"; reason: {html.escape(item.reason)}" if item.reason is not None else ""
        rows.append(
            "<li>"
            f"{html.escape(item.dimension_id)} / {html.escape(item.stage)} / "
            f"{html.escape(item.owner_kind)} / {html.escape(item.indexing_view)}: "
            f"{value}{currency}; basis: {html.escape(item.basis)}; "
            f"proof: {html.escape(item.proof_status.value)}; "
            f"source records: {len(item.source_record_ids)}{reason}"
            "</li>"
        )
    return "".join(rows)


def _limitations(model: OfflineReportModel) -> tuple[str, ...]:
    if isinstance(model, ExternalHistoricalEvidenceReport):
        return historical_limitation_texts(model.limitation_codes)
    return model.limitations


def _asset_text(name: str) -> str:
    return resources.files("oamb.reporting").joinpath("assets", name).read_text(encoding="utf-8")


def _csp_hash(content: str) -> str:
    digest = hashlib.sha256(content.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def _safe_json(content: bytes) -> str:
    return (
        content.decode("utf-8")
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _canonical_display_model_bytes(model: OfflineReportModel) -> bytes:
    output = bytearray()
    _append_canonical_display_value(model, output)
    return bytes(output)


def _append_canonical_display_value(value: object, output: bytearray) -> None:
    if isinstance(value, RootModel):
        _append_canonical_display_value(value.root, output)
        return
    if isinstance(value, BaseModel):
        output.extend(b"{")
        first = True
        fields = sorted(
            type(value).model_fields.items(),
            key=lambda item: item[1].serialization_alias or item[1].alias or item[0],
        )
        for name, field in fields:
            field_value = getattr(value, name)
            if field_value is None:
                continue
            key = field.serialization_alias or field.alias or name
            if not first:
                output.extend(b",")
            first = False
            _append_json_string(key, output)
            output.extend(b":")
            _append_canonical_display_value(field_value, output)
        output.extend(b"}")
        return
    if isinstance(value, Enum):
        _append_canonical_display_value(value.value, output)
        return
    if isinstance(value, Decimal):
        _append_json_string(canonical_decimal_text(value), output)
        return
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("datetime must include an explicit timezone offset")
        _append_json_string(value.isoformat(), output)
        return
    if isinstance(value, date):
        _append_json_string(value.isoformat(), output)
        return
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise TypeError("canonical JSON map keys must be strings")
        output.extend(b"{")
        for index, key in enumerate(sorted(value)):
            if index:
                output.extend(b",")
            _append_json_string(key, output)
            output.extend(b":")
            _append_canonical_display_value(value[key], output)
        output.extend(b"}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        output.extend(b"[")
        for index, item in enumerate(value):
            if index:
                output.extend(b",")
            _append_canonical_display_value(item, output)
        output.extend(b"]")
        return
    if value is None or isinstance(value, (bool, int)):
        output.extend(json.dumps(value, separators=(",", ":")).encode("utf-8"))
        return
    if isinstance(value, str):
        _append_json_string(value, output)
        return
    if isinstance(value, float):
        raise TypeError("floating-point values are not canonical OAMB values")
    raise TypeError(f"unsupported canonical display value: {type(value).__name__}")


def _append_json_string(value: str, output: bytearray) -> None:
    output.extend(json.dumps(value, ensure_ascii=False).encode("utf-8"))


def offline_asset_hashes() -> tuple[str, str]:
    hashes = tuple(
        hashlib.sha256(_asset_text(name).encode("utf-8")).hexdigest()
        for name in ("report-v1.css", "report-v1.js")
    )
    return hashes[0], hashes[1]


def offline_renderer_hash() -> str:
    source_hash = hashlib.sha256(
        resources.files("oamb.reporting").joinpath("offline_renderer.py").read_bytes()
    ).hexdigest()
    return canonical_sha256(
        [
            "oamb-offline-renderer-v1",
            source_hash,
            offline_asset_hashes(),
            browser_acceptance_contract_hash(),
            performance_acceptance_contract_hash(),
            "typed-model-no-browser-reduction",
        ]
    )


def _browser_contract_summary() -> str:
    engine_contracts = cast(tuple[dict[str, str], ...], BROWSER_ACCEPTANCE_CONTRACT["engines"])
    engines = ", ".join(
        f"{item['name']} {item['version']} (revision {item['revision']})"
        for item in engine_contracts
    )
    return (
        f"Browser contract: {browser_acceptance_contract_hash()}; "
        f"Playwright {BROWSER_ACCEPTANCE_CONTRACT['playwright_version']}; {engines}; "
        "file:// with zero network requests."
    )


def _performance_contract_summary() -> str:
    return (
        f"Performance contract: {performance_acceptance_contract_hash()}; "
        f"cases: {PERFORMANCE_ACCEPTANCE_CONTRACT['case_counts']}; "
        f"HTML ceiling: {PERFORMANCE_ACCEPTANCE_CONTRACT['max_html_bytes']} bytes; "
        f"10x growth ceiling: {PERFORMANCE_ACCEPTANCE_CONTRACT['max_10x_growth_ratio']}x."
    )


__all__ = [
    "OfflineReportModel",
    "offline_asset_hashes",
    "offline_renderer_hash",
    "render_offline_report",
]
