"""Pure deterministic renderer shared by report construction and export validation."""

from __future__ import annotations

import base64
import hashlib
import html as html_module
from pathlib import Path

from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.reporting import RunReportModelV2


def load_fake_report_css() -> bytes:
    return (Path(__file__).parent / "assets" / "report.css").read_bytes()


def fake_report_renderer_hash(css: bytes) -> str:
    return canonical_sha256(
        [
            "oamb-fake-html-renderer-v1",
            hashlib.sha256(css).hexdigest(),
            "data-script-no-executable-js",
        ]
    )


def render_fake_report_html(
    model: RunReportModelV2,
    css: bytes,
    *,
    diagnostic: bool,
) -> bytes:
    css_text = css.decode("utf-8")
    css_csp_hash = base64.b64encode(hashlib.sha256(css).digest()).decode("ascii")
    csp = (
        "default-src 'none'; connect-src 'none'; object-src 'none'; font-src 'none'; "
        "img-src data:; base-uri 'none'; form-action 'none'; script-src 'none'; "
        f"style-src 'sha256-{css_csp_hash}'"
    )
    report_json = _safe_json_script(canonical_json_bytes(model))
    summary = model.summary
    diagnostic_banner = (
        '<section class="diagnostic" role="status">DIAGNOSTIC — INVALID EVIDENCE</section>'
        if diagnostic
        else ""
    )
    limitations = "".join(f"<li>{html_module.escape(item)}</li>" for item in model.limitations)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="color-scheme" content="light dark">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<title>OAMB offline run report</title>
<style>{css_text}</style>
</head>
<body>
<main>
<h1>OAMB offline run report</h1>
{diagnostic_banner}
<section aria-labelledby="identity-heading"><h2 id="identity-heading">Identity and origin</h2>
<dl><dt>Run</dt><dd>{html_module.escape(summary.run_id)}</dd>
<dt>Source manifest</dt><dd><code>{model.source_manifest_hash}</code></dd></dl></section>
<section aria-labelledby="validation-heading"><h2 id="validation-heading">Validation and claim boundary</h2>
<p><span class="badge">{"DIAGNOSTIC" if diagnostic else "PASS"}</span></p></section>
<section aria-labelledby="completion-heading"><h2 id="completion-heading">Execution completeness</h2>
<dl><dt>Logical contexts</dt><dd>{summary.intended_logical_contexts}</dd>
<dt>Ready ingestion plans</dt><dd>{summary.ready_ingestion_plans}/{summary.intended_ingestion_plans}</dd>
<dt>Terminal cases</dt><dd>{summary.terminal_cases}/{summary.intended_cases}</dd>
<dt>Completed cases</dt><dd>{summary.completed_cases}</dd>
<dt>Errored cases</dt><dd>{summary.errored_cases}</dd>
<dt>Parsed cases</dt><dd>{summary.parsed_cases}</dd>
<dt>Evaluated cases</dt><dd>{summary.evaluated_cases}</dd>
<dt>Judged cases</dt><dd>{summary.judged_cases}</dd>
<dt>Unjudged cases</dt><dd>{summary.unjudged_cases}</dd></dl></section>
<section aria-labelledby="limitations-heading"><h2 id="limitations-heading">Limitations</h2><ul>{limitations}</ul></section>
<noscript><p>This report is fully readable without JavaScript.</p></noscript>
<script type="application/json" id="oamb-report-data">{report_json}</script>
</main>
</body>
</html>
"""
    return document.encode("utf-8")


def _safe_json_script(content: bytes) -> str:
    return (
        content.decode("utf-8")
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


__all__ = [
    "fake_report_renderer_hash",
    "load_fake_report_css",
    "render_fake_report_html",
]
