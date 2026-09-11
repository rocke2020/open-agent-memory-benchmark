"""Regenerate HTML from saved report data and compare it with the original.

Run from the repository with the locked environment::

    uv run --locked python tests/reporting/regenerate_offline_report.py \
        path/to/comparison --output path/to/report.regenerated.html

Inputs are report.json, report.html, and report-analysis.json when present.
No dataset, provider service, model credentials, or sibling run is needed.
The output must be a new file; the original comparison remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from oamb.reporting.comparison_project import REPORT_EXPORT_NAME, REPORT_HTML_NAME, _render_html
from oamb.reporting.report_analysis import REPORT_ANALYSIS_NAME, parse_report_analysis


def _freeze_arrays(value: Any) -> Any:
    """Restore the tuple collections expected by the existing renderer."""
    if isinstance(value, list):
        return tuple(_freeze_arrays(item) for item in value)
    if isinstance(value, dict):
        return {key: _freeze_arrays(item) for key, item in value.items()}
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "comparison_dir", type=Path, help="Directory containing saved report files."
    )
    parser.add_argument("--output", type=Path, required=True, help="New HTML file to write.")
    args = parser.parse_args()

    try:
        export = _freeze_arrays(json.loads((args.comparison_dir / REPORT_EXPORT_NAME).read_bytes()))
        if not isinstance(export, dict):
            raise ValueError("report.json must contain an object")
        analysis_path = args.comparison_dir / REPORT_ANALYSIS_NAME
        analysis = (
            parse_report_analysis(analysis_path.read_bytes(), export)
            if analysis_path.exists()
            else None
        )
        rendered = _render_html(export, analysis)
        original = (args.comparison_dir / REPORT_HTML_NAME).read_bytes()
        with args.output.open("xb") as output:
            output.write(rendered)
    except (OSError, ValueError, TypeError, KeyError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"Generated: {args.output}")
    print(f"SHA-256: {hashlib.sha256(rendered).hexdigest()}")
    if rendered != original:
        print("FAIL: regenerated HTML differs from the saved report.html", file=sys.stderr)
        return 1
    print("PASS: regenerated HTML is byte-identical to the saved report.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
