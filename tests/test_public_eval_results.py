from __future__ import annotations

import json
from pathlib import Path

from oamb.reporting.report_analysis import parse_report_analysis

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = REPOSITORY_ROOT / "eval_results/lme60"
COMPARISON_NAME = "comparison-20260913-133717-82378"
MAX_TRACKED_FILE_BYTES = 16 * 1024 * 1024
PROVIDER_RESULTS = {
    "hindsight": "hindsight-20260911-57-of-60/results/hindsight.json",
    "mem0": "mem0-lme60-20260913-oamb/results/mem0.json",
    "openviking": "openviking-lme60-20260913-ov-oamb/results/openviking.json",
}


def _is_local_report_output(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    if not parts:
        return False
    return (parts[0].startswith("comparison") and parts[0] != COMPARISON_NAME) or parts[
        0
    ] == "report-analysis-cache"


def test_lme60_public_results_are_complete_and_portable() -> None:
    files = tuple(
        path
        for path in sorted(PUBLIC_ROOT.rglob("*"))
        if path.is_file() and not _is_local_report_output(path.relative_to(PUBLIC_ROOT).as_posix())
    )
    relative_files = tuple(path.relative_to(PUBLIC_ROOT).as_posix() for path in files)

    assert "README.md" in relative_files
    assert "case-manifest.json" in relative_files
    assert "LICENSES/LongMemEval-MIT.txt" in relative_files
    assert not any(
        path.name in {"result-map-first.json", "result-map-second.json"} for path in files
    )
    assert all(
        not path.is_symlink() and path.stat().st_size <= MAX_TRACKED_FILE_BYTES for path in files
    )
    for provider_id, relative_path in PROVIDER_RESULTS.items():
        result = json.loads((PUBLIC_ROOT / relative_path).read_bytes())
        assert len(result) == 60, provider_id

    comparison_root = PUBLIC_ROOT / COMPARISON_NAME
    report = json.loads((comparison_root / "report.json").read_bytes())
    analysis_payload = (comparison_root / "report-analysis.json").read_bytes()

    assert report["coverage"] == {
        "cell_count": 3,
        "provider_specific_result_count": 180,
        "unique_case_count": 60,
    }
    assert report["dataset_details"]["distribution_scope"] == "public"
    assert report["dataset_details"]["license_id"] == "MIT"
    assert report["dataset_details"]["payload_policy"] == "redistribution-allowed-under-MIT"
    assert all(
        key not in result
        for cell in report["cells"]
        for result in cell["results"]
        for key in (
            "model_answer",
            "injected_context",
            "judge_decision",
            "answer_unavailable_reason",
        )
    )
    assert all(len(question["provider_results"]) == 3 for question in report["questions"])
    archived_manifest = json.loads((PUBLIC_ROOT / "case-manifest.json").read_bytes())
    assert archived_manifest["manifest_hash"] == report["dataset"]["case_manifest_hash"]
    assert [item["raw_question_id"] for item in archived_manifest["cases"]] == [
        item["raw_question_id"] for item in report["questions"]
    ]
    assert any(
        path.read_bytes() == analysis_payload
        for path in (PUBLIC_ROOT / "report-analysis-cache").glob("*/report-analysis.json")
    )
    parse_report_analysis(analysis_payload, report)

    public_bytes = b"\n".join(
        path.read_bytes() for path in files if path.suffix in {".json", ".md", ".html"}
    )
    assert b"/Users/" not in public_bytes
    assert b"rocke_dong" not in public_bytes
