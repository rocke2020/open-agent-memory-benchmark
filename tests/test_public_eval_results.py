from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from oamb.reporting.report_analysis import parse_report_analysis

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLIC_RESULTS_ROOT = REPOSITORY_ROOT / "eval_results"
PUBLIC_ROOT = REPOSITORY_ROOT / "eval_results/v0.1.0"
COMPARISON_NAME = "comparison-20260913-133717-82378"
MAX_TRACKED_FILE_BYTES = 16 * 1024 * 1024
CHECKSUM_EXCLUDED_FILENAMES = frozenset({"README.md", "SHA256SUMS"})
PROVIDER_RESULTS = {
    "hindsight": "hindsight-20260911-57-of-60/results/hindsight.json",
    "mem0": "mem0-lme60-20260913-oamb/results/mem0.json",
    "openviking": "openviking-lme60-20260913-ov-oamb/results/openviking.json",
}


def _checksum_entries(release_root: Path) -> tuple[tuple[str, str], ...]:
    entries: list[tuple[str, str]] = []
    for line in (release_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        assert match is not None
        entries.append((match.group(1), match.group(2)))
    return tuple(entries)


def _is_local_report_output(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    if not parts:
        return False
    return (parts[0].startswith("comparison") and parts[0] != COMPARISON_NAME) or parts[
        0
    ] == "report-analysis-cache"


def test_public_result_checksums_cover_evidence_but_exclude_human_readmes() -> None:
    release_roots = tuple(sorted(path for path in PUBLIC_RESULTS_ROOT.iterdir() if path.is_dir()))

    assert release_roots
    for release_root in release_roots:
        relative_files = tuple(
            path.relative_to(release_root).as_posix()
            for path in sorted(release_root.rglob("*"))
            if path.is_file()
        )
        entries = _checksum_entries(release_root)
        checksummed_files = tuple(relative_path for _digest, relative_path in entries)
        assert checksummed_files == tuple(sorted(checksummed_files))
        unexpected_files = tuple(
            path
            for path in relative_files
            if Path(path).name not in CHECKSUM_EXCLUDED_FILENAMES and path not in checksummed_files
        )
        assert all(_is_local_report_output(path) for path in unexpected_files), release_root.name
        for digest, relative_path in entries:
            assert hashlib.sha256((release_root / relative_path).read_bytes()).hexdigest() == digest


def test_v010_public_results_are_complete_and_portable() -> None:
    entries = _checksum_entries(PUBLIC_ROOT)
    published_paths = tuple(relative_path for _digest, relative_path in entries)
    files = tuple(
        sorted(
            {
                *(PUBLIC_ROOT / relative_path for relative_path in published_paths),
                PUBLIC_ROOT / "README.md",
                PUBLIC_ROOT / "SHA256SUMS",
            }
        )
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
    published_cache_analyses = tuple(
        path for path in published_paths if path.startswith("report-analysis-cache/")
    )
    assert len(published_cache_analyses) == 1

    for provider_id, relative_path in PROVIDER_RESULTS.items():
        result = json.loads((PUBLIC_ROOT / relative_path).read_bytes())
        assert len(result) == 60, provider_id

    comparison_root = PUBLIC_ROOT / COMPARISON_NAME
    report = json.loads((comparison_root / "report.json").read_bytes())
    analysis_payload = (comparison_root / "report-analysis.json").read_bytes()
    cached_analysis = PUBLIC_ROOT / published_cache_analyses[0]

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
    assert cached_analysis.read_bytes() == analysis_payload
    parse_report_analysis(analysis_payload, report)

    public_bytes = b"\n".join(
        path.read_bytes() for path in files if path.suffix in {".json", ".md", ".html"}
    )
    assert b"/Users/" not in public_bytes
    assert b"rocke_dong" not in public_bytes


def test_v010_readme_documents_the_direct_report_command_and_authoritative_results() -> None:
    readme = (PUBLIC_ROOT / "README.md").read_text(encoding="utf-8")

    assert "./run.sh --generate-report --result-dir=./eval_results/v0.1.0" in readme
    assert "https://rocke2020.github.io/open-agent-memory-benchmark/" in readme
    assert "results/hindsight.json" in readme
    assert "results/mem0.json" in readme
    assert "results/openviking.json" in readme
    assert "result-map-first.json" not in readme
    assert "result-map-second.json" not in readme


def test_local_report_outputs_are_outside_the_frozen_release_inventory() -> None:
    assert _is_local_report_output("comparison/report.json")
    assert _is_local_report_output("comparison-20990101-000000-1/report.html")
    assert _is_local_report_output("report-analysis-cache/local/report-analysis.json")
    assert not _is_local_report_output("mem0-lme60-20260913-oamb/results/extra.json")


def test_longmemeval_source_contains_no_stale_private_distribution_metadata() -> None:
    source = b"\n".join(path.read_bytes() for path in (REPOSITORY_ROOT / "src").rglob("*.py"))

    assert b"download-required-not-redistributed" not in source
    assert b"local-only" not in source
