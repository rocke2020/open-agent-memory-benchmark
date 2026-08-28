from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from importlib import resources
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from oamb.contracts.ids import canonical_json_bytes, canonical_sha256
from oamb.contracts.schema import parse_contract
from oamb.contracts.states import ValidationDisposition
from oamb.reporting.offline_renderer import (
    offline_asset_hashes,
    offline_renderer_hash,
    render_offline_report,
)
from oamb.reporting.roots import build_report_spec

FIXTURE = Path(
    str(
        resources.files("oamb.external_evidence").joinpath(
            "data", "amb-longmemeval-historical-v1.json"
        )
    )
)
SOURCE_SHA256 = "4e94268f30bda9dedf45853c37b1aa9b4385c6e8a4121f2eb3141e77c1d4fb23"
SOURCE_BYTES = 306_744_666
EXPECTED_CASES = 500
EXPECTED_CORRECT = 448
EXPECTED_CONTEXT_TOKENS = 24_812_616
EXPECTED_RETRIEVAL_TIME_MS = "315476.1"
EXPECTED_CATEGORIES = {
    "knowledge-update": (78, 75, 3_841_877, "47391.5"),
    "multi-session": (133, 107, 6_606_764, "83130.6"),
    "single-session-assistant": (56, 54, 2_758_246, "40933.3"),
    "single-session-preference": (30, 24, 1_485_501, "19295.6"),
    "single-session-user": (70, 66, 3_439_798, "39371.8"),
    "temporal-reasoning": (133, 122, 6_680_430, "85353.3"),
}


def _module(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pytest.fail(f"{name} is not implemented", pytrace=False)


@pytest.fixture(scope="module")
def record() -> Any:
    return _module("oamb.external_evidence.amb").load_curated_evidence(FIXTURE)


def _report_spec() -> Any:
    return build_report_spec(
        report_kind="run",
        audience="public",
        preview_max_field_bytes=4096,
        preview_total_bytes=65536,
        display_field_ids=("identity", "completion", "quality", "usage", "limitations"),
        renderer_hash=offline_renderer_hash(),
        asset_hashes=offline_asset_hashes(),
        export_profile_selector_id="public-run-v1",
        export_profile_selector_version=1,
    )


def test_curated_pack_preserves_only_allowlisted_facts(record: Any) -> None:
    assert record.origin_class == "external_amb_generated"
    assert record.source_sha256 == SOURCE_SHA256
    assert record.source_byte_count == SOURCE_BYTES
    assert record.compatibility.status == "unknown"
    assert record.comparison_eligible is False
    assert record.billing_complete is False
    assert record.cost_complete is False
    assert record.oamb_attempt_ledger_present is False
    assert record.transformation.algorithm == "amb-factual-projection-v1"
    assert record.transformation.allowed_case_fields == (
        "case_id",
        "category",
        "verdict",
        "context_tokens",
        "retrieval_time_ms",
    )
    assert (
        record.transformation.importer_implementation_hash
        == _module("oamb.external_evidence.amb").importer_implementation_hash()
    )
    assert len(record.cases) == EXPECTED_CASES
    assert len({case.case_id for case in record.cases}) == EXPECTED_CASES
    assert sum(case.verdict == "correct" for case in record.cases) == EXPECTED_CORRECT
    assert sum(case.context_tokens for case in record.cases) == EXPECTED_CONTEXT_TOKENS
    assert str(sum(case.retrieval_time_ms for case in record.cases)) == (EXPECTED_RETRIEVAL_TIME_MS)

    aggregates = {
        item.category: (
            item.total_cases,
            item.correct_cases,
            item.context_tokens_total,
            str(item.retrieval_time_ms_total),
        )
        for item in record.category_aggregates
    }
    assert aggregates == EXPECTED_CATEGORIES

    raw = json.loads(FIXTURE.read_bytes())
    serialized = json.dumps(raw, ensure_ascii=False, sort_keys=True)
    for forbidden in (
        '"question"',
        '"answer"',
        '"gold_answers"',
        '"context"',
        '"judge_reason"',
        '"raw_response"',
        '"reasoning"',
        "oamb" + "-plan",
        "agent-memory" + "-eval",
        "/" + "Users/",
    ):
        assert forbidden not in serialized
    assert not any(
        character in serialized
        for character in "\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069"
    )


def test_curated_pack_is_canonical_and_pinned_by_repository_hash(record: Any) -> None:
    amb = _module("oamb.external_evidence.amb")

    assert amb.load_packaged_curated_evidence() == record
    assert FIXTURE.read_bytes() == canonical_json_bytes(record)
    assert canonical_sha256(record) == amb.CURATED_EVIDENCE_PACK_SHA256

    tampered = record.model_copy(
        update={"source_sha256": "a" * 64, "external_evidence_id": "b" * 64}
    )
    with pytest.raises(ValueError, match="curated evidence pack hash"):
        amb.require_curated_evidence(tampered)


def test_contract_registry_parses_external_evidence_and_report(record: Any) -> None:
    parsed = parse_contract(record.model_dump(mode="json"))

    assert parsed == record


def test_external_evidence_contract_rejects_free_text_unknown_private_and_bidi_fields(
    record: Any,
) -> None:
    document = record.model_dump(mode="python")
    planted_values = (
        ("question", "private question text"),
        ("answer", "private answer text"),
        ("local_path", "/private/tmp/source.json"),
        ("notes", "caller supplied prose"),
    )
    for key, value in planted_values:
        planted = {**document, key: value}
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            type(record).model_validate(planted)

    planted_case = dict(document["cases"][0])
    planted_case["case_id"] += "\u202e"
    planted = {**document, "cases": (planted_case, *document["cases"][1:])}
    with pytest.raises(ValidationError, match="String should match pattern|bidirectional control"):
        type(record).model_validate(planted)

    planted_case = dict(document["cases"][0])
    planted_case["case_id"] = "A" * 18
    planted = {**document, "cases": (planted_case, *document["cases"][1:])}
    with pytest.raises(ValidationError, match="at most 17 characters|String should match pattern"):
        type(record).model_validate(planted)


def test_raw_projector_rejects_unknown_source_fields_before_copying_facts() -> None:
    amb = _module("oamb.external_evidence.amb")
    minimal = {
        "query_id": "abcdef12",
        "category_axes": {"Question Type": ["single-session-user"]},
        "context_tokens": 10,
        "correct": True,
        "meta": {
            "question_type": "single-session-user",
            "query_timestamp": "2023-01-01T00:00:00+00:00",
        },
        "retrieve_time_ms": Decimal("1.5"),
        "answer": "ignored",
        "context": "ignored",
        "gold_answers": ["ignored"],
        "judge_reason": "ignored",
        "query": "ignored",
        "raw_response": {},
        "reasoning": "ignored",
    }

    assert amb._project_amb_case(minimal).case_id == "abcdef12"
    with pytest.raises(ValueError, match="unknown AMB result fields"):
        amb._project_amb_case({**minimal, "private_path": "/private/source"})
    with pytest.raises(ValueError, match="bidirectional control"):
        amb._project_amb_case({**minimal, "query_id": "abcdef12\u2066"})


def test_closed_external_profile_accepts_curated_pack_and_rejects_each_tamper(
    record: Any,
) -> None:
    validation_module = _module("oamb.external_evidence.validation")
    valid = validation_module.validate_external_historical_evidence(record)

    assert valid.disposition == ValidationDisposition.VALIDATED
    assert valid.target_hash == canonical_sha256(record)
    assert valid.required_rule_ids == valid.executed_rule_ids == valid.passed_rule_ids
    assert len(valid.required_rule_ids) == 5

    first = record.cases[0]
    flipped = first.model_copy(
        update={"verdict": "incorrect" if first.verdict == "correct" else "correct"}
    )
    bad_cases = record.model_copy(update={"cases": (flipped, *record.cases[1:])})
    bad_compatibility = record.model_copy(
        update={"compatibility": record.compatibility.model_copy(update={"status": "compatible"})}
    )
    bad_transform = record.model_copy(
        update={
            "transformation": record.transformation.model_copy(
                update={"importer_implementation_hash": "a" * 64}
            )
        }
    )
    bad_usage = record.model_copy(update={"billing_complete": True})
    bad_pack_identity = record.model_copy(update={"external_evidence_id": "a" * 64})

    for planted in (
        bad_cases,
        bad_compatibility,
        bad_transform,
        bad_usage,
        bad_pack_identity,
    ):
        result = validation_module.validate_external_historical_evidence(planted)
        assert result.disposition == ValidationDisposition.INVALID
        assert result.failed_rule_ids


def test_fixed_profile_rejects_every_provenance_and_inventory_drift(record: Any) -> None:
    validation_module = _module("oamb.external_evidence.validation")
    first = record.cases[0]
    second = record.cases[1]
    drifted_category = first.model_copy(update={"category": "multi-session"})
    drifted_context = first.model_copy(update={"context_tokens": first.context_tokens + 1})
    drifted_aggregate = record.aggregate.model_copy(
        update={"context_tokens_total": record.aggregate.context_tokens_total + 1}
    )
    mutations = (
        record.model_copy(update={"producer_repository": "example.invalid/producer"}),
        record.model_copy(update={"producer_code_revision": "a" * 40}),
        record.model_copy(update={"producer_base_revision": "b" * 40}),
        record.model_copy(update={"source_path": "outputs/other.json"}),
        record.model_copy(update={"source_sha256": "c" * 64}),
        record.model_copy(update={"source_byte_count": SOURCE_BYTES - 1}),
        record.model_copy(update={"source_attestation_path": "evidence/other.json"}),
        record.model_copy(update={"source_attestation_sha256": "d" * 64}),
        record.model_copy(update={"source_attestation_revision": "e" * 40}),
        record.model_copy(
            update={
                "transformation": record.transformation.model_copy(
                    update={"importer_implementation_hash": "f" * 64}
                )
            }
        ),
        record.model_copy(update={"external_evidence_id": "1" * 64}),
        record.model_copy(update={"cases": (second, first, *record.cases[2:])}),
        record.model_copy(update={"cases": record.cases[:-1]}),
        record.model_copy(update={"cases": (first, first, *record.cases[2:])}),
        record.model_copy(update={"cases": (drifted_context, *record.cases[1:])}),
        record.model_copy(update={"cases": (drifted_category, *record.cases[1:])}),
        record.model_copy(update={"aggregate": drifted_aggregate}),
    )

    for planted in mutations:
        result = validation_module.validate_external_historical_evidence(planted)
        assert result.disposition == ValidationDisposition.INVALID
        assert result.failed_rule_ids


def test_reducer_emits_external_only_report_without_native_evidence(record: Any) -> None:
    validation_module = _module("oamb.external_evidence.validation")
    reducer = _module("oamb.external_evidence.reduce")
    validation = validation_module.validate_external_historical_evidence(record)
    report = reducer.reduce_external_historical_report(
        record,
        validation,
        report_spec=_report_spec(),
    )

    assert report.origin_class == "external_amb_generated"
    assert report.external_evidence_id == record.external_evidence_id
    assert report.producer_code_revision == record.producer_code_revision
    assert report.producer_base_revision == record.producer_base_revision
    assert report.source_sha256 == SOURCE_SHA256
    assert report.source_byte_count == SOURCE_BYTES
    assert report.source_attestation_sha256 == record.source_attestation_sha256
    assert report.source_attestation_revision == record.source_attestation_revision
    assert report.importer_implementation_hash == record.transformation.importer_implementation_hash
    assert report.compatibility_status == "unknown"
    assert report.comparison_eligible is False
    assert report.billing_complete is False
    assert report.cost_complete is False
    assert report.total_cases == EXPECTED_CASES
    assert report.correct_cases == EXPECTED_CORRECT
    assert (report.accuracy.numerator, report.accuracy.denominator) == (112, 125)
    assert report.amb_formatted_view_context_tokens_total == EXPECTED_CONTEXT_TOKENS
    assert (
        report.amb_formatted_view_context_tokens_mean.numerator,
        report.amb_formatted_view_context_tokens_mean.denominator,
    ) == (6_203_154, 125)
    assert str(report.retrieval_time_ms_total) == EXPECTED_RETRIEVAL_TIME_MS
    assert str(report.retrieval_time_ms_mean) == "630.9522"
    assert report.cases == record.cases
    assert len(report.record_projections) == EXPECTED_CASES
    assert len({item.record_id for item in report.record_projections}) == EXPECTED_CASES
    first_projection = report.record_projections[0]
    assert first_projection.record_id == (
        "d1f5c2a834acffebceb66cf848a1c95c0ff270a4862d4da4e4f9851b62902be7"
    )
    assert first_projection.axis == "case"
    assert first_projection.status == "judged"
    assert first_projection.evaluation_status == "judged"
    assert first_projection.verdict == "correct"
    assert first_projection.capabilities_or_types == ("single-session-user",)
    assert first_projection.metric_ids == ("amb-historical-judge-verdict-v1",)
    assert first_projection.proof_statuses == ()
    assert first_projection.raw_evidence_present is False
    assert first_projection.latency_microseconds == 727_300
    assert first_projection.context_view_tokens is None
    assert first_projection.declared_usage is None
    assert first_projection.detail_items == (
        ("source_order", "1"),
        ("producer_case_id", "e47becba"),
        ("category", "single-session-user"),
        ("verdict", "correct"),
        ("amb_formatted_view_context_tokens", "48453"),
        ("retrieval_time_ms", "727.3"),
        ("compatibility_status", "unknown"),
    )
    assert not hasattr(report, "capsule_id")
    assert not hasattr(report, "ingestion_occurrence_ids")
    assert not hasattr(report, "attempt_ids")
    assert not hasattr(report, "measurement_lines")
    assert not hasattr(report, "indexing_usage")
    rendered = render_offline_report(report)
    assert b'"origin_class":"external_amb_generated"' in rendered
    assert b"No OAMB attempt ledger exists for this external producer run." in rendered
    assert b"no-oamb-attempt-ledger" not in rendered.split(b'<script type="application/json"')[0]

    with pytest.raises(ValidationError, match="exactly one source binding"):
        type(report).model_validate(
            {**report.model_dump(mode="python"), "ordered_source_bindings": ()}
        )
    with pytest.raises(ValidationError, match="case projections do not close"):
        type(report).model_validate(
            {
                **report.model_dump(mode="python"),
                "record_projections": report.record_projections[:-1],
            }
        )


def test_reducer_requires_fresh_validation_and_recomputes_exact_report(record: Any) -> None:
    validation_module = _module("oamb.external_evidence.validation")
    reducer = _module("oamb.external_evidence.reduce")
    validation = validation_module.validate_external_historical_evidence(record)
    spec = _report_spec()
    expected = reducer.reduce_external_historical_report(record, validation, report_spec=spec)

    assert (
        reducer.reduce_external_historical_report(
            record,
            validation,
            report_spec=spec,
        )
        == expected
    )

    stale = record.model_copy(update={"cases": tuple(reversed(record.cases))})
    with pytest.raises(ValueError, match="fresh validation"):
        reducer.reduce_external_historical_report(stale, validation, report_spec=spec)


def test_importer_rejects_wrong_raw_source_hash_without_parsing(tmp_path: Path) -> None:
    amb = _module("oamb.external_evidence.amb")
    source = tmp_path / "s.json"
    source.write_bytes(b'{"private_path":"/private/source"}')

    with pytest.raises(ValueError, match="source SHA-256"):
        amb.import_historical_amb_result(source)


def test_historical_cli_rejects_tampered_input_without_writing_output(
    record: Any,
    tmp_path: Path,
) -> None:
    cli = _module("oamb.historical_cli")
    tampered = record.model_copy(
        update={
            "aggregate": record.aggregate.model_copy(
                update={"correct_cases": record.aggregate.correct_cases - 1}
            )
        }
    )
    input_path = tmp_path / "tampered.json"
    input_path.write_bytes(canonical_json_bytes(tampered))
    output_path = tmp_path / "validation.json"

    result = CliRunner().invoke(
        cli.external_app,
        ["validate", "--input", str(input_path), "--output", str(output_path)],
    )

    assert result.exit_code != 0
    assert "external evidence input is invalid" in result.output
    assert "Traceback" not in result.output
    assert not output_path.exists()


def test_external_report_publication_reduces_fresh_source(record: Any, tmp_path: Path) -> None:
    validation_module = _module("oamb.external_evidence.validation")
    reducer = _module("oamb.external_evidence.reduce")
    publication = _module("oamb.reporting.publication")
    validation = validation_module.validate_external_historical_evidence(record)
    spec = _report_spec()
    report = reducer.reduce_external_historical_report(record, validation, report_spec=spec)
    result = publication.build_report_derivation(
        model=report,
        report_spec=spec,
        ordered_source_bindings=report.ordered_source_bindings,
        evidence_validations=(validation,),
        evidence_validation_targets=(record,),
        transform_spec_hash=canonical_sha256(
            ["amb-historical-report-transform-v1", report.reducer_binding]
        ),
        schema_versions=(
            "external_historical_evidence_report@1",
            "report_artifact_manifest@2",
        ),
        output_root=tmp_path,
        committed_at=datetime(2026, 8, 29, tzinfo=UTC),
    )

    assert result.export_validation.disposition == ValidationDisposition.VALIDATED

    forged = report.model_copy(update={"correct_cases": EXPECTED_CORRECT - 1})
    with pytest.raises(publication.ReportExportError):
        publication.build_report_derivation(
            model=forged,
            report_spec=spec,
            ordered_source_bindings=forged.ordered_source_bindings,
            evidence_validations=(validation,),
            evidence_validation_targets=(record,),
            transform_spec_hash=canonical_sha256(
                ["amb-historical-report-transform-v1", forged.reducer_binding]
            ),
            schema_versions=(
                "external_historical_evidence_report@1",
                "report_artifact_manifest@2",
            ),
            output_root=tmp_path / "forged",
            committed_at=datetime(2026, 8, 29, tzinfo=UTC),
        )
