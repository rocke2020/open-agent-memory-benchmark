"""Strict read-only import of the pinned AMB LongMemEval historical result."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from importlib import resources
from pathlib import Path
from typing import Any, cast

from oamb.contracts.external import (
    ExternalCompatibilityAssessment,
    ExternalHistoricalAggregate,
    ExternalHistoricalCase,
    ExternalHistoricalCategoryAggregate,
    ExternalHistoricalEvidence,
    ExternalTransformationRecord,
    HistoricalCategory,
    external_historical_evidence_id,
)
from oamb.contracts.ids import canonical_json_bytes, canonical_sha256

from .pins import (
    AMB_PRODUCER_BASE_REVISION,
    AMB_PRODUCER_CODE_REVISION,
    AMB_SOURCE_ATTESTATION_REVISION,
    AMB_SOURCE_ATTESTATION_SHA256,
    AMB_SOURCE_BYTE_COUNT,
    AMB_SOURCE_SHA256,
    CURATED_IMPORTED_AT,
)
from .pins import CURATED_EVIDENCE_PACK_SHA256 as PINNED_CURATED_EVIDENCE_PACK_SHA256

SOURCE_PATH = "outputs/longmemeval/hindsight-deepseek/rag/s.json"
SOURCE_ATTESTATION_PATH = "eval_analysis/evidence/official-comparison.json"
CURATED_EVIDENCE_PACK_SHA256 = PINNED_CURATED_EVIDENCE_PACK_SHA256

_ROOT_FIELDS = frozenset(
    {
        "accuracy",
        "answer_llm",
        "avg_context_tokens",
        "avg_retrieve_time_ms",
        "category",
        "correct",
        "dataset",
        "description",
        "ingested_docs",
        "ingestion_time_ms",
        "judge_llm",
        "memory_provider",
        "mode",
        "oracle",
        "results",
        "run_name",
        "split",
        "total_queries",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "answer",
        "category_axes",
        "context",
        "context_tokens",
        "correct",
        "gold_answers",
        "judge_reason",
        "meta",
        "query",
        "query_id",
        "raw_response",
        "reasoning",
        "retrieve_time_ms",
    }
)
_META_FIELDS = frozenset({"question_type", "query_timestamp"})
_CATEGORY_AXIS_FIELDS = frozenset({"Question Type"})
_CATEGORY_ORDER: tuple[HistoricalCategory, ...] = (
    "knowledge-update",
    "multi-session",
    "single-session-assistant",
    "single-session-preference",
    "single-session-user",
    "temporal-reasoning",
)
_BIDI_CONTROLS = frozenset("\u200e\u200f\u202a\u202b\u202c\u202d\u202e\u2066\u2067\u2068\u2069")


def importer_implementation_hash() -> str:
    """Bind the exact factual projection algorithm, independent of caller input."""

    return canonical_sha256(
        [
            "oamb-amb-factual-importer-implementation-v1",
            tuple(sorted(_ROOT_FIELDS)),
            tuple(sorted(_RESULT_FIELDS)),
            tuple(sorted(_META_FIELDS)),
            tuple(sorted(_CATEGORY_AXIS_FIELDS)),
            _CATEGORY_ORDER,
            inspect.getsource(_project_amb_case),
            inspect.getsource(_project_amb_document),
        ]
    )


def import_historical_amb_result(source_path: Path) -> ExternalHistoricalEvidence:
    """Import only the pinned factual fields after verifying the complete raw bytes."""

    source_path = Path(source_path)
    source_byte_count, source_sha256 = _hash_file(source_path)
    if source_sha256 != AMB_SOURCE_SHA256:
        raise ValueError("historical AMB source SHA-256 does not match the fixed profile")
    if source_byte_count != AMB_SOURCE_BYTE_COUNT:
        raise ValueError("historical AMB source byte count does not match the fixed profile")
    with source_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle, parse_float=Decimal)
    if not isinstance(document, dict):
        raise ValueError("historical AMB source root must be an object")
    return _project_amb_document(document)


def load_curated_evidence(path: Path) -> ExternalHistoricalEvidence:
    record = ExternalHistoricalEvidence.model_validate_json(Path(path).read_bytes())
    require_curated_evidence(record)
    return record


def load_packaged_curated_evidence() -> ExternalHistoricalEvidence:
    content = (
        resources.files("oamb.external_evidence")
        .joinpath("data", "amb-longmemeval-historical-v1.json")
        .read_bytes()
    )
    record = ExternalHistoricalEvidence.model_validate_json(content)
    return require_curated_evidence(record)


def require_curated_evidence(
    record: ExternalHistoricalEvidence,
) -> ExternalHistoricalEvidence:
    expected = (
        record.origin_class.value == "external_amb_generated"
        and record.producer_repository == "rocke2020/agent-memory-benchmark"
        and record.producer_code_revision == AMB_PRODUCER_CODE_REVISION
        and record.producer_base_revision == AMB_PRODUCER_BASE_REVISION
        and record.producer_protocol == "amb-longmemeval-rag"
        and record.imported_at == datetime.fromisoformat(CURATED_IMPORTED_AT)
        and record.source_path == SOURCE_PATH
        and record.source_sha256 == AMB_SOURCE_SHA256
        and record.source_byte_count == AMB_SOURCE_BYTE_COUNT
        and record.source_attestation_path == SOURCE_ATTESTATION_PATH
        and record.source_attestation_sha256 == AMB_SOURCE_ATTESTATION_SHA256
        and record.source_attestation_revision == AMB_SOURCE_ATTESTATION_REVISION
        and record.transformation.importer_implementation_hash == importer_implementation_hash()
        and canonical_sha256(record) == CURATED_EVIDENCE_PACK_SHA256
    )
    if not expected:
        raise ValueError("external curated evidence pack hash or provenance does not match")
    return record


def _project_amb_document(document: Mapping[str, Any]) -> ExternalHistoricalEvidence:
    _require_exact_fields(document, _ROOT_FIELDS, "AMB root")
    _require_root_metadata(document)
    results = document["results"]
    if not isinstance(results, list):
        raise ValueError("AMB results must be an array")
    cases = tuple(_project_amb_case(row) for row in results)
    aggregate = _aggregate(cases)
    if (
        document["total_queries"] != aggregate.total_cases
        or document["correct"] != aggregate.correct_cases
        or document["accuracy"] != Decimal("0.896")
        or document["avg_context_tokens"] != Decimal("49625.2")
        or document["avg_retrieve_time_ms"] != Decimal("631")
    ):
        raise ValueError("AMB top-level aggregate does not match the pinned producer view")
    category_aggregates = tuple(
        _category_aggregate(category, cases) for category in _CATEGORY_ORDER
    )
    transformation = ExternalTransformationRecord(
        algorithm="amb-factual-projection-v1",
        allowed_case_fields=(
            "case_id",
            "category",
            "verdict",
            "context_tokens",
            "retrieval_time_ms",
        ),
        allowed_aggregate_fields=(
            "total_cases",
            "correct_cases",
            "context_tokens_total",
            "retrieval_time_ms_total",
        ),
        unchanged_source_root_sha256=AMB_SOURCE_SHA256,
        importer_implementation_hash=importer_implementation_hash(),
    )
    provisional = ExternalHistoricalEvidence.model_construct(
        external_evidence_id="0" * 64,
        origin_class="external_amb_generated",
        producer_repository="rocke2020/agent-memory-benchmark",
        producer_code_revision=AMB_PRODUCER_CODE_REVISION,
        producer_base_revision=AMB_PRODUCER_BASE_REVISION,
        producer_protocol="amb-longmemeval-rag",
        imported_at=datetime.fromisoformat(CURATED_IMPORTED_AT),
        source_path=SOURCE_PATH,
        source_sha256=AMB_SOURCE_SHA256,
        source_byte_count=AMB_SOURCE_BYTE_COUNT,
        source_attestation_path=SOURCE_ATTESTATION_PATH,
        source_attestation_sha256=AMB_SOURCE_ATTESTATION_SHA256,
        source_attestation_revision=AMB_SOURCE_ATTESTATION_REVISION,
        run_name="hindsight-deepseek",
        dataset="longmemeval",
        split="s",
        memory_provider="hindsight",
        mode="rag",
        oracle=False,
        answer_model="openai:deepseek-v4-pro",
        judge_model="openai:deepseek-v4-flash",
        compatibility=ExternalCompatibilityAssessment(
            target_protocol="oamb-longmemeval-protocol",
            status="unknown",
            assessment_code="not-evaluated-by-oamb-comparison-predicate",
        ),
        transformation=transformation,
        aggregate=aggregate,
        category_aggregates=category_aggregates,
        cases=cases,
        oamb_attempt_ledger_present=False,
        comparison_eligible=False,
        billing_complete=False,
        cost_complete=False,
        limitation_codes=(
            "no-oamb-attempt-ledger",
            "indexing-usage-unavailable",
            "external-llm-usage-unavailable",
            "prompt-compatibility-unknown",
            "amb-context-view-not-oamb-context-view",
            "no-causal-attribution",
        ),
    )
    fields = provisional.model_dump(
        mode="python",
        exclude={"schema_name", "schema_version", "external_evidence_id"},
    )
    return ExternalHistoricalEvidence.model_validate(
        {
            **provisional.model_dump(mode="python"),
            "external_evidence_id": external_historical_evidence_id(**fields),
        }
    )


def _project_amb_case(document: Mapping[str, Any]) -> ExternalHistoricalCase:
    _require_exact_fields(document, _RESULT_FIELDS, "AMB result")
    meta = document["meta"]
    axes = document["category_axes"]
    if not isinstance(meta, dict) or not isinstance(axes, dict):
        raise ValueError("AMB result metadata and category axes must be objects")
    _require_exact_fields(meta, _META_FIELDS, "AMB result metadata")
    _require_exact_fields(axes, _CATEGORY_AXIS_FIELDS, "AMB result category axes")
    category = meta["question_type"]
    if category not in _CATEGORY_ORDER or axes["Question Type"] != [category]:
        raise ValueError("AMB result category metadata does not agree")
    case_id = document["query_id"]
    context_tokens = document["context_tokens"]
    correct = document["correct"]
    retrieval_time = document["retrieve_time_ms"]
    if not isinstance(case_id, str) or not case_id:
        raise ValueError("AMB result case ID must be nonempty text")
    _reject_bidi(case_id)
    if type(context_tokens) is not int or context_tokens < 0:
        raise ValueError("AMB result context tokens must be a non-negative integer")
    if type(correct) is not bool:
        raise ValueError("AMB result verdict must be boolean")
    if not isinstance(retrieval_time, (int, Decimal)) or isinstance(retrieval_time, bool):
        raise ValueError("AMB result retrieval timing must be numeric")
    return ExternalHistoricalCase(
        case_id=case_id,
        category=cast(HistoricalCategory, category),
        verdict="correct" if correct else "incorrect",
        context_tokens=context_tokens,
        retrieval_time_ms=Decimal(retrieval_time),
    )


def _aggregate(cases: tuple[ExternalHistoricalCase, ...]) -> ExternalHistoricalAggregate:
    return ExternalHistoricalAggregate(
        total_cases=len(cases),
        correct_cases=sum(case.verdict == "correct" for case in cases),
        context_tokens_total=sum(case.context_tokens for case in cases),
        retrieval_time_ms_total=sum((case.retrieval_time_ms for case in cases), Decimal()),
    )


def _category_aggregate(
    category: HistoricalCategory,
    cases: tuple[ExternalHistoricalCase, ...],
) -> ExternalHistoricalCategoryAggregate:
    members = tuple(case for case in cases if case.category == category)
    return ExternalHistoricalCategoryAggregate(
        category=category,
        total_cases=len(members),
        correct_cases=sum(case.verdict == "correct" for case in members),
        context_tokens_total=sum(case.context_tokens for case in members),
        retrieval_time_ms_total=sum((case.retrieval_time_ms for case in members), Decimal()),
    )


def _require_root_metadata(document: Mapping[str, Any]) -> None:
    expected = {
        "dataset": "longmemeval",
        "split": "s",
        "category": None,
        "memory_provider": "hindsight",
        "run_name": "hindsight-deepseek",
        "mode": "rag",
        "oracle": False,
        "ingestion_time_ms": Decimal("14425418.6"),
        "ingested_docs": 5861,
        "description": None,
        "answer_llm": "openai:deepseek-v4-pro",
        "judge_llm": "openai:deepseek-v4-flash",
    }
    if any(document[key] != value for key, value in expected.items()):
        raise ValueError("AMB producer metadata does not match the fixed profile")
    for key in (
        "dataset",
        "split",
        "memory_provider",
        "run_name",
        "mode",
        "answer_llm",
        "judge_llm",
    ):
        _reject_bidi(cast(str, document[key]))


def _require_exact_fields(
    document: Mapping[str, Any], expected: frozenset[str], label: str
) -> None:
    actual = frozenset(document)
    unknown = tuple(sorted(actual - expected))
    missing = tuple(sorted(expected - actual))
    if unknown:
        raise ValueError(f"unknown {label} fields: {unknown!r}")
    if missing:
        raise ValueError(f"missing {label} fields: {missing!r}")


def _reject_bidi(value: str) -> None:
    if any(character in value for character in _BIDI_CONTROLS):
        raise ValueError("external evidence text contains a bidirectional control")


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def canonical_curated_evidence_bytes(record: ExternalHistoricalEvidence) -> bytes:
    return canonical_json_bytes(record)


__all__ = [
    "CURATED_EVIDENCE_PACK_SHA256",
    "canonical_curated_evidence_bytes",
    "import_historical_amb_result",
    "importer_implementation_hash",
    "load_curated_evidence",
    "load_packaged_curated_evidence",
    "require_curated_evidence",
]
