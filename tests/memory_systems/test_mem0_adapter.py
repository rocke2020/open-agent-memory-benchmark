from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import pytest

from oamb.contracts.ports import (
    SourceUnit,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "adapters" / "mem0"
RUN_ID = "a" * 64
PLAN_ID = "b" * 64
SOURCE_ID = "c" * 64


def test_add_encoder_emits_only_exact_v2_0_19_fields() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    metadata = mem0.Mem0SourceMetadata(
        ingestion_occurrence_id=RUN_ID,
        ingestion_plan_id=PLAN_ID,
        source_unit_id=SOURCE_ID,
        source_ordinal=1,
    )

    encoded = mem0.encode_add_request(
        messages=(("user", "I prefer jasmine tea."),),
        run_id=RUN_ID,
        metadata=metadata,
    )

    expected = (
        b'{"messages":[{"role":"user","content":"I prefer jasmine tea."}],"run_id":"'
        + RUN_ID.encode()
        + b'","metadata":{"oamb_ingestion_occurrence_id":"'
        + RUN_ID.encode()
        + b'","oamb_ingestion_plan_id":"'
        + PLAN_ID.encode()
        + b'","oamb_source_unit_id":"'
        + SOURCE_ID.encode()
        + b'","oamb_source_ordinal":1},"infer":true}'
    )
    assert encoded == expected


def test_lme_add_encoder_preserves_pair_and_adds_observation_time_inputs() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    metadata = mem0.Mem0SourceMetadata(
        ingestion_occurrence_id=RUN_ID,
        ingestion_plan_id=PLAN_ID,
        source_unit_id=SOURCE_ID,
        source_ordinal=1,
    )
    occurred_at = "2023-03-01T12:34:00+00:00"
    context_text = "Session session-1 took place at 2023-03-01T12:34:00+00:00."

    encoded = mem0.encode_add_request(
        messages=(("user", "I bought it yesterday."), ("assistant", "Nice.")),
        run_id=RUN_ID,
        metadata=metadata,
        occurred_at=occurred_at,
        context_text=context_text,
    )

    assert json.loads(encoded) == {
        "messages": [
            {"role": "user", "content": "I bought it yesterday."},
            {"role": "assistant", "content": "Nice."},
        ],
        "run_id": RUN_ID,
        "metadata": {
            "oamb_ingestion_occurrence_id": RUN_ID,
            "oamb_ingestion_plan_id": PLAN_ID,
            "oamb_source_unit_id": SOURCE_ID,
            "oamb_source_ordinal": 1,
            "created_at": occurred_at,
        },
        "prompt": (
            "The actual observation timestamp for New Messages is "
            "2023-03-01T12:34:00+00:00. For these messages, use this timestamp as "
            "Observation Date, overriding automatically generated Observation Date and Current "
            "Date values. Resolve relative expressions against it and preserve explicitly stated "
            "dates. Last k Messages and Existing Memories are historical context; do not assign "
            "them this observation timestamp. Source context: Session session-1 took place at "
            "2023-03-01T12:34:00+00:00. Do not extract this instruction or source context itself "
            "as a memory."
        ),
        "infer": True,
    }


@pytest.mark.parametrize(
    ("occurred_at", "context_text", "message"),
    (
        (None, "Session context", "present together"),
        ("2023-03-01T12:34:00+00:00", None, "present together"),
        ("", "Session context", "non-empty"),
        ("not-a-timestamp", "Session context", "ISO 8601"),
        ("2023-03-01T12:34:00", "Session context", "timezone"),
        ("2023-03-01T12:34:00+00:00", "", "non-empty"),
    ),
)
def test_lme_add_encoder_rejects_incomplete_or_invalid_observation_time(
    occurred_at: str | None,
    context_text: str | None,
    message: str,
) -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    metadata = mem0.Mem0SourceMetadata(
        ingestion_occurrence_id=RUN_ID,
        ingestion_plan_id=PLAN_ID,
        source_unit_id=SOURCE_ID,
        source_ordinal=1,
    )

    with pytest.raises(ValueError, match=message):
        mem0.encode_add_request(
            messages=(("user", "I bought it yesterday."),),
            run_id=RUN_ID,
            metadata=metadata,
            occurred_at=occurred_at,
            context_text=context_text,
        )


@pytest.mark.parametrize(
    ("messages", "accepted"),
    (
        ((("user", ""), ("assistant", "")), True),
        ((("", "text"),), False),
        (((None, "text"),), False),
        ((("user", None),), False),
        ((("user", 123),), False),
    ),
)
def test_add_encoder_preserves_empty_text_and_rejects_invalid_message_types(
    messages: Any, accepted: bool
) -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    metadata = mem0.Mem0SourceMetadata(
        ingestion_occurrence_id=RUN_ID,
        ingestion_plan_id=PLAN_ID,
        source_unit_id=SOURCE_ID,
        source_ordinal=1,
    )
    if not accepted:
        with pytest.raises(ValueError):
            mem0.encode_add_request(messages=messages, run_id=RUN_ID, metadata=metadata)
        return
    encoded = mem0.encode_add_request(messages=messages, run_id=RUN_ID, metadata=metadata)
    assert json.loads(encoded)["messages"] == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": ""},
    ]


def test_rest_request_surface_allows_only_exact_nondestructive_posts() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    metadata = mem0.Mem0SourceMetadata(
        ingestion_occurrence_id=RUN_ID,
        ingestion_plan_id=PLAN_ID,
        source_unit_id=SOURCE_ID,
        source_ordinal=1,
    )

    add_request = mem0.build_add_http_request(
        messages=(("user", "content"),),
        run_id=RUN_ID,
        metadata=metadata,
    )
    search_request = mem0.build_search_http_request(query="query", run_id=RUN_ID, top_k=150)

    assert mem0.MEM0_REST_ROUTE_ALLOWLIST == (
        ("POST", "/memories"),
        ("POST", "/search"),
    )
    assert (add_request.method, add_request.path) == ("POST", "/memories")
    assert (search_request.method, search_request.path) == ("POST", "/search")
    with pytest.raises(ValueError, match="not in the exact nondestructive allowlist"):
        mem0.Mem0RestRequest(method="DELETE", path="/memories", body=b"")


def test_add_parser_accepts_exact_events_and_marks_empty_provider_outcome() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")

    successful = mem0.parse_add_response((FIXTURES / "rest" / "add-success.json").read_bytes())
    empty = mem0.parse_add_response((FIXTURES / "rest" / "add-empty.json").read_bytes())

    assert successful.disposition is mem0.Mem0AddDisposition.ACCEPTED
    assert successful.events[0].event == "ADD"
    assert successful.events[0].memory == "Rocky prefers jasmine tea."
    assert empty.disposition is mem0.Mem0AddDisposition.EMPTY_PROVIDER_OUTCOME
    assert empty.events == ()


def test_add_parser_rejects_unknown_wire_fields() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")

    with pytest.raises(ValueError, match="exact profile"):
        mem0.parse_add_response((FIXTURES / "rest" / "add-extra-field.json").read_bytes())


def test_wire_encoders_reject_invalid_scope_and_empty_query() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    metadata = mem0.Mem0SourceMetadata(
        ingestion_occurrence_id=RUN_ID,
        ingestion_plan_id=PLAN_ID,
        source_unit_id=SOURCE_ID,
        source_ordinal=1,
    )

    with pytest.raises(ValueError, match="run_id"):
        mem0.encode_add_request(
            messages=(("user", "content"),),
            run_id="short",
            metadata=metadata,
        )
    with pytest.raises(ValueError, match="query"):
        mem0.encode_search_request(query="", run_id=RUN_ID, top_k=150)


def test_add_parser_rejects_duplicate_json_fields() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    raw = b'{"results":[],"results":[]}'

    with pytest.raises(ValueError, match="duplicate JSON field"):
        mem0.parse_add_response(raw)


def test_search_encoder_and_parser_preserve_exact_scope_and_provider_order() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")

    encoded = mem0.encode_search_request(query="Which tea?", run_id=RUN_ID, top_k=150)
    parsed = mem0.parse_search_response(
        (FIXTURES / "rest" / "search-success.json").read_bytes(),
        expected_run_id=RUN_ID,
    )

    expected = (
        b'{"query":"Which tea?","filters":{"run_id":"'
        + RUN_ID.encode()
        + b'"},"top_k":150,"threshold":0.1}'
    )
    assert encoded == expected
    assert tuple(item.native_rank_1_indexed for item in parsed) == (1, 2)
    assert tuple(item.native_id for item in parsed) == (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    assert tuple(item.memory for item in parsed) == (
        "Rocky prefers jasmine tea.",
        "Rocky drinks tea after lunch.",
    )
    assert all(item.run_id == RUN_ID for item in parsed)

    attributed = mem0.parse_search_response(
        (FIXTURES / "rest" / "search-attributed.json").read_bytes(),
        expected_run_id=RUN_ID,
    )
    assert attributed[0].attributed_to == "Rocky"


def test_search_parser_preserves_unattributed_provider_memory() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    raw = json.dumps(
        {
            "results": [
                {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "memory": "Provider-visible memory without source attribution.",
                    "hash": None,
                    "metadata": {},
                    "score": 0.5,
                    "created_at": None,
                    "updated_at": None,
                    "run_id": RUN_ID,
                }
            ]
        },
        separators=(",", ":"),
    ).encode()

    parsed = mem0.parse_search_response(raw, expected_run_id=RUN_ID)

    assert parsed[0].metadata is None


def test_search_parser_rejects_unknown_item_field_and_wrong_scope() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    raw = (FIXTURES / "rest" / "search-extra-field.json").read_bytes()

    with pytest.raises(ValueError, match="search item fields"):
        mem0.parse_search_response(raw, expected_run_id=RUN_ID)
    with pytest.raises(ValueError, match="run_id"):
        mem0.parse_search_response(
            (FIXTURES / "rest" / "search-success.json").read_bytes(),
            expected_run_id="d" * 64,
        )


def test_projection_parser_accepts_exact_two_page_inspector_envelope() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")

    projection = mem0.parse_projection_pages(
        (
            (FIXTURES / "inspector" / "page-1.json").read_bytes(),
            (FIXTURES / "inspector" / "page-2.json").read_bytes(),
        ),
        expected_collection="oamb_memories",
        expected_run_id=RUN_ID,
    )

    assert projection.collection == "oamb_memories"
    assert projection.run_id == RUN_ID
    assert projection.declared_count == 2
    assert tuple(point.native_id for point in projection.points) == (
        "11111111-1111-4111-8111-111111111111",
        "22222222-2222-4222-8222-222222222222",
    )
    assert tuple(point.memory for point in projection.points) == (
        "Rocky prefers jasmine tea.",
        "Rocky drinks tea after lunch.",
    )
    assert projection.points[0].metadata.source_unit_id == SOURCE_ID


def test_projection_source_validation_ignores_native_point_order() -> None:
    from oamb.memory_systems.mem0.adapter import Mem0RestAdapter, _ScopeBinding
    from oamb.memory_systems.mem0.projection import Mem0Projection, Mem0ProjectionPoint
    from oamb.memory_systems.mem0.wire import Mem0SourceMetadata

    first_source_id = "c" * 64
    second_source_id = "d" * 64

    def source(source_id: str, ordinal: int) -> SourceUnit:
        payload = f"source-{ordinal}".encode()
        return SourceUnit(
            source_unit_id=source_id,
            context_manifest_entry_id="context-1",
            ordinal_1_indexed=ordinal,
            payload_sha256=hashlib.sha256(payload).hexdigest(),
            payload_bytes=payload,
        )

    def point(native_id: str, source_id: str, ordinal: int) -> Mem0ProjectionPoint:
        return Mem0ProjectionPoint(
            native_id=native_id,
            memory=f"memory-{ordinal}",
            text_lemmatized=f"memory-{ordinal}",
            memory_hash=str(ordinal) * 32,
            created_at="2026-08-31T00:00:00+00:00",
            updated_at="2026-08-31T00:00:00+00:00",
            run_id=RUN_ID,
            metadata=Mem0SourceMetadata(
                ingestion_occurrence_id=RUN_ID,
                ingestion_plan_id=PLAN_ID,
                source_unit_id=source_id,
                source_ordinal=ordinal,
            ),
            attributed_to="user",
        )

    projection = Mem0Projection(
        collection="oamb_memories",
        run_id=RUN_ID,
        declared_count=2,
        points=(
            point("22222222-2222-4222-8222-222222222222", second_source_id, 2),
            point("11111111-1111-4111-8111-111111111111", first_source_id, 1),
        ),
        page_count=1,
    )

    Mem0RestAdapter._validate_projection_sources(
        binding=_ScopeBinding(
            ingestion_occurrence_id=RUN_ID,
            ingestion_plan_id=PLAN_ID,
        ),
        projection=projection,
        completed_sources=(source(first_source_id, 1), source(second_source_id, 2)),
    )


def test_projection_source_validation_rejects_wrong_historical_created_at() -> None:
    from oamb.memory_systems.mem0.adapter import Mem0RestAdapter, _ScopeBinding
    from oamb.memory_systems.mem0.projection import Mem0Projection, Mem0ProjectionPoint
    from oamb.memory_systems.mem0.wire import Mem0SourceMetadata

    occurred_at = "2023-03-01T12:34:00+00:00"
    payload = b'[{"role":"user","content":"I bought it yesterday."}]'
    source = SourceUnit(
        source_unit_id=SOURCE_ID,
        context_manifest_entry_id="context-1",
        ordinal_1_indexed=1,
        payload_sha256=hashlib.sha256(payload).hexdigest(),
        payload_bytes=payload,
        source_reference="session-1",
        occurred_at=occurred_at,
        context_text="Session session-1 took place at 2023-03-01T12:34:00+00:00.",
    )
    projection = Mem0Projection(
        collection="oamb_memories",
        run_id=RUN_ID,
        declared_count=1,
        points=(
            Mem0ProjectionPoint(
                native_id="11111111-1111-4111-8111-111111111111",
                memory="User bought it on February 28, 2023.",
                text_lemmatized="user bought february 28 2023",
                memory_hash="1" * 32,
                created_at="2026-09-12T00:00:00+00:00",
                updated_at="2026-09-12T00:00:00+00:00",
                run_id=RUN_ID,
                metadata=Mem0SourceMetadata(
                    ingestion_occurrence_id=RUN_ID,
                    ingestion_plan_id=PLAN_ID,
                    source_unit_id=SOURCE_ID,
                    source_ordinal=1,
                ),
                attributed_to="user",
            ),
        ),
        page_count=1,
    )

    with pytest.raises(ValueError, match="created_at"):
        Mem0RestAdapter._validate_projection_sources(
            binding=_ScopeBinding(
                ingestion_occurrence_id=RUN_ID,
                ingestion_plan_id=PLAN_ID,
            ),
            projection=projection,
            completed_sources=(source,),
        )


def test_independent_mem0_candidate_reconstruction_rejects_date_drift() -> None:
    from oamb.artifacts.validation.mem0_evidence import (
        Mem0PlanEvidence,
        Mem0ProjectionEvidence,
        reconstruct_mem0_candidates,
    )
    from oamb.memory_systems.mem0.projection import (
        parse_projection_pages,
        projection_state_sha256,
    )

    projection = parse_projection_pages(
        (
            (FIXTURES / "inspector" / "page-1.json").read_bytes(),
            (FIXTURES / "inspector" / "page-2.json").read_bytes(),
        ),
        expected_collection="oamb_memories",
        expected_run_id=RUN_ID,
    )
    search = json.loads((FIXTURES / "rest" / "search-success.json").read_bytes())
    search["results"][0]["created_at"] = "2027-01-01T00:00:00+00:00"

    class Plan:
        ingestion_occurrence_id = RUN_ID

    class Case:
        retrieval_raw_ref = "search-response"

    evidence = Mem0PlanEvidence(
        plan=Plan(),  # type: ignore[arg-type]
        projection=Mem0ProjectionEvidence(
            projection=projection,
            ordered_source_unit_ids=(SOURCE_ID,),
            state_sha256=projection_state_sha256(projection),
            capture_sequence=1,
            summary_raw_ref="projection-summary",
            page_raw_refs=("projection-page-1", "projection-page-2"),
        ),
    )

    with pytest.raises(ValueError, match="sealed projection"):
        reconstruct_mem0_candidates(
            raw_payloads={
                "search-response": json.dumps(search, separators=(",", ":")).encode(),
            },
            case=Case(),  # type: ignore[arg-type]
            plan=evidence,
        )


def test_projection_parser_hashes_unattributed_visible_memory() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    raw = json.dumps(
        {
            "collection": "oamb_memories",
            "run_id": RUN_ID,
            "count": 1,
            "points": [
                {
                    "id": "11111111-1111-4111-8111-111111111111",
                    "payload": {
                        "data": "Provider-visible memory without source attribution.",
                        "text_lemmatized": "provider visible memory",
                        "hash": "11111111111111111111111111111111",
                        "created_at": "2026-08-28T01:02:03+00:00",
                        "updated_at": "2026-08-28T01:02:03+00:00",
                        "run_id": RUN_ID,
                    },
                }
            ],
            "next_cursor": None,
        },
        separators=(",", ":"),
    ).encode()

    projection = mem0.parse_projection_pages(
        (raw,),
        expected_collection="oamb_memories",
        expected_run_id=RUN_ID,
    )

    assert projection.points[0].metadata is None


@pytest.mark.parametrize(
    ("fixture_name", "message"),
    (
        ("point-extra-field.json", "projection point fields"),
        ("payload-extra-field.json", "projection payload fields"),
        ("nonterminal-empty.json", "non-terminal projection page is empty"),
    ),
)
def test_projection_parser_rejects_planted_envelope_failures(
    fixture_name: str,
    message: str,
) -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")

    with pytest.raises(ValueError, match=message):
        mem0.parse_projection_pages(
            ((FIXTURES / "inspector" / fixture_name).read_bytes(),),
            expected_collection="oamb_memories",
            expected_run_id=RUN_ID,
        )


def test_projection_parser_rejects_incomplete_duplicate_and_count_drift_pages() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    page_1 = (FIXTURES / "inspector" / "page-1.json").read_bytes()

    with pytest.raises(ValueError, match="terminal projection page"):
        mem0.parse_projection_pages(
            (page_1,),
            expected_collection="oamb_memories",
            expected_run_id=RUN_ID,
        )
    with pytest.raises(ValueError, match="duplicate projection point id"):
        mem0.parse_projection_pages(
            (
                page_1,
                (FIXTURES / "inspector" / "duplicate-point-page-2.json").read_bytes(),
            ),
            expected_collection="oamb_memories",
            expected_run_id=RUN_ID,
        )
    with pytest.raises(ValueError, match="count changed"):
        mem0.parse_projection_pages(
            (
                page_1,
                (FIXTURES / "inspector" / "count-drift-page-2.json").read_bytes(),
            ),
            expected_collection="oamb_memories",
            expected_run_id=RUN_ID,
        )


def test_projection_parser_fails_closed_at_page_and_point_limits() -> None:
    mem0 = importlib.import_module("oamb.memory_systems.mem0")
    pages = (
        (FIXTURES / "inspector" / "page-1.json").read_bytes(),
        (FIXTURES / "inspector" / "page-2.json").read_bytes(),
    )

    with pytest.raises(ValueError, match="page limit"):
        mem0.parse_projection_pages(
            pages,
            expected_collection="oamb_memories",
            expected_run_id=RUN_ID,
            max_pages=1,
        )
    with pytest.raises(ValueError, match="point limit"):
        mem0.parse_projection_pages(
            pages,
            expected_collection="oamb_memories",
            expected_run_id=RUN_ID,
            max_points=1,
        )
