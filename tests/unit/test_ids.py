from __future__ import annotations

import importlib
from datetime import UTC, datetime
from decimal import Decimal
from types import ModuleType

import pytest
from hypothesis import given
from hypothesis import strategies as st


def require_ids() -> ModuleType:
    try:
        return importlib.import_module("oamb.contracts.ids")
    except ModuleNotFoundError:
        pytest.fail("oamb.contracts.ids is not implemented", pytrace=False)


def test_canonical_json_bytes_freezes_supported_primitives() -> None:
    ids = require_ids()

    actual = ids.canonical_json_bytes(
        {
            "when": datetime(2026, 8, 27, 12, 30, tzinfo=UTC),
            "amount": Decimal("1.20"),
            "label": "雪\u0000山",
        }
    )

    assert actual == (
        b'{"amount":"1.2","label":"\xe9\x9b\xaa\\u0000\xe5\xb1\xb1",'
        b'"when":"2026-08-27T12:30:00+00:00"}'
    )
    assert not actual.endswith(b"\n")


@given(st.dictionaries(st.text(min_size=1), st.integers(), max_size=12))
def test_canonical_json_is_independent_of_map_insertion_order(values: dict[str, int]) -> None:
    ids = require_ids()

    assert ids.canonical_json_bytes(values) == ids.canonical_json_bytes(
        dict(reversed(tuple(values.items())))
    )


def test_canonical_json_preserves_sequence_order_and_rejects_naive_time() -> None:
    ids = require_ids()

    assert ids.canonical_json_bytes(["a", "b"]) != ids.canonical_json_bytes(["b", "a"])
    with pytest.raises(ValueError, match="timezone"):
        ids.canonical_json_bytes(datetime(2026, 8, 27, 12, 30))


def test_length_safe_identity_does_not_have_delimiter_collisions() -> None:
    ids = require_ids()

    assert ids.sha256_identity("example-v1", ("a|b", "c")) != ids.sha256_identity(
        "example-v1", ("a", "b|c")
    )


def test_frozen_identity_formulas_match_protocol_goldens() -> None:
    ids = require_ids()

    context_content_id = ids.context_content_id(
        "dataset-r1", "test", "fixtures/source.json", "a" * 64
    )
    context_manifest_entry_id = ids.context_manifest_entry_id(
        "b" * 64, "c" * 64, 1, context_content_id
    )
    case_manifest_entry_id = ids.case_manifest_entry_id(
        context_manifest_entry_id, 1, "d" * 64, "q-1"
    )
    plan_manifest_entry_id = ids.plan_manifest_entry_id("fake-v1", (context_manifest_entry_id,))
    ingestion_payload_hash = ids.ingestion_payload_hash(("e" * 64,))
    ingestion_plan_id = ids.ingestion_plan_id(plan_manifest_entry_id, ingestion_payload_hash)
    ingestion_occurrence_id = ids.ingestion_occurrence_id("run-1", "fake-memory", ingestion_plan_id)
    case_occurrence_id = ids.case_occurrence_id(ingestion_occurrence_id, case_manifest_entry_id)

    assert context_content_id == "156e9b4cfe4686e2ac1823ade21092790b1f49c219808cfd76053065cdd5c215"
    assert (
        context_manifest_entry_id
        == "1db3e1ffe249c379cce331b71de741f28faf967a7c5aaca7d222a08b8e086a39"
    )
    assert (
        case_manifest_entry_id == "1db4f1d2215baeb4fd99efa8e46f25b2f7b16e3ea709aaa36fcb919d58f1424b"
    )
    assert (
        plan_manifest_entry_id == "0ab28721088d77fab7d74028828864a9c2dbf866e0de29da2d310cb781bd95d3"
    )
    assert (
        ingestion_payload_hash == "d130c84d1c0bedfbde0c17817e7725d5a4fbd5af2605df89aa9de2a000acc8de"
    )
    assert ingestion_plan_id == "e458c187aced04e29828ebe7eef6e3099dec82ba8d7ed64ff812c1f31e0fe371"
    assert (
        ingestion_occurrence_id
        == "c2b951f6752499598d2a7ed8c21d41958a8f17a2ae03b014a871de5f493a41de"
    )
    assert case_occurrence_id == "b1b7d5cb12ac90674bc17bf37f1c183a32f4e1cbb103658a120cf452c30f6cbc"
    assert ids.attempt_id(case_occurrence_id, "answer", 1, "f" * 64) == (
        "ab674df8c3f6ba8f1ec3f58090512284d6355512638267b3379723f9e31eefba"
    )


def test_logical_rows_and_repeated_raw_ids_remain_distinct() -> None:
    ids = require_ids()

    content_id = ids.context_content_id("r1", "test", "source.json", "a" * 64)
    row_one = ids.context_manifest_entry_id("b" * 64, "c" * 64, 1, content_id)
    row_two = ids.context_manifest_entry_id("b" * 64, "c" * 64, 2, content_id)

    assert row_one != row_two
    assert ids.case_manifest_entry_id(row_one, 1, "d" * 64, "repeated") != (
        ids.case_manifest_entry_id(row_two, 1, "d" * 64, "repeated")
    )


def test_decimal_uses_the_same_canonical_path_inside_and_outside_models() -> None:
    ids = require_ids()
    accounting = importlib.import_module("oamb.contracts.accounting")
    record = accounting.CostRecord(
        cost_record_id="d" * 64,
        parent_kind="run",
        parent_id="run-1",
        basis=accounting.CostBasis.ESTIMATE_FROM_MEASURED_USAGE,
        indexing_view=accounting.IndexingView.NOT_APPLICABLE,
        amount=Decimal("1.20"),
        currency="USD",
        price_snapshot_id="price-1",
        source_usage_record_ids=(),
        source_resource_record_ids=(),
        proof_status=accounting.ProofStatus.MEASURED_COMPLETE,
        reason=None,
    )

    assert ids.canonical_json_bytes({"amount": Decimal("1.20")}) == (
        ids.canonical_json_bytes({"amount": record.amount})
    )
